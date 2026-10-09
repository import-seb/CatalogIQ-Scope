"""Conservative, label-blind family resolution with deterministic blocking.

Normalization and corroboration live in ``split_matching`` so the rule and
cosine approaches apply the same formulation and brand safety checks. These
rules never consult classification targets, source row numbers, or annotations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re

from .split_matching import brand_block_key, compatible, name_similarity, prepare_products
from .splitting import Components, MISSING, grouping_view, gtin, listing_url, text


def refine_rule_groups(frame, ids, config):
    """Return family components, accepted links, and comparison diagnostics.

    Identifier matches need compatible name/brand/formulation evidence. Exact
    comparison titles and pack-free family titles have independent passes;
    approximate comparisons use infrequent informative title tokens within a
    canonical brand. All decisions are ordered by immutable record identities.
    """
    if len(frame) != len(ids):
        raise ValueError("Record identities must align with the input frame")
    view = grouping_view(frame)
    products = prepare_products(frame, config)
    uf = Components(len(view))
    edges = []
    stats = Counter()
    linked_pairs = set()
    component_ingredients = {
        i: {p.ingredients} if p.ingredients else set()
        for i, p in enumerate(products)
    }

    def pair_order(a, b):
        return (a, b) if ids[a] < ids[b] else (b, a)

    def ingredient_conflict(a, b):
        # Missing evidence cannot bridge two positively contradictory formulas.
        # Partial lists can overlap, so only disjoint nonempty signatures veto.
        left = component_ingredients[uf.find(a)]
        right = component_ingredients[uf.find(b)]
        return any(x.isdisjoint(y) for x in left for y in right)

    def link(a, b, reason, score=1.0, allow_brand_override=False):
        a, b = pair_order(a, b)
        if (a, b) in linked_pairs:
            return True
        if not compatible(products[a], products[b], config,
                          allow_brand_override=allow_brand_override):
            stats["pair_compatibility_rejections"] += 1
            return False
        root_a, root_b = uf.find(a), uf.find(b)
        if root_a != root_b:
            if ingredient_conflict(a, b):
                stats["component_ingredient_conflict_rejections"] += 1
                return False
            signatures = (component_ingredients.pop(root_a)
                          | component_ingredients.pop(root_b))
            uf.union(a, b)
            component_ingredients[uf.find(a)] = signatures
        linked_pairs.add((a, b))
        edges.append((a, b, reason, float(score)))
        stats[reason] += 1
        return True

    def by_key(keys, reason, name_guard=False, allow_brand_override=False):
        buckets = defaultdict(list)
        for i, key in enumerate(keys):
            if key:
                buckets[key].append(i)
        for key in sorted(buckets):
            members = sorted(buckets[key], key=lambda i: ids[i])
            if len(members) < 2:
                continue
            # For exceptionally repetitive keys, compare against representatives
            # rather than accept an unguarded star or perform quadratic work.
            if len(members) > config.max_block_size:
                stats["oversized_key_buckets_representative_pass"] += 1
                anchors = []
                for b in members:
                    matched = False
                    for a in anchors:
                        stats["key_comparisons"] += 1
                        score = name_similarity(products[a], products[b])
                        if name_guard and score < config.identifier_name_threshold:
                            continue
                        if link(a, b, reason, score, allow_brand_override):
                            matched = True
                            break
                    if not matched:
                        anchors.append(b)
                continue
            for pos, a in enumerate(members):
                for b in members[pos + 1:]:
                    stats["key_comparisons"] += 1
                    score = name_similarity(products[a], products[b])
                    if name_guard and score < config.identifier_name_threshold:
                        stats["identifier_name_rejections"] += 1
                        continue
                    link(a, b, reason, score, allow_brand_override)

    brands = [p.brand for p in products]
    retailers = [text(v) for v in view["Retailer"]]
    valid_gtins = [gtin(v) for v in view["Upc"]]
    stats["valid_gtin_rows"] = sum(bool(v) for v in valid_gtins)
    by_key(valid_gtins, "corroborated_gtin", name_guard=True)
    by_key([listing_url(v) for v in view["ProductUrl"]],
           "corroborated_listing_url", name_guard=True)
    skus = [str(v).strip().casefold() for v in view["Sku"]]
    by_key([(r, s) if r and s not in MISSING else None
            for r, s in zip(retailers, skus)],
           "retailer_sku_name", name_guard=True)
    models = [str(v).strip().casefold() for v in view["ProductModelNumber"]]
    by_key([(b, m) if b and re.fullmatch(r"[\w-]{4,}", m)
            and re.search(r"\d", m)
            and not re.fullmatch(r"\d+(?:\.\d+)?e[+-]?\d+", m) else None
            for b, m in zip(brands, models)],
           "brand_model_name", name_guard=True)

    # The global exact passes can recover manufacturer/sub-brand aliases only
    # when compatible() finds positive identity support; copied generic titles
    # across distinct private labels do not receive a free pass.
    # Identical full titles can still be useful for branded generic names whose
    # tokens are all marketing scaffolding. Keep brand/formulation safeguards;
    # the pack-free family pass still requires an informative core.
    by_key([p.name if len(p.name) >= 12 else None for p in products],
           "compatible_exact_name", allow_brand_override=True)
    by_key([p.family if len(p.family) >= 12 and p.core else None for p in products],
           "compatible_exact_family", allow_brand_override=True)

    frequencies = Counter(t for p in products for t in p.core)
    blocks = defaultdict(list)
    for i, p in enumerate(products):
        if not p.brand:
            continue
        eligible = sorted((t for t in p.core if len(t) >= 3 and not t.isdigit()),
                          key=lambda t: (frequencies[t], t))[:config.block_tokens]
        for token in eligible:
            blocks["brand", brand_block_key(p.brand), token].append(i)
            # An additional title-prefix pass proposes parent/sub-brand pairs.
            # compatible() still requires positive brand evidence and rejects
            # uncorroborated generic titles across unrelated private labels.
            if p.tokens and p.family:
                blocks["title_prefix", p.family.split()[0], token].append(i)
    seen = set()
    for key in sorted(blocks):
        members = sorted(blocks[key], key=lambda i: ids[i])
        if len(members) > config.max_block_size:
            stats["oversized_blocks_skipped"] += 1
            continue
        for pos, a in enumerate(members):
            for b in members[pos + 1:]:
                pair = pair_order(a, b)
                if pair in seen or pair in linked_pairs:
                    continue
                seen.add(pair)
                stats["blocked_comparisons"] += 1
                score = name_similarity(products[a], products[b])
                if score < config.rule_family_threshold:
                    continue
                left, right = sorted((products[a].family, products[b].family))
                edit = SequenceMatcher(None, left, right, autojunk=False).ratio()
                if edit >= config.rule_family_edit_threshold:
                    link(a, b, "blocked_family_name", score, allow_brand_override=True)
    stats["accepted_edges"] = len(edges)
    return uf.groups(ids, "rule"), edges, dict(stats)
