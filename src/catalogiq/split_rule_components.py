"""Product-only candidate and component engine for versioned rule evidence.

This extracts the frozen v3 engine into a separately versioned helper: the v3
implementation and its saved historical behavior are left intact. Confirmation
and optional component constraints belong to product grouping, not allocation.
"""
from collections import Counter, defaultdict
import re

from .split_matching import brand_block_key
from .split_rule_candidates_v3 import generate_candidates
from .splitting import Components, MISSING, grouping_view, gtin, listing_url, text


def confirmed_rule_components(frame, ids, split_config, config, *,
                              prepare_products, confirm_pair, component_compatible=None):
    """Use v3 proposals, evidence ordering and deterministic representative checks.

``component_compatible`` receives every member's prepared product evidence,
so evidence-specific constraints cannot be bypassed by an unsampled bridge.
The helper does not read targets, split roles, or reviewed pair IDs.
"""
    if len(frame) != len(ids):
        raise ValueError("Record identities must align with the input frame")
    ids = [str(value) for value in ids]
    if len(set(ids)) != len(ids) or any(not value.strip() for value in ids):
        raise ValueError("Record identities must be unique and nonblank")
    view = grouping_view(frame)
    products, evidence_stats = prepare_products(view, split_config, config)
    candidates, candidate_stats = generate_candidates([p.name for p in products],
        [p.brand for p in products], ids, [p.core for p in products], config.candidate)
    stats = Counter(evidence_stats)
    for key, value in candidate_stats.items():
        stats[key] = value
    uf = Components(len(ids))
    members = {index: [index] for index in range(len(ids))}
    anchors = {index: products[index].anchors for index in range(len(ids))}
    representatives = {index: [index] for index in range(len(ids))}
    edges, accepted, proposals = [], set(), set()

    def order(left, right):
        return (left, right) if ids[left] < ids[right] else (right, left)

    retailers = [text(value) for value in view["Retailer"]]
    gtins = [gtin(value) for value in view["Upc"]]
    stats["valid_gtin_rows"] = sum(bool(value) for value in gtins)
    keys_by_reason = {
        "classifier_payload": [p.payload_key for p in products],
        "gtin": gtins,
        "listing_url": [listing_url(value) for value in view["ProductUrl"]],
        "retailer_sku": [(retailer, sku.strip().casefold())
            if retailer and sku.strip().casefold() not in MISSING else None
            for retailer, sku in zip(retailers, view["Sku"])],
        "brand_model": [(brand_block_key(product.brand), model.strip().casefold())
            if product.brand and re.fullmatch(r"[\w-]{4,}", model.strip())
            and re.search(r"\d", model)
            and not re.fullmatch(r"\d+(?:\.\d+)?e[+-]?\d+", model.strip(), flags=re.I)
            else None for product, model in zip(products, view["ProductModelNumber"])],
    }
    for keys in keys_by_reason.values():
        buckets = defaultdict(list)
        for index, key in enumerate(keys):
            if key:
                buckets[key].append(index)
        for key in sorted(buckets):
            group = sorted(buckets[key], key=lambda index: ids[index])
            for right in group[1:]:
                proposals.add(order(group[0], right))
    for left, right, _ in candidates:
        proposals.add(order(left, right))
    del candidates
    pair_cache = {}

    def confirmed(left, right):
        if left == right:
            return "self", 1.0
        key = order(left, right)
        if key not in pair_cache:
            pair_cache[key] = confirm_pair(products[key[0]], products[key[1]], config)
        return pair_cache[key]

    def select_representatives(group):
        ordered = sorted(group, key=lambda index: ids[index])
        selected = [ordered[0]]
        if len(ordered) > 1 and config.component_representatives > 1:
            selected.append(ordered[-1])
        while len(selected) < min(len(ordered), config.component_representatives):
            remaining = [index for index in ordered if index not in selected]

            def diversity(index):
                similarity = max(len(products[index].core & products[other].core) /
                    max(1, len(products[index].core | products[other].core)) for other in selected)
                return similarity, ids[index]

            selected.append(min(remaining, key=diversity))
        return selected

    evidence_order = {"exact_classifier_payload": 0, "exact_product_family": 0,
        "exact_branded_generic_line": 0, "equal_informative_name": 1,
        "distinctive_named_line": 2, "substantive_shared_line_head": 2,
        "active_or_strength_line": 3, "informative_name_family": 4,
        "substantive_cross_brand_template": 5, "copied_substantive_item_phrase": 5,
        "copied_distinctive_named_line": 5}
    matches = []
    for left, right in sorted(proposals, key=lambda pair: (ids[pair[0]], ids[pair[1]])):
        stats["candidate_pair_comparisons"] += 1
        evidence = confirm_pair(products[left], products[right], config)
        if evidence:
            pair_cache[(left, right)] = evidence
            reason, score = evidence
            matches.append((evidence_order[reason], -score, ids[left], ids[right],
                            left, right, reason, score))
        else:
            stats["pair_evidence_rejections"] += 1
    del proposals
    for _, _, _, _, left_index, right_index, reason, score in sorted(matches):
        root_left, root_right = uf.find(left_index), uf.find(right_index)
        if root_left != root_right:
            common_anchors = anchors[root_left] & anchors[root_right]
            if not common_anchors:
                stats["component_family_anchor_rejections"] += 1
                continue
            if component_compatible is not None and not component_compatible(
                    [products[index] for index in members[root_left]],
                    [products[index] for index in members[root_right]], config):
                stats["component_evidence_guard_rejections"] += 1
                continue
            left = sorted(set(representatives[root_left]) | {left_index}, key=lambda index: ids[index])
            right = sorted(set(representatives[root_right]) | {right_index}, key=lambda index: ids[index])
            support = [[bool(confirmed(a, b)) for b in right] for a in left]
            stats["component_representative_comparisons"] += len(left) * len(right)
            threshold = config.component_pair_support_fraction
            if (any(sum(row) / len(right) < threshold for row in support)
                    or any(sum(row[column] for row in support) / len(left) < threshold
                           for column in range(len(right)))):
                stats["component_representative_rejections"] += 1
                continue
            combined = members.pop(root_left) + members.pop(root_right)
            anchors.pop(root_left)
            anchors.pop(root_right)
            representatives.pop(root_left)
            representatives.pop(root_right)
            uf.union(left_index, right_index)
            root = uf.find(left_index)
            members[root] = combined
            anchors[root] = common_anchors
            representatives[root] = select_representatives(combined)
            stats["validated_component_merges"] += 1
        key = order(left_index, right_index)
        if key not in accepted:
            accepted.add(key)
            edges.append((key[0], key[1], reason, float(score)))
            stats[reason] += 1
    edges.sort(key=lambda edge: (ids[edge[0]], ids[edge[1]], edge[2]))
    stats["accepted_edges"] = len(edges)
    stats["groups_with_multiple_members"] = sum(len(group) > 1 for group in members.values())
    stats["largest_group"] = max((len(group) for group in members.values()), default=0)
    stats["representative_pair_cache_entries"] = len(pair_cache)
    return uf.groups(ids, "rule"), edges, dict(stats)
