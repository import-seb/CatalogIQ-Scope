"""Rule-only refinement with item evidence and component-aware confirmation.

Proposals are not must-link constraints. They become accepted edges only after
both pair evidence and existing component representatives agree. The shared
allocator and the frozen TF-IDF baseline are deliberately outside this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import math
import re

from .split_matching import ACTIVE_NAMES, MINERALS, SALTS, brand_block_key
from .split_rule_candidates_v3 import RuleCandidateConfig, generate_candidates
from .split_rule_evidence_v3 import prepare_rule_products_v3
from .splitting import Components, MISSING, grouping_view, gtin, listing_url, text


@dataclass(frozen=True)
class RuleRefinementConfig:
    candidate: RuleCandidateConfig = field(default_factory=RuleCandidateConfig)
    line_anchor_tokens: int = 2
    generic_line_min_tokens: int = 3
    informative_containment: float = 0.75
    informative_dice: float = 0.55
    shared_line_min_tokens: int = 2
    shared_head_min_tokens: int = 3
    cross_brand_min_shared_tokens: int = 4
    cross_brand_run_min_shared_tokens: int = 2
    cross_brand_run_min_tokens: int = 12
    cross_brand_min_name_chars: int = 90
    cross_brand_name_similarity: float = 0.70
    template_min_titles: int = 12
    template_min_distinct_heads: int = 4
    template_document_fraction: float = 0.70
    template_head_fraction: float = 0.10
    component_representatives: int = 4
    component_pair_support_fraction: float = 1.0

    def __post_init__(self):
        if not isinstance(self.candidate, RuleCandidateConfig):
            raise ValueError("candidate must be RuleCandidateConfig")
        for name in ("line_anchor_tokens", "generic_line_min_tokens", "shared_line_min_tokens", "shared_head_min_tokens",
                     "cross_brand_min_shared_tokens", "cross_brand_min_name_chars", "template_min_titles",
                     "template_min_distinct_heads", "component_representatives", "cross_brand_run_min_shared_tokens",
                     "cross_brand_run_min_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("informative_containment", "informative_dice", "cross_brand_name_similarity",
                     "template_document_fraction", "component_pair_support_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if not math.isfinite(self.template_head_fraction) or not 0 <= self.template_head_fraction <= 1:
            raise ValueError("template_head_fraction must be finite and in [0, 1]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "candidate" in values:
            values["candidate"] = RuleCandidateConfig(**values["candidate"])
        return cls(**values)


def _edit(a, b):
    a, b = sorted((a, b))
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def pair_evidence_v3(a, b, config):
    """Return a confirmed family relation and its explainable evidence channel.

    Ingredient differences are not an automatic veto: a distinctive brand-line
    or long copied item template can expose evaluation information across such
    variants. Same-brand advertising and generic category terms cannot do so.
    """
    if a.payload_key and a.payload_key == b.payload_key:
        return "exact_classifier_payload", 1.0
    if not a.name or not b.name:
        return None
    same_brand = bool(a.brand and brand_block_key(a.brand) == brand_block_key(b.brand))
    shared = a.core & b.core
    containment = len(shared) / min(len(a.core), len(b.core)) if a.core and b.core else 0.0
    dice = 2 * len(shared) / (len(a.core) + len(b.core)) if a.core or b.core else 0.0
    if a.name == b.name:
        if shared and same_brand:
            return "exact_product_family", 1.0
        if same_brand and a.branded_generic_line and b.branded_generic_line:
            return "exact_branded_generic_line", 1.0
    # Prefix anchors are product cores, not supplier words, forms or adjectives.
    named_line = (len(a.head) >= config.line_anchor_tokens and a.head == b.head)
    if same_brand and named_line:
        return "distinctive_named_line", max(dice, containment)
    if (same_brand and a.head and b.head and a.head[0] == b.head[0]
            and a.head[0] not in MINERALS and len(shared) >= config.shared_head_min_tokens):
        return "substantive_shared_line_head", max(dice, containment)
    # Shared substantive words tolerate word order and extra variant wording.
    if (same_brand and len(shared) >= config.shared_line_min_tokens
            and containment >= config.informative_containment and dice >= config.informative_dice):
        return "informative_name_family", containment
    # One substantive head can identify a manufacturer-specific active line,
    # when supported by an active name or explicitly matched strength. This does
    # not admit shared body words or an arbitrary supplier suffix.
    if (same_brand and a.head and b.head and a.head[0] == b.head[0]
            and (a.head[0] in ACTIVE_NAMES or a.strengths & b.strengths)):
        def compounds(product):
            words = product.name.split()
            return {left + " " + right for left, right in zip(words, words[1:])
                    if left in MINERALS and right in SALTS}
        left_compounds, right_compounds = compounds(a), compounds(b)
        # A broad mineral head and dose alone cannot establish a distinctive
        # line when the compounds positively differ. Strong named-line evidence
        # above remains able to keep a manufacturer's formulation variants.
        if not (left_compounds and right_compounds and left_compounds.isdisjoint(right_compounds)):
            return "active_or_strength_line", max(dice, containment)
    # A title can omit its declared brand (retailer abbreviations). Compare its
    # residual core against a fully named same-brand line rather than require
    # literal full-title similarity.
    if same_brand and a.core and a.core == b.core:
        return "equal_informative_name", 1.0
    # Different declared private labels need long, item-specific template
    # evidence. Four generic words or a shared manufacturer slogan do not count.
    if (not same_brand and len(shared) >= config.cross_brand_min_shared_tokens
            and min(len(a.name), len(b.name)) >= config.cross_brand_min_name_chars
            and containment >= config.informative_containment
            and _edit(a.name, b.name) >= config.cross_brand_name_similarity):
        return "substantive_cross_brand_template", _edit(a.name, b.name)
    if (not same_brand and named_line and len(shared) >= config.line_anchor_tokens
            and min(len(a.name), len(b.name)) >= config.cross_brand_min_name_chars
            and containment >= config.informative_containment
            and _edit(a.name, b.name) >= config.cross_brand_name_similarity):
        return "copied_distinctive_named_line", _edit(a.name, b.name)
    if (not same_brand and len(shared) >= config.cross_brand_run_min_shared_tokens
            and min(len(a.name), len(b.name)) >= config.cross_brand_min_name_chars
            and containment >= config.informative_containment):
        left_words, right_words = sorted((a.name.split(), b.name.split()))
        blocks = SequenceMatcher(None, left_words, right_words, autojunk=False).get_matching_blocks()
        if any(block.size >= config.cross_brand_run_min_tokens
               and len(set(left_words[block.a:block.a + block.size]) & shared)
               >= config.cross_brand_run_min_shared_tokens for block in blocks):
            return "copied_substantive_item_phrase", containment
    return None


def refine_rule_groups_v3(frame, ids, split_config, config=None):
    """Return ``(group_ids, accepted_edges, diagnostics)`` in input row order.

    IDs order proposals and representatives, making components reproducible
    even when input rows are permuted. Representative count limits comparison
    work; it never limits the number of records in a legitimate group.
    """
    config = config or RuleRefinementConfig()
    if len(frame) != len(ids):
        raise ValueError("Record identities must align with the input frame")
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids) or any(not i.strip() for i in ids):
        raise ValueError("Record identities must be unique and nonblank")
    view = grouping_view(frame)
    products, evidence_stats = prepare_rule_products_v3(view, split_config, config)
    candidates, candidate_stats = generate_candidates([p.name for p in products],
                [p.brand for p in products], ids, [p.core for p in products], config.candidate)
    stats = Counter(evidence_stats)
    # Candidate diagnostics include nested metadata. Counter.update performs
    # arithmetic when already populated, so assign those fields explicitly.
    for key, value in candidate_stats.items():
        stats[key] = value
    uf = Components(len(ids))
    members = {i: [i] for i in range(len(ids))}
    anchors = {i: products[i].anchors for i in range(len(ids))}
    representatives = {i: [i] for i in range(len(ids))}
    edges = []
    accepted = set()
    proposals = set()

    def order(a, b):
        return (a, b) if ids[a] < ids[b] else (b, a)

    def propose(a, b, channel):
        a, b = order(a, b)
        proposals.add((a, b))

    # Intact identifiers are proposals, never sufficient authority by themselves.
    brands = [p.brand for p in products]
    retailers = [text(v) for v in view["Retailer"]]
    gtins = [gtin(v) for v in view["Upc"]]
    stats["valid_gtin_rows"] = sum(bool(v) for v in gtins)
    keys_by_reason = {
        "classifier_payload": [p.payload_key for p in products],
        "gtin": gtins,
        "listing_url": [listing_url(v) for v in view["ProductUrl"]],
        "retailer_sku": [(r, s.strip().casefold()) if r and s.strip().casefold() not in MISSING else None
                         for r, s in zip(retailers, view["Sku"])],
        "brand_model": [(brand_block_key(b), m.strip().casefold())
                        if b and re.fullmatch(r"[\w-]{4,}", m.strip()) and re.search(r"\d", m)
                        and not re.fullmatch(r"\d+(?:\.\d+)?e[+-]?\d+", m.strip(), flags=re.I)
                        else None for b, m in zip(brands, view["ProductModelNumber"])],
    }
    for reason, keys in keys_by_reason.items():
        buckets = defaultdict(list)
        for i, key in enumerate(keys):
            if key:
                buckets[key].append(i)
        for key in sorted(buckets):
            group = sorted(buckets[key], key=lambda i: ids[i])
            # A deterministic star proposes shared-identifier copies without
            # quadratic comparisons; rejected members may still join through
            # independent name candidates or another identifier.
            for b in group[1:]:
                propose(group[0], b, "identifier_" + reason)
    for a, b, channels in candidates:
        proposals.add(order(a, b))
    del candidates

    pair_cache = {}

    def confirmed(a, b):
        if a == b:
            return "self", 1.0
        key = order(a, b)
        if key not in pair_cache:
            pair_cache[key] = pair_evidence_v3(products[key[0]], products[key[1]], config)
        return pair_cache[key]

    def select_representatives(group):
        # Include immutable first/last identities plus lexical extremes, then
        # farthest core representatives. This is stable and samples multiple
        # variants instead of only the endpoint of the latest proposed link.
        ordered = sorted(group, key=lambda i: ids[i])
        selected = [ordered[0]]
        if len(ordered) > 1 and config.component_representatives > 1:
            selected.append(ordered[-1])
        while len(selected) < min(len(ordered), config.component_representatives):
            remaining = [i for i in ordered if i not in selected]
            def diversity(i):
                similarity = max(len(products[i].core & products[j].core) /
                    max(1, len(products[i].core | products[j].core)) for j in selected)
                return similarity, ids[i]
            selected.append(min(remaining, key=diversity))
        return selected

    evidence_order = {"exact_classifier_payload": 0, "exact_product_family": 0, "exact_branded_generic_line": 0,
                      "equal_informative_name": 1, "distinctive_named_line": 2,
                      "substantive_shared_line_head": 2,
                      "active_or_strength_line": 3, "informative_name_family": 4,
                      "substantive_cross_brand_template": 5, "copied_substantive_item_phrase": 5,
                      "copied_distinctive_named_line": 5}
    matches = []
    for a, b in sorted(proposals, key=lambda pair: (ids[pair[0]], ids[pair[1]])):
        stats["candidate_pair_comparisons"] += 1
        evidence = pair_evidence_v3(products[a], products[b], config)
        if evidence:
            pair_cache[(a, b)] = evidence
            reason, score = evidence
            matches.append((evidence_order[reason], -score, ids[a], ids[b], a, b, reason, score))
        else:
            stats["pair_evidence_rejections"] += 1
    del proposals
    for _, _, _, _, a, b, reason, score in sorted(matches):
        root_a, root_b = uf.find(a), uf.find(b)
        if root_a != root_b:
            common_anchors = anchors[root_a] & anchors[root_b]
            if not common_anchors:
                stats["component_family_anchor_rejections"] += 1
                continue
            # New endpoints and every stable representative must receive
            # positive evidence from the other component's representatives.
            left = sorted(set(representatives[root_a]) | {a}, key=lambda i: ids[i])
            right = sorted(set(representatives[root_b]) | {b}, key=lambda i: ids[i])
            support = [[bool(confirmed(x, y)) for y in right] for x in left]
            stats["component_representative_comparisons"] += len(left) * len(right)
            threshold = config.component_pair_support_fraction
            if (any(sum(row) / len(right) < threshold for row in support)
                    or any(sum(row[j] for row in support) / len(left) < threshold for j in range(len(right)))):
                stats["component_representative_rejections"] += 1
                continue
            combined = members.pop(root_a) + members.pop(root_b)
            anchors.pop(root_a)
            anchors.pop(root_b)
            representatives.pop(root_a)
            representatives.pop(root_b)
            uf.union(a, b)
            root = uf.find(a)
            members[root] = combined
            anchors[root] = common_anchors
            representatives[root] = select_representatives(combined)
            stats["validated_component_merges"] += 1
        key = order(a, b)
        if key not in accepted:
            accepted.add(key)
            edges.append((key[0], key[1], reason, float(score)))
            stats[reason] += 1
    edges.sort(key=lambda e: (ids[e[0]], ids[e[1]], e[2]))
    stats["accepted_edges"] = len(edges)
    stats["groups_with_multiple_members"] = sum(len(v) > 1 for v in members.values())
    stats["largest_group"] = max((len(v) for v in members.values()), default=0)
    stats["representative_pair_cache_entries"] = len(pair_cache)
    return uf.groups(ids, "rule"), edges, dict(stats)
