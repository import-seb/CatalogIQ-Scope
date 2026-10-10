"""Focused product-evidence fixes following the leakage audit.

The frozen v3 baseline is preserved. V4 changes only demonstrated generic
supplier/category evidence: domain suffixes and extraction scaffolding cannot
identify an ingredient; category-first device names cannot merge different
device families through an untyped bridge. Targets and allocation are absent.
"""
from dataclasses import dataclass

from .split_rule_candidates_v3 import RuleCandidateConfig
from .split_rule_components import confirmed_rule_components
from .split_rule_evidence_v3 import RuleProductEvidence, prepare_rule_products_v3
from .split_rules_v3 import RuleRefinementConfig, pair_evidence_v3

VERSION = "rule-final-v4"


@dataclass(frozen=True)
class RuleFinalConfig(RuleRefinementConfig):
    comparison_noise: tuple[str, ...] = ("com", "net", "org", "www", "extract", "extracts",
                                         "herbal", "herb", "herbs", "root", "roots")
    generic_device_categories: tuple[str, ...] = ("motion sickness",)
    device_families: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("eyewear", ("glasses", "eyeglasses", "goggles")),
        ("acupressure", ("wristband", "wristbands", "band", "bands", "bracelet", "bracelets")),
    )

    def __post_init__(self):
        super().__post_init__()
        for field in ("comparison_noise", "generic_device_categories"):
            values = getattr(self, field)
            if (not isinstance(values, tuple) or any(not isinstance(value, str)
                    or not value.strip() or value != value.casefold().strip() for value in values)):
                raise ValueError(f"{field} must contain lowercase nonblank strings")
        if any(len(value.split()) != 1 for value in self.comparison_noise):
            raise ValueError("comparison_noise must contain individual words")
        if not isinstance(self.device_families, tuple):
            raise ValueError("device_families must be a tuple")
        labels = []
        for entry in self.device_families:
            if (not isinstance(entry, tuple) or len(entry) != 2
                    or not isinstance(entry[0], str) or not entry[0].strip()
                    or not isinstance(entry[1], tuple) or not entry[1]
                    or any(not isinstance(word, str) or len(word.split()) != 1
                           or word != word.casefold().strip() for word in entry[1])):
                raise ValueError("device_families requires named lowercase token lists")
            labels.append(entry[0])
        if len(set(labels)) != len(labels):
            raise ValueError("device family names must be distinct")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "candidate" in values:
            values["candidate"] = RuleCandidateConfig(**values["candidate"])
        for field in ("comparison_noise", "generic_device_categories"):
            if field in values:
                values[field] = tuple(values[field])
        if "device_families" in values:
            values["device_families"] = tuple((label, tuple(tokens))
                                              for label, tokens in values["device_families"])
        return cls(**values)


@dataclass(frozen=True)
class FinalProductEvidence(RuleProductEvidence):
    category_device_families: frozenset[str] = frozenset()


def prepare_rule_products_v4(frame, split_config, config):
    """Change comparison evidence only; preserve every source and payload value."""
    products, stats = prepare_rule_products_v3(frame, split_config, config)
    noise = frozenset(config.comparison_noise)
    categories = tuple(tuple(phrase.split()) for phrase in config.generic_device_categories)
    prepared = []
    changed, category_rows = 0, 0
    for product in products:
        ordered = tuple(word for word in product.ordered_core if word not in noise)
        is_category_first = any(ordered[:len(category)] == category for category in categories)
        families = frozenset()
        if is_category_first:
            category_rows += 1
            families = frozenset(label for label, tokens in config.device_families
                                 if product.tokens.intersection(tokens))
            for category in categories:
                if ordered[:len(category)] == category:
                    ordered = ordered[len(category):]
                    break
        changed += ordered != product.ordered_core
        core = frozenset(ordered)
        anchors = frozenset(anchor for anchor in product.anchors if not anchor.startswith("word:"))
        anchors |= frozenset("word:" + word for word in core)
        prepared.append(FinalProductEvidence(**{
            **product.__dict__, "core": core, "ordered_core": ordered,
            "head": ordered[:config.line_anchor_tokens], "anchors": anchors,
            "category_device_families": families,
        }))
    stats.update({"generic_evidence_rows_changed": changed,
                  "category_first_device_rows": category_rows,
                  "group_size_cap_applied": False})
    return prepared, stats


def pair_evidence_v4(left, right, config):
    """Keep v3 substantive line rules; reject conflicting category-only devices."""
    if left.payload_key and left.payload_key == right.payload_key:
        return "exact_classifier_payload", 1.0
    if (left.category_device_families and right.category_device_families
            and left.category_device_families.isdisjoint(right.category_device_families)):
        return None
    return pair_evidence_v3(left, right, config)


def component_compatible_v4(left, right, config):
    """Prevent unknown intermediate records from bridging separate device types."""
    products = (*left, *right)
    payloads = {product.payload_key for product in products}
    if len(payloads) == 1 and "" not in payloads:
        return True
    families = set().union(*(product.category_device_families for product in products))
    return len(families) <= 1


def refine_rule_groups_v4(frame, ids, split_config, config=None):
    """Return deterministic target-free groups, accepted edges and diagnostics."""
    config = config or RuleFinalConfig()
    if not isinstance(config, RuleFinalConfig):
        raise ValueError("config must be RuleFinalConfig")
    groups, edges, stats = confirmed_rule_components(frame, ids, split_config, config,
        prepare_products=prepare_rule_products_v4, confirm_pair=pair_evidence_v4,
        component_compatible=component_compatible_v4)
    stats["version"] = VERSION
    stats["rule_final_config"] = config.to_dict()
    return groups, edges, stats
