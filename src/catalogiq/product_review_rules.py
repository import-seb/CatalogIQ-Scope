"""Arrival-time product gates, exploratory discovery and group-aware scoring.

Never trains a classifier, fits calibration, changes confidence cutoffs or
discovers rules from policy-audit labels. Bounds reuse clustered_accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clustered_accuracy import aggregate_groups, analytic_bounds


MIN_DISCOVERY_CLASS_GROUPS = 25
MIN_POLICY_GROUPS = 300
MIN_POLICY_EFFECTIVE_ROWS = 300
COMMON_TRAINING_ROWS = 1000
FIXED_THRESHOLDS = {"segment": .93, "brand_lookup_probabilistic_v1": .955}
CLASS_MODES = ("all", "discovered_99", "discovered_995", "discovered_error_free", "training_common")
GATES = {
    "segment": ("all", "description", "category", "description_category", "seen_brand", "seen_brand_description_category"),
    "brand_lookup_probabilistic_v1": ("all", "description", "category", "description_category", "unambiguous_key", "unambiguous_key_support20"),
}
SEARCHED_RULES = sum(len(g)*len(CLASS_MODES) for g in GATES.values())
ARRIVAL_FIELDS = (
    "predicted_label", "confidence", "base_eligible", "description_present", "category_present",
    "brand_seen", "brand_key_unambiguous", "brand_training_groups", "predicted_class_training_rows",
)
RULE_KEYS = {"id", "model", "class_mode", "gate", "threshold", "allowed_labels", "vocabulary"}


def require_role(frame, role):
    if (frame.empty or "experiment_role" not in frame
            or not frame.experiment_role.eq(role).all()
            or frame.record_id.duplicated().any()
            or frame.group_id.isna().any() or frame.group_id.astype(str).str.strip().eq("").any()):
        raise ValueError(f"Only nonempty unique records in {role} may be used here")
    if "split" in frame and not frame.split.eq("validation").all():
        raise ValueError("Only existing validation records may enter rule evaluation")


def check_phase_isolation(discovery, selection):
    require_role(discovery, "calibration_fit")
    require_role(selection, "threshold_selection")
    if (set(discovery.record_id) & set(selection.record_id)
            or set(discovery.group_id) & set(selection.group_id)):
        raise ValueError("Discovery and selection must be record- and group-disjoint")


def apply_rule(frame, rule, *, expected_threshold):
    """Only whitelisted arrival-time columns are inspected by this decision."""
    if set(rule) != RULE_KEYS or rule["model"] not in GATES:
        raise ValueError("Unknown rule fields/model; truth-dependent conditions are forbidden")
    if expected_threshold != FIXED_THRESHOLDS[rule["model"]] or rule["threshold"] != expected_threshold:
        raise ValueError("Existing confidence threshold must remain exactly fixed")
    if rule["class_mode"] not in CLASS_MODES or rule["gate"] not in GATES[rule["model"]]:
        raise ValueError("Unsupported class screen or information gate")
    if not set(ARRIVAL_FIELDS).issubset(frame):
        raise ValueError("Missing arrival-time fields")
    part = frame[list(ARRIVAL_FIELDS)]
    conf = pd.to_numeric(part.confidence, errors="coerce").to_numpy(float)
    accepted = (part.base_eligible.eq(True).to_numpy() & np.isfinite(conf)
                & (conf >= expected_threshold) & (conf <= 1)
                & part.predicted_label.isin(rule["vocabulary"]).to_numpy())
    if rule["allowed_labels"] is not None:
        if not set(rule["allowed_labels"]).issubset(rule["vocabulary"]):
            raise ValueError("Rule whitelist contains unsupported classes")
        accepted &= part.predicted_label.isin(rule["allowed_labels"]).to_numpy()
    if rule["class_mode"] == "training_common":
        accepted &= part.predicted_class_training_rows.ge(COMMON_TRAINING_ROWS).to_numpy()
    gate = rule["gate"]
    if gate in ("description", "description_category", "seen_brand_description_category"):
        accepted &= part.description_present.eq(True).to_numpy()
    if gate in ("category", "description_category", "seen_brand_description_category"):
        accepted &= part.category_present.eq(True).to_numpy()
    if gate in ("seen_brand", "seen_brand_description_category"):
        accepted &= part.brand_seen.eq(True).to_numpy()
    if gate in ("unambiguous_key", "unambiguous_key_support20"):
        accepted &= part.brand_key_unambiguous.eq(True).to_numpy()
    if gate == "unambiguous_key_support20":
        accepted &= part.brand_training_groups.ge(20).to_numpy()
    return accepted


def summarize_policy(frame, accepted, *, alpha=.05):
    correct = frame.predicted_label.eq(frame.true_label).to_numpy()
    accepted = np.asarray(accepted, bool)
    groups = aggregate_groups(accepted, correct, frame.group_id)
    result = analytic_bounds(groups.accepted_rows, groups.errors, alpha)
    return {"population_rows": len(frame), "population_groups": int(frame.group_id.nunique()),
            "accepted_rows": result["rows"], "review_rows": len(frame)-result["rows"],
            "coverage": result["rows"]/len(frame), "review_fraction": 1-result["rows"]/len(frame),
            "errors": result["errors"], "accepted_accuracy": result["accuracy"],
            "accepted_error_rate": 1-result["accuracy"] if result["accuracy"] is not None else None,
            "accepted_groups": result["groups"], "n_eff_df": result["n_eff_df"],
            "max_accepted_group_rows": result["max_group_rows"], "beta_lower": result["beta_effective_sample"],
            "sufficient_groups": bool(result["groups"] >= MIN_POLICY_GROUPS
                                      and (result["n_eff_df"] or 0) >= MIN_POLICY_EFFECTIVE_ROWS),
            "alpha": alpha}


def discover_rules(discovery, model, vocabulary, threshold):
    require_role(discovery, "calibration_fit")
    baseline = {"id": "all__all", "model": model, "class_mode": "all", "gate": "all",
                "threshold": threshold, "allowed_labels": None, "vocabulary": list(vocabulary)}
    accepted = apply_rule(discovery, baseline, expected_threshold=threshold)
    accepted_frame = discovery.loc[accepted]
    screens = {"all": None, "training_common": None,
               "discovered_99": [], "discovered_995": [], "discovered_error_free": []}
    evidence = []
    for label in vocabulary:
        part = accepted_frame.loc[accepted_frame.predicted_label.eq(label)]
        n, g = len(part), part.group_id.nunique()
        e = int(part.predicted_label.ne(part.true_label).sum())
        accuracy = 1-e/n if n else None
        enough = g >= MIN_DISCOVERY_CLASS_GROUPS
        if enough:
            for mode, target in (("discovered_99", .99), ("discovered_995", .995), ("discovered_error_free", 1.)):
                if accuracy >= target:
                    screens[mode].append(label)
        evidence.append({"predicted_class": label, "discovery_accepted_rows": n,
                         "discovery_accepted_groups": int(g), "discovery_errors": e,
                         "discovery_accuracy": accuracy, "eligible_for_screen": enough,
                         "per_class_99_percent_guarantee": False})
    registry = [{**baseline, "id": f"{mode}__{gate}", "class_mode": mode, "gate": gate,
                 "allowed_labels": screens[mode]} for mode in CLASS_MODES for gate in GATES[model]]
    return registry, pd.DataFrame(evidence)


def score_registry(selection, registry):
    require_role(selection, "threshold_selection")
    rows = []
    for rule in registry:
        accepted = apply_rule(selection, rule, expected_threshold=rule["threshold"])
        nominal = summarize_policy(selection, accepted)
        adjusted = summarize_policy(selection, accepted, alpha=.05/SEARCHED_RULES)
        rows.append({"rule_id": rule["id"], "model": rule["model"], **nominal,
                     "beta_lower_search_adjusted": adjusted["beta_lower"],
                     "search_alpha": .05/SEARCHED_RULES})
    return pd.DataFrame(rows)


def choose_rule(selection_scores):
    """Selection rows only; no audit-based fallback or tuning is permitted."""
    baseline = selection_scores.loc[selection_scores.rule_id.eq("all__all")].iloc[0]
    pool = selection_scores.loc[selection_scores.rule_id.ne("all__all") & selection_scores.sufficient_groups].copy()
    supported = pool.loc[pool.beta_lower_search_adjusted.ge(.99)]
    status = "search_adjusted_selection_support"
    if supported.empty:
        supported = pool.loc[pool.beta_lower.ge(.99)]
        status = "nominal_selection_support_only"
    if supported.empty:
        supported = pool.loc[pool.accepted_accuracy.ge(.99)
                             & pool.accepted_error_rate.lt(baseline.accepted_error_rate)]
        status = "exploratory_improvement_only"
        order = ["beta_lower", "coverage", "rule_id"]
        ascending = [False, False, True]
    else:
        order = ["coverage", "beta_lower", "rule_id"]
        ascending = [False, False, True]
    if supported.empty:
        return {"rule_id": None, "status": "no_supported_improvement", "enabled": False,
                "independently_confirmed": False}
    chosen = supported.sort_values(order, ascending=ascending, kind="stable").iloc[0]
    return {"rule_id": chosen.rule_id, "status": status, "enabled": False,
            "independently_confirmed": False, "selection_summary": chosen.to_dict()}


def diagnostic_tables(frame, baseline, *, role):
    """Descriptive associations; truth-based facets are explicitly non-operational."""
    require_role(frame, role)
    facets = {
        "predicted_class": (frame.predicted_label, True),
        "true_class": (frame.true_label, False),
        "predicted_class_frequency": (np.where(frame.predicted_class_training_rows.ge(COMMON_TRAINING_ROWS), "common", "uncommon"), True),
        "true_class_frequency": (np.where(frame.true_class_training_rows.ge(COMMON_TRAINING_ROWS), "common", "uncommon"), False),
        "brand_familiarity": (frame.brand_familiarity, True),
        "description": (np.where(frame.description_present, "present", "missing"), True),
        "product_category": (np.where(frame.category_present, "present", "missing"), True),
        "brand_key_relationship": (np.where(frame.brand_key_unambiguous, "unambiguous", "mixed_or_unseen"), True),
        "brand_training_support": (pd.cut(frame.brand_training_groups, [-1, 0, 4, 19, np.inf],
                                           labels=["unseen_or_missing", "1_to_4_groups", "5_to_19_groups", "20_plus_groups"]).astype(str), True),
        "confidence_range": (pd.cut(frame.confidence, [0, .5, .9, .93, .955, .97, .99, .995, 1.000001],
            right=False, labels=["below_50", "50_to_90", "90_to_93", "93_to_95.5", "95.5_to_97", "97_to_99", "99_to_99.5", "99.5_to_100"]).astype(str), True),
    }
    baseline_mask = apply_rule(frame, baseline, expected_threshold=baseline["threshold"])
    rows = []
    for facet, (values, available) in facets.items():
        values = np.asarray(values)
        for value in sorted(set(values)):
            mask = values == value
            part = frame.loc[mask]
            summary = summarize_policy(part, baseline_mask[mask])
            rows.append({"role": role, "characteristic": facet, "value": value,
                         "available_at_arrival": available, "descriptive_not_confirmatory": True,
                         "all_predictions_errors": int(part.predicted_label.ne(part.true_label).sum()),
                         "all_predictions_accuracy": float(part.predicted_label.eq(part.true_label).mean()),
                         **summary})
    return pd.DataFrame(rows)


def uncommon_class_table(frame, accepted, vocabulary, *, policy_name, training_counts):
    rows = []
    for label in [*vocabulary, *sorted(set(frame.true_label)-set(vocabulary))]:
        truth = frame.true_label.eq(label).to_numpy()
        emitted = frame.predicted_label.eq(label).to_numpy()
        train_count = int(training_counts.get(label, 0))
        n = int((accepted & emitted).sum())
        e = int((accepted & emitted & ~truth).sum())
        rows.append({"policy": policy_name, "class": label, "training_rows": train_count,
            "uncommon_true_class": train_count < COMMON_TRAINING_ROWS,
            "truth_rows": int(truth.sum()), "accepted_truth_rows": int((accepted & truth).sum()),
            "truth_class_coverage": float(accepted[truth].mean()) if truth.any() else None,
            "accepted_predictions": n, "incorrect_accepted_predictions": e,
            "accepted_precision": 1-e/n if n else None,
            "missed_true_class_errors": int((accepted & truth & ~emitted).sum()),
            "per_class_99_percent_confirmed": False})
    return pd.DataFrame(rows)
