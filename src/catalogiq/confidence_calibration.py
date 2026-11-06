"""Development-only probability calibration, selective risk, and review rules.

No classifier fitting, inference, split reassignment, or protected-test loading.
All temperature parameters must be fitted on a separate group-disjoint subset.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, xlogy
from scipy.stats import beta

from .structural import DESCRIPTIVE_FIELDS, StructuralConfig, missing


ROLES = ("calibration_fit", "threshold_selection", "policy_audit")
GROUP_SALT = "catalogiq-confidence-v1-20261103"
PROBABILITY_FLOOR = 1e-15


def group_role(group_id: str, salt: str = GROUP_SALT) -> str:
    """Shared deterministic assignment across targets; never stratify on labels."""
    digest = hashlib.sha256((salt + "\0" + str(group_id)).encode("utf8")).digest()
    return ROLES[int.from_bytes(digest[:8], "big") % len(ROLES)]


def partition_context(context: pd.DataFrame) -> pd.DataFrame:
    required = {"record_id", "group_id", "split", "true_label"}
    if (not required.issubset(context) or context.empty
            or context.record_id.duplicated().any() or context[list(required)].isna().any().any()
            or context.group_id.astype(str).str.strip().eq("").any()
            or not context.split.eq("validation").all()):
        raise ValueError("Unique labeled validation context with nonblank groups required")
    result = context.copy()
    result["experiment_role"] = result.group_id.map(group_role)
    if set(result.experiment_role) != set(ROLES):
        raise ValueError("All three group-disjoint development subsets need data")
    if result.groupby("group_id").experiment_role.nunique().gt(1).any():
        raise ValueError("Experiment groups overlap")
    return result


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Finite normalized multiclass probabilities required")
    # Correct only floating-point serialization noise; never change source files.
    return p / p.sum(axis=1, keepdims=True)


def temperature_scale(probabilities, temperature):
    p = validate_probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if temperature == 1:
        return p.copy()
    # Exports have probabilities rather than original logits. log(p)/T is
    # equivalent up to a shared constant for positive probabilities. Zeros stay
    # zero, so exact zeros/ties cannot acquire a new winning class by clipping.
    with np.errstate(divide="ignore"):
        z = np.log(p) / temperature
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def truth_indices(truth, labels):
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError("Unique ordered vocabulary required")
    lookup = {label: i for i, label in enumerate(labels)}
    return np.asarray([lookup.get(label, -1) for label in truth], dtype=int)


def negative_log_likelihood(p, truth, labels):
    """Infinite for OOV truth or zero true probability; finite clipped sensitivity."""
    y = truth_indices(truth, labels)
    if len(y) != len(p) or not len(y) or p.shape[1] != len(labels):
        raise ValueError("Nonempty probability/truth rows must match")
    true_p = np.zeros(len(y))
    known = y >= 0
    true_p[known] = p[np.arange(len(y))[known], y[known]]
    infinite = bool((true_p == 0).any())
    return {
        "nll": None if infinite else float(-np.log(true_p).mean()),
        "nll_is_infinite": infinite,
        "nll_clipped_1e15": float(-np.log(np.maximum(true_p, PROBABILITY_FLOOR)).mean()),
        "zero_true_probability_rows": int((true_p == 0).sum()),
    }


def fit_temperature(probabilities, truth, labels):
    """Fit the scalar on calibration-fit rows only; never silently discard OOV."""
    p = validate_probabilities(probabilities)
    y = truth_indices(truth, labels)
    if len(p) != len(y) or not len(y) or p.shape[1] != len(labels):
        raise ValueError("Nonempty fitting rows must match")
    if (y < 0).any() or (p[np.arange(len(y)), y] == 0).any():
        return {"temperature": 1., "status": "unidentifiable_infinite_nll",
                "reason": "OOV/zero true probability retained; no finite NLL fit"}

    def objective(log_t):
        q = temperature_scale(p, float(np.exp(log_t)))
        return float(-np.log(np.maximum(q[np.arange(len(y)), y], PROBABILITY_FLOOR)).mean())

    bounds = (math.log(.05), math.log(20.))
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-9})
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    options = [(0., objective(0.)), (result.x, result.fun),
               (bounds[0], objective(bounds[0])), (bounds[1], objective(bounds[1]))]
    best = min(options, key=lambda item: item[1])[0]
    return {"temperature": float(np.exp(best)), "status": "fitted",
            "bounds": [.05, 20.], "at_boundary": bool(best in bounds),
            "fit_nll_before": objective(0.), "fit_nll_after": objective(best)}


def reliability(confidence, correct, bins=10):
    conf, correct = np.asarray(confidence, float), np.asarray(correct, float)
    if (conf.shape != correct.shape or not len(conf) or bins < 1
            or not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any()):
        raise ValueError("Matching nonempty confidences/correctness required")
    edges = np.linspace(0, 1, bins + 1)
    # Same right-closed bins as Maria's pinned definitions, including 0 and 1.
    indices = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    rows = []
    for i in range(bins):
        mask = indices == i
        rows.append({"lower": float(edges[i]), "upper": float(edges[i+1]), "rows": int(mask.sum()),
                     "mean_confidence": float(conf[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    ece = sum(r["rows"] / len(conf) * abs(r["accuracy"] - r["mean_confidence"])
              for r in rows if r["rows"])
    return rows, float(ece)


def calibration_metrics(probabilities, truth, labels):
    p = validate_probabilities(probabilities)
    y = truth_indices(truth, labels)
    if len(y) != len(p) or not len(y) or p.shape[1] != len(labels):
        raise ValueError("Nonempty probability/truth rows must match")
    correct = p.argmax(axis=1) == y
    rows, ece = reliability(p.max(axis=1), correct)
    # Extend the vector with an OOV truth coordinate whose predicted p is zero.
    # Brier = sum(p^2) + 1 - 2*p_true also handles unknown truth correctly.
    true_p = np.zeros(len(y))
    known = y >= 0
    true_p[known] = p[np.arange(len(y))[known], y[known]]
    accuracy, confidence = float(correct.mean()), float(p.max(axis=1).mean())
    return {"rows": len(p), "accuracy": accuracy, "mean_confidence": confidence,
            "confidence_minus_accuracy": confidence - accuracy, "ece_10_bins": ece,
            "brier_multiclass_sum": float((np.square(p).sum(axis=1) + 1 - 2*true_p).mean()),
            "oov_truth_rows": int((~known).sum()), **negative_log_likelihood(p, truth, labels)}, rows


def threshold_grid():
    """50--99.9%; 0.1 percentage point spacing, 0.01 points above 99%."""
    return np.unique(np.r_[np.arange(500, 1000) / 1000, np.arange(9900, 9991) / 10000])


def bernoulli_kl(a, b):
    return float(xlogy(a, a / b) + xlogy(1-a, (1-a) / (1-b)))


def cluster_accuracy_lower(errors, accepted, groups, max_group_rows, alpha=.05):
    """One-sided bounded-cluster KL concentration lower bound on row accuracy.

    Given fixed accepted group volumes, X_g=errors_g/M lies in [0,1].
    For independent groups, P(mean(X) <= x) <= exp(-G*KL(x, E mean(X))).
    Invert that bound, then multiply by G*M/accepted to bound row error.
    No independence inside a group is assumed. Can be very conservative for
    large unequal groups. This is conditional development evidence; future
    group size/distribution changes are not covered by the calculation.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if accepted == 0:
        return None
    if not (0 <= errors <= accepted and 0 < groups <= accepted <= groups * max_group_rows):
        raise ValueError("Invalid accepted group counts")
    x = errors / (groups * max_group_rows)
    if x == 0:
        upper = -math.expm1(math.log(alpha) / groups)
    elif x == 1:
        upper = 1.
    else:
        upper = brentq(lambda u: groups * bernoulli_kl(x, u) + math.log(alpha),
                       x + 1e-14, 1 - 1e-14)
    return max(0., 1 - upper * groups * max_group_rows / accepted)


def accepted_summary(accepted, correct, groups, alpha=.05):
    accepted, correct, groups = np.asarray(accepted, bool), np.asarray(correct, bool), np.asarray(groups)
    if accepted.shape != correct.shape or accepted.shape != groups.shape or not len(accepted):
        raise ValueError("Matching nonempty masks/groups required")
    n = int(accepted.sum())
    errors = int((accepted & ~correct).sum())
    sizes = pd.Series(groups[accepted]).value_counts()
    maximum = int(sizes.max()) if n else 0
    # Reference only: an exact binomial bound wrongly assumes independent rows.
    iid_lower = float(beta.ppf(alpha, n-errors, errors+1)) if n > errors else (0. if n else None)
    return {"rows": len(accepted), "accepted_rows": n, "review_rows": len(accepted)-n,
            "coverage": n/len(accepted), "review_fraction": 1-n/len(accepted),
            "accepted_errors": errors, "accepted_accuracy": (n-errors)/n if n else None,
            "accepted_error_rate": errors/n if n else None,
            "accepted_groups": len(sizes), "max_accepted_group_rows": maximum,
            "cluster_accuracy_lower": cluster_accuracy_lower(errors, n, len(sizes), maximum, alpha),
            "iid_row_accuracy_lower_reference_only": iid_lower, "alpha": alpha}


def threshold_table(confidence, correct, groups, eligible=None, thresholds=None, alpha=.05):
    confidence = np.asarray(confidence, float)
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Finite confidence in [0,1] required")
    eligible = np.ones(len(confidence), bool) if eligible is None else np.asarray(eligible, bool)
    thresholds = threshold_grid() if thresholds is None else np.asarray(thresholds, float)
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Invalid threshold")
    return pd.DataFrame([{"threshold": float(t), **accepted_summary(eligible & (confidence >= t), correct, groups, alpha)}
                         for t in thresholds])


def bootstrap_accuracy(accepted, correct, groups, seed=20261103, samples=2000):
    """Resample whole groups. Descriptive only; zero-error samples can degenerate."""
    frame = pd.DataFrame({"group": groups, "n": np.asarray(accepted, int),
                          "errors": np.asarray(accepted, bool) & ~np.asarray(correct, bool)})
    totals = frame.groupby("group")[["n", "errors"]].sum().to_numpy()
    if not totals[:, 0].sum():
        return {"lower": None, "upper": None, "samples": samples, "degenerate_zero_errors": False}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        n, errors = totals[rng.integers(len(totals), size=len(totals))].sum(axis=0)
        if n:
            values.append(1-errors/n)
    lo, hi = np.quantile(values, [.025, .975])
    return {"lower": float(lo), "upper": float(hi), "samples": samples,
            "degenerate_zero_errors": bool(totals[:, 1].sum() == 0)}


def per_class_table(truth, predicted, accepted, labels):
    truth, predicted, accepted = np.asarray(truth), np.asarray(predicted), np.asarray(accepted, bool)
    classes = [*labels, *sorted(set(truth) - set(labels))]
    indices = {label: i for i, label in enumerate(classes)}
    y = np.asarray([indices[v] for v in truth])
    pred = np.asarray([indices[v] for v in predicted])
    count = len(classes)
    support = np.bincount(y, minlength=count)
    accepted_truth = np.bincount(y[accepted], minlength=count)
    emitted = np.bincount(pred[accepted], minlength=count)
    errors = accepted & (y != pred)
    emitted_errors = np.bincount(pred[errors], minlength=count)
    missed_errors = np.bincount(y[errors], minlength=count)
    rows = []
    for i, label in enumerate(classes):
        n, wrong, actual = int(emitted[i]), int(emitted_errors[i]), int(support[i])
        rows.append({"class": label, "truth_rows": actual,
                     "accepted_truth_rows": int(accepted_truth[i]),
                     "truth_class_coverage": float(accepted_truth[i]/actual) if actual else None,
                     "accepted_predictions": n, "incorrect_accepted_predictions": wrong,
                     "missed_true_class_errors": int(missed_errors[i]),
                     "accepted_precision": (n-wrong)/n if n else None,
                     "minority_under_50_rows": actual < 50})
    return pd.DataFrame(rows)


def essential_information_reasons(features):
    config = StructuralConfig()
    def absent(value):
        # Native JSON null / DataFrame NA are missing too. Literal strings
        # such as "NA" still follow the existing blank/null-only policy.
        return value is None or bool(pd.isna(value)) or missing(str(value), config)

    return [["missing_essential_product_text"] if all(absent(row.get(c, ""))
                                                       for c in DESCRIPTIVE_FIELDS) else []
            for row in features.to_dict("records")]


def review_decision(probabilities, predicted_label, labels, threshold, product,
                    *, enabled=False, provisional=False, approved_relations=(),
                    classifications=None, lookup_status=None, quarantined=False):
    """Fail closed serving primitive; experiments never enable the returned policies.

    Approved relations must explicitly carry status/evidence and allowed pairs.
    No relation is inferred from label names, validation truth, or co-occurrence.
    """
    reasons = []
    if not enabled:
        reasons.append("automatic_acceptance_disabled")
    if provisional:
        reasons.append("provisional_model")
    valid_threshold = (isinstance(threshold, (int, float, np.integer, np.floating))
                       and not isinstance(threshold, (bool, np.bool_))
                       and np.isfinite(threshold) and 0 <= threshold <= 1)
    if not valid_threshold:
        reasons.append("no_approved_threshold")
    try:
        p = validate_probabilities(np.asarray(probabilities, float).reshape(1, -1))[0]
        if len(p) != len(labels) or len(set(labels)) != len(labels):
            raise ValueError("Vocabulary mismatch")
        if predicted_label not in labels:
            reasons.append("unsupported_classification")
        elif predicted_label != labels[int(p.argmax())]:
            reasons.append("conflicting_probability_classification")
        if valid_threshold and p.max() < threshold:
            reasons.append("below_confidence_threshold")
    except (ValueError, TypeError):
        reasons.append("invalid_probabilities")
    reasons.extend(essential_information_reasons(pd.DataFrame([product]))[0])
    if quarantined:
        reasons.append("existing_quarantine")
    if lookup_status in ("missing", "unseen"):
        reasons.append("lookup_without_matching_training_key")
    for relation in approved_relations:
        if relation.get("status") != "approved" or not relation.get("evidence"):
            reasons.append("unapproved_relationship_rule")
            continue
        values = classifications or {}
        try:
            pair = (values.get(relation["child_target"]), values.get(relation["parent_target"]))
            allowed = {tuple(p) for p in relation["allowed_pairs"]}
            if any(len(p) != 2 for p in allowed):
                raise ValueError("Relationship pairs need two classes")
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_approved_relationship_rule")
            continue
        if None in pair or any(str(v).strip() == "" for v in pair):
            reasons.append("missing_relationship_classification")
        elif pair not in allowed:
            reasons.append("invalid_approved_classification_relationship")
    return {"accept": not reasons, "review_reasons": sorted(set(reasons))}
