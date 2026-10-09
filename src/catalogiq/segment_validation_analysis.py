"""Validation-only diagnostics for the controlled Segment baseline experiment.

These functions do not train, choose splits, inspect final-test records, or alter
product groups. Confidence intervals resample frozen product groups; subgroup
comparisons describe associations rather than causal effects of leakage.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


COHORT_COLUMNS = (
    "has_rule_group_train_neighbor", "has_independent_near_train_neighbor",
    "has_exact_payload_train_neighbor",
)


def _labels(label_names):
    labels = [str(label) for label in label_names]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("label_names must be nonempty and unique")
    return labels


def _predictions(frame, label_names):
    labels = _labels(label_names)
    required = {"record_id", "true_label", "predicted_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions require {sorted(required)}")
    if (frame.record_id.isna().any() or frame.record_id.astype(str).eq("").any()
            or frame.record_id.astype(str).duplicated().any()):
        raise ValueError("prediction record IDs must be nonempty and unique")
    if frame[["true_label", "predicted_label"]].isna().any().any():
        raise ValueError("prediction labels cannot be missing")
    positions = {label: i for i, label in enumerate(labels)}
    true = frame.true_label.astype(str).map(positions)
    predicted = frame.predicted_label.astype(str).map(positions)
    if true.isna().any() or predicted.isna().any():
        raise ValueError("prediction labels are outside label_names")
    return labels, true.to_numpy(dtype=int), predicted.to_numpy(dtype=int)


def _confusion_metrics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted_support, out=np.zeros_like(diagonal),
                          where=predicted_support > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    denominator = support + predicted_support
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    size = float(support.sum())
    metrics = {
        "records": int(size),
        "accuracy": float(diagonal.sum() / size) if size else None,
        "macro_f1": float(f1.mean()) if size else None,
        "weighted_f1": float(np.dot(f1, support) / size) if size else None,
        "balanced_accuracy": float(recall[support > 0].mean()) if size else None,
        "present_class_macro_f1": float(f1[support > 0].mean()) if size else None,
        "classes_with_true_support": int((support > 0).sum()),
        "macro_f1_uses_all_declared_labels": True,
    }
    return metrics, precision, recall, f1, support, predicted_support


def evaluate_predictions(predictions, label_names, probability_columns=None, bins=10):
    """Return scalar metrics and per-class, confusion, and calibration tables.

    The order of ``probability_0 ... probability_(K-1)`` is ``label_names``.
    Macro F1 always includes every declared label, even when a cohort lacks one;
    balanced accuracy and present-class F1 include only labels with true support.
    Multiclass Brier is the mean sum of squared errors, with range [0, 2].
    """
    labels, true, predicted = _predictions(predictions, label_names)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("calibration bins must be a positive integer")
    # Empty fixed cohorts are valid and remain unscored. Newer scikit-learn
    # rejects empty inputs, so construct their declared-label matrix directly.
    matrix = (confusion_matrix(true, predicted, labels=np.arange(len(labels)))
              if len(true) else np.zeros((len(labels), len(labels)), dtype=np.int64))
    metrics, precision, recall, f1, support, predicted_support = _confusion_metrics(matrix)
    metrics["unsupported_true_labels"] = [label for label, n in zip(labels, support) if n == 0]
    metrics.update({"negative_log_likelihood": None, "multiclass_brier": None,
                    "expected_calibration_error": None})
    columns = list(probability_columns) if probability_columns is not None else [
        f"probability_{i}" for i in range(len(labels))]
    present = [column in predictions for column in columns]
    if any(present) and not all(present):
        raise ValueError("probability columns must be provided for every label")
    probability = None
    if all(present):
        if len(columns) != len(labels) or len(set(columns)) != len(labels):
            raise ValueError("probability columns must match label_names exactly")
        probability = predictions[columns].to_numpy(dtype=float)
        if (not np.isfinite(probability).all() or (probability < 0).any()
                or (probability > 1).any()
                or not np.allclose(probability.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("probabilities must be finite, bounded, and sum to one")
        if len(true) and not np.all(probability[np.arange(len(true)), predicted]
                                     >= probability.max(axis=1) - 1e-7):
            raise ValueError("predicted labels must maximize their probability")
        if len(true):
            metrics["negative_log_likelihood"] = float(-np.log(
                np.clip(probability[np.arange(len(true)), true], 1e-12, 1)).mean())
            targets = np.eye(len(labels))[true]
            metrics["multiclass_brier"] = float(np.square(probability - targets).sum(axis=1).mean())
    confidence = None
    if "confidence" in predictions:
        confidence = predictions.confidence.to_numpy(dtype=float)
        if (not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
            raise ValueError("confidence must be finite and in [0, 1]")
        if probability is not None and not np.allclose(confidence, probability.max(axis=1),
                                                       atol=1e-5, rtol=0):
            raise ValueError("confidence must equal the maximum label probability")
    elif probability is not None:
        confidence = probability.max(axis=1)
    calibration_rows = []
    if confidence is not None:
        indices = np.minimum((confidence * bins).astype(int), bins - 1)
        error = 0.0
        for index in range(bins):
            selected = indices == index
            count = int(selected.sum())
            accuracy = float((true[selected] == predicted[selected]).mean()) if count else None
            average = float(confidence[selected].mean()) if count else None
            if count:
                error += count / len(true) * abs(accuracy - average)
            calibration_rows.append({"bin": index, "lower": index / bins,
                "upper": (index + 1) / bins, "records": count,
                "mean_confidence": average, "accuracy": accuracy})
        metrics["expected_calibration_error"] = float(error) if len(true) else None
    per_class = pd.DataFrame({"label": labels, "precision": precision, "recall": recall,
                             "f1": f1, "true_support": support.astype(int),
                             "predicted_support": predicted_support.astype(int)})
    confusion = pd.DataFrame(matrix, index=pd.Index(labels, name="true_label"),
                              columns=pd.Index(labels, name="predicted_label"))
    return {"metrics": metrics, "per_class": per_class, "confusion": confusion,
            "calibration": pd.DataFrame(calibration_rows, columns=["bin", "lower", "upper",
                "records", "mean_confidence", "accuracy"])}


def majority_baseline(train_labels: Sequence, validation_labels: Sequence, label_names):
    """Fit the constant class on training labels only, then score validation."""
    labels = _labels(label_names)
    positions = {label: i for i, label in enumerate(labels)}
    train = [str(label) for label in train_labels]
    validation = [str(label) for label in validation_labels]
    if not train or any(label not in positions for label in train + validation):
        raise ValueError("training labels must be nonempty and all labels declared")
    counts = np.bincount([positions[label] for label in train], minlength=len(labels))
    # Ties use the frozen declared class order, rather than validation information.
    winner = labels[int(counts.argmax())]
    frame = pd.DataFrame({"record_id": [f"baseline_{i}" for i in range(len(validation))],
        "true_label": validation, "predicted_label": winner})
    return {"majority_label": winner, "training_records": len(train),
            "training_class_counts": dict(zip(labels, counts.astype(int).tolist())),
            **evaluate_predictions(frame, labels)["metrics"]}


def _group_mapping(groups):
    if not {"record_id", "group_id"}.issubset(groups.columns):
        raise ValueError("groups require record_id and group_id")
    if (groups[["record_id", "group_id"]].isna().any().any()
            or groups.record_id.astype(str).duplicated().any()
            or groups.record_id.astype(str).eq("").any()
            or groups.group_id.astype(str).eq("").any()):
        raise ValueError("group mapping must have unique nonempty record IDs and groups")
    return dict(zip(groups.record_id.astype(str), groups.group_id.astype(str)))


def _training_pair_neighbors(pairs, train_ids, universe):
    required = {"left_record_id", "right_record_id"}
    if not required.issubset(pairs.columns):
        raise ValueError("audit pairs require left_record_id and right_record_id")
    neighbors = defaultdict(set)
    for left, right in pairs[["left_record_id", "right_record_id"]].itertuples(index=False, name=None):
        if pd.isna(left) or pd.isna(right):
            raise ValueError("audit pair endpoints cannot be missing")
        left, right = str(left), str(right)
        if left not in universe or right not in universe or left == right:
            raise ValueError("audit pair endpoints must be distinct members of the group universe")
        if left in train_ids:
            neighbors[right].add(left)
        if right in train_ids:
            neighbors[left].add(right)
    return {record: len(peers) for record, peers in neighbors.items()}


def annotate_training_neighbors(predictions, train_ids, frozen_groups, near_pairs,
                                exact_pairs=None):
    """Annotate validation-only rows using frozen groups and independent audits.

    Only training neighbors count. Pair direction and repeated audit edges have
    no effect. Missing exact audit evidence is marked unavailable, not negative.
    Near-neighbor absence only means absence from this fixed audit candidate pool.
    """
    mapping = _group_mapping(frozen_groups)
    train = {str(record) for record in train_ids}
    if not train.issubset(mapping):
        raise ValueError("training IDs are outside the frozen group universe")
    if ("record_id" not in predictions or predictions.record_id.isna().any()
            or predictions.record_id.astype(str).duplicated().any()):
        raise ValueError("prediction IDs must be unique and nonmissing")
    ids = predictions.record_id.astype(str)
    if not set(ids).issubset(mapping) or set(ids) & train:
        raise ValueError("validation IDs must be mapped and disjoint from training IDs")
    group_counts = defaultdict(int)
    for record in train:
        group_counts[mapping[record]] += 1
    result = predictions.copy()
    result["group_id"] = ids.map(mapping)
    result["rule_group_train_neighbors"] = result.group_id.map(group_counts).fillna(0).astype(int)
    result["has_rule_group_train_neighbor"] = result.rule_group_train_neighbors.gt(0)
    near_counts = _training_pair_neighbors(near_pairs, train, mapping)
    result["independent_near_train_neighbors"] = ids.map(near_counts).fillna(0).astype(int)
    result["has_independent_near_train_neighbor"] = result.independent_near_train_neighbors.gt(0)
    result["exact_payload_audit_available"] = exact_pairs is not None
    if exact_pairs is None:
        result["exact_payload_train_neighbors"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["has_exact_payload_train_neighbor"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    else:
        exact_counts = _training_pair_neighbors(exact_pairs, train, mapping)
        result["exact_payload_train_neighbors"] = ids.map(exact_counts).fillna(0).astype(int)
        result["has_exact_payload_train_neighbor"] = result.exact_payload_train_neighbors.gt(0)
    return result


def _cluster_confusions(predictions, label_names, group_column):
    labels, true, predicted = _predictions(predictions, label_names)
    if group_column not in predictions or predictions[group_column].isna().any():
        raise ValueError("cluster bootstrap requires a nonmissing group column")
    if predictions[group_column].astype(str).eq("").any():
        raise ValueError("cluster IDs cannot be empty")
    ordered = predictions.assign(_record=predictions.record_id.astype(str),
                                  _group=predictions[group_column].astype(str)).sort_values("_record")
    _, true, predicted = _predictions(ordered, labels)
    groups = sorted(ordered._group.unique())
    lookup = {group: i for i, group in enumerate(groups)}
    indices = ordered._group.map(lookup).to_numpy(dtype=int)
    k = len(labels)
    flattened = indices * k * k + true * k + predicted
    matrices = np.bincount(flattened, minlength=len(groups) * k * k).reshape(len(groups), k, k)
    return labels, groups, matrices


def _bootstrap_configuration(n_bootstrap, confidence):
    if (not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 1
            or not 0 < confidence < 1):
        raise ValueError("bootstrap resamples must be positive and confidence in (0, 1)")


def bootstrap_predictions(predictions, label_names, group_column="group_id",
                          n_bootstrap=500, seed=42, confidence=0.95):
    """Cluster-resampled accuracy and all-label macro-F1 intervals.

    Every member of a sampled group receives the same multiplicity. These
    intervals cover validation sampling uncertainty, not training-seed variation,
    grouping correctness, or uncertainty about the production population.
    """
    _bootstrap_configuration(n_bootstrap, confidence)
    labels, groups, matrices = _cluster_confusions(predictions, label_names, group_column)
    point = _confusion_metrics(matrices.sum(axis=0))[0]
    rows = []
    rng = np.random.default_rng(seed)
    values = np.empty((n_bootstrap, 2))
    if groups:
        for i in range(n_bootstrap):
            selected = rng.integers(0, len(groups), size=len(groups))
            metric = _confusion_metrics(matrices[selected].sum(axis=0))[0]
            values[i] = metric["accuracy"], metric["macro_f1"]
    alpha = (1 - confidence) / 2
    for index, metric in enumerate(("accuracy", "macro_f1")):
        bounds = np.quantile(values[:, index], [alpha, 1 - alpha]) if groups else [None, None]
        rows.append({"metric": metric, "estimate": point[metric],
                     "lower": None if bounds[0] is None else float(bounds[0]),
                     "upper": None if bounds[1] is None else float(bounds[1])})
    return {"records": len(predictions), "groups": len(groups), "resamples": n_bootstrap,
            "seed": int(seed), "confidence": float(confidence), "label_names": labels,
            "method": "frozen_group_cluster_percentile_bootstrap",
            "caveat": "Validation sampling uncertainty only; training-seed and production uncertainty excluded.",
            "intervals": pd.DataFrame(rows)}


def compare_paired_predictions(left, right, label_names, frozen_groups,
                               n_bootstrap=500, seed=42, confidence=0.95):
    """Compare models only on identical validation rows, with paired resampling.

    Positive differences favor ``left``. Different validation pools are described
    by their intersection coverage; their raw overall score difference is not a
    causal leakage estimate. No prediction outside the intersection is scored.
    """
    _bootstrap_configuration(n_bootstrap, confidence)
    labels, _, _ = _predictions(left, label_names)
    _predictions(right, labels)
    left_lookup = left.assign(record_id=left.record_id.astype(str)).set_index("record_id")
    right_lookup = right.assign(record_id=right.record_id.astype(str)).set_index("record_id")
    common = sorted(set(left_lookup.index) & set(right_lookup.index))
    a = left_lookup.loc[common].reset_index()
    b = right_lookup.loc[common].reset_index()
    if a.true_label.astype(str).tolist() != b.true_label.astype(str).tolist():
        raise ValueError("paired prediction truth labels disagree")
    mapping = _group_mapping(frozen_groups)
    if not set(common).issubset(mapping):
        raise ValueError("paired IDs are outside frozen groups")
    a["group_id"] = a.record_id.map(mapping)
    b["group_id"] = b.record_id.map(mapping)
    _, groups, matrices_a = _cluster_confusions(a, labels, "group_id")
    _, groups_b, matrices_b = _cluster_confusions(b, labels, "group_id")
    if groups != groups_b:
        raise ValueError("paired group order mismatch")
    point_a = _confusion_metrics(matrices_a.sum(axis=0))[0]
    point_b = _confusion_metrics(matrices_b.sum(axis=0))[0]
    rng = np.random.default_rng(seed)
    samples = np.empty((n_bootstrap, 2))
    if groups:
        for i in range(n_bootstrap):
            selection = rng.integers(0, len(groups), size=len(groups))
            ma = _confusion_metrics(matrices_a[selection].sum(axis=0))[0]
            mb = _confusion_metrics(matrices_b[selection].sum(axis=0))[0]
            samples[i] = (ma["accuracy"] - mb["accuracy"], ma["macro_f1"] - mb["macro_f1"])
    alpha = (1 - confidence) / 2
    rows = []
    for index, metric in enumerate(("accuracy", "macro_f1")):
        low, high = np.quantile(samples[:, index], [alpha, 1 - alpha]) if groups else (None, None)
        rows.append({"metric": metric, "left": point_a[metric], "right": point_b[metric],
            "left_minus_right": point_a[metric] - point_b[metric] if groups else None,
            "lower": None if low is None else float(low), "upper": None if high is None else float(high)})
    return {"common_records": len(common), "common_groups": len(groups),
            "left_records": len(left), "right_records": len(right),
            "left_coverage": len(common) / len(left) if len(left) else None,
            "right_coverage": len(common) / len(right) if len(right) else None,
            "resamples": n_bootstrap, "seed": int(seed), "confidence": float(confidence),
            "comparison": "left_minus_right_on_identical_validation_records",
            "intervals": pd.DataFrame(rows), "record_ids": common}


def summarize_cohorts(predictions, label_names, cohort_columns=COHORT_COLUMNS):
    """Report supported classes and metrics for each neighbor-present/absent cohort."""
    labels = _labels(label_names)
    metrics_rows, class_rows = [], []
    subsets = [("all_validation", predictions)]
    for column in cohort_columns:
        if column not in predictions:
            continue
        valid = predictions[column].dropna()
        if not valid.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("cohort indicators must be boolean or missing")
        for value in (False, True):
            selected = predictions[column].eq(value).fillna(False)
            subsets.append((f"{column}={str(value).lower()}", predictions.loc[selected]))
    for name, subset in subsets:
        evaluated = evaluate_predictions(subset, labels)
        row = {"cohort": name, **evaluated["metrics"]}
        row["record_share"] = len(subset) / len(predictions) if len(predictions) else None
        row["unsupported_class_caveat"] = bool(row["unsupported_true_labels"])
        metrics_rows.append(row)
        for item in evaluated["per_class"].to_dict(orient="records"):
            class_rows.append({"cohort": name, **item})
    return {"metrics": pd.DataFrame(metrics_rows), "per_class": pd.DataFrame(class_rows),
            "interpretation": "Descriptive cohort association; class composition and selection differ. "
                "All-label macro F1 assigns zero to unsupported labels; present-class F1 is also supplied."}


def affected_row_bound(predictions, indicator="has_independent_near_train_neighbor"):
    """Describe the conditional direct-only accuracy bound for an audited cohort.

    If only the identified rows' correctness could change, their share is the
    largest possible total accuracy change. This is NOT a causal or global bound:
    related training data can affect other predictions, and the audit misses pairs.
    """
    if indicator not in predictions:
        raise ValueError("affected-row indicator is missing")
    observed = predictions[indicator].dropna()
    if not observed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("affected-row indicator must be boolean")
    affected = int(predictions[indicator].fillna(False).astype(bool).sum())
    size = len(predictions)
    share = affected / size if size else None
    return {"validation_records": size, "audited_affected_records": affected,
            "indicator_missing_records": int(predictions[indicator].isna().sum()),
            "audited_affected_row_share": share,
            "conditional_direct_only_accuracy_bound_percentage_points": 100 * share if size else None,
            "assumption": "Only the identified validation rows' correctness changes.",
            "is_causal_or_population_bound": False,
            "caveat": "The independent candidate audit is incomplete; changing training data can also affect "
                "other validation rows. This is not a bound on total leakage-induced optimism."}
