"""Pre-training population checks for the full-pool Segment comparison.

This module does not train models or consume predictions. Protected source rows
contribute only to aggregate feature distributions and label-availability counts;
class and group analyses use the permitted development records.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .segment_experiment import METHODS, validate_development_assignments


VERSION = "segment-development-profile-v1"
FEATURES = ("ProductBrand", "ProductDescription", "ProductContents")
SIZE_BUCKETS = ("1", "2", "3-5", "6-10", "11+")
MISSING = ("", "null", "<missing>")


def _missing(values):
    return values.isna() | values.fillna("").astype(str).str.strip().str.casefold().isin(MISSING)


def _retailer(values):
    result = values.fillna("").astype(str).str.strip().str.casefold()
    result = result.str.replace(r"\s+", " ", regex=True)
    return result.mask(_missing(values), "<MISSING>")


def _bucket(size):
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return "11+"


def _validate_source(frame, assignments, exclusions, expected_development_ids):
    required = {"record_id", "Segment", "Retailer", *FEATURES}
    if not required.issubset(frame):
        raise ValueError(f"source frame requires {sorted(required)}")
    if frame.record_id.duplicated().any() or _missing(frame.record_id).any():
        raise ValueError("source record identities must be unique and nonmissing")
    if not {"record_id", "reason"}.issubset(exclusions):
        raise ValueError("exclusions require record_id and reason")
    if exclusions.record_id.duplicated().any() or _missing(exclusions.reason).any():
        raise ValueError("each exclusion needs one nonmissing reason")
    validate_development_assignments(assignments, exclusions)
    source_ids = set(frame.record_id)
    development_ids = set(assignments.record_id)
    excluded_ids = set(exclusions.record_id)
    if source_ids != development_ids | excluded_ids:
        raise ValueError("development and exclusions must cover exactly the complete source")
    labeled = frame.loc[~_missing(frame.Segment)].set_index("record_id")
    if not development_ids.issubset(labeled.index):
        raise ValueError("an unlabeled source record reached development")
    assigned = assignments.loc[assignments.strategy.eq("group_aware")].set_index("record_id")
    if not assigned.Segment.equals(labeled.Segment.reindex(assigned.index)):
        raise ValueError("development labels differ from their source records")
    if expected_development_ids is not None:
        expected = set(expected_development_ids)
        if development_ids != expected:
            raise ValueError("development universe differs from labeled frozen-v3 non-test records")
    indexed = frame.set_index("record_id")
    unlabeled = exclusions.loc[exclusions.reason.eq("missing_Segment"), "record_id"]
    if not _missing(indexed.loc[unlabeled, "Segment"]).all():
        raise ValueError("a labeled record was excluded as missing Segment")
    return labeled, assigned


def _historical_partitions(previous_assignments, development, group_index):
    if previous_assignments is None:
        return {}, {}
    previous = previous_assignments.copy()
    validate_development_assignments(previous, pd.DataFrame(columns=["record_id"]))
    if not set(previous.record_id).issubset(development.index):
        raise ValueError("previous reduced pool must be a subset of current development")
    historical = previous.loc[previous.strategy.eq("group_aware")].set_index("record_id")
    wanted = group_index.loc[historical.index, ["group_id", "Segment"]]
    if not historical[["group_id", "Segment"]].equals(wanted):
        raise ValueError("previous reduced identities or labels differ from frozen development")
    features = {"previous_development_pool": development.loc[historical.index].copy()}
    groups = {"previous_development_pool": historical.copy()}
    for method in METHODS:
        for role in ("train", "validation"):
            name = f"previous_{method}_{role}"
            subset = previous.loc[previous.strategy.eq(method) & previous.split.eq(role)]
            features[name] = development.loc[subset.record_id].copy()
            groups[name] = subset.set_index("record_id").copy()
    return features, groups


def _original_partitions(frozen_role_assignments, source_ids, development, group_index):
    if frozen_role_assignments is None:
        return {}, {}
    frozen = frozen_role_assignments
    if not {"record_id", "group_id", "split"}.issubset(frozen):
        raise ValueError("frozen roles require record_id, group_id and split")
    if frozen.record_id.duplicated().any() or set(frozen.record_id) != source_ids:
        raise ValueError("frozen roles must cover the complete source exactly once")
    if not frozen.split.isin(("train", "validation", "test")).all():
        raise ValueError("unknown frozen partition")
    if frozen.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("frozen groups cross partitions")
    # Do not read historical/test target values. Use the guarded development map.
    selected = frozen.loc[frozen.record_id.isin(development.index),
                          ["record_id", "group_id", "split"]].set_index("record_id")
    if selected.split.eq("test").any():
        raise ValueError("frozen final-test record reached development profiles")
    if not selected.group_id.equals(group_index.group_id.reindex(selected.index)):
        raise ValueError("original roles changed the frozen product groups")
    selected["Segment"] = group_index.Segment.reindex(selected.index)
    features, groups = {}, {}
    for role in ("train", "validation"):
        name = f"original_frozen_{role}"
        groups[name] = selected.loc[selected.split.eq(role)].copy()
        features[name] = development.loc[groups[name].index].copy()
    return features, groups


def _tvd(distributions, left, right):
    return .5 * float((distributions[left] - distributions[right]).abs().sum())


def profile_development(frame, assignments, exclusions, *, reference_features=None,
                        previous_assignments=None, expected_development_ids: Iterable | None = None,
                        frozen_role_assignments=None):
    """Describe the full development population before any training.

    ``expected_development_ids`` is the caller's independently derived set of
    labeled frozen-v3 train/validation IDs. Supply it to verify full-pool coverage.
    Production ``reference_features`` is immediately reduced to feature columns;
    its supplied targets are never accessed. All returned tables are deterministic
    under source and assignment row reordering.
    """
    labeled, group_index = _validate_source(frame, assignments, exclusions, expected_development_ids)
    group_index = group_index.sort_index(kind="stable")
    development = frame.set_index("record_id").loc[group_index.index].copy()
    feature_columns = ["Retailer", *FEATURES]
    populations = {"full_labeled_source": labeled.loc[:, feature_columns].copy(),
                   "development_pool": development.loc[:, feature_columns].copy()}
    grouped_populations = {"development_pool": group_index.copy()}
    for method in METHODS:
        for role in ("train", "validation"):
            name = f"{method}_{role}"
            subset = assignments.loc[assignments.strategy.eq(method) & assignments.split.eq(role)]
            grouped_populations[name] = subset.set_index("record_id").sort_index(kind="stable")
            populations[name] = development.loc[grouped_populations[name].index, feature_columns].copy()
    previous_features, previous_groups = _historical_partitions(previous_assignments, development, group_index)
    original_features, original_groups = _original_partitions(
        frozen_role_assignments, set(frame.record_id), development, group_index)
    populations.update({name: rows.loc[:, feature_columns] for name, rows in previous_features.items()})
    populations.update({name: rows.loc[:, feature_columns] for name, rows in original_features.items()})
    grouped_populations.update(previous_groups)
    grouped_populations.update(original_groups)
    if reference_features is not None:
        if not set(feature_columns).issubset(reference_features):
            raise ValueError(f"production reference requires feature columns {feature_columns}")
        populations["production_reference"] = reference_features.loc[:, feature_columns].copy()

    retailer_counts = {name: _retailer(rows.Retailer).value_counts()
                       for name, rows in populations.items()}
    retailers = sorted(set().union(*(set(counts.index) for counts in retailer_counts.values())))
    distributions = {name: counts.reindex(retailers, fill_value=0).astype(float) / len(populations[name])
                     if len(populations[name]) else pd.Series(0., index=retailers)
                     for name, counts in retailer_counts.items()}
    retailer_rows, missing_rows = [], []
    for name in sorted(populations):
        rows = populations[name]
        for retailer in retailers:
            retailer_rows.append({"population": name, "retailer": retailer,
                "records": int(retailer_counts[name].get(retailer, 0)), "population_records": len(rows),
                "fraction": float(distributions[name].loc[retailer])})
        for feature in FEATURES:
            missing = int(_missing(rows[feature]).sum())
            missing_rows.append({"population": name, "feature": feature, "population_records": len(rows),
                "missing_records": missing, "missing_fraction": missing / len(rows) if len(rows) else None})
    comparisons = [("full_labeled_source", "development_pool"),
                   ("random_train", "random_validation"), ("group_aware_train", "group_aware_validation"),
                   ("development_pool", "random_validation"), ("development_pool", "group_aware_validation")]
    if previous_assignments is not None:
        comparisons.extend([("full_labeled_source", "previous_development_pool"),
                            ("development_pool", "previous_development_pool")])
    if frozen_role_assignments is not None:
        comparisons.extend([("development_pool", "original_frozen_validation"),
                            ("original_frozen_train", "original_frozen_validation")])
    if reference_features is not None:
        comparisons.extend([("production_reference", name) for name in
                            ("full_labeled_source", "development_pool", "random_validation", "group_aware_validation")])
    tvd_rows, observations = [], []
    for left, right in comparisons:
        if not len(populations[left]) or not len(populations[right]):
            tvd_rows.append({"reference": left, "comparison": right, "total_variation_distance": None,
                             "largest_shift_retailer": None, "comparison_minus_reference_fraction": None})
            continue
        difference = distributions[right] - distributions[left]
        largest = difference.abs().sort_index().idxmax()
        tvd_rows.append({"reference": left, "comparison": right,
            "total_variation_distance": _tvd(distributions, left, right),
            "largest_shift_retailer": largest,
            "comparison_minus_reference_fraction": float(difference.loc[largest])})
        if float(difference.abs().max()) > 1e-12:
            observations.append({"kind": "observed_retailer_mix_difference", **tvd_rows[-1]})

    labels = sorted(group_index.Segment.unique())
    development_sizes = group_index.groupby("group_id").size()
    group_rows, class_rows, partition_summaries = [], [], []
    for name in sorted(grouped_populations):
        rows = grouped_populations[name]
        sizes = rows.groupby("group_id").size()
        size_counts = sizes.value_counts().sort_index()
        for size, count in size_counts.items():
            group_rows.append({"population": name, "group_size": int(size), "size_bucket": _bucket(size),
                "groups": int(count), "records": int(size * count),
                "fraction_of_partition_groups": count / len(sizes),
                "fraction_of_partition_records": size * count / len(rows)})
        buckets = {bucket: int(sizes.loc[sizes.map(_bucket).eq(bucket)].sum()) for bucket in SIZE_BUCKETS}
        whole_size_buckets = rows.group_id.map(development_sizes).map(_bucket).value_counts()
        summary = {"population": name, "records": len(rows), "unique_frozen_groups": len(sizes),
            "fraction_of_development_groups_present": len(sizes) / len(development_sizes),
            "within_partition_singleton_groups": int(sizes.eq(1).sum()),
            "development_singleton_group_records": int(rows.group_id.map(development_sizes).eq(1).sum()),
            "development_singleton_record_fraction": float(rows.group_id.map(development_sizes).eq(1).mean())
                if len(rows) else None,
            "mean_within_partition_group_size": float(sizes.mean()) if len(sizes) else None,
            "p95_within_partition_group_size": float(sizes.quantile(.95)) if len(sizes) else None,
            "max_within_partition_group_size": int(sizes.max()) if len(sizes) else 0,
            "size_bucket_records": buckets,
            "development_size_bucket_records": {bucket: int(whole_size_buckets.get(bucket, 0))
                                                for bucket in SIZE_BUCKETS}}
        partition_summaries.append(summary)
        counts = rows.Segment.value_counts()
        for label in labels:
            class_rows.append({"population": name, "Segment": label, "records": int(counts.get(label, 0)),
                "population_records": len(rows), "fraction": counts.get(label, 0) / len(rows) if len(rows) else None})
    by_name = {row["population"]: row for row in partition_summaries}
    group_size_comparisons = [("development_pool", "random_train"),
                             ("development_pool", "random_validation"),
                             ("development_pool", "group_aware_train"),
                             ("development_pool", "group_aware_validation")]
    if previous_assignments is not None:
        group_size_comparisons.append(("development_pool", "previous_development_pool"))
    if frozen_role_assignments is not None:
        group_size_comparisons.append(("development_pool", "original_frozen_validation"))
    group_size_tvd = []
    for left, right in group_size_comparisons:
        left_count, right_count = by_name[left]["records"], by_name[right]["records"]
        value = .5 * sum(abs(by_name[left]["development_size_bucket_records"][bucket] / left_count
                            - by_name[right]["development_size_bucket_records"][bucket] / right_count)
                         for bucket in SIZE_BUCKETS) if left_count and right_count else None
        group_size_tvd.append({"reference": left, "comparison": right,
                              "whole_development_size_bucket_record_tvd": value})
        if value is not None and value > 1e-12:
            observations.append({"kind": "observed_group_size_mix_difference", **group_size_tvd[-1]})
    random_groups = by_name["random_train"]["unique_frozen_groups"]
    aware_groups = by_name["group_aware_train"]["unique_frozen_groups"]
    if random_groups != aware_groups:
        observations.append({"kind": "observed_training_group_coverage_difference",
            "random_groups": random_groups, "group_aware_groups": aware_groups,
            "random_relative_difference": random_groups / aware_groups - 1 if aware_groups else None})
    reason_counts = exclusions.reason.value_counts().sort_index()
    exclusion_labels = frame.set_index("record_id").Segment.reindex(exclusions.record_id)
    exclusion_details = []
    for reason, count in reason_counts.items():
        matching = exclusions.reason.eq(reason).to_numpy()
        missing_count = int(_missing(exclusion_labels.iloc[matching]).sum())
        exclusion_details.append({"reason": reason, "records": int(count),
                                  "missing_Segment_records": missing_count,
                                  "labeled_records": int(count) - missing_count})
    summary = {"version": VERSION, "source_records": len(frame), "labeled_source_records": len(labeled),
        "missing_Segment_source_records": len(frame) - len(labeled),
        "development_records": len(development), "excluded_records": len(exclusions),
        "development_fraction_of_labeled_source": len(development) / len(labeled) if len(labeled) else None,
        "complete_source_coverage_verified": True,
        "expected_frozen_v3_non_test_universe_verified": expected_development_ids is not None,
        "both_strategies_share_development_and_exclusions": True,
        "final_test_predictions_or_metrics_consumed": False,
        "protected_source_use": "aggregate feature mix and label availability only",
        "production_reference_records": len(reference_features) if reference_features is not None else None,
        "previous_development_records": len(previous_groups["previous_development_pool"])
            if previous_assignments is not None else None,
        "excluded_by_reason": exclusion_details, "retailer_total_variation": tvd_rows,
        "group_size_total_variation": group_size_tvd,
        "partition_group_summary": partition_summaries, "observed_composition_differences": observations}
    return {"summary": summary, "retailer_distribution": pd.DataFrame(retailer_rows),
            "feature_missingness": pd.DataFrame(missing_rows),
            "partition_group_sizes": pd.DataFrame(group_rows),
            "class_distribution": pd.DataFrame(class_rows)}
