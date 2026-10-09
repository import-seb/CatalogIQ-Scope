"""Supplemental development-population diagnostics for frozen Segment models.

These slices describe associations, not causal leakage effects. Only development
features are normalized, and predictions must exactly cover validation records.
The three frozen model experiment modules are imported, never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .segment_experiment import _jsonable, validate_development_assignments, verify_protocol, write_json
from .segment_validation_analysis import annotate_training_neighbors, evaluate_predictions
from .splitting import record_ids, text

VERSION = "segment-population-diagnostics-v1"
METHODS = ("random", "group_aware")
FEATURES = ("ProductBrand", "Retailer", "ProductDescription", "ProductContents")
FIXED_VALUES = {
    "development_group_size_bucket": ("1", "2", "3-5", "6-10", "11+"),
    "product_brand_train_seen": ("seen", "unseen", "missing"),
    "description_presence": ("present", "missing"),
    "contents_presence": ("present", "missing"),
    "group_label_homogeneity": ("single_Segment", "mixed_Segment"),
}
DIMENSIONS = (*FIXED_VALUES, "retailer")
REFERENCE_DIMENSIONS = ("random_has_rule_group_train_neighbor",
                        "random_has_independent_near_train_neighbor")


def _read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _size_bucket(size):
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return "11+"


def _missing(values):
    return values.fillna("").astype(str).str.strip().str.casefold().isin(("", "null"))


def population_annotations(assignments, exclusions, features):
    """Annotate shared eligible groups and strategy-relative training coverage.

    Input features may include other source rows. Filter them out before calling
    normalization or inspecting text. Segment comes only from guarded frozen
    development assignments, never from raw source labels.
    """
    validate_development_assignments(assignments, exclusions)
    required = {"record_id", *FEATURES}
    if not required.issubset(features):
        raise ValueError(f"features require {sorted(required)}")
    if features.record_id.duplicated().any():
        raise ValueError("feature record IDs must be unique")
    wanted = set(assignments.record_id)
    selected = features.loc[features.record_id.isin(wanted), ["record_id", *FEATURES]].copy()
    if set(selected.record_id) != wanted:
        raise ValueError("features do not cover every eligible development record")
    if set(selected.record_id) & set(exclusions.record_id):
        raise ValueError("excluded or final-test feature reached diagnostics")
    selected = selected.set_index("record_id")
    selected["normalized_brand"] = selected.ProductBrand.map(text)
    selected["retailer"] = selected.Retailer.map(text).replace("", "<MISSING>")
    selected["description_presence"] = _missing(selected.ProductDescription).map(
        {False: "present", True: "missing"})
    selected["contents_presence"] = _missing(selected.ProductContents).map(
        {False: "present", True: "missing"})
    base = assignments.loc[assignments.strategy.eq("group_aware")].copy()
    sizes = base.groupby("group_id").size()
    labels = base.groupby("group_id").Segment.nunique()
    result = assignments.copy()
    result["development_group_size"] = result.group_id.map(sizes).astype(int)
    result["development_group_size_bucket"] = result.development_group_size.map(_size_bucket)
    result["group_distinct_Segment_labels"] = result.group_id.map(labels).astype(int)
    result["group_label_homogeneity"] = result.group_distinct_Segment_labels.gt(1).map(
        {False: "single_Segment", True: "mixed_Segment"})
    for column in ("normalized_brand", "retailer", "description_presence", "contents_presence"):
        result[column] = result.record_id.map(selected[column])
    result["product_brand_train_seen"] = "unseen"
    for strategy in METHODS:
        mask = result.strategy.eq(strategy)
        known = set(result.loc[mask & result.split.eq("train"), "normalized_brand"]) - {""}
        result.loc[mask & result.normalized_brand.isin(known), "product_brand_train_seen"] = "seen"
    result.loc[result.normalized_brand.eq(""), "product_brand_train_seen"] = "missing"
    return result


def validate_validation_predictions(predictions, assignments, exclusions, strategy):
    """Reject train, protected, unknown, duplicate, or mislabeled predictions."""
    if strategy not in METHODS:
        raise ValueError("unknown strategy")
    required = {"record_id", "true_label", "predicted_label"}
    if not required.issubset(predictions):
        raise ValueError("predictions require record_id, true_label, predicted_label")
    if predictions.record_id.duplicated().any():
        raise ValueError("duplicate validation prediction")
    current = assignments.loc[assignments.strategy.eq(strategy)]
    wanted = set(current.loc[current.split.eq("validation"), "record_id"])
    actual = set(predictions.record_id)
    if actual & set(exclusions.record_id):
        raise ValueError("excluded or final-test prediction detected")
    if actual != wanted:
        raise ValueError("predictions must cover exactly the strategy validation IDs")
    truth = current.set_index("record_id").Segment
    if not predictions.true_label.astype(str).equals(predictions.record_id.map(truth).astype(str)):
        raise ValueError("prediction truth differs from frozen development labels")
    return True


def _slices(frame, values):
    yield "all", "all", frame
    for dimension in DIMENSIONS:
        for value in values[dimension]:
            yield dimension, value, frame.loc[frame[dimension].eq(value)]


def _composition(frame, labels, metadata):
    counts = frame.Segment.value_counts()
    size = len(frame)
    return [{**metadata, "label": label, "true_support": int(counts.get(label, 0)),
             "class_fraction": float(counts.get(label, 0) / size) if size else None,
             "cohort_records": size, "cohort_groups": int(frame.group_id.nunique())}
            for label in labels]


def summarize_population(assignments, exclusions, features, predictions, label_names, near_pairs=None):
    """Return composition, validation metrics, per-class metrics, and annotations."""
    labels = list(label_names)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("label_names must be nonempty and unique")
    validate_development_assignments(assignments, exclusions)
    if set(predictions) != set(METHODS):
        raise ValueError("both frozen strategy prediction files are required")
    if not assignments.Segment.isin(labels).all():
        raise ValueError("development labels are outside declared labels")
    for strategy in METHODS:
        validate_validation_predictions(predictions[strategy], assignments, exclusions, strategy)
    annotated = population_annotations(assignments, exclusions, features)
    values = {**FIXED_VALUES, "retailer": tuple(sorted(annotated.retailer.unique()))}
    composition, metrics, per_class, partition_stats = [], [], [], []
    common = set.intersection(*(set(predictions[strategy].record_id) for strategy in METHODS))
    random_train = set(annotated.loc[annotated.strategy.eq("random") & annotated.split.eq("train"), "record_id"])
    frozen_groups = annotated.loc[annotated.strategy.eq("group_aware"), ["record_id", "group_id"]]
    if near_pairs is None:
        near_pairs = pd.DataFrame(columns=["left_record_id", "right_record_id"])
        near_available = False
    else:
        near_available = True
        wanted = set(assignments.record_id)
        near_pairs = near_pairs.loc[near_pairs.left_record_id.isin(wanted)
                                    & near_pairs.right_record_id.isin(wanted)].copy()
    reference = annotate_training_neighbors(pd.DataFrame({"record_id": sorted(common)}),
        random_train, frozen_groups, near_pairs)
    reference[REFERENCE_DIMENSIONS[0]] = reference.has_rule_group_train_neighbor.map({True: "yes", False: "no"})
    reference[REFERENCE_DIMENSIONS[1]] = (reference.has_independent_near_train_neighbor.map(
        {True: "yes", False: "no"}) if near_available else "unavailable")
    reference = reference[["record_id", "group_id", *REFERENCE_DIMENSIONS,
                           "rule_group_train_neighbors", "independent_near_train_neighbors"]]
    reference_index = reference.set_index("record_id")
    development_groups = int(frozen_groups.group_id.nunique())
    for strategy in METHODS:
        current = annotated.loc[annotated.strategy.eq(strategy)].copy()
        for population, frame in (("development_pool", current),
                                  ("strategy_train", current.loc[current.split.eq("train")]),
                                  ("strategy_validation", current.loc[current.split.eq("validation")])):
            for dimension, value, cohort in _slices(frame, values):
                metadata = {"strategy": strategy, "population": population,
                            "dimension": dimension, "value": value}
                composition.extend(_composition(cohort, labels, metadata))
            sizes = frame.groupby("group_id").size()
            partition_stats.append({"strategy": strategy, "population": population,
                "records": len(frame), "unique_frozen_groups": len(sizes),
                "fraction_of_development_groups_present": len(sizes) / development_groups,
                "within_partition_singleton_groups": int(sizes.eq(1).sum()),
                "singleton_development_group_records": int(frame.development_group_size.eq(1).sum()),
                "mean_within_partition_group_size": float(sizes.mean()) if len(sizes) else None,
                "p95_within_partition_group_size": float(sizes.quantile(.95)) if len(sizes) else None,
                "max_within_partition_group_size": int(sizes.max()) if len(sizes) else 0,
                **{f"development_size_{bucket}_records": int(frame.development_group_size_bucket.eq(bucket).sum())
                   for bucket in FIXED_VALUES["development_group_size_bucket"]}})
        scored = predictions[strategy].merge(current.drop(columns=["strategy"]),
            on="record_id", how="left", validate="one_to_one", sort=False)
        for column in REFERENCE_DIMENSIONS:
            scored[column] = scored.record_id.map(reference_index[column])
        for population, frame in (("strategy_validation", scored),
                                  ("shared_validation_intersection", scored.loc[scored.record_id.isin(common)])):
            slices = list(_slices(frame, values))
            if population == "shared_validation_intersection":
                for column in REFERENCE_DIMENSIONS:
                    for value in ("yes", "no") if near_available or column == REFERENCE_DIMENSIONS[0] else ("unavailable",):
                        slices.append((column, value, frame.loc[frame[column].eq(value)]))
            for dimension, value, cohort in slices:
                metadata = {"strategy": strategy, "population": population,
                            "dimension": dimension, "value": value}
                evaluated = evaluate_predictions(cohort, labels)
                scalar = dict(evaluated["metrics"])
                scalar["unsupported_true_labels"] = json.dumps(scalar["unsupported_true_labels"])
                metrics.append({**metadata, "groups": int(cohort.group_id.nunique()), **scalar})
                for row in evaluated["per_class"].to_dict(orient="records"):
                    per_class.append({**metadata, "cohort_records": len(cohort), **row,
                        "class_fraction": float(row["true_support"] / len(cohort)) if len(cohort) else None})
    return {"population_records": annotated, "class_composition": pd.DataFrame(composition),
            "cohort_metrics": pd.DataFrame(metrics), "cohort_per_class": pd.DataFrame(per_class),
            "partition_group_summary": pd.DataFrame(partition_stats),
            "shared_reference_cohort_records": reference,
            "shared_validation_records": len(common), "slice_values": values,
            "independent_near_audit_available": near_available}


def freeze_implementation(output_dir):
    """Pin this supplemental implementation before inspecting slice outcomes."""
    output_dir = Path(output_dir).resolve()
    plan_path = output_dir / "analysis_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["version"] != VERSION:
        raise ValueError("unexpected supplemental plan version")
    destination = output_dir / "analysis_implementation_freeze.json"
    if destination.exists():
        raise ValueError("supplemental implementation already frozen")
    source = Path(__file__).resolve()
    wrapper = source.parents[2] / "scripts/analyze_segment_population.py"
    pins = {str(p): sha256(p) for p in (source, wrapper, plan_path)}
    write_json(destination, {"version": VERSION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_supplemental_slice_result_inspection", "sha256": pins,
        "master_model_protocol_unchanged": True,
        "aggregate_epoch_log_may_have_been_seen_before_supplemental_plan": True})
    return pins


def run_population_diagnostics(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    plan_path = output_dir / "analysis_plan.json"
    freeze_path = output_dir / "analysis_implementation_freeze.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (plan["version"] != VERSION or freeze["version"] != VERSION
            or Path(plan["model_run_dir"]).resolve() != run_dir):
        raise ValueError("supplemental plan does not match frozen experiment")
    pins = {**plan["reference_sha256"], **freeze["sha256"]}
    for path, digest in pins.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"supplemental planned input or implementation changed: {path}")
    if (output_dir / "summary.json").exists():
        raise ValueError("supplemental results already exist")
    protocol = verify_protocol(run_dir)
    assignments = _read_csv(run_dir / "development_assignments.csv")
    exclusions = _read_csv(run_dir / "excluded_records.csv")
    validate_development_assignments(assignments, exclusions)
    predictions = {}
    for strategy in METHODS:
        path = run_dir / strategy / "predictions.csv"
        completed_path = path.parent / "completed.json"
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed["predictions_sha256"] != sha256(path):
            raise ValueError("completed validation predictions changed")
        pins[str(path)] = sha256(path)
        pins[str(completed_path)] = sha256(completed_path)
        predictions[strategy] = pd.read_csv(path, keep_default_na=False)
        validate_validation_predictions(predictions[strategy], assignments, exclusions, strategy)
    input_path = Path(protocol["input_path"])
    # Raw supplied labels are not read. Identity calculation is the only operation
    # before strict development filtering; text normalization occurs afterward.
    raw = pd.read_csv(input_path, usecols=[*FEATURES, *PROVENANCE], dtype=str, keep_default_na=False)
    raw["record_id"] = record_ids(raw)
    features = raw.loc[raw.record_id.isin(set(assignments.record_id))].copy()
    del raw
    pins[str(input_path)] = sha256(input_path)
    split_dir = Path(protocol["split_dir"])
    for name in ("rule_v2_assignments.csv", "rule_assignments.csv", "tfidf_assignments.csv"):
        path = split_dir / name
        pins[str(path)] = sha256(path)
    near_path = split_dir / "independent_near_duplicate_pairs.csv"
    pins[str(near_path)] = sha256(near_path)
    near_pairs = pd.read_csv(near_path, usecols=["left_record_id", "right_record_id"], dtype=str, keep_default_na=False)
    tables = summarize_population(assignments, exclusions, features, predictions, protocol["label_names"], near_pairs)
    for name in ("population_records", "class_composition", "cohort_metrics", "cohort_per_class",
                 "partition_group_summary", "shared_reference_cohort_records"):
        tables[name].to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
    for path, digest in pins.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"supplemental input changed during analysis: {path}")
    verify_protocol(run_dir)
    summary = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan": str(plan_path), "implementation_freeze": str(freeze_path),
        "model_run_dir": str(run_dir), "input_sha256": pins, "label_names": protocol["label_names"],
        "eligible_development_records": int(assignments.record_id.nunique()),
        "largest_development_group": int(tables["population_records"].development_group_size.max()),
        "shared_validation_records": tables["shared_validation_records"],
        "slice_values": tables["slice_values"], "raw_source_label_columns_read": [],
        "partition_group_summary": tables["partition_group_summary"],
        "common_row_reference_cohorts": {"reference_strategy": "random",
            "dimensions": REFERENCE_DIMENSIONS, "same_record_masks_applied_to_both_models": True,
            "independent_near_audit_available": tables["independent_near_audit_available"],
            "candidate_pool_limitation": "No audited near neighbor means absent from the fixed candidate pool, not proven unrelated."},
        "checks": {"validation_prediction_universe_only": True, "excluded_prediction_records": 0,
            "features_filtered_before_normalization": True, "frozen_model_protocol_unchanged": True,
            "three_protected_test_assignment_inputs_pinned": True},
        "interpretation": "Descriptive cohort associations, not causal leakage inflation. Class composition and per-class support accompany every cohort. Training family diversity and validation family novelty differ despite matched row and class counts.",
        "limitations": plan["limitations"],
        "output_sha256": {path.name: sha256(path) for path in output_dir.glob("*.csv")}}
    write_json(output_dir / "summary.json", summary)
    return summary


def verify_existing_population_diagnostics(run_dir, output_dir):
    """Verify and display saved results, allowing a pinned presentation fix only.

    Existing analytical artifacts are never rewritten. The original source and
    freeze remain saved so the documented stdout-only amendment is reviewable.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["version"] != VERSION or Path(summary["model_run_dir"]).resolve() != run_dir:
        raise ValueError("saved supplemental results belong to another run")
    amendment_path = output_dir / "presentation_fix_amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.exists() else None
    if amendment:
        if (amendment.get("version") != VERSION or amendment.get("scope") != "presentation_only"
                or amendment["original_summary_sha256"] != sha256(summary_path)
                or amendment["original_implementation_freeze_sha256"]
                   != sha256(output_dir / "analysis_implementation_freeze.json")
                or amendment["original_source_sha256"]
                   != sha256(output_dir / "segment_population_diagnostics_original_frozen.py")
                or amendment["amended_source_sha256"] != sha256(Path(__file__).resolve())):
            raise ValueError("invalid presentation-only amendment chain")
    for path, digest in summary["input_sha256"].items():
        actual = sha256(Path(path))
        amended_source = (amendment and Path(path).resolve() == Path(__file__).resolve()
            and digest == amendment["original_source_sha256"]
            and actual == amendment["amended_source_sha256"])
        if actual != digest and not amended_source:
            raise ValueError(f"saved supplemental input changed: {path}")
    for name, digest in summary["output_sha256"].items():
        if sha256(output_dir / name) != digest:
            raise ValueError(f"saved supplemental analytical output changed: {name}")
    verify_protocol(run_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze:
        print(json.dumps(freeze_implementation(args.output_dir), indent=2))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required for analysis")
    if (args.output_dir / "summary.json").exists():
        result = verify_existing_population_diagnostics(args.run_dir, args.output_dir)
    else:
        result = run_population_diagnostics(args.run_dir, args.output_dir)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
