"""A frozen, validation-only comparison of Segment evaluation strategies.

The current three final-test snapshots are embargoed before tokenization. The
grouping algorithms and allocation engine are neither called nor changed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .features import sha256
from .splitting import record_ids

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "data/processed/split_rule_v3_20261008"
DEFAULT_INPUT = ROOT / "data/processed/training_cleaned.csv"
METHODS = ("random", "group_aware")
VERSION = "segment-validation-baseline-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_development_assignments(frame, group_assignment, test_assignments, seed=42):
    """Protect the union of current tests and its whole-group closure.

    The random control exactly matches frozen group-aware class counts. Missing
    labels never become a supervised class. Sorting immutable IDs precedes RNG.
    Returns (eligible feature rows, long assignments, embargo reasons, counts).
    """
    if "record_id" not in frame:
        raise ValueError("feature rows need immutable record_id")
    if frame.record_id.duplicated().any() or group_assignment.record_id.duplicated().any():
        raise ValueError("record identities must be unique")
    universe = set(frame.record_id)
    if set(group_assignment.record_id) != universe:
        raise ValueError("frozen grouping must cover the same input records")
    if not group_assignment.split.isin(["train", "validation", "test"]).all():
        raise ValueError("unknown frozen role")
    if group_assignment.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("frozen rule group crosses a boundary")
    direct = set()
    for assignment in test_assignments:
        if assignment.record_id.duplicated().any() or set(assignment.record_id) != universe:
            raise ValueError("protected assignment must cover the shared universe")
        direct.update(assignment.loc[assignment.split.eq("test"), "record_id"])
    if not direct:
        raise ValueError("at least one protected final-test record is required")
    protected_groups = set(group_assignment.loc[group_assignment.record_id.isin(direct), "group_id"])
    embargo_ids = set(group_assignment.loc[group_assignment.group_id.isin(protected_groups), "record_id"])
    indexed = group_assignment.set_index("record_id")
    ordered = frame.sort_values("record_id", kind="stable").reset_index(drop=True).copy()
    roles = ordered.record_id.map(indexed.split)
    groups = ordered.record_id.map(indexed.group_id)
    labels = ordered.Segment.astype(str)
    labeled = ~labels.str.strip().str.casefold().isin(["", "null", "<missing>"])
    eligible = ~ordered.record_id.isin(embargo_ids) & labeled
    development = ordered.loc[eligible].copy().reset_index(drop=True)
    group_rows = pd.DataFrame({"record_id": ordered.loc[eligible, "record_id"].to_numpy(),
                               "group_id": groups.loc[eligible].to_numpy(),
                               "Segment": labels.loc[eligible].to_numpy(),
                               "split": roles.loc[eligible].to_numpy()})
    if not group_rows.split.isin(["train", "validation"]).all():
        raise ValueError("a protected test record reached the development pool")
    if set(group_rows.groupby("split").size().index) != {"train", "validation"}:
        raise ValueError("both development partitions are required")
    random_rows = group_rows.copy()
    rng = np.random.default_rng(seed)
    for label, indices in random_rows.groupby("Segment", sort=True).groups.items():
        indices = np.array(sorted(indices), dtype=int)
        permutation = rng.permutation(indices)
        n_validation = int((group_rows.loc[indices, "split"] == "validation").sum())
        if n_validation == 0 or n_validation == len(indices):
            raise ValueError(f"class {label!r} needs training and validation support")
        random_rows.loc[indices, "split"] = "train"
        random_rows.loc[permutation[:n_validation], "split"] = "validation"
    group_rows["strategy"] = "group_aware"
    random_rows["strategy"] = "random"
    assignments = pd.concat([random_rows, group_rows], ignore_index=True)
    exclusions = ordered.loc[~eligible, ["record_id"]].copy()
    exclusions["reason"] = np.where(exclusions.record_id.isin(direct), "protected_final_test",
        np.where(exclusions.record_id.isin(embargo_ids), "protected_rule_group_sibling", "missing_Segment"))
    metadata = {"input_records": len(frame), "direct_test_union_records": len(direct),
                "group_closed_test_embargo_records": len(embargo_ids),
                "missing_Segment_in_unprotected_pool": int((~labeled & ~ordered.record_id.isin(embargo_ids)).sum()),
                "eligible_labeled_development_records": len(development),
                "train_records_per_strategy": int(group_rows.split.eq("train").sum()),
                "validation_records_per_strategy": int(group_rows.split.eq("validation").sum()),
                "group_aware_development_train_fraction": float(group_rows.split.eq("train").mean()),
                "group_aware_groups": int(group_rows.group_id.nunique()),
                "largest_eligible_group": int(group_rows.groupby("group_id").size().max()),
                "random_rule_groups_crossing_train_validation": int(random_rows.groupby("group_id").split.nunique().gt(1).sum()),
                "protected_snapshots": len(test_assignments)}
    validate_development_assignments(assignments, exclusions)
    return development, assignments, exclusions, metadata


def validate_development_assignments(assignments, exclusions):
    if set(assignments.strategy) != set(METHODS):
        raise ValueError("both strategies must be present")
    if not assignments.split.isin(["train", "validation"]).all():
        raise ValueError("model experiment permits training and validation only")
    if set(assignments.record_id) & set(exclusions.record_id):
        raise ValueError("excluded record reached model experiment")
    parts = {method: assignments.loc[assignments.strategy.eq(method)].copy() for method in METHODS}
    if any(part.record_id.duplicated().any() for part in parts.values()):
        raise ValueError("each strategy must assign a record exactly once")
    if set(parts["random"].record_id) != set(parts["group_aware"].record_id):
        raise ValueError("strategies must use exactly the same records")
    left = parts["random"].set_index("record_id").sort_index()
    right = parts["group_aware"].set_index("record_id").sort_index()
    if not left[["group_id", "Segment"]].equals(right[["group_id", "Segment"]]):
        raise ValueError("strategies disagree on identities or labels")
    if not left.groupby(["Segment", "split"]).size().equals(right.groupby(["Segment", "split"]).size()):
        raise ValueError("strategies must match per-class training and validation counts")
    if right.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("group-aware development partition has a crossing")
    return True


def _protocol_sources():
    return [Path(__file__), ROOT / "scripts/compare_segment_validation.py",
            ROOT / "requirements-training.txt",
            Path(__file__).with_name("segment_transformer.py"),
            Path(__file__).with_name("segment_validation_analysis.py")]


def prepare_experiment(output_dir, model_config, input_path=DEFAULT_INPUT,
                       split_dir=DEFAULT_SPLITS, bootstrap_samples=500):
    from .rule_refinement_experiment import verify_rule_refinement_freeze
    split_dir, output_dir, input_path = map(lambda p: Path(p).resolve(), (split_dir, output_dir, input_path))
    verify_rule_refinement_freeze(split_dir)
    if output_dir.exists():
        raise ValueError("Use a new output directory; a frozen protocol cannot be overwritten")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    manifest = json.loads((split_dir / "input_manifest.json").read_text(encoding="utf-8"))
    if sha256(input_path) != manifest["sha256"]:
        raise ValueError("input differs from frozen grouping input")
    frame = read_csv(input_path)
    frame["record_id"] = record_ids(frame)
    snapshot_names = ("rule_assignments.csv", "rule_v2_assignments.csv", "tfidf_assignments.csv")
    snapshots = [read_csv(split_dir / name) for name in snapshot_names]
    development, assignments, exclusions, counts = make_development_assignments(
        frame, snapshots[0], snapshots, model_config.seed)
    # Rebuild the seeded control to verify assignments, before any model outcome.
    _, repeated, _, repeated_counts = make_development_assignments(
        frame.iloc[::-1], snapshots[0].iloc[::-1], snapshots, model_config.seed)
    if not assignments.equals(repeated) or repeated_counts != counts:
        raise AssertionError("development assignments are not seed/row-order reproducible")
    output_dir.mkdir(parents=True)
    assignments.to_csv(output_dir / "development_assignments.csv", index=False, lineterminator="\n")
    exclusions.to_csv(output_dir / "excluded_records.csv", index=False, lineterminator="\n")
    class_counts = assignments.groupby(["strategy", "Segment", "split"]).size().rename("records").reset_index()
    class_counts.to_csv(output_dir / "class_counts.csv", index=False)
    sources = [input_path, split_dir / "input_manifest.json", split_dir / "rule_refinement_freeze.json",
               *[split_dir / name for name in snapshot_names],
               split_dir / "independent_near_duplicate_pairs.csv",
               split_dir / "exact_payload_duplicate_pairs.csv", *_protocol_sources()]
    # Pin cached encoder/tokenizer bytes as well as their public revision.
    from huggingface_hub import snapshot_download
    model_snapshot = Path(snapshot_download(model_config.model_name,
        revision=model_config.revision, local_files_only=model_config.local_files_only))
    sources.extend(path for path in model_snapshot.rglob("*") if path.is_file())
    artifacts = [output_dir / name for name in ("development_assignments.csv", "excluded_records.csv", "class_counts.csv")]
    protocol = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_path": str(input_path), "split_dir": str(split_dir), "model_config": asdict(model_config),
                "target": "Segment", "label_names": sorted(assignments.loc[
                    assignments.strategy.eq("group_aware") & assignments.split.eq("train"), "Segment"].unique().tolist()),
                "primary_metric": "macro_f1", "checkpoint_policy": "fixed final epoch; no model/threshold search",
                "pretrained_snapshot_path": str(model_snapshot),
                "features": ["ProductName", "ProductBrand", "ProductDescription", "ProductContents"],
                "counts": counts, "bootstrap_samples": bootstrap_samples,
                "test_protection": "union of current rule_v2/rule_v3/tfidf_v2 tests, closed over frozen rule_v3 groups",
                "validation_policy": "original frozen v3 roles; random matches exact per-class role counts",
                "split_engine_and_grouping_unchanged": True, "test_predictions_permitted": False,
                "interpretation": "own-validation difference is a strategy comparison, not causal leakage inflation",
                "limitations": ["single initialization/training seed", "group closure changes development population",
                    "not all 291 original crossings are in the eligible development population",
                    "embargoed broad groups cannot be evaluated here", "production family mix is not established"],
                "protected_inputs_and_sources": {str(path): sha256(path) for path in sources},
                "prepared_artifacts": {str(path): sha256(path) for path in artifacts}}
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "protocol_freeze.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
                                                     "frozen_before_training": True})
    return protocol


def verify_protocol(output_dir):
    output_dir = Path(output_dir).resolve()
    seal = json.loads((output_dir / "protocol_freeze.json").read_text(encoding="utf-8"))
    if not seal.get("frozen_before_training") or sha256(output_dir / "protocol.json") != seal["protocol_sha256"]:
        raise ValueError("experiment protocol has changed")
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    for path, expected in {**protocol["protected_inputs_and_sources"], **protocol["prepared_artifacts"]}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"frozen experiment input or implementation changed: {path}")
    assignments = read_csv(output_dir / "development_assignments.csv")
    exclusions = read_csv(output_dir / "excluded_records.csv")
    validate_development_assignments(assignments, exclusions)
    return protocol


def _load_development(protocol, assignments):
    frame = read_csv(protocol["input_path"])
    frame["record_id"] = record_ids(frame)
    wanted = sorted(set(assignments.record_id))
    frame = frame.set_index("record_id").loc[wanted].reset_index()
    labels = assignments.loc[assignments.strategy.eq("group_aware")].set_index("record_id").Segment
    if frame.Segment.tolist() != frame.record_id.map(labels).tolist():
        raise ValueError("development labels changed")
    return frame


def run_training(output_dir, progress=print):
    from .segment_transformer import ModelConfig, tokenize_corpus, train_transformer
    output_dir = Path(output_dir).resolve()
    protocol = verify_protocol(output_dir)
    config = ModelConfig.from_dict(protocol["model_config"])
    assignments = read_csv(output_dir / "development_assignments.csv")
    development = _load_development(protocol, assignments)
    progress(f"Tokenizing only {len(development):,} eligible development records")
    corpus = tokenize_corpus(development, protocol["label_names"], config)
    positions = {record: i for i, record in enumerate(development.record_id)}
    initialization_hashes = []
    for strategy in METHODS:
        destination = output_dir / strategy
        if (destination / "completed.json").exists():
            completed = json.loads((destination / "completed.json").read_text(encoding="utf-8"))
            if completed["predictions_sha256"] != sha256(destination / "predictions.csv"):
                raise ValueError("saved model predictions changed")
            initialization_hashes.append(completed["initial_state_sha256"])
            progress(f"Using completed {strategy} run without retraining")
            continue
        if destination.exists():
            raise ValueError(f"Incomplete {strategy} model directory; preserve it and use a new experiment directory")
        assignment = assignments.loc[assignments.strategy.eq(strategy)]
        train_indices = [positions[key] for key in assignment.loc[assignment.split.eq("train"), "record_id"]]
        validation_indices = [positions[key] for key in assignment.loc[assignment.split.eq("validation"), "record_id"]]
        progress(f"Training {strategy}: {len(train_indices):,} train, {len(validation_indices):,} validation")
        predictions, manifest = train_transformer(corpus, train_indices, validation_indices,
                                                   protocol["label_names"], config, destination, progress=progress)
        if set(predictions.record_id) != set(assignment.loc[assignment.split.eq("validation"), "record_id"]):
            raise ValueError("model produced predictions outside its validation partition")
        initialization = manifest["initial_state_sha256"]
        initialization_hashes.append(initialization)
        write_json(destination / "completed.json", {"initial_state_sha256": initialization,
                   "predictions_sha256": sha256(destination / "predictions.csv"), "manifest": manifest})
    if len(set(initialization_hashes)) != 1:
        raise AssertionError("comparison models did not start from the identical initialization")
    verify_protocol(output_dir)
    return analyze_experiment(output_dir, progress)


def analyze_experiment(output_dir, progress=print):
    from .segment_validation_analysis import (evaluate_predictions, annotate_training_neighbors,
        summarize_cohorts, bootstrap_predictions, compare_paired_predictions, affected_row_bound,
        majority_baseline)
    output_dir = Path(output_dir).resolve()
    protocol = verify_protocol(output_dir)
    assignments = read_csv(output_dir / "development_assignments.csv")
    exclusions = read_csv(output_dir / "excluded_records.csv")
    split_dir = Path(protocol["split_dir"])
    frozen_groups = read_csv(split_dir / "rule_assignments.csv")[["record_id", "group_id"]]
    near = read_csv(split_dir / "independent_near_duplicate_pairs.csv")
    exact = read_csv(split_dir / "exact_payload_duplicate_pairs.csv")
    eligible_ids = set(assignments.record_id)
    near = near.loc[near.left_record_id.isin(eligible_ids) & near.right_record_id.isin(eligible_ids)].copy()
    exact = exact.loc[exact.left_record_id.isin(eligible_ids) & exact.right_record_id.isin(eligible_ids)].copy()
    labels = protocol["label_names"]
    rows, predictions, bootstrap_summaries, initialization_hashes = [], {}, {}, []
    for strategy in METHODS:
        assignment = assignments.loc[assignments.strategy.eq(strategy)]
        saved = pd.read_csv(output_dir / strategy / "predictions.csv", keep_default_na=False)
        completed = json.loads((output_dir / strategy / "completed.json").read_text(encoding="utf-8"))
        if completed["predictions_sha256"] != sha256(output_dir / strategy / "predictions.csv"):
            raise ValueError("completed predictions changed")
        initialization_hashes.append(completed["initial_state_sha256"])
        if saved.record_id.duplicated().any() or set(saved.record_id) != set(assignment.loc[assignment.split.eq("validation"), "record_id"]):
            raise ValueError("prediction universe differs from validation IDs")
        if set(saved.record_id) & set(exclusions.record_id):
            raise ValueError("excluded or test prediction detected")
        true_labels = assignment.set_index("record_id").Segment
        if saved.true_label.tolist() != saved.record_id.map(true_labels).tolist():
            raise ValueError("prediction truth differs from frozen Segment labels")
        trains = set(assignment.loc[assignment.split.eq("train"), "record_id"])
        annotated = annotate_training_neighbors(saved, trains, frozen_groups, near, exact)
        annotated.to_csv(output_dir / strategy / "validation_neighbor_predictions.csv", index=False)
        predictions[strategy] = annotated
        evaluation = evaluate_predictions(annotated, labels)
        row = {"strategy": strategy, **evaluation["metrics"]}
        majority = majority_baseline(assignment.loc[assignment.split.eq("train"), "Segment"],
                                     saved.true_label, labels)
        row["majority_label"] = majority["majority_label"]
        row["majority_accuracy"] = majority["accuracy"]
        row["majority_macro_f1"] = majority["macro_f1"]
        write_json(output_dir / strategy / "majority_baseline.json", majority)
        rows.append(row)
        write_json(output_dir / strategy / "evaluation_metrics.json", row)
        for name in ("per_class", "confusion", "calibration"):
            evaluation[name].to_csv(output_dir / strategy / f"{name}.csv", index=(name == "confusion"))
        cohorts = summarize_cohorts(annotated, labels)
        cohorts["metrics"].to_csv(output_dir / strategy / "neighbor_cohort_metrics.csv", index=False)
        cohorts["per_class"].to_csv(output_dir / strategy / "neighbor_cohort_per_class.csv", index=False)
        bootstrap_summaries[strategy] = bootstrap_predictions(annotated, labels,
            n_bootstrap=protocol["bootstrap_samples"], seed=protocol["model_config"]["seed"])
        bootstrap_summaries[strategy]["intervals"].to_csv(output_dir / strategy / "cluster_bootstrap_intervals.csv", index=False)
    if len(set(initialization_hashes)) != 1:
        raise ValueError("models did not start from identical parameter values")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
    paired = compare_paired_predictions(predictions["random"], predictions["group_aware"], labels,
        frozen_groups, n_bootstrap=protocol["bootstrap_samples"], seed=protocol["model_config"]["seed"])
    paired["intervals"].to_csv(output_dir / "common_validation_comparison.csv", index=False)
    write_json(output_dir / "common_validation_comparison.json", paired)
    write_json(output_dir / "cluster_bootstrap_intervals.json", bootstrap_summaries)
    indexed = comparison.set_index("strategy")
    difference = {key: float(indexed.at["random", key] - indexed.at["group_aware", key])
                  for key in ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1")}
    summary = {"version": VERSION, "counts": protocol["counts"],
               "label_names": labels, "model_config": protocol["model_config"],
               "validation_strategy_difference_random_minus_group": difference,
               "eligible_independent_audit_pairs": len(near),
               "group_aware_near_neighbor_affected_row_bound": affected_row_bound(predictions["group_aware"]),
               "test_predictions_created": False, "identical_initialization": True,
               "splitting_and_grouping_unchanged": True, "single_seed_diagnostic": True,
               "causal_leakage_inflation_estimated": False,
               "common_validation": paired, "limitations": protocol["limitations"],
               "environment": {"python": platform.python_version(), "platform": platform.platform()},
               "protocol_sha256": sha256(output_dir / "protocol.json")}
    write_json(output_dir / "summary.json", summary)
    progress(comparison.to_string(index=False))
    return _jsonable(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args(argv)
    from .segment_transformer import ModelConfig
    if args.analyze_only:
        analyze_experiment(args.output_dir)
        return
    if not args.resume:
        config = ModelConfig(epochs=args.epochs, batch_size=args.batch_size,
                             sequence_length=args.max_length, learning_rate=args.learning_rate,
                             seed=args.seed, device=args.device)
        protocol = prepare_experiment(args.output_dir, config, args.input, args.split_dir, args.bootstrap_samples)
        print(json.dumps(protocol["counts"], indent=2), flush=True)
    if not args.prepare_only:
        run_training(args.output_dir)


if __name__ == "__main__":
    main()
