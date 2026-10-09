"""Segment baseline using all labeled development records and one fixed test.

Product groups, the final-test assignment, preprocessing, and the transformer
remain frozen. An experiment-local allocator balances validation composition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .features import sha256
from .segment_experiment import (DEFAULT_INPUT, DEFAULT_SPLITS, ROOT, analyze_experiment,
    read_csv, run_training, validate_development_assignments, verify_protocol, write_json)
from .splitting import record_ids

VERSION = "segment-full-development-baseline-v2"
DEFAULT_PREVIOUS = ROOT / "data/processed/segment_validation_baseline_20261008"
DEFAULT_REFERENCE = ROOT / "data/processed/integrated_policy_v4_final_20261003/target_features.csv"


def full_development_pool(frame, frozen_assignment):
    """Return every labeled frozen-v3 non-test row, with no other embargo."""
    required = {"record_id", "Segment", "Retailer"}
    if not required.issubset(frame) or not {"record_id", "group_id", "split"}.issubset(frozen_assignment):
        raise ValueError("missing input identity, target, retailer, or frozen assignment")
    if frame.record_id.duplicated().any() or frozen_assignment.record_id.duplicated().any():
        raise ValueError("record identities must be unique")
    if set(frame.record_id) != set(frozen_assignment.record_id):
        raise ValueError("frozen assignment must cover the exact input universe")
    if not frozen_assignment.split.isin(["train", "validation", "test"]).all():
        raise ValueError("invalid frozen role")
    if (frozen_assignment.group_id.isna().any()
            or frozen_assignment.group_id.astype(str).str.strip().eq("").any()
            or frozen_assignment.groupby("group_id").split.nunique().gt(1).any()):
        raise ValueError("invalid or crossing frozen product group")
    ordered = frame.sort_values("record_id", kind="stable").reset_index(drop=True).copy()
    indexed = frozen_assignment.set_index("record_id")
    roles = ordered.record_id.map(indexed.split)
    labeled = ~ordered.Segment.str.strip().str.casefold().isin(["", "null", "<missing>"])
    eligible = roles.ne("test") & labeled
    development = ordered.loc[eligible].copy().reset_index(drop=True)
    development["group_id"] = development.record_id.map(indexed.group_id)
    exclusions = ordered.loc[~eligible, ["record_id"]].copy()
    exclusions["reason"] = roles.loc[~eligible].eq("test").map(
        {True: "protected_rule_v3_final_test", False: "missing_Segment"})
    if development.empty or not roles.eq("test").any():
        raise ValueError("development and protected test populations are both required")
    if set(development.group_id) & set(frozen_assignment.loc[frozen_assignment.split.eq("test"), "group_id"]):
        raise ValueError("development shares a product group with the final test")
    counts = {"input_records": len(frame), "protected_final_test_records": int(roles.eq("test").sum()),
              "full_development_records_including_unlabeled": int(roles.ne("test").sum()),
              "missing_Segment_development_records": int((roles.ne("test") & ~labeled).sum()),
              "eligible_labeled_development_records": len(development),
              "product_groups": int(development.group_id.nunique()),
              "largest_labeled_development_group": int(development.groupby("group_id").size().max()),
              "protected_final_test_snapshots": 1, "other_strategy_test_embargo_applied": False}
    return development, exclusions.reset_index(drop=True), counts


def prepare_full_development(output_dir, model_config, allocation_config=None,
        input_path=DEFAULT_INPUT, split_dir=DEFAULT_SPLITS, previous_dir=DEFAULT_PREVIOUS,
        reference_path=DEFAULT_REFERENCE, bootstrap_samples=500):
    from .rule_refinement_experiment import verify_rule_refinement_freeze
    from .segment_development_allocation import (AllocationConfig, allocate_development_groups,
        random_control, validate_development_population)
    from .segment_development_profile import profile_development

    output_dir, input_path, split_dir, previous_dir = [Path(path).resolve()
        for path in (output_dir, input_path, split_dir, previous_dir)]
    if output_dir.exists():
        raise ValueError("Use a new output directory; preserved runs cannot be overwritten")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    allocation_config = allocation_config or AllocationConfig(seed=model_config.seed)
    verify_rule_refinement_freeze(split_dir)
    previous_protocol = verify_protocol(previous_dir)
    manifest_path = split_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(input_path) != manifest["sha256"]:
        raise ValueError("input differs from the frozen cleaned dataset")
    frame = read_csv(input_path)
    frame["record_id"] = record_ids(frame)
    frozen = read_csv(split_dir / "rule_assignments.csv")
    development, exclusions, counts = full_development_pool(frame, frozen)
    grouped, allocation_stats = allocate_development_groups(development, allocation_config)
    random = random_control(grouped, model_config.seed)
    group_audit = validate_development_population(development, grouped, allocation_config)
    random_audit = validate_development_population(development, random, allocation_config,
                                                   require_group_isolation=False)
    grouped = grouped.assign(strategy="group_aware")
    random = random.assign(strategy="random")
    assignments = pd.concat([random, grouped], ignore_index=True)
    validate_development_assignments(assignments, exclusions)
    # No source rows can silently disappear under a second method's test policy.
    if set(assignments.record_id) != set(development.record_id):
        raise AssertionError("experiment dropped an eligible development record")
    repeated_grouped, _ = allocate_development_groups(development.iloc[::-1], allocation_config)
    repeated_random = random_control(repeated_grouped, model_config.seed)
    pd.testing.assert_frame_equal(grouped.drop(columns="strategy"), repeated_grouped)
    pd.testing.assert_frame_equal(random.drop(columns="strategy"), repeated_random)
    previous_assignment_path = previous_dir / "development_assignments.csv"
    previous_assignments = read_csv(previous_assignment_path)
    previous_ids = set(previous_assignments.record_id)
    if not previous_ids.issubset(set(development.record_id)):
        raise ValueError("previous development population is not a subset of full development")
    reference = None
    reference_path = Path(reference_path).resolve() if reference_path else None
    if reference_path is not None:
        # Feature-only supplied prediction export, never labels or predictions.
        reference = read_csv(reference_path)
        forbidden = {"Segment", "Brand", "Mnfr", "Sub-Segment", "TargetAgeGroup", "Platform"}
        if forbidden & set(reference.columns):
            raise ValueError("production reference must be a feature-only export")
    profile = profile_development(frame, assignments, exclusions,
        reference_features=reference, previous_assignments=previous_assignments,
        expected_development_ids=set(development.record_id), frozen_role_assignments=frozen)
    counts.update({"train_records_per_strategy": int(grouped.split.eq("train").sum()),
                   "validation_records_per_strategy": int(grouped.split.eq("validation").sum()),
                   "previous_reduced_development_records": len(previous_ids),
                   "restored_labeled_records": len(set(development.record_id) - previous_ids)})
    output_dir.mkdir(parents=True)
    assignments.to_csv(output_dir / "development_assignments.csv", index=False, lineterminator="\n")
    exclusions.to_csv(output_dir / "excluded_records.csv", index=False, lineterminator="\n")
    frozen.loc[frozen.split.eq("test"), ["record_id", "group_id", "split"]].to_csv(
        output_dir / "protected_final_test.csv", index=False)
    class_counts = assignments.groupby(["strategy", "Segment", "split"]).size().rename("records").reset_index()
    class_counts.to_csv(output_dir / "class_counts.csv", index=False)
    tables = ("retailer_distribution", "feature_missingness", "partition_group_sizes", "class_distribution")
    for name in tables:
        profile[name].to_csv(output_dir / f"pretraining_{name}.csv", index=False)
    write_json(output_dir / "pretraining_population_audit.json", profile["summary"])
    write_json(output_dir / "allocation_audit.json", {"configuration": asdict(allocation_config),
        "grouped": group_audit, "random": random_audit, "optimization": allocation_stats,
        "assignments_seed_and_row_order_reproducible": True})
    source_paths = [input_path, manifest_path, split_dir / "rule_refinement_freeze.json",
        split_dir / "rule_assignments.csv", split_dir / "independent_near_duplicate_pairs.csv",
        split_dir / "exact_payload_duplicate_pairs.csv", previous_dir / "protocol.json",
        previous_dir / "protocol_freeze.json", previous_assignment_path,
        Path(__file__), ROOT / "scripts/compare_segment_full_development.py",
        Path(__file__).with_name("segment_development_allocation.py"),
        Path(__file__).with_name("segment_development_profile.py"),
        Path(__file__).with_name("segment_experiment.py"),
        Path(__file__).with_name("segment_transformer.py"),
        Path(__file__).with_name("segment_validation_analysis.py"), ROOT / "requirements-training.txt"]
    if reference_path:
        source_paths.append(reference_path)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(model_config.model_name, revision=model_config.revision,
                                     local_files_only=model_config.local_files_only))
    source_paths.extend(path for path in snapshot.rglob("*") if path.is_file())
    protocol = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path), "split_dir": str(split_dir), "target": "Segment",
        "model_config": asdict(model_config), "allocation_config": asdict(allocation_config),
        "counts": counts, "label_names": sorted(grouped.loc[grouped.split.eq("train"), "Segment"].unique()),
        "features": ["ProductName", "ProductBrand", "ProductDescription", "ProductContents"],
        "primary_metric": "macro_f1", "bootstrap_samples": bootstrap_samples,
        "checkpoint_policy": "fixed final epoch; no model or threshold search",
        "test_protection": "only the fixed rule-v3 final test; other methods' old test memberships are eligible development",
        "validation_policy": "seeded whole-group allocation balancing Segment, retailer, and group-size margins; class-matched random control",
        "development_pool_definition": "every labeled frozen rule-v3 train/validation record, without subsampling or multi-method closure",
        "pretraining_population_checks_passed": True,
        "grouping_algorithms_unchanged": True, "existing_split_engine_source_unchanged": True,
        "development_validation_allocator_changed": True, "test_predictions_permitted": False,
        "previous_reduced_run_preserved": str(previous_dir),
        "previous_protocol_sha256": sha256(previous_dir / "protocol.json"),
        "previous_model_config": previous_protocol["model_config"],
        "interpretation": "strategy comparison on representative development partitions; not a causal leakage estimate",
        "limitations": ["single training seed", "fixed final-test exclusion remains",
                        "incoming production family mixture is unconfirmed",
                        "group membership can contain matching errors",
                        "model differences do not isolate leakage from training-family coverage"],
        "protected_inputs_and_sources": {str(path): sha256(path) for path in source_paths},
        "prepared_artifacts": {str(path): sha256(path) for path in output_dir.iterdir() if path.is_file()}}
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "protocol_freeze.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
                                                      "frozen_before_training": True})
    verify_protocol(output_dir)
    return protocol


def finish_comparison(output_dir, summary):
    """Add v2 provenance to the shared, unchanged validation scoring results."""
    output_dir = Path(output_dir).resolve()
    protocol = verify_protocol(output_dir)
    summary = dict(summary)
    summary.update({"version": VERSION, "full_development_pool_used": True,
        "grouping_algorithms_unchanged": True, "existing_split_engine_source_unchanged": True,
        "development_validation_allocator_changed": True,
        "pretraining_population_audit": str(output_dir / "pretraining_population_audit.json"),
        "previous_reduced_run_preserved": protocol["previous_reduced_run_preserved"],
        "previous_2_42_point_gap_is_not_used_as_a_design_selection_criterion": True})
    # Clarify the reused v1 scorer's legacy field: partitions were intentionally regenerated.
    summary["splitting_and_grouping_unchanged"] = False
    write_json(output_dir / "summary.json", summary)
    old_comparison = Path(protocol["previous_reduced_run_preserved"]) / "validation_comparison.csv"
    old = pd.read_csv(old_comparison).assign(experiment="previous_reduced_pool")
    current = pd.read_csv(output_dir / "validation_comparison.csv").assign(experiment="full_development_pool")
    pd.concat([old, current], ignore_index=True).to_csv(output_dir / "experiment_comparison.csv", index=False)
    return summary


def main(argv=None):
    from .segment_transformer import ModelConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--reference-features", type=Path, default=DEFAULT_REFERENCE)
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
    if args.analyze_only:
        finish_comparison(args.output_dir, analyze_experiment(args.output_dir))
        return
    if not args.resume:
        model_config = ModelConfig(epochs=args.epochs, batch_size=args.batch_size,
            sequence_length=args.max_length, learning_rate=args.learning_rate, seed=args.seed, device=args.device)
        protocol = prepare_full_development(args.output_dir, model_config, input_path=args.input,
            split_dir=args.split_dir, previous_dir=args.previous_dir, reference_path=args.reference_features,
            bootstrap_samples=args.bootstrap_samples)
        print(json.dumps(protocol["counts"], indent=2), flush=True)
    if not args.prepare_only:
        finish_comparison(args.output_dir, run_training(args.output_dir))


if __name__ == "__main__":
    main()
