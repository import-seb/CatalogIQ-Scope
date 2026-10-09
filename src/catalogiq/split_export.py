"""Export full product records with auditable, rule-v3 split assignments."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

from .cleaning import PROVENANCE, TARGET_COLUMNS
from .features import sha256
from .split_evaluation import leakage_metrics, near_duplicate_pairs, segment_distribution
from .split_rules_v3 import RuleRefinementConfig, refine_rule_groups_v3
from .splitting import GROUP_FIELDS, SPLITS, SplitConfig, check_assignments, record_ids
from .splitting_experiment import _assign, _edges_frame, _environment, _write_json, load_input

VERSION = "rule-v3-split-export-v1"


def export_splits(input_path, output_dir, config=None, identifier_path=None,
                  grouping_config=None, config_paths=(), progress=print):
    """Save whole-group partitions, preserving all input fields and string values.

    The existing matcher and Segment-stratified allocator own grouping and
    allocation. This function adds full-row exports, diagnostics and provenance.
    Missing Segment records are retained; supervised callers decide eligibility.
    """
    started = perf_counter()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new run directory")
    config = config or SplitConfig(grouping_version=2)
    if config.grouping_version != 2:
        raise ValueError("split_data uses rule-v3; the shared grouping_version setting must be 2")
    grouping_config = grouping_config or RuleRefinementConfig()
    grouping_config.candidate.validate()
    if set(GROUP_FIELDS) & set(TARGET_COLUMNS):
        raise RuntimeError("Grouping attributes must exclude classification targets")

    input_columns = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if set(input_columns) & {"group_id", "split"}:
        raise ValueError("Input contains reserved group_id or split columns; use the cleaned source export")
    if "record_id" in input_columns:
        original = pd.read_csv(input_path, usecols=["record_id", *PROVENANCE],
                               dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if original.record_id.tolist() != record_ids(original).tolist():
            raise ValueError("Input record_id values do not match their source identities")

    frame, input_metadata = load_input(input_path, identifier_path)
    config_fingerprints = {str(Path(path).resolve()): sha256(Path(path)) for path in config_paths}
    progress(f"Grouping {len(frame):,} records with rule-v3")
    stage = perf_counter()
    groups, edges, grouping_stats = refine_rule_groups_v3(
        frame, frame.record_id.to_numpy(), config, grouping_config)
    grouping_seconds = perf_counter() - stage
    stage = perf_counter()
    assignments, allocation_stats = _assign(frame, groups, config)
    check_assignments(assignments, frame.record_id)
    repeated, _ = _assign(frame, groups, config)
    if not assignments.equals(repeated):
        raise RuntimeError("Seeded whole-group allocation did not reproduce")
    allocation_seconds = perf_counter() - stage

    progress("Auditing exact duplicates and independent near-duplicate name pairs")
    stage = perf_counter()
    audit_pairs, evaluation = near_duplicate_pairs(frame, config, progress=progress)
    metrics = leakage_metrics(frame, assignments, audit_pairs, config)
    distributions = segment_distribution(frame, assignments)
    audit_seconds = perf_counter() - stage
    for path, expected in {**input_metadata["fingerprints"], **config_fingerprints}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input or configuration artifact changed during splitting: {path}")

    exports = frame.copy()
    exports["group_id"] = assignments.group_id.to_numpy()
    exports["split"] = assignments.split.to_numpy()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "split_config.json", config.to_dict())
    _write_json(output_dir / "rule_refinement_config.json", grouping_config.to_dict())
    _write_json(output_dir / "input_manifest.json", input_metadata)
    assignments.to_csv(output_dir / "rule_assignments.csv", index=False, lineterminator="\n")
    _edges_frame(edges, frame).to_csv(output_dir / "rule_matching_edges.csv",
                                      index=False, lineterminator="\n")
    assignments.groupby("group_id", sort=True).size().rename("rows").reset_index().to_csv(
        output_dir / "rule_group_sizes.csv", index=False, lineterminator="\n")
    distributions.to_csv(output_dir / "segment_distributions.csv", index=False, lineterminator="\n")
    pd.DataFrame([
        {"left_record_id": frame.at[a, "record_id"], "right_record_id": frame.at[b, "record_id"],
         "name_jaccard": score, "left_split": assignments.at[a, "split"],
         "right_split": assignments.at[b, "split"],
         "crosses_split": assignments.at[a, "split"] != assignments.at[b, "split"]}
        for a, b, score in audit_pairs], columns=["left_record_id", "right_record_id", "name_jaccard",
            "left_split", "right_split", "crosses_split"]).to_csv(
        output_dir / "independent_near_duplicate_pairs.csv", index=False, lineterminator="\n")
    for split in SPLITS:
        subset = exports.loc[exports.split.eq(split)].reset_index(drop=True)
        path = output_dir / f"{split}.csv"
        subset.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        restored = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        pd.testing.assert_frame_equal(restored, subset, check_dtype=False)

    # Recheck after exporting so the completed metadata certifies stable inputs.
    for path, expected in {**input_metadata["fingerprints"], **config_fingerprints}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input or configuration artifact changed during export: {path}")
    environment = _environment()
    environment["code_sha256"]["cli.py"] = sha256(Path(__file__).with_name("cli.py"))
    summary = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "grouping_method": "rule_v3", "target": "Segment", "seed": config.seed,
        "input": input_metadata, "configuration_input_sha256": config_fingerprints,
        "split_config": config.to_dict(), "rule_refinement_config": grouping_config.to_dict(),
        "environment": environment, "grouping": grouping_stats,
        "split_allocator": allocation_stats, "metrics": metrics, "evaluation": evaluation,
        "exports": {split: {"path": f"{split}.csv", "rows": metrics["splits"][split]["rows"]}
                    for split in SPLITS},
        "export_columns": exports.columns.tolist(), "original_input_columns": input_columns,
        "supervised_eligibility": "Missing Segment values are preserved; filter them before supervised fitting.",
        "model_input_policy": "Select product features explicitly; assignment IDs, provenance and other targets are not model features.",
        "checks": {"complete_unique_coverage": True, "group_isolation": True,
                   "labels_excluded_from_grouping": True, "allocation_reproducibility": True,
                   "export_values_preserved": True, "input_artifacts_unchanged": True},
        "runtime_seconds": {"grouping": grouping_seconds, "allocation_and_reproduction": allocation_seconds,
                            "independent_audit": audit_seconds, "total": perf_counter() - started},
        "output_sha256": {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
    }
    _write_json(output_dir / "summary.json", summary)
    progress(f"Saved full-record splits in {output_dir}")
    progress(" | ".join(f"{split}: {metrics['splits'][split]['rows']:,}" for split in SPLITS))
    return summary
