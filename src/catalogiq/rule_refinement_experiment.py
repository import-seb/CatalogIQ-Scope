"""Refine rules without changing the allocator, TF-IDF baseline, or audit pool.

The 75 inspected pairs are regression diagnostics. A successful run is sealed
before the separately reserved product evidence is reviewed or scored.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .group_graph_audit import _verify_edge_snapshot
from .grouping_regression import compare_review_regression
from .independent_review_reservation import verify_reservation
from .paths import ROOT_PATH
from .split_evaluation import PAYLOAD_FIELDS, evaluation_name, leakage_metrics, segment_distribution
from .split_refinement_comparison import AUDIT_PARAMETERS, _canonical_audit
from .split_review_validation import verify_refinement_seal
from .splitting import Components, SplitConfig, check_assignments, grouping_view
from .splitting_experiment import _assign, _edges_frame, _environment, comparison_table, load_input


VERSION = "rule-refinement-v3"
ROOT = ROOT_PATH
DEFAULT_BASELINE = ROOT / "data/processed/split_refinement_20261008_final"
DEFAULT_RESERVATION = ROOT / "data/processed/split_future_review_reservation_20261008"
DEFAULT_EDGE_SNAPSHOT = ROOT / "data/processed/split_balanced_graph_edge_snapshot_20261008.json"
DEFAULT_FIXTURE = ROOT / "tests/fixtures/grouping_review_75.json"
DEFAULT_REFERENCE = ROOT / "tests/fixtures/grouping_review_75_baseline.json"
SOURCE_NAMES = ("split_rules_v3.py", "split_rule_evidence_v3.py", "split_rule_candidates_v3.py",
                "rule_refinement_experiment.py", "grouping_regression.py")
METHODS = ("rule_v2", "rule_v3", "tfidf_v2")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def _csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _hashes(paths):
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def _assert_hashes(hashes):
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"protected input or implementation changed: {name}")


def _configuration(config):
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("refinement configuration must be a dataclass or provide to_dict")


def _saved_assignment(path, frame):
    assignment = pd.read_csv(path, dtype=str, keep_default_na=False)
    check_assignments(assignment, frame.record_id)
    assignment = assignment.set_index("record_id").loc[frame.record_id].reset_index()
    for column in (*PROVENANCE, "Segment"):
        if column not in assignment or assignment[column].tolist() != frame[column].tolist():
            raise ValueError(f"saved baseline disagrees with shared input: {column}")
    return assignment


def load_fixed_audit(path, frame):
    """Read the frozen independent pool; no grouping graph supplies candidates."""
    canonical = _canonical_audit(path)
    positions = {record: i for i, record in enumerate(frame.record_id)}
    if any(a not in positions or b not in positions for a, b in canonical):
        raise ValueError("independent audit references a record outside the shared input")
    audit = pd.read_csv(path, usecols=["left_record_id", "right_record_id", "name_jaccard"],
                        dtype=str, keep_default_na=False)
    pairs = [(positions[a], positions[b], float(score))
             for a, b, score in audit.itertuples(index=False, name=None)]
    return audit, pairs


def check_edge_components(assignment, edge_frame):
    """Verify every accepted edge and reconstruct the entire saved partition."""
    ids = assignment.record_id.tolist()
    positions = {record: i for i, record in enumerate(ids)}
    components = Components(len(ids))
    for left, right, reason, score in edge_frame.itertuples(index=False, name=None):
        if (left not in positions or right not in positions or left == right or not str(reason)
                or not np.isfinite(float(score)) or not 0 <= float(score) <= 1):
            raise ValueError("invalid accepted matching edge")
        a, b = positions[left], positions[right]
        if assignment.at[a, "group_id"] != assignment.at[b, "group_id"]:
            raise ValueError("accepted edge crosses a saved group")
        components.union(a, b)
    roots = [components.find(i) for i in range(len(ids))]
    reconstruction = pd.DataFrame({"group": assignment.group_id, "root": roots})
    if (reconstruction.groupby("group").root.nunique().ne(1).any()
            or reconstruction.groupby("root").group.nunique().ne(1).any()):
        raise ValueError("accepted-edge components do not reconstruct saved grouping")
    return True


def _near_audit_outcomes(audit, assignments):
    result = audit.copy()
    for method, assignment in assignments.items():
        indexed = assignment.set_index("record_id")
        for side in ("left", "right"):
            result[f"{method}_{side}_split"] = result[f"{side}_record_id"].map(indexed.split)
            result[f"{method}_{side}_group_id"] = result[f"{side}_record_id"].map(indexed.group_id)
        result[f"{method}_crosses_split"] = result[f"{method}_left_split"].ne(result[f"{method}_right_split"])
        result[f"{method}_co_grouped"] = result[f"{method}_left_group_id"].eq(result[f"{method}_right_group_id"])
    return result


def _exact_audit_outcomes(frame, assignments):
    """Enumerate raw payload duplicate pairs under the existing evaluator scope."""
    from collections import defaultdict
    from itertools import combinations
    buckets = defaultdict(list)
    for i, row in enumerate(frame.reindex(columns=PAYLOAD_FIELDS).itertuples(index=False, name=None)):
        key = tuple(None if pd.isna(value) else str(value) for value in row)
        if any(evaluation_name(value) for value in key):
            buckets[key].append(i)
    rows = []
    for members in buckets.values():
        for a, b in combinations(members, 2):
            row = {"left_record_id": frame.at[a, "record_id"], "right_record_id": frame.at[b, "record_id"]}
            for method, assignment in assignments.items():
                row[f"{method}_left_split"] = assignment.at[a, "split"]
                row[f"{method}_right_split"] = assignment.at[b, "split"]
                row[f"{method}_crosses_split"] = assignment.at[a, "split"] != assignment.at[b, "split"]
                row[f"{method}_co_grouped"] = assignment.at[a, "group_id"] == assignment.at[b, "group_id"]
            rows.append(row)
    columns = ["left_record_id", "right_record_id", *[column for method in assignments for column in
               (f"{method}_left_split", f"{method}_right_split", f"{method}_crosses_split", f"{method}_co_grouped")]]
    return pd.DataFrame(rows, columns=columns)


def _save_group_sizes(output_dir, method, assignment):
    sizes = assignment.groupby("group_id", sort=True).size().rename("rows").reset_index()
    sizes.to_csv(output_dir / f"{method}_group_sizes.csv", index=False, lineterminator="\n")
    sizes.rows.value_counts().sort_index().rename_axis("group_size").rename("groups").to_csv(
        output_dir / f"{method}_group_size_distribution.csv", lineterminator="\n")


def run_rule_refinement(baseline_dir, output_dir, refinement_config=None, *,
                        reservation_dir=DEFAULT_RESERVATION, edge_snapshot=DEFAULT_EDGE_SNAPSHOT,
                        fixture_path=DEFAULT_FIXTURE, reference_path=DEFAULT_REFERENCE, progress=print):
    """Run and repeat v3 rules, audit all methods identically, then seal the run.

    Saved TF-IDF bytes are copied unchanged. No fresh review examples or judgments
    are read by the algorithm, regression evaluator, or split allocator.
    """
    from .split_rules_v3 import RuleRefinementConfig, refine_rule_groups_v3
    baseline_dir, output_dir, reservation_dir = map(lambda p: Path(p).resolve(),
                                                   (baseline_dir, output_dir, reservation_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite a grouping experiment: {output_dir}")
    if any(output_dir.is_relative_to(path) for path in (baseline_dir, reservation_dir)):
        raise ValueError("new experiment must be outside protected baseline and reservation directories")
    started = perf_counter()
    old_seal = verify_refinement_seal(baseline_dir)
    _verify_edge_snapshot(baseline_dir, edge_snapshot)
    reservation = verify_reservation(reservation_dir)
    baseline_summary = _json(baseline_dir / "summary.json")
    input_metadata = _json(baseline_dir / "input_manifest.json")
    split_config = SplitConfig.from_dict(_json(baseline_dir / "config.json"))
    if split_config.seed != 42 or split_config.ratios != (0.70, 0.15, 0.15) or split_config.grouping_version != 2:
        raise ValueError("baseline must retain seed42,70/15/15 allocation and v2 comparison methods")
    refinement_config = refinement_config or RuleRefinementConfig()
    source_paths = [Path(__file__).parent / name for name in SOURCE_NAMES]
    source_paths.append(ROOT / "scripts/compare_rule_refinement.py")
    reservation_paths = list(reservation_dir.iterdir())
    protected = _hashes([*map(Path, old_seal["sha256"]), baseline_dir / "refinement_freeze.json",
                         baseline_dir / "rule_matching_edges.csv", baseline_dir / "tfidf_matching_edges.csv",
                         Path(edge_snapshot), *reservation_paths, *source_paths, Path(fixture_path), Path(reference_path),
                         *map(Path, input_metadata["fingerprints"])])
    frame, actual_metadata = load_input(Path(input_metadata["path"]),
                                        Path(input_metadata["identifier_source"]) if input_metadata["identifier_source"] else None)
    if actual_metadata != input_metadata or actual_metadata != baseline_summary["input"]:
        raise ValueError("v3 input differs from the frozen baseline manifest")
    assignments = {"rule_v2": _saved_assignment(baseline_dir / "rule_assignments.csv", frame),
                   "tfidf_v2": _saved_assignment(baseline_dir / "tfidf_assignments.csv", frame)}
    audit, pairs = load_fixed_audit(baseline_dir / "independent_near_duplicate_pairs.csv", frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", split_config.to_dict())
    _write_json(output_dir / "rule_refinement_config.json", _configuration(refinement_config))
    _write_json(output_dir / "input_manifest.json", input_metadata)
    for original, saved in (("rule_assignments.csv", "rule_v2_assignments.csv"),
                            ("rule_matching_edges.csv", "rule_v2_matching_edges.csv"),
                            ("tfidf_assignments.csv", "tfidf_assignments.csv"),
                            ("tfidf_matching_edges.csv", "tfidf_matching_edges.csv")):
        shutil.copyfile(baseline_dir / original, output_dir / saved)
    progress(f"Grouping {len(frame):,} records with refined rules; TF-IDF remains the saved v2 baseline")
    features = grouping_view(frame)
    ids = frame.record_id.to_numpy()
    start = perf_counter()
    groups, edges, grouping_stats = refine_rule_groups_v3(features, ids, split_config, refinement_config)
    grouping_seconds = perf_counter() - start
    start = perf_counter()
    assignment, allocator_stats = _assign(frame, groups, split_config)
    splitting_seconds = perf_counter() - start
    assignments["rule_v3"] = assignment
    assignments = {method: assignments[method] for method in METHODS}
    edge_frame = _edges_frame(edges, frame)
    check_edge_components(assignment, edge_frame)
    (output_dir / "rule_assignments.csv").write_bytes(_csv_bytes(assignment))
    (output_dir / "rule_matching_edges.csv").write_bytes(_csv_bytes(edge_frame))
    progress("Repeating the full refined grouping and allocation to verify identical saved bytes")
    start = perf_counter()
    repeated_groups, repeated_edges, _ = refine_rule_groups_v3(grouping_view(frame), ids, split_config, refinement_config)
    repeated, _ = _assign(frame, repeated_groups, split_config)
    repeat_seconds = perf_counter() - start
    if _csv_bytes(assignment) != _csv_bytes(repeated) or _csv_bytes(edge_frame) != _csv_bytes(_edges_frame(repeated_edges, frame)):
        raise RuntimeError("refined grouping, edge evidence, or split assignments are not reproducible")
    progress("Applying the fixed independent near-duplicate pool and unchanged exact-payload audit to all three methods")
    start = perf_counter()
    distributions, method_stats = [], {}
    for method, current in assignments.items():
        check_assignments(current, ids)
        _save_group_sizes(output_dir, {"rule_v2": "rule_v2", "rule_v3": "rule", "tfidf_v2": "tfidf"}[method], current)
        metrics = leakage_metrics(frame, current, pairs, split_config)
        if method != "rule_v3":
            original_method = "rule" if method == "rule_v2" else "tfidf"
            previous = baseline_summary["approaches"][original_method]
            if metrics != previous["metrics"]:
                raise ValueError(f"fixed evaluator results differ from frozen baseline: {method}")
            method_stats[method] = {"metrics": metrics, "runtime_seconds": previous["runtime_seconds"],
                                    "runtime_source": "original frozen v2 experiment; grouping was not rerun",
                                    "version": "rule-v2" if method == "rule_v2" else "tfidf-v2"}
        else:
            method_stats[method] = {"metrics": metrics, "grouping": grouping_stats,
                                    "split_allocator": allocator_stats, "version": VERSION,
                                    "runtime_seconds": {"grouping": grouping_seconds, "splitting": splitting_seconds,
                                                        "primary_total": grouping_seconds + splitting_seconds},
                                    "runtime_source": "first v3 execution; full-repeat time reported separately"}
        distribution = segment_distribution(frame, current)
        distribution.insert(0, "approach", method)
        distributions.append(distribution)
    pd.concat(distributions, ignore_index=True).to_csv(output_dir / "segment_distributions.csv", index=False, lineterminator="\n")
    near = _near_audit_outcomes(audit, assignments)
    near.to_csv(output_dir / "independent_near_duplicate_pairs.csv", index=False, lineterminator="\n")
    if _canonical_audit(output_dir / "independent_near_duplicate_pairs.csv") != _canonical_audit(baseline_dir / "independent_near_duplicate_pairs.csv"):
        raise ValueError("saved independent audit pool or exact recorded scores changed")
    exact = _exact_audit_outcomes(frame, assignments)
    exact.to_csv(output_dir / "exact_payload_duplicate_pairs.csv", index=False, lineterminator="\n")
    for method in assignments:
        if (len(exact) != method_stats[method]["metrics"]["exact_payload"]["duplicate_pairs"]
                or int(exact[f"{method}_crosses_split"].sum()) != method_stats[method]["metrics"]["exact_payload"]["crossing_pairs"]):
            raise ValueError("enumerated exact-payload pairs disagree with the shared evaluator")
    audit_seconds = perf_counter() - start
    regression, regression_metrics = compare_review_regression(fixture_path, reference_path,
        {"rule": assignments["rule_v3"], "tfidf": assignments["tfidf_v2"]})
    regression.to_csv(output_dir / "regression_pair_outcomes.csv", index=False, lineterminator="\n")
    regression_metrics.to_csv(output_dir / "regression_metrics.csv", index=False, lineterminator="\n")
    new_clear_errors = int(regression.new_clear_error.sum())
    _assert_hashes(protected)
    verify_refinement_seal(baseline_dir)
    _verify_edge_snapshot(baseline_dir, edge_snapshot)
    verify_reservation(reservation_dir)
    checks = {"same_input_manifest": True, "same_records": True, "group_isolation": True,
              "accepted_edges_reconstruct_groups": True, "target_columns_excluded_from_grouping": True,
              "input_and_frozen_baselines_unchanged": True, "split_engine_unchanged": True,
              "tfidf_assignment_bytes_unchanged": sha256(output_dir / "tfidf_assignments.csv") == sha256(baseline_dir / "tfidf_assignments.csv"),
              "tfidf_edge_bytes_unchanged": sha256(output_dir / "tfidf_matching_edges.csv") == sha256(baseline_dir / "tfidf_matching_edges.csv"),
              "same_independent_pool_and_recorded_scores": True,
              "full_grouping_allocation_and_edges_byte_reproducible": True,
              "reserved_sample_unchanged_and_unreviewed": True, "no_new_clear_regression_errors": new_clear_errors == 0}
    summary = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
               "input": input_metadata, "config": split_config.to_dict(),
               "rule_refinement_config": _configuration(refinement_config), "baseline_dir": str(baseline_dir),
               "reservation_dir": str(reservation_dir), "edge_snapshot": str(Path(edge_snapshot).resolve()),
               "environment": _environment(), "approaches": method_stats, "checks": checks,
               "evaluation": {"method": "fixed independent character-shingle Jaccard pool and raw payload duplicates",
                   "near_pairs": len(pairs), "exact_payload_pairs": len(exact),
                   "parameters": {key: getattr(split_config, key) for key in AUDIT_PARAMETERS},
                   "pool_source_sha256": sha256(baseline_dir / "independent_near_duplicate_pairs.csv"),
                   "runtime_seconds": audit_seconds},
               "regression": {"new_clear_errors": new_clear_errors, "pairs": len(regression) // 2,
                   "interpretation": "Inspected development regression cases; known-error fixes are permitted; this is not independent or population accuracy."},
               "independent_review": {"reserved_pairs": reservation["reserved_pairs"], "status": "unreviewed_until_this_run_is_frozen"},
               "repeat_runtime_seconds": repeat_seconds,
               "total_wall_seconds_including_io_evaluation_and_reproduction": perf_counter() - started}
    comparison_table(summary).to_csv(output_dir / "comparison.csv", index_label="metric", lineterminator="\n")
    summary["output_sha256"] = {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    _write_json(output_dir / "summary.json", summary)
    if new_clear_errors:
        progress(f"Saved results, but freeze blocked by {new_clear_errors} new clear regression errors; fresh review remains unopened")
        return summary
    protected.update(_hashes(path for path in output_dir.iterdir() if path.is_file()))
    freeze = {"status": "frozen_before_fresh_independent_review", "version": VERSION,
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "baseline_dir": str(baseline_dir),
              "reservation_dir": str(reservation_dir), "edge_snapshot": str(Path(edge_snapshot).resolve()),
              "protocol": "Review reserved product evidence only after verifying this seal. Freeze independent dual judgments before scoring. Do not tune this version after review.",
              "sha256": protected}
    _write_json(output_dir / "rule_refinement_freeze.json", freeze)
    verify_rule_refinement_freeze(output_dir)
    progress(f"Sealed refined rule experiment before independent review: {output_dir}")
    return summary


def verify_rule_refinement_freeze(run_dir):
    """Gate fresh evidence review and subsequent scoring on an unchanged run."""
    run_dir = Path(run_dir).resolve()
    freeze = _json(run_dir / "rule_refinement_freeze.json")
    if freeze.get("status") != "frozen_before_fresh_independent_review" or freeze.get("version") != VERSION:
        raise ValueError("invalid rule refinement freeze status or version")
    _assert_hashes(freeze["sha256"])
    verify_refinement_seal(Path(freeze["baseline_dir"]))
    _verify_edge_snapshot(Path(freeze["baseline_dir"]), Path(freeze["edge_snapshot"]))
    verify_reservation(Path(freeze["reservation_dir"]))
    summary = _json(run_dir / "summary.json")
    if not all(summary["checks"].values()) or summary["regression"]["new_clear_errors"]:
        raise ValueError("frozen experiment has incomplete checks or clear regressions")
    return {"version": VERSION, "sha256_verified": True, "frozen_baselines_unchanged": True,
            "new_rules_config_and_outputs_unchanged": True, "reserved_sample_unchanged": True,
            "reserved_pairs": summary["independent_review"]["reserved_pairs"],
            "records": summary["input"]["rows"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rule-config", type=Path)
    parser.add_argument("--reservation-dir", type=Path, default=DEFAULT_RESERVATION)
    parser.add_argument("--edge-snapshot", type=Path, default=DEFAULT_EDGE_SNAPSHOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_rule_refinement_freeze(args.verify), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required when not using --verify")
    from .split_rules_v3 import RuleRefinementConfig
    config = None
    if args.rule_config:
        values = _json(args.rule_config)
        config = RuleRefinementConfig.from_dict(values) if hasattr(RuleRefinementConfig, "from_dict") else RuleRefinementConfig(**values)
    summary = run_rule_refinement(args.baseline, args.output_dir, config,
        reservation_dir=args.reservation_dir, edge_snapshot=args.edge_snapshot,
        fixture_path=args.fixture, reference_path=args.reference)
    print(comparison_table(summary).to_string(float_format=lambda number: f"{number:.5g}"))
    print(pd.read_csv(args.output_dir / "regression_metrics.csv").drop(columns="interpretation").to_string(index=False))
    return 0 if summary["checks"]["no_new_clear_regression_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
