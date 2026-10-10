"""Freeze corrected product groups, reserve unexposed groups, and seal splits.

This is a target-specific orchestration layer. The product grouping and existing
development allocation engine remain independently usable and unchanged. Test
eligibility is an exposure constraint, never a model-performance optimization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .grouping_regression import evaluate_review_groups, load_review_fixture
from .model_input_groups import load_model_input_cache, merge_model_input_groups
from .segment_development_allocation import AllocationConfig, allocate_development_groups
from .segment_transformer import ModelConfig
from .split_evaluation import _exact_metrics, leakage_metrics, near_duplicate_pairs, segment_distribution
from .split_rules_v4 import RuleFinalConfig, refine_rule_groups_v4
from .splitting import SplitConfig, check_assignments
from .splitting_experiment import _edges_frame, _write_json, load_input

VERSION = "segment-final-splits-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data/processed/split_finalization_20261009"
MISSING_ALLOCATION_LABEL = "UNLABELED_ALLOCATION_ONLY"


def _csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _source_fingerprints():
    # Pin dependencies affecting grouping, representation, allocation, and audit.
    folder = Path(__file__).parent
    names = {"cleaning.py", "features.py", "segment_transformer.py",
             "segment_development_allocation.py", "model_input_groups.py",
             "grouping_regression.py", "balanced_review_evaluation.py"}
    names |= {path.name for path in folder.glob("split*.py")}
    return {str(folder / name): sha256(folder / name) for name in sorted(names)}


def verify_hashes(fingerprints):
    for name, expected in fingerprints.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")


def reserve_unexposed_groups(ids, groups, eligible_ids, original_test_ids):
    """Reserve every wholly eligible group, without inspecting labels or text."""
    ids, groups = np.asarray(ids, dtype=str), np.asarray(groups, dtype=str)
    eligible_ids, original_test_ids = set(eligible_ids), set(original_test_ids)
    if (len(ids) != len(groups) or not len(ids) or len(set(ids)) != len(ids)
            or any(not value.strip() for value in ids) or any(not value.strip() for value in groups)):
        raise ValueError("Unique records and aligned nonblank groups are required")
    if not eligible_ids <= original_test_ids <= set(ids):
        raise ValueError("Eligibility must be a subset of the original unscored test")
    membership = pd.DataFrame({"record_id": ids, "group_id": groups})
    membership["eligible"] = membership.record_id.isin(eligible_ids)
    entire = membership.groupby("group_id", sort=True).eligible.all()
    reserved = membership.group_id.map(entire).to_numpy(dtype=bool)
    if not reserved.any() or reserved.all():
        raise ValueError("Both unexposed test and development populations must be nonempty")
    return reserved, {
        "policy": "reserve_all_wholly_eligible_groups_without_label_or_outcome_selection",
        "original_unscored_test_records": len(original_test_ids),
        "historically_eligible_records": len(eligible_ids),
        "removed_by_corrected_or_effective_input_group_closure": int(len(eligible_ids) - reserved.sum()),
        "final_test_records": int(reserved.sum()),
        "final_test_groups": int(membership.loc[reserved, "group_id"].nunique()),
        "previously_model_seen_records_used_to_pad_test": 0,
    }


def allocate_remaining(frame, groups, test_mask, config=None):
    """Reuse the existing development allocator; preserve actual missing labels."""
    config = config or AllocationConfig()
    view = frame.loc[~test_mask, ["record_id", "Segment"]].copy()
    view["group_id"] = np.asarray(groups)[~test_mask]
    retailer = frame.get("Retailer", pd.Series("", index=frame.index))
    view["Retailer"] = retailer.loc[~test_mask].fillna("").to_numpy()
    missing = view.Segment.str.strip().str.casefold().isin(("", "null", "none", "nan", "<missing>"))
    if view.Segment.eq(MISSING_ALLOCATION_LABEL).any():
        raise ValueError("Allocation placeholder conflicts with a real Segment label")
    # This temporary margin is never exported or passed to a model.
    view.loc[missing, "Segment"] = MISSING_ALLOCATION_LABEL
    development, stats = allocate_development_groups(view, config)
    assignment = frame[[*PROVENANCE, "record_id", "Segment"]].copy()
    assignment["group_id"] = groups
    roles = development.set_index("record_id").split
    assignment["split"] = frame.record_id.map(roles).fillna("test")
    check_assignments(assignment, frame.record_id)
    if not np.array_equal(assignment.split.eq("test").to_numpy(), test_mask):
        raise AssertionError("Development allocation changed the reserved test")
    stats["missing_target_policy"] = "Temporary allocation margin only; original labels preserved; excluded from supervised fitting."
    return assignment, stats


def _regression(frame, groups, old_assignments, witness_path, fixture_path):
    mapping = pd.Series(groups, index=frame.record_id)
    old = _csv(old_assignments).set_index("record_id").group_id
    _, pairs, _ = load_review_fixture(fixture_path)
    before, after = evaluate_review_groups(pairs, old), evaluate_review_groups(pairs, mapping)
    clear = pairs.review_category.ne("borderline")
    after["previous_grouped"] = before.grouped
    after["new_clear_error"] = clear & before.reviewed_error.eq(False) & after.reviewed_error.eq(True)
    if after.new_clear_error.any():
        raise ValueError("Corrected grouping broke a previously correct clear regression case")
    witnesses = _csv(witness_path)
    confirmed = witnesses.loc[witnesses.method.eq("rule_v3") & witnesses.evaluation_leakage.eq("safe_to_separate")].copy()
    if len(confirmed) != 4:
        raise ValueError("Exactly four frozen confirmed incorrect connections are required")
    confirmed["corrected_grouped"] = confirmed.left_record_id.map(mapping).eq(confirmed.right_record_id.map(mapping))
    if confirmed.corrected_grouped.any():
        raise ValueError("A confirmed incorrect relationship remains connected through the full graph")
    return after, confirmed


def verify_final_splits(directory, *, verify_exports=True):
    """Verify immutable policy and integrity evidence; never evaluate a model."""
    directory = Path(directory).resolve()
    completion = json.loads((directory / "completion.json").read_text(encoding="utf8"))
    freeze = json.loads((directory / "grouping_freeze.json").read_text(encoding="utf8"))
    verify_hashes({**freeze["source_sha256"], **freeze["input_sha256"]})
    for name, expected in completion["artifact_sha256"].items():
        if name == "test.csv" and not verify_exports:
            continue  # Development entry points never open even the test export.
        if sha256(directory / name) != expected:
            raise ValueError(f"Sealed split output changed: {name}")
    assignment = pd.read_csv(directory / "assignments.csv", dtype=str, keep_default_na=False,
                             usecols=["record_id", "group_id", "split"])
    check_assignments(assignment, assignment.record_id)
    protected = _csv(directory / "protected_final_test.csv")
    if set(protected.record_id) != set(assignment.loc[assignment.split.eq("test"), "record_id"]):
        raise ValueError("Reserved test identity changed")
    if protected.set_index("record_id").group_id.to_dict() != assignment.loc[assignment.split.eq("test")].set_index("record_id").group_id.to_dict():
        raise ValueError("Reserved test groups changed")
    signatures = _csv(Path(freeze["model_input_cache"]) / "record_signatures.csv")
    role_map = assignment.set_index("record_id").split
    signatures["split"] = signatures.record_id.map(role_map)
    if signatures.split.isna().any() or signatures.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Identical effective transformer inputs cross split boundaries")
    if verify_exports:
        for role in ("train", "validation", "test"):
            exported = _csv(directory / f"{role}.csv")
            expected = assignment.loc[assignment.split.eq(role), ["record_id", "group_id", "split"]]
            actual = exported[["record_id", "group_id", "split"]]
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    return json.loads((directory / "summary.json").read_text(encoding="utf8"))


def finalize_splits(input_path, output_dir, *, exposure_dir, model_cache, tokenizer_dir,
                    old_assignments, witness_path, fixture_path=None,
                    split_config=None, rule_config=None, model_config=None,
                    allocation_config=None, progress=print):
    started = perf_counter()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Use a new output directory; a sealed test cannot be overwritten")
    split_config = split_config or SplitConfig(grouping_version=2)
    rule_config = rule_config or RuleFinalConfig()
    model_config = model_config or ModelConfig()
    allocation_config = allocation_config or AllocationConfig(seed=split_config.seed,
        validation_fraction=split_config.ratios[1] / sum(split_config.ratios[:2]))
    fixture_path = Path(fixture_path or ROOT / "tests/fixtures/grouping_review_75.json")
    exposure_dir, model_cache = Path(exposure_dir).resolve(), Path(model_cache).resolve()
    frame, manifest = load_input(Path(input_path))
    ids = frame.record_id.to_numpy()
    signatures, cache_manifest = load_model_input_cache(model_cache, ids, model_config,
        tokenizer_dir=tokenizer_dir, expected_input_sha256=manifest["sha256"])
    eligible_path = exposure_dir / "eligible_strict_historical_family_record_ids.csv"
    eligible_ids = set(_csv(eligible_path).record_id)
    old = _csv(old_assignments)
    if old.record_id.duplicated().any() or set(old.record_id) != set(ids):
        raise ValueError("Original assignments must cover the identical input population")
    old_test = set(old.loc[old.split.eq("test"), "record_id"])
    inputs = {**manifest["fingerprints"], str(Path(old_assignments).resolve()): sha256(Path(old_assignments)),
              str(Path(witness_path).resolve()): sha256(Path(witness_path)), str(fixture_path.resolve()): sha256(fixture_path)}
    inputs.update({str(path): sha256(path) for folder in (exposure_dir, model_cache)
                   for path in sorted(folder.iterdir()) if path.is_file()})
    inputs.update({str(path): sha256(path) for path in sorted(Path(tokenizer_dir).resolve().iterdir())
                   if path.is_file()})
    exposure = json.loads((exposure_dir / "summary.json").read_text(encoding="utf8"))
    inputs.update({str(ROOT / name): expected for name, expected in exposure["input_file_sha256"].items()})
    verify_hashes(inputs)
    # This is written BEFORE grouping or choosing any final test records.
    freeze = {"version": VERSION, "frozen_utc": _now(), "target": "Segment",
        "source_sha256": _source_fingerprints(), "input_sha256": inputs,
        "split_config": split_config.to_dict(), "rule_final_config": rule_config.to_dict(),
        "development_allocation_config": allocation_config.to_dict(),
        "model_config": model_config.to_dict(), "model_view_contract": cache_manifest["model_view_contract"],
        "model_input_cache": str(model_cache), "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        "test_selection_policy": "All corrected components wholly within strict historical-family eligible old unscored test IDs. No label/outcome selection; no padding with model-seen records.",
        "eligibility_csv": str(eligible_path), "target_ratios": [0.70, 0.15, 0.15],
        "development_validation_fraction": allocation_config.validation_fraction,
        "stop_rule": "No grouping refinement using the reserved test. Develop models with training and validation only.",
        "representation_change_policy": "Input construction/tokenizer/sequence limits are frozen. A representation change requires a separate exposure-aware split protocol, not reuse of this seal."}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "grouping_freeze.json", freeze)
    progress("Grouping with frozen v4 rules and exact effective-input constraints")
    stage = perf_counter()
    product_groups, edges, grouping = refine_rule_groups_v4(frame, ids, split_config, rule_config)
    groups, equality = merge_model_input_groups(product_groups, ids, signatures)
    # Full-corpus order-invariance checks include alternative transitive paths.
    reverse = frame.iloc[::-1].reset_index(drop=True)
    other, _, _ = refine_rule_groups_v4(reverse, reverse.record_id.to_numpy(), split_config, rule_config)
    repeated_groups, _ = merge_model_input_groups(other, reverse.record_id, signatures[::-1])
    if dict(zip(ids, groups)) != dict(zip(reverse.record_id, repeated_groups)):
        raise AssertionError("Full-corpus grouping did not reproduce after row reversal")
    reviewed, fixed = _regression(frame, groups, old_assignments, witness_path, fixture_path)
    grouping_seconds = perf_counter() - stage
    verify_hashes({**freeze["source_sha256"], **inputs})
    mask, reservation = reserve_unexposed_groups(ids, groups, eligible_ids, old_test)
    reservation["reserved_utc"] = _now()
    reservation["grouping_freeze_sha256"] = sha256(output_dir / "grouping_freeze.json")
    _write_json(output_dir / "test_reservation.json", reservation)
    progress(f"Reserved {int(mask.sum()):,} unexposed test records; allocating development only")
    stage = perf_counter()
    assignment, allocation = allocate_remaining(frame, groups, mask, allocation_config)
    repeat, _ = allocate_remaining(frame, groups, mask, allocation_config)
    if not assignment.equals(repeat):
        raise AssertionError("Seeded development allocation did not reproduce")
    allocation_seconds = perf_counter() - stage
    progress("Running frozen independent duplicate and class-distribution checks")
    stage = perf_counter()
    pairs, evaluation = near_duplicate_pairs(frame, split_config, progress=progress)
    metrics = leakage_metrics(frame, assignment, pairs, split_config)
    effective = _exact_metrics(signatures, assignment.split.tolist(), "actual frozen transformer token arrays")
    if metrics["exact_payload"]["crossing_pairs"] or effective["crossing_pairs"]:
        raise ValueError("Exact payload or effective transformer inputs cross split boundaries")
    exposed = set(_csv(exposure_dir / "historical_family_exposed_records.csv").record_id)
    model_seen = set(_csv(exposure_dir / "historical_model_seen_records.csv").record_id)
    final_ids = set(ids[mask])
    if final_ids & (exposed | model_seen):
        raise ValueError("Reserved final test overlaps historical review or model exposure")
    audit_seconds = perf_counter() - stage
    exports = frame.copy()
    exports["group_id"], exports["split"] = groups, assignment.split
    for role in ("train", "validation", "test"):
        subset = exports.loc[exports.split.eq(role)].reset_index(drop=True)
        path = output_dir / f"{role}.csv"
        subset.to_csv(path, index=False, lineterminator="\n")
        pd.testing.assert_frame_equal(_csv(path), subset, check_dtype=False)
    # Public reservation contains IDs only, never names or target labels.
    assignment.loc[mask, ["record_id", "group_id", "split"]].to_csv(output_dir / "protected_final_test.csv", index=False)
    assignment.to_csv(output_dir / "assignments.csv", index=False)
    _edges_frame(edges, frame).to_csv(output_dir / "product_matching_edges.csv", index=False)
    pd.DataFrame({"record_id": ids, "product_group_id": product_groups,
                  "effective_input_group_id": signatures, "group_id": groups}).to_csv(output_dir / "grouping_assignments.csv", index=False)
    assignment.groupby("group_id").size().rename("rows").reset_index().to_csv(output_dir / "group_sizes.csv", index=False)
    segment_distribution(frame, assignment).to_csv(output_dir / "segment_distributions.csv", index=False)
    reviewed.to_csv(output_dir / "regression_75_outcomes.csv", index=False)
    fixed.to_csv(output_dir / "confirmed_four_connections.csv", index=False)
    audit = pd.DataFrame([{"left_record_id": ids[a], "right_record_id": ids[b], "name_jaccard": score,
        "left_split": assignment.at[a, "split"], "right_split": assignment.at[b, "split"],
        "crosses_split": assignment.at[a, "split"] != assignment.at[b, "split"]} for a, b, score in pairs])
    audit.to_csv(output_dir / "independent_near_pairs.csv", index=False)
    _write_json(output_dir / "input_manifest.json", manifest)
    _write_json(output_dir / "effective_input_merges.json", equality)
    summary = {"version": VERSION, "created_utc": _now(), "input": manifest,
        "grouping_freeze_sha256": reservation["grouping_freeze_sha256"],
        "grouping": grouping, "effective_input_union": {k: v for k, v in equality.items() if k != "merge_evidence"},
        "test_reservation": reservation, "development_allocation": allocation,
        "metrics": metrics, "effective_input_duplicates": effective,
        "independent_evaluation": evaluation,
        "supervised_split_counts": {role: int((assignment.split.eq(role) & ~assignment.Segment.str.strip().str.casefold().isin(("", "null", "none", "nan"))).sum()) for role in ("train", "validation", "test")},
        "regression": {"pairs": len(reviewed), "new_clear_errors": int(reviewed.new_clear_error.sum()),
                       "confirmed_incorrect_connections_separated": len(fixed)},
        "checks": {"complete_unique_coverage": True, "product_and_effective_input_group_isolation": True,
            "full_corpus_grouping_reproduces_after_row_reversal": True,
            "seeded_allocation_reproduces": True, "exact_payload_crossings_zero": True,
            "exact_effective_input_crossings_zero": True, "confirmed_four_connections_fixed": True,
            "previous_model_seen_test_overlap_zero": True, "historical_review_family_test_overlap_zero": True,
            "grouping_frozen_before_test_reservation": True, "export_values_preserved": True},
        "remaining_risks": [
            "Independent lexical matches are suspicious candidates, not confirmed leakage; zero exact crossings does not prove zero leakage.",
            "Broad named umbrellas remain intentionally unresolved; no arbitrary size cap was imposed.",
            "Reviewed regression pairs were selected, so their error rates are not population accuracy estimates.",
            "The test is unreviewed/model-unseen according to recorded history. Earlier automatic whole-universe audits and unknown unrecorded exposure cannot be undone.",
            "Some historical review artifacts presented Segment labels; affected records and historical connected families are excluded conservatively.",
            "Exposure exclusions change test composition. This is an in-catalog evaluation, not a demonstrated future-retailer or time-shift deployment estimate."],
        "historical_exposure_registry": {k: v for k, v in exposure.items() if k != "input_file_sha256"},
        "runtime_seconds": {"grouping_and_reproduction": grouping_seconds,
            "allocation_and_reproduction": allocation_seconds, "independent_audit": audit_seconds,
            "total": perf_counter() - started},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    verify_hashes({**freeze["source_sha256"], **inputs})
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "completion.json", {"completed_utc": _now(), "artifact_sha256": {
        path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}})
    verify_final_splits(output_dir)
    progress("Sealed final splits; subsequent model work must load only train and validation")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/training_cleaned.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "frozen")
    parser.add_argument("--exposure-dir", type=Path, default=DEFAULT_ROOT / "exposure")
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_ROOT / "model_view")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "data/processed/segment_full_development_20261009/group_aware/tokenizer")
    parser.add_argument("--old-assignments", type=Path, default=ROOT / "data/processed/split_rule_v3_20261008/rule_assignments.csv")
    parser.add_argument("--witnesses", type=Path, default=ROOT / "data/processed/split_leakage_audit_20261009/groups/reviewed_group_witnesses.csv")
    parser.add_argument("--config", type=Path, help="JSON with split, rule, model, and allocation configuration overrides")
    parser.add_argument("--verify", type=Path, help="Verify an existing seal; does not create a new test")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_final_splits(args.verify)
    else:
        values = json.loads(args.config.read_text(encoding="utf8")) if args.config else {}
        unknown = set(values) - {"split", "rule", "model", "allocation"}
        if unknown:
            parser.error(f"Unknown configuration sections: {sorted(unknown)}")
        result = finalize_splits(args.input, args.output_dir, exposure_dir=args.exposure_dir,
            model_cache=args.model_cache, tokenizer_dir=args.tokenizer_dir,
            old_assignments=args.old_assignments, witness_path=args.witnesses,
            split_config=SplitConfig.from_dict({"grouping_version": 2, **values.get("split", {})}),
            rule_config=RuleFinalConfig.from_dict(values.get("rule", {})),
            model_config=ModelConfig.from_dict(values.get("model", {})),
            allocation_config=AllocationConfig.from_dict({"seed": values.get("split", {}).get("seed", 42), **values.get("allocation", {})}))
    print(json.dumps({"checks": result["checks"], "splits": result["metrics"]["splits"],
                      "near_crossings": result["metrics"]["near_duplicate"]["crossing_pairs"]}, indent=2))
    return 0
