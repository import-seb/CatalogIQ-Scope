"""Reserve unseen review candidates without examining method correctness.

Saved method group IDs and independent near-pair IDs are used only to exclude
previously reviewed families and to keep new cases component-disjoint. Candidate
retrieval then uses the existing independent name/brand lexical sampler. No
target values, split assignments, prior judgments or method outcomes select or
label the reserved pairs. Full product evidence is saved only for later blinded
review, after a future method version has been frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from .balanced_pair_sample import (
    BUCKETS, CandidateConfig, EVIDENCE_FIELDS, PAIR_COLUMNS, blind_evidence,
    candidate_pool,
)
from .features import sha256
from .paths import ROOT_PATH
from .split_review_validation import verify_refinement_seal
from .splitting import Components
from .splitting_experiment import load_input

SEED = 20261012
METHODS = ("rule", "tfidf")
REGISTRY_COLUMNS = ["pair_id", *PAIR_COLUMNS, "provisional_stratum",
                    "left_exclusion_component", "right_exclusion_component"]
CAUTION = ("Unreviewed purposively retrieved lexical challenge candidates. Provisional "
           "strata are not identity or leakage labels; this reservation supplies no "
           "accuracy, prevalence or method-comparison estimates.")


def _rank(seed, *values):
    return hashlib.sha256((str(seed) + "\0" + "\0".join(map(str, values))).encode()).hexdigest()


def _read_pairs(path):
    return pd.read_csv(path, usecols=PAIR_COLUMNS, dtype=str, keep_default_na=False)


def _validate_ids(values, name):
    values = list(map(str, values))
    if not values or any(not value.strip() for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must have unique nonempty record IDs")
    return set(values)


def union_component_map(groupings, independent_pairs):
    """Conservative family union; only record/group IDs can affect this map."""
    if set(groupings) != set(METHODS):
        raise ValueError("both rule and tfidf saved groupings are required")
    allowed = {}
    for method in METHODS:
        frame = groupings[method]
        if not {"record_id", "group_id"} <= set(frame):
            raise ValueError(f"{method} grouping requires record_id and group_id")
        projected = frame[["record_id", "group_id"]].astype(str).copy()
        _validate_ids(projected.record_id, method)
        if projected.group_id.str.strip().eq("").any():
            raise ValueError("group IDs must be nonempty")
        allowed[method] = projected
    ids = sorted(allowed["rule"].record_id)
    if set(ids) != set(allowed["tfidf"].record_id):
        raise ValueError("saved methods must cover identical records")
    positions = {key: index for index, key in enumerate(ids)}
    components = Components(len(ids))
    for method in METHODS:
        for _, members in allowed[method].groupby("group_id", sort=True):
            keys = sorted(members.record_id)
            for key in keys[1:]:
                components.union(positions[keys[0]], positions[key])
    if not set(PAIR_COLUMNS) <= set(independent_pairs):
        raise ValueError("independent pairs require both endpoints")
    for left, right in independent_pairs[PAIR_COLUMNS].astype(str).itertuples(index=False, name=None):
        if left not in positions or right not in positions or left == right:
            raise ValueError("independent pair has unknown or identical endpoints")
        components.union(positions[left], positions[right])
    return dict(zip(ids, components.groups(ids, "union")))


def _select_component_independent(pool, component_map, *, seed, per_bucket):
    required = {*PAIR_COLUMNS, "provisional_stratum"}
    if not required <= set(pool):
        raise ValueError("candidate pool lacks registered endpoints or lexical strata")
    # Scores, names, group predictions and labels cannot reach this selector.
    rows = pool[[*PAIR_COLUMNS, "provisional_stratum"]].to_dict(orient="records")
    queues = {}
    for bucket in BUCKETS:
        eligible = [row for row in rows if row["provisional_stratum"] == bucket]
        queues[bucket] = iter(sorted(eligible, key=lambda row: _rank(seed, *[row[c] for c in PAIR_COLUMNS])))
    used, selected, exhausted = set(), [], set()
    counts = Counter()
    while len(exhausted) < len(BUCKETS):
        for bucket in BUCKETS:
            if bucket in exhausted:
                continue
            if counts[bucket] >= per_bucket:
                exhausted.add(bucket)
                continue
            for row in queues[bucket]:
                endpoints = {component_map[row[c]] for c in PAIR_COLUMNS}
                if endpoints & used:
                    continue
                selected.append(row)
                used.update(endpoints)
                counts[bucket] += 1
                break
            else:
                exhausted.add(bucket)
    selected.sort(key=lambda row: _rank(seed, "neutral_future_pair", *[row[c] for c in PAIR_COLUMNS]))
    for index, row in enumerate(selected, 1):
        row["pair_id"] = f"F{index:03d}"
        row["left_exclusion_component"] = component_map[row["left_record_id"]]
        row["right_exclusion_component"] = component_map[row["right_record_id"]]
    return pd.DataFrame(selected, columns=REGISTRY_COLUMNS)


def reserve_from_frames(frame, prior_pairs, groupings, independent_pairs, *, seed=SEED,
                        per_bucket=20, max_oversample_factor=4):
    """Pure deterministic reservation; no judgments or predictions are consumed."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if type(per_bucket) is not int or per_bucket < 1:
        raise ValueError("per-bucket quota must be positive")
    if type(max_oversample_factor) is not int or max_oversample_factor < 1:
        raise ValueError("maximum oversample factor must be positive")
    required = {"record_id", "ProductName", "ProductBrand"}
    if not required <= set(frame):
        raise ValueError("input requires record IDs, names and brands")
    # Project before any retrieval, including when a caller supplies targets.
    # Build absent optional fields explicitly for pandas 3 string-block support.
    source = pd.DataFrame({field: frame[field] if field in frame else ""
                           for field in ["record_id", *EVIDENCE_FIELDS]},
                          index=frame.index).fillna("").astype(str)
    ids = _validate_ids(source.record_id, "input")
    component_map = union_component_map(groupings, independent_pairs)
    if ids != set(component_map):
        raise ValueError("saved groups must cover exactly the input records")
    if not set(PAIR_COLUMNS) <= set(prior_pairs):
        raise ValueError("prior reviews require both endpoint columns")
    prior = prior_pairs[PAIR_COLUMNS].astype(str)
    prior_ids = set(prior.left_record_id) | set(prior.right_record_id)
    if any(not value.strip() for value in prior_ids) or not prior_ids <= ids:
        raise ValueError("prior review contains missing or unknown endpoints")
    excluded_components = {component_map[key] for key in prior_ids}
    excluded_ids = {key for key, component in component_map.items() if component in excluded_components}
    remaining = source.loc[~source.record_id.isin(excluded_ids)].copy()
    empty_prior = pd.DataFrame(columns=PAIR_COLUMNS)
    attempts = []
    factors = [1]
    while factors[-1] < max_oversample_factor:
        factors.append(min(factors[-1] * 2, max_oversample_factor))
    registry = pd.DataFrame(columns=REGISTRY_COLUMNS)
    pool = pd.DataFrame()
    for factor in factors:
        config = replace(CandidateConfig(), seed=seed, per_bucket=per_bucket * factor)
        pool, retrieval = candidate_pool(remaining, empty_prior, config)
        registry = _select_component_independent(pool, component_map, seed=seed, per_bucket=per_bucket)
        counts = registry.provisional_stratum.value_counts().to_dict()
        attempts.append({"oversample_factor": factor, "retrieval": retrieval,
                         "selected_counts": counts})
        if all(counts.get(bucket, 0) == per_bucket for bucket in BUCKETS):
            break
    selected_pairs = {(row.left_record_id, row.right_record_id) for row in registry.itertuples(index=False)}
    pool_by_pair = pool.set_index(PAIR_COLUMNS)
    evidence_rows = []
    for row in registry.itertuples(index=False):
        evidence = pool_by_pair.loc[(row.left_record_id, row.right_record_id)].to_dict()
        evidence.update({"pair_id": row.pair_id, "left_record_id": row.left_record_id,
                         "right_record_id": row.right_record_id})
        evidence_rows.append(evidence)
    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        evidence = pd.DataFrame(columns=["pair_id", *PAIR_COLUMNS,
            *(f"{side}_{field}" for side in ("left", "right") for field in EVIDENCE_FIELDS)])
    blind = blind_evidence(evidence)
    used_components = []
    for row in registry.itertuples(index=False):
        used_components.extend({row.left_exclusion_component, row.right_exclusion_component})
    counts = registry.provisional_stratum.value_counts().to_dict()
    proof = {
        "no_prior_endpoint_overlap": not (set(registry.left_record_id) | set(registry.right_record_id)) & prior_ids,
        "no_prior_union_component_overlap": not set(used_components) & excluded_components,
        "no_union_component_reused_between_pairs": len(used_components) == len(set(used_components)),
        "no_duplicate_selected_pairs": len(selected_pairs) == len(registry),
        "labels_and_method_correctness_excluded_from_retrieval": True,
        "no_new_pairs_reviewed_or_scored": True,
    }
    if not all(proof.values()):
        raise RuntimeError("reservation independence proof failed")
    report = {"protocol": "independent-unreviewed-reservation-v1", "seed": seed,
              "per_lexical_bucket": per_bucket, "max_oversample_factor": max_oversample_factor,
              "input_records": len(source), "prior_review_rows": len(prior),
              "prior_unique_pairs": len({tuple(sorted(pair)) for pair in prior.itertuples(index=False, name=None)}),
              "prior_unique_endpoints": len(prior_ids), "union_components": len(set(component_map.values())),
              "excluded_union_components": len(excluded_components), "excluded_records": len(excluded_ids),
              "remaining_records": len(remaining), "reserved_pairs": len(registry),
              "reserved_union_components": len(set(used_components)), "selected_counts": counts,
              "quota_filled": all(counts.get(bucket, 0) == per_bucket for bucket in BUCKETS),
              "retrieval_attempts": attempts, "proof": proof, "interpretation": CAUTION}
    return registry, blind, report


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def reserve_independent_review(run_dir, prior_paths, output_dir, *, seed=SEED,
                               per_bucket=20, max_oversample_factor=4):
    """Save a sealed reservation; refuse overwrites and verify repeated retrieval."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    prior_paths = list(map(Path, prior_paths))
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an independent review reservation")
    if not prior_paths:
        raise ValueError("at least one complete prior-review registry is required")
    verify_refinement_seal(run_dir)
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    input_path = Path(run_summary["input"]["path"])
    identifier = run_summary["input"].get("identifier_source")
    frame, metadata = load_input(input_path, Path(identifier) if identifier else None)
    groupings = {method: pd.read_csv(run_dir / f"{method}_assignments.csv",
                                    usecols=["record_id", "group_id"], dtype=str, keep_default_na=False)
                 for method in METHODS}
    near_path = run_dir / "independent_near_duplicate_pairs.csv"
    near = _read_pairs(near_path)
    prior = pd.concat([_read_pairs(path) for path in prior_paths], ignore_index=True)
    source_paths = [run_dir / f"{method}_assignments.csv" for method in METHODS]
    source_paths += [near_path, run_dir / "refinement_freeze.json", *prior_paths,
                     *[Path(path) for path in metadata["fingerprints"]]]
    fingerprints = {str(path.resolve()): sha256(path) for path in source_paths}
    kwargs = {"seed": seed, "per_bucket": per_bucket, "max_oversample_factor": max_oversample_factor}
    registry, blind, report = reserve_from_frames(frame, prior, groupings, near, **kwargs)
    repeated_registry, repeated_blind, repeated_report = reserve_from_frames(frame, prior, groupings, near, **kwargs)
    if not registry.equals(repeated_registry) or blind != repeated_blind or report != repeated_report:
        raise RuntimeError("independent reservation does not reproduce")
    verify_refinement_seal(run_dir)
    for path, expected in fingerprints.items():
        if sha256(Path(path)) != expected:
            raise ValueError("reservation input changed during retrieval")
    report["proof"].update({"repeated_retrieval_identical": True, "frozen_methods_unchanged": True,
                             "all_input_fingerprints_unchanged": True})
    code_paths = [Path(__file__), Path(__file__).with_name("balanced_pair_sample.py"),
                  ROOT_PATH / "scripts" / "reserve_independent_review.py"]
    report["source_sha256"] = fingerprints
    report["code_sha256"] = {str(path.resolve()): sha256(path) for path in code_paths}
    report["frozen_run"] = str(run_dir.resolve())
    report["prior_registry_sources"] = [str(path.resolve()) for path in prior_paths]
    protocol = {
        "status": "reserved_unreviewed; no future-method outcomes inspected",
        "seed": seed, "retrieval_fields": ["ProductName", "ProductBrand"],
        "exclusion_evidence": "union of saved rule groups, TF-IDF groups and fixed independent near-pair IDs",
        "prior_exclusions": "all endpoints in every supplied prior-review registry plus every record in their union components",
        "case_independence": "one selected pair per union component globally; reserve both components for a pair spanning two",
        "strata": "provisional lexical retrieval strata, never gold labels or method-correctness strata",
        "review_gate": "Keep blind product evidence unreviewed until a new grouping implementation/version is frozen. Then obtain independent identity and evaluation-leakage judgments without labels or method outcomes; freeze judgments before scoring that version.",
        "no_current_evaluation": "Do not annotate, inspect names manually, join outcomes or score accuracy from this reservation now.",
        "interpretation": CAUTION,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    registry.to_csv(output_dir / "reserved_pairs.csv", index=False, lineterminator="\n")
    _write_json(output_dir / "blind_pairs.json", blind)
    _write_json(output_dir / "protocol.json", protocol)
    report["output_sha256"] = {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    _write_json(output_dir / "summary.json", report)
    return report


def verify_reservation(output_dir):
    output_dir = Path(output_dir)
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    verify_refinement_seal(Path(report["frozen_run"]))
    for path, expected in report["source_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("reservation source changed")
    for path, expected in report["code_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("reservation code changed")
    for name, expected in report["output_sha256"].items():
        if sha256(output_dir / name) != expected:
            raise ValueError("reserved artifact changed")
    if not all(report["proof"].values()):
        raise ValueError("reservation proof is incomplete")
    return {"artifacts_verified": True, "frozen_methods_unchanged": True,
            "unreviewed_reservation": True, "reserved_pairs": report["reserved_pairs"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reserve outcome-blind, component-disjoint future review candidates.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prior-review", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--max-oversample-factor", type=int, default=4)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_reservation(args.verify), sort_keys=True))
        return 0
    if not args.run_dir or not args.output_dir or not args.prior_review:
        parser.error("--run-dir, --output-dir and at least one --prior-review are required")
    report = reserve_independent_review(args.run_dir, args.prior_review, args.output_dir,
                                        seed=args.seed, per_bucket=args.per_bucket,
                                        max_oversample_factor=args.max_oversample_factor)
    # Print only counts and proofs; full future product names stay uninspected.
    print(json.dumps({key: report[key] for key in ("reserved_pairs", "selected_counts", "quota_filled",
                      "excluded_records", "excluded_union_components", "proof")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
