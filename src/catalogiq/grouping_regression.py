"""Regression checks for reviewed leakage grouping, independent of split allocation.

The retained cases are a development diagnostic set. Known misses are recorded,
not declared correct, and improvements are permitted. Passing these checks is
not evidence of accuracy on new products; use separately reserved examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .balanced_pair_sample import EVIDENCE_FIELDS, REVIEW_FIELDS
from .balanced_review_evaluation import validate_balanced_reviews
from .features import sha256


def load_review_fixture(path):
    """Return target-free product features and separate, immutable review labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported reviewed-grouping fixture")
    records = pd.DataFrame(payload["records"])
    allowed = {"record_id", *EVIDENCE_FIELDS}
    if set(records) != allowed or records.record_id.eq("").any() or records.record_id.duplicated().any():
        raise ValueError("fixture records must be unique and contain only permitted product features")
    pairs = pd.DataFrame(payload["pairs"])
    names = records.set_index("record_id").ProductName
    if not (set(pairs.left_record_id) | set(pairs.right_record_id)) <= set(names.index):
        raise ValueError("fixture pair references missing records")
    for side in ("left", "right"):
        pairs[f"{side}_ProductName"] = pairs[f"{side}_record_id"].map(names)
    validate_balanced_reviews(pairs, quota=payload["metadata"]["pairs_per_category"])
    if len(pairs) != payload["metadata"]["pairs"]:
        raise ValueError("review fixture coverage changed")
    # Reviews are never attached to the feature frame passed to a grouper.
    return records, pairs, payload["metadata"]


def _group_map(groups):
    if isinstance(groups, pd.DataFrame):
        if not {"record_id", "group_id"} <= set(groups):
            raise ValueError("group assignments require record_id and group_id")
        if groups.record_id.duplicated().any():
            raise ValueError("duplicate group-assignment records")
        groups = groups.set_index("record_id").group_id
    elif not isinstance(groups, pd.Series):
        groups = pd.Series(groups, dtype=object)
    if (groups.index.has_duplicates or groups.isna().any()
            or (groups.index.astype(str).str.strip() == "").any()
            or groups.astype(str).str.strip().eq("").any()):
        raise ValueError("invalid record-to-group mapping")
    groups = groups.copy()
    groups.index = groups.index.astype(str)
    if groups.index.has_duplicates:
        raise ValueError("duplicate normalized record IDs")
    return groups.astype(str)


def evaluate_review_groups(pairs, groups):
    """Score group membership only; target labels and split membership are absent."""
    validate_balanced_reviews(pairs)
    groups = _group_map(groups)
    if not (set(pairs.left_record_id) | set(pairs.right_record_id)) <= set(groups.index):
        raise ValueError("review endpoint absent from grouping")
    result = pairs.copy()
    expected = result.leakage_decision.map({"keep_together": True, "keep_separate": False}).astype("boolean")
    grouped = result.left_record_id.map(groups).eq(result.right_record_id.map(groups))
    result["grouped"] = grouped
    result["false_negative"] = (expected & ~grouped).where(expected.notna())
    result["false_positive"] = (~expected & grouped).where(expected.notna())
    result["reviewed_error"] = (expected.ne(grouped)).where(expected.notna())
    return result


def compare_review_regression(fixture_path, reference_path, candidates):
    """Permit known-error fixes, flag newly broken clear cases and boundary changes.

    The reference records frozen full-corpus outcomes, not desired truth. Never
    run TF-IDF on just these150 records and compare with that reference: corpus
    IDF, candidate blocks and transitive paths depend on the full input corpus.
    """
    _, pairs, metadata = load_review_fixture(fixture_path)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if sha256(Path(fixture_path)) != reference["fixture_sha256"]:
        raise ValueError("review fixture changed from the regression reference")
    if set(candidates) != set(reference["methods"]):
        raise ValueError("candidate methods differ from regression reference")
    rows, metrics = [], []
    for method, groups in candidates.items():
        scored = evaluate_review_groups(pairs, groups)
        previous = reference["methods"][method]
        if set(previous) != set(pairs.pair_id) or any(type(value) is not bool for value in previous.values()):
            raise ValueError("invalid reference pair outcomes")
        expected = scored.leakage_decision.map({"keep_together": True, "keep_separate": False}).astype("boolean")
        before_grouped = scored.pair_id.map(previous)
        before_error = expected.ne(before_grouped).where(expected.notna())
        decisive = scored.review_category.ne("borderline")
        scored["method"] = method
        scored["reference_grouped"] = before_grouped
        scored["reference_reviewed_error"] = before_error
        scored["new_clear_error"] = (decisive & before_error.eq(False) & scored.reviewed_error.eq(True)).fillna(False)
        scored["fixed_clear_error"] = (decisive & before_error.eq(True) & scored.reviewed_error.eq(False)).fillna(False)
        scored["borderline_connection_changed"] = ~decisive & before_grouped.ne(scored.grouped)
        rows.append(scored)
        for category in ("decisive_core", "borderline"):
            subset = scored.loc[decisive if category == "decisive_core" else ~decisive]
            metrics.append({"method": method, "review_category": category, "pairs": len(subset),
                            "false_negative_pairs": int(subset.false_negative.sum()),
                            "false_positive_pairs": int(subset.false_positive.sum()),
                            "unresolved_pairs": int(subset.reviewed_error.isna().sum()),
                            "new_clear_errors": int(subset.new_clear_error.sum()),
                            "fixed_clear_errors": int(subset.fixed_clear_error.sum()),
                            "borderline_connections_changed": int(subset.borderline_connection_changed.sum()),
                            "interpretation": metadata["interpretation"]})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def assert_no_new_clear_errors(outcomes):
    failed = outcomes.loc[outcomes.new_clear_error]
    if len(failed):
        cases = ", ".join(f"{row.method}:{row.pair_id}" for row in failed.itertuples())
        raise AssertionError(f"new grouping errors on previously satisfied clear review cases: {cases}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/grouping_review_75.json"))
    parser.add_argument("--reference", type=Path, default=Path("tests/fixtures/grouping_review_75_baseline.json"))
    parser.add_argument("--rule-assignments", type=Path, required=True)
    parser.add_argument("--tfidf-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite regression results: {args.output_dir}")
    candidates = {method: pd.read_csv(path, usecols=["record_id", "group_id"], dtype=str, keep_default_na=False)
                  for method, path in (("rule", args.rule_assignments), ("tfidf", args.tfidf_assignments))}
    outcomes, metrics = compare_review_regression(args.fixture, args.reference, candidates)
    args.output_dir.mkdir(parents=True)
    outcomes.to_csv(args.output_dir / "pair_regressions.csv", index=False)
    metrics.to_csv(args.output_dir / "regression_metrics.csv", index=False)
    print(metrics.drop(columns="interpretation").to_string(index=False))
    print("Selected reviewed regression cases; evaluate proposed improvements separately on new independent examples.")
    try:
        assert_no_new_clear_errors(outcomes)
    except AssertionError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
