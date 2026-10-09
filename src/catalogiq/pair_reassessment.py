"""Score saved product groups against independent identity/leakage reviews.

This changes review criteria only. It never changes product groups, splitting,
target labels, source records, or the independent text-leakage evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .features import sha256
from .split_refinement_comparison import validate_runs
from .split_review_validation import (
    METHODS, PAIR_COLUMNS, RELATED_ANNOTATIONS, UNRELATED_ANNOTATIONS,
    _assignments, verify_holdout, verify_refinement_seal,
)


IDENTITIES = {"same_product", "different_product", "uncertain"}
DETAILS = {"identical_item", "pack_size_variant", "strength_variant",
           "flavor_style_variant", "formulation_variant", "distinct_product", "uncertain"}
LEAKAGE = {"keep_together", "keep_separate", "uncertain"}
CONFIDENCE = {"high", "medium", "low"}
REQUIRED = ["pair_id", "review_set", *PAIR_COLUMNS, "product_identity", "identity_detail",
            "identity_confidence", "identity_rationale", "leakage_decision",
            "leakage_confidence", "leakage_rationale", "judgment_uses_Segment"]


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _canonical(frame):
    return [tuple(sorted(pair)) for pair in frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None)]


def validate_annotations(frame, registry=None):
    """Reject ambiguous encodings and mismatched or incomplete review samples."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"missing dual-review columns: {sorted(missing)}")
    if frame.empty or frame.pair_id.eq("").any() or frame.pair_id.duplicated().any():
        raise ValueError("pair IDs must be nonempty and unique")
    pairs = _canonical(frame)
    if len(set(pairs)) != len(pairs) or any(not a or not b or a == b for a, b in pairs):
        raise ValueError("review endpoints must be unique nonempty distinct pairs")
    for field, allowed in (("review_set", {"diagnostic", "heldout"}),
                           ("product_identity", IDENTITIES), ("identity_detail", DETAILS),
                           ("identity_confidence", CONFIDENCE), ("leakage_decision", LEAKAGE),
                           ("leakage_confidence", CONFIDENCE)):
        if not frame[field].isin(allowed).all():
            raise ValueError(f"invalid {field}")
    for field in ("identity_rationale", "leakage_rationale"):
        if frame[field].str.strip().eq("").any():
            raise ValueError(f"empty {field}")
    if not frame.judgment_uses_Segment.astype(str).str.casefold().eq("false").all():
        raise ValueError("judgments must exclude Segment labels")
    if frame.loc[frame.product_identity.eq("same_product"), "leakage_decision"].eq("keep_separate").any():
        raise ValueError("same-product identity contradicts keep_separate; adjudicate before scoring")
    if registry is not None:
        required = ["pair_id", "review_set", *PAIR_COLUMNS]
        if not set(required) <= set(registry):
            raise ValueError("registry is incomplete")
        registered = registry[required].copy()
        if registered.pair_id.duplicated().any() or set(registered.pair_id) != set(frame.pair_id):
            raise ValueError("review pair IDs differ from registry")
        reference = registered.set_index("pair_id").loc[frame.pair_id]
        if (list(reference.review_set) != list(frame.review_set)
                or _canonical(reference) != pairs):
            raise ValueError("review cohorts or endpoints differ from registry")


def score_annotations(frame, phases):
    """Return pair outcomes and metrics using ONLY the leakage decision.

    A grouping false negative is a keep-together pair assigned to different
    groups, even when those groups happen to land in the same split. Actual
    cross-split leakage is reported separately. Unknown decisions never enter
    either error denominator. Identity confidence does not filter leakage.
    """
    validate_annotations(frame)
    result = frame.copy()
    expected = result.leakage_decision.map({"keep_together": True, "keep_separate": False}).astype("boolean")
    result["expected_together"] = expected
    previous = None
    for phase, assignments in phases.items():
        if set(assignments) != set(METHODS):
            raise ValueError("each phase must contain rule and tfidf assignments")
        for method, assignment in assignments.items():
            if (assignment.index.has_duplicates or (assignment.index.astype(str) == "").any()
                    or assignment.group_id.eq("").any()
                    or not assignment.split.isin(("train", "validation", "test")).all()):
                raise ValueError("invalid assignments")
            ids = set(assignment.index)
            if previous is not None and ids != previous:
                raise ValueError("runs must contain identical record IDs")
            previous = ids
            if not (set(result.left_record_id) | set(result.right_record_id)) <= ids:
                raise ValueError("review endpoint absent from assignments")
            if assignment.groupby("group_id").split.nunique().gt(1).any():
                raise ValueError("group isolation failed")
            grouped = result.left_record_id.map(assignment.group_id).eq(result.right_record_id.map(assignment.group_id))
            crosses = result.left_record_id.map(assignment.split).ne(result.right_record_id.map(assignment.split))
            prefix = f"{phase}_{method}"
            result[f"{prefix}_grouped"] = grouped
            result[f"{prefix}_crosses_split"] = crosses
            result[f"{prefix}_false_negative"] = (expected & ~grouped).where(expected.notna())
            result[f"{prefix}_false_positive"] = (~expected & grouped).where(expected.notna())
            result[f"{prefix}_actual_leakage"] = (expected & crosses).where(expected.notna())

    metrics = []
    # Keep cohorts separate; intentionally do not pool the diagnostic/holdout sample.
    for review_set in sorted(result.review_set.unique()):
        cohort = result.loc[result.review_set.eq(review_set)]
        for confidence in ["all", *sorted(cohort.leakage_confidence.unique())]:
            subset = cohort if confidence == "all" else cohort.loc[cohort.leakage_confidence.eq(confidence)]
            positive = subset.expected_together.eq(True).fillna(False)
            negative = subset.expected_together.eq(False).fillna(False)
            for phase, assignments in phases.items():
                for method in assignments:
                    prefix = f"{phase}_{method}"
                    fn = int(subset.loc[positive, f"{prefix}_false_negative"].sum())
                    fp = int(subset.loc[negative, f"{prefix}_false_positive"].sum())
                    crossing = int(subset.loc[positive, f"{prefix}_actual_leakage"].sum())
                    metrics.append({
                        "review_set": review_set, "leakage_confidence": confidence,
                        "phase": phase, "method": method, "pairs": len(subset),
                        "keep_together_pairs": int(positive.sum()), "keep_separate_pairs": int(negative.sum()),
                        "uncertain_pairs": int(subset.expected_together.isna().sum()),
                        "true_positive_pairs": int(subset.loc[positive, f"{prefix}_grouped"].sum()),
                        "true_negative_pairs": int((~subset.loc[negative, f"{prefix}_grouped"]).sum()),
                        "false_negative_pairs": fn, "false_positive_pairs": fp,
                        "false_negative_rate": fn / positive.sum() if positive.any() else None,
                        "false_positive_rate": fp / negative.sum() if negative.any() else None,
                        "actual_cross_split_leakage_pairs": crossing,
                        "missed_group_pairs_coincidentally_in_same_split": fn - crossing,
                        "different_products_requiring_grouping": int((subset.product_identity.eq("different_product") & positive).sum()),
                    })
    return result, pd.DataFrame(metrics)


def previous_criterion_comparison(outcomes, originals):
    """Explain old-to-new errors without calling the old rule strict identity."""
    combined = pd.concat([frame.assign(review_set=cohort) for cohort, frame in originals.items()], ignore_index=True)
    old = {pair: row for pair, (_, row) in zip(_canonical(combined), combined.iterrows())}
    if len(old) != len(combined) or set(old) != set(_canonical(outcomes)):
        raise ValueError("original reviewed pair coverage differs")
    result = outcomes.copy()
    records = [old[key] for key in _canonical(result)]
    if any(row.review_set != review_set for row, review_set in zip(records, result.review_set)):
        raise ValueError("original review cohort differs")
    result["previous_relation"] = [row.relation for row in records]
    result["previous_confidence"] = [row.confidence for row in records]
    result["previous_expected_together"] = result.previous_relation.map(
        lambda value: True if value in RELATED_ANNOTATIONS else False if value in UNRELATED_ANNOTATIONS else pd.NA).astype("boolean")
    comparisons = []
    for cohort in sorted(result.review_set.unique()):
        subset = result.loc[result.review_set.eq(cohort)]
        for phase in ("baseline", "refined"):
            for method in METHODS:
                prefix = f"{phase}_{method}"
                for criterion, expected in (("previous_review", subset.previous_expected_together),
                                             ("evaluation_leakage", subset.expected_together)):
                    positive, negative = expected.eq(True).fillna(False), expected.eq(False).fillna(False)
                    comparisons.append({
                        "review_set": cohort, "phase": phase, "method": method, "criterion": criterion,
                        "keep_together_pairs": int(positive.sum()), "keep_separate_pairs": int(negative.sum()),
                        "uncertain_pairs": int(expected.isna().sum()),
                        "false_negative_pairs": int((~subset.loc[positive, f"{prefix}_grouped"]).sum()),
                        "false_positive_pairs": int(subset.loc[negative, f"{prefix}_grouped"].sum()),
                    })
    return result, pd.DataFrame(comparisons)


def compare_saved_reviews(review_dir, baseline_dir, refined_dir, holdout_dir, output_dir):
    review_dir, baseline_dir, refined_dir, holdout_dir, output_dir = map(
        Path, (review_dir, baseline_dir, refined_dir, holdout_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reassessment: {output_dir}")
    verify_refinement_seal(refined_dir)
    reserved = verify_holdout(holdout_dir)
    verification = validate_runs(baseline_dir, refined_dir)
    protocol_path = review_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for path, expected in {**protocol["input_sha256"], **protocol["input_record_fingerprints"]}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"review source changed: {path}")
    for name, expected in protocol["blind_file_sha256"].items():
        if sha256(review_dir / name) != expected:
            raise ValueError(f"blind evidence changed: {name}")
    registry_path = review_dir / "pair_registry.csv"
    if sha256(registry_path) != protocol["registry_sha256"]:
        raise ValueError("pair registry changed")
    annotation_path = review_dir / "dual_judgments.csv"
    annotation_seal = json.loads((review_dir / "judgment_freeze.json").read_text(encoding="utf-8"))
    if sha256(annotation_path) != annotation_seal["annotations_sha256"]:
        raise ValueError("judgments changed after pre-score freeze")
    if sha256(protocol_path) != annotation_seal["protocol_sha256"]:
        raise ValueError("review policy changed after pre-score freeze")
    for name, expected in annotation_seal.get("source_review_sha256", {}).items():
        if sha256(review_dir / name) != expected:
            raise ValueError(f"primary blind review changed: {name}")
    if "boundary_review_sha256" in annotation_seal:
        if sha256(review_dir / "boundary_review.json") != annotation_seal["boundary_review_sha256"]:
            raise ValueError("boundary review changed after judgment freeze")
    annotations, registry = _read(annotation_path), _read(registry_path)
    validate_annotations(annotations, registry)
    heldout = annotations.loc[annotations.review_set.eq("heldout")]
    if set(_canonical(heldout)) != set(_canonical(reserved)):
        raise ValueError("held-out reassessment differs from originally reserved sample")
    for cohort, count_key in (("diagnostic", "diagnostic_pairs"), ("heldout", "heldout_pairs")):
        if int(annotations.review_set.eq(cohort).sum()) != protocol[count_key]:
            raise ValueError("review cohort count differs from policy")
    phases = {"baseline": _assignments(baseline_dir), "refined": _assignments(refined_dir)}
    outcomes, metrics = score_annotations(annotations, phases)
    original_frames = {}
    for path in protocol["input_sha256"]:
        original = _read(path)
        cohort = "heldout" if set(_canonical(original)) == set(_canonical(reserved)) else "diagnostic"
        if cohort in original_frames:
            raise ValueError("ambiguous original review sources")
        original_frames[cohort] = original
    outcomes, criterion_comparison = previous_criterion_comparison(outcomes, original_frames)
    verify_refinement_seal(refined_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    outcomes.to_csv(output_dir / "pair_outcomes.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "leakage_metrics.csv", index=False, lineterminator="\n")
    criterion_comparison.to_csv(output_dir / "previous_vs_leakage_criterion.csv", index=False, lineterminator="\n")
    counts = annotations.groupby(["review_set", "product_identity", "leakage_decision"]).size().rename("pairs").reset_index()
    counts.to_csv(output_dir / "identity_vs_leakage_counts.csv", index=False, lineterminator="\n")
    # Deterministic examples of policy changes and remaining errors, with full names.
    reasons = {}
    for cohort in sorted(outcomes.review_set.unique()):
        sample = outcomes.loc[outcomes.review_set.eq(cohort)]
        selections = {"different products that should stay together": sample.product_identity.eq("different_product") & sample.expected_together.fillna(False)}
        for method in METHODS:
            selections[f"remaining {method} false negative"] = sample[f"refined_{method}_false_negative"].fillna(False)
            selections[f"remaining {method} false positive"] = sample[f"refined_{method}_false_positive"].fillna(False)
            selections[f"{method} old false positive now appropriate"] = sample.previous_expected_together.eq(False).fillna(False) & sample.expected_together.fillna(False) & sample[f"refined_{method}_grouped"]
        for reason, mask in selections.items():
            chosen = sample.loc[mask].assign(rank=lambda x: x.leakage_confidence.map({"high": 0, "medium": 1, "low": 2})).sort_values(["rank", "pair_id"]).head(2)
            for pair_id in chosen.pair_id:
                reasons.setdefault(pair_id, []).append(reason)
    examples = outcomes.loc[outcomes.pair_id.isin(reasons)].copy()
    examples["example_types"] = examples.pair_id.map(lambda key: "; ".join(reasons[key]))
    example_columns = ["pair_id", "review_set", "original_review_scope", "original_review_id", "example_types",
                       "left_ProductName", "right_ProductName", "product_identity", "identity_detail", "identity_confidence",
                       "identity_rationale", "leakage_decision", "leakage_confidence", "leakage_rationale",
                       *[f"{phase}_{method}_{status}" for phase in phases for method in METHODS for status in ("grouped", "crosses_split")]]
    examples[example_columns].to_csv(output_dir / "small_product_comparison.csv", index=False, lineterminator="\n")
    overall = metrics.loc[metrics.leakage_confidence.eq("all")].copy()
    overall.to_csv(output_dir / "comparison.csv", index=False, lineterminator="\n")
    verification.update({
        "created_utc": datetime.now(timezone.utc).isoformat(), "judgments_frozen_utc": annotation_seal["frozen_utc"],
        "grouping_and_splitting_unchanged": True, "previous_reviews_unchanged": True,
        "false_negative_definition": "keep_together pair in different groups, including groups coincidentally in the same split",
        "false_positive_definition": "keep_separate pair in the same group",
        "unknown_policy": "identity and leakage unknown independently; only leakage unknown excluded from error counts",
        "interpretation": protocol["interpretation"], "review_policy": protocol,
        "scoring_code_sha256": sha256(Path(__file__)),
        "dual_annotations_sha256": sha256(annotation_path),
        "output_sha256": {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
    })
    (output_dir / "summary.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return outcomes, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--refined", required=True, type=Path)
    parser.add_argument("--holdout-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _, metrics = compare_saved_reviews(args.review_dir, args.baseline, args.refined, args.holdout_dir, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(metrics.loc[metrics.leakage_confidence.eq("all"),
        ["review_set", "phase", "method", "keep_together_pairs", "keep_separate_pairs", "uncertain_pairs",
         "false_negative_pairs", "false_positive_pairs", "actual_cross_split_leakage_pairs"]].to_string(index=False))
    print(f"Saved dual-judgment comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
