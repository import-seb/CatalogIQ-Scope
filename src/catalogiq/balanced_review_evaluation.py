"""Evaluate a selected, balanced challenge set against frozen split assignments.

This analysis never constructs groups or changes a split. Selected pair error
fractions describe this challenge set, not population accuracy or prevalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .features import sha256
from .pair_reassessment import IDENTITIES, DETAILS, LEAKAGE, CONFIDENCE
from .split_review_validation import METHODS, PAIR_COLUMNS, _assignments, verify_refinement_seal


CATEGORIES = ("related", "safe_to_separate", "borderline")
CAUTION = ("Purposively selected, balanced challenge pairs. Fractions describe these reviewed "
           "pairs only; they do not estimate population accuracy, error prevalence, or precision. "
           "Judgments are qualitative text-based assessments, not independently verified product identity.")


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_balanced_reviews(frame, registry=None, quota=None):
    required = {"pair_id", *PAIR_COLUMNS, "review_category", "product_identity", "identity_detail",
                "identity_confidence", "identity_rationale", "leakage_decision", "leakage_confidence",
                "leakage_rationale", "judgment_uses_Segment", "judgment_uses_method_outcomes",
                "left_ProductName", "right_ProductName"}
    if not required <= set(frame):
        raise ValueError(f"missing balanced-review columns: {sorted(required - set(frame))}")
    if frame.empty or frame.pair_id.eq("").any() or frame.pair_id.duplicated().any():
        raise ValueError("review pair IDs must be nonempty and unique")
    keys = [tuple(sorted(x)) for x in frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None)]
    if len(set(keys)) != len(keys) or any(not a or not b or a == b for a, b in keys):
        raise ValueError("review endpoints must be distinct, unique, and nonempty")
    for field, values in (("review_category", CATEGORIES), ("product_identity", IDENTITIES),
                          ("identity_detail", DETAILS), ("identity_confidence", CONFIDENCE),
                          ("leakage_decision", LEAKAGE), ("leakage_confidence", CONFIDENCE)):
        if not frame[field].isin(values).all():
            raise ValueError(f"invalid {field}")
    for field in ("identity_rationale", "leakage_rationale", "left_ProductName", "right_ProductName"):
        if frame[field].str.strip().eq("").any():
            raise ValueError(f"empty {field}")
    if not frame.judgment_uses_Segment.astype(str).str.casefold().eq("false").all():
        raise ValueError("Segment must not inform reviews")
    if not frame.judgment_uses_method_outcomes.astype(str).str.casefold().eq("false").all():
        raise ValueError("method outcomes must not inform reviews")
    if frame.loc[frame.product_identity.eq("same_product"), "leakage_decision"].eq("keep_separate").any():
        raise ValueError("same product cannot be safe to separate")
    for category, decision in (("related", "keep_together"), ("safe_to_separate", "keep_separate")):
        selected = frame.loc[frame.review_category.eq(category)]
        if not (selected.leakage_decision.eq(decision) & selected.leakage_confidence.eq("high")).all():
            raise ValueError(f"{category} requires a clear high-confidence leakage judgment")
    borderline = frame.loc[frame.review_category.eq("borderline")]
    if (borderline.leakage_decision.ne("uncertain") & borderline.leakage_confidence.eq("high")).any():
        raise ValueError("borderline cases must retain review ambiguity")
    if quota is not None and any(int(frame.review_category.eq(c).sum()) != quota for c in CATEGORIES):
        raise ValueError("balanced sample category counts differ from the required quota")
    if registry is not None:
        if registry.pair_id.duplicated().any() or set(frame.pair_id) != set(registry.pair_id):
            raise ValueError("selected review registry differs")
        ordered = registry.set_index("pair_id").loc[frame.pair_id]
        for column in (*PAIR_COLUMNS, "review_category", "left_ProductName", "right_ProductName"):
            if list(frame[column]) != list(ordered[column]):
                raise ValueError(f"review registry changed: {column}")


def score_balanced_reviews(frame, assignments):
    """Score decisive and borderline strata separately without using identity labels."""
    validate_balanced_reviews(frame)
    if set(assignments) != set(METHODS):
        raise ValueError("both rule and tfidf assignments are required")
    outcomes = frame.copy()
    expected = frame.leakage_decision.map({"keep_together": True, "keep_separate": False}).astype("boolean")
    outcomes["expected_together"] = expected
    common_ids = None
    for method in METHODS:
        assignment = assignments[method]
        if (assignment.index.has_duplicates or (assignment.index.astype(str) == "").any()
                or assignment.group_id.eq("").any()
                or not assignment.split.isin(("train", "validation", "test")).all()):
            raise ValueError("invalid split assignments")
        ids = set(assignment.index)
        if common_ids is not None and ids != common_ids:
            raise ValueError("methods must contain the same input records")
        common_ids = ids
        if not (set(frame.left_record_id) | set(frame.right_record_id)) <= ids:
            raise ValueError("review record missing from assignments")
        if assignment.groupby("group_id").split.nunique().gt(1).any():
            raise ValueError("group crosses split boundaries")
        grouped = frame.left_record_id.map(assignment.group_id).eq(frame.right_record_id.map(assignment.group_id))
        crosses = frame.left_record_id.map(assignment.split).ne(frame.right_record_id.map(assignment.split))
        sizes = assignment.groupby("group_id").size()
        for side in ("left", "right"):
            groups = frame[f"{side}_record_id"].map(assignment.group_id)
            outcomes[f"{method}_{side}_group_id"] = groups
            outcomes[f"{method}_{side}_group_size"] = groups.map(sizes)
        outcomes[f"{method}_grouped"] = grouped
        outcomes[f"{method}_crosses_split"] = crosses
        outcomes[f"{method}_false_negative"] = (expected & ~grouped).where(expected.notna())
        outcomes[f"{method}_false_positive"] = (~expected & grouped).where(expected.notna())
        outcomes[f"{method}_actual_leakage"] = (expected & crosses).where(expected.notna())

    metrics = []
    for category in ("decisive_core", *CATEGORIES):
        selected = outcomes.loc[outcomes.review_category.ne("borderline") if category == "decisive_core"
                                else outcomes.review_category.eq(category)]
        positive = selected.expected_together.eq(True).fillna(False)
        negative = selected.expected_together.eq(False).fillna(False)
        unknown = selected.expected_together.isna()
        for method in METHODS:
            grouped = selected[f"{method}_grouped"]
            fn = int((~grouped.loc[positive]).sum())
            fp = int(grouped.loc[negative].sum())
            crossing = int(selected.loc[positive, f"{method}_actual_leakage"].sum())
            metrics.append({
                "review_category": category, "method": method, "pairs": len(selected),
                "keep_together_pairs": int(positive.sum()), "keep_separate_pairs": int(negative.sum()),
                "uncertain_pairs": int(unknown.sum()), "true_positive_pairs": int(grouped.loc[positive].sum()),
                "true_negative_pairs": int((~grouped.loc[negative]).sum()),
                "false_negative_pairs": fn, "false_positive_pairs": fp,
                "selected_positive_miss_fraction": fn / positive.sum() if positive.any() else None,
                "selected_negative_overgroup_fraction": fp / negative.sum() if negative.any() else None,
                "actual_cross_split_leakage_pairs": crossing,
                "missed_pairs_coincidentally_same_split": fn - crossing,
                "uncertain_pairs_grouped": int(grouped.loc[unknown].sum()),
                "uncertain_pairs_separated": int((~grouped.loc[unknown]).sum()),
                "interpretation": CAUTION,
            })
    # Unknown labels are not forced. Enumerate possible error totals for this
    # stratum if all unresolved pairs are adjudicated either way; not CIs.
    sensitivity = []
    borderline = outcomes.loc[outcomes.review_category.eq("borderline")]
    for method in METHODS:
        unknown = borderline.expected_together.isna()
        grouped = borderline[f"{method}_grouped"]
        sensitivity.append({
            "method": method, "borderline_pairs": len(borderline), "unresolved_pairs": int(unknown.sum()),
            "false_negative_lower_bound": int(borderline[f"{method}_false_negative"].sum()),
            "false_negative_upper_bound": int(borderline[f"{method}_false_negative"].sum()) + int((~grouped.loc[unknown]).sum()),
            "false_positive_lower_bound": int(borderline[f"{method}_false_positive"].sum()),
            "false_positive_upper_bound": int(borderline[f"{method}_false_positive"].sum()) + int(grouped.loc[unknown].sum()),
            "note": "Logical adjudication bounds for these pairs, not confidence intervals; extremes need not occur jointly.",
        })
    return outcomes, pd.DataFrame(metrics), pd.DataFrame(sensitivity)


def adjudicate_secondary_reviews(primary, secondary):
    """Preserve honest ambiguity from an independent second text review.

    No method predictions are used. Opposite leakage judgments remain
    unresolved; confidence disagreement retains the more cautious confidence.
    Identity disagreement never changes the leakage decision on its own.
    """
    from .balanced_pair_sample import _validate_gold

    _validate_gold(primary)
    _validate_gold(secondary)
    if not set(secondary.pair_id) <= set(primary.pair_id):
        raise ValueError("secondary pair absent from primary review")
    result = primary.copy().set_index("pair_id")
    agreement = []
    rank = {"low": 0, "medium": 1, "high": 2}
    for row in secondary.to_dict("records"):
        original = result.loc[row["pair_id"]].copy()
        if set(original[list(PAIR_COLUMNS)]) != {row[c] for c in PAIR_COLUMNS}:
            raise ValueError("secondary review endpoints differ")
        identity_agrees = original.product_identity == row["product_identity"]
        leakage_agrees = original.leakage_decision == row["leakage_decision"]
        agreement.append({"pair_id": row["pair_id"], **{c: original[c] for c in PAIR_COLUMNS},
                          "identity_agrees": identity_agrees, "leakage_agrees": leakage_agrees,
                          **{f"primary_{c}": original[c] for c in
                             ("product_identity", "identity_detail", "identity_rationale", "leakage_decision", "leakage_confidence", "leakage_rationale", "review_category")},
                          **{f"secondary_{c}": row[c] for c in
                             ("product_identity", "identity_detail", "identity_rationale", "leakage_decision", "leakage_confidence", "leakage_rationale", "review_category")}})
        if not identity_agrees:
            result.loc[row["pair_id"], ["product_identity", "identity_detail", "identity_confidence"]] = ["uncertain", "uncertain", "medium"]
            result.at[row["pair_id"], "identity_rationale"] = (
                "Independent identity reviews disagree. Primary: " + original.identity_rationale
                + " Secondary: " + row["identity_rationale"])
        if not leakage_agrees:
            result.loc[row["pair_id"], ["leakage_decision", "leakage_confidence", "review_category"]] = ["uncertain", "medium", "borderline"]
            result.at[row["pair_id"], "leakage_rationale"] = (
                "Independent leakage reviews disagree; no forced decision. Primary: " + original.leakage_rationale
                + " Secondary: " + row["leakage_rationale"])
        elif "borderline" in (original.review_category, row["review_category"]):
            confidence = min((original.leakage_confidence, row["leakage_confidence"]), key=rank.get)
            result.loc[row["pair_id"], ["leakage_confidence", "review_category"]] = [confidence, "borderline"]
            result.at[row["pair_id"], "leakage_rationale"] = (
                "Independent reviews retain boundary ambiguity. Primary: " + original.leakage_rationale
                + " Secondary: " + row["leakage_rationale"])
    result = result.reset_index()
    _validate_gold(result)
    return result, pd.DataFrame(agreement)


def freeze_balanced_reviews(review_dir, annotation_paths, *, per_category=30, seed=20261011):
    """Register explicit reviews and freeze deterministic selection before scoring.

    This consumes independently produced annotations. It neither infers labels
    from lexical seeds nor reads saved method outcomes to choose the sample.
    Every candidate is retained in all_candidate_judgments.csv, including pairs
    not selected for the balanced challenge set.
    """
    from .balanced_pair_sample import select_balanced_review

    review_dir = Path(review_dir)
    protocol_path = review_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol["category_quota"] != per_category or protocol["seed"] != seed):
        raise ValueError("selection differs from pre-review policy")
    if (review_dir / "judgment_freeze.json").exists():
        raise FileExistsError("refusing to replace frozen judgments")
    if sha256(review_dir / "candidate_protocol.json") != protocol["candidate_protocol_sha256"]:
        raise ValueError("candidate protocol changed")
    for name, digest in protocol["candidate_artifact_sha256"].items():
        if sha256(review_dir / name) != digest:
            raise ValueError(f"candidate evidence changed: {name}")
    for name, digest in protocol.get("annotation_source_sha256", {}).items():
        if sha256(review_dir / name) != digest:
            raise ValueError(f"independent source review changed: {name}")
    pool = _read(review_dir / "candidate_pool.csv")
    paths = [Path(path) for path in annotation_paths]
    for path in paths:
        if not path.resolve().is_relative_to(review_dir.resolve()):
            raise ValueError("annotation files must be retained in the review directory")
    review = pd.concat([pd.DataFrame(json.loads(path.read_text(encoding="utf-8-sig")))
                        for path in paths], ignore_index=True)
    if set(review.pair_id) != set(pool.pair_id) or len(review) != len(pool):
        raise ValueError("review must cover the entire registered candidate pool exactly once")
    selected = select_balanced_review(pool, review, per_category=per_category, seed=seed)
    validate_balanced_reviews(selected, quota=per_category)
    extra_review_columns = [c for c in review.columns if c not in pool.columns]
    full = pool.merge(review[["pair_id", *extra_review_columns]], on="pair_id", validate="one_to_one", sort=False)
    full["selected_for_balanced_sample"] = full.pair_id.isin(selected.pair_id)
    full.to_csv(review_dir / "all_candidate_judgments.csv", index=False)
    selected.to_csv(review_dir / "annotated_balanced_pairs.csv", index=False)
    registry_columns = ["pair_id", *PAIR_COLUMNS, "review_category", "left_ProductName", "right_ProductName", "selection_rank"]
    selected[registry_columns].to_csv(review_dir / "selection_registry.csv", index=False)
    names = ["protocol.json", "candidate_protocol.json", *protocol["candidate_artifact_sha256"],
             *protocol.get("annotation_source_sha256", {}),
             "all_candidate_judgments.csv", "annotated_balanced_pairs.csv", "selection_registry.csv"]
    hashes = {name: sha256(review_dir / name) for name in names}
    for path in paths:
        try:
            name = str(path.resolve().relative_to(review_dir.resolve()))
        except ValueError:
            raise ValueError("annotation files must be retained in the review directory") from None
        hashes[name] = sha256(path)
    seal = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "pre_score_freeze": True,
        "selection_seed": seed, "category_quota": per_category, "candidate_pairs": len(review),
        "reviewed_category_counts": review.review_category.value_counts().sort_index().to_dict(),
        "selected_category_counts": selected.review_category.value_counts().sort_index().to_dict(),
        "selection_uses_method_outcomes": False, "review_file_sha256": hashes,
        "interpretation": CAUTION,
    }
    (review_dir / "judgment_freeze.json").write_text(json.dumps(seal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected, seal


def compare_balanced_sample(review_dir, run_dir, output_dir):
    review_dir, run_dir, output_dir = map(Path, (review_dir, run_dir, output_dir))
    if output_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("analysis output must be outside the frozen run")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output_dir}")
    verify_refinement_seal(run_dir)
    protocol = json.loads((review_dir / "protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((review_dir / "judgment_freeze.json").read_text(encoding="utf-8"))
    for name, digest in freeze["review_file_sha256"].items():
        if sha256(review_dir / name) != digest:
            raise ValueError(f"review changed after pre-score freeze: {name}")
    for path, digest in protocol["input_record_fingerprints"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"source input changed: {path}")
    if protocol.get("no_method_refinement") is not True:
        raise ValueError("review must evaluate frozen implementations")
    if sha256(run_dir / "refinement_freeze.json") != protocol["method_freeze_sha256"]:
        raise ValueError("wrong frozen method version")
    frame = _read(review_dir / "annotated_balanced_pairs.csv")
    registry = _read(review_dir / "selection_registry.csv")
    validate_balanced_reviews(frame, registry, protocol["category_quota"])
    outcomes, metrics, sensitivity = score_balanced_reviews(frame, _assignments(run_dir))
    output_dir.mkdir(parents=True)
    outcomes.to_csv(output_dir / "pair_outcomes.csv", index=False)
    metrics.to_csv(output_dir / "selected_sample_metrics.csv", index=False)
    sensitivity.to_csv(output_dir / "borderline_sensitivity.csv", index=False)
    columns = ["pair_id", "review_category", "left_ProductName", "right_ProductName",
               "product_identity", "leakage_decision", "leakage_confidence", "leakage_rationale",
               *[f"{method}_{field}" for method in METHODS for field in
                 ("grouped", "false_negative", "false_positive", "crosses_split", "left_group_size", "right_group_size")]]
    outcomes[columns].to_csv(output_dir / "product_comparison.csv", index=False)
    counts = outcomes.groupby(["review_category", "product_identity", "leakage_decision"]).size().rename("pairs").reset_index()
    counts.to_csv(output_dir / "identity_vs_leakage_counts.csv", index=False)
    verify_refinement_seal(run_dir)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "interpretation": CAUTION,
        "records": int(len(_assignments(run_dir)["rule"])), "review_pairs": len(frame),
        "category_counts": frame.review_category.value_counts().sort_index().to_dict(),
        "no_method_refinement": True, "method_freeze_sha256": protocol["method_freeze_sha256"],
        "evaluator_source_sha256": sha256(Path(__file__)),
        "review_file_sha256": freeze["review_file_sha256"],
        "result_sha256": {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))},
        "selected_sample_metrics": json.loads(metrics.to_json(orient="records")),
        "borderline_sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = compare_balanced_sample(args.review_dir, args.run_dir, args.output_dir)
    display = pd.DataFrame(summary["selected_sample_metrics"])
    print(display[["review_category", "method", "pairs", "keep_together_pairs", "keep_separate_pairs",
                   "uncertain_pairs", "false_negative_pairs", "false_positive_pairs", "actual_cross_split_leakage_pairs"]].to_string(index=False))
    print(CAUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
