"""Reproducible before/after comparisons with a fixed independent audit pool."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pandas as pd

from .split_review_validation import (
    METHODS, PAIR_COLUMNS, diagnostic_comparison, verify_holdout,
    verify_refinement_seal,
)


AUDIT_PARAMETERS = ("evaluation_threshold", "evaluation_shingle_size", "evaluation_min_name_chars")
DISPLAY_METRICS = ("groups", "singleton groups", "largest group", "train proportion",
                   "validation proportion", "test proportion", "exact payload crossing pairs",
                   "near duplicate crossing pairs", "near duplicate crossing pair rate",
                   "independent near duplicate co-grouped rate", "primary total seconds")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_audit(path):
    frame = pd.read_csv(path, usecols=[*PAIR_COLUMNS, "name_jaccard"], dtype=str, keep_default_na=False)
    result = {}
    for left, right, similarity in frame[list(PAIR_COLUMNS) + ["name_jaccard"]].itertuples(index=False, name=None):
        pair = tuple(sorted((left, right)))
        score = Decimal(similarity)
        if not left or not right or left == right or not score.is_finite() or not 0 <= score <= 1:
            raise ValueError("invalid independent audit identity or Jaccard similarity")
        if pair in result:
            raise ValueError("duplicate independent audit pair")
        result[pair] = score
    return result


def validate_runs(baseline_dir, refined_dir):
    """Require the same input records, split setup and independent evaluation.

    Decimal comparisons preserve the exact recorded Jaccard values. Pair order
    and endpoint orientation can differ without changing the evaluation pool.
    """
    baseline_dir, refined_dir = Path(baseline_dir), Path(refined_dir)
    baseline_manifest = _json(baseline_dir / "input_manifest.json")
    refined_manifest = _json(refined_dir / "input_manifest.json")
    if baseline_manifest != refined_manifest:
        raise ValueError("input manifests differ between baseline and refined run")
    baseline_config, refined_config = (_json(path / "config.json") for path in (baseline_dir, refined_dir))
    for key in (*AUDIT_PARAMETERS, "seed", "ratios"):
        if key not in baseline_config or baseline_config[key] != refined_config.get(key):
            raise ValueError(f"comparison parameter differs or is missing: {key}")

    ids = None
    for directory in (baseline_dir, refined_dir):
        for method in METHODS:
            frame = pd.read_csv(directory / f"{method}_assignments.csv", usecols=["record_id"],
                                dtype=str, keep_default_na=False)
            current = set(frame.record_id)
            if frame.record_id.eq("").any() or len(current) != len(frame):
                raise ValueError("assignment IDs must be nonempty and unique")
            if ids is None:
                ids = current
            elif current != ids:
                raise ValueError("baseline/refined methods do not have identical record IDs")
    if baseline_manifest.get("rows") != len(ids):
        raise ValueError("manifest row count does not match assignments")
    baseline_pairs = _canonical_audit(baseline_dir / "independent_near_duplicate_pairs.csv")
    refined_pairs = _canonical_audit(refined_dir / "independent_near_duplicate_pairs.csv")
    if baseline_pairs != refined_pairs:
        raise ValueError("independent audit pair IDs or exact Jaccard scores differ")
    if any(left not in ids or right not in ids for left, right in baseline_pairs):
        raise ValueError("independent audit references a record absent from assignments")
    canonical = [[left, right, str(score.normalize())] for (left, right), score in sorted(baseline_pairs.items())]
    return {
        "same_input_manifest": True, "same_record_ids": True,
        "same_independent_pair_ids_and_jaccards": True,
        "records": len(ids), "independent_pairs": len(baseline_pairs),
        "independent_pool_sha256": hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest(),
        "audit_parameters": {key: baseline_config[key] for key in AUDIT_PARAMETERS},
        "seed": baseline_config["seed"], "ratios": baseline_config["ratios"],
    }


def _aggregate_comparison(baseline_dir, refined_dir):
    frames = {}
    for phase, directory in (("baseline", baseline_dir), ("refined", refined_dir)):
        frame = pd.read_csv(Path(directory) / "comparison.csv")
        if frame.metric.duplicated().any():
            raise ValueError("aggregate comparison metrics must be unique")
        frames[phase] = frame.set_index("metric")
    metrics = list(frames["baseline"].index)
    metrics += [metric for metric in frames["refined"].index if metric not in metrics]
    result = pd.DataFrame(index=pd.Index(metrics, name="metric"))
    for method in METHODS:
        for phase in frames:
            result[f"{phase}_{method}"] = frames[phase][method]
        result[f"delta_{method}"] = result[f"refined_{method}"] - result[f"baseline_{method}"]
    return result.reset_index()


def _canonical_review_pairs(path):
    pairs = pd.read_csv(path, usecols=list(PAIR_COLUMNS), dtype=str, keep_default_na=False)
    canonical = [tuple(sorted(pair)) for pair in pairs[list(PAIR_COLUMNS)].itertuples(index=False, name=None)]
    if len(set(canonical)) != len(canonical):
        raise ValueError("review sample contains duplicate pairs")
    return set(canonical)


def _with_review_names(outcomes, review_csv):
    """Attach original reviewed names through a validated endpoint identity join.

    This is called for held-out annotations only after the freeze is verified.
    Identifier-only diagnostic scoring and reservation remain separate.
    """
    header = pd.read_csv(review_csv, nrows=0).columns
    names = [column for column in ("left_ProductName", "right_ProductName",
                                  "left_ProductBrand", "right_ProductBrand") if column in header]
    if not names:
        return outcomes
    details = pd.read_csv(review_csv, usecols=[*PAIR_COLUMNS, *names], dtype=str, keep_default_na=False)
    if details[list(PAIR_COLUMNS)].duplicated().any():
        raise ValueError("reviewed name attachment requires unique pair endpoints")
    merged = outcomes.merge(details, on=list(PAIR_COLUMNS), how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("reviewed names could not be joined to scored pair identities")
    return merged.drop(columns="_merge")


def compare_refinement(baseline_dir, refined_dir, diagnostic_csv, output_dir,
                       holdout_dir=None, heldout_annotations=None):
    """Save before/after results without choosing or ranking a winning method.

    A held-out annotation file is accepted only after the experiment is frozen,
    and must cover exactly the previously reserved sample. Its metrics are saved
    separately from inspected diagnostic annotations.
    """
    baseline_dir, refined_dir, diagnostic_csv, output_dir = map(
        Path, (baseline_dir, refined_dir, diagnostic_csv, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite refinement comparison: {output_dir}")
    verification = validate_runs(baseline_dir, refined_dir)
    if heldout_annotations is not None and holdout_dir is None:
        raise ValueError("held-out annotations require the original holdout directory and freeze")
    if holdout_dir is not None:
        heldout = verify_holdout(holdout_dir)
        verify_refinement_seal(refined_dir)
        verification["holdout_reservation_unchanged"] = True
        verification["refinement_freeze_unchanged"] = True
        verification["heldout_reserved_pairs"] = len(heldout)
    aggregate = _aggregate_comparison(baseline_dir, refined_dir)
    diagnostic_pairs, diagnostic_metrics = diagnostic_comparison(diagnostic_csv, baseline_dir, refined_dir)
    diagnostic_pairs = _with_review_names(diagnostic_pairs, diagnostic_csv)
    heldout_results = None
    if heldout_annotations is not None:
        reserved_pairs = _canonical_review_pairs(Path(holdout_dir) / "holdout_pairs.csv")
        annotated_pairs = _canonical_review_pairs(heldout_annotations)
        if annotated_pairs != reserved_pairs:
            raise ValueError("held-out annotation IDs must exactly match the sealed review sample")
        if annotated_pairs & _canonical_review_pairs(diagnostic_csv):
            raise ValueError("held-out annotations overlap inspected diagnostic pairs")
        heldout_results = diagnostic_comparison(heldout_annotations, baseline_dir, refined_dir)
        heldout_results = (_with_review_names(heldout_results[0], heldout_annotations), heldout_results[1])
        heldout_results[1]["evaluation_scope"] = "reserved held-out weak labels"
        verification["heldout_review_pairs"] = len(annotated_pairs)

    output_dir.mkdir(parents=True, exist_ok=False)
    aggregate.to_csv(output_dir / "aggregate_comparison.csv", index=False)
    diagnostic_pairs.to_csv(output_dir / "diagnostic_pair_outcomes.csv", index=False)
    diagnostic_metrics.to_csv(output_dir / "diagnostic_metrics.csv", index=False)
    if heldout_results is not None:
        heldout_results[0].to_csv(output_dir / "heldout_pair_outcomes.csv", index=False)
        heldout_results[1].to_csv(output_dir / "heldout_metrics.csv", index=False)
    segment_paths = [path / "segment_distributions.csv" for path in (baseline_dir, refined_dir)]
    if all(path.is_file() for path in segment_paths):
        segments = [pd.read_csv(path) for path in segment_paths]
        combined = segments[0].merge(segments[1], on=["approach", "Segment"],
                                     how="outer", suffixes=("_baseline", "_refined"), validate="one_to_one")
        combined.to_csv(output_dir / "segment_comparison.csv", index=False)
    verification["interpretation"] = "Selected reviewed pairs are weak diagnostic labels; rates are sample diagnostics, not population accuracy or precision."
    input_paths = [diagnostic_csv, *[path / name for path in (baseline_dir, refined_dir)
                   for name in ("input_manifest.json", "config.json", "comparison.csv",
                                "rule_assignments.csv", "tfidf_assignments.csv", "independent_near_duplicate_pairs.csv")]]
    if heldout_annotations is not None:
        input_paths.append(Path(heldout_annotations))
    verification["input_sha256"] = {str(path.resolve()): _hash(path) for path in input_paths}
    (output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return aggregate, diagnostic_metrics, heldout_results[1] if heldout_results is not None else None


def _display(aggregate, metrics, label):
    print(label)
    if aggregate is not None:
        shown = aggregate.loc[aggregate.metric.isin(DISPLAY_METRICS),
                              ["metric", "baseline_rule", "refined_rule", "baseline_tfidf", "refined_tfidf"]]
        print(shown.to_string(index=False, float_format=lambda number: f"{number:.5g}"))
    overall = metrics.loc[metrics.review_scope.eq("all") & metrics.confidence.eq("all"),
                          ["phase", "method", "related_pairs", "unrelated_pairs", "uncertain_or_unknown_pairs",
                           "false_negative_pairs", "false_positive_pairs"]]
    print(overall.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--refined", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-dir", type=Path)
    parser.add_argument("--heldout-annotations", type=Path)
    arguments = parser.parse_args(argv)
    aggregate, diagnostic, heldout = compare_refinement(
        arguments.baseline, arguments.refined, arguments.diagnostic, arguments.output,
        arguments.holdout_dir, arguments.heldout_annotations)
    _display(aggregate, diagnostic, "Baseline/refined aggregate and inspected diagnostic pairs")
    if heldout is not None:
        _display(None, heldout, "Separate held-out review sample")
    print(f"Saved comparison: {arguments.output}")


if __name__ == "__main__":
    main()
