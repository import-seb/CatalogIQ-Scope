"""Reserve an unseen review set and compare weak diagnostic pair annotations.

Reservation reads identifiers, baseline assignments and baseline edges only. It
never reads product text or Segment, and never consults refined assignments.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import pandas as pd


METHODS = ("rule", "tfidf")
STRATA = ("rule_only_grouped", "tfidf_only_grouped", "both_grouped",
          "independent_ungrouped")
PAIR_COLUMNS = ("left_record_id", "right_record_id")
ASSIGNMENT_COLUMNS = ("record_id", "group_id", "split")
PROTOCOL = {
    "version": 1,
    "candidate_source": "original baseline accepted matching edges and independent near pairs",
    "family_components": "union of every baseline rule group, TF-IDF group and independent near pair",
    "exclude": "whole components touching either endpoint of any diagnostic annotation",
    "sampling": "sorted canonical pairs, seeded shuffle in fixed stratum order; one pair per component globally",
    "blindness": "identifier-only reservation; no product attributes, Segment, judgments or refined outcomes",
    "interpretation": "stratified diagnostic review sample, not a population accuracy estimate",
}
HOLDOUT_COLUMNS = [*PAIR_COLUMNS, "baseline_stratum", "baseline_component_id",
                   *[f"baseline_{method}_{status}" for method in METHODS
                     for status in ("grouped", "crosses_split")]]
RELATED_ANNOTATIONS = {"same_product_likely", "related_variant_likely", "related", "same_product", "related_variant"}
UNRELATED_ANNOTATIONS = {"unrelated_likely", "unrelated"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path, columns):
    return pd.read_csv(path, usecols=list(columns), dtype=str, keep_default_na=False)


def _assignments(directory):
    result = {}
    for method in METHODS:
        frame = _read(Path(directory) / f"{method}_assignments.csv", ASSIGNMENT_COLUMNS)
        if frame.record_id.eq("").any() or frame.record_id.duplicated().any():
            raise ValueError(f"{method} assignments require unique nonempty record IDs")
        if frame.group_id.eq("").any() or not frame.split.isin(("train", "validation", "test")).all():
            raise ValueError(f"invalid {method} group or split")
        if frame.groupby("group_id").split.nunique().gt(1).any():
            raise ValueError(f"{method} group crosses splits")
        result[method] = frame.set_index("record_id")
    if set(result["rule"].index) != set(result["tfidf"].index):
        raise ValueError("methods must have identical record IDs")
    return result


class _Components:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}

    def find(self, key):
        if key not in self.parent:
            raise ValueError(f"unknown record ID in pair: {key}")
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            next_key = self.parent[key]
            self.parent[key] = root
            key = next_key
        return root

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            low, high = sorted((left, right))
            self.parent[high] = low


def _pairs(frame):
    return {tuple(sorted((left, right)))
            for left, right in frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
            if left != right}


def reserve_holdout(baseline_dir, diagnostic_path, output_dir, seed=20261008,
                    pairs_per_stratum=16):
    """Write an immutable identifier-only holdout selected from the original run.

    Existing output directories are rejected, including empty ones. All baseline
    records in a known family containing a diagnostic endpoint are ineligible.
    This conservative separation can leave fewer than the requested pairs.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(pairs_per_stratum) is not int or pairs_per_stratum < 1:
        raise ValueError("pairs_per_stratum must be a positive integer")
    baseline_dir, diagnostic_path, output_dir = map(Path, (baseline_dir, diagnostic_path, output_dir))
    if not baseline_dir.is_dir() or not diagnostic_path.is_file():
        raise ValueError("baseline directory and diagnostic CSV must exist")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite held-out reservation: {output_dir}")

    assignments = _assignments(baseline_dir)
    ids = set(assignments["rule"].index)
    components = _Components(ids)
    for frame in assignments.values():
        for group in frame.groupby("group_id", sort=True):
            members = list(group[1].index)
            for member in members[1:]:
                components.join(members[0], member)

    near_path = baseline_dir / "independent_near_duplicate_pairs.csv"
    near_pairs = _pairs(_read(near_path, PAIR_COLUMNS))
    for pair in sorted(near_pairs):
        components.join(*pair)
    diagnostic = _read(diagnostic_path, PAIR_COLUMNS)
    diagnostic_ids = set(diagnostic.left_record_id) | set(diagnostic.right_record_id)
    if not diagnostic_ids <= ids:
        raise ValueError("diagnostic pair contains records absent from baseline")
    excluded = {components.find(key) for key in diagnostic_ids}
    component_map = {key: components.find(key) for key in ids}
    all_components = set(component_map.values())

    edge_paths = [baseline_dir / f"{method}_matching_edges.csv" for method in METHODS]
    edge_pairs = set()
    for path in edge_paths:
        edge_pairs.update(_pairs(_read(path, PAIR_COLUMNS)))
    candidates = defaultdict(set)
    for left, right in sorted(edge_pairs | near_pairs):
        if left not in ids or right not in ids:
            raise ValueError("candidate pair contains records absent from baseline")
        rule_same = assignments["rule"].at[left, "group_id"] == assignments["rule"].at[right, "group_id"]
        tfidf_same = assignments["tfidf"].at[left, "group_id"] == assignments["tfidf"].at[right, "group_id"]
        if (left, right) in edge_pairs:
            if rule_same and tfidf_same:
                candidates["both_grouped"].add((left, right))
            elif rule_same:
                candidates["rule_only_grouped"].add((left, right))
            elif tfidf_same:
                candidates["tfidf_only_grouped"].add((left, right))
        if (left, right) in near_pairs and not rule_same and not tfidf_same:
            candidates["independent_ungrouped"].add((left, right))

    generator, used, selected, counts = random.Random(seed), set(), [], {}
    for stratum in STRATA:
        pairs = sorted(candidates[stratum])
        eligible = [pair for pair in pairs if component_map[pair[0]] not in excluded]
        generator.shuffle(eligible)
        eligible_components = {component_map[pair[0]] for pair in eligible}
        chosen = 0
        for left, right in eligible:
            component = component_map[left]
            if component in used:
                continue
            used.add(component)
            row = {"left_record_id": left, "right_record_id": right,
                   "baseline_stratum": stratum, "baseline_component_id": component}
            for method in METHODS:
                frame = assignments[method]
                row[f"baseline_{method}_grouped"] = frame.at[left, "group_id"] == frame.at[right, "group_id"]
                row[f"baseline_{method}_crosses_split"] = frame.at[left, "split"] != frame.at[right, "split"]
            selected.append(row)
            chosen += 1
            if chosen == pairs_per_stratum:
                break
        counts[stratum] = {"candidate_pairs": len(pairs), "eligible_pairs": len(eligible),
                           "eligible_components": len(eligible_components), "selected_pairs": chosen}

    holdout = pd.DataFrame(selected, columns=HOLDOUT_COLUMNS)
    input_paths = [baseline_dir / f"{method}_assignments.csv" for method in METHODS]
    input_paths += [*edge_paths, near_path, diagnostic_path]
    summary = {
        "protocol": PROTOCOL, "seed": seed, "pairs_per_stratum": pairs_per_stratum,
        "records": len(ids), "baseline_components": len(all_components),
        "diagnostic_pairs": len(diagnostic), "diagnostic_record_endpoints": len(diagnostic_ids),
        "excluded_components": len(excluded),
        "excluded_records": sum(value in excluded for value in component_map.values()),
        "eligible_components": len(all_components - excluded),
        "selected_pairs": len(selected), "selected_components": len(used), "strata": counts,
        "input_sha256": {str(path.resolve()): _sha256(path) for path in input_paths},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "holdout_pairs.csv"
    holdout.to_csv(output_path, index=False, lineterminator="\n")
    summary["holdout_pairs_sha256"] = _sha256(output_path)
    summary["protocol_sha256"] = hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True).encode("utf-8")).hexdigest()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def verify_holdout(holdout_dir):
    """Verify the reserved sample and original inputs have not changed.

    Returns identifier-only rows; callers must keep these sealed until their
    proposed methods and parameters are frozen.
    """
    holdout_dir = Path(holdout_dir)
    summary = json.loads((holdout_dir / "summary.json").read_text(encoding="utf-8"))
    pairs_path = holdout_dir / "holdout_pairs.csv"
    if _sha256(pairs_path) != summary["holdout_pairs_sha256"]:
        raise ValueError("held-out pair file changed after reservation")
    protocol_hash = hashlib.sha256(json.dumps(summary["protocol"], sort_keys=True).encode("utf-8")).hexdigest()
    if protocol_hash != summary["protocol_sha256"]:
        raise ValueError("held-out protocol changed after reservation")
    for path, expected in summary["input_sha256"].items():
        if _sha256(path) != expected:
            raise ValueError(f"reservation input changed: {path}")
    pairs = pd.read_csv(pairs_path, dtype=str, keep_default_na=False)
    if list(pairs.columns) != HOLDOUT_COLUMNS:
        raise ValueError("held-out file includes unexpected attributes")
    if (len(pairs) != summary["selected_pairs"]
            or pairs.baseline_component_id.duplicated().any()
            or pairs[list(PAIR_COLUMNS)].duplicated().any()
            or not pairs.baseline_stratum.isin(STRATA).all()):
        raise ValueError("invalid held-out identities or strata")
    return pairs


def diagnostic_comparison(diagnostic_csv, baseline_dir, refined_dir):
    """Return (pair outcomes, diagnostic metrics) without reading product text.

    Related variants count as related. Unknown/uncertain annotations are retained
    in outputs and excluded from error-rate denominators. Scopes and confidence
    strata are reported separately. These intentionally selected annotations are
    weak diagnostic labels, not a representative accuracy or precision estimate.
    """
    columns = ["review_id", "review_scope", "relation", "confidence", *PAIR_COLUMNS]
    header = pd.read_csv(diagnostic_csv, nrows=0).columns
    if "review_id" not in header:
        columns.remove("review_id")
    pairs = _read(diagnostic_csv, columns)
    if "review_id" not in pairs:
        pairs.insert(0, "review_id", [str(value) for value in range(len(pairs))])
    pairs["review_scope"] = pairs.review_scope.replace("", "unspecified")
    pairs["confidence"] = pairs.confidence.replace("", "unspecified")
    expected = pd.Series(pd.array(pairs.relation.map(
        lambda value: True if value in RELATED_ANNOTATIONS
        else False if value in UNRELATED_ANNOTATIONS else pd.NA), dtype="boolean"), index=pairs.index)
    pairs["expected_related"] = expected
    phases = {"baseline": _assignments(baseline_dir), "refined": _assignments(refined_dir)}
    baseline_ids = set(phases["baseline"]["rule"].index)
    if baseline_ids != set(phases["refined"]["rule"].index):
        raise ValueError("baseline and refined runs must use the same records")
    if not (set(pairs.left_record_id) | set(pairs.right_record_id)) <= baseline_ids:
        raise ValueError("diagnostic endpoint missing from split assignments")
    for phase, methods in phases.items():
        for method, frame in methods.items():
            grouped = pairs.left_record_id.map(frame.group_id).eq(pairs.right_record_id.map(frame.group_id))
            crosses = pairs.left_record_id.map(frame.split).ne(pairs.right_record_id.map(frame.split))
            prefix = f"{phase}_{method}"
            pairs[f"{prefix}_grouped"] = grouped
            pairs[f"{prefix}_crosses_split"] = crosses
            pairs[f"{prefix}_false_negative"] = (expected & ~grouped).where(expected.notna())
            pairs[f"{prefix}_false_positive"] = (~expected & grouped).where(expected.notna())

    metrics = []
    for scope in ["all", *sorted(pairs.review_scope.unique())]:
        scope_pairs = pairs if scope == "all" else pairs.loc[pairs.review_scope.eq(scope)]
        for confidence in ["all", *sorted(scope_pairs.confidence.unique())]:
            subset = scope_pairs if confidence == "all" else scope_pairs.loc[scope_pairs.confidence.eq(confidence)]
            related = subset.expected_related.eq(True).fillna(False)
            unrelated = subset.expected_related.eq(False).fillna(False)
            for phase in phases:
                for method in METHODS:
                    prefix = f"{phase}_{method}"
                    fn = int(subset.loc[related, f"{prefix}_false_negative"].sum())
                    fp = int(subset.loc[unrelated, f"{prefix}_false_positive"].sum())
                    metrics.append({
                        "evaluation_scope": "selected diagnostic weak labels",
                        "review_scope": scope, "confidence": confidence,
                        "phase": phase, "method": method, "pair_count": len(subset),
                        "related_pairs": int(related.sum()), "unrelated_pairs": int(unrelated.sum()),
                        "uncertain_or_unknown_pairs": int(subset.expected_related.isna().sum()),
                        "false_negative_pairs": fn, "false_positive_pairs": fp,
                        "false_negative_rate_on_reviewed_related": fn / related.sum() if related.any() else None,
                        "false_positive_rate_on_reviewed_unrelated": fp / unrelated.sum() if unrelated.any() else None,
                        "reviewed_related_cross_split_pairs": int(subset.loc[related, f"{prefix}_crosses_split"].sum()),
                    })
    return pairs, pd.DataFrame(metrics)


def seal_refinement(refined_dir, holdout_dir, code_paths=()):
    """Freeze completed experimental assignments/config before held-out review.

    The root orchestrator calls this before reading names for held-out IDs. A
    rerun or parameter change invalidates the seal; it must be treated as another
    experiment and requires a new unseen review set for confirmatory assessment.
    """
    refined_dir, holdout_dir = Path(refined_dir), Path(holdout_dir)
    seal_path = refined_dir / "refinement_freeze.json"
    if seal_path.exists():
        raise FileExistsError("refinement already frozen; refusing overwrite")
    verify_holdout(holdout_dir)
    _assignments(refined_dir)
    paths = [refined_dir / f"{method}_assignments.csv" for method in METHODS]
    for name in ("config.json", "input_manifest.json", "summary.json", "independent_near_duplicate_pairs.csv"):
        path = refined_dir / name
        if not path.is_file():
            raise ValueError(f"completed experimental artifact missing: {name}")
        paths.append(path)
    paths += [holdout_dir / "summary.json", holdout_dir / "holdout_pairs.csv"]
    paths += [Path(path) for path in code_paths]
    seal = {
        "status": "frozen_before_heldout_product_review",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Review sealed pairs after freeze; do not tune this experiment on held-out judgments.",
        "sha256": {str(path.resolve()): _sha256(path) for path in paths},
    }
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    return seal


def verify_refinement_seal(refined_dir):
    """Reject a post-review change to any frozen artifact, protocol, or code."""
    seal = json.loads((Path(refined_dir) / "refinement_freeze.json").read_text(encoding="utf-8"))
    for path, expected in seal["sha256"].items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen experimental artifact changed: {path}")
    return seal
