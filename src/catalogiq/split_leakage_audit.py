"""Read-only duplicate audit of frozen splitting artifacts.

The independent name comparator is exhaustive within its registered lexical
scope. Exact normalized description clusters are counted combinatorially;
shared supplier copy is a diagnostic flag, never an automatic leakage label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .features import sha256
from .split_evaluation import evaluation_name, leakage_metrics, near_duplicate_pairs
from .splitting import SPLITS, SplitConfig
from .splitting_experiment import load_input

REVIEW_FIELDS = ("ProductName", "ProductBrand", "ProductDescription", "ProductContents",
                 "Upc", "Sku", "ProductModelNumber", "Retailer", "ProductUrl")
VERSION = "frozen-split-leakage-audit-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))


def pair_rank(pair, seed=20261009):
    return hashlib.sha256((str(seed) + ":" + ":".join(pair)).encode()).hexdigest()


def aligned_assignment(frame, assignment):
    """Reject missing, extra, duplicate or unknown role IDs before auditing."""
    required = {"record_id", "group_id", "split"}
    if not required.issubset(assignment):
        raise ValueError("Assignment requires record_id, group_id and split")
    if frame.record_id.duplicated().any() or assignment.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs")
    if set(frame.record_id) != set(assignment.record_id):
        raise ValueError("Assignment IDs must exactly cover this audit universe")
    if not assignment.split.isin(SPLITS).all():
        raise ValueError("Unknown split role")
    return assignment.set_index("record_id").loc[frame.record_id].reset_index()


def isolation_summary(assignment, expected_isolated):
    crossing = assignment.groupby("group_id").split.nunique().gt(1)
    crossing_ids = set(crossing[crossing].index)
    result = {"expected_group_isolation": expected_isolated,
              "crossing_groups": len(crossing_ids),
              "records_in_crossing_groups": int(assignment.group_id.isin(crossing_ids).sum()),
              "groups_isolated": not crossing_ids}
    if expected_isolated and crossing_ids:
        raise ValueError("Frozen grouped strategy has groups crossing roles")
    return result


def description_probe(frame, assignment, minimum_chars=200, candidate_limit=200):
    """Count every exact normalized body pair without expanding large clusters.

    Candidate witnesses are bounded hash-ordered train/validation products per
    cluster. Their selection does not consume labels, matching outcomes or test
    text. Distinct-name clusters can be supplier boilerplate or true line variants.
    """
    assignment = aligned_assignment(frame, assignment)
    buckets = defaultdict(list)
    names = frame.ProductName.map(evaluation_name).tolist()
    brands = frame.ProductBrand.map(evaluation_name).tolist()
    for row, description in enumerate(frame.ProductDescription.map(evaluation_name)):
        if len(description) >= minimum_chars:
            buckets[description].append(row)
    roles = assignment.split.tolist()
    ids = frame.record_id.tolist()
    group_ids = assignment.group_id.tolist()
    clusters, candidates, crossing_rows = [], [], set()
    totals = {"eligible_rows": sum(map(len, buckets.values())), "duplicate_clusters": 0,
              "duplicate_pairs": 0, "crossing_clusters": 0, "crossing_pairs": 0,
              "crossing_rows": 0, "possible_boilerplate_clusters": 0,
              "crossing_split_pairs": {f"{a}__{b}": 0 for a, b in combinations(SPLITS, 2)}}
    for description, members in sorted(buckets.items(), key=lambda item: hashlib.sha256(item[0].encode()).hexdigest()):
        if len(members) < 2:
            continue
        digest = hashlib.sha256(description.encode()).hexdigest()
        counts = Counter(roles[row] for row in members)
        split_pairs = {f"{a}__{b}": counts[a] * counts[b] for a, b in combinations(SPLITS, 2)}
        name_count = len({names[row] for row in members if names[row]})
        brand_count = len({brands[row] for row in members if brands[row]})
        crossed = sum(split_pairs.values())
        totals["duplicate_clusters"] += 1
        totals["duplicate_pairs"] += len(members) * (len(members) - 1) // 2
        totals["crossing_clusters"] += bool(crossed)
        totals["crossing_pairs"] += crossed
        totals["possible_boilerplate_clusters"] += name_count > 1
        if crossed:
            crossing_rows.update(members)
        for key, count in split_pairs.items():
            totals["crossing_split_pairs"][key] += count
        clusters.append({"description_sha256": digest, "normalized_chars": len(description),
                         "records": len(members), "distinct_normalized_names": name_count,
                         "distinct_normalized_brands": brand_count,
                         "distinct_groups": len({group_ids[row] for row in members}),
                         "possible_boilerplate": name_count > 1,
                         "crossing_pairs": crossed, **{f"{split}_records": counts[split] for split in SPLITS},
                         **split_pairs})
        train = sorted((row for row in members if roles[row] == "train"), key=lambda row: pair_rank((ids[row],)))[:8]
        validation = sorted((row for row in members if roles[row] == "validation"), key=lambda row: pair_rank((ids[row],)))[:8]
        witnesses = sorted((pair_key(ids[a], ids[b]) for a in train for b in validation
                            if names[a] != names[b]), key=pair_rank)[:8]
        candidates.extend({"left_record_id": a, "right_record_id": b, "route": "exact_normalized_description",
                           "name_jaccard": None, "description_sha256": digest} for a, b in witnesses)
    totals["crossing_rows"] = len(crossing_rows)
    totals.update({"minimum_normalized_chars": minimum_chars,
                   "normalization": "same independent HTML/NFKC/casefold/alphanumeric normalization as name audit",
                   "all_pair_counts_exact": True, "pair_expansion_used_for_counts": False,
                   "meaning": "Shared descriptions are candidates; distinct names flag possible supplier boilerplate, not established leakage."})
    candidates = sorted(candidates, key=lambda row: pair_rank(pair_key(row["left_record_id"], row["right_record_id"])))[:candidate_limit]
    return totals, pd.DataFrame(clusters), candidates


def blind_review_sample(frame, assignment, candidates, prior_pairs=(), size=24, seed=20261009):
    """Round-robin route/brand/score strata, with no test rows or prior pairs."""
    assignment = aligned_assignment(frame, assignment)
    roles = dict(zip(assignment.record_id, assignment.split))
    evidence = frame.set_index("record_id")
    prior = {pair_key(*pair) for pair in prior_pairs}
    seen, eligible = set(), []
    for row in sorted(candidates, key=lambda row: (row["route"], pair_rank(pair_key(row["left_record_id"], row["right_record_id"]), seed))):
        key = pair_key(row["left_record_id"], row["right_record_id"])
        if key in seen or key in prior or {roles.get(key[0]), roles.get(key[1])} != {"train", "validation"}:
            continue
        seen.add(key)
        brands = [evaluation_name(evidence.at[id_, "ProductBrand"]) for id_ in key]
        brand_stratum = "same_brand" if brands[0] and brands[0] == brands[1] else "different_or_missing_brand"
        score = row.get("name_jaccard")
        score_stratum = "description_route" if score is None else "identical_name" if score == 1 else "high" if score >= .95 else "middle" if score >= .90 else "boundary"
        eligible.append({**row, "left_record_id": key[0], "right_record_id": key[1],
                         "stratum": f"{row['route']}__{brand_stratum}__{score_stratum}"})
    strata = defaultdict(list)
    for row in eligible:
        strata[row["stratum"]].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: pair_rank(pair_key(row["left_record_id"], row["right_record_id"]), seed))
    chosen = []
    routes = sorted({row["route"] for row in eligible})
    # Balanced routes where available, with equal turns for brand/score strata.
    for route in routes:
        route_strata = [key for key in sorted(strata) if key.startswith(route + "__")]
        quota = size // max(len(routes), 1)
        taken = 0
        while taken < quota and any(strata[key] for key in route_strata):
            for key in route_strata:
                if strata[key] and taken < quota:
                    chosen.append(strata[key].pop(0))
                    taken += 1
    while len(chosen) < size and any(strata.values()):
        for stratum in sorted(strata):
            if strata[stratum] and len(chosen) < size:
                chosen.append(strata[stratum].pop(0))
    chosen.sort(key=lambda row: pair_rank(pair_key(row["left_record_id"], row["right_record_id"]), seed))
    blind, mapping = [], []
    for index, row in enumerate(chosen, 1):
        review_id = f"leakage_{index:02d}"
        blind_row = {"review_id": review_id}
        for side in ("left", "right"):
            id_ = row[f"{side}_record_id"]
            for field in REVIEW_FIELDS:
                blind_row[f"{side}_{field}"] = evidence.at[id_, field] if field in evidence else ""
        blind.append(blind_row)
        mapping.append({"review_id": review_id, **row})
    return pd.DataFrame(blind), pd.DataFrame(mapping), {
        "seed": seed, "requested_pairs": size, "selected_pairs": len(chosen),
        "eligible_candidates_after_prior_pair_and_test_exclusion": len(eligible),
        "previously_reviewed_exact_pairs_excluded": len(prior),
        "all_final_test_records_excluded": True,
        "stratum_counts": dict(Counter(row["stratum"] for row in chosen)),
        "selection": "equal route quotas where available; deterministic round-robin brand/name-score strata then seeded hash ordering; description witnesses require distinct normalized names",
        "interpretation": "Selected crossing witnesses, not a random population error-rate sample."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_cleaned.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/split_rule_v3_20261008"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/processed/segment_full_development_20261009"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/split_leakage_audit_20261009/automatic"))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Audit writes only to a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = SplitConfig(evaluation_threshold=.85, evaluation_shingle_size=5, evaluation_min_name_chars=12)
    started = time.perf_counter()
    sources = [args.input, args.split_dir / "rule_assignments.csv", args.split_dir / "tfidf_assignments.csv",
               args.split_dir / "independent_near_duplicate_pairs.csv", args.development_dir / "development_assignments.csv",
               args.development_dir / "protected_final_test.csv", Path(__file__), Path("scripts/audit_split_leakage.py"),
               Path("src/catalogiq/split_evaluation.py"), Path("src/catalogiq/splitting_experiment.py")]
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    protocol = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": {"name_shingle_size": 5, "name_jaccard_threshold": .85,
                               "minimum_normalized_name_chars": 12, "description_minimum_normalized_chars": 200,
                               "review_pairs": 24, "review_seed": 20261009, "description_candidate_limit": 200},
                "parameters_frozen_before_data_audit": True, "source_sha256": hashes,
                "no_grouping_or_assignments_changed": True, "no_model_or_test_predictions_consumed": True,
                "test_role_IDS_audited_for_duplicate_crossings_only": True}
    write_json(output / "protocol.json", protocol)
    frame, input_metadata = load_input(args.input)
    hashes.update(input_metadata["fingerprints"])
    pairs, pair_metadata = near_duplicate_pairs(frame, config, progress=print)
    fresh = {(frame.at[a, "record_id"], frame.at[b, "record_id"]): score for a, b, score in pairs}
    saved = pd.read_csv(args.split_dir / "independent_near_duplicate_pairs.csv", keep_default_na=False)
    old = {(row.left_record_id, row.right_record_id): row.name_jaccard for row in saved.itertuples(index=False)}
    equal = fresh.keys() == old.keys() and all(abs(fresh[key] - float(old[key])) <= 1e-12 for key in fresh)
    if not equal:
        raise ValueError("Fresh exhaustive audit differs from frozen independent pair pool")
    pd.DataFrame([{"left_record_id": key[0], "right_record_id": key[1], "name_jaccard": score}
                  for key, score in fresh.items()]).to_csv(output / "recomputed_near_pairs.csv", index=False)
    strategies = {}
    for name, file in (("rule_v3", "rule_assignments.csv"), ("tfidf_v2", "tfidf_assignments.csv")):
        strategies[name] = (frame, pd.read_csv(args.split_dir / file, dtype=str, keep_default_na=False), True)
    development = pd.read_csv(args.development_dir / "development_assignments.csv", dtype=str, keep_default_na=False)
    protected = pd.read_csv(args.development_dir / "protected_final_test.csv", dtype=str, keep_default_na=False)
    for name in ("group_aware", "random"):
        assignments = pd.concat([development.loc[development.strategy.eq(name), ["record_id", "group_id", "split"]],
                                 protected[["record_id", "group_id", "split"]]], ignore_index=True)
        subset = frame.loc[frame.record_id.isin(assignments.record_id)].reset_index(drop=True)
        strategies[f"full_development_{name}"] = (subset, assignments, name != "random")
    rows, crossing_frames, description_frames, sample_candidates = [], [], [], []
    results = {}
    for name, (subset, assignments, isolated) in strategies.items():
        strategy_started = time.perf_counter()
        assignments = aligned_assignment(subset, assignments)
        position = {id_: row for row, id_ in enumerate(subset.record_id)}
        eligible_pairs = [(position[a], position[b], score) for (a, b), score in fresh.items() if a in position and b in position]
        metrics = leakage_metrics(subset, assignments, eligible_pairs, config)
        isolation = isolation_summary(assignments, isolated)
        description, clusters, description_candidates = description_probe(subset, assignments)
        if len(clusters):
            clusters.insert(0, "strategy", name)
            description_frames.append(clusters)
        near_group_ids = set()
        crossings = []
        for a, b, score in eligible_pairs:
            if assignments.at[a, "split"] == assignments.at[b, "split"]:
                continue
            near_group_ids.update((assignments.at[a, "group_id"], assignments.at[b, "group_id"]))
            crossings.append({"strategy": name, "left_record_id": subset.at[a, "record_id"],
                              "right_record_id": subset.at[b, "record_id"], "name_jaccard": score,
                              "left_split": assignments.at[a, "split"], "right_split": assignments.at[b, "split"]})
        if crossings:
            crossing_frames.append(pd.DataFrame(crossings))
        metrics["near_duplicate"]["crossing_groups"] = len(near_group_ids)
        result = {"records": len(subset), "groups": metrics["groups"], "splits": metrics["splits"],
                  "isolation": isolation, "exact_payload": metrics["exact_payload"],
                  "exact_brand_name": metrics["exact_brand_name"], "near_duplicate": metrics["near_duplicate"],
                  "exact_normalized_description": description,
                  "runtime_seconds": time.perf_counter() - strategy_started}
        results[name] = result
        for method in ("exact_payload", "exact_brand_name", "near_duplicate", "exact_normalized_description"):
            m = result[method]
            rows.append({"strategy": name, "audit_method": method, "records": len(subset),
                         "product_groups": metrics["groups"]["count"],
                         "duplicate_pairs": m.get("duplicate_pairs", m.get("pairs")),
                         "crossing_pairs": m["crossing_pairs"], "crossing_rows": m["crossing_rows"],
                         **m["crossing_split_pairs"]})
        if name == "full_development_group_aware":
            sample_candidates = [{"left_record_id": r["left_record_id"], "right_record_id": r["right_record_id"],
                                  "route": "independent_name", "name_jaccard": r["name_jaccard"],
                                  "description_sha256": ""} for r in crossings] + description_candidates
    prior = set()
    review_paths = [Path("data/processed") / p for p in (
        "split_pair_review_20261008/annotated_pair_review.csv", "split_refinement_review_20261008/annotated_heldout_pairs.csv",
        "split_dual_review_20261008/dual_judgments.csv", "split_balanced_review_20261008/annotated_balanced_pairs.csv",
        "split_balanced_review_20261008/all_candidate_judgments.csv",
        "split_balanced_graph_review_20261008/annotated_graph_pairs.csv", "split_rule_v3_independent_review_20261008/annotated_reserved_pairs.csv")]
    for path in review_paths:
        if path.exists():
            hashes[str(path.resolve())] = sha256(path)
            reviewed = pd.read_csv(path, dtype=str, keep_default_na=False)
            prior.update(pair_key(row.left_record_id, row.right_record_id) for row in reviewed.itertuples(index=False))
    subset, assignments, _ = strategies["full_development_group_aware"]
    blind, mapping, selection = blind_review_sample(subset, assignments, sample_candidates, prior)
    if len(blind) != 24:
        raise ValueError("Insufficient fresh train-validation witnesses for registered24 review sample")
    blind.to_csv(output / "blind_review_24.csv", index=False)
    mapping.to_csv(output / "review_selection_mapping.csv", index=False)
    write_json(output / "review_selection.json", selection)
    pd.DataFrame(rows).to_csv(output / "leakage_comparison.csv", index=False)
    pd.concat(crossing_frames, ignore_index=True).to_csv(output / "near_crossing_pairs.csv", index=False)
    pd.concat(description_frames, ignore_index=True).to_csv(output / "description_duplicate_clusters.csv", index=False)
    unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    if not unchanged:
        raise ValueError("Source artifacts changed during independent audit")
    summary = {"version": VERSION, "input": input_metadata, "name_audit": pair_metadata,
               "recomputed_pool_equals_saved_pairs_and_scores": equal, "strategies": results,
               "review_selection": selection, "source_artifacts_unchanged": unchanged,
               "source_sha256": hashes, "runtime_seconds": time.perf_counter() - started,
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}},
               "no_final_test_model_metrics_or_predictions_read": True,
               "limitations": ["Lexical witnesses and exact shared description bodies are not verified product identities or leakage decisions.",
                               "Name audit is exhaustive only for normalized names with at least12 chars at registered Jaccard threshold.",
                               "Description probe detects exact normalized bodies only, not all semantic or approximate description matches.",
                               "Four strategy rows use two different universes; compare methods within matched universes.",
                               "The random control intentionally allows product families across training and validation.",
                               "The selected review sample cannot estimate population-wide error rates."]}
    write_json(output / "summary.json", summary)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Audit complete;24 blind review pairs: {output / 'blind_review_24.csv'}")
