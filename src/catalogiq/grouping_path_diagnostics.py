"""Explain saved grouping links and paths without changing a grouping algorithm.

Product features are read from the frozen experiment input, never its target.
Counterfactual edge removals concern graph connectivity only. An unreviewed
edge on a path is not automatically an erroneous match.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .group_graph_audit import (
    _components, _cuts, _load_method, _pair, _path, _verify_edge_snapshot,
)
from .split_matching import (
    brand_block_key, compatible, name_similarity, prepare_products,
)
from .split_review_validation import verify_refinement_seal
from .split_tfidf_v2 import _supporting_matrix
from .splitting import SplitConfig
from .splitting_experiment import load_input


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edge_disjoint_paths(left, right, adjacency, removed_edges=frozenset()):
    """Deterministic maximum edge-disjoint paths in a simple undirected graph.

    Unit-capacity Edmonds--Karp and integral-flow decomposition provide a real
    connectivity count, rather than a greedy collection of alternative paths.
    Opposite directed flows cancel, so an undirected edge appears at most once.
    """
    if left == right:
        raise ValueError("path endpoints must be different")
    if left not in adjacency or right not in adjacency:
        raise ValueError("path endpoint absent from graph")
    residual = {node: {other: 1 for other in sorted(neighbors)
                       if _pair(node, other) not in removed_edges}
                for node, neighbors in adjacency.items()}
    for node, neighbors in residual.items():
        for other in neighbors:
            if node not in residual.get(other, {}):
                raise ValueError("edge-disjoint paths require an undirected graph")
    while True:
        parents, pending = {left: None}, deque([left])
        while pending and right not in parents:
            node = pending.popleft()
            for other in sorted(residual[node]):
                if residual[node][other] > 0 and other not in parents:
                    parents[other] = node
                    pending.append(other)
        if right not in parents:
            break
        node = right
        while node != left:
            previous = parents[node]
            residual[previous][node] -= 1
            residual[node][previous] += 1
            node = previous
    positive = {node: {other: 1 - capacity for other, capacity in neighbors.items()
                       if 1 - capacity > 0}
                for node, neighbors in residual.items()}
    result = []
    while True:
        parents, pending = {left: None}, deque([left])
        while pending and right not in parents:
            node = pending.popleft()
            for other in sorted(positive[node]):
                if positive[node][other] > 0 and other not in parents:
                    parents[other] = node
                    pending.append(other)
        if right not in parents:
            break
        path = [right]
        while path[-1] != left:
            path.append(parents[path[-1]])
        path.reverse()
        for a, b in zip(path, path[1:]):
            positive[a][b] -= 1
        result.append(path)
    used = [_pair(a, b) for path in result for a, b in zip(path, path[1:])]
    if len(set(used)) != len(used):
        raise AssertionError("flow decomposition reused an undirected edge")
    return result


def _candidate_blocks(products, config):
    """Reconstruct candidate-block membership, not accepted-match decisions."""
    frequencies = Counter(token for product in products for token in product.core)
    keys, counts = {}, Counter()
    for index, product in enumerate(products):
        member = []
        if product.brand:
            selected = sorted((token for token in product.core
                               if len(token) >= 3 and not token.isdigit()),
                              key=lambda token: (frequencies[token], token))[:config.block_tokens]
            for token in selected:
                member.append(("brand", brand_block_key(product.brand), token))
                if product.tokens and product.family:
                    member.append(("title_prefix", product.family.split()[0], token))
        keys[index] = set(member)
        counts.update(member)
    return keys, counts


def lexical_evidence(left, right, config):
    """Explain the unchanged compatibility guard for two prepared products."""
    common = sorted(left.core & right.core)
    core_overlap = (len(common) / min(len(left.core), len(right.core))
                    if left.core and right.core else None)
    # This is an explanatory annotation only: the matcher still uses its own
    # unchanged core tokens and ingredient extraction.
    claims = []
    for product in (left, right):
        claims.append(re.findall(r"\bno\s+([a-z]+(?:\s+(?:or|and)\s+[a-z]+)*)\s+fillers\b",
                                 product.name))
    negated_words = [set(re.findall(r"[a-z]+", " ".join(side))) - {"and", "or"}
                     for side in claims]
    signatures_conflict = bool(left.ingredients and right.ingredients
                               and left.ingredients.isdisjoint(right.ingredients))
    return {
        "left_normalized_name": left.name, "right_normalized_name": right.name,
        "left_family_name": left.family, "right_family_name": right.family,
        "left_normalized_brand": left.brand, "right_normalized_brand": right.brand,
        "left_core_json": _json(sorted(left.core)), "right_core_json": _json(sorted(right.core)),
        "shared_core_json": _json(common), "core_overlap_over_smaller_core": core_overlap,
        "shared_negated_filler_words_counted_as_core_json": _json(
            sorted(set(common) & negated_words[0] & negated_words[1])),
        "left_ingredient_signature_json": _json(sorted(left.ingredients)),
        "right_ingredient_signature_json": _json(sorted(right.ingredients)),
        "ingredient_pair_conflict": signatures_conflict,
        "ingredient_guard_has_two_nonempty_signatures": bool(left.ingredients and right.ingredients),
        "pair_guard_passes": compatible(left, right, config, allow_brand_override=True),
        "family_token_jaccard": name_similarity(left, right),
        "family_edit_similarity": SequenceMatcher(
            None, *sorted((left.family, right.family)), autojunk=False).ratio(),
    }


def named_path_rows(path, method, scenario, path_number, records, evidence, review_by_pair):
    """Orient names by actual path traversal while preserving canonical evidence."""
    rows = []
    for step, (left, right) in enumerate(zip(path, path[1:]), start=1):
        pair = _pair(left, right)
        review = review_by_pair.get(pair, {})
        rows.append({
            "method": method, "scenario": scenario, "path_number": path_number,
            "step": step, "path_hops": len(path) - 1,
            "left_record_id": left, "right_record_id": right,
            "left_ProductName": records[left]["ProductName"],
            "right_ProductName": records[right]["ProductName"],
            "canonical_left_record_id": pair[0], "canonical_right_record_id": pair[1],
            "saved_edge_evidence_json": _json(evidence[pair]),
            "saved_reasons_json": _json(sorted({item.get("reason", "") for item in evidence[pair]})),
            "saved_reason": evidence[pair][0].get("reason", "") if len(evidence[pair]) == 1 else "multiple_saved_reasons",
            "saved_score": evidence[pair][0].get("score", "") if len(evidence[pair]) == 1 else "",
            "reviewed_edge_pair_id": review.get("pair_id", ""),
            "reviewed_edge_leakage_decision": review.get("leakage_decision", "unreviewed"),
            "reviewed_edge_leakage_rationale": review.get("leakage_rationale", ""),
        })
    return rows


def _fit_selected(values, indices, max_features):
    """Fit the frozen corpus with the v2 vectorizer; retain inspected rows only."""
    model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float64,
                           max_features=max_features, max_df=1.0)
    matrix = model.fit_transform(values)
    return matrix[indices].tocsr(), model.get_feature_names_out()


def _cosine_features(matrix, features, left, right):
    shared = matrix[left].multiply(matrix[right]).tocoo()
    terms = sorted(((str(features[column]), float(value))
                    for column, value in zip(shared.col, shared.data)),
                   key=lambda item: (-item[1], item[0]))
    return float(shared.sum()), [{"feature": feature, "contribution": value}
                                for feature, value in terms[:20]]


def _tfidf_evidence(frame, products, config, pairs):
    """Recompute actual frozen-corpus similarities, without proposing new links."""
    endpoint_ids = sorted({record for pair in pairs for record in pair})
    full_index = dict(zip(frame.record_id, range(len(frame))))
    indices = [full_index[record] for record in endpoint_ids]
    local = {record: index for index, record in enumerate(endpoint_ids)}
    result = {pair: {} for pair in pairs}
    for field, values in (("title", [product.name for product in products]),
                          ("family", [product.family for product in products])):
        matrix, features = _fit_selected(values, indices, config.tfidf_name_max_features)
        for left, right in sorted(pairs):
            score, contributions = _cosine_features(matrix, features, local[left], local[right])
            result[(left, right)][f"recomputed_{field}_cosine"] = score
            result[(left, right)][f"{field}_shared_feature_contributions_json"] = _json(contributions)
    support, sizes = _supporting_matrix(frame, config)
    selected = support[indices].tocsr()
    for left, right in sorted(pairs):
        result[(left, right)]["recomputed_support_cosine"] = float(
            selected[local[left]].multiply(selected[local[right]]).sum())
    return result, sizes


def investigate_saved_paths(run_dir, review_path, output_dir=None, *, pair_id="G012",
                            edge_snapshot=None, verify_seal=True, recompute_cosines=True):
    """Trace both frozen methods for a reviewed pair and their entire groups."""
    run_dir, review_path = Path(run_dir), Path(review_path)
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError("refusing to overwrite path investigation")
    if verify_seal:
        verify_refinement_seal(run_dir)
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    source_paths = [review_path, run_dir / "config.json", run_dir / "input_manifest.json"]
    source_paths += [run_dir / f"{method}_{suffix}.csv"
                     for method in ("rule", "tfidf")
                     for suffix in ("assignments", "matching_edges", "group_sizes")]
    before = {str(path.resolve()): _hash(path) for path in source_paths}
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    frame, loaded = load_input(Path(manifest["path"]), Path(manifest["identifier_source"])
                                if manifest.get("identifier_source") else None)
    if loaded["sha256"] != manifest["sha256"] or loaded["rows"] != manifest["rows"]:
        raise ValueError("path evidence input differs from frozen experiment")
    for path, expected in manifest.get("fingerprints", {}).items():
        if _hash(path) != expected:
            raise ValueError(f"frozen input source fingerprint changed: {path}")
    before.update(loaded["fingerprints"])
    # Drop target columns immediately; product evidence only is exported.
    feature_columns = ["record_id", "ProductName", "ProductBrand", "ProductDescription",
                       "ProductContents", "Sku", "Upc", "ProductModelNumber", "Retailer", "ProductUrl"]
    frame = frame.loc[:, [column for column in feature_columns if column in frame]].copy()
    config = SplitConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    products = prepare_products(frame, config)
    product_by_id = dict(zip(frame.record_id, products))
    index_by_id = dict(zip(frame.record_id, range(len(frame))))
    records = frame.set_index("record_id").to_dict("index")
    block_keys, block_sizes = _candidate_blocks(products, config)
    reviews = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    if not reviews.pair_id.eq(pair_id).sum() == 1:
        raise ValueError("requested review pair must occur exactly once")
    if "judgment_uses_Segment" in reviews and not reviews.judgment_uses_Segment.str.casefold().eq("false").all():
        raise ValueError("review judgments must exclude Segment")
    review_by_pair = {_pair(row["left_record_id"], row["right_record_id"]): row
                      for row in reviews.to_dict("records")}
    endpoint = reviews.loc[reviews.pair_id.eq(pair_id)].iloc[0]
    left, right = endpoint.left_record_id, endpoint.right_record_id
    methods, all_edges, paths, nodes = {}, [], [], set()
    tf_pairs = set()
    for method in ("rule", "tfidf"):
        groups, group_by_id, adjacency, evidence, source_rows = _load_method(run_dir, method)
        if set(group_by_id) != set(records):
            raise ValueError("saved assignments differ from input record identities")
        if group_by_id[left] != group_by_id[right]:
            methods[method] = {"grouped": False, "endpoint_left_group": group_by_id[left],
                               "endpoint_right_group": group_by_id[right]}
            continue
        group_id = group_by_id[left]
        group = groups[group_id]
        nodes.update(group)
        group_edges = {pair: values for pair, values in evidence.items()
                       if pair[0] in set(group)}
        group_adj = {record: adjacency[record] for record in group}
        bad_edges = {pair for pair in group_edges
                     if review_by_pair.get(pair, {}).get("leakage_decision") == "keep_separate"}
        bridges, articulation = _cuts(group, group_adj)
        scenario_results = {}
        scenarios = [("saved", frozenset()),
                     ("reviewed_bad_direct_edges_removed", bad_edges)]
        if _pair(left, right) in group_edges:
            scenarios.append(("endpoint_direct_edge_removed", {_pair(left, right)}))
        for scenario, removed in scenarios:
            shortest = _path(left, right, group_adj, removed)
            alternatives = edge_disjoint_paths(left, right, group_adj, removed)
            components = _components(group, group_adj, removed)
            remaining_conflicts = sum(
                review.get("leakage_decision") == "keep_separate"
                and pair[0] in set(group) and pair[1] in set(group)
                and _path(pair[0], pair[1], group_adj, removed) is not None
                for pair, review in review_by_pair.items())
            scenario_results[scenario] = {
                "removed_edges": len(removed),
                "component_sizes": [len(component) for component in components],
                "endpoint_connected": shortest is not None,
                "shortest_path_hops": len(shortest) - 1 if shortest else None,
                "maximum_edge_disjoint_endpoint_paths": len(alternatives),
                "reviewed_keep_separate_pairs_still_connected": int(remaining_conflicts),
                "shortest_path_names": [records[record]["ProductName"] for record in shortest] if shortest else [],
            }
            if shortest:
                paths.extend(named_path_rows(shortest, method, scenario + "_shortest", 1,
                                             records, evidence, review_by_pair))
            for number, path in enumerate(alternatives, start=1):
                paths.extend(named_path_rows(path, method, scenario + "_maximum_edge_disjoint", number,
                                             records, evidence, review_by_pair))
        signatures = sorted({tuple(sorted(product_by_id[record].ingredients))
                             for record in group if product_by_id[record].ingredients})
        methods[method] = {
            "grouped": True, "group_id": group_id, "group_size": len(group),
            "saved_direct_edges": len(group_edges),
            "implied_co_group_pairs": len(group) * (len(group) - 1) // 2,
            "direct_edge_density": len(group_edges) / (len(group) * (len(group) - 1) / 2),
            "saved_reason_counts": dict(Counter(item.get("reason", "")
                                                  for values in group_edges.values() for item in values)),
            "group_bridge_edges": len(bridges), "group_articulation_nodes": len(articulation),
            "known_component_ingredient_signatures": signatures,
            "records_with_empty_ingredient_signature": sum(not product_by_id[record].ingredients for record in group),
            "reviewed_bad_direct_edges": len(bad_edges), "scenarios": scenario_results,
        }
        for pair, values in sorted(group_edges.items()):
            a, b = pair
            common_blocks = sorted(block_keys[index_by_id[a]] & block_keys[index_by_id[b]])
            usable = [key for key in common_blocks if block_sizes[key] <= config.max_block_size]
            reviewed = review_by_pair.get(pair, {})
            for value in values:
                all_edges.append({
                    "method": method, "group_id": group_id, "group_size": len(group),
                    "left_record_id": a, "right_record_id": b,
                    "left_ProductName": records[a]["ProductName"],
                    "right_ProductName": records[b]["ProductName"],
                    "saved_reason": value.get("reason", ""), "saved_score": value.get("score", ""),
                    "reviewed_pair_id": reviewed.get("pair_id", ""),
                    "reviewed_leakage_decision": reviewed.get("leakage_decision", "unreviewed"),
                    "reviewed_leakage_rationale": reviewed.get("leakage_rationale", ""),
                    "reconstructed_usable_candidate_blocks_json": _json(
                        [{"key": key, "block_size": block_sizes[key]} for key in usable]),
                    **lexical_evidence(product_by_id[a], product_by_id[b], config),
                })
            if method == "tfidf":
                tf_pairs.add(pair)
    vector_scores, support_sizes = (_tfidf_evidence(frame, products, config, tf_pairs)
                                     if recompute_cosines and tf_pairs else ({}, {}))
    for row in all_edges:
        if row["method"] != "tfidf" or not recompute_cosines:
            continue
        row.update(vector_scores[_pair(row["left_record_id"], row["right_record_id"])])
        score_field = "family" if row["saved_reason"] == "tfidf_family_cosine" else "title"
        delta = abs(row[f"recomputed_{score_field}_cosine"] - float(row["saved_score"]))
        row["saved_cosine_absolute_recomputation_difference"] = delta
        if delta > 1e-10:
            raise ValueError("frozen accepted cosine differs from full-corpus recomputation")
    edges_frame, paths_frame = pd.DataFrame(all_edges), pd.DataFrame(paths)
    features = frame.loc[frame.record_id.isin(nodes)].copy()
    after = {path: _hash(path) for path in before}
    if before != after:
        raise ValueError("path investigation changed an input artifact")
    if verify_seal:
        verify_refinement_seal(run_dir)
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    summary = {
        "pair_id": pair_id, "left_record_id": left, "right_record_id": right,
        "left_ProductName": records[left]["ProductName"], "right_ProductName": records[right]["ProductName"],
        "reviewed_endpoint_leakage_decision": endpoint.leakage_decision,
        "methods": methods, "supporting_vocabulary_sizes": support_sizes,
        "unchanged_guard_parameters": {name: getattr(config, name) for name in (
            "rule_family_threshold", "rule_family_edit_threshold", "core_min_overlap",
            "tfidf_name_threshold", "tfidf_family_threshold", "tfidf_supported_name_threshold",
            "tfidf_support_threshold")},
        "frozen_seal_verified": verify_seal, "accepted_edge_snapshot_verified": edge_snapshot is not None,
        "input_sha256": before, "input_hashes_unchanged": True,
        "tfidf_cosines_recomputed_from_full_frozen_corpus": recompute_cosines,
        "uses_target_labels": False, "changes_grouping_or_split_assignments": False,
        "interpretation": "Saved accepted-edge reasons are actual matcher provenance. Reconstructed lexical and candidate-block evidence explains the unchanged guards; no new match is proposed. Only reviewed keep_separate decisions establish known bad direct edges. Connectivity removals are counterfactual diagnostics; remaining unreviewed edges are not thereby proven incorrect. These selected examples do not estimate population-wide accuracy.",
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in (("all_group_edges", edges_frame), ("named_paths", paths_frame),
                           ("group_product_features", features)):
            data.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                            allow_nan=False) + "\n", encoding="utf-8")
    return {"all_group_edges": edges_frame, "named_paths": paths_frame,
            "group_product_features": features, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="G012")
    parser.add_argument("--edge-snapshot", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = investigate_saved_paths(arguments.run_dir, arguments.review, arguments.output_dir,
                                     pair_id=arguments.pair_id, edge_snapshot=arguments.edge_snapshot)
    for method, data in result["summary"]["methods"].items():
        saved = data["scenarios"]["saved"]
        removed = data["scenarios"]["reviewed_bad_direct_edges_removed"]
        print(f"{method}: {data['group_size']} records, {data['saved_direct_edges']} direct links; "
              f"shortest endpoint path {saved['shortest_path_hops']} hops, "
              f"{saved['maximum_edge_disjoint_endpoint_paths']} edge-disjoint paths.")
        print("  " + " -> ".join(saved["shortest_path_names"]))
        print(f"  Remove {removed['removed_edges']} reviewed bad direct links: "
              f"component sizes {removed['component_sizes']}; "
              f"endpoints connected={removed['endpoint_connected']}.")
    return result


if __name__ == "__main__":
    main()
