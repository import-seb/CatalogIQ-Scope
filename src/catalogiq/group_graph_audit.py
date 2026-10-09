"""Audit transitivity in saved grouping graphs without running a matcher.

Edges and assignments are read-only inputs. Edge-removal results are diagnostic
counterfactuals, never replacement assignments. Size thresholds screen a tail;
only reviewed leakage decisions provide evidence of a grouping error.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random

import pandas as pd

from .split_review_validation import METHODS, PAIR_COLUMNS, verify_refinement_seal


EMPTY_COLUMNS = {
    "canonical_edges": ["method", "group_id", *PAIR_COLUMNS, "source_edge_rows", "reasons_json", "evidence_json", "reviewed_keep_separate"],
    "articulation_points": ["method", "group_id", "group_size", "record_id", "component_sizes_after_node_removal_json", "lost_pairs_between_remaining_nodes"],
    "bridge_edges": ["method", "group_id", "group_size", *PAIR_COLUMNS, "smaller_component_size", "larger_component_size", "pair_amplification", "evidence_json", "reviewed_keep_separate"],
    "reviewed_pair_paths": ["method", "pair_id", "review_set", *PAIR_COLUMNS, "leakage_decision", "leakage_confidence", "grouped", "left_group_id", "right_group_id", "group_size", "direct_edge", "connection_type", "shortest_path_hops", "path_record_ids_json", "path_edge_evidence_json", "connected_after_reviewed_bad_edge_removal"],
    "removal_counterfactuals": ["method", "group_id", "group_size", "scenario", "removed_direct_edges", "removed_edge_pairs_json", "component_sizes_json", "components_after_removal", "implied_pairs_before", "implied_pairs_after", "implied_pair_reduction", "reviewed_keep_separate_pairs_before", "reviewed_keep_separate_pairs_still_connected", "reviewed_keep_separate_pairs_disconnected"],
    "candidates": ["method", "group_id", "group_size", "candidate_type", "shortest_path_hops", "pair_amplification", *PAIR_COLUMNS, "graph_contexts_json", "methods"],
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(left, right):
    return tuple(sorted((left, right)))


def _choose_two(size):
    return size * (size - 1) // 2


def _components(nodes, adjacency, removed_edges=frozenset(), removed_node=None):
    unseen = set(nodes) - {removed_node}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, pending = {start}, [start]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen and _pair(current, neighbor) not in removed_edges:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        result.append(component)
    return sorted(result, key=lambda component: (-len(component), min(component)))


def _bfs(start, adjacency, removed_edges=frozenset()):
    distances, parents = {start: 0}, {start: None}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in distances and _pair(current, neighbor) not in removed_edges:
                distances[neighbor] = distances[current] + 1
                parents[neighbor] = current
                pending.append(neighbor)
    return distances, parents


def _path(left, right, adjacency, removed_edges=frozenset()):
    distances, parents = _bfs(left, adjacency, removed_edges)
    if right not in distances:
        return None
    result = [right]
    while result[-1] != left:
        result.append(parents[result[-1]])
    return list(reversed(result))


def _diameter(nodes, adjacency):
    """Exact diameter and deterministic endpoints, not a double-sweep estimate."""
    best_distance, best_pair, best_path = 0, None, []
    for left in sorted(nodes):
        distances, parents = _bfs(left, adjacency)
        for right in sorted(nodes):
            if left >= right:
                continue
            distance = distances[right]
            if best_pair is None or distance > best_distance:
                best_distance, best_pair = distance, (left, right)
                best_path = [right]
                while best_path[-1] != left:
                    best_path.append(parents[best_path[-1]])
                best_path.reverse()
    return best_distance, best_pair, best_path


def _cuts(nodes, adjacency):
    """Iterative Tarjan traversal: bridges, side sizes and articulation nodes."""
    start = min(nodes)
    discovery, low, parent, subtree = {start: 0}, {start: 0}, {start: None}, {start: 1}
    children = Counter()
    stack = [(start, iter(sorted(adjacency[start])))]
    bridges, articulation = [], set()
    while stack:
        node, neighbors = stack[-1]
        neighbor = next(neighbors, None)
        if neighbor is not None:
            if neighbor == parent[node]:
                continue
            if neighbor not in discovery:
                parent[neighbor] = node
                children[node] += 1
                discovery[neighbor] = low[neighbor] = len(discovery)
                subtree[neighbor] = 1
                stack.append((neighbor, iter(sorted(adjacency[neighbor]))))
            else:
                low[node] = min(low[node], discovery[neighbor])
            continue
        stack.pop()
        ancestor = parent[node]
        if ancestor is None:
            if children[node] > 1:
                articulation.add(node)
        else:
            subtree[ancestor] += subtree[node]
            low[ancestor] = min(low[ancestor], low[node])
            if low[node] > discovery[ancestor]:
                side = subtree[node]
                bridges.append((_pair(node, ancestor), min(side, len(nodes) - side),
                                max(side, len(nodes) - side)))
            if parent[ancestor] is not None and low[node] >= discovery[ancestor]:
                articulation.add(ancestor)
    return sorted(bridges), sorted(articulation)


def _load_method(run_dir, method):
    assignment = pd.read_csv(run_dir / f"{method}_assignments.csv",
                             usecols=["record_id", "group_id", "split"],
                             dtype=str, keep_default_na=False)
    if (assignment.empty or assignment.record_id.eq("").any()
            or assignment.record_id.duplicated().any() or assignment.group_id.eq("").any()
            or not assignment.split.isin(("train", "validation", "test")).all()):
        raise ValueError(f"invalid {method} assignments")
    if assignment.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError(f"{method} group spans saved splits")
    groups = defaultdict(list)
    group_by_record = dict(zip(assignment.record_id, assignment.group_id))
    for record, group in group_by_record.items():
        groups[group].append(record)
    adjacency = {record: set() for record in group_by_record}
    source = pd.read_csv(run_dir / f"{method}_matching_edges.csv", dtype=str,
                         keep_default_na=False)
    if not set(PAIR_COLUMNS) <= set(source.columns):
        raise ValueError(f"missing {method} edge endpoints")
    evidence = defaultdict(list)
    for row in source.to_dict("records"):
        left, right = (row[column] for column in PAIR_COLUMNS)
        if not left or not right or left == right:
            raise ValueError(f"invalid {method} edge endpoints")
        if left not in group_by_record or right not in group_by_record:
            raise ValueError(f"{method} edge endpoint absent from assignments")
        if group_by_record[left] != group_by_record[right]:
            raise ValueError(f"{method} edge crosses saved groups")
        pair = _pair(left, right)
        evidence[pair].append({key: str(value) for key, value in row.items() if key not in PAIR_COLUMNS})
        adjacency[left].add(right)
        adjacency[right].add(left)
    for group, nodes in groups.items():
        if len(_components(nodes, adjacency)) != 1:
            raise ValueError(f"{method} graph components do not reconstruct saved group: {group}")
    size_path = run_dir / f"{method}_group_sizes.csv"
    if size_path.exists():
        sizes = pd.read_csv(size_path, dtype={"group_id": str})
        if (sizes.group_id.duplicated().any()
                or dict(zip(sizes.group_id, sizes.rows)) != {group: len(nodes) for group, nodes in groups.items()}):
            raise ValueError(f"{method} saved group sizes differ from assignments")
    return groups, group_by_record, adjacency, evidence, len(source)


def _load_annotations(annotations, records):
    if annotations is None:
        return []
    frame = (annotations.copy() if isinstance(annotations, pd.DataFrame)
             else pd.read_csv(annotations, dtype=str, keep_default_na=False))
    if not {*PAIR_COLUMNS, "leakage_decision"} <= set(frame.columns):
        raise ValueError("annotations require endpoints and leakage_decision")
    if not frame.leakage_decision.isin(("keep_together", "keep_separate", "uncertain")).all():
        raise ValueError("invalid leakage_decision")
    if "judgment_uses_Segment" in frame and not frame.judgment_uses_Segment.astype(str).str.casefold().eq("false").all():
        raise ValueError("review judgments must exclude Segment")
    seen, result = set(), []
    for row in frame.to_dict("records"):
        left, right = (str(row[column]) for column in PAIR_COLUMNS)
        pair = _pair(left, right)
        if not left or not right or left == right or pair in seen:
            raise ValueError("review pairs must be unique, nonempty and distinct")
        if not set(pair) <= records:
            raise ValueError("review endpoint absent from assignments")
        seen.add(pair)
        if left > right:
            # Keep the permitted listing evidence aligned with canonical IDs.
            swapped = row.copy()
            for column in row:
                if column.startswith("left_") and "right_" + column[5:] in row:
                    other = "right_" + column[5:]
                    swapped[column], swapped[other] = row[other], row[column]
            row = swapped
        result.append({**row, "left_record_id": pair[0], "right_record_id": pair[1]})
    return result


def _path_evidence(path, evidence):
    return [{"left_record_id": _pair(left, right)[0], "right_record_id": _pair(left, right)[1],
             "evidence": sorted(evidence[_pair(left, right)], key=_json)}
            for left, right in zip(path, path[1:])]


def _candidates(paths, bridges, limit=30):
    """Identifier-only endpoint/bridge sample from each method's largest groups."""
    selected, order = {}, []
    largest = {(row["method"], row["group_id"]) for row in paths}
    choices = []
    for row in paths:
        if row["left_record_id"]:
            choices.append({**row, "candidate_type": "maximum_graph_distance", "pair_amplification": ""})
    for row in sorted(bridges, key=lambda row: (-row["pair_amplification"], row["method"], row["group_id"], row["left_record_id"], row["right_record_id"])):
        if (row["method"], row["group_id"]) in largest:
            choices.append({**row, "candidate_type": "bridge_connection", "shortest_path_hops": 1})
    for row in choices:
        pair = _pair(row["left_record_id"], row["right_record_id"])
        context = {key: row[key] for key in ("method", "group_id", "group_size", "candidate_type", "shortest_path_hops", "pair_amplification")}
        for key in ("path_record_ids_json", "path_edge_evidence_json", "evidence_json",
                    "smaller_component_size", "larger_component_size"):
            if key in row:
                context[key] = row[key]
        if pair not in selected:
            if len(selected) == limit:
                continue
            selected[pair] = {**context, "left_record_id": pair[0], "right_record_id": pair[1], "contexts": []}
            order.append(pair)
        if context not in selected[pair]["contexts"]:
            selected[pair]["contexts"].append(context)
    result = []
    for pair in order:
        row = selected[pair]
        contexts = row.pop("contexts")
        row["graph_contexts_json"] = _json(contexts)
        row["methods"] = ";".join(sorted({context["method"] for context in contexts}))
        result.append(row)
    return result


def _verify_edge_snapshot(run_dir, snapshot_path):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    for method in METHODS:
        path = Path(run_dir) / f"{method}_matching_edges.csv"
        expected = snapshot.get("input_sha256", {}).get(str(path.resolve()))
        if expected is None or _hash(path) != expected:
            raise ValueError(f"accepted edge snapshot changed or incomplete: {method}")
    return snapshot


def sample_saved_path_edges(run_dir, registry_csv, endpoint_pair_ids, *,
                            extra_edges_per_group=4, seed=20261011, edge_snapshot=None,
                            verify_seal=True):
    """Select direct path edges and endpoint-diverse extras using identifiers only.

    Registry text fields and annotations are never read. Random tie breaking
    spreads the additional direct edges across as many new endpoints as possible.
    Returned canonical pairs may carry contexts for more than one saved method.
    """
    run_dir = Path(run_dir)
    if (not isinstance(extra_edges_per_group, int) or isinstance(extra_edges_per_group, bool)
            or extra_edges_per_group < 0):
        raise ValueError("extra_edges_per_group must be a nonnegative integer")
    if verify_seal:
        verify_refinement_seal(run_dir)
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    registry = pd.read_csv(registry_csv, usecols=["pair_id", "method", "group_id", *PAIR_COLUMNS],
                           dtype=str, keep_default_na=False)
    selected_registry = registry.loc[registry.pair_id.isin(endpoint_pair_ids)].copy()
    if (selected_registry.pair_id.duplicated().any()
            or set(selected_registry.pair_id) != set(endpoint_pair_ids)):
        raise ValueError("requested endpoint pairs are missing or duplicated in registry")
    if not selected_registry.method.isin(METHODS).all():
        raise ValueError("invalid registry method")
    graphs = {method: _load_method(run_dir, method) for method in selected_registry.method.unique()}
    rng, pairs, order = random.Random(seed), {}, []
    for source in selected_registry.sort_values("pair_id").to_dict("records"):
        method, group = source["method"], source["group_id"]
        groups, by_record, adjacency, evidence, _ = graphs[method]
        endpoints = (source["left_record_id"], source["right_record_id"])
        if any(by_record.get(endpoint) != group for endpoint in endpoints):
            raise ValueError("registry endpoints do not share the saved registry group")
        path = _path(*endpoints, adjacency)
        path_pairs = [_pair(left, right) for left, right in zip(path, path[1:])]
        chosen = [(pair, "accepted_shortest_path_edge", position)
                  for position, pair in enumerate(path_pairs, 1)]
        pool = [pair for pair in sorted(evidence) if by_record[pair[0]] == group and pair not in path_pairs]
        rng.shuffle(pool)
        covered = set(path)
        for _ in range(min(extra_edges_per_group, len(pool))):
            index = max(range(len(pool)), key=lambda index: len(set(pool[index]) - covered))
            pair = pool.pop(index)
            covered.update(pair)
            chosen.append((pair, "additional_direct_edge_endpoint_coverage", ""))
        for pair, candidate_type, position in chosen:
            context = {"source_endpoint_pair_id": source["pair_id"], "method": method,
                       "group_id": group, "group_size": len(groups[group]), "candidate_type": candidate_type,
                       "path_edge_position": position, "source_endpoint_shortest_path_hops": len(path) - 1,
                       "evidence": sorted(evidence[pair], key=_json)}
            if pair not in pairs:
                pairs[pair] = {"left_record_id": pair[0], "right_record_id": pair[1],
                               "method": method, "group_id": group, "group_size": len(groups[group]),
                               "candidate_type": candidate_type, "contexts": []}
                order.append(pair)
            pairs[pair]["contexts"].append(context)
    result = []
    for pair in order:
        row = pairs[pair]
        contexts = row.pop("contexts")
        row["graph_contexts_json"] = _json(contexts)
        row["methods"] = ";".join(sorted({context["method"] for context in contexts}))
        row["selection_seed"] = seed
        result.append(row)
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    if verify_seal:
        verify_refinement_seal(run_dir)
    return pd.DataFrame(result)


def audit_group_graph(run_dir, annotations=None, output_dir=None, *,
                      screening_thresholds=(10, 20, 50), largest_groups=10,
                      edge_snapshot=None, verify_seal=True):
    """Return tables and optionally save an audit of frozen grouping artifacts.

    ``annotations`` accepts a CSV path or DataFrame with endpoints and the
    separate leakage decision. Product identity never determines bad edges.
    ``verify_seal=False`` supports synthetic test fixtures; real audits require
    the existing experiment seal. Existing output folders cannot be overwritten.
    """
    run_dir = Path(run_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite graph audit: {output_dir}")
        if output_dir.resolve() == run_dir.resolve() or run_dir.resolve() in output_dir.resolve().parents:
            raise ValueError("graph audit output must be outside the frozen run")
    thresholds = tuple(sorted(set(screening_thresholds)))
    if (not thresholds or any(not isinstance(value, int) or isinstance(value, bool) or value < 2 for value in thresholds)
            or not isinstance(largest_groups, int) or isinstance(largest_groups, bool) or largest_groups < 1):
        raise ValueError("screening thresholds must be integers >=2 and largest_groups >=1")
    seal = verify_refinement_seal(run_dir) if verify_seal else None
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    graphs = {method: _load_method(run_dir, method) for method in METHODS}
    records = set(graphs[METHODS[0]][1])
    if any(set(graph[1]) != records for graph in graphs.values()):
        raise ValueError("methods do not contain identical record IDs")
    reviewed = _load_annotations(annotations, records)
    tables = {key: [] for key in ("group_metrics", "canonical_edges", "articulation_points", "bridge_edges",
                                  "largest_group_paths", "reviewed_pair_paths", "removal_counterfactuals")}
    summaries = {}
    for method, (groups, group_by_record, adjacency, evidence, source_edge_rows) in graphs.items():
        largest = set(sorted(groups, key=lambda group: (-len(groups[group]), group))[:largest_groups])
        reviews_by_group = defaultdict(list)
        removed_by_group = defaultdict(set)
        for row in reviewed:
            pair = (row["left_record_id"], row["right_record_id"])
            left_group, right_group = (group_by_record[endpoint] for endpoint in pair)
            together = left_group == right_group
            path = _path(*pair, adjacency) if together else None
            if together:
                reviews_by_group[left_group].append(row)
                if row["leakage_decision"] == "keep_separate" and pair in evidence:
                    removed_by_group[left_group].add(pair)
            tables["reviewed_pair_paths"].append({
                "method": method, "pair_id": str(row.get("pair_id", "")),
                "review_set": str(row.get("review_set", "")), "left_record_id": pair[0], "right_record_id": pair[1],
                "leakage_decision": row["leakage_decision"], "leakage_confidence": str(row.get("leakage_confidence", "")),
                "grouped": together, "left_group_id": left_group, "right_group_id": right_group,
                "group_size": len(groups[left_group]) if together else "", "direct_edge": pair in evidence,
                "connection_type": "direct" if pair in evidence else "transitive_only" if together else "separate_groups",
                "shortest_path_hops": len(path) - 1 if path else "", "path_record_ids_json": _json(path or []),
                "path_edge_evidence_json": _json(_path_evidence(path, evidence)) if path else "[]",
            })
        group_edges = defaultdict(list)
        for pair, rows in sorted(evidence.items()):
            group = group_by_record[pair[0]]
            group_edges[group].append(pair)
            tables["canonical_edges"].append({
                "method": method, "group_id": group, "left_record_id": pair[0], "right_record_id": pair[1],
                "source_edge_rows": len(rows), "reasons_json": _json(sorted({row.get("reason", "") for row in rows})),
                "evidence_json": _json(sorted(rows, key=_json)), "reviewed_keep_separate": pair in removed_by_group[group],
            })
        for group in sorted(groups):
            nodes, edges, reviews = groups[group], group_edges[group], reviews_by_group[group]
            size, implied = len(nodes), _choose_two(len(nodes))
            counts = Counter(row["leakage_decision"] for row in reviews)
            status = ("reviewed_conflict" if counts["keep_separate"] else
                      "reviewed_all_pairs_supported" if implied and counts["keep_together"] == implied else
                      "partially_reviewed_no_conflict" if counts["keep_together"] else "unknown")
            cuts, articulation = _cuts(nodes, adjacency)
            metrics = {"method": method, "group_id": group, "group_size": size,
                       "direct_pairs": len(edges), "implied_co_group_pairs": implied,
                       "transitive_only_pairs": implied - len(edges), "density": len(edges) / implied if implied else 0.0,
                       "bridge_edges": len(cuts), "articulation_points": len(articulation),
                       "diameter": "", "diameter_computed": group in largest,
                       "group_review_status": status, "reviewed_co_group_pairs": len(reviews),
                       "reviewed_keep_together_pairs": counts["keep_together"],
                       "reviewed_keep_separate_pairs": counts["keep_separate"], "reviewed_uncertain_pairs": counts["uncertain"],
                       "reviewed_pair_fraction": len(reviews) / implied if implied else 0.0}
            if group in largest:
                distance, pair, path = _diameter(nodes, adjacency)
                metrics["diameter"] = distance
                tables["largest_group_paths"].append({
                    "method": method, "group_id": group, "group_size": size,
                    "left_record_id": pair[0] if pair else "", "right_record_id": pair[1] if pair else "",
                    "shortest_path_hops": distance, "path_record_ids_json": _json(path),
                    "path_edge_evidence_json": _json(_path_evidence(path, evidence)),
                })
            tables["group_metrics"].append(metrics)
            for pair, smaller, larger in cuts:
                tables["bridge_edges"].append({
                    "method": method, "group_id": group, "group_size": size,
                    "left_record_id": pair[0], "right_record_id": pair[1],
                    "smaller_component_size": smaller, "larger_component_size": larger,
                    "pair_amplification": smaller * larger,
                    "evidence_json": _json(sorted(evidence[pair], key=_json)),
                    "reviewed_keep_separate": pair in removed_by_group[group],
                })
            for node in articulation:
                components = _components(nodes, adjacency, removed_node=node)
                sizes = [len(component) for component in components]
                tables["articulation_points"].append({
                    "method": method, "group_id": group, "group_size": size, "record_id": node,
                    "component_sizes_after_node_removal_json": _json(sizes),
                    "lost_pairs_between_remaining_nodes": _choose_two(size - 1) - sum(map(_choose_two, sizes)),
                })
            if counts["keep_separate"]:
                removed = removed_by_group[group]
                components = _components(nodes, adjacency, removed)
                sizes = [len(component) for component in components]
                component_by_node = {node: index for index, component in enumerate(components) for node in component}
                remaining = sum(map(_choose_two, sizes))
                negatives = [row for row in reviews if row["leakage_decision"] == "keep_separate"]
                still_together = sum(component_by_node[row["left_record_id"]] == component_by_node[row["right_record_id"]] for row in negatives)
                tables["removal_counterfactuals"].append({
                    "method": method, "group_id": group, "group_size": size,
                    "scenario": "simultaneous_removal_of_all_reviewed_keep_separate_direct_edges",
                    "removed_direct_edges": len(removed), "removed_edge_pairs_json": _json(sorted(removed)),
                    "component_sizes_json": _json(sizes), "components_after_removal": len(sizes),
                    "implied_pairs_before": implied, "implied_pairs_after": remaining,
                    "implied_pair_reduction": implied - remaining,
                    "reviewed_keep_separate_pairs_before": len(negatives), "reviewed_keep_separate_pairs_still_connected": still_together,
                    "reviewed_keep_separate_pairs_disconnected": len(negatives) - still_together,
                })
        for row in tables["reviewed_pair_paths"]:
            if row["method"] == method:
                removed = removed_by_group[row["left_group_id"]] if row["grouped"] else frozenset()
                row["connected_after_reviewed_bad_edge_removal"] = bool(row["grouped"] and _path(
                    row["left_record_id"], row["right_record_id"], adjacency, removed))
        method_metrics = [row for row in tables["group_metrics"] if row["method"] == method]
        size_series = pd.Series([len(nodes) for nodes in groups.values()])
        counterfactuals = [row for row in tables["removal_counterfactuals"] if row["method"] == method]
        summaries[method] = {
            "records": len(records), "groups": len(groups), "singleton_groups": int(size_series.eq(1).sum()),
            "largest_group": int(size_series.max()), "source_edge_rows": source_edge_rows, "canonical_direct_edges": len(evidence),
            "duplicate_edge_rows": source_edge_rows - len(evidence),
            "implied_co_group_pairs": sum(row["implied_co_group_pairs"] for row in method_metrics),
            "transitive_only_pairs": sum(row["transitive_only_pairs"] for row in method_metrics),
            "bridge_edges": sum(row["bridge_edges"] for row in method_metrics),
            "articulation_points": sum(row["articulation_points"] for row in method_metrics),
            "group_size_quantiles": {str(quantile): float(size_series.quantile(quantile)) for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)},
            "screening_tail": [{"minimum_group_size": threshold, "groups": int(size_series.ge(threshold).sum()),
                                "records": int(size_series.loc[size_series.ge(threshold)].sum()),
                                "implied_co_group_pairs": sum(row["implied_co_group_pairs"] for row in method_metrics if row["group_size"] >= threshold)} for threshold in thresholds],
            "reviewed_conflict_groups": sum(row["group_review_status"] == "reviewed_conflict" for row in method_metrics),
            "reviewed_keep_separate_co_group_pairs": sum(row["reviewed_keep_separate_pairs"] for row in method_metrics),
            "counterfactual_removed_direct_edges": sum(row["removed_direct_edges"] for row in counterfactuals),
            "counterfactual_implied_pair_reduction": sum(row["implied_pair_reduction"] for row in counterfactuals),
            "counterfactual_reviewed_keep_separate_pairs_still_connected": sum(row["reviewed_keep_separate_pairs_still_connected"] for row in counterfactuals),
        }
    tables["candidates"] = _candidates(tables["largest_group_paths"], tables["bridge_edges"])
    summary = {"methods": summaries, "screening_thresholds": list(thresholds), "largest_groups_per_method": largest_groups,
               "reviewed_pairs": len(reviewed), "interpretation": (
                   "Size thresholds and graph structure screen for review; they do not establish excessive or incorrect groups. "
                   "Only leakage_decision=keep_separate supplies reviewed negative evidence. Partial review does not certify an entire group. "
                   "All reviewed bad direct links are removed simultaneously in diagnostic counterfactuals; unreviewed path edges remain. "
                   "Implied-pair reduction measures connectivity lost, not a count of proven incorrect pairs. Actual saved assignments are unchanged.")}
    input_paths = [run_dir / f"{method}_{suffix}.csv" for method in METHODS for suffix in ("assignments", "matching_edges")]
    input_paths += [run_dir / f"{method}_group_sizes.csv" for method in METHODS if (run_dir / f"{method}_group_sizes.csv").exists()]
    input_paths += [run_dir / name for name in ("config.json", "input_manifest.json", "refinement_freeze.json") if (run_dir / name).exists()]
    if annotations is not None and not isinstance(annotations, pd.DataFrame):
        input_paths.append(Path(annotations))
    if edge_snapshot is not None:
        input_paths.append(Path(edge_snapshot))
    verification = {"frozen_seal_verified": verify_seal, "edges_within_saved_groups": True,
                    "components_reconstruct_saved_groups_including_singletons": True, "identical_method_record_ids": True,
                    "no_matching_or_assignment_changes": True,
                    "accepted_edge_snapshot_verified": edge_snapshot is not None,
                    "audit_module_sha256": _hash(Path(__file__)),
                    "annotations_canonical_sha256": hashlib.sha256(_json(sorted(reviewed, key=lambda row: (row["left_record_id"], row["right_record_id"]))).encode()).hexdigest(),
                    "accepted_edge_files_in_original_seal": {
                        method: bool(seal and str((run_dir / f"{method}_matching_edges.csv").resolve()) in seal["sha256"])
                        for method in METHODS},
                    "input_sha256": {str(path.resolve()): _hash(path) for path in input_paths}}
    if verify_seal:
        verify_refinement_seal(run_dir)
    if edge_snapshot is not None:
        _verify_edge_snapshot(run_dir, edge_snapshot)
    frames = {key: pd.DataFrame(rows) if rows else pd.DataFrame(columns=EMPTY_COLUMNS.get(key, []))
              for key, rows in tables.items()}
    if reviewed and any("left_ProductName" in row and "right_ProductName" in row for row in reviewed):
        names = {row[side + "_record_id"]: str(row.get(side + "_ProductName", ""))
                 for row in reviewed for side in ("left", "right")}
        decisions = {_pair(row["left_record_id"], row["right_record_id"]): row["leakage_decision"]
                     for row in reviewed}
        rationales = {_pair(row["left_record_id"], row["right_record_id"]): str(row.get("leakage_rationale", ""))
                      for row in reviewed}
        named = frames["reviewed_pair_paths"].copy()
        for side in ("left", "right"):
            named[side + "_ProductName"] = named[side + "_record_id"].map(names).fillna("")
        named["leakage_rationale"] = [rationales[_pair(a, b)] for a, b in named[list(PAIR_COLUMNS)].itertuples(index=False, name=None)]
        frames["named_reviewed_pairs"] = named
        steps = []
        for row in named.to_dict("records"):
            path = json.loads(row["path_record_ids_json"])
            for position, (left, right) in enumerate(zip(path, path[1:]), 1):
                steps.append({"method": row["method"], "endpoint_pair_id": row["pair_id"],
                              "path_step": position, "left_record_id": left, "right_record_id": right,
                              "left_ProductName": names.get(left, ""), "right_ProductName": names.get(right, ""),
                              "reviewed_edge_leakage_decision": decisions.get(_pair(left, right), "unreviewed")})
        frames["named_connection_paths"] = pd.DataFrame(steps, columns=["method", "endpoint_pair_id", "path_step",
            *PAIR_COLUMNS, "left_ProductName", "right_ProductName", "reviewed_edge_leakage_decision"])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        for key, frame in frames.items():
            frame.to_csv(output_dir / f"{key}.csv", index=False, lineterminator="\n")
        for name, value in (("summary", summary), ("verification", verification)):
            (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {**frames, "summary": summary, "verification": verification}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screening-thresholds", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--largest-groups", type=int, default=10)
    parser.add_argument("--edge-snapshot", type=Path)
    arguments = parser.parse_args(argv)
    result = audit_group_graph(arguments.run, arguments.annotations, arguments.output,
                               screening_thresholds=arguments.screening_thresholds, largest_groups=arguments.largest_groups,
                               edge_snapshot=arguments.edge_snapshot)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
