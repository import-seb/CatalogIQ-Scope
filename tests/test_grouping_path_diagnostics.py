"""Path provenance, maximum connectivity, and unchanged guard explanations."""
import itertools
import json
import unittest

import pandas as pd

from catalogiq.grouping_path_diagnostics import (
    edge_disjoint_paths, lexical_evidence, named_path_rows,
)
from catalogiq.split_matching import prepare_products
from catalogiq.splitting import SplitConfig


def graph(edges):
    result = {}
    for left, right in edges:
        result.setdefault(left, set()).add(right)
        result.setdefault(right, set()).add(left)
    return result


class GroupingPathDiagnosticsTests(unittest.TestCase):
    def test_chain_requires_sole_edge_and_removal_disconnects(self):
        adjacency = graph([(1, 2), (2, 3)])
        self.assertEqual(edge_disjoint_paths(1, 3, adjacency), [[1, 2, 3]])
        self.assertEqual(edge_disjoint_paths(1, 3, adjacency, {(2, 3)}), [])

    def test_dense_group_has_multiple_edge_disjoint_paths(self):
        adjacency = graph(itertools.combinations(range(6), 2))
        paths = edge_disjoint_paths(0, 5, adjacency)
        self.assertEqual(len(paths), 5)
        used = [tuple(sorted(pair)) for path in paths for pair in zip(path, path[1:])]
        self.assertEqual(len(used), len(set(used)))
        removed = edge_disjoint_paths(0, 5, adjacency, {(0, 5), (0, 4)})
        self.assertEqual(len(removed), 3)

    def test_augmenting_flow_beats_greedy_shortest_path_removal(self):
        # Greedily taking s-a-b-t would consume edges needed by both alternatives.
        adjacency = graph([("s", "a"), ("a", "b"), ("b", "t"),
                           ("a", "x"), ("x", "t"), ("s", "y"), ("y", "b")])
        paths = edge_disjoint_paths("s", "t", adjacency)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths, edge_disjoint_paths("s", "t", adjacency))

    def test_flow_count_matches_bruteforce_minimum_cuts(self):
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
        adjacency = graph(edges)
        for removed_count in range(3):
            for removed in itertools.combinations(edges, removed_count):
                retained = [edge for edge in edges if edge not in removed]
                minimum_cut = min(sum((a in side) != (b in side) for a, b in retained)
                                  for count in range(5)
                                  for inside in itertools.combinations([1, 2, 3, 4], count)
                                  for side in [{0, *inside}])
                self.assertEqual(len(edge_disjoint_paths(0, 5, adjacency, set(removed))), minimum_cut)

    def test_names_follow_path_direction_and_saved_evidence_stays_canonical(self):
        evidence = {("a", "b"): [{"reason": "accepted_rule", "score": "0.9012345678901234"}]}
        before = repr(evidence)
        rows = named_path_rows(["b", "a"], "rule", "saved", 1,
                               {"a": {"ProductName": "Alpha"}, "b": {"ProductName": "Beta"}},
                               evidence, {("a", "b"): {"pair_id": "R1", "leakage_decision": "keep_separate"}})
        self.assertEqual(rows[0]["left_ProductName"], "Beta")
        self.assertEqual(rows[0]["right_ProductName"], "Alpha")
        self.assertEqual(rows[0]["canonical_left_record_id"], "a")
        self.assertEqual(json.loads(rows[0]["saved_edge_evidence_json"])[0]["score"], "0.9012345678901234")
        self.assertEqual(rows[0]["reviewed_edge_leakage_decision"], "keep_separate")
        self.assertEqual(repr(evidence), before)

    def test_unreviewed_edges_remain_unreviewed(self):
        rows = named_path_rows(["a", "b"], "tfidf", "saved", 1,
                               {"a": {"ProductName": "Alpha"}, "b": {"ProductName": "Beta"}},
                               {("a", "b"): [{"reason": "tfidf_family_cosine", "score": "1.0"}]}, {})
        self.assertEqual(rows[0]["reviewed_edge_leakage_decision"], "unreviewed")

    def test_negated_filler_words_explain_core_guard_without_altering_parser(self):
        names = ["Pure Original Ingredients TMG (365 Capsules) No Magnesium Or Rice Fillers, Always Pure, Lab Verified",
                 "Pure Original Ingredients Milk Thistle (365 Capsules) No Magnesium Or Rice Fillers, Always Pure, Lab Verified"]
        frame = pd.DataFrame({"ProductName": names, "ProductBrand": ["Pure Organic Ingredients"] * 2})
        config = SplitConfig(grouping_version=2)
        left, right = prepare_products(frame, config)
        evidence = lexical_evidence(left, right, config)
        self.assertEqual(json.loads(evidence["shared_negated_filler_words_counted_as_core_json"]), ["magnesium", "rice"])
        self.assertTrue(evidence["pair_guard_passes"])
        self.assertFalse(evidence["ingredient_guard_has_two_nonempty_signatures"])
        self.assertGreaterEqual(evidence["family_token_jaccard"], config.rule_family_threshold)
        self.assertGreaterEqual(evidence["family_edit_similarity"], config.rule_family_edit_threshold)
        self.assertEqual(left.ingredients, frozenset())
        self.assertEqual(right.ingredients, frozenset())

    def test_malformed_directed_graph_and_equal_endpoints_rejected(self):
        with self.assertRaises(ValueError):
            edge_disjoint_paths("a", "b", {"a": {"b"}, "b": set()})
        with self.assertRaises(ValueError):
            edge_disjoint_paths("a", "a", {"a": set()})


if __name__ == "__main__":
    unittest.main()
