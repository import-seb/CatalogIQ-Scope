"""Synthetic discovery tests; no reviewed or held-out product pairs enter here."""
from dataclasses import replace
import inspect
import random
import unittest

from catalogiq.split_rule_candidates_v3 import RuleCandidateConfig, generate_candidates


def id_candidates(rows, config=None):
    names, brands, ids, cores = zip(*rows) if rows else ((), (), (), ())
    pairs, stats = generate_candidates(names, brands, ids, cores, config)
    return {(ids[left], ids[right]): channels for left, right, channels in pairs}, stats


class CandidateDiscoveryV3Tests(unittest.TestCase):
    def test_rare_variant_tokens_no_longer_hide_shared_line(self):
        rows = [
            ("cedar frostshield coolant blue winter", "cedar", "b", {"frostshield", "blue", "winter"}),
            ("cedar frostshield coolant red summer", "cedar", "a", {"frostshield", "red", "summer"}),
            # Make the line token less rare than either exclusive variant.
            ("cedar frostshield refill", "cedar", "z", {"frostshield", "refill"}),
        ]
        pairs, _ = id_candidates(rows, replace(RuleCandidateConfig(), block_tokens=1))
        self.assertIn(("a", "b"), pairs)
        self.assertNotIn("rare_token:brand", pairs["a", "b"])
        self.assertIn("name_prefix:brand_line", pairs["a", "b"])

    def test_empty_cores_receive_exact_and_brand_line_proposals(self):
        rows = [
            ("alder maximum strength softgel", "alder", "2", set()),
            ("alder maximum strength softgel", "alder", "1", set()),
            ("alder maximum strength rapid softgel", "alder", "3", set()),
        ]
        pairs, stats = id_candidates(rows)
        self.assertIn("normalized_exact_name", pairs["1", "2"])
        self.assertIn("name_prefix:brand_line", pairs["1", "3"])
        self.assertEqual(stats["empty_core_rows"], 3)
        self.assertEqual(stats["rows_with_candidates"], 3)

    def test_character_fallback_does_not_require_token_overlap_or_prefix(self):
        rows = [
            ("alder thermashield lubricant coolant", "alder", "a", {"thermashield"}),
            ("alder thermoshield lubricant coolant", "alder", "b", {"thermoshield"}),
        ]
        pairs, stats = id_candidates(rows)
        self.assertEqual(pairs["a", "b"], ("character_trigram_neighborhood",))
        self.assertEqual(stats["character_comparisons"], 1)

    def test_global_rare_core_discovers_missing_title_brand(self):
        rows = [
            ("ocean ocularveil preservative free mist", "ocean", "a", {"ocularveil"}),
            ("ocularveil preservative free mist", "ocean health", "b", {"ocularveil"}),
        ]
        pairs, _ = id_candidates(rows)
        self.assertIn("rare_token:global_rare", pairs["a", "b"])

    def test_channels_combine_without_repeated_pairs_or_self_pairs(self):
        rows = [
            ("beech lumivex powder", "beech", "z", {"lumivex"}),
            ("beech lumivex powder", "beech", "a", {"lumivex"}),
        ]
        names, brands, ids, cores = zip(*rows)
        pairs, stats = generate_candidates(names, brands, ids, cores)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((ids[pairs[0][0]], ids[pairs[0][1]]), ("a", "z"))
        self.assertEqual(len(pairs[0][2]), len(set(pairs[0][2])))
        self.assertIn("normalized_exact_name", pairs[0][2])
        self.assertIn("character_trigram_neighborhood", pairs[0][2])
        self.assertEqual(stats["candidate_pairs"], 1)

    def test_record_permutations_preserve_pairs_provenance_and_statistics(self):
        rows = [(f"oak prism{i % 4} mist variant{i}", "oak", f"id{i:03d}", {f"prism{i % 4}", f"variant{i}"})
                for i in range(34)]
        config = replace(RuleCandidateConfig(), max_exhaustive_block=6, char_top_k=3,
                         max_character_comparisons=100)
        expected = id_candidates(rows, config)
        for seed in (7, 11, 32):
            shuffled = rows.copy()
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(id_candidates(shuffled, config), expected)

    def test_large_exact_block_keeps_every_record_and_does_not_cap_groups(self):
        rows = [("elm crystallux lotion", "elm", f"id{i:04d}", {"crystallux"}) for i in range(210)]
        pairs, stats = id_candidates(rows, replace(RuleCandidateConfig(), max_exhaustive_block=8,
                                                  large_block_window=2, sorted_neighbor_window=2,
                                                  char_top_k=2))
        adjacency = {row[2]: set() for row in rows}
        for left, right in pairs:
            adjacency[left].add(right)
            adjacency[right].add(left)
        reached, stack = set(), [rows[0][2]]
        while stack:
            node = stack.pop()
            if node not in reached:
                reached.add(node)
                stack.extend(adjacency[node] - reached)
        self.assertEqual(len(reached), len(rows))
        self.assertEqual(stats["rows_with_candidates"], 210)
        self.assertFalse(stats["group_size_cap_applied"])
        self.assertGreater(stats["normalized_exact_name_block_comparisons_omitted"], 0)
        self.assertLess(stats["character_comparisons"], 210 * 10)

    def test_character_budget_is_disclosed_and_preserves_other_channels(self):
        rows = [(f"fir lucent mist {i}", "fir", f"id{i}", {"lucent"}) for i in range(6)]
        pairs, stats = id_candidates(rows, replace(RuleCandidateConfig(), max_character_comparisons=2))
        self.assertEqual(stats["character_comparisons"], 2)
        self.assertTrue(stats["character_comparison_budget_exhausted"])
        self.assertEqual(stats["character_proposals_omitted_by_budget"], 13)
        self.assertEqual(len(pairs), 15)

    def test_character_top_k_ties_are_resolved_by_identity(self):
        rows = [("fir lucent mist", "fir", record_id, set()) for record_id in ("c", "a", "d", "b")]
        pairs, stats = id_candidates(rows, replace(RuleCandidateConfig(), char_top_k=1))
        char_pairs = {pair for pair, channels in pairs.items() if "character_trigram_neighborhood" in channels}
        self.assertEqual(char_pairs, {("a", "b"), ("a", "c"), ("a", "d")})
        self.assertEqual(stats["character_directed_neighbors_omitted_by_top_k"], 8)

    def test_empty_names_do_not_generate_generic_empty_candidates(self):
        rows = [("", "fir", "a", set()), ("", "fir", "b", set()),
                ("fir lumen mist", "fir", "c", {"lumen"})]
        pairs, stats = id_candidates(rows)
        self.assertEqual(pairs, {})
        self.assertEqual(stats["candidate_pairs"], 0)

    def test_feature_only_api_and_inputs_are_preserved(self):
        rows = [("aspen verilium powder", "aspen", "a", {"verilium"}),
                ("aspen verilium mist", "aspen", "b", {"verilium"})]
        before = [(name, brand, identity, core.copy()) for name, brand, identity, core in rows]
        _, stats = id_candidates(rows)
        self.assertEqual(rows, before)
        self.assertEqual(tuple(inspect.signature(generate_candidates).parameters),
                         ("names", "brands", "ids", "cores", "config"))
        self.assertFalse(stats["targets_used"])
        self.assertFalse(stats["annotations_used"])
        self.assertFalse(stats["candidate_similarity_is_confirmation"])

    def test_invalid_arrays_identities_and_configurations_are_rejected(self):
        for args in ((["x"], [], ["a"], [set()]),
                     (["x", "y"], ["", ""], ["a", "a"], [set(), set()]),
                     (["x"], [""], [" "], [set()])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_candidates(*args)
        for config in (replace(RuleCandidateConfig(), block_tokens=0),
                       replace(RuleCandidateConfig(), char_top_k=True),
                       replace(RuleCandidateConfig(), char_candidate_threshold=float("nan")),
                       replace(RuleCandidateConfig(), max_character_comparisons=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                generate_candidates([], [], [], [], config)


if __name__ == "__main__":
    unittest.main()
