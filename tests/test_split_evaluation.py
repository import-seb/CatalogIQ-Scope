"""Independent duplicate audits and a brute-force oracle for prefix filtering."""
import random
import unittest

import pandas as pd

from catalogiq.split_evaluation import evaluation_name, leakage_metrics, near_duplicate_pairs, segment_distribution
from catalogiq.splitting import SplitConfig


def brute_force(names, config):
    normalized = [evaluation_name(name) for name in names]
    shingles = []
    for name in normalized:
        if len(name) < max(config.evaluation_min_name_chars, config.evaluation_shingle_size):
            shingles.append(set())
        else:
            shingles.append({name[i:i + config.evaluation_shingle_size]
                             for i in range(len(name) - config.evaluation_shingle_size + 1)})
    result = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if not shingles[a] or not shingles[b]:
                continue
            similarity = len(shingles[a] & shingles[b]) / len(shingles[a] | shingles[b])
            if similarity + 1e-12 >= config.evaluation_threshold:
                result[a, b] = similarity
    return result


class IndependentEvaluationTests(unittest.TestCase):
    def assert_matches_oracle(self, names, config):
        frame = pd.DataFrame({"ProductName": names, "Segment": ["ignored"] * len(names)},
                             index=range(100, 100 + len(names)))
        pairs, stats = near_duplicate_pairs(frame, config)
        actual = {(a, b): score for a, b, score in pairs}
        self.assertEqual(actual, brute_force(names, config))
        self.assertEqual(len(pairs), len(actual))
        self.assertEqual(pairs, sorted(pairs))
        self.assertTrue(stats["exhaustive_within_scope"])
        pairs_again, stats_again = near_duplicate_pairs(frame, config)
        self.assertEqual(pairs, pairs_again)
        stats.pop("runtime_seconds")
        stats_again.pop("runtime_seconds")
        self.assertEqual(stats, stats_again)

    def test_normalization_preserves_numbers_nonascii_and_accents(self):
        self.assertEqual(evaluation_name("  Café—１２ &amp; cream\n50mL  "), "café 12 cream 50ml")
        self.assertEqual(evaluation_name("ＡＢＣ_123"), "abc 123")
        for value in (None, pd.NA, float("nan"), "null", " NULL ", ""):
            self.assertEqual(evaluation_name(value), "")
        self.assertNotEqual(evaluation_name("Bottle 120 ml"), evaluation_name("Bottle 150 ml"))

    def test_identical_short_missing_and_numeric_variants(self):
        self.assert_matches_oracle([
            "long product name cream bottle 120 ml", "LONG PRODUCT NAME CREAM BOTTLE 120 ML",
            "long product name cream bottle 150 ml", "long product name cream bottle 120ml",
            "Café crème délicate 120 ml", "Café crème délicate 125 ml", "ＡＢＣ_123",
            "tiny", "tiny", "", None, "null", "independent completely unrelated title",
        ], SplitConfig())

    def test_lossless_randomized_prefix_join_at_multiple_thresholds(self):
        rng = random.Random(451)
        alphabet = "abcdef0123456789 é"
        names = []
        for _ in range(45):
            base = "".join(rng.choices(alphabet, k=rng.randrange(12, 65)))
            names.extend([base, base])
            for _ in range(3):
                chars = list(base)
                chars[rng.randrange(len(chars))] = rng.choice(alphabet)
                names.append("".join(chars))
        for threshold in (0.3, 0.5, 0.7, 0.85, 0.95, 1.0):
            for shingle_size in (1, 3, 5):
                with self.subTest(threshold=threshold, shingle_size=shingle_size):
                    self.assert_matches_oracle(names, SplitConfig(evaluation_threshold=threshold,
                                              evaluation_shingle_size=shingle_size))

    def test_threshold_boundary_and_different_strings_with_same_shingle_set(self):
        self.assert_matches_oracle(["abababababab", "babababababa", "aaaaaaaaaaaa",
                                   "aaaaabaaaaaa", "aaaaabaaaaaaa", "aaaaabcaaaaa"],
                                  SplitConfig(evaluation_shingle_size=1, evaluation_threshold=0.5))

    def test_labels_and_non_name_features_cannot_change_audit(self):
        frame = pd.DataFrame({"ProductName": ["long product name alpha", "Long product name alpha",
                                            "long product name alphaa"]})
        expected, _ = near_duplicate_pairs(frame, SplitConfig())
        frame["Segment"] = ["SecretA", "SecretB", "SecretC"]
        frame["ProductDescription"] = ["unrelated", "identical", "identical"]
        frame["Sku"] = "same"
        actual, _ = near_duplicate_pairs(frame, SplitConfig())
        self.assertEqual(actual, expected)

    def test_empty_and_all_ineligible_inputs(self):
        for frame in (pd.DataFrame(), pd.DataFrame({"ProductName": ["x", "", None]})):
            pairs, stats = near_duplicate_pairs(frame, SplitConfig())
            self.assertEqual(pairs, [])
            self.assertEqual(stats["eligible_rows"], 0)

    def test_metrics_distinguish_exact_payload_normalized_name_and_near_pairs(self):
        frame = pd.DataFrame({
            "ProductBrand": ["Brand"] * 4,
            "ProductName": ["long product name alpha", "long product name alpha",
                            "LONG PRODUCT NAME ALPHA", "long product name alphaa"],
            "ProductDescription": ["same", "same", "changed", "changed"],
            "ProductContents": [""] * 4,
            "Segment": ["Common", "Common", "Common", "Rare"],
            "ProductUrl": ["a", "b", "c", "d"],
        }, index=[10, 20, 30, 40])
        assignments = pd.DataFrame({"group_id": ["a", "a", "b", "b"],
                                    "split": ["train", "validation", "train", "test"]})
        pairs, _ = near_duplicate_pairs(frame, SplitConfig())
        result = leakage_metrics(frame, assignments, pairs, SplitConfig())
        self.assertEqual(result["exact_payload"]["duplicate_pairs"], 1)
        self.assertEqual(result["exact_payload"]["crossing_pairs"], 1)
        self.assertEqual(result["exact_brand_name"]["duplicate_pairs"], 3)
        self.assertEqual(result["exact_brand_name"]["crossing_pairs"], 2)
        self.assertEqual(result["near_duplicate"]["pairs"], 6)
        self.assertEqual(result["near_duplicate"]["crossing_pairs"], 5)
        self.assertEqual(result["near_duplicate"]["co_grouped_pairs"], 2)
        self.assertEqual(result["near_duplicate"]["co_grouped_pair_rate"], 2 / 6)
        self.assertEqual(result["near_duplicate"]["nonidentical_name_pairs"], 3)
        self.assertEqual(result["near_duplicate"]["nonidentical_name_crossing_pairs"], 3)
        self.assertEqual(result["groups"]["count"], 2)
        self.assertEqual(result["groups"]["mixed_segment_groups"], 1)
        self.assertIn("Rare", result["rare_segment_classes_missing_validation"])
        self.assertNotIn("Rare", result["rare_segment_classes_missing_test"])
        distribution = segment_distribution(frame, assignments).set_index("Segment")
        self.assertEqual(int(distribution.loc["Common", "train_rows"]), 2)
        self.assertEqual(int(distribution.loc["Rare", "test_rows"]), 1)
        self.assertEqual(float(distribution.loc["Rare", "test_allocation_proportion"]), 1)
        self.assertEqual(float(distribution.loc["Common", "global_class_proportion"]), 0.75)
        self.assertEqual(float(distribution.loc["Common", "train_delta_from_global"]), 0.25)
        self.assertEqual(int(distribution.loc["Common", "supporting_groups"]), 2)
        self.assertEqual(result["segment_supporting_groups"], {"Common": 2, "Rare": 1})
        self.assertEqual(result["classes_not_representable_in_all_three_splits"], ["Common", "Rare"])
        self.assertEqual(result["rare_segment_class_count"], 2)

    def test_record_ids_must_match_when_both_frames_provide_them(self):
        frame = pd.DataFrame({"ProductName": ["long name product alpha", "long name product alpha"],
                              "record_id": ["first", "second"], "Segment": ["A", "A"]})
        assignments = pd.DataFrame({"record_id": ["second", "first"],
                                    "group_id": ["g", "g"], "split": ["train", "train"]})
        with self.assertRaisesRegex(ValueError, "record_id order"):
            leakage_metrics(frame, assignments, [], SplitConfig())
        with self.assertRaisesRegex(ValueError, "record_id order"):
            segment_distribution(frame, assignments)
        assignments["record_id"] = ["first", "second"]
        self.assertEqual(leakage_metrics(frame, assignments, [], SplitConfig())["rows"], 2)


if __name__ == "__main__":
    unittest.main()
