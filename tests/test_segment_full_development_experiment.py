import unittest

import pandas as pd

from catalogiq.segment_full_development_experiment import full_development_pool


class FullDevelopmentPoolTests(unittest.TestCase):
    def fixture(self):
        source = pd.DataFrame({"record_id": ["a", "b", "c", "d", "e", "f"],
            "Segment": ["A", "A", "B", "B", "null", "A"],
            "Retailer": ["Amazon", "CVS", "Amazon", "Target", "CVS", "Walmart"]})
        frozen = pd.DataFrame({"record_id": source.record_id,
            "group_id": ["g1", "g1", "g2", "g3", "g4", "g5"],
            "split": ["train", "train", "validation", "validation", "train", "test"]})
        return source, frozen

    def test_all_labeled_non_test_rows_retained_without_other_method_embargo(self):
        source, frozen = self.fixture()
        development, excluded, counts = full_development_pool(source, frozen)
        self.assertEqual(set(development.record_id), {"a", "b", "c", "d"})
        self.assertEqual(set(excluded.record_id), {"e", "f"})
        self.assertEqual(counts["full_development_records_including_unlabeled"], 5)
        self.assertEqual(counts["missing_Segment_development_records"], 1)
        self.assertFalse(counts["other_strategy_test_embargo_applied"])
        self.assertEqual(excluded.set_index("record_id").at["f", "reason"], "protected_rule_v3_final_test")

    def test_pool_is_invariant_to_input_order(self):
        source, frozen = self.fixture()
        first = full_development_pool(source, frozen)
        second = full_development_pool(source.iloc[::-1], frozen.iloc[::-1])
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_crossing_groups_and_incomplete_snapshots_are_rejected(self):
        source, frozen = self.fixture()
        broken = frozen.copy()
        broken.loc[broken.record_id.eq("a"), "split"] = "test"
        with self.assertRaises(ValueError):
            full_development_pool(source, broken)
        with self.assertRaises(ValueError):
            full_development_pool(source, frozen.iloc[1:])


if __name__ == "__main__":
    unittest.main()
