import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.segment_experiment import (make_development_assignments,
    validate_development_assignments, write_json)


class DevelopmentProtocolTests(unittest.TestCase):
    def fixture(self):
        ids = [f"r{i:02d}" for i in range(16)]
        frame = pd.DataFrame({"record_id": ids,
            "Segment": ["A"] * 8 + ["B"] * 7 + ["null"]})
        roles = ["test", "test", "train", "train", "validation", "train", "train", "validation",
                 "train", "train", "validation", "train", "train", "validation", "train", "train"]
        groups = ["g0", "g0", "g1", "g1", "g2", "g3", "g4", "g5",
                  "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13"]
        frozen = pd.DataFrame({"record_id": ids, "group_id": groups, "split": roles})
        other = frozen.copy()
        other.loc[other.record_id.eq("r02"), "split"] = "test"
        return frame, frozen, other

    def test_union_embargo_is_closed_over_groups_and_missing_labels_excluded(self):
        frame, frozen, other = self.fixture()
        development, assignments, excluded, counts = make_development_assignments(frame, frozen, [frozen, other])
        self.assertEqual(set(excluded.record_id), {"r00", "r01", "r02", "r03", "r15"})
        self.assertEqual(excluded.set_index("record_id").at["r03", "reason"], "protected_rule_group_sibling")
        self.assertEqual(counts["direct_test_union_records"], 3)
        self.assertEqual(counts["group_closed_test_embargo_records"], 4)
        self.assertFalse(development.Segment.eq("null").any())
        self.assertEqual(set(development.record_id), set(assignments.record_id))
        self.assertTrue(validate_development_assignments(assignments, excluded))

    def test_random_control_matches_every_class_count_and_is_row_order_reproducible(self):
        frame, frozen, other = self.fixture()
        first = make_development_assignments(frame, frozen, [frozen, other], seed=11)
        second = make_development_assignments(frame.sample(frac=1, random_state=91), frozen.iloc[::-1], [other, frozen], seed=11)
        pd.testing.assert_frame_equal(first[1], second[1])
        counts = first[1].groupby(["strategy", "Segment", "split"]).size()
        pd.testing.assert_series_equal(counts.loc["random"], counts.loc["group_aware"])

    def test_protocol_rejects_test_and_embargo_contamination(self):
        frame, frozen, other = self.fixture()
        _, assignment, excluded, _ = make_development_assignments(frame, frozen, [frozen, other])
        bad = assignment.copy()
        bad.at[0, "split"] = "test"
        with self.assertRaises(ValueError):
            validate_development_assignments(bad, excluded)
        with self.assertRaises(ValueError):
            validate_development_assignments(assignment, pd.DataFrame({"record_id": [assignment.at[0, "record_id"]]}))

    def test_missing_snapshot_record_or_invalid_group_rejected(self):
        frame, frozen, other = self.fixture()
        with self.assertRaises(ValueError):
            make_development_assignments(frame, frozen, [other.iloc[1:]])
        broken = frozen.copy()
        broken.loc[broken.record_id.eq("r01"), "split"] = "validation"
        with self.assertRaises(ValueError):
            make_development_assignments(frame, broken, [other])

    def test_tables_are_saved_as_json_records_for_bootstrap_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "intervals.json"
            write_json(path, {"intervals": pd.DataFrame({"metric": ["accuracy"], "lower": [0.8]})})
            self.assertEqual(json.loads(path.read_text())["intervals"], [{"metric": "accuracy", "lower": 0.8}])


if __name__ == "__main__":
    unittest.main()
