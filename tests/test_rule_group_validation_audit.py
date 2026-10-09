"""Meaningful invariant checks on synthetic saved partitions, without tuning."""
import unittest

import pandas as pd

from catalogiq.rule_group_validation_audit import audit_group_partition
from catalogiq.split_rules_v3 import RuleRefinementConfig
from catalogiq.splitting import SplitConfig


class FrozenRuleGroupAuditTests(unittest.TestCase):
    def setUp(self):
        self.split = SplitConfig(grouping_version=2)
        self.rule = RuleRefinementConfig()
        self.frame = pd.DataFrame([
            {"record_id": f"r{i}", "ProductBrand": "Willow",
             "ProductName": f"Willow Valerian Root Capsule {60 + i} Count",
             "Segment": "irrelevant"} for i in range(5)])
        self.assignment = pd.DataFrame({"record_id": self.frame.record_id,
            "group_id": ["large"] * 4 + ["single"], "Segment": "irrelevant", "split": "train"})
        self.edges = pd.DataFrame([
            ("r0", "r1", "distinctive_named_line", 1.0),
            ("r1", "r2", "distinctive_named_line", 1.0),
            ("r2", "r3", "distinctive_named_line", 1.0)],
            columns=["left_record_id", "right_record_id", "reason", "score"])

    def audit(self, frame=None, assignment=None, edges=None):
        return audit_group_partition(self.frame if frame is None else frame,
            self.assignment if assignment is None else assignment,
            self.edges if edges is None else edges, self.split, self.rule)

    def test_graph_amplification_common_anchors_and_actual_names(self):
        tables, summary = self.audit()
        row = tables["group_audit.csv"].iloc[0]
        self.assertEqual((row.group_size, row.direct_edges, row.implied_pairs, row.transitive_only_pairs), (4, 3, 6, 3))
        self.assertEqual(row.implied_to_direct_pair_ratio, 2)
        self.assertIn("word:valerian", row.common_anchors_json)
        self.assertEqual(row.representative_unsupported_pairs, 0)
        self.assertEqual(len(tables["largest_group_all_members.csv"]), 4)
        self.assertEqual(len(tables["largest_group_named_edges.csv"]), 3)
        self.assertTrue(tables["largest_group_representatives.csv"].ProductName.str.contains("Willow").all())
        self.assertFalse(summary["proof"]["group_size_cap_applied"])

    def test_row_edge_permutations_and_changed_targets_do_not_affect_audit(self):
        expected, proof = self.audit()
        changed = self.frame.sample(frac=1, random_state=8).copy()
        changed["Segment"] = range(len(changed))
        changed["Family"] = "arbitrary target"
        assignment = self.assignment.sample(frac=1, random_state=9).copy()
        assignment["Segment"] = "changed"
        assignment["split"] = "test"
        actual, actual_proof = self.audit(changed, assignment, self.edges.iloc[::-1])
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], actual[name])
            self.assertNotIn("Segment", actual[name])
            self.assertNotIn("Family", actual[name])
            self.assertNotIn("split", actual[name])
        self.assertEqual(proof, actual_proof)

    def test_graph_with_missing_connection_or_duplicate_edge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            self.audit(edges=self.edges.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "one saved row"):
            self.audit(edges=pd.concat([self.edges, self.edges.iloc[[0]]]))

    def test_connected_but_no_common_anchor_partition_is_rejected(self):
        changed = self.frame.copy()
        changed.loc[changed.record_id.eq("r3"), "ProductName"] = "Willow Copper Kitchen Thermometer"
        with self.assertRaisesRegex(ValueError, "common to every member"):
            self.audit(frame=changed)

    def test_negative_claim_overlap_is_review_flag_with_positive_context(self):
        changed = self.frame.copy()
        changed["ProductName"] = "Willow No Magnesium Or Rice Fillers - With Magnesium Citrate Capsules"
        tables, summary = self.audit(frame=changed)
        row = tables["group_audit.csv"].iloc[0]
        self.assertEqual(row.negative_core_overlap_rows, 4)
        self.assertIn("magnesium", row.negative_core_overlap_tokens_json)
        self.assertTrue(summary["proof"]["all_non_singleton_groups_have_common_anchor"])

    def test_all_singleton_partition_and_empty_tables_are_supported(self):
        assignment = self.assignment.copy()
        assignment["group_id"] = assignment.record_id
        tables, summary = self.audit(assignment=assignment, edges=self.edges.iloc[:0])
        self.assertEqual(summary["non_singleton_groups"], 0)
        self.assertEqual(summary["singleton_groups"], 5)
        self.assertTrue(all(table.empty for table in tables.values()))


if __name__ == "__main__":
    unittest.main()
