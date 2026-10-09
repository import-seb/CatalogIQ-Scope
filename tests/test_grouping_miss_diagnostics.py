import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq import split_rules_v2
from catalogiq.grouping_miss_diagnostics import (
    compatibility_evidence, cause_from_events, identifier_evidence,
    observe_replay, text_overlap, explain_case,
)
from catalogiq.split_matching import prepare_products
from catalogiq.splitting import SplitConfig


class MissDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = SplitConfig(grouping_version=2)

    def products(self, left, right, **columns):
        frame = pd.DataFrame({"ProductName": [left, right], "ProductBrand": columns.pop("brands", ["ACME", "ACME"]),
                              **columns})
        return frame, prepare_products(frame, self.config)

    def test_identifier_integrity_and_scopes(self):
        _, (a,b) = self.products("ACME Mineral Capsules", "ACME Mineral Capsules")
        left = {"Retailer": "Amazon", "Sku": "XYZ", "ProductModelNumber": "X123", "Upc": "3.0573E+11"}
        right = {**left, "Retailer": "Walgreens"}
        result = identifier_evidence(left, right, a, b)
        self.assertTrue(result["same_raw_upc"])
        self.assertFalse(result["shared_valid_gtin"])
        self.assertFalse(result["shared_scoped_sku"])
        self.assertTrue(result["shared_scoped_model"])
        right["ProductModelNumber"] = "1.23e+11"
        self.assertFalse(identifier_evidence(left, right, a, b)["shared_scoped_model"])

    def test_guards_separate_brand_evidence_from_overlap(self):
        _, (a,b) = self.products("Calming Melatonin 5 mg Capsules", "Calming Melatonin 5 mg Capsules", brands=["ACME", "OTHER"])
        result = compatibility_evidence(a,b,self.config,True)
        self.assertEqual(result["core_overlap"], 1)
        self.assertFalse(result["brand_override_passes"])
        self.assertEqual(result["failed_guards"], ["brand_guard"])

    def test_actual_replay_matches_original_and_restores_observers(self):
        frame, _ = self.products("ACME Calming Melatonin 5 mg 60 Capsules", "ACME Calming Melatonin 5 mg 120 Capsules")
        ids = np.array(["a", "b"])
        original = (split_rules_v2.prepare_products, split_rules_v2.name_similarity, split_rules_v2.compatible)
        expected = split_rules_v2.refine_rule_groups(frame, ids, self.config)
        groups, edges, stats, events, _, _ = observe_replay(frame, ids, self.config, {"p": (0,1)})
        self.assertTrue(np.array_equal(groups, expected[0]))
        self.assertEqual(edges, expected[1])
        self.assertEqual(stats, expected[2])
        self.assertTrue(any(e["event"] == "link_result" and e["accepted"] for e in events["p"]))
        self.assertEqual(original, (split_rules_v2.prepare_products, split_rules_v2.name_similarity, split_rules_v2.compatible))
        self.assertIsNone(sys.gettrace())

    def test_replay_component_guard_distinct_from_pair_guard(self):
        # A missing-signature record can pass each pair but cannot bridge the
        # two contradictory signatures already collected by its component.
        frame = pd.DataFrame({"ProductName": ["ACME Calm acetaminophen medicine", "ACME Calm medicine", "ACME Calm ibuprofen medicine"],
                              "ProductBrand": ["ACME"]*3, "Sku": ["SAME"]*3, "Retailer": ["Shop"]*3})
        config = SplitConfig(grouping_version=2, identifier_name_threshold=.4, core_min_overlap=.3)
        _, _, _, events, _, _ = observe_replay(frame, np.array(["a","b","c"]), config, {"bc": (1,2)})
        self.assertTrue(any(e.get("component_guard_rejected") for e in events["bc"]))
        self.assertEqual(cause_from_events(events["bc"], {"shared_blocks":[]}), "component_ingredient_guard")

    def test_exception_restores_globals_and_trace(self):
        frame, _ = self.products("ACME Calming Melatonin", "ACME Calming Melatonin")
        original = split_rules_v2.name_similarity
        with patch.object(split_rules_v2, "name_similarity", side_effect=RuntimeError("test")) as broken:
            with self.assertRaises(RuntimeError):
                observe_replay(frame, np.array(["a","b"]), self.config, {"p":(0,1)})
            self.assertIs(split_rules_v2.name_similarity, broken)
            self.assertIsNone(sys.gettrace())
        self.assertIs(split_rules_v2.name_similarity, original)

    def test_cause_requires_executed_gate(self):
        blocks = {"shared_blocks": []}
        self.assertEqual(cause_from_events([], blocks), "candidate_blocking:no_shared_selected_key")
        events = [{"event":"candidate_comparison", "family_name_gate_passes":False, "family_edit_gate_passes":True}]
        self.assertEqual(cause_from_events(events, blocks), "blocked_family_jaccard_gate")
        events = [{"event":"candidate_comparison", "family_name_gate_passes":True, "family_edit_gate_passes":False}]
        self.assertEqual(cause_from_events(events, blocks), "blocked_family_edit_gate")

    def test_text_missing_not_exact_copy_and_boilerplate_caution(self):
        absent = text_overlap("null", "null")
        self.assertFalse(absent["normalized_exact_copy"])
        copy = text_overlap("Premium tested natural product for your daily health", "Premium tested natural product for your daily health")
        self.assertTrue(copy["normalized_exact_copy"])
        self.assertIn("boilerplate", copy["caution"])

    def test_exact_family_empty_core_explanation_does_not_claim_guard_rejection(self):
        _, (a,b) = self.products("ACME Maximum Strength Softgels, 36 Each", "ACME Maximum Strength Softgels, 36 ea (Pack of 2)")
        self.assertEqual(a.family,b.family)
        self.assertFalse(a.core)
        identifiers = {"same_raw_upc":False}
        missing = text_overlap("null","null")
        result = explain_case(a,b,[],{"shared_blocks":[], "common_core_tokens":[]},identifiers,missing,missing,self.config)
        self.assertIn("exact-family pass requires",result["executed_branch_explanation"])
        self.assertFalse(result["compatibility_guard_reached"])


if __name__ == "__main__":
    unittest.main()
