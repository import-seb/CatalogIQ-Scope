"""Meaningful regression checks for the focused product-evidence guards."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import pandas as pd

from catalogiq.grouping_regression import evaluate_review_groups, load_review_fixture
from catalogiq.split_rule_components import confirmed_rule_components
from catalogiq.split_rule_evidence_v3 import prepare_rule_products_v3
from catalogiq.split_rules_v3 import RuleRefinementConfig, pair_evidence_v3, refine_rule_groups_v3
from catalogiq.split_rules_v4 import (
    RuleFinalConfig, component_compatible_v4, prepare_rule_products_v4, refine_rule_groups_v4,
)
from catalogiq.splitting import SplitConfig


class FinalRuleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.split = SplitConfig(grouping_version=2)
        self.config = RuleFinalConfig()

    def run_groups(self, names, brand="Willow", extra=None, config=None):
        frame = pd.DataFrame({"ProductName": names, "ProductBrand": [brand] * len(names)})
        for key, value in (extra or {}).items():
            frame[key] = value
        ids = [f"record{index:03}" for index in range(len(frame))]
        return refine_rule_groups_v4(frame, ids, self.split, config or self.config)

    def test_supplier_domain_and_strength_cannot_connect_different_ingredients(self):
        groups, _, _ = self.run_groups([
            "Willow.com Lemon Balm Extract Powder Herbal Supplement 1000mg 250 Grams",
            "Willow.com American Ginseng Extract Powder Herbal Supplement 1000mg 100 Grams",
            "Willow.com Ginseng Root Extract Powder Panax Ginseng 1000mg 100 Servings",
            "Willow.com Fish Oil Softgels 1000mg 240 Count",
            "Willow.com Lemon Balm Extract Powder 1000mg 500 Grams",
        ])
        self.assertNotEqual(groups[0], groups[3])
        self.assertNotEqual(groups[1], groups[3])
        self.assertEqual(groups[0], groups[4])

    def test_generic_extract_words_do_not_join_distinct_botanicals(self):
        groups, _, _ = self.run_groups([
            "Willow.com Ashwagandha Root Extract Powder Herbal Supplement 450mg",
            "Willow.com Valerian Root Extract Powder Herbal Supplement 500mg",
            "Willow.com Red Yeast Rice Extract Powder Herbal Supplement 600mg",
            "Willow.com Cayenne Extract Powder Herbal Supplement 500mg",
            "Willow.com Senna Leaf Extract Powder Herbal Supplement 500mg",
        ])
        self.assertEqual(len(set(groups)), 5)

    def test_generic_category_device_families_do_not_merge_through_shared_body(self):
        groups, _, stats = self.run_groups([
            "Willow Motion Sickness Smart Glasses Portable Nausea Cruise Travel",
            "Willow Motion Sickness Smart Portable Nausea Cruise Travel Device",
            "Willow Motion Sickness Smart Wristbands Portable Nausea Cruise Travel",
            "Willow Motion Sickness Smart Glasses Portable Nausea Cruise Travel 2 Pack",
            "Willow Motion Sickness Smart Wristbands Portable Nausea Cruise Travel 4 Pack",
        ])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(groups[0], groups[3])
        self.assertEqual(groups[2], groups[4])
        self.assertGreater(stats.get("component_evidence_guard_rejections", 0), 0)

    def test_component_guard_checks_whole_component_not_only_representatives(self):
        frame = pd.DataFrame({"ProductBrand": ["Willow"] * 3, "ProductName": [
            "Willow Motion Sickness Smart Glasses", "Willow Motion Sickness Smart Device",
            "Willow Motion Sickness Smart Wristbands"]})
        products, _ = prepare_rule_products_v4(frame, self.split, self.config)
        self.assertFalse(component_compatible_v4(products[:2], products[2:], self.config))
        self.assertTrue(component_compatible_v4(products[:1], products[1:2], self.config))

    def test_exact_bundle_payloads_stay_together_despite_multiple_device_tags(self):
        groups, edges, _ = self.run_groups([
            "Willow Motion Sickness Glasses and Acupressure Wristbands Kit",
            "Willow Motion Sickness Glasses and Acupressure Wristbands Kit",
        ], extra={"ProductDescription": ["Specific bundle contents"] * 2})
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(edges[0][2], "exact_classifier_payload")

    def test_named_lines_and_same_ingredient_variants_remain_related(self):
        groups, _, _ = self.run_groups([
            "Willow Northern Edge Vitamin C Tablets 30 Count",
            "Willow Northern Edge Vitamin D3 Gummies 60 Count",
            "Willow.com Lemon Balm Extract Capsules 1000mg 30 Count",
            "Willow.com Lemon Balm Extract Powder 1000mg 250 Grams",
        ])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])

    def test_large_coherent_line_has_no_size_cap(self):
        groups, _, stats = self.run_groups([
            f"Willow Northern Edge Vitamin C Tablets {index + 1} Count" for index in range(105)])
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(stats["largest_group"], 105)
        self.assertFalse(stats["group_size_cap_applied"])

    def test_engine_reproduces_v3_with_v3_evidence_hooks(self):
        frame = pd.DataFrame({"ProductBrand": ["Willow"] * 4, "ProductName": [
            "Willow Northern Edge Tablets 30 Count", "Willow Northern Edge Gummies 60 Count",
            "Willow Anchor Amber", "Willow Anchor Amber Gamma"]})
        ids = ["d", "b", "a", "c"]
        expected = refine_rule_groups_v3(frame, ids, self.split, RuleRefinementConfig())
        actual = confirmed_rule_components(frame, ids, self.split, RuleRefinementConfig(),
            prepare_products=prepare_rule_products_v3, confirm_pair=pair_evidence_v3)
        self.assertEqual(expected[0].tolist(), actual[0].tolist())
        self.assertEqual(expected[1:], actual[1:])

    def test_target_mutation_and_row_permutation_do_not_change_decisions(self):
        frame = pd.DataFrame({"ProductName": ["Willow.com Lemon Balm Powder 1000mg",
            "Willow.com Fish Oil 1000mg", "Willow.com Lemon Balm Capsules 1000mg"],
            "ProductBrand": ["Willow"] * 3, "Segment": ["one", "two", "three"]})
        before = frame.copy(deep=True)
        ids = ["c", "a", "b"]
        groups, edges, _ = refine_rule_groups_v4(frame, ids, self.split)
        pd.testing.assert_frame_equal(frame, before)
        order = [2, 0, 1]
        other = frame.iloc[order].copy()
        other["Segment"] = "mutated target"
        other_ids = [ids[index] for index in order]
        new_groups, new_edges, _ = refine_rule_groups_v4(other, other_ids, self.split)
        self.assertEqual(dict(zip(ids, groups)), dict(zip(other_ids, new_groups)))
        named = lambda rows, values: [(values[left], values[right], reason, score)
                                     for left, right, reason, score in rows]
        self.assertEqual(named(edges, ids), named(new_edges, other_ids))

    def test_75_existing_reviews_remain_regression_checks(self):
        path = Path(__file__).parent / "fixtures" / "grouping_review_75.json"
        frame, pairs, _ = load_review_fixture(path)
        ids = frame.record_id.tolist()
        old_groups, _, _ = refine_rule_groups_v3(frame, ids, self.split)
        new_groups, _, _ = refine_rule_groups_v4(frame, ids, self.split)
        old = evaluate_review_groups(pairs, pd.Series(old_groups, index=ids))
        new = evaluate_review_groups(pairs, pd.Series(new_groups, index=ids))
        clear = pairs.review_category.ne("borderline")
        regressions = clear & old.reviewed_error.eq(False) & new.reviewed_error.eq(True)
        self.assertFalse(regressions.any(), pairs.loc[regressions, "pair_id"].tolist())
        self.assertEqual(len(pairs), 75)
        # This checks the fixture partition only. Corpus-level outputs must also
        # be checked because candidate blocks and transitive paths depend on it.

    def test_config_roundtrip_validation_and_empty_input(self):
        self.assertEqual(RuleFinalConfig.from_dict(json.loads(json.dumps(self.config.to_dict()))), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, comparison_noise=("",))
        with self.assertRaises(ValueError):
            replace(self.config, comparison_noise=("two words",))
        with self.assertRaises(ValueError):
            replace(self.config, device_families=(("eyewear", ("",)),))
        groups, edges, stats = refine_rule_groups_v4(pd.DataFrame(), [], self.split)
        self.assertEqual(len(groups), 0)
        self.assertFalse(edges)
        self.assertEqual(stats["largest_group"], 0)


if __name__ == "__main__":
    unittest.main()
