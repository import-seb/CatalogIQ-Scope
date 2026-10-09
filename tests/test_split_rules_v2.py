"""Public fixtures for version-two rule resolution, independent of review IDs."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from catalogiq.cleaning import TARGET_COLUMNS
from catalogiq.split_rules_v2 import refine_rule_groups
from catalogiq.splitting import SplitConfig, check_assignments, record_ids, stratified_group_split


def records(rows):
    frame = pd.DataFrame(rows)
    frame["dataset"] = "novel_rule_fixture.csv"
    frame["source_sha256"] = "public-rule-fixture"
    frame["source_row"] = range(len(frame))
    if "Segment" not in frame:
        frame["Segment"] = "General"
    return frame


class RefinedRuleTests(unittest.TestCase):
    def setUp(self):
        self.config = SplitConfig(grouping_version=2)

    def group(self, rows, config=None):
        frame = records(rows)
        return refine_rule_groups(frame, record_ids(frame), config or self.config)

    def test_generic_brand_suffix_and_pack_variants(self):
        groups, edges, _ = self.group([
            {"ProductBrand": "Cedar Health Products",
             "ProductName": "Cedar Valerian Root Sleep Support 60 Capsules (Pack of 2)"},
            {"ProductBrand": "CEDAR HEALTH",
             "ProductName": "Cedar Valerian Root Sleep Support 120 Capsules (Pack of 4)"},
        ])
        self.assertEqual(groups[0], groups[1])
        self.assertTrue(edges)

    def test_apostrophe_and_vitamin_spelling_normalization(self):
        groups, _, _ = self.group([
            {"ProductBrand": "Pine's", "ProductName": "Pine's Vitamin B-12 500 mcg 60 tablets"},
            {"ProductBrand": "Pines", "ProductName": "Pines Vitamin B12 500mcg 120 tablets"},
        ])
        self.assertEqual(groups[0], groups[1])

    def test_units_and_quantity_are_family_variants(self):
        groups, _, _ = self.group([
            {"ProductBrand": "Mistral", "ProductName": "Mistral Aloe Moisturizing Lotion 100 mL"},
            {"ProductBrand": "Mistral", "ProductName": "Mistral Aloe Moisturizing Lotion 8 fl. oz."},
        ])
        self.assertEqual(groups[0], groups[1])

    def test_distinct_active_ingredients_cannot_be_overwhelmed_by_marketing(self):
        scaffold = "Harbor Daily Pain Fever Relief Medicine Proven Active Ingredients "
        groups, _, _ = self.group([
            {"ProductBrand": "Harbor", "ProductName": scaffold + "Acetaminophen 500mg"},
            {"ProductBrand": "Harbor", "ProductName": scaffold + "Ibuprofen 200mg"},
        ])
        self.assertNotEqual(groups[0], groups[1])

    def test_distinct_named_formulas_do_not_merge(self):
        groups, _, _ = self.group([
            {"ProductBrand": "Juniper", "ProductName": "Juniper Luminex 90 Tablets"},
            {"ProductBrand": "Juniper", "ProductName": "Juniper Solavex 90 Tablets"},
        ], replace(self.config, rule_family_threshold=0.2, rule_family_edit_threshold=0.2))
        self.assertNotEqual(groups[0], groups[1])

    def test_cross_brand_bad_gtin_and_copied_private_label_title_are_rejected(self):
        groups, _, stats = self.group([
            {"ProductBrand": "Northstar", "Upc": "036000291452",
             "ProductName": "Northstar Iron Supplement 27 mg 110 Tablets"},
            {"ProductBrand": "Willow", "Upc": "036000291452",
             "ProductName": "Willow Iron Supplement 28mg 90 Tablets"},
            {"ProductBrand": "One Label", "ProductName": "Reusable Cooling Gel Pad Recovery Kit"},
            {"ProductBrand": "Another Label", "ProductName": "Reusable Cooling Gel Pad Recovery Kit"},
        ])
        self.assertEqual(len(set(groups)), 4)
        self.assertGreater(stats["pair_compatibility_rejections"], 0)

    def test_same_brand_gtin_requires_corroboration(self):
        groups, edges, _ = self.group([
            {"ProductBrand": "Orchard", "Upc": "036000291452",
             "ProductName": "Orchard Marine Collagen Capsules"},
            {"ProductBrand": "Orchard", "Upc": "036000291452",
             "ProductName": "Orchard Marine Collagen Dietary Supplement Capsules"},
            {"ProductBrand": "Orchard", "Upc": "036000291452",
             "ProductName": "Orchard Digital Kitchen Thermometer"},
        ])
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertIn("corroborated_gtin", {reason for _, _, reason, _ in edges})

    def test_empty_evidence_cannot_bridge_contradictory_ingredient_components(self):
        frame = records([
            {"ProductBrand": "Nimbus", "Upc": "036000291452",
             "ProductName": "Nimbus Comfort Pain Fever Relief Acetaminophen 60 Tablets"},
            {"ProductBrand": "Nimbus", "Upc": "036000291452",
             "ProductName": "Nimbus Comfort Pain Fever Relief 60 Tablets"},
            {"ProductBrand": "Nimbus", "Upc": "036000291452",
             "ProductName": "Nimbus Comfort Pain Fever Relief Ibuprofen 60 Tablets"},
        ])
        baseline = None
        for input_frame in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=21)):
            ids = record_ids(input_frame)
            groups, _, _ = refine_rule_groups(input_frame, ids, self.config)
            mapping = dict(zip(ids, groups))
            if baseline is None:
                baseline = mapping
            self.assertEqual(mapping, baseline)
            original_ids = record_ids(frame)
            self.assertNotEqual(mapping[original_ids[0]], mapping[original_ids[2]])

    def test_rule_thresholds_are_configurable(self):
        rows = [
            {"ProductBrand": "Mistral", "ProductName": "Mistral Aloe Moisturizing Lotion"},
            {"ProductBrand": "Mistral", "ProductName": "Mistral Aloe Moisturizing Lotion Refill"},
        ]
        permissive, _, _ = self.group(rows, replace(
            self.config, rule_family_threshold=0.7, rule_family_edit_threshold=0.7))
        strict, _, _ = self.group(rows, replace(
            self.config, rule_family_threshold=1.0, rule_family_edit_threshold=1.0))
        self.assertEqual(permissive[0], permissive[1])
        self.assertNotEqual(strict[0], strict[1])

    def test_repeat_permutation_label_blindness_source_preservation_and_isolation(self):
        frame = records([
            {"ProductBrand": f"Label {i // 3}",
             "ProductName": f"Label {i // 3} Aloe Moisturizing Lotion {30 + i} ml",
             "Segment": "Skin" if i % 2 else "Body"}
            for i in range(30)
        ])
        untouched = frame.copy(deep=True)
        baseline = None
        for input_frame in (frame, frame.copy(), frame.sample(frac=1, random_state=19)):
            ids = record_ids(input_frame)
            groups, _, _ = refine_rule_groups(input_frame, ids, self.config)
            splits, _ = stratified_group_split(groups, input_frame["Segment"], self.config)
            check_assignments(pd.DataFrame({
                "record_id": ids, "group_id": groups, "split": splits}), ids)
            actual = (dict(zip(ids, groups)), dict(zip(ids, splits)))
            if baseline is None:
                baseline = actual
            self.assertEqual(actual, baseline)
        ids = record_ids(frame)
        original, _, _ = refine_rule_groups(frame, ids, self.config)
        changed = frame.copy()
        for field in [*TARGET_COLUMNS, "Category"]:
            changed[field] = [f"changed-{i}" for i in range(len(changed))]
        altered, _, _ = refine_rule_groups(changed, ids, self.config)
        np.testing.assert_array_equal(original, altered)
        pd.testing.assert_frame_equal(frame, untouched)

    def test_large_exact_bucket_uses_guarded_representatives_without_fragmenting(self):
        groups, _, stats = self.group([
            {"ProductBrand": "Cobalt", "ProductName": "Cobalt Magnesium Supplement 60 Tablets"}
            for _ in range(8)
        ], replace(self.config, max_block_size=3))
        self.assertEqual(len(set(groups)), 1)
        self.assertGreater(stats["oversized_key_buckets_representative_pass"], 0)

    def test_branded_generic_exact_duplicates_keep_formulation_guards(self):
        title = "Original natural health vitamin supplement"
        rows = [
            {"ProductName": title, "ProductBrand": "Acme", "ProductDescription": "Daily use"},
            {"ProductName": title, "ProductBrand": "Acme", "ProductDescription": "Daily use"},
            {"ProductName": title, "ProductBrand": "Different Maker", "ProductDescription": "Daily use"},
            {"ProductName": title, "ProductBrand": "Formula Maker", "ProductContents": "Active ingredient: acetaminophen"},
            {"ProductName": title, "ProductBrand": "Formula Maker", "ProductContents": "Active ingredient: ibuprofen"},
        ]
        groups, _, _ = self.group(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertNotEqual(groups[3], groups[4])

    def test_empty_fields_remain_singletons(self):
        groups, edges, _ = self.group([{}, {}, {}])
        self.assertEqual(len(set(groups)), 3)
        self.assertEqual(edges, [])


if __name__ == "__main__":
    unittest.main()
