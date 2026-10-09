"""Generic comparison-view evidence, without private products or pair IDs."""
import unittest

import pandas as pd

from catalogiq.split_matching import (
    brand_block_key, canonical_brand, compatible, family_text, matching_text, prepare_products,
)
from catalogiq.splitting import SplitConfig


class MatchingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = SplitConfig(grouping_version=2)

    def products(self, rows):
        return prepare_products(pd.DataFrame(rows), self.config)

    def test_possessives_vitamin_symbols_and_count_units(self):
        self.assertEqual(matching_text("Women's B-12, 500mcg, 120Count"),
                         matching_text("Womens B12, 500 mcg, 120 count"))

    def test_packaging_is_removed_but_active_strength_retained(self):
        self.assertEqual(family_text("Example Herb 500mg, 120 tablets (Pack of 3)", self.config),
                         family_text("Example Herb 500 mg, 60Tablets", self.config))
        self.assertIn("500 mg", family_text("Example Herb 500mg, 120 tablets", self.config))
        self.assertNotEqual(family_text("Example Herb 500mg", self.config),
                            family_text("Example Herb 100mg", self.config))

    def test_generic_brand_aliases_domain_and_spacing(self):
        self.assertEqual(canonical_brand("Example Health Products", self.config), "example")
        self.assertEqual(canonical_brand("Example Research", self.config), "example")
        self.assertEqual(canonical_brand("Example.COM", self.config), "example")
        self.assertEqual(brand_block_key(canonical_brand("Demo Care", self.config)),
                         brand_block_key(canonical_brand("DemoCare", self.config)))
        self.assertNotEqual(canonical_brand("Example", self.config), canonical_brand("Other", self.config))

    def test_unknown_named_formulas_not_lost_to_marketing(self):
        a, b = self.products([
            {"ProductBrand": "Example", "ProductName": "Example Organic Zorvex 1500mg Vegetarian 90 Tablets"},
            {"ProductBrand": "Example", "ProductName": "Example Organic Quelorin 1500mg Vegetarian 90 Tablets"},
        ])
        self.assertFalse(compatible(a, b, self.config))

    def test_active_section_is_distinct_from_inactive_section(self):
        a, b = self.products([
            {"ProductBrand": "Example", "ProductName": "Example named cream", "ProductContents":
             "Inactive ingredients: lidocaine and glycerin"},
            {"ProductBrand": "Example", "ProductName": "Example named cream", "ProductContents":
             "Active Ingredients: lidocaine 4%. Inactive ingredients: glycerin"},
        ])
        self.assertEqual(a.ingredients, frozenset())
        self.assertEqual(b.ingredients, frozenset({"lidocaine"}))

    def test_salt_compounds_detected_at_any_token_position(self):
        for prefix in ("Demo", "Demo Example"):
            a, b = self.products([
                {"ProductBrand": prefix, "ProductName": prefix + " Ferrous Gluconate Powder"},
                {"ProductBrand": prefix, "ProductName": prefix + " Ferrous Fumarate Powder"},
            ])
            self.assertEqual(a.ingredients, frozenset({"ferrous gluconate"}))
            self.assertFalse(compatible(a, b, self.config))

    def test_parent_brand_title_evidence_and_private_label_conflict(self):
        a, b = self.products([
            {"ProductBrand": "ParentName", "ProductName": "ChildName berry sleep formula 30 Tablets"},
            {"ProductBrand": "ChildName", "ProductName": "ChildName berry sleep formula 60 tablets"},
        ])
        self.assertTrue(compatible(a, b, self.config, allow_brand_override=True))
        a, b = self.products([
            {"ProductBrand": "MakerA", "ProductName": "Generic cooling cushion"},
            {"ProductBrand": "MakerB", "ProductName": "Generic cooling cushion"},
        ])
        self.assertFalse(compatible(a, b, self.config, allow_brand_override=True))


if __name__ == "__main__":
    unittest.main()
