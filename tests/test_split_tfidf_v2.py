"""Novel synthetic fixtures for title-first matching and leakage safeguards."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.cleaning import TARGET_COLUMNS
from catalogiq.splitting import SplitConfig, record_ids
from catalogiq.split_tfidf_v2 import _fit, _threshold_pairs, refine_tfidf_groups


def records(rows):
    frame = pd.DataFrame(rows)
    frame["dataset"] = "novel_tfidf_review_fixture.csv"
    frame["source_sha256"] = "novel-tfidf-fixture"
    frame["source_row"] = range(len(frame))
    frame["Segment"] = "Synthetic"
    return frame


class RefinedTfidfTests(unittest.TestCase):
    def setUp(self):
        self.config = SplitConfig(grouping_version=2, cosine_chunk_size=2)

    def test_distinguishing_formula_words_survive_the_support_vocabulary_cap(self):
        frame = records([
            {"ProductBrand": "Cedar", "ProductName": "Cedar Quorvexa Botanical 60 capsules",
             "ProductDescription": "Premium daily natural wellbeing health support"},
            {"ProductBrand": "Cedar", "ProductName": "Cedar Plenovira Botanical 60 capsules",
             "ProductDescription": "Premium daily natural wellbeing health support"},
        ])
        config = replace(self.config, tfidf_max_features=1)
        groups, edges, stats = refine_tfidf_groups(frame, record_ids(frame), config)
        self.assertNotEqual(groups[0], groups[1])
        self.assertEqual(edges, [])
        self.assertTrue(stats["name_features_uncapped"])
        self.assertGreater(stats["vocabulary_sizes"]["ProductName"], 1)
        matrix, size = _fit(["cedar quorvexa botanical", "cedar plenovira botanical"])
        self.assertGreater(size, 1)
        self.assertLess(float(matrix[0].multiply(matrix[1]).sum()), 1)

    def test_strong_title_match_is_not_diluted_by_disjoint_descriptions(self):
        frame = records([
            {"ProductBrand": "Cedar", "ProductName": "Cedar Luna Mineral Complex 60 Capsules",
             "ProductDescription": "Retailer delivery policies customer store benefits"},
            {"ProductBrand": "Cedar", "ProductName": "CEDAR Luna Mineral Complex, 60 capsules",
             "ProductDescription": "Travel gym routine portable lightweight bottle"},
        ])
        groups, edges, _ = refine_tfidf_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(edges), 1)

    def test_package_counts_and_generic_brand_suffix_variations_are_related(self):
        frame = records([
            {"ProductBrand": "Cedar Health Products",
             "ProductName": "Cedar Luna Mineral Complex 60 capsules (Pack of 3)"},
            {"ProductBrand": "Cedar Health",
             "ProductName": "Cedar Luna Mineral Complex 120 capsules (Pack of 4)"},
        ])
        groups, edges, _ = refine_tfidf_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][2], "tfidf_family_cosine")

    def test_vitamin_alphanumeric_names_and_unit_spacing_normalize_consistently(self):
        frame = records([
            {"ProductBrand": "Northstar", "ProductName": "Northstar Vitamin B-12 1000 mcg, 60 tablets"},
            {"ProductBrand": "Northstar", "ProductName": "Northstar Vitamin B12 1000mcg 120 tablets"},
        ])
        groups, _, _ = refine_tfidf_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])

    def test_different_active_ingredients_cannot_be_overridden_by_marketing_text(self):
        shared = "Easy convenient relief product pain wellbeing daily care " * 10
        frame = records([
            {"ProductBrand": "Cedar", "ProductName": "Cedar relief acetaminophen 500 mg 60 tablets",
             "ProductDescription": shared, "ProductContents": "acetaminophen 500mg"},
            {"ProductBrand": "Cedar", "ProductName": "Cedar relief ibuprofen 200 mg 60 tablets",
             "ProductDescription": shared, "ProductContents": "ibuprofen 200mg"},
        ])
        config = replace(self.config, tfidf_name_threshold=.5, tfidf_family_threshold=.5,
                         tfidf_supported_name_threshold=.4)
        groups, edges, _ = refine_tfidf_groups(frame, record_ids(frame), config)
        self.assertNotEqual(groups[0], groups[1])
        self.assertEqual(edges, [])

    def test_identifiers_never_create_text_group_edges(self):
        frame = records([
            {"ProductBrand": "Cedar", "ProductName": "Digital camera lens filter",
             "Upc": "036000291452", "ProductUrl": "https://example.org/collision"},
            {"ProductBrand": "Cedar", "ProductName": "Cotton fleece blanket",
             "Upc": "036000291452", "ProductUrl": "https://example.org/collision"},
        ])
        groups, edges, _ = refine_tfidf_groups(frame, record_ids(frame), self.config)
        self.assertNotEqual(groups[0], groups[1])
        self.assertEqual(edges, [])

    def test_labels_row_order_and_chunk_size_do_not_change_groups(self):
        frame = records([
            {"ProductBrand": "Cedar", "ProductName": "Cedar Luna Mineral Complex 60 capsules"},
            {"ProductBrand": "Cedar", "ProductName": "Cedar Luna Mineral Complex 120 capsules"},
            {"ProductBrand": "Maple", "ProductName": "Maple Cotton Fleece Blanket"},
            {"ProductBrand": "Maple", "ProductName": "Maple Cotton Fleece Blanket"},
            {"ProductBrand": "Juniper", "ProductName": "Juniper Digital Lens Filter"},
        ])
        ids = record_ids(frame)
        groups, _, _ = refine_tfidf_groups(frame, ids, self.config)
        repeated, _, _ = refine_tfidf_groups(frame, ids, self.config)
        np.testing.assert_array_equal(groups, repeated)
        for field in (*TARGET_COLUMNS, "Category"):
            frame[field] = ["different", "labels", "entirely", "new", "classes"]
        changed, _, _ = refine_tfidf_groups(frame, ids, self.config)
        np.testing.assert_array_equal(groups, changed)
        shuffled = frame.iloc[[4, 2, 0, 3, 1]]
        shuffled_ids = record_ids(shuffled)
        reordered, _, _ = refine_tfidf_groups(
            shuffled, shuffled_ids, replace(self.config, cosine_chunk_size=1),
        )
        self.assertEqual(dict(zip(ids, groups)), dict(zip(shuffled_ids, reordered)))

    def test_threshold_graph_is_exhaustive_including_low_ranked_neighbors(self):
        values = ["copper insulated hiking bottle"] * 8 + ["ceramic garden pot"]
        matrix, _ = _fit(values)
        pairs = list(_threshold_pairs(matrix, .9, 2, None, "fixture"))
        self.assertEqual({(a, b) for a, b, _ in pairs},
                         {(a, b) for a in range(8) for b in range(a + 1, 8)})
        self.assertEqual(len(pairs), 28)

    def test_unknown_formulation_bridge_cannot_join_conflicting_components(self):
        frame = records([{"ProductName": "Cedar relief tablets"}] * 3)
        ids = record_ids(frame)
        products = [SimpleNamespace(name="cedar relief tablets", family="cedar relief tablets",
                                   ingredients=ingredients)
                    for ingredients in (frozenset({"active-a"}), frozenset(),
                                        frozenset({"active-b"}))]
        with patch("catalogiq.split_tfidf_v2.prepare_products", return_value=products), \
             patch("catalogiq.split_tfidf_v2.compatible", return_value=True):
            groups, _, stats = refine_tfidf_groups(frame, ids, self.config)
        self.assertNotEqual(groups[0], groups[2])
        self.assertGreater(stats["component_formulation_rejections"], 0)
        self.assertEqual(len(set(groups)), 2)

    def test_empty_names_stay_singleton_even_with_identical_descriptions(self):
        frame = records([
            {"ProductName": "", "ProductDescription": "Same shared retailer description"},
            {"ProductName": "", "ProductDescription": "Same shared retailer description"},
        ])
        groups, edges, stats = refine_tfidf_groups(frame, record_ids(frame), self.config)
        self.assertNotEqual(groups[0], groups[1])
        self.assertEqual(edges, [])
        self.assertEqual(stats["zero_vector_rows"], 2)


if __name__ == "__main__":
    unittest.main()
