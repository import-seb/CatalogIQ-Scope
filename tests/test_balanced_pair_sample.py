"""Independent lexical retrieval, blind evidence, and balanced gold selection."""
from copy import deepcopy
import unittest

import pandas as pd

from catalogiq.balanced_pair_sample import (
    BUCKETS, CATEGORIES, CandidateConfig, EVIDENCE_FIELDS, assemble_evidence,
    blind_evidence, candidate_pool, family_keys, lexical_similarity, normalize_raw,
    select_balanced_review,
)


def products():
    records = []
    for index in range(8):
        brand = f"Maker{index}"
        names = [f"{brand} Premium Alpha Turmeric Joint Herbal Supplement 30 Capsules",
                 f"{brand} Premium Alpha Turmeric Joint Herbal Supplement 60 Capsules",
                 f"{brand} Gold Beta Magnesium Sleep Herbal Supplement 30 Capsules",
                 f"{brand} Gold Gamma Cranberry Urinary Herbal Supplement 60 Capsules",
                 f"{brand} Ancient Omega Fish Oil Pure Lemon Flavor 100 Softgels",
                 f"{brand} Ancient Omega Fish Oil Pure Orange Flavor 200 Softgels"]
        for i, name in enumerate(names):
            records.append({"record_id": f"m{index}r{i}", "ProductName": name,
                            "ProductBrand": brand, "ProductDescription": f"Full description {index}-{i}",
                            "ProductContents": "Full ingredients", "Sku": str(i), "Upc": "01234",
                            "ProductModelNumber": "M12", "Retailer": "Store", "ProductUrl": "https://example.test/item",
                            "Segment": "target-do-not-use", "rule_group_id": "group-do-not-use"})
    return pd.DataFrame(records)


def registry_and_reviews():
    records, pair_rows, annotations = [], [], []
    for category_i, category in enumerate(CATEGORIES):
        for i in range(4):
            pair_id = f"P{category_i}_{i}"
            left, right = f"{pair_id}a", f"{pair_id}b"
            for endpoint in (left, right):
                records.append({"record_id": endpoint, "ProductBrand": f"Brand{pair_id}",
                                "ProductName": f"Item{pair_id} name", "ProductDescription": "complete",
                                "ProductContents": "complete ingredients", "Segment": "excluded"})
            pair_rows.append({"pair_id": pair_id, "left_record_id": left, "right_record_id": right,
                              "left_family_key": pair_id, "right_family_key": pair_id,
                              "provisional_stratum": BUCKETS[(category_i + 1) % 3],
                              "name_char3_jaccard": .8, "refined_rule_false_positive": True})
            annotations.append({"pair_id": pair_id, "left_record_id": left, "right_record_id": right,
                                "product_identity": "different_product", "identity_detail": "distinct_product",
                                "identity_confidence": "high", "identity_rationale": "Independent product evidence.",
                                "leakage_decision": {"related": "keep_together", "safe_to_separate": "keep_separate",
                                                     "borderline": "uncertain"}[category],
                                "leakage_confidence": "medium" if category == "borderline" else "high",
                                "leakage_rationale": "Independent information overlap evidence.",
                                "review_category": category, "judgment_uses_Segment": "False",
                                "judgment_uses_method_outcomes": "False"})
    return assemble_evidence(pd.DataFrame(records), pd.DataFrame(pair_rows)), pd.DataFrame(annotations)


class BalancedPairSampleTests(unittest.TestCase):
    def test_simple_raw_normalization_and_similarity(self):
        self.assertEqual(normalize_raw(" MÁKER'S  Vitamin-C "), "máker s vitamin c")
        self.assertEqual(normalize_raw("null"), "")
        self.assertEqual(lexical_similarity("Maker Pure C", "maker pure c"), (1.0, 1.0))
        self.assertEqual(family_keys("Maker Pure C 30 Capsules", "Maker"),
                         family_keys("Maker Pure C 60 Capsules", "Maker"))

    def test_candidates_ignore_labels_saved_outcomes_and_row_order(self):
        frame = products()
        prior = pd.DataFrame(columns=["left_record_id", "right_record_id", "Segment", "rule_group_id"])
        config = CandidateConfig(per_bucket=3)
        first, first_report = candidate_pool(frame, prior, config)
        changed = frame.copy()
        changed["Segment"] = "completely different"
        changed["rule_group_id"] = range(len(changed))
        changed["tfidf_outcome"] = "wrong"
        second, second_report = candidate_pool(changed.iloc[::-1], prior, config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_report, second_report)
        self.assertFalse(first_report["candidate_targets_used"])
        self.assertFalse(first_report["candidate_method_outcomes_used"])
        self.assertNotIn("Segment", " ".join(first.columns))

    def test_prior_endpoints_and_their_simple_families_are_excluded(self):
        frame = products()
        prior = pd.DataFrame([{"left_record_id": "m0r0", "right_record_id": "m0r2"}])
        pool, report = candidate_pool(frame, prior, CandidateConfig(per_bucket=5))
        endpoints = set(pool.left_record_id) | set(pool.right_record_id)
        self.assertFalse({"m0r0", "m0r1", "m0r2"} & endpoints)
        self.assertEqual(report["prior_endpoints_excluded"], 2)
        self.assertGreaterEqual(report["prior_family_rows_excluded"], 1)
        self.assertEqual(len(endpoints), len(pool) * 2)
        used = set()
        for left, right in pool[["left_family_key", "right_family_key"]].itertuples(index=False, name=None):
            self.assertFalse(used & {left, right})
            used.update((left, right))

    def test_blind_evidence_contains_full_fields_and_no_seeds_scores_targets_or_outcomes(self):
        pool, _ = registry_and_reviews()
        blind = blind_evidence(pool)
        expected = {"pair_id", "left_record_id", "right_record_id",
                    *(f"{side}_{field}" for side in ("left", "right") for field in EVIDENCE_FIELDS)}
        self.assertEqual(set(blind[0]), expected)
        self.assertEqual(blind[0]["left_ProductDescription"], "complete")
        self.assertEqual(blind[0]["left_ProductContents"], "complete ingredients")
        self.assertFalse(any("Segment" in column or "false_positive" in column for column in blind[0]))

    def test_final_selection_is_balanced_deterministic_and_independent_of_seed_strata_or_correctness(self):
        pool, review = registry_and_reviews()
        first = select_balanced_review(pool, review, per_category=2)
        changed = pool.iloc[::-1].copy()
        changed["provisional_stratum"] = "different seed"
        changed["name_char3_jaccard"] = 0
        changed["refined_rule_false_positive"] = False
        second = select_balanced_review(changed, review.iloc[::-1], per_category=2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.review_category.value_counts().to_dict(), dict.fromkeys(CATEGORIES, 2))
        self.assertNotIn("provisional_stratum", first)
        self.assertNotIn("refined_rule_false_positive", first)
        self.assertIn("leakage_rationale", first)

    def test_review_is_explicit_and_category_quota_is_never_filled_by_inference(self):
        pool, review = registry_and_reviews()
        for field, value in [("review_category", ""), ("judgment_uses_Segment", "True"),
                             ("judgment_uses_method_outcomes", "True"), ("leakage_rationale", ""),
                             ("identity_detail", "")]:
            changed = review.copy()
            changed.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_balanced_review(pool, changed, per_category=2)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            select_balanced_review(pool, review.loc[review.review_category.ne("borderline")], per_category=2)
        with self.assertRaisesRegex(ValueError, "missing independent"):
            select_balanced_review(pool, review.drop(columns="judgment_uses_method_outcomes"), per_category=2)

    def test_borderline_can_have_independently_decided_direction_without_high_confidence(self):
        pool, review = registry_and_reviews()
        review.loc[review.review_category.eq("borderline"), "leakage_decision"] = "keep_separate"
        chosen = select_balanced_review(pool, review, per_category=2)
        self.assertTrue(chosen.loc[chosen.review_category.eq("borderline"), "leakage_decision"].eq("keep_separate").all())
        review.loc[review.review_category.eq("borderline"), "leakage_confidence"] = "high"
        with self.assertRaisesRegex(ValueError, "borderline"):
            select_balanced_review(pool, review, per_category=2)

    def test_duplicate_pairs_endpoint_registry_mismatches_and_duplicate_record_ids_rejected(self):
        pool, review = registry_and_reviews()
        with self.assertRaisesRegex(ValueError, "unique"):
            select_balanced_review(pool, pd.concat([review, review.iloc[[0]]]), per_category=2)
        changed = review.copy()
        changed.loc[0, "right_record_id"] = "not_registered"
        with self.assertRaisesRegex(ValueError, "endpoints differ"):
            select_balanced_review(pool, changed, per_category=2)
        changed = products()
        changed.loc[1, "record_id"] = changed.loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            candidate_pool(changed, pd.DataFrame(columns=["left_record_id", "right_record_id"]))

    def test_swapped_review_orientation_preserves_registered_evidence(self):
        pool, review = registry_and_reviews()
        first = select_balanced_review(pool, review, per_category=2)
        swapped = review.copy()
        swapped[["left_record_id", "right_record_id"]] = review[["right_record_id", "left_record_id"]].to_numpy()
        second = select_balanced_review(pool, swapped, per_category=2)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
