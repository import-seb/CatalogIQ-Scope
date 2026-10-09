import unittest

import pandas as pd

from catalogiq.segment_development_profile import profile_development


def fixture():
    ids = list("abcdefgh")
    base = pd.DataFrame({"record_id": ids, "group_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
                         "Segment": ["A", "A", "A", "B", "A", "B", "B", "B"]})
    grouped = base.assign(split=["train"] * 4 + ["validation"] * 4, strategy="group_aware")
    random = base.assign(split=["train", "validation"] * 4, strategy="random")
    assignments = pd.concat([random, grouped], ignore_index=True)
    exclusions = pd.DataFrame({"record_id": ["x", "u"],
                               "reason": ["protected_final_test", "missing_Segment"]})
    frame = pd.DataFrame({"record_id": ids + ["x", "u"], "Segment": base.Segment.to_list() + ["B", "null"],
        "Retailer": ["Amazon", " AMAZON ", "Shop", "Shop", "Amazon", "Amazon", "Amazon", "Amazon", "Shop", "Other"],
        "ProductBrand": ["", "null", "Alpha", "Beta", "Alpha", "Alpha", "Alpha", "Alpha", "Test", "Missing"],
        "ProductDescription": [None, "text", "text", "text", "text", "text", "text", "text", "Test", "Missing"],
        "ProductContents": ["contents"] * 8 + ["Test", "Missing"]})
    historical = assignments.loc[assignments.record_id.isin(list("abef"))].copy()
    # Identical historical roles satisfy matched per-class control counts.
    historical.loc[historical.record_id.isin(list("ab")), "split"] = "train"
    historical.loc[historical.record_id.isin(list("ef")), "split"] = "validation"
    frozen = pd.concat([grouped.drop(columns="strategy"), pd.DataFrame({"record_id": ["x", "u"],
        "group_id": ["gx", "gu"], "Segment": ["B", "null"], "split": ["test", "train"]})], ignore_index=True)
    return frame, assignments, exclusions, historical, frozen


class UntouchableTarget:
    def __str__(self):
        raise AssertionError("production or frozen target was inspected")


class SegmentDevelopmentProfileTests(unittest.TestCase):
    def test_complete_pool_coverage_and_no_label_invention(self):
        frame, assignments, exclusions, _, _ = fixture()
        result = profile_development(frame, assignments, exclusions, expected_development_ids=list("abcdefgh"))
        summary = result["summary"]
        self.assertEqual(summary["source_records"], 10)
        self.assertEqual(summary["labeled_source_records"], 9)
        self.assertEqual(summary["development_records"], 8)
        self.assertTrue(summary["expected_frozen_v3_non_test_universe_verified"])
        self.assertFalse(summary["final_test_predictions_or_metrics_consumed"])
        self.assertEqual(set(result["class_distribution"].Segment), {"A", "B"})
        self.assertNotIn("full_labeled_source", set(result["class_distribution"].population))
        reason = {row["reason"]: row for row in summary["excluded_by_reason"]}
        self.assertEqual(reason["missing_Segment"]["missing_Segment_records"], 1)
        self.assertEqual(reason["protected_final_test"]["labeled_records"], 1)

    def test_reduced_pool_retailer_bias_is_quantified_not_assumed(self):
        frame, assignments, exclusions, historical, _ = fixture()
        result = profile_development(frame, assignments, exclusions, previous_assignments=historical)
        counts = result["retailer_distribution"]
        source = counts.loc[counts.population.eq("full_labeled_source") & counts.retailer.eq("amazon")].iloc[0]
        previous = counts.loc[counts.population.eq("previous_development_pool") & counts.retailer.eq("amazon")].iloc[0]
        self.assertAlmostEqual(source.fraction, 2 / 3)
        self.assertEqual(previous.fraction, 1)
        tvd = next(row for row in result["summary"]["retailer_total_variation"]
                   if row["reference"] == "full_labeled_source" and row["comparison"] == "previous_development_pool")
        self.assertAlmostEqual(tvd["total_variation_distance"], 1 / 3)
        self.assertEqual(tvd["largest_shift_retailer"], "amazon")
        self.assertAlmostEqual(tvd["comparison_minus_reference_fraction"], 1 / 3)

    def test_same_retailer_mix_does_not_emit_retailer_bias_observation(self):
        frame, assignments, exclusions, _, _ = fixture()
        frame.Retailer = "Amazon"
        result = profile_development(frame, assignments, exclusions)
        self.assertFalse(any(row["kind"] == "observed_retailer_mix_difference"
                             for row in result["summary"]["observed_composition_differences"]))

    def test_group_sizes_distinguish_partial_groups_from_development_singletons(self):
        frame, assignments, exclusions, _, _ = fixture()
        result = profile_development(frame, assignments, exclusions)
        summary = {row["population"]: row for row in result["summary"]["partition_group_summary"]}
        self.assertEqual(summary["random_train"]["unique_frozen_groups"], 4)
        self.assertEqual(summary["group_aware_train"]["unique_frozen_groups"], 2)
        self.assertEqual(summary["random_train"]["within_partition_singleton_groups"], 4)
        self.assertEqual(summary["random_train"]["development_singleton_group_records"], 0)
        self.assertEqual(summary["development_pool"]["max_within_partition_group_size"], 2)
        self.assertEqual(summary["development_pool"]["size_bucket_records"],
                         {"1": 0, "2": 8, "3-5": 0, "6-10": 0, "11+": 0})
        self.assertEqual(summary["random_train"]["size_bucket_records"]["1"], 4)
        self.assertEqual(summary["random_train"]["development_size_bucket_records"]["2"], 4)
        self.assertTrue(all(row["whole_development_size_bucket_record_tvd"] == 0
                            for row in result["summary"]["group_size_total_variation"]))
        sizes = result["partition_group_sizes"]
        pool = sizes.loc[sizes.population.eq("development_pool")].iloc[0]
        self.assertEqual((pool.group_size, pool.groups, pool.records), (2, 4, 8))

    def test_production_and_frozen_targets_are_ignored(self):
        frame, assignments, exclusions, _, frozen = fixture()
        reference = pd.DataFrame({"Retailer": ["Shop", "Shop"], "ProductBrand": ["", "brand"],
            "ProductDescription": ["null", "text"], "ProductContents": [None, "contents"],
            "Segment": [UntouchableTarget(), UntouchableTarget()]})
        frozen["Segment"] = UntouchableTarget()
        result = profile_development(frame, assignments, exclusions, reference_features=reference,
                                     frozen_role_assignments=frozen)
        self.assertEqual(result["summary"]["production_reference_records"], 2)
        classes = result["class_distribution"]
        self.assertNotIn("production_reference", set(classes.population))
        self.assertIn("original_frozen_validation", set(classes.population))
        missing = result["feature_missingness"]
        production = missing.loc[missing.population.eq("production_reference")]
        self.assertTrue(production.missing_fraction.eq(.5).all())

    def test_validation_singleton_concentration_uses_whole_group_record_mix(self):
        ids = list("abcdef")
        base = pd.DataFrame({"record_id": ids, "group_id": ["g1"] * 4 + ["g2", "g3"], "Segment": ["A"] * 6})
        grouped = base.assign(split=["train"] * 4 + ["validation"] * 2, strategy="group_aware")
        random = base.assign(split=["train", "validation", "train", "train", "train", "validation"], strategy="random")
        assignments = pd.concat([random, grouped], ignore_index=True)
        frame = pd.DataFrame({"record_id": ids + ["x"], "Segment": ["A"] * 7,
            "Retailer": ["Amazon"] * 7, "ProductBrand": ["Brand"] * 7,
            "ProductDescription": ["text"] * 7, "ProductContents": ["text"] * 7})
        exclusions = pd.DataFrame({"record_id": ["x"], "reason": ["protected_final_test"]})
        result = profile_development(frame, assignments, exclusions)
        summary = {row["population"]: row for row in result["summary"]["partition_group_summary"]}
        self.assertAlmostEqual(summary["development_pool"]["development_singleton_record_fraction"], 1 / 3)
        self.assertEqual(summary["group_aware_validation"]["development_singleton_record_fraction"], 1)
        self.assertEqual(summary["group_aware_train"]["development_size_bucket_records"]["3-5"], 4)
        comparison = next(row for row in result["summary"]["group_size_total_variation"]
                          if row["comparison"] == "group_aware_validation")
        self.assertAlmostEqual(comparison["whole_development_size_bucket_record_tvd"], 2 / 3)

    def test_class_support_and_source_missingness_are_exact(self):
        frame, assignments, exclusions, historical, _ = fixture()
        result = profile_development(frame, assignments, exclusions, previous_assignments=historical)
        classes = result["class_distribution"]
        absent = classes.loc[classes.population.eq("previous_group_aware_train") & classes.Segment.eq("B")].iloc[0]
        self.assertEqual(absent.records, 0)
        random = classes.loc[classes.population.eq("random_train")].set_index("Segment")
        aware = classes.loc[classes.population.eq("group_aware_train")].set_index("Segment")
        self.assertTrue(random.records.equals(aware.records))
        missing = result["feature_missingness"]
        brands = missing.loc[missing.population.eq("development_pool") & missing.feature.eq("ProductBrand")].iloc[0]
        self.assertEqual(brands.missing_records, 2)
        self.assertEqual(brands.missing_fraction, .25)

    def test_rejects_incomplete_pool_unlabeled_pool_bad_exclusions_and_crossed_groups(self):
        frame, assignments, exclusions, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "frozen-v3"):
            profile_development(frame, assignments, exclusions, expected_development_ids=list("abcdefg"))
        with self.assertRaisesRegex(ValueError, "complete source"):
            profile_development(frame, assignments, exclusions.iloc[:1])
        invalid_labels = frame.copy()
        invalid_labels.loc[invalid_labels.record_id.eq("a"), "Segment"] = "null"
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            profile_development(invalid_labels, assignments, exclusions)
        bad_reasons = exclusions.copy()
        bad_reasons.loc[bad_reasons.record_id.eq("x"), "reason"] = "missing_Segment"
        with self.assertRaisesRegex(ValueError, "labeled record"):
            profile_development(frame, assignments, bad_reasons)
        crossed = assignments.copy()
        # Swap two equally labeled records, preserving the class budget but crossing groups.
        crossed.loc[crossed.strategy.eq("group_aware") & crossed.record_id.eq("b"), "split"] = "validation"
        crossed.loc[crossed.strategy.eq("group_aware") & crossed.record_id.eq("e"), "split"] = "train"
        with self.assertRaisesRegex(ValueError, "crossing"):
            profile_development(frame, crossed, exclusions)

    def test_results_are_invariant_to_record_order_and_inputs_are_not_modified(self):
        frame, assignments, exclusions, historical, frozen = fixture()
        before = [part.copy(deep=True) for part in (frame, assignments, exclusions, historical, frozen)]
        reference = profile_development(frame, assignments, exclusions, previous_assignments=historical,
                                        frozen_role_assignments=frozen)
        reordered = profile_development(frame.iloc[::-1], assignments.iloc[::-1], exclusions.iloc[::-1],
            previous_assignments=historical.iloc[::-1], frozen_role_assignments=frozen.iloc[::-1])
        self.assertEqual(reference["summary"], reordered["summary"])
        for table in ("retailer_distribution", "feature_missingness", "partition_group_sizes", "class_distribution"):
            pd.testing.assert_frame_equal(reference[table], reordered[table])
        for original, current in zip(before, (frame, assignments, exclusions, historical, frozen)):
            pd.testing.assert_frame_equal(original, current)


if __name__ == "__main__":
    unittest.main()
