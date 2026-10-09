import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.segment_population_diagnostics import (
    population_annotations, summarize_population, validate_validation_predictions,
)
from catalogiq.splitting import text as normalize_text


def fixture():
    base = pd.DataFrame({"record_id": ["a", "b", "c", "d"],
        "group_id": ["g1", "g1", "g2", "g2"], "Segment": ["A", "B", "A", "B"]})
    grouped = base.assign(split=["train", "train", "validation", "validation"], strategy="group_aware")
    random = base.assign(split=["train", "validation", "validation", "train"], strategy="random")
    assignments = pd.concat([random, grouped], ignore_index=True)
    exclusions = pd.DataFrame({"record_id": ["protected_test"]})
    features = pd.DataFrame({"record_id": ["a", "b", "c", "d", "protected_test"],
        "ProductBrand": ["Alpha", "Beta", "Beta", "Gamma", "NEVER_NORMALIZE_TEST"],
        "Retailer": ["R1", "R2", "R1", "R3", "NEVER_NORMALIZE_TEST"],
        "ProductDescription": ["null", "description", "!!!", "", "NEVER_NORMALIZE_TEST"],
        "ProductContents": ["", "contents", " null ", "contents", "NEVER_NORMALIZE_TEST"]})
    predictions = {}
    for strategy in ("random", "group_aware"):
        selected = assignments.loc[assignments.strategy.eq(strategy) & assignments.split.eq("validation")]
        predictions[strategy] = pd.DataFrame({"record_id": selected.record_id.to_list(),
            "true_label": selected.Segment.to_list(), "predicted_label": selected.Segment.to_list()})
    return assignments, exclusions, features, predictions


class SegmentPopulationDiagnosticsTests(unittest.TestCase):
    def test_rejects_test_train_duplicate_unknown_and_changed_truth_predictions(self):
        assignments, exclusions, _, predictions = fixture()
        good = predictions["group_aware"]
        self.assertTrue(validate_validation_predictions(good, assignments, exclusions, "group_aware"))
        invalid = [good.assign(record_id=["protected_test", "d"]),
                   good.assign(record_id=["a", "d"]), good.assign(record_id=["unknown", "d"]),
                   good.assign(record_id=["c", "c"]), good.assign(true_label=["B", "B"])]
        for frame in invalid:
            with self.subTest(records=frame.record_id.to_list()):
                with self.assertRaises(ValueError):
                    validate_validation_predictions(frame, assignments, exclusions, "group_aware")

    def test_filters_test_features_before_normalization_and_uses_train_only_brand_coverage(self):
        assignments, exclusions, features, _ = fixture()

        def guarded_normalization(value):
            self.assertNotEqual(value, "NEVER_NORMALIZE_TEST")
            return normalize_text(value)

        with patch("catalogiq.segment_population_diagnostics.text", new=guarded_normalization):
            annotated = population_annotations(assignments, exclusions, features)
        self.assertNotIn("protected_test", set(annotated.record_id))
        grouped = annotated.loc[annotated.strategy.eq("group_aware")].set_index("record_id")
        random = annotated.loc[annotated.strategy.eq("random")].set_index("record_id")
        self.assertEqual(grouped.loc["c", "product_brand_train_seen"], "seen")
        self.assertEqual(grouped.loc["d", "product_brand_train_seen"], "unseen")
        self.assertEqual(random.loc["c", "product_brand_train_seen"], "unseen")
        self.assertEqual(grouped.loc["c", "description_presence"], "present")
        self.assertEqual(grouped.loc["a", "description_presence"], "missing")
        self.assertTrue(annotated.development_group_size.eq(2).all())
        self.assertTrue(annotated.group_label_homogeneity.eq("mixed_Segment").all())

    def test_cohort_composition_reports_absent_classes_empty_fixed_buckets_and_common_rows(self):
        assignments, exclusions, features, predictions = fixture()
        near = pd.DataFrame({"left_record_id": ["a"], "right_record_id": ["c"]})
        result = summarize_population(assignments, exclusions, features, predictions, ["A", "B", "C"], near)
        self.assertEqual(result["shared_validation_records"], 1)
        comp = result["class_composition"]
        base = comp.loc[comp.strategy.eq("group_aware") & comp.population.eq("strategy_validation")
                        & comp.dimension.eq("all")]
        self.assertEqual(dict(zip(base.label, base.true_support)), {"A": 1, "B": 1, "C": 0})
        self.assertEqual(base.class_fraction.sum(), 1)
        metrics = result["cohort_metrics"]
        row = metrics.loc[metrics.strategy.eq("group_aware") & metrics.population.eq("strategy_validation")
                          & metrics.dimension.eq("all")].iloc[0]
        self.assertEqual(row.records, 2)
        self.assertAlmostEqual(row.macro_f1, 2 / 3)
        self.assertEqual(row.present_class_macro_f1, 1)
        empty = metrics.loc[metrics.dimension.eq("development_group_size_bucket") & metrics.value.eq("11+")]
        self.assertTrue(empty.records.eq(0).all())
        per_class = result["cohort_per_class"]
        absent = per_class.loc[per_class.label.eq("C")]
        self.assertTrue(absent.true_support.eq(0).all())
        shared = metrics.loc[metrics.population.eq("shared_validation_intersection")
                             & metrics.dimension.eq("random_has_independent_near_train_neighbor")
                             & metrics.value.eq("yes")]
        self.assertEqual(set(shared.strategy), {"random", "group_aware"})
        self.assertTrue(shared.records.eq(1).all())
        groups = result["partition_group_summary"]
        grouped_train = groups.loc[groups.strategy.eq("group_aware") & groups.population.eq("strategy_train")].iloc[0]
        random_train = groups.loc[groups.strategy.eq("random") & groups.population.eq("strategy_train")].iloc[0]
        self.assertEqual(grouped_train.unique_frozen_groups, 1)
        self.assertEqual(random_train.unique_frozen_groups, 2)


if __name__ == "__main__":
    unittest.main()
