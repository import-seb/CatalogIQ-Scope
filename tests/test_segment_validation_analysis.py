import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from catalogiq.segment_validation_analysis import (
    affected_row_bound, annotate_training_neighbors, bootstrap_predictions,
    compare_paired_predictions, evaluate_predictions, majority_baseline,
    summarize_cohorts,
)


LABELS = ["A", "B", "C"]


def predictions():
    return pd.DataFrame({"record_id": ["a", "b", "c", "d"],
        "true_label": ["A", "A", "B", "C"], "predicted_label": ["A", "B", "B", "A"],
        "confidence": [.8, .6, .7, .6], "probability_0": [.8, .3, .2, .6],
        "probability_1": [.1, .6, .7, .1], "probability_2": [.1, .1, .1, .3],
        "group_id": ["ab", "ab", "cd", "cd"]})


class SegmentValidationAnalysisTests(unittest.TestCase):
    def test_metrics_match_sklearn_and_probability_metrics_have_declared_semantics(self):
        frame = predictions()
        result = evaluate_predictions(frame, LABELS)
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["accuracy"], accuracy_score(frame.true_label, frame.predicted_label))
        self.assertAlmostEqual(metrics["macro_f1"], f1_score(frame.true_label, frame.predicted_label,
            labels=LABELS, average="macro", zero_division=0))
        self.assertAlmostEqual(metrics["weighted_f1"], f1_score(frame.true_label, frame.predicted_label,
            labels=LABELS, average="weighted", zero_division=0))
        self.assertAlmostEqual(metrics["balanced_accuracy"],
            balanced_accuracy_score(frame.true_label, frame.predicted_label))
        probability = frame[[f"probability_{i}" for i in range(3)]].to_numpy()
        self.assertAlmostEqual(metrics["negative_log_likelihood"],
            log_loss(frame.true_label, probability, labels=LABELS))
        true = np.eye(3)[[0, 0, 1, 2]]
        self.assertAlmostEqual(metrics["multiclass_brier"], np.square(probability - true).sum(axis=1).mean())
        self.assertEqual(result["confusion"].to_numpy().tolist(), [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(result["per_class"].true_support.tolist(), [2, 1, 1])
        self.assertEqual(len(result["calibration"]), 10)
        self.assertEqual(int(result["calibration"].records.sum()), len(frame))

    def test_confidence_one_and_zero_are_counted_and_ece_uses_bin_weighting(self):
        frame = pd.DataFrame({"record_id": ["x", "y", "z"], "true_label": ["A", "A", "B"],
            "predicted_label": ["A", "B", "B"], "confidence": [1, 0, .75]})
        result = evaluate_predictions(frame, ["A", "B"], bins=2)
        self.assertEqual(result["calibration"].records.tolist(), [1, 2])
        self.assertAlmostEqual(result["metrics"]["expected_calibration_error"], 1 / 12)
        self.assertIsNone(result["metrics"]["negative_log_likelihood"])

    def test_unsupported_classes_are_retained_and_empty_cohorts_are_unscored(self):
        frame = predictions().iloc[:1]
        result = evaluate_predictions(frame, LABELS)
        self.assertEqual(result["metrics"]["macro_f1"], 1 / 3)
        self.assertEqual(result["metrics"]["present_class_macro_f1"], 1)
        self.assertEqual(result["metrics"]["unsupported_true_labels"], ["B", "C"])
        empty = evaluate_predictions(frame.iloc[:0], LABELS)
        self.assertIsNone(empty["metrics"]["accuracy"])
        self.assertIsNone(empty["metrics"]["negative_log_likelihood"])
        self.assertEqual(empty["confusion"].shape, (3, 3))

    def test_invalid_records_labels_or_probabilities_cannot_silently_score(self):
        changes = [lambda f: f.assign(record_id=["a", "a", "c", "d"]),
                   lambda f: f.assign(true_label=["Z", "A", "B", "C"]),
                   lambda f: f.assign(probability_0=[.9, .3, .2, .6]),
                   lambda f: f.drop(columns="probability_2"),
                   lambda f: f.assign(confidence=[.9, .6, .7, .6]),
                   lambda f: f.assign(predicted_label=["B", "B", "B", "A"])]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    evaluate_predictions(change(predictions()), LABELS)

    def test_majority_classifier_uses_train_only_and_deterministic_tie_order(self):
        result = majority_baseline(["B", "A"], ["B", "B", "B"], LABELS)
        self.assertEqual(result["majority_label"], "A")
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["training_class_counts"], {"A": 1, "B": 1, "C": 0})
        with self.assertRaises(ValueError):
            majority_baseline([], ["A"], LABELS)

    def test_neighbor_annotation_counts_unique_training_peers_only(self):
        groups = pd.DataFrame({"record_id": ["a", "b", "c", "d", "t1", "t2", "v2"],
            "group_id": ["g1", "g2", "g3", "g4", "g1", "g1", "g2"]})
        near = pd.DataFrame({"left_record_id": ["a", "t1", "a", "b", "d"],
            "right_record_id": ["t1", "a", "t2", "v2", "c"]})
        exact = pd.DataFrame({"left_record_id": ["t2"], "right_record_id": ["c"]})
        frame = predictions().iloc[::-1].copy()
        result = annotate_training_neighbors(frame, ["t1", "t2"], groups, near, exact).set_index("record_id")
        self.assertEqual(result.index.tolist(), ["d", "c", "b", "a"])
        self.assertEqual(result.at["a", "rule_group_train_neighbors"], 2)
        self.assertEqual(result.at["a", "independent_near_train_neighbors"], 2)
        self.assertTrue(result.at["a", "has_independent_near_train_neighbor"])
        self.assertFalse(result.at["b", "has_independent_near_train_neighbor"])
        self.assertTrue(result.at["c", "has_exact_payload_train_neighbor"])
        self.assertFalse(result.at["c", "has_rule_group_train_neighbor"])
        unobserved = annotate_training_neighbors(frame, ["t1"], groups, near)
        self.assertTrue(unobserved.has_exact_payload_train_neighbor.isna().all())
        self.assertFalse(unobserved.exact_payload_audit_available.any())

    def test_neighbor_annotation_rejects_train_validation_overlap_and_bad_audit(self):
        groups = pd.DataFrame({"record_id": ["a", "b", "c", "d", "t"],
            "group_id": ["a", "b", "c", "d", "t"]})
        near = pd.DataFrame({"left_record_id": ["a"], "right_record_id": ["t"]})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            annotate_training_neighbors(predictions(), ["a"], groups, near)
        with self.assertRaisesRegex(ValueError, "outside"):
            annotate_training_neighbors(predictions(), ["unknown"], groups, near)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            annotate_training_neighbors(predictions(), ["t"], groups, near.assign(right_record_id="unknown"))

    def test_group_bootstrap_is_reproducible_and_insensitive_to_row_order(self):
        frame = predictions()
        first = bootstrap_predictions(frame, LABELS, n_bootstrap=50, seed=17)
        second = bootstrap_predictions(frame.iloc[::-1], LABELS, n_bootstrap=50, seed=17)
        pd.testing.assert_frame_equal(first["intervals"], second["intervals"])
        self.assertEqual(first["groups"], 2)
        self.assertEqual(first["intervals"].iloc[0].estimate, .5)

    def test_one_cluster_interval_cannot_pretend_rows_are_independent(self):
        frame = predictions().assign(group_id="single_family")
        result = bootstrap_predictions(frame, LABELS, n_bootstrap=25)["intervals"]
        self.assertTrue(np.allclose(result.estimate, result.lower))
        self.assertTrue(np.allclose(result.estimate, result.upper))
        with self.assertRaises(ValueError):
            bootstrap_predictions(frame.assign(group_id=None), LABELS)

    def test_paired_comparison_uses_only_same_rows_and_same_cluster_draws(self):
        frame = predictions().drop(columns=["confidence", "probability_0", "probability_1", "probability_2"])
        groups = frame[["record_id", "group_id"]]
        left = frame.iloc[:3].copy()
        right = frame.iloc[1:].copy()
        right.loc[right.record_id.eq("b"), "predicted_label"] = "A"
        result = compare_paired_predictions(left, right, LABELS, groups, n_bootstrap=50, seed=1)
        self.assertEqual(result["record_ids"], ["b", "c"])
        self.assertEqual(result["common_records"], 2)
        self.assertAlmostEqual(result["left_coverage"], 2 / 3)
        accuracy = result["intervals"].set_index("metric").loc["accuracy"]
        self.assertEqual(accuracy.left, .5)
        self.assertEqual(accuracy.right, 1)
        self.assertEqual(accuracy.left_minus_right, -.5)
        same = compare_paired_predictions(frame, frame.iloc[::-1], LABELS, groups, n_bootstrap=20)
        self.assertTrue(same["intervals"][["left_minus_right", "lower", "upper"]].eq(0).all().all())
        wrong = right.assign(true_label="C")
        with self.assertRaisesRegex(ValueError, "truth"):
            compare_paired_predictions(left, wrong, LABELS, groups)

    def test_empty_intersection_is_reported_without_inventing_a_gap(self):
        frame = predictions()
        result = compare_paired_predictions(frame.iloc[:1], frame.iloc[1:], LABELS,
            frame[["record_id", "group_id"]], n_bootstrap=5)
        self.assertEqual(result["common_records"], 0)
        self.assertTrue(result["intervals"].left_minus_right.isna().all())

    def test_cohorts_include_class_counts_and_do_not_force_unknown_audit_to_absent(self):
        frame = predictions().assign(has_independent_near_train_neighbor=[True, True, False, False],
                                    has_exact_payload_train_neighbor=pd.NA)
        result = summarize_cohorts(frame, LABELS)
        indexed = result["metrics"].set_index("cohort")
        present = indexed.loc["has_independent_near_train_neighbor=true"]
        self.assertEqual(present.records, 2)
        self.assertEqual(present.classes_with_true_support, 1)
        self.assertTrue(present.unsupported_class_caveat)
        self.assertEqual(indexed.loc["has_exact_payload_train_neighbor=false"].records, 0)
        self.assertEqual(indexed.loc["has_exact_payload_train_neighbor=true"].records, 0)
        self.assertIn("composition", result["interpretation"])

    def test_affected_row_bound_is_explicitly_conditional_and_not_population_claim(self):
        frame = predictions().assign(has_independent_near_train_neighbor=[True, False, False, False])
        bound = affected_row_bound(frame)
        self.assertEqual(bound["audited_affected_records"], 1)
        self.assertEqual(bound["audited_affected_row_share"], .25)
        self.assertEqual(bound["conditional_direct_only_accuracy_bound_percentage_points"], 25)
        self.assertFalse(bound["is_causal_or_population_bound"])
        self.assertIn("incomplete", bound["caveat"])


if __name__ == "__main__":
    unittest.main()
