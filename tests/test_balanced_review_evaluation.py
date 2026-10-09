import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from catalogiq.balanced_review_evaluation import (
    adjudicate_secondary_reviews, compare_balanced_sample, freeze_balanced_reviews,
    score_balanced_reviews, validate_balanced_reviews,
)
from catalogiq.balanced_pair_sample import EVIDENCE_FIELDS
from catalogiq.features import sha256


def reviews():
    rows = []
    for pair_id, left, right, category, decision, confidence in (
        ("B1", "a", "b", "related", "keep_together", "high"),
        ("B2", "b", "c", "safe_to_separate", "keep_separate", "high"),
        ("B3", "a", "c", "borderline", "uncertain", "medium"),
        ("B4", "d", "e", "borderline", "keep_together", "medium"),
    ):
        rows.append(dict(pair_id=pair_id, left_record_id=left, right_record_id=right,
                         review_category=category, leakage_decision=decision, leakage_confidence=confidence,
                         product_identity="different_product", identity_detail="distinct_product",
                         identity_confidence="high", identity_rationale="Distinct sellable variants.",
                         leakage_rationale="Review of informative product text.", judgment_uses_Segment=False,
                         judgment_uses_method_outcomes=False,
                         left_ProductName="Item " + left, right_ProductName="Item " + right))
    return pd.DataFrame(rows)


def assignments():
    rule = pd.DataFrame({"group_id": ["g1", "g2", "g2", "g3", "g4"],
                         "split": ["train", "train", "train", "validation", "test"]}, index=list("abcde"))
    tfidf = pd.DataFrame({"group_id": ["h1", "h1", "h2", "h3", "h3"],
                          "split": ["train", "train", "test", "validation", "validation"]}, index=list("abcde"))
    return {"rule": rule, "tfidf": tfidf}


class BalancedEvaluationTests(unittest.TestCase):
    def test_independent_leakage_disagreement_retained_without_forcing_labels(self):
        primary = reviews()
        secondary = primary.iloc[:1].copy()
        secondary["leakage_decision"] = "keep_separate"
        secondary["review_category"] = "safe_to_separate"
        result, agreement = adjudicate_secondary_reviews(primary, secondary)
        self.assertEqual(result.loc[0, "leakage_decision"], "uncertain")
        self.assertEqual(result.loc[0, "review_category"], "borderline")
        self.assertFalse(agreement.loc[0, "leakage_agrees"])
        self.assertEqual(result.loc[1, "leakage_decision"], primary.loc[1, "leakage_decision"])

    def test_identity_disagreement_does_not_change_leakage(self):
        primary = reviews()
        secondary = primary.iloc[:1].copy()
        secondary["product_identity"] = "same_product"
        secondary["identity_detail"] = "identical_item"
        result, agreement = adjudicate_secondary_reviews(primary, secondary)
        self.assertEqual(result.loc[0, "product_identity"], "uncertain")
        self.assertEqual(result.loc[0, "leakage_decision"], "keep_together")
        self.assertEqual(result.loc[0, "review_category"], "related")
        self.assertFalse(agreement.loc[0, "identity_agrees"])

    def test_secondary_review_preserves_boundary_confidence_and_registry(self):
        primary = reviews()
        secondary = primary.iloc[:1].copy()
        secondary["leakage_confidence"] = "medium"
        secondary["review_category"] = "borderline"
        result, _ = adjudicate_secondary_reviews(primary, secondary)
        self.assertEqual(result.loc[0, "leakage_confidence"], "medium")
        self.assertEqual(result.loc[0, "leakage_decision"], "keep_together")
        secondary.loc[0, "right_record_id"] = "z"
        with self.assertRaisesRegex(ValueError, "endpoints differ"):
            adjudicate_secondary_reviews(primary, secondary)

    def test_frozen_review_end_to_end_repeat_and_tamper_detection(self):
        # Synthetic saved assignments: exercising review lifecycle, not matchers.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            review_dir, run = root / "reviews", root / "run"
            review_dir.mkdir()
            run.mkdir()
            frame = reviews().iloc[:3].copy()
            frame["left_record_id"] = ["a", "c", "e"]
            frame["right_record_id"] = ["b", "d", "f"]
            pool = frame[["pair_id", *[f"{s}_record_id" for s in ("left", "right")]]].copy()
            for side in ("left", "right"):
                for field in EVIDENCE_FIELDS:
                    pool[f"{side}_{field}"] = ("Product " + pool[f"{side}_record_id"] if field == "ProductName" else "")
            pool.to_csv(review_dir / "candidate_pool.csv", index=False)
            (review_dir / "candidate_protocol.json").write_text("{}", encoding="utf-8")
            source = root / "input.csv"
            source.write_text("fixed source\n", encoding="utf-8")
            (run / "refinement_freeze.json").write_text("{}", encoding="utf-8")
            for method in ("rule", "tfidf"):
                pd.DataFrame({"record_id": list("abcdef"), "group_id": ["g1", "g1", "g2", "g3", "g4", "g4"],
                              "split": ["train", "train", "train", "test", "validation", "validation"]}).to_csv(run / f"{method}_assignments.csv", index=False)
            protocol = {"category_quota": 1, "seed": 20261011, "no_method_refinement": True,
                        "candidate_protocol_sha256": sha256(review_dir / "candidate_protocol.json"),
                        "candidate_artifact_sha256": {"candidate_pool.csv": sha256(review_dir / "candidate_pool.csv")},
                        "method_freeze_sha256": sha256(run / "refinement_freeze.json"),
                        "input_record_fingerprints": {str(source): sha256(source)}}
            (review_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            annotations = review_dir / "review.json"
            annotations.write_text(json.dumps(frame.to_dict("records")), encoding="utf-8")
            selected, seal = freeze_balanced_reviews(review_dir, [annotations], per_category=1)
            self.assertEqual(len(selected), 3)
            self.assertTrue(seal["pre_score_freeze"])
            with self.assertRaises(FileExistsError):
                freeze_balanced_reviews(review_dir, [annotations], per_category=1)
            with patch("catalogiq.balanced_review_evaluation.verify_refinement_seal"):
                with self.assertRaisesRegex(ValueError, "outside the frozen run"):
                    compare_balanced_sample(review_dir, run, run / "new_analysis")
                first = compare_balanced_sample(review_dir, run, root / "first")
                second = compare_balanced_sample(review_dir, run, root / "second")
                self.assertEqual(first["result_sha256"], second["result_sha256"])
                for name in first["result_sha256"]:
                    self.assertEqual((root / "first" / name).read_bytes(), (root / "second" / name).read_bytes())
                annotations.write_text("[]", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pre-score freeze"):
                    compare_balanced_sample(review_dir, run, root / "third")

    def test_errors_are_leakage_decisions_not_identity(self):
        outcomes, metrics, sensitivity = score_balanced_reviews(reviews(), assignments())
        core = metrics.loc[metrics.review_category.eq("decisive_core")].set_index("method")
        self.assertEqual(core.at["rule", "false_negative_pairs"], 1)
        self.assertEqual(core.at["rule", "false_positive_pairs"], 1)
        self.assertEqual(core.at["rule", "actual_cross_split_leakage_pairs"], 0)
        self.assertEqual(core.at["rule", "missed_pairs_coincidentally_same_split"], 1)
        self.assertEqual(core.at["tfidf", "false_negative_pairs"], 0)
        self.assertEqual(core.at["tfidf", "false_positive_pairs"], 0)
        self.assertEqual(outcomes.loc[0, "product_identity"], "different_product")
        self.assertTrue(outcomes.loc[0, "tfidf_grouped"])
        self.assertIn("not confidence intervals", sensitivity.loc[0, "note"])

    def test_unknowns_excluded_and_borderline_separate(self):
        outcomes, metrics, bounds = score_balanced_reviews(reviews(), assignments())
        self.assertTrue(pd.isna(outcomes.loc[2, "rule_false_negative"]))
        self.assertTrue(pd.isna(outcomes.loc[2, "rule_false_positive"]))
        borderline = metrics.loc[metrics.review_category.eq("borderline")].set_index("method")
        self.assertEqual(borderline.at["rule", "uncertain_pairs"], 1)
        self.assertEqual(borderline.at["rule", "false_negative_pairs"], 1)
        self.assertEqual(borderline.at["tfidf", "false_negative_pairs"], 0)
        bounds = bounds.set_index("method")
        self.assertEqual(bounds.at["rule", "false_negative_lower_bound"], 1)
        self.assertEqual(bounds.at["rule", "false_negative_upper_bound"], 2)
        self.assertEqual(bounds.at["tfidf", "false_positive_upper_bound"], 0)

    def test_deterministic_and_input_order_invariant(self):
        first = score_balanced_reviews(reviews(), assignments())
        reversed_inputs = {method: frame.iloc[::-1] for method, frame in assignments().items()}
        second = score_balanced_reviews(reviews(), reversed_inputs)
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)

    def test_target_label_review_rejected(self):
        frame = reviews()
        frame.loc[0, "judgment_uses_Segment"] = True
        with self.assertRaisesRegex(ValueError, "Segment"):
            validate_balanced_reviews(frame)
        frame = reviews()
        frame.loc[0, "judgment_uses_method_outcomes"] = True
        with self.assertRaisesRegex(ValueError, "method outcomes"):
            validate_balanced_reviews(frame)

    def test_ambiguous_core_and_duplicate_reviews_rejected(self):
        frame = reviews()
        frame.loc[0, "leakage_confidence"] = "medium"
        with self.assertRaisesRegex(ValueError, "high-confidence"):
            validate_balanced_reviews(frame)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_balanced_reviews(pd.concat([reviews(), reviews().iloc[:1]], ignore_index=True))

    def test_group_isolation_record_coverage_and_shared_input(self):
        assigned = assignments()
        assigned["rule"].loc["c", "split"] = "test"
        with self.assertRaisesRegex(ValueError, "boundaries"):
            score_balanced_reviews(reviews(), assigned)
        assigned = assignments()
        assigned["tfidf"] = assigned["tfidf"].drop(index="e")
        with self.assertRaisesRegex(ValueError, "same input"):
            score_balanced_reviews(reviews(), assigned)

    def test_quota_registry_and_identity_contradiction(self):
        with self.assertRaisesRegex(ValueError, "quota"):
            validate_balanced_reviews(reviews(), quota=1)
        registry = reviews().copy()
        registry.loc[0, "left_ProductName"] = "Changed"
        with self.assertRaisesRegex(ValueError, "registry changed"):
            validate_balanced_reviews(reviews(), registry)
        frame = reviews()
        frame.loc[1, "product_identity"] = "same_product"
        with self.assertRaisesRegex(ValueError, "same product"):
            validate_balanced_reviews(frame)


if __name__ == "__main__":
    unittest.main()
