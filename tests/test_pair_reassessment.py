"""Leakage judgments, identity independence, uncertainty, and sample integrity."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.pair_reassessment import (
    compare_saved_reviews, main, previous_criterion_comparison, score_annotations, validate_annotations,
)
from catalogiq.features import sha256
from catalogiq.split_review_validation import reserve_holdout, seal_refinement
from test_split_refinement_comparison import write_runs


def annotations():
    rows = []
    # Identity differs for a packaging/formula variant that nevertheless must
    # stay together; identity uncertainty is independent of leakage uncertainty.
    for i, (identity, leakage, left, right) in enumerate([
        ("different_product", "keep_together", "a", "b"),
        ("different_product", "keep_separate", "c", "d"),
        ("uncertain", "keep_together", "e", "f"),
        ("different_product", "uncertain", "g", "h"),
    ]):
        rows.append({"pair_id": str(i), "review_set": "heldout", "left_record_id": left,
                     "right_record_id": right, "product_identity": identity,
                     "identity_detail": "pack_size_variant" if i == 0 else "uncertain",
                     "identity_confidence": "low" if identity == "uncertain" else "high",
                     "identity_rationale": "Variant or incomplete identity information.",
                     "leakage_decision": leakage, "leakage_confidence": "high",
                     "leakage_rationale": "Distinctive repeated information assessed independently.",
                     "judgment_uses_Segment": "False"})
    return pd.DataFrame(rows)


def assignments():
    index = list("abcdefgh")
    rule = pd.DataFrame({"group_id": list("aacdefgh"),
                         "split": ["train", "train", "train", "test", "test", "test", "train", "test"]}, index=index)
    tfidf = pd.DataFrame({"group_id": list("abccdefg"),
                          "split": ["train", "test", "train", "train", "train", "test", "train", "test"]}, index=index)
    return {"baseline": {"rule": rule, "tfidf": tfidf}, "refined": {"rule": rule.copy(), "tfidf": tfidf.copy()}}


class PairReassessmentTests(unittest.TestCase):
    def test_different_identity_and_unknown_identity_can_require_grouping(self):
        out, metrics = score_annotations(annotations(), assignments())
        summary = metrics.loc[metrics.phase.eq("refined") & metrics.leakage_confidence.eq("all")].set_index("method")
        self.assertEqual(summary.at["rule", "keep_together_pairs"], 2)
        self.assertEqual(summary.at["rule", "keep_separate_pairs"], 1)
        self.assertEqual(summary.at["rule", "uncertain_pairs"], 1)
        self.assertEqual(summary.at["rule", "false_negative_pairs"], 1)
        self.assertEqual(summary.at["rule", "false_positive_pairs"], 0)
        self.assertEqual(summary.at["tfidf", "false_negative_pairs"], 2)
        self.assertEqual(summary.at["tfidf", "false_positive_pairs"], 1)
        self.assertTrue(out.loc[0, "expected_together"])
        self.assertFalse(out.loc[0, "refined_rule_false_positive"])

    def test_identity_and_identity_confidence_do_not_change_leakage_scores(self):
        original = annotations()
        changed = original.copy()
        changed["product_identity"] = "different_product"
        changed["identity_confidence"] = "low"
        _, old = score_annotations(original, assignments())
        _, new = score_annotations(changed, assignments())
        columns = [c for c in old if c != "different_products_requiring_grouping"]
        pd.testing.assert_frame_equal(old[columns], new[columns])

    def test_unknown_leakage_is_excluded_and_flags_remain_missing(self):
        out, metrics = score_annotations(annotations(), assignments())
        for method in ("rule", "tfidf"):
            self.assertTrue(pd.isna(out.loc[3, f"refined_{method}_false_negative"]))
            self.assertTrue(pd.isna(out.loc[3, f"refined_{method}_false_positive"]))
        self.assertEqual(int(metrics.iloc[0].pairs), 4)
        self.assertEqual(int(metrics.iloc[0].keep_together_pairs + metrics.iloc[0].keep_separate_pairs), 3)

    def test_actual_cross_split_leakage_is_distinct_from_grouping_false_negative(self):
        _, metrics = score_annotations(annotations(), assignments())
        rows = metrics.loc[metrics.phase.eq("refined") & metrics.leakage_confidence.eq("all")].set_index("method")
        self.assertEqual(rows.at["rule", "false_negative_pairs"], 1)
        self.assertEqual(rows.at["rule", "actual_cross_split_leakage_pairs"], 0)
        self.assertEqual(rows.at["rule", "missed_group_pairs_coincidentally_in_same_split"], 1)
        self.assertEqual(rows.at["tfidf", "actual_cross_split_leakage_pairs"], 2)

    def test_invalid_labels_target_use_missing_endpoint_or_crossing_group_rejected(self):
        for field, value in [("leakage_decision", "yes"), ("identity_confidence", "certain"),
                             ("judgment_uses_Segment", "True"), ("leakage_rationale", "")]:
            changed = annotations()
            changed.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_annotations(changed, assignments())
        changed = annotations()
        changed.loc[0, "right_record_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "absent"):
            score_annotations(changed, assignments())
        phases = assignments()
        phases["refined"]["rule"].loc["b", "split"] = "test"
        with self.assertRaisesRegex(ValueError, "isolation"):
            score_annotations(annotations(), phases)

    def test_pair_registry_allows_orientation_but_rejects_changed_cohort_or_identity(self):
        frame = annotations()
        registry = frame[["pair_id", "review_set", "left_record_id", "right_record_id"]].copy()
        swapped = frame.copy()
        swapped[["left_record_id", "right_record_id"]] = frame[["right_record_id", "left_record_id"]].to_numpy()
        validate_annotations(swapped, registry)
        swapped.loc[0, "review_set"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "cohorts"):
            validate_annotations(swapped, registry)
        changed = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_annotations(changed, registry)

    def test_previous_false_positive_can_be_appropriate_under_leakage_policy(self):
        out, _ = score_annotations(annotations(), assignments())
        original = annotations().copy()
        original["relation"] = ["unrelated_likely", "unrelated_likely", "uncertain", "uncertain"]
        original["confidence"] = "medium"
        out, metrics = previous_criterion_comparison(out, {"heldout": original})
        rule = metrics.loc[metrics.phase.eq("refined") & metrics.method.eq("rule")].set_index("criterion")
        self.assertEqual(rule.at["previous_review", "false_positive_pairs"], 1)
        self.assertEqual(rule.at["evaluation_leakage", "false_positive_pairs"], 0)
        self.assertTrue(out.loc[0, "expected_together"])
        self.assertFalse(out.loc[0, "previous_expected_together"])

    def test_scoring_repeat_and_row_permutation_preserve_pair_results(self):
        frame, phases = annotations(), assignments()
        original, metrics = score_annotations(frame, phases)
        repeated, again = score_annotations(frame.copy(), deepcopy(phases))
        pd.testing.assert_frame_equal(original, repeated)
        pd.testing.assert_frame_equal(metrics, again)
        permuted, _ = score_annotations(frame.sample(frac=1, random_state=42), phases)
        pd.testing.assert_frame_equal(original.set_index("pair_id").sort_index(), permuted.set_index("pair_id").sort_index())

    def test_cli_help(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as context:
            main(["--help"])
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--review-dir", stdout.getvalue())

    def test_saved_comparison_reproduces_and_rejects_judgment_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, diagnostic = write_runs(root)
            holdout = root / "holdout"
            reserve_holdout(baseline, diagnostic, holdout)
            seal_refinement(refined, holdout)
            heldout_source = root / "heldout_source.csv"
            pd.DataFrame([{"left_record_id": "d", "right_record_id": "e",
                           "relation": "related_variant_likely", "confidence": "high"}]).to_csv(heldout_source, index=False)
            review = root / "review"
            review.mkdir()
            frame = annotations().iloc[:3].copy()
            frame["left_record_id"], frame["right_record_id"] = ["a", "b", "d"], ["b", "c", "e"]
            frame["review_set"] = ["diagnostic", "diagnostic", "heldout"]
            frame["leakage_decision"] = ["keep_together", "keep_separate", "uncertain"]
            frame["original_review_scope"] = "synthetic"
            frame["original_review_id"] = frame.pair_id
            frame["left_ProductName"], frame["right_ProductName"] = "Left", "Right"
            registry = review / "pair_registry.csv"
            frame[["pair_id", "review_set", "left_record_id", "right_record_id"]].to_csv(registry, index=False)
            annotation_path = review / "dual_judgments.csv"
            frame.to_csv(annotation_path, index=False)
            blind = review / "blind_pairs_A.json"
            blind.write_text("[]", encoding="utf-8")
            protocol = {
                "input_sha256": {str(path): sha256(path) for path in (diagnostic, heldout_source)},
                "input_record_fingerprints": {}, "blind_file_sha256": {blind.name: sha256(blind)},
                "registry_sha256": sha256(registry), "diagnostic_pairs": 2, "heldout_pairs": 1,
                "interpretation": "Selected synthetic examples.",
            }
            protocol_path = review / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            (review / "judgment_freeze.json").write_text(json.dumps({
                "annotations_sha256": sha256(annotation_path), "protocol_sha256": sha256(protocol_path),
                "frozen_utc": "2026-10-08T00:00:00+00:00",
            }), encoding="utf-8")
            before = {path: sha256(path) for path in (diagnostic, heldout_source, annotation_path)}
            out, metrics = compare_saved_reviews(review, baseline, refined, holdout, root / "first")
            compare_saved_reviews(review, baseline, refined, holdout, root / "second")
            for file in (root / "first").glob("*.csv"):
                self.assertEqual(file.read_bytes(), (root / "second" / file.name).read_bytes())
            self.assertEqual(before, {path: sha256(path) for path in before})
            self.assertEqual(len(out), 3)
            unknown = metrics.loc[metrics.review_set.eq("heldout") & metrics.leakage_confidence.eq("all")]
            self.assertTrue(unknown.false_positive_rate.isna().all())
            self.assertTrue(unknown.false_negative_rate.isna().all())
            annotation_path.write_bytes(annotation_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "judgments changed"):
                compare_saved_reviews(review, baseline, refined, holdout, root / "tampered")
            self.assertFalse((root / "tampered").exists())


if __name__ == "__main__":
    unittest.main()
