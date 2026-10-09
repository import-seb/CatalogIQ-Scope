"""Identifier-only, family-separated reservation and weak-label diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.split_review_validation import (
    HOLDOUT_COLUMNS, PAIR_COLUMNS, STRATA, diagnostic_comparison,
    reserve_holdout, seal_refinement, verify_holdout, verify_refinement_seal,
)


def write_fixture(root):
    """Create four strata without names, attributes, or Segment columns."""
    baseline = root / "baseline"
    baseline.mkdir()
    rule = {key: key for key in ("a", "b", "c", "d")}
    tfidf = dict(rule)
    rule["b"] = "a"
    tfidf["c"] = "b"
    rule_edges, tfidf_edges, near = [("a", "b")], [("b", "c")], [("c", "d")]
    for i in range(8):
        for prefix in ("ro", "to", "bo", "un"):
            left, right = f"{prefix}{i}l", f"{prefix}{i}r"
            rule[left], rule[right] = left, right
            tfidf[left], tfidf[right] = left, right
            if prefix in ("ro", "bo"):
                rule[right] = left
                rule_edges.append((left, right))
            if prefix in ("to", "bo"):
                tfidf[right] = left
                tfidf_edges.append((left, right))
            if prefix == "un":
                near.append((left, right))
    for method, groups in (("rule", rule), ("tfidf", tfidf)):
        pd.DataFrame([{"record_id": key, "group_id": value, "split": "train"}
                      for key, value in groups.items()]).to_csv(baseline / f"{method}_assignments.csv", index=False)
    for method, edges in (("rule", rule_edges), ("tfidf", tfidf_edges)):
        pd.DataFrame(edges, columns=PAIR_COLUMNS).to_csv(baseline / f"{method}_matching_edges.csv", index=False)
    pd.DataFrame(near, columns=PAIR_COLUMNS).to_csv(baseline / "independent_near_duplicate_pairs.csv", index=False)
    diagnostic = root / "diagnostic.csv"
    pd.DataFrame([{"review_id": "0", "review_scope": "diagnostic", "relation": "uncertain",
                   "confidence": "low", "left_record_id": "a", "right_record_id": "b"}]).to_csv(diagnostic, index=False)
    return baseline, diagnostic


class ReviewValidationTests(unittest.TestCase):
    def test_reservation_is_identifier_only_disjoint_deterministic_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            first, second = root / "holdout1", root / "holdout2"
            summary = reserve_holdout(baseline, diagnostic, first, pairs_per_stratum=3)
            reserve_holdout(baseline, diagnostic, second, pairs_per_stratum=3)
            self.assertEqual((first / "holdout_pairs.csv").read_bytes(), (second / "holdout_pairs.csv").read_bytes())
            pairs = verify_holdout(first)
            self.assertEqual(list(pairs.columns), HOLDOUT_COLUMNS)
            self.assertEqual(len(pairs), 12)
            self.assertEqual(set(pairs.baseline_stratum), set(STRATA))
            self.assertFalse(pairs.baseline_component_id.duplicated().any())
            self.assertFalse((set(pairs.left_record_id) | set(pairs.right_record_id)) & {"a", "b", "c", "d"})
            # c and d were never directly annotated, but share the union family.
            self.assertEqual(summary["excluded_records"], 4)
            self.assertEqual(summary["excluded_components"], 1)
            with self.assertRaises(FileExistsError):
                reserve_holdout(baseline, diagnostic, first)

    def test_selection_ignores_labels_attributes_order_and_refined_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            reserve_holdout(baseline, diagnostic, root / "first", pairs_per_stratum=3)
            expected = (root / "first/holdout_pairs.csv").read_bytes()
            # Change annotation judgment and add inaccessible-to-selection data.
            annotations = pd.read_csv(diagnostic)
            annotations["relation"] = "unrelated_likely"
            annotations["left_ProductName"] = "secret held-out content"
            annotations.to_csv(diagnostic, index=False)
            for method in ("rule", "tfidf"):
                path = baseline / f"{method}_assignments.csv"
                frame = pd.read_csv(path).iloc[::-1]
                frame["Segment"] = "arbitrary labels"
                frame["ProductName"] = "not a sampling input"
                frame.to_csv(path, index=False)
                edge_path = baseline / f"{method}_matching_edges.csv"
                edges = pd.read_csv(edge_path).iloc[::-1]
                edges[list(PAIR_COLUMNS)] = edges[list(PAIR_COLUMNS[::-1])].to_numpy()
                edges.to_csv(edge_path, index=False)
            (root / "unrelated_refinement").mkdir()
            reserve_holdout(baseline, diagnostic, root / "second", pairs_per_stratum=3)
            self.assertEqual(expected, (root / "second/holdout_pairs.csv").read_bytes())

    def test_holdout_detects_pair_or_original_input_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            reserve_holdout(baseline, diagnostic, root / "holdout")
            with diagnostic.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "reservation input changed"):
                verify_holdout(root / "holdout")
            with (root / "holdout/holdout_pairs.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "pair file changed"):
                verify_holdout(root / "holdout")

    def test_insufficient_eligible_components_returns_shortfall(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            summary = reserve_holdout(baseline, diagnostic, root / "holdout", pairs_per_stratum=100)
            self.assertEqual(summary["selected_pairs"], 32)
            self.assertTrue(all(value["selected_pairs"] == 8 for value in summary["strata"].values()))

    def test_global_one_family_limit_across_strata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            near_path = baseline / "independent_near_duplicate_pairs.csv"
            near = pd.read_csv(near_path)
            near.loc[len(near)] = ["ro0l", "to0l"]
            near.to_csv(near_path, index=False)
            summary = reserve_holdout(baseline, diagnostic, root / "holdout", pairs_per_stratum=100)
            self.assertEqual(summary["selected_components"], 31)
            self.assertEqual(summary["selected_pairs"], 31)

    def test_unknown_annotations_excluded_from_rates_and_scope_confidence_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            pd.DataFrame([
                {"left_record_id": "ro0l", "right_record_id": "ro0r", "relation": "related_variant_likely", "review_scope": "disagreement", "confidence": "high"},
                {"left_record_id": "to0l", "right_record_id": "to0r", "relation": "unrelated_likely", "review_scope": "disagreement", "confidence": "medium"},
                {"left_record_id": "bo0l", "right_record_id": "bo0r", "relation": "uncertain", "review_scope": "leakage", "confidence": "low"},
                {"left_record_id": "un0l", "right_record_id": "un0r", "relation": "new_unknown_label", "review_scope": "leakage", "confidence": "low"},
            ]).to_csv(diagnostic, index=False)
            pairs, metrics = diagnostic_comparison(diagnostic, baseline, baseline)
            self.assertEqual(pairs.relation.tolist(), ["related_variant_likely", "unrelated_likely", "uncertain", "new_unknown_label"])
            self.assertTrue(pairs.loc[2:, "baseline_rule_false_positive"].isna().all())
            overall = metrics.loc[metrics.review_scope.eq("all") & metrics.confidence.eq("all") & metrics.phase.eq("baseline")].set_index("method")
            self.assertEqual(overall.loc["rule", "related_pairs"], 1)
            self.assertEqual(overall.loc["rule", "unrelated_pairs"], 1)
            self.assertEqual(overall.loc["rule", "uncertain_or_unknown_pairs"], 2)
            self.assertEqual(overall.loc["rule", "false_negative_pairs"], 0)
            self.assertEqual(overall.loc["tfidf", "false_negative_pairs"], 1)
            self.assertEqual(overall.loc["tfidf", "false_positive_pairs"], 1)
            self.assertTrue(metrics.loc[metrics.review_scope.eq("leakage"), "false_positive_rate_on_reviewed_unrelated"].isna().all())
            self.assertIn("medium", set(metrics.confidence))

    def test_refinement_freeze_prevents_postreview_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            reserve_holdout(baseline, diagnostic, root / "holdout")
            for name in ("config.json", "input_manifest.json", "summary.json"):
                (baseline / name).write_text(json.dumps({"complete": True}), encoding="utf-8")
            code = root / "method.py"
            code.write_text("# frozen methods\n", encoding="utf-8")
            seal_refinement(baseline, root / "holdout", code_paths=[code])
            self.assertEqual(verify_refinement_seal(baseline)["status"], "frozen_before_heldout_product_review")
            self.assertIn("+00:00", verify_refinement_seal(baseline)["frozen_utc"])
            with self.assertRaises(FileExistsError):
                seal_refinement(baseline, root / "holdout")
            code.write_text("# changed after review\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen experimental artifact changed"):
                verify_refinement_seal(baseline)

    def test_pair_ids_missing_from_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, diagnostic = write_fixture(root)
            annotations = pd.read_csv(diagnostic)
            annotations.loc[0, "left_record_id"] = "absent"
            annotations.to_csv(diagnostic, index=False)
            with self.assertRaisesRegex(ValueError, "absent from baseline"):
                reserve_holdout(baseline, diagnostic, root / "holdout")


if __name__ == "__main__":
    unittest.main()
