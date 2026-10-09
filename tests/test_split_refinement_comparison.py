"""Fair audit-pool comparison and held-out freeze requirements."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest

import pandas as pd

from catalogiq.split_refinement_comparison import compare_refinement, main, validate_runs
from catalogiq.split_review_validation import reserve_holdout, seal_refinement


def write_runs(root):
    baseline, refined = root / "baseline", root / "refined"
    baseline.mkdir()
    refined.mkdir()
    record_ids = list("abcde")
    groupings = {
        "baseline": {"rule": list("aacdd"), "tfidf": record_ids},
        "refined": {"rule": list("abbdd"), "tfidf": list("aacde")},
    }
    configuration = {"seed": 42, "ratios": [0.7, 0.15, 0.15], "evaluation_threshold": 0.85,
                     "evaluation_shingle_size": 5, "evaluation_min_name_chars": 12}
    for directory in (baseline, refined):
        config = {**configuration, "grouping_version": 1 if directory == baseline else 2}
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "input_manifest.json").write_text(json.dumps({"rows": 5, "sha256": "identical-source", "target": "Segment"}), encoding="utf-8")
        (directory / "summary.json").write_text("{}", encoding="utf-8")
        groups = groupings[directory.name]
        for method in ("rule", "tfidf"):
            pd.DataFrame({"record_id": record_ids, "group_id": groups[method], "split": ["train"] * 5}).to_csv(directory / f"{method}_assignments.csv", index=False)
        pd.DataFrame({"metric": ["groups", "near duplicate crossing pairs"],
                      "rule": [len(set(groups["rule"])), 0], "tfidf": [len(set(groups["tfidf"])), 0]}).to_csv(directory / "comparison.csv", index=False)
        pd.DataFrame({"left_record_id": ["a", "d"], "right_record_id": ["b", "e"],
                      "name_jaccard": ["0.95", "0.9"]}).to_csv(directory / "independent_near_duplicate_pairs.csv", index=False)
    pd.DataFrame({"left_record_id": ["a", "d"], "right_record_id": ["b", "e"]}).to_csv(baseline / "rule_matching_edges.csv", index=False)
    pd.DataFrame(columns=["left_record_id", "right_record_id"]).to_csv(baseline / "tfidf_matching_edges.csv", index=False)
    diagnostic = root / "diagnostic.csv"
    pd.DataFrame([
        {"left_record_id": "a", "right_record_id": "b", "relation": "same_product_likely", "confidence": "high", "review_scope": "diagnostic"},
        {"left_record_id": "b", "right_record_id": "c", "relation": "unrelated_likely", "confidence": "medium", "review_scope": "diagnostic"},
    ]).to_csv(diagnostic, index=False)
    return baseline, refined, diagnostic


class RefinementComparisonTests(unittest.TestCase):
    def test_command_help_describes_inputs_and_heldout_option(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--heldout-annotations", output.getvalue())
        self.assertIn("--diagnostic", output.getvalue())

    def test_command_runs_completed_version_one_and_two_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, diagnostic = write_runs(root)
            output = io.StringIO()
            with redirect_stdout(output):
                main(["--baseline", str(baseline), "--refined", str(refined), "--diagnostic", str(diagnostic),
                      "--output", str(root / "comparison")])
            self.assertTrue((root / "comparison/verification.json").is_file())
            self.assertIn("baseline_rule", output.getvalue())
            self.assertIn("false_negative_pairs", output.getvalue())

    def test_aggregate_and_weak_diagnostic_outputs_reproduce(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, diagnostic = write_runs(root)
            reviewed = pd.read_csv(diagnostic)
            reviewed["left_ProductName"] = ["Reviewed product A", "Reviewed product B"]
            reviewed["right_ProductName"] = ["Reviewed product B", "Reviewed product C"]
            reviewed.to_csv(diagnostic, index=False)
            aggregate, metrics, heldout = compare_refinement(baseline, refined, diagnostic, root / "first")
            compare_refinement(baseline, refined, diagnostic, root / "second")
            for name in ("aggregate_comparison.csv", "diagnostic_pair_outcomes.csv", "diagnostic_metrics.csv", "verification.json"):
                self.assertEqual((root / "first" / name).read_bytes(), (root / "second" / name).read_bytes())
            self.assertIsNone(heldout)
            pair_outputs = pd.read_csv(root / "first/diagnostic_pair_outcomes.csv")
            self.assertEqual(pair_outputs.left_ProductName.tolist(), reviewed.left_ProductName.tolist())
            self.assertEqual(aggregate.set_index("metric").loc["groups", "delta_tfidf"], -1)
            overall = metrics.loc[metrics.review_scope.eq("all") & metrics.confidence.eq("all") & metrics.phase.eq("refined")].set_index("method")
            self.assertEqual(overall.loc["rule", "false_negative_pairs"], 1)
            self.assertEqual(overall.loc["rule", "false_positive_pairs"], 1)
            with self.assertRaises(FileExistsError):
                compare_refinement(baseline, refined, diagnostic, root / "first")

    def test_audit_pool_allows_order_and_orientation_but_requires_exact_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, _ = write_runs(root)
            path = refined / "independent_near_duplicate_pairs.csv"
            frame = pd.read_csv(path).iloc[::-1]
            frame[["left_record_id", "right_record_id"]] = frame[["right_record_id", "left_record_id"]].to_numpy()
            frame.to_csv(path, index=False)
            self.assertEqual(validate_runs(baseline, refined)["independent_pairs"], 2)
            frame.loc[0, "name_jaccard"] = 0.95000000001
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exact Jaccard scores differ"):
                validate_runs(baseline, refined)

    def test_changed_input_manifest_records_or_evaluator_rejected(self):
        for mutation in ("manifest", "ids", "evaluator", "audit_ids"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                baseline, refined, _ = write_runs(root)
                if mutation == "manifest":
                    (refined / "input_manifest.json").write_text('{"rows": 5, "sha256": "other-source"}', encoding="utf-8")
                elif mutation == "ids":
                    path = refined / "rule_assignments.csv"
                    frame = pd.read_csv(path)
                    frame.loc[0, "record_id"] = "other-record"
                    frame.to_csv(path, index=False)
                elif mutation == "evaluator":
                    path = refined / "config.json"
                    config = json.loads(path.read_text())
                    config["evaluation_threshold"] = 0.80
                    path.write_text(json.dumps(config), encoding="utf-8")
                else:
                    path = refined / "independent_near_duplicate_pairs.csv"
                    frame = pd.read_csv(path)
                    frame.loc[0, "right_record_id"] = "c"
                    frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    validate_runs(baseline, refined)

    def test_heldout_requires_freeze_and_exact_sample_ids_saved_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, diagnostic = write_runs(root)
            holdout = root / "holdout"
            reserve_holdout(baseline, diagnostic, holdout)
            with self.assertRaises(FileNotFoundError):
                compare_refinement(baseline, refined, diagnostic, root / "not_frozen", holdout_dir=holdout)
            self.assertFalse((root / "not_frozen").exists())
            seal_refinement(refined, holdout)
            annotations = root / "heldout_annotations.csv"
            frame = pd.DataFrame([{"left_record_id": "d", "right_record_id": "e", "relation": "related_variant_likely",
                                   "confidence": "medium", "review_scope": "heldout_rule_only_grouped"}])
            frame.to_csv(annotations, index=False)
            _, _, metrics = compare_refinement(baseline, refined, diagnostic, root / "complete", holdout, annotations)
            self.assertIsNotNone(metrics)
            self.assertTrue((root / "complete/heldout_metrics.csv").is_file())
            self.assertTrue((root / "complete/diagnostic_metrics.csv").is_file())
            frame.loc[0, "right_record_id"] = "c"
            frame.to_csv(annotations, index=False)
            with self.assertRaisesRegex(ValueError, "exactly match the sealed review sample"):
                compare_refinement(baseline, refined, diagnostic, root / "bad_sample", holdout, annotations)
            self.assertFalse((root / "bad_sample").exists())

    def test_duplicate_audit_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, refined, _ = write_runs(root)
            path = refined / "independent_near_duplicate_pairs.csv"
            frame = pd.read_csv(path)
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate independent audit pair"):
                validate_runs(baseline, refined)


if __name__ == "__main__":
    unittest.main()
