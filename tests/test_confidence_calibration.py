"""Synthetic calculations, leakage guards and fail-closed review policy tests."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.confidence_calibration import (
    ROLES, accepted_summary, bootstrap_accuracy, calibration_metrics, cluster_accuracy_lower,
    essential_information_reasons, fit_temperature, group_role, partition_context,
    per_class_table, reliability, review_decision, temperature_scale, threshold_grid,
    threshold_table, validate_probabilities,
)
from catalogiq.confidence_experiment import (
    SUMMARY_THRESHOLDS, _summarize_saved_run, choose_policy, evaluate_model,
    load_run, model_gates, run_experiment, summarize_saved,
)
from catalogiq.prediction_contract import Evidence, REGISTRY_NAME, contract_metadata, write_json
from catalogiq.features import sha256
from catalogiq.maria_harness_compatibility import REVIEW_REFERENCE, review_namespace


def synthetic_frame():
    groups = []
    for role in ROLES:
        values = [f"g{i}" for i in range(1000) if group_role(f"g{i}") == role][:12]
        groups.extend(values)
    frame = pd.DataFrame({"record_id": [f"r{i}" for i in range(len(groups))],
                          "group_id": groups, "split": "validation", "true_label": "A",
                          "predicted_label": "A"})
    return frame


class CalibrationCalculationTests(unittest.TestCase):
    def test_metrics_match_hand_calculation_and_preserve_oov_errors(self):
        metric, bins = calibration_metrics([[.8, .2], [.3, .7]], ["A", "A"], ["A", "B"])
        self.assertAlmostEqual(metric["accuracy"], .5)
        self.assertAlmostEqual(metric["ece_10_bins"], .45)
        self.assertAlmostEqual(metric["brier_multiclass_sum"], (.08+.98)/2)
        self.assertAlmostEqual(metric["nll"], -(np.log(.8)+np.log(.3))/2)
        self.assertEqual(sum(r["rows"] for r in bins), 2)
        oov, _ = calibration_metrics([[.8, .2]], ["unknown"], ["A", "B"])
        self.assertEqual(oov["accuracy"], 0)
        self.assertAlmostEqual(oov["brier_multiclass_sum"], 1.68)
        self.assertIsNone(oov["nll"])
        self.assertTrue(oov["nll_is_infinite"])
        self.assertEqual(oov["oov_truth_rows"], 1)
        self.assertEqual(fit_temperature([[.8, .2]], ["unknown"], ["A", "B"])["status"],
                         "unidentifiable_infinite_nll")

    def test_reliability_matches_maria_pinned_definition(self):
        path = REVIEW_REFERENCE
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ece_and_reliability")
        namespace = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        confidence = np.array([0., .1, .2, .5, .9, 1.])
        correct = np.array([0, 0, 1, 1, 0, 1])
        _, expected = namespace["ece_and_reliability"](confidence, correct)
        bins, actual = reliability(confidence, correct)
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(sum(b["rows"] for b in bins), len(confidence))

    def test_temperature_fit_can_soften_or_sharpen(self):
        over = np.tile([.99, .01], (100, 1))
        truth = ["A"]*80 + ["B"]*20
        result = fit_temperature(over, truth, ["A", "B"])
        self.assertGreater(result["temperature"], 1)
        self.assertLess(result["fit_nll_after"], result["fit_nll_before"])
        under = fit_temperature(np.tile([.6, .4], (100, 1)), ["A"]*98+["B"]*2, ["A", "B"])
        self.assertLess(under["temperature"], 1)
        q = temperature_scale(over, result["temperature"])
        np.testing.assert_array_equal(q.argmax(axis=1), over.argmax(axis=1))
        np.testing.assert_allclose(q.sum(axis=1), 1)

    def test_zero_probabilities_ties_and_extreme_values(self):
        p = np.array([[1., 0., 0.], [.5, .5, 0.], [1e-100, .8, .2]])
        for t in (.05, 1, 20):
            q = temperature_scale(p, t)
            np.testing.assert_array_equal(q.argmax(axis=1), p.argmax(axis=1))
            self.assertTrue(np.isfinite(q).all())
        metric, _ = calibration_metrics([[1., 0.]], ["B"], ["A", "B"])
        self.assertTrue(metric["nll_is_infinite"])
        self.assertEqual(metric["brier_multiclass_sum"], 2)

    def test_invalid_probabilities_and_temperature_refused(self):
        for p in ([[.5, .6]], [[np.nan, 0]], [[-.1, 1.1]], [[1.]], [[.2, np.inf]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                validate_probabilities(p)
        for t in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                temperature_scale([[.5, .5]], t)

    def test_threshold_grid_endpoints_precision_and_inclusive_boundary(self):
        grid = threshold_grid()
        self.assertEqual(grid.min(), .5)
        self.assertEqual(grid.max(), .999)
        self.assertIn(.9951, grid)
        table = threshold_table([.5, .99, .999], [1, 0, 1], ["a", "b", "c"],
                                thresholds=[.5, .99, .999, 1.])
        self.assertEqual(table.accepted_rows.tolist(), [3, 2, 1, 0])
        self.assertEqual(table.accepted_errors.tolist(), [1, 1, 0, 0])
        self.assertTrue(pd.isna(table.iloc[-1].accepted_accuracy))
        self.assertTrue(pd.isna(table.iloc[-1].cluster_accuracy_lower))
        self.assertEqual(table.iloc[-1].review_fraction, 1)

    def test_cluster_bound_is_conservative_for_duplicates_and_search(self):
        single = cluster_accuracy_lower(0, 1000, 1000, 1)
        duplicate = cluster_accuracy_lower(0, 1000, 100, 10)
        self.assertAlmostEqual(single, .05**(1/1000))
        self.assertLess(duplicate, single)
        self.assertLess(cluster_accuracy_lower(0, 1000, 1000, 1, .05/1000), single)
        self.assertLess(cluster_accuracy_lower(2, 1000, 1000, 1), single)
        self.assertIsNone(cluster_accuracy_lower(0, 0, 0, 0))
        self.assertEqual(cluster_accuracy_lower(1000, 1000, 1000, 1), 0)
        with self.assertRaises(ValueError):
            cluster_accuracy_lower(2, 1, 1, 1)

    def test_bootstrap_resamples_groups_and_cannot_authorize_zero_error(self):
        result = bootstrap_accuracy([True]*4, [True]*4, ["g", "g", "h", "h"], samples=100)
        self.assertTrue(result["degenerate_zero_errors"])
        bound = accepted_summary([True]*4, [True]*4, ["g", "g", "h", "h"])
        self.assertLess(bound["cluster_accuracy_lower"], .99)
        again = bootstrap_accuracy([True]*4, [True, False, True, True], ["g", "g", "h", "h"], samples=100)
        self.assertLess(again["lower"], 1)

    def test_per_class_includes_zero_support_and_minority_errors(self):
        table = per_class_table(["A", "B", "OOV"], ["A", "A", "B"], [True, True, True], ["A", "B", "C"])
        indexed = table.set_index("class")
        self.assertEqual(indexed.loc["C", "truth_rows"], 0)
        self.assertTrue(pd.isna(indexed.loc["C", "truth_class_coverage"]))
        self.assertEqual(indexed.loc["A", "incorrect_accepted_predictions"], 1)
        self.assertEqual(indexed.loc["OOV", "missed_true_class_errors"], 1)
        self.assertEqual(indexed.incorrect_accepted_predictions.sum(), 2)


class LeakageTests(unittest.TestCase):
    def make_run(self, directory):
        from test_prediction_contract import fixture
        manifest, predictions, context, protected = fixture()
        manifest = contract_metadata(manifest, predictions, manifest["label_names"])
        identity = {"name": "fixture-frozen-v4", "completion_sha256": "fixture-seal"}
        manifest["split_allocation_identity"] = identity
        directory.mkdir()
        predictions.to_csv(directory/"predictions.csv", index=False)
        context.to_csv(directory/"validation_context.csv", index=False)
        protected.to_csv(directory/REGISTRY_NAME, index=False)
        write_json(directory/"label_mapping.json", manifest["label_mapping"])
        manifest["files_sha256"] = {p.name: sha256(p) for p in directory.iterdir()}
        write_json(directory/"manifest.json", manifest)
        return manifest, context, protected, identity

    def test_real_loader_validates_seals_approved_context_and_extra_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"synthetic"
            manifest, context, protected, identity = self.make_run(directory)
            args = (directory, Evidence(), {"Segment": context}, {"Segment": identity}, protected, {})
            result = load_run(*args, expected_manifest_sha256=sha256(directory/"manifest.json"))
            self.assertEqual(len(result[1]), 2)
            with self.assertRaisesRegex(ValueError, "supplied SHA"):
                load_run(*args, expected_manifest_sha256="wrong")
            wrong = context.copy()
            wrong.loc[0, "true_label"] = "wrong"
            with self.assertRaisesRegex(ValueError, "approved population"):
                load_run(directory, Evidence(), {"Segment": wrong}, {"Segment": identity}, protected, {})
            predictions = pd.read_csv(directory/"predictions.csv")
            predictions["group_id"] = "wrong"
            predictions.to_csv(directory/"predictions.csv", index=False)
            manifest["files_sha256"]["predictions.csv"] = sha256(directory/"predictions.csv")
            write_json(directory/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "group_id"):
                load_run(directory, Evidence(), {"Segment": context}, {"Segment": identity}, protected, {})

    def test_loader_rejects_resealed_protected_registry_and_wrong_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"synthetic"
            manifest, context, protected, identity = self.make_run(directory)
            with self.assertRaisesRegex(ValueError, "allocation identity"):
                load_run(directory, Evidence(), {"Segment": context}, {"Segment": {}}, protected, {})
            protected.assign(group_id="tampered").to_csv(directory/REGISTRY_NAME, index=False)
            manifest["files_sha256"][REGISTRY_NAME] = sha256(directory/REGISTRY_NAME)
            write_json(directory/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "original V4"):
                load_run(directory, Evidence(), {"Segment": context}, {"Segment": identity}, protected, {})

    def test_measured_essential_gate_never_uses_target_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = synthetic_frame()
            features = frame.set_index("record_id").drop(columns=["true_label", "predicted_label", "split"])
            features["ProductName"] = "Some product"
            features.loc[frame.record_id.iloc[0], "ProductName"] = " null "
            eligible, reasons, _ = model_gates(Path(tmp), frame, features, Evidence())
            self.assertFalse(eligible[0])
            self.assertTrue(eligible[1:].all())
            changed, _, _ = model_gates(Path(tmp), frame.assign(true_label="unknown"), features, Evidence())
            np.testing.assert_array_equal(eligible, changed)

    def test_partition_is_label_independent_stable_and_group_disjoint(self):
        frame = synthetic_frame()
        frame = pd.concat([frame, frame.iloc[[0]].assign(record_id="duplicate-group")], ignore_index=True)
        a = partition_context(frame)
        b = partition_context(frame.sample(frac=1, random_state=1).assign(true_label="different"))
        self.assertEqual(a.set_index("record_id").experiment_role.to_dict(),
                         b.set_index("record_id").experiment_role.to_dict())
        self.assertEqual(a.groupby("group_id").experiment_role.nunique().max(), 1)
        for role in ROLES:
            for other in set(ROLES)-{role}:
                self.assertFalse(set(a.loc[a.experiment_role.eq(role), "group_id"]) &
                                 set(a.loc[a.experiment_role.eq(other), "group_id"]))
        with self.assertRaises(ValueError):
            partition_context(frame.assign(split="test"))

    def test_audit_labels_cannot_change_fitted_temperature_or_selected_policy(self):
        frame = synthetic_frame()
        probabilities = np.tile([.9, .1], (len(frame), 1))
        changed = frame.copy()
        audit = frame.group_id.map(group_role).eq("policy_audit")
        changed.loc[audit, "true_label"] = "B"
        with tempfile.TemporaryDirectory() as tmp:
            with patch("catalogiq.confidence_experiment.threshold_grid", return_value=np.array([.5, .9, .999])):
                first, _ = evaluate_model(frame, probabilities, ["A", "B"], np.ones(len(frame), bool),
                    np.repeat("neural", len(frame)), Path(tmp)/"first", bootstrap_samples=100)
                second, _ = evaluate_model(changed, probabilities, ["A", "B"], np.ones(len(frame), bool),
                    np.repeat("neural", len(frame)), Path(tmp)/"second", bootstrap_samples=100)
            self.assertEqual(json.loads((Path(tmp)/"first/temperature.json").read_text()),
                             json.loads((Path(tmp)/"second/temperature.json").read_text()))
            self.assertEqual(first["selected_candidate"], second["selected_candidate"])
            self.assertNotEqual(first["audit"]["accepted_errors"], second["audit"]["accepted_errors"])
            self.assertFalse(second["enabled"])
            self.assertFalse(second["development_supports_99_percent"])

    def test_selection_prefers_supported_coverage_never_observed_only_claim(self):
        table = pd.DataFrame([
            {"variant": "raw", "threshold": .5, "coverage": 1., "cluster_accuracy_lower": .98,
             "accepted_accuracy": .999, "accepted_rows": 100, "selection_nll": .1},
            {"variant": "temperature", "threshold": .95, "coverage": .8, "cluster_accuracy_lower": .991,
             "accepted_accuracy": 1., "accepted_rows": 80, "selection_nll": .2}])
        chosen, status = choose_policy(table)
        self.assertEqual(chosen, {"variant": "temperature", "threshold": .95})
        self.assertEqual(status, "selection_supported")
        chosen, status = choose_policy(table.iloc[:1])
        self.assertEqual(status, "exploratory_only")

    def test_existing_output_refused_before_source_reads(self):
        with tempfile.TemporaryDirectory() as tmp, patch("catalogiq.confidence_experiment.frozen_identity_context") as loader:
            with self.assertRaises(FileExistsError):
                run_experiment(tmp, Path(tmp)/"missing", tmp)
            loader.assert_not_called()

    def test_test_csv_is_never_opened_or_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.csv"
            path.write_text("DO NOT READ")
            with patch("catalogiq.prediction_contract.sha256", side_effect=AssertionError("must not hash")):
                with self.assertRaises(PermissionError):
                    Evidence().capture(path)

    def test_input_hash_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"validation.csv"
            path.write_text("original")
            evidence = Evidence()
            evidence.capture(path)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                evidence.reverify()


class ReviewRuleTests(unittest.TestCase):
    def decision(self, **changes):
        kwargs = dict(probabilities=[.999, .001], predicted_label="A", labels=["A", "B"],
                      threshold=.99, product={"ProductName": "Product"}, enabled=True)
        kwargs.update(changes)
        return review_decision(**kwargs)

    def test_high_confidence_accepts_only_when_enabled_and_valid(self):
        self.assertTrue(self.decision()["accept"])
        self.assertFalse(self.decision(enabled=False)["accept"])
        self.assertFalse(self.decision(provisional=True)["accept"])
        self.assertFalse(self.decision(threshold=None)["accept"])
        self.assertFalse(self.decision(quarantined=True)["accept"])

    def test_confidence_boundary_and_invalid_probabilities(self):
        self.assertTrue(self.decision(probabilities=[.99, .01])["accept"])
        self.assertIn("below_confidence_threshold", self.decision(probabilities=[.98, .02])["review_reasons"])
        for p in ([np.nan, 0], [-.1, 1.1], [.8, .8], [1.]):
            self.assertIn("invalid_probabilities", self.decision(probabilities=p)["review_reasons"])
        self.assertIn("unsupported_classification", self.decision(predicted_label="C")["review_reasons"])
        self.assertIn("conflicting_probability_classification", self.decision(predicted_label="B")["review_reasons"])

    def test_essential_information_uses_approved_missing_semantics(self):
        self.assertFalse(self.decision(product={"ProductBrand": "Present", "ProductCategory": "Present"})["accept"])
        self.assertFalse(self.decision(product={"ProductName": " null "})["accept"])
        for value in (None, np.nan, pd.NA):
            self.assertFalse(self.decision(product={"ProductName": value})["accept"])
        for key in ("ProductName", "ProductDescription", "ProductContents"):
            self.assertTrue(self.decision(product={key: "Some text"})["accept"])
        for threshold in ("bad", True, np.inf, -1, 1.01):
            self.assertIn("no_approved_threshold", self.decision(threshold=threshold)["review_reasons"])

    def test_lookup_fallback_must_be_reviewed_after_sharpening(self):
        for status in ("missing", "unseen"):
            self.assertFalse(self.decision(lookup_status=status)["accept"])
        self.assertTrue(self.decision(lookup_status="seen")["accept"])

    def test_only_explicit_approved_relationships_are_enforced(self):
        relation = {"status": "approved", "evidence": "synthetic test only",
                    "child_target": "Child", "parent_target": "Parent", "allowed_pairs": [["a", "p"], ["a", "q"]]}
        self.assertTrue(self.decision(approved_relations=[relation], classifications={"Child": "a", "Parent": "q"})["accept"])
        self.assertIn("invalid_approved_classification_relationship", self.decision(approved_relations=[relation],
                      classifications={"Child": "a", "Parent": "bad"})["review_reasons"])
        self.assertIn("missing_relationship_classification", self.decision(approved_relations=[relation])["review_reasons"])
        relation["status"] = "hypothesis"
        self.assertIn("unapproved_relationship_rule", self.decision(approved_relations=[relation])["review_reasons"])
        self.assertIn("invalid_approved_relationship_rule", self.decision(approved_relations=[
            {"status": "approved", "evidence": "synthetic test"}])["review_reasons"])


class SavedSummaryTests(unittest.TestCase):
    def inputs(self):
        frame = synthetic_frame()
        matrix = np.tile([[.99, .01], [.95, .05], [.9, .1], [.2, .8]], (len(frame)//4, 1))
        frame["predicted_label"] = np.asarray(["A", "B"])[matrix.argmax(axis=1)]
        frame["probability_0"], frame["probability_1"] = matrix.T
        assignments = frame[["record_id", "group_id", "split"]].copy()
        assignments["experiment_role"] = frame.group_id.map(group_role)
        assignments["eligible_measured_gates"] = "True"
        assignments["cohort"] = "neural"
        metrics, tables = [], []
        for variant in ("raw", "temperature"):
            for role in ROLES:
                mask = assignments.experiment_role.eq(role).to_numpy()
                score, _ = calibration_metrics(matrix[mask], frame.true_label[mask], ["A", "B"])
                metrics.append({"variant": variant, "subset": role, **score})
                if role != "calibration_fit":
                    table = threshold_table(matrix[mask].max(axis=1),
                        frame.true_label[mask].eq(frame.predicted_label[mask]), frame.group_id[mask],
                        thresholds=SUMMARY_THRESHOLDS)
                    table["variant"], table["subset"] = variant, role
                    tables.append(table)
        return dict(name="segment", frame=frame, labels=["A", "B"],
                    columns=["probability_0", "probability_1"],
                    eligible=np.ones(len(frame), bool), cohorts=np.repeat("neural", len(frame)),
                    assignments=assignments, temperature=1.,
                    policy={"enabled": False, "operational_threshold": None,
                            "selected_candidate": {"variant": "temperature"}},
                    existing=pd.concat(tables, ignore_index=True), saved_cal=pd.DataFrame(metrics),
                    reference=review_namespace(), bootstrap_samples=100, seed=20261106)

    def test_saved_replay_retains_counts_without_fitting_and_handles_empty_denominators(self):
        inputs = self.inputs()
        inputs["reference"]["temperature_scale"] = lambda *a: self.fail("Must not fit Maria temperature")
        with patch("catalogiq.confidence_experiment.fit_temperature", side_effect=AssertionError("No fitting")):
            shortlist, distributions, practical, coverage, calibration_checks, threshold_checks = _summarize_saved_run(**inputs)
        self.assertEqual((len(shortlist), len(distributions), len(practical)), (44, 6, 4))
        self.assertEqual((calibration_checks, threshold_checks), (6, 8))
        first = next(r for r in practical if r["threshold"] == .8)
        self.assertEqual(first["review_rows"], 0)
        self.assertIsNone(first["queue_flag_precision_percent"])
        self.assertGreater(first["accepted_errors"], 0)
        self.assertEqual(coverage.iloc[0].answered, first["rows"])

    def test_saved_replay_rejects_changed_assignments_results_and_enabled_policy(self):
        for field in ("assignments", "existing", "saved_cal", "policy"):
            inputs = self.inputs()
            if field == "assignments":
                inputs[field].loc[0, "group_id"] = "tampered"
            elif field == "existing":
                inputs[field].loc[:, "accepted_rows"] += 1
            elif field == "saved_cal":
                inputs[field].loc[:, "nll"] += .1
            else:
                inputs[field]["enabled"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                _summarize_saved_run(**inputs)

    def test_reference_functions_are_pinned_and_top_level_code_is_never_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"reference.py"
            source = REVIEW_REFERENCE.read_text(encoding="utf8")
            path.write_text("raise AssertionError('Do not execute module setup')\n" + source, encoding="utf8")
            self.assertIn("queue_metrics", review_namespace(path))
            path.write_text(source.replace("def queue_metrics(", "def changed_metrics("), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                review_namespace(path)

    def test_saved_summary_refuses_overwrites_and_source_overlap_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp, patch("catalogiq.confidence_experiment.frozen_identity_context") as loader:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                summarize_saved(root, root/"source", root/"previous", root)
            with self.assertRaisesRegex(ValueError, "separate"):
                summarize_saved(root, root/"source", root/"previous", root/"source"/"new")
            loader.assert_not_called()

    def test_summary_cli_reuses_entrypoint_and_requires_a_fresh_explicit_output(self):
        from catalogiq.cli import confidence_review_main
        with patch("catalogiq.confidence_experiment.summarize_saved", return_value={"runs":6, "output_root":"new"}) as summary:
            self.assertEqual(confidence_review_main(["--summarize-saved", "old", "--output-root", "new"]), 0)
            self.assertEqual(summary.call_args.args[2:], (Path("old"), Path("new")))
            with self.assertRaises(SystemExit):
                confidence_review_main(["--summarize-saved", "old"])


if __name__ == "__main__":
    unittest.main()
