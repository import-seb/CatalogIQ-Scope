"""Synthetic policy behavior, arithmetic, rare classes and phase protection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.product_review_rules import (
    SEARCHED_RULES, apply_rule, check_phase_isolation, choose_rule, diagnostic_tables,
    discover_rules, require_role, score_registry, summarize_policy, uncommon_class_table,
)
from catalogiq.product_rule_experiment import audit_product_rules, code_hashes, discover_product_rules, training_reference
from catalogiq.accuracy_reassessment import seal_output, verify_seal
from catalogiq.prediction_contract import Evidence, write_json


def frame(role="calibration_fit", n=1000, prefix="d"):
    return pd.DataFrame({"record_id": [f"{prefix}r{i}" for i in range(n)],
        "group_id": [f"{prefix}g{i}" for i in range(n)], "split": "validation", "experiment_role": role,
        "predicted_label": ["A"]*int(.8*n)+["B"]*(n-int(.8*n)),
        "true_label": ["A"]*int(.8*n)+["B"]*(n-int(.8*n)),
        "confidence": .98, "base_eligible": True, "description_present": True, "category_present": True,
        "brand_seen": True, "brand_familiarity": "seen", "brand_key_unambiguous": True,
        "brand_training_groups": 25, "predicted_class_training_rows": 1500, "true_class_training_rows": 1500})


def registry_fixture():
    discovery = frame()
    discovery.loc[800:809, "true_label"] = "A"
    return discover_rules(discovery, "segment", ["A", "B", "Absent"], .93)[0]


class RuleBehaviorTests(unittest.TestCase):
    def test_baseline_preserves_floor_equality_and_existing_gates(self):
        f = frame(n=5)
        rule = registry_fixture()[0]
        f["confidence"] = [.93, .92999, .99, np.nan, 1.01]
        f.loc[2, "base_eligible"] = False
        self.assertEqual(apply_rule(f, rule, expected_threshold=.93).tolist(), [True, False, False, False, False])

    def test_fixed_floor_cannot_be_changed_in_either_direction(self):
        for value in (.90, .95):
            rule = deepcopy(registry_fixture()[0])
            rule["threshold"] = value
            with self.assertRaisesRegex(ValueError, "fixed"):
                apply_rule(frame(), rule, expected_threshold=value)

    def test_true_labels_and_audit_answers_cannot_change_arrival_rule_decisions(self):
        f = frame(role="policy_audit")
        rule = next(r for r in registry_fixture() if r["id"] == "discovered_995__all")
        original = apply_rule(f, rule, expected_threshold=.93)
        changed = apply_rule(f.assign(true_label="wrong", true_class_training_rows=0), rule, expected_threshold=.93)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(original.sum(), 800)

    def test_truth_conditions_and_unknown_predicted_labels_are_refused(self):
        rule = deepcopy(registry_fixture()[0])
        rule["true_label"] = "A"
        with self.assertRaisesRegex(ValueError, "truth"):
            apply_rule(frame(), rule, expected_threshold=.93)
        rule.pop("true_label")
        self.assertFalse(apply_rule(frame().assign(predicted_label="Unsupported"), rule, expected_threshold=.93).any())

    def test_description_and_category_presence_are_additional_gates(self):
        f = frame(n=5)
        f.loc[0, "description_present"] = False
        f.loc[1, "category_present"] = False
        rule = next(r for r in registry_fixture() if r["id"] == "all__description_category")
        self.assertEqual(apply_rule(f, rule, expected_threshold=.93).tolist(), [False, False, True, True, True])

    def test_brand_familiarity_and_training_support_use_arrival_information(self):
        f = frame(n=5)
        f["brand_seen"] = [True, False, False, True, True]
        rule = next(r for r in registry_fixture() if r["id"] == "all__seen_brand")
        self.assertEqual(apply_rule(f, rule, expected_threshold=.93).sum(), 3)
        lookup = discover_rules(frame(), "brand_lookup_probabilistic_v1", ["A", "B"], .955)[0]
        gate = next(r for r in lookup if r["id"] == "all__unambiguous_key_support20")
        f["brand_training_groups"] = [19, 20, 21, 30, 0]
        f.loc[3, "brand_key_unambiguous"] = False
        self.assertEqual(apply_rule(f, gate, expected_threshold=.955).tolist(), [False, True, True, False, False])

    def test_common_class_gate_uses_predicted_class_training_count(self):
        f = frame(n=5)
        f["predicted_class_training_rows"] = [999, 1000, 2000, 0, 500]
        f["true_class_training_rows"] = 50000
        rule = next(r for r in registry_fixture() if r["id"] == "training_common__all")
        self.assertEqual(apply_rule(f, rule, expected_threshold=.93).tolist(), [False, True, True, False, False])

    def test_sparse_error_free_classes_are_not_admitted_by_class_screens(self):
        f = frame(n=30)
        registry, evidence = discover_rules(f, "segment", ["A", "B", "Absent"], .93)
        rule = next(r for r in registry if r["id"] == "discovered_error_free__all")
        self.assertEqual(rule["allowed_labels"], [])
        self.assertEqual(len(evidence), 3)
        self.assertFalse(evidence.per_class_99_percent_guarantee.any())
        self.assertEqual(len(registry), 30)
        self.assertEqual(SEARCHED_RULES, 60)


class CalculationAndSelectionTests(unittest.TestCase):
    def test_counts_coverage_review_errors_and_family_weights(self):
        f = frame(n=5)
        f["predicted_label"] = "A"
        f["true_label"] = ["A", "B", "A", "A", "A"]
        f["group_id"] = ["large", "large", "g2", "g3", "g4"]
        result = summarize_policy(f, [True, True, True, False, False])
        self.assertEqual(result["accepted_rows"], 3)
        self.assertEqual(result["review_rows"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertAlmostEqual(result["coverage"], .6)
        self.assertAlmostEqual(result["accepted_accuracy"], 2/3)
        self.assertEqual(result["accepted_groups"], 2)
        self.assertLess(result["n_eff_df"], 9/5)

    def test_no_acceptance_is_not_reported_as_perfect_accuracy(self):
        result = summarize_policy(frame(n=5), np.zeros(5, bool))
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["review_fraction"], 1)
        self.assertIsNone(result["accepted_accuracy"])
        self.assertIsNone(result["beta_lower"])
        self.assertFalse(result["sufficient_groups"])

    def test_selection_corrects_search_and_requires_sufficient_families(self):
        f = frame(role="threshold_selection", prefix="s")
        f.loc[800:809, "true_label"] = "A"
        scores = score_registry(f, registry_fixture())
        self.assertTrue((scores.beta_lower_search_adjusted.dropna() <= scores.beta_lower.dropna()+1e-12).all())
        selected = choose_rule(scores)
        self.assertEqual(selected["status"], "search_adjusted_selection_support")
        self.assertEqual(selected["selection_summary"]["accepted_rows"], 800)
        self.assertFalse(selected["enabled"])
        small = score_registry(f.iloc[:100].copy(), registry_fixture())
        self.assertIsNone(choose_rule(small)["rule_id"])

    def test_diagnostics_mark_true_class_information_unavailable_at_arrival(self):
        f = frame(n=50)
        diagnostics = diagnostic_tables(f, registry_fixture()[0], role="calibration_fit")
        self.assertFalse(diagnostics.loc[diagnostics.characteristic.isin(["true_class", "true_class_frequency"]), "available_at_arrival"].any())
        self.assertTrue(diagnostics.loc[diagnostics.characteristic.eq("predicted_class"), "available_at_arrival"].all())
        self.assertTrue(diagnostics.descriptive_not_confirmatory.all())

    def test_uncommon_class_table_keeps_absent_vocabulary_and_reconciles_errors(self):
        f = frame(n=50)
        f.loc[0, "true_label"] = "B"
        table = uncommon_class_table(f, np.ones(50, bool), ["A", "B", "Absent"],
                                    policy_name="test", training_counts={"A": 10000, "B": 100, "Absent": 1})
        self.assertEqual(table.incorrect_accepted_predictions.sum(), 1)
        self.assertTrue(table.set_index("class").loc["Absent", "uncommon_true_class"])
        self.assertEqual(table.set_index("class").loc["Absent", "truth_rows"], 0)
        self.assertFalse(table.per_class_99_percent_confirmed.any())


class LeakageAndEvidenceTests(unittest.TestCase):
    def test_audit_rows_cannot_discover_or_select_rules(self):
        for role in ("policy_audit", "train", "test"):
            f = frame(role=role)
            with self.assertRaises(ValueError):
                discover_rules(f, "segment", ["A", "B"], .93)
            with self.assertRaises(ValueError):
                score_registry(f, registry_fixture())
        with self.assertRaises(ValueError):
            discover_rules(frame().assign(split="test"), "segment", ["A", "B"], .93)

    def test_discovery_selection_must_not_share_families_or_records(self):
        discovery = frame(prefix="d")
        selection = frame(role="threshold_selection", prefix="s")
        check_phase_isolation(discovery, selection)
        selection.loc[0, "group_id"] = discovery.group_id.iloc[0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            check_phase_isolation(discovery, selection)

    def test_training_reference_refuses_validation_rows(self):
        bad = pd.DataFrame({"record_id": ["r"], "group_id": ["g"], "split": ["validation"], "Segment": ["A"]})
        with patch.object(Evidence, "csv", return_value=bad):
            with self.assertRaisesRegex(ValueError, "training"):
                training_reference(Path.cwd(), "segment", ["A"], pd.DataFrame(), Evidence())

    def test_existing_directories_and_source_overlap_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"existing").mkdir()
            with self.assertRaises(FileExistsError):
                discover_product_rules(root, root/"existing")
            old = root/"data/processed/confidence_review_v3_20261103"
            with self.assertRaisesRegex(ValueError, "separate"):
                discover_product_rules(root, old/"new")

    def test_audit_requires_sealed_discovery_before_loading_real_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch("catalogiq.product_rule_experiment.load_inputs") as loader:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                audit_product_rules(root, root/"missing", root/"audit")
            loader.assert_not_called()
            self.assertFalse((root/"audit").exists())

    def test_bad_audit_cannot_reselect_a_rule_or_enable_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovery = root/"discovery"
            discovery.mkdir()
            write_json(discovery/"protocol.json", {"code_sha256": code_hashes()})
            choices, frames, references, originals = {}, {}, {}, {}
            for model, floor in (("segment", .93), ("brand_lookup_probabilistic_v1", .955)):
                registry = discover_rules(frame(), model, ["A", "B"], floor)[0]
                chosen = deepcopy(registry[0])
                chosen["allowed_labels"] = ["A"]
                chosen["id"] = "frozen_before_bad_audit"
                choices[model] = {"rule_id": chosen["id"], "rule": chosen, "status": "experimental", "enabled": False}
                (discovery/model).mkdir()
                write_json(discovery/model/"registry.json", registry)
                frames[model] = frame(role="policy_audit").assign(true_label="B")
                references[model] = {"class_training_rows": {"A": 1500, "B": 1500}}
                originals[model] = {"audit": {"accepted_rows": 1000, "accepted_errors": 800}}
            write_json(discovery/"selected_rules.json", choices)
            seal_output(discovery)
            with patch("catalogiq.product_rule_experiment.load_inputs", return_value=(frames, references, originals, {})), \
                    patch("catalogiq.product_rule_experiment.choose_rule", side_effect=AssertionError("No reselection")), \
                    patch("catalogiq.product_rule_plots.plot_audit"):
                destination = audit_product_rules(root, discovery, root/"audit")
            verify_seal(destination, Evidence())
            results = pd.read_csv(destination/"audit_policy_comparison.csv")
            self.assertFalse(results.independently_confirmed.any())
            self.assertFalse(results.automatic_acceptance_enabled.any())
            selected = results.loc[results.policy.eq("selected_experimental_rule")]
            self.assertEqual(set(selected.rule_id), {"frozen_before_bad_audit"})
            self.assertTrue(selected.errors.eq(800).all())


if __name__ == "__main__":
    unittest.main()
