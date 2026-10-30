"""CPU-only frequency-estimation and actual harness contracts; no neural runs."""
from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.brand_probabilistic_lookup import (
    ALPHA, SMOOTHING, calibration_compatibility, evaluate_frequency_lookup,
    export_probabilistic_lookup, fit_frequency_lookup, predict_frequency_lookup,
)
from catalogiq.features import sha256
from catalogiq.minilm_prediction_export import ExportInputs, write_json
from catalogiq.segment_transformer import ModelConfig, _hash_items
from catalogiq.target_minilm_experiment import brand_key, classification_metrics


FIXTURES = Path(__file__).parent / "fixtures"
LOADER = FIXTURES / "maria_harness_loader_05e52e2.py"
CALIBRATION = FIXTURES / "maria_harness_calibration_05e52e2.py"
MODULE = "catalogiq.brand_probabilistic_lookup"
LABELS = ("Rare", "A", "UnItemised brand", "B")


def populations():
    training = pd.DataFrame({"record_id": [f"t{i}" for i in range(9)], "group_id": [f"gt{i}" for i in range(9)],
        "split": ["train"] * 9, "ProductBrand": ["tie", "tie", "b", "b", "r", "", "other", "other", "other"],
        "Brand": ["A", "B", "B", "B", "Rare", "UnItemised brand", "UnItemised brand", "UnItemised brand", "UnItemised brand"]})
    validation = pd.DataFrame({"record_id": [f"v{i}" for i in range(6)], "group_id": [f"gv{i}" for i in range(6)],
        "split": ["validation"] * 6, "ProductBrand": ["TIE", "b", "new", "", "r", "other"],
        "Brand": ["B", "B", "UnItemised brand", "UnItemised brand", "Rare", "UnItemised brand"]})
    return training, validation


def original_artifacts(model, validation):
    original = {"label_names": list(LABELS), "mapping": {"tie": "A", "b": "B", "r": "Rare", "other": "UnItemised brand"},
        "fallback_label": "UnItemised brand", "training_record_ids_sha256": model["training_record_ids_sha256"],
        "training_class_counts": dict(zip(LABELS, model["training_class_counts"]))}
    encoding = {label: i for i, label in enumerate(LABELS)}
    labels = validation.ProductBrand.map(brand_key).map(original["mapping"]).fillna(original["fallback_label"])
    evaluation = {"validation_record_ids_sha256": _hash_items(validation.record_id.tolist()), "coverage_records": 4,
        "full_validation": classification_metrics(validation.Brand.map(encoding).to_numpy(), labels.map(encoding).to_numpy(), LABELS)}
    return original, evaluation


class ProbabilisticBrandTests(unittest.TestCase):
    def setUp(self):
        self.training, self.validation = populations()
        self.model, _ = fit_frequency_lookup(self.training, LABELS)

    def test_counts_prior_and_fixed_smoothing_are_training_only_and_deterministic(self):
        again, _ = fit_frequency_lookup(self.training, LABELS)
        self.assertEqual(self.model, again)
        self.assertEqual(self.model["training_class_counts"], [1, 1, 4, 3])
        self.assertEqual(self.model["smoothing"], SMOOTHING)
        self.assertEqual(ALPHA, 1.)
        self.assertNotIn("new", self.model["key_counts"])
        self.assertEqual(self.model["training_nonmissing_feature_records"], 8)
        self.assertFalse(self.model["validation_used_for_smoothing_selection"])
        with self.assertRaisesRegex(ValueError, "role"):
            fit_frequency_lookup(pd.concat([self.training, self.validation]), LABELS)

    def test_unseen_and_all_missing_forms_receive_the_exact_training_prior(self):
        values = ["new", None, "", "null", "NaN", " n/a "]
        probabilities, cohorts, _ = predict_frequency_lookup(self.model, values)
        np.testing.assert_allclose(probabilities, np.tile(self.model["class_prior"], (len(values), 1)), rtol=0, atol=0)
        self.assertEqual(cohorts["unseen_nonmissing_values"].sum(), 1)
        self.assertEqual(cohorts["missing_values"].sum(), 5)
        self.assertFalse(cohorts["seen_feature_values"].any())
        self.assertTrue((probabilities.argmax(axis=1) == LABELS.index("UnItemised brand")).all())

    def test_ambiguous_mapping_uses_prior_and_can_change_the_original_tie_break(self):
        probabilities, _, _ = predict_frequency_lookup(self.model, ["TIE"])
        prior = np.asarray(self.model["class_prior"])
        expected = prior.copy()
        expected[[LABELS.index("A"), LABELS.index("B")]] += 1
        np.testing.assert_allclose(probabilities[0], expected / 3)
        self.assertEqual(LABELS[int(probabilities.argmax())], "B")
        # The saved deterministic winner is A; no probability edits force it back.
        old, metrics = original_artifacts(self.model, self.validation)
        _, result, differences, _ = evaluate_frequency_lookup(self.model, self.validation, old, metrics)
        self.assertEqual(result["changed_prediction_records"], 1)
        self.assertEqual(result["old_wrong_new_correct"], 1)
        self.assertEqual(differences.deterministic_label.tolist(), ["A"])
        self.assertEqual(differences.probabilistic_label.tolist(), ["B"])

    def test_rare_classes_keep_positive_support_and_no_estimate_is_one_hot(self):
        probabilities, _, _ = predict_frequency_lookup(self.model, ["r", "b", "other", "new"])
        self.assertTrue((probabilities > 0).all())
        self.assertTrue((probabilities < 1).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
        rare = LABELS.index("Rare")
        self.assertAlmostEqual(probabilities[0, rare], (1 + 1 / 9) / 2)

    def test_saved_label_order_and_unicode_html_key_normalization_are_preserved(self):
        first, _, _ = predict_frequency_lookup(self.model, ["<b> ＴＩＥ </b>", "b"])
        reversed_model, _ = fit_frequency_lookup(self.training, LABELS[::-1])
        second, _, _ = predict_frequency_lookup(reversed_model, ["tie", "b"])
        np.testing.assert_allclose(first, second[:, ::-1])
        restored = json.loads(json.dumps(self.model))
        np.testing.assert_array_equal(first, predict_frequency_lookup(restored, ["tie", "b"])[0])

    def test_protected_records_and_groups_are_rejected_in_both_roles(self):
        for field in ("protected_ids", "protected_groups"):
            source = self.training.record_id if field == "protected_ids" else self.training.group_id
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Protected-test"):
                fit_frequency_lookup(self.training, LABELS, **{field: [source.iloc[0]]})
            old, metrics = original_artifacts(self.model, self.validation)
            source = self.validation.record_id if field == "protected_ids" else self.validation.group_id
            with self.assertRaisesRegex(ValueError, "Protected-test"):
                evaluate_frequency_lookup(self.model, self.validation, old, metrics, **{field: [source.iloc[0]]})

    def test_malformed_vocabulary_prior_or_population_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "supported class"):
            fit_frequency_lookup(self.training, (*LABELS, "No training support"))
        wrong = json.loads(json.dumps(self.model))
        wrong["class_prior"][0] = -1
        with self.assertRaisesRegex(ValueError, "Invalid lookup"):
            predict_frequency_lookup(wrong, ["b"])
        duplicate = self.training.copy()
        duplicate.loc[1, "record_id"] = duplicate.loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "identity"):
            fit_frequency_lookup(duplicate, LABELS)

    def make_export_fixture(self, root):
        source, mini = root / "saved", root / "minilm"
        source.mkdir(); mini.mkdir()
        old, metrics = original_artifacts(self.model, self.validation)
        write_json(source / "lookup_model.json", old)
        write_json(source / "lookup_baseline.json", metrics)
        snapshots = {p: sha256(p) for p in source.iterdir()}
        frame = pd.concat([self.training, self.validation], ignore_index=True)
        protected = pd.DataFrame({"record_id": ["protected"], "group_id": ["gp"]})
        inputs = ExportInputs("Brand", source, {}, replace(ModelConfig(), sequence_length=256), LABELS, frame,
            np.arange(len(self.training)), np.arange(len(self.training), len(frame)),
            frame[["record_id", "group_id", "split"]], protected, snapshots, "test-brand-minilm", {"allocation": "approved fixture"}, None)
        predictions, _, _, _ = evaluate_frequency_lookup(self.model, self.validation, old, metrics)
        predictions.to_csv(mini / "predictions.csv", index=False, lineterminator="\n")
        write_json(mini / "manifest.json", {"target": "Brand", "model_identity": inputs.model_identity,
            "label_names": list(LABELS), "files_sha256": {"predictions.csv": sha256(mini / "predictions.csv")}})
        return inputs, mini

    def test_export_actual_loader_scoring_and_calibration_leave_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, mini = self.make_export_fixture(root)
            before = {p: p.read_bytes() for p in (*inputs.snapshots, *mini.iterdir())}
            output = root / "new-probabilistic"
            with patch("catalogiq.target_minilm_experiment.fit_brand_lookup", side_effect=AssertionError("Do not refit original")), \
                    patch("catalogiq.segment_transformer.train_transformer", side_effect=AssertionError("No neural training")):
                result = export_probabilistic_lookup(inputs, output, mini, LOADER, calibration_source=CALIBRATION)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result["status"], "completed_and_verified")
            self.assertEqual(result["changed_prediction_records"], 1)
            self.assertFalse(result["calibration"]["export_probabilities_temperature_scaled"])
            self.assertEqual(result["calibration"]["rows_retained"], 6)
            self.assertEqual(manifest["label_mapping"], {f"probability_{i}": label for i, label in enumerate(LABELS)})
            self.assertEqual(manifest["validation_record_ids_sha256"], _hash_items(self.validation.record_id.tolist()))
            self.assertIsNone(manifest["inference_latency_ms"])
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(sha256(output / name), digest)
            for path in output.iterdir():
                self.assertNotIn(b"\r", path.read_bytes(), path.name)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            model = json.loads((output / "model.json").read_text())
            predictions = pd.read_csv(output / "predictions.csv", keep_default_na=False)
            expected = predict_frequency_lookup(model, self.validation.ProductBrand)[0]
            np.testing.assert_allclose(predictions[[f"probability_{i}" for i in range(4)]], expected, atol=1e-15, rtol=0)
            with self.assertRaises(FileExistsError):
                export_probabilistic_lookup(inputs, output, mini, LOADER, calibration_source=CALIBRATION)

    def test_changed_saved_baseline_or_minilm_alignment_blocks_export(self):
        for defect in ("saved_baseline", "groups", "minilm"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, mini = self.make_export_fixture(root)
                if defect == "saved_baseline":
                    (inputs.directory / "lookup_model.json").write_text('{}')
                elif defect == "groups":
                    inputs.frame.loc[len(self.training), "group_id"] = inputs.frame.group_id.iloc[0]
                else:
                    (mini / "predictions.csv").write_text('changed')
                with self.assertRaises(ValueError):
                    export_probabilistic_lookup(inputs, root / "new", mini, LOADER, calibration_source=CALIBRATION)
                self.assertFalse((root / "new").exists())

    def test_unreviewed_calibration_definitions_are_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.py"
            changed.write_text(CALIBRATION.read_text().replace('np.clip(np.vstack(probs), 1e-12, 1.0)', 'np.clip(np.vstack(probs), 1e-3, 1.0)'))
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                calibration_compatibility({}, changed)


if __name__ == "__main__":
    unittest.main()
