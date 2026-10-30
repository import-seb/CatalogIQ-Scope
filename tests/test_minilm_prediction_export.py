"""Offline inference/export contract tests; no fine tuning or model downloads."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.maria_harness_compatibility import load_with_maria_harness, loader_definitions
from catalogiq.minilm_prediction_export import (
    ALLOCATIONS, TARGETS, ExportInputs, compare_original, export_target,
    predict_validation, selected_source, validate_predictions, verify_snapshots,
    audit_handoff, handoff_cost_metadata, repair_export_handoff,
)
from catalogiq.segment_category_ablation import tokenize_category_corpus
from catalogiq.segment_transformer import ModelConfig, _hash_items, _make_classifier, _state_sha256
from catalogiq.target_minilm_preparation import INPUT_FIELDS
from test_segment_transformer import TinyTokenizer, tiny_encoder


MODULE = "catalogiq.minilm_prediction_export"
HARNESS = Path(__file__).parent / "fixtures/maria_harness_loader_05e52e2.py"


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        # Deliberately non-alphabetic vocabulary and non-sorted validation order.
        self.labels = ("Zeta", "Alpha")
        self.validation = pd.DataFrame({"record_id": ["z", "a"], "group_id": ["gz", "ga"],
            "split": ["validation", "validation"], "true_label": ["Zeta", "Alpha"]})
        self.predictions = pd.DataFrame({"record_id": ["z", "a"], "true_label": ["Zeta", "Alpha"],
            "predicted_label": ["Zeta", "Alpha"], "confidence": [.8, .9],
            "probability_0": [.8, .1], "probability_1": [.2, .9]})

    def validate(self, frame=None, validation=None, labels=None, protected_groups=()):
        return validate_predictions(self.predictions if frame is None else frame,
            self.validation if validation is None else validation,
            self.labels if labels is None else labels, ["protected"], protected_groups)

    def inputs(self, target="Brand"):
        reference = self.model / "predictions.csv"
        self.predictions.to_csv(reference, index=False)
        checkpoint = self.model / "checkpoint.pt"
        checkpoint.write_bytes(b"unchanged test checkpoint")
        frame = self.validation.rename(columns={"true_label": target})
        config = replace(ModelConfig(), sequence_length=256, device="cpu")
        metadata = {"validation_record_ids_sha256": _hash_items(["z", "a"]),
            "final_state_sha256": "state", "text_sha256": "texts", "token_sha256": "tokens"}
        assignments = pd.concat([frame[["record_id", "group_id", "split"]], pd.DataFrame({
            "record_id": ["missing", "train", "protected"], "group_id": ["gm", "gt", "gp"],
            "split": ["validation", "train", "test"]})], ignore_index=True)
        return ExportInputs(target, self.model, metadata, config, self.labels, frame,
            np.array([], dtype=int), np.array([0, 1]), assignments,
            pd.DataFrame({"record_id": ["protected"], "group_id": ["gp"]}),
            {checkpoint: sha256(checkpoint), reference: sha256(reference)}, "test-model",
            {"allocation": "approved test fixture"}, reference)

    def test_all_supported_targets_and_platform_exclusion(self):
        self.assertEqual(TARGETS, ("Segment", "Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment"))
        self.assertEqual(set(ALLOCATIONS), {"Brand", "Sub-Segment"})
        with self.assertRaisesRegex(ValueError, "Platform is excluded"):
            selected_source(self.root, "Platform")

    def test_probabilities_and_numeric_class_order_for_all_vocabulary_sizes(self):
        for count in (2, 3, 8, 49, 143):
            with self.subTest(count=count):
                labels = tuple(f"Class {i}" for i in range(count))
                matrix = np.zeros((2, count))
                matrix[0, -1] = 1
                matrix[1, 0] = 1
                frame = pd.DataFrame(matrix, columns=[f"probability_{i}" for i in range(count)])
                for key, values in reversed(list({"record_id": ["z", "a"], "true_label": [labels[-1], labels[0]],
                    "predicted_label": [labels[-1], labels[0]], "confidence": [1., 1.]}.items())):
                    frame.insert(0, key, values)
                validation = self.validation.assign(true_label=frame.true_label)
                self.assertEqual(self.validate(frame, validation, labels)["classes"], count)

    def test_invalid_ids_membership_truth_and_groups_are_rejected(self):
        changes = [("record_id", ["z", "z"]), ("record_id", ["z", ""]),
            ("record_id", ["z", "train"]), ("record_id", ["z", "protected"]),
            ("true_label", ["Alpha", "Alpha"]), ("predicted_label", ["Unknown", "Alpha"])]
        for column, values in changes:
            with self.subTest(column=column, values=values), self.assertRaises(ValueError):
                self.validate(self.predictions.assign(**{column: values}))
        with self.assertRaises(ValueError):
            self.validate(self.predictions.iloc[:1])
        with self.assertRaisesRegex(ValueError, "Protected-test"):
            self.validate(protected_groups=["ga"])

    def test_nonfinite_unnormalized_and_misaligned_probabilities_are_rejected(self):
        for column, value in (("probability_0", np.nan), ("probability_0", -.1),
            ("probability_0", 1.1), ("probability_0", .7), ("confidence", .7), ("confidence", np.inf)):
            frame = self.predictions.copy()
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.validate(frame)
        with self.assertRaisesRegex(ValueError, "class order"):
            self.validate(labels=self.labels[::-1])
        with self.assertRaisesRegex(ValueError, "columns/order"):
            self.validate(self.predictions[list(self.predictions.columns)[::-1]])

    def write_harness_input(self, directory, labels=None, digest=None):
        directory.mkdir()
        self.predictions.to_csv(directory / "predictions.csv", index=False)
        (directory / "manifest.json").write_text(json.dumps({"label_names": labels or self.labels,
            "validation_record_ids_sha256": digest or _hash_items(sorted(self.validation.record_id))}))

    def test_actual_harness_consumes_all_rows_with_saved_label_order(self):
        directory = self.root / "export"
        self.write_harness_input(directory)
        run, report = load_with_maria_harness(directory, HARNESS, self.validation, ["protected"], ["z", "a", "missing"])
        self.assertEqual(run["labels"], list(self.labels))
        self.assertEqual(report["rows_consumed"], 2)
        self.assertTrue(report["split_checked"])
        self.assertEqual(report["metrics"]["macro_f1_full_vocabulary"], 1.)

    def test_actual_harness_rejects_wrong_allocation_and_class_order(self):
        for name, labels, digest in (("wrong_split", self.labels, "0" * 64),
                                    ("wrong_order", self.labels[::-1], None)):
            with self.subTest(name=name):
                directory = self.root / name
                self.write_harness_input(directory, labels, digest)
                with self.assertRaises(ValueError):
                    load_with_maria_harness(directory, HARNESS, self.validation, ["protected"], ["z", "a"])

    def test_harness_retains_zero_support_classes_and_reports_both_macro_scopes(self):
        directory = self.root / "export"
        self.write_harness_input(directory, labels=(*self.labels, "Training only"))
        predictions = self.predictions.assign(probability_2=0.)
        predictions.to_csv(directory / "predictions.csv", index=False)
        run, report = load_with_maria_harness(directory, HARNESS, self.validation, ["protected"], ["z", "a"])
        self.assertEqual(len(run["labels"]), 3)
        self.assertEqual(len(run["frame"]._probs.iloc[0]), 3)
        self.assertEqual(report["metrics"]["macro_f1_observed_labels"], 1.)
        self.assertAlmostEqual(report["metrics"]["macro_f1_full_vocabulary"], 2 / 3)

    def test_harness_source_changes_are_not_executed(self):
        modified = self.root / "unreviewed.py"
        modified.write_text(HARNESS.read_text().replace('d = hashlib.sha256()', 'd = hashlib.sha512()'))
        with self.assertRaisesRegex(ValueError, "reviewed current commit"):
            loader_definitions(modified)

    def test_export_roundtrip_all_targets_preserves_originals_and_hash_conventions(self):
        for target in TARGETS:
            with self.subTest(target=target):
                inputs = self.inputs(target)
                before = {path: path.read_bytes() for path in inputs.snapshots}
                output = self.root / "exports"
                with patch(MODULE + ".predict_validation", return_value=(self.predictions.copy(), {"model_training_performed": False})):
                    result = export_target(inputs, output, HARNESS)
                directory = Path(result["directory"])
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["target"], target)
                self.assertEqual(manifest["input_fields"], list(INPUT_FIELDS))
                self.assertEqual(manifest["label_names"], list(self.labels))
                self.assertEqual(manifest["validation_record_ids_sha256"], _hash_items(["a", "z"]))
                self.assertEqual(manifest["original_training_validation_record_ids_sha256"], _hash_items(["z", "a"]))
                self.assertNotEqual(manifest["validation_record_ids_sha256"], manifest["original_training_validation_record_ids_sha256"])
                self.assertEqual(result["harness_compatibility"]["rows_dropped"], 0)
                self.assertEqual(result["comparison_to_original"]["label_disagreements"], 0)
                for name, digest in manifest["files_sha256"].items():
                    self.assertEqual(sha256(directory / name), digest)
                for path in directory.iterdir():
                    self.assertNotIn(b"\r", path.read_bytes(), path.name)
                self.assertIsNone(manifest["training_seconds"])
                self.assertIsNone(manifest["inference_latency_ms"])
                self.assertIn("hardware", manifest)
                self.assertEqual(before, {path: path.read_bytes() for path in before})
                with self.assertRaises(FileExistsError):
                    export_target(inputs, output, HARNESS)

    def test_accented_brand_label_survives_windows_harness_json_loading(self):
        self.labels = ("Horb\u00e4ach", "Alpha")
        self.validation.loc[0, "true_label"] = self.labels[0]
        self.predictions.loc[0, ["true_label", "predicted_label"]] = self.labels[0]
        inputs = self.inputs()
        with patch(MODULE + ".predict_validation", return_value=(self.predictions.copy(), {})):
            result = export_target(inputs, self.root / "exports", HARNESS)
        directory = Path(result["directory"])
        for name in ("label_mapping.json", "manifest.json"):
            encoded = (directory / name).read_bytes()
            self.assertTrue(encoded.isascii())
            self.assertEqual(json.loads(encoded.decode("cp1252"))["probability_0"] if name == "label_mapping.json"
                else json.loads(encoded.decode("cp1252"))["label_names"][0], self.labels[0])
        self.assertEqual(result["harness_compatibility"]["labels"], list(self.labels))

    def test_changed_sources_or_reference_fail_closed(self):
        inputs = self.inputs()
        changed = self.predictions.copy()
        changed.loc[0, "predicted_label"] = "Alpha"
        with self.assertRaisesRegex(ValueError, "truth/decisions"):
            compare_original(changed, inputs)
        (self.model / "checkpoint.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed during export"):
            verify_snapshots(inputs)

    def test_historical_costs_have_provenance_and_unmeasured_latency_is_null(self):
        inputs = self.inputs()
        training_path = self.model / "source_training_manifest.json"
        training_path.write_text(json.dumps({"runtime_seconds": 123.4, "device": "cuda",
            "cuda_device_name": "Historical GPU", "cuda_version": "12.8", "torch_version": "saved torch"}))
        inputs.snapshots[training_path] = sha256(training_path)
        result = handoff_cost_metadata(inputs, {"device": "cuda", "validation_inference_seconds": 2.}, 100)
        self.assertEqual(result["training_seconds"], 123.4)
        self.assertEqual(result["runtime_seconds"], 123.4)
        self.assertEqual(result["hardware"]["training"]["gpu_name"], "Historical GPU")
        self.assertIsNone(result["hardware"]["inference"]["gpu_name"])
        self.assertEqual(result["amortized_inference_latency_ms_per_record"], 20.)
        self.assertIsNone(result["inference_latency_ms"])
        self.assertIsNone(result["training_optimizer_only_seconds"])
        self.assertEqual(result["training_metadata_source"]["sha256"], sha256(training_path))
        training_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, "not sealed"):
            handoff_cost_metadata(inputs, {}, 100)

    def test_metadata_and_line_ending_repair_keeps_every_csv_field_verbatim(self):
        inputs = self.inputs()
        with patch(MODULE + ".predict_validation", return_value=(self.predictions.copy(), {"validation_inference_seconds": 1.})):
            result = export_target(inputs, self.root / "exports", HARNESS)
        directory = Path(result["directory"])
        # Recreate the original deficient export: missing cost fields and CRLF.
        manifest = json.loads((directory / "manifest.json").read_text())
        for key in handoff_cost_metadata(inputs, manifest["inference"], 2):
            manifest.pop(key)
        for path in directory.iterdir():
            if path.name != "manifest.json":
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        manifest["files_sha256"] = {path.name: sha256(path) for path in directory.iterdir() if path.name != "manifest.json"}
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").replace("\n", "\r\n").encode())
        before = {name: (directory / name).read_bytes() for name in ("predictions.csv", "validation_context.csv")}
        sources = {path: path.read_bytes() for path in inputs.snapshots}
        with patch(MODULE + ".predict_validation", side_effect=AssertionError("No new predictions")):
            repaired = repair_export_handoff(inputs, directory, HARNESS)
        self.assertEqual(repaired["before"]["status"], "FAIL")
        self.assertFalse(repaired["before"]["checks"]["LF_artifacts"])
        self.assertFalse(repaired["before"]["checks"]["cost_and_hardware_metadata"])
        self.assertEqual(repaired["after"]["status"], "PASS")
        self.assertFalse(repaired["inference_rerun"])
        for name, raw in before.items():
            self.assertEqual((directory / name).read_bytes(), raw.replace(b"\r\n", b"\n"))
        self.assertEqual(sources, {path: path.read_bytes() for path in sources})
        self.assertEqual(audit_handoff(inputs, directory, HARNESS)["status"], "PASS")

    def test_metadata_repair_rejects_changed_class_mapping(self):
        inputs = self.inputs()
        with patch(MODULE + ".predict_validation", return_value=(self.predictions.copy(), {})):
            result = export_target(inputs, self.root / "exports", HARNESS)
        directory = Path(result["directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["label_mapping"]["probability_0"] = "Wrong class"
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "beyond metadata/line endings"):
            repair_export_handoff(inputs, directory, HARNESS)

    def test_loader_failure_leaves_no_final_or_staging_export(self):
        inputs = self.inputs()
        output = self.root / "exports"
        with patch(MODULE + ".predict_validation", return_value=(self.predictions.copy(), {})), \
                patch(MODULE + ".load_with_maria_harness", side_effect=ValueError("bad harness")):
            with self.assertRaisesRegex(ValueError, "bad harness"):
                export_target(inputs, output, HARNESS)
        self.assertEqual(list(output.iterdir()), [])

    def test_existing_classifier_performs_inference_only_from_saved_state(self):
        import torch
        inputs = self.inputs("Mnfr")
        for field in INPUT_FIELDS:
            inputs.frame[field] = "<b>Example</b> product"
        config = replace(inputs.config, evaluation_batch_size=1)
        inputs.config = config
        legacy = inputs.frame[["record_id", "Mnfr", *INPUT_FIELDS]].rename(columns={"Mnfr": "Segment"})
        corpus = tokenize_category_corpus(legacy, self.labels, config, tokenizer=TinyTokenizer())
        model = _make_classifier(torch, tiny_encoder(), len(self.labels), config.dropout)
        state = model.state_dict()
        torch.save({"state_dict": state, "model_config": config.to_dict(), "label_names": self.labels}, self.model / "checkpoint.pt")
        inputs.metadata.update(text_sha256=corpus.text_sha256, token_sha256=corpus.token_sha256, final_state_sha256=_state_sha256(state))
        before = sha256(self.model / "checkpoint.pt")
        with patch("torch.optim.AdamW", side_effect=AssertionError("Training forbidden")), \
                patch("catalogiq.segment_transformer.train_transformer", side_effect=AssertionError("Training forbidden")):
            first, runtime = predict_validation(inputs, tokenizer=TinyTokenizer(), encoder_factory=tiny_encoder)
            second, _ = predict_validation(inputs, tokenizer=TinyTokenizer(), encoder_factory=tiny_encoder)
        pd.testing.assert_frame_equal(first, second)
        self.validate(first)
        self.assertFalse(runtime["model_training_performed"])
        self.assertFalse(runtime["optimizer_created"])
        self.assertEqual(before, sha256(self.model / "checkpoint.pt"))
        inputs.labels = inputs.labels[::-1]
        with self.assertRaisesRegex(ValueError, "checkpoint state, labels"):
            predict_validation(inputs, tokenizer=TinyTokenizer(), encoder_factory=tiny_encoder)


if __name__ == "__main__":
    unittest.main()
