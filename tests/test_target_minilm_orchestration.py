"""Shared dry-run orchestration and Mnfr regression; no model training."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from catalogiq.target_minilm_preparation import (
    INPUT_FIELDS, _annotate_target_manifest, run_mnfr_minilm, run_target_minilm,
)
from test_segment_transformer import TinyTokenizer
import test_target_minilm_preparation as preparation_fixtures


MODULE = "catalogiq.target_minilm_preparation"


class TargetMiniLMOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = preparation_fixtures.TargetMiniLMPreparationTests("test_v4_targets_select_their_own_labels_without_segment_filtering")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.enterContext(patch(MODULE + "._training_readiness", return_value={
            "ready_for_training": True, "blockers": [], "resolved_device": "cpu"}))

    def prepare_all(self):
        return {target: self.fixture.prepare(target, self.fixture.proposal(target) if target in ("Brand", "Sub-Segment") else None)
                for target in ("Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")}

    def test_all_targets_default_to_read_only_dry_runs_with_their_own_metadata(self):
        prepared = self.prepare_all()
        before = {str(path.relative_to(self.fixture.root)): path.read_bytes()
                  for path in self.fixture.root.rglob("*") if path.is_file()}
        reports = {}
        with patch(MODULE + ".train_transformer", autospec=True, side_effect=AssertionError("Training forbidden")) as trainer:
            for target, value in prepared.items():
                report = run_target_minilm(value, tokenizer=TinyTokenizer())
                reports[target] = report
                self.assertTrue(report["dry_run"])
                self.assertFalse(report["training_authorized"])
                self.assertFalse(report["model_trained"])
                self.assertEqual(report["target"], target)
                self.assertEqual(report["label_names"], list(value.label_names))
                self.assertEqual(report["label_to_id"], {label: i for i, label in enumerate(value.label_names)})
                self.assertEqual(report["input_fields"], list(INPUT_FIELDS))
                self.assertEqual(report["sequence_length"], 256)
                self.assertEqual(report["config"], value.config.to_dict())
                self.assertEqual(report["model_identity"], value.model_identity)
                self.assertEqual(report["output_directory"], str(value.output_directory))
                self.assertEqual(report["ordered_supervised_membership_sha256"], value.audit["ordered_supervised_membership_sha256"])
                self.assertEqual(sum(report["train_class_counts"]), len(value.train_indices))
                self.assertEqual(sum(report["validation_class_counts"]), len(value.validation_indices))
                self.assertFalse(value.output_directory.exists())
                if target in ("Brand", "Sub-Segment"):
                    self.assertEqual(report["split_identity"]["allocation_completion_sha256"], value.audit["allocation_completion_sha256"])
                    self.assertEqual(report["split_identity"]["allocation_assignments_sha256"], value.audit["allocation_sha256"]["assignments.csv"])
                else:
                    self.assertEqual(report["split_identity"]["allocation"], "frozen V4; no reallocation")
        trainer.assert_not_called()
        self.assertEqual(len({report["model_identity"] for report in reports.values()}), 4)
        self.assertEqual(len({report["output_directory"] for report in reports.values()}), 4)
        self.assertEqual(before, {str(path.relative_to(self.fixture.root)): path.read_bytes()
                                 for path in self.fixture.root.rglob("*") if path.is_file()})

    def test_training_requires_a_second_explicit_authorization_flag(self):
        value = self.fixture.prepare("Mnfr")
        with patch(MODULE + ".tokenize_target_minilm") as tokenize, patch(MODULE + ".train_transformer") as trainer:
            with self.assertRaisesRegex(PermissionError, "explicit authorization"):
                run_target_minilm(value, dry_run=False)
            with self.assertRaisesRegex(PermissionError, "explicit authorization"):
                run_mnfr_minilm(value, dry_run=False, benchmark_steps=20)
            for flag in ("yes", 1, None):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "training_authorized must be boolean"):
                    run_target_minilm(value, training_authorized=flag)
        tokenize.assert_not_called()
        trainer.assert_not_called()
        self.assertFalse(value.output_directory.exists())

    def test_legacy_manifest_correction_is_target_aware_for_all_targets(self):
        for target, prepared in self.prepare_all().items():
            report = run_target_minilm(prepared, tokenizer=TinyTokenizer())
            path = self.fixture.root / f"{target}-legacy-manifest.json"
            path.write_text(json.dumps({"target": "Segment", "input_fields": ["ProductName"],
                "config": report["config"], "label_names": report["label_names"],
                "train_record_ids_sha256": report["train_record_ids_sha256"],
                "validation_record_ids_sha256": report["validation_record_ids_sha256"],
                "initial_state_sha256": "unchanged-state"}), encoding="utf8")
            manifest = _annotate_target_manifest(path, report)
            self.assertEqual(manifest["target"], target)
            self.assertEqual(manifest["label_names"], list(prepared.label_names))
            self.assertEqual(manifest["input_fields"], list(INPUT_FIELDS))
            self.assertEqual(manifest["sequence_length"], 256)
            self.assertEqual(manifest["split_identity"], report["split_identity"])
            self.assertEqual(manifest["initial_state_sha256"], "unchanged-state")

    def test_mnfr_wrapper_and_shared_entry_point_return_identical_dry_run_metadata(self):
        value = self.fixture.prepare("Mnfr")
        generic = run_target_minilm(value, tokenizer=TinyTokenizer())
        compatible = run_mnfr_minilm(value, tokenizer=TinyTokenizer())
        self.assertEqual(generic, compatible)

    def test_all_targets_delegate_unchanged_arguments_and_correct_saved_manifests_using_only_a_stub(self):
        for target, prepared in self.prepare_all().items():
            report = run_target_minilm(prepared, tokenizer=TinyTokenizer())

            def recording_trainer(corpus, train_indices, validation_indices, label_names, config, output_dir, progress):
                self.assertEqual(corpus.label_names, prepared.label_names)
                self.assertEqual(tuple(label_names), prepared.label_names)
                self.assertEqual(train_indices.tolist(), prepared.train_indices.tolist())
                self.assertEqual(validation_indices.tolist(), prepared.validation_indices.tolist())
                self.assertEqual(config, prepared.config)
                self.assertEqual(output_dir, prepared.output_directory)
                legacy = {key: report[key] for key in (
                    "config", "label_names", "train_record_ids_sha256", "validation_record_ids_sha256")}
                legacy.update(target="Segment", initial_state_sha256="stub-state")
                (output_dir / "initialization.json").write_text(json.dumps(legacy), encoding="utf8")
                progress("Stub initialized; no neural execution")
                initial = json.loads((output_dir / "initialization.json").read_text(encoding="utf8"))
                self.assertEqual(initial["target"], target)
                (output_dir / "training_manifest.json").write_text(json.dumps(legacy), encoding="utf8")
                return None, legacy

            with patch(MODULE + ".train_transformer", side_effect=recording_trainer) as trainer:
                result = run_target_minilm(prepared, dry_run=False, training_authorized=True,
                    tokenizer=TinyTokenizer(), progress=None)
            trainer.assert_called_once()
            self.assertEqual(result["status"], "completed")
            for name in ("run_protocol.json", "initialization.json", "training_manifest.json"):
                manifest = json.loads((prepared.output_directory / name).read_text(encoding="utf8"))
                for key in ("target", "label_names", "label_to_id", "model_identity", "input_fields",
                            "sequence_length", "representation", "split_identity"):
                    self.assertEqual(manifest[key], report[key])

    def test_mnfr_pre_generalization_fingerprints_and_configuration_are_preserved(self):
        value = self.fixture.prepare("Mnfr")
        report = run_target_minilm(value, tokenizer=TinyTokenizer())
        # Captured from the original Mnfr orchestrator before generalization.
        expected = {
            "model_identity": "minilm-mnfr-category256-a63a4f163e1cc176",
            "ordered_supervised_membership_sha256": "16417072960118e9104a2073d28fa9b8948fcb847cd5a2732fe4e469776eceeb",
            "train_record_ids_sha256": "f05f55fd9714d43352bc27d95e5d8a0f26a9cb896b352a1f5d70a82e45ad23ef",
            "validation_record_ids_sha256": "6bdc2a9e4321741c5fb4d350e7d9f8a7d3e70c340d3958e51cc1d10b93a06540",
            "label_ids_sha256": "42a6ad36c1324362e50470477878f03c80e95a67e33adc2b1bde5679efb57508",
            "text_sha256": "d988435ce628e7186cae3cb83721aa555b64fc50472d7baa5f2360581977fb73",
            "token_sha256": "4767b9b57f9867fe089862d65dbd48e9e67510c0a4f29406e4948181201c274f",
        }
        for name, expected_value in expected.items():
            self.assertEqual(report[name], expected_value)
        config = report["config"]
        self.assertEqual((config["epochs"], config["batch_size"], config["evaluation_batch_size"], config["seed"], config["threads"]), (3, 32, 64, 42, 4))
        self.assertEqual((config["learning_rate"], config["weight_decay"], config["dropout"], config["warmup_fraction"]), (3e-5, .01, .10, .10))
        self.assertEqual((config["sequence_length"], config["precision"], config["device"], config["local_files_only"]), (256, "fp32", "auto", True))

    def test_changed_allocation_or_membership_is_rejected_before_training(self):
        path = self.fixture.proposal("Brand")
        value = self.fixture.prepare("Brand", path)
        with (path / "class_coverage.csv").open("a", encoding="utf8") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(ValueError, "allocation changed"):
            run_target_minilm(value, tokenizer=TinyTokenizer())
        age = self.fixture.prepare("TargetAgeGroup")
        changed = age.frame.copy()
        changed.loc[changed.record_id.eq("a"), "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "membership or configuration changed"):
            run_target_minilm(replace(age, frame=changed), tokenizer=TinyTokenizer())

    def test_other_targets_cannot_accidentally_use_mnfr_benchmark_mode(self):
        age = self.fixture.prepare("TargetAgeGroup")
        with patch(MODULE + ".train_transformer") as trainer, self.assertRaisesRegex(ValueError, "Mnfr only"):
            run_target_minilm(age, dry_run=False, training_authorized=True, benchmark_steps=1, tokenizer=TinyTokenizer())
        trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
