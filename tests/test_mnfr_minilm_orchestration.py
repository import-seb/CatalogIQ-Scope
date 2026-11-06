"""Mnfr orchestration with a recording trainer stub; no neural training runs."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from catalogiq.features import sha256
from catalogiq.segment_transformer import MODEL_FIELDS, _hash_items
from catalogiq.target_minilm_preparation import INPUT_FIELDS, _training_readiness, run_mnfr_minilm
from test_segment_transformer import TinyTokenizer
import test_target_minilm_preparation as preparation_fixtures
import test_mnfr_minilm_benchmark as benchmark_doubles


MODULE = "catalogiq.target_minilm_preparation"


class MnfrMiniLMOrchestrationTests(unittest.TestCase):
    def setUp(self):
        # Reuse only fixture methods, without inheriting/re-running their tests.
        self.fixture = preparation_fixtures.TargetMiniLMPreparationTests("test_v4_targets_select_their_own_labels_without_segment_filtering")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture.prepare("Mnfr")
        self.readiness = self.enterContext(patch(MODULE + "._training_readiness", return_value={
            "ready_for_training": True, "blockers": [], "software_versions": {"test": "offline"},
            "pretrained_artifact_sha256": {"cached-config.json": "test-hash"},
            "encoder_initialized": False, "weight_downloads_allowed": False}))

    def test_default_dry_run_has_correct_target_metadata_and_no_writes(self):
        before = {str(path.relative_to(self.fixture.root)): path.read_bytes()
                  for path in self.fixture.root.rglob("*") if path.is_file()}
        with patch(MODULE + ".train_transformer", side_effect=AssertionError("Training forbidden")) as trainer:
            result = run_mnfr_minilm(self.prepared, tokenizer=TinyTokenizer())
        trainer.assert_not_called()
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["model_trained"])
        self.assertEqual(result["target"], "Mnfr")
        self.assertEqual(result["label_names"], ["All others", "J&J"])
        self.assertEqual(result["label_to_id"], {"All others": 0, "J&J": 1})
        self.assertEqual(result["input_fields"], list(INPUT_FIELDS))
        self.assertEqual(result["sequence_length"], 256)
        self.assertEqual(result["config"], self.prepared.config.to_dict())
        self.assertEqual(result["representation"]["frozen_grouping_token_budget"], 128)
        self.assertEqual(result["split_identity"]["protocol_id"], "synthetic-v4")
        self.assertEqual(result["split_identity"]["allocation"], "frozen V4; no reallocation")
        self.assertEqual(result["training_rows"], 3)
        self.assertEqual(result["validation_rows"], 3)
        self.assertEqual(result["train_class_counts"], [2, 1])
        self.assertEqual(result["validation_class_counts"], [2, 1])
        self.assertEqual(result["output_directory"], str(self.prepared.output_directory))
        self.assertFalse(self.prepared.output_directory.exists())
        self.assertEqual(before, {str(path.relative_to(self.fixture.root)): path.read_bytes()
                                 for path in self.fixture.root.rglob("*") if path.is_file()})

    def test_explicit_vocabulary_order_is_preserved_in_counts_and_encoding(self):
        prepared = self.fixture.prepare("Mnfr", label_names=("J&J", "All others"))
        result = run_mnfr_minilm(prepared, tokenizer=TinyTokenizer())
        self.assertEqual(result["label_names"], ["J&J", "All others"])
        self.assertEqual(result["label_to_id"], {"J&J": 0, "All others": 1})
        self.assertEqual(result["train_class_counts"], [1, 2])
        self.assertEqual(result["validation_class_counts"], [1, 2])

    def test_non_mnfr_and_configuration_or_label_drift_are_rejected(self):
        for value in ("false", 0, None):
            with self.subTest(dry_run=value), self.assertRaisesRegex(ValueError, "dry_run must be boolean"):
                run_mnfr_minilm(self.prepared, dry_run=value, tokenizer=TinyTokenizer())
        other = self.fixture.prepare("TargetAgeGroup")
        with self.assertRaisesRegex(ValueError, "Mnfr only"):
            run_mnfr_minilm(other, tokenizer=TinyTokenizer())
        changed_config = replace(self.prepared.config, epochs=7)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            run_mnfr_minilm(replace(self.prepared, config=changed_config), tokenizer=TinyTokenizer())
        changed_frame = self.prepared.frame.copy()
        changed_frame.loc[changed_frame.record_id.eq("a"), "Mnfr"] = "J&J"
        with self.assertRaisesRegex(ValueError, "Corpus does not match"):
            run_mnfr_minilm(replace(self.prepared, frame=changed_frame), tokenizer=TinyTokenizer())

    def test_indices_cannot_be_swapped_away_from_frozen_roles(self):
        with self.assertRaisesRegex(ValueError, "Corpus does not match"):
            run_mnfr_minilm(replace(self.prepared, train_indices=self.prepared.validation_indices,
                validation_indices=self.prepared.train_indices), tokenizer=TinyTokenizer())

    def test_mnfr_output_cannot_be_redirected_into_segment_or_frozen_artifacts(self):
        for output in (self.fixture.root / "segment" / "existing_run", self.fixture.source / "mnfr" / self.prepared.model_identity):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "own prepared directory"):
                run_mnfr_minilm(replace(self.prepared, output_directory=output), tokenizer=TinyTokenizer())

    def test_readiness_blockers_are_reported_in_dry_run_and_block_training(self):
        self.readiness.return_value = {"ready_for_training": False, "blockers": ["Pinned cache absent"]}
        dry = run_mnfr_minilm(self.prepared, tokenizer=TinyTokenizer())
        self.assertEqual(dry["readiness"]["blockers"], ["Pinned cache absent"])
        with patch(MODULE + ".train_transformer") as trainer, self.assertRaisesRegex(ValueError, "Pinned cache absent"):
            run_mnfr_minilm(self.prepared, dry_run=False, training_authorized=True, tokenizer=TinyTokenizer())
        trainer.assert_not_called()
        self.assertFalse(self.prepared.output_directory.exists())

    def test_training_delegates_once_to_stub_and_corrects_legacy_manifests(self):
        received = {}
        def recording_trainer(corpus, train_indices, validation_indices, label_names, config, output_dir, progress):
            received.update(corpus=corpus, train_indices=train_indices, validation_indices=validation_indices,
                            label_names=label_names, config=config, output_dir=output_dir)
            manifest = {"target": "Segment", "input_fields": list(MODEL_FIELDS), "config": config.to_dict(),
                "label_names": list(label_names), "initial_state_sha256": "stub-initial", "final_state_sha256": "stub-final",
                "train_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in train_indices]),
                "validation_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in validation_indices])}
            (output_dir / "initialization.json").write_text(json.dumps(manifest), encoding="utf8")
            progress("Stub initialized; no neural execution")
            initial = json.loads((output_dir / "initialization.json").read_text(encoding="utf8"))
            self.assertEqual(initial["target"], "Mnfr")
            self.assertEqual(initial["input_fields"], list(INPUT_FIELDS))
            (output_dir / "training_manifest.json").write_text(json.dumps({**manifest, "artifacts": {}}), encoding="utf8")
            return None, manifest

        with patch(MODULE + ".train_transformer", side_effect=recording_trainer) as trainer:
            result = run_mnfr_minilm(self.prepared, dry_run=False, training_authorized=True, tokenizer=TinyTokenizer(), progress=None)
        trainer.assert_called_once()
        self.assertTrue(result["model_trained"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(received["output_dir"], self.prepared.output_directory)
        np.testing.assert_array_equal(received["train_indices"], self.prepared.train_indices)
        np.testing.assert_array_equal(received["validation_indices"], self.prepared.validation_indices)
        self.assertEqual(received["corpus"].label_names, self.prepared.label_names)
        output = self.prepared.output_directory
        for name in ("initialization.json", "training_manifest.json", "run_protocol.json"):
            value = json.loads((output / name).read_text(encoding="utf8"))
            self.assertEqual(value["target"], "Mnfr")
            self.assertEqual(value["label_names"], ["All others", "J&J"])
            self.assertEqual(value["input_fields"], list(INPUT_FIELDS))
            self.assertEqual(value["sequence_length"], 256)
            self.assertEqual(value["model_identity"], self.prepared.model_identity)
        manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf8"))
        self.assertEqual(manifest["initial_state_sha256"], "stub-initial")
        self.assertEqual(manifest["final_state_sha256"], "stub-final")
        self.assertTrue(manifest["model_trained"])
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(sha256(output / name), digest)
        self.assertFalse((output / "checkpoint.pt").exists())
        self.assertFalse((output / "predictions.csv").exists())

    def test_saved_tokenizer_change_during_verification_is_rejected(self):
        def change_cache_check(config):
            path = self.prepared.tokenizer_directory / "tokenizer.json"
            path.write_text('{"changed":true}', encoding="utf8")
            return {"ready_for_training": True}
        self.readiness.side_effect = change_cache_check
        with self.assertRaisesRegex(ValueError, "tokenizer changed"):
            run_mnfr_minilm(self.prepared, tokenizer=TinyTokenizer())

    def test_benchmark_stops_legacy_delegation_without_completed_model_artifacts(self):
        from catalogiq.mnfr_minilm_benchmark import MnfrBenchmark
        optimizer = benchmark_doubles.Optimizer()
        def recording_trainer(corpus, train_indices, validation_indices, label_names, config, output_dir, progress):
            manifest = {"target": "Segment", "input_fields": list(MODEL_FIELDS), "config": config.to_dict(),
                "label_names": list(label_names),
                "train_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in train_indices]),
                "validation_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in validation_indices])}
            (output_dir / "initialization.json").write_text(json.dumps(manifest), encoding="utf8")
            scheduler = benchmark_doubles.Scheduler(optimizer)
            progress("Transformer initialized: recording double")
            labels = benchmark_doubles.Tensor(corpus.label_ids[train_indices])
            benchmark_doubles.Loss().forward(SimpleNamespace(shape=(len(train_indices), 2)), labels)
            optimizer.step()
            scheduler.step()
            raise AssertionError("Benchmark stop was missed")
        def monitor(*args, **kwargs):
            return MnfrBenchmark(*args, **kwargs, torch=benchmark_doubles.FAKE_TORCH)
        before = self.prepared.config.to_dict()
        with patch(MODULE + ".train_transformer", side_effect=recording_trainer), \
                patch("catalogiq.mnfr_minilm_benchmark.MnfrBenchmark", side_effect=monitor):
            result = run_mnfr_minilm(self.prepared, dry_run=False, training_authorized=True, tokenizer=TinyTokenizer(), progress=None, benchmark_steps=1)
        self.assertEqual(result["status"], "benchmark_completed")
        self.assertFalse(result["full_training_completed"])
        self.assertEqual(result["benchmark"]["completed_optimizer_steps"], 1)
        self.assertEqual(result["config"], before)
        self.assertEqual(optimizer.updates, 1)
        output = self.prepared.output_directory
        self.assertTrue((output / "benchmark_report.json").exists())
        self.assertFalse((output / "training_manifest.json").exists())
        self.assertFalse((output / "checkpoint.pt").exists())
        self.assertFalse((output / "predictions.csv").exists())

    def test_readiness_uses_cached_artifacts_and_resolves_device_without_encoder(self):
        class Device:
            def __init__(self, value):
                self.type, self.index, self.value = value.split(":")[0], None, value
            def __str__(self):
                return self.value
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0), device=Device)
        with patch(MODULE + ".package_metadata.version", return_value="test-version"), \
                patch(MODULE + ".pretrained_artifacts", return_value={"cached-model.safetensors": "cached-hash"}) as cached, \
                patch.dict("sys.modules", {"torch": fake_torch}):
            result = _training_readiness(self.prepared.config)
            cached.assert_called_once_with(self.prepared.config)
            gpu = _training_readiness(replace(self.prepared.config, device="cuda"))
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["resolved_device"], "cpu")
        self.assertFalse(result["encoder_initialized"])
        self.assertFalse(result["weight_downloads_allowed"])
        self.assertFalse(gpu["ready_for_training"])
        self.assertTrue(any("CUDA device is unavailable" in value for value in gpu["blockers"]))

    def test_missing_runtime_and_cached_weights_are_readiness_blockers(self):
        from importlib.metadata import PackageNotFoundError
        with patch(MODULE + ".package_metadata.version", side_effect=PackageNotFoundError), \
                patch(MODULE + ".pretrained_artifacts", side_effect=FileNotFoundError("No local snapshot")):
            result = _training_readiness(self.prepared.config)
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(result["pretrained_artifact_sha256"], {})
        self.assertEqual(len(result["blockers"]), 6)


if __name__ == "__main__":
    unittest.main()
