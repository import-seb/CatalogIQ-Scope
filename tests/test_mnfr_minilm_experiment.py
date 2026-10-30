"""Mnfr loss/selection checks; no encoder or optimizer training."""
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from catalogiq.mnfr_minilm_experiment import (
    LABELS, candidate_preparation, choose_candidate, evaluate_candidate,
    mnfr_loss_scope, run_candidate, select_and_package,
)
from catalogiq.segment_transformer import _state_sha256
from catalogiq.target_minilm_preparation import INPUT_FIELDS, run_mnfr_minilm
from test_segment_transformer import TinyTokenizer
import test_target_minilm_preparation as fixtures


class MnfrExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TargetMiniLMPreparationTests("test_v4_targets_select_their_own_labels_without_segment_filtering")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture.prepare("Mnfr")

    def test_candidate_identity_changes_only_for_weighting_and_inputs_are_identical(self):
        baseline = candidate_preparation(self.prepared, "unweighted")
        weighted = candidate_preparation(self.prepared, "weighted_1_5")
        self.assertEqual(baseline.model_identity, self.prepared.model_identity)
        self.assertNotEqual(weighted.model_identity, baseline.model_identity)
        self.assertEqual(weighted.audit["loss_configuration"]["training_class_weights"], [1., 5.])
        self.assertEqual(weighted.config, baseline.config)
        pd.testing.assert_frame_equal(weighted.frame, baseline.frame)
        np.testing.assert_array_equal(weighted.train_indices, baseline.train_indices)
        with patch("catalogiq.target_minilm_preparation._training_readiness", return_value={"ready_for_training": True}):
            report = run_mnfr_minilm(weighted, tokenizer=TinyTokenizer())
        self.assertEqual(report["model_identity"], weighted.model_identity)
        self.assertTrue(report["dry_run"])

    def test_other_targets_reordered_labels_and_unapproved_candidates_are_rejected(self):
        for prepared, candidate in ((replace(self.prepared, target_column="Brand"), "weighted_1_5"),
                                    (replace(self.prepared, label_names=tuple(reversed(LABELS))), "weighted_1_5"),
                                    (self.prepared, "other_weight")):
            with self.assertRaises(ValueError):
                candidate_preparation(prepared, candidate)

    def test_loss_matches_pytorch_weights_for_training_and_unweighted_validation(self):
        logits = torch.tensor([[1., .5], [.8, .2], [.1, .9]], requires_grad=True)
        labels = torch.tensor([0, 1, 1])
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=torch.tensor([1., 5.]))
        unweighted = torch.nn.functional.cross_entropy(logits, labels)
        original = torch.nn.CrossEntropyLoss.forward
        with mnfr_loss_scope(torch, (1., 5.)) as counts:
            criterion = torch.nn.CrossEntropyLoss()
            self.assertTrue(torch.equal(criterion(logits, labels), expected))
            with torch.no_grad():
                self.assertTrue(torch.equal(criterion(logits, labels), unweighted))
        self.assertIs(torch.nn.CrossEntropyLoss.forward, original)
        self.assertEqual(counts, {"training_batches": 1, "training_examples": 3,
                                  "validation_batches": 1, "validation_examples": 3})

    def test_unweighted_loss_is_exact_and_patch_restores_after_failure(self):
        logits = torch.tensor([[1., .5], [.8, .2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        original = torch.nn.CrossEntropyLoss.forward
        expected = torch.nn.CrossEntropyLoss()(logits, labels)
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            with mnfr_loss_scope(torch, None):
                self.assertTrue(torch.equal(torch.nn.CrossEntropyLoss()(logits, labels), expected))
                raise RuntimeError("simulated failure")
        self.assertIs(torch.nn.CrossEntropyLoss.forward, original)

    def test_authorization_and_cuda_required_before_any_training(self):
        with patch("catalogiq.mnfr_minilm_experiment.run_mnfr_minilm") as train:
            with self.assertRaises(PermissionError):
                run_candidate(self.prepared, "unweighted")
            with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(ValueError, "CUDA"):
                run_candidate(self.prepared, "weighted_1_5", training_authorized=True)
        train.assert_not_called()

    def test_validation_metrics_use_only_exact_validation_rows_and_order(self):
        prepared = self.prepared
        frame = prepared.supervised_frame.iloc[prepared.validation_indices]
        labels = frame.Mnfr.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
        prepared.output_directory.mkdir(parents=True)
        predictions = pd.DataFrame({"record_id": frame.record_id.tolist(), "true_label": frame.Mnfr.tolist(),
            "predicted_label": frame.Mnfr.tolist(), "probability_0": 1-labels, "probability_1": labels})
        path = prepared.output_directory / "predictions.csv"
        predictions.to_csv(path, index=False)
        self.assertEqual(evaluate_candidate(prepared)["macro_f1"], 1.)
        predictions.iloc[::-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "frozen Mnfr validation"):
            evaluate_candidate(prepared)

    def test_selection_tolerance_predeclares_unweighted_tie_preference(self):
        result = lambda score: {"evaluation": {"macro_f1": score}}
        self.assertEqual(choose_candidate(result(.8), result(.802)), "weighted_1_5")
        self.assertEqual(choose_candidate(result(.8), result(.8009)), "unweighted")
        self.assertEqual(choose_candidate(result(.8), result(.801)), "unweighted")
        self.assertEqual(choose_candidate(result(.8), result(.8)), "unweighted")
        self.assertEqual(choose_candidate(result(.8), result(.79)), "unweighted")
        with self.assertRaises(ValueError):
            choose_candidate(result(float("nan")), result(.8))

    def test_only_selected_checkpoint_is_packaged_with_target_and_loss_metadata(self):
        root = self.fixture.root / "experiment"
        state = {"test_weight": torch.arange(6).reshape(2, 3).float()}
        for name, score in (("unweighted", .8), ("weighted_1_5", .85)):
            path = root / name / "mnfr" / name
            (path / "tokenizer").mkdir(parents=True)
            (path / "tokenizer" / "tokenizer.json").write_text("{}")
            checkpoint = path / "checkpoint.pt"
            torch.save({"state_dict": state, "label_names": list(LABELS)}, checkpoint)
            from catalogiq.features import sha256
            result = {"target": "Mnfr", "candidate": name, "output_directory": str(path),
                "model_identity": name, "label_names": list(LABELS), "label_to_id": {"All others": 0, "J&J": 1},
                "input_fields": list(INPUT_FIELDS), "sequence_length": 256, "config": {"seed": 42},
                "representation": {"combined_token_budget": 256}, "split_identity": {"allocation": "V4"},
                "loss_configuration": {"training_class_weights": None if name == "unweighted" else [1., 5.]},
                "initial_state_sha256": "same-initial-state", "final_state_sha256": _state_sha256(state),
                "checkpoint_sha256": sha256(checkpoint), "evaluation": {"macro_f1": score, "validation_record_ids_sha256": "same"}}
            for filename in ("encoder_config.json", "model_config.json", "training_manifest.json", "evaluation.json"):
                (path / filename).write_text("{}")
            (path / "candidate_result.json").write_text(json.dumps(result))
        selection = select_and_package(root)
        self.assertEqual(selection["selected_candidate"], "weighted_1_5")
        packaged = torch.load(root / "maria_handoff/checkpoint.pt", weights_only=True)
        self.assertEqual(packaged["target"], "Mnfr")
        self.assertEqual(packaged["label_names"], list(LABELS))
        self.assertEqual(packaged["loss_configuration"]["training_class_weights"], [1., 5.])
        self.assertEqual(packaged["input_fields"], list(INPUT_FIELDS))
        self.assertEqual(len(list((root / "maria_handoff").glob("*.pt"))), 1)


if __name__ == "__main__":
    unittest.main()
