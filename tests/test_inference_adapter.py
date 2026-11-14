"""Synthetic adapter/contract tests; no private data or pretrained downloads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from catalogiq.confidence_calibration import review_decision, temperature_scale
from catalogiq.features import sha256
from catalogiq.inference_adapter import infer_request, parse_request
from catalogiq.inference_engine import InferenceEngine, ProductRecord, TARGETS
from catalogiq.inference_models import ArtifactMismatch, FrozenModelLoader, MiniLMRoute, ordered_labels
from catalogiq.minilm_prediction_export import validate_predictions
from catalogiq.segment_transformer import ModelConfig, _make_classifier, _state_sha256
from catalogiq.target_minilm_preparation import INPUT_FIELDS
from test_segment_transformer import tiny_encoder


def product(record_id="001", name="Claritin RediTabs"):
    return {"record_id": record_id, **dict.fromkeys(INPUT_FIELDS, None), "ProductName": name,
            "ProductBrand": "Claritin", "ProductCategory": "Retailer allergy breadcrumb"}


class FakeLoader:
    def __init__(self):
        self.loads = []
        self.failures = {}

    def load(self, target):
        self.loads.append(target)
        if target in self.failures:
            raise self.failures[target]

        def predict(frame):
            # Deliberately unsorted rows and nonalphabetic vocabulary. Confidence
            # depends on the row so a positional association bug is observable.
            matrix = np.asarray([[.8, .2] if "Claritin" in name else [.3, .7]
                                 for name in frame.ProductName.fillna("")])
            cohorts = ["missing" if brand is None else "seen" if brand == "Claritin" else "unseen"
                       for brand in frame.ProductBrand] if target == "Brand" else [None] * len(frame)
            return matrix, cohorts

        return SimpleNamespace(labels=("Zeta", "Alpha"), temperature=1.7, provenance={"model_id": target},
                               predict=predict)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.loader = FakeLoader()
        self.engine = InferenceEngine("unused", loader=self.loader)

    def test_single_is_batch_of_one_and_repeated_loading_is_cached(self):
        row = product()
        single = infer_request(self.engine, row)
        self.assertEqual(single, infer_request(self.engine, {"products": [row]}))
        self.assertEqual(self.loader.loads, list(TARGETS))
        self.assertEqual(single["products"][0]["record_id"], "001")
        json.dumps(single, allow_nan=False)

    def test_batch_preserves_ids_order_and_prediction_confidence_alignment(self):
        rows = [product("z"), product("a", "Culturelle Daily Probiotic")]
        original = deepcopy(rows)
        results = infer_request(self.engine, {"products": rows})["products"]
        self.assertEqual([r["record_id"] for r in results], ["z", "a"])
        self.assertEqual(rows, original)
        for result, label, confidence in zip(results, ("Zeta", "Alpha"), (.8, .7)):
            self.assertTrue(result["review_required"])
            for target in TARGETS:
                prediction = result["predictions"][target]
                self.assertEqual(prediction["predicted_label"], label)
                self.assertEqual(prediction["confidence"], confidence)
                self.assertEqual(max(prediction["probabilities"].values()), confidence)
                self.assertEqual(prediction["label_mapping"], {"probability_0": "Zeta", "probability_1": "Alpha"})
                self.assertTrue(prediction["review_required"])
                self.assertIn("automatic_acceptance_disabled", prediction["review_reasons"])
            self.assertEqual(result["predictions"]["Platform"]["status"], "unavailable")
            self.assertEqual(result["predictions"]["Sub-Segment"]["selection_status"], "provisional")

    def test_existing_review_primitive_and_fixed_calibration_are_reused(self):
        row = product()
        row.update(ProductName=None, ProductCategory="Vitamins", ProductBrand=None, quarantined=True)
        result = infer_request(self.engine, row)["products"][0]["predictions"]["Brand"]
        expected = review_decision([.3, .7], "Alpha", ["Zeta", "Alpha"], None,
                                  row, enabled=False, lookup_status="missing", quarantined=True)
        self.assertEqual(result["review_reasons"], expected["review_reasons"])
        np.testing.assert_allclose(list(result["calibrated_probabilities"].values()),
                                   temperature_scale([[.3, .7]], 1.7)[0])
        self.assertEqual(result["confidence"], .7)

    def test_unseen_brand_and_provisional_reasons_propagate(self):
        row = product(); row["ProductBrand"] = "Never seen"
        results = infer_request(self.engine, row)["products"][0]["predictions"]
        self.assertEqual(results["Brand"]["lookup_status"], "unseen")
        self.assertIn("lookup_without_matching_training_key", results["Brand"]["review_reasons"])
        self.assertIn("provisional_model", results["Sub-Segment"]["review_reasons"])

    def test_bad_envelopes_and_correlation_keys_are_explicit(self):
        invalid = [None, [], {}, {"products": []}, {"products": "bad"}, {"products": [None]},
                   product(None), product(123), product(" "),
                   {"products": [product(), product()]}, {"products": [product()], "targets": ["Unknown"]},
                   {"products": [product()], "schema_version": "unknown"}, dict(product(), Mnfr="J&J")]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                infer_request(self.engine, request)
        self.assertEqual(self.loader.loads, [])

    def test_bad_features_fail_only_affected_row(self):
        for bad in ({"record_id": "bad"}, dict(product("bad"), ProductName=[]),
                    dict(product("bad"), ProductDescription=3), dict(product("bad"), quarantined="false")):
            with self.subTest(bad=bad):
                results = infer_request(self.engine, {"products": [bad, product()]})["products"]
                for target in TARGETS:
                    failed = results[0]["predictions"][target]
                    self.assertEqual(failed["status"], "failed")
                    self.assertIn("required_input_invalid", failed["review_reasons"])
                    self.assertIsNone(failed["confidence"])
                    self.assertEqual(results[1]["predictions"][target]["status"], "suggestion")

    def test_missing_mapping_and_direct_engine_validation(self):
        with self.assertRaises(ValueError):
            self.engine.predict([ProductRecord("x", {}), ProductRecord("x", {})])
        result = self.engine.predict([ProductRecord("x", None)])[0]
        self.assertIn("required_input_invalid", result.predictions[0].review_reasons)

    def test_target_load_failures_are_isolated_and_can_retry(self):
        for error, reason in [(FileNotFoundError("missing"), "model_unavailable"),
                              (ArtifactMismatch("checksum"), "artifact_mismatch"),
                              (RuntimeError("inference"), "prediction_failed")]:
            loader = FakeLoader(); loader.failures["Mnfr"] = error
            engine = InferenceEngine("unused", loader=loader)
            predictions = infer_request(engine, product())["products"][0]["predictions"]
            self.assertIn(reason, predictions["Mnfr"]["review_reasons"])
            for key in ("predicted_label", "confidence", "probabilities"):
                self.assertIsNone(predictions["Mnfr"][key])
            self.assertEqual(predictions["Segment"]["status"], "suggestion")
            loader.failures.clear()
            self.assertEqual(infer_request(engine, product())["products"][0]["predictions"]["Mnfr"]["status"], "suggestion")

    def test_invalid_matrix_or_row_count_never_becomes_success(self):
        infer_request(self.engine, product())
        route = self.engine._models["Segment"]
        for matrix in ([[np.nan, 0]], [[-.1, 1.1]], [[.6, .6]], [[1, 0, 0]], [[.4, .6], [.2, .8]]):
            route.predict = lambda frame, matrix=matrix: (matrix, [None])
            result = infer_request(self.engine, product())["products"][0]["predictions"]["Segment"]
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["predicted_label"])

    def test_registry_filters_approved_only_and_preserves_target_keys(self):
        relation = {"child_target": "Sub-Segment", "parent_target": "Segment", "status": "candidate",
                    "evidence": "synthetic", "allowed_pairs": [["Alpha", "Alpha"]]}
        self.assertEqual(self.engine.approved_relations, [])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            for status in ("candidate", "rejected", "approved"):
                relation["status"] = status
                path.write_text(json.dumps({"registry_version": "synthetic", "relations": [relation]}))
                engine = InferenceEngine("unused", loader=FakeLoader(), registry_path=path)
                results = infer_request(engine, product())["products"][0]["predictions"]
                self.assertEqual(len(engine.approved_relations), int(status == "approved"))
                for target in ("Sub-Segment", "Segment"):
                    self.assertEqual("invalid_approved_classification_relationship" in results[target]["review_reasons"],
                                     status == "approved")
                self.assertNotIn("invalid_approved_classification_relationship", results["Brand"]["review_reasons"])
                self.assertEqual(results["Segment"]["predicted_label"], "Zeta")

    def test_culturelle_taxonomy_does_not_manufacture_relationship_reasons(self):
        row = product(name="Culturelle Daily Probiotic")
        row["ProductCategory"] = "Vitamins, Minerals & Supplements"
        results = infer_request(self.engine, row)["products"][0]["predictions"]
        self.assertEqual(results["Segment"]["review_reasons"],
                         ["automatic_acceptance_disabled", "no_approved_threshold"])

    def test_numeric_mapping_order_and_existing_prediction_validator(self):
        labels = [f"Class {i}" for i in range(143)]
        mapping = {f"probability_{i}": labels[i] for i in reversed(range(143))}
        self.assertEqual(ordered_labels({"label_mapping": mapping, "label_names": labels}), tuple(labels))
        with self.assertRaises(ArtifactMismatch):
            ordered_labels({"label_mapping": {"probability_1": "a"}, "label_names": ["a"]})
        results = self.engine.predict(parse_request({"products": [product("z"), product("a", "Culturelle Daily Probiotic")]}))
        rows = [r.predictions[0] for r in results]
        frame = pd.DataFrame({"record_id": [r.record_id for r in results], "true_label": [r.predicted_label for r in rows],
                              "predicted_label": [r.predicted_label for r in rows], "confidence": [r.confidence for r in rows],
                              "probability_0": [r.probabilities[0] for r in rows], "probability_1": [r.probabilities[1] for r in rows]})
        context = frame[["record_id", "true_label"]].assign(group_id=["gz", "ga"], split="validation")
        self.assertEqual(validate_predictions(frame, context, ("Zeta", "Alpha"), ["protected"])["rows"], 2)


class TinyServingTokenizer:
    def __call__(self, texts, **kwargs):
        import torch
        assert kwargs == dict(truncation=True, max_length=256, padding=True, return_tensors="pt")
        return {"input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long)}


class ModelLoadingTests(unittest.TestCase):
    def test_tiny_checkpoint_strict_load_eval_and_repeated_forward(self):
        import torch
        torch.set_num_threads(2)
        config = replace(ModelConfig(), sequence_length=256, device="cpu", evaluation_batch_size=1)
        model = _make_classifier(torch, tiny_encoder(), 2, config.dropout)
        state = {"label_names": ["Zeta", "Alpha"], "model_config": config.to_dict(), "state_dict": model.state_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            torch.save(state, directory / "checkpoint.pt")
            route = MiniLMRoute(directory, config, ("Zeta", "Alpha"), _state_sha256(model.state_dict()), 1.2, {},
                                tokenizer=TinyServingTokenizer(), encoder_factory=tiny_encoder)
            self.assertFalse(route.model.training)
            frame = pd.DataFrame([product(), product("2", "Culturelle Daily Probiotic")])
            matrix, _ = route.predict(frame)
            np.testing.assert_array_equal(matrix, route.predict(frame)[0])
            self.assertEqual(matrix.shape, (2, 2))
            self.assertTrue(all(p.grad is None for p in route.model.parameters()))
            with self.assertRaises(ArtifactMismatch):
                MiniLMRoute(directory, config, ("Alpha", "Zeta"), _state_sha256(model.state_dict()), 1.2, {},
                            tokenizer=TinyServingTokenizer(), encoder_factory=tiny_encoder)

    def test_pinned_loader_checks_members_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.json"; path.write_text("original")
            loader = FrozenModelLoader(root)
            loader.pins = {"routes": {"Brand": {"files_sha256": {"model.json": sha256(path)}}}}
            path.write_text("changed")
            with self.assertRaises(ArtifactMismatch):
                loader.load("Brand")
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                loader.load("Brand")


if __name__ == "__main__":
    unittest.main()
