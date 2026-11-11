"""Opt-in integration against private frozen artifacts, using synthetic products.

No validation/training/protected dataset is opened. No expected class or score
is invented; check consistency, safeguards, and immutable serving members.
"""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from catalogiq.features import sha256
from catalogiq.inference_adapter import infer_request
from catalogiq.inference_engine import InferenceEngine, TARGETS
from catalogiq.inference_models import FrozenModelLoader


@unittest.skipUnless(os.environ.get("CATALOGIQ_FROZEN_INFERENCE_TESTS") == "1",
                     "Set CATALOGIQ_FROZEN_INFERENCE_TESTS=1 with the private serving artifacts")
class FrozenInferenceTests(unittest.TestCase):
    def test_real_routes_batch_single_repeat_and_unchanged_artifacts(self):
        import torch
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        request = json.loads((root / "docs/integration/inference_adapter_request.json").read_text())
        loader = FrozenModelLoader(root / "data/processed")
        engine = InferenceEngine(root / "data/processed", loader=loader)
        batch = infer_request(engine, request)
        self.assertEqual(engine.approved_relations, [])
        cached = {target: id(route) for target, route in engine._models.items()}
        self.assertEqual(set(cached), set(TARGETS))
        repeated = infer_request(engine, request)
        for index, row in enumerate(request["products"]):
            single = infer_request(engine, row)["products"][0]
            result = batch["products"][index]
            self.assertEqual(result["record_id"], row["record_id"])
            for target in TARGETS:
                prediction = result["predictions"][target]
                self.assertEqual(prediction["status"], "suggestion", prediction)
                self.assertEqual(prediction["predicted_label"], single["predictions"][target]["predicted_label"])
                probabilities = list(prediction["probabilities"].values())
                np.testing.assert_allclose(probabilities, list(single["predictions"][target]["probabilities"].values()), atol=1e-5, rtol=0)
                np.testing.assert_array_equal(probabilities, list(repeated["products"][index]["predictions"][target]["probabilities"].values()))
                self.assertAlmostEqual(prediction["confidence"], max(probabilities))
                winner = f"probability_{int(np.argmax(probabilities))}"
                self.assertEqual(prediction["predicted_label"], prediction["label_mapping"][winner])
                self.assertIn("automatic_acceptance_disabled", prediction["review_reasons"])
                self.assertIn("no_approved_threshold", prediction["review_reasons"])
                self.assertFalse(prediction["provenance"]["policy_approved"])
                self.assertTrue(prediction["review_required"])
                self.assertFalse(any("relationship" in reason for reason in prediction["review_reasons"]))
        self.assertEqual(cached, {target: id(route) for target, route in engine._models.items()})
        for spec in loader.pins["routes"].values():
            for relative, digest in spec["files_sha256"].items():
                self.assertEqual(sha256(loader.processed_dir / relative), digest)


if __name__ == "__main__":
    unittest.main()
