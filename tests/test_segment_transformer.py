"""Protocol and tiny offline fine-tuning checks, with no model download."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from catalogiq.segment_transformer import (
    MODEL_FIELDS, ModelConfig, _make_classifier, _state_sha256,
    build_product_texts, tokenize_corpus, train_transformer,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation, max_length, padding, return_attention_mask):
        inputs = []
        for text in texts:
            tokens = [1] + [2 + int(hashlib.sha256(word.encode()).hexdigest()[:4], 16) % 62
                            for word in text.split()] + [1]
            inputs.append(tokens[:max_length])
        return {"input_ids": inputs, "attention_mask": [[1] * len(row) for row in inputs],
                "token_type_ids": [[0] * len(row) for row in inputs]}

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_stub.json").write_text('{"test": true}', encoding="utf-8")


def tiny_encoder():
    import torch

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = torch.nn.Embedding(64, 8)
            self.projection = torch.nn.Linear(8, 8)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            return SimpleNamespace(last_hidden_state=torch.tanh(self.projection(self.embedding(input_ids))))

    return TinyEncoder()


class TransformerFixtures:
    def frame(self):
        return pd.DataFrame({
            "record_id": [f"record{i}" for i in range(6)],
            "ProductName": ["Mint capsules", "Lemon capsules", "Berry vitamins", "Green vitamins",
                            "Mint vitamins", "Berry capsules"],
            "ProductBrand": ["Willow"] * 6,
            "ProductDescription": ["<b>Useful</b> &amp; tidy\n text"] * 6,
            "ProductContents": ["Ingredient one"] * 6,
            "Segment": ["A", "B", "A", "B", "A", "B"],
            "Category": ["FORBIDDEN_CATEGORY"] * 6,
            "Class": ["FORBIDDEN_CLASS"] * 6,
            "ProductCategory": ["FORBIDDEN_RETAIL_CATEGORY"] * 6,
            "Sku": ["FORBIDDEN_ID"] * 6,
        })


class TransformerTextTests(TransformerFixtures, unittest.TestCase):
    def test_text_allowlist_html_and_frame_preservation(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        texts = build_product_texts(frame)
        self.assertTrue(texts[0].startswith("[ProductName] Mint capsules [ProductBrand] Willow"))
        self.assertIn("Useful & tidy text", texts[0])
        for marker in ("FORBIDDEN", "record0", "[Segment]", "[Category]", "<b>"):
            self.assertNotIn(marker, texts[0])
        self.assertEqual(tuple(name for name in MODEL_FIELDS),
                         ("ProductName", "ProductBrand", "ProductDescription", "ProductContents"))
        pd.testing.assert_frame_equal(frame, original)

    def test_labels_do_not_change_model_text_or_tokens(self):
        frame = self.frame()
        other = frame.copy()
        other["Segment"] = other["Segment"].map({"A": "B", "B": "A"})
        first = tokenize_corpus(frame, ["A", "B"], tokenizer=TinyTokenizer())
        second = tokenize_corpus(other, ["A", "B"], tokenizer=TinyTokenizer())
        self.assertEqual(first.text_sha256, second.text_sha256)
        self.assertEqual(first.token_sha256, second.token_sha256)
        np.testing.assert_array_equal(first.label_ids, 1 - second.label_ids)

    def test_character_limits_and_missing_fields(self):
        frame = pd.DataFrame({"ProductName": ["Longer product"], "ProductDescription": [None]})
        text = build_product_texts(frame, ModelConfig(field_character_limits=(4, 2, 2, 2)))[0]
        self.assertEqual(text, "[ProductName] Long")
        self.assertEqual(build_product_texts(pd.DataFrame(index=[0])), [""])
        text = build_product_texts(pd.DataFrame({"ProductName": ["Name"],
            "ProductDescription": ["<script>target='secret'</script>normal<style>label</style>"]}))[0]
        self.assertNotIn("secret", text)
        self.assertNotIn("label", text)
        self.assertIn("normal", text)

    def test_config_roundtrip_and_invalid_values(self):
        config = ModelConfig()
        self.assertEqual(ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)
        for kwargs in ({"sequence_length": 1}, {"epochs": 0}, {"batch_size": True},
                       {"learning_rate": float("nan")}, {"warmup_fraction": 1},
                       {"field_character_limits": (1, 2)}, {"precision": "bf16"},
                       {"device": "magic"}, {"revision": ""}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_duplicate_ids_missing_or_unknown_labels_are_rejected(self):
        for field, value in (("record_id", "record0"), ("Segment", None), ("Segment", "unknown")):
            frame = self.frame()
            frame.at[1, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                tokenize_corpus(frame, ["A", "B"], tokenizer=TinyTokenizer())
        with self.assertRaises(ValueError):
            tokenize_corpus(self.frame(), ["A", "A"], tokenizer=TinyTokenizer())


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch is not installed")
class TransformerFineTuningTests(TransformerFixtures, unittest.TestCase):
    def config(self):
        return ModelConfig(epochs=2, learning_rate=0.03, batch_size=2, evaluation_batch_size=2,
                           threads=1, device="cpu", warmup_fraction=0, log_every_steps=100)

    def run_tiny(self, output, frame=None, config=None, train=None, validation=None):
        config = config or self.config()
        corpus = tokenize_corpus(frame if frame is not None else self.frame(), ["A", "B"],
                                 config, tokenizer=TinyTokenizer())
        predictions, manifest = train_transformer(corpus, [0, 1, 2, 3] if train is None else train,
            [4, 5] if validation is None else validation, ["A", "B"], config, output,
            progress=None, encoder_factory=tiny_encoder)
        return predictions, manifest

    def test_tiny_training_reproduces_and_fine_tunes_encoder(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            first_predictions, first = self.run_tiny(Path(temporary) / "first")
            second_predictions, second = self.run_tiny(Path(temporary) / "second")
            pd.testing.assert_frame_equal(first_predictions, second_predictions, check_exact=True)
            self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
            self.assertEqual(first["final_state_sha256"], second["final_state_sha256"])
            self.assertNotEqual(first["initial_state_sha256"], first["final_state_sha256"])
            self.assertEqual(first["completed_epochs"], 2)
            self.assertEqual(first["optimizer_steps"], 4)
            self.assertGreater(first["encoder_trainable_parameters"], 0)
            checkpoint = torch.load(Path(temporary) / "first" / "checkpoint.pt", weights_only=True)
            torch.manual_seed(self.config().seed)
            initial_model = _make_classifier(torch, tiny_encoder(), 2, self.config().dropout)
            initial_encoder = initial_model.state_dict()["encoder.embedding.weight"]
            self.assertFalse(torch.equal(initial_encoder, checkpoint["state_dict"]["encoder.embedding.weight"]))
            self.assertEqual(_state_sha256(initial_model.state_dict()), first["initial_state_sha256"])
            np.testing.assert_allclose(first_predictions[["probability_0", "probability_1"]].sum(axis=1), 1)
            self.assertEqual(first_predictions["record_id"].tolist(), ["record4", "record5"])
            self.assertTrue((Path(temporary) / "first" / "tokenizer" / "tokenizer_stub.json").exists())
            history = pd.read_csv(Path(temporary) / "first" / "training_history.csv")
            self.assertEqual(history["epoch"].tolist(), [1, 2])

    def test_validation_labels_do_not_change_training_or_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_predictions, first = self.run_tiny(Path(temporary) / "first")
            changed = self.frame()
            changed.loc[[4, 5], "Segment"] = ["B", "A"]
            second_predictions, second = self.run_tiny(Path(temporary) / "second", frame=changed)
            self.assertEqual(first["final_state_sha256"], second["final_state_sha256"])
            pd.testing.assert_frame_equal(first_predictions.drop(columns="true_label"),
                                          second_predictions.drop(columns="true_label"), check_exact=True)
            self.assertEqual(second["checkpoint_selection"],
                             "fixed final epoch; validation is observational only")
            self.assertEqual(first["completed_epochs"], second["completed_epochs"])

    def test_initialization_is_identical_across_different_training_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, first = self.run_tiny(Path(temporary) / "first")
            _, second = self.run_tiny(Path(temporary) / "second", train=[0, 1, 4, 5], validation=[2, 3])
            self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
            self.assertEqual(first["train_class_counts"], second["train_class_counts"])
            self.assertEqual(first["validation_class_counts"], second["validation_class_counts"])

    def test_partition_overlap_repeated_and_out_of_range_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for train, validation in (([0, 1], [1, 2]), ([0, 0], [1]), ([6], [1]),
                                      ([], [1]), ([0.5], [1])):
                with self.subTest(train=train, validation=validation), self.assertRaises(ValueError):
                    self.run_tiny(Path(temporary), train=train, validation=validation)

    def test_completed_output_is_protected_and_fp16_cpu_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            self.run_tiny(path)
            with self.assertRaises(FileExistsError):
                self.run_tiny(path)
            with self.assertRaises(ValueError):
                self.run_tiny(Path(temporary) / "fp16", config=replace(self.config(), precision="fp16"))


if __name__ == "__main__":
    unittest.main()
