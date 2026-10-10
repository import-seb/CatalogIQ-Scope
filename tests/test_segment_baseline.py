"""Synthetic, offline checks; no private records, downloads or test-set scoring."""
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from catalogiq.cli import segment_baseline_main
from catalogiq.features import sha256
from catalogiq.model_input_groups import model_view_contract
from catalogiq.segment_baseline import (BaselineConfig, fit_baseline, run_baseline,
                                       token_documents)
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, tokenize_corpus


class TinyTokenizer:
    """Fixed external vocabulary: validation-only token has an ID but no learned TF-IDF feature."""
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        inputs = []
        for text in texts:
            ids = [101] + [10 if word == "allergy" else 20 if word == "cold" else
                           999 if word == "validationonly" else 30 for word in text.split()] + [102]
            inputs.append(ids[:kwargs["max_length"]])
        return {"input_ids": inputs, "attention_mask": [[1] * len(ids) for ids in inputs]}


def corpus():
    frame = pd.DataFrame({"record_id": list("abcdef"), "Segment": ["Allergy", "CCFS"] * 3,
                          "ProductName": ["allergy", "cold", "allergy", "cold",
                                          "allergy validationonly", "cold validationonly"]})
    return tokenize_corpus(frame, ("Allergy", "CCFS"), tokenizer=TinyTokenizer())


def snapshot(directory):
    """Minimal synthetic snapshot verified through the production loader/seal code."""
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text('{"fixture": true}', encoding="utf-8")
    model_config = directory / "model_config.json"
    model_config.write_text(json.dumps(ModelConfig().to_dict()), encoding="utf-8")
    frozen = directory / "frozen"
    frozen.mkdir()
    names = ("Allergy", "CCFS", "Digestive Health", "External Analgesics", "Internal Analgesics",
             "Lifestyle CHC", "Other Self Care", "Vitamins, Minerals & Supplements")
    assignments = []
    for role in ("train", "validation"):
        rows = [{"record_id": f"{role}-{i}", "group_id": f"group-{role}-{i}", "Segment": label,
                 "split": role, "ProductName": "allergy" if i % 2 else "cold",
                 **{field: "" for field in MODEL_FIELDS[1:]},
                 "Category": "FORBIDDEN_LABEL", "MDM_InsertDateTime": "FORBIDDEN_DATE"}
                for i, label in enumerate((*names, "", "null"))]
        pd.DataFrame(rows).to_csv(frozen / f"{role}.csv", index=False)
        assignments.extend({key: row[key] for key in ("record_id", "group_id", "split")} for row in rows)
    pd.DataFrame({"record_id": ["protected"], "group_id": ["protected-group"]}).to_csv(
        frozen / "protected_final_test.csv", index=False)
    (frozen / "test.csv").write_text("NEVER OPEN THIS", encoding="utf-8")
    assignments.append({"record_id": "protected", "group_id": "protected-group", "split": "test"})
    pd.DataFrame(assignments).to_csv(frozen / "assignments.csv", index=False)
    cache = directory / "model_view"
    cache.mkdir()
    pd.DataFrame(assignments)[["record_id", "group_id"]].to_csv(cache / "record_signatures.csv", index=False)
    (frozen / "summary.json").write_text('{"synthetic_fixture": true}', encoding="utf-8")
    (frozen / "grouping_freeze.json").write_text(json.dumps({
        "model_view_contract": model_view_contract(tokenizer_dir=tokenizer),
        "model_input_cache": str(cache), "source_sha256": {},
        "input_sha256": {str(cache / "record_signatures.csv"): sha256(cache / "record_signatures.csv")}}), encoding="utf-8")
    reseal(frozen)
    return frozen, tokenizer, model_config


def reseal(directory):
    (directory / "completion.json").write_text(json.dumps({"artifact_sha256": {
        name: sha256(directory / name) for name in ("grouping_freeze.json", "train.csv", "validation.csv",
                                                   "protected_final_test.csv", "assignments.csv", "summary.json")}}), encoding="utf-8")


class SegmentBaselineTests(unittest.TestCase):
    def test_train_only_vocabulary_and_idf_and_persisted_predictions(self):
        data = corpus()
        model, predictions, metrics = fit_baseline(data, [0, 1, 2, 3], [4, 5], BaselineConfig(min_df=1))
        tfidf = model.named_steps["tfidf"]
        self.assertNotIn("t999", tfidf.vocabulary_)
        # t10 occurs in 2 of 4 training documents, irrespective of validation.
        self.assertAlmostEqual(tfidf.idf_[tfidf.vocabulary_["t10"]], np.log(5 / 3) + 1)
        self.assertEqual(predictions.record_id.tolist(), ["e", "f"])
        self.assertEqual(metrics["baseline"]["records"], 2)
        self.assertEqual(metrics["majority_class_selected_on_training"], "Allergy")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            probabilities = restored.predict_proba(token_documents(data)[4:])
        np.testing.assert_allclose(probabilities, predictions[["probability_0", "probability_1"]])

    def test_fixed_seed_repeats_predictions(self):
        first = fit_baseline(corpus(), [0, 1, 2, 3], [4, 5], BaselineConfig(min_df=1))[1]
        second = fit_baseline(corpus(), [0, 1, 2, 3], [4, 5], BaselineConfig(min_df=1))[1]
        pd.testing.assert_frame_equal(first, second)

    def test_metadata_and_other_labels_cannot_enter_tokens(self):
        frame = pd.DataFrame({"record_id": ["a", "b"], "Segment": ["Allergy", "CCFS"],
                              "ProductName": ["allergy", "cold"], "Category": ["validationonly"] * 2,
                              "Brand": ["validationonly"] * 2, "MDM_InsertDateTime": ["validationonly"] * 2})
        data = tokenize_corpus(frame, ("Allergy", "CCFS"), tokenizer=TinyTokenizer())
        self.assertNotIn("t999", " ".join(token_documents(data)))
        # Missing declared fields are explicit empty strings in the shared builder.
        self.assertEqual(len(data), 2)

    def test_rejects_overlap_absent_training_class_and_invalid_sequences(self):
        data = corpus()
        for training, validation in (([0, 1], [1, 4]), ([0, 2], [4, 5]), ([0, 0, 1], [4, 5])):
            with self.subTest(training=training, validation=validation), self.assertRaises(ValueError):
                fit_baseline(data, training, validation, BaselineConfig(min_df=1))
        broken = replace(data, input_ids=(np.array([-1]), *data.input_ids[1:]))
        with self.assertRaises(ValueError):
            token_documents(broken)

    def test_rejects_bad_config(self):
        for values in ({"c": 0}, {"c": float("nan")}, {"min_df": True}, {"max_iter": 0},
                       {"ngram_max": 4}, {"seed": -1}, {"class_weight": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BaselineConfig(**values)

    def test_missing_snapshot_fails_before_output_or_tokenizer_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(FileNotFoundError, "Do not substitute"):
                run_baseline(root / "missing", root / "tokenizer", root / "config.json", root / "out")
            self.assertFalse((root / "out").exists())

    def test_cli_reports_missing_shared_data_without_traceback(self):
        with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", new_callable=io.StringIO) as error:
            with self.assertRaises(SystemExit) as result:
                segment_baseline_main(["--split-dir", str(Path(folder) / "frozen"),
                                       "--tokenizer-dir", str(Path(folder) / "tokenizer"),
                                       "--model-config", str(Path(folder) / "missing.json"),
                                       "--output-dir", str(Path(folder) / "out")])
            self.assertEqual(result.exception.code, 1)
            self.assertIn("Frozen development inputs", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())

    def test_complete_run_keeps_test_closed_and_exports_maria_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frozen, tokenizer, config = snapshot(root)
            real_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.name == "test.csv":
                    raise AssertionError("Final test was opened")
                return real_open(path, *args, **kwargs)
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyTokenizer())
            with patch.dict("sys.modules", {"transformers": fake_transformers}), patch.object(Path, "open", guarded_open):
                summary = run_baseline(frozen, tokenizer, config, root / "run", BaselineConfig(min_df=1), progress=None)
            predictions = pd.read_csv(root / "run/predictions.csv")
            self.assertEqual(len(predictions), 8)
            self.assertTrue(predictions.split.eq("validation").all())
            self.assertEqual(summary["training_rows"], 8)
            self.assertFalse(summary["final_test_read_or_scored"])
            self.assertFalse(summary["new_splits_created"])
            np.testing.assert_allclose(predictions.filter(like="probability_").sum(axis=1), 1)
            self.assertTrue(set(predictions.record_id).isdisjoint({"protected"}))
            self.assertEqual(pd.read_csv(root / "run/confusion_matrix.csv", index_col=0).shape, (8, 8))
            for name, expected in summary["artifact_sha256"].items():
                self.assertEqual(sha256(root / "run" / name), expected)
            with self.assertRaises(FileExistsError):
                run_baseline(frozen, tokenizer, config, root / "run")

    def test_changed_tokenizer_and_group_overlap_stop_before_fit(self):
        for fault in ("tokenizer", "group_overlap"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                frozen, tokenizer, config = snapshot(root)
                if fault == "tokenizer":
                    (tokenizer / "tokenizer.json").write_text("changed", encoding="utf-8")
                else:
                    validation = pd.read_csv(frozen / "validation.csv", keep_default_na=False)
                    validation.loc[0, "group_id"] = "group-train-0"
                    validation.to_csv(frozen / "validation.csv", index=False)
                    reseal(frozen)
                with patch("catalogiq.segment_baseline.fit_baseline", side_effect=AssertionError("Fit called")):
                    with self.assertRaises(ValueError):
                        run_baseline(frozen, tokenizer, config, root / "out", progress=None)
                self.assertFalse((root / "out").exists())

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Install [baseline] for real offline tokenizer check")
    def test_real_saved_tokenizer_and_full_seal_without_network(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frozen, tokenizer, config = snapshot(root)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "allergy": 1, "cold": 2, "ProductName": 3}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(tokenizer)
            freeze = json.loads((frozen / "grouping_freeze.json").read_text(encoding="utf-8"))
            freeze["model_view_contract"] = model_view_contract(tokenizer_dir=tokenizer)
            (frozen / "grouping_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
            reseal(frozen)
            with patch("socket.socket.connect", side_effect=AssertionError("Network access")):
                summary = run_baseline(frozen, tokenizer, config, root / "real", BaselineConfig(min_df=1), progress=None)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["validation_rows"], 8)


if __name__ == "__main__":
    unittest.main()
