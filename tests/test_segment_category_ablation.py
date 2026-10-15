"""Category ablation uses development only and keeps its input contract explicit.

Public synthetic exports, the packaged tokenizer, and an injected trainer avoid
private data, downloads, model training, and final-test access in these tests.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.segment_category_ablation import (
    build_category_texts, load_category_development, main, run_experiment, tokenize_category_corpus,
)
from catalogiq.segment_frozen_development import load_frozen_development
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items, build_product_texts, tokenize_corpus
from catalogiq.split_protocol import DEFAULT_PROTOCOL, OfflineTokenizer, portable_model_view_contract


class HiddenTarget:
    def __str__(self):
        raise AssertionError("A classification target entered product text")


class PublicTokenizer(OfflineTokenizer):
    pad_token_id = 0


def verifier(directory, *, verify_exports):
    if verify_exports:
        raise AssertionError("Development requested the final-test export")
    return {"checks_passed": True}


def reseal(directory):
    names = ("train.csv", "validation.csv", "protected_final_test.csv", "grouping_freeze.json")
    completion = {"artifact_sha256": {name: sha256(directory / name) for name in names}}
    (directory / "completion.json").write_text(json.dumps(completion), encoding="utf8")


def synthetic_exports(directory):
    tokenizer = directory / "tokenizer"
    shutil.copytree(DEFAULT_PROTOCOL / "tokenizer", tokenizer)
    config = ModelConfig(device="cpu", epochs=1)
    for role, ids in (("train", list("abc")), ("validation", list("def"))):
        frame = pd.DataFrame({"record_id": ids, "group_id": ["g" + value for value in ids],
            "Segment": ["Allergy", "CCFS", "null"], "split": role,
            "ProductName": ["Alpha medicine", "Beta medicine", "Unknown medicine"],
            "ProductBrand": "Synthetic brand", "ProductDescription": "Synthetic evidence",
            "ProductContents": "", "Retailer": "Public synthetic store",
            "ProductCategory": ["Retailer > Vitamins & Minerals", "Retailer > Cold and Flu", "Unlabeled item"],
            "Category": "DO_NOT_USE_TARGET_CATEGORY", "Mnfr": "DO_NOT_USE_TARGET_MANUFACTURER"})
        # Category joins must use record IDs, not the physical export order.
        frame.iloc[::-1].to_csv(directory / (role + ".csv"), index=False, lineterminator="\n")
    pd.DataFrame({"record_id": ["x"], "group_id": ["gx"], "split": ["test"],
                  "ProductCategory": ["PRIVATE_FINAL_CATEGORY"], "Segment": ["PRIVATE_FINAL_LABEL"]}).to_csv(
                      directory / "protected_final_test.csv", index=False)
    (directory / "test.csv").write_bytes(b"NEVER READ OR HASH THIS FILE")
    (directory / "model_config.json").write_text(json.dumps(config.to_dict()), encoding="utf8")
    freeze = {"format": "catalogiq-portable-split-v1",
              "model_view_contract": portable_model_view_contract(config, tokenizer_dir=tokenizer)}
    (directory / "grouping_freeze.json").write_text(json.dumps(freeze), encoding="utf8")
    reseal(directory)
    return config


class SegmentCategoryAblationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = synthetic_exports(self.root)

    def development(self):
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier):
            return load_frozen_development(self.root, self.config)

    def test_category_formatter_excludes_targets_and_preserves_four_field_text(self):
        frame = pd.DataFrame({"record_id": ["a"], "ProductName": ["<b>Alpha</b> tablets"],
            "ProductBrand": ["Example"], "ProductDescription": ["Original description"],
            "ProductContents": ["Contents"],
            "ProductCategory": ["<script>wrong</script> Retailer &gt; Vitamins &amp; Minerals"],
            "Category": [HiddenTarget()], "Segment": [HiddenTarget()], "Mnfr": [HiddenTarget()]})
        before = frame.copy(deep=True)
        baseline = build_product_texts(frame, self.config)[0]
        actual = build_category_texts(frame, self.config)[0]
        self.assertEqual(actual, "[ProductCategory] Retailer > Vitamins & Minerals " + baseline)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(build_product_texts(frame, self.config)[0], baseline)

    def test_blank_category_does_not_change_baseline_text(self):
        frame = pd.DataFrame({"ProductName": ["Alpha"] * 5,
                              "ProductCategory": ["", " null ", None, np.nan, "none"]})
        self.assertEqual(build_category_texts(frame, self.config), build_product_texts(frame, self.config))

    def test_category_limit_is_explicit_and_invalid_limits_are_rejected(self):
        frame = pd.DataFrame({"ProductName": ["Alpha"], "ProductCategory": ["X" * 600]})
        actual = build_category_texts(frame, self.config, category_character_limit=12)[0]
        self.assertEqual(actual, "[ProductCategory] " + "X" * 12 + " " + build_product_texts(frame, self.config)[0])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                build_category_texts(frame, self.config, category_character_limit=invalid)

    def test_product_category_is_visible_with_a_full_128_token_product_name(self):
        frame = pd.DataFrame({"record_id": ["a", "b"], "Segment": ["A", "B"],
            "ProductName": ["hello " * 170, "short product"],
            "ProductCategory": ["Mineral remedies", "Cold treatment"],
            "Category": [HiddenTarget(), HiddenTarget()]})
        before = frame.copy(deep=True)
        tokenizer = PublicTokenizer(DEFAULT_PROTOCOL / "tokenizer")
        baseline = tokenize_corpus(frame, ["A", "B"], self.config, tokenizer=tokenizer)
        category = tokenize_category_corpus(frame, ["A", "B"], self.config, tokenizer=tokenizer)
        self.assertEqual(len(baseline.input_ids[0]), 128)
        self.assertEqual(len(category.input_ids[0]), 128)
        prefix = tokenizer(["[ProductCategory] Mineral remedies"], max_length=128)["input_ids"][0][:-1]
        self.assertEqual(category.input_ids[0][:len(prefix)].tolist(), prefix)
        self.assertNotEqual(baseline.token_sha256, category.token_sha256)
        self.assertEqual(category.record_ids, baseline.record_ids)
        np.testing.assert_array_equal(category.label_ids, baseline.label_ids)
        pd.testing.assert_frame_equal(frame, before)

    def test_product_category_is_visible_with_the_shared_256_token_budget(self):
        config = replace(self.config, sequence_length=256)
        frame = pd.DataFrame({"record_id": ["a", "b"], "Segment": ["A", "B"],
            "ProductName": ["hello " * 400, "short product"],
            "ProductDescription": ["medicine " * 400, "short description"],
            "ProductCategory": ["Mineral remedies", "Cold treatment"],
            "Category": [HiddenTarget(), HiddenTarget()]})
        before = frame.copy(deep=True)
        tokenizer = PublicTokenizer(DEFAULT_PROTOCOL / "tokenizer")
        baseline = tokenize_corpus(frame, ["A", "B"], config, tokenizer=tokenizer)
        category = tokenize_category_corpus(frame, ["A", "B"], config, tokenizer=tokenizer)
        self.assertEqual(baseline.sequence_length, 256)
        self.assertEqual(category.sequence_length, 256)
        self.assertEqual(len(baseline.input_ids[0]), 256)
        self.assertEqual(len(category.input_ids[0]), 256)
        prefix = tokenizer(["[ProductCategory] Mineral remedies"], max_length=256)["input_ids"][0][:-1]
        self.assertEqual(category.input_ids[0][:len(prefix)].tolist(), prefix)
        self.assertNotEqual(baseline.token_sha256, category.token_sha256)
        self.assertEqual(category.record_ids, baseline.record_ids)
        np.testing.assert_array_equal(category.label_ids, baseline.label_ids)
        pd.testing.assert_frame_equal(frame, before)

    def test_category_tokenization_is_reproducible_and_keeps_record_order(self):
        frame = pd.DataFrame({"record_id": ["b", "a"], "Segment": ["B", "A"],
                              "ProductName": ["Beta", "Alpha"], "ProductCategory": ["Flu", "Vitamins"]})
        tokenizer = PublicTokenizer(DEFAULT_PROTOCOL / "tokenizer")
        first = tokenize_category_corpus(frame, ["A", "B"], self.config, tokenizer=tokenizer)
        second = tokenize_category_corpus(frame, ["A", "B"], self.config, tokenizer=tokenizer)
        self.assertEqual(first.record_ids, ("b", "a"))
        self.assertEqual(first.label_ids.tolist(), [1, 0])
        self.assertEqual(first.text_sha256, second.text_sha256)
        self.assertEqual(first.token_sha256, second.token_sha256)
        self.assertEqual(first.text_sha256, _hash_items(build_category_texts(frame, self.config)))
        shuffled = tokenize_category_corpus(frame.iloc[::-1], ["A", "B"], self.config, tokenizer=tokenizer)
        self.assertEqual(dict(zip(first.record_ids, [row.tolist() for row in first.input_ids])),
                         dict(zip(shuffled.record_ids, [row.tolist() for row in shuffled.input_ids])))
        empty = frame.assign(ProductCategory="null")
        category_empty = tokenize_category_corpus(empty, ["A", "B"], self.config, tokenizer=tokenizer)
        baseline_empty = tokenize_corpus(empty, ["A", "B"], self.config, tokenizer=tokenizer)
        self.assertEqual(category_empty.text_sha256, baseline_empty.text_sha256)
        self.assertEqual(category_empty.token_sha256, baseline_empty.token_sha256)

    def test_loader_aligns_only_supervised_categories_and_never_reads_the_test(self):
        real_read, real_sha = pd.read_csv, sha256

        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            if Path(path).name == "protected_final_test.csv":
                self.assertEqual(kwargs.get("usecols"), ["record_id", "group_id"])
            return real_read(path, *args, **kwargs)

        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return real_sha(path)

        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pd.read_csv", side_effect=guarded_read), \
                patch("catalogiq.segment_category_ablation.sha256", side_effect=guarded_hash), \
                patch("catalogiq.segment_frozen_development.sha256", side_effect=guarded_hash):
            loaded = load_category_development(self.root, self.config)
        self.assertEqual(loaded.frame.record_id.tolist(), list("abde"))
        self.assertEqual(loaded.frame.ProductCategory.tolist(),
            ["Retailer > Vitamins & Minerals", "Retailer > Cold and Flu"] * 2)
        self.assertEqual(loaded.train_indices.tolist(), [0, 1])
        self.assertEqual(loaded.validation_indices.tolist(), [2, 3])
        self.assertNotIn("Category", loaded.frame)
        self.assertNotIn("Mnfr", loaded.frame)
        self.assertFalse(set(loaded.frame.record_id) & loaded.protected_record_ids)

    def test_sequence_override_verifies_the_original_sealed_128_token_contract(self):
        before = {name: sha256(self.root / name)
                  for name in ("model_config.json", "grouping_freeze.json", "train.csv", "validation.csv")}
        for config in (None, replace(self.config, sequence_length=256),
                       replace(self.config, sequence_length=512)):
            with self.subTest(config=config):
                with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                        patch("catalogiq.segment_category_ablation.load_frozen_development",
                              wraps=load_frozen_development) as guarded:
                    loaded = load_category_development(self.root, config)
                guarded.assert_called_once_with(self.root, self.config)
                self.assertEqual(loaded.audit["sealed_sequence_length"], 128)
                self.assertEqual(loaded.audit["experimental_sequence_length"],
                                 256 if config is None else config.sequence_length)
                self.assertTrue(loaded.audit["frozen_model_contract_unchanged"])
        self.assertEqual(before, {name: sha256(self.root / name) for name in before})

    def test_sequence_override_does_not_relax_other_frozen_representation_settings(self):
        config = replace(self.config, sequence_length=256)
        altered = (replace(config, revision="different-pinned-revision"),
                   replace(config, model_name="different-public-model"),
                   replace(config, field_character_limits=(1023, 512, 4000, 2000)))
        for invalid in altered:
            with self.subTest(config=invalid), \
                    patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier):
                with self.assertRaisesRegex(ValueError, "representation|contract|frozen|Frozen"):
                    load_category_development(self.root, invalid)

    def test_missing_product_category_cannot_be_replaced_with_category_target(self):
        frame = pd.read_csv(self.root / "train.csv", dtype=str, keep_default_na=False)
        frame.drop(columns="ProductCategory").to_csv(self.root / "train.csv", index=False)
        reseal(self.root)
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier):
            with self.assertRaisesRegex(ValueError, "ProductCategory"):
                load_category_development(self.root, self.config)

    def test_category_population_mismatch_and_duplicates_are_rejected(self):
        base = self.development()
        missing = base.frame.copy()
        missing.loc[0, "record_id"] = "outside_population"
        with patch("catalogiq.segment_category_ablation.load_frozen_development", return_value=replace(base, frame=missing)):
            with self.assertRaises(ValueError):
                load_category_development(self.root, self.config)
        exported = pd.read_csv(self.root / "train.csv", dtype=str, keep_default_na=False)
        pd.concat([exported, exported.iloc[[0]]], ignore_index=True).to_csv(self.root / "train.csv", index=False)
        reseal(self.root)
        with patch("catalogiq.segment_category_ablation.load_frozen_development", return_value=base):
            with self.assertRaises(ValueError):
                load_category_development(self.root, self.config)

    def test_category_rejoin_rejects_changed_export_bytes_after_guarded_loading(self):
        base = self.development()
        frame = pd.read_csv(self.root / "train.csv", dtype=str, keep_default_na=False)
        frame.loc[0, "ProductCategory"] = "Changed after the sealed loader returned"
        frame.to_csv(self.root / "train.csv", index=False)
        with patch("catalogiq.segment_category_ablation.load_frozen_development", return_value=base):
            with self.assertRaisesRegex(ValueError, "changed|modified|checksum|seal"):
                load_category_development(self.root, self.config)

    def fake_trainer(self, calls, *, changed_initialization=False, wrong_population=False, wrong_membership=False):
        """Return trainer-shaped artifacts, without importing any ML framework."""
        def train(corpus, train_indices, validation_indices, names, config, directory, progress=None):
            calls.append({"record_ids": corpus.record_ids, "train": np.asarray(train_indices).copy(),
                          "validation": np.asarray(validation_indices).copy(), "config": config,
                          "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
                          "sequence_length": corpus.sequence_length})
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            true = [names[int(corpus.label_ids[index])] for index in validation_indices]
            predicted = pd.DataFrame({"record_id": [corpus.record_ids[index] for index in validation_indices],
                                      "true_label": true, "predicted_label": true, "confidence": .9})
            for class_index, name in enumerate(names):
                predicted[f"probability_{class_index}"] = [.9 if label == name else .1 / (len(names) - 1) for label in true]
            if wrong_population:
                predicted.loc[0, "record_id"] = "x"  # Protected test identity.
            predicted.to_csv(directory / "predictions.csv", index=False)
            manifest = {"config": config.to_dict(), "label_names": list(names),
                "input_fields": list(MODEL_FIELDS), "target": "Segment",
                "initial_state_sha256": ("2" if changed_initialization and len(calls) == 2 else "1") * 64,
                "train_record_ids_sha256": _hash_items([corpus.record_ids[index] for index in train_indices]),
                "validation_record_ids_sha256": _hash_items([corpus.record_ids[index] for index in validation_indices]),
                "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
                "training_rows": len(train_indices), "validation_rows": len(validation_indices), "runtime_seconds": 0.0}
            if wrong_membership:
                manifest["train_record_ids_sha256"] = "0" * 64
            (directory / "initialization.json").write_text(json.dumps(manifest), encoding="utf8")
            (directory / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            (directory / "checkpoint.pt").write_bytes(b"Synthetic placeholder; contains no model weights.")
            return predicted, manifest
        return train

    def test_both_arms_use_identical_rows_initialization_and_explicit_field_metadata(self):
        calls = []
        config = replace(self.config, sequence_length=256)
        original = {name: sha256(self.root / name)
                    for name in ("train.csv", "validation.csv", "grouping_freeze.json", "model_config.json")}
        output = self.root / "ablation"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=self.fake_trainer(calls)):
            result = run_experiment(self.root, output, config, bootstrap_samples=10, progress=lambda _: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["record_ids"], tuple("abde"))
        self.assertEqual(calls[0]["record_ids"], calls[1]["record_ids"])
        np.testing.assert_array_equal(calls[0]["train"], calls[1]["train"])
        np.testing.assert_array_equal(calls[0]["validation"], calls[1]["validation"])
        self.assertEqual(calls[0]["train"].tolist(), [0, 1])
        self.assertEqual(calls[0]["validation"].tolist(), [2, 3])
        self.assertEqual(calls[0]["config"], calls[1]["config"])
        self.assertEqual(calls[0]["config"].sequence_length, 256)
        self.assertEqual([call["sequence_length"] for call in calls], [256, 256])
        self.assertNotEqual(calls[0]["text_sha256"], calls[1]["text_sha256"])
        self.assertNotEqual(calls[0]["token_sha256"], calls[1]["token_sha256"])
        self.assertTrue(result["same_initial_weights"])
        self.assertTrue(result["same_training_validation_membership"])
        self.assertFalse(result["final_test_read_or_scored"])
        self.assertFalse(result["representation"]["final_test_representation_certified"])
        self.assertEqual(result["representation"]["combined_token_limit"], 256)
        self.assertEqual(result["representation"]["sealed_grouping_token_limit"], 128)
        for arm, fields in (("baseline", list(MODEL_FIELDS)), ("with_category", ["ProductCategory", *MODEL_FIELDS])):
            for filename in ("initialization.json", "training_manifest.json"):
                manifest = json.loads((output / arm / filename).read_text(encoding="utf8"))
                self.assertEqual(manifest["input_fields"], fields)
                self.assertEqual(manifest["trainer_default_input_fields"], list(MODEL_FIELDS))
                self.assertEqual(manifest["config"]["sequence_length"], 256)
            predictions = pd.read_csv(output / arm / "predictions.csv", dtype={"record_id": str})
            self.assertEqual(set(predictions.record_id), set("de"))
        self.assertEqual(original, {name: sha256(self.root / name) for name in original})

    def test_experiment_rejects_nonidentical_initial_weights_and_changed_membership(self):
        for name, trainer_options, pattern in (("initialization", {"changed_initialization": True}, "identical weights"),
                                               ("membership", {"wrong_membership": True}, "membership"),
                                               ("protected", {"wrong_population": True}, "population|Protected")):
            with self.subTest(case=name):
                calls = []
                output = self.root / name
                with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                        patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                        patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                        patch("catalogiq.segment_category_ablation.train_transformer", side_effect=self.fake_trainer(calls, **trainer_options)):
                    with self.assertRaisesRegex(ValueError, pattern):
                        run_experiment(self.root, output, self.config, bootstrap_samples=10, progress=lambda _: None)
                self.assertFalse((output / "completion.json").exists())

    def test_prepare_only_freezes_actual_corpus_metadata_without_training(self):
        output = self.root / "prepared"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=AssertionError("Training was requested")):
            protocol = run_experiment(self.root, output, bootstrap_samples=10,
                                      prepare_only=True, progress=lambda _: None)
        self.assertEqual(protocol["training_rows"], 2)
        self.assertEqual(protocol["validation_rows"], 2)
        self.assertEqual(protocol["model_config"]["sequence_length"], 256)
        self.assertEqual(protocol["development_audit"]["sealed_sequence_length"], 128)
        self.assertEqual(protocol["development_audit"]["experimental_sequence_length"], 256)
        self.assertFalse(protocol["final_test_read_or_scored"])
        self.assertEqual(protocol["representation"]["category_fields"], ["ProductCategory", *MODEL_FIELDS])
        self.assertNotEqual(protocol["tokenization"]["baseline"]["text_sha256"],
                            protocol["tokenization"]["with_category"]["text_sha256"])
        freeze = json.loads((output / "protocol_freeze.json").read_text(encoding="utf8"))
        self.assertTrue(freeze["frozen_before_training"])
        self.assertEqual(freeze["protocol_sha256"], sha256(output / "protocol.json"))

    def test_command_defaults_to_256_and_honors_explicit_sequence_configuration(self):
        original = sha256(self.root / "model_config.json")
        cases = (([], 256), (["--sequence-length", "128"], 128),
                 (["--model-config", str(self.root / "model_config.json")], 128),
                 (["--model-config", str(self.root / "model_config.json"), "--sequence-length", "512"], 512))
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), \
                    patch("catalogiq.segment_category_ablation.run_experiment") as run:
                main(["--split-dir", str(self.root), "--output-dir", str(self.root / "command-output"),
                      "--prepare-only", *arguments])
                run.assert_called_once()
                split_dir, output_dir, config = run.call_args.args
                self.assertEqual(split_dir, self.root)
                self.assertEqual(output_dir, self.root / "command-output")
                self.assertEqual(config, replace(self.config, sequence_length=expected))
                self.assertTrue(run.call_args.kwargs["prepare_only"])
        self.assertEqual(sha256(self.root / "model_config.json"), original)

    def test_resume_reuses_only_completed_arms_and_rejects_config_drift(self):
        output = self.root / "resumable"
        calls = []
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=self.fake_trainer(calls)):
            first = run_experiment(self.root, output, self.config, bootstrap_samples=10, progress=lambda _: None)
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=AssertionError("Completed arm was retrained")):
            resumed = run_experiment(self.root, output, self.config, bootstrap_samples=10, resume=True, progress=lambda _: None)
            with self.assertRaisesRegex(ValueError, "Resume"):
                run_experiment(self.root, output, replace(self.config, learning_rate=1e-5),
                               bootstrap_samples=10, resume=True, progress=lambda _: None)
        for arm, metrics in first["validation"].items():
            for name, value in metrics.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, resumed["validation"][arm][name], places=12)
                else:
                    self.assertEqual(value, resumed["validation"][arm][name])


if __name__ == "__main__":
    unittest.main()
