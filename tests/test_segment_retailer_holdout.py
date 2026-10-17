"""Retailer generalization uses sealed development records, never final-test data.

Public synthetic exports, an offline public tokenizer, and trainer-shaped fake
artifacts exercise the real selection and orchestration without model training.
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
from catalogiq.segment_category_ablation import load_category_development, run_experiment as run_category_experiment
from catalogiq.segment_retailer_holdout import ARMS, DEFAULT_HELD_OUT, main, run_experiment, select_retailer_development
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items
from catalogiq.split_protocol import DEFAULT_PROTOCOL, OfflineTokenizer, portable_model_view_contract


class PublicTokenizer(OfflineTokenizer):
    pad_token_id = 0


def verifier(directory, *, verify_exports):
    if verify_exports:
        raise AssertionError("Retailer development requested the final-test export")
    return {"checks_passed": True}


def reseal(directory):
    names = ("train.csv", "validation.csv", "protected_final_test.csv", "grouping_freeze.json")
    completion = {"artifact_sha256": {name: sha256(directory / name) for name in names}}
    (directory / "completion.json").write_text(json.dumps(completion), encoding="utf8")


def synthetic_exports(directory):
    tokenizer = directory / "tokenizer"
    shutil.copytree(DEFAULT_PROTOCOL / "tokenizer", tokenizer)
    config = ModelConfig(device="cpu", epochs=3)
    rows = []
    for retailer_index, retailer in enumerate(("Amazon", *DEFAULT_HELD_OUT)):
        for role in ("train", "validation"):
            count = 4 if role == "train" and retailer == "Amazon" else 2
            for index in range(count):
                identity = f"{role}-{retailer_index}-{index}"
                rows.append({"record_id": identity, "group_id": "group-" + identity,
                    "Segment": ("Allergy", "CCFS")[index % 2], "split": role,
                    "ProductName": f"Synthetic medicine {identity}", "ProductBrand": "Public brand",
                    "ProductDescription": f"Description for {identity}", "ProductContents": "",
                    "ProductCategory": f"{retailer} > {'Allergy care' if index % 2 == 0 else 'Cold care'}",
                    "Retailer": retailer, "Category": "FORBIDDEN_TARGET_CATEGORY",
                    "Mnfr": "FORBIDDEN_TARGET_MANUFACTURER"})
    for role in ("train", "validation"):
        identity = role + "-unlabeled"
        rows.append({"record_id": identity, "group_id": "group-" + identity, "Segment": "null",
            "split": role, "ProductName": "Unlabeled public item", "ProductBrand": "Public brand",
            "ProductDescription": "Missing target is never imputed", "ProductContents": "",
            "ProductCategory": "Amazon > Public category", "Retailer": "Amazon",
            "Category": "FORBIDDEN_TARGET_CATEGORY", "Mnfr": "FORBIDDEN_TARGET_MANUFACTURER"})
    frame = pd.DataFrame(rows)
    for role in ("train", "validation"):
        frame.loc[frame.split.eq(role)].iloc[::-1].to_csv(directory / f"{role}.csv",
                                                       index=False, lineterminator="\n")
    pd.DataFrame({"record_id": ["protected"], "group_id": ["protected-group"],
                  "split": ["test"]}).to_csv(directory / "protected_final_test.csv", index=False)
    (directory / "test.csv").write_bytes(b"NEVER OPEN OR HASH THE FINAL TEST")
    (directory / "model_config.json").write_text(json.dumps(config.to_dict()), encoding="utf8")
    freeze = {"format": "catalogiq-portable-split-v1",
              "model_view_contract": portable_model_view_contract(config, tokenizer_dir=tokenizer)}
    (directory / "grouping_freeze.json").write_text(json.dumps(freeze), encoding="utf8")
    reseal(directory)
    return config


class SegmentRetailerHoldoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = synthetic_exports(self.root)

    def development(self):
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier):
            return load_category_development(self.root, replace(self.config, sequence_length=256))

    def test_selector_uses_every_amazon_train_record_and_the_entire_frozen_validation(self):
        original = self.development()
        before = original.frame.copy(deep=True)
        selected = select_retailer_development(original)
        expected_training = original.frame.loc[original.frame.split.eq("train") & original.frame.Retailer.eq("Amazon")]
        expected_validation = original.frame.loc[original.frame.split.eq("validation")]
        training = selected.frame.iloc[selected.train_indices]
        validation = selected.frame.iloc[selected.validation_indices]
        self.assertEqual(len(training), 4)
        self.assertEqual(len(validation), 12)
        self.assertEqual(set(training.record_id), set(expected_training.record_id))
        self.assertEqual(set(validation.record_id), set(expected_validation.record_id))
        self.assertEqual(set(training.Retailer), {"Amazon"})
        self.assertEqual(set(validation.Retailer), {"Amazon", *DEFAULT_HELD_OUT})
        self.assertEqual(selected.label_names, original.label_names)
        self.assertEqual(selected.frame.index.tolist(), list(range(len(selected.frame))))
        self.assertFalse(set(training.group_id) & set(validation.group_id))
        self.assertFalse(set(selected.frame.record_id) & selected.protected_record_ids)
        self.assertFalse(set(selected.frame.group_id) & selected.protected_group_ids)
        audit = selected.audit["retailer_selection"]
        self.assertEqual(audit["training_retailer"], "Amazon")
        self.assertEqual(audit["source_training_records"], 14)
        self.assertEqual(audit["selected_training_records"], 4)
        self.assertEqual(audit["excluded_training_records"], 10)
        self.assertEqual(audit["validation_records"], 12)
        self.assertEqual(audit["training_class_counts"], {"Allergy": 2, "CCFS": 2})
        self.assertEqual(set(audit["missing_training_classes"]), set(original.label_names) - {"Allergy", "CCFS"})
        self.assertEqual(audit["validation_retailer_counts"], {retailer: 2 for retailer in ("Amazon", *DEFAULT_HELD_OUT)})
        self.assertTrue(audit["no_downsampling"])
        self.assertTrue(audit["record_and_group_isolation_passed"])
        pd.testing.assert_frame_equal(original.frame, before)

    def test_selection_is_reproducible_and_never_promotes_validation_into_training(self):
        development = self.development()
        first = select_retailer_development(development)
        second = select_retailer_development(development)
        pd.testing.assert_frame_equal(first.frame, second.frame)
        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
        validation_ids = set(development.frame.loc[development.frame.split.eq("validation"), "record_id"])
        self.assertFalse(set(first.frame.iloc[first.train_indices].record_id) & validation_ids)

    def test_selector_normalizes_case_and_whitespace_without_losing_records(self):
        original = self.development()
        frame = original.frame.copy()
        frame["Retailer"] = frame.Retailer.map(lambda name: " " + name.swapcase() + " ")
        selected = select_retailer_development(replace(original, frame=frame))
        self.assertEqual(len(selected.train_indices), 4)
        self.assertEqual(len(selected.validation_indices), 12)
        self.assertEqual(set(selected.frame.iloc[selected.train_indices].Retailer), {"Amazon"})

    def test_selector_rejects_invalid_or_incomplete_retailer_populations(self):
        original = self.development()
        wrong_role = original.frame.copy()
        wrong_role.loc[0, "split"] = "test"
        unknown = original.frame.copy()
        unknown.loc[0, "Retailer"] = "Unregistered retailer"
        blank = original.frame.copy()
        blank.loc[0, "Retailer"] = " "
        missing = original.frame.loc[~(original.frame.split.eq("validation") &
                                      original.frame.Retailer.eq(DEFAULT_HELD_OUT[0]))].copy()
        empty_train = original.frame.copy()
        empty_train.loc[empty_train.split.eq("train") & empty_train.Retailer.eq("Amazon"), "Retailer"] = DEFAULT_HELD_OUT[0]
        one_class = original.frame.copy()
        one_class.loc[one_class.split.eq("train") & one_class.Retailer.eq("Amazon"), "Segment"] = "Allergy"
        for name, frame in (("role", wrong_role), ("unknown", unknown), ("blank", blank),
                            ("missing_validation", missing), ("empty_training", empty_train),
                            ("one_training_class", one_class)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                select_retailer_development(replace(original, frame=frame))
        with self.assertRaises(ValueError):
            select_retailer_development(original, train_retailer=DEFAULT_HELD_OUT[0])
        with self.assertRaises(ValueError):
            select_retailer_development(original, held_out_retailers=(*DEFAULT_HELD_OUT, DEFAULT_HELD_OUT[0].lower()))

    def test_selector_rechecks_record_group_and_protected_isolation(self):
        original = self.development()
        amazon_index = original.frame.index[original.frame.split.eq("train") & original.frame.Retailer.eq("Amazon")][0]
        validation_index = original.frame.index[original.frame.split.eq("validation")][0]
        for column, value in (("record_id", original.frame.loc[validation_index, "record_id"]),
                              ("group_id", original.frame.loc[validation_index, "group_id"]),
                              ("record_id", "protected"), ("group_id", "protected-group")):
            frame = original.frame.copy()
            frame.loc[amazon_index, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                select_retailer_development(replace(original, frame=frame))

    def fake_trainer(self, calls, *, changed_initialization=False, wrong_membership=False):
        def train(corpus, train_indices, validation_indices, names, config, directory, progress=None):
            calls.append({"record_ids": corpus.record_ids, "train": np.asarray(train_indices).copy(),
                          "validation": np.asarray(validation_indices).copy(), "config": config,
                          "sequence_length": corpus.sequence_length, "token_sha256": corpus.token_sha256,
                          "arm": Path(directory).name})
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            truth = [names[int(corpus.label_ids[index])] for index in validation_indices]
            predicted = pd.DataFrame({"record_id": [corpus.record_ids[index] for index in validation_indices],
                "true_label": truth, "predicted_label": truth, "confidence": .9})
            for class_index, name in enumerate(names):
                predicted[f"probability_{class_index}"] = [.9 if label == name else .1 / (len(names) - 1)
                                                          for label in truth]
            predicted.to_csv(directory / "predictions.csv", index=False)
            manifest = {"config": config.to_dict(), "label_names": list(names), "input_fields": list(MODEL_FIELDS),
                "target": "Segment", "initial_state_sha256":
                    ("2" if changed_initialization and len(calls) == 2 else "1") * 64,
                "train_record_ids_sha256": _hash_items([corpus.record_ids[index] for index in train_indices]),
                "validation_record_ids_sha256": _hash_items([corpus.record_ids[index] for index in validation_indices]),
                "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
                "training_rows": len(train_indices), "validation_rows": len(validation_indices), "runtime_seconds": 0.0}
            if wrong_membership:
                manifest["train_record_ids_sha256"] = "0" * 64
            (directory / "initialization.json").write_text(json.dumps(manifest), encoding="utf8")
            (directory / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            (directory / "checkpoint.pt").write_bytes(b"Public synthetic placeholder; contains no model weights")
            return predicted, manifest
        return train

    def test_two_arms_share_full_amazon_training_and_score_every_validation_retailer(self):
        calls = []
        output = self.root / "retailer-experiment"
        original = {name: sha256(self.root / name) for name in
                    ("train.csv", "validation.csv", "model_config.json", "grouping_freeze.json")}
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
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer(calls)), \
                patch("catalogiq.segment_retailer_holdout.pd.read_csv", side_effect=guarded_read), \
                patch("catalogiq.segment_retailer_holdout.sha256", side_effect=guarded_hash), \
                patch("catalogiq.segment_category_ablation.sha256", side_effect=guarded_hash), \
                patch("catalogiq.segment_frozen_development.sha256", side_effect=guarded_hash):
            result = run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None, arms=ARMS)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["record_ids"], calls[1]["record_ids"])
        np.testing.assert_array_equal(calls[0]["train"], calls[1]["train"])
        np.testing.assert_array_equal(calls[0]["validation"], calls[1]["validation"])
        self.assertEqual(len(calls[0]["train"]), 4)
        self.assertEqual(len(calls[0]["validation"]), 12)
        self.assertEqual(calls[0]["config"], calls[1]["config"])
        self.assertEqual(calls[0]["config"].sequence_length, 256)
        self.assertEqual(calls[0]["config"].epochs, 3)
        self.assertEqual(calls[0]["config"].seed, 42)
        self.assertEqual([call["sequence_length"] for call in calls], [256, 256])
        self.assertNotEqual(calls[0]["token_sha256"], calls[1]["token_sha256"])
        self.assertTrue(result["same_initial_weights"])
        self.assertTrue(result["same_training_validation_membership"])
        self.assertFalse(result["final_test_read_or_scored"])
        for arm, fields in (("baseline", list(MODEL_FIELDS)), ("with_category", ["ProductCategory", *MODEL_FIELDS])):
            for filename in ("initialization.json", "training_manifest.json"):
                manifest = json.loads((output / arm / filename).read_text(encoding="utf8"))
                self.assertEqual(manifest["input_fields"], fields)
                self.assertEqual(manifest["config"]["sequence_length"], 256)
                self.assertEqual(manifest["training_rows"], 4)
                self.assertEqual(manifest["validation_rows"], 12)
            predictions = pd.read_csv(output / arm / "predictions.csv", dtype={"record_id": str})
            self.assertEqual(len(predictions), 12)
            self.assertFalse(predictions.record_id.str.startswith("train-").any())
        assignments = pd.read_csv(output / "retailer_assignments.csv", dtype=str, keep_default_na=False)
        self.assertEqual(len(assignments), 16)
        self.assertEqual(set(assignments.record_id), set(calls[0]["record_ids"]))
        self.assertEqual(assignments.experiment_role.value_counts().to_dict(),
                         {"retailer_holdout": 10, "train": 4, "amazon_control": 2})
        self.assertEqual(set(assignments.loc[assignments.experiment_role.eq("train"), "retailer"]), {"Amazon"})
        self.assertEqual(set(assignments.loc[assignments.experiment_role.ne("train"), "original_split"]), {"validation"})
        self.assertNotIn("Segment", assignments)
        metrics = pd.read_csv(output / "validation_metrics.csv", keep_default_na=False)
        expected_cohorts = {"all_validation", "amazon_control", "retailer_holdout",
                            *("retailer:" + retailer for retailer in DEFAULT_HELD_OUT)}
        self.assertEqual(set(metrics.cohort), expected_cohorts)
        self.assertEqual(set(metrics.arm), {"baseline", "with_category"})
        for arm in ("baseline", "with_category"):
            arm_metrics = metrics.loc[metrics.arm.eq(arm)].set_index("cohort")
            self.assertEqual(set(arm_metrics.index), expected_cohorts)
            self.assertEqual(arm_metrics.loc["all_validation", "records"], 12)
            self.assertEqual(arm_metrics.loc["amazon_control", "records"], 2)
            self.assertEqual(arm_metrics.loc["retailer_holdout", "records"], 10)
            self.assertAlmostEqual(arm_metrics.loc["retailer_holdout", "accuracy"], 1.0)
            self.assertAlmostEqual(arm_metrics.loc["retailer_holdout", "macro_f1"], .25)
        self.assertEqual(metrics.training_regime.nunique(), 1)
        self.assertTrue((output / "category_coverage.csv").is_file())
        self.assertTrue((output / "category_slices.csv").is_file())
        paired = pd.read_csv(output / "paired_differences.csv", keep_default_na=False)
        self.assertEqual(set(paired.comparison), {"category_effect_amazon_only"})
        self.assertEqual(set(paired.cohort), {"all_validation", "amazon_control", "retailer_holdout"})
        self.assertEqual(original, {name: sha256(self.root / name) for name in original})

    def test_default_trains_only_product_category_and_creates_no_baseline_corpus_or_outputs(self):
        calls = []
        output = self.root / "category-only"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.tokenize_corpus", side_effect=AssertionError("Baseline tokenization was requested")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer(calls)):
            result = run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arm"], "with_category")
        self.assertEqual(len(calls[0]["train"]), 4)
        self.assertEqual(len(calls[0]["validation"]), 12)
        self.assertEqual(calls[0]["sequence_length"], 256)
        self.assertEqual(result["arms"], ["with_category"])
        self.assertEqual(set(result["input_audit"]), {"with_category"})
        self.assertFalse((output / "baseline").exists())
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf8"))
        self.assertEqual(protocol["arms"], ["with_category"])
        self.assertEqual(set(protocol["tokenization"]), {"with_category"})
        manifest = json.loads((output / "with_category" / "training_manifest.json").read_text(encoding="utf8"))
        self.assertEqual(manifest["input_fields"], ["ProductCategory", *MODEL_FIELDS])
        metrics = pd.read_csv(output / "validation_metrics.csv", keep_default_na=False)
        self.assertEqual(set(metrics.arm), {"with_category"})
        self.assertEqual(len(metrics), 8)
        self.assertEqual(result["paired_differences"], [])
        self.assertTrue(pd.read_csv(output / "paired_differences.csv").empty)
        self.assertEqual(json.loads((output / "paired_comparisons.json").read_text(encoding="utf8")), [])

    def test_invalid_arm_requests_are_rejected_before_loading_or_training(self):
        for arms in ((), ("with_category", "with_category"), ("unknown",), "with_category"):
            with self.subTest(arms=arms), \
                    patch("catalogiq.segment_retailer_holdout.load_category_development", side_effect=AssertionError("Loading was requested")):
                with self.assertRaises(ValueError):
                    run_experiment(self.root, self.root / "invalid-arms", arms=arms, progress=lambda _: None)
        self.assertFalse((self.root / "invalid-arms").exists())

    def test_prepare_only_saves_selection_and_freezes_inputs_without_training(self):
        output = self.root / "prepared"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=AssertionError("Training was requested")):
            protocol = run_experiment(self.root, output, bootstrap_samples=10,
                                      prepare_only=True, progress=lambda _: None)
        self.assertEqual(protocol["training_rows"], 4)
        self.assertEqual(protocol["validation_rows"], 12)
        self.assertEqual(protocol["model_config"]["sequence_length"], 256)
        self.assertFalse(protocol["final_test_read_or_scored"])
        self.assertEqual(protocol["arms"], ["with_category"])
        self.assertEqual(set(protocol["tokenization"]), {"with_category"})
        self.assertTrue((output / "retailer_assignments.csv").is_file())
        freeze = json.loads((output / "protocol_freeze.json").read_text(encoding="utf8"))
        self.assertTrue(freeze["frozen_before_training"])
        self.assertEqual(freeze["protocol_sha256"], sha256(output / "protocol.json"))

    def test_independent_near_audit_counts_only_selected_training_crossings_and_reads_only_ids(self):
        development = self.development()
        amazon_train = development.frame.loc[development.frame.split.eq("train") & development.frame.Retailer.eq("Amazon"), "record_id"].tolist()
        nonamazon_train = development.frame.loc[development.frame.split.eq("train") & development.frame.Retailer.ne("Amazon"), "record_id"].tolist()
        control = development.frame.loc[development.frame.split.eq("validation") & development.frame.Retailer.eq("Amazon"), "record_id"].tolist()
        held = development.frame.loc[development.frame.split.eq("validation") & development.frame.Retailer.ne("Amazon"), "record_id"].tolist()
        pairs = pd.DataFrame({"left_record_id": [amazon_train[0], held[0], nonamazon_train[0], "protected", control[0], amazon_train[1]],
                              "right_record_id": [held[0], amazon_train[0], held[1], held[1], held[1], control[1]],
                              "left_name": "DO NOT READ NAMES", "right_name": "DO NOT READ NAMES"})
        path = self.root / "independent_near_pairs.csv"
        pairs.to_csv(path, index=False)
        completion = json.loads((self.root / "completion.json").read_text(encoding="utf8"))
        completion["artifact_sha256"][path.name] = sha256(path)
        (self.root / "completion.json").write_text(json.dumps(completion), encoding="utf8")
        read_csv = pd.read_csv

        def read_identity_only(filename, *args, **kwargs):
            if Path(filename).name == "independent_near_pairs.csv":
                self.assertEqual(kwargs.get("usecols"), ["left_record_id", "right_record_id"])
            self.assertNotEqual(Path(filename).name, "test.csv")
            return read_csv(filename, *args, **kwargs)

        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.pd.read_csv", side_effect=read_identity_only):
            protocol = run_experiment(self.root, self.root / "near-audit", bootstrap_samples=10,
                                      prepare_only=True, progress=lambda _: None)
        audit = protocol["independent_near_audit"]
        self.assertTrue(audit["available"])
        self.assertEqual(audit["candidate_crossings_all_validation"], 2)
        self.assertEqual(audit["candidate_crossings_held_out_retailers"], 1)
        self.assertEqual(audit["validation_rows_with_candidate_training_neighbor"], 2)

    def test_resume_reuses_completed_models_and_rejects_configuration_or_assignment_drift(self):
        calls = []
        output = self.root / "resumable"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer(calls)):
            first = run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None)
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=AssertionError("Completed model was retrained")):
            resumed = run_experiment(self.root, output, bootstrap_samples=10, resume=True, progress=lambda _: None)
            self.assertEqual(first["selection"], resumed["selection"])
            with self.assertRaisesRegex(ValueError, "Resume|configuration|protocol"):
                run_experiment(self.root, output, replace(self.config, sequence_length=256, learning_rate=1e-5),
                               bootstrap_samples=10, resume=True, progress=lambda _: None)
            with self.assertRaisesRegex(ValueError, "Resume|configuration|protocol"):
                run_experiment(self.root, output, bootstrap_samples=10, resume=True,
                               arms=ARMS, progress=lambda _: None)
            assignments = pd.read_csv(output / "retailer_assignments.csv", dtype=str, keep_default_na=False)
            assignments.loc[0, "experiment_role"] = "tampered"
            assignments.to_csv(output / "retailer_assignments.csv", index=False)
            with self.assertRaises(ValueError):
                run_experiment(self.root, output, bootstrap_samples=10, resume=True, progress=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_resume_rejects_copied_or_mislabeled_representation_and_corpus_metadata(self):
        output = self.root / "invalid-completed-model"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer([])):
            run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None)
        path = output / "with_category" / "training_manifest.json"
        original = json.loads(path.read_text(encoding="utf8"))
        wrong_config = {**original["config"], "sequence_length": 128}
        cases = (("input_fields", list(MODEL_FIELDS)), ("text_sha256", "0" * 64),
                 ("token_sha256", "0" * 64), ("label_names", original["label_names"][::-1]),
                 ("config", wrong_config))
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=AssertionError("Invalid completed model was retrained")):
            for key, value in cases:
                manifest = {**original, key: value}
                path.write_text(json.dumps(manifest), encoding="utf8")
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, "configuration|representation|corpus"):
                    run_experiment(self.root, output, bootstrap_samples=10, resume=True, progress=lambda _: None)

    def test_experiment_rejects_initialization_or_training_membership_changes(self):
        for name, options, pattern in (("initialization", {"changed_initialization": True}, "identical weights"),
                                       ("membership", {"wrong_membership": True}, "membership")):
            with self.subTest(case=name):
                output = self.root / name
                with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                        patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                        patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                        patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer([], **options)):
                    with self.assertRaisesRegex(ValueError, pattern):
                        run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None, arms=ARMS)
                self.assertFalse((output / "completion.json").exists())

    def test_experiment_detects_source_changes_between_training_arms(self):
        calls = []
        trainer = self.fake_trainer(calls)

        def mutate_source(*args, **kwargs):
            result = trainer(*args, **kwargs)
            with (self.root / "train.csv").open("ab") as stream:
                stream.write(b"\n")
            return result

        output = self.root / "source-change"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=mutate_source):
            with self.assertRaisesRegex(ValueError, "source|artifact|changed"):
                run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None, arms=ARMS)
        self.assertEqual(len(calls), 1)
        self.assertFalse((output / "completion.json").exists())

    def test_empty_amazon_control_is_explicitly_unscored(self):
        development = self.development()
        frame = development.frame.loc[~(development.frame.split.eq("validation") &
                                       development.frame.Retailer.eq("Amazon"))].copy()
        no_control = replace(development, frame=frame)
        output = self.root / "empty-control"
        with patch("catalogiq.segment_retailer_holdout.load_category_development", return_value=no_control), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer([])):
            result = run_experiment(self.root, output, bootstrap_samples=10, progress=lambda _: None, arms=ARMS)
        control = [row for row in result["metrics"] if row["cohort"] == "amazon_control"]
        self.assertEqual(len(control), 2)
        for row in control:
            self.assertEqual(row["records"], 0)
            self.assertIsNone(row["accuracy"])
            self.assertIsNone(row["macro_f1"])
        paired = json.loads((output / "paired_comparisons.json").read_text(encoding="utf8"))
        for row in paired:
            if row["cohort"] == "amazon_control":
                self.assertEqual(row["common_records"], 0)

    def test_reference_comparison_uses_identical_validation_records_for_both_training_regimes(self):
        reference = self.root / "mixed-retailer-reference"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=self.fake_trainer([])):
            run_category_experiment(self.root, reference, bootstrap_samples=10, progress=lambda _: None)
        output = self.root / "with-reference"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer([])):
            run_experiment(self.root, output, reference_dir=reference, bootstrap_samples=10,
                           progress=lambda _: None, arms=ARMS)
        metrics = pd.read_csv(output / "validation_metrics.csv", keep_default_na=False)
        self.assertEqual(set(metrics.training_regime), {"amazon_only", "mixed_retailer_reference"})
        for cohort in ("all_validation", "amazon_control", "retailer_holdout"):
            subset = metrics.loc[metrics.cohort.eq(cohort)]
            self.assertEqual(len(subset), 4)
            self.assertEqual(subset.records.nunique(), 1)
        paired = json.loads((output / "paired_comparisons.json").read_text(encoding="utf8"))
        expected = {"category_effect_amazon_only", "holdout_training_vs_reference:baseline",
                    "holdout_training_vs_reference:with_category"}
        self.assertEqual({row["comparison"] for row in paired}, expected)
        for row in paired:
            self.assertEqual(row["left_coverage"], 1.0)
            self.assertEqual(row["right_coverage"], 1.0)
            self.assertEqual(row["common_records"], {"all_validation": 12, "amazon_control": 2,
                                                     "retailer_holdout": 10}[row["cohort"]])

    def test_default_reference_loads_only_completed_product_category_artifacts(self):
        reference = self.root / "category-reference"
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_category_ablation.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_category_ablation.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_category_ablation.train_transformer", side_effect=self.fake_trainer([])):
            run_category_experiment(self.root, reference, bootstrap_samples=10, progress=lambda _: None)
        # The requested reference arm is independently sealed. The unused arm
        # must neither be read nor hashed, even if its files are unavailable.
        for name in ("training_manifest.json", "predictions.csv"):
            (reference / "baseline" / name).write_bytes(b"UNUSED BASELINE MUST NOT BE CONSUMED")
        read_csv, hash_file = pd.read_csv, sha256

        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).parent, reference / "baseline")
            self.assertNotEqual(Path(path).name, "test.csv")
            return read_csv(path, *args, **kwargs)

        def guarded_hash(path):
            self.assertNotEqual(Path(path).parent, reference / "baseline")
            self.assertNotEqual(Path(path).name, "test.csv")
            return hash_file(path)

        output = self.root / "category-with-reference"
        calls = []
        with patch("catalogiq.split_protocol.verify_protocol_splits", side_effect=verifier), \
                patch("catalogiq.segment_retailer_holdout.pretrained_artifacts", return_value={}), \
                patch("catalogiq.segment_retailer_holdout.load_tokenizer", return_value=PublicTokenizer(self.root / "tokenizer")), \
                patch("catalogiq.segment_retailer_holdout.tokenize_corpus", side_effect=AssertionError("Baseline tokenization was requested")), \
                patch("catalogiq.segment_retailer_holdout.train_transformer", side_effect=self.fake_trainer(calls)), \
                patch("catalogiq.segment_retailer_holdout.pd.read_csv", side_effect=guarded_read), \
                patch("catalogiq.segment_retailer_holdout.sha256", side_effect=guarded_hash), \
                patch("catalogiq.segment_category_ablation.sha256", side_effect=guarded_hash):
            result = run_experiment(self.root, output, reference_dir=reference,
                                    bootstrap_samples=10, progress=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arm"], "with_category")
        self.assertEqual(result["arms"], ["with_category"])
        self.assertFalse((output / "baseline").exists())
        metrics = pd.read_csv(output / "validation_metrics.csv", keep_default_na=False)
        self.assertEqual(set(metrics.training_regime), {"amazon_only", "mixed_retailer_reference"})
        self.assertEqual(set(metrics.arm), {"with_category"})
        paired = json.loads((output / "paired_comparisons.json").read_text(encoding="utf8"))
        self.assertEqual({row["comparison"] for row in paired}, {"holdout_training_vs_reference:with_category"})
        self.assertEqual({row["cohort"] for row in paired}, {"all_validation", "amazon_control", "retailer_holdout"})
        self.assertTrue(all(row["left_coverage"] == row["right_coverage"] == 1 for row in paired))

    def test_command_defaults_to_256_and_passes_reference_resume_and_preparation_flags(self):
        original = sha256(self.root / "model_config.json")
        reference = self.root / "reference"
        for arguments, length, arms in (([], 256, ("with_category",)),
                                        (["--sequence-length", "128"], 128, ("with_category",)),
                                        (["--model-config", str(self.root / "model_config.json")], 128, ("with_category",)),
                                        (["--arms", *ARMS], 256, ARMS)):
            with self.subTest(arguments=arguments), patch("catalogiq.segment_retailer_holdout.run_experiment") as run:
                main(["--split-dir", str(self.root), "--output-dir", str(self.root / "command"),
                      "--reference-dir", str(reference), "--prepare-only", "--resume", *arguments])
                run.assert_called_once()
                split_dir, output_dir, config = run.call_args.args
                self.assertEqual(split_dir, self.root)
                self.assertEqual(output_dir, self.root / "command")
                self.assertEqual(config, replace(self.config, sequence_length=length))
                self.assertEqual(run.call_args.kwargs["reference_dir"], reference)
                self.assertTrue(run.call_args.kwargs["resume"])
                self.assertTrue(run.call_args.kwargs["prepare_only"])
                self.assertEqual(tuple(run.call_args.kwargs["arms"]), tuple(arms))
        self.assertEqual(sha256(self.root / "model_config.json"), original)


if __name__ == "__main__":
    unittest.main()
