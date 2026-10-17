"""Public, label-blind checks for a separate experimental representation audit.

No private data, model weights, training, downloads, or real final-test records
are used. The saved public tokenizer exercises effective input equality.
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
from catalogiq.segment_category_ablation import build_category_texts, tokenize_category_corpus
from catalogiq.segment_representation_audit import (
    FEATURE_FIELDS, _corpus_digests, _tokenize, audit_experiment,
    load_partition_features, summarize_signatures,
    token_signature, tokenize_signatures,
)
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, build_product_texts
from catalogiq.split_protocol import DEFAULT_PROTOCOL, OfflineTokenizer


class HiddenTarget:
    def __str__(self):
        raise AssertionError("A classification target entered the equality audit")


def public_records():
    """One cross-role 2/1/2 collision, one same-role pair, two unique inputs."""
    roles = ["train", "train", "validation", "test", "test", "train", "train", "validation", "test"]
    ids = ["t1", "t2", "v1", "h1", "h2", "t3", "t4", "v2", "h3"]
    return pd.DataFrame({
        "record_id": ids, "group_id": ["group_" + value for value in ids], "split": roles,
        "ProductName": ["Shared medicine"] * 5 + ["Train-only medicine"] * 2 + ["Unique validation", "Unique held-out"],
        "ProductBrand": "Public brand", "ProductDescription": "Public descriptive text",
        "ProductContents": "", "ProductCategory": "Public > Category",
    })


def write_public_exports(directory):
    directory.mkdir(parents=True, exist_ok=True)
    frame = public_records()
    for role in ("train", "validation", "test"):
        selected = frame.loc[frame.split.eq(role)].copy()
        selected["Segment"] = "PRIVATE_LABEL_MUST_NOT_BE_LOADED"
        selected["Category"] = "ANOTHER_TARGET_MUST_NOT_BE_LOADED"
        selected["Retailer"] = "Not a model field"
        selected.iloc[::-1].to_csv(directory / f"{role}.csv", index=False, lineterminator="\n")
    frame[["record_id", "group_id", "split"]].to_csv(directory / "assignments.csv", index=False, lineterminator="\n")
    shutil.copytree(DEFAULT_PROTOCOL / "tokenizer", directory / "tokenizer")
    (directory / "model_config.json").write_text(json.dumps(ModelConfig().to_dict()), encoding="utf8")
    names = ["assignments.csv", "train.csv", "validation.csv", "test.csv", "model_config.json"]
    (directory / "completion.json").write_text(json.dumps({
        "artifact_sha256": {name: sha256(directory / name) for name in names}
    }), encoding="utf8")
    return frame


def reseal(directory, *names):
    path = directory / "completion.json"
    completion = json.loads(path.read_text(encoding="utf8"))
    for name in names:
        completion["artifact_sha256"][name] = sha256(directory / name)
    path.write_text(json.dumps(completion), encoding="utf8")


class SegmentRepresentationAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frame = write_public_exports(self.root / "splits")
        self.directory = self.root / "splits"
        self.config = replace(ModelConfig(), sequence_length=256)
        self.tokenizer = OfflineTokenizer(self.directory / "tokenizer")

    def signatures(self, frame=None, **kwargs):
        return tokenize_signatures(self.frame if frame is None else frame, self.config,
                                   tokenizer=self.tokenizer, **kwargs)

    def test_signature_distinguishes_every_effective_input_array(self):
        encoded = {"input_ids": [[101, 5, 102]] * 3,
                   "attention_mask": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                   "token_type_ids": [[0, 0, 0], [0, 0, 0], [0, 1, 0]]}
        signatures = [token_signature(encoded, row)[0] for row in range(3)]
        self.assertEqual(len(set(signatures)), 3)
        self.assertEqual(token_signature(encoded, 0), token_signature(encoded, 0))
        # Absent token-type IDs differ from an explicitly supplied array.
        no_types = {key: value for key, value in encoded.items() if key != "token_type_ids"}
        self.assertNotEqual(token_signature(no_types, 0)[0], signatures[0])

    def test_signatures_are_stable_across_order_and_tokenization_batch_size(self):
        first = self.signatures(token_batch_size=1)
        shuffled = self.signatures(self.frame.iloc[[8, 3, 5, 0, 7, 1, 4, 6, 2]], token_batch_size=4)
        pd.testing.assert_frame_equal(first.sort_values("record_id").reset_index(drop=True),
                                      shuffled.sort_values("record_id").reset_index(drop=True))
        self.assertTrue(first.input_signature_sha256.str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertEqual(set(first.columns), {"record_id", "group_id", "split", "input_signature_sha256", "token_count"})
        self.assertTrue(first.token_count.between(1, 256).all())

    def test_missing_values_match_and_targets_never_enter_preprocessing(self):
        frame = self.frame.iloc[:2].copy()
        frame["ProductName"] = ["<b>Identical</b> medicine", "Identical medicine"]
        frame["ProductContents"] = [None, np.nan]
        frame["ProductCategory"] = ["null", " none "]
        frame["Segment"] = [HiddenTarget(), HiddenTarget()]
        frame["Category"] = [HiddenTarget(), HiddenTarget()]
        frame["Retailer"] = [HiddenTarget(), HiddenTarget()]
        before = frame.copy(deep=True)
        actual = self.signatures(frame)
        self.assertEqual(actual.input_signature_sha256.nunique(), 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_real_truncation_can_create_input_equality(self):
        frame = self.frame.iloc[:2].copy()
        frame["ProductName"] = "Shared medicine"
        frame["ProductDescription"] = ["hello " * 300 + "alpha diagnostic",
                                       "hello " * 300 + "omega diagnostic"]
        texts = build_category_texts(frame, self.config)
        self.assertNotEqual(texts[0], texts[1])
        actual = self.signatures(frame)
        self.assertEqual(actual.token_count.tolist(), [256, 256])
        self.assertEqual(actual.input_signature_sha256.nunique(), 1)

    def test_category_breaks_an_original_four_field_tie(self):
        frame = self.frame.iloc[:2].copy()
        frame["ProductCategory"] = ["Health > Vitamins", "Health > Allergy"]
        baseline = build_product_texts(frame, replace(self.config, sequence_length=128))
        self.assertEqual(baseline[0], baseline[1])
        actual = self.signatures(frame)
        self.assertEqual(actual.input_signature_sha256.nunique(), 2)

    def test_reconstructed_corpus_digest_matches_original_experiment_tokenizer_path(self):
        # Audit all feature-only rows in one order, then reconstruct the saved
        # development subset in another order. Labels do not enter either hash.
        _, arrays, texts = _tokenize(self.frame, self.config, tokenizer=self.tokenizer,
                                    token_batch_size=2)
        subset = self.frame.iloc[[8, 3, 0, 7]].copy()
        subset["Segment"] = ["A", "B", "A", "B"]
        original = tokenize_category_corpus(subset, ["A", "B"], self.config,
                                             tokenizer=self.tokenizer)
        positions = dict(zip(self.frame.record_id, range(len(self.frame))))
        actual = _corpus_digests(list(original.record_ids), positions, arrays, texts)
        self.assertEqual(actual, {"text_sha256": original.text_sha256,
                                  "token_sha256": original.token_sha256})

    def test_malformed_tokenizer_arrays_and_bad_batch_sizes_are_rejected(self):
        class BrokenTokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[101, 102]] * len(texts),
                        "attention_mask": [[1]] * len(texts)}
        with self.assertRaises(ValueError):
            tokenize_signatures(self.frame, self.config, tokenizer=BrokenTokenizer())
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                self.signatures(token_batch_size=batch_size)

    def test_crossing_counts_include_all_role_pairs_and_correct_group_evidence(self):
        signatures = self.signatures()
        summary, crossings, affected = summarize_signatures(signatures)
        self.assertEqual(summary["crossing_pairs"], 8)
        self.assertEqual(summary["crossing_clusters"], 1)
        self.assertEqual(summary["crossing_rows"], 5)
        self.assertEqual(summary["crossing_split_pairs"], {
            "train__validation": 2, "train__test": 4, "validation__test": 2})
        self.assertEqual(len(crossings), 8)
        self.assertEqual(set(affected.record_id), {"t1", "t2", "v1", "h1", "h2"})
        self.assertNotIn("t3", set(affected.record_id))
        groups = self.frame.set_index("record_id").group_id.to_dict()
        for row in crossings.itertuples(index=False):
            self.assertNotEqual(row.left_split, row.right_split)
            self.assertEqual(row.left_group_id, groups[row.left_record_id])
            self.assertEqual(row.right_group_id, groups[row.right_record_id])
        self.assertEqual(len({frozenset((row.left_record_id, row.right_record_id))
                              for row in crossings.itertuples(index=False)}), 8)
        _, other_crossings, other_affected = summarize_signatures(signatures.iloc[::-1])
        keys = ["input_signature_sha256", "left_record_id", "right_record_id"]
        pd.testing.assert_frame_equal(crossings.sort_values(keys).reset_index(drop=True),
                                      other_crossings.sort_values(keys).reset_index(drop=True))
        pd.testing.assert_frame_equal(affected.sort_values("record_id").reset_index(drop=True),
                                      other_affected.sort_values("record_id").reset_index(drop=True))

    def test_only_same_partition_duplicates_produce_no_crossings(self):
        actual = self.signatures(self.frame.loc[self.frame.record_id.isin(["t3", "t4", "v2", "h3"])])
        summary, crossings, affected = summarize_signatures(actual)
        self.assertEqual(summary["crossing_pairs"], 0)
        self.assertEqual(summary["crossing_clusters"], 0)
        self.assertEqual(summary["crossing_rows"], 0)
        self.assertTrue(crossings.empty)
        self.assertTrue(affected.empty)

    def test_loader_requests_exact_feature_and_identity_columns_never_targets(self):
        real_read = pd.read_csv
        before = {str(path): sha256(path) for path in self.directory.rglob("*") if path.is_file()}
        reads = []

        def guarded(path, *args, **kwargs):
            filename = Path(path).name
            columns = kwargs.get("usecols")
            self.assertIsInstance(columns, (tuple, list))
            self.assertEqual(set(columns), {"record_id", "group_id", "split"} |
                             (set(FEATURE_FIELDS) if filename in ("train.csv", "validation.csv", "test.csv") else set()))
            self.assertFalse({"Segment", "Category", "Retailer", "Mnfr"} & set(columns))
            reads.append(filename)
            return real_read(path, *args, **kwargs)

        with patch("catalogiq.segment_representation_audit.pd.read_csv", side_effect=guarded):
            actual = load_partition_features(self.directory)
        self.assertEqual(set(reads), {"assignments.csv", "train.csv", "validation.csv", "test.csv"})
        self.assertEqual(set(actual.columns), {"record_id", "group_id", "split", *FEATURE_FIELDS})
        pd.testing.assert_frame_equal(actual.sort_values("record_id").reset_index(drop=True).sort_index(axis=1),
                                      self.frame.sort_values("record_id").reset_index(drop=True).sort_index(axis=1))
        self.assertEqual(before, {str(path): sha256(path) for path in self.directory.rglob("*") if path.is_file()})
        self.assertEqual(tuple(FEATURE_FIELDS), ("ProductCategory", *MODEL_FIELDS))
        self.assertEqual(ModelConfig().sequence_length, 128)

    def test_loader_refuses_changed_export_even_without_membership_changes(self):
        path = self.directory / "test.csv"
        with path.open("a", encoding="utf8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed|hash|seal|checksum|modified"):
            load_partition_features(self.directory)

    def test_loader_refuses_resealed_record_or_group_membership_changes(self):
        path = self.directory / "test.csv"
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column, value in (("record_id", "outside_assignments"),
                              ("group_id", "outside_group"), ("split", "train")):
            changed = original.copy()
            changed.loc[0, column] = value
            changed.to_csv(path, index=False, lineterminator="\n")
            reseal(self.directory, "test.csv")
            with self.subTest(column=column), self.assertRaises(ValueError):
                load_partition_features(self.directory)

    def test_missing_product_category_cannot_be_substituted_with_target_category(self):
        path = self.directory / "test.csv"
        exported = pd.read_csv(path, dtype=str, keep_default_na=False)
        exported.drop(columns="ProductCategory").to_csv(path, index=False, lineterminator="\n")
        reseal(self.directory, "test.csv")
        with self.assertRaises(ValueError):
            load_partition_features(self.directory)

    def test_loader_refuses_duplicate_records_and_cross_partition_groups(self):
        path = self.directory / "assignments.csv"
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        duplicated = pd.concat([original, original.iloc[[0]]], ignore_index=True)
        duplicated.to_csv(path, index=False, lineterminator="\n")
        reseal(self.directory, "assignments.csv")
        with self.assertRaises(ValueError):
            load_partition_features(self.directory)
        changed = original.copy()
        changed.loc[changed.record_id.eq("h1"), "group_id"] = "group_t1"
        changed.to_csv(path, index=False, lineterminator="\n")
        exported = pd.read_csv(self.directory / "test.csv", dtype=str, keep_default_na=False)
        exported.loc[exported.record_id.eq("h1"), "group_id"] = "group_t1"
        exported.to_csv(self.directory / "test.csv", index=False, lineterminator="\n")
        reseal(self.directory, "assignments.csv", "test.csv")
        with self.assertRaises(ValueError):
            load_partition_features(self.directory)

    def test_existing_output_is_rejected_before_reading_any_source_or_checkpoint(self):
        output = self.root / "existing experiment"
        output.mkdir()
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(b"ORIGINAL CHECKPOINT MUST REMAIN AVAILABLE")
        before = sha256(checkpoint)
        # Missing source paths ensure the overwrite guard runs before any input
        # access, rather than merely failing partway through verification.
        with self.assertRaises((FileExistsError, ValueError)):
            audit_experiment(self.root / "missing splits", self.root / "missing experiment", output)
        self.assertEqual(sha256(checkpoint), before)


if __name__ == "__main__":
    unittest.main()
