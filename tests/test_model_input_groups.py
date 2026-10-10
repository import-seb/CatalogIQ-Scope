from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.model_input_groups import (
    SIGNATURE_PREFIX, exact_model_input_groups, load_model_input_cache,
    merge_model_input_groups, model_view_contract,
)
from catalogiq.segment_transformer import ModelConfig, _hash_items


FINGERPRINT = {"test_tokenizer": "fixed-test-v1"}


class FakeTokenizer:
    def __call__(self, texts, truncation, max_length, padding, return_attention_mask):
        rows = []
        for text in texts:
            # Deterministic lowercase tokenizer, with special tokens and truncation.
            tokens = [101] + [int(hashlib.sha256(word.casefold().encode()).hexdigest()[:6], 16)
                              for word in text.split()] + [102]
            rows.append(tokens[:max_length])
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows],
                "token_type_ids": [[0] * len(row) for row in rows]}


class HiddenTarget:
    def __str__(self):
        raise AssertionError("Target accessed")


def fixture():
    return pd.DataFrame({"record_id": list("abcd"),
                         "ProductName": ["Alpha Product", "ALPHA PRODUCT", "Separate Product", "alpha product"],
                         "ProductBrand": ["Brand", "BRAND", "Brand", "Brand"],
                         "ProductDescription": ["<b>Text</b>", "Text", "Other", "Text"],
                         "Segment": [HiddenTarget()] * 4, "Category": [HiddenTarget()] * 4})


def group(frame, config=None, **kwargs):
    return exact_model_input_groups(frame, frame.record_id, config,
                                   tokenizer=FakeTokenizer(), tokenizer_fingerprint=FINGERPRINT, **kwargs)


class ModelInputGroupTests(unittest.TestCase):
    def test_identical_effective_inputs_get_stable_hard_groups(self):
        frame = fixture()
        original = frame.copy(deep=True)
        groups, edges, stats = group(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[3])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(edges, [(0, 1, "exact_effective_model_input", 1.0), (0, 3, "exact_effective_model_input", 1.0)])
        self.assertEqual(stats["duplicate_pairs"], 3)
        self.assertEqual(stats["grouping_target_columns_used"], [])
        pd.testing.assert_frame_equal(frame, original)

    def test_row_permutation_and_batch_size_do_not_change_groups_or_edges(self):
        frame = fixture()
        groups, edges, _ = group(frame, token_batch_size=1)
        shuffled = frame.iloc[[3, 1, 2, 0]].reset_index(drop=True)
        other, other_edges, _ = group(shuffled, token_batch_size=3)
        self.assertEqual(dict(zip(frame.record_id, groups)), dict(zip(shuffled.record_id, other)))
        original_pairs = [(frame.at[a, "record_id"], frame.at[b, "record_id"]) for a, b, _, _ in edges]
        shuffled_pairs = [(shuffled.at[a, "record_id"], shuffled.at[b, "record_id"]) for a, b, _, _ in other_edges]
        self.assertEqual(original_pairs, shuffled_pairs)

    def test_truncation_is_actual_model_input_equality(self):
        frame = pd.DataFrame({"record_id": ["a", "b"], "ProductName": ["same repeated words laterA", "same repeated words laterB"]})
        short, _, _ = group(frame, ModelConfig(sequence_length=4))
        long, _, _ = group(frame, ModelConfig(sequence_length=32))
        self.assertEqual(short[0], short[1])
        self.assertNotEqual(long[0], long[1])

    def test_signature_includes_attention_masks_and_token_type_ids(self):
        class ArrayTokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1, 2]] * 3, "attention_mask": [[1, 1], [1, 0], [1, 1]],
                        "token_type_ids": [[0, 0], [0, 0], [0, 1]]}
        frame = pd.DataFrame({"record_id": list("abc"), "ProductName": ["Same"] * 3})
        groups, _, _ = exact_model_input_groups(frame, frame.record_id, tokenizer=ArrayTokenizer(), tokenizer_fingerprint=FINGERPRINT)
        self.assertEqual(len(set(groups)), 3)

    def test_malformed_tokenizer_arrays_and_misaligned_ids_are_rejected(self):
        class InvalidTokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1]] * len(texts), "attention_mask": [[]] * len(texts)}
        frame = fixture()
        with self.assertRaises(ValueError):
            exact_model_input_groups(frame, frame.record_id, tokenizer=InvalidTokenizer(), tokenizer_fingerprint=FINGERPRINT)
        with self.assertRaises(ValueError):
            exact_model_input_groups(frame, list("dcba"), tokenizer=FakeTokenizer(), tokenizer_fingerprint=FINGERPRINT)
        with self.assertRaises(ValueError):
            group(frame, token_batch_size=0)
        with self.assertRaises(ValueError):
            group(frame, replace(ModelConfig(), local_files_only=False))

    def test_must_link_union_preserves_existing_components_and_is_permutation_stable(self):
        ids = np.array(list("abcde"))
        rule = np.array(["r1", "r1", "r2", "r3", "r4"])
        exact = np.array(["s1", "s2", "s2", "s3", "s3"])
        groups, stats = merge_model_input_groups(rule, ids, exact)
        self.assertEqual(groups.tolist(), ["final_a"] * 3 + ["final_d"] * 2)
        self.assertEqual(stats["components_reduced"], 2)
        self.assertEqual(stats["equality_classes_connecting_different_product_groups"], 2)
        order = [4, 2, 0, 3, 1]
        other, _ = merge_model_input_groups(rule[order], ids[order], exact[order])
        self.assertEqual(dict(zip(ids, groups)), dict(zip(ids[order], other)))

    def test_union_rejects_blank_missing_duplicate_and_mismatched_inputs(self):
        for rule, ids, exact in ((["r", ""], list("ab"), ["s", "s"]),
                                  (["r", None], list("ab"), ["s", "s"]),
                                  (["r", "r"], list("aa"), ["s", "s"]),
                                  (["r"], list("ab"), ["s", "s"]),
                                  (["r", "r"], ["a", None], ["s", "s"])):
            with self.subTest(ids=ids, rule=rule), self.assertRaises(ValueError):
                merge_model_input_groups(rule, ids, exact)

    def test_contract_detects_limits_tokenizer_and_revision_changes(self):
        base = model_view_contract(tokenizer_fingerprint=FINGERPRINT)
        for cfg in (replace(ModelConfig(), sequence_length=64),
                    replace(ModelConfig(), field_character_limits=(1024, 512, 1000, 2000)),
                    replace(ModelConfig(), revision="new-revision")):
            self.assertNotEqual(base, model_view_contract(cfg, tokenizer_fingerprint=FINGERPRINT))
        self.assertNotEqual(base, model_view_contract(tokenizer_fingerprint={"test_tokenizer": "changed"}))
        self.assertEqual(base, model_view_contract(replace(ModelConfig(), learning_rate=1e-5), tokenizer_fingerprint=FINGERPRINT))

    def test_cache_reorders_ids_but_rejects_source_representation_universe_and_tampering(self):
        frame = fixture()
        groups, _, stats = group(frame)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            signatures = pd.DataFrame({"record_id": frame.record_id, "group_id": groups,
                                       "input_signature_sha256": [g[len(SIGNATURE_PREFIX):] for g in groups]})
            path = folder / "record_signatures.csv"
            signatures.to_csv(path, index=False)
            manifest = {"model_view_contract": stats["model_view_contract"], "input_sha256": "source",
                        "sorted_record_ids_sha256": _hash_items(sorted(frame.record_id)),
                        "signature_csv_sha256": sha256(path)}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            loaded, _ = load_model_input_cache(folder, list("dcba"), tokenizer_fingerprint=FINGERPRINT, expected_input_sha256="source")
            self.assertEqual(loaded.tolist(), groups[::-1].tolist())
            for ids, cfg, source in ((list("abcd"), ModelConfig(), "other-source"),
                                     (list("abc"), ModelConfig(), "source"),
                                     (list("abcd"), replace(ModelConfig(), sequence_length=64), "source")):
                with self.assertRaises(ValueError):
                    load_model_input_cache(folder, ids, cfg, tokenizer_fingerprint=FINGERPRINT, expected_input_sha256=source)
            signatures.loc[0, "group_id"] = "tampered"
            signatures.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_model_input_cache(folder, list("abcd"), tokenizer_fingerprint=FINGERPRINT, expected_input_sha256="source")


if __name__ == "__main__":
    unittest.main()
