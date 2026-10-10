import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.features import sha256
from catalogiq.model_input_groups import model_view_contract
from catalogiq.segment_frozen_development import load_frozen_development
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig


def fixture(directory):
    directory = Path(directory)
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf8")
    (tokenizer / "tokenizer_config.json").write_text('{"synthetic":true}', encoding="utf8")
    for role, ids in (("train", list("abc")), ("validation", list("def"))):
        frame = pd.DataFrame({"record_id": ids, "group_id": ["g" + id_ for id_ in ids],
                              "Segment": ["Allergy", "CCFS", "null"], "split": role,
                              "Category": ["FORBIDDEN_TARGET"] * 3, "Mnfr": ["FORBIDDEN_MANUFACTURER"] * 3})
        for field in MODEL_FIELDS:
            frame[field] = "Product evidence"
        frame.to_csv(directory / (role + ".csv"), index=False)
    # Protected file contains malicious feature/label sentinels: only IDs/groups may be loaded.
    pd.DataFrame({"record_id": ["x"], "group_id": ["gx"], "Segment": ["SECRET_TEST_LABEL"],
                  "ProductName": ["SECRET_TEST_NAME"]}).to_csv(directory / "protected_final_test.csv", index=False)
    (directory / "test.csv").write_text("DO NOT READ OR HASH THIS FILE", encoding="utf8")
    freeze = {"model_view_contract": model_view_contract(tokenizer_dir=tokenizer)}
    (directory / "grouping_freeze.json").write_text(json.dumps(freeze), encoding="utf8")
    reseal(directory)
    return tokenizer


def reseal(directory):
    manifest = {"artifact_sha256": {name: sha256(directory / name) for name in
                    ("train.csv", "validation.csv", "protected_final_test.csv", "grouping_freeze.json")}}
    (directory / "completion.json").write_text(json.dumps(manifest), encoding="utf8")


def fake_verifier(directory, *, verify_exports):
    if verify_exports:
        raise AssertionError("Training loader requested final-test exports")
    return {"checks_passed": True}


class FrozenDevelopmentTests(unittest.TestCase):
    def test_development_only_preserves_unknown_targets_and_excludes_metadata_features(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tokenizer = fixture(directory)
            development = load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)
            self.assertEqual(development.frame.record_id.tolist(), list("abde"))
            self.assertEqual(development.train_indices.tolist(), [0, 1])
            self.assertEqual(development.validation_indices.tolist(), [2, 3])
            self.assertNotIn("Category", development.frame)
            self.assertNotIn("Mnfr", development.frame)
            self.assertEqual(development.audit["missing_target_records_excluded_from_supervised_view"], {"train": 1, "validation": 1})
            self.assertEqual(pd.read_csv(directory / "train.csv", keep_default_na=False).Segment.tolist(), ["Allergy", "CCFS", "null"])
            self.assertEqual(development.protected_record_ids, frozenset(["x"]))

    def test_test_csv_never_opened_or_hashed_protected_columns_are_ID_only(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tokenizer = fixture(directory)
            real_read = pd.read_csv
            real_hash = sha256
            def guarded_read(path, **kwargs):
                if Path(path).name == "test.csv":
                    raise AssertionError("Read final test export")
                if Path(path).name == "protected_final_test.csv":
                    self.assertEqual(kwargs.get("usecols"), ["record_id", "group_id"])
                return real_read(path, **kwargs)
            def guarded_hash(path):
                if Path(path).name == "test.csv":
                    raise AssertionError("Hashed final test export")
                return real_hash(path)
            with patch("catalogiq.segment_frozen_development.pd.read_csv", side_effect=guarded_read), patch("catalogiq.segment_frozen_development.sha256", side_effect=guarded_hash):
                load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)

    def test_modified_non_test_export_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tokenizer = fixture(directory)
            with (directory / "train.csv").open("a", encoding="utf8") as stream:
                stream.write("modified\n")
            with self.assertRaisesRegex(ValueError, "train.csv"):
                load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)

    def test_group_or_record_overlap_rejected_even_for_unknown_target_rows(self):
        for column, value in (("record_id", "a"), ("group_id", "ga"), ("record_id", "x"), ("group_id", "gx")):
            with self.subTest(column=column, value=value), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                tokenizer = fixture(directory)
                frame = pd.read_csv(directory / "validation.csv", keep_default_na=False)
                frame.loc[2, column] = value  # Unlabelled row still participates in isolation guards.
                frame.to_csv(directory / "validation.csv", index=False)
                reseal(directory)
                with self.assertRaises(ValueError):
                    load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)

    def test_unknown_nonmissing_target_and_role_contamination_are_rejected(self):
        for field, value in (("Segment", "UNLABELED_ALLOCATION_ONLY"), ("split", "test"), ("group_id", "")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                tokenizer = fixture(directory)
                frame = pd.read_csv(directory / "train.csv", keep_default_na=False)
                frame.loc[0, field] = value
                frame.to_csv(directory / "train.csv", index=False)
                reseal(directory)
                with self.assertRaises(ValueError):
                    load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)

    def test_model_representation_drift_and_feature_allowlist_contamination_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tokenizer = fixture(directory)
            with self.assertRaisesRegex(ValueError, "representation"):
                load_frozen_development(directory, ModelConfig(sequence_length=64), tokenizer_dir=tokenizer, verifier=fake_verifier)
            freeze_path = directory / "grouping_freeze.json"
            freeze = json.loads(freeze_path.read_text())
            freeze["model_view_contract"]["features"].append("Segment")
            freeze_path.write_text(json.dumps(freeze), encoding="utf8")
            reseal(directory)
            with self.assertRaisesRegex(ValueError, "representation"):
                load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=fake_verifier)

    def test_upstream_source_or_cache_verification_failure_blocks_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tokenizer = fixture(directory)
            def rejected(*args, **kwargs):
                raise ValueError("Stale source or model-input cache")
            with self.assertRaisesRegex(ValueError, "Stale source"):
                load_frozen_development(directory, tokenizer_dir=tokenizer, verifier=rejected)


if __name__ == "__main__":
    unittest.main()
