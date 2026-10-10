"""Portable identity verification and final-test protection using public data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.features import sha256
from catalogiq.split_protocol import (
    OfflineTokenizer, _source_digest, canonical_digest, export_final_splits,
    load_protocol, verify_protocol_splits,
)
from test_split_data_command import fixture_builder, read_csv


class PortableSplitProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "synthetic protocol"
        production, _ = load_protocol()
        frame, cls.manifest = fixture_builder().materialize(cls.bundle, production)
        candidate = cls.root / "training_candidate.csv"
        frame.drop(columns="record_id").to_csv(candidate, index=False, lineterminator="\n")
        cls.output = cls.root / "valid generated splits"
        export_final_splits(candidate, cls.output, protocol_dir=cls.bundle, progress=lambda _: None)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def copy_output(self):
        copied = self.folder / "relocated splits"
        shutil.copytree(self.output, copied)
        return copied

    def reseal(self, directory, *names):
        """Simulate someone updating their own checksums after corrupting files."""
        path = directory / "completion.json"
        completion = json.loads(path.read_text(encoding="utf8"))
        for name in names:
            completion["artifact_sha256"][name] = sha256(directory / name)
        path.write_text(json.dumps(completion) + "\n", encoding="utf8")

    def test_canonical_digest_uses_documented_values_order_and_utf8_recipe(self):
        frame = pd.DataFrame({"record_id": ["b", "a"], "value": ["001", 'café\n"quoted"']})
        expected = hashlib.sha256()
        for row in (["a", 'café\n"quoted"', ""], ["b", "001", ""]):
            expected.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf8") + b"\n")
        fields = ["record_id", "value", "missing"]
        self.assertEqual(canonical_digest(frame, fields), expected.hexdigest())
        self.assertEqual(canonical_digest(frame.iloc[::-1], fields), expected.hexdigest())
        with self.assertRaisesRegex(ValueError, "unique"):
            canonical_digest(pd.concat([frame, frame.iloc[[0]]]), fields)

    def test_algorithm_hashes_normalize_only_line_endings(self):
        windows = self.folder / "windows.py"
        unix = self.folder / "unix.py"
        windows.write_bytes(b'VALUE = "caf\xc3\xa9"\r\n\r\n')
        unix.write_bytes(b'VALUE = "caf\xc3\xa9"\n\n')
        self.assertEqual(_source_digest(windows), _source_digest(unix))
        unix.write_bytes(b'VALUE = "different"\n\n')
        self.assertNotEqual(_source_digest(windows), _source_digest(unix))

    def test_offline_tokenizer_has_actual_special_tokens_and_frozen_truncation(self):
        tokenizer = OfflineTokenizer(self.bundle / "tokenizer")
        encoded = tokenizer(["Hello world", "HELLO WORLD", "hello " * 150], max_length=128)
        self.assertEqual(encoded["input_ids"][0], [101, 7592, 2088, 102])
        self.assertEqual(encoded["input_ids"][0], encoded["input_ids"][1])
        self.assertEqual(len(encoded["input_ids"][2]), 128)
        self.assertEqual(encoded["input_ids"][2][-1], 102)
        for ids, mask, types in zip(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]):
            self.assertEqual(mask, [1] * len(ids))
            self.assertEqual(types, [0] * len(ids))
        with self.assertRaises(ValueError):
            tokenizer(["hello"], padding=True)

    def test_relocated_bundle_and_outputs_verify_without_original_paths(self):
        relocated_bundle = self.folder / "another machine protocol"
        shutil.copytree(self.bundle, relocated_bundle)
        relocated = self.copy_output()
        result = verify_protocol_splits(relocated, protocol_dir=relocated_bundle)
        self.assertTrue(result["checks"]["exact_frozen_assignments"])
        self.assertEqual(result["authoritative"]["assignment_sha256"], self.manifest["expected"]["assignment_sha256"])
        for name in ("grouping_freeze.json", "completion.json", "input_manifest.json"):
            self.assertNotIn(str(self.root), (relocated / name).read_text(encoding="utf8"))

    def test_resealed_membership_changes_still_fail_against_authoritative_hashes(self):
        copied = self.copy_output()
        frame = read_csv(copied / "assignments.csv")
        group = frame.loc[frame.split.eq("train"), "group_id"].iloc[0]
        frame.loc[frame.group_id.eq(group), "split"] = "validation"
        frame.to_csv(copied / "assignments.csv", index=False, lineterminator="\n")
        shutil.copyfile(copied / "assignments.csv", copied / "rule_assignments.csv")
        self.reseal(copied, "assignments.csv", "rule_assignments.csv")
        self.assertEqual(frame.groupby("group_id").split.nunique().max(), 1)
        with self.assertRaisesRegex(ValueError, "membership|authoritative|October 9"):
            verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)

    def test_resealed_provenance_and_input_signatures_cannot_be_substituted(self):
        copied = self.copy_output()
        frame = read_csv(copied / "assignments.csv")
        frame.loc[0, "source_row"] = "999999"
        frame.to_csv(copied / "assignments.csv", index=False, lineterminator="\n")
        shutil.copyfile(copied / "assignments.csv", copied / "rule_assignments.csv")
        self.reseal(copied, "assignments.csv", "rule_assignments.csv")
        with self.assertRaisesRegex(ValueError, "provenance|stable"):
            verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)
        copied = self.folder / "signature tampering"
        shutil.copytree(self.output, copied)
        frame = read_csv(copied / "model_inputs.csv")
        frame.loc[0, "effective_input_group_id"] = "model_input_" + "0" * 64
        frame.to_csv(copied / "model_inputs.csv", index=False, lineterminator="\n")
        self.reseal(copied, "model_inputs.csv")
        with self.assertRaisesRegex(ValueError, "effective-input"):
            verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)

    def test_changed_tokenizer_or_exposure_resource_is_rejected(self):
        for name in ("tokenizer/tokenizer_config.json", "eligibility.json"):
            with self.subTest(resource=name):
                bundle = self.folder / name.replace("/", "-")
                shutil.copytree(self.bundle, bundle)
                with (bundle / name).open("ab") as stream:
                    stream.write(b"\n ")
                with self.assertRaisesRegex(ValueError, "resource changed"):
                    load_protocol(bundle)

    def test_resealed_grouping_configuration_and_exposure_outputs_are_rejected(self):
        for filename, pattern in (("grouping_assignments.csv", "grouping memberships"),
                                  ("split_config.json", "configuration changed"),
                                  ("eligibility.json", "exposure exclusions")):
            with self.subTest(artifact=filename):
                copied = self.folder / filename.replace(".", "-")
                shutil.copytree(self.output, copied)
                if filename.endswith(".csv"):
                    frame = read_csv(copied / filename)
                    frame.loc[0, "product_group_id"] = "different_component"
                    frame.to_csv(copied / filename, index=False, lineterminator="\n")
                else:
                    document = json.loads((copied / filename).read_text(encoding="utf8"))
                    if filename == "split_config.json":
                        document["seed"] = 123
                    else:
                        document["selectors"]["strict_eligible"]["source_rows"].append(0)
                    (copied / filename).write_text(json.dumps(document) + "\n", encoding="utf8")
                self.reseal(copied, filename)
                with self.assertRaisesRegex(ValueError, pattern):
                    verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)

    def test_missing_seal_entries_cannot_hide_modified_exports(self):
        copied = self.copy_output()
        completion_path = copied / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf8"))
        del completion["artifact_sha256"]["train.csv"]
        completion_path.write_text(json.dumps(completion) + "\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)

    def test_resealed_assignment_alias_and_protected_ids_cannot_diverge(self):
        for filename, pattern in (("rule_assignments.csv", "identical final assignment alias"),
                                  ("protected_final_test.csv", "identity/group mapping")):
            with self.subTest(artifact=filename):
                copied = self.folder / filename.replace(".", "-")
                shutil.copytree(self.output, copied)
                frame = read_csv(copied / filename)
                frame.loc[0, "group_id"] = "different_group"
                frame.to_csv(copied / filename, index=False, lineterminator="\n")
                self.reseal(copied, filename)
                with self.assertRaisesRegex(ValueError, pattern):
                    verify_protocol_splits(copied, verify_exports=False, protocol_dir=self.bundle)

    def test_development_verification_never_opens_test_export_or_target_columns(self):
        original_read, original_sha = pd.read_csv, sha256

        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            if Path(path).name in ("assignments.csv", "protected_final_test.csv"):
                self.assertIsNotNone(kwargs.get("usecols"))
                self.assertNotIn("Segment", kwargs["usecols"])
                self.assertNotIn("ProductName", kwargs["usecols"])
            return original_read(path, *args, **kwargs)

        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_sha(path)

        with patch("catalogiq.split_protocol.pd.read_csv", guarded_read), patch("catalogiq.split_protocol.sha256", guarded_hash):
            result = verify_protocol_splits(self.output, verify_exports=False, protocol_dir=self.bundle)
        self.assertTrue(result["checks"]["exact_final_test_membership"])


if __name__ == "__main__":
    unittest.main()
