"""Run the authoritative CLI against a frozen, public synthetic protocol.

Actual v4 grouping, packaged tokenization, eligibility and allocation are used.
No matching is patched, private data required, or classifier trained.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from catalogiq.split_protocol import load_protocol

REPOSITORY = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
PROVENANCE = ["dataset", "source_sha256", "source_row"]


def fixture_builder():
    path = Path(__file__).parent / "fixtures/portable_split_protocol/build_fixture.py"
    spec = importlib.util.spec_from_file_location("portable_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


class SplitDataCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "relocated protocol"
        production_bundle, _ = load_protocol()
        self.frame, self.manifest = fixture_builder().materialize(self.bundle, production_bundle)
        self.input = self.root / "training_candidate.csv"
        self.frame.drop(columns="record_id").to_csv(self.input, index=False, lineterminator="\n")

    def command(self, output=None, *options, input_path=None, use_fixture=True):
        args = [sys.executable, "-m", "scripts.split_data"]
        if output is not None:
            args += ["--input", str(input_path or self.input), "--output-dir", str(output)]
        if use_fixture:
            args += ["--protocol-dir", str(self.bundle)]
        args += list(map(str, options))
        # In CI catalogiq comes from pip install .; this does not inject src.
        return subprocess.run(args, cwd=REPOSITORY, capture_output=True, text=True,
                              encoding="utf8", errors="replace", timeout=90, check=False)

    def run_command(self, output, *options, input_path=None):
        result = self.command(output, *options, input_path=input_path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((output / "summary.json").read_text(encoding="utf8"))

    def test_command_exports_final_protocol_and_full_source_rows(self):
        output = self.root / "splits"
        before = self.input.read_bytes()
        self.run_command(output)
        required = {"assignments.csv", "rule_assignments.csv", "train.csv", "validation.csv", "test.csv",
                    "protected_final_test.csv", "grouping_freeze.json", "summary.json", "completion.json"}
        self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
        self.assertEqual((output / "assignments.csv").read_bytes(), (output / "rule_assignments.csv").read_bytes())
        assignments = read_csv(output / "assignments.csv")
        self.assertEqual(assignments.columns.tolist(), [*PROVENANCE, "record_id", "group_id", "split"])
        self.assertEqual(len(assignments), 360)
        self.assertTrue(assignments.record_id.is_unique)
        self.assertEqual(assignments.groupby("group_id").split.nunique().max(), 1)
        roles = []
        for role in SPLITS:
            frame = read_csv(output / (role + ".csv"))
            self.assertTrue(frame.split.eq(role).all())
            self.assertEqual(len(frame), self.manifest["expected"]["splits"][role]["records"])
            roles.append(frame)
        combined = pd.concat(roles, ignore_index=True).set_index("record_id").sort_index()
        original = self.frame.set_index("record_id").sort_index()
        pd.testing.assert_frame_equal(combined[original.columns], original)
        pd.testing.assert_frame_equal(combined[["group_id", "split"]],
                                      assignments.set_index("record_id")[["group_id", "split"]].sort_index())
        by_source = combined.set_index("source_row")
        self.assertNotEqual(by_source.at["0", "group_id"], by_source.at["2", "group_id"])
        self.assertEqual(by_source.at["72", "group_id"], by_source.at["78", "group_id"])
        self.assertEqual(by_source.at["72", "split"], by_source.at["78", "split"])
        grouping = read_csv(output / "grouping_assignments.csv").set_index("record_id")
        id_by_source = self.frame.set_index("source_row").record_id
        left, right = id_by_source["72"], id_by_source["78"]
        self.assertNotEqual(grouping.at[left, "product_group_id"], grouping.at[right, "product_group_id"])
        self.assertEqual(grouping.at[left, "effective_input_group_id"], grouping.at[right, "effective_input_group_id"])
        self.assertNotEqual(by_source.at["305", "split"], "test")
        protected = read_csv(output / "protected_final_test.csv")
        self.assertEqual(len(protected), 54)
        self.assertNotIn("Segment", protected)
        self.assertNotIn("ProductName", protected)
        self.assertEqual(set(protected.record_id), set(self.frame.loc[self.frame.source_row.astype(int).ge(306), "record_id"]))
        self.assertEqual(self.input.read_bytes(), before)
        for row in assignments.to_dict("records"):
            identity = json.dumps([row[field] for field in PROVENANCE], ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(row["record_id"], hashlib.sha256(identity.encode()).hexdigest())
        verified = self.command(None, "--verify", output)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

    def test_row_order_and_csv_line_endings_reproduce_assignment_bytes(self):
        first, second = self.root / "first", self.root / "second"
        self.run_command(first, "--seed", "42")
        shuffled = self.root / "shuffled_candidate.csv"
        self.frame.drop(columns="record_id").sample(frac=1, random_state=19).to_csv(
            shuffled, index=False, lineterminator="\r\n")
        self.run_command(second, input_path=shuffled)
        for name in ("assignments.csv", "rule_assignments.csv", "protected_final_test.csv", "train.csv", "validation.csv", "test.csv"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_default_protocol_rejects_another_population_without_creating_a_split(self):
        output = self.root / "wrong-population"
        result = self.command(output, use_fixture=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertNotIn("Traceback", result.stderr)

    def test_frozen_settings_cannot_be_overridden(self):
        config = self.root / "changed.json"
        config.write_text('{"seed": 123}', encoding="utf8")
        grouping = self.root / "grouping.json"
        grouping.write_text('{"component_representatives": 3}', encoding="utf8")
        for name, options in (("seed", ("--seed", "123")), ("split-config", ("--config", config)),
                              ("group-config", ("--grouping-config", grouping))):
            with self.subTest(name=name):
                output = self.root / name
                result = self.command(output, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_existing_output_is_refused_without_modification(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "train.csv").write_bytes(b"An existing experiment must stay untouched.\n")
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
        self.assertNotIn("Traceback", result.stderr)

    def test_input_identity_features_and_record_ids_fail_closed(self):
        source = self.frame.drop(columns="record_id")
        cases = {"missing-provenance": source.drop(columns="source_sha256"),
                 "missing-target": source.drop(columns="Segment")}
        altered = source.copy()
        altered.loc[0, "ProductName"] = "Unexpected different formula"
        cases["changed-features"] = altered
        altered = source.copy()
        altered.loc[0, "source_sha256"] = "0" * 64
        cases["changed-source"] = altered
        altered = self.frame.copy()
        altered.loc[0, "record_id"] = "0" * 64
        cases["invalid-record-id"] = altered
        cases["duplicate-source-row"] = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        for name, frame in cases.items():
            with self.subTest(name=name):
                candidate = self.root / (name + ".csv")
                frame.to_csv(candidate, index=False, lineterminator="\n")
                output = self.root / name
                result = self.command(output, input_path=candidate)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
