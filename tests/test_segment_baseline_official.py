"""Exercise the baseline through the real portable verifier on public fixtures."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.cli import segment_baseline_main
from catalogiq.segment_baseline import BaselineConfig, run_baseline
from catalogiq.split_protocol import export_final_splits


class OfficialBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        fixture_path = Path(__file__).parent / "fixtures/portable_split_protocol/build_fixture.py"
        spec = importlib.util.spec_from_file_location("baseline_protocol_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cls.bundle = cls.root / "bundle"
        production = Path(__file__).parents[1] / "src/catalogiq/resources/segment_final_20261009"
        frame, _ = fixture.materialize(cls.bundle, production)
        frame["ProductCategory"] = "Synthetic > Product"
        candidate = cls.root / "candidate.csv"
        frame.drop(columns="record_id").to_csv(candidate, index=False)
        cls.splits = cls.root / "splits"
        export_final_splits(candidate, cls.splits, protocol_dir=cls.bundle, progress=lambda _: None)

    def test_complete_comparison_uses_real_seal_without_test_or_hf_access(self):
        real_open = Path.open
        real_read = pd.read_csv
        def guarded_open(path, *args, **kwargs):
            if path.name == "test.csv":
                raise AssertionError("Test file opened or hashed")
            return real_open(path, *args, **kwargs)
        def guarded_read(path, *args, **kwargs):
            if Path(path).name == "test.csv":
                raise AssertionError("Test file read")
            return real_read(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with patch("catalogiq.split_protocol.DEFAULT_PROTOCOL", self.bundle), \
                 patch.object(Path, "open", guarded_open), patch("pandas.read_csv", guarded_read), \
                 patch.dict("sys.modules", {"transformers": None, "torch": None}), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network access")):
                summary = run_baseline(None, None, None, output, BaselineConfig(min_df=1),
                                       split_dir=self.splits, train=True, progress=None)
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["test_scored"])
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertTrue(protocol["development"]["final_snapshot_seal_or_exposure_history_verified"])
            self.assertEqual(protocol["development"]["protocol_id"], "synthetic-segment-final-protocol-v1")
            a = pd.read_csv(output / "base/predictions.csv")
            b = pd.read_csv(output / "category/predictions.csv")
            self.assertEqual(a.record_id.tolist(), b.record_id.tolist())
            self.assertTrue(a.split.eq("validation").all())

    def test_modified_export_fails_before_output_or_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            copied = root / "modified"
            shutil.copytree(self.splits, copied)
            with (copied / "train.csv").open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with patch("catalogiq.split_protocol.DEFAULT_PROTOCOL", self.bundle), \
                 patch("catalogiq.segment_baseline.fit_baseline", side_effect=AssertionError("Fit called")):
                with self.assertRaises(ValueError):
                    run_baseline(None, None, None, root / "run", split_dir=copied, train=True, progress=None)
            self.assertFalse((root / "run").exists())

    def test_cli_rejects_training_from_historical_csvs(self):
        with patch("sys.stderr", new_callable=io.StringIO) as error:
            with self.assertRaises(SystemExit):
                segment_baseline_main(["--input", "old.csv", "--assignments", "old-assignments.csv",
                                       "--assignments-sha256", "a" * 64, "--train", "--output-dir", "unused"])
            self.assertIn("Training requires --split-dir", error.getvalue())


if __name__ == "__main__":
    unittest.main()
