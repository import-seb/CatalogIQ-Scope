"""Exercise private handoffs against the real portable verifier and public fixtures."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from catalogiq.cli import segment_baseline_handoff_main
from catalogiq.segment_baseline import BaselineConfig, run_baseline
from catalogiq.segment_baseline_handoff import package_baseline_handoff
from catalogiq.split_protocol import export_final_splits


class BaselineHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        fixture_path = Path(__file__).parent / "fixtures/portable_split_protocol/build_fixture.py"
        spec = importlib.util.spec_from_file_location("handoff_protocol_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cls.bundle = cls.root / "bundle"
        production = Path(__file__).parents[1] / "src/catalogiq/resources/segment_final_20261009"
        frame, _ = fixture.materialize(cls.bundle, production)
        frame["ProductCategory"] = "Synthetic > Product"
        candidate = cls.root / "candidate.csv"
        frame.drop(columns="record_id").to_csv(candidate, index=False)
        cls.splits, cls.run_dir = cls.root / "splits", cls.root / "run"
        export_final_splits(candidate, cls.splits, protocol_dir=cls.bundle, progress=lambda _: None)
        with patch("catalogiq.split_protocol.DEFAULT_PROTOCOL", cls.bundle):
            run_baseline(None, None, None, cls.run_dir, BaselineConfig(min_df=1),
                         split_dir=cls.splits, train=True, progress=None)

    def setUp(self):
        self.protocol_patch = patch("catalogiq.split_protocol.DEFAULT_PROTOCOL", self.bundle)
        self.protocol_patch.start()
        self.addCleanup(self.protocol_patch.stop)

    def reseal(self, run, name):
        path = run / "summary.json"
        summary = json.loads(path.read_text())
        summary["artifact_sha256"][name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(summary), encoding="utf-8")

    def test_cli_portable_deterministic_bundle_without_models_test_or_network(self):
        real_open = Path.open
        def guard(path, *args, **kwargs):
            if path.name in {"test.csv", "model.joblib"}:
                raise AssertionError(f"Prohibited file access: {path.name}")
            return real_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            with patch.object(Path, "open", guard), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network access")), \
                 patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("Refitting")), \
                 patch("joblib.load", side_effect=AssertionError("Loading a model")), \
                 patch("sys.stdout", new_callable=io.StringIO):
                code = segment_baseline_handoff_main(["--run-dir", str(self.run_dir), "--split-dir", str(self.splits),
                                                      "--output-dir", str(first)])
                package_baseline_handoff(self.run_dir, self.splits, second)
            self.assertEqual(code, 0)
            self.assertEqual((first / "evaluation.zip").read_bytes(), (second / "evaluation.zip").read_bytes())
            with zipfile.ZipFile(first / "evaluation.zip") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertFalse(manifest["test_opened"])
                self.assertFalse(manifest["probabilities_calibrated"])
                self.assertNotIn(str(self.root), archive.read("manifest.json").decode())
                self.assertNotIn("protocol.json", archive.namelist())
                self.assertFalse(any(name.endswith(".joblib") for name in archive.namelist()))
                for name, digest in manifest["files_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

    def test_checksum_and_incomplete_run_fail_without_output(self):
        for fault in ("checksum", "prepared"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                copied = root / "run"
                shutil.copytree(self.run_dir, copied)
                if fault == "checksum":
                    with (copied / "base/predictions.csv").open("a") as stream:
                        stream.write("tampered\n")
                else:
                    summary = json.loads((copied / "summary.json").read_text())
                    summary["status"] = "prepared"
                    (copied / "summary.json").write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    package_baseline_handoff(copied, self.splits, root / "out")
                self.assertFalse((root / "out").exists())

    def test_resealed_bad_predictions_cannot_pass_validation(self):
        for fault in ("duplicate", "missing", "train", "test", "source_key", "group", "label",
                      "extra_text", "negative", "nan", "sum", "confidence", "prediction"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                copied = root / "run"
                shutil.copytree(self.run_dir, copied)
                relative = "base/predictions.csv"
                predictions = pd.read_csv(copied / relative, dtype=str, keep_default_na=False)
                if fault == "duplicate":
                    predictions = pd.concat([predictions, predictions.iloc[:1]], ignore_index=True)
                elif fault == "missing":
                    predictions = predictions.iloc[1:]
                elif fault in {"train", "test"}:
                    predictions.loc[0, "split"] = fault
                elif fault == "source_key":
                    predictions.loc[0, "source_row"] = "999999"
                elif fault == "group":
                    predictions.loc[0, "group_id"] = "foreign-group"
                elif fault == "label":
                    predictions.loc[0, "true_label"] = "wrong label"
                elif fault == "extra_text":
                    predictions["ProductName"] = "PRIVATE TEXT"
                elif fault in {"negative", "nan", "sum"}:
                    predictions.loc[0, "probability_0"] = {"negative": "-0.5", "nan": "NaN", "sum": "1"}[fault]
                elif fault == "confidence":
                    predictions.loc[0, "confidence"] = "0"
                else:
                    predictions.loc[0, "predicted_label"] = "wrong prediction"
                predictions.to_csv(copied / relative, index=False)
                self.reseal(copied, relative)
                with self.assertRaises(ValueError):
                    package_baseline_handoff(copied, self.splits, root / "out")
                self.assertFalse((root / "out").exists())

    def test_class_order_protocol_and_metric_mismatch_fail(self):
        for fault in ("mapping", "protocol", "metric", "confusion", "errors"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                copied = root / "run"
                shutil.copytree(self.run_dir, copied)
                if fault == "mapping":
                    name = "category/label_mapping.json"
                    value = json.loads((copied / name).read_text())
                    value["probability_0"], value["probability_1"] = value["probability_1"], value["probability_0"]
                elif fault == "protocol":
                    name = "protocol.json"
                    value = json.loads((copied / name).read_text())
                    value["development"]["protocol_id"] = "superseded-v3"
                elif fault == "metric":
                    name = "base/metrics.json"
                    value = json.loads((copied / name).read_text())
                    value["baseline"]["macro_f1"] = -1
                elif fault == "confusion":
                    name = "base/confusion_matrix.csv"
                    frame = pd.read_csv(copied / name, index_col=0)
                    frame.iloc[0, 0] += 1
                    frame.to_csv(copied / name)
                else:
                    name = "base/validation_errors.csv"
                    frame = pd.read_csv(copied / "base/predictions.csv", dtype=str, keep_default_na=False)
                    frame.iloc[0:0].assign(ProductName=[]).to_csv(copied / name, index=False)
                if fault in {"mapping", "protocol", "metric"}:
                    (copied / name).write_text(json.dumps(value))
                self.reseal(copied, name)
                with self.assertRaises(ValueError):
                    package_baseline_handoff(copied, self.splits, root / "out")
                self.assertFalse((root / "out").exists())

    def test_output_guards_preserve_existing_and_source_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                package_baseline_handoff(self.run_dir, self.splits, folder)
        for path in (self.run_dir / "new", self.splits / "new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                package_baseline_handoff(self.run_dir, self.splits, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
