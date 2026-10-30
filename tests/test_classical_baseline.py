"""Target-specific labels, train-only fitting, portable manifests and handoffs."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
import pandas as pd

from catalogiq.classical_baseline import (fit_target, package_classical, predict_classical,
                                         run_classical, source_fingerprints)
from catalogiq.classical_data import TARGETS, load_target_development
from catalogiq.cli import classical_main
from catalogiq.segment_baseline import BaselineConfig, build_baseline_texts
from catalogiq.split_protocol import export_final_splits
from catalogiq.splitting import record_ids


def development():
    frame = pd.DataFrame([
        {"dataset": "synthetic", "source_sha256": "a" * 64, "source_row": str(i),
         "record_id": "", "group_id": f"group-{i}", "split": "train" if i < 8 else "validation",
         "ProductName": ("allergy tablet" if i % 2 else "cold syrup") + (" validationonly" if i >= 8 else ""),
         "ProductBrand": "", "ProductDescription": "", "ProductContents": "",
         "ProductCategory": "Health > allergy" if i % 2 else "Health > cold",
         **{target: "class-a" if i % 2 else "class-b" for target in TARGETS}}
        for i in range(12)])
    frame["record_id"] = record_ids(frame)
    return frame


class ClassicalFitTests(unittest.TestCase):
    def test_every_target_uses_its_own_labels_and_persisted_prediction_adapter(self):
        for target in TARGETS:
            with self.subTest(target=target):
                frame = development()
                for other in set(TARGETS) - {target}:
                    frame[other] = "FORBIDDEN_TARGET_TEXT"
                bundle, predictions, metrics, performance = fit_target(
                    frame, target, BaselineConfig(min_df=1), include_category=True, latency_repeats=1)
                vocabulary = bundle["pipeline"].named_steps["tfidf"].vocabulary_
                self.assertNotIn("validationonly", vocabulary)
                self.assertNotIn("forbidden_target_text", vocabulary)
                self.assertEqual(set(bundle["label_mapping"].values()), {"class-a", "class-b"})
                self.assertEqual(len(predictions), 4)
                self.assertEqual(metrics["baseline"]["unseen_validation_rows"], 0)
                self.assertGreaterEqual(performance["training_seconds"], 0)
                self.assertEqual(performance["inference_latency"]["batch_size"], 4)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.joblib"
                    joblib.dump(bundle, path)
                    restored = joblib.load(path)
                    inputs = frame.loc[frame.split.eq("validation")].drop(columns=list(TARGETS))
                    actual = predict_classical(restored, inputs)
                np.testing.assert_allclose(actual.filter(like="probability_"), predictions.filter(like="probability_"))

    def test_unseen_classes_are_never_learned_or_silently_removed(self):
        frame = development()
        frame.loc[8, "Platform"] = "unseen-platform"
        with self.assertRaisesRegex(ValueError, "absent from train"):
            fit_target(frame, "Platform", BaselineConfig(min_df=1))
        bundle, predictions, metrics, _ = fit_target(frame, "Platform", BaselineConfig(min_df=1),
                                                    allow_unseen_validation=True, latency_repeats=1)
        self.assertEqual(len(predictions), 4)
        self.assertIn("unseen-platform", predictions.true_label.tolist())
        self.assertNotIn("unseen-platform", bundle["label_mapping"].values())
        self.assertEqual(metrics["baseline"]["unseen_validation_rows"], 1)
        self.assertIsNone(metrics["baseline"]["log_loss"])
        self.assertIsNone(metrics["baseline"]["multiclass_brier_sum"])
        self.assertLessEqual(metrics["baseline"]["accuracy"], 0.75)

    def test_invalid_supervised_views_and_leakage_fail(self):
        for fault in ("test", "duplicate", "group", "missing", "one_class"):
            with self.subTest(fault=fault):
                frame = development()
                if fault == "test":
                    frame.loc[8, "split"] = "test"
                elif fault == "duplicate":
                    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
                elif fault == "group":
                    frame.loc[8, "group_id"] = frame.loc[0, "group_id"]
                elif fault == "missing":
                    frame.loc[8, "Mnfr"] = " NULL "
                else:
                    frame.loc[frame.split.eq("train"), "Mnfr"] = "one"
                with self.assertRaises(ValueError):
                    fit_target(frame, "Mnfr", BaselineConfig(min_df=1))

    def test_hashes_explain_crlf_without_changing_files(self):
        import hashlib
        from catalogiq import classical_baseline
        hashes = source_fingerprints()
        for name, item in hashes.items():
            data = Path(classical_baseline.__file__).with_name(name).read_bytes()
            self.assertEqual(item["sha256_raw"], hashlib.sha256(data).hexdigest())
            self.assertEqual(item["sha256_lf"], hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest())


class ClassicalPortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        path = Path(__file__).parent / "fixtures/portable_split_protocol/build_fixture.py"
        spec = importlib.util.spec_from_file_location("multitarget_fixture", path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cls.protocol = cls.root / "protocol"
        production = Path(__file__).parents[1] / "src/catalogiq/resources/segment_final_20261009"
        frame, _ = fixture.materialize(cls.protocol, production)
        frame["ProductCategory"] = "Synthetic > Health"
        for target in set(TARGETS) - {"Segment"}:
            frame[target] = np.where(frame.source_row.astype(int).mod(4).lt(2), "first", "second")
        # Missing labels are independent of Segment and of each other.
        frame.loc[frame.source_row.astype(int).mod(7).eq(0), "Platform"] = " NULL "
        frame.loc[frame.source_row.astype(int).mod(9).eq(0), "Mnfr"] = ""
        cls.candidate = cls.root / "candidate.csv"
        frame.drop(columns="record_id").to_csv(cls.candidate, index=False)
        cls.splits = cls.root / "splits"
        export_final_splits(cls.candidate, cls.splits, protocol_dir=cls.protocol, progress=lambda _: None)

    def test_target_masks_not_segment_mask_and_test_is_never_read(self):
        real_open = Path.open
        def guard(path, *args, **kwargs):
            if path.name == "test.csv":
                raise AssertionError("Read reserved test")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard), patch("socket.socket.connect", side_effect=AssertionError("network")):
            views, audit = load_target_development(self.splits, TARGETS, protocol_dir=self.protocol)
        expected = pd.concat([pd.read_csv(self.splits / f"{r}.csv", dtype=str, keep_default_na=False)
                              for r in ("train", "validation")], ignore_index=True)
        for target in TARGETS:
            mask = ~expected[target].str.strip().str.casefold().isin(["", "null", "none", "nan"])
            self.assertEqual(set(views[target].record_id), set(expected.loc[mask, "record_id"]))
        self.assertNotEqual(set(views["Platform"].record_id), set(views["Mnfr"].record_id))
        self.assertFalse(audit["test_opened"])

    def test_cli_dry_run_and_complete_handoff_with_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("dry run fitted")), \
                 patch("sys.stdout", new_callable=io.StringIO):
                result = classical_main(["--split-dir", str(self.splits), "--protocol-dir", str(self.protocol),
                    "--targets", *TARGETS, "--output-dir", str(root / "prepared")])
            self.assertEqual(result, 0)
            run = root / "run"
            summary = run_classical(self.splits, run, targets=("Mnfr", "TargetAgeGroup"),
                protocol_dir=self.protocol, config=BaselineConfig(min_df=1), train=True, progress=None)
            self.assertEqual(summary["status"], "completed")
            archive = root / "models.zip"
            package_classical(run, archive, include_models=True)
            with zipfile.ZipFile(archive) as z:
                for target in ("Mnfr", "TargetAgeGroup"):
                    for arm in ("base", "category"):
                        manifest = json.loads(z.read(f"{target}/{arm}/manifest.json"))
                        predictions = pd.read_csv(io.BytesIO(z.read(f"{target}/{arm}/predictions.csv")), dtype=str)
                        from catalogiq.segment_transformer import _hash_items
                        self.assertEqual(manifest["validation_record_ids_sha256"], _hash_items(sorted(predictions.record_id)))
                        self.assertTrue(manifest["label_mapping"])
                        self.assertNotIn(b"\r\n", z.read(f"{target}/{arm}/manifest.json"))
                        self.assertNotIn(b"\r\n", z.read(f"{target}/{arm}/predictions.csv"))
                        self.assertIn(f"{target}/{arm}/model.joblib", z.namelist())
                self.assertFalse(any(name.endswith("test.csv") for name in z.namelist()))
            with self.assertRaises(FileExistsError):
                package_classical(run, archive)
            # Corrupting a published output must prevent creating a new handoff.
            with (run / "Mnfr/base/predictions.csv").open("a") as stream:
                stream.write("tampered\n")
            with self.assertRaises(ValueError):
                package_classical(run, root / "bad.zip")
            self.assertFalse((root / "bad.zip").exists())

    def test_unknown_protocol_or_changed_export_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "splits"
            shutil.copytree(self.splits, copied)
            with (copied / "train.csv").open("a") as stream:
                stream.write("tampered\n")
            with self.assertRaises(ValueError):
                load_target_development(copied, ("Mnfr",), protocol_dir=self.protocol)


if __name__ == "__main__":
    unittest.main()
