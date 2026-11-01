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
                                         run_classical, source_fingerprints, measure_scores, write_json)
from catalogiq.classical_allocations import load_approved_targets
from catalogiq.classical_contract import align_classical_contract
from catalogiq.classical_data import TARGETS, load_target_development
from catalogiq.features import sha256
from catalogiq.cli import classical_main
from catalogiq.segment_baseline import BaselineConfig, build_baseline_texts
from catalogiq.split_protocol import export_final_splits
from catalogiq.splitting import record_ids
from catalogiq.segment_transformer import _hash_items


def allocation_fixture(package, source):
    directory = package / "brand-fixture"
    directory.mkdir(parents=True)
    original = pd.concat([pd.read_csv(source / f"{r}.csv", dtype=str, keep_default_na=False)
                          for r in ("train", "validation")], ignore_index=True).sort_values("record_id")
    metadata = original[["record_id", "group_id", "split", "Brand"]].rename(columns={"split": "original_split"})
    metadata["split"] = metadata.original_split.map({"train": "validation", "validation": "train"})
    metadata["is_labeled"] = True
    protected = pd.read_csv(source / "protected_final_test.csv", dtype=str, keep_default_na=False)
    identities = ["record_id", "group_id", "split"]
    pd.concat([metadata[identities], protected[identities]], ignore_index=True).to_csv(directory / "assignments.csv", index=False)
    protected.to_csv(directory / "protected_final_test.csv", index=False)
    metadata.to_csv(directory / "development_assignments.csv", index=False)
    train = metadata.loc[metadata.split.eq("train")]
    validation = metadata.loc[metadata.split.eq("validation")]
    names = sorted(train.Brand.unique())
    pd.DataFrame({"label": names}).to_csv(directory / "class_coverage.csv", index=False)
    source_hashes = {n: sha256(source / n) for n in ("train.csv", "validation.csv", "assignments.csv", "protected_final_test.csv", "grouping_freeze.json")}
    source_seal = sha256(source / "completion.json")
    write_json(directory / "protocol.json", {"preparation": {"target_column": "Brand",
        "completion_sha256": source_seal, "consumed_artifact_sha256": source_hashes}})
    write_json(directory / "completion.json", {"artifact_sha256": {p.name: sha256(p) for p in directory.iterdir()}})
    handoff = {"format": "catalogiq-approved-classical-split-handoff-v1",
        "source": {"completion_sha256": source_seal, "protected_test_rows": len(protected), "protected_test_groups": protected.group_id.nunique()},
        "targets": {"Brand": {"directory": directory.name, "approved_completion_sha256": sha256(directory / "completion.json"),
            "source_consumed_artifact_sha256": source_hashes, "label_names": names,
            "development_rows_including_missing": len(original), "labeled_training_rows": len(train),
            "labeled_validation_rows": len(validation), "ordered_training_record_ids_sha256": _hash_items(train.record_id.tolist()),
            "validation_record_ids_sha256": _hash_items(validation.record_id.tolist())}},
        "artifact_sha256": {p.relative_to(package).as_posix(): sha256(p) for p in directory.iterdir()}}
    write_json(package / "handoff_manifest.json", handoff)
    return sha256(package / "handoff_manifest.json"), handoff


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
    def test_union_macro_includes_zero_support_and_unseen_truth(self):
        scores = measure_scores(["a", "unseen"], ["a", "a"], np.array([[.8, .1, .1], [.8, .1, .1]]), ["a", "b", "c"])
        self.assertAlmostEqual(scores["macro_f1"], 1 / 6)
        self.assertAlmostEqual(scores["observed_label_macro_f1"], 1 / 3)
        self.assertEqual(scores["macro_f1_labels"], ["a", "b", "c", "unseen"])
        self.assertEqual(scores["unseen_validation_rows"], 1)
        self.assertIsNone(scores["multiclass_brier_sum"])
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

    def test_probability_mapping_is_independent_of_json_key_order(self):
        frame = pd.concat([development()] * 3, ignore_index=True)
        frame["source_row"] = [str(i) for i in range(len(frame))]
        frame["group_id"] = [f"group-{i}" for i in range(len(frame))]
        frame["record_id"] = record_ids(frame)
        frame["split"] = ["train" if i < 24 else "validation" for i in range(len(frame))]
        frame["Segment"] = [f"class-{i % 12:02d}" for i in range(len(frame))]
        frame["ProductName"] = [f"product variant{i % 12} tablets" for i in range(len(frame))]
        bundle, _, _, _ = fit_target(frame, "Segment", BaselineConfig(min_df=1), latency_repeats=1)
        inputs = frame.loc[frame.split.eq("validation")].drop(columns=list(TARGETS))
        expected = predict_classical(bundle, inputs)
        # JSON key sorting places probability_10 before probability_2.
        bundle["label_mapping"] = json.loads(json.dumps(bundle["label_mapping"], sort_keys=True))
        actual = predict_classical(bundle, inputs)
        pd.testing.assert_frame_equal(actual, expected)

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

    def test_approved_allocation_reassigns_whole_groups_and_refuses_other_source(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            with self.assertRaisesRegex(ValueError, "requires the approved allocation"):
                run_classical(self.splits, package / "unapproved", targets=["Brand"], train=True,
                    allow_unseen_validation=True, protocol_dir=self.protocol, progress=None)
            self.assertFalse((package / "unapproved").exists())
            pin, handoff = allocation_fixture(package, self.splits)
            views, audit = load_approved_targets(self.splits, package, ["Brand"], manifest_sha256=pin, protocol_dir=self.protocol)
            profile = audit["profiles"]["Brand"]
            self.assertEqual(profile["allocation_id"], "brand-fixture")
            self.assertEqual(profile["validation_record_ids_sha256"], handoff["targets"]["Brand"]["validation_record_ids_sha256"])
            self.assertFalse(set(views["Brand"].query("split == 'train'").group_id) & set(views["Brand"].query("split == 'validation'").group_id))
            handoff["source"]["completion_sha256"] = "0" * 64
            write_json(package / "handoff_manifest.json", handoff)
            with self.assertRaisesRegex(ValueError, "Different pinned V4 source"):
                load_approved_targets(self.splits, package, ["Brand"], manifest_sha256=sha256(package / "handoff_manifest.json"), protocol_dir=self.protocol)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_approved_targets(self.splits, package, ["Brand"], manifest_sha256=pin, protocol_dir=self.protocol)

    def test_contract_revision_preserves_artifacts_and_exports_ids_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "old"
            run_classical(self.splits, old, targets=["Mnfr"], protocol_dir=self.protocol,
                config=BaselineConfig(min_df=1), train=True, progress=None)
            original = {p.relative_to(old): sha256(p) for p in old.rglob("*") if p.is_file()}
            with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("refit")), patch("joblib.load", side_effect=AssertionError("pickle")):
                aligned = align_classical_contract(old, self.splits, root / "new", protocol_dir=self.protocol)
            self.assertFalse(aligned["revision_training_performed"])
            for name, digest in original.items():
                self.assertEqual(sha256(old / name), digest)
            for arm in ("base", "category"):
                path = root / "new" / "Mnfr" / arm
                manifest = json.loads((path / "manifest.json").read_text())
                self.assertEqual(manifest["macro_f1_class_set"], "model_vocabulary_union_truth")
                self.assertIsNone(manifest["inference_latency_ms"])
                self.assertEqual(manifest["out_of_vocabulary_true_labels"], 0)
                self.assertEqual(manifest["source_completion_sha256"], sha256(self.splits / "completion.json"))
                self.assertEqual(list(pd.read_csv(path / "protected_test_identity_registry.csv")), ["record_id", "group_id"])
                for name in ("predictions.csv", "model.joblib"):
                    self.assertEqual(sha256(path / name), sha256(old / "Mnfr" / arm / name))
            package_classical(root / "new", root / "predictions.zip")
            with zipfile.ZipFile(root / "predictions.zip") as z:
                self.assertFalse(any(n.endswith("model.joblib") for n in z.namelist()))
                self.assertIn("Mnfr/base/protected_test_identity_registry.csv", z.namelist())


if __name__ == "__main__":
    unittest.main()
