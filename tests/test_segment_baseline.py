"""Offline synthetic tests for the classical baseline and explicit split joins."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from catalogiq.cleaning import PROVENANCE
from catalogiq.cli import segment_baseline_main
from catalogiq.features import sha256
from catalogiq.segment_baseline import (BaselineConfig, build_baseline_texts, fit_baseline, run_baseline)
from catalogiq.segment_baseline_data import load_development
from catalogiq.segment_transformer import MODEL_FIELDS
from catalogiq.splitting import record_ids


def records():
    rows = []
    for i, (role, label) in enumerate([
        ("train", "Allergy"), ("train", "CCFS"), ("train", "Allergy"), ("train", "CCFS"),
        ("validation", "Allergy"), ("validation", "CCFS"), ("train", ""), ("validation", " NULL "),
        ("test", "TEST_ONLY_LABEL"),
    ], start=1):
        rows.append({
            "dataset": "training", "source_sha256": "a" * 64, "source_row": str(i),
            "group_id": f"group-{i}", "split": role, "Segment": label,
            "ProductName": ("allergy" if label == "Allergy" else "cold") + (" validationonly" if role == "validation" else ""),
            "ProductBrand": "", "ProductDescription": "", "ProductContents": "",
            "ProductCategory": "Health>categoryword" if label == "Allergy" else "Health > remedies",
            "Retailer": "FORBIDDEN_RETAILER", "Brand": "FORBIDDEN_TARGET",
            "MDM_InsertDateTime": "FORBIDDEN_DATE",
        })
    frame = pd.DataFrame(rows)
    frame["record_id"] = record_ids(frame)
    frame.loc[frame.split.eq("test"), [*MODEL_FIELDS, "ProductCategory"]] = "TEST_ONLY_TEXT"
    return frame


def development():
    return records().iloc[:6].copy()


def fixture(root):
    frame = records()
    candidate = root / "training_candidate.csv"
    assignments = root / "rule_assignments.csv"
    frame.drop(columns=["record_id", "group_id", "split"]).to_csv(candidate, index=False)
    # A different row order proves this is a key join, not a positional join.
    frame[[*PROVENANCE, "record_id", "group_id", "split", "Segment"]].iloc[::-1].to_csv(assignments, index=False)
    (root / "test.csv").write_text("THIS MUST NEVER BE OPENED", encoding="utf-8")
    return candidate, assignments


class SegmentBaselineTests(unittest.TestCase):
    def test_train_only_word_vocabulary_idf_and_persistence(self):
        frame = development()
        model, predictions, metrics = fit_baseline(frame, BaselineConfig(min_df=1))
        tfidf = model.named_steps["tfidf"]
        self.assertNotIn("validationonly", tfidf.vocabulary_)
        self.assertNotIn("categoryword", tfidf.vocabulary_)
        self.assertAlmostEqual(tfidf.idf_[tfidf.vocabulary_["allergy"]], np.log(5 / 3) + 1)
        self.assertEqual(predictions.source_row.tolist(), ["5", "6"])
        self.assertEqual(metrics["majority_class_selected_on_training"], "Allergy")
        self.assertEqual(metrics["baseline"]["records"], 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            probabilities = restored.predict_proba(build_baseline_texts(frame.iloc[4:]))
        np.testing.assert_allclose(probabilities, predictions.filter(like="probability_"))

    def test_category_uses_same_vectorizer_without_transformer_truncation(self):
        frame = development()
        frame["ProductDescription"] = "word " * 150 + "lateword"
        model, _, _ = fit_baseline(frame, BaselineConfig(min_df=1), include_category=True)
        self.assertEqual(list(model.named_steps), ["tfidf", "classifier"])
        self.assertIn("categoryword", model.named_steps["tfidf"].vocabulary_)
        self.assertIn("lateword", model.named_steps["tfidf"].vocabulary_)
        self.assertIn("[ProductCategory] Health > categoryword", build_baseline_texts(frame, include_category=True)[0])

    def test_allowlist_null_categories_and_no_mutation(self):
        frame = development()
        frame["ProductCategory"] = [None, "", "null", pd.NA, "A>B", "A > B"]
        original = frame.copy(deep=True)
        text = build_baseline_texts(frame, include_category=True)
        self.assertTrue(all(value.endswith("[ProductCategory] ") for value in text[:4]))
        self.assertTrue(all(value.endswith("[ProductCategory] A > B") for value in text[4:]))
        self.assertNotIn("FORBIDDEN", " ".join(text))
        self.assertNotIn("source_sha256", " ".join(text))
        pd.testing.assert_frame_equal(frame, original)
        with self.assertRaisesRegex(ValueError, "ProductCategory"):
            build_baseline_texts(frame.drop(columns="ProductCategory"), include_category=True)

    def test_validation_labels_do_not_change_model_or_probabilities(self):
        first = development()
        second = first.copy()
        second.loc[second.split.eq("validation"), "Segment"] = ["CCFS", "Allergy"]
        a, predictions_a, _ = fit_baseline(first, BaselineConfig(min_df=1), include_category=True)
        b, predictions_b, _ = fit_baseline(second, BaselineConfig(min_df=1), include_category=True)
        self.assertEqual(a.named_steps["tfidf"].vocabulary_, b.named_steps["tfidf"].vocabulary_)
        np.testing.assert_array_equal(a.named_steps["classifier"].coef_, b.named_steps["classifier"].coef_)
        np.testing.assert_array_equal(predictions_a.filter(like="probability_"), predictions_b.filter(like="probability_"))

    def test_source_key_join_is_order_independent_and_filters_missing_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, assignments = fixture(Path(folder))
            first = load_development(candidate, assignments, sha256(assignments), sha256(candidate))
            data = pd.read_csv(candidate, dtype=str, keep_default_na=False)
            data.iloc[::-1].to_csv(candidate, index=False)
            second = load_development(candidate, assignments, sha256(assignments))
            pd.testing.assert_frame_equal(first.frame, second.frame)
            self.assertEqual(first.audit["complete_one_to_one_rows"], 9)
            self.assertEqual(first.audit["labeled_development_counts"], {"train": 4, "validation": 2})
            self.assertEqual(first.audit["excluded_missing_label_counts"], {"train": 1, "validation": 1})
            self.assertFalse(first.audit["final_snapshot_seal_or_exposure_history_verified"])
            self.assertNotIn("TEST_ONLY", first.frame.to_string())

    def test_hash_mismatch_and_short_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, assignments = fixture(Path(folder))
            for value in ("a" * 64, sha256(assignments)[:8], None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_development(candidate, assignments, value)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_development(candidate, assignments, sha256(assignments), "b" * 64)

    def test_invalid_assignment_or_candidate_rejected_before_fit(self):
        for fault in ("duplicate_candidate", "duplicate_assignment", "missing_candidate", "extra_candidate",
                      "bad_record_id", "cross_group", "unknown_split", "missing_field", "changed_key"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                candidate, assignments = fixture(root)
                c = pd.read_csv(candidate, dtype=str, keep_default_na=False)
                a = pd.read_csv(assignments, dtype=str, keep_default_na=False)
                if fault == "duplicate_candidate":
                    c = pd.concat([c, c.iloc[:1]])
                elif fault == "duplicate_assignment":
                    a = pd.concat([a, a.iloc[:1]])
                elif fault == "missing_candidate":
                    c = c.iloc[1:]
                elif fault == "extra_candidate":
                    extra = c.iloc[:1].copy()
                    extra["source_row"] = "999"
                    c = pd.concat([c, extra])
                elif fault == "bad_record_id":
                    a.loc[0, "record_id"] = "forged"
                elif fault == "cross_group":
                    a.loc[a.source_row.isin(["1", "9"]), "group_id"] = "overlap-with-test"
                elif fault == "unknown_split":
                    a.loc[0, "split"] = "unknown"
                elif fault == "missing_field":
                    c = c.drop(columns="ProductName")
                else:
                    c.loc[0, "source_sha256"] = "b" * 64
                c.to_csv(candidate, index=False)
                a.to_csv(assignments, index=False)
                with patch("catalogiq.segment_baseline.fit_baseline", side_effect=AssertionError("Fit called")):
                    with self.assertRaises(ValueError):
                        run_baseline(candidate, assignments, sha256(assignments), root / "run", train=True, progress=None)
                self.assertFalse((root / "run").exists())

    def test_fit_rejects_test_rows_overlap_missing_labels_and_unseen_classes(self):
        for fault in ("test", "group_overlap", "duplicate", "missing_label", "unseen_class", "one_class"):
            with self.subTest(fault=fault):
                frame = development()
                if fault == "test":
                    frame.loc[5, "split"] = "test"
                elif fault == "group_overlap":
                    frame.loc[5, "group_id"] = frame.loc[0, "group_id"]
                elif fault == "duplicate":
                    frame = pd.concat([frame, frame.iloc[:1]])
                elif fault == "missing_label":
                    frame.loc[0, "Segment"] = "null"
                elif fault == "unseen_class":
                    frame.loc[5, "Segment"] = "Not in training"
                else:
                    frame.loc[frame.split.eq("train"), "Segment"] = "Allergy"
                with self.assertRaises(ValueError):
                    fit_baseline(frame, BaselineConfig(min_df=1))

    def test_cli_defaults_to_preparation_without_fit_or_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, assignments = fixture(root)
            with patch("catalogiq.segment_baseline.fit_baseline", side_effect=AssertionError("Fit called")), patch("sys.stdout", new_callable=io.StringIO):
                result = segment_baseline_main(["--input", str(candidate), "--assignments", str(assignments),
                                                "--assignments-sha256", sha256(assignments),
                                                "--output-dir", str(root / "run")])
            self.assertEqual(result, 0)
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertFalse(summary["model_training_performed"])
            self.assertEqual(summary["status"], "prepared")
            self.assertEqual({path.name for path in (root / "run").iterdir()}, {"protocol.json", "summary.json"})

    def test_complete_comparison_offline_without_hugging_face_or_test_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, assignments = fixture(root)
            real_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.name == "test.csv":
                    raise AssertionError("Test export was opened")
                return real_open(path, *args, **kwargs)
            with patch.dict("sys.modules", {"transformers": None, "tokenizers": None, "torch": None}), \
                 patch("socket.socket.connect", side_effect=AssertionError("Network access")), \
                 patch.object(Path, "open", guarded_open):
                summary = run_baseline(candidate, assignments, sha256(assignments), root / "run",
                                       BaselineConfig(min_df=1), train=True, progress=None)
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["test_scored"])
            self.assertFalse(summary["new_splits_created"])
            self.assertEqual(set(summary["variants"]), {"base", "category"})
            prediction_frames = []
            for variant in ("base", "category"):
                path = root / "run" / variant
                p = pd.read_csv(path / "predictions.csv", dtype={"source_row": str})
                prediction_frames.append(p)
                self.assertEqual(set(p.source_row), {"5", "6"})
                self.assertTrue(p.split.eq("validation").all())
                np.testing.assert_allclose(p.filter(like="probability_").sum(axis=1), 1)
                self.assertEqual(pd.read_csv(path / "confusion_matrix.csv", index_col=0).shape, (2, 2))
                saved = joblib.load(path / "model.joblib")
                self.assertEqual(saved["include_category"], variant == "category")
                self.assertNotIn("test_only_text", saved["pipeline"].named_steps["tfidf"].vocabulary_)
                dev = load_development(candidate, assignments, sha256(assignments)).frame
                val = dev.loc[dev.split.eq("validation")]
                np.testing.assert_allclose(saved["pipeline"].predict_proba(build_baseline_texts(val, include_category=saved["include_category"])),
                                           p.filter(like="probability_"))
            self.assertEqual(prediction_frames[0].record_id.tolist(), prediction_frames[1].record_id.tolist())
            comparison = pd.read_csv(root / "run/comparison.csv").set_index("variant")
            self.assertAlmostEqual(summary["category_minus_base_percentage_points"]["macro_f1"],
                                   100 * (comparison.loc["category", "macro_f1"] - comparison.loc["base", "macro_f1"]))
            for name, expected in summary["artifact_sha256"].items():
                self.assertEqual(sha256(root / "run" / name), expected)
            with self.assertRaises(FileExistsError):
                run_baseline(candidate, assignments, sha256(assignments), root / "run", progress=None)

    def test_cli_failure_has_no_traceback_or_output(self):
        with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", new_callable=io.StringIO) as error:
            root = Path(folder)
            with self.assertRaises(SystemExit) as result:
                segment_baseline_main(["--input", str(root / "missing"), "--assignments", str(root / "absent"),
                                       "--assignments-sha256", "a" * 64, "--output-dir", str(root / "run")])
            self.assertEqual(result.exception.code, 1)
            self.assertNotIn("Traceback", error.getvalue())
            self.assertFalse((root / "run").exists())

    def test_nonconvergence_cannot_produce_completed_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, assignments = fixture(root)
            with patch("catalogiq.segment_baseline.LogisticRegression.fit", side_effect=ConvergenceWarning("Not converged")):
                with self.assertRaises(ConvergenceWarning):
                    run_baseline(candidate, assignments, sha256(assignments), root / "run",
                                 BaselineConfig(min_df=1), train=True, progress=None)
            self.assertFalse((root / "run/summary.json").exists())

    def test_rejects_invalid_configuration(self):
        for values in ({"c": 0}, {"c": float("nan")}, {"min_df": True}, {"max_iter": 0},
                       {"ngram_max": 4}, {"seed": -1}, {"class_weight": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BaselineConfig(**values)


if __name__ == "__main__":
    unittest.main()
