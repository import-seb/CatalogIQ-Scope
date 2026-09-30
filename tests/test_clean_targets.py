"""Behavioral checks for notebook-derived label cleaning."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.cleaning import (
    TARGET_COLUMNS, check_single_parent_counts, clean_training, load_source, run, sha256,
    target_distributions,
)


def sample_frame():
    rows = []
    for i in range(40):
        rows.append({
            "Mnfr": "All others", "Brand": "Example", "Platform": pd.NA,
            "Segment": "Lifestyle CHC", "Sub-Segment": "Other", "TargetAgeGroup": "Adult",
            "ProductBrand": "Example", "ProductName": "Example product " + "x" * (10 * (i % 10)), "Retailer": "Shop",
            "ProductCategory": "Health", "Sku": "000123", "Upc": "001234567890",
            "MDM_Id": str(i + 100),
        })
    frame = pd.DataFrame(rows, dtype="string")
    frame["dataset"] = "training.csv"
    frame["source_sha256"] = "fixture"
    frame["source_row"] = range(len(frame))
    return frame


class CleanTargetsTests(unittest.TestCase):
    def test_confirmed_rules_only_and_input_not_mutated(self):
        frame = sample_frame()
        frame.loc[0, "Sub-Segment"] = "Cold / Flu"
        frame.loc[1, "Sub-Segment"] = "Other Lifestyle"
        frame.loc[2, "Brand"] = "Tylenol"
        frame.loc[3, ["Brand", "Mnfr"]] = ["Benadryl", pd.NA]
        frame.loc[4, "Brand"] = "UnItemised brand"
        frame.loc[5, "Brand"] = "Other Brands"
        frame.loc[6, "Brand"] = pd.NA
        frame.loc[6, "ProductBrand"] = "Tylenol"
        original = frame.copy(deep=True)
        result = clean_training(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result.cleaned.loc[0, "Sub-Segment"], "Cold/Flu")
        self.assertEqual(result.cleaned.loc[1, "Sub-Segment"], "Other Lifestyle CHC")
        self.assertEqual(result.cleaned.loc[2, "Mnfr"], "All others")
        self.assertTrue(pd.isna(result.cleaned.loc[3, "Mnfr"]))
        self.assertEqual(len(result.changes), 2)
        for column in ["Mnfr", "Brand", "Platform", "Segment", "TargetAgeGroup", "Sku", "Upc"]:
            pd.testing.assert_series_equal(result.cleaned[column], original.loc[result.cleaned.index, column])
        self.assertEqual(result.cleaned.loc[6, "Mnfr"], "All others")

    def test_multiple_reasons_and_provenance_not_numeric_evidence(self):
        frame = sample_frame()
        for column in [c for c in frame if c not in ["dataset", "source_sha256", "source_row"]]:
            frame.loc[0, column] = "Text"
        frame.loc[0, "Brand"] = "lowercase"
        frame.loc[1, "Mnfr"] = "Unexpected"
        result = clean_training(frame)
        self.assertTrue({2, 5, 6}.issubset(result.quarantine.loc[0, "Q_REASON"]))
        self.assertIn(6, result.quarantine.loc[1, "Q_REASON"])
        self.assertEqual(result.quarantine.loc[1, "Mnfr"], "Unexpected")
        self.assertNotIn(0, result.cleaned.index)
        self.assertEqual(set(result.cleaned.index) | set(result.quarantine.index), set(frame.index))
        self.assertFalse(set(result.cleaned.index) & set(result.quarantine.index))
        self.assertEqual(len(result.quarantine), 2)

    def test_flag_does_not_block_approved_correction(self):
        frame = sample_frame()
        frame.loc[0, "Brand"] = "lowercase"
        frame.loc[0, "Sub-Segment"] = "Cold / Flu"
        result = clean_training(frame)
        self.assertIn(2, result.cleaned.loc[0, "Q_REASON"])
        self.assertEqual(result.cleaned.loc[0, "Sub-Segment"], "Cold/Flu")
        self.assertEqual(len(result.cleaned), len(frame))

    def test_rarity_alone_is_review_only(self):
        frame = sample_frame().iloc[:31].copy()
        frame["Brand"] = pd.array(["Rare"] + [f"Common{x}" for x in range(6) for _ in range(5)], dtype="string")
        result = clean_training(frame)
        self.assertIn(0, result.cleaned.index)
        self.assertIn(3, result.cleaned.loc[0, "Q_REASON"])
        self.assertEqual(result.cleaned.loc[0, "Brand"], "Rare")

    def test_hierarchy_shared_parents_are_not_violations(self):
        frame = pd.DataFrame({"Segment": ["A", "B", "B", "C", None, None],
                              "Sub-Segment": ["Shared", "Shared", "Shared", "Single", "Single", "Single"]})
        violations = check_single_parent_counts(frame)
        self.assertEqual(violations["Sub-Segment"].tolist(), ["Single"])
        self.assertEqual(violations["difference"].tolist(), [2])

    def test_roundtrip_preserves_identifiers_target_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = sample_frame().drop(columns=["dataset", "source_sha256", "source_row"])
            train_path, target_path = root / "train.csv", root / "target.csv"
            source.loc[0, "Brand"] = "lowercase"
            source.to_csv(train_path, index=False)
            source[TARGET_COLUMNS] = pd.NA
            source.to_csv(target_path, index=False)
            before = [sha256(train_path), sha256(target_path)]
            report = run(train_path, target_path, root / "result")
            self.assertEqual(before, [sha256(train_path), sha256(target_path)])
            target = pd.read_csv(root / "result/target_unchanged.csv", dtype="string")
            pd.testing.assert_frame_equal(target[source.columns], source.astype("string"))
            cleaned = pd.read_csv(root / "result/training_cleaned.csv", dtype="string")
            self.assertTrue(cleaned["Sku"].eq("000123").all())
            self.assertEqual(report["rows"]["training_input"], 40)
            self.assertEqual(report["rows"]["training_cleaned"], 40)
            self.assertEqual(report["rows"]["training_flagged"], 1)
            self.assertEqual(report["rows"]["training_quarantine"], 0)
            self.assertTrue((root / "result/training_quarantine.csv").exists())
            self.assertIn(2, json.loads(cleaned.loc[0, "Q_REASON"]))
            saved = json.loads((root / "result/summary.json").read_text())
            self.assertEqual(saved["target_distributions"], report["target_distributions"])
            for column in TARGET_COLUMNS:
                distribution = saved["target_distributions"][column]
                self.assertEqual(sum(distribution["before"].values()), 40)
                self.assertEqual(sum(distribution["after"].values()), len(cleaned))
            self.assertEqual(saved["target_distributions"]["Platform"]["before"]["<missing>"], 40)
            self.assertEqual(saved["target_distributions"]["Mnfr"]["after"]["<missing>"], 0)
            self.assertEqual(json.loads((root / "result/summary.json").read_text())["label_changes"], 0)
            with self.assertRaises(FileExistsError):
                run(train_path, target_path, root / "result")

    def test_input_schema_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({"Brand": ["Example"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing target columns"):
                load_source(path)

    def test_distributions_include_missing_removed_and_new_labels(self):
        before = pd.DataFrame({c: ["Old", "Old", pd.NA, "Removed"] for c in TARGET_COLUMNS})
        after = pd.DataFrame({c: ["New", "Old", pd.NA] for c in TARGET_COLUMNS})
        report = target_distributions(before, after)
        self.assertEqual(set(report), set(TARGET_COLUMNS))
        for column in TARGET_COLUMNS:
            self.assertEqual(report[column]["before"],
                             {"<missing>": 1, "New": 0, "Old": 2, "Removed": 1})
            self.assertEqual(report[column]["after"],
                             {"<missing>": 1, "New": 1, "Old": 1, "Removed": 0})
        empty = target_distributions(before, after.iloc[:0])
        self.assertEqual(sum(empty["Brand"]["after"].values()), 0)

    def test_summary_marker_cannot_hide_a_literal_label(self):
        frame = pd.DataFrame({c: ["<missing>", pd.NA] for c in TARGET_COLUMNS})
        with self.assertRaisesRegex(ValueError, "reserved summary label"):
            target_distributions(frame, frame)


if __name__ == "__main__":
    unittest.main()
