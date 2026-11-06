"""Shared allocation engine: target isolation and preserved Platform outputs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.features import sha256
from catalogiq.platform_development_allocation import allocate_platform_development
from catalogiq.target_development_allocation import (
    TargetAllocationConfig, allocate_target_development, write_target_proposal,
)
from catalogiq.target_development_preparation import prepare_target_development
from test_target_development_preparation import fixture, reseal


class TargetDevelopmentAllocationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "frozen"
        self.source.mkdir()
        fixture(self.source)
        self.enterContext(patch("catalogiq.target_development_preparation.verify_protocol_splits", return_value={"verified": True}))
        self.config = TargetAllocationConfig(restarts=2, refinement_passes=3, exchange_passes=2, solver_time_limit=15)

    def test_brand_and_subsegment_coverage_integrity_and_determinism(self):
        for target in ("Brand", "Sub-Segment"):
            with self.subTest(target=target):
                prepared = prepare_target_development(self.source, target)
                first = allocate_target_development(prepared, self.config)
                second = allocate_target_development(replace(prepared, frame=prepared.frame.iloc[::-1]), self.config)
                for name in ("assignments", "development_assignments", "class_coverage", "distribution_differences"):
                    pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
                self.assertEqual(first.summary, second.summary)
                self.assertEqual(first.summary["target_column"], target)
                self.assertEqual(first.summary["proposed"]["train"]["classes"], 2)
                self.assertEqual(first.summary["proposed"]["validation"]["classes"], 1)
                self.assertTrue(first.assignments.groupby("group_id").split.nunique().eq(1).all())
                frame = first.development_assignments
                self.assertEqual(len(frame), 10)
                self.assertEqual(frame.is_labeled.sum(), 5)
                self.assertEqual(set(frame.loc[frame.group_id.eq("g3"), "split"]), {"train"})
                pd.testing.assert_frame_equal(first.assignments.loc[first.assignments.split.eq("test")].reset_index(drop=True), prepared.protected_test)

    def test_platform_matches_pre_refactor_golden_outputs(self):
        prepared = prepare_target_development(self.source, "Platform")
        result = allocate_target_development(prepared, self.config)
        expected = {
            "assignments": "db75a2e1b7d9b18869e2c49d07211ab11de6bf9cff16954345396c702ec4b370",
            "development_assignments": "521dbca2fe451a20ddcb208e36b9ad4ec3236d847cb148e883349212f34b6379",
            "class_coverage": "38f3d6f58c4f6cc08666e30164f481af64845d998eb8be6aa560cb546fac466c",
            "distribution_differences": "271b4d96285d1391be761b447a65631002ed577c6b931ea9bd67438fd0303b5d",
        }
        for name, digest in expected.items():
            encoded = getattr(result, name).to_csv(index=False, lineterminator="\n").encode("utf8")
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
        legacy = allocate_platform_development(prepared, self.config)
        self.assertEqual(result.summary, legacy.summary)
        pd.testing.assert_frame_equal(result.assignments, legacy.assignments)

    def test_target_selection_does_not_depend_on_other_labels(self):
        for role in ("train", "validation"):
            path = self.source / f"{role}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["Platform"] = ""
            frame["Sub-Segment"] = frame["Sub-Segment"].replace({"Rare": "Different"})
            frame.to_csv(path, index=False)
        reseal(self.source)
        brand = allocate_target_development(prepare_target_development(self.source, "Brand"), self.config)
        subsegment = allocate_target_development(prepare_target_development(self.source, "Sub-Segment"), self.config)
        self.assertEqual(set(brand.class_coverage.label), {"Common", "Rare"})
        self.assertEqual(set(subsegment.class_coverage.label), {"Common", "Different"})

    def test_shared_singleton_groups_block_validation_for_both_new_targets(self):
        # Common spans g1 and g3, each pinned by a different singleton class.
        for role in ("train", "validation"):
            path = self.source / f"{role}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["group_id"] = frame.group_id.replace({"g2": "g3"})
            for target in ("Brand", "Sub-Segment"):
                frame.loc[frame.record_id.eq("b"), target] = "RareTrain"
            frame.to_csv(path, index=False)
        path = self.source / "assignments.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["group_id"] = frame.group_id.replace({"g2": "g3"})
        frame.to_csv(path, index=False)
        reseal(self.source)
        for target in ("Brand", "Sub-Segment"):
            with self.subTest(target=target):
                result = allocate_target_development(prepare_target_development(self.source, target), self.config)
                common = result.class_coverage.set_index("label").loc["Common"]
                self.assertEqual(common.development_groups, 2)
                self.assertEqual(common.validation_records, 0)
                self.assertEqual(common.validation_exclusion_reason, "shares_required_training_groups")
                self.assertEqual(result.summary["training_unsupported_classes"], [])

    def test_new_target_writers_never_read_or_hash_test_csv(self):
        original_read, original_hash = pd.read_csv, sha256
        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_read(path, *args, **kwargs)
        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_hash(path)
        with patch("catalogiq.target_development_allocation.pd.read_csv", side_effect=guarded_read), \
                patch("catalogiq.target_development_allocation.sha256", side_effect=guarded_hash), \
                patch("catalogiq.target_development_preparation.sha256", side_effect=guarded_hash):
            for target in ("Brand", "Sub-Segment"):
                write_target_proposal(self.source, self.root / target, target, self.config)

    def test_reports_include_consistent_low_support_counts(self):
        for target in ("Brand", "Sub-Segment"):
            result = allocate_target_development(prepare_target_development(self.source, target), self.config)
            for version, prefix in (("original", "original_validation"), ("proposed", "validation")):
                report = result.summary["validation_support"][version]
                present = result.class_coverage.loc[result.class_coverage[f"{prefix}_records"].gt(0)]
                self.assertEqual(sum(report["record_support_bins"].values()), len(present))
                self.assertEqual(sum(report["group_support_bins"].values()), len(present))
                expected = present.loc[present[f"{prefix}_records"].lt(5), f"{prefix}_records"].sum()
                self.assertEqual(report["records_in_classes_below_five"], expected)

    def test_separate_target_proposals_are_sealed_and_do_not_touch_source(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        for target, slug in (("Brand", "brand"), ("Sub-Segment", "sub_segment")):
            output = self.root / f"{slug}_v1"
            result = write_target_proposal(self.source, output, target, self.config)
            protocol = json.loads((output / "protocol.json").read_text(encoding="utf8"))
            self.assertEqual(protocol["preparation"]["target_column"], target)
            self.assertEqual(result.summary["target_column"], target)
            completion = json.loads((output / "completion.json").read_text(encoding="utf8"))
            for name, digest in completion["artifact_sha256"].items():
                self.assertEqual(sha256(output / name), digest)
            self.assertEqual((output / "protected_final_test.csv").read_bytes(), before["protected_final_test.csv"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.source.iterdir()})

    def test_non_supported_targets_fail_without_output(self):
        for target in ("Segment", "Mnfr", "TargetAgeGroup", "brand"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                write_target_proposal(self.source, self.root / "invalid", target, self.config)
        self.assertFalse((self.root / "invalid").exists())

    def test_large_population_path_is_deterministic_and_preserves_constraints(self):
        # Exercise the bounded exchange path with >1,000 independent groups.
        frame = pd.DataFrame({"record_id": [f"r{i:04d}" for i in range(1021)],
            "group_id": [f"g{i:04d}" for i in range(1021)],
            "split": ["validation" if i % 6 == 0 else "train" for i in range(1021)],
            "Brand": ["A" if i % 2 else "B" for i in range(1021)],
            "Retailer": ["Amazon" if i % 3 else "Walmart" for i in range(1021)]})
        for role in ("train", "validation"):
            frame.loc[frame.split.eq(role)].to_csv(self.source / f"{role}.csv", index=False)
        protected = pd.read_csv(self.source / "protected_final_test.csv", dtype=str, keep_default_na=False)
        pd.concat([frame[["record_id", "group_id", "split"]], protected[["record_id", "group_id", "split"]]], ignore_index=True).to_csv(self.source / "assignments.csv", index=False)
        reseal(self.source)
        prepared = prepare_target_development(self.source, "Brand")
        first = allocate_target_development(prepared, self.config)
        second = allocate_target_development(prepared, self.config)
        pd.testing.assert_frame_equal(first.assignments, second.assignments, check_exact=True)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.summary["proposed"]["validation"]["labeled_records"], 180)
        self.assertEqual(first.summary["proposed"]["train"]["classes"], 2)
        self.assertEqual(first.summary["proposed"]["validation"]["classes"], 2)
        self.assertEqual(first.summary["optimization"]["exchange_search"], "seeded equal-labeled-size group pairs")


if __name__ == "__main__":
    unittest.main()
