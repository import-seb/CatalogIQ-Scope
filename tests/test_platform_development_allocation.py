"""Whole-group Platform constraints and proposal sealing; no model execution."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.features import sha256
from catalogiq.platform_development_allocation import (
    PlatformAllocationConfig, allocate_platform_development, write_platform_proposal,
)
from catalogiq.target_development_preparation import prepare_target_development
from test_target_development_preparation import fixture, reseal


MODULE = "catalogiq.platform_development_allocation"


class PlatformDevelopmentAllocationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "frozen"
        self.source.mkdir()
        fixture(self.source)
        self.enterContext(patch("catalogiq.target_development_preparation.verify_protocol_splits", return_value={"verified": True}))
        self.config = PlatformAllocationConfig(restarts=2, refinement_passes=3, exchange_passes=2, solver_time_limit=15)

    def prepared(self):
        return prepare_target_development(self.source, "Platform")

    def population(self, data):
        """Write synthetic source exports only; never call a split generator."""
        rows = [{"record_id": f"r{index:03d}", "group_id": group, "Platform": label,
                 "split": role, "Retailer": retailer}
                for index, (group, label, role, retailer) in enumerate(data)]
        frame = pd.DataFrame(rows)
        for role in ("train", "validation"):
            frame.loc[frame.split.eq(role)].to_csv(self.source / f"{role}.csv", index=False)
        protected = pd.read_csv(self.source / "protected_final_test.csv", dtype=str, keep_default_na=False)
        pd.concat([frame[["record_id", "group_id", "split"]], protected[["record_id", "group_id", "split"]]],
                  ignore_index=True).to_csv(self.source / "assignments.csv", index=False)
        reseal(self.source)
        return self.prepared()

    def test_training_coverage_group_isolation_missing_members_and_protected_test(self):
        prepared = self.prepared()
        result = allocate_platform_development(prepared, self.config)
        frame = result.development_assignments
        self.assertEqual(len(frame), len(prepared.frame))
        self.assertTrue(frame.groupby("group_id").split.nunique().eq(1).all())
        self.assertEqual(result.summary["training_unsupported_classes"], [])
        self.assertEqual(result.summary["proposed"]["train"]["classes"], 2)
        self.assertEqual(result.summary["proposed"]["validation"]["classes"], 1)
        self.assertTrue(frame.loc[frame.group_id.eq("g3"), "split"].eq("train").all())
        self.assertEqual(set(frame.loc[frame.group_id.eq("g3"), "record_id"]), {"d", "e", "f"})
        self.assertFalse(frame.loc[frame.record_id.eq("f"), "is_labeled"].iloc[0])
        self.assertEqual(result.summary["proposed"]["validation"]["labeled_records"], 1)
        self.assertTrue(result.assignments.loc[result.assignments.split.eq("test")].reset_index(drop=True).equals(prepared.protected_test))
        missing_only_groups = set(prepared.frame.group_id) - set(prepared.labeled_frame.group_id)
        self.assertTrue(frame.loc[frame.group_id.isin(missing_only_groups), "split"].equals(
            frame.loc[frame.group_id.isin(missing_only_groups), "original_split"]))

    def test_seed_and_input_order_are_deterministic(self):
        prepared = self.prepared()
        first = allocate_platform_development(prepared, self.config)
        second = allocate_platform_development(replace(prepared, frame=prepared.frame.iloc[::-1]), self.config)
        for name in ("development_assignments", "assignments", "class_coverage", "distribution_differences"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
        self.assertEqual(first.summary, second.summary)

    def test_shared_groups_block_sinus_headache_without_losing_training_classes(self):
        prepared = self.population([
            ("p1", "Coated Tabs", "train", "Amazon"),
            ("p1", "Sinus + Headache", "train", "Amazon"),
            ("p2", "Sinus Plus", "validation", "Walmart"),
            ("p2", "Sinus + Headache", "validation", "Walmart"),
            ("c1", "Common", "train", "Amazon"),
            ("c2", "Common", "validation", "Walmart"),
        ])
        result = allocate_platform_development(prepared, self.config)
        coverage = result.class_coverage.set_index("label")
        self.assertEqual(coverage.loc["Sinus + Headache", "validation_records"], 0)
        self.assertEqual(coverage.loc["Sinus + Headache", "train_records"], 2)
        self.assertEqual(coverage.loc["Sinus + Headache", "validation_exclusion_reason"], "shares_required_training_groups")
        self.assertTrue(result.development_assignments.loc[result.development_assignments.group_id.isin(["p1", "p2"]), "split"].eq("train").all())
        self.assertEqual(result.summary["optimization"]["maximum_validation_classes"], 1)

    def test_joint_coverage_optimum_matches_exhaustive_small_example(self):
        # Odd cycle: all classes individually have two groups; representing all
        # three in validation would remove some class entirely from training.
        prepared = self.population([
            ("g1", "A", "train", "Amazon"), ("g1", "B", "train", "Amazon"),
            ("g2", "B", "validation", "Walmart"), ("g2", "C", "validation", "Walmart"),
            ("g3", "C", "train", "Amazon"), ("g3", "A", "train", "Amazon"),
        ])
        result = allocate_platform_development(prepared, self.config)
        feasible = []
        groups = sorted(prepared.frame.group_id.unique())
        all_labels = set(prepared.labeled_frame.Platform)
        for bits in itertools.product((False, True), repeat=len(groups)):
            validation = set(group for group, selected in zip(groups, bits) if selected)
            train_labels = set(prepared.frame.loc[~prepared.frame.group_id.isin(validation), "Platform"])
            if train_labels == all_labels:
                subset = prepared.frame.loc[prepared.frame.group_id.isin(validation)]
                feasible.append((subset.Platform.nunique(), len(subset)))
        optimum = max(count for count, _ in feasible)
        desired = len(prepared.labeled_frame) * self.config.validation_fraction
        nearest = min(abs(records - desired) for count, records in feasible if count == optimum)
        self.assertEqual(result.summary["optimization"]["maximum_validation_classes"], optimum)
        self.assertEqual(optimum, 2)
        self.assertAlmostEqual(result.summary["optimization"]["labeled_validation_record_error"], nearest)
        self.assertEqual(result.summary["training_unsupported_classes"], [])
        self.assertEqual(len(result.summary["validation_unsupported_classes"]), 1)
        self.assertEqual(result.summary["validation_unsupported_classes"][0]["validation_exclusion_reason"], "joint_class_coverage_constraints")

    def test_distribution_reports_reconcile_counts_and_roles(self):
        result = allocate_platform_development(self.prepared(), self.config)
        self.assertEqual(result.summary["optimization"]["margin_families"][0]["name"], "Platform")
        differences = result.distribution_differences
        for dimension in ("class", "retailer"):
            for role in ("train", "validation"):
                selected = differences.loc[differences.dimension.eq(dimension) & differences.role.eq(role)]
                self.assertEqual(selected.proposed_records.sum(), result.summary["proposed"][role]["labeled_records"])
                self.assertAlmostEqual(selected.pool_fraction.sum(), 1)
                self.assertAlmostEqual(selected.proposed_fraction.sum(), 1)
                self.assertAlmostEqual(selected.proposed_delta_pp.abs().max(), result.summary["proposed"][role][f"max_{dimension}_fraction_delta_pp"])

    def test_no_labeled_platform_records_fail_before_optimization(self):
        self.population([("g1", "", "train", "Amazon"),
                         ("g2", "null", "validation", "Walmart")])
        with patch(MODULE + ".milp") as solver, self.assertRaisesRegex(ValueError, "no labeled records"):
            allocate_platform_development(self.prepared(), self.config)
        solver.assert_not_called()

    def test_nonplatform_and_invalid_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Platform"):
            allocate_platform_development(prepare_target_development(self.source, "Brand"), self.config)
        for options in ({"seed": -1}, {"seed": True}, {"validation_fraction": 1},
                        {"restarts": 0}, {"solver_time_limit": 0}, {"solver_time_limit": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PlatformAllocationConfig(**options)

    def test_mutated_protected_group_or_stale_preparation_is_rejected(self):
        prepared = self.prepared()
        contaminated = prepared.frame.copy()
        contaminated.loc[0, "group_id"] = "gz"
        with self.assertRaisesRegex(ValueError, "Protected test"):
            allocate_platform_development(replace(prepared, frame=contaminated), self.config)
        stale = prepared.frame.copy()
        stale.loc[0, "Platform"] = "New class"
        with self.assertRaisesRegex(ValueError, "support changed"):
            allocate_platform_development(replace(prepared, frame=stale), self.config)

    def test_solver_failure_creates_no_proposal(self):
        output = self.root / "proposal-v1"
        with patch(MODULE + ".milp") as solver:
            solver.return_value.success = False
            solver.return_value.message = "time limit"
            with self.assertRaisesRegex(ValueError, "time limit"):
                write_platform_proposal(self.source, output, self.config)
        self.assertFalse(output.exists())

    def test_writer_preserves_inputs_and_seals_review_outputs(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        output = self.root / "proposal-v1"
        result = write_platform_proposal(self.source, output, self.config)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.source.iterdir()})
        self.assertEqual((output / "protected_final_test.csv").read_bytes(), before["protected_final_test.csv"])
        self.assertFalse((output / "test.csv").exists())
        completion = json.loads((output / "completion.json").read_text(encoding="utf8"))
        self.assertEqual(completion["status"], "proposed; pending review")
        for name, expected in completion["artifact_sha256"].items():
            self.assertEqual(sha256(output / name), expected)
        assignments = pd.read_csv(output / "assignments.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(assignments, result.assignments)
        with self.assertRaisesRegex(ValueError, "new proposal"):
            write_platform_proposal(self.source, output, self.config)

    def test_writer_cannot_write_into_source_and_never_reads_test(self):
        for output in (self.source, self.source / "nested-proposal"):
            with self.assertRaisesRegex(ValueError, "outside"):
                write_platform_proposal(self.source, output, self.config)
        original_read, original_hash = pd.read_csv, sha256
        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_read(path, *args, **kwargs)
        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_hash(path)
        with patch(MODULE + ".pd.read_csv", side_effect=guarded_read), \
                patch(MODULE + ".sha256", side_effect=guarded_hash), \
                patch("catalogiq.target_development_preparation.sha256", side_effect=guarded_hash):
            write_platform_proposal(self.source, self.root / "proposal-v1", self.config)

    def test_source_change_during_allocation_blocks_writing(self):
        original_allocate = allocate_platform_development
        def changing_allocate(prepared, config):
            result = original_allocate(prepared, config)
            with (self.source / "train.csv").open("a", encoding="utf8") as stream:
                stream.write("changed\n")
            return result
        output = self.root / "proposal-v1"
        with patch(MODULE + ".allocate_platform_development", side_effect=changing_allocate), self.assertRaisesRegex(ValueError, "Frozen inputs changed"):
            write_platform_proposal(self.source, output, self.config)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
