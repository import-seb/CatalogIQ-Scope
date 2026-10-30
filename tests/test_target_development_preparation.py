"""Preparation constraints using synthetic exports; no allocator or model runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.features import sha256
from catalogiq.target_development_preparation import SUPPORTED_TARGETS, prepare_target_development


MODULE = "catalogiq.target_development_preparation"


def reseal(directory):
    names = ("grouping_freeze.json", "assignments.csv", "protected_final_test.csv", "train.csv", "validation.csv")
    completion = {"format": "catalogiq-portable-split-v1",
                  "artifact_sha256": {name: sha256(directory / name) for name in names}}
    (directory / "completion.json").write_text(json.dumps(completion), encoding="utf8")


def fixture(directory):
    # Several rows in one group still count as one independent example group.
    data = [
        ("a", "g1", "train", "Common"), ("b", "g1", "train", "Common"),
        ("c", "g2", "validation", "Common"), ("d", "g3", "validation", "Rare"),
        ("e", "g3", "validation", "Rare"), ("f", "g3", "validation", "null"),
        ("g", "g4", "train", ""), ("h", "g5", "validation", " NoNe "),
        ("i", "g6", "train", "nAn"), ("j", "g7", "train", "  "),
    ]
    frame = pd.DataFrame(data, columns=["record_id", "group_id", "split", "Brand"])
    frame["Platform"] = frame.Brand
    frame["Sub-Segment"] = frame.Brand
    frame["Retailer"] = "Amazon"
    frame["Segment"] = ""  # Other targets must not depend on Segment eligibility.
    frame["ProductName"] = "FORBIDDEN_FEATURE"
    for role in ("train", "validation"):
        frame.loc[frame.split.eq(role)].to_csv(directory / f"{role}.csv", index=False)
    protected = pd.DataFrame({"record_id": ["z"], "group_id": ["gz"], "split": ["test"],
                              "Brand": ["SECRET_TEST_LABEL"], "ProductName": ["SECRET_TEST_FEATURE"]})
    protected.to_csv(directory / "protected_final_test.csv", index=False)
    pd.concat([frame[["record_id", "group_id", "split"]], protected[["record_id", "group_id", "split"]]],
              ignore_index=True).to_csv(directory / "assignments.csv", index=False)
    (directory / "grouping_freeze.json").write_text(json.dumps({
        "format": "catalogiq-portable-split-v1", "grouping_version": "rule-final-v4",
        "protocol_id": "synthetic-v4"}), encoding="utf8")
    (directory / "test.csv").write_text("DO NOT READ OR HASH", encoding="utf8")
    reseal(directory)


class TargetDevelopmentPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        fixture(self.directory)
        self.verifier = self.enterContext(patch(MODULE + ".verify_protocol_splits", return_value={"verified": True}))

    def prepare(self, target="Brand"):
        return prepare_target_development(self.directory, target)

    def edit(self, name, transform, *, seal=True):
        path = self.directory / name
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        transform(frame).to_csv(path, index=False)
        if seal:
            reseal(self.directory)

    def test_support_uses_groups_and_preserves_original_roles_and_missing_rows(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        result = self.prepare()
        self.verifier.assert_called_once_with(self.directory.resolve(), verify_exports=False)
        self.assertEqual(len(result.frame), 10)
        self.assertEqual(len(result.labeled_frame), 5)
        self.assertEqual(result.frame.set_index("record_id").loc["h", "Brand"], " NoNe ")
        self.assertEqual(result.required_training_group_ids, frozenset(["g3"]))
        self.assertEqual(result.training_only_classes, ("Rare",))
        self.assertEqual(result.eligible_for_both_classes, ("Common",))
        support = result.class_support.set_index("label")
        self.assertEqual(support.loc["Common", "labeled_records"], 3)
        self.assertEqual(support.loc["Common", "development_groups"], 2)
        self.assertEqual(support.loc["Rare", "development_groups"], 1)
        self.assertEqual(support.loc["Rare", "original_validation_records"], 2)
        self.assertTrue(result.frame.loc[result.frame.group_id.eq("g3"), "original_split"].eq("validation").all())
        self.assertEqual(set(result.frame.loc[result.frame.group_id.eq("g3"), "record_id"]), {"d", "e", "f"})
        self.assertEqual(result.protected_test.to_dict("records"), [{"record_id": "z", "group_id": "gz", "split": "test"}])
        self.assertFalse(result.audit["allocation_performed"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_all_supported_targets_and_target_specific_eligibility(self):
        for target in SUPPORTED_TARGETS:
            with self.subTest(target=target):
                result = self.prepare(target)
                self.assertEqual(result.target_column, target)
                self.assertEqual(result.audit["labeled_records"], 5)
                self.assertNotIn("Segment", result.frame)
                self.assertNotIn("ProductName", result.frame)
        self.edit("train.csv", lambda frame: frame.assign(Platform=""))
        self.assertEqual(self.prepare("Brand").audit["labeled_records"], 5)
        self.assertEqual(self.prepare("Platform").audit["labeled_records"], 3)

    def test_deterministic_under_export_row_reordering(self):
        first = self.prepare()
        for name in ("train.csv", "validation.csv", "assignments.csv", "protected_final_test.csv"):
            self.edit(name, lambda frame: frame.iloc[::-1])
        second = self.prepare()
        for name in ("frame", "class_support", "class_group_support", "protected_test"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
        self.assertEqual(first.required_training_group_ids, second.required_training_group_ids)
        # Byte hashes intentionally change when exports are reordered/resealed.
        excluded = {"completion_sha256", "consumed_artifact_sha256"}
        self.assertEqual({k: v for k, v in first.audit.items() if k not in excluded},
                         {k: v for k, v in second.audit.items() if k not in excluded})
        self.assertEqual(second.audit, self.prepare().audit)

    def test_no_test_features_labels_or_other_targets_read_and_no_test_hash(self):
        original_read, original_hash = pd.read_csv, sha256

        def guarded_read(path, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            columns = kwargs["usecols"]
            if Path(path).name in ("assignments.csv", "protected_final_test.csv"):
                self.assertEqual(columns, ["record_id", "group_id", "split"])
            else:
                self.assertEqual(columns, ["record_id", "group_id", "split", "Retailer", "Brand"])
            return original_read(path, **kwargs)

        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_hash(path)

        with patch(MODULE + ".pd.read_csv", side_effect=guarded_read), patch(MODULE + ".sha256", side_effect=guarded_hash):
            self.prepare()

    def test_singleton_constraints_include_other_labels_and_missing_group_members(self):
        # Common now occupies g1 and g3. Pinning g3 for Rare must not split it.
        self.edit("validation.csv", lambda frame: frame.assign(group_id=frame.group_id.replace({"g2": "g3"})))
        self.edit("assignments.csv", lambda frame: frame.assign(group_id=frame.group_id.replace({"g2": "g3"})))
        result = self.prepare()
        self.assertEqual(result.required_training_group_ids, frozenset(["g3"]))
        self.assertEqual(result.class_support.set_index("label").loc["Common", "available_validation_groups"], 1)
        self.assertEqual(set(result.frame.loc[result.frame.group_id.eq("g3"), "record_id"]), {"c", "d", "e", "f"})

    def test_multi_group_class_blocked_when_every_group_is_required_for_training(self):
        # Each Rare* class has one group; Common shares both pinned groups.
        def add_rare(frame):
            frame.loc[frame.record_id.eq("b"), "Brand"] = "RareTrain"
            return frame
        self.edit("train.csv", add_rare)
        self.edit("validation.csv", lambda frame: frame.assign(group_id=frame.group_id.replace({"g2": "g3"})))
        self.edit("assignments.csv", lambda frame: frame.assign(group_id=frame.group_id.replace({"g2": "g3"})))
        result = self.prepare()
        support = result.class_support.set_index("label")
        self.assertEqual(support.loc["Common", "development_groups"], 2)
        self.assertTrue(support.loc["Common", "blocked_by_training_only_groups"])
        self.assertFalse(support.loc["Common", "eligible_for_both"])
        self.assertEqual(result.eligible_for_both_classes, ())

    def test_all_missing_target_keeps_population_and_has_empty_support(self):
        for name in ("train.csv", "validation.csv"):
            self.edit(name, lambda frame: frame.assign(Brand="null"))
        result = self.prepare()
        self.assertEqual(len(result.frame), 10)
        self.assertTrue(result.labeled_frame.empty)
        self.assertTrue(result.class_support.empty)
        self.assertTrue(result.class_group_support.empty)
        self.assertEqual(result.training_only_classes, ())
        self.assertEqual(result.required_training_group_ids, frozenset())

    def test_nonmissing_label_strings_and_placeholders_are_preserved(self):
        self.edit("train.csv", lambda frame: frame.assign(Brand=frame.Brand.replace({
            "Common": " UnItemised brand ", "": "Other Brands", "nAn": "Generic", "  ": "lowercase"})))
        result = self.prepare()
        self.assertTrue({" UnItemised brand ", "Other Brands", "Generic", "lowercase"} <= set(result.class_support.label))

    def test_unsupported_targets_fail_before_access(self):
        for target in ("Segment", "Mnfr", "TargetAgeGroup", "brand", "", None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                prepare_target_development(self.directory / "does-not-exist", target)
        self.verifier.assert_not_called()

    def test_authoritative_verification_failure_is_not_bypassed(self):
        self.verifier.side_effect = ValueError("Frozen membership differs")
        with self.assertRaisesRegex(ValueError, "Frozen membership"):
            self.prepare()

    def test_modified_export_without_reseal_is_rejected(self):
        with (self.directory / "train.csv").open("a", encoding="utf8") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(ValueError, "train.csv"):
            self.prepare()

    def test_resealed_protected_mapping_changes_are_rejected(self):
        self.edit("protected_final_test.csv", lambda frame: frame.assign(group_id="other"))
        with self.assertRaisesRegex(ValueError, "Protected test record/group mapping"):
            self.prepare()

    def test_protected_record_or_group_including_unlabeled_rows_is_rejected(self):
        for column, value in (("record_id", "z"), ("group_id", "gz")):
            with self.subTest(column=column):
                fixture(self.directory)
                def contaminate(frame):
                    frame.loc[frame.record_id.eq("f"), column] = value
                    return frame
                self.edit("validation.csv", contaminate)
                with self.assertRaisesRegex(ValueError, "Protected test"):
                    self.prepare()

    def test_group_identity_changes_missing_rows_and_duplicate_ids_are_rejected(self):
        transformations = (
            lambda frame: frame.assign(group_id="replacement"),
            lambda frame: frame.iloc[:-1],
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            lambda frame: frame.assign(split="test"),
            lambda frame: frame.assign(group_id=""),
        )
        for transform in transformations:
            with self.subTest(transform=transform):
                fixture(self.directory)
                self.edit("train.csv", transform)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_frozen_group_crossing_original_roles_is_rejected(self):
        self.edit("assignments.csv", lambda frame: frame.assign(group_id=frame.group_id.replace({"g2": "g1"})))
        with self.assertRaisesRegex(ValueError, "group crosses"):
            self.prepare()

    def test_input_change_during_loading_is_rejected(self):
        original_read = pd.read_csv
        def changing_read(path, **kwargs):
            frame = original_read(path, **kwargs)
            if Path(path).name == "validation.csv":
                with Path(path).open("a", encoding="utf8") as stream:
                    stream.write("changed\n")
            return frame
        with patch(MODULE + ".pd.read_csv", side_effect=changing_read), self.assertRaisesRegex(ValueError, "changed during loading"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
