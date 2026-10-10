"""Exposure-aware reservation and sealed protocol integrity behavior."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.segment_development_allocation import AllocationConfig
from catalogiq.split_finalization import (
    allocate_remaining, reserve_unexposed_groups, verify_final_splits,
)


class FinalSplitTests(unittest.TestCase):
    def test_test_reservation_excludes_complete_transitive_components(self):
        ids = list("abcdefghi")
        groups = ["related", "related", "clean", "clean", "review", "review", "train", "single", "outside"]
        # b is exposed despite a being eligible. e is eligible but f was trained.
        mask, stats = reserve_unexposed_groups(ids, groups, {"a", "c", "d", "e", "h"}, set("abcdeh"))
        self.assertEqual(np.asarray(ids)[mask].tolist(), ["c", "d", "h"])
        self.assertEqual(stats["removed_by_corrected_or_effective_input_group_closure"], 2)
        self.assertEqual(stats["final_test_groups"], 2)

    def test_previously_trained_records_cannot_pad_test(self):
        with self.assertRaisesRegex(ValueError, "subset"):
            reserve_unexposed_groups(["oldtest", "trained"], ["a", "b"], {"trained"}, {"oldtest"})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            reserve_unexposed_groups(["test", "trained"], ["bridge", "bridge"], {"test"}, {"test"})

    def test_missing_targets_preserved_and_development_repeatable(self):
        n = 180
        frame = pd.DataFrame({"record_id": [f"r{i:03}" for i in range(n)],
            "dataset": "training", "source_sha256": "source", "source_row": [str(i) for i in range(n)],
            "Segment": ["" if i % 3 == 0 else "A" if i % 3 == 1 else "B" for i in range(n)],
            "Retailer": ["shop" if i % 2 else "market" for i in range(n)]})
        groups = np.array([f"g{i // 2}" for i in range(n)])
        mask = np.arange(n) >= 150
        config = replace(AllocationConfig(), max_validation_fraction_delta=.03,
                         max_segment_fraction_delta=.04, max_retailer_fraction_delta=.08,
                         max_group_size_fraction_delta=.08)
        first, stats = allocate_remaining(frame, groups, mask, config)
        second, _ = allocate_remaining(frame, groups, mask, config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.Segment.tolist(), frame.Segment.tolist())
        self.assertEqual(first.loc[mask, "split"].unique().tolist(), ["test"])
        self.assertEqual(first.groupby("group_id").split.nunique().max(), 1)
        self.assertNotIn("UNLABELED_ALLOCATION_ONLY", first.Segment.tolist())

    def _sealed_fixture(self, directory):
        directory = Path(directory)
        cache = directory / "cache"
        cache.mkdir()
        signatures = pd.DataFrame({"record_id": ["a", "b", "c"], "group_id": ["x", "y", "z"]})
        signatures.to_csv(cache / "record_signatures.csv", index=False)
        assignments = pd.DataFrame({"record_id": ["a", "b", "c"], "group_id": ["g1", "g2", "g3"],
            "split": ["train", "validation", "test"], "Segment": ["A", "A", "PRIVATE_FINAL_LABEL"]})
        assignments.to_csv(directory / "assignments.csv", index=False)
        for role in ("train", "validation", "test"):
            assignments.loc[assignments.split.eq(role)].to_csv(directory / f"{role}.csv", index=False)
        assignments.loc[assignments.split.eq("test"), ["record_id", "group_id", "split"]].to_csv(directory / "protected_final_test.csv", index=False)
        freeze = {"source_sha256": {}, "input_sha256": {str(cache / "record_signatures.csv"): sha256(cache / "record_signatures.csv")}, "model_input_cache": str(cache)}
        (directory / "grouping_freeze.json").write_text(json.dumps(freeze), encoding="utf8")
        (directory / "summary.json").write_text(json.dumps({"checks": {"group_isolation": True}}), encoding="utf8")
        seal = {"artifact_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
        (directory / "completion.json").write_text(json.dumps(seal), encoding="utf8")
        return directory

    def test_development_verification_does_not_open_test_or_read_its_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self._sealed_fixture(folder)
            real_read, real_sha = pd.read_csv, sha256
            def guarded_read(path, *args, **kwargs):
                self.assertNotEqual(Path(path).name, "test.csv")
                if Path(path).name == "assignments.csv":
                    self.assertNotIn("Segment", kwargs["usecols"])
                return real_read(path, *args, **kwargs)
            def guarded_sha(path):
                self.assertNotEqual(Path(path).name, "test.csv")
                return real_sha(path)
            with patch("catalogiq.split_finalization.pd.read_csv", guarded_read), patch("catalogiq.split_finalization.sha256", guarded_sha):
                self.assertTrue(verify_final_splits(directory, verify_exports=False)["checks"]["group_isolation"])

    def test_seal_detects_changed_ids_and_feature_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self._sealed_fixture(folder)
            with (directory / "train.csv").open("a", encoding="utf8") as output:
                output.write("altered,group,train,A\n")
            with self.assertRaisesRegex(ValueError, "Sealed split output changed"):
                verify_final_splits(directory)
        with tempfile.TemporaryDirectory() as folder:
            directory = self._sealed_fixture(folder)
            (directory / "cache/record_signatures.csv").write_text("changed", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                verify_final_splits(directory)


if __name__ == "__main__":
    unittest.main()
