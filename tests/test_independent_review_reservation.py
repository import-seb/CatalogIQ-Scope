"""Independent future reservation: family separation, blindness and repeatability."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.balanced_pair_sample import BUCKETS, EVIDENCE_FIELDS, PAIR_COLUMNS
from catalogiq.independent_review_reservation import (
    REGISTRY_COLUMNS, _select_component_independent, reserve_from_frames,
    reserve_independent_review, union_component_map, verify_reservation,
)


def records():
    rows = []
    for maker in range(12):
        brand = f"Maker{maker}"
        names = [f"{brand} Premium Alpha Turmeric Joint Herbal Supplement 30 Capsules",
                 f"{brand} Premium Alpha Turmeric Joint Herbal Supplement 60 Capsules",
                 f"{brand} Gold Beta Magnesium Sleep Herbal Supplement 30 Capsules",
                 f"{brand} Gold Gamma Cranberry Urinary Herbal Supplement 60 Capsules",
                 f"{brand} Ancient Omega Fish Oil Pure Lemon Flavor 100 Softgels",
                 f"{brand} Ancient Omega Fish Oil Pure Orange Flavor 200 Softgels"]
        for index, name in enumerate(names):
            rows.append({"record_id": f"m{maker:02d}r{index}", "ProductName": name,
                         "ProductBrand": brand, "ProductDescription": f"Full evidence {maker} {index}",
                         "ProductContents": "Complete product contents", "Segment": "unused target",
                         "rule_false_positive": True, "split": "train"})
    return pd.DataFrame(rows)


def fixture():
    frame = records()
    groupings = {method: pd.DataFrame({"record_id": frame.record_id,
                                      "group_id": [f"{method}_{key}" for key in frame.record_id]})
                 for method in ("rule", "tfidf")}
    # A chain across both methods and independent near-pairs must be excluded.
    groupings["rule"].loc[1, "group_id"] = groupings["rule"].loc[0, "group_id"]
    groupings["tfidf"].loc[2, "group_id"] = groupings["tfidf"].loc[1, "group_id"]
    near = pd.DataFrame([{"left_record_id": "m00r2", "right_record_id": "m00r3"}])
    prior = pd.DataFrame([{"left_record_id": "m00r0", "right_record_id": "m00r4",
                           "leakage_decision": "keep_together", "rule_grouped": False}])
    return frame, prior, groupings, near


class IndependentReviewReservationTests(unittest.TestCase):
    def test_union_map_combines_both_methods_and_independent_pairs(self):
        _, _, groupings, near = fixture()
        mapping = union_component_map(groupings, near)
        self.assertEqual(len({mapping[f"m00r{i}"] for i in range(4)}), 1)
        self.assertNotEqual(mapping["m00r3"], mapping["m00r4"])

    def test_reservation_excludes_every_prior_union_component(self):
        frame, prior, groupings, near = fixture()
        registry, _, report = reserve_from_frames(frame, prior, groupings, near, per_bucket=2)
        endpoints = set(registry.left_record_id) | set(registry.right_record_id)
        self.assertFalse({f"m00r{i}" for i in range(5)} & endpoints)
        self.assertEqual(report["excluded_records"], 5)
        self.assertEqual(report["excluded_union_components"], 2)
        self.assertTrue(all(report["proof"].values()))
        self.assertEqual(set(registry), set(REGISTRY_COLUMNS))

    def test_labels_prior_judgments_correctness_and_split_values_do_not_select_pairs(self):
        frame, prior, groupings, near = fixture()
        first = reserve_from_frames(frame, prior, groupings, near, per_bucket=2)
        changed = frame.copy()
        changed["Segment"] = [f"new label {i}" for i in range(len(frame))]
        changed["rule_false_positive"] = False
        changed["tfidf_false_negative"] = True
        changed["split"] = "test"
        changed_prior = prior.copy()
        changed_prior["leakage_decision"] = "keep_separate"
        changed_prior["rule_grouped"] = True
        changed_groups = deepcopy(groupings)
        for method in changed_groups:
            changed_groups[method]["Segment"] = "different"
            changed_groups[method]["split"] = "validation"
            changed_groups[method]["false_positive"] = True
        second = reserve_from_frames(changed, changed_prior, changed_groups, near, per_bucket=2)
        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])

    def test_retrieval_reproduces_under_row_and_group_name_permutations(self):
        frame, prior, groupings, near = fixture()
        first = reserve_from_frames(frame, prior, groupings, near, per_bucket=2)
        changed = {}
        for method, grouping in groupings.items():
            shuffled = grouping.iloc[::-1].copy()
            names = {value: f"renamed_{i}" for i, value in enumerate(sorted(shuffled.group_id.unique(), reverse=True))}
            shuffled["group_id"] = shuffled.group_id.map(names)
            changed[method] = shuffled
        second = reserve_from_frames(frame.iloc[::-1], prior.iloc[::-1], changed, near.iloc[::-1], per_bucket=2)
        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])

    def test_blind_evidence_has_full_fields_and_no_strata_labels_or_outcomes(self):
        frame, prior, groupings, near = fixture()
        _, blind, _ = reserve_from_frames(frame, prior, groupings, near, per_bucket=2)
        expected = {"pair_id", *PAIR_COLUMNS,
                    *(f"{side}_{field}" for side in ("left", "right") for field in EVIDENCE_FIELDS)}
        self.assertGreater(len(blind), 0)
        self.assertEqual(set(blind[0]), expected)
        self.assertTrue(blind[0]["left_ProductDescription"].startswith("Full evidence"))
        self.assertFalse(any("group" in key or "Segment" in key or "stratum" in key for key in blind[0]))

    def test_selector_reserves_both_components_and_never_reuses_them(self):
        pool = pd.DataFrame([
            {"left_record_id": "a", "right_record_id": "b", "provisional_stratum": BUCKETS[0]},
            {"left_record_id": "c", "right_record_id": "d", "provisional_stratum": BUCKETS[1]},
            {"left_record_id": "e", "right_record_id": "f", "provisional_stratum": BUCKETS[2]},
        ])
        components = {"a": "x", "b": "y", "c": "y", "d": "z", "e": "w", "f": "w"}
        selected = _select_component_independent(pool, components, seed=1, per_bucket=1)
        self.assertEqual(len(selected), 2)
        used = set()
        for row in selected.itertuples(index=False):
            current = {row.left_exclusion_component, row.right_exclusion_component}
            self.assertFalse(current & used)
            used.update(current)

    def test_incomplete_quota_is_reported_without_invented_labels(self):
        frame, prior, groupings, near = fixture()
        _, _, report = reserve_from_frames(frame, prior, groupings, near, per_bucket=100, max_oversample_factor=1)
        self.assertFalse(report["quota_filled"])
        self.assertNotIn("accuracy", report)
        self.assertEqual(len(report["retrieval_attempts"]), 1)

    def test_invalid_records_groups_prior_and_configuration_rejected(self):
        frame, prior, groupings, near = fixture()
        with self.assertRaisesRegex(ValueError, "unknown"):
            reserve_from_frames(frame, pd.DataFrame([{"left_record_id": "unknown", "right_record_id": "m00r4"}]), groupings, near)
        bad_groups = deepcopy(groupings)
        bad_groups["tfidf"] = bad_groups["tfidf"].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "identical"):
            reserve_from_frames(frame, prior, bad_groups, near)
        bad_near = pd.DataFrame([{"left_record_id": "unknown", "right_record_id": "m00r4"}])
        with self.assertRaisesRegex(ValueError, "unknown"):
            union_component_map(groupings, bad_near)
        for kwargs in ({"seed": -1}, {"per_bucket": 0}, {"max_oversample_factor": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reserve_from_frames(frame, prior, groupings, near, **kwargs)

    def test_artifact_reservation_is_reproducible_refuses_overwrite_and_detects_tampering(self):
        frame, prior, groupings, near = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for method, grouping in groupings.items():
                grouping.to_csv(run / f"{method}_assignments.csv", index=False)
            near.to_csv(run / "independent_near_duplicate_pairs.csv", index=False)
            prior_path = root / "prior.csv"
            prior.to_csv(prior_path, index=False)
            (run / "refinement_freeze.json").write_text("{}", encoding="utf-8")
            (run / "summary.json").write_text(json.dumps({"input": {"path": str(root / "input.csv"),
                                                                   "identifier_source": None}}), encoding="utf-8")
            with patch("catalogiq.independent_review_reservation.verify_refinement_seal"), \
                    patch("catalogiq.independent_review_reservation.load_input", return_value=(frame, {"fingerprints": {}})):
                first = root / "first"
                second = root / "second"
                reserve_independent_review(run, [prior_path], first, per_bucket=2)
                reserve_independent_review(run, [prior_path], second, per_bucket=2)
                for name in ("reserved_pairs.csv", "blind_pairs.json", "protocol.json", "summary.json"):
                    self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertTrue(verify_reservation(first)["artifacts_verified"])
                with self.assertRaises(FileExistsError):
                    reserve_independent_review(run, [prior_path], first, per_bucket=2)
                (first / "reserved_pairs.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    verify_reservation(first)


if __name__ == "__main__":
    unittest.main()
