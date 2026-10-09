import unittest

import pandas as pd

from catalogiq.segment_development_allocation import (
    AllocationConfig, allocate_development_groups, random_control,
    validate_development_population,
)


def population_fixture():
    rows = []
    for repetition in range(17):
        for label in ("A", "B"):
            for retailer in ("amazon", "walmart", "cvs"):
                for size in (1, 2, 3, 6, 12):
                    group = f"{repetition:02d}-{label}-{retailer}-{size:02d}"
                    for member in range(size):
                        rows.append({"record_id": f"{group}-{member:02d}",
                                     "group_id": group, "Segment": label,
                                     "Retailer": retailer})
    return pd.DataFrame(rows)


class DevelopmentAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = population_fixture()
        cls.assignment, cls.stats = allocate_development_groups(cls.frame)

    def test_full_population_and_group_isolation(self):
        self.assertEqual(set(self.assignment.record_id), set(self.frame.record_id))
        self.assertFalse(self.assignment.record_id.duplicated().any())
        self.assertEqual(set(self.assignment.split), {"train", "validation"})
        self.assertTrue(self.assignment.groupby("group_id").split.nunique().eq(1).all())
        self.assertTrue(self.stats["all_input_records_assigned"])
        self.assertTrue(self.stats["grouping_unchanged"])
        self.assertEqual(self.stats["groups_crossing_train_validation"], 0)

    def test_validation_represents_all_family_sizes_and_retailers(self):
        sizes = self.frame.groupby("group_id").size()
        validation = self.assignment.loc[self.assignment.split.eq("validation")]
        self.assertEqual(set(validation.group_id.map(sizes)), {1, 2, 3, 6, 12})
        for field in ("retailer", "group_size_bucket", "Segment"):
            for role in ("train", "validation"):
                limit = .005 if field == "Segment" else .02
                self.assertLessEqual(self.stats["margins"][field][role]["maximum_fraction_delta"], limit)
        self.assertLessEqual(self.stats["validation_fraction_delta"], .001)

    def test_allocation_is_row_order_and_seed_reproducible(self):
        reordered, stats = allocate_development_groups(
            self.frame.sample(frac=1, random_state=17))
        pd.testing.assert_frame_equal(self.assignment, reordered)
        self.assertEqual(self.stats, stats)

    def test_random_control_exact_class_counts_and_reproducibility(self):
        first = random_control(self.assignment)
        second = random_control(self.assignment.iloc[::-1])
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_series_equal(
            self.assignment.groupby(["Segment", "split"]).size(),
            first.groupby(["Segment", "split"]).size())
        self.assertEqual(set(first.record_id), set(self.frame.record_id))
        self.assertTrue(first.groupby("group_id").split.nunique().gt(1).any())
        validate_development_population(self.frame, first,
                                        require_group_isolation=False)
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_development_population(self.frame, first)

    def test_mixed_label_groups_are_not_broken(self):
        frame = self.frame.copy()
        frame.loc[frame.record_id.str.endswith("-01"), "Segment"] = "B"
        assignment, _ = allocate_development_groups(frame)
        original = frame.set_index("record_id").Segment
        self.assertEqual(assignment.Segment.tolist(),
                         assignment.record_id.map(original).tolist())
        self.assertTrue(assignment.groupby("group_id").split.nunique().eq(1).all())
        self.assertEqual(set(assignment.loc[assignment.split.eq("train"), "Segment"]), {"A", "B"})
        self.assertEqual(set(assignment.loc[assignment.split.eq("validation"), "Segment"]), {"A", "B"})

    def test_rejects_unallocatable_class_and_missing_labels(self):
        frame = self.frame.copy()
        group = frame.group_id.iloc[0]
        frame.loc[frame.group_id.eq(group), "Segment"] = "rare"
        with self.assertRaisesRegex(ValueError, "at least two"):
            allocate_development_groups(frame)
        for missing in ("null", "", "<MISSING>", None):
            frame = self.frame.copy()
            frame.at[0, "Segment"] = missing
            with self.assertRaises(ValueError):
                allocate_development_groups(frame)

    def test_feasible_rare_class_has_training_and_validation_support(self):
        rare = pd.DataFrame([{"record_id": f"rare-{i:02d}",
            "group_id": f"rare-group-{i:02d}", "Segment": "rare",
            "Retailer": "amazon"} for i in range(17)])
        frame = pd.concat([self.frame, rare], ignore_index=True)
        assignment, stats = allocate_development_groups(frame)
        support = assignment.loc[assignment.Segment.eq("rare")].split.value_counts()
        self.assertGreater(support.get("train", 0), 0)
        self.assertGreater(support.get("validation", 0), 0)
        self.assertFalse(stats["failed_checks"])

    def test_rejects_missing_duplicate_and_changed_identities(self):
        with self.assertRaises(ValueError):
            allocate_development_groups(self.frame.drop(columns="Retailer"))
        with self.assertRaises(ValueError):
            allocate_development_groups(pd.concat([self.frame, self.frame.iloc[[0]]]))
        for field in ("Segment", "group_id"):
            altered = self.assignment.copy()
            altered.at[0, field] = "changed"
            with self.assertRaisesRegex(ValueError, "unchanged"):
                validate_development_population(self.frame, altered)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_development_population(self.frame, self.assignment.iloc[1:])

    def test_rejects_final_test_roles_and_biased_validation(self):
        altered = self.assignment.copy()
        altered.at[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "training and validation only"):
            validate_development_population(self.frame, altered)
        altered = self.assignment.copy()
        sizes = self.frame.groupby("group_id").size()
        altered["split"] = altered.group_id.map(sizes).eq(1).map(
            {True: "validation", False: "train"})
        with self.assertRaisesRegex(ValueError, "safeguards"):
            validate_development_population(self.frame, altered)

    def test_config_roundtrip_and_invalid_parameters(self):
        config = AllocationConfig()
        self.assertEqual(config, AllocationConfig.from_dict(config.to_dict()))
        for values in ({"seed": True}, {"seed": -1}, {"validation_fraction": True},
                       {"validation_fraction": 1}, {"restarts": 0},
                       {"refinement_passes": True}, {"exchange_passes": -1},
                       {"margin_weights": (1., 1., 1.)},
                       {"margin_weights": (1., True, 1., 1.)},
                       {"max_retailer_fraction_delta": True},
                       {"max_group_size_fraction_delta": -1},
                       {"max_segment_fraction_delta": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AllocationConfig(**values)


if __name__ == "__main__":
    unittest.main()
