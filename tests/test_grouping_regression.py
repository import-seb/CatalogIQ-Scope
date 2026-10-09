import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from catalogiq.balanced_pair_sample import EVIDENCE_FIELDS
from catalogiq.grouping_regression import (
    assert_no_new_clear_errors, compare_review_regression, evaluate_review_groups,
    load_review_fixture, main,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/grouping_review_75.json"
REFERENCE = ROOT / "tests/fixtures/grouping_review_75_baseline.json"


def reference_groups():
    """Real reference pair outcomes represented as disjoint synthetic groups.

    This tests review-regression semantics. It does not rerun a corpus-dependent
    grouper on a subset or pretend the reference predictions are desired truth.
    """
    _, pairs, _ = load_review_fixture(FIXTURE)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["methods"]
    mappings = {}
    for method, predictions in reference.items():
        mapping = {}
        for row in pairs.itertuples():
            mapping[row.left_record_id] = row.pair_id + "_left"
            mapping[row.right_record_id] = mapping[row.left_record_id] if predictions[row.pair_id] else row.pair_id + "_right"
        mappings[method] = mapping
    return mappings


class ReviewedGroupingRegressionTests(unittest.TestCase):
    def test_all75_cases_and_separate150_label_free_records_retained(self):
        records, pairs, metadata = load_review_fixture(FIXTURE)
        self.assertEqual(len(pairs), 75)
        self.assertEqual(len(records), 150)
        self.assertEqual(pairs.review_category.value_counts().to_dict(),
                         {"related": 25, "safe_to_separate": 25, "borderline": 25})
        self.assertEqual(set(records), {"record_id", *EVIDENCE_FIELDS})
        self.assertEqual(metadata["review_policy_target"], "Segment")
        for row in pairs.itertuples():
            with self.subTest(pair_id=row.pair_id):
                self.assertFalse(row.judgment_uses_Segment)
                self.assertFalse(row.judgment_uses_method_outcomes)
                self.assertTrue(row.identity_rationale)
                self.assertTrue(row.leakage_rationale)
        self.assertEqual(pairs.leakage_decision.eq("uncertain").sum(), 24)

    def test_current_errors_remain_visible_and_are_not_asserted_correct(self):
        outcomes, metrics = compare_review_regression(FIXTURE, REFERENCE, reference_groups())
        assert_no_new_clear_errors(outcomes)
        decisive = metrics.loc[metrics.review_category.eq("decisive_core")].set_index("method")
        self.assertEqual(decisive.at["rule", "false_negative_pairs"], 19)
        self.assertEqual(decisive.at["rule", "false_positive_pairs"], 1)
        self.assertEqual(decisive.at["tfidf", "false_negative_pairs"], 20)
        self.assertEqual(decisive.at["tfidf", "false_positive_pairs"], 0)
        self.assertEqual(outcomes.new_clear_error.sum(), 0)

    def test_known_miss_fix_is_allowed(self):
        groups = reference_groups()
        _, pairs, _ = load_review_fixture(FIXTURE)
        row = pairs.set_index("pair_id").loc["B075"]  # Phazyme pack variants.
        groups["rule"][row.right_record_id] = groups["rule"][row.left_record_id]
        outcomes, _ = compare_review_regression(FIXTURE, REFERENCE, groups)
        assert_no_new_clear_errors(outcomes)
        fixed = outcomes.loc[outcomes.fixed_clear_error]
        self.assertEqual(fixed[["method", "pair_id"]].values.tolist(), [["rule", "B075"]])

    def test_newly_broken_related_case_is_a_regression(self):
        groups = reference_groups()
        _, pairs, _ = load_review_fixture(FIXTURE)
        row = pairs.set_index("pair_id").loc["B147"]  # Matching heating-pad template.
        groups["rule"][row.right_record_id] = "newly_separated"
        outcomes, _ = compare_review_regression(FIXTURE, REFERENCE, groups)
        with self.assertRaisesRegex(AssertionError, "rule:B147"):
            assert_no_new_clear_errors(outcomes)

    def test_new_overmerge_is_a_regression_even_if_other_errors_improve(self):
        groups = reference_groups()
        _, pairs, _ = load_review_fixture(FIXTURE)
        reference = json.loads(REFERENCE.read_text())["methods"]["rule"]
        separate = pairs.loc[pairs.review_category.eq("safe_to_separate") & pairs.pair_id.map(reference).eq(False)].iloc[0]
        groups["rule"][separate.right_record_id] = groups["rule"][separate.left_record_id]
        missed = pairs.set_index("pair_id").loc["B075"]
        groups["rule"][missed.right_record_id] = groups["rule"][missed.left_record_id]
        outcomes, _ = compare_review_regression(FIXTURE, REFERENCE, groups)
        self.assertEqual(outcomes.fixed_clear_error.sum(), 1)
        self.assertEqual(outcomes.new_clear_error.sum(), 1)
        with self.assertRaises(AssertionError):
            assert_no_new_clear_errors(outcomes)

    def test_unresolved_boundary_changes_require_review_and_do_not_force_labels(self):
        groups = reference_groups()
        _, pairs, _ = load_review_fixture(FIXTURE)
        row = pairs.loc[pairs.leakage_decision.eq("uncertain")].iloc[0]
        groups["tfidf"][row.right_record_id] = groups["tfidf"][row.left_record_id]
        outcomes, _ = compare_review_regression(FIXTURE, REFERENCE, groups)
        assert_no_new_clear_errors(outcomes)
        scored = outcomes.loc[outcomes.method.eq("tfidf") & outcomes.pair_id.eq(row.pair_id)].iloc[0]
        self.assertTrue(scored.borderline_connection_changed)
        self.assertTrue(pd.isna(scored.reviewed_error))

    def test_split_membership_and_class_labels_do_not_affect_group_regression(self):
        groups = reference_groups()["rule"]
        assignments = pd.DataFrame({"record_id": list(groups), "group_id": list(groups.values()),
                                    "Segment": "class1", "Family": "other_target", "split": "train"})
        _, pairs, _ = load_review_fixture(FIXTURE)
        original = evaluate_review_groups(pairs, assignments)
        assignments["Segment"] = "different"
        assignments["Family"] = "different"
        assignments["split"] = "test"
        pd.testing.assert_frame_equal(original, evaluate_review_groups(pairs, assignments))

    def test_invalid_group_coverage_and_changed_fixture_are_rejected(self):
        _, pairs, _ = load_review_fixture(FIXTURE)
        groups = reference_groups()["rule"]
        groups.pop(pairs.left_record_id.iloc[0])
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluate_review_groups(pairs, groups)
        groups = reference_groups()["rule"]
        groups[next(iter(groups))] = None
        with self.assertRaisesRegex(ValueError, "invalid"):
            evaluate_review_groups(pairs, groups)
        with TemporaryDirectory() as folder:
            changed = Path(folder) / "fixture.json"
            changed.write_text(FIXTURE.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                compare_review_regression(changed, REFERENCE, reference_groups())

    def test_fixture_rejects_target_columns_in_grouping_features(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["records"][0]["Segment"] = "forbidden"
        with TemporaryDirectory() as folder:
            changed = Path(folder) / "fixture.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "permitted product features"):
                load_review_fixture(changed)

    def test_cli_reproducible_and_input_files_preserved(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            paths = {}
            for method, groups in reference_groups().items():
                path = directory / f"{method}.csv"
                pd.DataFrame({"record_id": list(groups), "group_id": list(groups.values())}).to_csv(path, index=False)
                paths[method] = path
            commands = ["--fixture", str(FIXTURE), "--reference", str(REFERENCE),
                        "--rule-assignments", str(paths["rule"]), "--tfidf-assignments", str(paths["tfidf"])]
            before = {method: path.read_bytes() for method, path in paths.items()}
            self.assertEqual(main([*commands, "--output-dir", str(directory / "first")]), 0)
            self.assertEqual(main([*commands, "--output-dir", str(directory / "second")]), 0)
            self.assertEqual((directory / "first/pair_regressions.csv").read_bytes(),
                             (directory / "second/pair_regressions.csv").read_bytes())
            self.assertEqual(before, {method: path.read_bytes() for method, path in paths.items()})


if __name__ == "__main__":
    unittest.main()
