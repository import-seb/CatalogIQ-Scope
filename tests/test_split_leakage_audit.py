import unittest

import pandas as pd

from catalogiq.split_leakage_audit import (
    aligned_assignment, blind_review_sample, description_probe, isolation_summary,
)
from catalogiq.split_evaluation import leakage_metrics, near_duplicate_pairs
from catalogiq.splitting import SplitConfig


def fixture():
    frame = pd.DataFrame({"record_id": list("abcde"), "ProductName": ["Alpha", "Beta", "Alpha", "Test", "Short"],
        "ProductBrand": ["Same"] * 5, "ProductDescription": ["Detailed product copy " * 20] * 4 + ["short"],
        "ProductContents": ["contents"] * 5, "Segment": ["hidden"] * 5})
    assignment = pd.DataFrame({"record_id": list("abcde"), "group_id": list("xyzww"),
                              "split": ["train", "train", "validation", "test", "train"]})
    return frame, assignment


class SplitLeakageAuditTests(unittest.TestCase):
    def test_description_counts_all_pairs_and_roles_combinatorially(self):
        frame, assignment = fixture()
        metrics, clusters, candidates = description_probe(frame, assignment)
        self.assertEqual(metrics["eligible_rows"], 4)
        self.assertEqual(metrics["duplicate_pairs"], 6)
        self.assertEqual(metrics["crossing_pairs"], 5)
        self.assertEqual(metrics["crossing_split_pairs"], {"train__validation": 2, "train__test": 2, "validation__test": 1})
        self.assertEqual(metrics["crossing_rows"], 4)
        self.assertEqual(clusters.iloc[0].distinct_normalized_names, 3)
        self.assertTrue(clusters.iloc[0].possible_boilerplate)
        self.assertEqual({(r["left_record_id"], r["right_record_id"]) for r in candidates}, {("b", "c")})

    def test_description_normalization_and_short_body_exclusion(self):
        frame, assignment = fixture()
        frame.loc[1, "ProductDescription"] = frame.loc[1, "ProductDescription"].upper().replace(" ", " / ")
        metrics, _, _ = description_probe(frame, assignment)
        self.assertEqual(metrics["duplicate_pairs"], 6)
        self.assertEqual(metrics["eligible_rows"], 4)

    def test_body_witness_limit_does_not_change_exact_counts(self):
        n = 100
        frame = pd.DataFrame({"record_id": [str(i) for i in range(n)], "ProductName": ["Product " + str(i) for i in range(n)],
                             "ProductBrand": ["Brand"] * n, "ProductDescription": ["Words " * 100] * n})
        assignment = frame[["record_id"]].assign(group_id="g", split=["train"] * 50 + ["validation"] * 50)
        metrics, _, candidates = description_probe(frame, assignment, candidate_limit=3)
        self.assertEqual(metrics["duplicate_pairs"], 4950)
        self.assertEqual(metrics["crossing_pairs"], 2500)
        self.assertEqual(len(candidates), 3)

    def test_missing_payloads_are_excluded_and_null_distinct_from_empty(self):
        frame = pd.DataFrame({"record_id": list("abcd"), "ProductName": [None, "", "Distinct product title", "Distinct product title"],
                              "ProductBrand": [None, "", None, ""], "ProductDescription": [None] * 4,
                              "ProductContents": [None] * 4})
        assignment = frame[["record_id"]].assign(group_id=list("abcd"), split=["train", "validation", "train", "validation"])
        metrics = leakage_metrics(frame, assignment, [], SplitConfig())
        self.assertEqual(metrics["exact_payload"]["eligible_rows"], 2)
        self.assertEqual(metrics["exact_payload"]["duplicate_pairs"], 0)
        self.assertEqual(metrics["exact_brand_name"]["crossing_pairs"], 1)

    def test_short_and_missing_names_do_not_enter_exhaustive_near_scope(self):
        frame = pd.DataFrame({"ProductName": [None, "Short", "Short", "Distinct long product", "Distinct long product"]})
        pairs, metadata = near_duplicate_pairs(frame, SplitConfig())
        self.assertEqual(pairs, [(3, 4, 1.0)])
        self.assertEqual(metadata["eligible_rows"], 2)
        self.assertTrue(metadata["exhaustive_within_scope"])

    def test_alignment_rejects_duplicate_and_missing_ids(self):
        frame, assignment = fixture()
        with self.assertRaises(ValueError):
            aligned_assignment(frame, assignment.iloc[:-1])
        with self.assertRaises(ValueError):
            aligned_assignment(frame, pd.concat([assignment.iloc[:-1], assignment.iloc[:1]]))
        self.assertEqual(aligned_assignment(frame, assignment.iloc[::-1]).record_id.tolist(), list("abcde"))

    def test_intentional_random_crossings_are_reported_and_grouped_rejected(self):
        _, assignment = fixture()
        result = isolation_summary(assignment, False)
        self.assertEqual(result["crossing_groups"], 1)
        self.assertEqual(result["records_in_crossing_groups"], 2)
        with self.assertRaises(ValueError):
            isolation_summary(assignment, True)

    def test_blind_sample_excludes_test_and_prior_pairs_and_target_fields(self):
        frame, assignment = fixture()
        candidates = [{"left_record_id": "a", "right_record_id": "c", "route": "independent_name", "name_jaccard": .9},
                      {"left_record_id": "b", "right_record_id": "c", "route": "independent_name", "name_jaccard": .95},
                      {"left_record_id": "a", "right_record_id": "d", "route": "independent_name", "name_jaccard": .9}]
        blind, mapping, meta = blind_review_sample(frame, assignment, candidates, prior_pairs=[("c", "a")])
        self.assertEqual(len(blind), 1)
        self.assertEqual(mapping.iloc[0].left_record_id, "b")
        self.assertEqual(meta["selected_pairs"], 1)
        self.assertFalse(any(any(token in field.lower() for token in ("segment", "split", "group", "score", "stratum", "record_id")) for field in blind))
        repeated, repeated_mapping, _ = blind_review_sample(frame.iloc[::-1], assignment.iloc[::-1], candidates[::-1], prior_pairs=[("a", "c")])
        pd.testing.assert_frame_equal(blind, repeated)
        pd.testing.assert_frame_equal(mapping, repeated_mapping)


if __name__ == "__main__":
    unittest.main()
