"""Public synthetic fixtures for label-blind, reproducible product splitting."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from catalogiq.cleaning import PROVENANCE, TARGET_COLUMNS
from catalogiq.splitting import (
    GROUP_FIELDS,
    SPLITS,
    SplitConfig,
    check_assignments,
    grouping_view,
    record_ids,
    rule_groups,
    stratified_group_split,
    tfidf_groups,
)


def records(rows):
    """Create independent source identities; no repository dataset is required."""
    frame = pd.DataFrame(rows)
    frame["dataset"] = "public_synthetic.csv"
    frame["source_sha256"] = "synthetic-source-identity"
    frame["source_row"] = range(len(frame))
    if "Segment" not in frame:
        frame["Segment"] = "General"
    return frame


def mapping(ids, values):
    return dict(zip(ids, values))


class SplittingTests(unittest.TestCase):
    def setUp(self):
        self.config = SplitConfig(tfidf_max_df=1.0)

    def test_grouping_view_preserves_records_when_optional_fields_are_absent(self):
        frame = pd.DataFrame({"ProductName": ["Alpha", None],
                              "ProductBrand": ["ACME", "ACME"],
                              "Segment": ["Excluded label", "Another label"]},
                             index=pd.Index([9, 2], name="source"))
        original = frame.copy(deep=True)
        view = grouping_view(frame)
        self.assertEqual(list(view), list(GROUP_FIELDS))
        pd.testing.assert_index_equal(view.index, frame.index)
        self.assertEqual(view.ProductName.tolist(), ["Alpha", ""])
        self.assertEqual(view.ProductBrand.tolist(), ["ACME", "ACME"])
        for field in set(GROUP_FIELDS) - {"ProductName", "ProductBrand"}:
            self.assertEqual(view[field].tolist(), ["", ""])
        pd.testing.assert_frame_equal(frame, original)

    def assert_isolated(self, ids, groups, splits):
        assignments = pd.DataFrame({
            "record_id": ids, "group_id": groups, "split": splits,
        })
        check_assignments(assignments, ids)
        self.assertTrue(assignments.groupby("group_id")["split"].nunique().eq(1).all())

    def test_rule_links_form_transitive_components_and_remain_isolated(self):
        frame = records([
            {"ProductName": "Unrelated alpha", "Upc": "036000291452"},
            {"ProductName": "Unrelated beta", "Upc": "0036000291452",
             "ProductUrl": "https://store.example/item?id=17"},
            {"ProductName": "Unrelated gamma", "ProductUrl": "https://store.example/item?id=17"},
            {"ProductName": "Independent delta"},
        ])
        ids = record_ids(frame)
        groups, edges, _ = rule_groups(frame, ids, self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertNotIn((0, 2), [(a, b) for a, b, _, _ in edges])
        splits, _ = stratified_group_split(groups, frame["Segment"], self.config)
        self.assert_isolated(ids, groups, splits)

    def test_rule_accepts_only_intact_checksum_valid_gtins(self):
        frame = records([
            {"ProductName": "First", "Upc": "036000291452"},
            {"ProductName": "Second", "Upc": "00036000291452"},
            {"ProductName": "Third", "Upc": "3.6000291452E+10"},
            {"ProductName": "Fourth", "Upc": "3.6000291452E+10"},
            {"ProductName": "Fifth", "Upc": "036000291453"},
            {"ProductName": "Sixth", "Upc": "036000291453"},
            {"ProductName": "Seventh", "Upc": "000000000000"},
            {"ProductName": "Eighth", "Upc": "000000000000"},
        ])
        groups, _, stats = rule_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(set(groups)), 7)
        self.assertEqual(stats["valid_gtin_rows"], 2)

    def test_sku_is_retailer_scoped_and_name_guarded(self):
        frame = records([
            {"Retailer": "Store A", "Sku": "0012", "ProductName": "Hydrating lotion cream"},
            {"Retailer": "store a", "Sku": "0012", "ProductName": "Hydrating lotion cream refill"},
            {"Retailer": "Store B", "Sku": "0012", "ProductName": "Hydrating lotion cream"},
            {"Retailer": "Store A", "Sku": "0012", "ProductName": "Digital camera tripod"},
        ])
        # Distinct comparison brands avoid an independent exact-name match.
        frame["ProductBrand"] = ["A", "B", "C", "D"]
        groups, edges, _ = rule_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(set(groups)), 3)
        self.assertIn("retailer_sku_name", {reason for _, _, reason, _ in edges})

    def test_url_keeps_product_variant_query_and_path_case(self):
        frame = records([
            {"ProductName": "First", "ProductUrl": "https://STORE.example/Item?variant=red#top"},
            {"ProductName": "Second", "ProductUrl": "https://store.example/Item?variant=red#reviews"},
            {"ProductName": "Third", "ProductUrl": "https://store.example/Item?variant=blue"},
            {"ProductName": "Fourth", "ProductUrl": "https://store.example/item?variant=red"},
        ])
        groups, _, _ = rule_groups(frame, record_ids(frame), self.config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(set(groups)), 3)

    def test_rule_normalized_text_and_brand_blocking(self):
        frame = records([
            {"ProductBrand": "Example", "ProductName": "Daily face cream & lotion 100 ml"},
            {"ProductBrand": "EXAMPLE", "ProductName": "  DAILY Face cream &amp; lotion 100 ml "},
            {"ProductBrand": "Example", "ProductName": "Daily face cream lotion 100 ml refill"},
            {"ProductBrand": "Other", "ProductName": "Daily face cream lotion 100 ml refill"},
        ])
        config = replace(self.config, rule_token_threshold=0.8, rule_edit_threshold=0.8)
        groups, edges, stats = rule_groups(frame, record_ids(frame), config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertGreater(stats["blocked_comparisons"], 0)
        self.assertIn("blocked_name", {reason for _, _, reason, _ in edges})

    def test_rule_edit_threshold_is_independent_of_pair_argument_order(self):
        # SequenceMatcher's ratio can differ when its arguments are reversed.
        # This threshold lies between those two scores for these names.
        frame = records([
            {"ProductBrand": "Example", "ProductName":
             "gentle daily fresh soft skin lotion cream face care beauty tide"},
            {"ProductBrand": "Example", "ProductName":
             "gentle daily fresh soft skin lotion cream face care beauty diet"},
        ])
        config = replace(self.config, rule_edit_threshold=0.96)
        ids = record_ids(frame)
        original, _, _ = rule_groups(frame, ids, config)
        shuffled = frame.iloc[::-1]
        shuffled_ids = record_ids(shuffled)
        repeated, _, _ = rule_groups(shuffled, shuffled_ids, config)
        self.assertEqual(mapping(ids, original), mapping(shuffled_ids, repeated))

    def test_both_groupers_are_blind_to_all_target_labels(self):
        frame = records([
            {"ProductBrand": "Example", "ProductName": "Daily gentle face cream"},
            {"ProductBrand": "Example", "ProductName": "DAILY gentle face cream"},
            {"ProductBrand": "Other", "ProductName": "Wooden camera tripod"},
            {"ProductBrand": "Other", "ProductName": "Digital lens filter"},
        ])
        ids = record_ids(frame)
        for field in [*TARGET_COLUMNS, "Category"]:
            frame[field] = ["Different A", "Different B", "Same", "Same"]
        changed = frame.copy(deep=True)
        for field in [*TARGET_COLUMNS, "Category"]:
            changed[field] = ["Same", "Same", "Different C", "Different D"]
        for grouper in (rule_groups, tfidf_groups):
            with self.subTest(approach=grouper.__name__):
                original, _, _ = grouper(frame, ids, self.config)
                altered, _, _ = grouper(changed, ids, self.config)
                np.testing.assert_array_equal(original, altered)
                self.assertEqual(original[0], original[1])
                self.assertNotEqual(original[2], original[3])

    def test_tfidf_transitive_similarity_chain(self):
        frame = records([
            {"ProductName": "alpha beta"},
            {"ProductName": "alpha beta gamma delta"},
            {"ProductName": "gamma delta"},
            {"ProductName": "epsilon zeta"},
        ])
        config = replace(self.config, tfidf_threshold=0.45, tfidf_weights=(1, 0, 0, 0))
        ids = record_ids(frame)
        groups, edges, _ = tfidf_groups(frame, ids, config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertNotIn((0, 2), [(a, b) for a, b, _, _ in edges])
        splits, _ = stratified_group_split(groups, frame["Segment"], config)
        self.assert_isolated(ids, groups, splits)

    def test_tfidf_uses_product_names_and_descriptions(self):
        frame = records([
            {"ProductName": "Daily gentle skin cream", "ProductDescription": "Original cream"},
            {"ProductName": "Daily gentle skin cream", "ProductDescription": "Refill jar"},
            {"ProductName": "Unique old listing", "ProductDescription": "Hydrating aloe glycerin moisturizer"},
            {"ProductName": "Unique new listing", "ProductDescription": "Hydrating aloe glycerin moisturizer"},
        ])
        ids = record_ids(frame)
        names, _, _ = tfidf_groups(frame, ids, replace(self.config, tfidf_weights=(1, 0, 0, 0)))
        descriptions, _, _ = tfidf_groups(frame, ids, replace(self.config, tfidf_weights=(0, 1, 0, 0)))
        self.assertEqual(names[0], names[1])
        self.assertNotEqual(names[2], names[3])
        self.assertEqual(descriptions[2], descriptions[3])
        self.assertNotEqual(descriptions[0], descriptions[1])

    def test_tfidf_threshold_graph_is_complete_and_chunk_invariant(self):
        frame = records([{"ProductName": "Daily gentle skin cream"} for _ in range(9)]
                        + [{"ProductName": "Independent wooden tripod"}])
        ids = record_ids(frame)
        baseline = None
        for chunk_size in (1, 3, 100):
            groups, edges, _ = tfidf_groups(
                frame, ids, replace(self.config, cosine_chunk_size=chunk_size))
            self.assertEqual(len(edges), 9 * 8 // 2)
            self.assertEqual(len(set(groups)), 2)
            pairs = {(ids[a], ids[b]) for a, b, _, _ in edges}
            result = (mapping(ids, groups), pairs)
            if baseline is None:
                baseline = result
            else:
                self.assertEqual(result, baseline)

    def test_repeat_and_row_permutation_reproduce_groups_and_splits(self):
        frame = records([
            {"ProductBrand": f"Brand {i // 3}", "ProductName": f"Care product series {i // 3}",
             "ProductDescription": f"Gentle daily cream series {i // 3}",
             "Segment": "Skin" if i % 2 else "Body"}
            for i in range(30)
        ])
        untouched = frame.copy(deep=True)
        for grouper in (rule_groups, tfidf_groups):
            with self.subTest(approach=grouper.__name__):
                baseline = None
                for input_frame in (frame, frame.copy(), frame.sample(frac=1, random_state=19)):
                    ids = record_ids(input_frame)
                    groups, _, _ = grouper(input_frame, ids, self.config)
                    splits, _ = stratified_group_split(groups, input_frame["Segment"], self.config)
                    self.assert_isolated(ids, groups, splits)
                    result = (mapping(ids, groups), mapping(ids, splits))
                    if baseline is None:
                        baseline = result
                    else:
                        self.assertEqual(result, baseline)
        pd.testing.assert_frame_equal(frame, untouched)

    def test_singleton_class_balance_hits_target_with_skewed_classes(self):
        labels = np.array(["Major"] * 100 + ["Minor"] * 20 + ["Rare"] * 7 + [None] * 10)
        groups = np.array([f"group_{i:04}" for i in range(len(labels))])
        splits, stats = stratified_group_split(groups, labels, self.config)
        normalized = np.array(["<MISSING>" if x is None else x for x in labels])
        self.assertIn("<MISSING>", stats["classes"])
        for split, ratio in zip(SPLITS, self.config.ratios):
            self.assertLessEqual(abs(np.sum(splits == split) / len(labels) - ratio), 0.025)
            for label in np.unique(normalized):
                count = np.sum((splits == split) & (normalized == label))
                self.assertGreater(count, 0)
                self.assertLessEqual(abs(count - np.sum(normalized == label) * ratio), 1.1)

    def test_mixed_label_groups_and_infeasible_rare_class_are_retained(self):
        groups = np.array([f"mixed_{i}" for i in range(20) for _ in range(2)]
                          + [f"major_{i}" for i in range(40)] + ["rare_only"] * 4)
        labels = np.array([x for _ in range(20) for x in ("Major", "Minor")]
                          + ["Major"] * 40 + ["Rare"] * 4)
        splits, _ = stratified_group_split(groups, labels, self.config)
        self.assert_isolated([f"row_{i}" for i in range(len(labels))], groups, splits)
        self.assertEqual(len(set(splits[labels == "Rare"])), 1)
        self.assertEqual(sum(not np.any((labels == "Rare") & (splits == split)) for split in SPLITS), 2)
        for split in SPLITS:
            self.assertTrue(np.any((labels == "Minor") & (splits == split)))

    def test_empty_matching_fields_stay_singletons(self):
        frame = records([{}, {}, {}])
        for grouper in (rule_groups, tfidf_groups):
            groups, edges, _ = grouper(frame, record_ids(frame), self.config)
            self.assertEqual(len(set(groups)), len(frame))
            self.assertEqual(edges, [])

    def test_config_roundtrip_and_validation(self):
        self.assertEqual(SplitConfig.from_dict(self.config.to_dict()), self.config)
        invalid = [
            {"seed": -1}, {"seed": 2**32}, {"seed": True},
            {"ratios": (0.7, 0.2, 0.2)}, {"ratios": (1, 0, 0)},
            {"tfidf_threshold": 0}, {"rule_token_threshold": 1.1},
            {"evaluation_threshold": float("nan")}, {"cosine_chunk_size": 0},
            {"max_block_size": -1}, {"split_refinement_passes": -1},
            {"tfidf_weights": (0, 0, 0, 0)}, {"tfidf_weights": (1, 1, 1, -1)},
            {"size_weight": float("inf")},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SplitConfig(**values)

    def test_source_identity_rejects_missing_blank_and_duplicate_provenance(self):
        frame = records([{}, {}])
        self.assertEqual(len(set(record_ids(frame))), 2)
        for field in PROVENANCE:
            with self.subTest(missing_column=field), self.assertRaises(ValueError):
                record_ids(frame.drop(columns=[field]))
            for invalid in ("", "  ", None, np.nan, pd.NA):
                bad = frame.astype(object).copy()
                bad.loc[0, field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    record_ids(bad)
        duplicated = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            record_ids(duplicated)

    def test_assignment_validator_rejects_incomplete_duplicate_or_crossing_groups(self):
        valid = pd.DataFrame({"record_id": ["a", "b", "c"],
                              "group_id": ["one", "one", "two"],
                              "split": ["train", "train", "test"]})
        check_assignments(valid, ["a", "b", "c"])
        invalid = [valid.iloc[:2], pd.concat([valid, valid.iloc[[0]]]),
                   valid.assign(record_id=["a", "b", "unexpected"]),
                   valid.assign(split=["train", "validation", "test"]),
                   valid.assign(split=["train", "train", "unknown"]),
                   valid.assign(group_id=["one", "one", None]),
                   valid.assign(group_id=["one", "one", "  "])]
        for index, assignments in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                check_assignments(assignments, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
