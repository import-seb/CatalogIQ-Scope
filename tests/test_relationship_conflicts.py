"""Warn-mode flagging stays a dry run and keeps conflicts apart from gaps."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.relationship_conflicts import (
    REGISTRY_PATH, enforceable_relations, flag_all, flag_relation, load_registry, summarize,
)

RELATION = {
    "relation_id": "T1_child_under_parent",
    "status": "candidate",
    "evidence": "synthetic test only",
    "child_target": "Sub-Segment",
    "parent_target": "Segment",
    "allowed_pairs": [["Probiotics", "Digestive Health"],
                      ["Probiotics", "Vitamins, Minerals & Supplements"],
                      ["Oral Antihistamine", "Allergy"]],
}


def predictions(pairs):
    child = {"record_id": [], "predicted_label": [], "confidence": []}
    parent = {"record_id": [], "predicted_label": [], "confidence": []}
    for index, (c, p) in enumerate(pairs):
        record = f"r{index}"
        if c is not None:
            child["record_id"].append(record)
            child["predicted_label"].append(c)
            child["confidence"].append(0.9)
        if p is not None:
            parent["record_id"].append(record)
            parent["predicted_label"].append(p)
            parent["confidence"].append(0.8)
    return {"Sub-Segment": pd.DataFrame(child), "Segment": pd.DataFrame(parent)}


class WarnModeTests(unittest.TestCase):
    def flags(self, pairs, relation=None):
        return flag_relation(relation or RELATION, predictions(pairs), "test-version")

    def test_an_allowed_pair_is_not_flagged(self):
        self.assertTrue(self.flags([("Oral Antihistamine", "Allergy")]).empty)

    def test_a_pair_outside_the_map_is_a_conflict(self):
        out = self.flags([("Oral Antihistamine", "Digestive Health")])
        self.assertEqual(list(out["observation"]), ["invalid_approved_classification_relationship"])
        self.assertEqual(out.loc[0, "child_prediction"], "Oral Antihistamine")
        self.assertEqual(out.loc[0, "parent_prediction"], "Digestive Health")

    def test_a_multi_parent_child_is_clean_under_either_parent(self):
        out = self.flags([("Probiotics", "Digestive Health"),
                          ("Probiotics", "Vitamins, Minerals & Supplements")])
        self.assertTrue(out.empty, "a shared parent must not be reported as a conflict")

    def test_a_missing_sibling_is_a_gap_and_not_a_conflict(self):
        for pair in (("Probiotics", None), (None, "Allergy")):
            out = self.flags([pair])
            self.assertEqual(list(out["observation"]), ["missing_relationship_classification"])

    def test_a_blank_prediction_counts_as_missing(self):
        out = self.flags([("   ", "Allergy")])
        self.assertEqual(list(out["observation"]), ["missing_relationship_classification"])

    def test_output_never_claims_to_be_a_decision(self):
        out = self.flags([("Oral Antihistamine", "Digestive Health"), ("Probiotics", None)])
        self.assertFalse(out["is_approved_decision"].any())
        self.assertTrue((out["relation_status"] == "candidate").all())

    def test_promotion_does_not_mutate_the_relation(self):
        relation = dict(RELATION)
        self.flags([("Oral Antihistamine", "Digestive Health")], relation)
        self.assertEqual(relation["status"], "candidate")

    def test_a_malformed_relation_is_rejected_loudly(self):
        broken = dict(RELATION, allowed_pairs=[["only-one-element"]])
        with self.assertRaises(ValueError):
            self.flags([("Probiotics", "Allergy")], broken)

    def test_missing_predictions_for_one_side_raise(self):
        with self.assertRaises(KeyError):
            flag_relation(RELATION, {"Sub-Segment": predictions([])["Sub-Segment"]}, "v")

    def test_rejected_relations_define_no_pairs_and_are_skipped(self):
        registry = load_registry()
        ids = {r["relation_id"] for r in enforceable_relations(registry)}
        self.assertNotIn("R3_brand_platform_not_a_relation", ids)
        self.assertEqual(len(ids), 2)

    def test_summary_separates_conflicts_from_gaps(self):
        registry = {"registry_version": "test-version", "approved_relations_count": 0,
                    "relations": [RELATION]}
        preds = predictions([("Oral Antihistamine", "Allergy"),
                             ("Oral Antihistamine", "Digestive Health"),
                             ("Probiotics", None)])
        flags = flag_all(preds, registry)
        summary = summarize(flags, preds, registry)
        row = summary["per_relation"]["T1_child_under_parent"]
        self.assertEqual(summary["mode"], "warn_only")
        self.assertEqual(summary["approved_relations_count"], 0)
        self.assertEqual(row["records_with_both_predictions"], 2)
        self.assertEqual(row["conflicts"], 1)
        self.assertEqual(row["not_checkable_one_side_missing"], 1)

    def test_the_shipped_registry_still_approves_nothing(self):
        registry = load_registry(REGISTRY_PATH)
        self.assertEqual(registry["approved_relations_count"], 0)
        for relation in registry["relations"]:
            self.assertNotEqual(relation["status"], "approved")


if __name__ == "__main__":
    unittest.main()
