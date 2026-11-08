"""The relationship registry stays fail-closed and matches the serving primitive.

These tests do not check taxonomy. They check that the file cannot quietly start
enforcing rules, that its pairs are shaped the way review_decision reads them,
and that the facts the integration doc quotes are still the facts in the file.
"""
import csv
import json
from pathlib import Path
import unittest

import catalogiq
from catalogiq.confidence_calibration import review_decision

RESOURCES = Path(catalogiq.__file__).parent / "resources" / "relationships"
REGISTRY = RESOURCES / "classification_relationship_registry.json"
CURATED = RESOURCES / "segment_subsegment_map.csv"
TARGETS = ("Segment", "Sub-Segment", "Brand", "Mnfr", "Platform", "TargetAgeGroup")


def decision(relations, classifications):
    return review_decision([0.9, 0.1], "a", ["a", "b"], 0.5,
                           {"ProductName": "Claritin RediTabs 24 Hour"}, enabled=True,
                           approved_relations=relations, classifications=classifications)


class RelationshipRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.relations = cls.registry["relations"]
        cls.enforceable = [r for r in cls.relations if r["allowed_pairs"]]

    def test_no_relation_claims_approval(self):
        """Approval needs a named approver and the two multi-parent decisions."""
        self.assertEqual(self.registry["approved_relations_count"], 0)
        for relation in self.relations:
            self.assertNotEqual(relation["status"], "approved", relation["relation_id"])

    def test_every_relation_rests_on_training_support(self):
        """v1 used validation co-occurrence, which the primitive's docstring excludes."""
        for relation in self.relations:
            provenance = relation["provenance"]
            self.assertTrue(provenance["derived_from_is_approvable"], relation["relation_id"])
            self.assertNotIn("validation", provenance["derived_from"])

    def test_loading_an_unapproved_relation_reviews_every_product(self):
        """The documented loading rule: filter to approved, or everything is flagged."""
        for relation in self.enforceable:
            child, parent = relation["allowed_pairs"][0]
            reasons = decision([relation], {relation["child_target"]: child,
                                            relation["parent_target"]: parent})["review_reasons"]
            self.assertIn("unapproved_relationship_rule", reasons, relation["relation_id"])

    def test_pairs_are_well_formed_and_child_first(self):
        for relation in self.enforceable:
            self.assertIn(relation["child_target"], TARGETS)
            self.assertIn(relation["parent_target"], TARGETS)
            pairs = [tuple(p) for p in relation["allowed_pairs"]]
            self.assertEqual(len(pairs), len(set(pairs)), "duplicate allowed pairs")
            for pair in relation["allowed_pairs"]:
                self.assertEqual(len(pair), 2)
                self.assertTrue(all(isinstance(v, str) and v.strip() for v in pair))
            self.assertEqual(set(relation["pair_support"]),
                             {f"{c} | {p}" for c, p in pairs}, "support keys do not match the pairs")
            approved = dict(relation, status="approved")
            child, parent = pairs[0]
            self.assertNotIn("invalid_approved_classification_relationship",
                             decision([approved], {relation["child_target"]: child,
                                                   relation["parent_target"]: parent})["review_reasons"])

    def test_multi_parent_children_are_valid_not_violations(self):
        """decisions.md: shared parents can be legitimate and must not be whitelisted away."""
        seen = 0
        for relation in self.enforceable:
            approved = dict(relation, status="approved")
            for child, parents in relation["multi_parent_children"].items():
                self.assertGreater(len(parents), 1)
                seen += 1
                for parent in parents:
                    reasons = decision([approved], {relation["child_target"]: child,
                                                    relation["parent_target"]: parent})["review_reasons"]
                    self.assertNotIn("invalid_approved_classification_relationship", reasons,
                                     f"{child} under {parent} must stay valid")
        self.assertEqual(seen, 2, "expected Probiotics and Internal Pain")

    def test_an_unobserved_pair_and_a_missing_sibling_are_distinguished(self):
        relation = dict(self.enforceable[0], status="approved")
        child = relation["allowed_pairs"][0][0]
        self.assertIn("invalid_approved_classification_relationship",
                      decision([relation], {relation["child_target"]: child,
                                            relation["parent_target"]: "not a real segment"})["review_reasons"])
        self.assertIn("missing_relationship_classification",
                      decision([relation], {relation["child_target"]: child})["review_reasons"])

    def test_brand_platform_stays_rejected(self):
        """One brand spans many platforms, so an unseen pair is thin coverage, not a violation."""
        r3 = [r for r in self.relations if r["relation_id"] == "R3_brand_platform_not_a_relation"][0]
        self.assertEqual(r3["status"], "rejected")
        self.assertEqual(r3["allowed_pairs"], [])
        self.assertGreater(max(r3["platforms_per_brand"].values()), 1)

    def test_recall_is_reported_and_modest(self):
        """The rule is an extra signal, not a detector. Nothing here may claim otherwise."""
        for relation in self.enforceable:
            measured = relation["measured"]
            self.assertTrue(measured["false_alarm_count_is_a_real_result"])
            self.assertLess(measured["recall_of_errors_pct"], 50.0)
            self.assertGreater(measured["rows_with_an_error"], measured["fires_rows"])

    def test_curated_map_matches_the_r1_pairs(self):
        """The map ships beside the registry so R1 is reproducible from the repo."""
        with CURATED.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        curated = {(r["Sub-Segment"].strip(), r["Segment"].strip()) for r in rows}
        r1 = [r for r in self.relations if r["relation_id"] == "R1_sub_segment_under_segment"][0]
        self.assertEqual(curated, {tuple(p) for p in r1["allowed_pairs"]})
        self.assertEqual(sum(int(r["n"]) for r in rows),
                         r1["provenance"]["training_rows_behind_the_map"])


if __name__ == "__main__":
    unittest.main()
