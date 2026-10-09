"""Artifact, shared-audit, regression-gate and freeze tests on a small corpus."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.balanced_pair_sample import EVIDENCE_FIELDS
from catalogiq.features import sha256
from catalogiq.rule_refinement_experiment import (
    check_edge_components, load_fixed_audit, run_rule_refinement, verify_rule_refinement_freeze,
)
from catalogiq.splitting import GROUP_FIELDS, SplitConfig, grouping_view
from catalogiq.splitting_experiment import load_input, run_experiment


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class RuleRefinementExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = []
        names = ["Hydrating oat facial lotion", "Copper travel drink bottle", "Silicone kitchen baking mat",
                 "Cotton woven beach towel", "Rose blossom scented candle", "Vitamin citrus cleansing gel",
                 "Aluminum folding picnic chair", "Ceramic glazed coffee mug", "Wireless optical computer mouse",
                 "Lavender mild hand soap", "Bamboo bristle hair brush", "Waterproof walking trail boots"]
        for i, name in enumerate(names):
            for variant in range(2):
                rows.append({"dataset": "synthetic_training.csv", "source_sha256": "synthetic-fingerprint",
                    "source_row": str(i * 2 + variant), "ProductName": name, "ProductBrand": f"Maker {i}",
                    "ProductDescription": f"{name} for daily use", "ProductContents": name,
                    "Retailer": "Store", "ProductUrl": f"https://store.example/products/{i}",
                    "Segment": "" if i == 10 else "null" if i == 11 else "Home" if i % 2 else "Care"})
        self.input = self.root / "training_cleaned.csv"
        pd.DataFrame(rows[::2] + rows[1::2]).to_csv(self.input, index=False)
        self.config = SplitConfig(grouping_version=2, tfidf_max_df=1.0, cosine_chunk_size=2,
                                  split_restarts=1, split_refinement_passes=1)
        self.baseline = self.root / "baseline"
        with redirect_stdout(StringIO()):
            run_experiment(self.input, self.baseline, self.config, verify_reproducibility=False,
                           progress=lambda _: None)
        self.frame, _ = load_input(self.input)
        self.baseline_assignment = pd.read_csv(self.baseline / "rule_assignments.csv", dtype=str,
                                               keep_default_na=False)
        self.baseline_edges = pd.read_csv(self.baseline / "rule_matching_edges.csv", keep_default_na=False)
        write_json(self.baseline / "refinement_freeze.json", {"sha256": {
            str(path.resolve()): sha256(path) for path in self.baseline.iterdir() if path.is_file()}})
        self.snapshot = self.root / "edge_snapshot.json"
        write_json(self.snapshot, {"input_sha256": {str((self.baseline / f"{method}_matching_edges.csv").resolve()):
            sha256(self.baseline / f"{method}_matching_edges.csv") for method in ("rule", "tfidf")}})
        self.reservation = self.root / "reservation"
        self.reservation.mkdir()
        # Neutral synthetic reservation tests the gate; the real future sample
        # is never opened or joined in these experiment tests.
        write_json(self.reservation / "blind_pairs.json", {"synthetic": True})
        (self.reservation / "reserved_pairs.csv").write_text("pair_id,left_record_id,right_record_id\nF001,a,b\n", encoding="utf-8")
        write_json(self.reservation / "summary.json", {"frozen_run": str(self.baseline), "source_sha256": {},
            "code_sha256": {}, "output_sha256": {path.name: sha256(path) for path in self.reservation.iterdir()},
            "proof": {"target_blind": True}, "reserved_pairs": 1})
        self.fixture, self.reference = self.make_reviews()
        self.seen_features = []

    def make_reviews(self):
        by_source = self.frame.set_index("source_row")
        specifications = [("related", "keep_together", "2", "4"),
                          ("safe_to_separate", "keep_separate", "6", "8"),
                          ("borderline", "uncertain", "10", "12")]
        records, pairs = [], []
        for i, (category, decision, left, right) in enumerate(specifications):
            a, b = by_source.loc[left], by_source.loc[right]
            for row in (a, b):
                records.append({"record_id": row.record_id, **{field: row.get(field, "") for field in EVIDENCE_FIELDS}})
            pairs.append({"pair_id": f"P{i}", "left_record_id": a.record_id, "right_record_id": b.record_id,
                "review_category": category, "product_identity": "different_product" if decision != "uncertain" else "uncertain",
                "identity_detail": "distinct_product" if decision != "uncertain" else "uncertain",
                "identity_confidence": "high" if category != "borderline" else "low", "identity_rationale": "Synthetic identity assessment",
                "leakage_decision": decision, "leakage_confidence": "high" if category != "borderline" else "low",
                "leakage_rationale": "Synthetic independent leakage judgment", "judgment_uses_Segment": False,
                "judgment_uses_method_outcomes": False})
        fixture = self.root / "fixture.json"
        write_json(fixture, {"schema_version": 1, "records": records, "pairs": pairs,
            "metadata": {"pairs_per_category": 1, "pairs": 3, "interpretation": "Synthetic development cases"}})
        methods = {}
        for method in ("rule", "tfidf"):
            groups = pd.read_csv(self.baseline / f"{method}_assignments.csv", dtype=str).set_index("record_id").group_id
            methods[method] = {row["pair_id"]: bool(groups[row["left_record_id"]] == groups[row["right_record_id"]]) for row in pairs}
        reference = self.root / "reference.json"
        write_json(reference, {"fixture_sha256": sha256(fixture), "methods": methods})
        return fixture, reference

    def grouper(self, features, ids, split_config, refinement_config):
        self.seen_features.append(features.copy())
        groups = self.baseline_assignment.set_index("record_id").group_id.loc[ids].to_numpy()
        positions = {record: i for i, record in enumerate(ids)}
        links = [(positions[row.left_record_id], positions[row.right_record_id], row.reason, float(row.score))
                 for row in self.baseline_edges.itertuples()]
        return groups, links, {"synthetic": True, "labels_used": []}

    def run_saved(self, name="refined", grouper=None):
        output = self.root / name
        with patch("catalogiq.split_rules_v3.refine_rule_groups_v3", side_effect=grouper or self.grouper):
            summary = run_rule_refinement(self.baseline, output, reservation_dir=self.reservation,
                edge_snapshot=self.snapshot, fixture_path=self.fixture, reference_path=self.reference,
                progress=lambda _: None)
        return output, summary

    def merged_grouper(self, pair_id):
        pair = next(row for row in json.loads(self.fixture.read_text())["pairs"] if row["pair_id"] == pair_id)
        def merged(features, ids, split_config, refinement_config):
            groups, links, stats = self.grouper(features, ids, split_config, refinement_config)
            positions = {record: i for i, record in enumerate(ids)}
            a, b = positions[pair["left_record_id"]], positions[pair["right_record_id"]]
            previous = groups[b]
            groups[groups == previous] = groups[a]
            links.append((a, b, "synthetic_confirmed_family", 1.0))
            return groups, links, stats
        return merged

    def test_saved_baselines_and_allocator_unchanged_shared_audit_and_full_repeat(self):
        before = {path: sha256(path) for directory in (self.baseline, self.reservation)
                  for path in directory.iterdir() if path.is_file()}
        output, summary = self.run_saved()
        self.assertEqual({path: sha256(path) for path in before}, before)
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual((output / "tfidf_assignments.csv").read_bytes(), (self.baseline / "tfidf_assignments.csv").read_bytes())
        self.assertEqual((output / "tfidf_matching_edges.csv").read_bytes(), (self.baseline / "tfidf_matching_edges.csv").read_bytes())
        self.assertEqual((output / "rule_v2_assignments.csv").read_bytes(), (self.baseline / "rule_assignments.csv").read_bytes())
        self.assertEqual(set(summary["approaches"]), {"rule_v2", "rule_v3", "tfidf_v2"})
        self.assertEqual(len(self.seen_features), 2)
        for seen in self.seen_features:
            self.assertEqual(set(seen), set(GROUP_FIELDS))
            self.assertNotIn("Segment", seen)
            self.assertNotIn("Brand", seen)
            self.assertNotIn("record_id", seen)
        expected = ["rule_assignments.csv", "rule_matching_edges.csv", "rule_group_sizes.csv",
                    "rule_group_size_distribution.csv", "comparison.csv", "segment_distributions.csv",
                    "exact_payload_duplicate_pairs.csv", "regression_pair_outcomes.csv", "regression_metrics.csv"]
        self.assertTrue(all((output / name).exists() for name in expected))
        self.assertTrue(verify_rule_refinement_freeze(output)["sha256_verified"])
        near = pd.read_csv(output / "independent_near_duplicate_pairs.csv", keep_default_na=False)
        exact = pd.read_csv(output / "exact_payload_duplicate_pairs.csv", keep_default_na=False)
        for method, details in summary["approaches"].items():
            self.assertEqual(int(near[f"{method}_crosses_split"].sum()), details["metrics"]["near_duplicate"]["crossing_pairs"])
            self.assertEqual(int(exact[f"{method}_crosses_split"].sum()), details["metrics"]["exact_payload"]["crossing_pairs"])
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertEqual(tuple(summary["config"]["ratios"]), (0.70, 0.15, 0.15))

    def test_known_miss_fix_is_allowed_and_recorded_separately(self):
        output, summary = self.run_saved(grouper=self.merged_grouper("P0"))
        self.assertTrue((output / "rule_refinement_freeze.json").exists())
        metrics = pd.read_csv(output / "regression_metrics.csv").set_index(["method", "review_category"])
        self.assertEqual(metrics.at[("rule", "decisive_core"), "fixed_clear_errors"], 1)
        self.assertEqual(metrics.at[("rule", "decisive_core"), "false_negative_pairs"], 0)
        self.assertEqual(summary["regression"]["new_clear_errors"], 0)

    def test_new_clear_error_is_saved_but_prevents_fresh_review_freeze(self):
        output, summary = self.run_saved(grouper=self.merged_grouper("P1"))
        self.assertFalse(summary["checks"]["no_new_clear_regression_errors"])
        self.assertEqual(summary["regression"]["new_clear_errors"], 1)
        self.assertTrue((output / "regression_pair_outcomes.csv").exists())
        self.assertFalse((output / "rule_refinement_freeze.json").exists())
        with self.assertRaises(FileNotFoundError):
            verify_rule_refinement_freeze(output)

    def test_second_full_run_changes_edge_evidence_and_is_rejected(self):
        count = 0
        def changing(features, ids, split_config, refinement_config):
            nonlocal count
            count += 1
            groups, links, stats = self.grouper(features, ids, split_config, refinement_config)
            if count == 2:
                a, b, reason, _ = links[0]
                links[0] = (a, b, reason, 0.99)
            return groups, links, stats
        with self.assertRaisesRegex(RuntimeError, "not reproducible"):
            self.run_saved(grouper=changing)
        self.assertFalse((self.root / "refined/rule_refinement_freeze.json").exists())

    def test_same_run_reproduces_saved_csv_bytes_without_rerunning_tfidf_or_audit_search(self):
        with patch("catalogiq.splitting_experiment.tfidf_groups", side_effect=AssertionError("TF-IDF rerun")), \
             patch("catalogiq.split_evaluation.near_duplicate_pairs", side_effect=AssertionError("new audit pool")):
            first, _ = self.run_saved("first")
            second, _ = self.run_saved("second")
        for name in ("rule_assignments.csv", "rule_matching_edges.csv", "independent_near_duplicate_pairs.csv",
                     "exact_payload_duplicate_pairs.csv", "segment_distributions.csv", "regression_pair_outcomes.csv"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_freeze_detects_output_and_reserved_evidence_changes(self):
        output, _ = self.run_saved()
        path = output / "rule_assignments.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_rule_refinement_freeze(output)
        path.write_bytes(original)
        (self.reservation / "blind_pairs.json").write_text('{"tampered":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_rule_refinement_freeze(output)

    def test_baseline_edge_snapshot_prevents_unsealed_edge_changes(self):
        path = self.baseline / "tfidf_matching_edges.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_saved()
        self.assertFalse((self.root / "refined").exists())

    def test_existing_output_and_baseline_child_output_refused(self):
        self.run_saved()
        with self.assertRaises(FileExistsError):
            self.run_saved()
        with self.assertRaisesRegex(ValueError, "outside protected"):
            self.run_saved("baseline/child")

    def test_fixed_audit_rejects_missing_endpoints_and_duplicate_pairs(self):
        bad = self.root / "bad_audit.csv"
        pd.DataFrame([{"left_record_id": "absent", "right_record_id": self.frame.record_id.iloc[0],
                       "name_jaccard": "0.9"}]).to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, "outside"):
            load_fixed_audit(bad, self.frame)
        row = {"left_record_id": self.frame.record_id.iloc[0], "right_record_id": self.frame.record_id.iloc[1],
               "name_jaccard": "0.9"}
        pd.DataFrame([row, row]).to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_fixed_audit(bad, self.frame)

    def test_edge_component_validation_rejects_unexplained_groups_and_cross_edges(self):
        assignment = self.baseline_assignment.copy()
        empty = self.baseline_edges.iloc[:0]
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            check_edge_components(assignment, empty)
        edge = self.baseline_edges.iloc[0]
        assignment.loc[assignment.record_id.eq(edge.right_record_id), "group_id"] = "separated"
        with self.assertRaisesRegex(ValueError, "crosses"):
            check_edge_components(assignment, self.baseline_edges)

    def test_grouping_input_unchanged_by_scrambling_all_class_targets(self):
        original = grouping_view(self.frame)
        changed = self.frame.copy()
        for column in ("Segment", "Brand", "Mnfr", "Platform", "Sub-Segment", "TargetAgeGroup"):
            changed[column] = [f"arbitrary-{i % 3}" for i in range(len(changed))]
        pd.testing.assert_frame_equal(original, grouping_view(changed))


if __name__ == "__main__":
    unittest.main()
