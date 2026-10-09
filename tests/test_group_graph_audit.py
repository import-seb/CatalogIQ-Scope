"""Saved graph validation and diagnostic transitivity counterfactuals."""
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq.group_graph_audit import audit_group_graph, main, sample_saved_path_edges


def write_run(root, edges=None, group_ids=None):
    directory = root / "run"
    directory.mkdir()
    ids = list("abcd")
    group_ids = group_ids or ["family", "family", "family", "singleton"]
    edges = edges if edges is not None else [("a", "b", "reason_a", "0.9"), ("b", "c", "reason_b", "0.8")]
    for method in ("rule", "tfidf"):
        pd.DataFrame({"record_id": ids, "group_id": group_ids, "split": ["train"] * 4}).to_csv(
            directory / f"{method}_assignments.csv", index=False)
        pd.DataFrame(edges, columns=["left_record_id", "right_record_id", "reason", "score"]).to_csv(
            directory / f"{method}_matching_edges.csv", index=False)
        pd.DataFrame({"group_id": group_ids}).groupby("group_id").size().rename("rows").reset_index().to_csv(
            directory / f"{method}_group_sizes.csv", index=False)
    return directory


def labels(*pairs):
    return pd.DataFrame([{"left_record_id": left, "right_record_id": right,
                          "leakage_decision": decision, "product_identity": "different_product",
                          "judgment_uses_Segment": False}
                         for left, right, decision in pairs])


def write_cycle(root, size=20):
    directory = root / "run"
    directory.mkdir()
    ids = [f"n{index:02d}" for index in range(size)]
    for method in ("rule", "tfidf"):
        pd.DataFrame({"record_id": ids, "group_id": ["family"] * size, "split": ["train"] * size}).to_csv(
            directory / f"{method}_assignments.csv", index=False)
        pd.DataFrame({"left_record_id": ids, "right_record_id": ids[1:] + ids[:1],
                      "reason": ["saved_edge"] * size, "score": ["0.9"] * size}).to_csv(
            directory / f"{method}_matching_edges.csv", index=False)
    return directory


class GroupGraphAuditTests(unittest.TestCase):
    def test_named_evidence_remains_aligned_when_review_endpoints_are_reversed(self):
        with tempfile.TemporaryDirectory() as folder:
            annotated = labels(("c", "a", "keep_separate"), ("b", "a", "keep_together"))
            annotated["pair_id"] = ["endpoints", "link"]
            annotated["left_ProductName"] = ["Gamma product", "Beta product"]
            annotated["right_ProductName"] = ["Alpha product", "Alpha product"]
            annotated["leakage_rationale"] = ["Distinct cores", "Specific shared information"]
            result = audit_group_graph(write_run(Path(folder)), annotated, verify_seal=False)
            named = result["named_reviewed_pairs"].query("method == 'rule' and pair_id == 'endpoints'").iloc[0]
            self.assertEqual(named.left_record_id, "a")
            self.assertEqual(named.left_ProductName, "Alpha product")
            self.assertEqual(named.right_ProductName, "Gamma product")
            steps = result["named_connection_paths"].query("method == 'rule' and endpoint_pair_id == 'endpoints'")
            self.assertEqual(steps.left_ProductName.tolist(), ["Alpha product", "Beta product"])
            self.assertEqual(steps.reviewed_edge_leakage_decision.tolist(), ["keep_together", "unreviewed"])

    def test_chain_reconstructs_singleton_and_quantifies_transitive_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_group_graph(write_run(Path(folder)), screening_thresholds=(3, 10),
                                       largest_groups=2, verify_seal=False)
            rule = result["group_metrics"].query("method == 'rule'").set_index("group_id")
            self.assertEqual(rule.loc["family", "direct_pairs"], 2)
            self.assertEqual(rule.loc["family", "implied_co_group_pairs"], 3)
            self.assertEqual(rule.loc["family", "transitive_only_pairs"], 1)
            self.assertAlmostEqual(rule.loc["family", "density"], 2 / 3)
            self.assertEqual(rule.loc["family", "diameter"], 2)
            self.assertEqual(rule.loc["singleton", "diameter"], 0)
            self.assertEqual(rule.loc["family", "group_review_status"], "unknown")
            bridges = result["bridge_edges"].query("method == 'rule'")
            self.assertEqual(bridges.pair_amplification.tolist(), [2, 2])
            self.assertEqual(bridges.smaller_component_size.tolist(), [1, 1])
            articulation = result["articulation_points"].query("method == 'rule'").iloc[0]
            self.assertEqual(articulation.record_id, "b")
            self.assertEqual(json.loads(articulation.component_sizes_after_node_removal_json), [1, 1])
            self.assertEqual(articulation.lost_pairs_between_remaining_nodes, 1)
            summary = result["summary"]["methods"]["rule"]
            self.assertEqual(summary["screening_tail"][0]["groups"], 1)
            self.assertEqual(summary["screening_tail"][1]["groups"], 0)
            candidate = result["candidates"].query("candidate_type == 'maximum_graph_distance'").iloc[0]
            self.assertEqual((candidate.left_record_id, candidate.right_record_id), ("a", "c"))
            self.assertEqual(set(candidate["methods"].split(";")), {"rule", "tfidf"})
            self.assertNotIn("product_identity", result["candidates"])

    def test_multiple_reviewed_bad_nonbridge_links_removed_simultaneously(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = write_run(root, [("a", "b", "one", "1"), ("a", "c", "two", "1"), ("b", "c", "three", "1")])
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            result = audit_group_graph(directory, labels(("a", "b", "keep_separate"), ("a", "c", "keep_separate")),
                                       root / "audit", verify_seal=False)
            self.assertTrue(result["bridge_edges"].empty)
            counterfactual = result["removal_counterfactuals"].query("method == 'rule'").iloc[0]
            self.assertEqual(counterfactual.removed_direct_edges, 2)
            self.assertEqual(json.loads(counterfactual.component_sizes_json), [2, 1])
            self.assertEqual(counterfactual.implied_pair_reduction, 2)
            self.assertEqual(counterfactual.reviewed_keep_separate_pairs_disconnected, 2)
            self.assertFalse(result["reviewed_pair_paths"].connected_after_reviewed_bad_edge_removal.any())
            self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})
            self.assertEqual(pd.read_csv(root / "audit/removal_counterfactuals.csv").shape[0], 2)
            with self.assertRaises(FileExistsError):
                audit_group_graph(directory, output_dir=root / "audit", verify_seal=False)

    def test_transitive_negative_does_not_delete_unreviewed_path_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_group_graph(write_run(Path(folder)), labels(("a", "c", "keep_separate")), verify_seal=False)
            row = result["reviewed_pair_paths"].query("method == 'rule'").iloc[0]
            self.assertEqual(row.connection_type, "transitive_only")
            self.assertEqual(row.shortest_path_hops, 2)
            self.assertEqual(json.loads(row.path_record_ids_json), ["a", "b", "c"])
            path_edges = json.loads(row.path_edge_evidence_json)
            self.assertEqual(path_edges[0]["evidence"][0]["reason"], "reason_a")
            self.assertTrue(row.connected_after_reviewed_bad_edge_removal)
            counterfactual = result["removal_counterfactuals"].query("method == 'rule'").iloc[0]
            self.assertEqual(counterfactual.removed_direct_edges, 0)
            self.assertEqual(counterfactual.implied_pair_reduction, 0)
            self.assertEqual(counterfactual.reviewed_keep_separate_pairs_still_connected, 1)

    def test_one_bad_direct_link_with_alternative_path_cannot_split_group(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = write_run(Path(folder), [("a", "b", "one", "1"), ("a", "c", "two", "1"), ("b", "c", "three", "1")])
            result = audit_group_graph(directory, labels(("a", "b", "keep_separate")), verify_seal=False)
            pair = result["reviewed_pair_paths"].query("method == 'rule'").iloc[0]
            self.assertTrue(pair.direct_edge)
            self.assertTrue(pair.connected_after_reviewed_bad_edge_removal)
            removal = result["removal_counterfactuals"].query("method == 'rule'").iloc[0]
            self.assertEqual(removal.removed_direct_edges, 1)
            self.assertEqual(removal.implied_pair_reduction, 0)
            self.assertEqual(json.loads(removal.component_sizes_json), [3])
            self.assertEqual(removal.reviewed_keep_separate_pairs_still_connected, 1)

    def test_large_nonbridge_group_transitive_error_and_simultaneous_cut(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = write_cycle(Path(folder))
            reviewed = labels(("n00", "n10", "keep_separate"), ("n00", "n01", "keep_separate"),
                              ("n10", "n11", "keep_separate"))
            result = audit_group_graph(directory, reviewed, verify_seal=False)
            self.assertTrue(result["bridge_edges"].empty)
            self.assertEqual(result["group_metrics"].diameter.tolist(), [10, 10])
            endpoint = result["reviewed_pair_paths"].query("method == 'rule' and right_record_id == 'n10'").iloc[0]
            self.assertEqual(endpoint.connection_type, "transitive_only")
            self.assertEqual(endpoint.shortest_path_hops, 10)
            self.assertFalse(endpoint.connected_after_reviewed_bad_edge_removal)
            removal = result["removal_counterfactuals"].query("method == 'rule'").iloc[0]
            self.assertEqual(removal.removed_direct_edges, 2)
            self.assertEqual(json.loads(removal.component_sizes_json), [10, 10])
            self.assertEqual(removal.implied_pair_reduction, 100)
            self.assertEqual(removal.reviewed_keep_separate_pairs_disconnected, 3)

    def test_path_edge_sampler_uses_identifiers_and_seed_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = write_cycle(root)
            registry = root / "registry.csv"
            row = {"pair_id": "G1", "method": "rule", "group_id": "family", "left_record_id": "n00",
                   "right_record_id": "n10", "left_ProductName": "unread text", "leakage_decision": "keep_separate"}
            pd.DataFrame([row]).to_csv(registry, index=False)
            first = sample_saved_path_edges(directory, registry, ["G1"], extra_edges_per_group=2, seed=42, verify_seal=False)
            self.assertEqual(first.candidate_type.eq("accepted_shortest_path_edge").sum(), 10)
            self.assertEqual(first.candidate_type.eq("additional_direct_edge_endpoint_coverage").sum(), 2)
            row["left_ProductName"] = "changed text"
            row["leakage_decision"] = "keep_together"
            pd.DataFrame([row]).to_csv(registry, index=False)
            second = sample_saved_path_edges(directory, registry, ["G1"], extra_edges_per_group=2, seed=42, verify_seal=False)
            pd.testing.assert_frame_equal(first, second)
            self.assertNotIn("leakage_decision", first)

    def test_edge_snapshot_detects_changes_outside_original_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = write_run(root)
            paths = [directory / f"{method}_matching_edges.csv" for method in ("rule", "tfidf")]
            snapshot = root / "edge_snapshot.json"
            snapshot.write_text(json.dumps({"input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                                               for path in paths}}), encoding="utf-8")
            result = audit_group_graph(directory, edge_snapshot=snapshot, verify_seal=False)
            self.assertTrue(result["verification"]["accepted_edge_snapshot_verified"])
            paths[0].write_text(paths[0].read_text().replace("0.9", "0.95"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "edge snapshot changed"):
                audit_group_graph(directory, edge_snapshot=snapshot, verify_seal=False)

    def test_identity_different_is_not_a_bad_leakage_edge(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_group_graph(write_run(Path(folder)), labels(("a", "b", "keep_together")), verify_seal=False)
            self.assertTrue(result["removal_counterfactuals"].empty)
            family = result["group_metrics"].query("method == 'rule' and group_id == 'family'").iloc[0]
            self.assertEqual(family.group_review_status, "partially_reviewed_no_conflict")
            self.assertEqual(family.reviewed_keep_separate_pairs, 0)

    def test_duplicate_oriented_edge_rows_preserve_all_reason_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = write_run(Path(folder), [("a", "b", "one", "0.9"), ("b", "a", "two", "0.8"),
                                                 ("b", "c", "three", "0.7")])
            result = audit_group_graph(directory, verify_seal=False)
            row = result["canonical_edges"].query("method == 'rule' and left_record_id == 'a'").iloc[0]
            self.assertEqual(row.source_edge_rows, 2)
            self.assertEqual(json.loads(row.reasons_json), ["one", "two"])
            self.assertEqual(len(json.loads(row.evidence_json)), 2)
            self.assertEqual(result["summary"]["methods"]["rule"]["duplicate_edge_rows"], 1)

    def test_invalid_graphs_fail_before_outputs_are_created(self):
        cases = [([], "do not reconstruct"),
                 ([("a", "d", "one", "1")], "crosses saved groups"),
                 ([("a", "missing", "one", "1")], "absent from assignments"),
                 ([("a", "a", "one", "1")], "invalid rule edge endpoints")]
        for edges, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with self.assertRaisesRegex(ValueError, error):
                    audit_group_graph(write_run(root, edges), output_dir=root / "audit", verify_seal=False)
                self.assertFalse((root / "audit").exists())

    def test_annotations_validate_target_exclusion_and_pair_identity(self):
        for mutation, error in (("duplicate", "unique"), ("target", "exclude Segment"),
                                ("unknown_id", "absent"), ("invalid_decision", "invalid leakage_decision")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                reviewed = labels(("a", "b", "keep_separate"))
                if mutation == "duplicate":
                    reviewed = pd.concat([reviewed, reviewed], ignore_index=True)
                elif mutation == "target":
                    reviewed.loc[0, "judgment_uses_Segment"] = True
                elif mutation == "unknown_id":
                    reviewed.loc[0, "left_record_id"] = "missing"
                else:
                    reviewed.loc[0, "leakage_decision"] = "different_product"
                with self.assertRaisesRegex(ValueError, error):
                    audit_group_graph(write_run(Path(folder)), reviewed, verify_seal=False)

    def test_real_audit_requires_seal_and_refuses_outputs_in_frozen_run(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = write_run(Path(folder))
            with self.assertRaises(FileNotFoundError):
                audit_group_graph(directory)
            with self.assertRaisesRegex(ValueError, "outside the frozen run"):
                audit_group_graph(directory, output_dir=directory / "audit", verify_seal=False)
            with self.assertRaisesRegex(ValueError, "screening thresholds"):
                audit_group_graph(directory, screening_thresholds=(1,), verify_seal=False)

    def test_cli_help_exposes_configurable_screening_thresholds(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit):
            main(["--help"])
        self.assertIn("--screening-thresholds", output.getvalue())


if __name__ == "__main__":
    unittest.main()
