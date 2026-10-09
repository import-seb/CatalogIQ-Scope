"""Synthetic end-to-end checks for the saved splitting comparison artifacts."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.cli import splitting_main
from catalogiq.features import sha256
from catalogiq.splitting import SplitConfig, check_assignments, record_ids
from catalogiq.splitting_experiment import load_input, run_experiment, verify_saved


def synthetic_rows():
    """Include repeated listings, both populated labels, and missing labels."""
    rows = []
    names = ["Hydrating oat facial lotion", "Copper travel drink bottle",
             "Silicone kitchen baking mat", "Cotton woven beach towel",
             "Rose blossom scented candle", "Vitamin citrus cleansing gel",
             "Aluminum folding picnic chair", "Ceramic glazed coffee mug",
             "Wireless optical computer mouse", "Lavender mild hand soap",
             "Bamboo bristle hair brush", "Waterproof walking trail boots"]
    for i, name in enumerate(names):
        for variant in range(2):
            rows.append({
                "dataset": "synthetic_training.csv",
                "source_sha256": "synthetic-training-fingerprint",
                "source_row": str(2 * i + variant),
                "ProductName": name,
                "ProductDescription": f"{name} for daily use",
                "ProductContents": name,
                "ProductBrand": f"Maker {i}",
                "Retailer": "Store",
                "ProductUrl": f"https://store.example/products/{i}",
                "Segment": ("" if i == 10 else "null" if i == 11
                            else "Personal care" if i % 2 else "Home"),
            })
    # Input order deliberately differs from identity sorting.
    return rows[::2] + rows[1::2]


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


class SplittingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "training_cleaned.csv"
        self.rows = synthetic_rows()
        write_csv(self.input, self.rows)
        self.config = SplitConfig(tfidf_max_df=1.0, cosine_chunk_size=2,
                                  split_restarts=1, split_refinement_passes=1)

    def run_saved(self, name="run", identifier_path=None):
        output = self.root / name
        with redirect_stdout(StringIO()):
            summary = run_experiment(self.input, output, self.config,
                                     identifier_path=identifier_path, progress=lambda _: None)
        return output, summary

    def identifiers(self):
        rows = [{**{c: row[c] for c in PROVENANCE},
                 "Sku": f"sku-{row['source_row']}",
                 "Upc": "036000291452" if row["source_row"] in {"0", "1"} else "",
                 "MDM_Id": f"mdm-{row['source_row']}"}
                for row in self.rows]
        # Extra unretained and prediction identities share row numbers. They
        # must never be accidentally matched or added to the training universe.
        rows.extend([
            {"dataset": "synthetic_target.csv", "source_sha256": "prediction-source",
             "source_row": "0", "Sku": "wrong-prediction", "Upc": "", "MDM_Id": "wrong"},
            {"dataset": "synthetic_training.csv", "source_sha256": "another-source",
             "source_row": "0", "Sku": "wrong-hash", "Upc": "", "MDM_Id": "wrong"},
            {"dataset": "synthetic_training.csv", "source_sha256": "synthetic-training-fingerprint",
             "source_row": "99", "Sku": "unretained", "Upc": "", "MDM_Id": "extra"},
        ])
        path = self.root / "training_candidate.csv"
        write_csv(path, list(reversed(rows)))
        return path, rows

    def test_end_to_end_saves_complete_comparison_and_preserves_inputs(self):
        identifiers, _ = self.identifiers()
        before = {path: sha256(path) for path in (self.input, identifiers)}
        output, summary = self.run_saved(identifier_path=identifiers)
        expected = {
            "config.json", "input_manifest.json", "attribute_profile.csv", "summary.json",
            "rule_assignments.csv", "tfidf_assignments.csv",
            "rule_matching_edges.csv", "tfidf_matching_edges.csv",
            "rule_group_sizes.csv", "tfidf_group_sizes.csv",
            "rule_group_size_distribution.csv", "tfidf_group_size_distribution.csv",
            "independent_near_duplicate_pairs.csv", "segment_distributions.csv",
            "disagreement_examples.csv",
        }
        self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
        for name, digest in summary["output_sha256"].items():
            self.assertEqual(sha256(output / name), digest)
        self.assertEqual({path: sha256(path) for path in before}, before)
        self.assertEqual(summary["input"]["rows"], len(self.rows))
        self.assertFalse(summary["input"]["prediction_target_records_used"])
        self.assertEqual(summary["identity_policy"]["grouping_label_columns_used"], [])
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertEqual(tuple(summary["config"]["ratios"]), (0.70, 0.15, 0.15))
        for key in ("same_records", "complete_unique_coverage", "group_isolation",
                    "labels_excluded_from_grouping", "input_artifacts_unchanged"):
            self.assertTrue(summary["checks"][key])
        frame, _ = load_input(self.input, identifiers)
        assignments = {}
        for method in ("rule", "tfidf"):
            self.assertTrue(summary["checks"]["reproducibility"][method]["passed"])
            assignment = pd.read_csv(output / f"{method}_assignments.csv", dtype=str,
                                     keep_default_na=False)
            check_assignments(assignment, frame["record_id"])
            self.assertEqual(len(assignment), len(self.rows))
            self.assertTrue(assignment.groupby("group_id")["split"].nunique().eq(1).all())
            self.assertEqual(assignment[PROVENANCE].to_dict("records"), frame[PROVENANCE].to_dict("records"))
            self.assertEqual(assignment["Segment"].tolist(), frame["Segment"].tolist())
            self.assertIn("metrics", summary["approaches"][method])
            self.assertGreaterEqual(summary["approaches"][method]["runtime_seconds"]["primary_total"], 0)
            assignments[method] = assignment
        self.assertEqual(assignments["rule"]["record_id"].tolist(), assignments["tfidf"]["record_id"].tolist())
        pairs = pd.read_csv(output / "independent_near_duplicate_pairs.csv", keep_default_na=False)
        for method in ("rule", "tfidf"):
            metrics = summary["approaches"][method]["metrics"]
            self.assertEqual(metrics["near_duplicate"]["pairs"], len(pairs))
            self.assertEqual(metrics["near_duplicate"]["crossing_pairs"],
                             pairs[f"{method}_crosses_split"].sum())
            self.assertEqual(metrics["groups"]["count"], assignments[method]["group_id"].nunique())
            self.assertEqual(metrics["rows"], len(frame))
        self.assertTrue(all(verify_saved(output).values()))

    def test_rerunning_same_configuration_produces_identical_assignments(self):
        first, _ = self.run_saved("first")
        second, _ = self.run_saved("second")
        for method in ("rule", "tfidf"):
            for suffix in ("assignments.csv", "matching_edges.csv", "group_sizes.csv",
                           "group_size_distribution.csv"):
                self.assertEqual((first / f"{method}_{suffix}").read_bytes(),
                                 (second / f"{method}_{suffix}").read_bytes())
        self.assertEqual((first / "independent_near_duplicate_pairs.csv").read_bytes(),
                         (second / "independent_near_duplicate_pairs.csv").read_bytes())

    def test_missing_segment_records_are_counted_and_saved(self):
        output, _ = self.run_saved()
        distribution = pd.read_csv(output / "segment_distributions.csv", keep_default_na=False)
        for method in ("rule", "tfidf"):
            missing = distribution.loc[distribution["approach"].eq(method)
                                       & distribution["Segment"].eq("<MISSING>")]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing.iloc[0]["total_rows"], 4)
            self.assertEqual(missing[["train_rows", "validation_rows", "test_rows"]].sum(axis=1).iloc[0], 4)
            assignment = pd.read_csv(output / f"{method}_assignments.csv", keep_default_na=False)
            self.assertEqual(assignment["Segment"].isin(["", "null"]).sum(), 4)

    def test_saved_artifact_tampering_is_detected(self):
        output, _ = self.run_saved()
        changed = output / "rule_assignments.csv"
        changed.write_bytes(changed.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Saved artifact changed: rule_assignments.csv"):
            verify_saved(output)

    def test_rehashed_cross_boundary_group_is_still_rejected(self):
        output, summary = self.run_saved()
        artifact = output / "rule_assignments.csv"
        assignment = pd.read_csv(artifact, dtype=str, keep_default_na=False)
        group = assignment["group_id"].value_counts().index[0]
        member = assignment.index[assignment["group_id"].eq(group)][0]
        old_split = assignment.at[member, "split"]
        assignment.at[member, "split"] = "validation" if old_split != "validation" else "test"
        assignment.to_csv(artifact, index=False)
        summary["output_sha256"][artifact.name] = sha256(artifact)
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Product group crosses split boundaries"):
            verify_saved(output)

    def test_saved_source_input_tampering_is_detected(self):
        output, _ = self.run_saved()
        self.input.write_bytes(self.input.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "fingerprint changed|artifact changed"):
            verify_saved(output)

    def test_saved_identifier_input_tampering_is_detected(self):
        identifiers, rows = self.identifiers()
        output, _ = self.run_saved(identifier_path=identifiers)
        rows[0]["Sku"] = "changed-after-run"
        write_csv(identifiers, rows)
        with self.assertRaisesRegex(ValueError, "fingerprint changed|artifact changed"):
            verify_saved(output)

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "marker.txt"
        marker.write_text("existing work", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_experiment(self.input, output, self.config, progress=lambda _: None)
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing work")
        self.assertEqual(list(output.iterdir()), [marker])

    def test_identifiers_join_by_full_identity_without_adding_or_dropping_rows(self):
        identifiers, _ = self.identifiers()
        before = self.input.read_bytes(), identifiers.read_bytes()
        frame, metadata = load_input(self.input, identifiers)
        expected_ids = set(record_ids(pd.DataFrame(self.rows)))
        self.assertEqual(set(frame["record_id"]), expected_ids)
        self.assertEqual(len(frame), len(self.rows))
        self.assertEqual(frame["record_id"].tolist(), sorted(expected_ids))
        self.assertEqual(frame["Sku"].tolist(), [f"sku-{row}" for row in frame["source_row"]])
        self.assertEqual(set(metadata["joined_identifier_columns"]), {"Sku", "Upc", "MDM_Id"})
        self.assertEqual((self.input.read_bytes(), identifiers.read_bytes()), before)

    def test_export_manifest_finds_identifier_source_in_original_or_copied_checkout(self):
        identifiers, _ = self.identifiers()
        run_dir = self.root / "integrated_run"
        run_dir.mkdir()
        identifiers = identifiers.rename(run_dir / identifiers.name)
        manifest = self.input.with_suffix(".summary.json")
        for source_run in (run_dir, self.root / "original_checkout" / run_dir.name):
            with self.subTest(source_run=str(source_run)):
                manifest.write_text(json.dumps({
                    "output_sha256": sha256(self.input), "rows": len(self.rows),
                    "source_run": str(source_run),
                    "input_sha256": {identifiers.name: sha256(identifiers)},
                }), encoding="utf-8")
                frame, metadata = load_input(self.input)
                self.assertEqual(len(frame), len(self.rows))
                self.assertEqual(metadata["identifier_source"], str(identifiers.resolve()))
                self.assertEqual(frame["Sku"].tolist(), [f"sku-{row}" for row in frame["source_row"]])
                self.assertEqual(metadata["fingerprints"][str(manifest.resolve())], sha256(manifest))

    def test_identifiers_missing_a_retained_identity_are_rejected(self):
        identifiers, rows = self.identifiers()
        write_csv(identifiers, rows[1:])
        with self.assertRaisesRegex(ValueError, "does not cover every retained record"):
            load_input(self.input, identifiers)

    def test_duplicate_identifier_identity_is_rejected(self):
        identifiers, rows = self.identifiers()
        write_csv(identifiers, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "unique"):
            load_input(self.input, identifiers)

    def test_conflicting_existing_identifier_is_rejected(self):
        identifiers, _ = self.identifiers()
        write_csv(self.input, [row | {"Sku": "conflicting"} for row in self.rows])
        with self.assertRaisesRegex(ValueError, "Conflicting identifier values in Sku"):
            load_input(self.input, identifiers)

    def test_cleaned_export_manifest_hash_mismatch_is_rejected(self):
        self.input.with_suffix(".summary.json").write_text(json.dumps({
            "output_sha256": "not-the-cleaned-file-hash", "rows": len(self.rows),
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from its export manifest"):
            load_input(self.input)

    def test_cleaned_export_manifest_row_count_mismatch_is_rejected(self):
        self.input.with_suffix(".summary.json").write_text(json.dumps({
            "output_sha256": sha256(self.input), "rows": len(self.rows) + 1,
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row count differs"):
            load_input(self.input)

    def test_integrated_identifier_manifest_hash_mismatch_is_rejected(self):
        identifiers, _ = self.identifiers()
        (self.root / "summary.json").write_text(json.dumps({
            "output_sha256": {identifiers.name: "not-the-identifier-file-hash"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from its integrated manifest"):
            load_input(self.input, identifiers)

    def test_cleaned_export_source_identifier_hash_mismatch_is_rejected(self):
        identifiers, _ = self.identifiers()
        self.input.with_suffix(".summary.json").write_text(json.dumps({
            "output_sha256": sha256(self.input), "rows": len(self.rows),
            "input_sha256": {identifiers.name: "not-the-original-source-hash"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the cleaned export's source"):
            load_input(self.input, identifiers)

    def test_cli_verify_reports_success_and_rejects_changed_artifact(self):
        output, _ = self.run_saved()
        stdout = StringIO()
        with patch("sys.argv", ["catalogiq-compare-splits", "--verify", str(output)]), redirect_stdout(stdout):
            splitting_main()
        self.assertTrue(json.loads(stdout.getvalue())["allocation_reproduced"])
        artifact = output / "tfidf_assignments.csv"
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        stderr = StringIO()
        with patch("sys.argv", ["catalogiq-compare-splits", "--verify", str(output)]), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                splitting_main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Saved artifact changed: tfidf_assignments.csv", stderr.getvalue())

    def test_cli_configuration_and_seed_are_saved(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config.to_dict()), encoding="utf-8")
        output = self.root / "cli-run"
        stdout = StringIO()
        with patch("sys.argv", ["catalogiq-compare-splits", "--input", str(self.input),
                                "--output-dir", str(output), "--config", str(config_path),
                                "--seed", "17"]), redirect_stdout(stdout):
            splitting_main()
        saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["seed"], 17)
        self.assertEqual(saved["tfidf_max_df"], 1.0)
        self.assertEqual(saved["cosine_chunk_size"], 2)
        self.assertIn("Side-by-side split comparison", stdout.getvalue())
        self.assertTrue(all(verify_saved(output).values()))

    def test_v2_dispatch_preserves_evaluator_and_saved_reproduction(self):
        baseline, old_summary = self.run_saved("baseline")
        self.config = SplitConfig.from_dict(self.config.to_dict() | {"grouping_version": 2})
        refined, summary = self.run_saved("refined")
        self.assertEqual(summary["version"], "split-comparison-v2")
        self.assertEqual(summary["input"], old_summary["input"])
        fields = ["left_record_id", "right_record_id", "name_jaccard"]
        old_pairs = pd.read_csv(baseline / "independent_near_duplicate_pairs.csv")[fields]
        new_pairs = pd.read_csv(refined / "independent_near_duplicate_pairs.csv")[fields]
        pd.testing.assert_frame_equal(old_pairs, new_pairs)
        for method in ("rule", "tfidf"):
            self.assertTrue(summary["checks"]["reproducibility"][method]["passed"])
        self.assertTrue(all(verify_saved(refined).values()))


if __name__ == "__main__":
    unittest.main()
