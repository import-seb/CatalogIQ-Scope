"""Exercise the public splitting command using small, synthetic CSV inputs.

These subprocess tests use the same invocation as teammates and CI. They do
not patch matching, bypass argparse, require project datasets, or train models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
PROVENANCE = ["dataset", "source_sha256", "source_row"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def synthetic_rows():
    """Eighteen independent families, each with two packaging variants."""
    families = [
        ("Willow", "Mint Strong Lozenges"),
        ("Acacia", "Northern Edge Vitamin C Tablets"),
        ("Birch", "Citrus Blossom Hand Soap"),
        ("Cedar", "Oat Meadow Facial Lotion"),
        ("Dogwood", "Lavender Valley Bath Salts"),
        ("Elm", "Chamomile Grove Tea Bags"),
        ("Fir", "Copper Trail Drink Bottles"),
        ("Hawthorn", "Bamboo Bristle Hair Brushes"),
        ("Juniper", "Rose Garden Scented Candles"),
        ("Larch", "Silicone Kitchen Baking Mats"),
        ("Maple", "Cotton Coastal Beach Towels"),
        ("Oak", "Ceramic Glazed Coffee Mugs"),
        ("Pine", "Aluminum Folding Picnic Chairs"),
        ("Redwood", "Wireless Optical Computer Mice"),
        ("Spruce", "Waterproof Walking Trail Boots"),
        ("Sycamore", "Peppermint Botanical Lip Balm"),
        ("Walnut", "Honey Apricot Cleansing Gel"),
        ("Yew", "Calendula Gentle Body Cream"),
    ]
    rows = []
    for family, (brand, line) in enumerate(families):
        for variant in range(2):
            index = 2 * family + variant
            name = (f"{brand} {line}, 38 Count" if variant == 0
                    else f"3 Pack {brand} {line} 38 Count")
            rows.append({
                "dataset": "synthetic_training.csv",
                "source_sha256": "synthetic-source-fingerprint",
                "source_row": str(index),
                "ProductName": name,
                "ProductBrand": brand,
                "ProductDescription": ("null" if family == 1 else
                                       f"{line} — a precise, synthetic description."),
                "ProductContents": "" if family == 2 else f"Contents for {line}",
                "Retailer": "Synthetic Store",
                "ProductUrl": f"https://store.example/products/{family}/{variant}",
                "Segment": ("Vitamins", "Personal care", "Household")[family % 3],
                "ProductCode": f"{index:06d}",
                "SourceNote": 'NA' if family == 3 else 'Keep commas, quotes "and"\nnewlines.',
            })
    # Input order must not become an accidental record identity.
    return rows[::2] + rows[1::2]


class SplitDataCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "cleaned.csv"
        self.rows = synthetic_rows()
        write_csv(self.input, self.rows)

    def command(self, output, *options, input_path=None):
        # In CI catalogiq comes from `pip install .`; do not inject src here.
        return subprocess.run(
            [sys.executable, "-m", "scripts.split_data", "--input",
             str(input_path or self.input), "--output-dir", str(output),
             *map(str, options)],
            cwd=REPOSITORY, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60, check=False,
        )

    def run_command(self, output, *options):
        result = self.command(output, *options)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))

    def exported_records(self, output):
        frames = []
        for split in SPLITS:
            frame = read_csv(output / f"{split}.csv")
            self.assertFalse(frame.empty, f"Synthetic {split} export is empty")
            self.assertTrue(frame["split"].eq(split).all())
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_command_exports_full_rows_and_reproducibility_metadata(self):
        output = self.root / "run"
        input_before = self.input.read_bytes()
        summary = self.run_command(output)
        expected = {
            "train.csv", "validation.csv", "test.csv", "rule_assignments.csv",
            "rule_matching_edges.csv", "rule_group_sizes.csv",
            "segment_distributions.csv", "split_config.json",
            "rule_refinement_config.json", "input_manifest.json", "summary.json",
        }
        self.assertTrue(expected.issubset({p.name for p in output.iterdir()}))

        combined = self.exported_records(output)
        assignments = read_csv(output / "rule_assignments.csv")
        self.assertEqual(len(combined), len(self.rows))
        self.assertTrue(combined["record_id"].is_unique)
        self.assertEqual(set(combined["source_row"]), {row["source_row"] for row in self.rows})
        self.assertTrue(combined.groupby("group_id")["split"].nunique().eq(1).all())
        pd.testing.assert_frame_equal(
            combined.set_index("record_id")[["group_id", "split"]].sort_index(),
            assignments.set_index("record_id")[["group_id", "split"]].sort_index(),
        )
        original = read_csv(self.input).set_index("source_row").sort_index()
        recovered = combined.set_index("source_row")[original.columns].sort_index()
        pd.testing.assert_frame_equal(recovered, original)
        indexed = combined.set_index("source_row")
        self.assertEqual(indexed.at["0", "group_id"], indexed.at["1", "group_id"])
        self.assertEqual(indexed.at["0", "split"], indexed.at["1", "split"])
        for row in combined.to_dict("records"):
            identity = json.dumps([row[c] for c in PROVENANCE],
                                  ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(row["record_id"], hashlib.sha256(identity.encode()).hexdigest())

        self.assertEqual(self.input.read_bytes(), input_before)
        self.assertTrue(summary["version"])
        self.assertEqual(summary["target"], "Segment")
        self.assertEqual(summary["grouping_method"], "rule_v3")
        self.assertEqual(summary["input"]["sha256"], digest(self.input))
        self.assertEqual(summary["input"]["rows"], len(self.rows))
        self.assertEqual(summary["split_config"]["seed"], 42)
        self.assertEqual(summary["split_config"]["ratios"], [0.70, 0.15, 0.15])
        for key, filename in (("input", "input_manifest.json"),
                              ("split_config", "split_config.json"),
                              ("rule_refinement_config", "rule_refinement_config.json")):
            self.assertEqual(summary[key], json.loads((output / filename).read_text(encoding="utf-8")))
        for check in ("complete_unique_coverage", "group_isolation", "export_values_preserved",
                      "input_artifacts_unchanged", "labels_excluded_from_grouping"):
            self.assertTrue(summary["checks"][check], check)
        self.assertTrue(summary["environment"])
        self.assertTrue(summary["grouping"])
        self.assertEqual(summary["metrics"]["rows"], len(self.rows))
        self.assertGreater(summary["metrics"]["groups"]["count"], 0)
        self.assertIn("size_summary", summary["metrics"]["groups"])
        self.assertIn("near_duplicate", summary["metrics"])
        self.assertIn("exact_payload", summary["metrics"])
        self.assertEqual(summary["evaluation"]["method"], "independent_character_shingle_jaccard")
        self.assertTrue(summary["evaluation"]["exhaustive_within_scope"])
        self.assertTrue(summary["runtime_seconds"])
        self.assertTrue(summary["output_sha256"])
        for filename, expected_hash in summary["output_sha256"].items():
            self.assertEqual(digest(output / filename), expected_hash, filename)

    def test_repeated_command_is_byte_reproducible_and_records_custom_settings(self):
        split_config = self.root / "split.json"
        split_config.write_text(json.dumps({"grouping_version": 2, "seed": 11,
                                           "split_restarts": 1, "split_refinement_passes": 1}),
                                encoding="utf-8")
        grouping_config = self.root / "grouping.json"
        grouping_config.write_text(json.dumps({"component_representatives": 3,
                                               "candidate": {"char_top_k": 6}}), encoding="utf-8")
        options = ("--config", split_config, "--grouping-config", grouping_config, "--seed", 123)
        first, second = self.root / "first", self.root / "second"
        summary = self.run_command(first, *options)
        self.run_command(second, *options)
        self.assertEqual(summary["split_config"]["seed"], 123)
        self.assertEqual(summary["split_config"]["split_restarts"], 1)
        self.assertEqual(summary["rule_refinement_config"]["component_representatives"], 3)
        self.assertEqual(summary["rule_refinement_config"]["candidate"]["char_top_k"], 6)
        csv_names = {path.name for path in first.glob("*.csv")}
        self.assertEqual(csv_names, {path.name for path in second.glob("*.csv")})
        for name in csv_names:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_optional_identifiers_are_joined_by_complete_provenance_and_exported(self):
        identifiers = self.root / "identifiers.csv"
        rows = [{**{key: row[key] for key in PROVENANCE},
                 "Sku": "000" + row["source_row"],
                 "Upc": "036000291452" if row["source_row"] in {"0", "1"} else "",
                 "MDM_Id": "trace-" + row["source_row"]}
                for row in reversed(self.rows)]
        rows.append({"dataset": "synthetic_prediction.csv", "source_sha256": "other-source",
                     "source_row": "0", "Sku": "wrong-record", "Upc": "", "MDM_Id": "wrong-record"})
        write_csv(identifiers, rows)
        before = {path: path.read_bytes() for path in (self.input, identifiers)}
        output = self.root / "with-identifiers"
        summary = self.run_command(output, "--identifiers", identifiers)
        combined = self.exported_records(output).set_index("source_row")
        self.assertEqual(len(combined), len(self.rows))
        for row in self.rows:
            self.assertEqual(combined.at[row["source_row"], "Sku"], "000" + row["source_row"])
            self.assertEqual(combined.at[row["source_row"], "MDM_Id"], "trace-" + row["source_row"])
        self.assertEqual(combined.at["0", "Upc"], "036000291452")
        self.assertEqual(combined.at["1", "Upc"], "036000291452")
        self.assertEqual(summary["input"]["fingerprints"][str(identifiers.resolve())], digest(identifiers))
        self.assertIn("Upc", summary["input"]["joined_identifier_columns"])
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_existing_output_is_refused_without_modifying_it(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "train.csv"
        sentinel.write_bytes(b"An existing experiment must stay untouched.\n")
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr.lower())
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def test_invalid_input_and_configs_fail_usefully_without_output(self):
        missing_segment = self.root / "missing-segment.csv"
        missing_identity = self.root / "missing-identity.csv"
        frame = read_csv(self.input)
        frame.drop(columns="Segment").to_csv(missing_segment, index=False)
        frame.drop(columns="source_sha256").to_csv(missing_identity, index=False)
        malformed = self.root / "malformed.json"
        malformed.write_text('{"seed":', encoding="utf-8")
        unknown = self.root / "unknown.json"
        unknown.write_text('{"misspelled_parameter": 1}', encoding="utf-8")
        invalid_candidate = self.root / "invalid-candidate.json"
        invalid_candidate.write_text('{"candidate": {"char_top_k": 0}}', encoding="utf-8")
        cases = [
            ("missing-segment", missing_segment, (), "Segment"),
            ("missing-identity", missing_identity, (), "source"),
            ("malformed-json", self.input, ("--config", malformed), "config"),
            ("unknown-config", self.input, ("--config", unknown), "misspelled_parameter"),
            ("invalid-candidate", self.input, ("--grouping-config", invalid_candidate), "char_top_k"),
        ]
        for name, input_path, options, expected_message in cases:
            with self.subTest(name=name):
                output = self.root / name
                result = self.command(output, *options, input_path=input_path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_message.lower(), result.stderr.lower())
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists(), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
