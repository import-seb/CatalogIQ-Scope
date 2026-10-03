"""Structural decisions using synthetic raw records, independent of cleaner masks."""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

from catalogiq.structural import (
    KEY, NUMERIC, QUARANTINE_RULES, TARGETS, StructuralConfig, assess_row,
    quantile, records, required_fields, run, sha256,
)
from collections import Counter


CONFIG = StructuralConfig()


def sample(**overrides):
    row = dict.fromkeys(sorted(required_fields(CONFIG) | {"JoiningKey", "Sku", "Upc", "Category", "ProductModelNumber"}), "null")
    row.update(Retailer="Shop", ProductCategory="Health > Care", ProductBrand="Example",
               ProductName="A product", ProductRating="4.5", ProductReviewsCount="10",
               ReviewsCount="10", XRatXRev="45", ProductUrl="https://example.test/p",
               ProductImageUrl="https://example.test/image", ProductDescription="Description",
               ProductContents="Contents", MDM_Id="00123", MDM_InsertDateTime="45117.83254",
               Mnfr="All others", Brand="up & up", TargetAgeGroup="Adult", JoiningKey="duplicate",
               Sku="00123", Upc="3.00054E+11")
    row.update(overrides)
    return row


def decide(row, role="training", config=CONFIG, upper=4):
    return assess_row(row, {"dataset": role, "source_sha256": "a" * 64, "source_row": 1},
                      config=config, missingness_upper_count=upper)


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(sample())
        for row in rows:
            writer.writerow([row[c] for c in sample()] if isinstance(row, dict) else row)


class StructuralTests(unittest.TestCase):
    def test_each_numeric_field_is_a_field_review_on_its_own(self):
        for role in ("training", "target"):
            for column in NUMERIC:
                with self.subTest(role=role, column=column):
                    row = sample(**{column: "ingredient text"})
                    if role == "target":
                        row.update(dict.fromkeys(TARGETS, "null"))
                    result = decide(row, role)
                    self.assertFalse(result["quarantine"])
                    self.assertIn("numeric_non_numeric", result["finding_codes"])
                    self.assertEqual(next(f for f in result["findings"] if f["reason_code"] == "numeric_non_numeric")
                                     ["evidence"]["raw_values"], {column: "ingredient text"})

    def test_corroborated_numeric_failures_in_both_datasets(self):
        cases = [({"ProductRating": "description", "ProductReviewsCount": "fragment"},
                  "text_in_multiple_numeric_anchors"),
                 ({"ProductRating": "description", "ReviewsCount": "2.5"},
                  "text_numeric_failure_with_corroboration"),
                 ({"XRatXRev": "fragment", "MDM_Id": "https://example.test/image"},
                  "text_numeric_failure_with_corroboration"),
                 ({"ReviewsCount": "description", "ProductUrl": "ingredient"},
                  "text_numeric_failure_with_corroboration")]
        for role in ("training", "target"):
            for values, reason in cases:
                with self.subTest(role=role, values=values):
                    row = sample(**values)
                    if role == "target":
                        row.update(dict.fromkeys(TARGETS, ""))
                    original = copy.deepcopy(row)
                    first, second = decide(row, role), decide(row, role)
                    self.assertEqual(first, second)
                    self.assertEqual(row, original)
                    self.assertTrue(first["quarantine"])
                    self.assertIn(reason, first["reason_codes"])

    def test_clean_values_and_numeric_boundaries(self):
        for values in ({"ProductRating": " 4.5 ", "ProductReviewsCount": "0", "ReviewsCount": "2.0"},
                       {"ProductReviewsCount": "1E3", "XRatXRev": "0"},
                       {"ProductRating": "null"},
                       {"MDM_Id": "ASIN123", "Upc": "3.0E+11", "Sku": "A-123"},
                       dict.fromkeys(TARGETS, "null")):
            result = decide(sample(**values))
            self.assertFalse(result["quarantine"])
            self.assertFalse(any(code.startswith("numeric_") for code in result["finding_codes"]))
        for values in ({"ReviewsCount": "-1"}, {"ProductRating": "NaN"},
                       {"ProductRating": "Infinity"},
                       {"ProductReviewsCount": "2.5", "ReviewsCount": "4.2"},
                       {"ProductRating": "NA"},
                       {"ProductRating": "bad", "MDM_InsertDateTime": "2026-09-30"}):
            with self.subTest(values=values):
                self.assertFalse(decide(sample(**values))["quarantine"])

    def test_shifted_metadata_needs_two_distinct_fields(self):
        for values in ({"Sun1": "ingredient"}, {"Exclude": "ingredient"},
                       {"Exclude": "Exclude", "Sun1": "ingredient"},
                       {"MDM_Id": "https://example.test/image"}, {"ProductUrl": "http://["}):
            self.assertFalse(decide(sample(**values))["quarantine"])
        result = decide(sample(Exclude="shifted text", Sun1="ingredient"))
        self.assertEqual(result["reason_codes"], ["unexpected_exclude_with_populated_unknown_columns"])

    def test_multiple_target_warnings_are_not_semantic_failures(self):
        result = decide(sample(Brand="up & up", Segment="lowercase valid", Platform=" Rare label"))
        self.assertFalse(result["quarantine"])
        self.assertIn("target_format_review", result["finding_codes"])
        self.assertFalse(decide(sample(Mnfr="new manufacturer"))["quarantine"])
        result = decide(sample(Mnfr="ingredient text", TargetAgeGroup="Paleo friendly"))
        self.assertEqual(result["reason_codes"], ["multiple_suspicious_targets"])
        result = decide(sample(Mnfr="ingredient text", ProductRating="shifted text"))
        self.assertEqual(result["reason_codes"], ["suspicious_target_with_displacement"])
        result = decide(sample(Brand="https://example.test/p", Segment="https://example.test/category"))
        self.assertIn("multiple_suspicious_targets", result["reason_codes"])
        config = replace(CONFIG, manufacturer_values=(*CONFIG.manufacturer_values, "new manufacturer"))
        self.assertNotIn("suspicious_target_value", decide(sample(Mnfr="new manufacturer"), config=config)["finding_codes"])

    def test_missingness_boundary_excludes_labels_and_metadata(self):
        row = sample(ProductName="", ProductDescription=" null ")
        self.assertNotIn("high_feature_missingness", decide(row, upper=2)["finding_codes"])
        row["ProductContents"] = ""
        result = decide(row, upper=2)
        self.assertIn("high_feature_missingness", result["finding_codes"])
        self.assertFalse(result["quarantine"])
        row.update(dict.fromkeys((*TARGETS, "Category", "MDM_Id"), ""))
        evidence = next(f["evidence"] for f in decide(row, upper=2)["findings"]
                        if f["reason_code"] == "high_feature_missingness")
        self.assertEqual(evidence["missing_count"], 3)
        row.update(dict.fromkeys(CONFIG.missingness_fields, ""))
        result = decide(row, upper=2)
        self.assertIn("no_valid_numeric_anchor", result["finding_codes"])
        self.assertFalse(result["quarantine"])
        self.assertEqual(quantile(Counter({0: 3, 4: 1}), .75), 1)
        self.assertEqual(quantile(Counter(), .75), 0)

    def test_configuration_disables_quarantine_not_evidence(self):
        row = sample(ProductRating="bad", ReviewsCount="text")
        config = replace(CONFIG, quarantine_rules=())
        result = decide(row, config=config)
        self.assertFalse(result["quarantine"])
        self.assertIn("text_in_multiple_numeric_anchors", result["finding_codes"])
        for values in ({"numeric_fields": ["source_row"]}, {"count_fields": ["unknown"]},
                       {"missingness_fields": ["Brand"]}, {"missingness_fields": []},
                       {"missingness_iqr_multiplier": -1}, {"missingness_iqr_multiplier": float("nan")},
                       {"quarantine_rules": ["unknown"]}, {"unknown": []}, {"numeric_fields": "rating"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                StructuralConfig.from_dict(values)

    def test_invalid_identity_rejected(self):
        valid = {"dataset": "training", "source_sha256": "a" * 64, "source_row": 1}
        for override in ({"dataset": ""}, {"source_sha256": "bad"}, {"source_row": -1},
                         {"dataset": "training.csv"}, {"source_row": 0},
                         {"source_row": True}, {"source_row": "0"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                assess_row(sample(), valid | override, config=CONFIG, missingness_upper_count=1)

    def test_full_run_identity_audit_preservation_and_repeatability(self):
        from catalogiq.cleaning import load_source
        from scripts.validate_structural import verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            rows = [sample(ProductName='Quoted "name"\nsecond line'),
                    sample(ProductRating="description", ReviewsCount="fragment"),
                    sample(Exclude="Exclude"), sample(Mnfr="bad", TargetAgeGroup="fragment"),
                    sample(**dict.fromkeys(CONFIG.missingness_fields, "null"))]
            train, target = raw / "train.csv", raw / "target.csv"
            write_source(train, rows)
            target_rows = [r | dict.fromkeys(TARGETS, "null") for r in rows]
            write_source(target, target_rows)
            originals = [train.read_bytes(), target.read_bytes()]
            first = run(train, target, root / "out1")
            second = run(train, target, root / "out2")
            self.assertEqual(first, second)
            verified = verify(root / "out1")
            self.assertEqual(verified["training"]["verified_rows"], len(rows))
            self.assertEqual(verified["target"]["verified_rows"], len(rows))
            self.assertEqual(originals, [train.read_bytes(), target.read_bytes()])
            for role, path, expected_rows in (("training", train, rows), ("target", target, target_rows)):
                with (root / "out1" / f"{role}_decisions.csv").open(newline="") as stream:
                    decisions = list(csv.DictReader(stream))
                details = [json.loads(line) for line in (root / "out1" / f"{role}_evidence.jsonl")
                           .read_text().splitlines()]
                self.assertEqual([int(r["source_row"]) for r in decisions], list(range(1, len(rows) + 1)))
                self.assertEqual(len({tuple(r[k] for k in KEY) for r in decisions}), len(rows))
                self.assertEqual([r["dataset"] for r in decisions], [role] * len(rows))
                self.assertEqual([r["source_sha256"] for r in decisions], [sha256(path)] * len(rows))
                loaded = load_source(path)
                self.assertEqual((loaded["source_row"] + 1).tolist(), [int(r["source_row"]) for r in decisions])
                self.assertEqual(decisions[1]["quarantine"], "1")
                self.assertEqual(decisions[2]["quarantine"], "0")
                self.assertEqual(first["datasets"][role]["decisions"]["quarantine"], 2 if role == "training" else 1)
                for detail in details:
                    self.assertEqual(dict(zip(detail["raw_columns"], detail["raw_values"])),
                                     expected_rows[detail["source_row"] - 1])
                    for finding in detail["findings"]:
                        self.assertEqual({k: finding[k] for k in KEY}, {k: detail[k] for k in KEY})
                for code, counts in first["datasets"][role]["rows_by_finding_and_decision"].items():
                    self.assertEqual(sum(counts.values()), first["datasets"][role]["rows_by_finding"][code])
            for filename in (*first["output_sha256"], "summary.json"):
                self.assertEqual((root / "out1" / filename).read_bytes(), (root / "out2" / filename).read_bytes())
            with self.assertRaises(FileExistsError):
                run(train, target, root / "out1")
            for unsafe in (raw, raw / "out", root, train):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    run(train, target, unsafe)
            with self.assertRaisesRegex(ValueError, "distinct source files"):
                run(train, train, root / "overlap")
            # Rehash a tampered artifact: validator must check keys, not just hashes.
            audit_path = root / "out1/training_decisions.csv"
            with audit_path.open(newline="") as stream:
                audit_rows = list(csv.DictReader(stream))
            audit_rows[0]["source_row"] = "0"
            with audit_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
                writer.writeheader()
                writer.writerows(audit_rows)
            first["output_sha256"][audit_path.name] = sha256(audit_path)
            (root / "out1/summary.json").write_text(json.dumps(first))
            with self.assertRaisesRegex(ValueError, "identity/order"):
                verify(root / "out1")

    def test_wrong_width_has_identity_and_raw_cells_not_silent_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            malformed = ["too", "few"]
            extra = list(sample().values()) + ["extra"]
            write_source(train, [sample(), malformed, extra])
            write_source(target, [malformed])
            report = run(train, target, root / "out")
            for role, count in (("training", 2), ("target", 1)):
                self.assertEqual(report["datasets"][role]["rows_by_quarantine_reason"], {"csv_field_count": count})
            details = [json.loads(line) for line in (root / "out/training_evidence.jsonl").read_text().splitlines()]
            bad = [d for d in details if "csv_field_count" in d["reason_codes"]]
            self.assertEqual([d["source_row"] for d in bad], [2, 3])
            self.assertEqual([d["raw_values"] for d in bad], [malformed, extra])

    def test_invalid_schema_or_syntax_never_produces_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            write_source(target, [sample()])
            for content in ("a,a\nx,x\n", "Brand\nExample\n", "", ",".join(sample()) + '\n"unterminated'):
                train.write_text(content)
                with self.subTest(content=content), self.assertRaises((ValueError, csv.Error)):
                    run(train, target, root / "out")
                self.assertFalse((root / "out").exists())

    def test_header_only_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            write_source(train, [])
            write_source(target, [sample(), sample()])
            with target.open("a") as stream:
                stream.write("\n\n")
            self.assertEqual([n for n, _, _ in records(target, CONFIG)], [1, 2])
            report = run(train, target, root / "out")
            self.assertEqual(report["datasets"]["training"]["decisions"], {"keep": 0, "quarantine": 0})

    def test_record_numbers_match_canonical_reader_with_blank_and_quoted_records(self):
        from scripts.validate_structural import source_records

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            write_source(path, [sample(ProductName="first\n\n   \nlast")])
            with path.open("a", newline="") as stream:
                stream.write('\n   \n""\n"   "\n')
            with path.open(newline="") as stream:
                expected = list(csv.DictReader(stream))
            actual = list(records(path, CONFIG))
            self.assertEqual([n for n, _, _ in actual], list(range(1, len(expected) + 1)))
            self.assertEqual(len(actual), 4)
            self.assertEqual(actual[1][2], ["   "])
            self.assertEqual(actual[2][2], [""])
            self.assertEqual(actual[3][2], ["   "])
            with path.open(newline="") as stream:
                independent = list(source_records(stream))
            self.assertEqual([v for _, _, v in actual], independent[1:])

    def test_merged_feature_and_integrated_contracts_match_without_applying_masks(self):
        from catalogiq.feature_decisions import run as run_features
        from catalogiq.integration import run as run_integrated
        from scripts.validate_structural import verify_feature_compatibility

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            rows = [sample(ProductName="line one\nline two", Brand="NA"),
                    sample(ProductRating="fragment", ReviewsCount="other fragment"),
                    sample(ProductRating="fragment", MDM_InsertDateTime="2026-10-02"),
                    sample(Mnfr="fragment", TargetAgeGroup="another fragment")]
            write_source(train, rows)
            write_source(target, [r | dict.fromkeys(TARGETS, "null") for r in rows])
            before = train.read_bytes(), target.read_bytes()
            run(train, target, root / "structural")
            feature_dirs = {role: root / f"feature_{role}" for role in ("training", "target")}
            for role, source in (("training", train), ("target", target)):
                run_features(source, feature_dirs[role], role)
            feature_bytes = {p: p.read_bytes() for d in feature_dirs.values() for p in d.iterdir()}
            compared = verify_feature_compatibility(root / "structural", feature_dirs)
            self.assertEqual(compared, verify_feature_compatibility(root / "structural", feature_dirs))
            self.assertEqual(compared["training"]["decisions_compared"],
                             {"both_quarantine": 1, "structural_only": 1, "feature_only": 1, "neither": 1})
            self.assertEqual(compared["target"]["decisions_compared"],
                             {"both_quarantine": 1, "structural_only": 0, "feature_only": 1, "neither": 2})
            self.assertFalse(compared["training"]["masks_combined"])
            self.assertTrue(all(p.read_bytes() == content for p, content in feature_bytes.items()))
            # Exercise Max's existing integration on synthetic data only. Structural
            # results are never passed into its mask or row-selection implementation.
            run_integrated(train, target, root / "integrated")
            with (root / "integrated/row_decisions.csv").open(newline="") as stream:
                integrated = list(csv.DictReader(stream))
            structural = []
            for role in ("training", "target"):
                with (root / f"structural/{role}_decisions.csv").open(newline="") as stream:
                    structural.extend(csv.DictReader(stream))
            self.assertEqual([tuple(r[k] for k in KEY) for r in structural],
                             [tuple(r[k] for k in KEY) for r in integrated])
            self.assertEqual(before, (train.read_bytes(), target.read_bytes()))
            with self.assertRaisesRegex(ValueError, "Both feature runs"):
                verify_feature_compatibility(root / "structural", {"training": feature_dirs["training"]})
            with self.assertRaisesRegex(ValueError, "Dataset mismatch"):
                verify_feature_compatibility(root / "structural", {"training": feature_dirs["target"],
                                                                   "target": feature_dirs["training"]})
            # A foreign source fingerprint must be rejected before comparing decisions.
            manifest_path = feature_dirs["training"] / "summary.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_sha256"] = "b" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_feature_compatibility(root / "structural", feature_dirs)

    def test_shared_cli_structural_mode_preserves_independent_entrypoint(self):
        from catalogiq.cli import main, structural_main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            write_source(train, [sample(ProductRating="bad", ReviewsCount="text")])
            write_source(target, [sample()])
            config = root / "config.json"
            config.write_text('{"quarantine_rules": []}')
            common = ["--train", str(train), "--target", str(target), "--config", str(config)]
            for entrypoint, extra, name in ((main, ["--mode", "structural"], "shared"),
                                             (structural_main, [], "standalone")):
                with patch("sys.argv", ["catalogiq", *extra, *common, "--output-dir", str(root / name)]), \
                     redirect_stdout(io.StringIO()):
                    entrypoint()
            for p in (root / "shared").iterdir():
                self.assertEqual(p.read_bytes(), (root / "standalone" / p.name).read_bytes())
            report = json.loads((root / "shared/summary.json").read_text())
            self.assertEqual(report["datasets"]["training"]["decisions"]["quarantine"], 0)
            for args in (["--mode", "targets", "--config", str(config)],
                         ["--mode", "integrated", "--config", str(config)],
                         ["--mode", "structural", "--exclude-policy", "quarantine"]):
                with self.subTest(args=args), patch("sys.argv", ["catalogiq", *args]), \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
