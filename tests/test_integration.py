import csv
import json
from pathlib import Path
import tempfile
import unittest

from catalogiq.cleaning import clean_training
from catalogiq.features import sha256, IDS
from catalogiq.integration import identity, row_decision, run, target_audit, load_structural_decisions
from test_feature_decisions import sample_row
import pandas as pd


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class IntegrationTests(unittest.TestCase):
    def test_current_policy_end_to_end_and_feature_projection(self):
        from catalogiq.integration_validation import verify
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            train, target = root / "raw/train.csv", root / "raw/target.csv"
            base = sample_row(Mnfr="J&J", Retailer="Shop", ProductName="x" * 10,
                              ProductDescription="x" * 10, ProductContents="x" * 10,
                              ProductBrand="x" * 10, ProductCategory="ABCD > EFG",
                              ProductRating="4", ReviewsCount="10", ProductReviewsCount="10", XRatXRev="40")
            empty_text = dict.fromkeys(("ProductName", "ProductDescription", "ProductContents"), " null ")
            cases = [({"Exclude": "Exclude"}, "candidate"),
                     ({"MDM_InsertDateTime": "not a date"}, "candidate"),
                     ({"MDM_InsertDateTime": "https://example.test/image"}, "candidate"),
                     ({"ProductUrl": "bad URL"}, "candidate"),
                     ({"ProductRating": "bad"}, "candidate"),
                     (empty_text, "quarantine"),
                     (empty_text | {"ProductUrl": "bad"}, "quarantine"),
                     ({"ProductRating": "text", "ReviewsCount": "text"}, "quarantine"),
                     (empty_text | {"ProductRating": "text", "ReviewsCount": "text"}, "quarantine"),
                     ({"ProductRating": "text", "MDM_InsertDateTime": "https://example.test/image"}, "quarantine"),
                     ({"MDM_InsertDateTime": "https://example.test/image", "ProductUrl": "bad"}, "quarantine"),
                     ({"Sun1": "ingredient", "Notes": "fragment", "Exclude": "shifted text"}, "quarantine")]
            rows = [dict(base) for _ in range(30)] + [base | changes for changes, _ in cases]
            write_rows(train, rows)
            write_rows(target, [r | dict.fromkeys(("Mnfr", "Brand", "Platform", "Segment", "Sub-Segment", "TargetAgeGroup"), "null") for r in rows])
            out = root / "out"
            result = run(train, target, out)
            checked = verify(train, target, out)
            for role in ("training", "target"):
                audit = [r for r in read_rows(out / "row_decisions.csv") if r["dataset"] == role]
                self.assertEqual([r["status"] for r in audit[30:]], [status for _, status in cases])
                for r in audit[31:33]:
                    self.assertNotIn("MDM_InsertDateTime", r["feature_flags"])
                    self.assertNotIn("timestamp", r["structural_finding_codes"])
                self.assertIn("ProductUrl:invalid_http_url", audit[33]["feature_flags"])
                self.assertEqual(audit[38]["hold_reasons"], "")
                self.assertIn("structural:missing_product_text", audit[35]["quarantine_reasons"])
                self.assertIn("structural:missing_product_text", audit[38]["quarantine_reasons"])
                exported = read_rows(out / f"{role}_features.csv")
                self.assertEqual(len(exported), result["rows"][role]["candidate"])
                self.assertTrue(all(c not in exported[0] for c in ("Sun1", "Sun2", "Sun3", "Sun4", "Sun5", "Notes", "MDM_InsertDateTime")))
                self.assertFalse(any(c.startswith("retailer_category_level_") for c in exported[0]))
                self.assertIn("ProductCategory", exported[0])
                self.assertIn("Retailer", exported[0])
                self.assertTrue(checked[role]["partitions_and_feature_exports_verified"])
            self.assertTrue(result["independent_structural_screening_applied"])
            self.assertEqual(result["exclude_policy"], "keep")
            # Rehash a tampered projection: the verifier must check actual values too.
            exported = read_rows(out / "training_features.csv")
            exported[0]["ProductName"] = "invented product"
            write_rows(out / "training_features.csv", exported)
            result["output_sha256"]["training_features.csv"] = sha256(out / "training_features.csv")
            (out / "summary.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Feature export value"):
                verify(train, target, out)

    def test_structural_identity_join_rejects_foreign_duplicates_and_ignores_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.csv"
            rows = [dict(dataset="training", source_sha256="digest", source_row=str(n),
                         quarantine="0", reason_codes="[]", finding_codes="[]") for n in (2, 1)]
            write_rows(path, rows)
            result = load_structural_decisions(path, "training", "digest")
            self.assertEqual(set(result), {("training", "digest", "1"), ("training", "digest", "2")})
            for bad in (rows + [rows[0]], [rows[0] | {"source_sha256": "foreign"}],
                        [rows[0] | {"dataset": "target"}], [rows[0] | {"source_row": "0"}],
                        [rows[0] | {"quarantine": "1"}]):
                write_rows(path, bad)
                with self.assertRaises(ValueError):
                    load_structural_decisions(path, "training", "digest")

    def test_union_precedence_and_marker_policies(self):
        feature = dict(reasons="", source_exclude_marker="0", requires_review="0",
                       recommend_quarantine="0", hold_reasons="")
        clean_structural = dict(reason_codes="[]", quarantine="0")
        def decision(f, codes=(), policy="keep", structural=clean_structural):
            return row_decision(f, codes, policy, structural=structural)
        self.assertEqual(decision(feature, [2, 3]), ("candidate", [], []))
        marked = {**feature, "source_exclude_marker": "1"}
        self.assertEqual(decision(marked)[0], "candidate")
        self.assertEqual(decision(marked, policy="hold")[0], "review_hold")
        self.assertEqual(decision(marked, policy="quarantine")[1], ["source:Exclude"])
        shifted = {**marked, "reasons": "displaced", "recommend_quarantine": "1",
                   "requires_review": "1", "hold_reasons": "missing_product_text"}
        status, reasons, holds = decision(shifted, [2, 6], structural={"reason_codes": '["shifted"]'})
        self.assertEqual(status, "quarantine")
        self.assertEqual(reasons, ["feature:displaced", "target:6", "structural:shifted"])
        self.assertEqual(holds, ["feature:missing_product_text"])
        self.assertEqual(decision({**feature, "requires_review": "1"})[0], "candidate")
        self.assertEqual(decision({**feature, "hold_reasons": "missing_product_text"})[0], "review_hold")
        self.assertEqual(decision(feature, structural={"reason_codes": '["shifted"]'})[0], "quarantine")
        with self.assertRaisesRegex(ValueError, "Unknown Exclude"):
            decision(feature, policy="guess")

    def test_target_join_checks_coverage_and_duplicates_not_position(self):
        rows = [sample_row(Mnfr="J&J") for _ in range(3)]
        for number, row in enumerate(rows, 1):
            row.update(dataset="training", source_sha256="digest", source_row=number)
        result = clean_training(pd.DataFrame(rows))
        expected = {identity(row) for row in rows}
        result.cleaned = result.cleaned.iloc[::-1]
        codes, _ = target_audit(result, expected)
        self.assertEqual(set(codes), expected)
        with self.assertRaisesRegex(ValueError, "source identities"):
            target_audit(result, expected | {("training", "digest", "4")})
        result.cleaned = pd.concat([result.cleaned, result.cleaned.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate target"):
            target_audit(result, expected)

    def test_full_pipeline_preserves_strings_and_has_one_partition_per_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            train, target = root / "raw" / "train.csv", root / "raw" / "predict.csv"
            # Duplicate JoiningKey and embedded newline cannot be used as row identity.
            base = sample_row(JoiningKey="duplicate", Mnfr="J&J", Upc="00123", Brand="NA",
                              ProductRating="4", ProductReviewsCount="10", ReviewsCount="10", XRatXRev="40")
            base["Sub-Segment"] = "Other Lifestyle CHC"
            rows = [dict(base) for _ in range(24)]
            rows[0]["ProductCategory"] = ""
            rows[1]["ProductCategory"] = " NULL "
            rows[2]["ProductName"] = " line one\nline two "
            rows[3]["Sub-Segment"] = "Cold / Flu"
            rows[4]["Sub-Segment"] = "Other Lifestyle"
            rows[5]["Exclude"] = "Exclude"
            rows[6]["ProductUrl"] = "not a url"
            rows[7].update(ProductRating="shifted text", ReviewsCount="more text")
            rows[8]["Mnfr"] = "bad manufacturer"
            rows[9].update(Mnfr="bad manufacturer", ProductRating="shifted text", ReviewsCount="more text")
            predicted = [sample_row(ProductCategory="", Upc="001"), sample_row(ProductCategory="null")]
            write_rows(train, rows)
            write_rows(target, predicted)
            raw_hashes = sha256(train), sha256(target)
            out = root / "run1"
            summary = run(train, target, out)
            self.assertEqual(raw_hashes, (sha256(train), sha256(target)))
            audit = read_rows(out / "row_decisions.csv")
            self.assertEqual(len(audit), 26)
            self.assertEqual(len({identity(r) for r in audit}), 26)
            for dataset, originals in (("training", rows), ("target", predicted)):
                combined = []
                for status in ("candidate", "quarantine", "review_hold"):
                    records = read_rows(out / f"{dataset}_{status}.csv")
                    self.assertEqual(len(records), summary["rows"][dataset][status])
                    combined.extend(records)
                self.assertEqual(sorted(int(r["source_row"]) for r in combined), list(range(1, len(originals) + 1)))
                for record in combined:
                    original = originals[int(record["source_row"]) - 1]
                    for column in IDS | {"Brand", "Mnfr", "MDM_InsertDateTime", "Category"}:
                        self.assertEqual(record[column], original[column])
                    if original["ProductCategory"] in {"", "null", " NULL "}:
                        self.assertEqual(record["ProductCategory"], original["ProductCategory"])
            statuses = {int(r["source_row"]): r["status"] for r in audit if r["dataset"] == "training"}
            # Structural content rules no longer run inside target cleaning.
            by_number = {int(r["source_row"]): r for r in audit if r["dataset"] == "training"}
            self.assertEqual(by_number[6]["hold_reasons"], "")
            self.assertEqual(by_number[7]["hold_reasons"], "")
            for number in (6, 7):
                self.assertEqual(statuses[number], "candidate")
            self.assertEqual([statuses[i] for i in (8, 9, 10)], ["quarantine"] * 3)
            self.assertEqual(summary["changes_by_rule"], {"slash_spacing": 1, "reviewed_other_lifestyle": 1})
            self.assertFalse(summary["modeling_ready"])
            self.assertFalse(summary["final_mask_agreed"])
            second = run(train, target, root / "run2")
            self.assertEqual(summary, second)
            for path, digest in summary["output_sha256"].items():
                self.assertEqual(sha256(out / path), digest)
            with self.assertRaises(FileExistsError):
                run(train, target, out)
            for unsafe in (train, train.parent, train.parent / "result", root):
                with self.assertRaises(ValueError):
                    run(train, target, unsafe)

    def test_unapproved_target_label_change_is_rejected(self):
        row = sample_row(Mnfr="J&J", ProductRating="4")
        row["Sub-Segment"] = "Cold / Flu"
        row.update(dataset="training", source_sha256="digest", source_row=1)
        result = clean_training(pd.DataFrame([row]))
        self.assertEqual(len(result.changes), 1)
        target_audit(result, {identity(row)})
        result.changes.loc[0, "new_value"] = "Invented label"
        with self.assertRaisesRegex(ValueError, "unapproved label change"):
            target_audit(result, {identity(row)})


if __name__ == "__main__":
    unittest.main()
