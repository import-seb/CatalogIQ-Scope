import csv
from pathlib import Path
import tempfile
import unittest

from catalogiq.cleaning import clean_training
from catalogiq.features import sha256, IDS
from catalogiq.integration import identity, row_decision, run, target_audit
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
    def test_union_precedence_and_marker_policies(self):
        feature = dict(reasons="", source_exclude_marker="0", requires_review="0", recommend_quarantine="0")
        self.assertEqual(row_decision(feature, [2, 3], "hold"), ("candidate", [], []))
        marked = {**feature, "source_exclude_marker": "1"}
        self.assertEqual(row_decision(marked, [], "hold")[0], "review_hold")
        self.assertEqual(row_decision(marked, [], "keep")[0], "candidate")
        self.assertEqual(row_decision(marked, [], "quarantine")[1], ["source:Exclude"])
        structural = {**marked, "reasons": "displaced", "recommend_quarantine": "1", "requires_review": "1"}
        status, reasons, holds = row_decision(structural, [2, 6], "hold")
        self.assertEqual(status, "quarantine")
        self.assertEqual(reasons, ["feature:displaced", "target:6"])
        self.assertEqual(holds, ["source_exclude_policy_unconfirmed"])
        self.assertEqual(row_decision({**feature, "requires_review": "1"}, [], "keep")[0], "review_hold")
        with self.assertRaisesRegex(ValueError, "Unknown Exclude"):
            row_decision(feature, [], "guess")

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
            # On a tiny synthetic distribution IQR reason 4 can supersede a
            # feature hold. The hold reason must still be preserved in the audit.
            by_number = {int(r["source_row"]): r for r in audit if r["dataset"] == "training"}
            self.assertIn("source_exclude_policy_unconfirmed", by_number[6]["hold_reasons"])
            self.assertIn("feature_field_review", by_number[7]["hold_reasons"])
            for number in (6, 7):
                self.assertEqual(statuses[number], "quarantine" if by_number[number]["quarantine_reasons"] else "review_hold")
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
