"""Eligibility comes from the agreed cleaning policy, not from a rule invented here."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from catalogiq import target_classification as tc
from catalogiq.inference_engine import ProductPrediction, TargetPrediction
from catalogiq.target_minilm_preparation import INPUT_FIELDS


def frame(rows):
    columns = ["Exclude", "JoiningKey", "Retailer", "Category", "Sku", "Upc", "ProductUrl",
               *INPUT_FIELDS]
    return pd.DataFrame([{c: r.get(c) for c in columns} for r in rows], dtype="string")


def good(**overrides):
    row = {"JoiningKey": "k1", "Retailer": "Amazon", "ProductName": "Claritin RediTabs",
           "ProductCategory": "Allergy", "ProductBrand": "Claritin",
           "ProductDescription": "24 hour", "ProductContents": "10 tablets"}
    row.update(overrides)
    return row


def decisions(*rows):
    """Rows shaped like the integrated cleaner's row_decisions.csv."""
    return pd.DataFrame([{"dataset": "target", "JoiningKey": key, "status": status,
                          "quarantine_reasons": reasons, "hold_reasons": holds}
                         for key, status, reasons, holds in rows], dtype="string")


def verdicts(frame_, status="candidate", reasons="", holds=""):
    return decisions(*[(k, status, reasons, holds) for k in frame_["JoiningKey"]])


class EligibilityTests(unittest.TestCase):
    def test_a_candidate_row_is_eligible(self):
        f = frame([good()])
        self.assertTrue(tc.eligibility(f, verdicts(f)).isna().all())

    def test_the_exclude_marker_alone_no_longer_disqualifies(self):
        """decisions.md policy 1: keep otherwise usable rows; the marker is auditable only."""
        f = frame([good(Exclude="Exclude")])
        self.assertTrue(tc.eligibility(f, verdicts(f)).isna().all())

    def test_a_missing_name_alone_no_longer_disqualifies(self):
        """decisions.md policy 4: any one of name, description or contents avoids the rule."""
        f = frame([good(ProductName="")])
        self.assertTrue(tc.eligibility(f, verdicts(f)).isna().all())

    def test_the_cleaner_decides_and_its_reason_codes_are_carried_through(self):
        f = frame([good(JoiningKey="keep"), good(JoiningKey="drop")])
        d = decisions(("keep", "candidate", "", ""),
                      ("drop", "quarantine", "structural:missing_product_text", ""))
        reasons = tc.eligibility(f, d)
        self.assertTrue(pd.isna(reasons.iloc[0]))
        self.assertEqual(reasons.iloc[1], "quarantine: structural:missing_product_text")

    def test_a_hold_is_also_ineligible_and_keeps_its_reason(self):
        f = frame([good(JoiningKey="held")])
        d = decisions(("held", "review_hold", "", "feature:some_hold"))
        self.assertEqual(tc.eligibility(f, d).iloc[0], "review_hold: feature:some_hold")

    def test_a_blank_identifier_is_reported_not_crashed_on(self):
        f = frame([good(JoiningKey="k1"), good(JoiningKey="   ")])
        reasons = tc.eligibility(f, decisions(("k1", "candidate", "", "")))
        self.assertTrue(pd.isna(reasons.iloc[0]))
        self.assertEqual(reasons.iloc[1], "missing_identifier")

    def test_a_row_with_no_decision_is_refused_loudly(self):
        f = frame([good(JoiningKey="k1"), good(JoiningKey="orphan")])
        with self.assertRaises(ValueError):
            tc.eligibility(f, decisions(("k1", "candidate", "", "")))

    def test_duplicate_decisions_are_refused(self):
        f = frame([good(JoiningKey="dup")])
        d = decisions(("dup", "candidate", "", ""), ("dup", "quarantine", "x", ""))
        with self.assertRaises(ValueError):
            tc.eligibility(f, d)

    def test_a_decisions_file_for_another_dataset_is_refused(self):
        d = decisions(("k1", "candidate", "", ""))
        d["dataset"] = "training"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "row_decisions.csv"
            d.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                tc.load_decisions(path, dataset="target")

    def test_a_file_that_is_not_a_decisions_export_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "something.csv"
            pd.DataFrame({"a": ["1"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                tc.load_decisions(path)

    def test_nothing_is_dropped_silently(self):
        rows = [good(JoiningKey=f"k{i}") for i in range(5)]
        f = frame(rows)
        d = decisions(*[(f"k{i}", "candidate" if i else "quarantine",
                         "" if i else "structural:missing_product_text", "")
                        for i in range(5)])
        reasons = tc.eligibility(f, d)
        report = tc.eligibility_report(f, reasons)
        self.assertEqual(report["eligible"] + report["ineligible"], report["rows_in_file"])
        self.assertEqual(report["eligible"], 4)
        self.assertEqual(report["ineligible_by_reason_code"],
                         {"structural:missing_product_text": 1})
        self.assertEqual(len(tc.to_records(f, reasons)) + len(tc.ineligible_table(f, reasons)),
                         len(f))

    def test_the_report_names_where_eligibility_came_from(self):
        f = frame([good()])
        report = tc.eligibility_report(f, tc.eligibility(f, verdicts(f)))
        self.assertIn("row_decisions", report["eligibility_source"])
        self.assertIn("decisions.md", report["policy_reference"])


class RecordTests(unittest.TestCase):
    def test_record_id_is_the_source_identifier(self):
        f = frame([good(JoiningKey="JK-7")])
        self.assertEqual([r.record_id for r in tc.to_records(f, tc.eligibility(f, verdicts(f)))], ["JK-7"])

    def test_features_are_exactly_the_model_input_fields(self):
        f = frame([good()])
        self.assertEqual(set(tc.to_records(f, tc.eligibility(f, verdicts(f)))[0].features), set(INPUT_FIELDS))

    def test_blank_feature_values_become_none_not_empty_strings(self):
        f = frame([good(ProductBrand="   ", ProductContents=None)])
        features = tc.to_records(f, tc.eligibility(f, verdicts(f)))[0].features
        self.assertIsNone(features["ProductBrand"])
        self.assertIsNone(features["ProductContents"])
        self.assertEqual(features["ProductName"], "Claritin RediTabs")

    def test_ineligible_rows_never_reach_a_model(self):
        f = frame([good(JoiningKey="keep"), good(JoiningKey="drop")])
        reasons = tc.eligibility(f, decisions(("keep", "candidate", "", ""),
                                              ("drop", "quarantine", "structural:shift", "")))
        self.assertEqual([r.record_id for r in tc.to_records(f, reasons)], ["keep"])


class FakeEngine:
    def __init__(self):
        self.batches = []

    def predict(self, records):
        self.batches.append(len(records))
        return [ProductPrediction(r.record_id,
                                  [TargetPrediction("Segment", "unavailable", "selected",
                                                    review_reasons=["model_unavailable"],
                                                    provenance={"model_id": "m1"})])
                for r in records]


class ClassifyTests(unittest.TestCase):
    def rows(self, n):
        return frame([good(JoiningKey=f"k{i}") for i in range(n)])

    def test_batching_covers_every_record_once_and_in_order(self):
        f = self.rows(7)
        records = tc.to_records(f, tc.eligibility(f, verdicts(f)))
        engine = FakeEngine()
        results = tc.classify(engine, records, batch_size=3)
        self.assertEqual(engine.batches, [3, 3, 1])
        self.assertEqual([r.record_id for r in results], [r.record_id for r in records])

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            tc.classify(FakeEngine(), tc.to_records(self.rows(1), tc.eligibility(self.rows(1), verdicts(self.rows(1)))), batch_size=0)

    def test_export_keeps_source_columns_and_uses_dashboard_naming(self):
        f = self.rows(2)
        results = tc.classify(FakeEngine(), tc.to_records(f, tc.eligibility(f, verdicts(f))), batch_size=10)
        envelope = tc.response(results, registry_version="relationships-test")
        table = tc.dashboard_table(f, results, envelope)
        self.assertEqual(len(table), 2)
        for column in ("JoiningKey", "Retailer", "ProductName"):
            self.assertIn(column, table.columns)
        self.assertEqual(list(table["JoiningKey"]), ["k0", "k1"])
        self.assertIn("catalogiq:Segment:original_label", table.columns)
        self.assertIn("catalogiq:Segment:final_label", table.columns)
        self.assertIn("catalogiq:Segment:model_version", table.columns)
        self.assertEqual(table.loc[0, "catalogiq:Segment:prediction_status"], "unavailable")
        self.assertEqual(table.loc[0, "catalogiq:Segment:review_reasons"], "model_unavailable")
        self.assertEqual(table.loc[0, "catalogiq:registry_version"], "relationships-test")
        self.assertEqual(table.loc[0, "catalogiq:eligibility_version"], tc.ELIGIBILITY_VERSION)

    def test_source_columns_are_copied_byte_for_byte_in_both_tables(self):
        """Padding is not ours to remove. An identifier must survive unchanged."""
        padded = good(JoiningKey=" k0 ", Sku="  S1", ProductUrl="http://x ")
        dropped = good(JoiningKey=" k1 ", Sku=" S2 ")
        f = frame([padded, dropped])
        reasons = tc.eligibility(f, decisions((" k0 ", "candidate", "", ""),
                                              (" k1 ", "quarantine", "structural:shift", "")))
        results = tc.classify(FakeEngine(), tc.to_records(f, reasons), batch_size=10)
        envelope = tc.response(results, registry_version="v")
        kept = tc.dashboard_table(f, results, envelope)
        skipped = tc.ineligible_table(f, reasons)
        self.assertEqual(kept.loc[0, "JoiningKey"], " k0 ")
        self.assertEqual(kept.loc[0, "Sku"], "  S1")
        self.assertEqual(kept.loc[0, "ProductUrl"], "http://x ")
        self.assertEqual(kept.loc[0, "ProductName"], "Claritin RediTabs")
        self.assertEqual(skipped.loc[0, "JoiningKey"], " k1 ")
        self.assertEqual(skipped.loc[0, "Sku"], " S2 ")

    def test_spreadsheet_formulas_are_neutralized_like_the_dashboard_does(self):
        """Three real target values start with + or -, among them the brand +1HEROLABS."""
        f = frame([good(JoiningKey="k0", ProductBrand="+1HEROLABS", Upc="-UPC:767963024671."),
                   good(JoiningKey="k1", ProductBrand="=SUM(A1)")])
        reasons = tc.eligibility(f, decisions(("k0", "candidate", "", ""),
                                              ("k1", "quarantine", "structural:shift", "")))
        results = tc.classify(FakeEngine(), tc.to_records(f, reasons), batch_size=10)
        envelope = tc.response(results, registry_version="v")
        kept = tc.dashboard_table(f, results, envelope)
        skipped = tc.ineligible_table(f, reasons)
        self.assertEqual(kept.loc[0, "ProductBrand"], "'+1HEROLABS")
        self.assertEqual(kept.loc[0, "Upc"], "'-UPC:767963024671.")
        self.assertEqual(skipped.loc[0, "ProductBrand"], "'=SUM(A1)")
        # the audit copy keeps the value untouched
        self.assertEqual(envelope["products"][0]["record_id"], "k0")

    def test_the_dashboard_column_contract_is_complete(self):
        f = frame([good()])
        results = tc.classify(FakeEngine(), tc.to_records(f, tc.eligibility(f, verdicts(f))), batch_size=1)
        envelope = tc.response(results, registry_version="v")
        columns = set(tc.dashboard_table(f, results, envelope).columns)
        for column in tc.REVIEW_COLUMNS:
            self.assertIn(column, columns)
        for field in tc.TARGET_FIELDS:
            self.assertIn(f"catalogiq:Segment:{field}", columns)

    def test_every_source_identifier_reaches_exactly_one_table(self):
        rows = [good(JoiningKey=f" k{i} ") for i in range(4)]
        f = frame(rows)
        reasons = tc.eligibility(f, decisions(
            (" k0 ", "candidate", "", ""), (" k1 ", "quarantine", "structural:shift", ""),
            (" k2 ", "candidate", "", ""), (" k3 ", "review_hold", "", "feature:flag")))
        results = tc.classify(FakeEngine(), tc.to_records(f, reasons), batch_size=2)
        envelope = tc.response(results, registry_version="v")
        kept = set(tc.dashboard_table(f, results, envelope)["JoiningKey"])
        skipped = set(tc.ineligible_table(f, reasons)["JoiningKey"])
        self.assertEqual(kept | skipped, set(f["JoiningKey"]))
        self.assertEqual(kept & skipped, set())

    def test_a_misbehaving_engine_is_caught_not_silently_truncated(self):
        class Short(FakeEngine):
            def predict(self, records):
                return super().predict(records)[:-1]

        class Shuffled(FakeEngine):
            def predict(self, records):
                return list(reversed(super().predict(records)))

        records = tc.to_records(self.rows(3), tc.eligibility(self.rows(3), verdicts(self.rows(3))))
        for engine in (Short(), Shuffled()):
            with self.assertRaises(ValueError):
                tc.classify(engine, records, batch_size=3)

    def test_export_carries_no_prediction_when_a_model_is_unavailable(self):
        f = self.rows(1)
        results = tc.classify(FakeEngine(), tc.to_records(f, tc.eligibility(f, verdicts(f))), batch_size=1)
        envelope = tc.response(results, registry_version="v")
        table = tc.dashboard_table(f, results, envelope)
        self.assertIsNone(table.loc[0, "catalogiq:Segment:original_label"])
        self.assertIsNone(table.loc[0, "catalogiq:Segment:raw_confidence"])


if __name__ == "__main__":
    unittest.main()
