"""Eligibility is explicit, identifiers survive, and nothing is dropped silently."""
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


class EligibilityTests(unittest.TestCase):
    def reasons(self, rows):
        f = frame(rows)
        return f, tc.eligibility(f)

    def test_a_clean_row_is_eligible(self):
        _, reasons = self.reasons([good()])
        self.assertTrue(reasons.isna().all())

    def test_each_rule_fires_with_its_own_code(self):
        cases = [
            ({"Exclude": "Exclude"}, "marked_exclude"),
            ({"JoiningKey": "   "}, "missing_identifier"),
            ({"ProductName": ""}, "missing_product_name"),
            ({"Retailer": "19 amino acids nourish the skins dermal layers"}, "unknown_retailer"),
        ]
        for overrides, code in cases:
            _, reasons = self.reasons([good(**overrides)])
            self.assertEqual(reasons.iloc[0], code, overrides)

    def test_duplicate_identifiers_are_caught_on_both_rows(self):
        _, reasons = self.reasons([good(JoiningKey="dup"), good(JoiningKey="dup")])
        self.assertEqual(list(reasons), ["duplicate_identifier"] * 2)

    def test_every_reason_code_is_documented(self):
        rows = [good(), good(Exclude="Exclude"), good(JoiningKey=""), good(ProductName=""),
                good(Retailer="nonsense")]
        _, reasons = self.reasons(rows)
        for code in reasons.dropna().unique():
            self.assertIn(code, tc.INELIGIBLE_REASONS)

    def test_nothing_is_dropped_silently(self):
        rows = [good(JoiningKey=f"k{i}") for i in range(5)] + [good(Exclude="Exclude", JoiningKey="x")]
        f, reasons = self.reasons(rows)
        report = tc.eligibility_report(f, reasons)
        self.assertEqual(report["eligible"] + report["ineligible"], report["rows_in_file"])
        self.assertEqual(len(tc.to_records(f, reasons)) + len(tc.ineligible_table(f, reasons)),
                         len(f))


class RecordTests(unittest.TestCase):
    def test_record_id_is_the_source_identifier(self):
        f = frame([good(JoiningKey="JK-7")])
        self.assertEqual([r.record_id for r in tc.to_records(f)], ["JK-7"])

    def test_features_are_exactly_the_model_input_fields(self):
        f = frame([good()])
        self.assertEqual(set(tc.to_records(f)[0].features), set(INPUT_FIELDS))

    def test_blank_feature_values_become_none_not_empty_strings(self):
        f = frame([good(ProductBrand="   ", ProductContents=None)])
        features = tc.to_records(f)[0].features
        self.assertIsNone(features["ProductBrand"])
        self.assertIsNone(features["ProductContents"])
        self.assertEqual(features["ProductName"], "Claritin RediTabs")

    def test_ineligible_rows_never_reach_a_model(self):
        f = frame([good(JoiningKey="keep"), good(JoiningKey="drop", Exclude="Exclude")])
        self.assertEqual([r.record_id for r in tc.to_records(f)], ["keep"])


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
        records = tc.to_records(f)
        engine = FakeEngine()
        results = tc.classify(engine, records, batch_size=3)
        self.assertEqual(engine.batches, [3, 3, 1])
        self.assertEqual([r.record_id for r in results], [r.record_id for r in records])

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            tc.classify(FakeEngine(), tc.to_records(self.rows(1)), batch_size=0)

    def test_export_keeps_source_columns_and_uses_dashboard_naming(self):
        f = self.rows(2)
        results = tc.classify(FakeEngine(), tc.to_records(f), batch_size=10)
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
        dropped = good(JoiningKey=" k1 ", Sku=" S2 ", Exclude="Exclude")
        f = frame([padded, dropped])
        reasons = tc.eligibility(f)
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
                   good(JoiningKey="k1", ProductBrand="=SUM(A1)", Exclude="Exclude")])
        reasons = tc.eligibility(f)
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
        results = tc.classify(FakeEngine(), tc.to_records(f), batch_size=1)
        envelope = tc.response(results, registry_version="v")
        columns = set(tc.dashboard_table(f, results, envelope).columns)
        for column in tc.REVIEW_COLUMNS:
            self.assertIn(column, columns)
        for field in tc.TARGET_FIELDS:
            self.assertIn(f"catalogiq:Segment:{field}", columns)

    def test_every_source_identifier_reaches_exactly_one_table(self):
        rows = [good(JoiningKey=f" k{i} ") for i in range(4)]
        rows[1]["Exclude"] = "Exclude"
        rows[2]["Retailer"] = "shifted text"
        f = frame(rows)
        reasons = tc.eligibility(f)
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

        records = tc.to_records(self.rows(3))
        for engine in (Short(), Shuffled()):
            with self.assertRaises(ValueError):
                tc.classify(engine, records, batch_size=3)

    def test_export_carries_no_prediction_when_a_model_is_unavailable(self):
        f = self.rows(1)
        results = tc.classify(FakeEngine(), tc.to_records(f), batch_size=1)
        envelope = tc.response(results, registry_version="v")
        table = tc.dashboard_table(f, results, envelope)
        self.assertIsNone(table.loc[0, "catalogiq:Segment:original_label"])
        self.assertIsNone(table.loc[0, "catalogiq:Segment:raw_confidence"])


if __name__ == "__main__":
    unittest.main()
