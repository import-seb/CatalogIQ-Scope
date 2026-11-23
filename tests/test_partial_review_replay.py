import unittest

import pandas as pd

from scripts.replay_partial_review import routes_for_review, summarize


class PartialReviewReplayTests(unittest.TestCase):
    def test_routes_use_only_arrival_fields_and_fail_closed(self):
        inputs = pd.DataFrame({
            "ProductBrand": ["A", "new", "A", "mixed", "", "A", "A", "A", "A"],
            "ProductDescription": ["known"] * 7 + ["", "known"],
            "ProductName": ["test"] * 9,
        })
        scores = pd.DataFrame({"score_pass": [True] * 9})
        audit = {
            "Mnfr": scores.copy(), "TargetAgeGroup": scores.copy(),
            "Brand": scores.assign(predicted_label=["A"] * 5 + ["Other Brands"] + ["A"] * 3),
            "Segment": scores.assign(predicted_label="allowed"),
        }
        audit["Mnfr"].loc[6, "score_pass"] = False
        audit["Segment"].loc[8, "predicted_label"] = "outside"
        lookup = {"key_counts": {"a": [[0, 3]], "mixed": [[0, 1], [1, 1]]},
                  "missing_normalized_values": [""]}
        _, two, four = routes_for_review(inputs, audit, lookup, [2], ["allowed"])
        self.assertEqual(inputs.index[two].tolist(), [0, 7, 8])
        self.assertEqual(inputs.index[four].tolist(), [0])

    def test_count_product_once_when_two_hidden_fields_are_wrong(self):
        wrong = pd.DataFrame({"true_label": ["A", "A"], "predicted_label": ["B", "B"]})
        result = summarize(pd.Series([True, False]), ("Brand", "Mnfr"),
                           {"Brand": wrong, "Mnfr": wrong}, pd.Series(["g", "h"]))
        self.assertEqual(result["rows_with_hidden_errors"], 1)
        self.assertEqual(result["field_checks_skipped"], 2)
        self.assertEqual(result["hidden_errors_by_field"], {"Brand": 1, "Mnfr": 1})
