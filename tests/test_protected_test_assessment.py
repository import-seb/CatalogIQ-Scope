"""Synthetic-only preparation/access/metric checks; never open project test.csv."""
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from catalogiq.features import sha256
from catalogiq.prediction_contract import write_json
from catalogiq.protected_test_assessment import (
    FORMAT, LIBRARIES, RUNS, SOURCE, assess, file_pin, freeze_protocol, preflight, score_target,
)
from catalogiq.segment_validation_analysis import _cluster_confusions, _confusion_metrics, bootstrap_predictions
from catalogiq.split_protocol import canonical_digest


class ProtectedAssessmentTests(unittest.TestCase):
    def protocol(self, root):
        directory = root/SOURCE
        directory.mkdir(parents=True)
        registry = pd.DataFrame({"record_id": ["a", "b", "c"], "group_id": ["g", "g", "h"]})
        registry.to_csv(directory/"protected_final_test.csv", index=False)
        # Deliberately not a real CSV. A preparation phase must never inspect it.
        (directory/"test.csv").write_text("MUST NEVER READ THIS DURING PREPARATION")
        p = {"format": FORMAT, "status": "frozen", "test": {"path": f"{SOURCE}/test.csv",
                "identity_registry": f"{SOURCE}/protected_final_test.csv", "records": 3, "groups": 2,
                "identity": {"record_ids_sha256": canonical_digest(registry, ["record_id"]),
                             "assignment_sha256": canonical_digest(registry.assign(split="test"), ["record_id", "group_id", "split"]) }},
             "models": [{"run": name, "target": target, "selection_status": status} for name, (target, status) in RUNS.items()],
             "artifacts": [file_pin(root, directory/"protected_final_test.csv")], "code": [],
             "runtime": {"python": "fixture", "libraries": {name: "fixture" for name in LIBRARIES},
                         "device":"cpu","precision":"fp32","threads":2,"batch_size":64,"seed":42}}
        path = root/"protocol.json"
        write_json(path, p)
        return path, p

    def test_preflight_validates_identity_without_opening_or_hashing_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.protocol(root)
            original_open = Path.open

            def guarded_open(p, *args, **kwargs):
                if p.name == "test.csv":
                    raise AssertionError("Protected test was opened")
                return original_open(p, *args, **kwargs)

            with patch.object(Path, "open", guarded_open), patch("catalogiq.protected_test_assessment.version", return_value="fixture"), patch("catalogiq.protected_test_assessment.platform.python_version", return_value="fixture"):
                _, registry, result = preflight(root, path, sha256(path))
            self.assertEqual(len(registry), 3)
            self.assertFalse(result["test_opened_or_hashed"])
            self.assertFalse(result["models_loaded"])

    def test_protocol_artifact_or_mapping_changes_fail_before_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, p = self.protocol(root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                preflight(root, path, "wrong")
            p["artifacts"].append({"path": f"{SOURCE}/test.csv", "sha256": "irrelevant", "normalization": "raw_bytes"})
            write_json(path, p)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                preflight(root, path, sha256(path))
            p["artifacts"].pop()
            p["models"][0]["target"] = "Platform"
            write_json(path, p)
            with self.assertRaisesRegex(ValueError, "routing"):
                preflight(root, path, sha256(path))

    def test_execution_needs_explicit_authorization_before_preflight(self):
        with patch("catalogiq.protected_test_assessment.preflight") as check:
            for kwargs in ({}, {"authorized": True}, {"authorized": True, "authorization_id": " "}):
                with self.assertRaises(PermissionError):
                    assess("missing", "missing", "missing", "missing", **kwargs)
            check.assert_not_called()

    def test_freeze_refuses_overwrite_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as tmp, patch("catalogiq.protected_test_assessment.frozen_identity_context") as loader:
            root = Path(tmp)
            path = root/"protocol.json"
            path.write_text("original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                freeze_protocol(root, path)
            loader.assert_not_called()
            self.assertEqual(path.read_text(), "original")

    def test_oov_and_zero_support_classes_remain_in_primary_score(self):
        result = score_target(["A", "OOV", "A"], ["A", "A", "B"], [[.8,.2],[.9,.1],[.1,.9]],
            ["A", "B"], ["g", "g", "h"], intervals={"samples":50,"seed":7,"confidence":.95},
            temperature=1., eligible=np.ones(3, bool))
        metrics = result["metrics"]
        self.assertEqual(metrics["oov_truth_rows"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], f1_score(["A","OOV","A"],["A","A","B"],labels=["A","B","OOV"],average="macro",zero_division=0))
        self.assertTrue(metrics["calibration_raw"]["nll_is_infinite"])
        self.assertIsNone(metrics["calibration_raw"]["nll"])
        self.assertNotIn("cluster_accuracy_lower", result["thresholds"])
        self.assertEqual(int(result["per_class"].set_index("label").loc["B","support"]), 0)

    def test_group_bootstrap_matches_previous_dense_draws_with_fixed_class_set(self):
        frame = pd.DataFrame({"record_id":["a","b","c","d"],"group_id":["g","g","h","i"],
                              "true_label":["A","A","B","B"],"predicted_label":["A","B","B","A"]})
        labels = ["A", "B", "zero_support"]
        _, groups, matrices = _cluster_confusions(frame, labels, "group_id")
        rng = np.random.default_rng(31)
        samples=[]
        for _ in range(100):
            metric = _confusion_metrics(matrices[rng.integers(0,len(groups),size=len(groups))].sum(axis=0))[0]
            samples.append([metric["accuracy"], metric["macro_f1"]])
        with patch("catalogiq.segment_validation_analysis._cluster_confusions", side_effect=AssertionError("No dense class cube")):
            result = bootstrap_predictions(frame, labels, n_bootstrap=100, seed=31)
        expected=np.quantile(samples,[.025,.975],axis=0)
        np.testing.assert_allclose(result["intervals"][["lower","upper"]].to_numpy(),expected.T,rtol=0,atol=0)

    def test_failed_authorized_attempt_keeps_receipt_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.protocol(root)
            output = root/"data/processed/attempt"
            protocol = {"runtime":{"threads":1,"seed":42}, "test":{"path":f"{SOURCE}/test.csv","expected_sha256_from_existing_seal":"wrong"}}
            with patch("catalogiq.protected_test_assessment.preflight", return_value=(protocol,None,{})):
                with self.assertRaisesRegex(ValueError,"Protected test bytes"):
                    assess(root,path,"fixture",output,authorized=True,authorization_id="synthetic test only")
            receipt = json.loads((output/"assessment_receipt.json").read_text())
            self.assertEqual(receipt["status"],"failed_preserve_attempt")
            self.assertFalse((output/"completion.json").exists())

    def test_authorized_synthetic_assessment_reconciles_all_rows_and_seals_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/SOURCE
            source.mkdir(parents=True)
            from catalogiq.protected_test_assessment import FIELDS
            frame = pd.DataFrame({"record_id":["a","b","c"],"group_id":["g","g","h"],
                                  "split":"test","Segment":["A","","OOV"],
                                  **{field:["some product"]*3 for field in FIELDS}})
            frame.to_csv(source/"test.csv",index=False)
            for role in ("train","validation"):
                pd.DataFrame({field:["development"] for field in FIELDS}).to_csv(source/f"{role}.csv",index=False)
            protocol={"runtime":{"threads":1,"seed":42},
                      "test":{"path":f"{SOURCE}/test.csv","expected_sha256_from_existing_seal":sha256(source/"test.csv")},
                      "models":[{"run":"segment","target":"Segment","model_identity":"synthetic",
                                 "selection_status":"selected","training_allocation":"synthetic",
                                 "temperature":1.,"label_mapping":{"probability_1":"B","probability_0":"A"}}],
                      "metrics":{"platform":{"status":"deferred"},"relationship_rates":{"valid_pair_rate":None},
                                 "missing_truth":["","null","none","nan"],
                                 "intervals":{"samples":10,"seed":42,"confidence":.95}}}
            fake_torch=SimpleNamespace(set_num_threads=lambda _:None,manual_seed=lambda _:None,use_deterministic_algorithms=lambda _:None)
            output=root/"data/processed/assessment"
            prediction=(np.asarray([[.8,.2],[.3,.7],[.1,.9]]),np.ones(3,bool),{"synthetic":True})
            with patch("catalogiq.protected_test_assessment.preflight",return_value=(protocol,frame[["record_id","group_id"]],{})), patch("catalogiq.protected_test_assessment.predict_frozen",return_value=prediction), patch.dict("sys.modules",{"torch":fake_torch}):
                result=assess(root,root/"protocol.json","fixture",output,authorized=True,authorization_id="synthetic fixture")
            entry=result["models"][0]
            self.assertEqual((entry["predicted_rows"],entry["labeled_rows"],entry["missing_truth_rows"]),(3,2,1))
            self.assertEqual(entry["metrics"]["oov_truth_rows"],1)
            self.assertIsNone(result["relationship_rates"]["valid_pair_rate"])
            saved=pd.read_csv(output/"segment/predictions.csv")
            self.assertEqual(saved.predicted_label.tolist(),["A","B","B"])
            self.assertTrue(saved.review_required.all())
            self.assertEqual(json.loads((output/"assessment_receipt.json").read_text())["status"],"completed")
            self.assertTrue((output/"completion.json").exists())


if __name__ == "__main__":
    unittest.main()
