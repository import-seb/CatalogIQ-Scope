"""Synthetic, offline handoff checks; no training, inference or private inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.prediction_contract import (
    CORE_FIELDS, POLICY_FIELDS, REGISTRY_NAME, RUNS, Evidence, allocation_name,
    align_prediction_packages, class_policy, contract_metadata, lookup_configuration,
    ordered_id_hash, validate_manifest, verify_protected_registry, write_identity_registry,
    write_json,
)
from catalogiq.split_protocol import canonical_digest


MODULE = "catalogiq.prediction_contract"


def fixture():
    labels = ["Zeta", "Alpha"]
    predictions = pd.DataFrame({"record_id": ["z", "a"], "true_label": labels,
        "predicted_label": labels, "confidence": [.8, .9],
        "probability_0": [.8, .1], "probability_1": [.2, .9]})
    context = pd.DataFrame({"record_id": ["z", "a"], "group_id": ["gz", "ga"],
        "split": ["validation", "validation"], "true_label": labels})
    protected = pd.DataFrame({"record_id": ["protected"], "group_id": ["gp"]})
    manifest = {"format": "synthetic-predictions", "target": "Segment", "partition": "validation",
        "protocol_id": "fixture-frozen-v4", "model": "synthetic classifier", "config": {"seed": 42},
        "input_fields": ["ProductName"], "label_mapping": {"probability_0": "Zeta", "probability_1": "Alpha"},
        "label_names": labels, "validation_rows": 2, "validation_record_ids_sha256": ordered_id_hash(["z", "a"]),
        "probabilities_calibrated": False, "training_seconds": 1.23, "hardware": None,
        "training_seconds_scope": "Synthetic measured scope", "inference_latency_ms": None,
        "inference_latency_scope": "Single request unmeasured",
        "amortized_inference_latency_ms_per_record": .2,
        "amortized_inference_latency_scope": "Synthetic batch time / rows",
        "split_identity": {"protocol_id": "fixture-frozen-v4", "allocation": "frozen V4; no reallocation"}}
    return manifest, predictions, context, protected


def registry_fixture():
    assignments = pd.DataFrame({"record_id": ["z", "a", "protected"], "group_id": ["gz", "ga", "gp"],
                                "split": ["validation", "train", "test"]})
    registry = assignments.loc[assignments.split.eq("test"), ["record_id", "group_id"]].reset_index(drop=True)
    test = assignments[assignments.split.eq("test")]
    protocol = {"protocol_id": "fixture-frozen-v4", "expected": {
        "assignment_sha256": canonical_digest(assignments, ["record_id", "group_id", "split"]),
        "splits": {"test": {"records": 1, "groups": 1,
            "record_ids_sha256": canonical_digest(test, ["record_id"]),
            "assignment_sha256": canonical_digest(test, ["record_id", "group_id", "split"])}}}}
    return registry, assignments, protocol


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.original, self.predictions, self.context, self.protected = fixture()
        self.manifest = contract_metadata(self.original, self.predictions, self.original["label_names"])

    def validate(self, manifest=None, predictions=None, context=None, protected=None):
        return validate_manifest(self.manifest if manifest is None else manifest,
            self.predictions if predictions is None else predictions,
            self.context if context is None else context,
            self.protected if protected is None else protected)

    def test_required_core_and_policy_fields_fail_closed(self):
        self.assertEqual(len(CORE_FIELDS), 13)
        self.validate()
        for field in (*CORE_FIELDS, *POLICY_FIELDS):
            changed = deepcopy(self.manifest)
            del changed[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Missing prediction contract"):
                self.validate(manifest=changed)

    def test_only_metadata_added_existing_values_and_input_frames_unchanged(self):
        before = deepcopy(self.original)
        before["evaluation"] = {"accuracy": 1., "macro_f1": 1.}
        p = self.predictions.copy(deep=True)
        result = contract_metadata(before, p, before["label_names"])
        for key, value in before.items():
            self.assertEqual(result[key], value)
        pd.testing.assert_frame_equal(p, self.predictions)
        self.assertNotIn("macro_f1_class_set", before)
        self.assertEqual(result["source_hash_normalization"], "raw_bytes")

    def test_lookup_config_and_protocol_come_from_actual_saved_fields(self):
        original = deepcopy(self.original)
        del original["protocol_id"], original["config"]
        model = {"format": "saved-frequency-model", "smoothing": {"alpha": 2., "method": "saved setting"},
            "normalization": "saved Unicode and case normalization", "missing_normalized_values": ["", "null"]}
        original["probability_estimation"] = deepcopy(model["smoothing"])
        result = contract_metadata(original, self.predictions, self.original["label_names"], lookup_model=model)
        self.assertEqual(result["protocol_id"], original["split_identity"]["protocol_id"])
        self.assertEqual(result["config"], lookup_configuration(model))
        self.assertEqual(result["config"]["smoothing"]["alpha"], 2.)
        self.assertNotIn("seed", result["config"])
        self.validate(manifest=result)
        model["smoothing"]["alpha"] = 3
        with self.assertRaisesRegex(ValueError, "smoothing"):
            contract_metadata(original, self.predictions, self.original["label_names"], lookup_model=model)

    def test_no_invented_protocol_or_conflicting_config(self):
        changed = deepcopy(self.original)
        changed.pop("protocol_id"); changed["split_identity"].pop("protocol_id")
        with self.assertRaisesRegex(ValueError, "Actual split protocol"):
            contract_metadata(changed, self.predictions, self.original["label_names"])
        changed = deepcopy(self.original)
        changed["split_identity"]["protocol_id"] = "other"
        with self.assertRaisesRegex(ValueError, "Conflicting split"):
            contract_metadata(changed, self.predictions, self.original["label_names"])

    def test_oov_truth_retained_as_error_and_in_union_macro_denominator(self):
        p = self.predictions.copy()
        p.loc[0, "true_label"] = "Novel truth"
        context = self.context.assign(true_label=p.true_label)
        m = contract_metadata(self.original, p, self.original["label_names"])
        result = self.validate(manifest=m, predictions=p, context=context)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["out_of_vocabulary_true_labels"], 1)
        self.assertEqual(result["macro_f1_classes"], 3)
        self.assertEqual(result["metrics"]["accuracy"], .5)
        self.assertAlmostEqual(result["metrics"]["macro_f1_model_vocabulary_union_truth"], 1/3)
        self.assertEqual(p.record_id.tolist(), ["z", "a"])
        with self.assertRaisesRegex(ValueError, "complete|membership"):
            self.validate(manifest=m, predictions=p.iloc[1:], context=context)

    def test_legacy_nonunion_metric_is_rejected_not_silently_replaced(self):
        p = self.predictions.copy()
        p.loc[0, "true_label"] = "Novel truth"
        m = contract_metadata(self.original, p, self.original["label_names"])
        m["evaluation"] = {"accuracy": .5, "macro_f1": .5}
        with self.assertRaisesRegex(ValueError, "explicit evaluation review"):
            self.validate(manifest=m, predictions=p, context=self.context.assign(true_label=p.true_label))
        self.assertEqual(m["evaluation"]["macro_f1"], .5)

    def test_numeric_143_class_mapping_and_zero_support_classes(self):
        labels = [f"Class {i}" for i in range(143)]
        labels[12] = "Horb\u00e4ach"
        p = pd.DataFrame(np.eye(143)[[12,142]], columns=[f"probability_{i}" for i in range(143)])
        for name, values in reversed(list({"record_id":["z","a"],"true_label":[labels[12],labels[142]],
            "predicted_label":[labels[12],labels[142]],"confidence":[1.,1.]}.items())):
            p.insert(0,name,values)
        original = deepcopy(self.original)
        original["label_names"] = labels
        original["label_mapping"] = {f"probability_{i}":v for i,v in enumerate(labels)}
        m = contract_metadata(original,p,labels)
        result = self.validate(manifest=m,predictions=p,context=self.context.assign(true_label=p.true_label))
        self.assertAlmostEqual(result["metrics"]["macro_f1_model_vocabulary_union_truth"],2/143)

    def test_wrong_class_mapping_probability_schema_and_argmax_rejected(self):
        for defect in ("mapping", "columns", "argmax", "confidence"):
            p, m = self.predictions.copy(), deepcopy(self.manifest)
            if defect=="mapping": m["label_mapping"]["probability_0"]="Alpha"
            elif defect=="columns": p=p.rename(columns={"probability_1":"probability_2"})
            elif defect=="argmax": p.loc[0,"predicted_label"]="Alpha"
            else: p.loc[0,"confidence"] = .5
            with self.subTest(defect=defect),self.assertRaises(ValueError): self.validate(manifest=m,predictions=p)

    def test_csv_order_fingerprint_not_sorted_and_relative_extra_columns_allowed(self):
        self.assertNotEqual(ordered_id_hash(["z","a"]),ordered_id_hash(["a","z"]))
        p=self.predictions.copy()
        p.insert(0,"group_id",self.context.group_id)
        p.insert(1,"split","validation")
        self.validate(predictions=p)
        p.loc[0,"group_id"]="wrong"
        with self.assertRaisesRegex(ValueError,"group_id"): self.validate(predictions=p)
        m=deepcopy(self.manifest);m["validation_record_ids_sha256"]=ordered_id_hash(["a","z"])
        with self.assertRaisesRegex(ValueError,"fingerprint"): self.validate(manifest=m)

    def test_protected_record_and_group_exclusion_independent(self):
        for field,value in (("record_id","z"),("group_id","ga")):
            p=self.protected.copy();p.loc[0,field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"Protected-test"):
                self.validate(protected=p)

    def test_context_truth_membership_and_allocation_cannot_be_substituted(self):
        for field,value in (("true_label","Different"),("record_id","Other"),("split","test")):
            c=self.context.copy();c.loc[0,field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.validate(context=c)
        m=deepcopy(self.manifest);m["split_allocation_target"]="wrong-allocation"
        with self.assertRaisesRegex(ValueError,"allocation"):self.validate(manifest=m)
        m=deepcopy(self.manifest)
        m["split_identity"]["allocation_directory"]=r"C:\private\brand_development_allocation_v1_20261029"
        self.assertEqual(allocation_name(m),"brand_development_allocation_v1_20261029")

    def test_cost_scopes_and_explicit_nulls(self):
        self.assertIsNone(self.manifest["inference_latency_ms"])
        self.assertIsNone(self.manifest["hardware"])
        for key,value in (("training_seconds",-1),("inference_latency_ms",float("nan")),
                          ("amortized_inference_latency_ms_per_record",True),("training_seconds_scope","")):
            m=deepcopy(self.manifest);m[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(manifest=m)
        original=deepcopy(self.original)
        original.pop("amortized_inference_latency_scope")
        original["inference"]={"scope":"Original CPU batch scope"}
        m=contract_metadata(original,self.predictions,original["label_names"])
        self.assertIn("Original CPU batch scope",m["amortized_inference_latency_scope"])

    def test_blank_truth_duplicate_ids_and_invalid_probabilities_fail(self):
        for column,value in (("true_label","null"),("record_id","a"),("probability_0",np.inf),
                             ("probability_0",-.1),("confidence",np.nan)):
            p=self.predictions.copy();p.loc[0,column]=value
            context=self.context.copy()
            if column=="true_label": context.loc[0,"true_label"]=value
            with self.subTest(column=column,value=value),self.assertRaises(ValueError):
                self.validate(predictions=p,context=context)


class ProtectedRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry,self.assignments,self.protocol=registry_fixture()

    def test_membership_and_groups_match_committed_canonical_digests(self):
        result=verify_protected_registry(self.registry,self.assignments,self.protocol)
        self.assertEqual(result["rows"],1)
        self.assertEqual(result["groups"],1)
        for field in ("record_id","group_id"):
            changed=self.registry.copy();changed.loc[0,field]="changed"
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"membership/group"):
                verify_protected_registry(changed,self.assignments,self.protocol)

    def test_resealed_assignment_and_registry_still_fail_original_protocol(self):
        changed=self.assignments.copy();changed.loc[2,"group_id"]="other"
        registry=self.registry.assign(group_id="other")
        with self.assertRaisesRegex(ValueError,"committed original"):
            verify_protected_registry(registry,changed,self.protocol)

    def test_labels_features_duplicates_and_blanks_not_allowed_in_registry(self):
        for changed in (self.registry.assign(Brand="Secret label"),pd.concat([self.registry,self.registry]),
                        self.registry.assign(group_id="")):
            with self.assertRaises(ValueError):verify_protected_registry(changed,self.assignments,self.protocol)

    def test_export_projects_only_two_id_columns_and_uses_lf(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/REGISTRY_NAME
            source=self.registry.assign(split="test",Brand="DO_NOT_EXPORT_LABEL",ProductName="DO_NOT_EXPORT_FEATURE")
            descriptor=write_identity_registry(path,source,self.protocol["protocol_id"])
            body=path.read_bytes()
            self.assertNotIn(b"DO_NOT_EXPORT",body)
            self.assertNotIn(b"\r",body)
            self.assertEqual(pd.read_csv(path).columns.tolist(),["record_id","group_id"])
            self.assertEqual(descriptor["sha256"],hashlib.sha256(body).hexdigest())

    def test_evidence_never_opens_or_hashes_test_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"test.csv"
            with patch(MODULE+".sha256",side_effect=AssertionError("must not read")):
                with self.assertRaises(PermissionError): Evidence().capture(path)

    def test_protected_reader_rejects_labels_before_reading(self):
        with patch(MODULE+".pd.read_csv",side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(PermissionError,"identity columns"):
                Evidence().csv(Path("protected_final_test.csv"),["record_id","Brand"])


class MetadataPackageTests(unittest.TestCase):
    def make_sources(self, root):
        source=root/"original"
        source.mkdir()
        base,p,context,protected=fixture()
        identities={}
        for target in set(RUNS.values()):
            name=("brand-fixture" if target=="Brand" else "sub-segment-fixture" if target=="Sub-Segment" else "fixture-frozen-v4")
            identities[target]={"name":name,"assignments_sha256":"synthetic-source-assignment-digest"}
        for slug,target in RUNS.items():
            directory=source/slug;directory.mkdir()
            m=deepcopy(base);m["target"]=target
            if target in ("Brand","Sub-Segment"):
                m["split_identity"]["allocation_directory"]="C:\\private\\"+identities[target]["name"]
            # Existing CRLF data must not be normalized during a metadata repair.
            p.to_csv(directory/"predictions.csv",index=False,lineterminator="\r\n")
            context.to_csv(directory/"validation_context.csv",index=False,lineterminator="\r\n")
            write_json(directory/"label_mapping.json",m["label_mapping"])
            m["evaluation"]={"accuracy":1.,"macro_f1":1.}
            if slug=="brand_lookup_probabilistic_v1":
                m.pop("protocol_id");m.pop("config");m.pop("amortized_inference_latency_scope")
                model={"format":"saved-model","smoothing":{"alpha":1.,"formula":"saved formula"},
                    "normalization":"saved normalizer","missing_normalized_values":["","null"]}
                write_json(directory/"model.json",model)
                m["probability_estimation"]=model["smoothing"]
                m["inference"]={"scope":"CPU batch assembly"}
            m["files_sha256"]={x.name:sha256(x) for x in directory.iterdir()}
            write_json(directory/"manifest.json",m)
        (source/"max_untouched").mkdir();(source/"max_untouched/model.joblib").write_bytes(b"Do not open Max model")
        protocol={"protocol_id":"fixture-frozen-v4"}
        contexts={target:context for target in set(RUNS.values())}
        frozen=(protocol,protected,contexts,identities,{"rows":1,"groups":1})
        return source,frozen

    def test_metadata_only_publication_is_complete_preserves_crlf_and_ignores_max(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,frozen=self.make_sources(root)
            before={p:p.read_bytes() for p in source.rglob("*") if p.is_file()}
            destination=root/"contract-v2"
            with patch(MODULE+".frozen_identity_context",return_value=frozen):
                result=align_prediction_packages(root,source,destination)
            self.assertEqual(result["runs"],6)
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            self.assertFalse((destination/"max_untouched").exists())
            self.assertTrue(destination.with_suffix(".zip").exists())
            for slug in RUNS:
                manifest=json.loads((destination/slug/"manifest.json").read_text())
                self.assertTrue(set(CORE_FIELDS).issubset(manifest))
                self.assertTrue(set(POLICY_FIELDS).issubset(manifest))
                self.assertEqual((destination/slug/"predictions.csv").read_bytes(),before[source/slug/"predictions.csv"])
                self.assertNotIn(b"\r",(destination/slug/"manifest.json").read_bytes())
                for name,digest in manifest["files_sha256"].items():self.assertEqual(sha256(destination/slug/name),digest)
            with patch(MODULE+".frozen_identity_context",side_effect=AssertionError("no new read")):
                with self.assertRaises(FileExistsError):align_prediction_packages(root,source,destination)

    def test_bad_probabilities_do_not_publish_or_modify_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,frozen=self.make_sources(root)
            p=source/"segment/predictions.csv"
            frame=pd.read_csv(p);frame.loc[0,"confidence"]=.3;frame.to_csv(p,index=False)
            m=json.loads((source/"segment/manifest.json").read_text())
            m["files_sha256"]["predictions.csv"]=sha256(p);write_json(source/"segment/manifest.json",m)
            before={p:p.read_bytes() for p in source.rglob("*") if p.is_file()}
            with patch(MODULE+".frozen_identity_context",return_value=frozen):
                with self.assertRaisesRegex(ValueError,"confidence"):
                    align_prediction_packages(root,source,root/"bad-output")
            self.assertFalse((root/"bad-output").exists())
            self.assertEqual(before,{p:p.read_bytes() for p in before})


if __name__=="__main__":unittest.main()
