"""Metadata-only handoff alignment with Maria's predictions contract.

Reads existing validation exports and ID/group registries. Never trains, loads a
model, runs inference, reads test.csv, or changes an existing prediction package.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .features import sha256
from .split_protocol import canonical_digest


CONTRACT_BLOB = "662bea5144c3fbf13ecf04f7e42c15eac614df44"
CONTRACT_URL = "https://github.com/import-seb/CatalogIQ-Scope/blob/maria/segment-evaluation-harness/docs/predictions-contract.md"
REGISTRY_NAME = "protected_test_identity_registry.csv"
CORE_FIELDS = (
    "format", "target", "partition", "protocol_id", "model", "config", "input_fields",
    "label_mapping", "validation_rows", "validation_record_ids_sha256",
    "probabilities_calibrated", "training_seconds", "hardware",
)
POLICY_FIELDS = (
    "macro_f1_class_set", "out_of_vocabulary_true_labels", "out_of_vocabulary_policy",
    "split_allocation_target", "source_hash_normalization", "training_seconds_scope",
    "inference_latency_ms", "inference_latency_scope",
    "amortized_inference_latency_ms_per_record", "amortized_inference_latency_scope",
)
IDENTITY = ["record_id", "group_id", "split"]
MISSING = frozenset(("", "null", "none", "nan"))
RUNS = {
    "segment": "Segment", "mnfr": "Mnfr", "targetagegroup": "TargetAgeGroup",
    "brand": "Brand", "sub_segment": "Sub-Segment",
    "brand_lookup_probabilistic_v1": "Brand",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                          encoding="utf8", newline="\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def ordered_id_hash(ids):
    result = hashlib.sha256()
    for value in ids:
        payload = str(value).encode("utf8")
        result.update(len(payload).to_bytes(8, "big"))
        result.update(payload)
    return result.hexdigest()


def class_policy(predictions, labels):
    """Keep every truth row; an unknown true class expands the F1 denominator."""
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(v, str) or not v.strip() for v in labels):
        raise ValueError("Model vocabulary must contain unique nonblank class names")
    truth, predicted = predictions.true_label, predictions.predicted_label
    if truth.isna().any() or truth.astype(str).str.strip().str.casefold().isin(MISSING).any():
        raise ValueError("Prediction population must contain labeled records")
    if not predicted.isin(labels).all():
        raise ValueError("Predicted labels must belong to the model vocabulary")
    union = [*labels, *sorted(set(truth) - set(labels))]
    return {
        "macro_f1_class_set": "model_vocabulary_union_truth",
        "out_of_vocabulary_true_labels": int((~truth.isin(labels)).sum()),
        "out_of_vocabulary_policy": "retained_as_errors",
        "macro_f1_labels": union,
    }


def lookup_configuration(model):
    """Use saved estimator settings, not transformer settings or new defaults."""
    return {
        "estimator": model["format"], "smoothing": deepcopy(model["smoothing"]),
        "normalization": model["normalization"],
        "missing_normalized_values": list(model["missing_normalized_values"]),
        "argmax_tie_break": "First maximum in the saved label order (numpy.argmax)",
    }


def allocation_name(manifest):
    identity = manifest["split_identity"]
    if identity.get("allocation_directory"):
        # Source manifests contain Windows provenance paths, even on POSIX hosts.
        return identity["allocation_directory"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return manifest.get("protocol_id") or identity["protocol_id"]


def contract_metadata(manifest, predictions, labels, *, lookup_model=None):
    """Add only missing declarations; preserve every pre-existing score and cost."""
    result = deepcopy(manifest)
    protocol = result.get("protocol_id") or result["split_identity"].get("protocol_id")
    if not protocol:
        raise ValueError("Actual split protocol_id is required")
    if result["split_identity"].get("protocol_id", protocol) != protocol:
        raise ValueError("Conflicting split protocol identities")
    result["protocol_id"] = protocol
    if lookup_model is not None:
        actual = lookup_configuration(lookup_model)
        if "config" in result and result["config"] != actual:
            raise ValueError("Lookup config differs from the saved estimator settings")
        if result.get("probability_estimation") != lookup_model["smoothing"]:
            raise ValueError("Lookup smoothing differs from its saved model")
        result["config"] = actual
    additions = class_policy(predictions, labels)
    additions.pop("macro_f1_labels")  # Derivable; preserve the existing saved model mapping.
    additions.update({
        "split_allocation_target": allocation_name(result),
        "source_hash_normalization": "raw_bytes",
    })
    for name, value in additions.items():
        if name in result and result[name] != value:
            raise ValueError(f"Existing contract declaration disagrees: {name}")
        result[name] = value
    result["source_hash_normalization_scope"] = (
        "source_input_sha256 and files_sha256 use raw file bytes, without line-ending normalization. "
        "Record-ID and frozen-assignment fingerprints retain their separately declared canonical recipes. "
        "Historical frozen algorithm seals use their original LF-normalized recipe, not raw file hashes."
    )
    result.setdefault("inference_latency_ms", None)
    result.setdefault("inference_latency_scope", "Single-request latency unmeasured")
    result.setdefault("amortized_inference_latency_ms_per_record", None)
    if "amortized_inference_latency_scope" not in result:
        scope = result.get("inference", {}).get("scope")
        if result["amortized_inference_latency_ms_per_record"] is not None and not scope:
            raise ValueError("Measured amortized cost needs its original measurement scope")
        result["amortized_inference_latency_scope"] = (
            "Recorded batched inference seconds divided by validation rows; " + scope
            if scope else "Amortized batch inference cost unmeasured"
        )
    return result


def check_identities(frame, *, require_split=False):
    columns = IDENTITY if require_split else IDENTITY[:2]
    if not set(columns).issubset(frame) or frame.empty:
        raise ValueError("Identity records/groups must be present and nonempty")
    if frame[columns].isna().any().any() or frame[columns].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Identity values must be nonblank")
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate record_id in identity registry")
    if require_split and (not frame.split.isin(["train", "validation", "test"]).all()
                          or frame.groupby("group_id").split.nunique().gt(1).any()):
        raise ValueError("Invalid roles or product groups crossing partitions")


def verify_protected_registry(registry, frozen_assignments, protocol):
    """Bind both membership and grouping to the committed original V4 digests."""
    if list(registry) != IDENTITY[:2]:
        raise ValueError("Protected registry must contain record_id, group_id only")
    check_identities(registry)
    check_identities(frozen_assignments, require_split=True)
    expected = protocol["expected"]
    if canonical_digest(frozen_assignments, IDENTITY) != expected["assignment_sha256"]:
        raise ValueError("Frozen assignments differ from the committed original protocol")
    original = frozen_assignments.loc[frozen_assignments.split.eq("test"), IDENTITY[:2]]
    left = registry.sort_values("record_id").reset_index(drop=True)
    right = original.sort_values("record_id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Protected registry membership/group mapping differs from original V4")
    test = expected["splits"]["test"]
    reconstructed = registry.assign(split="test")
    if (len(registry) != test["records"] or registry.group_id.nunique() != test["groups"]
            or canonical_digest(registry, ["record_id"]) != test["record_ids_sha256"]
            or canonical_digest(reconstructed, IDENTITY) != test["assignment_sha256"]):
        raise ValueError("Protected identity digest differs from the original frozen V4 test")
    return {"rows": len(registry), "groups": int(registry.group_id.nunique()),
            "canonical_record_ids_sha256": test["record_ids_sha256"],
            "canonical_assignment_sha256": test["assignment_sha256"]}


def write_identity_registry(path, protected, protocol_id):
    registry = protected[IDENTITY[:2]].copy()
    check_identities(registry)
    registry.to_csv(path, index=False, lineterminator="\n")
    return {"path": Path(path).name, "sha256": sha256(path), "columns": IDENTITY[:2],
            "rows": len(registry), "groups": int(registry.group_id.nunique()),
            "protocol_id": protocol_id, "contents": "Frozen protected test identities only; no labels or product features"}


def validate_manifest(manifest, predictions, context, protected):
    """Validate Sections 2/3 without filtering, sorting or rewriting predictions."""
    missing = [k for k in (*CORE_FIELDS, *POLICY_FIELDS) if k not in manifest]
    if missing:
        raise ValueError(f"Missing prediction contract fields: {', '.join(missing)}")
    if any(not isinstance(manifest[k], str) or not manifest[k].strip() for k in ("format", "target", "model", "protocol_id")):
        raise ValueError("Format, target, model and protocol must be nonblank strings")
    if isinstance(manifest["validation_rows"], bool) or not isinstance(manifest["validation_rows"], int) or manifest["validation_rows"] < 1:
        raise ValueError("Validation row count must be a positive integer")
    if manifest["partition"] != "validation" or not isinstance(manifest["config"], dict):
        raise ValueError("Validation partition and config object required")
    if not isinstance(manifest["probabilities_calibrated"], bool):
        raise ValueError("Calibration declaration must be boolean")
    if not manifest["protocol_id"] or manifest["protocol_id"] != manifest["split_identity"]["protocol_id"]:
        raise ValueError("Split protocol identity mismatch")
    mapping = manifest["label_mapping"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Explicit probability-to-class mapping required")
    columns = [f"probability_{i}" for i in range(len(mapping))]
    if set(mapping) != set(columns):
        raise ValueError("Class mapping must use contiguous numeric probability columns")
    labels = [mapping[c] for c in columns]
    if manifest.get("label_names", labels) != labels:
        raise ValueError("Saved class order disagrees with explicit mapping")
    required = ["record_id", "true_label", "predicted_label", "confidence", *columns]
    if not set(required).issubset(predictions) or [c for c in predictions if c in required] != required:
        raise ValueError("Required prediction columns must retain their relative order")
    actual_probabilities = [c for c in predictions if c.startswith("probability_")]
    if actual_probabilities != columns:
        raise ValueError("Prediction probability columns differ from class mapping")
    if len(predictions) != manifest["validation_rows"] or predictions.record_id.duplicated().any() or predictions.empty:
        raise ValueError("Prediction rows must be complete and unique")
    if not set([*IDENTITY, "true_label"]).issubset(context):
        raise ValueError("Context requires record_id, group_id, split, true_label")
    check_identities(context, require_split=True)
    check_identities(protected)
    if not context.split.eq("validation").all() or set(context.record_id) != set(predictions.record_id):
        raise ValueError("Exact validation context membership required")
    if set(context.record_id) & set(protected.record_id) or set(context.group_id) & set(protected.group_id):
        raise ValueError("Protected-test record/group in validation predictions")
    expected = context.set_index("record_id").loc[predictions.record_id]
    if predictions.true_label.tolist() != expected.true_label.tolist():
        raise ValueError("Prediction truth differs from declared validation context")
    if "group_id" in predictions and predictions.group_id.tolist() != expected.group_id.tolist():
        raise ValueError("Prediction group_id differs from validation context")
    if "split" in predictions and not predictions.split.eq("validation").all():
        raise ValueError("Prediction split must be validation")
    if ordered_id_hash(predictions.record_id) != manifest["validation_record_ids_sha256"]:
        raise ValueError("CSV-order record ID fingerprint mismatch")
    policy = class_policy(predictions, labels)
    for key in ("macro_f1_class_set", "out_of_vocabulary_true_labels", "out_of_vocabulary_policy"):
        if manifest[key] != policy[key]:
            raise ValueError(f"Class policy mismatch: {key}")
    if manifest["split_allocation_target"] != allocation_name(manifest):
        raise ValueError("Target-specific allocation declaration mismatch")
    if manifest["source_hash_normalization"] != "raw_bytes":
        raise ValueError("These source file hashes use raw_bytes")
    matrix = predictions[columns].to_numpy(float)
    if (not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid prediction probabilities")
    if not np.array_equal(np.asarray(labels)[matrix.argmax(axis=1)], predictions.predicted_label):
        raise ValueError("Probability class mapping/argmax disagreement")
    if not np.allclose(matrix.max(axis=1), predictions.confidence.to_numpy(float), atol=1e-6, rtol=0):
        raise ValueError("Prediction confidence must be maximum probability")
    for key in ("training_seconds", "inference_latency_ms", "amortized_inference_latency_ms_per_record"):
        value = manifest[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid cost measurement: {key}")
    for key in ("training_seconds_scope", "inference_latency_scope", "amortized_inference_latency_scope"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"Cost measurement scope required: {key}")
    metrics = {
        "accuracy": float(accuracy_score(predictions.true_label, predictions.predicted_label)),
        "macro_f1_model_vocabulary_union_truth": float(f1_score(predictions.true_label, predictions.predicted_label,
                                                                labels=policy["macro_f1_labels"], average="macro", zero_division=0)),
    }
    original = manifest.get("evaluation") or manifest.get("harness_compatibility", {}).get("metrics", {})
    for key, value in original.items():
        comparison = metrics["accuracy"] if key == "accuracy" else (
            metrics["macro_f1_model_vocabulary_union_truth"] if key in ("macro_f1", "macro_f1_full_vocabulary") else None)
        if comparison is not None and not np.isclose(value, comparison, atol=1e-10, rtol=0):
            raise ValueError("Existing reported metric disagrees with contract class set; explicit evaluation review required")
    return {"rows": len(predictions), "model_classes": len(labels), "macro_f1_classes": len(policy["macro_f1_labels"]),
            "out_of_vocabulary_true_labels": policy["out_of_vocabulary_true_labels"],
            "protected_record_overlap": 0, "protected_group_overlap": 0, "metrics": metrics}


class Evidence:
    """Snapshot only explicitly selected files; never open or hash test.csv."""
    def __init__(self):
        self.hashes = {}

    def capture(self, path):
        path = Path(path).resolve()
        if path.name.casefold() == "test.csv":
            raise PermissionError("Protected test.csv must not be opened or hashed")
        digest = sha256(path)
        if path in self.hashes and self.hashes[path] != digest:
            raise ValueError(f"Source changed during contract alignment: {path}")
        self.hashes[path] = digest
        return path

    def json(self, path):
        return read_json(self.capture(path))

    def csv(self, path, columns):
        # Identity sources always use a narrow projection. No protected targets/features.
        if Path(path).name in ("protected_final_test.csv", REGISTRY_NAME) and not set(columns).issubset(IDENTITY):
            raise PermissionError("Protected metadata permits identity columns only")
        return pd.read_csv(self.capture(path), dtype=str, keep_default_na=False,
                           encoding="utf-8-sig", usecols=columns)[columns]

    def reverify(self):
        for path, expected in list(self.hashes.items()):
            if sha256(path) != expected:
                raise ValueError(f"Consumed source changed: {path}")


def _sealed_csv(evidence, directory, seal, name, columns):
    path = evidence.capture(directory / name)
    if evidence.hashes[path] != seal["artifact_sha256"].get(name):
        raise ValueError(f"Source seal mismatch: {directory.name}/{name}")
    return evidence.csv(path, columns)


def frozen_identity_context(root, evidence):
    """Verify existing original/portable registries and approved allocations only."""
    from .minilm_prediction_export import ALLOCATIONS

    source = root / "data/processed/split_integration_reproduced_20261010"
    protocol = evidence.json(Path(__file__).parent / "resources/segment_final_20261009/manifest.json")
    seal = evidence.json(source / "completion.json")
    local_protocol = evidence.json(source / "protocol_manifest.json")
    if local_protocol != protocol or sha256(source / "protocol_manifest.json") != seal["protocol_manifest_sha256"]:
        raise ValueError("Source split protocol differs from committed frozen V4")
    assignments = _sealed_csv(evidence, source, seal, "assignments.csv", IDENTITY)
    original_registry = _sealed_csv(evidence, source, seal, "protected_final_test.csv", IDENTITY[:2])
    registry_checks = verify_protected_registry(original_registry, assignments, protocol)
    historical = evidence.csv(root / "data/processed/split_finalization_20261009/frozen/protected_final_test.csv", IDENTITY[:2])
    verify_protected_registry(historical, assignments, protocol)
    validation = _sealed_csv(evidence, source, seal, "validation.csv", [*IDENTITY, "Segment", "Mnfr", "TargetAgeGroup"])
    expected_validation = assignments.loc[assignments.split.eq("validation"), IDENTITY].sort_values("record_id").reset_index(drop=True)
    if not validation[IDENTITY].sort_values("record_id").reset_index(drop=True).equals(expected_validation):
        raise ValueError("Frozen validation identity mismatch")
    contexts, identities = {}, {}
    for target in ("Segment", "Mnfr", "TargetAgeGroup"):
        labeled = ~validation[target].str.strip().str.casefold().isin(MISSING)
        contexts[target] = validation.loc[labeled, [*IDENTITY, target]].rename(columns={target: "true_label"})
        identities[target] = {
            "name": protocol["protocol_id"], "assignments_sha256": evidence.hashes[(source / "assignments.csv").resolve()],
            "completion_sha256": evidence.hashes[(source / "completion.json").resolve()],
        }
    for target, (name, completion_hash) in ALLOCATIONS.items():
        directory = root / "data/processed" / name
        completion = evidence.capture(directory / "completion.json")
        if evidence.hashes[completion] != completion_hash:
            raise ValueError("Allocation completion differs from the approved identity")
        allocation_seal = evidence.json(completion)
        allocation = _sealed_csv(evidence, directory, allocation_seal, "assignments.csv", IDENTITY)
        check_identities(allocation, require_split=True)
        if set(allocation.record_id) != set(assignments.record_id):
            raise ValueError("Allocation changed the frozen record population")
        original = assignments.set_index("record_id").loc[allocation.record_id]
        if allocation.group_id.tolist() != original.group_id.tolist():
            raise ValueError("Allocation changed frozen product group IDs")
        reserved = _sealed_csv(evidence, directory, allocation_seal, "protected_final_test.csv", IDENTITY[:2])
        verify_protected_registry(reserved, assignments, protocol)
        allocation_test = allocation.loc[allocation.split.eq("test"), IDENTITY[:2]]
        verify_protected_registry(allocation_test, assignments, protocol)
        metadata = _sealed_csv(evidence, directory, allocation_seal, "development_assignments.csv",
                               [*IDENTITY, target, "is_labeled"])
        check_identities(metadata, require_split=True)
        if (not metadata.split.isin(["train", "validation"]).all()
                or set(metadata.record_id) != set(assignments.loc[assignments.split.ne("test"), "record_id"])):
            raise ValueError("Allocation development metadata changed membership")
        assigned = allocation.set_index("record_id").loc[metadata.record_id]
        if not np.array_equal(metadata[IDENTITY[1:]].to_numpy(), assigned[IDENTITY[1:]].to_numpy()):
            raise ValueError("Allocation role/group metadata disagrees")
        labeled = ~metadata[target].str.strip().str.casefold().isin(MISSING)
        if metadata.is_labeled.tolist() != labeled.map({True: "True", False: "False"}).tolist():
            raise ValueError("Allocation supervision eligibility changed")
        contexts[target] = metadata.loc[labeled & metadata.split.eq("validation"), [*IDENTITY, target]].rename(columns={target: "true_label"})
        allocation_protocol = evidence.json(directory / "protocol.json")
        if sha256(directory / "protocol.json") != allocation_seal["artifact_sha256"]["protocol.json"]:
            raise ValueError("Allocation protocol seal mismatch")
        if allocation_protocol["preparation"]["completion_sha256"] != evidence.hashes[(source / "completion.json").resolve()]:
            raise ValueError("Allocation source split seal mismatch")
        identities[target] = {"name": name, "completion_sha256": completion_hash,
            "assignments_sha256": evidence.hashes[(directory / "assignments.csv").resolve()],
            "development_assignments_sha256": evidence.hashes[(directory / "development_assignments.csv").resolve()],
            "protocol_sha256": evidence.hashes[(directory / "protocol.json").resolve()]}
    return protocol, original_registry, contexts, identities, registry_checks


def _safe_member(directory, name):
    result = (directory / name).resolve()
    if Path(name).is_absolute() or not result.is_relative_to(directory.resolve()) or result.name.casefold() == "test.csv":
        raise ValueError("Unsafe source artifact path")
    return result


def package_readme(target, allocation):
    return f"""# Predictions contract alignment: {target}

This is a metadata-only revision of an existing prediction export. Predictions,
validation context, class mapping, probabilities, metrics and measured costs
were preserved. No model training, inference, test scoring or label access ran.

Use this directory's manifest.json and validation_context.csv together.
Exact allocation: {allocation}. The manifest also pins its assignment/seal hashes.
The local {REGISTRY_NAME} contains record_id and group_id only and is checked
against both the committed V4 membership/group digests and original registry.
Use those protected IDs AND groups in evaluation exclusion guards.

Headline macro-F1 class set: model_vocabulary_union_truth. Unknown true labels
stay in the population as errors and enter the macro denominator. Saved observed-
label scores remain observed-label scores. The manifest records the exact OOV
truth-row count; when it is zero, the full-vocabulary and union scores agree.
Older compatibility reports remain historical reports about
the pinned harness; they do not claim a new final comparative evaluation.

Source/file hashes use raw_bytes. Original CSV/context files, including any
CRLF, remain byte-for-byte unchanged. New metadata and ID registry files use LF.
Historical canonical membership and LF-normalized algorithm seals keep their
original, separately described hash recipes. Absolute paths are provenance.

Single-request latency is null because it was not measured. Amortized batch
time has its own value and scope. CPU lookup and GPU MiniLM costs do not form
a controlled device comparison. No calibration or review threshold is implied.

Maria: evaluate one target context at a time, preserve zero-support classes,
compare an arriving group_id with the registry instead of overwriting it,
and handle OOV truth explicitly in Brier/calibration code. The harness itself
was not modified. Max's TF-IDF exports are outside this package's scope.
"""


def align_prediction_packages(root, source_root, output_root):
    """Publish fresh, self-contained packages; never mutate existing artifacts."""
    root, source_root, output_root = map(lambda p: Path(p).resolve(), (root, source_root, output_root))
    if output_root.exists() or output_root.with_suffix(".zip").exists():
        raise FileExistsError("Use a new contract handoff output directory/archive")
    if (output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root)
            or output_root.is_relative_to(root / "src") or output_root.is_relative_to(root / "tests")
            or output_root.is_relative_to(root / "notebooks")
            or any(output_root.is_relative_to(root / "data/processed" / name) for name in (
                "split_integration_reproduced_20261010", "split_finalization_20261009",
                "brand_development_allocation_v1_20261029", "sub_segment_development_allocation_v1_20261029_r2"))):
        raise ValueError("Contract handoff must be separate from existing sources and frozen artifacts")
    evidence = Evidence()
    protocol, protected, contexts, identities, registry_checks = frozen_identity_context(root, evidence)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prediction-contract-", dir=output_root.parent))
    report = {"status": "passed", "contract_git_blob": CONTRACT_BLOB, "contract_url": CONTRACT_URL,
        "protocol_id": protocol["protocol_id"], "protected_registry": registry_checks,
        "protected_test_labels_read": False, "test_csv_opened_or_hashed": False,
        "training_or_inference_performed": False, "original_exports_modified": False, "runs": {}}
    try:
        write_identity_registry(stage / REGISTRY_NAME, protected, protocol["protocol_id"])
        for slug, target in RUNS.items():
            source = source_root / slug
            original = evidence.json(source / "manifest.json")
            if original["target"] != target:
                raise ValueError("Prediction directory target mismatch")
            destination = stage / slug
            destination.mkdir()
            for name, expected in original["files_sha256"].items():
                member = evidence.capture(_safe_member(source, name))
                if evidence.hashes[member] != expected:
                    raise ValueError(f"Existing source artifact seal mismatch: {slug}/{name}")
                shutil.copyfile(member, destination / name)
            predictions = pd.read_csv(evidence.capture(source / "predictions.csv"), dtype={
                "record_id": str, "true_label": str, "predicted_label": str}, keep_default_na=False)
            context = evidence.csv(source / "validation_context.csv", [*IDENTITY, "true_label"])
            expected = contexts[target].set_index("record_id").sort_index()
            actual = context.set_index("record_id").sort_index()
            if not actual.equals(expected):
                raise ValueError(f"{slug}: Context differs from the exact approved allocation/truth")
            model = evidence.json(source / "model.json") if slug == "brand_lookup_probabilistic_v1" else None
            labels = [original["label_mapping"][f"probability_{i}"] for i in range(len(original["label_mapping"]))]
            manifest = contract_metadata(original, predictions, labels, lookup_model=model)
            if manifest["protocol_id"] != protocol["protocol_id"] or manifest["split_allocation_target"] != identities[target]["name"]:
                raise ValueError("Manifest allocation differs from the approved source")
            mapping = evidence.json(source / "label_mapping.json")
            if mapping != manifest["label_mapping"]:
                raise ValueError("External label mapping disagrees with manifest")
            manifest["split_allocation_identity"] = identities[target]
            manifest["protected_test_registry"] = write_identity_registry(destination / REGISTRY_NAME, protected, protocol["protocol_id"])
            manifest["contract_alignment"] = {"contract_git_blob": CONTRACT_BLOB,
                "previous_manifest_sha256": evidence.hashes[(source / "manifest.json").resolve()],
                "source_directory_provenance": str(source), "predictions_and_context_unchanged": True,
                "reported_metrics_and_costs_unchanged": True, "metadata_only": True}
            for key, value in original.items():
                if key != "files_sha256" and manifest[key] != value:
                    raise ValueError(f"Existing manifest value changed: {key}")
            checks = validate_manifest(manifest, predictions, context, protected)
            (destination / "CONTRACT_README.md").write_text(package_readme(target, identities[target]["name"]), encoding="utf8", newline="\n")
            manifest["files_sha256"] = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
            write_json(destination / "manifest.json", manifest)
            restored = read_json(destination / "manifest.json")
            validate_manifest(restored, predictions, context, protected)
            for name, expected_hash in original["files_sha256"].items():
                if sha256(destination / name) != expected_hash:
                    raise ValueError("Existing prediction/context/metric artifacts must remain byte-identical")
            report["runs"][slug] = {"target": target, "checks": checks,
                "already_satisfied_core_fields": [k for k in CORE_FIELDS if k in original],
                "added_core_fields": [k for k in CORE_FIELDS if k not in original],
                "added_policy_fields": [k for k in POLICY_FIELDS if k not in original],
                "source_manifest_sha256": evidence.hashes[(source / "manifest.json").resolve()],
                "manifest_sha256": sha256(destination / "manifest.json"),
                "predictions_sha256": sha256(destination / "predictions.csv"),
                "predictions_unchanged": True, "validation_context_unchanged": True,
                "split_allocation_identity": identities[target]}
        evidence.reverify()
        report["consumed_source_files_unchanged"] = len(evidence.hashes)
        report["source_sha256"] = {str(path): value for path, value in sorted(evidence.hashes.items())}
        write_json(stage / "verification_report.json", report)
        (stage / "README.md").write_text(
            "# CatalogIQ predictions contract handoff\n\n"
            f"Six owned exports, frozen V4 protocol `{protocol['protocol_id']}`.\n"
            f"The root and per-run `{REGISTRY_NAME}` contain protected IDs/groups only.\n"
            "Use each run's manifest, context and registry together. Original exports,\n"
            "predictions, metrics, allocations and all Max/Maria files are unchanged.\n"
            "See per-run CONTRACT_README.md and verification_report.json for scopes.\n"
            "Maria's final comparative evaluation remains open.\n", encoding="utf8", newline="\n")
        write_json(stage / "completion.json", {"format": "catalogiq-prediction-contract-handoff-v1",
            "contract_git_blob": CONTRACT_BLOB,
            "files_sha256": {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()}})
        stage.rename(output_root)
    except Exception:
        if stage.parent.resolve() != output_root.parent or not stage.name.startswith(".prediction-contract-"):
            raise RuntimeError("Refusing cleanup outside the contract staging directory")
        if stage.exists():
            shutil.rmtree(stage)
        raise
    archive = output_root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(output_root).as_posix())
    return {"directory": str(output_root), "archive": str(archive), "archive_sha256": sha256(archive),
            "runs": len(RUNS), "protected_records": len(protected), "status": "passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("data/processed/minilm_validation_exports_v1"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(align_prediction_packages(args.root, args.source_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
