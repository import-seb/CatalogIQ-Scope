"""Checkpoint-based validation exports for all five selected CatalogIQ targets.

No training, pretrained-weight downloads or protected-test label/feature access.
The frozen formatter, tokenizer, classifier and input guards are reused intact.
"""
from dataclasses import dataclass, replace
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from .features import sha256
from .maria_harness_compatibility import load_with_maria_harness
from .segment_category_ablation import load_category_development, load_tokenizer, tokenize_category_corpus
from .segment_transformer import ModelConfig, _hash_items, _initialize_torch, _make_classifier, _state_sha256
from .target_minilm_preparation import INPUT_FIELDS, prepare_target_minilm


TARGETS = ("Segment", "Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")
ALLOCATIONS = {
    "Brand": ("brand_development_allocation_v1_20261029", "94c64e5bd80219595f3c8764acd67a5d64496a63164276807d42b14d56624c6e"),
    "Sub-Segment": ("sub_segment_development_allocation_v1_20261029_r2", "26f52bd5360af96d0271c998f13c846dfeaa325bf0fb90b4c85011df5cabdc6a"),
}
BASE_COLUMNS = ["record_id", "true_label", "predicted_label", "confidence"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    # Maria's loader uses Path.read_text() without an encoding. Escaped JSON
    # preserves accented class names on both UTF-8 and Windows CP1252 hosts.
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def slug(target):
    return target.casefold().replace("-", "_")


def normalize_json(value):
    return json.loads(json.dumps(value))


@dataclass
class ExportInputs:
    target: str
    directory: Path
    metadata: dict
    config: ModelConfig
    labels: tuple
    frame: pd.DataFrame
    train_indices: np.ndarray
    validation_indices: np.ndarray
    assignments: pd.DataFrame
    protected: pd.DataFrame
    snapshots: dict
    model_identity: str
    split_identity: dict
    reference_predictions: Path | None

    @property
    def validation(self):
        part = self.frame.iloc[self.validation_indices]
        return part[["record_id", "group_id", "split", self.target]].rename(columns={self.target: "true_label"}).copy()


def selected_source(root, target):
    """Resolve only the reviewed selections; never pick the latest checkpoint."""
    root = Path(root)
    if target == "Segment":
        experiment = root / "data/processed/segment_category_ablation_256_20261010"
        directory = experiment / "with_category"
        seal = read_json(experiment / "completion.json")
        metadata = read_json(directory / "training_manifest.json")
        metadata["representation"] = read_json(experiment / "protocol.json")["representation"]
        files = {key.removeprefix("with_category/"): digest for key, digest in seal["artifact_sha256"].items() if key.startswith("with_category/")}
        return directory, metadata, files, directory / "predictions.csv", "segment_category_ablation_256_20261010/with_category"
    if target == "Mnfr":
        batch = root / "data/processed/mnfr_minilm_selection_v1_20261029"
        selection = read_json(batch / "selection.json")
        if selection["selected_candidate"] != "unweighted":
            raise ValueError("Mnfr selection changed; review its checkpoint explicitly")
        directory = batch / "maria_handoff"
        metadata = read_json(directory / "manifest.json")
        files = metadata["artifact_sha256"]
        original = read_json(directory / "training_manifest.json")
        for key in ("target", "config", "label_names", "input_fields", "model_identity", "final_state_sha256"):
            if key in metadata and normalize_json(metadata[key]) != normalize_json(original[key]):
                raise ValueError(f"Mnfr handoff/training metadata disagree: {key}")
        metadata = {**original, **metadata}
        delivery = read_json(batch / "handoff_delivery.json")
        if files["checkpoint.pt"] != delivery["bundle_checkpoint_sha256"]:
            raise ValueError("Selected Mnfr checkpoint seal differs")
        reference = Path(selection["candidates"]["unweighted"]["output_directory"]) / "predictions.csv"
    elif target in ("TargetAgeGroup", "Brand", "Sub-Segment"):
        batch = root / "data/processed/minilm_remaining_targets_v1_20261029"
        result = read_json(batch / f"{slug(target)}_result.json")
        if result["status"] != "completed_and_verified":
            raise ValueError("Selected target training did not complete verification")
        directory = batch / "handoff" / slug(target)
        metadata = read_json(directory / "manifest.json")
        files = metadata["artifact_sha256"]
        if files["checkpoint.pt"] != result["handoff"]["packaged_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint seal differs")
        reference = Path(result["output_directory"]) / "predictions.csv"
    else:
        raise ValueError(f"Unsupported MiniLM target: {target}; Platform is excluded")
    return directory, metadata, files, reference, metadata["model_identity"]


def prepare_export(root, target):
    root = Path(root).resolve()
    directory, metadata, files, reference, identity = selected_source(root, target)
    snapshots = {}
    for name, expected in files.items():
        path = directory / name
        if sha256(path) != expected:
            raise ValueError(f"Selected model artifact changed: {path}")
        snapshots[path] = expected
    meta_name = "training_manifest.json" if target == "Segment" else "manifest.json"
    snapshots[directory / meta_name] = sha256(directory / meta_name)
    if target == "Segment":
        for name in ("protocol.json", "completion.json"):
            path = directory.parent / name
            snapshots[path] = sha256(path)
    if metadata["target"] != target or metadata["input_fields"] != list(INPUT_FIELDS):
        raise ValueError("Selected model target or five-field representation differs")
    config = ModelConfig.from_dict(metadata["config"])
    labels = tuple(metadata["label_names"])
    if config.sequence_length != 256 or config.precision != "fp32" or not config.local_files_only or not labels or len(set(labels)) != len(labels):
        raise ValueError("Selected model configuration/vocabulary is invalid")
    source = root / "data/processed/split_integration_reproduced_20261010"
    allocation = None
    if target == "Segment":
        development = load_category_development(source, config)
        frame, train, validation = development.frame, development.train_indices, development.validation_indices
        if development.label_names != labels:
            raise ValueError("Segment checkpoint label order differs from its approved vocabulary")
        split_identity = {"protocol_id": "segment-final-20261009", "grouping_version": "rule-final-v4",
            "allocation": "frozen V4; no reallocation", "source_directory": str(source)}
    else:
        kwargs = {}
        if target in ALLOCATIONS:
            name, expected = ALLOCATIONS[target]
            allocation = root / "data/processed" / name
            kwargs = {"allocation_directory": allocation, "allocation_completion_sha256": expected}
        prepared = prepare_target_minilm(source, target, root / ".cache/minilm_export_input_identity", label_names=labels, config=config, **kwargs)
        if prepared.model_identity != identity:
            raise ValueError("Selected model identity differs from approved target input identity")
        frame, train, validation = prepared.supervised_frame, prepared.train_indices, prepared.validation_indices
        split_identity = metadata["split_identity"]
    for parent, seal_names in ((source, ("train.csv", "validation.csv", "grouping_freeze.json", "model_config.json", "protected_final_test.csv", "assignments.csv")),
                              (allocation, ("assignments.csv", "development_assignments.csv", "class_coverage.csv", "protected_final_test.csv", "protocol.json"))):
        if parent is None:
            continue
        seal = read_json(parent / "completion.json")
        for name in seal_names:
            expected = seal["artifact_sha256"][name]
            path = parent / name
            if sha256(path) != expected:
                raise ValueError(f"Approved input changed: {name}")
            snapshots[path] = expected
        snapshots[parent / "completion.json"] = sha256(parent / "completion.json")
    assignment_path = (allocation or source) / "assignments.csv"
    assignments = pd.read_csv(assignment_path, dtype=str, keep_default_na=False, usecols=["record_id", "group_id", "split"])
    protected = pd.read_csv(source / "protected_final_test.csv", dtype=str, keep_default_na=False, usecols=["record_id", "group_id"])
    if (assignments.record_id.duplicated().any() or assignments.groupby("group_id").split.nunique().gt(1).any()
            or frame.groupby("group_id").split.nunique().gt(1).any()
            or set(frame.record_id) & set(protected.record_id) or set(frame.group_id) & set(protected.group_id)):
        raise ValueError("Approved record/group/protected-test isolation failed")
    assigned = assignments.set_index("record_id").loc[frame.record_id]
    if frame.group_id.tolist() != assigned.group_id.tolist() or frame.split.tolist() != assigned.split.tolist():
        raise ValueError("Prepared population differs from approved assignments")
    if (len(train) != metadata["training_rows"] or len(validation) != metadata["validation_rows"]
            or _hash_items(frame.iloc[train].record_id.tolist()) != metadata["train_record_ids_sha256"]
            or _hash_items(frame.iloc[validation].record_id.tolist()) != metadata["validation_record_ids_sha256"]):
        raise ValueError("Selected checkpoint's ordered development population differs")
    if reference.exists():
        original_manifest = read_json(reference.parent / "training_manifest.json")
        expected = original_manifest["artifacts"]["predictions.csv"]
        if sha256(reference) != expected:
            raise ValueError("Original validation reference changed")
        snapshots[reference] = expected
    return ExportInputs(target, directory, metadata, config, labels, frame, train, validation, assignments, protected, snapshots, identity, split_identity, reference)


def verify_snapshots(inputs):
    for path, expected in inputs.snapshots.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Model/split input changed during export: {path}")


def handoff_cost_metadata(inputs, runtime, rows):
    """Copy historical measurements; never substitute current machine readings."""
    training = inputs.metadata
    source = None
    for name in ("source_training_manifest.json", "training_manifest.json"):
        path = inputs.directory / name
        if path.exists():
            if sha256(path) != inputs.snapshots.get(path):
                raise ValueError("Training-cost source is not sealed with the selected model")
            training = read_json(path)
            source = {"path": str(path), "sha256": inputs.snapshots[path]}
            break
    seconds = training.get("runtime_seconds", inputs.metadata.get("trainer_runtime_seconds"))
    total = runtime.get("validation_inference_seconds")
    for value in (seconds, total):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError("Invalid recorded runtime")
    return {"training_seconds": seconds, "runtime_seconds": seconds,
        "training_seconds_scope": "Original trainer runtime: initialization, epochs, validation and bookkeeping; input preparation before the trainer is excluded.",
        "training_optimizer_only_seconds": None, "training_metadata_source": source,
        "hardware": {
            "training": {"device": training.get("device"), "gpu_name": training.get("cuda_device_name"),
                "torch_version": training.get("torch_version"), "cuda_runtime": training.get("cuda_version"),
                "peak_allocated_gpu_bytes": inputs.metadata.get("cuda_peak_allocated_bytes"),
                "peak_reserved_gpu_bytes": inputs.metadata.get("cuda_peak_reserved_bytes")},
            "inference": {"device": runtime.get("device"), "gpu_name": runtime.get("gpu_name"),
                "gpu_total_memory_bytes": runtime.get("gpu_total_memory_bytes"),
                "torch_version": runtime.get("torch_version"), "cuda_runtime": runtime.get("cuda_runtime")}},
        "inference_latency_ms": None,
        "inference_latency_scope": "Single-record request latency was not benchmarked.",
        "amortized_inference_latency_ms_per_record": 1000 * total / rows if total is not None and rows else None,
        "amortized_inference_latency_scope": "Derived from recorded batched validation inference seconds / validation rows; includes transfers and output assembly, excludes tokenizer/model initialization. This is an amortized cost, not request latency."}


def predict_validation(inputs, *, device=None, tokenizer=None, encoder_factory=None):
    """Rebuild architecture, strictly load selected state, and run eval only."""
    import torch
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(inputs.directory / "tokenizer")
    legacy = inputs.frame[["record_id", inputs.target, *INPUT_FIELDS]].rename(columns={inputs.target: "Segment"})
    corpus = tokenize_category_corpus(legacy, inputs.labels, inputs.config, tokenizer=tokenizer, category_character_limit=512)
    for name in ("text_sha256", "token_sha256"):
        if getattr(corpus, name) != inputs.metadata[name]:
            raise ValueError(f"Original tokenizer/representation reconstruction failed: {name}")
    state = torch.load(inputs.directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (tuple(state["label_names"]) != inputs.labels or normalize_json(state["model_config"]) != normalize_json(inputs.config.to_dict())
            or _state_sha256(state["state_dict"]) != inputs.metadata["final_state_sha256"]):
        raise ValueError("Selected checkpoint state, labels or configuration differs")
    runtime = replace(inputs.config, device=device) if device is not None else inputs.config
    torch, resolved = _initialize_torch(runtime)
    if encoder_factory is None:
        from transformers import AutoConfig, AutoModel
        values = read_json(inputs.directory / "encoder_config.json")
        model_type = values.pop("model_type")
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
    else:
        encoder = encoder_factory()
    model = _make_classifier(torch, encoder, len(inputs.labels), inputs.config.dropout)
    model.load_state_dict(state["state_dict"], strict=True)
    if _state_sha256(model.state_dict()) != inputs.metadata["final_state_sha256"]:
        raise ValueError("Reconstructed classifier differs from selected state")
    model.to(resolved).eval()
    probabilities = []
    if resolved.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs.validation_indices), inputs.config.evaluation_batch_size):
            indices = inputs.validation_indices[start:start + inputs.config.evaluation_batch_size]
            width = max(len(corpus.input_ids[index]) for index in indices)
            batch = {}
            for name, rows, fill in (("input_ids", corpus.input_ids, corpus.pad_token_id), ("attention_mask", corpus.attention_mask, 0), ("token_type_ids", corpus.token_type_ids, 0)):
                if rows is None:
                    continue
                array = np.full((len(indices), width), fill, dtype=np.int64)
                for position, index in enumerate(indices):
                    array[position, :len(rows[index])] = rows[index]
                batch[name] = torch.from_numpy(array).to(resolved)
            probabilities.append(torch.softmax(model(**batch).float(), dim=1).cpu().numpy())
    if resolved.type == "cuda":
        torch.cuda.synchronize()
    matrix = np.concatenate(probabilities)
    validation = inputs.validation
    result = pd.DataFrame({"record_id": validation.record_id.tolist(), "true_label": validation.true_label.tolist(),
        "predicted_label": [inputs.labels[index] for index in matrix.argmax(axis=1)], "confidence": matrix.max(axis=1)})
    result = pd.concat([result, pd.DataFrame(matrix, columns=[f"probability_{index}" for index in range(len(inputs.labels))])], axis=1)
    return result, {"device": str(resolved), "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(resolved).total_memory if resolved.type == "cuda" else None,
        "precision": inputs.config.precision, "evaluation_batch_size": inputs.config.evaluation_batch_size,
        "validation_inference_seconds": time.perf_counter() - started, "optimizer_created": False,
        "model_training_performed": False, "weights_downloaded": False, "model_state_sha256": inputs.metadata["final_state_sha256"]}


def validate_predictions(predictions, validation, labels, protected_ids, protected_groups=()):
    expected_columns = [*BASE_COLUMNS, *[f"probability_{index}" for index in range(len(labels))]]
    if list(predictions.columns) != expected_columns:
        raise ValueError("Prediction columns/order differ from the harness contract")
    ids = predictions.record_id
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Prediction IDs must be nonblank and unique")
    if set(ids) & set(protected_ids) or set(validation.group_id) & set(protected_groups):
        raise ValueError("Protected-test record/group in validation predictions")
    if len(predictions) != len(validation) or set(ids) != set(validation.record_id):
        raise ValueError("Predictions must cover the exact approved validation population")
    expected = validation.set_index("record_id").loc[ids]
    if predictions.true_label.tolist() != expected.true_label.tolist():
        raise ValueError("Prediction truth differs from approved target labels")
    if not predictions.true_label.isin(labels).all() or not predictions.predicted_label.isin(labels).all():
        raise ValueError("Prediction labels differ from ordered vocabulary")
    probabilities = predictions[expected_columns[4:]].to_numpy(dtype=float)
    confidence = predictions.confidence.to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid or unnormalized probabilities")
    if not np.array_equal(np.asarray(labels)[probabilities.argmax(axis=1)], predictions.predicted_label.to_numpy()):
        raise ValueError("Probability class order/argmax disagrees with predicted labels")
    if not np.isfinite(confidence).all() or not np.allclose(confidence, probabilities.max(axis=1), atol=1e-7, rtol=0):
        raise ValueError("Confidence differs from maximum probability")
    return {"rows": len(predictions), "classes": len(labels), "unique_record_ids": True,
        "probability_columns": expected_columns[4:], "maximum_probability_sum_deviation": float(np.abs(probabilities.sum(axis=1) - 1).max()),
        "protected_test_records_scored": 0, "exact_validation_membership_and_truth": True, "class_alignment": True}


def compare_original(predictions, inputs):
    if inputs.reference_predictions is None or not inputs.reference_predictions.exists():
        raise ValueError("Selected model has no verified original validation reference")
    previous = pd.read_csv(inputs.reference_predictions, dtype={"record_id": str}, keep_default_na=False).set_index("record_id")
    current = predictions.set_index("record_id")
    if set(previous.index) != set(current.index):
        raise ValueError("Regenerated predictions differ from original validation record set")
    previous = previous.loc[current.index]
    if previous.true_label.tolist() != current.true_label.tolist() or previous.predicted_label.tolist() != current.predicted_label.tolist():
        raise ValueError("Regenerated validation truth/decisions differ from selected checkpoint's reference")
    columns = [f"probability_{index}" for index in range(len(inputs.labels))]
    difference = float(np.abs(previous[columns].to_numpy() - current[columns].to_numpy()).max())
    if not np.allclose(previous[columns].to_numpy(), current[columns].to_numpy(), atol=2e-5, rtol=1e-5):
        raise ValueError("Regenerated probabilities differ from selected checkpoint's reference")
    return {"rows": len(previous), "label_disagreements": 0, "maximum_probability_difference": difference,
        "reference_predictions_sha256": inputs.snapshots[inputs.reference_predictions]}


def export_target(inputs, output_root, harness_notebook, *, device=None):
    output_root = Path(output_root).resolve()
    destination = output_root / slug(inputs.target)
    source_paths = [inputs.directory.resolve(), *[Path(path).resolve().parent for path in inputs.snapshots]]
    if destination.exists():
        raise FileExistsError("Use a new export target directory")
    if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in source_paths):
        raise ValueError("Exports must be outside selected model and frozen input directories")
    verify_snapshots(inputs)
    predictions, runtime = predict_validation(inputs, device=device)
    checks = validate_predictions(predictions, inputs.validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
    reference = compare_original(predictions, inputs)
    predictions = predictions.sort_values("record_id", kind="stable").reset_index(drop=True)
    validation = inputs.validation.set_index("record_id").loc[predictions.record_id].reset_index()
    whole_ids = inputs.assignments.loc[inputs.assignments.split.eq("validation"), "record_id"].tolist()
    fingerprint = _hash_items(predictions.record_id.tolist())
    manifest = {"format": "catalogiq-minilm-validation-predictions-v1", "target": inputs.target,
        "model": inputs.config.model_name, "model_identity": inputs.model_identity,
        "checkpoint": {"source": str(inputs.directory / "checkpoint.pt"), "sha256": inputs.snapshots[inputs.directory / "checkpoint.pt"], "state_sha256": inputs.metadata["final_state_sha256"]},
        "tokenizer": {"source": str(inputs.directory / "tokenizer"), "local_files_only": True,
            "files_sha256": {str(Path(path).relative_to(inputs.directory)): digest for path, digest in inputs.snapshots.items()
                if Path(path).is_relative_to(inputs.directory / "tokenizer")}},
        "label_names": list(inputs.labels), "label_to_id": {label: index for index, label in enumerate(inputs.labels)},
        "label_mapping": {f"probability_{index}": label for index, label in enumerate(inputs.labels)},
        "config": inputs.config.to_dict(), "input_fields": list(INPUT_FIELDS), "sequence_length": inputs.config.sequence_length,
        "representation": {"formatter": "catalogiq.segment_category_ablation.build_category_texts", "category_character_limit": 512,
            "field_character_limits": list(inputs.config.field_character_limits), "combined_token_budget": inputs.config.sequence_length,
            "original_representation": inputs.metadata.get("representation"), "final_test_representation_certified": False},
        "split_identity": inputs.split_identity, "protocol_id": "segment-final-20261009", "partition": "validation",
        "validation_rows": len(predictions), "validation_record_ids_sha256": fingerprint,
        "validation_record_ids_hash_recipe": "Sorted IDs, each prefixed with an 8-byte big-endian UTF-8 byte length; SHA-256",
        "prediction_record_ids_sha256": fingerprint,
        "original_training_validation_record_ids_sha256": inputs.metadata["validation_record_ids_sha256"],
        "whole_validation_partition_record_ids_sha256": _hash_items(sorted(whole_ids)),
        "validation_group_ids_sha256": _hash_items(validation.group_id.tolist()),
        "text_sha256": inputs.metadata["text_sha256"], "token_sha256": inputs.metadata["token_sha256"],
        "source_input_sha256": {str(path): digest for path, digest in inputs.snapshots.items()},
        "inference": runtime, "checks": checks, "comparison_to_original": reference,
        "probabilities_calibrated": False, "model_training_performed": False, "test_csv_read_or_hashed": False}
    manifest.update(handoff_cost_metadata(inputs, runtime, len(predictions)))
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{slug(inputs.target)}-", dir=output_root))
    try:
        predictions.to_csv(stage / "predictions.csv", index=False, lineterminator="\n")
        validation.to_csv(stage / "validation_context.csv", index=False, lineterminator="\n")
        write_json(stage / "label_mapping.json", manifest["label_mapping"])
        write_json(stage / "manifest.json", manifest)
        restored = pd.read_csv(stage / "predictions.csv", dtype={"record_id": str}, keep_default_na=False)
        validate_predictions(restored, inputs.validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
        run, compatibility = load_with_maria_harness(stage, harness_notebook, inputs.validation, inputs.protected.record_id, whole_ids)
        if tuple(run["labels"]) != inputs.labels:
            raise ValueError("Maria's loader read a different ordered vocabulary")
        write_json(stage / "harness_compatibility.json", compatibility)
        (stage / "README.txt").write_text(
            f"CatalogIQ {inputs.target}: selected MiniLM validation predictions only.\n"
            "Use Maria's unchanged load_predictions() with this target's approved validation context. "
            "Set VA_IDS from validation_context.csv, GROUP_OF from its record_id/group_id columns, "
            "and TEST_IDS from the frozen protected_final_test.csv ID metadata only. "
            "Set VA_ID_HASHES to the labelled/whole-validation hashes in manifest.json after verifying "
            "them against the approved allocation. Do not run the notebook's original candidate/test-label "
            "setup or its baseline-training cells for this compatibility check. Brand/Sub-Segment use "
            "their approved target-specific assignments, so the default Segment V4 context is unsuitable.\n"
            "The probability columns follow label_mapping.json and manifest.json label_names. "
            "Rows are ordered by record_id; join by record_id, never row position. "
            "Confidence is uncalibrated softmax maximum. Harness observed-label macro-F1 can differ "
            "from full-vocabulary macro-F1 when classes have zero validation support. "
            "No model training, checkpoint modification or protected-test scoring was performed.\n",
            encoding="utf8", newline="\n")
        manifest["harness_compatibility"] = {key: compatibility[key] for key in ("status", "harness_commit", "rows_consumed", "split_checked", "metrics", "scope")}
        manifest["files_sha256"] = {path.name: sha256(path) for path in stage.iterdir() if path.is_file() and path.name != "manifest.json"}
        write_json(stage / "manifest.json", manifest)
        verify_snapshots(inputs)
        stage.rename(destination)
    except Exception:
        if stage.parent.resolve() != output_root or not stage.name.startswith(f".{slug(inputs.target)}-"):
            raise RuntimeError("Refusing cleanup outside the export staging directory")
        shutil.rmtree(stage)
        raise
    return {"target": inputs.target, "directory": str(destination), "rows": len(predictions), "classes": len(inputs.labels),
        "validation_record_ids_sha256": fingerprint, "manifest_sha256": sha256(destination / "manifest.json"),
        "harness_compatibility": compatibility, "comparison_to_original": reference, "inference": runtime}


def audit_handoff(inputs, directory, harness_notebook):
    """Inspect actual export bytes and exercise the unchanged loader."""
    directory = Path(directory)
    names = {path.name for path in directory.iterdir() if path.is_file()}
    manifest = read_json(directory / "manifest.json")
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    validation = inputs.validation
    probability_mapping = {f"probability_{i}": label for i, label in enumerate(inputs.labels)}
    checks = {"filenames": {"predictions.csv", "manifest.json"}.issubset(names),
        "prediction_columns": list(predictions) == [*BASE_COLUMNS, *probability_mapping],
        "original_class_mapping": manifest.get("label_mapping") == probability_mapping
            and tuple(manifest.get("label_names", ())) == inputs.labels
            and read_json(directory / "label_mapping.json") == probability_mapping}
    digest = hashlib.sha256()
    for record_id in predictions.record_id:
        value = record_id.encode("utf8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    checks["ordered_record_id_hash"] = manifest.get("validation_record_ids_sha256") == digest.hexdigest()
    checks["identity"] = (manifest.get("target") == inputs.target and manifest.get("model_identity") == inputs.model_identity
        and manifest["checkpoint"]["sha256"] == inputs.snapshots[inputs.directory / "checkpoint.pt"])
    expected_costs = handoff_cost_metadata(inputs, manifest["inference"], len(predictions))
    checks["cost_and_hardware_metadata"] = all(key in manifest and manifest[key] == value for key, value in expected_costs.items())
    line_endings = {path.name: {"CRLF": path.read_bytes().count(b"\r\n"),
        "CR": path.read_bytes().count(b"\r")} for path in directory.iterdir()
        if path.is_file() and path.suffix in (".csv", ".json", ".txt")}
    checks["LF_artifacts"] = all(counts["CR"] == 0 for counts in line_endings.values())
    validation_checks = validate_predictions(predictions, validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
    checks["validation_population_and_probabilities"] = True
    context = pd.read_csv(directory / "validation_context.csv", dtype=str, keep_default_na=False).set_index("record_id")
    expected = validation.set_index("record_id")
    checks["validation_context"] = (len(context) == len(expected) and set(context.index) == set(expected.index)
        and context.loc[expected.index].equals(expected))
    whole_ids = inputs.assignments.loc[inputs.assignments.split.eq("validation"), "record_id"]
    run, compatibility = load_with_maria_harness(directory, harness_notebook, validation, inputs.protected.record_id, whole_ids)
    checks["unchanged_harness"] = tuple(run["labels"]) == inputs.labels and compatibility["rows_dropped"] == 0
    checks["artifact_hashes"] = all(sha256(directory / name) == expected for name, expected in manifest["files_sha256"].items())
    return {"target": inputs.target, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "rows": len(predictions), "classes": len(inputs.labels),
        "validation_record_ids_sha256": digest.hexdigest(), "line_endings": line_endings,
        "validation_checks": validation_checks, "harness_compatibility": compatibility}


def repair_export_handoff(inputs, directory, harness_notebook):
    """Repair metadata/newlines only; retain every original CSV field verbatim."""
    directory = Path(directory).resolve()
    sources = [inputs.directory.resolve(), *[Path(path).resolve().parent for path in inputs.snapshots]]
    if any(directory.is_relative_to(path) or path.is_relative_to(directory) for path in sources):
        raise ValueError("Cannot repair model or frozen input directories")
    verify_snapshots(inputs)
    before = audit_handoff(inputs, directory, harness_notebook)
    allowed = {"cost_and_hardware_metadata", "LF_artifacts"}
    if any(not passed and name not in allowed for name, passed in before["checks"].items()):
        raise ValueError("Export has integrity defects beyond metadata/line endings")
    originals = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    stage = Path(tempfile.mkdtemp(prefix=f".{slug(inputs.target)}-repair-", dir=directory.parent))
    try:
        for name, raw in originals.items():
            (stage / name).write_bytes(raw.replace(b"\r\n", b"\n") if Path(name).suffix in (".csv", ".json", ".txt") else raw)
        manifest = read_json(stage / "manifest.json")
        manifest.update(handoff_cost_metadata(inputs, manifest["inference"], before["rows"]))
        manifest["handoff_correction"] = {"previous_manifest_sha256": hashlib.sha256(originals["manifest.json"]).hexdigest(),
            "previous_predictions_csv_sha256": hashlib.sha256(originals["predictions.csv"]).hexdigest(),
            "LF_predictions_csv_sha256": sha256(stage / "predictions.csv"), "prediction_values_changed": False,
            "inference_rerun": False, "corrections": ["Explicit historical cost/hardware metadata with null unmeasured values", "LF CSV/JSON/text line endings"]}
        write_json(stage / "manifest.json", manifest)
        whole_ids = inputs.assignments.loc[inputs.assignments.split.eq("validation"), "record_id"]
        _, compatibility = load_with_maria_harness(stage, harness_notebook, inputs.validation, inputs.protected.record_id, whole_ids)
        write_json(stage / "harness_compatibility.json", compatibility)
        manifest["harness_compatibility"] = {key: compatibility[key] for key in ("status", "harness_commit", "rows_consumed", "split_checked", "metrics", "scope")}
        manifest["files_sha256"] = {path.name: sha256(path) for path in stage.iterdir() if path.is_file() and path.name != "manifest.json"}
        write_json(stage / "manifest.json", manifest)
        after = audit_handoff(inputs, stage, harness_notebook)
        if after["status"] != "PASS":
            raise ValueError(f"Corrected handoff failed validation: {after['checks']}")
        for name in ("predictions.csv", "validation_context.csv"):
            if (stage / name).read_bytes() != originals[name].replace(b"\r\n", b"\n"):
                raise ValueError("Repair changed CSV fields")
        verify_snapshots(inputs)
        for name, raw in originals.items():
            if (directory / name).read_bytes() != raw:
                raise ValueError("Export was concurrently modified")
        # Atomic file replacements, manifest last so it certifies completed files.
        for name in (*[name for name in originals if name != "manifest.json"], "manifest.json"):
            (stage / name).replace(directory / name)
        return {"before": before, "after": after, "prediction_values_unchanged": True,
            "predictions_csv_LF_sha256": sha256(directory / "predictions.csv"),
            "manifest_sha256": sha256(directory / "manifest.json"), "inference_rerun": False}
    finally:
        if stage.parent.resolve() != directory.parent or not stage.name.startswith(f".{slug(inputs.target)}-repair-"):
            raise RuntimeError("Refusing cleanup outside the export repair staging directory")
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="all")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--harness-notebook", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    root = Path.cwd()
    summary_path = Path(args.output_root) / "export_summary.json"
    if summary_path.exists():
        raise FileExistsError("Batch summary already exists; use a new output root")
    summaries = []
    for target in TARGETS if args.target == "all" else (args.target,):
        print(f"Preparing validation-only export: {target}", flush=True)
        inputs = prepare_export(root, target)
        result = export_target(inputs, args.output_root, args.harness_notebook, device=args.device)
        summaries.append(result)
        print(json.dumps(result, indent=2), flush=True)
    write_json(summary_path, {"status": "completed_and_verified", "targets": summaries})


if __name__ == "__main__":
    main()
