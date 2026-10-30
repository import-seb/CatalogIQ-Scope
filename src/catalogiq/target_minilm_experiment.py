"""Sequential unweighted MiniLM experiments and aggregate validation handoffs.

Neural training is delegated unchanged to run_target_minilm. The only lookup
baseline is the explicitly authorized raw-to-curated Brand normalization task.
No new record-level prediction writer or inference endpoint is implemented.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import unicodedata
import zipfile

import numpy as np
import pandas as pd

from .features import sha256
from .segment_transformer import _hash_items, _plain_text, _state_sha256
from .target_minilm_preparation import INPUT_FIELDS, prepare_target_minilm, run_target_minilm


ORDER = ("TargetAgeGroup", "Sub-Segment", "Brand")
EXPECTED = {"TargetAgeGroup": (54282, 11624, 3), "Sub-Segment": (53956, 11562, 49), "Brand": (58888, 12619, 143)}
ALLOCATIONS = {
    "Brand": ("brand_development_allocation_v1_20261029", "94c64e5bd80219595f3c8764acd67a5d64496a63164276807d42b14d56624c6e"),
    "Sub-Segment": ("sub_segment_development_allocation_v1_20261029_r2", "26f52bd5360af96d0271c998f13c846dfeaa325bf0fb90b4c85011df5cabdc6a"),
}
FEATURE_CONTRACT = {
    "input_fields": list(INPUT_FIELDS), "combined_token_budget": 256,
    "Brand": "Raw-to-curated brand normalization. Retailer-supplied ProductBrand is explicitly authorized at inference; exact target echo is expected, not a blocking issue under this contract.",
    "TargetAgeGroup": "Product age cues in approved product fields are authorized; preflight found no blocking leakage.",
    "Sub-Segment": "Retailer-supplied ProductCategory breadcrumbs are authorized semantic evidence; preflight found no blocking leakage.",
    "protected_test_used": False,
}
MISSING_FEATURE = frozenset(("", "null", "none", "nan", "n/a", "na"))


def slug(target):
    return target.casefold().replace("-", "_")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def normalized_json(value):
    return json.loads(json.dumps(value))


def classification_metrics(truth, predicted, label_names):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    truth, predicted = np.asarray(truth, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    labels = np.arange(len(label_names))
    if truth.shape != predicted.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Metrics require matching nonempty label arrays")
    if not np.isin(truth, labels).all() or not np.isin(predicted, labels).all():
        raise ValueError("Encoded labels differ from the ordered vocabulary")
    supported = np.unique(truth)
    return {"accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1_validation_supported_classes": float(f1_score(truth, predicted, labels=supported, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "per_class": classification_report(truth, predicted, labels=labels, target_names=list(label_names), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "ordered_labels": list(label_names), "rows": len(truth), "supported_classes": len(supported),
        "macro_f1_scope": "Entire ordered training vocabulary; zero validation-support classes contribute zero F1.",
        "confusion_matrix_axes": "rows=true labels; columns=predicted labels"}


def brand_key(value):
    return unicodedata.normalize("NFKC", _plain_text(str(value))).casefold().strip()


def fit_brand_lookup(prepared):
    """Fit using labeled training records only; never read validation labels."""
    if prepared.target_column != "Brand":
        raise ValueError("The ProductBrand lookup is authorized for Brand only")
    started = time.perf_counter()
    frame = prepared.supervised_frame.iloc[prepared.train_indices].copy()
    names = prepared.label_names
    label_to_id = {label: index for index, label in enumerate(names)}
    class_counts = frame.Brand.value_counts().reindex(names, fill_value=0)
    fallback = names[int(np.argmax(class_counts.to_numpy()))]
    frame["key"] = frame.ProductBrand.map(brand_key)
    valid = frame.loc[~frame.key.isin(MISSING_FEATURE)]
    counts = valid.groupby(["key", "Brand"]).size().rename("count").reset_index()
    counts["label_id"] = counts.Brand.map(label_to_id)
    chosen = counts.sort_values(["key", "count", "label_id"], ascending=[True, False, True]).drop_duplicates("key")
    mapping = dict(zip(chosen.key, chosen.Brand))
    return {"target": "Brand", "version": "training-only-productbrand-normalization-v1", "label_names": list(names),
        "mapping": mapping, "fallback_label": fallback, "fallback_rule": "Most frequent labeled training class; ties use saved label order. Applies to both missing and unseen ProductBrand values.",
        "normalization": "Existing plain-text cleanup, Unicode NFKC, casefold, strip; no validation-fitted transformations.",
        "missing_normalized_values": sorted(MISSING_FEATURE), "mapping_tie_break": "Smallest saved label ID among tied training-majority labels",
        "training_class_counts": {label: int(count) for label, count in class_counts.items()},
        "training_records": len(frame), "training_nonmissing_feature_records": len(valid),
        "training_feature_values": len(mapping), "ambiguous_training_feature_values": int(counts.groupby("key").Brand.nunique().gt(1).sum()),
        "training_record_ids_sha256": _hash_items(frame.record_id.tolist()),
        "training_group_ids_sha256": _hash_items(frame.group_id.tolist()),
        "fit_seconds": time.perf_counter() - started, "validation_labels_used_to_fit": False}


def evaluate_brand_lookup(prepared, model):
    if prepared.target_column != "Brand" or list(prepared.label_names) != model["label_names"]:
        raise ValueError("Lookup target or label order differs")
    training = prepared.supervised_frame.iloc[prepared.train_indices]
    if _hash_items(training.record_id.tolist()) != model["training_record_ids_sha256"]:
        raise ValueError("Lookup was fitted on different training records")
    started = time.perf_counter()
    frame = prepared.supervised_frame.iloc[prepared.validation_indices]
    keys = frame.ProductBrand.map(brand_key)
    missing = keys.isin(MISSING_FEATURE).to_numpy()
    covered = keys.isin(model["mapping"]).to_numpy() & ~missing
    unseen = ~covered & ~missing
    predictions = keys.map(model["mapping"]).fillna(model["fallback_label"])
    label_to_id = {label: index for index, label in enumerate(prepared.label_names)}
    truth = frame.Brand.map(label_to_id).to_numpy()
    predicted = predictions.map(label_to_id).to_numpy()
    metrics = classification_metrics(truth, predicted, prepared.label_names)
    cohorts = {}
    for name, mask in (("seen_feature_values", covered), ("unseen_nonmissing_values", unseen), ("missing_values", missing), ("all_fallback_records", ~covered)):
        cohorts[name] = {"records": int(mask.sum()), "fraction": float(mask.mean()),
            "metrics": classification_metrics(truth[mask], predicted[mask], prepared.label_names) if mask.any() else None}
    return {"full_validation": metrics, "coverage_records": int(covered.sum()), "coverage_fraction": float(covered.mean()),
        "fallback_label": model["fallback_label"], "fallback_records": int((~covered).sum()), "cohorts": cohorts,
        "validation_record_ids_sha256": _hash_items(frame.record_id.tolist()), "evaluation_seconds": time.perf_counter() - started,
        "includes_catch_all_labels": True, "excludes_no_labeled_validation_records": True, "record_predictions_exported": False}


def evaluate_minilm(prepared):
    """Read only the existing trainer's internal validation artifact."""
    frame = prepared.supervised_frame.iloc[prepared.validation_indices]
    path = prepared.output_directory / "predictions.csv"
    predicted = pd.read_csv(path, dtype={"record_id": str}, keep_default_na=False)
    target, names = prepared.target_column, prepared.label_names
    if predicted.record_id.tolist() != frame.record_id.tolist() or predicted.true_label.tolist() != frame[target].tolist():
        raise ValueError("Validation export differs from approved ordered membership/labels")
    probabilities = predicted[[f"probability_{index}" for index in range(len(names))]].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid validation probabilities")
    encoded = probabilities.argmax(axis=1)
    if predicted.predicted_label.tolist() != [names[index] for index in encoded]:
        raise ValueError("Validation decision differs from unchanged argmax")
    truth = frame[target].map({label: index for index, label in enumerate(names)}).to_numpy()
    return {**classification_metrics(truth, encoded, names), "validation_record_ids_sha256": _hash_items(frame.record_id.tolist()),
        "validation_group_ids_sha256": _hash_items(frame.group_id.tolist()), "internal_validation_artifact_sha256": sha256(path)}


def write_aggregate_metrics(directory, metrics, prefix=""):
    """Write class aggregates/matrices only, never individual predictions."""
    names = metrics["ordered_labels"]
    rows = [{"label_id": index, "label": name, **metrics["per_class"][name]} for index, name in enumerate(names)]
    pd.DataFrame(rows).to_csv(Path(directory) / f"{prefix}per_class_metrics.csv", index=False)
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=names, columns=names)
    matrix.index.name = "true_label"
    matrix.to_csv(Path(directory) / f"{prefix}confusion_matrix.csv")


def verify_prepared(prepared):
    frame, reserved = prepared.frame, prepared.protected_test
    if (frame.groupby("group_id").split.nunique().gt(1).any() or set(frame.record_id) & set(reserved.record_id)
            or set(frame.group_id) & set(reserved.group_id)):
        raise ValueError("Group or protected-record isolation failed")
    source = prepared.tokenizer_directory.parent
    if sha256(source / "completion.json") != prepared.audit["source_completion_sha256"]:
        raise ValueError("Frozen completion seal changed")
    for name, expected in prepared.audit["source_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Frozen development input changed: {name}")
    if prepared.audit["allocation_directory"]:
        allocation = Path(prepared.audit["allocation_directory"])
        for name, expected in prepared.audit["allocation_sha256"].items():
            if sha256(allocation / name) != expected:
                raise ValueError(f"Approved allocation changed: {name}")


def package_model(prepared, result, root):
    import torch
    source = prepared.output_directory
    destination = Path(root) / "handoff" / slug(prepared.target_column)
    if destination.exists():
        raise FileExistsError("Handoff destination already exists")
    state = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (state["label_names"] != list(prepared.label_names)
            or normalized_json(state["model_config"]) != normalized_json(prepared.config.to_dict())
            or state["state_dict"]["classifier.weight"].shape[0] != len(prepared.label_names)
            or _state_sha256(state["state_dict"]) != result["final_state_sha256"]):
        raise ValueError("Checkpoint state/configuration/label alignment failed")
    metadata = {key: result[key] for key in ("target", "model_identity", "label_names", "label_to_id", "input_fields",
        "sequence_length", "representation", "split_identity", "config", "feature_contract", "loss_configuration")}
    destination.mkdir(parents=True)
    torch.save({**state, **metadata, "selection": "Only authorized candidate; fixed final third epoch"}, destination / "checkpoint.pt")
    shutil.copytree(source / "tokenizer", destination / "tokenizer")
    for name in ("encoder_config.json", "model_config.json", "run_protocol.json", "evaluation.json", "per_class_metrics.csv", "confusion_matrix.csv", "label_vocabulary.json", "feature_contract.json"):
        shutil.copy2(source / name, destination / name)
    shutil.copy2(source / "training_manifest.json", destination / "source_training_manifest.json")
    if prepared.target_column == "Brand":
        for name in ("lookup_model.json", "lookup_baseline.json", "lookup_per_class_metrics.csv", "lookup_confusion_matrix.csv"):
            shutil.copy2(source / name, destination / name)
    (destination / "README.md").write_text(
        f"# {prepared.target_column} MiniLM for Maria\n\n"
        "One unweighted model, fixed final third epoch. See manifest.json and label_vocabulary.json for the target, label order, "
        "representation, allocation identity, configuration, metrics, runtime and checksums. Use the existing ProductCategory-first "
        "five-field formatter and its character limits, with a combined 256-token budget. Architecture is the pinned encoder, "
        "attention-mask mean pooling, dropout and a linear classification head. This bundle implements no inference endpoint.\n\n"
        "Only aggregate validation results are included. No protected-test evaluation or new prediction-export workflow was performed. "
        "source_training_manifest.json records the original experiment artifacts; manifest.json hashes the packaged files. "
        "Brand uses retailer-supplied ProductBrand for explicitly authorized raw-to-curated normalization. If present, lookup_baseline.json "
        "reports the full validation population and fallback cohorts; lookup_model.json was fitted on training records only.\n",
        encoding="utf8")
    hashes = {path.relative_to(destination).as_posix(): sha256(path) for path in destination.rglob("*") if path.is_file()}
    write_json(destination / "manifest.json", {**result, "source_checkpoint_sha256": sha256(source / "checkpoint.pt"), "artifact_sha256": hashes,
        "checkpoint_selection": "Fixed final third epoch; no tuning or epoch selection", "external_delivery_performed": False})
    for name, expected in hashes.items():
        if sha256(destination / name) != expected:
            raise ValueError("Packaged artifact changed")
    reloaded = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=True)
    if _state_sha256(reloaded["state_dict"]) != result["final_state_sha256"]:
        raise ValueError("Packaged state differs from evaluated checkpoint")
    if any(normalized_json(reloaded[key]) != normalized_json(value) for key, value in metadata.items()):
        raise ValueError("Packaged target/representation metadata differs")
    archive = Path(root) / f"{slug(prepared.target_column)}_minilm_maria_v1.zip"
    if archive.exists():
        raise FileExistsError("Handoff ZIP already exists")
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=destination)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or sum(name.endswith("checkpoint.pt") for name in zipped.namelist()) != 1:
            raise ValueError("Handoff ZIP verification failed")
    return {"directory": str(destination), "archive": str(archive), "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
        "packaged_checkpoint_sha256": sha256(destination / "checkpoint.pt"), "checksums_and_state_verified": True}


def run_experiment(prepared, root, *, training_authorized=False, progress=print):
    if training_authorized is not True:
        raise PermissionError("Explicit authorization is required for training")
    if prepared.target_column not in ORDER:
        raise ValueError("This batch supports only TargetAgeGroup, Sub-Segment and Brand")
    import torch
    if (not torch.cuda.is_available() or torch.__version__ != "2.11.0+cu128"
            or "RTX 3060 Laptop" not in torch.cuda.get_device_name(0)):
        raise ValueError("The approved RTX 3060/CUDA environment is required; no fallback")
    verify_prepared(prepared)
    lookup_model = fit_brand_lookup(prepared) if prepared.target_column == "Brand" else None
    lookup_evaluation = evaluate_brand_lookup(prepared, lookup_model) if lookup_model else None
    initialized = False

    def notify(message):
        nonlocal initialized
        if not initialized and (prepared.output_directory / "initialization.json").exists():
            value = read_json(prepared.output_directory / "initialization.json")
            for key, expected in (("device", "cuda"), ("target", prepared.target_column), ("label_names", list(prepared.label_names)),
                                  ("input_fields", list(INPUT_FIELDS)), ("sequence_length", 256)):
                if value[key] != expected:
                    raise ValueError(f"Initialization metadata mismatch: {key}")
            initialized = True
        if progress:
            progress(message)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = run_target_minilm(prepared, dry_run=False, training_authorized=True, progress=notify)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    verify_prepared(prepared)
    source = prepared.output_directory
    manifest = read_json(source / "training_manifest.json")
    history = pd.read_csv(source / "training_history.csv")
    expected_steps = math.ceil(len(prepared.train_indices) / prepared.config.batch_size) * 3
    if (manifest["completed_epochs"] != 3 or manifest["optimizer_steps"] != expected_steps
            or len(history) != 3 or not np.isfinite(history.select_dtypes(include="number").to_numpy()).all()):
        raise ValueError("Incomplete or nonfinite training history")
    evaluation = evaluate_minilm(prepared)
    if not math.isclose(evaluation["macro_f1"], manifest["final_validation"]["macro_f1"], abs_tol=1e-12):
        raise ValueError("Aggregate metrics differ from trainer validation")
    if lookup_evaluation and lookup_evaluation["validation_record_ids_sha256"] != evaluation["validation_record_ids_sha256"]:
        raise ValueError("Brand lookup and MiniLM evaluated different records")
    write_json(source / "evaluation.json", evaluation)
    write_aggregate_metrics(source, evaluation)
    write_json(source / "label_vocabulary.json", {"target": prepared.target_column, "label_names": list(prepared.label_names), "label_to_id": report["label_to_id"]})
    write_json(source / "feature_contract.json", FEATURE_CONTRACT)
    if lookup_model:
        write_json(source / "lookup_model.json", lookup_model)
        write_json(source / "lookup_baseline.json", lookup_evaluation)
        write_aggregate_metrics(source, lookup_evaluation["full_validation"], "lookup_")
    result = {key: report[key] for key in ("target", "model_identity", "model_identity_sha256", "label_names", "label_to_id", "input_fields",
        "sequence_length", "representation", "split_identity", "config", "training_rows", "validation_rows", "schedule",
        "ordered_supervised_membership_sha256", "train_record_ids_sha256", "validation_record_ids_sha256", "train_group_ids_sha256", "validation_group_ids_sha256",
        "label_ids_sha256", "text_sha256", "token_sha256", "tokenizer_sha256", "readiness")}
    result.update(status="completed_and_verified", feature_contract=FEATURE_CONTRACT,
        loss_configuration={"name": "unweighted mean cross_entropy", "class_weights": None},
        output_directory=str(source), final_state_sha256=manifest["final_state_sha256"], initial_state_sha256=manifest["initial_state_sha256"],
        evaluation=evaluation, lookup_baseline=lookup_evaluation,
        trainer_runtime_seconds=manifest["runtime_seconds"], wall_seconds_including_preflight_and_saving=elapsed,
        epoch_training_validation_seconds=float(history.epoch_seconds.sum()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        completed_epochs=3, optimizer_steps=expected_steps, test_csv_read_or_hashed=False,
        reproducibility={"source_sha256": prepared.audit["source_sha256"], "source_completion_sha256": prepared.audit["source_completion_sha256"],
            "allocation_sha256": prepared.audit["allocation_sha256"], "software_versions": report["readiness"]["software_versions"], "experiment_module_sha256": sha256(Path(__file__))})
    manifest.update(feature_contract=FEATURE_CONTRACT, loss_configuration=result["loss_configuration"])
    manifest["artifacts"] = {path.name: sha256(path) for path in source.iterdir() if path.is_file() and path.name != "training_manifest.json"}
    write_json(source / "training_manifest.json", manifest)
    for name, expected in manifest["artifacts"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Final artifact mismatch: {name}")
    result["handoff"] = package_model(prepared, result, root)
    write_json(Path(root) / f"{slug(prepared.target_column)}_result.json", result)
    return result


def verify_predecessors(batch, target):
    """Verify saved predecessor archives after their paths round-trip via JSON."""
    for previous in ORDER[:ORDER.index(target)]:
        result = read_json(Path(batch) / f"{slug(previous)}_result.json")
        archive = Path(result["handoff"]["archive"])
        if result["status"] != "completed_and_verified" or sha256(archive) != result["handoff"]["archive_sha256"]:
            raise ValueError("A preceding target has not completed verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=ORDER)
    parser.add_argument("--experiment-directory", required=True)
    parser.add_argument("--training-authorized", action="store_true")
    args = parser.parse_args()
    if not args.training_authorized:
        parser.error("Actual training requires explicit --training-authorized")
    root, batch = Path.cwd(), Path(args.experiment_directory).resolve()
    protocol = read_json(batch / "experiment_protocol.json")
    if list(batch.glob("*_failure.json")):
        raise ValueError("Batch stopped after a failure; no automatic retry or continuation")
    if protocol["target_order"] != list(ORDER) or protocol["feature_contract"] != FEATURE_CONTRACT:
        raise ValueError("Batch order or authorized feature contract differs")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(root / "src/catalogiq" / name) != expected:
            raise ValueError(f"Experiment implementation changed: {name}")
    verify_predecessors(batch, args.target)
    kwargs = {}
    if args.target in ALLOCATIONS:
        directory, digest = ALLOCATIONS[args.target]
        kwargs = {"allocation_directory": root / "data/processed" / directory, "allocation_completion_sha256": digest}
    prepared = prepare_target_minilm(root / "data/processed/split_integration_reproduced_20261010", args.target, batch / "models", **kwargs)
    if (len(prepared.train_indices), len(prepared.validation_indices), len(prepared.label_names)) != EXPECTED[args.target]:
        raise ValueError("Prepared target population differs from approval")
    if normalized_json(prepared.config.to_dict()) != protocol["config"]:
        raise ValueError("Hyperparameters differ from the approved configuration")
    try:
        result = run_experiment(prepared, batch, training_authorized=True, progress=lambda message: print(message, flush=True))
    except Exception as error:
        write_json(batch / f"{slug(args.target)}_failure.json", {"target": args.target, "status": "failed_stop_batch", "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps({"target": args.target, "status": result["status"], "accuracy": result["evaluation"]["accuracy"],
        "macro_f1": result["evaluation"]["macro_f1"], "trainer_runtime_seconds": result["trainer_runtime_seconds"],
        "cuda_peak_allocated_bytes": result["cuda_peak_allocated_bytes"], "cuda_peak_reserved_bytes": result["cuda_peak_reserved_bytes"],
        "handoff": result["handoff"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
