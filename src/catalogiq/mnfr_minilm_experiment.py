"""Two fixed Mnfr loss candidates around the unchanged legacy MiniLM trainer."""
from contextlib import contextmanager
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from .features import sha256
from .segment_transformer import _hash_items, _metrics, _state_sha256
from .target_minilm_preparation import prepare_target_minilm, run_mnfr_minilm


LABELS = ("All others", "J&J")
WEIGHTS = {"unweighted": None, "weighted_1_5": (1.0, 5.0)}
TIE_TOLERANCE = 0.001


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def candidate_preparation(prepared, candidate):
    """Keep input/config identities intact, adding loss identity for weighting."""
    if prepared.target_column != "Mnfr" or prepared.label_names != LABELS:
        raise ValueError("This experiment requires Mnfr with ordered labels All others, J&J")
    if candidate not in WEIGHTS:
        raise ValueError("Only the two predeclared Mnfr loss candidates are supported")
    spec = {"name": "cross_entropy", "ordered_labels": list(LABELS),
            "training_class_weights": list(WEIGHTS[candidate]) if WEIGHTS[candidate] else None,
            "reduction": "mean", "validation_class_weights": None}
    audit = {**prepared.audit, "base_model_identity": prepared.model_identity, "loss_configuration": spec}
    if WEIGHTS[candidate] is None:
        return replace(prepared, audit=audit)
    identity_sha = hashlib.sha256(json.dumps({"base_model_identity_sha256": prepared.audit["model_identity_sha256"],
        "loss_configuration": spec}, sort_keys=True).encode("utf8")).hexdigest()
    identity = f"minilm-mnfr-category256-{identity_sha[:16]}"
    audit.update(model_identity=identity, model_identity_sha256=identity_sha)
    return replace(prepared, audit=audit, model_identity=identity,
                   output_directory=prepared.output_directory.with_name(identity))


@contextmanager
def mnfr_loss_scope(torch, weights):
    """Scope loss interception to one synchronous Mnfr call; restore on failure.

    Training changes only the CE weight argument. The trainer's no_grad
    validation path delegates to the original unweighted CE implementation.
    No optimizer, sampling, architecture or shared configuration is replaced.
    """
    original = torch.nn.CrossEntropyLoss.forward
    counts = {"training_batches": 0, "training_examples": 0,
              "validation_batches": 0, "validation_examples": 0}
    cached = {}

    def forward(criterion, logits, labels):
        if criterion.weight is not None or criterion.reduction != "mean" or criterion.label_smoothing != 0:
            raise ValueError("Unexpected CE configuration in the fixed Mnfr experiment")
        if logits.shape[-1] != 2:
            raise ValueError("Mnfr loss adapter requires exactly two classes")
        training = torch.is_grad_enabled() and logits.requires_grad
        role = "training" if training else "validation"
        counts[role + "_batches"] += 1
        counts[role + "_examples"] += labels.numel()
        if not training or weights is None:
            return original(criterion, logits, labels)
        key = (logits.device, logits.dtype)
        if key not in cached:
            cached[key] = logits.new_tensor(weights)
        return torch.nn.functional.cross_entropy(logits, labels, weight=cached[key],
            ignore_index=criterion.ignore_index, reduction=criterion.reduction,
            label_smoothing=criterion.label_smoothing)

    with patch.object(torch.nn.CrossEntropyLoss, "forward", forward):
        yield counts


def evaluate_candidate(prepared):
    """Score only the trainer's exact, ordered Mnfr validation export."""
    from sklearn.metrics import confusion_matrix
    frame = prepared.supervised_frame.iloc[prepared.validation_indices]
    predictions = pd.read_csv(prepared.output_directory / "predictions.csv", dtype={"record_id": str})
    if (predictions.record_id.tolist() != frame.record_id.tolist()
            or predictions.true_label.tolist() != frame.Mnfr.tolist()):
        raise ValueError("Validation predictions differ from the frozen Mnfr validation rows")
    probabilities = predictions[["probability_0", "probability_1"]].to_numpy()
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid validation probabilities")
    truth = frame.Mnfr.map({name: index for index, name in enumerate(LABELS)}).to_numpy()
    predicted = probabilities.argmax(axis=1)
    if predictions.predicted_label.tolist() != [LABELS[index] for index in predicted]:
        raise ValueError("Validation labels disagree with the unchanged argmax decision")
    return {**_metrics(truth, probabilities, LABELS), "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_rows": "true labels", "confusion_matrix_columns": "predicted labels",
        "ordered_labels": list(LABELS), "validation_rows": len(frame),
        "validation_record_ids_sha256": _hash_items(frame.record_id.tolist()),
        "predictions_sha256": sha256(prepared.output_directory / "predictions.csv")}


def run_candidate(prepared, candidate, *, training_authorized=False, progress=print):
    """Train one fixed final-epoch candidate, with no protected-test access."""
    if training_authorized is not True:
        raise PermissionError("Actual Mnfr training requires explicit authorization")
    prepared = candidate_preparation(prepared, candidate)
    import torch
    if not torch.cuda.is_available() or prepared.config.precision != "fp32" or prepared.config.device != "auto":
        raise ValueError("This experiment requires CUDA execution with the unchanged auto/FP32 configuration")
    if prepared.output_directory.exists():
        raise FileExistsError("A new candidate output directory is required")
    loss_spec = prepared.audit["loss_configuration"]
    extra = {"candidate": candidate, "loss_configuration": loss_spec,
             "base_model_identity": prepared.audit["base_model_identity"]}
    initialized = False

    def notify(message):
        nonlocal initialized
        if not initialized and (prepared.output_directory / "initialization.json").exists():
            for name in ("initialization.json", "run_protocol.json"):
                path = prepared.output_directory / name
                value = _read(path)
                if name == "initialization.json" and value["device"] != "cuda":
                    raise ValueError("CPU fallback is forbidden for this experiment")
                _write(path, {**value, **extra})
            initialized = True
        if progress:
            progress(message)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with mnfr_loss_scope(torch, WEIGHTS[candidate]) as counts:
            report = run_mnfr_minilm(prepared, dry_run=False, training_authorized=True, progress=notify)
        torch.cuda.synchronize()
    except Exception as error:
        if prepared.output_directory.exists():
            _write(prepared.output_directory / "candidate_failure.json", {**extra,
                "error_type": type(error).__name__, "error": str(error), "full_training_completed": False})
        raise
    whole_seconds = time.perf_counter() - started
    manifest_path = prepared.output_directory / "training_manifest.json"
    manifest = _read(manifest_path)
    expected_steps = math.ceil(len(prepared.train_indices) / prepared.config.batch_size) * prepared.config.epochs
    if (manifest["optimizer_steps"] != expected_steps or manifest["completed_epochs"] != 3
            or counts["training_batches"] != expected_steps
            or counts["training_examples"] != len(prepared.train_indices) * 3
            or counts["validation_examples"] != len(prepared.validation_indices) * 3):
        raise ValueError("Candidate did not complete the exact three-epoch training/validation schedule")
    evaluation = evaluate_candidate(prepared)
    if not np.isclose(evaluation["macro_f1"], manifest["final_validation"]["macro_f1"], atol=1e-12):
        raise ValueError("Recomputed validation metrics differ from the trainer")
    _write(prepared.output_directory / "evaluation.json", evaluation)
    result = {**extra, "target": "Mnfr", "model_identity": prepared.model_identity,
        "output_directory": str(prepared.output_directory), "config": report["config"],
        "label_names": list(LABELS), "label_to_id": report["label_to_id"],
        "input_fields": report["input_fields"], "sequence_length": 256,
        "representation": report["representation"], "split_identity": report["split_identity"],
        "initial_state_sha256": manifest["initial_state_sha256"], "final_state_sha256": manifest["final_state_sha256"],
        "checkpoint_sha256": sha256(prepared.output_directory / "checkpoint.pt"),
        "candidate_wall_seconds_including_preflight": whole_seconds,
        "trainer_runtime_seconds": manifest["runtime_seconds"],
        "epoch_training_and_validation_seconds": float(pd.read_csv(prepared.output_directory / "training_history.csv").epoch_seconds.sum()),
        "timing_scope": "Trainer runtime includes initialization, training, validation and CPU state hashing; excludes final artifact writing. Candidate wall time additionally includes preflight and saving.",
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loss_call_counts": counts, "evaluation": evaluation,
        "optimizer_steps": manifest["optimizer_steps"], "completed_epochs": 3,
        "test_csv_read_or_hashed": False, "full_training_completed": True,
        "implementation_sha256": sha256(Path(__file__))}
    _write(prepared.output_directory / "candidate_result.json", result)
    manifest.update(extra, full_training_completed=True)
    manifest["artifacts"] = {path.name: sha256(path) for path in prepared.output_directory.iterdir()
        if path.is_file() and path.name != "training_manifest.json"}
    _write(manifest_path, manifest)
    return result


def choose_candidate(unweighted, weighted, tolerance=TIE_TOLERANCE):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    first, second = unweighted["evaluation"]["macro_f1"], weighted["evaluation"]["macro_f1"]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in (first, second)):
        raise ValueError("Invalid candidate macro-F1")
    return "weighted_1_5" if second > first + tolerance else "unweighted"


def select_and_package(experiment_directory):
    """Designate exactly one final checkpoint in a local Maria handoff bundle."""
    import torch
    root = Path(experiment_directory).resolve()
    results = {}
    for name in WEIGHTS:
        paths = list((root / name / "mnfr").glob("*/candidate_result.json"))
        if len(paths) != 1:
            raise ValueError("Exactly one completed result per candidate is required")
        results[name] = _read(paths[0])
    baseline, weighted = results.values()
    for key in ("config", "label_names", "input_fields", "representation", "split_identity", "initial_state_sha256"):
        if baseline[key] != weighted[key]:
            raise ValueError(f"Candidate inputs/configuration differ: {key}")
    if baseline["evaluation"]["validation_record_ids_sha256"] != weighted["evaluation"]["validation_record_ids_sha256"]:
        raise ValueError("Candidates evaluated different validation records")
    selected = choose_candidate(baseline, weighted)
    winner = results[selected]
    source = Path(winner["output_directory"])
    if sha256(source / "checkpoint.pt") != winner["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after evaluation")
    destination = root / "maria_handoff"
    if destination.exists():
        raise FileExistsError("Handoff already exists; do not overwrite it")
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (checkpoint["label_names"] != list(LABELS)
            or _state_sha256(checkpoint["state_dict"]) != winner["final_state_sha256"]):
        raise ValueError("Selected checkpoint labels/state do not match the evaluated candidate")
    metadata = {key: winner[key] for key in ("target", "model_identity", "label_names", "label_to_id", "input_fields",
        "sequence_length", "representation", "split_identity", "config", "loss_configuration", "candidate")}
    selection = {"selected_candidate": selected, "metric": "validation macro_f1", "tie_tolerance": TIE_TOLERANCE,
        "tie_preference": "unweighted", "epoch_selection": "fixed final third epoch for both candidates",
        "weighted_minus_unweighted_macro_f1": weighted["evaluation"]["macro_f1"] - baseline["evaluation"]["macro_f1"],
        "candidates": results, "protected_test_used": False, "external_delivery_performed": False}
    destination.mkdir()
    torch.save({**checkpoint, **metadata, "selection": {k: v for k, v in selection.items() if k != "candidates"}}, destination / "checkpoint.pt")
    for name in ("encoder_config.json", "model_config.json", "training_manifest.json", "evaluation.json"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "tokenizer", destination / "tokenizer")
    (destination / "README.md").write_text(
        "# Selected Mnfr MiniLM checkpoint for Maria\n\n"
        f"Selected candidate: {selected}. Target: Mnfr; All others = 0, J&J = 1.\n\n"
        "Use ProductCategory, ProductName, ProductBrand, ProductDescription, ProductContents "
        "in that order, with the existing HTML/whitespace preprocessing and character limits "
        "recorded in manifest.json. The combined token limit is 256. Architecture is the pinned "
        "MiniLM encoder plus attention-mask mean pooling, dropout and a two-class linear head. "
        "Use ordinary softmax/argmax; training weights are not applied to predictions.\n\n"
        "This is an experimental development-selected model, not protected-test certification. "
        "No protected test evaluation was performed. Both experiment results are retained in "
        "the parent directory; only this checkpoint is designated for handoff. No external "
        "delivery was performed. Source training_manifest.json refers to the original experiment "
        "artifacts; manifest.json contains checksums for this bundle.\n", encoding="utf8")
    artifact_hashes = {str(path.relative_to(destination)).replace("\\", "/"): sha256(path)
        for path in destination.rglob("*") if path.is_file()}
    _write(destination / "manifest.json", {**metadata, "selection": selection,
        "source_checkpoint_sha256": winner["checkpoint_sha256"], "artifact_sha256": artifact_hashes})
    _write(root / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "select"))
    parser.add_argument("--experiment-directory", required=True)
    parser.add_argument("--candidate", choices=tuple(WEIGHTS))
    parser.add_argument("--training-authorized", action="store_true")
    args = parser.parse_args()
    if args.action == "select":
        selection = select_and_package(args.experiment_directory)
        print(json.dumps({key: value for key, value in selection.items() if key != "candidates"}, indent=2), flush=True)
        return
    if not args.training_authorized or args.candidate is None:
        parser.error("train requires --candidate and explicit --training-authorized")
    root = Path.cwd()
    prepared = prepare_target_minilm(root / "data/processed/split_integration_reproduced_20261010", "Mnfr",
        Path(args.experiment_directory) / args.candidate)
    approved = _read(root / "data/processed/mnfr_minilm_gpu_benchmark_v1_20261029/mnfr/minilm-mnfr-category256-c3458037a670c8f0/benchmark_report.json")
    if json.loads(json.dumps(prepared.config.to_dict())) != approved["config"]:
        raise ValueError("Configuration differs from the approved GPU benchmark")
    if (prepared.model_identity != approved["model_identity"]
            or prepared.audit["ordered_supervised_membership_sha256"] != approved["ordered_supervised_membership_sha256"]):
        raise ValueError("Mnfr inputs differ from the approved GPU benchmark")
    result = run_candidate(prepared, args.candidate, training_authorized=True,
        progress=lambda message: print(message, flush=True))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
