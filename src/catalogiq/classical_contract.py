"""Revise classical evaluation metadata without refitting or rewriting predictions."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .classical_baseline import measure_scores, write_json
from .classical_data import load_target_development
from .features import sha256
from .segment_transformer import _hash_items


def align_classical_contract(run_dir, source_dir, output_dir, *, protocol_dir=None):
    """Create an independently validated revision; old outputs remain immutable."""
    root, source, output = map(lambda p: Path(p).resolve(), (run_dir, source_dir, output_dir))
    if output.exists():
        raise FileExistsError("Use a new contract revision directory")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["test_opened"] is not False:
        raise ValueError("A completed validation-only run is required")
    targets = tuple(summary["targets"])
    # This upgrader is deliberately restricted to original V4 allocations.
    # Reallocated targets must be exported through the guarded allocation trainer.
    views, audit = load_target_development(source, targets, protocol_dir=protocol_dir)
    registry_path = source / "protected_final_test.csv"
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False, usecols=["record_id", "group_id"])
    if sha256(registry_path) != audit["protected_registry_source_sha256"]:
        raise ValueError("Protected registry changed")
    prepared = []
    for target, variants in summary["targets"].items():
        for arm, result in variants.items():
            if arm not in {"base", "category"} or result["status"] != "completed":
                raise ValueError("Invalid run variant")
            directory = root / target / arm
            manifest_path = directory / "manifest.json"
            relative = manifest_path.relative_to(root).as_posix()
            if sha256(manifest_path) != summary["artifact_sha256"][relative]:
                raise ValueError("Changed original manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["target"] != target or manifest["partition"] != "validation" or manifest["test_opened"] is not False:
                raise ValueError("Wrong original prediction context")
            allowed = {"predictions.csv", "model.joblib", "metrics.json", "validation_errors.csv", "confusion_matrix.csv"}
            for name in allowed:
                path = directory / name
                if sha256(path) != manifest["artifacts"][name] or sha256(path) != summary["artifact_sha256"][f"{target}/{arm}/{name}"]:
                    raise ValueError(f"Changed original artifact: {target}/{arm}/{name}")
            frame = pd.read_csv(directory / "predictions.csv", dtype=str, keep_default_na=False)
            expected = views[target].loc[views[target].split.eq("validation")].reset_index(drop=True)
            if (frame.record_id.tolist() != expected.record_id.tolist()
                    or not frame.record_id.is_unique or not frame.split.eq("validation").all()
                    or len(frame) != manifest["validation_rows"]
                    or _hash_items(frame.record_id.tolist()) != manifest["validation_record_ids_sha256"]):
                raise ValueError("Wrong validation identities or fingerprint")
            for column in ("group_id", "dataset", "source_sha256", "source_row"):
                if frame[column].tolist() != expected[column].tolist():
                    raise ValueError(f"Prediction context mismatch: {column}")
            if frame.true_label.tolist() != expected[target].tolist():
                raise ValueError("Prediction truth mismatch")
            mapping = manifest["label_mapping"]
            columns = [f"probability_{i}" for i in range(len(mapping))]
            names = [mapping[c] for c in columns]
            if set(mapping) != set(columns) or len(set(names)) != len(names) or set(names) != set(audit["profiles"][target]["training_class_counts"]):
                raise ValueError("Class mapping differs from training vocabulary")
            probabilities = frame[columns].to_numpy(dtype=float)
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)
                    or not np.array_equal(np.asarray(names)[probabilities.argmax(axis=1)], frame.predicted_label)
                    or not np.allclose(probabilities.max(axis=1), frame.confidence.astype(float), atol=1e-8)):
                raise ValueError("Invalid probabilities, class order or confidence")
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            before = metrics["baseline"]
            scores = measure_scores(frame.true_label, frame.predicted_label, probabilities, names)
            if not np.isclose(before["accuracy"], scores["accuracy"], atol=1e-12):
                raise ValueError("Original accuracy cannot be reproduced")
            majority = metrics["majority_class_selected_on_training"]
            if majority not in names or majority != views[target].loc[views[target].split.eq("train"), target].value_counts().index[0]:
                raise ValueError("Majority class differs from training")
            dummy = np.zeros_like(probabilities)
            dummy[:, names.index(majority)] = 1
            metrics.update(baseline=scores, majority_class=measure_scores(frame.true_label,
                np.full(len(frame), majority), dummy, names))
            manifest.update(metrics=scores, macro_f1_class_set="model_vocabulary_union_truth",
                out_of_vocabulary_true_labels=scores["unseen_validation_rows"], out_of_vocabulary_policy="retained_as_errors",
                split_allocation_target=audit["allocation_id"], source_completion_sha256=audit["source_completion_sha256"],
                source_assignments_sha256=audit["source_assignments_sha256"],
                training_seconds_scope=manifest["training_time_scope"], inference_latency_ms=None,
                inference_latency_scope="single-request latency, unmeasured",
                amortized_inference_latency_ms=1000 * manifest["inference_latency"]["seconds_per_record"],
                amortized_inference_latency_scope=manifest["inference_latency"]["scope"],
                source_hash_normalization="raw_bytes",
                source_fingerprint_normalization="sha256_lf replaces CRLF bytes with LF only; sha256_raw preserves training bytes",
                previous_manifest_sha256=sha256(manifest_path),
                contract_revision={"date": "2026-10-31", "training_performed": False,
                    "inference_performed": False, "predictions_rewritten": False,
                    "implementation_sha256": sha256(Path(__file__))})
            prepared.append((target, arm, directory, manifest, metrics, allowed))
    output.mkdir(parents=True, exist_ok=False)
    for target, arm, old, manifest, metrics, allowed in prepared:
        new = output / target / arm
        new.mkdir(parents=True)
        for name in allowed - {"metrics.json"}:
            shutil.copyfile(old / name, new / name)
            if sha256(new / name) != manifest["artifacts"][name]:
                raise ValueError("Artifact changed while copying")
        write_json(new / "metrics.json", metrics)
        registry.to_csv(new / "protected_test_identity_registry.csv", index=False, lineterminator="\n")
        manifest["artifacts"] = {p.name: sha256(p) for p in new.iterdir() if p.is_file()}
        write_json(new / "manifest.json", manifest)
        summary["targets"][target][arm]["metrics"] = metrics["baseline"]
    summary.update(audit=audit, contract_revision="2026-10-31", metadata_revision_only=True,
                   revision_training_performed=False, revision_inference_performed=False,
                   previous_summary_sha256=sha256(root / "summary.json"))
    summary["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "summary.json", summary)
    return summary
