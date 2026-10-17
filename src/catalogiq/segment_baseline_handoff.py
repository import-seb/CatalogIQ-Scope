"""Verify and package a completed classical comparison for private evaluation.

Reuses the frozen development guard; never opens test.csv or loads a model.
Only explicitly allowlisted prediction artifacts enter the deterministic ZIP.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .cleaning import PROVENANCE
from .segment_baseline import score_predictions
from .segment_baseline_data import load_official_development
from .segment_transformer import _hash_items


ARMS = ("base", "category")
FILES = ("predictions.csv", "label_mapping.json", "metrics.json",
         "validation_errors.csv", "confusion_matrix.csv")
SCORES = ("accuracy", "macro_f1", "weighted_f1")
IDENTITY = [*PROVENANCE, "record_id", "group_id", "split"]


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact is outside the run directory: {relative}")
    return path.read_bytes()


def _csv(content, **kwargs):
    return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, **kwargs)


def _same_rows(actual, expected, description):
    if (list(actual.columns) != list(expected.columns)
            or actual.record_id.duplicated().any()
            or set(actual.record_id) != set(expected.record_id)):
        raise ValueError(f"{description}: columns or record coverage differ")
    a = actual.set_index("record_id").sort_index()
    b = expected.set_index("record_id").sort_index()
    if not a.equals(b):
        raise ValueError(f"{description}: values differ from verified validation records")


def _check_scores(actual, expected, description):
    for key in SCORES:
        if not np.isclose(float(actual[key]), expected[key], rtol=0, atol=1e-12):
            raise ValueError(f"{description}: {key} disagrees with predictions")


def package_baseline_handoff(run_dir, split_dir, output_dir):
    """Validate both saved arms against sealed V4 development and write a new bundle.

    This is an integrity check of existing predictions, not a new model-quality
    experiment. It does not fit, calibrate, select a model or score the final test.
    """
    run_dir, split_dir, output_dir = (Path(p).resolve() for p in (run_dir, split_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; existing artifacts are never overwritten")
    if any(output_dir.is_relative_to(p) for p in (run_dir, split_dir)):
        raise ValueError("Output must be outside the source run and split directories")
    summary_bytes = _read(run_dir, "summary.json")
    summary = json.loads(summary_bytes)
    if (summary.get("status") != "completed" or summary.get("model_training_performed") is not True
            or summary.get("test_scored") is not False or summary.get("new_splits_created") is not False
            or summary.get("target") != "Segment" or set(summary.get("variants", {})) != set(ARMS)):
        raise ValueError("A completed, validation-only base/category comparison is required")
    contents = {}
    for name in ("protocol.json", "comparison.csv", *(f"{a}/{f}" for a in ARMS for f in FILES)):
        data = _read(run_dir, name)
        if _digest(data) != summary["artifact_sha256"].get(name):
            raise ValueError(f"Artifact checksum mismatch: {name}")
        contents[name] = data
    protocol = json.loads(contents["protocol.json"])
    if protocol.get("target") != "Segment" or protocol.get("features") != "compare":
        raise ValueError("Expected a Segment comparison protocol")
    development = load_official_development(split_dir)
    audit = development.audit
    guarded = audit["guarded_development"]
    if protocol["development"]["protocol_id"] != audit["protocol_id"]:
        raise ValueError("Run and split protocol IDs differ")
    for role in ("train", "validation"):
        count_key = "training_rows" if role == "train" else "validation_rows"
        if summary[count_key] != audit["labeled_development_counts"][role]:
            raise ValueError(f"Incorrect {role} record count")
        frame = development.frame.loc[development.frame.split.eq(role)]
        key = "training_record_ids_sha256" if role == "train" else "validation_record_ids_sha256"
        if protocol[key] != _hash_items(frame.record_id.tolist()):
            raise ValueError(f"Run and split {role} membership differ")
        before = protocol["development"]["guarded_development"]["consumed_artifact_sha256"][f"{role}.csv"]
        if before != guarded["consumed_artifact_sha256"][f"{role}.csv"]:
            raise ValueError(f"Run and split {role} export hashes differ")
    validation = development.frame.loc[development.frame.split.eq("validation")]
    expected = validation[IDENTITY].copy()
    expected["true_label"] = validation.Segment.to_numpy()
    # The baseline learns sorted training classes; synthetic fixtures may contain
    # only a subset of the guard's declared Segment taxonomy.
    names = sorted(development.frame.loc[development.frame.split.eq("train"), "Segment"].unique())
    if len(names) < 2 or not set(validation.Segment).issubset(names):
        raise ValueError("Validation classes must be represented in training")
    mapping = {f"probability_{i}": label for i, label in enumerate(names)}
    columns = [*IDENTITY, "true_label", "predicted_label", "confidence", *mapping]
    comparison = _csv(contents["comparison.csv"])
    if (list(comparison.columns) != ["variant", *SCORES]
            or len(comparison) != 2 or set(comparison.variant) != set(ARMS)):
        raise ValueError("Comparison must contain exactly base and category scores")
    results = {}
    for arm in ARMS:
        if json.loads(contents[f"{arm}/label_mapping.json"]) != mapping:
            raise ValueError(f"{arm}: probability label mapping differs from V4 classes")
        predictions = _csv(contents[f"{arm}/predictions.csv"])
        if list(predictions.columns) != columns:
            raise ValueError(f"{arm}: unexpected prediction columns")
        _same_rows(predictions[[*IDENTITY, "true_label"]], expected, f"{arm} predictions")
        probabilities = predictions[list(mapping)].to_numpy(dtype=float)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
            raise ValueError(f"{arm}: invalid probability distribution")
        predicted = np.asarray(names)[probabilities.argmax(axis=1)]
        if not np.array_equal(predicted, predictions.predicted_label.to_numpy()):
            raise ValueError(f"{arm}: predicted labels disagree with probabilities")
        if not np.allclose(predictions.confidence.to_numpy(dtype=float), probabilities.max(axis=1),
                           rtol=0, atol=1e-12):
            raise ValueError(f"{arm}: confidence disagrees with probabilities")
        labels = predictions.true_label.map({name: i for i, name in enumerate(names)}).to_numpy(dtype=int)
        measured = score_predictions(labels, probabilities, names)
        metrics = json.loads(contents[f"{arm}/metrics.json"])
        if metrics["partition"] != "validation" or metrics["label_names"] != names:
            raise ValueError(f"{arm}: wrong metric partition or class order")
        _check_scores(metrics["baseline"], measured, f"{arm} metrics")
        _check_scores(summary["variants"][arm]["validation"], measured, f"{arm} summary")
        _check_scores(comparison.set_index("variant").loc[arm], measured, f"{arm} comparison")
        errors = _csv(contents[f"{arm}/validation_errors.csv"])
        _same_rows(errors, predictions.loc[predictions.true_label.ne(predictions.predicted_label)],
                   f"{arm} error report")
        matrix = _csv(contents[f"{arm}/confusion_matrix.csv"], index_col=0)
        counts = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=names)
        if (matrix.index.tolist() != names or matrix.columns.tolist() != names
                or not np.array_equal(matrix.to_numpy(dtype=float), counts)):
            raise ValueError(f"{arm}: confusion matrix disagrees with predictions")
        results[arm] = {key: measured[key] for key in SCORES}
    # The original protocol contains local filesystem paths; publish only scoped metadata.
    protocol_hash = _digest(contents.pop("protocol.json"))
    manifest = {
        "format": "catalogiq-classical-evaluation-handoff-v1", "target": "Segment",
        "protocol_id": audit["protocol_id"], "partition": "validation",
        "training_records": summary["training_rows"], "validation_records_per_variant": len(validation),
        "validation_record_ids_sha256": protocol["validation_record_ids_sha256"],
        "source_protocol_sha256": protocol_hash, "source_summary_sha256": _digest(summary_bytes),
        "label_mapping": mapping, "metrics": results,
        "probabilities_calibrated": False, "model_training_performed": False,
        "test_opened": False, "new_splits_created": False,
        "checks": ["authoritative_development_seal", "source_key_and_group_alignment",
                   "validation_label_alignment", "class_order_and_probabilities",
                   "headline_metrics", "error_rows_and_confusion_counts", "artifact_checksums"],
        "files_sha256": {name: _digest(data) for name, data in contents.items()},
    }
    contents["manifest.json"] = _json_bytes(manifest)
    contents["README.txt"] = (
        "CatalogIQ classical Segment baseline - private validation handoff\n\n"
        "base/ uses the four agreed product fields. category/ additionally concatenates\n"
        "ProductCategory into the same TF-IDF document. Both use logistic regression.\n"
        "Compare on record_id, or join dataset + source_sha256 + source_row one-to-one.\n"
        "Use label_mapping.json for probability-column order; do not join by row position.\n"
        "Probabilities are uncalibrated. No confidence threshold or model was selected.\n"
        "Only validation predictions are included. Do not use this as final-test evidence.\n"
        "manifest.json contains protocol identity, checksums and verified headline scores.\n"
        "Model binaries, raw product text, training data, test data and local paths are omitted.\n"
        "Checksums detect accidental changes; they are not digital signatures.\n"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data, compresslevel=9)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    result = {"status": "completed", **{key: manifest[key] for key in
              ("protocol_id", "validation_records_per_variant", "model_training_performed", "test_opened")},
              "metrics": results, "archive": "evaluation.zip", "archive_bytes": len(payload),
              "archive_sha256": _digest(payload), "files": sorted(contents)}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evaluation.zip").write_bytes(payload)
    (output_dir / "summary.json").write_bytes(_json_bytes(result))
    return result
