"""Reusable classical classifiers and validation handoffs for six CatalogIQ targets."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from threadpoolctl import threadpool_info, threadpool_limits

from .classical_data import TARGETS, load_target_development
from .cleaning import PROVENANCE
from .features import sha256
from .segment_baseline import BaselineConfig, build_baseline_texts, make_pipeline
from .segment_baseline_data import MISSING_LABELS
from .segment_transformer import MODEL_FIELDS, _hash_items
from .splitting import check_assignments


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def source_fingerprints():
    """Raw hashes explain history; LF-normalized hashes compare across platforms."""
    names = ("classical_baseline.py", "classical_data.py", "segment_baseline.py",
             "segment_baseline_data.py", "segment_transformer.py", "split_protocol.py")
    result = {}
    for name in names:
        data = Path(__file__).with_name(name).read_bytes()
        result[name] = {
            "sha256_raw": hashlib.sha256(data).hexdigest(),
            "sha256_lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
            "crlf_count": data.count(b"\r\n"),
        }
    return result


def hardware_info(threads):
    return {"device": "cpu", "gpu_used": False, "os": platform.system(),
            "os_release": platform.release(), "architecture": platform.machine(),
            "cpu_description": platform.processor() or None,
            "logical_cpus": os.cpu_count(), "numerical_thread_limit": threads,
            "threadpools": [{key: pool.get(key) for key in
                             ("internal_api", "num_threads", "version", "architecture")}
                            for pool in threadpool_info()]}


def predict_classical(bundle, frame):
    """Predict new product rows without requiring any ground-truth target columns.

    Only load joblib artifacts from a trusted teammate: joblib uses pickle.
    Timing this function includes text formatting, TF-IDF and classification.
    """
    required = {"record_id", *MODEL_FIELDS}
    if bundle["include_category"]:
        required.add("ProductCategory")
    if not required <= set(frame):
        raise ValueError("Prediction input lacks record_id or required product fields")
    if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any() or frame.record_id.duplicated().any():
        raise ValueError("Prediction record IDs must be nonblank and unique")
    mapping = bundle["label_mapping"]
    probability_columns = [f"probability_{i}" for i in range(len(mapping))]
    names = [mapping[column] for column in probability_columns]
    if not np.array_equal(bundle["pipeline"].classes_, np.arange(len(names))):
        raise ValueError("Saved probability order differs from label_mapping")
    texts = build_baseline_texts(frame, include_category=bundle["include_category"])
    probabilities = bundle["pipeline"].predict_proba(texts)
    out = frame[["record_id"]].reset_index(drop=True).copy()
    out["predicted_label"] = np.asarray(names)[probabilities.argmax(axis=1)]
    out["confidence"] = probabilities.max(axis=1)
    out = pd.concat([out, pd.DataFrame(probabilities, columns=probability_columns)], axis=1)
    return out


def measure_scores(true, predictions, probabilities, names):
    """Keep unseen validation labels as errors; never fit or invent their class."""
    true = np.asarray(true)
    predicted = np.asarray(predictions)
    known = np.isin(true, names)
    observed = sorted(set(true) | set(predicted))
    all_classes = sorted(set(names) | set(true))
    scores = {
        "records": len(true), "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "macro_f1_label_scope": "union of observed true and predicted validation labels (sklearn default)",
        "macro_f1_labels": observed,
        "macro_f1_all_training_and_validation_labels": float(f1_score(true, predicted, labels=all_classes, average="macro", zero_division=0)),
        "per_class": classification_report(true, predicted, labels=all_classes, zero_division=0, output_dict=True),
        "unseen_validation_labels": sorted(set(true[~known])),
        "unseen_validation_rows": int((~known).sum()),
        "log_loss": None, "multiclass_brier_sum": None,
    }
    if known.all():
        ids = pd.Series(true).map({name: i for i, name in enumerate(names)}).to_numpy(dtype=int)
        scores["log_loss"] = float(log_loss(ids, probabilities, labels=np.arange(len(names))))
        scores["multiclass_brier_sum"] = float(np.mean(((probabilities - np.eye(len(names))[ids]) ** 2).sum(axis=1)))
    return scores


def fit_target(frame, target, config=None, *, include_category=False,
               allow_unseen_validation=False, threads=2, latency_repeats=3, latency_batch_size=256):
    config = config or BaselineConfig()
    required = {*PROVENANCE, "record_id", "group_id", "split", target, *MODEL_FIELDS}
    if target not in TARGETS or not required <= set(frame) or set(frame.split) != {"train", "validation"}:
        raise ValueError("Provide a supported target and labeled train/validation records only")
    if any(type(value) is not int or value < 1 for value in (threads, latency_repeats, latency_batch_size)):
        raise ValueError("Threads, repeats and latency batch size must be positive integers")
    check_assignments(frame, frame.record_id)
    labels = frame[target].astype("string").str.strip()
    if frame[PROVENANCE].duplicated().any() or labels.isna().any() or labels.str.casefold().isin(MISSING_LABELS).any():
        raise ValueError("Duplicate keys or missing labels in supervised view")
    train = frame.split.eq("train")
    names = sorted(labels.loc[train].unique())
    if len(names) < 2:
        raise ValueError("Training requires at least two classes")
    unseen = sorted(set(labels.loc[~train]) - set(names))
    if unseen and not allow_unseen_validation:
        raise ValueError("Validation has classes absent from train; obtain the agreed target split or explicitly report them as errors")
    training_labels = labels.loc[train].map({name: i for i, name in enumerate(names)}).to_numpy(dtype=int)
    preparation_start = time.perf_counter()
    training_text = build_baseline_texts(frame.loc[train], include_category=include_category)
    formatting_seconds = time.perf_counter() - preparation_start
    model = make_pipeline(config)
    bundle = {"format": "catalogiq-classical-model-v1", "target": target, "pipeline": model,
              "label_mapping": {f"probability_{i}": name for i, name in enumerate(names)},
              "config": asdict(config), "include_category": include_category,
              "input_adapter": "catalogiq.classical_baseline.predict_classical"}
    validation = frame.loc[~train].reset_index(drop=True)
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        start = time.perf_counter()
        model.fit(training_text, training_labels)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = predict_classical(bundle, validation)
        validation_seconds = time.perf_counter() - start
        batch = validation.iloc[:latency_batch_size]
        predict_classical(bundle, batch)  # Explicit, excluded warm-up.
        timings = []
        for _ in range(latency_repeats):
            start = time.perf_counter()
            predict_classical(bundle, batch)
            timings.append(time.perf_counter() - start)
        hardware = hardware_info(threads)
    columns = list(bundle["label_mapping"])
    probabilities = predicted[columns].to_numpy(dtype=float)
    out = validation[[*PROVENANCE, "record_id", "group_id", "split"]].copy()
    out["true_label"] = labels.loc[~train].to_numpy()
    for column in predicted.columns.drop("record_id"):
        out[column] = predicted[column].to_numpy()
    baseline = measure_scores(out.true_label, out.predicted_label, probabilities, names)
    most_common = labels.loc[train].value_counts().index[0]
    majority_predictions = np.full(len(out), most_common)
    dummy = np.zeros_like(probabilities)
    dummy[:, names.index(most_common)] = 1
    metrics = {"target": target, "partition": "validation", "primary_metric": "macro_f1",
               "baseline": baseline, "majority_class": measure_scores(out.true_label, majority_predictions, dummy, names),
               "majority_class_selected_on_training": most_common,
               "confidence_status": "uncalibrated; no decision threshold selected"}
    median = float(np.median(timings))
    performance = {
        "training_seconds": training_seconds,
        "training_time_scope": "pipeline.fit on preformatted training text: TF-IDF vocabulary/IDF and logistic regression",
        "training_text_formatting_seconds": formatting_seconds,
        "validation_prediction_seconds": validation_seconds,
        "hardware": hardware,
        "inference_latency": {"batch_size": len(batch), "repeats": latency_repeats,
            "warmup_batches": 1, "batch_seconds": timings, "median_batch_seconds": median,
            "seconds_per_record": median / len(batch), "records_per_second": len(batch) / median,
            "scope": "warm in-memory batch: formatting + vectorization + predict_proba + output frame; excludes model load and disk IO",
            "sample": "first records in sorted validation order; labels unused for selection",
            "comparison_caveat": "Compare only matching hardware, thread limit, batch size and measurement scope"},
    }
    return bundle, out, metrics, performance


def _write_variant(directory, frame, target, config, audit, include_category, **options):
    bundle, predictions, metrics, performance = fit_target(frame, target, config,
                                                           include_category=include_category, **options)
    directory.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(directory / "predictions.csv", index=False, lineterminator="\n")
    predictions.loc[predictions.true_label.ne(predictions.predicted_label)].to_csv(
        directory / "validation_errors.csv", index=False, lineterminator="\n")
    matrix_labels = sorted(set(bundle["label_mapping"].values()) | set(predictions.true_label))
    matrix = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=matrix_labels)
    pd.DataFrame(matrix, index=pd.Index(matrix_labels, name="true_label"), columns=matrix_labels).to_csv(
        directory / "confusion_matrix.csv", lineterminator="\n")
    write_json(directory / "metrics.json", metrics)
    joblib.dump(bundle, directory / "model.joblib", compress=3)
    profile = audit["profiles"][target]
    manifest = {
        "schema_version": 1, "format": "catalogiq-classical-run-v1", "target": target,
        "model": "word TF-IDF + logistic regression", "variant": directory.name,
        "partition": "validation", "label_mapping": bundle["label_mapping"],
        "protocol_id": audit["protocol_id"], "protocol_manifest_sha256": audit["protocol_manifest_sha256"],
        "split_allocation_target": audit["allocation_target"],
        "training_rows": profile["training_rows"], "validation_rows": profile["validation_rows"],
        "training_record_ids_sha256": profile["training_record_ids_sha256"],
        "validation_record_ids_sha256": profile["validation_record_ids_sha256"],
        "training_class_counts": profile["training_class_counts"],
        "validation_class_counts": profile["validation_class_counts"],
        "missing_labels_excluded": profile["missing_labels_excluded"],
        "unseen_validation_labels": profile["unseen_validation_labels"],
        "unseen_validation_rows": profile["unseen_validation_rows"],
        "calibration_requires_unseen_label_policy": bool(profile["unseen_validation_rows"]),
        "probabilities_calibrated": False, "test_opened": False, "new_splits_created": False,
        "input_fields": list(MODEL_FIELDS) + (["ProductCategory"] if include_category else []),
        "category_mode": "same text/vectorizer" if include_category else "excluded",
        "config": asdict(config), **performance,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "source_fingerprints": source_fingerprints(),
        "source_hash_normalization": "sha256_lf replaces CRLF bytes with LF only; sha256_raw preserves actual bytes",
        "metrics": metrics["baseline"],
        "artifacts": {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()},
    }
    write_json(directory / "manifest.json", manifest)
    return {"status": "completed", "metrics": metrics["baseline"],
            "training_seconds": performance["training_seconds"],
            "inference_latency": performance["inference_latency"]}


def run_classical(split_dir, output_dir, *, targets=("Segment",), config=None,
                  features="compare", train=False, protocol_dir=None,
                  allow_unseen_validation=False, threads=2, progress=print):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; previous artifacts are immutable")
    if features not in ("base", "category", "compare"):
        raise ValueError("Unknown feature variant")
    config = config or BaselineConfig()
    if progress:
        progress("Verifying shared exports and building target-specific supervised views.")
    views, audit = load_target_development(split_dir, targets, protocol_dir=protocol_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
               "model_training_performed": False, "test_opened": False, "new_splits_created": False,
               "features": features, "config": asdict(config), "audit": audit, "targets": {}}
    write_json(output / "summary.json", summary)
    variants = {"base": False, "category": True} if features == "compare" else {features: features == "category"}
    if train:
        for target, frame in views.items():
            summary["targets"][target] = {}
            for arm, category in variants.items():
                if progress:
                    profile = audit["profiles"][target]
                    progress(f"Fitting {target}/{arm}: {profile['training_rows']} train / {profile['validation_rows']} validation.")
                try:
                    summary["targets"][target][arm] = _write_variant(
                        output / target / arm, frame, target, config, audit, category,
                        allow_unseen_validation=allow_unseen_validation, threads=threads)
                except (ValueError, RuntimeError, Warning) as error:
                    summary["targets"][target][arm] = {"status": "failed", "error": str(error)}
                write_json(output / "summary.json", summary)
        outcomes = [run["status"] for arms in summary["targets"].values() for run in arms.values()]
        summary["status"] = "completed" if all(value == "completed" for value in outcomes) else "incomplete"
        summary["model_training_performed"] = any(value == "completed" for value in outcomes)
    summary["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256(p)
                                    for p in sorted(output.rglob("*")) if p.is_file() and p != output / "summary.json"}
    write_json(output / "summary.json", summary)
    return summary


def package_classical(run_dir, output_zip, *, include_models=False):
    """Package saved artifacts with verified hashes and the agreed per-arm manifest."""
    root, output = Path(run_dir).resolve(), Path(output_zip).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing handoff")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["test_opened"] is not False:
        raise ValueError("A completed validation run is required")
    files = {}
    allowed = {"predictions.csv", "manifest.json", "metrics.json", "validation_errors.csv", "confusion_matrix.csv"}
    if include_models:
        allowed.add("model.joblib")
    for target, variants in summary["targets"].items():
        if target not in TARGETS or not set(variants) <= {"base", "category"}:
            raise ValueError("Unexpected target or variant")
        for arm in variants:
            directory = root / target / arm
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["target"] != target or manifest["partition"] != "validation" or manifest["test_opened"] is not False:
                raise ValueError("Invalid prediction manifest")
            for filename in sorted(allowed):
                relative = f"{target}/{arm}/{filename}"
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or sha256(path) != summary["artifact_sha256"][relative]:
                    raise ValueError(f"Changed artifact: {relative}")
                files[relative] = path.read_bytes()
    files["README.txt"] = (
        "Classical CatalogIQ validation models\n"
        "Each <target>/<base|category>/ has predictions.csv and manifest.json.\n"
        "Use manifest.label_mapping for probability columns. True labels are target-specific.\n"
        "Unseen validation labels remain errors; inspect calibration_requires_unseen_label_policy\n"
        "before computing probabilistic metrics. Do not silently drop those records.\n"
        "Only validation is scored. These are not final-test results.\n"
        "Training seconds and warm batch inference timings include their measurement scope.\n"
        "Compare cost only with the same hardware, threads, batch size and timing boundaries.\n"
        "When models are included, load only these trusted teammate joblib files.\n"
        "Use catalogiq.classical_baseline.predict_classical(bundle, product_frame).\n"
    ).encode("utf-8")
    files["manifest.json"] = (json.dumps({"format": "catalogiq-classical-handoff-v1",
        "protocol_id": summary["audit"]["protocol_id"], "targets": list(summary["targets"]),
        "includes_models": include_models, "test_opened": False,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Handoff archive failed CRC verification")
    return {"archive": output.name, "sha256": sha256(output), "bytes": output.stat().st_size}
