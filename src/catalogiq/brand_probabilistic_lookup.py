"""Separate training-frequency Brand baseline; no neural fitting or inference."""
import argparse
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from .features import sha256
from .maria_harness_compatibility import HARNESS_NOTEBOOK_SHA256, load_with_maria_harness
from .minilm_prediction_export import prepare_export, read_json, write_json, validate_predictions, verify_snapshots
from .segment_transformer import _hash_items
from .target_minilm_experiment import MISSING_FEATURE, brand_key, classification_metrics


VERSION = "brand-productbrand-empirical-prior-v1"
ALPHA = 1.0
CALIBRATION_FUNCTIONS = ("ece_and_reliability", "multiclass_brier", "temperature_scale")
CALIBRATION_SHA256 = "22fe29f9151f43316e6f20c4f1ff97b08ceff83c57e21233740d4beccec35fbe"
SMOOTHING = {
    "method": "Dirichlet posterior predictive centered on the empirical training class prior",
    "alpha": ALPHA, "formula": "P(c|key) = (n_key_c + alpha * prior_c) / (n_key + alpha)",
    "prior_formula": "prior_c = labeled_training_class_count_c / labeled_training_records",
    "parameter_selection": "Unit prior strength fixed before validation evaluation; no parameter search. Only training counts estimate the prior and conditional distributions.",
    "fallback": "Full training class-prior distribution for missing or unseen normalized ProductBrand",
}


def check_population(frame, role, labels, protected_ids=(), protected_groups=()):
    required = {"record_id", "group_id", "split", "ProductBrand", "Brand"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Brand population requires nonempty records, groups, roles, features and labels")
    if (frame.record_id.isna().any() or frame.record_id.str.strip().eq("").any() or frame.record_id.duplicated().any()
            or frame.group_id.isna().any() or frame.group_id.str.strip().eq("").any()
            or not frame.split.eq(role).all() or not frame.Brand.isin(labels).all()):
        raise ValueError("Invalid Brand population identity, role or vocabulary")
    if set(frame.record_id) & set(protected_ids) or set(frame.group_id) & set(protected_groups):
        raise ValueError("Protected-test record/group in development population")


def fit_frequency_lookup(training, labels, *, protected_ids=(), protected_groups=()):
    """The estimator accepts training records only; alpha is fixed, not searched."""
    started = time.perf_counter()
    labels = tuple(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("An ordered unique vocabulary is required")
    check_population(training, "train", labels, protected_ids, protected_groups)
    class_counts = training.Brand.value_counts().reindex(labels, fill_value=0).to_numpy(dtype=np.int64)
    if (class_counts <= 0).any():
        raise ValueError("Every supported class requires training records")
    label_to_id = {label: i for i, label in enumerate(labels)}
    keys = training.ProductBrand.map(brand_key)
    valid = ~keys.isin(MISSING_FEATURE)
    table = pd.DataFrame({"key": keys.loc[valid], "label_id": training.loc[valid, "Brand"].map(label_to_id)})
    counts = table.groupby(["key", "label_id"], sort=True).size()
    sparse = {}
    for (key, label_id), count in counts.items():
        sparse.setdefault(key, []).append([int(label_id), int(count)])
    model = {"format": VERSION, "target": "Brand", "label_names": list(labels),
        "smoothing": dict(SMOOTHING), "training_class_counts": class_counts.tolist(),
        "class_prior": (class_counts / class_counts.sum()).tolist(), "key_counts": sparse,
        "normalization": "Existing brand_key: plain-text cleanup, Unicode NFKC, casefold, strip",
        "missing_normalized_values": sorted(MISSING_FEATURE), "training_rows": len(training),
        "training_nonmissing_feature_records": int(valid.sum()),
        "training_record_ids_sha256": _hash_items(training.record_id.tolist()),
        "training_group_ids_sha256": _hash_items(training.group_id.tolist()),
        "validation_labels_used_to_estimate_probabilities": False,
        "validation_used_for_smoothing_selection": False}
    return model, time.perf_counter() - started


def predict_frequency_lookup(model, feature_values):
    """Return genuine smoothed frequency estimates and disjoint lookup cohorts."""
    labels = model["label_names"]
    prior = np.asarray(model["class_prior"], dtype=float)
    if (model["smoothing"] != SMOOTHING or prior.shape != (len(labels),)
            or not np.isfinite(prior).all() or (prior <= 0).any()
            or not np.isclose(prior.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid lookup prior or smoothing specification")
    keys = pd.Series(list(feature_values), dtype=object).map(brand_key)
    missing = keys.isin(MISSING_FEATURE).to_numpy()
    covered = keys.isin(model["key_counts"]).to_numpy() & ~missing
    probabilities = np.tile(prior, (len(keys), 1))
    cached = {}
    for row in np.flatnonzero(covered):
        key = keys.iloc[row]
        if key not in cached:
            pairs = np.asarray(model["key_counts"][key], dtype=np.int64)
            if (pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs[:, 1] <= 0).any()
                    or (pairs[:, 0] < 0).any() or (pairs[:, 0] >= len(labels)).any()
                    or len(set(pairs[:, 0])) != len(pairs)):
                raise ValueError("Invalid training key counts")
            estimate = ALPHA * prior.copy()
            estimate[pairs[:, 0]] += pairs[:, 1]
            cached[key] = estimate / (pairs[:, 1].sum() + ALPHA)
        probabilities[row] = cached[key]
    if (not np.isfinite(probabilities).all() or (probabilities <= 0).any() or (probabilities >= 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid smoothed probability distributions")
    return probabilities, {"seen_feature_values": covered, "missing_values": missing,
        "unseen_nonmissing_values": ~covered & ~missing, "all_fallback_records": ~covered}, keys


def evaluate_frequency_lookup(model, validation, deterministic, original_evaluation, *, protected_ids=(), protected_groups=()):
    labels = tuple(model["label_names"])
    check_population(validation, "validation", labels, protected_ids, protected_groups)
    if (deterministic["label_names"] != list(labels) or deterministic["training_record_ids_sha256"] != model["training_record_ids_sha256"]
            or set(deterministic["mapping"]) != set(model["key_counts"])
            or [deterministic["training_class_counts"][label] for label in labels] != model["training_class_counts"]):
        raise ValueError("Saved deterministic lookup differs from the approved training population")
    started = time.perf_counter()
    probabilities, masks, keys = predict_frequency_lookup(model, validation.ProductBrand)
    inferred = probabilities.argmax(axis=1)
    predicted_labels = np.asarray(labels)[inferred]
    predictions = pd.DataFrame({"record_id": validation.record_id.tolist(), "true_label": validation.Brand.tolist(),
        "predicted_label": predicted_labels, "confidence": probabilities.max(axis=1)})
    predictions = pd.concat([predictions, pd.DataFrame(probabilities, columns=[f"probability_{i}" for i in range(len(labels))])], axis=1)
    inference_seconds = time.perf_counter() - started
    encoding = {label: i for i, label in enumerate(labels)}
    truth = validation.Brand.map(encoding).to_numpy()
    old_labels = keys.map(deterministic["mapping"]).fillna(deterministic["fallback_label"])
    old_ids = old_labels.map(encoding).to_numpy()
    old_metrics = classification_metrics(truth, old_ids, labels)
    if (_hash_items(validation.record_id.tolist()) != original_evaluation["validation_record_ids_sha256"]
            or int(masks["seen_feature_values"].sum()) != original_evaluation["coverage_records"]
            or old_metrics["confusion_matrix"] != original_evaluation["full_validation"]["confusion_matrix"]):
        raise ValueError("Deterministic comparison does not reproduce its saved evaluation")
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        if not np.isclose(old_metrics[name], original_evaluation["full_validation"][name], atol=1e-12, rtol=0):
            raise ValueError("Deterministic saved metrics differ")
    metrics = classification_metrics(truth, inferred, labels)
    cohorts = {}
    for name, mask in masks.items():
        cohorts[name] = {"records": int(mask.sum()), "fraction": float(mask.mean()),
            "probabilistic": classification_metrics(truth[mask], inferred[mask], labels) if mask.any() else None,
            "deterministic": classification_metrics(truth[mask], old_ids[mask], labels) if mask.any() else None,
            "mean_confidence": float(probabilities[mask].max(axis=1).mean()) if mask.any() else None}
    changed = old_ids != inferred
    differences = pd.DataFrame({"record_id": validation.record_id.tolist(), "true_label": validation.Brand.tolist(),
        "normalized_ProductBrand": keys, "deterministic_label": old_labels, "probabilistic_label": predicted_labels,
        "old_correct": old_ids == truth, "new_correct": inferred == truth})
    differences = differences.loc[changed].reset_index(drop=True)
    evaluation = {"probabilistic": metrics, "deterministic_reproduced": old_metrics,
        "coverage_records": int(masks["seen_feature_values"].sum()), "coverage_fraction": float(masks["seen_feature_values"].mean()),
        "fallback_records": int(masks["all_fallback_records"].sum()), "cohorts": cohorts,
        "changed_prediction_records": int(changed.sum()),
        "old_correct_new_wrong": int(((old_ids == truth) & (inferred != truth)).sum()),
        "old_wrong_new_correct": int(((old_ids != truth) & (inferred == truth)).sum()),
        "accuracy_delta": metrics["accuracy"] - old_metrics["accuracy"],
        "macro_f1_delta": metrics["macro_f1"] - old_metrics["macro_f1"],
        "original_majority_fallback_label": deterministic["fallback_label"],
        "probabilistic_fallback_argmax_label": labels[int(np.argmax(model["class_prior"]))],
        "includes_all_labeled_validation_records_and_catch_all_labels": True}
    return predictions, evaluation, differences, inference_seconds


def calibration_compatibility(run, notebook):
    """Run Maria's exact probability diagnostics; never apply Q to the export."""
    notebook = Path(notebook)
    if notebook.suffix == ".ipynb":
        if sha256(notebook) != HARNESS_NOTEBOOK_SHA256:
            raise ValueError("Unreviewed harness notebook")
        blocks = ["".join(c["source"]) for c in read_json(notebook)["cells"] if c["cell_type"] == "code"]
    else:
        blocks = [notebook.read_text(encoding="utf8")]
    definitions = {}
    for block in blocks:
        for node in ast.parse(block).body:
            if isinstance(node, ast.FunctionDef) and node.name in CALIBRATION_FUNCTIONS:
                if node.name in definitions:
                    raise ValueError("Duplicate calibration definition")
                definitions[node.name] = ast.get_source_segment(block, node)
    source = "\n\n".join(definitions[name] for name in CALIBRATION_FUNCTIONS)
    if hashlib.sha256(source.encode()).hexdigest() != CALIBRATION_SHA256:
        raise ValueError("Unreviewed harness calibration definitions")
    namespace = {"np": np, "pd": pd}
    exec(compile(source, str(notebook), "exec"), namespace)
    frame, labels = run["frame"], run["labels"]
    matrix = np.vstack(frame._probs)
    reliability, ece = namespace["ece_and_reliability"](frame.top1_prob, frame.correct)
    brier = namespace["multiclass_brier"](frame._probs.tolist(), frame.true_label, labels)
    temperature, scaled, keep = namespace["temperature_scale"](frame._probs.tolist(), frame.true_label, labels)
    _, scaled_ece = namespace["ece_and_reliability"](scaled.max(axis=1), frame.loc[keep, "correct"])
    if (not keep.all() or not np.array_equal(scaled.argmax(axis=1), matrix.argmax(axis=1))
            or not np.allclose(scaled.sum(axis=1), 1, atol=1e-12, rtol=0)
            or not np.isfinite([ece, brier, temperature, scaled_ece]).all()):
        raise ValueError("Maria's calibration diagnostics failed")
    return {"status": "compatible", "definitions_sha256": CALIBRATION_SHA256,
        "raw_ece": float(ece), "raw_multiclass_brier": float(brier),
        "temperature_diagnostic": temperature, "scaled_ece_diagnostic": float(scaled_ece),
        "rows_retained": int(keep.sum()), "temperature_argmax_unchanged": True,
        "raw_reliability_bins": reliability.to_dict(orient="records"),
        "export_probabilities_temperature_scaled": False,
        "scope": "Unchanged calibration definitions exercised for compatibility. Temperature is fitted and evaluated on these validation records, so scaled scores are diagnostic, not independent calibration evidence; no estimates or smoothing parameters were changed."}


def export_probabilistic_lookup(inputs, output_directory, minilm_directory, harness_notebook, *, calibration_source=None):
    if inputs.target != "Brand":
        raise ValueError("Probabilistic ProductBrand lookup supports Brand only")
    output_directory, minilm_directory = Path(output_directory).resolve(), Path(minilm_directory).resolve()
    if output_directory.exists():
        raise FileExistsError("Use a new probabilistic lookup directory")
    sources = [inputs.directory.resolve(), minilm_directory, *[Path(p).resolve().parent for p in inputs.snapshots]]
    if any(output_directory.is_relative_to(p) or p.is_relative_to(output_directory) for p in sources):
        raise ValueError("Lookup output must be separate from existing sources")
    verify_snapshots(inputs)
    source_hashes = dict(inputs.snapshots)
    for name in ("predictions.csv", "manifest.json"):
        source_hashes[minilm_directory / name] = sha256(minilm_directory / name)
    source_hashes[Path(__file__).resolve()] = sha256(Path(__file__))
    for name in ("target_minilm_experiment.py", "segment_transformer.py", "maria_harness_compatibility.py"):
        path = Path(__file__).resolve().parent / name
        source_hashes[path] = sha256(path)
    original_model = read_json(inputs.directory / "lookup_model.json")
    original_evaluation = read_json(inputs.directory / "lookup_baseline.json")
    reference = read_json(minilm_directory / "manifest.json")
    if reference["target"] != "Brand" or reference["model_identity"] != inputs.model_identity or tuple(reference["label_names"]) != inputs.labels:
        raise ValueError("Brand MiniLM reference identity/vocabulary differs")
    if sha256(minilm_directory / "predictions.csv") != reference["files_sha256"]["predictions.csv"]:
        raise ValueError("Brand MiniLM predictions changed")
    mini = pd.read_csv(minilm_directory / "predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    validate_predictions(mini, inputs.validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
    training, validation = inputs.frame.iloc[inputs.train_indices], inputs.frame.iloc[inputs.validation_indices]
    if set(training.group_id) & set(validation.group_id):
        raise ValueError("Training and validation groups overlap")
    model, fit_seconds = fit_frequency_lookup(training, inputs.labels, protected_ids=inputs.protected.record_id, protected_groups=inputs.protected.group_id)
    canonical = json.dumps(model, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    identity = VERSION + "-" + hashlib.sha256(canonical).hexdigest()[:16]
    predictions, evaluation, differences, inference_seconds = evaluate_frequency_lookup(model, validation, original_model, original_evaluation,
        protected_ids=inputs.protected.record_id, protected_groups=inputs.protected.group_id)
    predictions = predictions.sort_values("record_id", kind="stable").reset_index(drop=True)
    if predictions.record_id.tolist() != mini.record_id.tolist() or predictions.true_label.tolist() != mini.true_label.tolist():
        raise ValueError("Probabilistic lookup differs from exact ordered Brand MiniLM validation population")
    checks = validate_predictions(predictions, inputs.validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
    context = inputs.validation.set_index("record_id").loc[predictions.record_id].reset_index()
    whole_ids = inputs.assignments.loc[inputs.assignments.split.eq("validation"), "record_id"]
    hardware = {"device": "cpu", "cpu_model": platform.processor() or None, "machine": platform.machine() or None,
        "platform": platform.platform(), "python_version": platform.python_version(), "gpu_used": False, "gpu_name": None}
    manifest = {"format": "catalogiq-probabilistic-brand-lookup-validation-v1", "target": "Brand",
        "model": "Training ProductBrand conditional frequencies with empirical-prior Dirichlet smoothing",
        "model_identity": identity, "label_names": list(inputs.labels),
        "label_to_id": {label: i for i, label in enumerate(inputs.labels)},
        "label_mapping": {f"probability_{i}": label for i, label in enumerate(inputs.labels)},
        "input_fields": ["ProductBrand"], "representation": {"normalization": model["normalization"], "missing_values": sorted(MISSING_FEATURE), "token_budget": None},
        "feature_contract": "Authorized retailer-supplied ProductBrand for raw-to-curated Brand normalization",
        "probability_estimation": SMOOTHING, "probabilities_calibrated": False,
        "confidence_semantics": "Maximum training-estimated conditional probability; no claim of calibration",
        "split_identity": inputs.split_identity, "partition": "validation", "training_rows": len(training), "validation_rows": len(predictions),
        "validation_record_ids_sha256": _hash_items(predictions.record_id.tolist()),
        "validation_record_ids_hash_recipe": "CSV record_id order: UTF-8 bytes prefixed by their 8-byte big-endian length, SHA-256; CSV rows sorted by record_id",
        "original_validation_record_ids_sha256": _hash_items(validation.record_id.tolist()),
        "whole_validation_partition_record_ids_sha256": _hash_items(sorted(whole_ids)),
        "training_record_ids_sha256": model["training_record_ids_sha256"], "training_group_ids_sha256": model["training_group_ids_sha256"],
        "training_seconds": fit_seconds, "runtime_seconds": fit_seconds,
        "training_seconds_scope": "CPU validation of training records, frequency aggregation and in-memory model construction; excludes source loading and file writing",
        "hardware": {"training": hardware, "inference": hardware},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": version("scikit-learn")},
        "inference": {"device": "cpu", "validation_inference_seconds": inference_seconds,
            "scope": "Key normalization, probability construction, argmax and prediction-frame assembly; excludes validation scoring and file writing"},
        "inference_latency_ms": None, "inference_latency_scope": "Single-request latency unmeasured",
        "amortized_inference_latency_ms_per_record": 1000 * inference_seconds / len(predictions),
        "coverage_records": evaluation["coverage_records"], "coverage_fraction": evaluation["coverage_fraction"],
        "fallback_records": evaluation["fallback_records"], "evaluation": {key: evaluation["probabilistic"][key] for key in ("accuracy", "macro_f1", "macro_f1_validation_supported_classes")},
        "deterministic_comparison": {key: evaluation[key] for key in ("changed_prediction_records", "accuracy_delta", "macro_f1_delta")},
        "original_deterministic_artifacts_unchanged": True, "neural_models_trained": False,
        "validation_used_for_estimation_or_parameter_selection": False, "protected_test_labels_read": False,
        "test_csv_read_or_hashed": False, "checks": checks, "minilm_alignment": {"rows": len(mini), "exact_order_and_truth": True,
            "predictions_sha256": source_hashes[minilm_directory / "predictions.csv"]},
        "source_input_sha256": {str(p): digest for p, digest in source_hashes.items()}}
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".brand-probabilistic-", dir=output_directory.parent))
    try:
        predictions.to_csv(stage / "predictions.csv", index=False, lineterminator="\n")
        context.to_csv(stage / "validation_context.csv", index=False, lineterminator="\n")
        differences.to_csv(stage / "prediction_differences.csv", index=False, lineterminator="\n")
        write_json(stage / "model.json", model)
        write_json(stage / "evaluation.json", evaluation)
        write_json(stage / "label_mapping.json", manifest["label_mapping"])
        per_class = evaluation["probabilistic"]["per_class"]
        pd.DataFrame([{"label": label, **per_class[label]} for label in inputs.labels]).to_csv(stage / "per_class_metrics.csv", index=False, lineterminator="\n")
        matrix = pd.DataFrame(evaluation["probabilistic"]["confusion_matrix"], index=inputs.labels, columns=inputs.labels)
        matrix.to_csv(stage / "confusion_matrix.csv", index_label="true_label", lineterminator="\n")
        write_json(stage / "manifest.json", manifest)
        restored = pd.read_csv(stage / "predictions.csv", dtype={"record_id": str}, keep_default_na=False)
        validate_predictions(restored, inputs.validation, inputs.labels, inputs.protected.record_id, inputs.protected.group_id)
        run, loader_report = load_with_maria_harness(stage, harness_notebook, inputs.validation, inputs.protected.record_id, whole_ids)
        if tuple(run["labels"]) != inputs.labels or not np.isclose(loader_report["metrics"]["macro_f1_full_vocabulary"], evaluation["probabilistic"]["macro_f1"], atol=1e-12, rtol=0):
            raise ValueError("Maria loader/scoring differs from the baseline")
        calibration = calibration_compatibility(run, calibration_source or harness_notebook)
        write_json(stage / "harness_compatibility.json", {"loader_and_scoring": loader_report, "calibration": calibration})
        manifest["harness_compatibility"] = {"status": "compatible", "rows_dropped": 0,
            "harness_commit": loader_report["harness_commit"], "calibration_definitions_sha256": CALIBRATION_SHA256}
        (stage / "README.md").write_text(reproducibility_text(manifest, evaluation, calibration), encoding="utf8", newline="\n")
        manifest["files_sha256"] = {p.name: sha256(p) for p in stage.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(stage / "manifest.json", manifest)
        for path, expected in source_hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Source changed during lookup export: {path}")
        for path in stage.iterdir():
            if path.suffix in (".csv", ".json", ".md") and b"\r" in path.read_bytes():
                raise ValueError("Export text artifacts must use LF")
        stage.rename(output_directory)
    except Exception:
        if stage.parent.resolve() != output_directory.parent or not stage.name.startswith(".brand-probabilistic-"):
            raise RuntimeError("Refusing cleanup outside the lookup staging directory")
        shutil.rmtree(stage)
        raise
    return {"directory": str(output_directory), "model_identity": identity,
        "metrics": manifest["evaluation"], "coverage_fraction": evaluation["coverage_fraction"],
        "changed_prediction_records": evaluation["changed_prediction_records"], "training_seconds": fit_seconds,
        "inference_seconds": inference_seconds, "calibration": calibration,
        "validation_rows": len(predictions), "classes": len(inputs.labels), "status": "completed_and_verified"}


def reproducibility_text(manifest, evaluation, calibration):
    return f"""# Probabilistic Brand ProductBrand lookup

Separate baseline: {manifest['model_identity']}. The original deterministic lookup,
its results and all MiniLM artifacts remain unchanged. Only ProductBrand is used.
The existing brand_key normalizer and missing-value definitions are reused.

All conditional label counts come from the {manifest['training_rows']:,} approved
labeled Brand training records. The full ordered vocabulary is retained. For a
seen key, P(c|key) = (n_key_c + prior_c)/(n_key+1); prior_c = N_c/N across all
labeled training records, including missing-feature records. Alpha=1 was fixed
before validation evaluation; no parameter candidates or validation optimization
were used. model.json contains all sparse counts, class counts and prior needed
to reproduce these predictions. Hashes and approved split identity are in the
manifest. Source functions: fit_frequency_lookup and predict_frequency_lookup
in catalogiq.brand_probabilistic_lookup. No neural models are loaded or fitted.

Missing and unseen keys use the entire training class-prior vector. Its argmax
is {evaluation['probabilistic_fallback_argmax_label']!r}, as for the original
majority fallback, but its confidence is the prior probability. Seen-key ties
can change because the prior replaces the deterministic saved-label-order tie
break; predicted labels always use probability argmax in the saved class order.

All {manifest['validation_rows']:,} labeled validation records, including catch-all
labels and missing/unseen ProductBrand values, match Brand MiniLM exactly by
ordered ID and truth. Coverage is {evaluation['coverage_fraction']:.10%}.
Original deterministic accuracy={evaluation['deterministic_reproduced']['accuracy']:.10f},
full-vocabulary macro-F1={evaluation['deterministic_reproduced']['macro_f1']:.10f}.
Probabilistic accuracy={evaluation['probabilistic']['accuracy']:.10f},
full-vocabulary macro-F1={evaluation['probabilistic']['macro_f1']:.10f}.
Changed predictions={evaluation['changed_prediction_records']}; original-correct
to new-wrong={evaluation['old_correct_new_wrong']}, original-wrong to new-correct=
{evaluation['old_wrong_new_correct']}. prediction_differences.csv identifies them.
evaluation.json includes full/cohort/per-class metrics and confusion matrices.
All {len(manifest['label_names'])} classes enter the original macro-F1 convention;
Maria's observed-label headline can differ when classes lack validation support.

These are training-frequency probability estimates, not calibrated probabilities.
Record counts may include correlated products within a training group. Rare keys
shrink strongly toward the prior; unseen keys have no key-specific evidence.
Training-to-validation conditional shifts and global-prior fallback shifts can
affect reliability. Maria should assess their calibration and review-rule value.

Maria's unchanged loader, scoring and exact calibration definitions passed.
Raw ECE={calibration['raw_ece']:.10f}, raw Brier={calibration['raw_multiclass_brier']:.10f}.
The temperature diagnostic was fitted/scored on the same validation population;
its results are compatibility diagnostics, not held-out calibration estimates.
No temperature-scaled probabilities were exported and no smoothing parameter
was changed. Use the approved Brand context rather than original V4/Segment.
Notebook dataset-loading, baseline-training and report-writing cells were not run.

CPU frequency-estimation time={manifest['training_seconds']:.6f}s; CPU batched
inference time={manifest['inference']['validation_inference_seconds']:.6f}s.
These scopes exclude file I/O and metric evaluation as recorded in manifest.json.
Single-request latency is unmeasured and null. Hardware was recorded during this
run; no GPU was used. CSV/JSON/Markdown files are LF, JSON labels are Unicode-escaped.

Reproduce from the repository root in the existing environment, with a NEW output:

    $env:PYTHONPATH = 'src'
    .venv-gpu/Scripts/python.exe -m catalogiq.brand_probabilistic_lookup --output-directory data/processed/minilm_validation_exports_v1/brand_lookup_probabilistic_v2

The estimator has no validation-selected parameters. No protected-test labels
or test.csv are read. Existing output directories are refused; source checksums
are verified before publication. Do not mix different targets in Maria's RUNS.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="data/processed/minilm_validation_exports_v1/brand_lookup_probabilistic_v1")
    parser.add_argument("--minilm-directory", default="data/processed/minilm_validation_exports_v1/brand")
    parser.add_argument("--harness-notebook", default=".cache/maria_harness/05e52e2e0203dee6d58dfc2d0e72d513eed9ad95/04-segment-evaluation.ipynb")
    args = parser.parse_args()
    inputs = prepare_export(Path.cwd(), "Brand")
    if (len(inputs.train_indices), len(inputs.validation_indices), len(inputs.labels)) != (58888, 12619, 143):
        raise ValueError("Approved Brand population changed")
    print(json.dumps(export_probabilistic_lookup(inputs, args.output_directory, args.minilm_directory, args.harness_notebook), indent=2))


if __name__ == "__main__":
    main()
