"""Word TF-IDF + logistic regression, with an optional category text ablation.

All learned preprocessing fits on train. Only validation is scored. No Hugging
Face tokenizer, model weights, split generation or test-scoring API is needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, recall_score)
from sklearn.pipeline import Pipeline

from .cleaning import PROVENANCE
from .features import sha256
from .segment_baseline_data import MISSING_LABELS, load_development
from .segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items, build_product_texts
from .splitting import check_assignments


@dataclass(frozen=True)
class BaselineConfig:
    c: float = 1.0
    max_iter: int = 1000
    min_df: int = 2
    max_features: int = 100_000
    ngram_max: int = 2
    class_weight: str | None = None
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.c, bool) or not math.isfinite(self.c) or self.c <= 0:
            raise ValueError("c must be positive and finite")
        for name in ("max_iter", "min_df", "max_features", "ngram_max"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ngram_max > 3:
            raise ValueError("ngram_max must be 1, 2, or 3")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be null or balanced")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")


def build_baseline_texts(frame, *, include_category=False):
    """Allowlisted four-field text; category joins the same document/vectorizer.

    Retains the shared builder's character limits (1024/512/4000/2000), but does
    not apply a transformer token limit. Category uses Maria's hierarchy spacing;
    nulls become empty. Input records are never modified.
    """
    texts = build_product_texts(frame)
    if include_category:
        if "ProductCategory" not in frame:
            raise ValueError("Category experiment requires a ProductCategory column")
        for i, value in enumerate(frame.ProductCategory):
            value = "" if pd.isna(value) else str(value).strip()
            value = "" if value.casefold() in MISSING_LABELS else re.sub(r"\s*>\s*", " > ", value)
            # Keep the same field marker in all category-experiment documents.
            texts[i] += " [ProductCategory] " + value
    return texts


def make_pipeline(config):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, config.ngram_max), min_df=config.min_df,
                                 max_features=config.max_features, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=config.c, solver="lbfgs", max_iter=config.max_iter,
                                          class_weight=config.class_weight, random_state=config.seed)),
    ])


def score_predictions(labels, probabilities, names):
    predicted = probabilities.argmax(axis=1)
    classes = np.arange(len(names))
    return {
        "records": len(labels),
        "macro_f1": float(f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, labels=classes, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_recall": float(recall_score(labels, predicted, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "multiclass_brier_sum": float(np.mean(np.sum((probabilities - np.eye(len(names))[labels]) ** 2, axis=1))),
        "per_class": classification_report(labels, predicted, labels=classes, target_names=list(names),
                                            zero_division=0, output_dict=True),
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_baseline(frame, config=None, *, include_category=False):
    """Fit training records and return validation predictions with source keys."""
    config = config or BaselineConfig()
    required = {*PROVENANCE, "record_id", "group_id", "split", "Segment", *MODEL_FIELDS}
    if not required.issubset(frame.columns) or set(frame.split) != {"train", "validation"}:
        raise ValueError("Provide labeled train/validation records only, with source keys and product fields")
    check_assignments(frame, frame.record_id)
    if frame[PROVENANCE].duplicated().any() or frame.Segment.isna().any():
        raise ValueError("Duplicate source keys or missing labels")
    labels = frame.Segment.astype(str).str.strip()
    if labels.str.casefold().isin(MISSING_LABELS).any():
        raise ValueError("Missing labels must be removed before supervised fitting")
    train = frame.split.eq("train").to_numpy()
    validation = ~train
    names = tuple(sorted(labels.loc[train].unique()))
    if len(names) < 2 or not set(labels.loc[validation]).issubset(names):
        raise ValueError("At least two classes are required; every validation class needs training examples")
    label_ids = labels.map({name: i for i, name in enumerate(names)}).to_numpy(dtype=int)
    documents = build_baseline_texts(frame, include_category=include_category)
    training_text = [text for text, selected in zip(documents, train) if selected]
    validation_text = [text for text, selected in zip(documents, validation) if selected]
    model = make_pipeline(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(training_text, label_ids[train])
    if not np.array_equal(model.classes_, np.arange(len(names))):
        raise AssertionError("Classifier probability columns changed class order")
    probabilities = model.predict_proba(validation_text)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((int(train.sum()), 1)), label_ids[train])
    dummy_probabilities = dummy.predict_proba(np.zeros((int(validation.sum()), 1)))
    predictions = frame.loc[validation, [*PROVENANCE, "record_id", "group_id", "split"]].reset_index(drop=True)
    predictions["true_label"] = labels.loc[validation].to_numpy()
    predictions["predicted_label"] = [names[i] for i in probabilities.argmax(axis=1)]
    predictions["confidence"] = probabilities.max(axis=1)
    for i in range(len(names)):
        predictions[f"probability_{i}"] = probabilities[:, i]
    metrics = {
        "primary_metric": "macro_f1", "partition": "validation", "label_names": list(names),
        "baseline": score_predictions(label_ids[validation], probabilities, names),
        "majority_class": score_predictions(label_ids[validation], dummy_probabilities, names),
        "majority_class_selected_on_training": names[int(dummy.class_prior_.argmax())],
        "confidence_status": "uncalibrated model probabilities; no automatic review or acceptance threshold",
    }
    return model, predictions, metrics


def _save_variant(directory, frame, config, include_category):
    directory.mkdir()
    model, predictions, metrics = fit_baseline(frame, config, include_category=include_category)
    names = metrics["label_names"]
    predictions.to_csv(directory / "predictions.csv", index=False)
    predictions.loc[predictions.true_label.ne(predictions.predicted_label)].to_csv(directory / "validation_errors.csv", index=False)
    matrix = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=names)
    pd.DataFrame(matrix, index=pd.Index(names, name="true_label"), columns=names).to_csv(directory / "confusion_matrix.csv")
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "label_mapping.json", {f"probability_{i}": label for i, label in enumerate(names)})
    joblib.dump({"pipeline": model, "label_names": names, "config": asdict(config),
                 "include_category": include_category,
                 "input_adapter": "catalogiq.segment_baseline.build_baseline_texts"}, directory / "model.joblib")
    return {
        "validation": metrics["baseline"], "majority_class": metrics["majority_class"],
        "vocabulary_features": len(model.named_steps["tfidf"].vocabulary_),
        "optimizer_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "training_text_sha256": _hash_items(build_baseline_texts(frame.loc[frame.split.eq("train")], include_category=include_category)),
        "validation_text_sha256": _hash_items(build_baseline_texts(frame.loc[frame.split.eq("validation")], include_category=include_category)),
    }


def run_baseline(input_path, assignments_path, expected_assignments_sha256, output_dir,
                 config=None, *, expected_input_sha256=None, features="compare", train=False, progress=print):
    """Prepare explicit shared assignments; fitting requires train=True.

    Preparation does not fit or score. Supplying an older research assignment
    does not verify or replace the project's protected final snapshot.
    """
    started = time.perf_counter()
    config = config or BaselineConfig()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; prior runs are never overwritten")
    if features not in {"base", "category", "compare"}:
        raise ValueError("features must be base, category or compare")
    if progress:
        progress("Checking input hashes, source-key coverage and group isolation.")
    development = load_development(input_path, assignments_path, expected_assignments_sha256, expected_input_sha256)
    frame = development.frame
    if features != "base" and "ProductCategory" not in frame:
        raise ValueError("Category experiment requires a ProductCategory column")
    counts = development.audit["labeled_development_counts"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "target": "Segment",
        "model": "word TF-IDF + logistic regression", "config": asdict(config),
        "development": development.audit, "features": features,
        "base_fields": list(MODEL_FIELDS), "category_mode": "concatenated into the same text/vectorizer",
        "field_character_limits": dict(zip(MODEL_FIELDS, ModelConfig().field_character_limits)),
        "transformer_token_limit_applied": False,
        "policy": "train fit only; validation evaluation only; no test-scoring API",
        "training_record_ids_sha256": _hash_items(frame.loc[frame.split.eq("train"), "record_id"].tolist()),
        "validation_record_ids_sha256": _hash_items(frame.loc[frame.split.eq("validation"), "record_id"].tolist()),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                          ("segment_baseline.py", "segment_baseline_data.py", "segment_transformer.py", "splitting.py")},
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "protocol.json", protocol)
    summary = {
        "status": "prepared", "model_training_performed": False, "target": "Segment",
        "primary_metric": "macro_f1", "training_rows": counts["train"], "validation_rows": counts["validation"],
        "test_scored": False, "new_splits_created": False, "variants": {},
    }
    if train:
        variants = {"base": False, "category": True} if features == "compare" else {features: features == "category"}
        for name, include_category in variants.items():
            if progress:
                progress(f"Fitting {name}: {counts['train']:,} train / {counts['validation']:,} validation records.")
            summary["variants"][name] = _save_variant(output_dir / name, frame, config, include_category)
        rows = [{"variant": name, **{key: result["validation"][key] for key in ("accuracy", "macro_f1", "weighted_f1")}}
                for name, result in summary["variants"].items()]
        pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)
        if features == "compare":
            base, category = (summary["variants"][name]["validation"] for name in ("base", "category"))
            summary["category_minus_base_percentage_points"] = {key: 100 * (category[key] - base[key]) for key in ("accuracy", "macro_f1")}
        summary.update(status="completed", model_training_performed=True)
    summary["runtime_seconds"] = time.perf_counter() - started
    summary["artifact_sha256"] = {path.relative_to(output_dir).as_posix(): sha256(path)
                                  for path in sorted(output_dir.rglob("*")) if path.is_file()}
    _write_json(output_dir / "summary.json", summary)
    if progress:
        progress("Validation artifacts saved; test not scored." if train else "Prepared only; no model trained or scored.")
    return summary
