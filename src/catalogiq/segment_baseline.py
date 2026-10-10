"""Classical Segment baseline on the sealed transformer development inputs.

TF-IDF and logistic regression learn from training only. The frozen tokenizer is
used as a deterministic text adapter, without a neural encoder or model weights.
There is no test-scoring, split-generation, or unsealed-CSV fallback here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
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

from .features import sha256
from .segment_frozen_development import load_frozen_development
from .segment_transformer import (MODEL_FIELDS, ModelConfig, TokenizedCorpus,
                                  _hash_items, _validate_indices, tokenize_corpus)


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


def token_documents(corpus: TokenizedCorpus) -> list[str]:
    """Preserve frozen token boundaries/IDs, including special tokens.

    A word token such as t123 is an opaque vocabulary ID, not an ordinal numeric
    feature. No decoding, second normalization, or extra product text is added.
    The resulting bag of uni/bigrams is deliberately a classical representation.
    """
    documents = []
    if len(corpus.input_ids) != len(corpus) or len(corpus.attention_mask) != len(corpus):
        raise ValueError("Token arrays do not cover the development records")
    for ids, mask in zip(corpus.input_ids, corpus.attention_mask):
        ids, mask = np.asarray(ids), np.asarray(mask)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
                or len(ids) > corpus.sequence_length or mask.shape != ids.shape
                or np.any(ids < 0) or not np.all(mask == 1)):
            raise ValueError("Expected nonnegative unpadded frozen token sequences")
        documents.append(" ".join(f"t{int(value)}" for value in ids))
    return documents


def make_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\bt\d+\b",
                                 ngram_range=(1, config.ngram_max), min_df=config.min_df,
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


def fit_baseline(corpus, train_indices, validation_indices, config=None):
    """Fit on the given training records and return validation predictions only."""
    config = config or BaselineConfig()
    train_indices, validation_indices = _validate_indices(corpus, train_indices, validation_indices)
    names = tuple(corpus.label_names)
    labels = np.asarray(corpus.label_ids)
    if (len(names) < 2 or len(set(names)) != len(names) or labels.shape != (len(corpus),)
            or labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= len(names)):
        raise ValueError("Invalid development labels or class order")
    if set(labels[train_indices]) != set(range(len(names))):
        raise ValueError("Every declared class must have training examples")
    documents = token_documents(corpus)
    training_text = [documents[i] for i in train_indices]
    validation_text = [documents[i] for i in validation_indices]
    model = make_pipeline(config)
    with warnings.catch_warnings():
        # A nonconverged fit is not a completed baseline. Retry with an explicit
        # recorded configuration in a new run directory, never silently continue.
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(training_text, labels[train_indices])
    if not np.array_equal(model.classes_, np.arange(len(names))):
        raise AssertionError("Classifier probability columns changed class order")
    probabilities = model.predict_proba(validation_text)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_indices), 1)), labels[train_indices])
    dummy_probabilities = dummy.predict_proba(np.zeros((len(validation_indices), 1)))
    predictions = pd.DataFrame({
        "record_id": [corpus.record_ids[i] for i in validation_indices],
        "true_label": [names[i] for i in labels[validation_indices]],
        "predicted_label": [names[i] for i in probabilities.argmax(axis=1)],
        "confidence": probabilities.max(axis=1),
    })
    for i in range(len(names)):
        predictions[f"probability_{i}"] = probabilities[:, i]
    metrics = {
        "primary_metric": "macro_f1", "partition": "validation", "label_names": list(names),
        "baseline": score_predictions(labels[validation_indices], probabilities, names),
        "majority_class": score_predictions(labels[validation_indices], dummy_probabilities, names),
        "majority_class_selected_on_training": names[int(dummy.class_prior_.argmax())],
        "confidence_status": "uncalibrated model probabilities; no automatic review or acceptance threshold",
    }
    return model, predictions, metrics


def run_baseline(split_dir, tokenizer_dir, model_config_path, output_dir, config=None, *, progress=print):
    """Run against the existing frozen snapshot, failing closed on missing files."""
    started = time.perf_counter()
    config = config or BaselineConfig()
    split_dir, tokenizer_dir = Path(split_dir).resolve(), Path(tokenizer_dir).resolve()
    model_config_path, output_dir = Path(model_config_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; prior runs are never overwritten")
    for source in (split_dir, tokenizer_dir):
        if output_dir == source or output_dir.is_relative_to(source):
            raise ValueError("Output must be outside the frozen inputs and tokenizer directories")
    required = [split_dir / name for name in ("completion.json", "grouping_freeze.json", "train.csv",
                                             "validation.csv", "protected_final_test.csv")]
    required += [tokenizer_dir / "tokenizer.json", tokenizer_dir / "tokenizer_config.json", model_config_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen development inputs are unavailable. Obtain the shared sealed snapshot, "
                                "its referenced verification artifacts, model_config.json and tokenizer from the split owner. "
                                "Do not substitute scripts.split_data or a fresh random split. Missing: " + ", ".join(missing))
    model_config = ModelConfig.from_dict(json.loads(model_config_path.read_text(encoding="utf-8")))
    if not model_config.local_files_only:
        raise ValueError("The baseline requires the frozen offline tokenizer")
    if progress:
        progress("Verifying the sealed training/validation snapshot; final test remains closed.")
    development = load_frozen_development(split_dir, model_config, tokenizer_dir=tokenizer_dir)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Install the baseline extra: python -m pip install -e ".[baseline]"') from error
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    corpus = tokenize_corpus(development.frame, development.label_names, model_config, tokenizer=tokenizer)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "target": "Segment",
        "model": "TF-IDF frozen token ngrams + logistic regression", "config": asdict(config),
        "model_input_config": model_config.to_dict(), "development": development.audit,
        "policy": "train fit only; validation evaluation only; no final-test API",
        "representation": "Same frozen product text/tokenization/128-token default as the transformer; different learned feature representation.",
        "features": list(MODEL_FIELDS), "feature_engineering": "No additional research features enabled",
        "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
        "training_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in development.train_indices]),
        "validation_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in development.validation_indices]),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                          ("segment_baseline.py", "segment_frozen_development.py", "segment_transformer.py", "model_input_groups.py")},
        "python_version": platform.python_version(), "package_versions": {},
        "reproducibility_scope": "same frozen inputs, tokenizer, configuration, software versions and ordering",
    }
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "transformers", "tokenizers"):
        try:
            protocol["package_versions"][package] = version(package)
        except PackageNotFoundError:
            protocol["package_versions"][package] = "unavailable"
    _write_json(output_dir / "protocol.json", protocol)
    if progress:
        progress(f"Fitting baseline: {len(development.train_indices):,} training, {len(development.validation_indices):,} validation records.")
    model, predictions, metrics = fit_baseline(corpus, development.train_indices, development.validation_indices, config)
    validation = development.frame.iloc[development.validation_indices]
    if (predictions.record_id.tolist() != validation.record_id.tolist()
            or set(predictions.record_id) & development.protected_record_ids):
        raise AssertionError("Predictions must cover validation only")
    predictions.insert(1, "group_id", validation.group_id.to_numpy())
    predictions.insert(2, "split", "validation")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions.loc[predictions.true_label.ne(predictions.predicted_label)].to_csv(output_dir / "validation_errors.csv", index=False)
    matrix = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=development.label_names)
    pd.DataFrame(matrix, index=pd.Index(development.label_names, name="true_label"),
                 columns=development.label_names).to_csv(output_dir / "confusion_matrix.csv")
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "label_mapping.json", {f"probability_{i}": label for i, label in enumerate(development.label_names)})
    joblib.dump({"pipeline": model, "label_names": development.label_names,
                 "model_input_config": model_config.to_dict(), "config": asdict(config),
                 "input_adapter": "catalogiq.segment_baseline.token_documents"}, output_dir / "model.joblib")
    summary = {
        "status": "completed", "target": "Segment", "primary_metric": "macro_f1",
        "training_rows": len(development.train_indices), "validation_rows": len(development.validation_indices),
        "final_test_read_or_scored": False, "new_splits_created": False,
        "vocabulary_features": len(model.named_steps["tfidf"].vocabulary_),
        "optimizer_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "validation": metrics["baseline"], "majority_class": metrics["majority_class"],
        "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
    }
    _write_json(output_dir / "summary.json", summary)
    if progress:
        progress(f"Validation macro-F1 {metrics['baseline']['macro_f1']:.4f}; accuracy {metrics['baseline']['accuracy']:.4f}. Final test not scored.")
    return summary
