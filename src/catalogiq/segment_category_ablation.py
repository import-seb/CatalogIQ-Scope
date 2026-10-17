"""Controlled ProductCategory ablation on frozen Segment development records.

The frozen grouper, allocation and original four-field model remain unchanged.
This experiment registers a separate category representation and never opens or
hashes test.csv. Both arms train from scratch and score validation only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .features import sha256
from .model_input_groups import _signature
from .segment_frozen_development import load_frozen_development
from .segment_transformer import (MODEL_FIELDS, ModelConfig, TokenizedCorpus,
    _hash_items, _plain_text, build_product_texts, tokenize_corpus, train_transformer)
from .segment_validation_analysis import compare_paired_predictions, evaluate_predictions
from .split_evaluation import _exact_metrics

VERSION = "segment-productcategory-ablation-v2"
ARMS = ("baseline", "with_category")
CATEGORY = "ProductCategory"
DEFAULT_SEQUENCE_LENGTH = 256


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n", encoding="utf8", newline="\n")


def load_category_development(split_dir, config=None):
    """Verify sealed splits, then add development-only experimental features.

    The frozen groups retain their original token-length contract. A different
    experiment length does not change that contract or certify the final test.
    All other representation settings must still pass the original loader.
    """
    split_dir = Path(split_dir)
    sealed_config = ModelConfig.from_dict(_json(split_dir / "model_config.json"))
    config = config or replace(sealed_config, sequence_length=DEFAULT_SEQUENCE_LENGTH)
    verification_config = replace(config, sequence_length=sealed_config.sequence_length)
    development = load_frozen_development(split_dir, verification_config)
    completion = _json(split_dir / "completion.json")
    pieces = []
    consumed = {}
    for role in ("train", "validation"):
        path = split_dir / f"{role}.csv"
        expected_hash = completion["artifact_sha256"][path.name]
        if sha256(path) != expected_hash:
            raise ValueError(f"Frozen {role} export changed before category loading")
        fields = pd.read_csv(path, dtype=str, keep_default_na=False,
            encoding="utf-8-sig", usecols=["record_id", "ProductCategory", "Retailer"])
        if fields.record_id.duplicated().any():
            raise ValueError("Category join requires unique development record IDs")
        # The guard counts the full exports before filtering missing targets.
        if len(fields) != development.audit["export_records"][role]:
            raise ValueError("Category export population differs from verified development")
        expected_ids = set(development.frame.loc[development.frame.split.eq(role), "record_id"])
        if not expected_ids <= set(fields.record_id):
            raise ValueError("Category export does not cover every supervised development record")
        if set(fields.record_id) & development.protected_record_ids:
            raise ValueError("Protected final-test record appears in category development")
        pieces.append(fields.loc[fields.record_id.isin(expected_ids)])
        consumed[path.name] = expected_hash
    categories = pd.concat(pieces, ignore_index=True)
    if categories.record_id.duplicated().any() or set(categories.record_id) != set(development.frame.record_id):
        raise ValueError("Category join population differs from supervised development")
    frame = development.frame.merge(categories, on="record_id", how="left", sort=False,
                                   validate="one_to_one")
    if frame.record_id.tolist() != development.frame.record_id.tolist():
        raise ValueError("Category join changed supervised record ordering")
    if any(sha256(split_dir / name) != digest for name, digest in consumed.items()):
        raise ValueError("Development exports changed during category loading")
    return replace(development, frame=frame, audit={**development.audit,
        "additional_development_features": [CATEGORY, "Retailer"],
        "category_source_sha256": consumed, "frozen_model_contract_unchanged": True,
        "sealed_sequence_length": sealed_config.sequence_length,
        "experimental_sequence_length": config.sequence_length})


def build_category_texts(frame, config=None, category_character_limit=512):
    """Prefix a retailer breadcrumb; preserve the baseline formatter verbatim."""
    if CATEGORY not in frame:
        raise ValueError("ProductCategory is required; Category is not a substitute")
    if type(category_character_limit) is not int or category_character_limit < 1:
        raise ValueError("category_character_limit must be a positive integer")
    baseline = build_product_texts(frame, config)
    categories = [_plain_text(value)[:category_character_limit].rstrip() for value in frame[CATEGORY]]
    return [f"[ProductCategory] {category} {text}".rstrip() if category else text
            for category, text in zip(categories, baseline)]


def load_tokenizer(directory):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


def tokenize_category_corpus(frame, label_names, config=None, *, tokenizer=None,
                             category_character_limit=512):
    config = config or ModelConfig()
    names = tuple(str(label) for label in label_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("label_names must contain at least two unique classes")
    if frame.empty or not {"record_id", "Segment"}.issubset(frame):
        raise ValueError("Development records require identity and Segment")
    ids = tuple(frame.record_id.astype(str))
    if frame.record_id.isna().any() or len(set(ids)) != len(ids) or any(not value.strip() for value in ids):
        raise ValueError("Development record IDs must be unique and nonblank")
    labels = frame.Segment.map({name: i for i, name in enumerate(names)})
    if labels.isna().any():
        raise ValueError("Segment contains a label outside label_names")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision,
                                                  local_files_only=config.local_files_only)
    texts = build_category_texts(frame, config, category_character_limit)
    encoded = tokenizer(texts, truncation=True, max_length=config.sequence_length,
                        padding=False, return_attention_mask=True)
    inputs = tuple(np.asarray(row, dtype=np.int32) for row in encoded["input_ids"])
    masks = tuple(np.asarray(row, dtype=np.int32) for row in encoded["attention_mask"])
    types = (tuple(np.asarray(row, dtype=np.int32) for row in encoded["token_type_ids"])
             if "token_type_ids" in encoded else None)
    if (len(inputs) != len(ids) or len(masks) != len(ids)
            or (types is not None and len(types) != len(ids))
            or any(not 0 < len(tokens) <= config.sequence_length or len(tokens) != len(mask)
                   for tokens, mask in zip(inputs, masks))
            or (types is not None and any(len(tokens) != len(kind) for tokens, kind in zip(inputs, types)))):
        raise ValueError("Tokenizer returned invalid development token arrays")
    digest = hashlib.sha256()
    digest.update(_hash_items(ids).encode("ascii"))
    for rows in (inputs, masks, types or ()):
        for row in rows:
            digest.update(len(row).to_bytes(8, "big"))
            digest.update(row.astype("<i4", copy=False).tobytes())
    return TokenizedCorpus(ids, inputs, masks, labels.to_numpy(dtype=np.int64), names,
        types, int(getattr(tokenizer, "pad_token_id", 0) or 0), _hash_items(texts),
        digest.hexdigest(), config.sequence_length, tokenizer)


def input_audit(corpus, frame):
    encoded = {"input_ids": corpus.input_ids, "attention_mask": corpus.attention_mask}
    if corpus.token_type_ids is not None:
        encoded["token_type_ids"] = corpus.token_type_ids
    keys = [_signature(encoded, index)[0] for index in range(len(corpus))]
    metrics = _exact_metrics(keys, frame.split.tolist(), "actual development token arrays")
    lengths = np.array([len(row) for row in corpus.input_ids])
    metrics["max_length_records"] = {role: int(((lengths == corpus.sequence_length) & frame.split.eq(role)).sum())
                                     for role in ("train", "validation")}
    return metrics


def _source_hashes(split_dir):
    folder = Path(__file__).parent
    sources = {str(folder / name): sha256(folder / name) for name in (
        "segment_category_ablation.py", "segment_transformer.py", "segment_frozen_development.py",
        "segment_validation_analysis.py", "split_protocol.py")}
    sources.update({str(split_dir / name): sha256(split_dir / name) for name in (
        "completion.json", "grouping_freeze.json", "model_config.json", "train.csv", "validation.csv")})
    sources.update({str(path): sha256(path) for path in (split_dir / "tokenizer").iterdir() if path.is_file()})
    return sources


def _check_sources(hashes):
    if any(sha256(Path(name)) != expected for name, expected in hashes.items()):
        raise ValueError("Frozen experiment source or development artifact changed")


def pretrained_artifacts(config):
    """Bind the cached public encoder bytes without downloading anything."""
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(config.model_name, revision=config.revision, local_files_only=True))
    paths = [directory / name for name in ("config.json", "model.safetensors")]
    return {str(path): sha256(path) for path in paths}


def _annotate_training(directory, manifest, fields, representation):
    """Record actual experimental fields instead of the trainer's fixed default.

    The unchanged trainer consumes token arrays and records its four-field
    default. This wrapper registers the actual input view in both manifests and
    updates artifact hashes before sealing the experimental arm.
    """
    initialization = _json(directory / "initialization.json")
    for value in (initialization, manifest):
        value["trainer_default_input_fields"] = list(MODEL_FIELDS)
        value["input_fields"] = list(fields)
        value["experimental_representation"] = representation
        value["final_test_representation_certified"] = False
    _write_json(directory / "initialization.json", initialization)
    manifest["artifacts"] = {path.name: sha256(path) for path in directory.iterdir()
                             if path.is_file() and path.name != "training_manifest.json"}
    _write_json(directory / "training_manifest.json", manifest)


def _read_completed_arm(directory):
    manifest = _json(directory / "training_manifest.json")
    for name, expected in manifest["artifacts"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Completed training artifact changed: {name}")
    return pd.read_csv(directory / "predictions.csv", dtype={"record_id": str,
        "true_label": str, "predicted_label": str}, float_precision="round_trip"), manifest


def run_experiment(split_dir, output_dir, config=None, *, bootstrap_samples=1000,
                   category_character_limit=512, progress=print, resume=False, prepare_only=False):
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError("Use a new ablation output directory; completed runs are preserved")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    config = config or replace(ModelConfig.from_dict(_json(split_dir / "model_config.json")),
                               sequence_length=DEFAULT_SEQUENCE_LENGTH)
    started = perf_counter()
    development = load_category_development(split_dir, config)
    frame = development.frame
    hashes = _source_hashes(split_dir)
    hashes.update(pretrained_artifacts(config))
    tokenizer = load_tokenizer(split_dir / "tokenizer")
    corpora = {"baseline": tokenize_corpus(frame, development.label_names, config, tokenizer=tokenizer),
        "with_category": tokenize_category_corpus(frame, development.label_names, config,
            tokenizer=tokenizer, category_character_limit=category_character_limit)}
    if (any(corpus.record_ids != tuple(frame.record_id) for corpus in corpora.values())
            or not np.array_equal(corpora["baseline"].label_ids, corpora["with_category"].label_ids)):
        raise ValueError("Input representations changed controlled record/label ordering")
    category_values = frame.ProductCategory.map(_plain_text)
    training_categories = set(category_values.iloc[development.train_indices]) - {""}
    representation = {"baseline_fields": list(MODEL_FIELDS), "category_fields": [CATEGORY, *MODEL_FIELDS],
        "category_position": "prefix", "category_character_limit": category_character_limit,
        "combined_token_limit": config.sequence_length,
        "sealed_grouping_token_limit": development.audit["sealed_sequence_length"],
        "category_scope": "Existing retailer breadcrumb only; supplied Category and all other targets are excluded.",
        "budget_caveat": "Category prefix can displace product text within the same token budget; comparison measures the net effect.",
        "split_contract": "Original split representation remains unchanged; category representation audited on development only.",
        "final_test_representation_certified": False}
    protocol = {"version": VERSION, "model_config": config.to_dict(), "representation": representation,
        "label_names": list(development.label_names), "bootstrap_samples": bootstrap_samples,
        "training_rows": len(development.train_indices), "validation_rows": len(development.validation_indices),
        "train_record_ids_sha256": _hash_items(frame.iloc[development.train_indices].record_id.tolist()),
        "validation_record_ids_sha256": _hash_items(frame.iloc[development.validation_indices].record_id.tolist()),
        "checkpoint_policy": "Fixed final epoch; no validation-based checkpoint or hyperparameter selection.",
        "controlled": "Same full development rows, initial weights, batch ordering, optimizer and training schedule; only input representation differs.",
        "source_sha256": hashes, "development_audit": development.audit,
        "tokenization": {arm: {"text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
            "input_audit": input_audit(corpus, frame)} for arm, corpus in corpora.items()},
        "category_coverage": {role: {"records": int(frame.split.eq(role).sum()),
            "missing": int((frame.split.eq(role) & category_values.eq("")).sum())} for role in ("train", "validation")},
        "final_test_read_or_scored": False}
    # Normalize dataclass tuples exactly as they will be serialized for resume.
    protocol = json.loads(json.dumps(protocol, ensure_ascii=False, allow_nan=False))
    _check_sources(hashes)
    if resume:
        existing = _json(output_dir / "protocol.json")
        if {key: value for key, value in existing.items() if key != "created_utc"} != protocol:
            raise ValueError("Resume configuration, corpus or frozen source differs from prepared protocol")
        if _json(output_dir / "protocol_freeze.json")["protocol_sha256"] != sha256(output_dir / "protocol.json"):
            raise ValueError("Prepared experiment protocol changed")
        protocol = existing
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        protocol["created_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "protocol.json", protocol)
        _write_json(output_dir / "protocol_freeze.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
            "frozen_before_training": True})
    progress(f"Prepared full category ablation: {protocol['training_rows']:,} train / {protocol['validation_rows']:,} validation")
    for arm in ARMS:
        progress(f"{arm} identical-input crossings: {protocol['tokenization'][arm]['input_audit']['crossing_pairs']}")
    if prepare_only:
        return protocol
    validation = frame.iloc[development.validation_indices].copy()
    validation_ids = set(validation.record_id)
    predictions, manifests, evaluated, rows, classes, slices = {}, {}, {}, [], [], []
    for arm in ARMS:
        _check_sources(hashes)
        directory = output_dir / arm
        if resume and (directory / "training_manifest.json").exists():
            predicted, manifest = _read_completed_arm(directory)
        else:
            progress(f"Training {arm} from the pinned pretrained encoder")
            def arm_progress(message):
                initialization = directory / "initialization.json"
                if arm == "with_category" and initialization.exists():
                    if _json(initialization)["initial_state_sha256"] != manifests["baseline"]["initial_state_sha256"]:
                        raise ValueError("The two models did not start from identical weights")
                progress(message)
            predicted, manifest = train_transformer(corpora[arm], development.train_indices,
                development.validation_indices, development.label_names, config, directory, progress=arm_progress)
            fields = MODEL_FIELDS if arm == "baseline" else (CATEGORY, *MODEL_FIELDS)
            _annotate_training(directory, manifest, fields, representation)
            # Score persisted probabilities identically on fresh and resumed runs.
            predicted, manifest = _read_completed_arm(directory)
        if set(predicted.record_id) & development.protected_record_ids:
            raise ValueError("Protected final-test record was predicted")
        if predicted.record_id.duplicated().any() or set(predicted.record_id) != validation_ids:
            raise ValueError("Prediction population differs from the entire frozen validation pool")
        if predicted.set_index("record_id").true_label.to_dict() != validation.set_index("record_id").Segment.to_dict():
            raise ValueError("Prediction truth differs from frozen validation targets")
        if (manifest["train_record_ids_sha256"] != protocol["train_record_ids_sha256"]
                or manifest["validation_record_ids_sha256"] != protocol["validation_record_ids_sha256"]):
            raise ValueError("Trainer changed controlled development membership")
        if arm == "with_category" and manifest["initial_state_sha256"] != manifests["baseline"]["initial_state_sha256"]:
            raise ValueError("The two models did not start from identical weights")
        predictions[arm], manifests[arm] = predicted, manifest
        evaluated[arm] = evaluate_predictions(predicted, development.label_names)
        rows.append({"arm": arm, **evaluated[arm]["metrics"], "training_seconds": manifest["runtime_seconds"]})
        classes.append(evaluated[arm]["per_class"].assign(arm=arm))
        predicted = predicted.merge(validation[["record_id", "Retailer", "ProductCategory"]],
                                    on="record_id", validate="one_to_one")
        category = predicted.ProductCategory.map(_plain_text)
        cohorts = [("category", "seen_in_training", predicted.loc[category.isin(training_categories)]),
                   ("category", "unseen_in_training", predicted.loc[category.ne("") & ~category.isin(training_categories)]),
                   ("category", "missing", predicted.loc[category.eq("")])]
        cohorts += [("retailer", str(retailer), subset) for retailer, subset in predicted.groupby("Retailer", sort=True)]
        for dimension, value, subset in cohorts:
            slices.append({"arm": arm, "dimension": dimension, "value": value,
                           **evaluate_predictions(subset, development.label_names)["metrics"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "validation_comparison.csv", index=False, lineterminator="\n")
    per_class = pd.concat(classes, ignore_index=True)
    per_class.to_csv(output_dir / "per_class_comparison.csv", index=False, lineterminator="\n")
    pd.DataFrame(slices).to_csv(output_dir / "validation_slices.csv", index=False, lineterminator="\n")
    paired = compare_paired_predictions(predictions["with_category"], predictions["baseline"],
        development.label_names, validation[["record_id", "group_id"]], n_bootstrap=bootstrap_samples, seed=config.seed)
    paired["intervals"].to_csv(output_dir / "paired_differences.csv", index=False, lineterminator="\n")
    _write_json(output_dir / "paired_comparison.json", {key: value for key, value in paired.items()
                                                       if key not in ("intervals", "record_ids")})
    _check_sources(hashes)
    summary = {"version": VERSION, "protocol_sha256": sha256(output_dir / "protocol.json"),
        "training_rows": protocol["training_rows"], "validation_rows": protocol["validation_rows"],
        "validation": {arm: evaluated[arm]["metrics"] for arm in ARMS},
        "paired_differences": paired["intervals"].to_dict("records"),
        "input_audit": {arm: protocol["tokenization"][arm]["input_audit"] for arm in ARMS},
        "same_initial_weights": True, "initial_state_sha256": manifests["baseline"]["initial_state_sha256"],
        "same_training_validation_membership": True, "full_development_pool_used": True,
        "split_artifacts_and_frozen_modules_unchanged": True, "final_test_read_or_scored": False,
        "category_coverage": protocol["category_coverage"], "representation": representation,
        "bootstrap_caveat": "Paired frozen-group validation sampling uncertainty only; training-seed and production uncertainty excluded.",
        "single_training_seed": True, "runtime_seconds": perf_counter() - started}
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "completion.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
        "artifact_sha256": {path.relative_to(output_dir).as_posix(): sha256(path)
            for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "completion.json"}})
    progress(comparison[["arm", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
    progress(paired["intervals"].to_string(index=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--sequence-length", type=int,
        help="Shared token limit; default 256, or the explicitly supplied model configuration")
    parser.add_argument("--category-character-limit", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = ModelConfig.from_dict(_json(args.model_config or args.split_dir / "model_config.json"))
    if args.sequence_length is not None:
        config = replace(config, sequence_length=args.sequence_length)
    elif args.model_config is None:
        config = replace(config, sequence_length=DEFAULT_SEQUENCE_LENGTH)
    if args.device:
        config = replace(config, device=args.device)
    run_experiment(args.split_dir, args.output_dir, config,
        category_character_limit=args.category_character_limit, bootstrap_samples=args.bootstrap_samples,
        resume=args.resume, prepare_only=args.prepare_only,
        progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
