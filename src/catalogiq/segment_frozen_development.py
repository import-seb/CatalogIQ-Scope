"""Guarded Segment development loader for sealed final product-group splits.

The loader reads train/validation features and labels only. Final-test evidence
is restricted to the protected ID/group mapping; test.csv is never opened or
hashed here. Model preparation defaults to checks only. Explicit training starts
fresh from the pinned pretrained encoder, with a fixed final-epoch policy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import sha256
from .model_input_groups import model_view_contract
from .segment_transformer import MODEL_FIELDS, ModelConfig

SEGMENT_LABELS = ("Allergy", "CCFS", "Digestive Health", "External Analgesics",
                  "Internal Analgesics", "Lifestyle CHC", "Other Self Care",
                  "Vitamins, Minerals & Supplements")
MISSING_TARGETS = {"", "null", "none", "nan"}


@dataclass(frozen=True)
class FrozenDevelopment:
    frame: pd.DataFrame
    train_indices: np.ndarray
    validation_indices: np.ndarray
    label_names: tuple[str, ...]
    protected_record_ids: frozenset[str]
    protected_group_ids: frozenset[str]
    audit: dict


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def _hash_from_completion(completion, name):
    expected = completion.get("artifact_sha256", {}).get(name)
    if not expected and name == "grouping_freeze.json":
        expected = completion.get("grouping_freeze_sha256")
    if not expected:
        raise ValueError(f"Final split seal has no checksum for {name}")
    return expected


def _validate_identity(frame, name):
    for column in ("record_id", "group_id"):
        if column not in frame or frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"{name} requires nonblank {column}")
    if frame.record_id.duplicated().any():
        raise ValueError(f"{name} has duplicate record IDs")


def load_frozen_development(directory, config=None, *, tokenizer_dir, verifier=None,
                            label_names=SEGMENT_LABELS):
    """Verify a split seal and construct supervised development without test data.

    Missing targets remain untouched in exports and are excluded only from the
    supervised view. Any nonmissing undeclared Segment label is rejected. All
    full exported groups are checked for isolation before excluding missing rows.
    The verifier injection point is for synthetic tests only.
    """
    directory = Path(directory).resolve()
    config = config or ModelConfig()
    if verifier is None:
        from .split_finalization import verify_final_splits
        verifier = verify_final_splits
    verified = verifier(directory, verify_exports=False)
    completion = json.loads((directory / "completion.json").read_text(encoding="utf8"))
    freeze_path = directory / "grouping_freeze.json"
    if sha256(freeze_path) != _hash_from_completion(completion, freeze_path.name):
        raise ValueError("Final grouping freeze has changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf8"))
    current = model_view_contract(config, tokenizer_dir=tokenizer_dir)
    if current != freeze.get("model_view_contract"):
        raise ValueError("Model representation differs from the sealed groups; rebuild grouping and final splits")
    if tuple(current["features"]) != MODEL_FIELDS:
        raise ValueError("Target/metadata columns cannot be model features")
    consumed = {}
    for name in ("train.csv", "validation.csv", "protected_final_test.csv"):
        path = directory / name
        if sha256(path) != _hash_from_completion(completion, name):
            raise ValueError(f"Frozen {name} was modified")
        consumed[name] = sha256(path)
    # Restrict the protected file to identities; never read its labels/features.
    protected = pd.read_csv(directory / "protected_final_test.csv", dtype=str, keep_default_na=False,
                            usecols=["record_id", "group_id"])
    _validate_identity(protected, "protected final test")
    roles = []
    counts = {}
    for role, name in (("train", "train.csv"), ("validation", "validation.csv")):
        frame = pd.read_csv(directory / name, dtype=str, keep_default_na=False)
        _validate_identity(frame, role)
        if "Segment" not in frame or not all(field in frame for field in MODEL_FIELDS):
            raise ValueError(f"{role} export lacks Segment or the fixed model fields")
        if "split" in frame and not frame.split.eq(role).all():
            raise ValueError(f"{role} export contains another split role")
        if set(frame.record_id) & set(protected.record_id):
            raise ValueError("Protected final-test record appears in development")
        if set(frame.group_id) & set(protected.group_id):
            raise ValueError("Protected final-test product group appears in development")
        # Keep only declared feature fields, the true target, identity and role.
        frame = frame[["record_id", "group_id", "Segment", *MODEL_FIELDS]].copy()
        frame["split"] = role
        counts[role] = len(frame)
        roles.append(frame)
    train, validation = roles
    if set(train.record_id) & set(validation.record_id):
        raise ValueError("Training and validation record overlap")
    if set(train.group_id) & set(validation.group_id):
        raise ValueError("Training and validation product-group overlap")
    joined = pd.concat(roles, ignore_index=True)
    missing = joined.Segment.str.strip().str.casefold().isin(MISSING_TARGETS)
    names = tuple(str(label) for label in label_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Class names must be unique and contain at least two classes")
    if not joined.loc[~missing, "Segment"].isin(names).all():
        raise ValueError("Nonmissing Segment outside the fixed declared classification target")
    supervised = joined.loc[~missing].sort_values("record_id", kind="stable").reset_index(drop=True)
    train_indices = np.flatnonzero(supervised.split.eq("train").to_numpy())
    validation_indices = np.flatnonzero(supervised.split.eq("validation").to_numpy())
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Both supervised training and validation must contain records")
    # Detect edits between verification and reading. test.csv is deliberately absent.
    if any(sha256(directory / name) != digest for name, digest in consumed.items()):
        raise ValueError("Development artifacts changed during loading")
    audit = {"version": "guarded-frozen-Segment-development-v1", "split_directory": str(directory),
             "grouping_freeze_sha256": sha256(freeze_path), "completion_sha256": sha256(directory / "completion.json"),
             "consumed_artifact_sha256": consumed, "model_view_contract_sha256": current["contract_sha256"],
             "export_records": counts, "supervised_records": {"train": len(train_indices), "validation": len(validation_indices)},
             "missing_target_records_excluded_from_supervised_view": {
                 role: int((missing & joined.split.eq(role)).sum()) for role in ("train", "validation")},
             "actual_missing_targets_imputed": False, "label_names": list(names),
             "feature_allowlist": list(MODEL_FIELDS), "final_test_ids_protected": len(protected),
             "no_final_test_csv_read_or_hashed": True, "final_test_target_features_or_predictions_read": False,
             "record_and_group_isolation_passed": True,
             "source_and_cache_verifier_passed": bool(verified is not None),
             "class_counts": {role: supervised.loc[supervised.split.eq(role), "Segment"].value_counts().sort_index().to_dict()
                              for role in ("train", "validation")}}
    return FrozenDevelopment(supervised, train_indices, validation_indices, names,
                             frozenset(protected.record_id), frozenset(protected.group_id), audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/split_finalization_20261009/frozen"))
    parser.add_argument("--model-config", type=Path, default=Path("data/processed/segment_full_development_20261009/random/model_config.json"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/processed/segment_full_development_20261009/random/tokenizer"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/split_finalization_20261009/development_preparation"))
    parser.add_argument("--tokenize", action="store_true", help="Verify development tokenization without training")
    parser.add_argument("--train", action="store_true", help="Explicitly run a fresh fixed-epoch Segment baseline")
    parser.add_argument("--device", help="Override training device, retaining frozen model representation")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty preparation/training output directory")
    config = ModelConfig.from_dict(json.loads(args.model_config.read_text(encoding="utf8")))
    if args.device:
        config = replace(config, device=args.device)
    development = load_frozen_development(args.split_dir, config, tokenizer_dir=args.tokenizer_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "policy": "validation only; no final-test API",
                "mode": "explicit_training" if args.train else "tokenization_only" if args.tokenize else "prepare_only",
                "fresh_pretrained_initialization": True, "checkpoint_policy": "fixed final epoch; no validation selection",
                "model_config": config.to_dict(), "development": development.audit,
                "implementation_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))}}
    _write_json(output / "protocol.json", protocol)
    development.frame[["record_id", "group_id", "Segment", "split"]].to_csv(output / "supervised_development_assignments.csv", index=False)
    if args.tokenize or args.train:
        from transformers import AutoTokenizer
        from .segment_transformer import tokenize_corpus
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
        frame = development.frame[["record_id", "Segment", *MODEL_FIELDS]]
        corpus = tokenize_corpus(frame, development.label_names, config, tokenizer=tokenizer)
        _write_json(output / "tokenization.json", {"records": len(corpus), "text_sha256": corpus.text_sha256,
                    "token_sha256": corpus.token_sha256, "development_only": True,
                    "train_records": len(development.train_indices), "validation_records": len(development.validation_indices)})
        if args.train:
            from .segment_transformer import train_transformer
            predictions, manifest = train_transformer(corpus, development.train_indices, development.validation_indices,
                                                     development.label_names, config, output / "model")
            if set(predictions.record_id) != set(development.frame.iloc[development.validation_indices].record_id):
                raise AssertionError("Model prediction IDs differ from validation")
            if set(predictions.record_id) & development.protected_record_ids:
                raise AssertionError("Final test prediction exposure")
            _write_json(output / "training_completion.json", {"fixed_final_epoch": config.epochs,
                        "validation_predictions": len(predictions), "protected_final_test_predictions": 0,
                        "initial_state_sha256": manifest["initial_state_sha256"]})
    _write_json(output / "ready.json", {"mode": protocol["mode"], "checks_passed": True,
                "model_trained": args.train, "final_test_read_or_scored": False, "audit": development.audit})
    print(json.dumps(development.audit, indent=2))


if __name__ == "__main__":
    main()
