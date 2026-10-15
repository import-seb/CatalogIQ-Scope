"""Label-free equality audit for a saved experimental model representation.

This reads final-test identity and feature columns only. It never changes
groups, memberships, defaults or model files, and performs no inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .features import sha256
from .model_input_groups import _signature as token_signature
from .segment_category_ablation import (
    CATEGORY, _check_sources, _json, _write_json, build_category_texts,
    load_category_development, load_tokenizer,
)
from .segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items

VERSION = "saved-experiment-input-equality-v1"
FEATURE_FIELDS = (CATEGORY, *MODEL_FIELDS)
IDENTITY_FIELDS = ("record_id", "group_id", "split")
ROLES = ("train", "validation", "test")
PAIR_COLUMNS = ("input_signature_sha256", "left_record_id", "right_record_id",
                "left_group_id", "right_group_id", "left_split", "right_split")


def _validate_membership(frame):
    if frame.empty or not set(IDENTITY_FIELDS) <= set(frame):
        raise ValueError("Nonempty partition identities are required")
    if (frame[list(IDENTITY_FIELDS)].isna().any().any()
            or any(frame[field].astype(str).str.strip().eq("").any() for field in IDENTITY_FIELDS)
            or frame.record_id.duplicated().any() or not frame.split.isin(ROLES).all()):
        raise ValueError("Invalid or duplicate record IDs, group IDs or split roles")
    if frame.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Existing product group crosses partition boundaries")


def load_partition_features(split_dir):
    """Read only identity/feature allowlists, checking the frozen export seals."""
    split_dir = Path(split_dir)
    expected = _json(split_dir / "completion.json")["artifact_sha256"]
    consumed = {str(split_dir / name): expected[name] for name in
                ("assignments.csv", "train.csv", "validation.csv", "test.csv")}
    _check_sources(consumed)
    assignments = pd.read_csv(split_dir / "assignments.csv", dtype=str,
        keep_default_na=False, encoding="utf-8-sig", usecols=list(IDENTITY_FIELDS))
    _validate_membership(assignments)
    parts = []
    for role in ROLES:
        piece = pd.read_csv(split_dir / f"{role}.csv", dtype=str, keep_default_na=False,
            encoding="utf-8-sig", usecols=[*IDENTITY_FIELDS, *FEATURE_FIELDS])
        _validate_membership(piece)
        if not piece.split.eq(role).all():
            raise ValueError("Export split role differs from its partition")
        expected_members = assignments.loc[assignments.split.eq(role), list(IDENTITY_FIELDS)].sort_values("record_id").reset_index(drop=True)
        actual = piece[list(IDENTITY_FIELDS)].sort_values("record_id").reset_index(drop=True)
        if not actual.equals(expected_members):
            raise ValueError("Export identity or group differs from frozen assignments")
        parts.append(piece)
    frame = pd.concat(parts, ignore_index=True).sort_values("record_id", kind="stable").reset_index(drop=True)
    _validate_membership(frame)
    if set(frame.record_id) != set(assignments.record_id):
        raise ValueError("Partition exports do not cover frozen record membership")
    _check_sources(consumed)
    return frame


def _tokenize(frame, config, *, tokenizer, category_character_limit=512,
              token_batch_size=4096, progress=None):
    _validate_membership(frame)
    if type(token_batch_size) is not int or token_batch_size < 1:
        raise ValueError("token_batch_size must be a positive integer")
    if CATEGORY not in frame:
        raise ValueError("ProductCategory is required; a target Category is not a substitute")
    # Strip every target, provenance field and other feature before formatting.
    texts = build_category_texts(frame.reindex(columns=FEATURE_FIELDS), config,
                                 category_character_limit)
    arrays = {"input_ids": [], "attention_mask": []}
    payloads, rows = {}, []
    types_present = None
    for offset in range(0, len(frame), token_batch_size):
        encoded = tokenizer(texts[offset:offset + token_batch_size], truncation=True,
            max_length=config.sequence_length, padding=False, return_attention_mask=True)
        count = min(token_batch_size, len(frame) - offset)
        present = "token_type_ids" in encoded
        if types_present is not None and types_present != present:
            raise ValueError("Tokenizer changed token-type availability between batches")
        types_present = present
        fields = ("input_ids", "attention_mask", *(("token_type_ids",) if present else ()))
        if any(len(encoded.get(field, [])) != count for field in fields):
            raise ValueError("Tokenizer changed input coverage")
        for row in range(count):
            length = len(encoded["input_ids"][row])
            if (not 0 < length <= config.sequence_length
                    or any(np.asarray(encoded[field][row]).ndim != 1 or len(encoded[field][row]) != length for field in fields)):
                raise ValueError("Invalid effective token arrays")
            signature, payload = token_signature(encoded, row)
            if signature in payloads and payloads[signature] != payload:
                raise ValueError("Effective input signature hash collision")
            payloads[signature] = payload
            for field in fields:
                arrays.setdefault(field, []).append(np.asarray(encoded[field][row], dtype=np.int32))
            identity = frame.iloc[offset + row]
            rows.append({**{field: identity[field] for field in IDENTITY_FIELDS},
                         "input_signature_sha256": signature, "token_count": length})
        if progress:
            progress(f"Experimental input equality: {offset + count:,}/{len(frame):,} records")
    return pd.DataFrame(rows), arrays, texts


def tokenize_signatures(frame, config, *, tokenizer, category_character_limit=512,
                        token_batch_size=4096, progress=None):
    """Sign the complete unpadded token arrays without reading target columns."""
    return _tokenize(frame, config, tokenizer=tokenizer,
        category_character_limit=category_character_limit,
        token_batch_size=token_batch_size, progress=progress)[0]


def summarize_signatures(signatures):
    """Enumerate every cross-partition pair and its original product groups."""
    _validate_membership(signatures)
    if not {"input_signature_sha256", "token_count"} <= set(signatures):
        raise ValueError("Token signatures and lengths are required")
    crossings, affected_ids = [], set()
    split_pairs = {f"{left}__{right}": 0 for left, right in combinations(ROLES, 2)}
    duplicates, duplicate_pairs, crossing_clusters, max_size = 0, 0, 0, 0
    for signature, members in signatures.groupby("input_signature_sha256", sort=True):
        max_size = max(max_size, len(members))
        duplicates += len(members) > 1
        duplicate_pairs += len(members) * (len(members) - 1) // 2
        if members.split.nunique() < 2:
            continue
        crossing_clusters += 1
        affected_ids.update(members.record_id)
        buckets = {role: members.loc[members.split.eq(role)].sort_values("record_id") for role in ROLES}
        for left, right in combinations(ROLES, 2):
            split_pairs[f"{left}__{right}"] += len(buckets[left]) * len(buckets[right])
            for first in buckets[left].itertuples(index=False):
                for second in buckets[right].itertuples(index=False):
                    crossings.append({"input_signature_sha256": signature,
                        "left_record_id": first.record_id, "right_record_id": second.record_id,
                        "left_group_id": first.group_id, "right_group_id": second.group_id,
                        "left_split": left, "right_split": right})
    pairs = pd.DataFrame(crossings, columns=PAIR_COLUMNS)
    affected = signatures.loc[signatures.record_id.isin(affected_ids)].sort_values("record_id").reset_index(drop=True)
    summary = {"records": len(signatures), "records_by_partition": signatures.split.value_counts().sort_index().to_dict(),
        "unique_signatures": signatures.input_signature_sha256.nunique(),
        "duplicate_clusters": int(duplicates), "duplicate_pairs": int(duplicate_pairs),
        "largest_equivalence_class": max_size, "crossing_clusters": crossing_clusters,
        "crossing_pairs": len(pairs), "crossing_rows": len(affected), "crossing_split_pairs": split_pairs,
        "affected_group_ids": sorted(set(affected.group_id)),
        "group_isolation_passed": True, "exact_input_isolation_passed": len(pairs) == 0}
    return summary, pairs, affected


def _verify_artifacts(directory, completion):
    hashes = {}
    for name, expected in completion["artifact_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes its experiment directory")
        hashes[str(path)] = expected
    _check_sources(hashes)
    return hashes


def _corpus_digests(ordered_ids, positions, arrays, texts):
    digest = hashlib.sha256()
    digest.update(_hash_items(ordered_ids).encode("ascii"))
    for field in ("input_ids", "attention_mask", "token_type_ids"):
        for record_id in ordered_ids if field in arrays else ():
            row = arrays[field][positions[record_id]].astype("<i4", copy=False)
            digest.update(len(row).to_bytes(8, "big"))
            digest.update(row.tobytes())
    return {"text_sha256": _hash_items([texts[positions[record_id]] for record_id in ordered_ids]),
            "token_sha256": digest.hexdigest()}


def audit_experiment(split_dir, experiment_dir, output_dir, *, token_batch_size=4096, progress=print):
    """Verify the successful experiment and audit all frozen partitions."""
    started = perf_counter()
    split_dir, experiment_dir, output_dir = (Path(path).resolve() for path in (split_dir, experiment_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new equality-audit output directory")
    protocol = _json(experiment_dir / "protocol.json")
    completion = _json(experiment_dir / "completion.json")
    if (sha256(experiment_dir / "protocol.json") != completion["protocol_sha256"]
            or _json(experiment_dir / "protocol_freeze.json")["protocol_sha256"] != completion["protocol_sha256"]):
        raise ValueError("Successful experiment protocol seal changed")
    hashes = _verify_artifacts(experiment_dir, completion)
    hashes.update(protocol["source_sha256"])
    split_completion = _json(split_dir / "completion.json")
    hashes.update(_verify_artifacts(split_dir, split_completion))
    hashes.update({str(path): sha256(path) for path in (Path(__file__).resolve(),
        Path(__file__).with_name("model_input_groups.py").resolve(),
        split_dir / "completion.json", experiment_dir / "completion.json")})
    _check_sources(hashes)
    manifest = _json(experiment_dir / "with_category" / "training_manifest.json")
    config = ModelConfig.from_dict(protocol["model_config"])
    representation = protocol["representation"]
    if (config.sequence_length != 256 or representation["category_position"] != "prefix"
            or representation["combined_token_limit"] != config.sequence_length
            or representation["category_fields"] != list(FEATURE_FIELDS)
            or manifest["input_fields"] != list(FEATURE_FIELDS)
            or ModelConfig.from_dict(manifest["config"]) != config):
        raise ValueError("Expected the saved five-field, 256-token ProductCategory experiment")
    if sha256(split_dir / "grouping_freeze.json") != protocol["development_audit"]["grouping_freeze_sha256"]:
        raise ValueError("Experiment does not use this frozen grouping protocol")
    frame = load_partition_features(split_dir)
    tokenizer = load_tokenizer(experiment_dir / "with_category" / "tokenizer")
    signatures, arrays, texts = _tokenize(frame, config, tokenizer=tokenizer,
        category_character_limit=representation["category_character_limit"],
        token_batch_size=token_batch_size, progress=progress)
    development = load_category_development(split_dir, config)
    ids = list(development.frame.record_id)
    positions = dict(zip(frame.record_id, range(len(frame))))
    digests = _corpus_digests(ids, positions, arrays, texts)
    expected_digests = protocol["tokenization"]["with_category"]
    if any(value != manifest[key] or value != expected_digests[key] for key, value in digests.items()):
        raise ValueError("Reconstructed text or token arrays differ from the successful experiment")
    for role, indices in (("train", development.train_indices), ("validation", development.validation_indices)):
        ordered = development.frame.iloc[indices].record_id.tolist()
        key = "train_record_ids_sha256" if role == "train" else "validation_record_ids_sha256"
        if _hash_items(ordered) != manifest[key] or _hash_items(ordered) != protocol[key]:
            raise ValueError("Successful experiment training or validation membership differs")
    predicted = pd.read_csv(experiment_dir / "with_category" / "predictions.csv", dtype=str,
        keep_default_na=False, usecols=["record_id", "true_label", "predicted_label"])
    validation = development.frame.iloc[development.validation_indices]
    if (predicted.record_id.duplicated().any() or set(predicted.record_id) != set(validation.record_id)
            or predicted.set_index("record_id").true_label.to_dict() != validation.set_index("record_id").Segment.to_dict()):
        raise ValueError("Saved validation prediction membership or truth differs")
    from sklearn.metrics import accuracy_score, f1_score
    metrics = {"validation_accuracy": accuracy_score(predicted.true_label, predicted.predicted_label),
        "validation_macro_f1": f1_score(predicted.true_label, predicted.predicted_label,
            labels=manifest["label_names"], average="macro", zero_division=0)}
    summary, crossings, affected = summarize_signatures(signatures)
    _check_sources(hashes)
    audit = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_directory": str(split_dir), "experiment_directory": str(experiment_dir),
        "feature_columns_read": list(FEATURE_FIELDS), "final_test_columns_read": [*IDENTITY_FIELDS, *FEATURE_FIELDS],
        "final_test_target_columns_read": [], "final_test_predictions_or_performance_used": False,
        "scope": "All frozen records, including development records without Segment labels; test eligibility is not inspected.",
        "signature_fields": ["input_ids", "attention_mask", "token_type_ids"],
        "signature_recipe": "SHA256 of field names, 4-byte big-endian lengths and little-endian int32 unpadded arrays; full-payload collision check",
        "tokenizer_directory": str(experiment_dir / "with_category" / "tokenizer"),
        "tokenizer_kwargs": {"truncation": True, "max_length": 256, "padding": False, "return_attention_mask": True},
        "representation": representation, "model_config": config.to_dict(), "corpus_digests": digests,
        "development_reconstruction_matches_saved_experiment": True,
        "training_records": len(development.train_indices), "validation_records": len(development.validation_indices),
        "training_manifest": {key: manifest[key] for key in ("initial_state_sha256", "final_state_sha256",
            "completed_epochs", "optimizer_steps", "warmup_steps")},
        "checkpoint_sha256": sha256(experiment_dir / "with_category" / "checkpoint.pt"),
        "model_selection_status": "Experimental comparison candidate; no official model has been selected.",
        "source_sha256": hashes, "sources_unchanged": True, "split_memberships_modified": False,
        "frozen_split_contract_modified": False, "project_defaults_modified": False,
        "checkpoints_modified": False, "retrained": False, **metrics, **summary,
        "interpretation": "Exact equality only; zero detected collisions does not establish absence of near-duplicate or other leakage.",
        "conflict_resolution": ("No membership changes needed for exact equality under this representation."
            if summary["exact_input_isolation_passed"] else
            "Preserve V4. Record conflicts and freeze a separately versioned grouping/split protocol if these pairs must be co-located; do not silently repair or tune against the existing test."),
        "runtime_seconds": perf_counter() - started}
    output_dir.mkdir(parents=True, exist_ok=False)
    signatures.to_csv(output_dir / "record_signatures.csv", index=False, lineterminator="\n")
    crossings.to_csv(output_dir / "cross_split_pairs.csv", index=False, lineterminator="\n")
    affected.to_csv(output_dir / "affected_records.csv", index=False, lineterminator="\n")
    _write_json(output_dir / "audit.json", audit)
    _check_sources(hashes)
    _write_json(output_dir / "completion.json", {"artifact_sha256": {
        path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}})
    progress(f"Cross-partition exact input pairs: {summary['crossing_split_pairs']}")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    audit_experiment(args.split_dir, args.experiment_dir, args.output_dir,
        token_batch_size=args.token_batch_size, progress=lambda value: print(value, flush=True))


if __name__ == "__main__":
    main()
