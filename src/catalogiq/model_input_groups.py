"""Hard equality constraints for a frozen classifier's actual text inputs.

Product grouping and target-specific allocation remain separate. These links
mean identical effective token inputs, not identical commercial products. A cache
is valid only for its registered tokenizer, preprocessing, limits and universe;
changing that representation invalidates the cached groups and final split seal.
No model, target labels or predictions are required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .features import sha256
from .segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items, build_product_texts
from .splitting import Components

SIGNATURE_PREFIX = "model_input_"
VERSION = "exact-effective-input-groups-v1"


def _ids(frame, ids):
    raw = np.asarray(ids)
    if raw.ndim != 1 or pd.isna(raw).any():
        raise ValueError("Record IDs must be present and one-dimensional")
    values = np.asarray(ids, dtype=str)
    if values.ndim != 1 or len(values) != len(frame) or not len(values):
        raise ValueError("One record ID per nonempty input row is required")
    if len(set(values)) != len(values) or any(not value.strip() for value in values):
        raise ValueError("Record IDs must be unique and nonblank")
    if "record_id" in frame and frame.record_id.astype(str).tolist() != values.tolist():
        raise ValueError("Record IDs do not align with input rows")
    return values


def model_view_contract(config=None, *, tokenizer_dir=None, tokenizer_fingerprint=None):
    """Register every representation choice that can invalidate must-link groups.

    Real cached tokenizers are identified by their saved file bytes. Injected
    test/custom tokenizers need an explicit fingerprint from the caller; cache
    correctness then depends on that caller accurately identifying its adapter.
    """
    config = config or ModelConfig()
    if tokenizer_dir is not None and tokenizer_fingerprint is not None:
        raise ValueError("Choose saved tokenizer files or an injected tokenizer fingerprint")
    if tokenizer_dir is not None:
        folder = Path(tokenizer_dir)
        paths = [path for path in sorted(folder.iterdir()) if path.is_file()]
        if not paths or not (folder / "tokenizer.json").exists() or not (folder / "tokenizer_config.json").exists():
            raise ValueError("A complete saved tokenizer directory is required")
        fingerprint = {path.name: sha256(path) for path in paths}
    elif tokenizer_fingerprint:
        fingerprint = dict(sorted(tokenizer_fingerprint.items()))
    else:
        raise ValueError("Tokenizer byte identity or an explicit injected fingerprint is required")
    contract = {"version": VERSION, "model_name": config.model_name, "revision": config.revision,
                "features": list(MODEL_FIELDS), "field_character_limits": list(config.field_character_limits),
                "sequence_length": config.sequence_length,
                "tokenizer_kwargs": {"truncation": True, "max_length": config.sequence_length,
                                     "padding": False, "return_attention_mask": True},
                "tokenizer_fingerprint": fingerprint,
                "preprocessor_source_sha256": sha256(Path(__file__).with_name("segment_transformer.py")),
                "signature_implementation_source_sha256": sha256(Path(__file__)),
                "signature_fields": ["input_ids", "attention_mask", "token_type_ids"],
                "signature_recipe": "field name; colon;4-bytebiglength;little-endianint32array;SHA256",
                "cache_invalidation": "Any representation, source-universe or tokenization implementation drift requires regenerating grouping and split seals."}
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract


def _signature(encoded, row):
    payload = b""
    for field in ("input_ids", "attention_mask", "token_type_ids"):
        values = np.asarray(encoded[field][row] if field in encoded else [], dtype="<i4")
        payload += field.encode() + b":" + len(values).to_bytes(4, "big") + values.tobytes()
    return hashlib.sha256(payload).hexdigest(), payload


def exact_model_input_groups(frame, ids, config=None, *, tokenizer=None,
                             tokenizer_dir=None, tokenizer_fingerprint=None,
                             token_batch_size=4096, progress=None):
    """Return (signature-group IDs, deterministic star edges, metadata).

    Star edges are an economical exact representation of every identical-input
    equivalence class. Each edge uses original row positions and score1.0; they
    take precedence as must-link constraints in a downstream component union.
    Text construction uses only the existing MODEL_FIELDS allowlist. Batch size
    changes throughput/memory only: there is no padding, fitting or inference.
    """
    started = time.perf_counter()
    config = config or ModelConfig()
    values = _ids(frame, ids)
    if type(token_batch_size) is not int or token_batch_size <= 0:
        raise ValueError("token_batch_size must be a positive integer")
    if not config.local_files_only:
        raise ValueError("Exact-input grouping requires a frozen offline tokenizer")
    if tokenizer is not None and tokenizer_dir is not None:
        raise ValueError("Do not combine an injected tokenizer with a saved tokenizer directory")
    if tokenizer is None and tokenizer_dir is None:
        raise ValueError("Supply a frozen saved tokenizer directory or an injected tokenizer")
    contract = model_view_contract(config, tokenizer_dir=tokenizer_dir, tokenizer_fingerprint=tokenizer_fingerprint)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    # Remove every other column before preprocessing, including all targets.
    view = frame.reindex(columns=MODEL_FIELDS)
    texts = build_product_texts(view, config)
    signature_values, payloads, lengths = [], {}, []
    for offset in range(0, len(frame), token_batch_size):
        batch = texts[offset:offset + token_batch_size]
        encoded = tokenizer(batch, truncation=True, max_length=config.sequence_length,
                            padding=False, return_attention_mask=True)
        if len(encoded.get("input_ids", [])) != len(batch) or len(encoded.get("attention_mask", [])) != len(batch):
            raise ValueError("Tokenizer changed batch coverage")
        for row in range(len(batch)):
            length = len(encoded["input_ids"][row])
            if (not 0 < length <= config.sequence_length or len(encoded["attention_mask"][row]) != length
                    or ("token_type_ids" in encoded and len(encoded["token_type_ids"][row]) != length)):
                raise ValueError("Invalid effective token arrays")
            signature, payload = _signature(encoded, row)
            if signature in payloads and payloads[signature] != payload:
                raise ValueError("Effective-input signature hash collision")
            payloads[signature] = payload
            signature_values.append(signature)
            lengths.append(length)
        if progress:
            progress(f"Frozen model input equality: {min(offset + len(batch), len(frame)):,}/{len(frame):,} records")
    buckets = defaultdict(list)
    for row, signature in enumerate(signature_values):
        buckets[signature].append(row)
    edges = []
    for members in buckets.values():
        ordered = sorted(members, key=lambda row: values[row])
        edges.extend((ordered[0], row, "exact_effective_model_input", 1.0) for row in ordered[1:])
    edges.sort(key=lambda edge: (values[edge[0]], values[edge[1]]))
    groups = np.asarray([SIGNATURE_PREFIX + signature for signature in signature_values], dtype=str)
    sizes = [len(members) for members in buckets.values()]
    stats = {"version": VERSION, "records": len(frame), "unique_input_groups": len(buckets),
             "duplicate_clusters": sum(size > 1 for size in sizes),
             "duplicate_records": sum(size for size in sizes if size > 1),
             "duplicate_pairs": sum(size * (size - 1) // 2 for size in sizes),
             "must_link_star_edges": len(edges), "largest_equivalence_class": max(sizes),
             "records_at_sequence_length_cap": sum(length == config.sequence_length for length in lengths),
             "model_view_contract": contract, "grouping_target_columns_used": [],
             "model_weights_predictions_or_inference_used": False,
             "runtime_seconds": time.perf_counter() - started}
    return groups, edges, stats


def merge_model_input_groups(rule_groups, ids, signature_groups, *, prefix="final"):
    """Union existing product groups with every hard input-equality constraint.

    This never splits an existing product group and never caps group sizes. It
    reports every equality class that connects different product components;
    reviewers must still audit resulting components and product-family regressions.
    """
    values = np.asarray(ids, dtype=str)
    original = np.asarray(rule_groups, dtype=str)
    exact = np.asarray(signature_groups, dtype=str)
    if (pd.isna(np.asarray(ids)).any() or not len(values) or values.ndim != 1
            or len(set(values)) != len(values) or any(not value.strip() for value in values)):
        raise ValueError("Record IDs must be nonempty, unique and one-dimensional")
    if (pd.isna(np.asarray(rule_groups)).any() or pd.isna(np.asarray(signature_groups)).any()
            or original.ndim != 1 or exact.ndim != 1 or len(original) != len(values) or len(exact) != len(values)
            or any(not group.strip() for group in original) or any(not group.strip() for group in exact)):
        raise ValueError("One nonblank product and model-input group per record is required")
    components = Components(len(values))
    for groups in (original, exact):
        first = {}
        for row, group in enumerate(groups):
            if group in first:
                components.union(first[group], row)
            else:
                first[group] = row
    result = components.groups(values, prefix)
    evidence = pd.DataFrame({"record_id": values, "rule_group_id": original,
                             "model_input_group_id": exact, "final_group_id": result})
    if (evidence.groupby("rule_group_id").final_group_id.nunique().gt(1).any()
            or evidence.groupby("model_input_group_id").final_group_id.nunique().gt(1).any()):
        raise AssertionError("Must-link union violated group isolation")
    merged = []
    for group, rows in evidence.groupby("model_input_group_id", sort=True):
        parents = sorted(set(rows.rule_group_id))
        if len(parents) > 1:
            merged.append({"model_input_group_id": group, "records": len(rows),
                           "product_components_connected": len(parents),
                           "prior_rule_group_ids": parents, "final_group_id": rows.final_group_id.iloc[0]})
    stats = {"records": len(values), "product_groups_before": len(set(original)),
             "groups_after_hard_equality_union": len(set(result)),
             "components_reduced": len(set(original)) - len(set(result)),
             "equality_classes_connecting_different_product_groups": len(merged),
             "merge_evidence": merged, "existing_groups_preserved": True,
             "all_exact_effective_inputs_together": True, "targets_used": [],
             "interpretation": "Hard input equality prevents deterministic feature duplication; final groups still require semantic/regression auditing."}
    return result, stats


def load_model_input_cache(directory, ids, config=None, *, tokenizer_dir=None,
                           tokenizer_fingerprint=None, expected_input_sha256):
    """Reject stale/tampered caches before using them in a split.

    The source checksum and sorted IDs bind the universe. The model-view
    contract binds tokenizer bytes, preprocessor bytes, max length and field
    limits. Final split protocols should pin this cache manifest as well.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    current = model_view_contract(config, tokenizer_dir=tokenizer_dir, tokenizer_fingerprint=tokenizer_fingerprint)
    if manifest["model_view_contract"] != current:
        raise ValueError("Model representation changed; cached grouping and final splits must be regenerated")
    if manifest["input_sha256"] != expected_input_sha256:
        raise ValueError("Source input changed; model-input cache is stale")
    if np.asarray(ids).ndim != 1 or pd.isna(np.asarray(ids)).any():
        raise ValueError("Cache IDs must be present and one-dimensional")
    values = np.asarray(ids, dtype=str)
    if len(set(values)) != len(values) or any(not value.strip() for value in values):
        raise ValueError("Cache record IDs must be unique and nonblank")
    if manifest["sorted_record_ids_sha256"] != _hash_items(sorted(values.tolist())):
        raise ValueError("Record universe changed; model-input cache is stale")
    path = directory / "record_signatures.csv"
    if sha256(path) != manifest["signature_csv_sha256"]:
        raise ValueError("Model-input signature cache was modified")
    signatures = pd.read_csv(path, dtype=str, keep_default_na=False)
    if signatures.record_id.duplicated().any() or set(signatures.record_id) != set(values):
        raise ValueError("Cache signature rows do not cover the universe")
    if not signatures.input_signature_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid cached signatures")
    if not signatures.group_id.eq(SIGNATURE_PREFIX + signatures.input_signature_sha256).all():
        raise ValueError("Cached group IDs conflict with signatures")
    return signatures.set_index("record_id").loc[values].group_id.to_numpy(), manifest
