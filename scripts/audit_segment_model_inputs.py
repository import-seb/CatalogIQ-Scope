"""Offline, label-free exact transformer-input duplicate audit; no inference.

Run from the repository root with PYTHONPATH=src. Frozen original audit outputs
are not modified. Counts concern identical effective token arrays after the
existing four-field HTML/whitespace processing and fixed128-token truncation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.features import sha256
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, build_product_texts
from catalogiq.splitting import SPLITS, record_ids


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def choose_two(value):
    return value * (value - 1) // 2


def token_signature(encoded, row):
    blocks = []
    for field in ("input_ids", "attention_mask", "token_type_ids"):
        values = np.asarray(encoded[field][row] if field in encoded else [], dtype="<i4")
        blocks.append(field.encode() + b":" + len(values).to_bytes(4, "big") + values.tobytes())
    payload = b"".join(blocks)
    return hashlib.sha256(payload).hexdigest(), payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_cleaned.csv"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/processed/segment_full_development_20261009"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/split_leakage_audit_20261009/model_inputs"))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = args.development_dir
    config_paths = [run / strategy / "model_config.json" for strategy in ("random", "group_aware")]
    values = [json.loads(path.read_text(encoding="utf8")) for path in config_paths]
    if values[0] != values[1]:
        raise ValueError("Model configuration differs between compared strategies")
    config = ModelConfig.from_dict(values[0])
    tokenizer_files = [run / strategy / "tokenizer" / file for strategy in ("random", "group_aware")
                       for file in ("tokenizer.json", "tokenizer_config.json")]
    for file in ("tokenizer.json", "tokenizer_config.json"):
        if sha256(run / "random" / "tokenizer" / file) != sha256(run / "group_aware" / "tokenizer" / file):
            raise ValueError("Stored tokenizers differ")
    source_paths = [args.input, *config_paths, *tokenizer_files, run / "development_assignments.csv",
                    run / "protected_final_test.csv", Path(__file__), Path("src/catalogiq/segment_transformer.py"),
                    Path("src/catalogiq/splitting.py")]
    hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    protocol = {"version": "exact-model-input-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_fields_read": [*MODEL_FIELDS, *PROVENANCE], "raw_target_columns_read": [],
                "model_config": values[0], "signature_fields": ["input_ids", "attention_mask", "token_type_ids"],
                "signature": "SHA256 of field names, length prefixes and little-endian32-bit effective unpadded arrays; fullbytes collision check",
                "tokenizer_kwargs": {"truncation": True, "max_length": config.sequence_length,
                                     "padding": False, "return_attention_mask": True},
                "reuses_existing_product_text_preprocessor": True, "no_model_weights_or_predictions_read": True,
                "no_final_test_inference": True, "source_sha256": hashes}
    write_json(output / "protocol.json", protocol)
    development = pd.read_csv(run / "development_assignments.csv", dtype=str, keep_default_na=False,
                              usecols=["record_id", "group_id", "split", "strategy"])
    protected = pd.read_csv(run / "protected_final_test.csv", dtype=str, keep_default_na=False,
                            usecols=["record_id", "group_id", "split"])
    roles = {}
    for strategy in ("random", "group_aware"):
        role = pd.concat([development.loc[development.strategy.eq(strategy), ["record_id", "group_id", "split"]],
                          protected], ignore_index=True)
        if role.record_id.duplicated().any() or not role.split.isin(SPLITS).all() or role.group_id.str.strip().eq("").any():
            raise ValueError("Invalid role or duplicate ID")
        roles[strategy] = role
    if set(roles["random"].record_id) != set(roles["group_aware"].record_id):
        raise ValueError("Compared role universes differ")
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                        usecols=[*MODEL_FIELDS, *PROVENANCE])
    frame["record_id"] = record_ids(frame)
    frame = frame.loc[frame.record_id.isin(roles["random"].record_id)].sort_values("record_id").reset_index(drop=True)
    if set(frame.record_id) != set(roles["random"].record_id):
        raise ValueError("Source coverage differs from audit universe")
    for strategy, role in roles.items():
        roles[strategy] = role.set_index("record_id").loc[frame.record_id].reset_index()
    texts = build_product_texts(frame, config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(run / "random" / "tokenizer"), local_files_only=True)
    encoded = tokenizer(texts, truncation=True, max_length=config.sequence_length,
                        padding=False, return_attention_mask=True)
    buckets, payloads = defaultdict(list), {}
    lengths = []
    for row in range(len(frame)):
        signature, payload = token_signature(encoded, row)
        if signature in payloads and payloads[signature] != payload:
            raise ValueError("Signature collision")
        payloads[signature] = payload
        buckets[signature].append(row)
        length = len(encoded["input_ids"][row])
        if length != len(encoded["attention_mask"][row]) or not 0 < length <= config.sequence_length:
            raise ValueError("Invalid token sequence")
        lengths.append(length)
    raw_keys = list(frame[list(MODEL_FIELDS)].itertuples(index=False, name=None))
    raw_counts = Counter(raw_keys)
    raw_pairs = sum(choose_two(value) for value in raw_counts.values())
    duplicate_buckets = {signature: members for signature, members in buckets.items() if len(members) > 1}
    cluster_rows, member_rows, summaries = [], [], {}
    split_pairs = list(combinations(SPLITS, 2))
    for signature, members in sorted(duplicate_buckets.items()):
        for row in members:
            member_rows.append({"input_signature_sha256": signature, "record_id": frame.at[row, "record_id"]})
    for strategy, role in roles.items():
        split_values = role.split.tolist()
        total = {"records": len(frame), "unique_input_sequences": len(buckets),
                 "duplicate_clusters": len(duplicate_buckets), "duplicate_records": sum(map(len, duplicate_buckets.values())),
                 "duplicate_pairs": 0, "raw_equal_pairs_within_token_clusters": 0, "raw_unequal_duplicate_pairs": 0,
                 "crossing_clusters": 0, "crossing_pairs": 0, "crossing_records": 0,
                 "raw_unequal_crossing_pairs": 0, "raw_unequal_crossing_records": 0,
                 "crossing_split_pairs": {f"{a}__{b}": 0 for a, b in split_pairs},
                 "raw_unequal_crossing_split_pairs": {f"{a}__{b}": 0 for a, b in split_pairs}}
        crossed_records, raw_unequal_records = set(), set()
        for signature, members in sorted(duplicate_buckets.items()):
            counts = Counter(split_values[row] for row in members)
            raw_by_split = defaultdict(Counter)
            for row in members:
                raw_by_split[raw_keys[row]][split_values[row]] += 1
            raw_equal_pairs = sum(choose_two(sum(values.values())) for values in raw_by_split.values())
            pairs = choose_two(len(members))
            crossing = {f"{a}__{b}": counts[a] * counts[b] for a, b in split_pairs}
            raw_unequal = {f"{a}__{b}": crossing[f"{a}__{b}"] - sum(values[a] * values[b] for values in raw_by_split.values())
                           for a, b in split_pairs}
            total["duplicate_pairs"] += pairs
            total["raw_equal_pairs_within_token_clusters"] += raw_equal_pairs
            total["raw_unequal_duplicate_pairs"] += pairs - raw_equal_pairs
            total["crossing_pairs"] += sum(crossing.values())
            total["raw_unequal_crossing_pairs"] += sum(raw_unequal.values())
            total["crossing_clusters"] += bool(sum(crossing.values()))
            if sum(crossing.values()):
                crossed_records.update(members)
            for row in members:
                own = split_values[row]
                # Has a differently valued raw payload in another role within this exact-input cluster.
                if any(counts[other] > raw_by_split[raw_keys[row]][other] for other in SPLITS if other != own):
                    raw_unequal_records.add(row)
            for key in crossing:
                total["crossing_split_pairs"][key] += crossing[key]
                total["raw_unequal_crossing_split_pairs"][key] += raw_unequal[key]
            cluster_rows.append({"strategy": strategy, "input_signature_sha256": signature,
                                 "token_length": lengths[members[0]], "records": len(members),
                                 "distinct_raw_four_field_payloads": len(raw_by_split), "duplicate_pairs": pairs,
                                 "raw_unequal_pairs": pairs - raw_equal_pairs,
                                 "crossing_pairs": sum(crossing.values()),
                                 "raw_unequal_crossing_pairs": sum(raw_unequal.values()),
                                 **{f"{split}_records": counts[split] for split in SPLITS}, **crossing,
                                 **{f"raw_unequal_{key}": value for key, value in raw_unequal.items()}})
        total["crossing_records"] = len(crossed_records)
        total["raw_unequal_crossing_records"] = len(raw_unequal_records)
        # Every raw-equal pair must necessarily have identical effective inputs.
        if total["raw_equal_pairs_within_token_clusters"] != raw_pairs:
            raise ValueError("Raw-equal pairs did not preserve deterministic token identity")
        summaries[strategy] = total
    pd.DataFrame(cluster_rows).to_csv(output / "duplicate_input_clusters.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output / "duplicate_input_members.csv", index=False)
    unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    if not unchanged:
        raise ValueError("Sources changed during token-input audit")
    summary = {"version": protocol["version"], "scope": "same79915currentdevelopment+protectedtestrecords; target columns not loaded",
               "strategies": summaries, "max_sequence_length": config.sequence_length,
               "records_at_sequence_length_cap": sum(length == config.sequence_length for length in lengths),
               "token_type_ids_present": "token_type_ids" in encoded,
               "token_type_ids_all_zero": all(not any(values) for values in encoded.get("token_type_ids", [])),
               "attention_masks_all_one_unpadded": all(all(values) for values in encoded["attention_mask"]),
               "raw_equal_pairs_in_entire_scope": raw_pairs, "source_artifacts_unchanged": unchanged,
               "runtime_seconds": time.perf_counter() - started,
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "transformers": version("transformers"), "tokenizers": version("tokenizers")},
               "no_target_columns_model_weights_predictions_or_test_inference_consumed": True,
               "limitations": ["Exact effective input equivalence, not all semantic product relationships.",
                               "Collisions can reflect harmless formatting or loss of distinguishing information through truncation.",
                               "Counts do not estimate how much model accuracy is inflated.",
                               "Saved CSVs contain only signatures/record IDs/roles/counts, no product names or test text."]}
    write_json(output / "summary.json", summary)
    print(json.dumps(summaries, indent=2))
    print(f"Token-input audit completed in {summary['runtime_seconds']:.1f}s: {output}")


if __name__ == "__main__":
    main()
