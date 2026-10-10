"""Freeze and cache exact model-input groups for the complete cleaned universe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.features import sha256
from catalogiq.model_input_groups import SIGNATURE_PREFIX, exact_model_input_groups, model_view_contract
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items
from catalogiq.splitting import record_ids


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_cleaned.csv"))
    parser.add_argument("--model-config", type=Path, default=Path("data/processed/segment_full_development_20261009/random/model_config.json"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/processed/segment_full_development_20261009/random/tokenizer"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/split_finalization_20261009/model_view"))
    parser.add_argument("--token-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty cache directory")
    output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.from_dict(json.loads(args.model_config.read_text(encoding="utf8")))
    contract = model_view_contract(config, tokenizer_dir=args.tokenizer_dir)
    sources = [args.input, args.model_config, Path(__file__), Path("src/catalogiq/model_input_groups.py"),
               Path("src/catalogiq/segment_transformer.py"), *[path for path in args.tokenizer_dir.iterdir() if path.is_file()]]
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    protocol = {"version": "full-universe-model-view-freeze-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "model_view_frozen_before_final_grouping_and_new_test_allocation",
                "input_sha256": sha256(args.input), "model_view_contract": contract,
                "source_sha256": hashes, "target_columns_read": [], "model_inference_or_predictions_used": False,
                "policy": "Identical effective input arrays are hard must-links. Any representation drift requires rebuilding groups and final splits."}
    write_json(output / "protocol.json", protocol)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=[*MODEL_FIELDS, *PROVENANCE])
    frame["record_id"] = record_ids(frame)
    frame = frame.sort_values("record_id").reset_index(drop=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    groups, edges, stats = exact_model_input_groups(frame, frame.record_id, config,
        tokenizer_dir=args.tokenizer_dir, token_batch_size=args.token_batch_size, progress=print)
    signatures = pd.DataFrame({"record_id": frame.record_id, "input_signature_sha256": [group[len(SIGNATURE_PREFIX):] for group in groups],
                               "group_id": groups})
    signatures.to_csv(output / "record_signatures.csv", index=False)
    pd.DataFrame([{"left_record_id": frame.at[a, "record_id"], "right_record_id": frame.at[b, "record_id"], "reason": reason, "score": score}
                  for a, b, reason, score in edges]).to_csv(output / "must_link_edges.csv", index=False)
    if not all(sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise ValueError("Frozen model-view sources changed during cache generation")
    manifest = {"version": protocol["version"], "input_sha256": protocol["input_sha256"],
                "sorted_record_ids_sha256": _hash_items(sorted(frame.record_id)),
                "model_view_contract": stats["model_view_contract"],
                "signature_csv_sha256": sha256(output / "record_signatures.csv"),
                "must_link_edges_sha256": sha256(output / "must_link_edges.csv"),
                "protocol_sha256": sha256(output / "protocol.json"), "source_artifacts_unchanged": True, "statistics": stats,
                "previous_development_corpus_validation": "data/processed/split_leakage_audit_20261009/model_inputs/development_corpus_verification.json"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in stats.items() if key != "model_view_contract"}, indent=2))


if __name__ == "__main__":
    main()
