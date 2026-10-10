"""Verify label-free rebuilt development tokens against both training manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.features import sha256
from catalogiq.segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items, build_product_texts
from catalogiq.splitting import record_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_cleaned.csv"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/processed/segment_full_development_20261009"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/split_leakage_audit_20261009/model_inputs/development_corpus_verification.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Write verification to a new file")
    started = time.perf_counter()
    run = args.development_dir
    manifests = {strategy: run / strategy / "training_manifest.json" for strategy in ("random", "group_aware")}
    sources = [args.input, run / "development_assignments.csv", run / "random" / "model_config.json",
               run / "random" / "tokenizer" / "tokenizer.json", run / "random" / "tokenizer" / "tokenizer_config.json",
               *manifests.values(), Path(__file__), Path("src/catalogiq/segment_transformer.py")]
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    ids = sorted(set(pd.read_csv(run / "development_assignments.csv", dtype=str, keep_default_na=False,
                                usecols=["record_id"]).record_id))
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                        usecols=[*MODEL_FIELDS, *PROVENANCE])
    frame["record_id"] = record_ids(frame)
    frame = frame.set_index("record_id").loc[ids].reset_index()
    config = ModelConfig.from_dict(json.loads((run / "random" / "model_config.json").read_text(encoding="utf8")))
    texts = build_product_texts(frame, config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(run / "random" / "tokenizer"), local_files_only=True)
    encoded = tokenizer(texts, truncation=True, max_length=config.sequence_length,
                        padding=False, return_attention_mask=True)
    text_digest = _hash_items(texts)
    digest = hashlib.sha256()
    digest.update(_hash_items(ids).encode("ascii"))
    for field in ("input_ids", "attention_mask", "token_type_ids"):
        for values in encoded.get(field, []):
            digest.update(len(values).to_bytes(8, "big"))
            digest.update(np.asarray(values, dtype="<i4").tobytes())
    token_digest = digest.hexdigest()
    comparisons = {}
    for strategy, path in manifests.items():
        manifest = json.loads(path.read_text(encoding="utf8"))
        comparisons[strategy] = {"text_sha256_matches": manifest["text_sha256"] == text_digest,
                                 "token_sha256_matches": manifest["token_sha256"] == token_digest}
        if not all(comparisons[strategy].values()):
            raise ValueError("Rebuilt tokenized inputs differ from saved training corpus")
    if not all(sha256(Path(path)) == expected for path, expected in hashes.items()):
        raise ValueError("Sources changed during verification")
    result = {"records": len(ids), "row_order": "sorted unique development record_id, matching segment_experiment._load_development",
              "text_sha256": text_digest, "token_sha256": token_digest,
              "training_manifest_comparisons": comparisons,
              "raw_target_columns_read": [], "model_weights_predictions_or_test_text_read": False,
              "hash_recipe": "Exact tokenize_corpus digest: _hash_items(record_ids), then per-array8-bytebiglength and little-endianint32bytes",
              "source_sha256": hashes, "source_artifacts_unchanged": True,
              "runtime_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
