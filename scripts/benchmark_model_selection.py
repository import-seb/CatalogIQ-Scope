"""Inference-only Sub-Segment comparison of the explicitly verified team handoff.

Run from the repository root. No training, downloads, artifact rewriting or
protected-test features/labels. See the November 6 model-selection findings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

MODEL_PINS = {
    "base": "c239c533f77b1d5fa4a3f8fb4a46fa90896506b508d595fa094d771ecea09b53",
    "category": "cce91a0765fd1f3ab93414da7b6b2b4ac6ff17d0ca039dea1efe75b7335d464d",
}
COMPARISON_PIN = "c78962ddf95c7fb0e052ce5ce2ab2accb146d0a62fa5c7b5c8c6bada9ecb94ff"
CHECKPOINT_PIN = "02f4fbb0f6ee55c80eb9068f0c8be9cc2a56982edd6bfdd8a2436918aa10997f"
FIELDS = ["record_id", "ProductName", "ProductBrand", "ProductDescription",
          "ProductContents", "ProductCategory"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new output file")
    root = Path(__file__).resolve().parents[1]
    handoff = args.handoff_dir.resolve()
    tracked = {}

    def verify(path, expected):
        path = Path(path).resolve()
        require(digest(path) == expected, f"Checksum mismatch: {path.name}")
        tracked[path] = expected

    # User identified this as Max's handoff. Verify the package seal, the
    # comparison's independently documented PR #23 fingerprint, and model pins
    # before importing joblib or deserializing anything.
    verify(handoff / "comparison/comparison.json", COMPARISON_PIN)
    package = read_json(handoff / "manifest.json")
    for name, expected in package["files_sha256"].items():
        path = (handoff / name).resolve()
        require(path.is_relative_to(handoff), "Package member escapes handoff")
        verify(path, expected)
    for arm, pin in MODEL_PINS.items():
        verify(handoff / "Sub-Segment" / arm / "model.joblib", pin)

    run = root / "data/processed/predictions_contract_v2_20261031/sub_segment"
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        verify(run / name, expected)
    model_dir = root / "data/processed/minilm_remaining_targets_v1_20261029/handoff/sub_segment"
    verify(model_dir / "checkpoint.pt", CHECKPOINT_PIN)
    for path, expected in manifest["source_input_sha256"].items():
        p = Path(path)
        if p.is_relative_to(model_dir):
            verify(p, expected)

    context = pd.read_csv(run / "validation_context.csv", dtype=str)
    protected = pd.read_csv(run / "protected_test_identity_registry.csv", dtype=str)
    require(not set(context.record_id) & set(protected.record_id), "Protected ID overlap")
    require(not set(context.group_id) & set(protected.group_id), "Protected group overlap")
    require(context.split.eq("validation").all(), "Nonvalidation context")
    ids = sorted(context.record_id)[:64]
    sources = root / "data/processed/split_integration_reproduced_20261010"
    pieces = []
    for role in ("train", "validation"):
        path = sources / f"{role}.csv"
        verify(path, manifest["source_input_sha256"][str(path)])
        frame = pd.read_csv(path, usecols=FIELDS, dtype=str, keep_default_na=False)
        pieces.append(frame.loc[frame.record_id.isin(ids)])
    sample = pd.concat(pieces).set_index("record_id").loc[ids].reset_index()
    require(len(sample) == 64 and not sample.record_id.duplicated().any(), "Incomplete sample")
    labels = [manifest["label_mapping"][f"probability_{i}"] for i in range(49)]
    expected = pd.read_csv(run / "predictions.csv", dtype={"record_id": str}).set_index("record_id").loc[ids]
    probability_columns = [f"probability_{i}" for i in range(49)]

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import joblib
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from catalogiq.classical_baseline import predict_classical
    from catalogiq.segment_category_ablation import build_category_texts
    from catalogiq.segment_transformer import ModelConfig, _make_classifier, _state_sha256

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = {
        "date": "2026-11-06", "target": "Sub-Segment", "device": "cpu",
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                     "logical_cpus": os.cpu_count(), "numerical_threads": 2},
        "versions": {n: importlib.metadata.version(n) for n in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib",
                      "threadpoolctl", "torch", "transformers", "tokenizers")},
        "python": platform.python_version(),
        "sample": "First 64 sorted approved Sub-Segment validation IDs; no label-based selection",
        "sample_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "scope": "In-memory product frame through original formatting, vectorization/tokenization, prediction and output frame with full probabilities; no disk IO or model load",
        "single_record_repeats": 3, "single_record_rows": 64,
        "batch_size": 64, "batch_repeats": 5, "warmup_batches": 3,
        "memory_bytes": None,
        "memory_scope": "Not measured: no reliable isolated per-model resident-memory measurement",
        "training_performed": False, "protected_test_features_or_labels_read": False,
        "limitations": ["Small deterministic sample; no service/network/concurrency overhead",
                       "CPU only; does not predict GPU or production performance",
                       "Native pipeline representations differ in ordering/truncation",
                       "Load time is a single warm-filesystem observation, not cold-start p95"],
        "models": {},
    }

    def measure(name, predict, load_seconds, reference, tolerance):
        for _ in range(3):
            predict(sample)
        actual = predict(sample)
        require(np.array_equal(actual.predicted_label.to_numpy(), reference.predicted_label.to_numpy()), "Prediction disagreement")
        require(np.allclose(actual[probability_columns].to_numpy(), reference[probability_columns].to_numpy(), atol=tolerance, rtol=0), "Probability disagreement")
        singles = [sample.iloc[[i]] for i in range(len(sample))]
        latency = []
        for _ in range(3):
            for row in singles:
                started = time.perf_counter()
                predict(row)
                latency.append(1000 * (time.perf_counter() - started))
        batches = []
        for _ in range(5):
            started = time.perf_counter()
            predict(sample)
            batches.append(time.perf_counter() - started)
        result = {"load_seconds": load_seconds,
                  "single_p50_ms": float(np.percentile(latency, 50)),
                  "single_p95_ms": float(np.percentile(latency, 95)),
                  "single_measurements_ms": latency,
                  "batch_seconds": batches,
                  "batch_records_per_second": len(sample) / float(np.median(batches)),
                  "saved_predictions_reproduced": True}
        output["models"][name] = result
        print(json.dumps({name: {k: v for k, v in result.items() if k not in ("single_measurements_ms", "batch_seconds")}}), flush=True)

    with threadpool_limits(limits=2):
        for arm in MODEL_PINS:
            path = handoff / "Sub-Segment" / arm
            started = time.perf_counter()
            bundle = joblib.load(path / "model.joblib")
            require(bundle["target"] == "Sub-Segment", "Wrong model target")
            require(bundle["label_mapping"] == manifest["label_mapping"], "Wrong class order")
            load_seconds = time.perf_counter() - started
            reference = pd.read_csv(path / "predictions.csv", dtype={"record_id": str}).set_index("record_id").loc[ids]
            measure("tfidf_" + arm, lambda rows: predict_classical(bundle, rows), load_seconds, reference, 1e-12)
            del bundle
            gc.collect()

        started = time.perf_counter()
        config = ModelConfig.from_dict(read_json(model_dir / "model_config.json"))
        state = torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        require(state["label_names"] == labels or tuple(state["label_names"]) == tuple(labels), "Wrong checkpoint vocabulary")
        encoder_config = read_json(model_dir / "encoder_config.json")
        model_type = encoder_config.pop("model_type")
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **encoder_config))
        model = _make_classifier(torch, encoder, len(labels), config.dropout)
        model.load_state_dict(state["state_dict"], strict=True)
        require(_state_sha256(model.state_dict()) == manifest["checkpoint"]["state_sha256"], "Wrong state hash")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
        load_seconds = time.perf_counter() - started
        del state

        def predict_minilm(rows):
            texts = build_category_texts(rows, config, category_character_limit=512)
            batch = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            with torch.inference_mode():
                probabilities = torch.softmax(model(**batch).float(), dim=1).numpy()
            frame = pd.DataFrame({"record_id": rows.record_id.tolist(),
                                  "predicted_label": np.asarray(labels)[probabilities.argmax(axis=1)],
                                  "confidence": probabilities.max(axis=1)})
            return pd.concat([frame, pd.DataFrame(probabilities, columns=probability_columns)], axis=1)

        measure("minilm", predict_minilm, load_seconds, expected, 1e-5)

    for path, expected_hash in tracked.items():
        require(digest(path) == expected_hash, "Consumed artifact changed")
    output["input_files_rehashed_unchanged"] = len(tracked)
    output["script_sha256"] = digest(__file__)
    # Only aggregate statistics and timing samples; no product text or record IDs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
