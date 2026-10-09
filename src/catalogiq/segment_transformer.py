"""Fixed-epoch, label-only Segment transformer baseline.

The caller supplies a development corpus and disjoint training/validation indices.
There is deliberately no final-test API and validation never chooses a checkpoint.
PyTorch and Transformers are imported only when tokenization or training needs them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


MODEL_FIELDS = ("ProductName", "ProductBrand", "ProductDescription", "ProductContents")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    sequence_length: int = 128
    epochs: int = 3
    learning_rate: float = 3e-5
    batch_size: int = 32
    evaluation_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    seed: int = 42
    device: str = "auto"
    threads: int = 4
    precision: str = "fp32"
    dropout: float = 0.10
    field_character_limits: tuple[int, int, int, int] = (1024, 512, 4000, 2000)
    local_files_only: bool = True
    log_every_steps: int = 50

    def __post_init__(self):
        for name in ("sequence_length", "epochs", "batch_size", "evaluation_batch_size",
                     "threads", "log_every_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 4:
            raise ValueError("sequence_length must leave room for text and special tokens")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        for name in ("warmup_fraction", "dropout"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16")
        if not self.model_name or not self.revision:
            raise ValueError("model_name and pinned revision must be nonempty")
        if (len(self.field_character_limits) != len(MODEL_FIELDS)
                or any(type(limit) is not int or limit < 1 for limit in self.field_character_limits)):
            raise ValueError("field_character_limits requires four positive integer limits")
        if type(self.local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "field_character_limits" in values:
            values["field_character_limits"] = tuple(values["field_character_limits"])
        return cls(**values)


def _plain_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "null", "none", "nan"}:
        return ""
    # HTML is removed only from this model view; the original frame is untouched.
    text = html.unescape(text)
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_product_texts(frame: pd.DataFrame, config: ModelConfig | None = None) -> list[str]:
    """Build name-first texts from the four fixed allowlisted product fields only."""
    config = config or ModelConfig()
    columns = [frame[name] if name in frame else pd.Series("", index=frame.index)
               for name in MODEL_FIELDS]
    result = []
    for values in zip(*columns):
        parts = []
        for name, value, limit in zip(MODEL_FIELDS, values, config.field_character_limits):
            value = _plain_text(value)[:limit].rstrip()
            if value:
                parts.append(f"[{name}] {value}")
        result.append(" ".join(parts))
    return result


@dataclass(frozen=True)
class TokenizedCorpus:
    record_ids: tuple[str, ...]
    input_ids: tuple[np.ndarray, ...]
    attention_mask: tuple[np.ndarray, ...]
    label_ids: np.ndarray
    label_names: tuple[str, ...]
    token_type_ids: tuple[np.ndarray, ...] | None = None
    pad_token_id: int = 0
    text_sha256: str = ""
    token_sha256: str = ""
    sequence_length: int = 128
    tokenizer: Any = field(default=None, compare=False, repr=False)

    def __len__(self):
        return len(self.record_ids)


def _hash_items(items: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        payload = item.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def tokenize_corpus(frame: pd.DataFrame, label_names: Sequence[str],
                    config: ModelConfig | None = None, *, tokenizer=None) -> TokenizedCorpus:
    """Tokenize a development-only frame once for reuse across controlled runs.

    Labels are encoded separately and never inserted into product text. Input IDs
    preserve row order; no vocabulary, normalizer, or parameters are fit on labels.
    """
    config = config or ModelConfig()
    names = tuple(str(name) for name in label_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("label_names must contain at least two unique classes")
    if frame.empty or "record_id" not in frame or "Segment" not in frame:
        raise ValueError("development frame requires record_id, Segment, and at least one row")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("record_id values must be nonempty")
    ids = tuple(frame["record_id"].astype(str))
    if len(set(ids)) != len(ids):
        raise ValueError("record_id values must be unique")
    if frame["Segment"].isna().any():
        raise ValueError("Segment labels must be present in the development frame")
    encoded_labels = frame["Segment"].astype(str).map({name: i for i, name in enumerate(names)})
    if encoded_labels.isna().any():
        raise ValueError("Segment contains a label outside label_names")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision,
                                                  local_files_only=config.local_files_only)
    texts = build_product_texts(frame, config)
    encoded = tokenizer(texts, truncation=True, max_length=config.sequence_length,
                        padding=False, return_attention_mask=True)
    inputs = tuple(np.asarray(row, dtype=np.int32) for row in encoded["input_ids"])
    masks = tuple(np.asarray(row, dtype=np.int32) for row in encoded["attention_mask"])
    token_types = (tuple(np.asarray(row, dtype=np.int32) for row in encoded["token_type_ids"])
                   if "token_type_ids" in encoded else None)
    if (len(inputs) != len(ids) or len(masks) != len(ids)
            or any(len(tokens) != len(mask) or not 0 < len(tokens) <= config.sequence_length
                   for tokens, mask in zip(inputs, masks))):
        raise ValueError("tokenizer returned invalid sequence lengths")
    digest = hashlib.sha256()
    digest.update(_hash_items(ids).encode("ascii"))
    for rows in (inputs, masks, token_types or ()):
        for row in rows:
            digest.update(len(row).to_bytes(8, "big"))
            digest.update(row.astype("<i4", copy=False).tobytes())
    return TokenizedCorpus(ids, inputs, masks, encoded_labels.to_numpy(dtype=np.int64), names,
                           token_types, int(tokenizer.pad_token_id or 0), _hash_items(texts),
                           digest.hexdigest(), config.sequence_length, tokenizer)


def _initialize_torch(config: ModelConfig):
    # Must precede CUDA initialization. Necessary for deterministic CUDA matmuls.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if config.precision == "fp16" and not device.startswith("cuda"):
        raise ValueError("fp16 AMP requires a CUDA device")
    return torch, torch.device(device)


def _make_classifier(torch, encoder, classes: int, dropout: float):
    class SegmentClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
            hidden = getattr(encoder.config, "hidden_size", None)
            if hidden is None:
                raise ValueError("encoder configuration must expose hidden_size")
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden, classes)

        def forward(self, **batch):
            outputs = self.encoder(**batch)
            hidden = outputs.last_hidden_state
            weights = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            return self.classifier(self.dropout(pooled))

    return SegmentClassifier()


def _state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_indices(corpus: TokenizedCorpus, train_indices, validation_indices):
    result = []
    for name, values in (("train", train_indices), ("validation", validation_indices)):
        indices = np.asarray(values)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
            raise ValueError(f"{name} indices must be a nonempty one-dimensional integer sequence")
        indices = indices.astype(np.int64, copy=False)
        if indices.min() < 0 or indices.max() >= len(corpus):
            raise ValueError(f"{name} indices are outside the development corpus")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} indices contain repeats")
        result.append(indices)
    if np.intersect1d(*result).size:
        raise ValueError("training and validation indices must be disjoint")
    if len(set(corpus.record_ids)) != len(corpus):
        raise ValueError("development corpus contains duplicate record IDs")
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray, names: Sequence[str]):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=np.arange(len(names)),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "per_class": classification_report(labels, predicted, labels=np.arange(len(names)),
                                             target_names=list(names), zero_division=0, output_dict=True),
    }


def train_transformer(corpus: TokenizedCorpus, train_indices, validation_indices,
                      label_names: Sequence[str], config: ModelConfig, output_dir,
                      progress: Callable[[str], Any] | None = print, *,
                      encoder_factory=None) -> tuple[pd.DataFrame, dict]:
    """Fine-tune every encoder parameter, then score only validation records.

    Fixed epochs, configuration, and class order control the comparison; validation
    metrics are observational and never select a checkpoint or change optimization.
    ``encoder_factory`` is an injection point for offline tiny-model tests only.
    """
    names = tuple(str(name) for name in label_names)
    if names != corpus.label_names:
        raise ValueError("label_names must exactly match the tokenized class ordering")
    if corpus.sequence_length != config.sequence_length:
        raise ValueError("configuration differs from the corpus tokenization length")
    train_indices, validation_indices = _validate_indices(corpus, train_indices, validation_indices)
    if (len(corpus.label_ids) != len(corpus)
            or np.asarray(corpus.label_ids).dtype.kind not in "iu"
            or np.min(corpus.label_ids) < 0 or np.max(corpus.label_ids) >= len(names)):
        raise ValueError("development corpus contains invalid label IDs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training_manifest.json").exists():
        raise FileExistsError("output directory already contains a completed training run")
    started = time.perf_counter()
    torch, device = _initialize_torch(config)
    if encoder_factory is None:
        import transformers
        encoder = transformers.AutoModel.from_pretrained(
            config.model_name, revision=config.revision, local_files_only=config.local_files_only,
            use_safetensors=True)
        transformers_version = transformers.__version__
    else:
        encoder = encoder_factory()
        transformers_version = "injected-test-encoder"
    model = _make_classifier(torch, encoder, len(names), config.dropout)
    initial_sha = _state_sha256(model.state_dict())
    manifest = {
        "config": config.to_dict(), "label_names": list(names),
        "input_fields": list(MODEL_FIELDS), "target": "Segment",
        "checkpoint_selection": "fixed final epoch; validation is observational only",
        "initial_state_sha256": initial_sha,
        "train_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in train_indices]),
        "validation_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in validation_indices]),
        "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
        "training_rows": len(train_indices), "validation_rows": len(validation_indices),
        "train_class_counts": np.bincount(corpus.label_ids[train_indices], minlength=len(names)).tolist(),
        "validation_class_counts": np.bincount(corpus.label_ids[validation_indices], minlength=len(names)).tolist(),
        "device": str(device), "torch_version": torch.__version__,
        "transformers_version": transformers_version, "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "reproducibility_scope": "same software, hardware, inputs, ordering, and configuration",
        "encoder_trainable_parameters": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if device.type == "cuda":
        manifest["cuda_device_name"] = torch.cuda.get_device_name(device)
    (output_dir / "initialization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches_per_epoch = math.ceil(len(train_indices) / config.batch_size)
    total_steps = batches_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)

    def lr_scale(step):
        if warmup_steps and step < warmup_steps:
            return float(step) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    criterion = torch.nn.CrossEntropyLoss()

    def make_batch(indices):
        width = max(len(corpus.input_ids[i]) for i in indices)
        features = {}
        for name, rows, fill in (("input_ids", corpus.input_ids, corpus.pad_token_id),
                                 ("attention_mask", corpus.attention_mask, 0),
                                 ("token_type_ids", corpus.token_type_ids, 0)):
            if rows is None:
                continue
            array = np.full((len(indices), width), fill, dtype=np.int64)
            for j, index in enumerate(indices):
                array[j, :len(rows[index])] = rows[index]
            features[name] = torch.from_numpy(array).to(device)
        labels = torch.as_tensor(corpus.label_ids[indices], dtype=torch.long, device=device)
        return features, labels

    def validation_pass():
        model.eval()
        all_probabilities, loss_total = [], 0.0
        with torch.no_grad():
            for start in range(0, len(validation_indices), config.evaluation_batch_size):
                indices = validation_indices[start:start + config.evaluation_batch_size]
                features, labels = make_batch(indices)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(**features)
                    loss = criterion(logits, labels)
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_probabilities.append(probabilities)
                loss_total += float(loss) * len(indices)
        probabilities = np.concatenate(all_probabilities)
        result = _metrics(corpus.label_ids[validation_indices], probabilities, names)
        result["loss"] = loss_total / len(validation_indices)
        return probabilities, result

    history = []
    if progress:
        progress(f"Transformer initialized: {len(train_indices)} train, {len(validation_indices)} validation; "
                 f"{device}, {config.precision}; initial state {initial_sha}")
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        ordering = torch.randperm(len(train_indices), generator=shuffle_generator).numpy()
        shuffled_indices = train_indices[ordering]
        running_loss, seen = 0.0, 0
        for batch_step, start in enumerate(range(0, len(shuffled_indices), config.batch_size), 1):
            indices = shuffled_indices[start:start + config.batch_size]
            features, labels = make_batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(**features), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss; experiment stopped")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
            if progress and batch_step % config.log_every_steps == 0:
                progress(f"Epoch {epoch}/{config.epochs}, batch {batch_step}/{batches_per_epoch}, "
                         f"training loss {running_loss / seen:.4f}, "
                         f"elapsed {time.perf_counter() - epoch_started:.1f}s")
        probabilities, final_metrics = validation_pass()
        history.append({"epoch": epoch, "training_loss": running_loss / seen,
                        "validation_loss": final_metrics["loss"],
                        "validation_accuracy": final_metrics["accuracy"],
                        "validation_macro_f1": final_metrics["macro_f1"],
                        "validation_weighted_f1": final_metrics["weighted_f1"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": time.perf_counter() - epoch_started})
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if progress:
            progress(f"Epoch {epoch} complete: validation accuracy {final_metrics['accuracy']:.4f}, "
                     f"macro F1 {final_metrics['macro_f1']:.4f}, "
                     f"{history[-1]['epoch_seconds']:.1f}s")

    predicted = probabilities.argmax(axis=1)
    predictions = pd.DataFrame({
        "record_id": [corpus.record_ids[i] for i in validation_indices],
        "true_label": [names[i] for i in corpus.label_ids[validation_indices]],
        "predicted_label": [names[i] for i in predicted],
        "confidence": probabilities.max(axis=1),
    })
    for i in range(len(names)):
        predictions[f"probability_{i}"] = probabilities[:, i]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    manifest["final_state_sha256"] = _state_sha256(state)
    manifest["completed_epochs"] = config.epochs
    manifest["optimizer_steps"] = total_steps
    manifest["warmup_steps"] = warmup_steps
    manifest["final_validation"] = final_metrics
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["validation_max_length_rows"] = sum(
        len(corpus.input_ids[i]) == config.sequence_length for i in validation_indices)
    manifest["training_max_length_rows"] = sum(
        len(corpus.input_ids[i]) == config.sequence_length for i in train_indices)
    torch.save({"state_dict": state, "model_config": config.to_dict(), "label_names": list(names),
                "pooling": "attention-mask mean pooling", "dropout": config.dropout,
                "initial_state_sha256": initial_sha, "final_state_sha256": manifest["final_state_sha256"]},
               output_dir / "checkpoint.pt")
    if hasattr(encoder.config, "to_json_file"):
        encoder.config.to_json_file(output_dir / "encoder_config.json")
    if corpus.tokenizer is not None and hasattr(corpus.tokenizer, "save_pretrained"):
        corpus.tokenizer.save_pretrained(output_dir / "tokenizer")
    manifest["artifacts"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in output_dir.iterdir() if path.is_file()
                             and path.name != "training_manifest.json"}
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return predictions, manifest
