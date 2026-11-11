"""Offline loading and forward paths for the five frozen inference routes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import sha256
from .segment_category_ablation import build_category_texts
from .segment_transformer import ModelConfig, _make_classifier, _state_sha256


PINS_PATH = Path(__file__).parent / "resources" / "inference" / "frozen_routes.json"


class ArtifactMismatch(ValueError):
    """A serving artifact disagrees with the reviewed frozen identity."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ArtifactMismatch(message)


def ordered_labels(manifest):
    mapping = manifest["label_mapping"]
    columns = [f"probability_{i}" for i in range(len(mapping))]
    require(set(mapping) == set(columns), "Noncontiguous probability mapping")
    labels = tuple(mapping[column] for column in columns)
    require(len(labels) >= 2 and len(set(labels)) == len(labels)
            and all(isinstance(label, str) and label.strip() for label in labels), "Invalid vocabulary")
    require(list(labels) == manifest["label_names"], "Vocabulary order differs")
    return labels


class FrozenModelLoader:
    def __init__(self, processed_dir, *, device="cpu"):
        self.processed_dir = Path(processed_dir)
        self.device = device
        self.pins = read_json(PINS_PATH)

    def describe(self, target):
        spec = self.pins["routes"][target]
        return {
            "model_id": spec["model_id"], "artifact_sha256": spec["artifact_sha256"],
            "configuration_version": self.pins["version"],
            "manifest_sha256": spec["files_sha256"][spec["manifest"]],
            "policy_version": "confidence_review_v3_20261103",
            "policy_sha256": spec["files_sha256"][spec["policy"]],
            "policy_approved": False, "operational_threshold": None,
            "probabilities_calibrated": False,
            "calibration_id": spec["files_sha256"][spec["temperature"]],
            "calibration_status": "research_only",
        }

    def load(self, target):
        spec = self.pins["routes"][target]
        # Only serving members are opened. Never traverse/hash a data directory
        # or consume the validation/protected CSVs named by historical manifests.
        for relative, digest in spec["files_sha256"].items():
            require(sha256(self.processed_dir / relative) == digest,
                    f"Frozen serving member checksum differs: {relative}")
        manifest = read_json(self.processed_dir / spec["manifest"])
        require(manifest["target"] == target and manifest["model_identity"] == spec["model_id"],
                "Wrong target or model identity")
        labels = ordered_labels(manifest)
        policy = read_json(self.processed_dir / spec["policy"])
        temperature = read_json(self.processed_dir / spec["temperature"])["temperature"]
        require(policy["target"] == target and policy["enabled"] is False
                and policy["operational_threshold"] is None, "Operational policy must remain disabled")
        require(policy["source_manifest_sha256"] == sha256(self.processed_dir / spec["manifest"]),
                "Calibration policy/model binding differs")
        require(isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
                and np.isfinite(temperature) and temperature > 0, "Invalid saved temperature")
        provenance = {**self.describe(target), "temperature": temperature}
        directory = self.processed_dir / spec["directory"]
        if target == "Brand":
            model = read_json(directory / "model.json")
            require(model["target"] == target and tuple(model["label_names"]) == labels,
                    "Brand vocabulary differs")
            return LookupRoute(model, labels, temperature, provenance)
        config = ModelConfig.from_dict(read_json(directory / "model_config.json"))
        require(config.to_dict() == ModelConfig.from_dict(manifest["config"]).to_dict(),
                "Model configuration differs from manifest")
        require(config.sequence_length == 256 and list(manifest["input_fields"]) ==
                ["ProductCategory", "ProductName", "ProductBrand", "ProductDescription", "ProductContents"],
                "Selected five-field representation required")
        return MiniLMRoute(directory, config, labels, manifest["checkpoint"]["state_sha256"],
                           temperature, provenance, device=self.device)


class LookupRoute:
    def __init__(self, model, labels, temperature, provenance):
        self.model, self.labels = model, labels
        self.temperature, self.provenance = temperature, provenance

    def predict(self, frame):
        from .brand_probabilistic_lookup import predict_frequency_lookup

        matrix, flags, _ = predict_frequency_lookup(self.model, frame.ProductBrand)
        cohorts = ["missing" if missing else "seen" if seen else "unseen"
                   for missing, seen in zip(flags["missing_values"], flags["seen_feature_values"])]
        return matrix, cohorts


class MiniLMRoute:
    def __init__(self, directory, config, labels, state_sha256, temperature, provenance,
                 *, device="cpu", tokenizer=None, encoder_factory=None):
        import torch

        self.config, self.labels = config, labels
        self.temperature, self.provenance = temperature, provenance
        self.device = torch.device(device)
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        require(tuple(state["label_names"]) == labels
                and ModelConfig.from_dict(state["model_config"]).to_dict() == config.to_dict()
                and _state_sha256(state["state_dict"]) == state_sha256,
                "Checkpoint state, configuration or labels differ")
        if encoder_factory is None:
            from transformers import AutoConfig, AutoModel

            values = read_json(directory / "encoder_config.json")
            model_type = values.pop("model_type")
            encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
        else:
            encoder = encoder_factory()
        self.model = _make_classifier(torch, encoder, len(labels), config.dropout)
        self.model.load_state_dict(state["state_dict"], strict=True)
        require(_state_sha256(self.model.state_dict()) == state_sha256, "Loaded state differs")
        self.model.to(self.device).eval()
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        self.tokenizer = tokenizer

    def predict(self, frame):
        import torch

        texts = build_category_texts(frame, self.config, category_character_limit=512)
        matrices = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.config.evaluation_batch_size):
                batch = self.tokenizer(texts[start:start + self.config.evaluation_batch_size],
                    truncation=True, max_length=256, padding=True, return_tensors="pt")
                batch = {name: value.to(self.device) for name, value in batch.items()}
                matrices.append(torch.softmax(self.model(**batch).float(), dim=1).cpu().numpy())
        return np.concatenate(matrices), [None] * len(frame)
