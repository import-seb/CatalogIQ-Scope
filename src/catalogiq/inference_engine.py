"""Framework-independent, reviewer-assisted inference over frozen selections."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Mapping, Sequence

import pandas as pd

from .confidence_calibration import review_decision, temperature_scale, validate_probabilities
from .relationship_conflicts import load_registry
from .target_minilm_preparation import INPUT_FIELDS


TARGETS = ("Segment", "Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")
ENGINE_VERSION = "catalogiq-inference-v1-20261111"


@dataclass(frozen=True)
class ProductRecord:
    record_id: str
    features: Mapping[str, str | None]
    quarantined: bool = False


@dataclass
class TargetPrediction:
    target: str
    status: str
    selection_status: str
    predicted_label: str | None = None
    confidence: float | None = None
    probabilities: tuple[float, ...] | None = None
    labels: tuple[str, ...] = ()
    calibrated_probabilities: tuple[float, ...] | None = None
    lookup_status: str | None = None
    review_required: bool = True
    review_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    provenance: dict = field(default_factory=dict)


@dataclass
class ProductPrediction:
    record_id: str
    predictions: list[TargetPrediction]
    review_required: bool = True


def validate_records(records):
    """Validate correlation keys globally; report feature errors per row."""
    if not records:
        raise ValueError("At least one product is required")
    ids = []
    for product in records:
        if not isinstance(product, ProductRecord):
            raise ValueError("Products must be ProductRecord objects")
        if not isinstance(product.record_id, str) or not product.record_id.strip():
            raise ValueError("record_id must be a nonblank string")
        ids.append(product.record_id)
    if len(ids) != len(set(ids)):
        raise ValueError("record_id must be unique within a request")


def input_error(product):
    if not isinstance(product.features, Mapping):
        return "features must be a mapping"
    missing = [name for name in INPUT_FIELDS if name not in product.features]
    if missing:
        return "Missing required feature fields: " + ", ".join(missing)
    if any(value is not None and not isinstance(value, str)
           for name, value in product.features.items() if name in INPUT_FIELDS):
        return "Feature values must be strings or null"
    if type(product.quarantined) is not bool:
        return "quarantined must be boolean"
    return None


class InferenceEngine:
    """Reuse loaded routes across requests; all acceptance remains disabled.

    Supply a loader with ``load(target)`` to embed another artifact store or to
    test offline. The default FrozenModelLoader verifies the pinned bundles.
    """

    def __init__(self, processed_dir: str | Path, *, registry_path=None, loader=None):
        from .inference_models import FrozenModelLoader

        self.loader = loader if loader is not None else FrozenModelLoader(processed_dir)
        registry = load_registry(registry_path)
        self.registry_version = registry["registry_version"]
        # enforceable_relations is a warn-mode research helper, not this filter.
        self.approved_relations = deepcopy([r for r in registry["relations"]
                                           if r.get("status") == "approved"])
        self._models = {}
        self._lock = RLock()

    def predict(self, records: Sequence[ProductRecord]) -> list[ProductPrediction]:
        records = list(records)
        validate_records(records)
        with self._lock:
            return self._predict(records)

    def _predict(self, records):
        from .inference_models import ArtifactMismatch

        errors = [input_error(product) for product in records]
        valid = [i for i, error in enumerate(errors) if error is None]
        frame = pd.DataFrame([{name: records[i].features[name] for name in INPUT_FIELDS}
                              for i in valid], columns=INPUT_FIELDS)
        outputs = [[] for _ in records]
        for target in TARGETS:
            selection = "provisional" if target == "Sub-Segment" else "selected"
            provenance = self.loader.describe(target) if hasattr(self.loader, "describe") else {}
            results = [TargetPrediction(target, "failed", selection,
                        review_reasons=["required_input_invalid"], error=error,
                        provenance=deepcopy(provenance))
                       for error in errors]
            if valid:
                try:
                    if target not in self._models:
                        self._models[target] = self.loader.load(target)
                    route = self._models[target]
                    matrix, cohorts = route.predict(frame)
                    matrix = validate_probabilities(matrix)
                    if matrix.shape != (len(valid), len(route.labels)):
                        raise ValueError("Prediction shape differs from rows or saved vocabulary")
                    if len(set(route.labels)) != len(route.labels) or any(
                            not isinstance(label, str) or not label.strip() for label in route.labels):
                        raise ValueError("Invalid model vocabulary")
                    if len(cohorts) != len(valid) or any(
                            value not in (None, "seen", "missing", "unseen") for value in cohorts):
                        raise ValueError("Lookup cohorts differ from prediction rows")
                    calibrated = temperature_scale(matrix, route.temperature)
                    for position, index in enumerate(valid):
                        probabilities = matrix[position]
                        results[index] = TargetPrediction(
                            target, "suggestion", selection,
                            predicted_label=route.labels[int(probabilities.argmax())],
                            confidence=float(probabilities.max()),
                            probabilities=tuple(map(float, probabilities)), labels=tuple(route.labels),
                            calibrated_probabilities=tuple(map(float, calibrated[position])),
                            lookup_status=cohorts[position], provenance=deepcopy(route.provenance))
                except Exception as exc:
                    # Keep other target suggestions and never leak a partial success.
                    code = ("artifact_mismatch" if isinstance(exc, ArtifactMismatch) else
                            "model_unavailable" if isinstance(exc, (OSError, ImportError)) else
                            "prediction_failed")
                    status = "unavailable" if code == "model_unavailable" else "failed"
                    for index in valid:
                        results[index] = TargetPrediction(target, status, selection,
                            review_reasons=[code], error=f"{type(exc).__name__}: {exc}",
                            provenance=deepcopy(provenance))
            for index, result in enumerate(results):
                outputs[index].append(result)

        for product, predictions in zip(records, outputs):
            classifications = {p.target: p.predicted_label for p in predictions}
            for prediction in predictions:
                # Call the existing primitive even for failures to retain missing
                # text, quarantine, policy and approved-relationship reasons.
                relations = [r for r in self.approved_relations
                             if prediction.target in (r.get("child_target"), r.get("parent_target"))]
                supplied = product.features if isinstance(product.features, Mapping) else {}
                features = {name: supplied.get(name) if isinstance(supplied.get(name), str) else None
                            for name in INPUT_FIELDS}
                decision = review_decision(prediction.probabilities or (),
                    prediction.predicted_label, prediction.labels, None, features,
                    enabled=False, provisional=prediction.selection_status == "provisional",
                    approved_relations=relations, classifications=classifications,
                    lookup_status=prediction.lookup_status, quarantined=product.quarantined is True)
                prediction.review_reasons = sorted(set(prediction.review_reasons + decision["review_reasons"]))
            predictions.append(TargetPrediction("Platform", "unavailable", "deferred",
                review_reasons=["target_unavailable"]))
        return [ProductPrediction(product.record_id, predictions)
                for product, predictions in zip(records, outputs)]
