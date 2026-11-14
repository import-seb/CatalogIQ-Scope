"""Provisional JSON interface. Keep transport/schema choices out of inference."""
from dataclasses import asdict

from .inference_engine import ENGINE_VERSION, ProductRecord, validate_records
from .target_minilm_preparation import INPUT_FIELDS


SCHEMA_VERSION = "catalogiq-inference-request-v1"


def parse_request(request):
    """A single flat product is a batch of one; IDs are never regenerated."""
    if not isinstance(request, dict):
        raise ValueError("Request must be an object")
    if "products" in request:
        if set(request) - {"products", "schema_version"}:
            raise ValueError("Unknown batch request fields")
        if request.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        rows = request["products"]
    else:
        rows = [request]
    if not isinstance(rows, list):
        raise ValueError("products must be a nonempty array")
    products = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each product must be an object")
        unknown = set(row) - {"record_id", "quarantined", *INPUT_FIELDS}
        if unknown:
            raise ValueError("Unknown product fields: " + ", ".join(sorted(unknown)))
        products.append(ProductRecord(row.get("record_id"),
            {name: row[name] for name in INPUT_FIELDS if name in row}, row.get("quarantined", False)))
    validate_records(products)
    return products


def format_response(results, *, registry_version):
    products = []
    for product in results:
        targets = {}
        for prediction in product.predictions:
            value = asdict(prediction)
            value.pop("target")
            labels = value.pop("labels")
            value["label_mapping"] = {f"probability_{i}": label for i, label in enumerate(labels)}
            for key in ("probabilities", "calibrated_probabilities"):
                probabilities = value[key]
                value[key] = (None if probabilities is None else
                              {f"probability_{i}": number for i, number in enumerate(probabilities)})
            value["raw_confidence"] = value["confidence"]
            scaled = prediction.calibrated_probabilities
            value["calibrated_confidence"] = max(scaled) if scaled is not None else None
            targets[prediction.target] = value
        products.append({"record_id": product.record_id, "review_required": product.review_required,
                         "predictions": targets})
    return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
            "relationship_registry_version": registry_version, "products": products}


def infer_request(engine, request):
    """Return a JSON-serializable batch response or raise ValueError for bad envelopes."""
    return format_response(engine.predict(parse_request(request)), registry_version=engine.registry_version)
