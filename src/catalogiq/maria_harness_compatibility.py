"""Pinned Maria loader and reference functions for validation-only checks.

Notebook setup, training, discovery and test-label reads never execute here.
Saved-findings replay uses reference metrics but never calls temperature fitting.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


HARNESS_COMMIT = "05e52e2e0203dee6d58dfc2d0e72d513eed9ad95"
HARNESS_NOTEBOOK_SHA256 = "8ab3b07a8f15bdcdeb565e4646ec606b4d4f2e4a535e6f991aeea492b60b27fe"
HARNESS_LOADER_SHA256 = "641afd3af422d3d4de54ed714c51b39665241659668dfcd779c1b1be848a5bc2"
FUNCTIONS = ("hash_items", "_labels_from", "load_predictions")

REVIEW_HARNESS_COMMIT = "a41608e83dd3f1420ce5cb63a28b022254f879d4"
REVIEW_NOTEBOOK_BLOB = "5670c9d55777f8f9b313b3f1e7835145fd5a6885"
REVIEW_REFERENCE = Path(__file__).parent / "resources" / "maria_review_a41608e.py"
REVIEW_FUNCTION_HASHES = {
    "queue_metrics": "d03500e93f64fd51bf447084b9efc7b5e22401410d064c8b81f01edde7f395b2",
    "coverage_risk": "707c663c69424539c088158b5636d6becf0e2fda559d7310e49938e93ce4fbaf",
    "class_set": "45abdbf95fa993bbf889076a07cf212e8b2d7d5813df25ba153b9c402138bc16",
    "macro_f1_on": "c66f0b456ac0cbb79ff5e19d72f216ee1f5906b5a31d20a83dedd903c41470b4",
    "ece_and_reliability": "7defe6cca4052a8b9239006b23dfe8039939bfd34027fe5514c9749b47dcd81e",
    "multiclass_brier": "afc46750e837276d3d1b494ea157e2267fa708e274a67dd628fdd51ab0b85c21",
    "temperature_scale": "439a7e3c7074eaff2bc787ac15c8bd825e40ef1af250b59942ffc57f53edfe89",
}


def review_namespace(path=None):
    """Load only fingerprinted reference definitions, available in wheels too.

    No top-level statements execute. Replay uses saved temperatures; the legacy
    temperature function remains available only for existing compatibility tests.
    """
    path = Path(path) if path is not None else REVIEW_REFERENCE
    text = path.read_text(encoding="utf8")
    definitions = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            if node.name not in REVIEW_FUNCTION_HASHES or node.name in definitions:
                raise ValueError("Unexpected or duplicate Maria review definition")
            source = ast.get_source_segment(text, node)
            if hashlib.sha256(source.encode()).hexdigest() != REVIEW_FUNCTION_HASHES[node.name]:
                raise ValueError(f"Unreviewed Maria definition: {node.name}")
            definitions[node.name] = source
    if set(definitions) != set(REVIEW_FUNCTION_HASHES):
        raise ValueError("Missing Maria review definitions")
    namespace = {"np": np, "pd": pd, "f1_score": f1_score}
    exec(compile("\n\n".join(definitions.values()), str(path), "exec"), namespace)
    return namespace


def loader_definitions(path):
    """Read the pinned notebook, or the identical definitions fixture in tests."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".ipynb":
        if hashlib.sha256(raw).hexdigest() != HARNESS_NOTEBOOK_SHA256:
            raise ValueError("Harness notebook differs from the reviewed current commit")
        blocks = ["".join(cell.get("source", [])) for cell in json.loads(raw)["cells"] if cell["cell_type"] == "code"]
    else:
        blocks = [raw.decode("utf8").replace("\r\n", "\n")]
    definitions, manifests = {}, None
    for block in blocks:
        for node in ast.parse(block).body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                if node.name in definitions:
                    raise ValueError("Duplicate harness loader definition")
                definitions[node.name] = ast.get_source_segment(block, node)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MANIFESTS" for target in node.targets):
                manifests = ast.literal_eval(node.value)
    source = "\n\n".join(definitions[name] for name in FUNCTIONS)
    if hashlib.sha256(source.encode()).hexdigest() != HARNESS_LOADER_SHA256:
        raise ValueError("Harness loader differs from the reviewed current commit")
    if manifests != ("training_manifest.json", "protocol.json", "model_config.json", "label_mapping.json", "manifest.json"):
        raise ValueError("Harness manifest discovery contract changed")
    return source, manifests


def validation_namespace(validation, protected_ids, whole_validation_ids):
    """Bind the real target-specific population, never the default Segment split."""
    if validation.record_id.duplicated().any() or set(validation.record_id) & set(protected_ids):
        raise ValueError("Invalid or protected validation population")
    from .segment_transformer import _hash_items
    return {"hashlib": hashlib, "json": json, "re": re, "Path": Path, "np": np, "pd": pd,
        "VA_IDS": set(validation.record_id), "TEST_IDS": set(protected_ids),
        "GROUP_OF": dict(zip(validation.record_id, validation.group_id)),
        "VA_ID_HASHES": {"labelled validation rows": _hash_items(sorted(validation.record_id)),
                         "whole validation partition": _hash_items(sorted(whole_validation_ids))}}


def load_with_maria_harness(directory, notebook, validation, protected_ids, whole_validation_ids):
    """Exercise the actual loader; reject drops, partial rows or changed truth."""
    source, manifests = loader_definitions(notebook)
    namespace = validation_namespace(validation, protected_ids, whole_validation_ids)
    namespace["MANIFESTS"] = manifests
    exec(compile(source, str(notebook), "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run = namespace["load_predictions"](directory)
    frame = run["frame"]
    expected = validation.set_index("record_id").loc[frame.record_id]
    if (len(frame) != len(validation) or set(frame.record_id) != set(validation.record_id)
            or frame.true_label.tolist() != expected.true_label.tolist()
            or frame.group_id.tolist() != expected.group_id.tolist() or not run["split_checked"]):
        raise ValueError("Harness did not consume the entire approved validation population")
    score = {"accuracy": float(accuracy_score(frame.true_label, frame.predicted_label)),
        "macro_f1_observed_labels": float(f1_score(frame.true_label, frame.predicted_label, average="macro", zero_division=0)),
        "macro_f1_full_vocabulary": float(f1_score(frame.true_label, frame.predicted_label, labels=run["labels"], average="macro", zero_division=0))}
    report = {"status": "compatible", "harness_commit": HARNESS_COMMIT,
        "harness_notebook_sha256": HARNESS_NOTEBOOK_SHA256, "loader_definitions_sha256": HARNESS_LOADER_SHA256,
        "rows_consumed": len(frame), "classes": len(run["labels"]), "split_checked": run["split_checked"],
        "protected_test_records_scored": 0, "rows_dropped": 0, "labels": run["labels"], "metrics": score,
        "loader_output": output.getvalue(),
        "scope": "Unchanged loader and headline scoring; guarded target-specific context. Notebook setup/training/calibration cells were not executed."}
    return run, report
