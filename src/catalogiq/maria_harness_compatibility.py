"""Run Maria's unchanged prediction loader with approved validation-only context.

Only the three reviewed loader definitions are extracted. Notebook setup,
training, discovery, calibration and test-label reads are never executed here.
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
