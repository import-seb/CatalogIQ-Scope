"""Consume reviewed target assignments without executing a handoff's scripts."""
import json
from pathlib import Path

import pandas as pd

from .classical_data import load_target_development, target_profile
from .features import sha256
from .segment_baseline_data import MISSING_LABELS
from .segment_transformer import _hash_items
from .splitting import check_assignments


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sealed_files(root, hashes):
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.name in {"test.csv", "training_candidate.csv"}:
            raise ValueError("Invalid sealed artifact path")
        if sha256(path) != expected:
            raise ValueError(f"Changed sealed artifact: {name}")


def load_approved_targets(source, package, targets, *, manifest_sha256, protocol_dir=None):
    """Apply exact approved assignments after checking the source and both seals.

    A different source completion is a hard failure, even when CSV bytes match.
    Source reconciliation is separate reviewed work, never an implicit fallback.
    """
    source, package = Path(source).resolve(), Path(package).resolve()
    manifest_path = package / "handoff_manifest.json"
    if not manifest_sha256 or sha256(manifest_path) != manifest_sha256:
        raise ValueError("A matching approved handoff manifest SHA-256 is required")
    handoff = _json(manifest_path)
    if handoff["format"] != "catalogiq-approved-classical-split-handoff-v1":
        raise ValueError("Unsupported allocation handoff format")
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= {"Brand", "Sub-Segment"}:
        raise ValueError("An allocation handoff supports distinct Brand/Sub-Segment targets only")
    _sealed_files(package, handoff["artifact_sha256"])
    if sha256(source / "completion.json") != handoff["source"]["completion_sha256"]:
        raise ValueError("Different pinned V4 source completion: obtain original sidecars or reconcile through review before training")
    views, audit = load_target_development(source, targets, protocol_dir=protocol_dir)
    protected = pd.read_csv(source / "protected_final_test.csv", dtype=str, keep_default_na=False,
                            usecols=["record_id", "group_id", "split"])
    canonical = lambda f: f[["record_id", "group_id", "split"]].sort_values("record_id").reset_index(drop=True)
    if len(protected) != handoff["source"]["protected_test_rows"] or protected.group_id.nunique() != handoff["source"]["protected_test_groups"]:
        raise ValueError("Protected population differs from handoff")
    for target in targets:
        expected = handoff["targets"][target]
        _sealed_files(source, expected["source_consumed_artifact_sha256"])
        directory = (package / expected["directory"]).resolve()
        if not directory.is_relative_to(package):
            raise ValueError("Invalid allocation directory")
        if sha256(directory / "completion.json") != expected["approved_completion_sha256"]:
            raise ValueError("Allocation approval pin differs")
        seal = _json(directory / "completion.json")
        _sealed_files(directory, seal["artifact_sha256"])
        preparation = _json(directory / "protocol.json")["preparation"]
        if (preparation["target_column"] != target
                or preparation["completion_sha256"] != handoff["source"]["completion_sha256"]
                or preparation["consumed_artifact_sha256"] != expected["source_consumed_artifact_sha256"]):
            raise ValueError("Allocation protocol has a different target or source")
        original = pd.concat([pd.read_csv(source / f"{r}.csv", dtype=str, keep_default_na=False,
            usecols=["record_id", "group_id", "split", target]) for r in ("train", "validation")], ignore_index=True)
        assignments = pd.read_csv(directory / "assignments.csv", dtype=str, keep_default_na=False)
        metadata = pd.read_csv(directory / "development_assignments.csv", dtype=str, keep_default_na=False)
        reserved = pd.read_csv(directory / "protected_final_test.csv", dtype=str, keep_default_na=False,
                               usecols=["record_id", "group_id", "split"])
        check_assignments(assignments, [*original.record_id, *protected.record_id])
        if (not canonical(reserved).equals(canonical(protected))
                or not canonical(assignments.loc[assignments.split.eq("test")]).equals(canonical(protected))):
            raise ValueError("Allocation changed protected test identities or groups")
        if (not metadata.record_id.is_unique or set(metadata.record_id) != set(original.record_id)
                or not metadata.split.isin(["train", "validation"]).all()):
            raise ValueError("Allocation changed development population")
        aligned = original.set_index("record_id").loc[metadata.record_id]
        for column, source_column in (("group_id", "group_id"), ("original_split", "split"), (target, target)):
            if metadata[column].tolist() != aligned[source_column].tolist():
                raise ValueError(f"Allocation changed original {column}")
        missing = metadata[target].str.strip().str.casefold().isin(MISSING_LABELS)
        if metadata.is_labeled.tolist() != (~missing).map({True: "True", False: "False"}).tolist():
            raise ValueError("Allocation changed label eligibility")
        assigned = assignments.set_index("record_id").loc[metadata.record_id]
        if (assigned.group_id.tolist() != metadata.group_id.tolist()
                or assigned.split.tolist() != metadata.split.tolist()):
            raise ValueError("Allocation exports disagree")
        view = views[target].copy()
        view["split"] = view.record_id.map(metadata.set_index("record_id").split)
        check_assignments(view, view.record_id)
        profile = target_profile(view, target,
            {r: int((missing & metadata.split.eq(r)).sum()) for r in ("train", "validation")})
        vocabulary = pd.read_csv(directory / "class_coverage.csv", dtype=str, keep_default_na=False).label.tolist()
        if (vocabulary != expected["label_names"] or vocabulary != sorted(view.loc[view.split.eq("train"), target].unique())
                or len(original) != expected["development_rows_including_missing"]
                or profile["training_rows"] != expected["labeled_training_rows"]
                or profile["validation_rows"] != expected["labeled_validation_rows"]
                or profile["training_record_ids_sha256"] != expected["ordered_training_record_ids_sha256"]
                or profile["validation_record_ids_sha256"] != expected["validation_record_ids_sha256"]):
            raise ValueError("Prepared target differs from approved comparison population")
        profile.update(allocation_id=expected["directory"], allocation_identity={
            "handoff_manifest_sha256": manifest_sha256,
            "completion_sha256": expected["approved_completion_sha256"],
            "assignments_sha256": sha256(directory / "assignments.csv")})
        _sealed_files(source, expected["source_consumed_artifact_sha256"])
        _sealed_files(directory, seal["artifact_sha256"])
        views[target], audit["profiles"][target] = view, profile
    _sealed_files(package, handoff["artifact_sha256"])
    if sha256(source / "completion.json") != handoff["source"]["completion_sha256"]:
        raise ValueError("Source completion changed during preparation")
    return views, audit
