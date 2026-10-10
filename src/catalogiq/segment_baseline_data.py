"""Join an explicitly selected assignment export to cleaned product records.

This checks membership and group isolation, not the final snapshot's exposure
history or seal. It neither generates splits nor substitutes for that protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .segment_transformer import MODEL_FIELDS
from .splitting import check_assignments, record_ids


MISSING_LABELS = {"", "null", "none", "nan"}


@dataclass(frozen=True)
class BaselineDevelopment:
    frame: pd.DataFrame
    audit: dict


def load_official_development(directory, *, include_category=False):
    """Reuse the team's sealed loader, then attach allowlisted development data.

    Only train.csv and validation.csv supply labels/features. The category arm
    is an explicit validation experiment; it never edits the split contract.
    """
    from .segment_frozen_development import load_frozen_development

    directory = Path(directory).resolve()
    guarded = load_frozen_development(directory)
    if guarded.audit["split_format"] != "catalogiq-portable-split-v1":
        raise ValueError("Use the authoritative portable split workflow")
    consumed = guarded.audit["consumed_artifact_sha256"]
    wanted = [*PROVENANCE, "record_id", "group_id", "split", "Segment", *MODEL_FIELDS]
    if include_category:
        wanted.append("ProductCategory")
    roles = []
    for role in ("train", "validation"):
        path = directory / f"{role}.csv"
        if sha256(path) != consumed[path.name]:
            raise ValueError("Sealed development export changed during loading")
        roles.append(pd.read_csv(path, dtype=str, keep_default_na=False, usecols=wanted))
        if sha256(path) != consumed[path.name]:
            raise ValueError("Sealed development export changed during loading")
    exported = pd.concat(roles, ignore_index=True).set_index("record_id", verify_integrity=True)
    frame = exported.loc[guarded.frame.record_id].reset_index()
    pd.testing.assert_frame_equal(frame[guarded.frame.columns], guarded.frame)
    freeze = json.loads((directory / "grouping_freeze.json").read_text(encoding="utf-8"))
    counts = guarded.audit["export_records"]
    return BaselineDevelopment(frame, {
        "split_directory": str(directory), "protocol_id": freeze["protocol_id"],
        "assignment_sha256": sha256(directory / "assignments.csv"),
        "complete_one_to_one_rows": sum(counts.values()) + len(guarded.protected_record_ids),
        "group_isolation": True, "assignment_counts": {**counts, "test": len(guarded.protected_record_ids)},
        "labeled_development_counts": guarded.audit["supervised_records"],
        "excluded_missing_label_counts": guarded.audit["missing_target_records_excluded_from_supervised_view"],
        "separate_test_export_opened": False,
        "final_snapshot_seal_or_exposure_history_verified": True,
        "guarded_development": guarded.audit,
        "category_mode": "validation feature experiment" if include_category else "base fields only",
    })


def _fingerprint(path, expected=None):
    if expected is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Expected SHA-256 must contain all 64 hexadecimal characters")
    actual = sha256(path)
    if expected is not None and actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}; check the shared version before training")
    return actual


def load_development(input_path, assignments_path, expected_assignments_sha256,
                     expected_input_sha256=None):
    """Read explicit membership, retain only labeled training/validation records.

    Reading the combined candidate CSV necessarily parses all rows; test rows
    are discarded before label normalization, text construction or model fitting.
    Assignment labels and separate test.csv exports are never loaded.
    """
    input_path, assignments_path = Path(input_path).resolve(), Path(assignments_path).resolve()
    if expected_assignments_sha256 is None:
        raise ValueError("An explicit expected assignment SHA-256 is required")
    hashes = {"assignments": _fingerprint(assignments_path, expected_assignments_sha256),
              "input": _fingerprint(input_path, expected_input_sha256)}
    columns = [*PROVENANCE, "record_id", "group_id", "split"]
    assignments = pd.read_csv(assignments_path, dtype=str, keep_default_na=False, usecols=columns)
    identities = record_ids(assignments)
    if assignments.empty or assignments.record_id.tolist() != list(identities):
        raise ValueError("Assignment record_id must match its canonical source key")
    check_assignments(assignments, identities)
    membership = assignments.set_index(PROVENANCE)
    expected_keys = set(membership.index)
    seen = set()
    development = []
    wanted = {*PROVENANCE, *MODEL_FIELDS, "ProductCategory", "Segment"}
    required = wanted - {"ProductCategory"}
    with pd.read_csv(input_path, dtype=str, keep_default_na=False,
                     usecols=lambda name: name in wanted, chunksize=10_000) as reader:
        for chunk in reader:
            if not required.issubset(chunk.columns):
                raise ValueError("Candidate is missing required source keys, Segment or product fields")
            keys = list(chunk[PROVENANCE].itertuples(index=False, name=None))
            chunk_keys = set(keys)
            if len(chunk_keys) != len(keys) or seen & chunk_keys:
                raise ValueError("Candidate source keys must be unique")
            if not chunk_keys.issubset(expected_keys):
                raise ValueError("Candidate contains source keys absent from assignments")
            seen.update(chunk_keys)
            # Labels in assignment files are not trusted or used as model inputs.
            joined = chunk.join(membership[["record_id", "group_id", "split"]],
                                on=PROVENANCE, validate="one_to_one")
            development.append(joined.loc[joined.split.isin(["train", "validation"])].copy())
    if seen != expected_keys:
        raise ValueError("Assignments and candidate must have complete one-to-one coverage")
    frame = pd.concat(development, ignore_index=True)
    frame["Segment"] = frame.Segment.str.strip()
    missing = frame.Segment.str.casefold().isin(MISSING_LABELS)
    excluded = frame.loc[missing].groupby("split").size().to_dict()
    frame = frame.loc[~missing].sort_values("record_id", kind="stable").reset_index(drop=True)
    if set(frame.split) != {"train", "validation"}:
        raise ValueError("Both training and validation need labeled records")
    # Fail if an input changed while it was being read.
    _fingerprint(input_path, hashes["input"])
    _fingerprint(assignments_path, hashes["assignments"])
    audit = {
        "input_path": str(input_path), "assignments_path": str(assignments_path),
        "input_sha256": hashes["input"], "assignment_sha256": hashes["assignments"],
        "join_keys": list(PROVENANCE), "complete_one_to_one_rows": len(seen),
        "group_isolation": True,
        "assignment_counts": assignments.groupby("split").size().to_dict(),
        "labeled_development_counts": frame.groupby("split").size().to_dict(),
        "excluded_missing_label_counts": {role: int(excluded.get(role, 0)) for role in ("train", "validation")},
        "test_rows_discarded_before_text_or_label_processing": True,
        "separate_test_export_opened": False,
        "final_snapshot_seal_or_exposure_history_verified": False,
        "scope": "Membership/hash checks only; use the team-agreed assignment version and evaluation protocol.",
    }
    return BaselineDevelopment(frame, audit)
