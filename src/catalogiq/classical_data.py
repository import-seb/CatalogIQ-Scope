"""Target-specific views of verified, shared train/validation exports."""
from pathlib import Path
import json

import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .segment_baseline_data import MISSING_LABELS
from .segment_transformer import MODEL_FIELDS, _hash_items
from .split_protocol import verify_protocol_splits
from .splitting import check_assignments, record_ids


TARGETS = ("Segment", "Sub-Segment", "Brand", "Mnfr", "Platform", "TargetAgeGroup")


def load_target_development(directory, targets, *, protocol_dir=None):
    """Verify the registered split, then filter missing labels separately per target.

    Segment's supervised mask must not be reused for another target. The existing
    verifier checks all exported identities/groups; test.csv is neither read nor
    hashed. A future split must have a compatible authoritative protocol bundle;
    an arbitrary self-sealed CSV directory is not sufficient.
    """
    directory = Path(directory).resolve()
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
        raise ValueError("Select unique targets from " + ", ".join(TARGETS))
    verify_protocol_splits(directory, verify_exports=False, protocol_dir=protocol_dir)
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol_manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((directory / "grouping_freeze.json").read_text(encoding="utf-8"))
    columns = list(dict.fromkeys([*PROVENANCE, "record_id", "group_id", "split",
                                 *MODEL_FIELDS, "ProductCategory", *targets]))
    parts, consumed = [], {}
    for role in ("train", "validation"):
        path = directory / f"{role}.csv"
        expected = completion["artifact_sha256"][path.name]
        if sha256(path) != expected:
            raise ValueError(f"Changed {role} export")
        part = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
        if not part.split.eq(role).all():
            raise ValueError(f"Wrong split role in {role} export")
        if part.record_id.tolist() != list(record_ids(part)) or part[PROVENANCE].duplicated().any():
            raise ValueError("Export identities must match unique canonical source keys")
        if sha256(path) != expected:
            raise ValueError(f"{role} export changed during reading")
        consumed[path.name] = expected
        parts.append(part)
    joined = pd.concat(parts, ignore_index=True)
    check_assignments(joined, joined.record_id)
    protected = pd.read_csv(directory / "protected_final_test.csv", dtype=str,
                            keep_default_na=False, usecols=["record_id", "group_id"])
    if set(joined.record_id) & set(protected.record_id) or set(joined.group_id) & set(protected.group_id):
        raise ValueError("Protected final-test identity or group appears in development")
    views, profiles = {}, {}
    for target in targets:
        labels = joined[target].str.strip()
        missing = labels.str.casefold().isin(MISSING_LABELS)
        view = joined.loc[~missing, [*PROVENANCE, "record_id", "group_id", "split",
                                    *MODEL_FIELDS, "ProductCategory", target]].copy()
        view[target] = labels.loc[~missing]
        view = view.sort_values("record_id", kind="stable").reset_index(drop=True)
        train = view.loc[view.split.eq("train"), target]
        validation = view.loc[view.split.eq("validation"), target]
        unseen = sorted(set(validation) - set(train))
        profiles[target] = {
            "training_rows": len(train), "validation_rows": len(validation),
            "training_classes": train.nunique(), "validation_classes": validation.nunique(),
            "training_class_counts": train.value_counts().sort_index().to_dict(),
            "validation_class_counts": validation.value_counts().sort_index().to_dict(),
            "unseen_validation_labels": unseen,
            "unseen_validation_rows": int(validation.isin(unseen).sum()),
            "missing_labels_excluded": {r: int((missing & joined.split.eq(r)).sum())
                                         for r in ("train", "validation")},
            "training_record_ids_sha256": _hash_items(view.loc[view.split.eq("train"), "record_id"].tolist()),
            "validation_record_ids_sha256": _hash_items(view.loc[view.split.eq("validation"), "record_id"].tolist()),
        }
        views[target] = view
    return views, {
        "protocol_id": freeze["protocol_id"], "protocol_manifest_sha256": sha256(directory / "protocol_manifest.json"),
        "split_format": freeze["format"], "allocation_target": protocol.get("target", "Segment"),
        "consumed_artifact_sha256": consumed,
        "protected_test_rows": len(protected), "test_opened": False,
        "record_and_group_isolation": True, "new_splits_created": False,
        "profiles": profiles,
    }
