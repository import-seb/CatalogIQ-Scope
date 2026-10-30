"""Read-only preparation for target-specific development allocation.

Reuse the authoritative V4 final groups and protected test mapping. No groups,
split assignments, or files are created. ``original_split`` always describes
the frozen source membership, including for groups required in future training.
Only development labels are read; test.csv is neither opened nor hashed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .features import sha256
from .split_protocol import verify_protocol_splits


SUPPORTED_TARGETS = ("Brand", "Platform", "Sub-Segment")
MISSING_TARGETS = frozenset(("", "null", "none", "nan"))
IDENTITY_FIELDS = ["record_id", "group_id", "split"]


@dataclass(frozen=True)
class PreparedTargetDevelopment:
    target_column: str
    frame: pd.DataFrame
    class_support: pd.DataFrame
    class_group_support: pd.DataFrame
    protected_test: pd.DataFrame
    required_training_group_ids: frozenset[str]
    audit: dict

    @property
    def labeled_frame(self) -> pd.DataFrame:
        return self.frame.loc[self.frame.is_labeled].copy()

    @property
    def training_only_classes(self) -> tuple[str, ...]:
        return tuple(self.class_support.loc[self.class_support.training_only, "label"])

    @property
    def eligible_for_both_classes(self) -> tuple[str, ...]:
        return tuple(self.class_support.loc[self.class_support.eligible_for_both, "label"])


def _validate_identity(frame, name):
    for column in IDENTITY_FIELDS:
        if column not in frame or frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"{name} requires nonblank {column}")
    if frame.record_id.duplicated().any():
        raise ValueError(f"{name} contains duplicate record IDs")


def _support(frame, target):
    labeled = frame.loc[frame.is_labeled]
    pairs = (labeled.groupby([target, "group_id", "original_split"], sort=True)
             .size().rename("labeled_records").reset_index().rename(columns={target: "label"}))
    counts = pairs.groupby("label", sort=True).group_id.nunique()
    single_group_classes = set(counts.index[counts.eq(1)])
    required = frozenset(pairs.loc[pairs.label.isin(single_group_classes), "group_id"])
    records = []
    for label, selected in pairs.groupby("label", sort=True):
        group_ids = set(selected.group_id)
        available = len(group_ids - required)
        single = len(group_ids) == 1
        records.append({
            "label": label,
            "labeled_records": int(selected.labeled_records.sum()),
            "development_groups": len(group_ids),
            "original_train_records": int(selected.loc[selected.original_split.eq("train"), "labeled_records"].sum()),
            "original_validation_records": int(selected.loc[selected.original_split.eq("validation"), "labeled_records"].sum()),
            "original_train_groups": int(selected.original_split.eq("train").sum()),
            "original_validation_groups": int(selected.original_split.eq("validation").sum()),
            "single_group_training_only": single,
            "available_validation_groups": available,
            "blocked_by_training_only_groups": not single and available == 0,
            "training_only": available == 0,
            "eligible_for_both": len(group_ids) >= 2 and available > 0,
        })
    support = pd.DataFrame(records, columns=[
        "label", "labeled_records", "development_groups", "original_train_records",
        "original_validation_records", "original_train_groups", "original_validation_groups",
        "single_group_training_only", "available_validation_groups",
        "blocked_by_training_only_groups", "training_only", "eligible_for_both",
    ]).astype({
        "label": str, **{column: "int64" for column in (
            "labeled_records", "development_groups", "original_train_records",
            "original_validation_records", "original_train_groups", "original_validation_groups",
            "available_validation_groups")},
        **{column: bool for column in ("single_group_training_only",
            "blocked_by_training_only_groups", "training_only", "eligible_for_both")},
    })
    return support, pairs, required


def prepare_target_development(directory, target_column: str) -> PreparedTargetDevelopment:
    """Verify portable V4 exports and describe their allocation constraints.

    All development records, missing labels, original roles and final group IDs
    are retained. Labels are not stripped, case-folded, merged or imputed; only
    missing detection strips/case-folds. Classes with a single development group
    require that entire group in training, including its other labels and missing
    rows. Multi-group classes sharing only required-training groups are flagged.

    ``eligible_for_both`` describes individual support after these constraints;
    it does not establish joint feasibility, balance or statistical reliability.
    No future assignment is selected, even for required-training groups.
    """
    if target_column not in SUPPORTED_TARGETS:
        raise ValueError(f"target_column must be one of {SUPPORTED_TARGETS}")
    directory = Path(directory).resolve()
    completion_path = directory / "completion.json"
    completion_digest = sha256(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf8"))
    if completion.get("format") != "catalogiq-portable-split-v1":
        raise ValueError("Preparation requires the authoritative portable V4 splits")
    # The existing verifier enforces repository-pinned groups, memberships and
    # exposure selectors, even if someone has resealed modified local artifacts.
    verify_protocol_splits(directory, verify_exports=False)
    names = ("grouping_freeze.json", "assignments.csv", "protected_final_test.csv",
             "train.csv", "validation.csv")
    consumed = {}
    for name in names:
        expected = completion.get("artifact_sha256", {}).get(name)
        if not expected or sha256(directory / name) != expected:
            raise ValueError(f"Frozen {name} was modified or has no checksum")
        consumed[name] = expected
    freeze = json.loads((directory / "grouping_freeze.json").read_text(encoding="utf8"))
    if (freeze.get("format") != "catalogiq-portable-split-v1"
            or freeze.get("grouping_version") != "rule-final-v4"):
        raise ValueError("Preparation requires frozen V4 groups")

    def read(name, columns):
        return pd.read_csv(directory / name, dtype=str, keep_default_na=False,
                           encoding="utf-8-sig", usecols=columns)[columns]

    assignments = read("assignments.csv", IDENTITY_FIELDS)
    protected = read("protected_final_test.csv", IDENTITY_FIELDS)
    for name, frame in (("assignments", assignments), ("protected test", protected)):
        _validate_identity(frame, name)
    if not assignments.split.isin(("train", "validation", "test")).all():
        raise ValueError("Invalid frozen assignment role")
    if assignments.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Frozen group crosses split boundaries")
    if protected.empty or not protected.split.eq("test").all():
        raise ValueError("Protected test must contain only test records")
    expected_test = assignments.loc[assignments.split.eq("test")]
    if (protected.set_index("record_id").group_id.to_dict()
            != expected_test.set_index("record_id").group_id.to_dict()):
        raise ValueError("Protected test record/group mapping differs from frozen assignments")
    protected_ids, protected_groups = set(protected.record_id), set(protected.group_id)
    pieces = []
    for role in ("train", "validation"):
        frame = read(f"{role}.csv", [*IDENTITY_FIELDS, "Retailer", target_column])
        _validate_identity(frame, role)
        if frame.empty or not frame.split.eq(role).all():
            raise ValueError(f"{role} export has empty or contaminated split roles")
        if set(frame.record_id) & protected_ids or set(frame.group_id) & protected_groups:
            raise ValueError("Protected test record or group appears in development")
        expected = assignments.loc[assignments.split.eq(role)]
        if (frame.set_index("record_id").group_id.to_dict()
                != expected.set_index("record_id").group_id.to_dict()):
            raise ValueError(f"{role} record/group mapping differs from frozen assignments")
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).sort_values("record_id", kind="stable").reset_index(drop=True)
    if frame.record_id.duplicated().any() or frame.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Development records or groups cross original split roles")
    frame = frame.rename(columns={"split": "original_split"})
    frame["is_labeled"] = ~frame[target_column].str.strip().str.casefold().isin(MISSING_TARGETS)
    support, pairs, required = _support(frame, target_column)
    protected = protected.sort_values("record_id", kind="stable").reset_index(drop=True)
    if (sha256(completion_path) != completion_digest
            or any(sha256(directory / name) != digest for name, digest in consumed.items())):
        raise ValueError("Frozen preparation inputs changed during loading")
    audit = {
        "version": "target-development-preparation-v1", "target_column": target_column,
        "protocol_id": freeze.get("protocol_id"), "grouping_version": freeze["grouping_version"],
        "completion_sha256": completion_digest, "consumed_artifact_sha256": consumed,
        "development_records": len(frame), "development_groups": int(frame.group_id.nunique()),
        "labeled_records": int(frame.is_labeled.sum()), "missing_target_records": int((~frame.is_labeled).sum()),
        "classes": len(support), "single_group_classes": int(support.single_group_training_only.sum()),
        "eligible_for_both_classes": int(support.eligible_for_both.sum()),
        "blocked_by_training_only_groups": int(support.blocked_by_training_only_groups.sum()),
        "required_training_groups": len(required),
        "protected_test_records": len(protected), "protected_test_groups": int(protected.group_id.nunique()),
        "original_membership_preserved": True, "missing_targets_imputed": False,
        "allocation_performed": False, "test_csv_read_or_hashed": False,
    }
    return PreparedTargetDevelopment(target_column, frame, support, pairs, protected, required, audit)
