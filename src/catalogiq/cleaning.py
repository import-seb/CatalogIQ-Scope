"""Auditable implementation of notebooks/02_clean_target_cols.ipynb.

Run from the repository root: python -m catalogiq
Reasons 2/3 are review-only; manufacturer validation can quarantine rows.
Row-content checks live in catalogiq.structural and must be run separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGET_COLUMNS = ["Mnfr", "Brand", "Platform", "Segment", "Sub-Segment", "TargetAgeGroup"]
PROVENANCE = ["dataset", "source_sha256", "source_row"]
REASONS = {
    2: "Target label starts with a space or lowercase character (review heuristic)",
    3: "Target label frequency below lower IQR bound (review only; not invalidity)",
    6: "Populated Mnfr outside J&J / All others",
}
CHANGE_COLUMNS = PROVENANCE + ["column", "old_value", "new_value", "rule"]
FLAG_COLUMNS = PROVENANCE + ["reason_code", "column", "value"]
QUARANTINE_REASONS = {6}
POLICY_VERSION = "target-labels-v2"


@dataclass
class CleaningResult:
    cleaned: pd.DataFrame
    quarantine: pd.DataFrame
    changes: pd.DataFrame
    flags: pd.DataFrame
    hierarchy_violations: pd.DataFrame


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(path: Path) -> pd.DataFrame:
    """Preserve source strings (including identifiers); use notebook NA parsing."""
    frame = pd.read_csv(path, dtype="string", low_memory=False)
    missing = set(TARGET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing target columns: {sorted(missing)}")
    reserved = set(PROVENANCE + ["Q_REASON"]) & set(frame.columns)
    if reserved:
        raise ValueError(f"{path}: reserved output columns already present: {sorted(reserved)}")
    frame["dataset"] = path.name
    frame["source_sha256"] = sha256(path)
    frame["source_row"] = range(len(frame))
    return frame


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    spread = q3 - q1
    return q1 - 1.5 * spread, q3 + 1.5 * spread


def check_single_parent_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: shared-parent Sub-Segments are allowed."""
    pairs = frame.dropna(subset=["Segment", "Sub-Segment"])
    num_parents = pairs.groupby("Sub-Segment")["Segment"].nunique()
    single = num_parents.index[num_parents.eq(1)]
    parents = pairs.loc[pairs["Sub-Segment"].isin(single)].groupby("Sub-Segment")["Segment"].first()
    result = parents.rename("Segment").reset_index()
    result["child_rows"] = result["Sub-Segment"].map(frame["Sub-Segment"].value_counts())
    result["parent_rows"] = result["Segment"].map(frame["Segment"].value_counts())
    result["difference"] = result["child_rows"] - result["parent_rows"]
    return result.loc[result["difference"].gt(0)].reset_index(drop=True)


def clean_training(frame: pd.DataFrame) -> CleaningResult:
    """Return cleaned and quarantined rows without mutating the input.

    Requires provenance from load_source. Reasons 2 and 3 are review-only;
    reason 6 quarantines a row even when reason 2 is also present.
    Former reasons 4/5 have moved to the independent structural checker.
    """
    required = set(TARGET_COLUMNS + PROVENANCE)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    if not frame.index.is_unique:
        raise ValueError("Training frame must have a unique row index")
    if "Q_REASON" in frame:
        raise ValueError("Pass source records, not an already cleaned output")
    working = frame.copy(deep=True)
    reasons = {index: set() for index in frame.index}
    flag_parts = []
    change_parts = []

    def flag(mask: pd.Series, reason: int, column: str = "") -> None:
        indexes = working.index[mask.fillna(False)]
        if not len(indexes):
            return
        for index in indexes:
            reasons[index].add(reason)
        details = working.loc[indexes, PROVENANCE].copy()
        details["reason_code"] = reason
        details["column"] = column
        details["value"] = working.loc[indexes, column] if column else pd.NA
        flag_parts.append(details)

    for column in TARGET_COLUMNS:
        labels = working[column].astype("string")
        # Preserve notebook behavior, including legitimate lowercase labels.
        abnormal = labels.map(lambda value: bool(value) and
                              (value.startswith(" ") or value[0].islower()), na_action="ignore")
        flag(abnormal.astype("boolean"), 2, column)
        counts = labels.value_counts()
        lower, _ = iqr_bounds(counts)
        flag(labels.map(counts).lt(lower), 3, column)

    invalid_mnfr = working["Mnfr"].notna() & ~working["Mnfr"].isin({"J&J", "All others"})
    flag(invalid_mnfr, 6, "Mnfr")
    excluded = pd.Series({index: bool(codes & QUARANTINE_REASONS)
                          for index, codes in reasons.items()}, dtype=bool)

    def replace(mask: pd.Series, column: str, value: str, rule: str) -> None:
        changed = ~excluded & mask.fillna(False) & working[column].ne(value).fillna(True)
        if not changed.any():
            return
        entries = working.loc[changed, PROVENANCE].copy()
        entries["column"] = column
        entries["old_value"] = working.loc[changed, column]
        entries["new_value"] = value
        entries["rule"] = rule
        change_parts.append(entries)
        working.loc[changed, column] = value

    replace(working["Sub-Segment"].eq("Cold / Flu"), "Sub-Segment", "Cold/Flu", "slash_spacing")
    replace(working["Sub-Segment"].eq("Other Lifestyle"), "Sub-Segment",
            "Other Lifestyle CHC", "reviewed_other_lifestyle")
    working["Q_REASON"] = [sorted(reasons[index]) for index in working.index]
    cleaned = working.loc[~excluded].copy()
    quarantine = working.loc[excluded].copy()
    changes = pd.concat(change_parts, ignore_index=True) if change_parts else pd.DataFrame(columns=CHANGE_COLUMNS)
    flags = pd.concat(flag_parts, ignore_index=True) if flag_parts else pd.DataFrame(columns=FLAG_COLUMNS)
    return CleaningResult(cleaned, quarantine, changes, flags, check_single_parent_counts(cleaned))


def target_distributions(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Count input labels and retained cleaned labels, excluding quarantine."""
    distributions = {}
    for column in TARGET_COLUMNS:
        counts = {}
        for stage, frame in (("before", before), ("after", after)):
            labels = frame[column].dropna().astype("string")
            if labels.eq("<missing>").any():
                raise ValueError(f"{column} contains the reserved summary label '<missing>'")
            counts[stage] = {str(label): int(count) for label, count in labels.value_counts().items()}
            counts[stage]["<missing>"] = int(frame[column].isna().sum())
        # Use matching keys so removed/new labels appear explicitly with zero counts.
        labels = sorted(set(counts["before"]) | set(counts["after"]))
        distributions[column] = {
            stage: {label: values.get(label, 0) for label in labels}
            for stage, values in counts.items()
        }
    return distributions


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    if "Q_REASON" in output:
        output["Q_REASON"] = output["Q_REASON"].map(json.dumps)
    output.to_csv(path, index=False, encoding="utf-8")


def run(train_path: Path, target_path: Path, output_dir: Path) -> dict:
    train_path, target_path, output_dir = (p.resolve() for p in (train_path, target_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new --output-dir: {output_dir}")
    train = load_source(train_path)
    target = load_source(target_path)
    result = clean_training(train)
    target_output = target.copy()
    target_output["Q_REASON"] = [[] for _ in range(len(target))]
    # The notebook cleans training labels, not the unlabelled prediction file.
    outputs = {
        "training_cleaned.csv": result.cleaned,
        "training_quarantine.csv": result.quarantine,
        "target_unchanged.csv": target_output,
        "label_changes.csv": result.changes,
        "review_flags.csv": result.flags,
        "hierarchy_violations.csv": result.hierarchy_violations,
    }
    counts = result.flags["reason_code"].value_counts().sort_index()
    summary = {
        "policy_version": POLICY_VERSION,
        "structural_screening_applied": False,
        "source_notebook": "notebooks/02_clean_target_cols.ipynb",
        "sources": {"training": {"path": str(train_path), "sha256": sha256(train_path)},
                    "target": {"path": str(target_path), "sha256": sha256(target_path)}},
        "rows": {"training_input": len(train), "training_cleaned": len(result.cleaned),
                 "training_quarantine": len(result.quarantine),
                 "training_flagged": int(result.cleaned["Q_REASON"].map(bool).sum()), "target_unchanged": len(target)},
        "label_changes": len(result.changes),
        "target_distributions": target_distributions(train, result.cleaned),
        "target_distributions_scope": {
            "dataset": "training",
            "before": "All parsed training input rows",
            "after": "Retained cleaned rows including review-only flags; quarantined rows excluded",
            "missing": "<missing> counts parsed null values, including zero counts",
        },
        "changes_by_rule": {str(k): int(v) for k, v in result.changes["rule"].value_counts().items()},
        "flag_events_by_reason": {str(k): int(v) for k, v in counts.items()},
        "reason_codes": REASONS,
        "quarantine_reason_codes": sorted(QUARANTINE_REASONS),
        "review_only_reason_codes": [2, 3],
        "manufacturer_policy": "Preserve supplied Mnfr, including missing values; validate vocabulary only",
        "hierarchy_violations": len(result.hierarchy_violations),
        "notes": [
            "source_row is the zero-based parsed CSV record, not a physical line number.",
            "Reasons 2/3 are review-only. Reason 6 quarantines rows, including mixed-reason rows.",
            "Frequency alone is review-only; lowercase labels can be legitimate.",
            "Source columns are read as strings with pandas default NA parsing.",
            "Former row-content reasons 4/5 moved to catalogiq.structural; run structural mode separately on both raw sources.",
            "Invalid Mnfr values are preserved in quarantine for review.",
            "Target data is passed through with provenance; it is not screened or filled.",
            "Brand, Platform, Segment and TargetAgeGroup are never inferred or replaced.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, output in outputs.items():
        write_csv(output, output_dir / filename)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
