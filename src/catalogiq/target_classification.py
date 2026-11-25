"""Apply the selected models to the unlabelled target dataset.

Ticket #1287. The acceptance criteria say "every eligible target product", and
nothing in the repo defined eligible, so the rule lives here with a version of
its own. Everything else is a thin pass over the frozen inference adapter: this
module chooses no labels, fits nothing and loads no model of its own.

When the serving members are absent the adapter reports `model_unavailable` per
route instead of failing, so this runs end to end today and produces the same
file shape it will produce with real predictions.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .inference_adapter import format_response
from .inference_engine import ProductRecord
from .target_minilm_preparation import INPUT_FIELDS

ELIGIBILITY_VERSION = "target-eligibility-20261116-v2"
ID_COLUMN = "JoiningKey"
DECISIONS_FILE = "row_decisions.csv"
ELIGIBLE_STATUS = "candidate"

# Kept beside every result so a reviewer can trace a row back to its source.
SOURCE_COLUMNS = (ID_COLUMN, "Retailer", "Category", "ProductCategory", "ProductBrand",
                  "ProductName", "Sku", "Upc", "ProductUrl")

# Columns the dashboard fills during review. Emitted empty so the file already has
# the shape its own export produces.
REVIEW_COLUMNS = ("catalogiq:mode", "catalogiq:review_status", "catalogiq:reviewer",
                  "catalogiq:reviewed_at", "catalogiq:review_note")
# Per-target column suffixes, in the dashboard's order and spelling.
TARGET_FIELDS = ("original_label", "final_label", "raw_confidence", "prediction_status",
                 "review_reasons", "model_version")
_FORMULA = re.compile(r"^[\s]*[=+\-@]|^[\t\r\n]")


def csv_cell(value):
    """Match the dashboard's csvCell: a leading quote stops a spreadsheet running the cell.

    Three real target values need this, among them the brand `+1HEROLABS`. The
    untouched value stays in the JSON sample, which is the audit copy.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return value
    text = str(value)
    return "'" + text if _FORMULA.search(text) else value


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string").str.strip()


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.str.len() == 0)


def load_target(path: str | Path) -> pd.DataFrame:
    """Read the target CSV as text. Nothing is coerced or imputed."""
    frame = pd.read_csv(path, dtype="string", low_memory=False, encoding="utf-8")
    return frame.reset_index(drop=True)


def load_decisions(path: str | Path, dataset: str = "target") -> pd.DataFrame:
    """Read one dataset's rows from the integrated cleaner's `row_decisions.csv`.

    Eligibility is not decided here. `docs/decisions.md` already settles it, and
    `integration.row_decision` implements it, so this module reads that verdict
    instead of keeping a second rule that can drift from the agreed one.
    """
    frame = pd.read_csv(path, dtype="string", low_memory=False, encoding="utf-8")
    missing = {"dataset", ID_COLUMN, "status"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is not a row_decisions export; missing {sorted(missing)}")
    frame = frame[frame["dataset"].astype("string").str.strip() == dataset].copy()
    if frame.empty:
        raise ValueError(f"{path} holds no rows for dataset {dataset!r}")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string").str.strip()
    return frame.reset_index(drop=True)


def eligibility(frame: pd.DataFrame, decisions: pd.DataFrame) -> pd.Series:
    """One reason per row, or NA when the integrated cleaner kept the row.

    A row is eligible when its status is `candidate`. Anything else carries the
    cleaner's own reason codes, so the reason a product was skipped is the reason
    the agreed policy gives, not one invented here.
    """
    identifier = _text(frame, ID_COLUMN)
    # Normalize both sides here so the join does not depend on the caller having
    # gone through load_decisions. The raw value is still what gets exported.
    decisions = decisions.copy()
    decisions[ID_COLUMN] = _text(decisions, ID_COLUMN)
    verdicts = decisions.drop_duplicates(subset=[ID_COLUMN], keep=False)
    if len(verdicts) != len(decisions):
        raise ValueError("row_decisions holds duplicate identifiers; cannot align by identity")
    present = identifier[~_blank(identifier)]
    unknown = set(present) - set(verdicts[ID_COLUMN])
    if unknown:
        raise ValueError(f"{len(unknown):,} target rows have no decision; "
                         "the decisions file and the target CSV disagree")

    status = verdicts.set_index(ID_COLUMN)["status"].astype("string").str.strip()
    reasons_by_id = verdicts.set_index(ID_COLUMN).get("quarantine_reasons")
    holds_by_id = verdicts.set_index(ID_COLUMN).get("hold_reasons")

    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    for position in frame.index:
        key = identifier[position]
        if pd.isna(key) or not str(key).strip():
            out[position] = "missing_identifier"
            continue
        verdict = status.get(key)
        if verdict == ELIGIBLE_STATUS:
            continue
        detail = ""
        for source in (reasons_by_id, holds_by_id):
            if source is not None:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    detail = value.strip()
                    break
        out[position] = f"{verdict}: {detail}" if detail else str(verdict)
    return out


def eligibility_report(frame: pd.DataFrame, reasons: pd.Series) -> dict:
    counts = reasons.value_counts(dropna=True).to_dict()
    codes: dict[str, int] = {}
    for reason, count in counts.items():
        _, _, detail = str(reason).partition(": ")
        for code in (detail.split(";") if detail else []):
            code = code.strip()
            if code:
                codes[code] = codes.get(code, 0) + int(count)
    eligible = int(reasons.isna().sum())
    fill = {}
    for field in INPUT_FIELDS:
        present = ~_blank(_text(frame.loc[reasons.isna()], field))
        fill[field] = int(present.sum())
    return {
        "eligibility_version": ELIGIBILITY_VERSION,
        "eligibility_source": f"integrated cleaner {DECISIONS_FILE}, status == {ELIGIBLE_STATUS!r}",
        "policy_reference": "docs/decisions.md, exclude policy 'keep'",
        "rows_in_file": int(len(frame)),
        "eligible": eligible,
        "ineligible": int(len(frame) - eligible),
        "ineligible_by_status": {k: int(v) for k, v in counts.items()},
        "ineligible_by_reason_code": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
        "model_input_fields_present_among_eligible": fill,
    }


def to_records(frame: pd.DataFrame, reasons: pd.Series) -> list[ProductRecord]:
    """Eligible rows as ProductRecord. record_id is the source identifier, never regenerated."""
    rows = frame.loc[reasons.isna()]
    identifier = _text(rows, ID_COLUMN)
    fields = {name: _text(rows, name) for name in INPUT_FIELDS}
    records = []
    for position in rows.index:
        features = {}
        for name in INPUT_FIELDS:
            value = fields[name].get(position)
            features[name] = None if pd.isna(value) or not str(value).strip() else str(value)
        records.append(ProductRecord(record_id=str(identifier[position]), features=features))
    return records


def classify(engine, records, *, batch_size: int = 256, progress=None):
    """Run the frozen engine in batches. Batching never changes a per-row result."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    results = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        batch = engine.predict(chunk)
        if len(batch) != len(chunk):
            raise ValueError(f"engine returned {len(batch)} results for {len(chunk)} records")
        if [r.record_id for r in batch] != [c.record_id for c in chunk]:
            raise ValueError("engine returned results in a different order than the records")
        results.extend(batch)
        if progress:
            progress(min(start + batch_size, len(records)), len(records))
    return results


def response(results, *, registry_version: str) -> dict:
    """The adapter's own documented envelope, which the dashboard already consumes."""
    return format_response(results, registry_version=registry_version)


def dashboard_table(frame: pd.DataFrame, results, envelope: dict) -> pd.DataFrame:
    """Flat CSV using the dashboard's `catalogiq:<target>:<field>` column naming.

    Source columns come first and are copied unchanged, so a reviewer can always
    see the original row beside the suggestion.
    """
    identifier = _text(frame, ID_COLUMN)
    by_id = {str(identifier[position]): position for position in frame.index
             if not pd.isna(identifier[position])}
    targets = sorted({name for product in envelope["products"] for name in product["predictions"]})

    rows = []
    for product in envelope["products"]:
        position = by_id.get(product["record_id"])
        row = {column: (frame.at[position, column] if position is not None
                        and column in frame.columns else pd.NA)
               for column in SOURCE_COLUMNS}
        for column in REVIEW_COLUMNS:
            row[column] = None
        row["catalogiq:mode"] = envelope.get("mode", "batch")
        row["catalogiq:eligibility"] = "eligible"
        row["catalogiq:review_required"] = product["review_required"]
        row["catalogiq:schema_version"] = envelope["schema_version"]
        row["catalogiq:engine_version"] = envelope["engine_version"]
        row["catalogiq:registry_version"] = envelope["relationship_registry_version"]
        row["catalogiq:eligibility_version"] = ELIGIBILITY_VERSION
        for target in targets:
            prediction = product["predictions"].get(target, {})
            prefix = f"catalogiq:{target}:"
            row[prefix + "original_label"] = prediction.get("predicted_label")
            row[prefix + "final_label"] = None
            row[prefix + "raw_confidence"] = prediction.get("raw_confidence")
            row[prefix + "calibrated_confidence"] = prediction.get("calibrated_confidence")
            row[prefix + "prediction_status"] = prediction.get("status")
            row[prefix + "selection_status"] = prediction.get("selection_status")
            row[prefix + "review_required"] = prediction.get("review_required")
            row[prefix + "review_reasons"] = "; ".join(prediction.get("review_reasons") or [])
            row[prefix + "model_version"] = (prediction.get("provenance") or {}).get("model_id")
        rows.append({k: csv_cell(v) for k, v in row.items()})
    return pd.DataFrame(rows)


def ineligible_table(frame: pd.DataFrame, reasons: pd.Series) -> pd.DataFrame:
    """The rows that were not sent to a model, with the reason, so nothing is silently dropped."""
    rows = frame.loc[reasons.notna()]
    # Copied raw, exactly as dashboard_table does. Stripping here would alter an
    # identifier, and "original identifiers are preserved" has to mean byte for byte
    # even on a row the shift corrupted.
    table = pd.DataFrame({column: (rows[column] if column in rows.columns else pd.NA)
                          for column in SOURCE_COLUMNS}).reset_index(drop=True)
    for column in SOURCE_COLUMNS:
        table[column] = table[column].map(csv_cell)
    table["catalogiq:eligibility"] = "ineligible"
    table["catalogiq:ineligible_reason"] = reasons.loc[rows.index].values
    table["catalogiq:eligibility_source"] = DECISIONS_FILE
    table["catalogiq:eligibility_version"] = ELIGIBILITY_VERSION
    return table
