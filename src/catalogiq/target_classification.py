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

ELIGIBILITY_VERSION = "target-eligibility-20261115-v1"
ID_COLUMN = "JoiningKey"
KNOWN_RETAILERS = ("Amazon", "Walmart", "Walgreens", "CVS", "Target", "Kroger")
EXCLUDE_MARKER = "Exclude"

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

INELIGIBLE_REASONS = {
    "marked_exclude": "The source row carries the Exclude marker.",
    "missing_product_name": "No ProductName, so there is nothing to classify.",
    "missing_identifier": f"No {ID_COLUMN}, so the result could not be traced back.",
    "duplicate_identifier": f"{ID_COLUMN} is not unique within the file.",
    "unknown_retailer": ("Retailer is not one of the six known retailers, which indicates a "
                         "shifted row rather than a product."),
}


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


def eligibility(frame: pd.DataFrame) -> pd.Series:
    """One reason code per row, or NA when the row is eligible.

    Order matters only for reporting; a row is ineligible if any rule fires.
    """
    identifier = _text(frame, ID_COLUMN)
    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")

    def mark(mask: pd.Series, code: str) -> None:
        reasons.loc[mask & reasons.isna()] = code

    mark(_text(frame, "Exclude").fillna("").eq(EXCLUDE_MARKER), "marked_exclude")
    mark(_blank(identifier), "missing_identifier")
    mark(identifier.duplicated(keep=False) & ~_blank(identifier), "duplicate_identifier")
    mark(_blank(_text(frame, "ProductName")), "missing_product_name")
    mark(~_text(frame, "Retailer").fillna("").isin(KNOWN_RETAILERS), "unknown_retailer")
    return reasons


def eligibility_report(frame: pd.DataFrame, reasons: pd.Series) -> dict:
    counts = reasons.value_counts(dropna=True).to_dict()
    eligible = int(reasons.isna().sum())
    fill = {}
    for field in INPUT_FIELDS:
        present = ~_blank(_text(frame.loc[reasons.isna()], field))
        fill[field] = int(present.sum())
    return {
        "eligibility_version": ELIGIBILITY_VERSION,
        "rows_in_file": int(len(frame)),
        "eligible": eligible,
        "ineligible": int(len(frame) - eligible),
        "ineligible_by_reason": {k: int(v) for k, v in counts.items()},
        "reason_meanings": INELIGIBLE_REASONS,
        "model_input_fields_present_among_eligible": fill,
        "known_retailers": list(KNOWN_RETAILERS),
    }


def to_records(frame: pd.DataFrame, reasons: pd.Series | None = None) -> list[ProductRecord]:
    """Eligible rows as ProductRecord. record_id is the source identifier, never regenerated."""
    reasons = eligibility(frame) if reasons is None else reasons
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
    table["catalogiq:eligibility_version"] = ELIGIBILITY_VERSION
    return table
