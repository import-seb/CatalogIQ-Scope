"""Independent structural screening of original CSVs; never cleans or drops rows.

The API accepts raw sources, not cleaner outputs or their quarantine masks.
All findings and decisions carry role/SHA-256/one-based-record identity,
matching the merged feature and integration contract without importing cleaners.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


POLICY_VERSION = "structural-v2"
KEY = ("dataset", "source_sha256", "source_row")
TARGETS = ("Mnfr", "Brand", "Platform", "Segment", "Sub-Segment", "TargetAgeGroup")
NUMERIC = ("ProductRating", "ProductReviewsCount", "ReviewsCount", "XRatXRev")
UNKNOWN = ("Sun1", "Sun2", "Sun3", "Sun4", "Sun5", "Notes")
QUARANTINE_RULES = (
    "csv_field_count", "text_in_multiple_numeric_anchors",
    "text_numeric_failure_with_corroboration",
    "unexpected_exclude_with_populated_unknown_columns",
    "multiple_suspicious_targets", "suspicious_target_with_displacement",
)
LIMITATIONS = [
    "Heuristic review quarantine, not confirmed corruption or an approved modeling mask.",
    "Missingness and absent numeric anchors alone never quarantine; sparse rows can be valid.",
    "Missing tokens are trimmed blank/null only; raw strings are preserved, including NA and NaN.",
    "Target vocabularies and numeric meanings are working assumptions, not sponsor-approved schema.",
    "Unknown timestamp encoding, Exclude semantics, and unknown metadata need team confirmation.",
    "No inference of shifts between fields of the same type; no attempt to reconstruct records.",
    "Unparseable CSV syntax aborts without a completion summary; no reliable record identity can be assigned.",
    "Cleaner masks are neither consumed nor combined; legacy zero-based artifacts require explicit conversion.",
]


@dataclass(frozen=True)
class StructuralConfig:
    """Serializable policy; statistical cutoffs are diagnostics, never exclusion gates."""

    numeric_fields: tuple[str, ...] = NUMERIC
    count_fields: tuple[str, ...] = ("ProductReviewsCount", "ReviewsCount")
    missingness_fields: tuple[str, ...] = (
        "Retailer", "ProductCategory", "ProductBrand", "ProductName", *NUMERIC,
        "ProductUrl", "ProductImageUrl", "ProductDescription", "ProductContents",
    )
    missing_tokens: tuple[str, ...] = ("", "null")
    missingness_iqr_multiplier: float = 1.5
    manufacturer_values: tuple[str, ...] = ("All others", "J&J")
    age_group_values: tuple[str, ...] = ("Adult", "Children", "Infant")
    quarantine_rules: tuple[str, ...] = QUARANTINE_RULES

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "missingness_iqr_multiplier":
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("missingness_iqr_multiplier must be finite and nonnegative")
            else:
                if (not isinstance(value, tuple) or any(not isinstance(v, str) for v in value)
                        or len(value) != len(set(value))):
                    raise ValueError(f"{name} must contain unique strings")
        if not self.numeric_fields or not self.missingness_fields:
            raise ValueError("Numeric and missingness field scopes cannot be empty")
        if not set(self.numeric_fields) <= set(NUMERIC):
            raise ValueError("Only documented measurement fields may be numeric anchors")
        if not set(self.count_fields) <= set(self.numeric_fields):
            raise ValueError("Count fields must be numeric anchors")
        if set(self.missingness_fields) & set((*TARGETS, "Category", *KEY)):
            raise ValueError("Missingness must exclude targets and provenance")
        if not set(self.quarantine_rules) <= set(QUARANTINE_RULES):
            raise ValueError("Unknown quarantine rule")
        if any(v != v.strip().lower() for v in self.missing_tokens) or "" not in self.missing_tokens:
            raise ValueError("Missing tokens must be normalized and include the empty string")

    @classmethod
    def from_dict(cls, values: dict) -> StructuralConfig:
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("Configuration must be an object with known policy fields")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def missing(value: str, config: StructuralConfig) -> bool:
    return value.strip().lower() in config.missing_tokens


def numeric_error(value: str, column: str, config: StructuralConfig) -> str | None:
    if missing(value, config):
        return None
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        return "non_numeric"
    if not number.is_finite():
        return "non_finite"
    if number < 0:
        return "negative_measurement"
    if column in config.count_fields and number != number.to_integral_value():
        return "fractional_count"
    return None


def is_url(value: str) -> bool:
    return value.strip().lower().startswith(("http://", "https://"))


def valid_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except ValueError:
        return False


def required_fields(config: StructuralConfig) -> set[str]:
    return set((*config.numeric_fields, *config.missingness_fields, *TARGETS, *UNKNOWN,
                "Exclude", "MDM_Id", "MDM_InsertDateTime", "ProductUrl", "ProductImageUrl"))


def literal_records(stream):
    """Match csv.DictReader data records used by feature/integrated cleaning.

    Empty physical lines are skipped. Whitespace-only records and quoted empty
    fields count, even if their width is malformed. Embedded newlines stay intact.
    The caller consumes and validates the header before invoking this iterator.
    """
    for values in csv.reader(stream, strict=True):
        if values:
            yield values


def records(path: Path, config: StructuralConfig):
    """Yield (one-based record, header, literal cell strings), skipping empty lines.

    Quoted multiline fields count once. Wrong widths are preserved for quarantine;
    ambiguous syntax and duplicate/missing schema columns fail the run.
    """
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream, strict=True), None)
        if not fields or len(fields) != len(set(fields)) or any(not f for f in fields):
            raise ValueError(f"{path}: missing or duplicate header columns")
        absent = required_fields(config) - set(fields)
        if absent:
            raise ValueError(f"{path}: missing structural columns: {sorted(absent)}")
        if set(fields) & set((*KEY, "Q_REASON")):
            raise ValueError(f"{path}: reserved provenance/cleaner columns; pass original source")
        for position, values in enumerate(literal_records(stream), 1):
            yield position, fields, values


def missing_count(row: dict[str, str], config: StructuralConfig) -> int:
    return sum(missing(row[c], config) for c in config.missingness_fields)


def quantile(counts: Counter, fraction: float) -> float:
    """Linear quantile of integer missing counts, without retaining source rows."""
    total = sum(counts.values())
    if not total:
        return 0.0
    position = (total - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    seen = 0
    endpoints = {}
    for value, count in sorted(counts.items()):
        for index in (low, high):
            if seen <= index < seen + count:
                endpoints[index] = value
        seen += count
    return endpoints[low] + (endpoints[high] - endpoints[low]) * (position - low)


def assess_row(row: dict[str, str], identity: dict, *, config: StructuralConfig,
               missingness_upper_count: float) -> dict:
    """Pure row decision with explicit dataset-profile context; no cleaner imports.

    Consumers use this result's keyed findings, never infer a decision from flags.
    Input cells must be original strings. A single field cannot corroborate itself.
    """
    if (set(identity) != set(KEY) or not isinstance(identity["dataset"], str)
            or identity["dataset"] not in {"training", "target"} or not isinstance(identity["source_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity["source_sha256"])
            or type(identity["source_row"]) is not int or identity["source_row"] < 1):
        raise ValueError("Invalid source identity")
    absent = required_fields(config) - set(row)
    if absent or any(not isinstance(v, str) for v in row.values()):
        raise ValueError("Expected complete raw string fields")
    if not math.isfinite(missingness_upper_count) or missingness_upper_count < 0:
        raise ValueError("Invalid missingness profile")
    findings, reasons = [], []

    def emit(code: str, columns=(), *, quarantine=False, **evidence):
        enabled = quarantine and code in config.quarantine_rules
        findings.append({**identity, "reason_code": code, "quarantine_rule": bool(enabled),
                         "evidence": {"raw_values": {c: row[c] for c in sorted(columns)}, **evidence}})
        if enabled:
            reasons.append(code)

    errors = {}
    for column in config.numeric_fields:
        error = numeric_error(row[column], column, config)
        if error:
            errors[column] = error
            emit("numeric_" + error, [column], column=column)
    text_numeric = {c for c, error in errors.items() if error == "non_numeric"}
    if all(missing(row[c], config) or c in errors for c in config.numeric_fields):
        emit("no_valid_numeric_anchor", config.numeric_fields)

    displaced = set()
    if is_url(row["MDM_Id"]):
        displaced.add("MDM_Id")
        emit("url_in_identifier", ["MDM_Id"])
    for column in ("ProductUrl", "ProductImageUrl"):
        if not missing(row[column], config) and not valid_url(row[column]):
            displaced.add(column)
            emit("invalid_url_field", [column], column=column)
    stamp = row["MDM_InsertDateTime"]
    if not missing(stamp, config):
        try:
            finite = Decimal(stamp.strip()).is_finite()
        except InvalidOperation:
            finite = False
        if not finite:
            emit("unconfirmed_timestamp_format", ["MDM_InsertDateTime"])

    populated_unknown = {c for c in UNKNOWN if not missing(row[c], config)}
    if populated_unknown:
        emit("populated_unknown_metadata", populated_unknown)
    unexpected_exclude = not missing(row["Exclude"], config) and row["Exclude"].strip() != "Exclude"
    if unexpected_exclude:
        emit("unexpected_exclude", ["Exclude"])

    suspicious_targets = set()
    vocabularies = {"Mnfr": config.manufacturer_values, "TargetAgeGroup": config.age_group_values}
    for column in TARGETS:
        value = row[column]
        if missing(value, config):
            continue
        if value.startswith(" ") or value[0].islower():
            emit("target_format_review", [column], column=column)
        # Formatting and frequency are not semantic evidence of displacement.
        if is_url(value) or (column in vocabularies and value.strip() not in vocabularies[column]):
            suspicious_targets.add(column)
            emit("suspicious_target_value", [column], column=column,
                 basis="url_in_label" if is_url(value) else "outside_working_vocabulary",
                 allowed_values=list(vocabularies.get(column, ())))

    count = missing_count(row, config)
    if count > missingness_upper_count:
        emit("high_feature_missingness", [c for c in config.missingness_fields if missing(row[c], config)],
             missing_count=count, field_count=len(config.missingness_fields),
             upper_count=missingness_upper_count, comparison="strictly_greater")

    if len(text_numeric) >= 2:
        emit("text_in_multiple_numeric_anchors", text_numeric, quarantine=True)
    elif text_numeric and (len(errors) >= 2 or displaced):
        emit("text_numeric_failure_with_corroboration", set(errors) | displaced, quarantine=True)
    if unexpected_exclude and populated_unknown:
        emit("unexpected_exclude_with_populated_unknown_columns", {"Exclude"} | populated_unknown,
             quarantine=True)
    if len(suspicious_targets) >= 2:
        emit("multiple_suspicious_targets", suspicious_targets, quarantine=True)
    if suspicious_targets and (text_numeric or displaced):
        emit("suspicious_target_with_displacement", suspicious_targets | text_numeric | displaced,
             quarantine=True)
    return {**identity, "quarantine": bool(reasons), "reason_codes": sorted(reasons),
            "finding_codes": sorted({f["reason_code"] for f in findings}), "findings": findings}


def run(train_path: Path, target_path: Path, output_dir: Path,
        config: StructuralConfig | None = None) -> dict:
    """Screen both full sources and write a new auditable run; no source edits."""
    config = config or StructuralConfig()
    paths = {"training": Path(train_path).resolve(), "target": Path(target_path).resolve()}
    output_dir = Path(output_dir).resolve()
    for path in paths.values():
        if output_dir.is_relative_to(path.parent) or path.is_relative_to(output_dir):
            raise ValueError("Output must be outside raw input directories and their ancestors")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    fingerprints = {role: sha256(path) for role, path in paths.items()}
    if paths["training"] == paths["target"]:
        raise ValueError("Training and target must be distinct source files")
    # Validate all syntax/schema and profile full-width feature records first.
    profiles = {}
    for role, path in paths.items():
        counts = Counter()
        total = 0
        for _, fields, values in records(path, config):
            total += 1
            if len(values) == len(fields):
                counts[missing_count(dict(zip(fields, values)), config)] += 1
        q1, q3 = quantile(counts, .25), quantile(counts, .75)
        profiles[role] = {"rows": total, "profile_rows": sum(counts.values()),
                          "q1_missing_count": q1, "q3_missing_count": q3,
                          "upper_missing_count": q3 + config.missingness_iqr_multiplier * (q3 - q1)}
        if sha256(path) != fingerprints[role]:
            raise RuntimeError("Source changed during profiling")

    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"policy_version": POLICY_VERSION, "config": asdict(config), "datasets": {},
               "identity": {"columns": list(KEY), "dataset": "training|target", "source_row_base": 1,
                            "source_row": "parsed CSV record excluding header and empty physical lines"},
               "rows_removed": 0, "rows_changed": 0, "masks_combined": False,
               "limitations": LIMITATIONS}
    for role, path in paths.items():
        decisions = Counter({"keep": 0, "quarantine": 0})
        finding_counts, reason_counts, by_decision = Counter(), Counter(), {}
        with (output_dir / f"{role}_decisions.csv").open("w", encoding="utf-8", newline="") as audit, \
             (output_dir / f"{role}_evidence.jsonl").open("w", encoding="utf-8", newline="\n") as detail:
            writer = csv.DictWriter(audit, fieldnames=[*KEY, "quarantine", "reason_codes", "finding_codes"])
            writer.writeheader()
            for position, fields, values in records(path, config):
                identity = dict(dataset=role, source_sha256=fingerprints[role], source_row=position)
                if len(values) != len(fields):
                    code = "csv_field_count"
                    quarantine = code in config.quarantine_rules
                    result = {**identity, "quarantine": quarantine,
                              "reason_codes": [code] if quarantine else [], "finding_codes": [code],
                              "findings": [{**identity, "reason_code": code, "quarantine_rule": quarantine,
                                            "evidence": {"expected_fields": len(fields), "actual_fields": len(values)}}]}
                else:
                    result = assess_row(dict(zip(fields, values)), identity, config=config,
                                        missingness_upper_count=profiles[role]["upper_missing_count"])
                decision = "quarantine" if result["quarantine"] else "keep"
                decisions[decision] += 1
                finding_counts.update(result["finding_codes"])
                reason_counts.update(result["reason_codes"])
                for code in result["finding_codes"]:
                    by_decision.setdefault(code, {"keep": 0, "quarantine": 0})[decision] += 1
                writer.writerow({**identity, "quarantine": int(result["quarantine"]),
                                 "reason_codes": json_text(result["reason_codes"]),
                                 "finding_codes": json_text(result["finding_codes"])})
                if result["findings"]:
                    detail.write(json_text({**result, "raw_columns": fields, "raw_values": values}) + "\n")
        if sum(decisions.values()) != profiles[role]["rows"] or sha256(path) != fingerprints[role]:
            raise RuntimeError("Source changed during screening; run incomplete")
        summary["datasets"][role] = {
            "dataset": role, "source_file": path.name, "source_path": str(path), "source_sha256": fingerprints[role],
            **profiles[role], "decisions": dict(decisions), "rows_by_finding": dict(sorted(finding_counts.items())),
            "rows_by_quarantine_reason": dict(sorted(reason_counts.items())),
            "rows_by_finding_and_decision": dict(sorted(by_decision.items())),
        }
    summary["output_sha256"] = {p.name: sha256(p) for p in sorted(output_dir.iterdir())}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
