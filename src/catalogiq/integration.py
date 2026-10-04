"""Join feature, target and independent structural decisions by source identity.

This produces a reproducible candidate mask, not an approved model dataset.
The legacy target-only entry point keeps its original parsing and row numbering.
"""
from collections import Counter
from contextlib import ExitStack
import csv
from itertools import zip_longest
import json
from pathlib import Path
import platform

import pandas as pd

from .cleaning import clean_training, TARGET_COLUMNS, QUARANTINE_REASONS, write_csv
from .feature_decisions import KEY, EXCLUDED_MODEL_FIELDS, run as run_features
from .feature_validation import verify
from .features import missing, sha256
from .structural import run as run_structural
from .structural_validation import verify as verify_structural

POLICY_VERSION = "integrated-review-v4"
STATUSES = ("candidate", "quarantine", "review_hold")
LABEL_RULES = {
    ("Sub-Segment", "Cold / Flu", "Cold/Flu", "slash_spacing"),
    ("Sub-Segment", "Other Lifestyle", "Other Lifestyle CHC", "reviewed_other_lifestyle"),
}


def identity(row):
    return tuple(str(row[column]) for column in KEY)


def target_audit(result, expected_keys):
    """Reject lost, duplicated or foreign identities and unapproved label edits."""
    codes, changes = {}, {}
    for frame in (result.cleaned, result.quarantine):
        for row in frame.to_dict("records"):
            key = identity(row)
            if key in codes:
                raise ValueError("Duplicate target identity")
            codes[key] = sorted(row["Q_REASON"])
    if set(codes) != expected_keys:
        raise ValueError("Target partitions do not match source identities")
    for row in result.changes.to_dict("records"):
        key = identity(row)
        change_key = (key, row["column"])
        rule = tuple(row[name] for name in ("column", "old_value", "new_value", "rule"))
        if key not in codes or change_key in changes or rule not in LABEL_RULES:
            raise ValueError("Unknown, duplicate or unapproved label change")
        if set(codes[key]) & QUARANTINE_REASONS:
            raise ValueError("Label change on a target-quarantined record")
        changes[change_key] = row
    return codes, changes


def load_structural_decisions(path, dataset, digest):
    """Consume a complete keyed decision table, independent of file row order.

    Integration checks exact source coverage by popping each expected key once.
    Raw evidence remains in the structural run, as with feature decision details.
    """
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = identity(row)
            if (key[:2] != (dataset, digest) or not key[2].isdigit()
                    or int(key[2]) < 1 or str(int(key[2])) != key[2] or key in result):
                raise ValueError("Duplicate/invalid structural identity")
            reasons, findings = json.loads(row["reason_codes"]), json.loads(row["finding_codes"])
            if (not isinstance(reasons, list) or not isinstance(findings, list)
                    or any(not isinstance(c, str) for c in reasons + findings)
                    or len(reasons) != len(set(reasons)) or not set(reasons) <= set(findings)
                    or row["quarantine"] not in {"0", "1"}
                    or (row["quarantine"] == "1") != bool(reasons)):
                raise ValueError("Invalid structural decision/reasons")
            result[key] = row
    return result


def row_decision(feature, target_codes, exclude_policy, *, structural):
    """Union quarantine recommendations; explicit holds only, never all flags."""
    if exclude_policy not in {"hold", "keep", "quarantine"}:
        raise ValueError("Unknown Exclude policy")
    reasons = ["feature:" + r for r in feature["reasons"].split(";") if r]
    reasons += ["target:" + str(code) for code in target_codes if code in QUARANTINE_REASONS]
    reasons += ["structural:" + code for code in json.loads(structural["reason_codes"])]
    holds = ["feature:" + code for code in feature["hold_reasons"].split(";") if code]
    if feature["source_exclude_marker"] == "1":
        if exclude_policy == "hold":
            holds.append("source_exclude_policy_override")
        elif exclude_policy == "quarantine":
            reasons.append("source:Exclude")
    status = "quarantine" if reasons else "review_hold" if holds else "candidate"
    return status, reasons, holds


def run(train_path, target_path, output_dir, exclude_policy="keep"):
    sources = {"training": Path(train_path).resolve(), "target": Path(target_path).resolve()}
    output = Path(output_dir).resolve()
    if exclude_policy not in {"hold", "keep", "quarantine"}:
        raise ValueError("Unknown Exclude policy")
    if sources["training"] == sources["target"]:
        raise ValueError("Training and target must be distinct source files")
    for source in sources.values():
        if output.is_relative_to(source.parent) or source.is_relative_to(output):
            raise ValueError("Output must be outside both raw directories and their ancestors")
    output.mkdir(parents=True, exist_ok=False)
    digests = {name: sha256(source) for name, source in sources.items()}
    structural_summary = run_structural(sources["training"], sources["target"], output / "structural")
    structural_check = verify_structural(output / "structural")
    feature_summaries, feature_checks = {}, {}
    for name, source in sources.items():
        folder = output / ("feature_" + name)
        feature_summaries[name] = run_features(source, folder, name)
        feature_checks[name] = verify(source, folder, name)
        if feature_summaries[name]["source_sha256"] != digests[name]:
            raise ValueError("Source changed between stages")

    # Parse without pandas' broad NA vocabulary. Convert only documented missing
    # tokens for computation; export from the original string records below.
    with sources["training"].open(encoding="utf-8-sig", newline="") as handle:
        raw_training = list(csv.DictReader(handle, strict=True))
    if not raw_training:
        raise ValueError("Training source must not be empty")
    frame = pd.DataFrame(raw_training, dtype="string")
    if set(KEY + ["Q_REASON"]) & set(frame.columns):
        raise ValueError("Reserved provenance or decision columns in source")
    for column in frame:
        frame[column] = frame[column].mask(frame[column].map(missing), pd.NA)
    frame["dataset"] = "training"
    frame["source_sha256"] = digests["training"]
    frame["source_row"] = range(1, len(frame) + 1)
    target_result = clean_training(frame)
    expected_keys = {("training", digests["training"], str(i)) for i in range(1, len(frame) + 1)}
    codes, label_changes = target_audit(target_result, expected_keys)
    write_csv(target_result.changes, output / "label_changes.csv")
    write_csv(target_result.flags, output / "target_review_flags.csv")
    write_csv(target_result.hierarchy_violations, output / "target_hierarchy_diagnostics.csv")
    del raw_training, frame

    totals, overlaps, distributions, reason_totals, hold_totals, feature_columns = {}, {}, {}, {}, {}, {}
    audit_fields = KEY + ["JoiningKey", "status", "keep_candidate", "quarantine_reasons",
                          "hold_reasons", "feature_disposition", "feature_flags", "target_reason_codes",
                          "structural_quarantine", "structural_reason_codes", "structural_finding_codes",
                          "changed_feature_columns", "changed_label_columns"]
    with (output / "row_decisions.csv").open("w", encoding="utf-8", newline="") as audit_handle:
        audit = csv.DictWriter(audit_handle, fieldnames=audit_fields)
        audit.writeheader()
        for dataset, source in sources.items():
            counts, overlap = Counter(dict.fromkeys(STATUSES, 0)), Counter()
            quarantine_counts, hold_counts = Counter(), Counter()
            structural_rows = load_structural_decisions(output / "structural" / f"{dataset}_decisions.csv",
                                                       dataset, digests[dataset])
            dist = {stage: {c: Counter() for c in TARGET_COLUMNS}
                    for stage in ("raw",) + STATUSES}
            with ExitStack() as stack:
                def reader(path):
                    return csv.DictReader(stack.enter_context(path.open(encoding="utf-8-sig", newline="")), strict=True)
                raw = reader(source)
                candidate = reader(output / ("feature_" + dataset) / "features_candidate.csv")
                features = reader(output / ("feature_" + dataset) / "feature_decisions.csv")
                fields = raw.fieldnames
                if set(KEY + ["Q_REASON"]) & set(fields) or fields != candidate.fieldnames:
                    raise ValueError("Invalid source/candidate schema")
                feature_columns[dataset] = [c for c in fields if c not in EXCLUDED_MODEL_FIELDS]
                model_export = csv.DictWriter(stack.enter_context(
                    (output / f"{dataset}_features.csv").open("w", encoding="utf-8", newline="")),
                    fieldnames=feature_columns[dataset] + KEY)
                model_export.writeheader()
                writers = {}
                for status in STATUSES:
                    handle = stack.enter_context((output / f"{dataset}_{status}.csv").open("w", encoding="utf-8", newline=""))
                    writers[status] = csv.DictWriter(handle, fieldnames=fields + KEY)
                    writers[status].writeheader()
                for number, triple in enumerate(zip_longest(raw, candidate, features), 1):
                    original, cleaned, feature = triple
                    if any(row is None for row in triple):
                        raise ValueError("Mismatched feature/source row counts")
                    key = (dataset, digests[dataset], str(number))
                    if identity(feature) != key:
                        raise ValueError("Feature identity/order mismatch")
                    if key not in structural_rows:
                        raise ValueError("Missing structural source identity")
                    structural = structural_rows.pop(key)
                    target_codes = codes[key] if dataset == "training" else []
                    changed_labels = []
                    for column in TARGET_COLUMNS:
                        change = label_changes.get((key, column))
                        if change:
                            if original[column] != change["old_value"] or cleaned[column] != original[column]:
                                raise ValueError("Label change does not match original source value")
                            cleaned[column] = change["new_value"]
                            changed_labels.append(column)
                    status, reasons, holds = row_decision(feature, target_codes, exclude_policy, structural=structural)
                    provenance = dict(zip(KEY, key))
                    writers[status].writerow({**cleaned, **provenance})
                    if status == "candidate":
                        model_export.writerow({**{c: cleaned[c] for c in feature_columns[dataset]}, **provenance})
                    audit.writerow({**provenance, "JoiningKey": original["JoiningKey"], "status": status,
                                    "keep_candidate": int(status == "candidate"),
                                    "quarantine_reasons": ";".join(reasons), "hold_reasons": ";".join(holds),
                                    "feature_disposition": feature["disposition"], "feature_flags": feature["flags"],
                                    "target_reason_codes": ";".join(map(str, target_codes)),
                                    "structural_quarantine": structural["quarantine"],
                                    "structural_reason_codes": structural["reason_codes"],
                                    "structural_finding_codes": structural["finding_codes"],
                                    "changed_feature_columns": feature["changed_columns"],
                                    "changed_label_columns": ";".join(changed_labels)})
                    counts[status] += 1
                    feature_q = feature["recommend_quarantine"] == "1"
                    target_q = bool(set(target_codes) & QUARANTINE_REASONS)
                    structural_q = structural["quarantine"] == "1"
                    overlap["+".join(name for name, flag in (("feature", feature_q), ("target", target_q),
                                                             ("structural", structural_q)) if flag) or "none"] += 1
                    quarantine_counts.update(reasons)
                    hold_counts.update(holds)
                    for stage, record in (("raw", original), (status, cleaned)):
                        for column in TARGET_COLUMNS:
                            value = record[column]
                            if value == "<missing>":
                                raise ValueError("Reserved summary label <missing> in source")
                            dist[stage][column]["<missing>" if missing(value) else value] += 1
            if structural_rows or sum(counts.values()) != feature_summaries[dataset]["rows"]:
                raise ValueError("Final partition does not reconcile with source")
            totals[dataset] = {"input": sum(counts.values()), **counts}
            overlaps[dataset] = dict(overlap)
            distributions[dataset] = dist
            reason_totals[dataset], hold_totals[dataset] = dict(quarantine_counts), dict(hold_counts)
    if any(sha256(path) != digests[name] for name, path in sources.items()):
        raise ValueError("Source changed during integration")
    summary = {
        "independent_structural_screening_applied": True,
        "structural_runs": structural_summary, "structural_validation": structural_check,
        "quarantine_reasons_by_dataset": reason_totals, "hold_reasons_by_dataset": hold_totals,
        "feature_exports": {d: {"file": f"{d}_features.csv", "rows": totals[d]["candidate"],
                                "input_columns": cols, "join_only_columns": KEY}
                            for d, cols in feature_columns.items()},
        "excluded_model_input_columns": EXCLUDED_MODEL_FIELDS,
        "policy_version": POLICY_VERSION, "sources": {k: {"file": p.name, "sha256": digests[k]} for k, p in sources.items()},
        "identity": {"columns": KEY, "source_row_base": 1, "record_not_physical_line": True},
        "missing_policy": "Blank/whitespace or case-insensitive null only; preserve literal source strings on export",
        "exclude_policy": exclude_policy, "rows": totals, "structural_overlap": overlaps,
        "label_changes": len(target_result.changes),
        "changes_by_rule": {str(k): int(v) for k, v in target_result.changes["rule"].value_counts().items()},
        "label_distributions": distributions, "feature_runs": feature_summaries, "feature_validation": feature_checks,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "target_screening": "Feature and structural rules; per-source feature-only profiles; target labels remain missing",
        "review_policy": "Field flags alone never hold a row. Missing name/description/contents is structural quarantine. Only explicit policy overrides create review holds.",
        "policy_authority": "User-supplied decision list; group agreement on Exclude keep and timestamp exclusion only",
        "category_level_features": "Deferred to feature engineering; ProductCategory and Retailer preserved",
        "final_mask_agreed": False, "modeling_ready": False,
        "limitations": ["Candidate mask needs team review and independent validation",
                        "Field flags can remain on candidate rows; downstream field handling is not implemented",
                        "Training IQR diagnostics use the full export; evaluate leakage before building model splits",
                        "MDM_InsertDateTime preserved unchanged; date system and timezone unconfirmed"],
        "output_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
