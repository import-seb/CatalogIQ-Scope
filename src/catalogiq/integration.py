"""Join feature and target decisions by source identity, preserving raw strings.

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
from .feature_decisions import KEY, run as run_features
from .feature_validation import verify
from .features import missing, sha256

POLICY_VERSION = "integrated-review-v1"
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


def row_decision(feature, target_codes, exclude_policy):
    if exclude_policy not in {"hold", "keep", "quarantine"}:
        raise ValueError("Unknown Exclude policy")
    reasons = ["feature:" + r for r in feature["reasons"].split(";") if r]
    reasons += ["target:" + str(code) for code in target_codes if code in QUARANTINE_REASONS]
    holds = []
    if feature["source_exclude_marker"] == "1":
        if exclude_policy == "hold":
            holds.append("source_exclude_policy_unconfirmed")
        elif exclude_policy == "quarantine":
            reasons.append("source:Exclude")
    if feature["requires_review"] == "1" and feature["recommend_quarantine"] != "1":
        holds.append("feature_field_review")
    status = "quarantine" if reasons else "review_hold" if holds else "candidate"
    return status, reasons, holds


def run(train_path, target_path, output_dir, exclude_policy="hold"):
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

    totals, overlaps, distributions = {}, {}, {}
    audit_fields = KEY + ["JoiningKey", "status", "keep_candidate", "quarantine_reasons",
                          "hold_reasons", "feature_disposition", "feature_flags", "target_reason_codes",
                          "changed_feature_columns", "changed_label_columns"]
    with (output / "row_decisions.csv").open("w", encoding="utf-8", newline="") as audit_handle:
        audit = csv.DictWriter(audit_handle, fieldnames=audit_fields)
        audit.writeheader()
        for dataset, source in sources.items():
            counts, overlap = Counter(dict.fromkeys(STATUSES, 0)), Counter()
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
                    target_codes = codes[key] if dataset == "training" else []
                    changed_labels = []
                    for column in TARGET_COLUMNS:
                        change = label_changes.get((key, column))
                        if change:
                            if original[column] != change["old_value"] or cleaned[column] != original[column]:
                                raise ValueError("Label change does not match original source value")
                            cleaned[column] = change["new_value"]
                            changed_labels.append(column)
                    status, reasons, holds = row_decision(feature, target_codes, exclude_policy)
                    provenance = dict(zip(KEY, key))
                    writers[status].writerow({**cleaned, **provenance})
                    audit.writerow({**provenance, "JoiningKey": original["JoiningKey"], "status": status,
                                    "keep_candidate": int(status == "candidate"),
                                    "quarantine_reasons": ";".join(reasons), "hold_reasons": ";".join(holds),
                                    "feature_disposition": feature["disposition"], "feature_flags": feature["flags"],
                                    "target_reason_codes": ";".join(map(str, target_codes)),
                                    "changed_feature_columns": feature["changed_columns"],
                                    "changed_label_columns": ";".join(changed_labels)})
                    counts[status] += 1
                    feature_q = feature["recommend_quarantine"] == "1"
                    target_q = bool(set(target_codes) & QUARANTINE_REASONS)
                    overlap["both" if feature_q and target_q else "feature_only" if feature_q
                            else "target_only" if target_q else "neither"] += 1
                    for stage, record in (("raw", original), (status, cleaned)):
                        for column in TARGET_COLUMNS:
                            value = record[column]
                            if value == "<missing>":
                                raise ValueError("Reserved summary label <missing> in source")
                            dist[stage][column]["<missing>" if missing(value) else value] += 1
            if sum(counts.values()) != feature_summaries[dataset]["rows"]:
                raise ValueError("Final partition does not reconcile with source")
            totals[dataset] = {"input": sum(counts.values()), **counts}
            overlaps[dataset] = dict(overlap)
            distributions[dataset] = dist
    if any(sha256(path) != digests[name] for name, path in sources.items()):
        raise ValueError("Source changed during integration")
    summary = {
        "policy_version": POLICY_VERSION, "sources": {k: {"file": p.name, "sha256": digests[k]} for k, p in sources.items()},
        "identity": {"columns": KEY, "source_row_base": 1, "record_not_physical_line": True},
        "missing_policy": "Blank/whitespace or case-insensitive null only; preserve literal source strings on export",
        "exclude_policy": exclude_policy, "rows": totals, "structural_overlap": overlaps,
        "label_changes": len(target_result.changes),
        "changes_by_rule": {str(k): int(v) for k, v in target_result.changes["rule"].value_counts().items()},
        "label_distributions": distributions, "feature_runs": feature_summaries, "feature_validation": feature_checks,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "target_screening": "Feature rules only; training label/empirical IQR rules are not fitted on prediction data",
        "review_policy": "Unresolved feature fields hold whole rows conservatively; target reasons 2/3 remain review-only",
        "final_mask_agreed": False, "modeling_ready": False,
        "limitations": ["Candidate mask needs team review and independent validation",
                        "Exclude meaning and field-review policy remain unconfirmed",
                        "Training IQR diagnostics use the full export; evaluate leakage before building model splits",
                        "MDM_InsertDateTime preserved unchanged; date system and timezone unconfirmed"],
        "output_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
