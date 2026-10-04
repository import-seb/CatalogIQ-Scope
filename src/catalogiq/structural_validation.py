"""Independently reconcile structural artifacts against every original CSV record.

Does not rerun rules or apply decisions. Optional feature compatibility uses the
merged feature pass's existing verifier; it never calls either cleaner.
Run: python -m scripts.validate_structural --output-dir data/processed/<run>
"""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


KEY = ("dataset", "source_sha256", "source_row")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_records(stream):
    """Independent CSV iteration matching the canonical DictReader convention."""
    for row in csv.reader(stream, strict=True):
        if row:
            yield row


def verify(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    expected = {f"{role}_{suffix}" for role in ("training", "target")
                for suffix in ("decisions.csv", "evidence.jsonl")}
    require(set(summary["output_sha256"]) == expected, "Unexpected artifact manifest")
    for filename, fingerprint in summary["output_sha256"].items():
        require(digest(output_dir / filename) == fingerprint, f"Artifact hash mismatch: {filename}")
    require(summary["policy_version"] in ("structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6"),
            "Expected structural-v2 through structural-v6; legacy runs need explicit migration")
    require(summary["identity"]["source_row_base"] == 1
            and summary["identity"]["dataset"] == "training|target"
            and summary["identity"]["columns"] == list(KEY), "Wrong record convention")
    require(summary["rows_removed"] == summary["rows_changed"] == 0 and not summary["masks_combined"],
            "Unexpected transformation claim")
    csv.field_size_limit(10_000_000)
    result = {}
    for role in ("training", "target"):
        report = summary["datasets"][role]
        source = Path(report["source_path"])
        require(report["dataset"] == role and source.name == report["source_file"], "Dataset mismatch")
        fingerprint = digest(source)
        require(fingerprint == report["source_sha256"], "Source hash mismatch")
        details = {}
        with (output_dir / f"{role}_evidence.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                detail = json.loads(line)
                position = detail["source_row"]
                require(type(position) is int and position >= 1 and position not in details,
                        "Duplicate/invalid evidence identity")
                details[position] = detail
        counts = Counter({"keep": 0, "quarantine": 0})
        findings, reasons, cross = Counter(), Counter(), {}
        total = 0
        with source.open(encoding="utf-8-sig", newline="") as raw_stream, \
             (output_dir / f"{role}_decisions.csv").open(encoding="utf-8", newline="") as audit_stream:
            columns = next(csv.reader(raw_stream, strict=True))
            raw_reader = source_records(raw_stream)
            for position, (raw, audit) in enumerate(zip_longest(raw_reader, csv.DictReader(audit_stream)), 1):
                require(raw is not None and audit is not None, "Decision coverage mismatch")
                identity = dict(dataset=role, source_sha256=fingerprint, source_row=position)
                require(all(audit[k] == str(identity[k]) for k in KEY), "Decision identity/order mismatch")
                require(audit["quarantine"] in ("0", "1"), "Invalid quarantine boolean")
                quarantine = audit["quarantine"] == "1"
                reason_codes, finding_codes = (json.loads(audit[k]) for k in ("reason_codes", "finding_codes"))
                require(quarantine == bool(reason_codes), "Quarantine/reason mismatch")
                require(set(reason_codes) <= set(finding_codes), "Missing reason evidence")
                require(reason_codes == sorted(set(reason_codes)) and finding_codes == sorted(set(finding_codes)),
                        "Nondeterministic or duplicated codes")
                detail = details.pop(position, None)
                require(bool(detail) == bool(finding_codes), "Evidence coverage mismatch")
                if detail:
                    require(all(detail[k] == identity[k] for k in KEY), "Evidence identity mismatch")
                    require(detail["raw_columns"] == columns and detail["raw_values"] == raw,
                            "Raw values changed in evidence")
                    require(detail["quarantine"] == quarantine and detail["reason_codes"] == reason_codes
                            and detail["finding_codes"] == finding_codes, "Audit/evidence decision mismatch")
                    require(sorted({f["reason_code"] for f in detail["findings"]}) == finding_codes,
                            "Finding details missing")
                    require(sorted({f["reason_code"] for f in detail["findings"] if f["quarantine_rule"]})
                            == reason_codes, "Rule/reason mismatch")
                    raw_map = dict(zip(columns, raw))
                    for finding in detail["findings"]:
                        require(all(finding[k] == identity[k] for k in KEY), "Finding identity mismatch")
                        require(all(raw_map.get(c) == v for c, v in finding["evidence"].get("raw_values", {}).items()),
                                "Finding raw evidence changed")
                decision = "quarantine" if quarantine else "keep"
                counts[decision] += 1
                findings.update(finding_codes)
                reasons.update(reason_codes)
                for code in finding_codes:
                    cross.setdefault(code, {"keep": 0, "quarantine": 0})[decision] += 1
                total += 1
        require(not details, "Orphan evidence rows")
        for key, actual in (("rows", total), ("decisions", dict(counts)), ("rows_by_finding", dict(findings)),
                            ("rows_by_quarantine_reason", dict(reasons)), ("rows_by_finding_and_decision", cross)):
            require(report[key] == actual, f"Summary mismatch: {key}")
        require(digest(source) == fingerprint, "Source changed during verification")
        result[role] = {"verified_rows": total, "decisions": dict(counts),
                        "source_sha256": fingerprint, "all_identities_and_evidence_verified": True}
    return result


def verify_feature_compatibility(output_dir: Path, feature_runs: dict[str, Path]) -> dict:
    """Validate complete keyed compatibility; compare decisions without a union mask.

    The structural engine stays independent. This optional validator consumes the
    merged feature pass's public artifacts and its existing raw-data verifier.
    Neither run nor its cells, dispositions, or masks are modified.
    """
    from catalogiq.feature_validation import verify as verify_features

    require(set(feature_runs) == {"training", "target"}, "Both feature runs are required")
    output_dir = Path(output_dir)
    verified = verify(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    result = {}
    for role in ("training", "target"):
        feature_dir = Path(feature_runs[role])
        report = summary["datasets"][role]
        feature_check = verify_features(Path(report["source_path"]), feature_dir, role)
        require(feature_check["verified_rows"] == verified[role]["verified_rows"], "Feature coverage mismatch")
        overlaps = Counter(dict.fromkeys(("both_quarantine", "structural_only", "feature_only", "neither"), 0))
        with (output_dir / f"{role}_decisions.csv").open(encoding="utf-8", newline="") as structural_stream, \
             (feature_dir / "feature_decisions.csv").open(encoding="utf-8", newline="") as feature_stream:
            for structural, feature in zip_longest(csv.DictReader(structural_stream), csv.DictReader(feature_stream)):
                require(structural is not None and feature is not None, "Feature decision coverage mismatch")
                require(all(structural[k] == feature[k] for k in KEY), "Structural/feature identity mismatch")
                structural_q = structural["quarantine"] == "1"
                feature_q = feature["recommend_quarantine"] == "1"
                overlaps["both_quarantine" if structural_q and feature_q else "structural_only" if structural_q
                         else "feature_only" if feature_q else "neither"] += 1
        result[role] = {"verified_rows": verified[role]["verified_rows"],
                        "source_sha256": report["source_sha256"], "decisions_compared": dict(overlaps),
                        "masks_combined": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-training-run", type=Path, help="Optional completed feature-training run")
    parser.add_argument("--feature-target-run", type=Path, help="Optional completed feature-target run")
    args = parser.parse_args()
    if bool(args.feature_training_run) != bool(args.feature_target_run):
        parser.error("Both --feature-training-run and --feature-target-run are required for comparison")
    report = (verify_feature_compatibility(args.output_dir, {"training": args.feature_training_run,
                                                           "target": args.feature_target_run})
              if args.feature_training_run else verify(args.output_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
