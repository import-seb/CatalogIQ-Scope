"""Reconcile integrated exports with every source record without rerunning cleaners."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
import json
from pathlib import Path

from .feature_decisions import EXCLUDED_MODEL_FIELDS, KEY
from .feature_validation import require, verify as verify_features
from .features import missing, sha256
from .structural_validation import verify as verify_structural


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, strict=True)


def identity(row):
    return tuple(row[c] for c in KEY)


def indexed(path):
    result = {}
    for row in records(path):
        require(None not in row and None not in row.values(), "Malformed output row")
        key = identity(row)
        require(key not in result, "Duplicate output identity")
        result[key] = row
    return result


def verify(train_path, target_path, output_dir):
    """Check hashes, union policy, complete partitions, projected values and joins."""
    csv.field_size_limit(10_000_000)
    output = Path(output_dir)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    require(summary["policy_version"] == "integrated-review-v4", "Unsupported integrated policy")
    require(summary["independent_structural_screening_applied"] is True, "Structural screening missing")
    for filename, digest in summary["output_sha256"].items():
        require(sha256(output / filename) == digest, f"Output hash mismatch: {filename}")
    structural_check = verify_structural(output / "structural")
    audit = indexed(output / "row_decisions.csv")
    labels = {}
    allowed_changes = {("Sub-Segment", "Cold / Flu", "Cold/Flu", "slash_spacing"),
                       ("Sub-Segment", "Other Lifestyle", "Other Lifestyle CHC", "reviewed_other_lifestyle")}
    for change in records(output / "label_changes.csv"):
        key = (identity(change), change["column"])
        require(key not in labels and key[0][0] == "training", "Invalid label change identity")
        require(tuple(change[c] for c in ("column", "old_value", "new_value", "rule")) in allowed_changes,
                "Unapproved label change")
        labels[key] = change
    target_codes = defaultdict(set)
    for flag in records(output / "target_review_flags.csv"):
        require(flag["reason_code"] in {"2", "3", "6"}, "Retired/unknown target reason")
        target_codes[identity(flag)].add(flag["reason_code"])
    report = {}
    for dataset, path in (("training", train_path), ("target", target_path)):
        source = Path(path)
        digest = sha256(source)
        require(digest == summary["sources"][dataset]["sha256"], "Source hash mismatch")
        feature_check = verify_features(source, output / f"feature_{dataset}", dataset)
        structural = indexed(output / "structural" / f"{dataset}_decisions.csv")
        partitions = {}
        counts, reason_counts, hold_counts = Counter(dict.fromkeys(("candidate", "quarantine", "review_hold"), 0)), Counter(), Counter()
        for status in counts:
            for key, row in indexed(output / f"{dataset}_{status}.csv").items():
                require(key not in partitions, "Overlapping partitions")
                partitions[key] = (status, row)
        model_rows = indexed(output / f"{dataset}_features.csv")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            source_fields = next(csv.reader(stream))
        input_fields = [c for c in source_fields if c not in EXCLUDED_MODEL_FIELDS]
        with (output / f"{dataset}_features.csv").open(encoding="utf-8", newline="") as stream:
            require(next(csv.reader(stream)) == input_fields + KEY, "Feature export contains forbidden/missing columns")
        total = 0
        for number, triple in enumerate(zip_longest(records(source),
                records(output / f"feature_{dataset}/features_candidate.csv"),
                records(output / f"feature_{dataset}/feature_decisions.csv")), 1):
            raw, cleaned, feature = triple
            require(all(r is not None for r in triple), "Source/feature coverage mismatch")
            key = (dataset, digest, str(number))
            require(identity(feature) == key and key in audit and key in structural and key in partitions,
                    "Missing/foreign source identity")
            decision, structure = audit.pop(key), structural.pop(key)
            codes = target_codes.pop(key, set())
            require(set(filter(None, decision["target_reason_codes"].split(";"))) == codes, "Target reasons mismatch")
            reasons = ["feature:" + c for c in feature["reasons"].split(";") if c]
            reasons += ["target:6"] if "6" in codes else []
            reasons += ["structural:" + c for c in json.loads(structure["reason_codes"])]
            holds = []
            empty_text = all(missing(raw[c]) for c in ("ProductName", "ProductDescription", "ProductContents"))
            require(("missing_product_text" in json.loads(structure["reason_codes"])) == empty_text,
                    "Missing product text quarantine mismatch")
            if raw["Exclude"].strip() == "Exclude":
                if summary["exclude_policy"] == "hold":
                    holds.append("source_exclude_policy_override")
                elif summary["exclude_policy"] == "quarantine":
                    reasons.append("source:Exclude")
            status = "quarantine" if reasons else "review_hold" if holds else "candidate"
            require(decision["status"] == status and decision["keep_candidate"] == str(int(status == "candidate")),
                    "Combined policy mismatch")
            require(decision["quarantine_reasons"] == ";".join(reasons) and decision["hold_reasons"] == ";".join(holds),
                    "Combined reasons mismatch")
            for column, source_column in (("structural_quarantine", "quarantine"),
                    ("structural_reason_codes", "reason_codes"), ("structural_finding_codes", "finding_codes")):
                require(decision[column] == structure[source_column], "Structural audit mismatch")
            require(decision["feature_flags"] == feature["flags"] and
                    decision["feature_disposition"] == feature["disposition"], "Feature audit mismatch")
            changed_labels = []
            for column in source_fields:
                change = labels.pop((key, column), None)
                if change:
                    require(raw[column] == change["old_value"] and cleaned[column] == raw[column], "Label source mismatch")
                    cleaned[column] = change["new_value"]
                    changed_labels.append(column)
            require(set(filter(None, decision["changed_label_columns"].split(";"))) == set(changed_labels),
                    "Label change log mismatch")
            actual_status, exported = partitions.pop(key)
            expected = {**cleaned, **dict(zip(KEY, key))}
            require(actual_status == status and exported == expected, "Partition value/status mismatch")
            if status == "candidate":
                require(key in model_rows and model_rows.pop(key) == {c: expected[c] for c in input_fields + KEY},
                        "Feature export value/coverage mismatch")
            counts[status] += 1
            reason_counts.update(reasons)
            hold_counts.update(holds)
            total += 1
        require(not structural and not partitions and not model_rows, "Extra/unmatched output rows")
        require(summary["rows"][dataset] == {"input": total, **counts}, "Partition counts mismatch")
        require(summary["quarantine_reasons_by_dataset"][dataset] == dict(reason_counts), "Reason counts mismatch")
        require(summary["hold_reasons_by_dataset"][dataset] == dict(hold_counts), "Hold counts mismatch")
        require(sha256(source) == digest, "Source changed during validation")
        report[dataset] = {"verified_rows": total, "counts": dict(counts), "source_sha256": digest,
                           "feature_validation": feature_check, "structural_validation": structural_check[dataset],
                           "partitions_and_feature_exports_verified": True}
    require(not audit and not labels and not target_codes, "Extra audit/label identities")
    return report


def main():
    from .paths import TRAIN_PATH, TARGET_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--target", type=Path, default=TARGET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.train, args.target, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
