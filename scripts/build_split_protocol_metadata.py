"""Maintainer utility: extract portable metadata from the October 9 seal.

This preserves historical decisions; it never chooses a test or changes rules.
Only source-row selectors, hashes, configuration and a public tokenizer are
exported. Product text, labels, assignments and weights are not exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.splitting import GROUP_FIELDS
from catalogiq.splitting_experiment import load_input


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(frame, fields):
    """SHA256 of compact UTF-8 JSON rows + LF, sorted by stable record ID."""
    digest = hashlib.sha256()
    selected = frame.sort_values("record_id", kind="stable").loc[:, fields]
    for values in selected.itertuples(index=False, name=None):
        digest.update(json.dumps(list(values), ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_csv(path, columns=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", usecols=columns)


def write_json(path, values, *, compact=False):
    path.write_text(json.dumps(values, ensure_ascii=False, allow_nan=False,
        separators=(",", ":") if compact else None, indent=None if compact else 2) + "\n",
        encoding="utf8", newline="\n")


def build_metadata(input_path, frozen, exposure, tokenizer, output, license_path):
    frozen, exposure, tokenizer, output = map(Path, (frozen, exposure, tokenizer, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty resource directory; do not overwrite an existing protocol")
    freeze = json.loads((frozen / "grouping_freeze.json").read_text(encoding="utf8"))
    completion = json.loads((frozen / "completion.json").read_text(encoding="utf8"))
    for name in ("assignments.csv", "grouping_assignments.csv", "protected_final_test.csv"):
        if file_digest(frozen / name) != completion["artifact_sha256"][name]:
            raise ValueError(f"Historical sealed identity artifact changed: {name}")
    frame, input_manifest = load_input(Path(input_path))
    # Do not inspect or export final individual labels. Only development labels
    # enter the label digest; target-free features enter the grouping digest.
    assignments = read_csv(frozen / "assignments.csv", [*PROVENANCE, "record_id", "group_id", "split"])
    grouping = read_csv(frozen / "grouping_assignments.csv")
    protected = read_csv(frozen / "protected_final_test.csv", ["record_id"])
    if set(frame.record_id) != set(assignments.record_id):
        raise ValueError("Input population differs from historical frozen assignments")
    source = assignments[["dataset", "source_sha256"]].drop_duplicates()
    if len(source) != 1:
        raise ValueError("This metadata schema requires the one historical training source")
    identity = source.iloc[0].to_dict()
    for column in PROVENANCE:
        actual = frame.set_index("record_id")[column].sort_index()
        expected = assignments.set_index("record_id")[column].sort_index()
        if not actual.equals(expected):
            raise ValueError(f"Historical provenance differs: {column}")
    old = read_csv(exposure / "old_test_eligibility.csv")
    strict = read_csv(exposure / "eligible_strict_historical_family_record_ids.csv", ["record_id"])
    groups = {
        "original_test": set(old.record_id),
        "strict_eligible": set(strict.record_id),
        "final_test": set(protected.record_id),
    }
    for flag in ("direct_review_exposed", "review_family_exposed", "direct_model_seen", "model_family_exposed"):
        groups["old_test_" + flag] = set(old.loc[old[flag].str.casefold().eq("true"), "record_id"])
    if not groups["final_test"] <= groups["strict_eligible"] <= groups["original_test"] <= set(frame.record_id):
        raise ValueError("Historical test/eligibility subset chain is invalid")
    selections = {}
    for name, ids in groups.items():
        records = assignments.loc[assignments.record_id.isin(ids)]
        if len(records) != len(ids):
            raise ValueError(f"Unrecognized selector record: {name}")
        selections[name] = {"count": len(records),
            "source_rows": sorted(records.source_row.astype(int).tolist()),
            "record_ids_sha256": canonical_digest(records, ["record_id"])}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "eligibility.json", {"schema_version": 1,
        "source_identity": identity, "selectors": selections}, compact=True)
    (output / "tokenizer").mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        expected = freeze["model_view_contract"]["tokenizer_fingerprint"][name]
        if file_digest(tokenizer / name) != expected:
            raise ValueError(f"Historical tokenizer bytes differ: {name}")
        shutil.copyfile(tokenizer / name, output / "tokenizer" / name)
    # Reproduce the standard Apache license text, excluding an unrelated
    # installed package's leading copyright notice.
    license_text = Path(license_path).read_text(encoding="utf8")
    start = license_text.find("                                 Apache License")
    if start < 0 or "Version 2.0, January 2004" not in license_text:
        raise ValueError("Supply the standard Apache License version 2.0 text")
    (output / "LICENSE").write_text(license_text[start:].replace("\r\n", "\n"),
                                   encoding="utf8", newline="\n")
    notice = """Public tokenizer attribution

Model: sentence-transformers/all-MiniLM-L6-v2
Upstream revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf
Upstream model card:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md

The upstream model card declares Apache-2.0. The license is included in LICENSE.
The two tokenizer files are byte-identical to the local tokenizer saved for the
October 9, 2026 Segment baseline (Transformers 5.3.0), including saved settings.
They contain the public WordPiece vocabulary and tokenizer configuration.
No CatalogIQ product data, target labels, model weights or predictions are
included. These assets make splitting offline and independent of remote caches.
"""
    (output / "NOTICE").write_text(notice, encoding="utf8", newline="\n")
    module_names = (
        "features.py", "cleaning.py", "splitting.py", "split_matching.py",
        "split_rules_v2.py", "split_rules_v3.py", "split_rules_v4.py",
        "split_rule_evidence_v3.py", "split_rule_candidates_v3.py", "split_rule_components.py",
        "segment_development_allocation.py", "segment_transformer.py", "model_input_groups.py",
    )
    module_dir = Path(__file__).resolve().parents[1] / "src/catalogiq"
    code_hashes = {}
    for name in module_names:
        path = module_dir / name
        original = next((expected for file, expected in freeze["source_sha256"].items()
                         if Path(file).name == name), None)
        if original is None:
            raise ValueError(f"Algorithm module absent from historical seal: {name}")
        raw = path.read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        candidates = (raw, normalized, normalized.replace(b"\n", b"\r\n"))
        if original not in {hashlib.sha256(value).hexdigest() for value in candidates}:
            raise ValueError(f"Algorithm module differs from historical seal: {name}")
        code_hashes[name] = hashlib.sha256(normalized).hexdigest()
    for column in GROUP_FIELDS:
        if column not in frame:
            raise ValueError(f"Historical input lacks required grouping feature: {column}")
    development = frame.loc[~frame.record_id.isin(groups["final_test"])]
    expected = {
        "assignment_sha256": canonical_digest(assignments, ["record_id", "group_id", "split"]),
        "product_group_sha256": canonical_digest(grouping, ["record_id", "product_group_id"]),
        "effective_input_group_sha256": canonical_digest(grouping, ["record_id", "effective_input_group_id"]),
        "group_sha256": canonical_digest(grouping, ["record_id", "group_id"]),
        "splits": {},
        "test_reservation": json.loads((frozen / "test_reservation.json").read_text(encoding="utf8")),
    }
    # Timestamps and machine-specific freeze references are historical evidence,
    # not portable allocation parameters.
    expected["test_reservation"].pop("reserved_utc", None)
    expected["test_reservation"].pop("grouping_freeze_sha256", None)
    for role in ("train", "validation", "test"):
        records = assignments.loc[assignments.split.eq(role)]
        expected["splits"][role] = {"records": len(records), "groups": int(records.group_id.nunique()),
            "record_ids_sha256": canonical_digest(records, ["record_id"]),
            "assignment_sha256": canonical_digest(records, ["record_id", "group_id", "split"])}
    manifest = {"schema_version": 1, "protocol_id": "segment-final-20261009",
        "grouping_version": "rule-final-v4", "split_version": "segment-final-splits-v1",
        "source": {**identity, "input_rows": len(frame),
            "record_ids_sha256": canonical_digest(frame, ["record_id"]),
            "provenance_sha256": canonical_digest(frame, PROVENANCE),
            "group_fields": list(GROUP_FIELDS),
            "group_features_sha256": canonical_digest(frame, ["record_id", *GROUP_FIELDS]),
            "development_labels_sha256": canonical_digest(development, ["record_id", "Segment"]),
            "development_rows": len(development),
            "historical_cleaned_input_sha256": input_manifest["sha256"]},
        **{name: freeze[name] for name in ("split_config", "rule_final_config",
           "development_allocation_config", "model_config", "model_view_contract")},
        "algorithm_source_sha256_lf": code_hashes,
        "historical_snapshot": {
            "assignments_csv_sha256": completion["artifact_sha256"]["assignments.csv"],
            "grouping_freeze_sha256": file_digest(frozen / "grouping_freeze.json"),
            "completion_sha256": file_digest(frozen / "completion.json"),
            "grouping_assignments_csv_sha256": completion["artifact_sha256"]["grouping_assignments.csv"],
            "protected_final_test_csv_sha256": completion["artifact_sha256"]["protected_final_test.csv"],
            "segment_distributions_csv_sha256": completion["artifact_sha256"]["segment_distributions.csv"]},
        "exposure_provenance": {"policy": json.loads((exposure / "policy.json").read_text(encoding="utf8")),
            "registry_artifact_sha256": {path.name: file_digest(path) for path in sorted(exposure.iterdir()) if path.is_file()},
            "meaning": "Selectors reproduce the recorded strict historical-family eligibility decision. Full local historical artifacts are not required on teammate machines."},
        "expected": expected,
        "canonicalization": {
            "version": "compact-json-lines-record-id-sorted-v1", "encoding": "UTF-8",
            "order": "ascending record_id; stable sort; all CSV values retained as strings",
            "row_encoding": "json.dumps(array,ensure_ascii=False,separators=(',',':'),allow_nan=False) + LF",
            "record_id_fields": ["dataset", "source_sha256", "source_row"],
            "record_ids_digest_fields": ["record_id"], "provenance_digest_fields": PROVENANCE,
            "group_features_digest_fields": ["record_id", *GROUP_FIELDS],
            "development_labels_digest_fields": ["record_id", "Segment"],
            "assignment_digest_fields": ["record_id", "group_id", "split"],
            "development_labels_scope": "All records outside the frozen actual final-test selector; no final-test individual labels are hashed or exported.",
            "source_hash_normalization": "Replace CRLF with LF before SHA256; retain other bytes."},
        "test_protection": {
            "rule": "Training/validation must match sealed role membership and exclude every final-test ID. No final-test scoring or grouping refinement during development.",
            "representation": "The text fields, character limits, tokenizer bytes and sequence limit are frozen; representation changes require a new exposure-aware protocol."},
        "resources_sha256": {str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
                              for path in sorted(output.rglob("*")) if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    return {"records": len(frame), "selectors": {key: value["count"] for key, value in selections.items()},
            "assignment_sha256": expected["assignment_sha256"],
            "resources_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apache-license", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_metadata(args.input, args.frozen_dir, args.exposure_dir,
        args.tokenizer_dir, args.output_dir, args.apache_license), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
