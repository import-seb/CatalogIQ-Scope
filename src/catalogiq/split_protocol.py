"""Portable reproduction and verification of the October 9 Segment protocol.

The registered protocol is immutable. This module connects unchanged product
grouping, exact-input constraints, exposure reservation, and allocation to the
public command. Canonical membership hashes reject plausible but different runs.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .model_input_groups import exact_model_input_groups, merge_model_input_groups
from .segment_development_allocation import AllocationConfig
from .segment_transformer import ModelConfig
from .split_evaluation import _exact_metrics, leakage_metrics, near_duplicate_pairs, segment_distribution
from .split_finalization import allocate_remaining, reserve_unexposed_groups
from .split_rules_v4 import RuleFinalConfig, refine_rule_groups_v4
from .splitting import GROUP_FIELDS, SPLITS, SplitConfig, check_assignments, record_ids
from .splitting_experiment import _edges_frame, _write_json, load_input

FORMAT = "catalogiq-portable-split-v1"
INTEGRATION_VERSION = "portable-segment-split-integration-v1"
DEFAULT_PROTOCOL = Path(__file__).parent / "resources/segment_final_20261009"
ASSIGNMENT_FIELDS = [*PROVENANCE, "record_id", "group_id", "split"]


def canonical_digest(frame, fields):
    """Hash compact UTF-8 JSON arrays plus LF in stable record-ID order."""
    fields = list(fields)
    if "record_id" not in frame or frame.record_id.duplicated().any():
        raise ValueError("Canonical digests require unique stable record IDs")
    ordered = frame.sort_values("record_id", kind="stable").reindex(columns=fields).fillna("")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        payload = json.dumps([str(value) for value in row], ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf8")
        digest.update(payload + b"\n")
    return digest.hexdigest()


def _read(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=usecols)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _within(folder, relative):
    path = (folder / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(folder.resolve()):
        raise ValueError("Artifact paths must be relative and remain inside their directory")
    return path


def _source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_protocol(protocol_dir=None):
    directory = Path(protocol_dir or DEFAULT_PROTOCOL).resolve()
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported frozen split protocol schema")
    if manifest.get("grouping_version") != "rule-final-v4" or manifest.get("split_version") != "segment-final-splits-v1":
        raise ValueError("The official command requires the finalized v4 grouping and split protocol")
    for name, expected in manifest["resources_sha256"].items():
        if sha256(_within(directory, name)) != expected:
            raise ValueError(f"Frozen protocol resource changed: {name}")
    source_dir = Path(__file__).parent
    for name, expected in manifest["algorithm_source_sha256_lf"].items():
        if _source_digest(_within(source_dir, name)) != expected:
            raise ValueError(f"Frozen algorithm source changed: {name}")
    return directory, manifest


class OfflineTokenizer:
    """Use the byte-pinned public tokenizer without model weights or Transformers."""
    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def __call__(self, texts, *, truncation=True, max_length=128, padding=False,
                 return_attention_mask=True):
        if not truncation or padding or not return_attention_mask:
            raise ValueError("Frozen tokenization requires truncation, no padding, and attention masks")
        self.backend.enable_truncation(max_length=max_length, stride=0,
                                       strategy="longest_first", direction="right")
        self.backend.no_padding()
        encoded = self.backend.encode_batch(list(texts), add_special_tokens=True)
        return {"input_ids": [row.ids for row in encoded],
                "attention_mask": [row.attention_mask for row in encoded],
                "token_type_ids": [row.type_ids for row in encoded]}


def portable_model_view_contract(config=None, tokenizer_dir=None):
    """Semantic representation identity with portable, LF-normalized code hashes."""
    config = config or ModelConfig()
    directory = Path(tokenizer_dir or DEFAULT_PROTOCOL / "tokenizer")
    # Reuse the original representation contract; normalize source identities only.
    from .model_input_groups import model_view_contract
    contract = model_view_contract(config, tokenizer_dir=directory)
    contract.pop("contract_sha256")
    source = Path(__file__).parent
    contract["preprocessor_source_sha256"] = _source_digest(source / "segment_transformer.py")
    contract["signature_implementation_source_sha256"] = _source_digest(source / "model_input_groups.py")
    contract["source_hash_normalization"] = "CRLF to LF; no other changes"
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True,
        separators=(",", ":")).encode("utf8")).hexdigest()
    return contract


def _selectors(directory, manifest, frame):
    policy = _json(directory / "eligibility.json")
    source = policy["source_identity"]
    if (source["dataset"] != manifest["source"]["dataset"]
            or source["source_sha256"] != manifest["source"]["source_sha256"]):
        raise ValueError("Exposure policy uses a different source dataset identity")
    by_row = frame.set_index("source_row").record_id
    selectors = {}
    for name, specification in policy["selectors"].items():
        rows = [str(row) for row in specification["source_rows"]]
        if len(set(rows)) != len(rows) or len(rows) != specification["count"]:
            raise ValueError(f"Invalid frozen exposure selector: {name}")
        if not set(rows) <= set(by_row.index):
            raise ValueError(f"Frozen exposure selector missing from input: {name}")
        selected = frame.loc[frame.source_row.isin(rows)]
        if canonical_digest(selected, ["record_id"]) != specification["record_ids_sha256"]:
            raise ValueError(f"Frozen exposure selector identities changed: {name}")
        selectors[name] = set(selected.record_id)
    if not selectors["final_test"] <= selectors["strict_eligible"] <= selectors["original_test"]:
        raise ValueError("Final-test and exposure selector populations are inconsistent")
    return selectors


def _validate_input(frame, manifest):
    specification = manifest["source"]
    ids = record_ids(frame)
    if frame.record_id.tolist() != ids.tolist():
        raise ValueError("Input record_id values conflict with their stable source provenance")
    if (len(frame) != specification["input_rows"]
            or not frame.dataset.eq(specification["dataset"]).all()
            or not frame.source_sha256.eq(specification["source_sha256"]).all()):
        raise ValueError("Input does not match the complete frozen source identity/universe; run integrated cleaning on the registered source data")
    if canonical_digest(frame, ["record_id"]) != specification["record_ids_sha256"]:
        raise ValueError("Input stable row universe differs from the finalized protocol")
    if canonical_digest(frame, PROVENANCE) != specification["provenance_sha256"]:
        raise ValueError("Input source-row provenance differs from the finalized protocol")
    if list(GROUP_FIELDS) != specification["group_fields"]:
        raise ValueError("Grouping feature allowlist changed")
    if canonical_digest(frame, ["record_id", *GROUP_FIELDS]) != specification["group_features_sha256"]:
        raise ValueError("Input grouping/model features differ from the finalized protocol")


def _check_expected(assignment, manifest):
    check_assignments(assignment, assignment.record_id)
    expected = manifest["expected"]
    if canonical_digest(assignment, ["record_id", "group_id", "split"]) != expected["assignment_sha256"]:
        raise ValueError("Regenerated split/group membership differs from October 9; no equivalent split will be claimed")
    if canonical_digest(assignment, ["record_id", "group_id"]) != expected["group_sha256"]:
        raise ValueError("Product groups differ from the authoritative version")
    for role in SPLITS:
        rows = assignment.loc[assignment.split.eq(role)]
        specification = expected["splits"][role]
        if (len(rows) != specification["records"] or rows.group_id.nunique() != specification["groups"]
                or canonical_digest(rows, ["record_id"]) != specification["record_ids_sha256"]
                or canonical_digest(rows, ["record_id", "group_id", "split"]) != specification["assignment_sha256"]):
            raise ValueError(f"Authoritative {role} membership verification failed")


def verify_protocol_splits(directory, *, verify_exports=True, protocol_dir=None):
    """Verify against repository truth, even if somebody recomputed an output seal.

    Development callers skip the test export entirely. Assignments and reserved
    identity files are always read without target/feature columns.
    """
    directory = Path(directory).resolve()
    bundle, manifest = load_protocol(protocol_dir)
    completion = _json(directory / "completion.json")
    required = {"assignments.csv", "rule_assignments.csv", "protected_final_test.csv",
        "grouping_assignments.csv", "model_inputs.csv", "grouping_freeze.json",
        "protocol_manifest.json", "eligibility.json", "model_config.json", "split_config.json",
        "rule_final_config.json", "development_allocation_config.json", "input_manifest.json",
        "summary.json", "test_reservation.json", "train.csv", "validation.csv", "test.csv",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if completion.get("format") != FORMAT or not required <= set(completion["artifact_sha256"]):
        raise ValueError("Incomplete portable split seal")
    if completion["protocol_manifest_sha256"] != sha256(bundle / "manifest.json"):
        raise ValueError("Generated splits use a different authoritative protocol manifest")
    for name, expected in completion["artifact_sha256"].items():
        if name == "test.csv" and not verify_exports:
            continue
        if sha256(_within(directory, name)) != expected:
            raise ValueError(f"Sealed split artifact changed: {name}")
    if sha256(directory / "protocol_manifest.json") != sha256(bundle / "manifest.json"):
        raise ValueError("Output protocol metadata differs from the committed source of truth")
    if sha256(directory / "eligibility.json") != sha256(bundle / "eligibility.json"):
        raise ValueError("Output exposure exclusions differ from the committed source of truth")
    for key in ("split_config", "rule_final_config", "development_allocation_config", "model_config"):
        if _json(directory / f"{key}.json") != manifest[key]:
            raise ValueError(f"Frozen output configuration changed: {key}")
    assignment = _read(directory / "assignments.csv", ASSIGNMENT_FIELDS)
    _validate_provenance = assignment.copy()
    if _validate_provenance.record_id.tolist() != record_ids(_validate_provenance).tolist():
        raise ValueError("Assignment stable IDs do not match their provenance")
    if (canonical_digest(assignment, ["record_id"]) != manifest["source"]["record_ids_sha256"]
            or canonical_digest(assignment, PROVENANCE) != manifest["source"]["provenance_sha256"]):
        raise ValueError("Assignment source universe/provenance changed")
    _check_expected(assignment, manifest)
    grouping = _read(directory / "grouping_assignments.csv")
    if (set(grouping.record_id) != set(assignment.record_id)
            or canonical_digest(grouping, ["record_id", "product_group_id"]) != manifest["expected"]["product_group_sha256"]
            or canonical_digest(grouping, ["record_id", "effective_input_group_id"]) != manifest["expected"]["effective_input_group_sha256"]
            or canonical_digest(grouping, ["record_id", "group_id"]) != manifest["expected"]["group_sha256"]):
        raise ValueError("Frozen grouping memberships changed")
    if sha256(directory / "rule_assignments.csv") != sha256(directory / "assignments.csv"):
        raise ValueError("rule_assignments.csv must be the identical final assignment alias")
    selectors = _selectors(bundle, manifest, assignment)
    actual_test = set(assignment.loc[assignment.split.eq("test"), "record_id"])
    if actual_test != selectors["final_test"] or not actual_test <= selectors["strict_eligible"]:
        raise ValueError("Final-test membership or frozen exposure exclusions changed")
    protected = _read(directory / "protected_final_test.csv", ["record_id", "group_id", "split"])
    if protected.record_id.duplicated().any() or not protected.split.eq("test").all():
        raise ValueError("Invalid protected final-test identities")
    if protected.set_index("record_id").group_id.to_dict() != assignment.loc[assignment.split.eq("test")].set_index("record_id").group_id.to_dict():
        raise ValueError("Protected final-test identity/group mapping changed")
    inputs = _read(directory / "model_inputs.csv", ["record_id", "effective_input_group_id"])
    if (inputs.record_id.duplicated().any()
            or set(inputs.record_id) != set(assignment.record_id)
            or canonical_digest(inputs, ["record_id", "effective_input_group_id"]) != manifest["expected"]["effective_input_group_sha256"]):
        raise ValueError("Frozen effective-input signatures changed")
    inputs["split"] = inputs.record_id.map(assignment.set_index("record_id").split)
    if inputs.groupby("effective_input_group_id").split.nunique().gt(1).any():
        raise ValueError("Identical transformer inputs cross split boundaries")
    model_config = ModelConfig.from_dict(manifest["model_config"])
    contract = portable_model_view_contract(model_config, directory / "tokenizer")
    if contract != portable_model_view_contract(model_config, bundle / "tokenizer"):
        raise ValueError("Generated tokenizer/model representation differs from the frozen protocol")
    freeze = _json(directory / "grouping_freeze.json")
    if freeze["format"] != FORMAT or freeze["model_view_contract"] != contract:
        raise ValueError("Generated grouping freeze/model contract changed")
    for role in SPLITS:
        if role == "test" and not verify_exports:
            continue
        # Verification never reads any final-test features or labels.
        exported = _read(directory / f"{role}.csv", ASSIGNMENT_FIELDS)
        expected = assignment.loc[assignment.split.eq(role), ASSIGNMENT_FIELDS]
        pd.testing.assert_frame_equal(exported.reset_index(drop=True), expected.reset_index(drop=True))
    return _json(directory / "summary.json")


def export_final_splits(input_path, output_dir, *, protocol_dir=None,
                        identifier_path=None, progress=print):
    started = perf_counter()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new directory and preserve historical runs")
    bundle, manifest = load_protocol(protocol_dir)
    source_hash = sha256(input_path)
    input_columns = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if set(input_columns) & {"group_id", "split"}:
        raise ValueError("Use cleaned input without existing group_id or split columns")
    if "record_id" in input_columns:
        original = _read(input_path, ["record_id", *PROVENANCE])
        if original.record_id.tolist() != record_ids(original).tolist():
            raise ValueError("Input record_id values conflict with source provenance")
    frame, input_metadata = load_input(input_path, identifier_path)
    _validate_input(frame, manifest)
    selectors = _selectors(bundle, manifest, frame)
    development = frame.loc[~frame.record_id.isin(selectors["final_test"])]
    if canonical_digest(development, ["record_id", "Segment"]) != manifest["source"]["development_labels_sha256"]:
        raise ValueError("Development Segment targets differ from the frozen allocation input")
    split_config = SplitConfig.from_dict(manifest["split_config"])
    rule_config = RuleFinalConfig.from_dict(manifest["rule_final_config"])
    model_config = ModelConfig.from_dict(manifest["model_config"])
    allocation_config = AllocationConfig.from_dict(manifest["development_allocation_config"])
    ids = frame.record_id.to_numpy()
    progress(f"Reproducing {manifest['protocol_id']} on {len(frame):,} records")
    stage = perf_counter()
    product_groups, edges, grouping = refine_rule_groups_v4(frame, ids, split_config, rule_config)
    product_map = pd.DataFrame({"record_id": ids, "product_group_id": product_groups})
    if canonical_digest(product_map, ["record_id", "product_group_id"]) != manifest["expected"]["product_group_sha256"]:
        raise ValueError("Frozen rule-v4 product groups did not reproduce")
    tokenizer_dir = bundle / "tokenizer"
    fingerprints = {path.name: sha256(path) for path in tokenizer_dir.iterdir() if path.is_file()}
    effective_groups, _, model_stats = exact_model_input_groups(frame, ids, model_config,
        tokenizer=OfflineTokenizer(tokenizer_dir), tokenizer_fingerprint=fingerprints)
    model_inputs = pd.DataFrame({"record_id": ids, "effective_input_group_id": effective_groups})
    if canonical_digest(model_inputs, ["record_id", "effective_input_group_id"]) != manifest["expected"]["effective_input_group_sha256"]:
        raise ValueError("Exact effective transformer-input groups did not reproduce")
    groups, unions = merge_model_input_groups(product_groups, ids, effective_groups)
    grouping_seconds = perf_counter() - stage
    mask, reservation = reserve_unexposed_groups(ids, groups, selectors["strict_eligible"], selectors["original_test"])
    if set(ids[mask]) != selectors["final_test"]:
        raise ValueError("Whole-group reservation changed the October 9 final test")
    stage = perf_counter()
    assignment, allocation = allocate_remaining(frame, groups, mask, allocation_config)
    repeated, _ = allocate_remaining(frame, groups, mask, allocation_config)
    if not assignment.equals(repeated):
        raise ValueError("Seeded development allocation did not reproduce")
    _check_expected(assignment, manifest)
    allocation_seconds = perf_counter() - stage
    progress("Exact October 9 memberships reproduced; auditing independent duplicates")
    stage = perf_counter()
    pairs, evaluation = near_duplicate_pairs(frame, split_config, progress=progress)
    metrics = leakage_metrics(frame, assignment, pairs, split_config)
    exact_inputs = _exact_metrics(effective_groups, assignment.split.tolist(), "frozen actual token arrays")
    if metrics["exact_payload"]["crossing_pairs"] or exact_inputs["crossing_pairs"]:
        raise ValueError("Exact product payload or transformer input leakage detected")
    distributions = segment_distribution(frame, assignment)
    audit_seconds = perf_counter() - stage
    # Fail before writing any output if the source/protocol changed during computation.
    if source_hash != sha256(input_path):
        raise ValueError("Input changed during split generation")
    load_protocol(bundle)
    exports = frame.copy()
    exports["group_id"], exports["split"] = groups, assignment.split
    assignment = assignment[ASSIGNMENT_FIELDS].copy()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "eligibility.json"):
        shutil.copyfile(bundle / name, output_dir / ("protocol_manifest.json" if name == "manifest.json" else name))
    shutil.copytree(tokenizer_dir, output_dir / "tokenizer")
    _write_json(output_dir / "model_config.json", model_config.to_dict())
    _write_json(output_dir / "split_config.json", split_config.to_dict())
    _write_json(output_dir / "rule_final_config.json", rule_config.to_dict())
    _write_json(output_dir / "development_allocation_config.json", allocation_config.to_dict())
    freeze = {"format": FORMAT, "protocol_id": manifest["protocol_id"],
        "split_version": manifest["split_version"], "grouping_version": manifest["grouping_version"],
        "original_frozen_utc": "2026-10-09T23:13:03.107666+00:00",
        "reproduced_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_manifest_sha256": sha256(bundle / "manifest.json"),
        "algorithm_source_sha256_lf": manifest["algorithm_source_sha256_lf"],
        "model_view_contract": portable_model_view_contract(model_config, tokenizer_dir),
        "test_policy": "Reproduce frozen exposure selectors and whole-group closure; never reserve a replacement test.",
        "final_test_record_ids_sha256": manifest["expected"]["splits"]["test"]["record_ids_sha256"]}
    _write_json(output_dir / "grouping_freeze.json", freeze)
    assignment.to_csv(output_dir / "assignments.csv", index=False, lineterminator="\n")
    shutil.copyfile(output_dir / "assignments.csv", output_dir / "rule_assignments.csv")
    assignment.loc[assignment.split.eq("test"), ["record_id", "group_id", "split"]].to_csv(output_dir / "protected_final_test.csv", index=False, lineterminator="\n")
    product_map["effective_input_group_id"], product_map["group_id"] = effective_groups, groups
    product_map.to_csv(output_dir / "grouping_assignments.csv", index=False, lineterminator="\n")
    model_inputs.to_csv(output_dir / "model_inputs.csv", index=False, lineterminator="\n")
    _edges_frame(edges, frame).to_csv(output_dir / "product_matching_edges.csv", index=False, lineterminator="\n")
    assignment.groupby("group_id").size().rename("rows").reset_index().to_csv(output_dir / "group_sizes.csv", index=False, lineterminator="\n")
    distributions.to_csv(output_dir / "segment_distributions.csv", index=False, lineterminator="\n")
    for role in SPLITS:
        subset = exports.loc[exports.split.eq(role)].reset_index(drop=True)
        subset.to_csv(output_dir / f"{role}.csv", index=False, lineterminator="\n")
        # Byte checks and ID-only reload protect final labels from inspection.
        restored_ids = _read(output_dir / f"{role}.csv", ASSIGNMENT_FIELDS)
        pd.testing.assert_frame_equal(restored_ids, subset[ASSIGNMENT_FIELDS].reset_index(drop=True))
    audit = pd.DataFrame([{"left_record_id": ids[a], "right_record_id": ids[b], "name_jaccard": score,
        "left_split": assignment.at[a, "split"], "right_split": assignment.at[b, "split"],
        "crosses_split": assignment.at[a, "split"] != assignment.at[b, "split"]} for a, b, score in pairs],
        columns=["left_record_id", "right_record_id", "name_jaccard", "left_split", "right_split", "crosses_split"])
    audit.to_csv(output_dir / "independent_near_pairs.csv", index=False, lineterminator="\n")
    input_info = {"filename": input_path.name, "file_sha256": source_hash,
        "source": manifest["source"], "input_columns": input_columns,
        "source_container_note": "Integrated candidate has additional preserved columns; identity and grouping/development content are verified canonically."}
    _write_json(output_dir / "input_manifest.json", input_info)
    _write_json(output_dir / "test_reservation.json", reservation)
    summary = {"format": FORMAT, "version": manifest["split_version"], "protocol_id": manifest["protocol_id"],
        "integration_version": INTEGRATION_VERSION, "grouping_method": "rule_v4_with_exact_model_input_constraints",
        "target": "Segment", "seed": split_config.seed, "input": input_info,
        "split_config": split_config.to_dict(), "rule_final_config": rule_config.to_dict(),
        "grouping": grouping, "effective_input_union": {k: v for k, v in unions.items() if k != "merge_evidence"},
        "effective_input_groups": {k: v for k, v in model_stats.items() if k != "model_view_contract"},
        "test_reservation": reservation, "development_allocation": allocation,
        "metrics": metrics, "effective_input_duplicates": exact_inputs, "evaluation": evaluation,
        "authoritative": {"manifest_sha256": sha256(bundle / "manifest.json"),
            "assignment_sha256": canonical_digest(assignment, ["record_id", "group_id", "split"]),
            "historical_snapshot": manifest["historical_snapshot"]},
        "checks": {"complete_unique_coverage": True, "group_isolation": True,
            "labels_excluded_from_grouping": True, "exact_frozen_assignments": True,
            "exact_final_test_membership": True, "frozen_exposure_exclusions": True,
            "seeded_allocation_reproduces": True, "input_artifacts_unchanged": True,
            "exact_payload_crossings_zero": True, "exact_effective_input_crossings_zero": True,
            "export_values_preserved": True},
        "runtime_seconds": {"grouping_and_input_constraints": grouping_seconds,
            "allocation_and_reproduction": allocation_seconds, "independent_audit": audit_seconds,
            "total": perf_counter() - started},
        "environment": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "tokenizers")},
        "final_test_policy": "Do not load final-test features/labels during development. Use the guarded train/validation loader."}
    _write_json(output_dir / "summary.json", summary)
    if source_hash != sha256(input_path):
        raise ValueError("Input changed during split export; outputs are not sealed")
    load_protocol(bundle)
    artifacts = {path.relative_to(output_dir).as_posix(): sha256(path) for path in sorted(output_dir.rglob("*")) if path.is_file()}
    _write_json(output_dir / "completion.json", {"format": FORMAT,
        "protocol_manifest_sha256": sha256(bundle / "manifest.json"), "artifact_sha256": artifacts})
    verify_protocol_splits(output_dir, protocol_dir=bundle)
    progress("Verified exact finalized assignments and protected test membership")
    progress(" | ".join(f"{role}: {metrics['splits'][role]['rows']:,}" for role in SPLITS))
    return summary
