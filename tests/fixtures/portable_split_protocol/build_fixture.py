"""Create the public, synthetic protocol used by splitting integration tests.

This fixture contains no CatalogIQ source records and never downloads or trains
a model. The expected memberships are created from the unchanged final engines.
Its tokenizer comes from the repository's public tokenizer resource.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from catalogiq.cleaning import PROVENANCE
from catalogiq.features import sha256
from catalogiq.model_input_groups import exact_model_input_groups, merge_model_input_groups
from catalogiq.segment_development_allocation import AllocationConfig
from catalogiq.segment_transformer import ModelConfig
from catalogiq.split_finalization import allocate_remaining, reserve_unexposed_groups
from catalogiq.split_rules_v4 import RuleFinalConfig, refine_rule_groups_v4
from catalogiq.splitting import GROUP_FIELDS, SplitConfig, record_ids


SOURCE_SHA256 = hashlib.sha256(b"Public synthetic CatalogIQ integration source v1").hexdigest()


def synthetic_frame():
    rows = []
    for family in range(180):
        suffix = chr(65 + family // 26) + chr(65 + family % 26)
        brand = "Botanica" + suffix
        line = f"Amber Nectar {suffix} Tablets"
        if family in (0, 1):
            brand = "Hion"
            line = ("Motion Sickness Glasses Travel Comfort" if family == 0
                    else "Motion Sickness Acupressure Wristbands Travel Comfort")
        elif family in (36, 39):
            # Full product names identify different ingredients. The frozen
            # classifier consumes only the common 1,024-character name prefix.
            line = "Pack " * 260 + ("Marshmallow" if family == 36 else "Peppermint")
        for variant in range(2):
            source_row = 2 * family + variant
            name = f"{line}, 38 Count" if variant == 0 else f"3 Pack {line} 38 Count"
            rows.append({
                "dataset": "synthetic_training.csv", "source_sha256": SOURCE_SHA256,
                "source_row": str(source_row), "ProductName": name, "ProductBrand": brand,
                "ProductDescription": "null" if family == 1 else f"Formula for {brand} {line[-70:]}.",
                "ProductContents": "" if family == 2 else f"Ingredients for {brand}",
                "Retailer": "Synthetic Market" if variant else "Synthetic Store",
                "ProductUrl": f"https://store.example/products/{family}/{variant}",
                "Segment": ("Allergy", "CCFS", "Digestive Health")[family % 3],
                "Sku": f"{source_row:06d}", "Upc": "", "ProductModelNumber": "",
                "MDM_Id": "", "JoiningKey": "", "ProductCode": f"{source_row:06d}",
                "SourceNote": "NA" if family == 3 else 'Preserve commas, quotes "and"\nUnicode: café.',
            })
    frame = pd.DataFrame(rows)
    frame["record_id"] = record_ids(frame)
    return frame.sort_values("record_id", kind="stable").reset_index(drop=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")


def build_manifest_once(directory, production_bundle):
    """Maintenance helper; tests consume the committed expected hashes instead."""
    from catalogiq.split_protocol import OfflineTokenizer, canonical_digest

    directory, production_bundle = Path(directory), Path(production_bundle)
    directory.mkdir(parents=True)
    production = json.loads((production_bundle / "manifest.json").read_text(encoding="utf8"))
    tokenizer_dir = directory / "tokenizer"
    shutil.copytree(production_bundle / "tokenizer", tokenizer_dir)
    for filename in ("NOTICE", "LICENSE"):
        shutil.copyfile(production_bundle / filename, directory / filename)
    frame = synthetic_frame()
    ids = frame.record_id.to_numpy()
    split_config, rule_config = SplitConfig.from_dict(production["split_config"]), RuleFinalConfig.from_dict(production["rule_final_config"])
    model_config = ModelConfig.from_dict(production["model_config"])
    allocation_config = AllocationConfig.from_dict(production["development_allocation_config"])
    groups, _, _ = refine_rule_groups_v4(frame, ids, split_config, rule_config)
    exact, _, _ = exact_model_input_groups(frame, ids, model_config,
        tokenizer=OfflineTokenizer(tokenizer_dir),
        tokenizer_fingerprint={path.name: sha256(path) for path in tokenizer_dir.iterdir() if path.is_file()})
    final_groups, _ = merge_model_input_groups(groups, ids, exact)
    numeric_rows = frame.source_row.astype(int)
    old_test = set(frame.loc[numeric_rows.ge(300), "record_id"])
    eligible = set(frame.loc[numeric_rows.ge(305), "record_id"])
    mask, reservation = reserve_unexposed_groups(ids, final_groups, eligible, old_test)
    if int(mask.sum()) != 54:
        raise AssertionError("The synthetic test reservation unexpectedly changed")
    assignment, _ = allocate_remaining(frame, final_groups, mask, allocation_config)
    by_source = pd.DataFrame({"source_row": frame.source_row, "product_group": groups,
                              "effective_group": exact, "final_group": final_groups}).set_index("source_row")
    if (by_source.at["72", "product_group"] == by_source.at["78", "product_group"]
            or by_source.at["72", "effective_group"] != by_source.at["78", "effective_group"]
            or by_source.at["72", "final_group"] != by_source.at["78", "final_group"]):
        raise AssertionError("The synthetic case must exercise a real hard-equality union")
    if by_source.at["0", "final_group"] == by_source.at["2", "final_group"]:
        raise AssertionError("The frozen v4 device-family safeguard failed")

    assignment_fields = ["record_id", "group_id", "split"]
    manifest = dict(production)
    manifest["protocol_id"] = "synthetic-segment-final-protocol-v1"
    manifest["source"] = {
        "dataset": "synthetic_training.csv", "source_sha256": SOURCE_SHA256,
        "input_rows": len(frame), "record_ids_sha256": canonical_digest(frame, ["record_id"]),
        "provenance_sha256": canonical_digest(frame, PROVENANCE),
        "group_fields": list(GROUP_FIELDS),
        "group_features_sha256": canonical_digest(frame, ["record_id", *GROUP_FIELDS]),
        "development_labels_sha256": canonical_digest(frame.loc[~mask], ["record_id", "Segment"]),
    }
    manifest["expected"] = {
        "assignment_sha256": canonical_digest(assignment, assignment_fields),
        "product_group_sha256": canonical_digest(pd.DataFrame({"record_id": ids, "product_group_id": groups}), ["record_id", "product_group_id"]),
        "effective_input_group_sha256": canonical_digest(pd.DataFrame({"record_id": ids, "effective_input_group_id": exact}), ["record_id", "effective_input_group_id"]),
        "group_sha256": canonical_digest(assignment, ["record_id", "group_id"]),
        "splits": {role: {
            "records": int(assignment.split.eq(role).sum()),
            "groups": int(assignment.loc[assignment.split.eq(role), "group_id"].nunique()),
            "record_ids_sha256": canonical_digest(assignment.loc[assignment.split.eq(role)], ["record_id"]),
            "assignment_sha256": canonical_digest(assignment.loc[assignment.split.eq(role)], assignment_fields),
        } for role in ("train", "validation", "test")},
        "test_reservation": reservation,
    }
    selectors = {}
    for name, selected_ids in (("original_test", old_test), ("strict_eligible", eligible),
                               ("final_test", set(ids[mask]))):
        chosen = frame.loc[frame.record_id.isin(selected_ids)]
        selectors[name] = {"count": len(chosen), "source_rows": sorted(chosen.source_row.astype(int).tolist()),
                           "record_ids_sha256": canonical_digest(chosen, ["record_id"])}
    write_json(directory / "eligibility.json", {
        "schema_version": 1, "source_identity": {"dataset": "synthetic_training.csv", "source_sha256": SOURCE_SHA256},
        "selectors": selectors,
    })
    manifest["resources_sha256"] = {name: sha256(directory / name) for name in production["resources_sha256"]}
    write_json(directory / "manifest.json", manifest)
    return frame, manifest


def materialize(directory, production_bundle):
    """Copy frozen fixture metadata and public tokenizer into a temporary path.

    Expected memberships are never recomputed by a test. This also exercises
    verification after moving the bundle away from its original repository path.
    """
    directory, production_bundle = Path(directory), Path(production_bundle)
    directory.mkdir(parents=True)
    fixture_dir = Path(__file__).resolve().parent
    for filename in ("manifest.json", "eligibility.json"):
        shutil.copyfile(fixture_dir / filename, directory / filename)
    for filename in ("NOTICE", "LICENSE"):
        shutil.copyfile(production_bundle / filename, directory / filename)
    shutil.copytree(production_bundle / "tokenizer", directory / "tokenizer")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    return synthetic_frame(), manifest
