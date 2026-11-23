"""Offline product-level review replay. Emits aggregate counts only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from catalogiq.confidence_calibration import essential_information_reasons, group_role, temperature_scale
from catalogiq.features import sha256
from catalogiq.prediction_contract import REGISTRY_NAME, validate_manifest
from catalogiq.relationship_conflicts import flag_all
from catalogiq.structural import StructuralConfig, missing
from catalogiq.target_minilm_experiment import brand_key


ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "docs/findings/03_modeling"
TARGETS = ("Segment", "Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")
HIDDEN = ("Segment", "Brand", "Mnfr", "TargetAgeGroup")


def verified_member(package, member, digest):
    raw = package.read(member)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Frozen file hash mismatch: {member}")
    return raw


def read_predictions(package, route, calibration):
    directory = Path(route["manifest"]).parent.name
    manifest = json.loads(verified_member(package, f"{directory}/manifest.json",
                                          route["files_sha256"][route["manifest"]]))

    def member(name):
        return verified_member(package, f"{directory}/{name}", manifest["files_sha256"][name])

    def csv(name):
        return pd.read_csv(io.BytesIO(member(name)), keep_default_na=False)

    predictions = csv("predictions.csv")
    context = csv("validation_context.csv")
    validate_manifest(manifest, predictions, context, csv(REGISTRY_NAME))
    if manifest["model_identity"] != route["model_id"] or manifest["probabilities_calibrated"]:
        raise ValueError("Expected raw probabilities from the selected model")
    probabilities = predictions.filter(regex=r"^probability_\d+$")
    params = calibration.loc[directory]
    if params.candidate_variant != "temperature":
        raise ValueError("Expected a frozen temperature recipe")
    confidence = temperature_scale(probabilities, params.temperature).max(axis=1)
    frame = predictions[["record_id", "predicted_label", "true_label"]].copy()
    frame["calibrated_confidence"] = confidence
    frame["score_pass"] = confidence >= params.candidate_threshold
    frame = frame.merge(context[["record_id", "group_id"]], on="record_id", validate="one_to_one")
    lookup = json.loads(member("model.json")) if manifest["target"] == "Brand" else None
    return frame.set_index("record_id"), lookup


def load_inputs(handoff, validation):
    routes = json.loads((ROOT / "src/catalogiq/resources/inference/frozen_routes.json").read_text())
    params = pd.read_csv(FINDINGS / "confidence_review_20261103/results.csv").set_index("run")
    frames = {}
    with ZipFile(handoff) as package:
        for target in TARGETS:
            frame, model = read_predictions(package, routes["routes"][target], params)
            frames[target] = frame
            if target == "Brand":
                lookup = model
    if validation.name != "validation.csv":
        raise ValueError("A sealed development validation.csv is required")
    seal = json.loads((validation.parent / "completion.json").read_text())
    if sha256(validation) != seal["artifact_sha256"]["validation.csv"]:
        raise ValueError("Development features differ from the split seal")
    fields = ["record_id", "group_id", "split", "ProductBrand", "ProductName",
              "ProductContents", "ProductCategory", "ProductDescription"]
    features = pd.read_csv(validation, usecols=fields, dtype=str, keep_default_na=False)
    if not features.split.eq("validation").all():
        raise ValueError("Unexpected split in validation features")
    return frames, lookup, features.set_index("record_id", verify_integrity=True)


def summarize(route, fields, frames, groups):
    ids = route.index[route]
    errors = pd.DataFrame({k: frames[k].loc[ids, "true_label"].ne(frames[k].loc[ids, "predicted_label"])
                           for k in fields}, index=ids)
    return {
        "partial_review_rows": len(ids),
        "partial_review_groups": int(groups.loc[ids].nunique()),
        "full_review_rows": len(route) - len(ids),
        "field_checks_skipped": len(ids) * len(fields),
        "rows_with_hidden_errors": int(errors.any(axis=1).sum()),
        "hidden_errors_by_field": {k: int(errors[k].sum()) for k in fields},
    }


def shared_audit(frames, features):
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if not common:
        raise ValueError("No jointly labeled validation records")
    common_groups = features.loc[common, "group_id"]
    for target, frame in frames.items():
        if not frame.loc[common, "group_id"].equals(common_groups):
            raise ValueError(f"Family mismatch for {target}")
    audit_ids = common_groups.index[common_groups.map(group_role).eq("policy_audit")]
    if audit_ids.empty:
        raise ValueError("No shared policy-audit records")
    return common, audit_ids


def routes_for_review(inputs, audit, lookup, warned_ids, allowed):
    """Select the fields to display using only information available at arrival."""
    keys = inputs.ProductBrand.map(brand_key)
    named_keys = {k for k, counts in lookup["key_counts"].items() if len(counts) == 1}
    named = (keys.isin(named_keys) & ~keys.isin(lookup["missing_normalized_values"])
             & ~audit["Brand"].predicted_label.isin(["Other Brands", "UnItemised brand", "Generic"]))
    complete = pd.Series([not r for r in essential_information_reasons(inputs)], index=inputs.index)
    common_gate = named & complete & ~inputs.index.isin(warned_ids)
    two_field_gate = common_gate & audit["Mnfr"].score_pass & audit["TargetAgeGroup"].score_pass
    config = StructuralConfig()
    has_description = ~inputs.ProductDescription.map(lambda value: missing(value, config))
    sub_only = (two_field_gate & audit["Brand"].score_pass & has_description
                & audit["Segment"].score_pass & audit["Segment"].predicted_label.isin(allowed))
    return named, two_field_gate, sub_only


def replay(handoff, validation):
    frames, lookup, features = load_inputs(handoff, validation)
    common, audit_ids = shared_audit(frames, features)
    inputs = features.loc[audit_ids]
    audit = {k: f.loc[audit_ids] for k, f in frames.items()}
    warnings = flag_all({k: f.reset_index() for k, f in audit.items()})
    segment_rule = json.loads((FINDINGS / "product_review_rules_20261103/segment_experimental_policy.json").read_text())
    named, two_field_gate, sub_only = routes_for_review(
        inputs, audit, lookup, warnings.record_id, segment_rule["rule"]["allowed_labels"])
    single = summarize(sub_only, HIDDEN, audit, inputs.group_id)
    three = summarize(two_field_gate & ~sub_only, ("Mnfr", "TargetAgeGroup"), audit, inputs.group_id)
    full = int((~two_field_gate).sum())
    return {
        "target_validation_rows": {k: len(f) for k, f in frames.items()},
        "validation_union_rows": len(set.union(*(set(f.index) for f in frames.values()))),
        "joint_validation_rows": len(common),
        "joint_audit_rows": len(inputs), "joint_audit_groups": int(inputs.group_id.nunique()),
        "named_unambiguous_brand_rows": int(named.sum()),
        "rows_with_candidate_warnings": int(warnings.record_id.nunique()),
        "baseline_field_checks": len(inputs) * len(TARGETS),
        "only_subsegment": single,
        "only_hide_manufacturer_and_age": summarize(two_field_gate, ("Mnfr", "TargetAgeGroup"), audit, inputs.group_id),
        "combined_routes": {
            "five_fields": full,
            "three_fields": three["partial_review_rows"],
            "one_field": single["partial_review_rows"],
            "field_checks_remaining": 5 * full + 3 * three["partial_review_rows"] + single["partial_review_rows"],
            "rows_with_hidden_errors": single["rows_with_hidden_errors"] + three["rows_with_hidden_errors"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path, help="Frozen predictions_contract_v2_20261031.zip")
    parser.add_argument("validation", type=Path, help="Sealed V4 validation.csv with adjacent completion.json")
    args = parser.parse_args()
    print(json.dumps(replay(args.handoff, args.validation), indent=2))
