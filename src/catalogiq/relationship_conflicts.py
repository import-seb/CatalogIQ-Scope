"""Warn-mode relationship conflict flagging.

This is a dry run, not a review decision. Every relation in the registry is a
candidate, so nothing here is approved and nothing here may gate a product. The
output says what an approved relation WOULD observe, so the taxonomy owner can
see the workload and the cases before signing anything off.

The pair semantics come from review_decision itself rather than a second
implementation, so there is one source of truth for what a conflict is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from catalogiq.confidence_calibration import review_decision

REGISTRY_PATH = Path(__file__).parent / "resources" / "relationships" / "classification_relationship_registry.json"
RELATIONSHIP_CODES = ("invalid_approved_classification_relationship",
                      "missing_relationship_classification")
CONFIG_CODES = ("unapproved_relationship_rule", "invalid_approved_relationship_rule")
# A valid single-class-pair probe so review_decision reaches the relationship
# block; these values carry no meaning for the pair outcome.
_PROBE = dict(probabilities=[0.9, 0.1], predicted_label="a", labels=["a", "b"],
              threshold=0.5, product={"ProductName": "probe"}, enabled=True)


def load_registry(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REGISTRY_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def enforceable_relations(registry: dict) -> list[dict]:
    """Relations that define pairs. R3 is recorded as rejected and defines none."""
    return [r for r in registry["relations"] if r.get("allowed_pairs")]


def _outcome(relation: dict, child_value, parent_value) -> list[str]:
    """What review_decision says about this one pair, with the relation promoted.

    Promotion is local to the probe. The registry on disk is never modified, and
    the caller is told the relation is unapproved in every output row.
    """
    approved = dict(relation, status="approved")
    classifications = {}
    if child_value is not None:
        classifications[relation["child_target"]] = child_value
    if parent_value is not None:
        classifications[relation["parent_target"]] = parent_value
    reasons = review_decision(**_PROBE, approved_relations=[approved],
                              classifications=classifications)["review_reasons"]
    bad_config = [r for r in reasons if r in CONFIG_CODES]
    if bad_config:
        raise ValueError(f"{relation['relation_id']} is malformed for the primitive: {bad_config}")
    return [r for r in reasons if r in RELATIONSHIP_CODES]


def _norm(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Normalize and name the columns here rather than leaning on join suffixes.

    DataFrame.join's lsuffix/rsuffix are deprecated and removed in pandas 4, and
    the replacement keyword is not available across the floor this project pins,
    so the columns carry their final names before the join.
    """
    out = frame.copy()
    out["record_id"] = out["record_id"].astype("string").str.strip()
    out["predicted_label"] = out["predicted_label"].astype("string").str.strip()
    out.loc[out["predicted_label"].str.len() == 0, "predicted_label"] = pd.NA
    if "confidence" not in out.columns:
        out["confidence"] = pd.NA
    out = out.set_index("record_id")[["predicted_label", "confidence"]]
    return out.rename(columns={"predicted_label": f"predicted_label{suffix}",
                               "confidence": f"confidence{suffix}"})


def flag_relation(relation: dict, predictions: dict[str, pd.DataFrame],
                  registry_version: str) -> pd.DataFrame:
    """Warn-mode observations for one relation over every record present."""
    child_target, parent_target = relation["child_target"], relation["parent_target"]
    for target in (child_target, parent_target):
        if target not in predictions:
            raise KeyError(f"no predictions supplied for {target!r}")
    child = _norm(predictions[child_target], "_child")
    parent = _norm(predictions[parent_target], "_parent")
    joined = child.join(parent, how="outer")

    cache: dict[tuple, list[str]] = {}
    rows = []
    for record_id, row in joined.iterrows():
        c = row["predicted_label_child"] if pd.notna(row["predicted_label_child"]) else None
        p = row["predicted_label_parent"] if pd.notna(row["predicted_label_parent"]) else None
        key = (c, p)
        if key not in cache:
            cache[key] = _outcome(relation, c, p)
        for code in cache[key]:
            rows.append({
                "record_id": record_id,
                "relation_id": relation["relation_id"],
                "observation": code,
                "child_target": child_target,
                "child_prediction": c,
                "child_confidence": row["confidence_child"],
                "parent_target": parent_target,
                "parent_prediction": p,
                "parent_confidence": row["confidence_parent"],
                "relation_status": relation["status"],
                "registry_version": registry_version,
                "is_approved_decision": False,
            })
    columns = ["record_id", "relation_id", "observation", "child_target", "child_prediction",
               "child_confidence", "parent_target", "parent_prediction", "parent_confidence",
               "relation_status", "registry_version", "is_approved_decision"]
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["relation_id", "observation", "record_id"]).reset_index(drop=True)


def flag_all(predictions: dict[str, pd.DataFrame], registry: dict | None = None) -> pd.DataFrame:
    registry = registry or load_registry()
    version = registry["registry_version"]
    frames = [flag_relation(r, predictions, version) for r in enforceable_relations(registry)
              if r["child_target"] in predictions and r["parent_target"] in predictions]
    if not frames:
        raise ValueError("no relation in the registry could be evaluated from these predictions")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["relation_id", "observation", "record_id"]).reset_index(drop=True)


def summarize(flags: pd.DataFrame, predictions: dict[str, pd.DataFrame],
              registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    per_relation = {}
    for relation in enforceable_relations(registry):
        rid = relation["relation_id"]
        if relation["child_target"] not in predictions or relation["parent_target"] not in predictions:
            per_relation[rid] = {"evaluated": False,
                                 "reason": "predictions missing for one side of the pair"}
            continue
        child = _norm(predictions[relation["child_target"]], "_child")
        parent = _norm(predictions[relation["parent_target"]], "_parent")
        both = child.join(parent, how="inner")
        both = both[both["predicted_label_child"].notna() & both["predicted_label_parent"].notna()]
        mine = flags[flags["relation_id"] == rid]
        conflicts = mine[mine["observation"] == "invalid_approved_classification_relationship"]
        per_relation[rid] = {
            "evaluated": True,
            "relation_status": relation["status"],
            "records_with_both_predictions": int(len(both)),
            "conflicts": int(len(conflicts)),
            "conflicts_pct_of_checkable": round(100 * len(conflicts) / len(both), 4) if len(both) else None,
            "not_checkable_one_side_missing": int((mine["observation"] ==
                                                   "missing_relationship_classification").sum()),
        }
    products = sorted({str(r) for r in flags["record_id"]})
    return {
        "registry_version": registry["registry_version"],
        "approved_relations_count": registry["approved_relations_count"],
        "mode": "warn_only",
        "mode_note": ("No relation in the registry is approved, so these are observations and not "
                      "review decisions. Nothing here may gate or reject a product."),
        "distinct_records_with_at_least_one_observation": len(products),
        "per_relation": per_relation,
    }
