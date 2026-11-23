"""Whole-card skip candidates across fixed confidence floors, with no queue quota."""
import argparse
import json
from pathlib import Path

import pandas as pd

from catalogiq.relationship_conflicts import flag_all, load_registry
from catalogiq.target_minilm_experiment import brand_key
from replay_partial_review import ROOT, TARGETS, load_inputs, shared_audit


def candidate_scores(features, scores, brand_labels, lookup, warnings):
    """Return a ranking signal only for complete, named, unambiguous products."""
    if (not scores.index.is_unique or not scores.index.equals(features.index)
            or not scores.index.equals(brand_labels.index)
            or not set(warnings).issubset(scores.index)):
        raise ValueError("Inconsistent product IDs")
    scores = scores.loc[:, list(TARGETS)]
    valid_scores = scores.ge(0).all(axis=1) & scores.le(1).all(axis=1)
    keys = features.ProductBrand.map(brand_key)
    unambiguous = {k for k, counts in lookup["key_counts"].items() if len(counts) == 1}
    eligible = (valid_scores & features.ProductName.map(brand_key).ne("")
                & keys.isin(unambiguous - set(lookup["missing_normalized_values"]))
                & ~brand_labels.isin(["Other Brands", "UnItemised brand", "Generic", ""])
                & ~scores.index.isin(warnings))
    return scores.min(axis=1).where(eligible)


def score_catalog(catalog_path, warnings_path, lookup):
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False).set_index("JoiningKey", verify_integrity=True)
    warnings = pd.read_csv(warnings_path, dtype=str, keep_default_na=False)
    registry_version = load_registry()["registry_version"]
    if not (catalog["catalogiq:registry_version"].eq(registry_version).all()
            and warnings.registry_version.eq(registry_version).all()):
        raise ValueError("Relationship registry mismatch")
    routes = json.loads((ROOT / "src/catalogiq/resources/inference/frozen_routes.json").read_text())["routes"]
    target_scores = {}
    for k in TARGETS:
        prefix = f"catalogiq:{k}:"
        if not catalog[prefix + "model_version"].eq(routes[k]["model_id"]).all():
            raise ValueError(f"Unexpected model version for {k}")
        available = catalog[prefix + "prediction_status"].eq("suggestion") & catalog[prefix + "original_label"].ne("")
        target_scores[k] = pd.to_numeric(catalog[prefix + "calibrated_confidence"], errors="coerce").where(available)
    catalog_score = candidate_scores(catalog, pd.DataFrame(target_scores), catalog["catalogiq:Brand:original_label"],
                                     lookup, warnings.record_id).where(catalog["catalogiq:eligibility"].eq("eligible"))
    return catalog, catalog_score


def replay(args):
    frames, lookup, features = load_inputs(args.handoff, args.validation)
    _, ids = shared_audit(frames, features)
    audit = {k: f.loc[ids] for k, f in frames.items()}
    warns = flag_all({k: f.reset_index() for k, f in audit.items()}).record_id
    scores = pd.DataFrame({k: f.calibrated_confidence for k, f in audit.items()})
    audit_score = candidate_scores(features.loc[ids], scores, audit["Brand"].predicted_label, lookup, warns)
    errors = pd.DataFrame({k: f.true_label.ne(f.predicted_label) for k, f in audit.items()}).any(axis=1)
    catalog, catalog_score = score_catalog(args.catalog, args.warnings, lookup)
    rows = []
    for floor in (0.95, 0.98, 0.99, 0.995):
        eligible = audit_score.ge(floor)
        skipped = int(catalog_score.ge(floor).sum())
        rows.append({"floor": floor, "validation_skip": int(eligible.sum()),
                     "validation_skip_families": int(features.loc[ids[eligible], "group_id"].nunique()),
                     "validation_errors_skipped": int((errors & eligible).sum()),
                     "catalog_skip": skipped, "catalog_review": len(catalog) - skipped})
    return {"validation_rows": len(ids), "validation_errors": int(errors.sum()),
            "catalog_rows": len(catalog), "scenarios": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "validation", "catalog", "warnings"):
        parser.add_argument(name, type=Path)
    print(json.dumps(replay(parser.parse_args()), indent=2))
