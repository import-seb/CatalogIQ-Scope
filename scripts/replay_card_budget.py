"""Simulate a fixed human-review budget, with one queue entry per product."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from catalogiq.relationship_conflicts import flag_all
from catalogiq.target_minilm_experiment import brand_key
from replay_partial_review import TARGETS, load_inputs, shared_audit


def rank_cards(scores, warned_ids, prioritize_warnings):
    if (scores.empty or not scores.index.is_unique or not np.isfinite(scores).all().all()
            or not ((scores >= 0) & (scores <= 1)).all().all()):
        raise ValueError("Each product needs one complete set of confidence scores")
    if not set(warned_ids).issubset(scores.index):
        raise ValueError("Warning does not match a product")
    # The minimum is a ranking signal, not a probability of whole-card success.
    rank = pd.DataFrame({"min_score": scores.min(axis=1),
                         "warning": scores.index.isin(warned_ids)}, index=scores.index)
    rank["key"] = rank.index
    order = ["warning", "min_score", "key"] if prioritize_warnings else ["min_score", "key"]
    ascending = [False, True, True] if prioritize_warnings else [True, True]
    return rank.sort_values(order, ascending=ascending).index


def run(args):
    frames, lookup, features = load_inputs(args.handoff, args.validation)
    _, ids = shared_audit(frames, features)
    audit = {k: f.loc[ids] for k, f in frames.items()}
    scores = pd.DataFrame({k: f.calibrated_confidence for k, f in audit.items()})
    errors = pd.DataFrame({k: f.true_label.ne(f.predicted_label) for k, f in audit.items()}).any(axis=1)
    warned = flag_all({k: f.reset_index() for k, f in audit.items()}).record_id.unique()

    catalog = pd.read_csv(args.catalog, dtype=str, keep_default_na=False).set_index("JoiningKey", verify_integrity=True)
    target_scores = pd.DataFrame({
        k: pd.to_numeric(catalog[f"catalogiq:{k}:calibrated_confidence"]) for k in TARGETS})
    conflicts = pd.read_csv(args.warnings, dtype=str, keep_default_na=False).record_id
    unknown = ~catalog.ProductBrand.map(brand_key).isin(set(lookup["key_counts"]) - set(lookup["missing_normalized_values"]))
    if not 1 <= args.budget <= len(catalog):
        raise ValueError("Review budget must fit the target catalog")
    n = round(len(ids) * args.budget / len(catalog))
    results = {}
    for name, warnings_first in [("lowest_score_first", False), ("warnings_then_lowest_score", True)]:
        queue = rank_cards(scores, warned, warnings_first)[:n]
        target_queue = rank_cards(target_scores, conflicts.unique(), warnings_first)[:args.budget]
        caught = int(errors.loc[queue].sum())
        results[name] = {"known_errors_in_queue": caught, "known_errors_outside": int(errors.sum()) - caught,
                         "target_warning_cards_queued": int(target_queue.isin(conflicts).sum()),
                         "target_unseen_or_missing_brand_queued": int(unknown.loc[target_queue].sum())}
    return {"target_products": len(catalog), "target_budget": args.budget,
            "target_warning_products": conflicts.nunique(), "target_unseen_or_missing_brand": int(unknown.sum()),
            "target_warnings_or_unseen_or_missing_brand": int((unknown | catalog.index.isin(conflicts)).sum()),
            "audit_products": len(ids), "audit_budget_nearest_integer": n, "audit_error_products": int(errors.sum()),
            "audit_wrong_by_field": {k: int(f.true_label.ne(f.predicted_label).sum()) for k, f in audit.items()},
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("handoff", "validation", "catalog", "warnings"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--budget", type=int, default=1000)
    print(json.dumps(run(parser.parse_args()), indent=2))
