"""Observe frozen grouping decisions without changing matching or splitting.

The replay calls the real rule implementation. Temporary observer wrappers and
Python tracing collect evidence only; all returned scores/decisions are the
original values. Reviewed labels select cases, never enter the matcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import inspect
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .features import sha256
from . import split_rules_v2
from .split_matching import (CORE_NOISE, brand_block_key, brand_compatible,
    matching_text, name_similarity, prepare_products, _title_brand_override)
from .split_review_validation import verify_refinement_seal
from .splitting import MISSING, SplitConfig, grouping_view, gtin, listing_url, text
from .splitting_experiment import load_input


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compatibility_evidence(a, b, config, allow_brand_override=False):
    """Explain separate guards; never replace the authoritative decision."""
    overlap = len(a.core & b.core) / min(len(a.core), len(b.core)) if a.core and b.core else None
    ingredients = bool(a.ingredients and b.ingredients and not (a.ingredients & b.ingredients))
    core = overlap is not None and overlap < config.core_min_overlap
    brands = brand_compatible(a, b, config)
    title = _title_brand_override(a, b)
    override_overlap = name_similarity(a, b)
    override = bool(allow_brand_override and title and override_overlap >= config.brand_override_name_threshold)
    failures = []
    if ingredients:
        failures.append("pair_ingredient_conflict")
    if core:
        failures.append("insufficient_informative_core_overlap")
    if not (brands or override):
        failures.append("brand_guard")
    return {"pair_ingredient_conflict": ingredients, "core_overlap": overlap,
            "brand_compatible": brands, "title_brand_override_evidence": title,
            "brand_override_enabled": allow_brand_override,
            "brand_override_title_jaccard": override_overlap,
            "brand_override_passes": override, "failed_guards": failures}


def identifier_evidence(left, right, a, b):
    """Use the exact identifier admission/scoping rules of the frozen matcher."""
    def sku(row):
        retailer, value = text(row.get("Retailer", "")), str(row.get("Sku", "")).strip().casefold()
        return (retailer, value) if retailer and value not in MISSING else None
    def model(row, p):
        value = str(row.get("ProductModelNumber", "")).strip().casefold()
        return ((p.brand, value) if p.brand and re.fullmatch(r"[\w-]{4,}", value)
                and re.search(r"\d", value)
                and not re.fullmatch(r"\d+(?:\.\d+)?e[+-]?\d+", value) else None)
    lgt, rgt = gtin(left.get("Upc", "")), gtin(right.get("Upc", ""))
    lu, ru = listing_url(left.get("ProductUrl", "")), listing_url(right.get("ProductUrl", ""))
    ls, rs, lm, rm = sku(left), sku(right), model(left, a), model(right, b)
    return {"left_valid_gtin": lgt, "right_valid_gtin": rgt,
            "shared_valid_gtin": bool(lgt and lgt == rgt),
            "same_raw_upc": bool(str(left.get("Upc", "")).strip().casefold() not in MISSING
                                 and str(left.get("Upc", "")).strip().casefold()
                                 == str(right.get("Upc", "")).strip().casefold()),
            "left_scoped_sku": ls, "right_scoped_sku": rs,
            "shared_scoped_sku": bool(ls and ls == rs),
            "left_scoped_model": lm, "right_scoped_model": rm,
            "shared_scoped_model": bool(lm and lm == rm),
            "shared_listing_url": bool(lu and lu == ru)}


def text_overlap(left, right, shared_line_tokens=()):
    """Descriptive lexical evidence, independent of matching decisions.

    Generic overlap is explicitly separated from tokens outside the matcher's
    marketing-noise vocabulary. This is an inspection aid, not a semantic
    ground truth or a new grouping predicate. Full source text is also saved.
    """
    left, right = matching_text(left), matching_text(right)
    lw, rw = left.split(), right.split()
    ls, rs = set(lw), set(rw)
    shared = ls & rs
    union = ls | rs
    meaningful = sorted(shared - CORE_NOISE)
    def ngrams(words, n):
        return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}
    common_phrases = ngrams(lw, 5) & ngrams(rw, 5)
    return {"left_present": bool(left), "right_present": bool(right),
            "normalized_exact_copy": bool(left and left == right),
            "token_jaccard": len(shared)/len(union) if union else None,
            "shared_tokens": sorted(shared), "shared_non_marketing_tokens": meaningful,
            "shared_line_tokens_in_both": sorted(shared & set(shared_line_tokens)),
            "shared_five_word_phrases": sorted(common_phrases),
            "caution": "Lexical overlap may include boilerplate; inspect the supplied full texts."}


def _block_evidence(products, ids, config, pairs):
    frequencies = Counter(t for p in products for t in p.core)
    selected = []
    counts = Counter()
    for p in products:
        keys = set()
        if p.brand:
            tokens = sorted((t for t in p.core if len(t) >= 3 and not t.isdigit()),
                            key=lambda t: (frequencies[t], t))[:config.block_tokens]
            for token in tokens:
                keys.add(("brand", brand_block_key(p.brand), token))
                if p.tokens and p.family:
                    keys.add(("title_prefix", p.family.split()[0], token))
        selected.append(keys)
        counts.update(keys)
    result = {}
    for key, (a, b) in pairs.items():
        shared = sorted(selected[a] & selected[b])
        result[key] = {
            "left_selected_blocks": sorted(selected[a]),
            "right_selected_blocks": sorted(selected[b]),
            "shared_blocks": [{"key": block, "rows": counts[block],
                                "skipped_oversized": counts[block] > config.max_block_size}
                              for block in shared],
            "common_core_tokens": sorted(products[a].core & products[b].core,
                                        key=lambda t: (frequencies[t], t)),
            "common_core_token_document_frequencies": {
                t: frequencies[t] for t in sorted(products[a].core & products[b].core)},
            "informative_title_line_tokens": sorted(
                (products[a].core & products[b].core) - set(products[a].brand.split())
                - set(products[b].brand.split())),
        }
    return result


def observe_replay(frame, ids, config, watched_pairs):
    """Replay the actual implementation, restoring globals/trace on any exit.

    ``watched_pairs`` maps review IDs to input-row index pairs. A trace of the
    nested link function distinguishes pair guards from component guards and
    captures the component ingredient signatures at the time of the attempt.
    """
    if sys.gettrace() is not None:
        raise RuntimeError("Replay requires no pre-existing debugger/trace hook")
    products = prepare_products(frame, config)
    object_indices = {id(p): i for i, p in enumerate(products)}
    lookup = {tuple(sorted(value)): key for key, value in watched_pairs.items()}
    events = {key: [] for key in watched_pairs}
    counter = {"comparisons": 0, "links": 0}
    original = (split_rules_v2.prepare_products, split_rules_v2.name_similarity,
                split_rules_v2.compatible)
    source_lines, first_line = inspect.getsourcelines(split_rules_v2.refine_rule_groups)
    ingredient_line = first_line + next(i for i, line in enumerate(source_lines)
                                        if "if ingredient_conflict(a, b):" in line)
    rejection_line = first_line + next(i for i, line in enumerate(source_lines)
                                        if 'stats["component_ingredient_conflict_rejections"]' in line)
    source_file = split_rules_v2.refine_rule_groups.__code__.co_filename

    def observed_similarity(a, b):
        value = original[1](a, b)
        counter["comparisons"] += 1
        pair = tuple(sorted((object_indices[id(a)], object_indices[id(b)])))
        key = lookup.get(pair)
        if key:
            caller = sys._getframe(1)
            local = caller.f_locals
            guard = bool(local.get("name_guard", False))
            phase = local.get("reason", "blocked_family_name")
            edit = SequenceMatcher(None, *sorted((a.family, b.family)), autojunk=False).ratio()
            events[key].append({"event": "candidate_comparison", "comparison_ordinal": counter["comparisons"],
                "pass": phase, "caller": caller.f_code.co_name,
                "key": local.get("key"), "name_guard_enabled": guard,
                "name_jaccard": value, "family_edit_similarity": edit,
                "identifier_name_gate_passes": (value >= config.identifier_name_threshold) if guard else None,
                "family_name_gate_passes": value >= config.rule_family_threshold if phase == "blocked_family_name" else None,
                "family_edit_gate_evaluated": phase == "blocked_family_name" and value >= config.rule_family_threshold,
                "family_edit_gate_passes": (edit >= config.rule_family_edit_threshold
                    if phase == "blocked_family_name" and value >= config.rule_family_threshold else None)})
        return value

    def observed_compatible(a, b, active_config, allow_brand_override=False):
        value = original[2](a, b, active_config, allow_brand_override=allow_brand_override)
        pair = tuple(sorted((object_indices[id(a)], object_indices[id(b)])))
        key = lookup.get(pair)
        if key:
            evidence = compatibility_evidence(a, b, active_config, allow_brand_override)
            evidence.update(event="pair_compatibility", passes=value)
            events[key].append(evidence)
        return value

    def trace(frame_, event, value):
        if event != "call" or frame_.f_code.co_filename != source_file or frame_.f_code.co_name != "link":
            return None
        counter["links"] += 1
        local = frame_.f_locals
        key = lookup.get(tuple(sorted((local["a"], local["b"]))))
        if not key:
            return None
        snapshot = {"event": "link_result", "link_ordinal": counter["links"],
                    "pass": local["reason"], "component_guard_rejected": False}
        def local_trace(f, kind, returned):
            loc = f.f_locals
            if kind == "line" and f.f_lineno == ingredient_line:
                snapshot["left_component_root"] = ids[loc["root_a"]]
                snapshot["right_component_root"] = ids[loc["root_b"]]
                snapshot["left_component_size"] = loc["uf"].size[loc["root_a"]]
                snapshot["right_component_size"] = loc["uf"].size[loc["root_b"]]
                for side in ("a", "b"):
                    snapshot[f"{side}_component_ingredients"] = sorted(
                        sorted(s) for s in loc["component_ingredients"][loc[f"root_{side}"]])
            if kind == "line" and f.f_lineno == rejection_line:
                snapshot["component_guard_rejected"] = True
            if kind == "return":
                snapshot["accepted"] = bool(returned)
                events[key].append(snapshot)
            return local_trace
        return local_trace

    start = perf_counter()
    try:
        split_rules_v2.prepare_products = lambda input_frame, active_config: products
        split_rules_v2.name_similarity = observed_similarity
        split_rules_v2.compatible = observed_compatible
        sys.settrace(trace)
        groups, edges, stats = split_rules_v2.refine_rule_groups(frame, ids, config)
    finally:
        sys.settrace(None)
        (split_rules_v2.prepare_products, split_rules_v2.name_similarity,
         split_rules_v2.compatible) = original
    return groups, edges, stats, events, products, perf_counter()-start


def cause_from_events(events, blocks):
    """Use executed branches, not hypothetical match scores, for the cause."""
    if any(e.get("component_guard_rejected") for e in events):
        return "component_ingredient_guard"
    compatibility = [e for e in events if e["event"] == "pair_compatibility"]
    if compatibility:
        failures = sorted({f for e in compatibility for f in e["failed_guards"]})
        return "pair_compatibility:" + ",".join(failures)
    candidates = [e for e in events if e["event"] == "candidate_comparison"]
    if not candidates:
        return "candidate_blocking:oversized_only" if blocks["shared_blocks"] else "candidate_blocking:no_shared_selected_key"
    if any(e.get("identifier_name_gate_passes") is False for e in candidates):
        return "identifier_name_gate"
    if any(e.get("family_name_gate_passes") is False for e in candidates):
        return "blocked_family_jaccard_gate"
    if any(e.get("family_edit_gate_passes") is False for e in candidates):
        return "blocked_family_edit_gate"
    return "investigate_unexplained"


def explain_case(p, q, events, blocking, identifiers, desc, contents, config):
    """Plain-language findings tied to observed decisions and source evidence."""
    cause = cause_from_events(events, blocking)
    score = name_similarity(p, q)
    edit = SequenceMatcher(None, *sorted((p.family, q.family)), autojunk=False).ratio()
    if not events:
        if p.family == q.family and (not p.core or not q.core):
            actual = ("The pack-free family titles are equal, but the exact-family pass requires a nonempty "
                      "informative core on each record. Brand, quantity, and marketing-token removal leaves "
                      "an empty core, so that pass does not admit the pair and neither record proposes fuzzy blocks.")
        else:
            actual = ("The records share title evidence, but their two globally least-frequent eligible core tokens "
                      "do not create any shared selected brand/token or title-prefix/token key. The pair is never compared.")
    elif cause == "blocked_family_jaccard_gate":
        actual = (f"The pair is proposed and its family-token Jaccard {score:.6f} is below the frozen "
                  f"{config.rule_family_threshold:.2f} requirement. The edit and compatibility gates are not reached.")
    elif cause == "blocked_family_edit_gate":
        actual = (f"The pair is proposed and family-token Jaccard {score:.6f} passes. Its sorted-family "
                  f"character edit similarity {edit:.6f} is below {config.rule_family_edit_threshold:.2f}; "
                  "the compatibility and component gates are not reached.")
    else:
        actual = "Observed branch: " + cause
    common = blocking["common_core_tokens"]
    name_line = ("Shared informative name tokens: " + ", ".join(common) if common
                 else "The canonical brand and normalized family title provide the name evidence; no core tokens survive.")
    identifier = ("No shared intact checksum-valid GTIN, retailer-scoped SKU, brand-scoped model, or listing URL.")
    if identifiers["same_raw_upc"] and not identifiers["shared_valid_gtin"]:
        identifier += " The raw UPC strings match, but scientific notation/truncated formatting is not an admissible GTIN."
    if not desc["left_present"] or not desc["right_present"]:
        description = "A description is missing on at least one side; copied-description evidence cannot be assessed for this pair."
    else:
        description = (f"Both descriptions are available; token Jaccard is {desc['token_jaccard']:.6f} with "
                       f"{len(desc['shared_five_word_phrases'])} shared five-word phrases. The full text is saved for separating product evidence from boilerplate.")
    if contents["left_present"] and contents["right_present"]:
        description += (f" Both contents fields are available; token Jaccard is {contents['token_jaccard']:.6f}, "
                        f"with {len(contents['shared_five_word_phrases'])} shared five-word phrases.")
    description += " Rule v2 does not use descriptions or contents to positively corroborate a match; contents only feed an ingredient veto."
    return {"executed_branch_explanation": actual, "name_line_evidence": name_line,
            "identifier_explanation": identifier, "supporting_text_explanation": description,
            "compatibility_guard_reached": any(e["event"] == "pair_compatibility" for e in events),
            "left_only_family_tokens": sorted(p.tokens-q.tokens),
            "right_only_family_tokens": sorted(q.tokens-p.tokens)}


def investigate_misses(run_dir, outcomes_path, output_dir):
    run_dir, outcomes_path, output_dir = map(Path, (run_dir, outcomes_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    freeze_before = verify_refinement_seal(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = SplitConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if config.grouping_version != 2:
        raise ValueError("This observer targets the frozen version-2 rule implementation")
    outcomes = _read(outcomes_path)
    selected = outcomes.loc[outcomes.review_category.eq("related")
                            & outcomes.rule_false_negative.str.casefold().eq("true")].copy()
    if selected.empty:
        raise ValueError("No clear related rule misses in reviewed outcomes")
    frame, metadata = load_input(Path(summary["input"]["path"]),
                                  Path(summary["input"]["identifier_source"]))
    # Remove every target before replay/inspection, preserving only grouping fields.
    view = grouping_view(frame)
    ids = frame.record_id.to_numpy()
    indices = {key: i for i, key in enumerate(ids)}
    watched = {row.pair_id: (indices[row.left_record_id], indices[row.right_record_id])
               for row in selected.itertuples(index=False)}
    groups, edges, stats, events, products, seconds = observe_replay(view, ids, config, watched)
    assignments = _read(run_dir / "rule_assignments.csv").set_index("record_id").loc[ids]
    if not np.array_equal(groups, assignments.group_id.to_numpy()):
        raise ValueError("Observed replay changed saved rule groups")
    accepted = pd.DataFrame([{"left_record_id": ids[a], "right_record_id": ids[b],
                             "reason": reason, "score": score} for a, b, reason, score in edges])
    saved_edges = pd.read_csv(run_dir / "rule_matching_edges.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(accepted, saved_edges)
    if accepted.to_csv(index=False).encode("utf-8") != (run_dir / "rule_matching_edges.csv").read_bytes():
        raise ValueError("Observed replay changed byte-for-byte edge serialization")
    blocks = _block_evidence(products, ids, config, watched)
    rows, detailed = [], []
    for row in selected.itertuples(index=False):
        a, b = watched[row.pair_id]
        left, right = view.iloc[a].to_dict(), view.iloc[b].to_dict()
        p, q = products[a], products[b]
        identifiers = identifier_evidence(left, right, p, q)
        cause = cause_from_events(events[row.pair_id], blocks[row.pair_id])
        if cause == "investigate_unexplained":
            raise ValueError(f"No causal branch identified for {row.pair_id}")
        desc = text_overlap(left["ProductDescription"], right["ProductDescription"], p.core & q.core)
        contents = text_overlap(left["ProductContents"], right["ProductContents"], p.core & q.core)
        explanation = explain_case(p, q, events[row.pair_id], blocks[row.pair_id], identifiers, desc, contents, config)
        evidence = {"pair_id": row.pair_id, "left_record_id": ids[a], "right_record_id": ids[b],
                    "left": left, "right": right, "left_comparison_evidence": {
                        "canonical_brand": p.brand, "name": p.name, "family": p.family,
                        "core": sorted(p.core), "ingredient_signature": sorted(p.ingredients)},
                    "right_comparison_evidence": {
                        "canonical_brand": q.brand, "name": q.name, "family": q.family,
                        "core": sorted(q.core), "ingredient_signature": sorted(q.ingredients)},
                    "identifiers": identifiers, "blocking": blocks[row.pair_id],
                    "executed_events": events[row.pair_id], "cause": cause,
                    "description_evidence": desc, "contents_evidence": contents,
                    "findings": explanation,
                    "text_policy": "Descriptions do not propose or confirm rule-v2 links. Contents only supply limited active-ingredient veto signatures.",
                    "reviewed_leakage_rationale": row.leakage_rationale}
        detailed.append(evidence)
        flat = {"pair_id": row.pair_id, "left_record_id": ids[a], "right_record_id": ids[b],
                "left_ProductName": left["ProductName"], "right_ProductName": right["ProductName"],
                "left_ProductBrand": left["ProductBrand"], "right_ProductBrand": right["ProductBrand"],
                "left_canonical_brand": p.brand, "right_canonical_brand": q.brand,
                "left_Upc": left["Upc"], "right_Upc": right["Upc"],
                "left_Sku": left["Sku"], "right_Sku": right["Sku"],
                "left_ProductModelNumber": left["ProductModelNumber"], "right_ProductModelNumber": right["ProductModelNumber"],
                "left_Retailer": left["Retailer"], "right_Retailer": right["Retailer"],
                "name_jaccard": name_similarity(p, q),
                "family_edit_similarity": SequenceMatcher(None, *sorted((p.family,q.family)), autojunk=False).ratio(),
                "cause": cause, "candidate_comparisons": sum(e["event"] == "candidate_comparison" for e in events[row.pair_id]),
                "link_attempts": sum(e["event"] == "link_result" for e in events[row.pair_id]),
                "common_core_tokens": "|".join(blocks[row.pair_id]["common_core_tokens"]),
                "shared_blocks": json.dumps(blocks[row.pair_id]["shared_blocks"], ensure_ascii=False),
                "description_both_present": desc["left_present"] and desc["right_present"],
                "description_exact_copy": desc["normalized_exact_copy"],
                "description_token_jaccard": desc["token_jaccard"],
                "description_shared_five_word_phrases": len(desc["shared_five_word_phrases"]),
                "contents_both_present": contents["left_present"] and contents["right_present"],
                "contents_exact_copy": contents["normalized_exact_copy"],
                "contents_token_jaccard": contents["token_jaccard"],
                "contents_shared_five_word_phrases": len(contents["shared_five_word_phrases"]),
                "reviewed_leakage_rationale": row.leakage_rationale,
                **{key: value for key, value in explanation.items() if not isinstance(value, list)},
                **{key: value for key, value in identifiers.items()
                   if key not in {"left_scoped_sku", "right_scoped_sku", "left_scoped_model", "right_scoped_model"}}}
        for side, source in (("left", left), ("right", right)):
            flat[f"{side}_ProductDescription"] = source["ProductDescription"]
            flat[f"{side}_ProductContents"] = source["ProductContents"]
        rows.append(flat)
    freeze_after = verify_refinement_seal(run_dir)
    output_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output_dir / "rule_miss_diagnostics.csv", index=False)
    (output_dir / "rule_miss_evidence.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    report = {"investigation_version": "frozen-rule-miss-observation-v1",
              "review_cases": len(rows), "records_replayed": len(frame),
              "cause_counts": dict(Counter(row["cause"] for row in rows)),
              "shared_valid_gtin_pairs": sum(row["shared_valid_gtin"] for row in rows),
              "shared_scoped_sku_pairs": sum(row["shared_scoped_sku"] for row in rows),
              "shared_scoped_model_pairs": sum(row["shared_scoped_model"] for row in rows),
              "same_raw_upc_pairs": sum(row["same_raw_upc"] for row in rows),
              "both_descriptions_available_pairs": sum(row["description_both_present"] for row in rows),
              "both_contents_available_pairs": sum(row["contents_both_present"] for row in rows),
              "rule_replay_seconds": seconds, "accepted_edges": len(edges),
              "rule_replay_statistics": stats,
              "all_saved_groups_equal": True, "all_saved_edges_equal_in_order_and_score": True,
              "freeze_verified_before_and_after": True,
              "targets_removed_before_grouping": True, "thresholds_changed": False,
              "input": metadata, "input_outcomes_sha256": sha256(outcomes_path),
              "observer_sha256": sha256(Path(__file__)),
              "frozen_rule_source_sha256": sha256(Path(split_rules_v2.__file__)),
              "configuration_sha256": sha256(run_dir/"config.json"),
              "freeze_checks": {"before": freeze_before, "after": freeze_after},
              "interpretation": "Purposively selected related pairs; explanations apply to these misses, not population prevalence.",
              "architecture": "Read-only grouping investigation. The target-specific splitting engine and persisted assignments are untouched.",
              "outputs_sha256": {name: sha256(output_dir/name) for name in ("rule_miss_diagnostics.csv", "rule_miss_evidence.json")}}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = investigate_misses(args.run_dir, args.outcomes, args.output_dir)
    print(json.dumps({key: result[key] for key in ("review_cases", "cause_counts", "shared_valid_gtin_pairs",
        "shared_scoped_sku_pairs", "shared_scoped_model_pairs", "same_raw_upc_pairs",
        "all_saved_groups_equal", "all_saved_edges_equal_in_order_and_score")}, indent=2))
    return 0
