"""Read-only evidence inspection of a frozen rule-refinement partition.

Size and heterogeneous heads are review triggers, never grouping errors or
automatic cuts. Common anchors and representative checks document mechanical
consistency; only inspection can judge whether those anchors are meaningful.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from time import perf_counter

import pandas as pd

from .features import sha256
from .rule_refinement_experiment import check_edge_components, verify_rule_refinement_freeze
from .split_matching import matching_text
from .split_rule_evidence_v3 import prepare_rule_products_v3, remove_negative_claims
from .split_rules_v3 import RuleRefinementConfig, pair_evidence_v3
from .splitting import SplitConfig, grouping_view
from .splitting_experiment import load_input

VERSION = "frozen-rule-group-evidence-audit-v1"
ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("ProductName", "ProductBrand", "ProductDescription", "ProductContents",
          "Sku", "Upc", "ProductModelNumber", "Retailer", "ProductUrl")
GROUP_COLUMNS = ["method", "group_id", "group_size", "direct_edges", "implied_pairs",
    "transitive_only_pairs", "implied_to_direct_pair_ratio", "direct_graph_density",
    "common_anchor_count", "common_word_anchor_count", "common_payload_anchor_count",
    "common_exact_branded_anchor_count", "common_anchors_json", "distinct_core_heads",
    "core_head_heterogeneity", "core_heads_json", "canonical_brands_json", "empty_core_rows",
    "accepted_reasons_json", "representative_checked_pairs", "representative_unsupported_pairs",
    "minimum_member_representative_support", "negative_claim_rows", "negative_core_overlap_rows",
    "negative_core_overlap_tokens_json", "review_trigger"]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _negative_evidence(value):
    """Literal claim overlap is an inspection flag, not a parsing verdict.

    Positive occurrences elsewhere can justify a shared token. Raw phrases are
    saved so that reviewers can assess this instead of treating flags as errors.
    This helper never proposes or removes a match.
    """
    value = str(value)
    prefix = re.compile(r"\b(?:no|without|free\s+from|free\s+of)\s+(.+?)"
        r"(?=\b(?:with|contains|containing|provides|providing)\b|[|;,().]|\s+[-–—]\s+|$)", re.I)
    phrases = [match.group(0).strip() for match in prefix.finditer(value)]
    phrases.extend(match.group(0) for match in re.finditer(
        r"\b(?:\w+(?:\s+(?:and|or)\s+\w+)*)[-\s]+free\b", value, re.I))
    tokens = frozenset(token for phrase in phrases for token in matching_text(phrase).split())
    return tuple(sorted(set(phrases))), tokens


def _representatives(members, products, ids, count):
    """Replicate the frozen deterministic diversity sampling for inspection."""
    ordered = sorted(members, key=lambda i: ids[i])
    chosen = [ordered[0]]
    if len(ordered) > 1 and count > 1:
        chosen.append(ordered[-1])
    while len(chosen) < min(count, len(ordered)):
        remaining = [i for i in ordered if i not in chosen]
        def priority(i):
            similarity = max(len(products[i].core & products[j].core) /
                max(1, len(products[i].core | products[j].core)) for j in chosen)
            return similarity, ids[i]
        chosen.append(min(remaining, key=priority))
    return chosen


def audit_group_partition(frame, assignment, edges, split_config, rule_config, *, top_groups=10):
    """Audit every non-singleton with product attributes and graph inputs only.

    Returns a mapping of deterministic CSV tables and a small proof summary.
    Label/split columns supplied by callers are projected away before use.
    """
    if type(top_groups) is not int or top_groups < 1:
        raise ValueError("top_groups must be a positive integer")
    if "record_id" not in frame or not {"record_id", "group_id"} <= set(assignment):
        raise ValueError("records and assignments must contain immutable identities")
    ids = frame.record_id.astype(str).tolist()
    if len(ids) != len(set(ids)) or any(not record.strip() for record in ids):
        raise ValueError("record IDs must be nonblank and unique")
    assignment = assignment[["record_id", "group_id"]].astype(str).copy()
    if (assignment.record_id.duplicated().any() or set(assignment.record_id) != set(ids)
            or assignment.group_id.str.strip().eq("").any()):
        raise ValueError("assignment identities must cover the input once with nonblank groups")
    assignment = assignment.set_index("record_id").loc[ids].reset_index()
    required = ["left_record_id", "right_record_id", "reason", "score"]
    if not set(required) <= set(edges):
        raise ValueError("accepted edges lack endpoint or evidence fields")
    edges = edges[required].copy()
    edge_pairs = [tuple(sorted((str(a), str(b)))) for a, b in
                  edges[["left_record_id", "right_record_id"]].itertuples(index=False, name=None)]
    if len(edge_pairs) != len(set(edge_pairs)):
        raise ValueError("accepted-edge audit requires one saved row per direct pair")
    check_edge_components(assignment, edges)
    features = grouping_view(frame)
    products, evidence_stats = prepare_rule_products_v3(features, split_config, rule_config)
    by_group = defaultdict(list)
    positions = {record: i for i, record in enumerate(ids)}
    for i, group in enumerate(assignment.group_id):
        by_group[group].append(i)
    reason_counts = defaultdict(Counter)
    direct_counts = Counter()
    for left, right, reason, score in edges.itertuples(index=False, name=None):
        group = assignment.at[positions[left], "group_id"]
        direct_counts[group] += 1
        reason_counts[group][str(reason)] += 1
    largest = sorted((group for group, members in by_group.items() if len(members) > 1),
                     key=lambda group: (-len(by_group[group]), group))[:top_groups]
    rank = {group: i + 1 for i, group in enumerate(largest)}
    representatives = {group: _representatives(by_group[group], products, ids,
                          rule_config.component_representatives) for group in largest}
    rows, member_rows, unsupported_rows = [], [], []
    common_by_group = {}
    for group in sorted(by_group):
        members = sorted(by_group[group], key=lambda i: ids[i])
        if len(members) == 1:
            continue
        common = set(products[members[0]].anchors)
        for i in members[1:]:
            common.intersection_update(products[i].anchors)
        if not common:
            raise ValueError(f"group lacks an anchor common to every member: {group}")
        common_by_group[group] = frozenset(common)
        heads = Counter(products[i].head for i in members)
        implied = len(members) * (len(members) - 1) // 2
        direct = direct_counts[group]
        negative_rows, overlap_rows, overlap_tokens = 0, 0, set()
        audit_representatives = _representatives(members, products, ids, rule_config.component_representatives)
        representative_checks, unsupported = 0, 0
        minimum_support = 1.0
        checked_pairs = {}
        for i in members:
            raw = features.iloc[i].ProductName
            phrases, negative_tokens = _negative_evidence(raw)
            overlap = negative_tokens & products[i].core
            negative_rows += bool(phrases)
            overlap_rows += bool(overlap)
            overlap_tokens.update(overlap)
            member_support = []
            for j in audit_representatives:
                if i == j:
                    member_support.append(True)
                    continue
                key = tuple(sorted((i, j)))
                if key not in checked_pairs:
                    checked_pairs[key] = bool(pair_evidence_v3(products[i], products[j], rule_config))
                    representative_checks += 1
                    unsupported += not checked_pairs[key]
                    if not checked_pairs[key]:
                        a, b = sorted((i, j), key=lambda index: ids[index])
                        unsupported_rows.append({"group_id": group, "group_size": len(members),
                            "left_record_id": ids[a], "right_record_id": ids[b],
                            "left_ProductName": features.iloc[a].ProductName,
                            "right_ProductName": features.iloc[b].ProductName,
                            "left_core_json": _json(sorted(products[a].core)),
                            "right_core_json": _json(sorted(products[b].core)),
                            "common_group_anchors_json": _json(sorted(common)),
                            "interpretation": "Final representative has no direct pair confirmation; this flags possible residual chaining, not an independent leakage judgment."})
                member_support.append(checked_pairs[key])
            minimum_support = min(minimum_support, sum(member_support) / len(member_support))
            if group in rank:
                member_rows.append({"method": "rule_v3", "largest_group_rank": rank[group],
                    "group_id": group, "group_size": len(members), "record_id": ids[i],
                    "is_representative": i in representatives[group],
                    "representative_order": representatives[group].index(i) + 1
                        if i in representatives[group] else 0,
                    **{field: features.iloc[i].get(field, "") for field in FIELDS},
                    "comparison_name": products[i].name, "canonical_brand": products[i].brand,
                    "core_head_json": _json(products[i].head), "core_tokens_json": _json(sorted(products[i].core)),
                    "strengths_json": _json(sorted(products[i].strengths)),
                    "common_group_anchors_json": _json(sorted(common)),
                    "member_anchors_json": _json(sorted(products[i].anchors)),
                    "negative_claim_phrases_json": _json(phrases),
                    "negative_claim_core_overlap_json": _json(sorted(overlap)),
                    "negative_claim_removed_from_name": matching_text(raw)
                        != matching_text(remove_negative_claims(raw))})
        rows.append({"method": "rule_v3", "group_id": group, "group_size": len(members),
            "direct_edges": direct, "implied_pairs": implied, "transitive_only_pairs": implied - direct,
            "implied_to_direct_pair_ratio": implied / direct, "direct_graph_density": direct / implied,
            "common_anchor_count": len(common), "common_word_anchor_count": sum(x.startswith("word:") for x in common),
            "common_payload_anchor_count": sum(x.startswith("payload:") for x in common),
            "common_exact_branded_anchor_count": sum(x.startswith("branded_exact:") for x in common),
            "common_anchors_json": _json(sorted(common)), "distinct_core_heads": len(heads),
            "core_head_heterogeneity": 1 - max(heads.values()) / len(members),
            "core_heads_json": _json({" ".join(head): count for head, count in sorted(heads.items())}),
            "canonical_brands_json": _json(sorted({products[i].brand for i in members})),
            "empty_core_rows": sum(not products[i].core for i in members),
            "accepted_reasons_json": _json(dict(sorted(reason_counts[group].items()))),
            "representative_checked_pairs": representative_checks,
            "representative_unsupported_pairs": unsupported,
            "minimum_member_representative_support": minimum_support,
            "negative_claim_rows": negative_rows, "negative_core_overlap_rows": overlap_rows,
            "negative_core_overlap_tokens_json": _json(sorted(overlap_tokens)),
            "review_trigger": ";".join(flag for condition, flag in (
                (group in rank, "largest_group"), (len(heads) > 1, "heterogeneous_heads"),
                (unsupported > 0, "final_representative_inconsistency"),
                (overlap_rows > 0, "negative_core_overlap")) if condition)})
    groups = pd.DataFrame(rows, columns=GROUP_COLUMNS).sort_values(
        ["group_size", "group_id"], ascending=[False, True]).reset_index(drop=True)
    member_columns = ["method", "largest_group_rank", "group_id", "group_size", "record_id",
        "is_representative", "representative_order", *FIELDS, "comparison_name", "canonical_brand",
        "core_head_json", "core_tokens_json", "strengths_json", "common_group_anchors_json",
        "member_anchors_json", "negative_claim_phrases_json", "negative_claim_core_overlap_json",
        "negative_claim_removed_from_name"]
    members_frame = pd.DataFrame(member_rows, columns=member_columns).sort_values(
        ["largest_group_rank", "record_id"]).reset_index(drop=True)
    representatives_frame = members_frame.loc[members_frame.is_representative].sort_values(
        ["largest_group_rank", "representative_order"]).reset_index(drop=True)
    named_edges = []
    for left, right, reason, score in edges.itertuples(index=False, name=None):
        group = assignment.at[positions[left], "group_id"]
        if group in rank:
            a, b = sorted((left, right))
            named_edges.append({"largest_group_rank": rank[group], "group_id": group,
                "left_record_id": a, "right_record_id": b,
                "left_ProductName": features.iloc[positions[a]].ProductName,
                "right_ProductName": features.iloc[positions[b]].ProductName,
                "accepted_reason": reason, "accepted_score": float(score)})
    edge_columns = ["largest_group_rank", "group_id", "left_record_id", "right_record_id",
                    "left_ProductName", "right_ProductName", "accepted_reason", "accepted_score"]
    named = pd.DataFrame(named_edges, columns=edge_columns).sort_values(
        ["largest_group_rank", "left_record_id", "right_record_id"]).reset_index(drop=True)
    unsupported_columns = ["group_id", "group_size", "left_record_id", "right_record_id",
        "left_ProductName", "right_ProductName", "left_core_json", "right_core_json",
        "common_group_anchors_json", "interpretation"]
    unsupported_table = pd.DataFrame(unsupported_rows, columns=unsupported_columns).sort_values(
        ["group_size", "group_id", "left_record_id", "right_record_id"],
        ascending=[False, True, True, True]).reset_index(drop=True)
    summary = {"records": len(ids), "groups": len(by_group), "non_singleton_groups": len(rows),
        "singleton_groups": sum(len(members) == 1 for members in by_group.values()),
        "largest_group": max(map(len, by_group.values()), default=0), "accepted_direct_edges": len(edges),
        "implied_pairs": int(groups.implied_pairs.sum()),
        "transitive_only_pairs": int(groups.transitive_only_pairs.sum()),
        "heterogeneous_head_groups": int(groups.distinct_core_heads.gt(1).sum()),
        "groups_with_final_representative_unsupported_flags": int(groups.representative_unsupported_pairs.gt(0).sum()),
        "final_representative_unsupported_pairs": int(groups.representative_unsupported_pairs.sum()),
        "groups_with_negative_core_overlap_flags": int(groups.negative_core_overlap_rows.gt(0).sum()),
        "largest_groups_presented": len(largest), "largest_group_member_rows": len(members_frame),
        "representative_rows": len(representatives_frame), "evidence_preparation": evidence_stats,
        "proof": {"accepted_edges_reconstruct_saved_partition": True,
                  "all_non_singleton_groups_have_common_anchor": True, "target_labels_used": False,
                  "group_assignments_changed": False, "group_size_cap_applied": False},
        "interpretation": "Common-anchor and graph checks prove mechanical consistency, not semantic correctness. Size, head diversity and negative-claim overlap are review flags; positive reoccurrences can explain overlap."}
    return {"group_audit.csv": groups, "largest_group_representatives.csv": representatives_frame,
            "largest_group_all_members.csv": members_frame, "largest_group_named_edges.csv": named,
            "final_representative_unsupported_pairs.csv": unsupported_table}, summary


def audit_frozen_rule_groups(run_dir, output_dir, *, top_groups=10):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    seal = verify_rule_refinement_freeze(run_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite group audit: {output_dir}")
    freeze = json.loads((run_dir / "rule_refinement_freeze.json").read_text(encoding="utf-8"))
    protected_dirs = [run_dir, Path(freeze["baseline_dir"]).resolve(), Path(freeze["reservation_dir"]).resolve()]
    if any(output_dir.is_relative_to(path) for path in protected_dirs):
        raise ValueError("audit output must be outside frozen experiment and reservation directories")
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    frame, actual = load_input(Path(manifest["path"]),
        Path(manifest["identifier_source"]) if manifest["identifier_source"] else None)
    if manifest != actual:
        raise ValueError("audit input differs from the frozen shared input")
    split_config = SplitConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    rule_config = RuleRefinementConfig.from_dict(json.loads(
        (run_dir / "rule_refinement_config.json").read_text(encoding="utf-8")))
    assignment = pd.read_csv(run_dir / "rule_assignments.csv", usecols=["record_id", "group_id"],
                             dtype=str, keep_default_na=False)
    edges = pd.read_csv(run_dir / "rule_matching_edges.csv", dtype=str, keep_default_na=False)
    started = perf_counter()
    tables, summary = audit_group_partition(frame, assignment, edges, split_config, rule_config, top_groups=top_groups)
    elapsed = perf_counter() - started
    verify_rule_refinement_freeze(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, lineterminator="\n")
    input_paths = [run_dir / name for name in ("rule_assignments.csv", "rule_matching_edges.csv", "config.json",
                 "rule_refinement_config.json", "input_manifest.json", "rule_refinement_freeze.json")]
    input_paths.extend(map(Path, manifest["fingerprints"]))
    source_paths = [Path(__file__), ROOT / "scripts/audit_rule_refinement_groups.py",
                    Path(__file__).parent / "split_rule_evidence_v3.py",
                    Path(__file__).parent / "split_rules_v3.py"]
    summary.update({"version": VERSION, "run_dir": str(run_dir), "method": "rule_v3",
        "schema": {"group_audit.csv": GROUP_COLUMNS, "product_fields": list(FIELDS)},
        "top_groups": top_groups, "runtime_seconds": elapsed, "frozen_input_verified": seal,
        "input_sha256": {str(path.resolve()): sha256(path) for path in input_paths},
        "code_sha256": {str(path.resolve()): sha256(path) for path in source_paths},
        "output_sha256": {name: sha256(output_dir / name) for name in tables}})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n", encoding="utf-8")
    verify_rule_refinement_freeze(run_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-groups", type=int, default=10)
    args = parser.parse_args(argv)
    summary = audit_frozen_rule_groups(args.run_dir, args.output_dir, top_groups=args.top_groups)
    print(json.dumps({key: summary[key] for key in ("records", "groups", "non_singleton_groups",
        "largest_group", "accepted_direct_edges", "implied_pairs", "transitive_only_pairs", "proof")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
