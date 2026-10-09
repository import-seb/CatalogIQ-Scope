"""Outcome-blind lexical candidates and independently reviewed balanced samples.

Retrieval uses only ProductName and ProductBrand, never grouping assignments,
split membership, targets, frozen matcher helpers, or saved method outcomes.
Provisional lexical buckets are retrieval hints, not inferred gold labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .features import sha256
from .splitting_experiment import load_input

SEED = 20261011
PAIR_COLUMNS = ["left_record_id", "right_record_id"]
EVIDENCE_FIELDS = ["ProductName", "ProductBrand", "ProductDescription", "ProductContents",
                   "Sku", "Upc", "ProductModelNumber", "Retailer", "ProductUrl"]
CATEGORIES = ("related", "safe_to_separate", "borderline")
BUCKETS = ("same_line_candidate", "distinct_core_candidate", "ambiguous_lexical_candidate")
REVIEW_FIELDS = ["product_identity", "identity_detail", "identity_confidence", "identity_rationale",
                 "leakage_decision", "leakage_confidence", "leakage_rationale", "review_category",
                 "judgment_uses_Segment", "judgment_uses_method_outcomes"]
_PACKAGING = {"pack", "packs", "count", "ct", "oz", "ounce", "ounces", "fl", "ml", "mg", "g",
              "gram", "grams", "capsule", "capsules", "tablet", "tablets", "softgel", "softgels",
              "bottle", "bottles", "of", "with", "and", "the", "for", "to", "a", "an"}


@dataclass(frozen=True)
class CandidateConfig:
    seed: int = SEED
    per_bucket: int = 70
    brand_record_cap: int = 128
    cross_brand_block_cap: int = 48
    max_pairs_per_brand_bucket: int = 5


def normalize_raw(value):
    """Simple text normalization independent of either saved matcher."""
    if value is None or str(value).strip().casefold() in {"", "null", "nan", "none"}:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def _rank(seed, *values):
    return hashlib.sha256((str(seed) + "\0" + "\0".join(map(str, values))).encode("utf-8")).hexdigest()


def _name_words(name, brand):
    brand_words = set(brand.split())
    return tuple(word for word in name.split()
                 if word not in brand_words and word not in _PACKAGING and not word.isdecimal())


def family_keys(name, brand):
    """Exact packaging-insensitive name and short title-prefix families.

    These deliberately simple lexical families are exclusion/deduplication
    guards, not product equivalence or leakage judgments.
    """
    words = _name_words(normalize_raw(name), normalize_raw(brand))
    prefix = words[:3]
    normalized_brand = normalize_raw(brand)
    return (normalized_brand + "|" + " ".join(words),
            normalized_brand + "|" + " ".join(prefix))


def _char3(value):
    value = " " + value + " "
    return frozenset(value[i:i + 3] for i in range(max(0, len(value) - 2)))


def lexical_similarity(left, right):
    """Raw normalized-name character-trigram Jaccard and token Dice."""
    a, b = normalize_raw(left), normalize_raw(right)
    grams_a, grams_b = _char3(a), _char3(b)
    tokens_a, tokens_b = set(a.split()), set(b.split())
    union = grams_a | grams_b
    return (len(grams_a & grams_b) / len(union) if union else 0.0,
            2 * len(tokens_a & tokens_b) / (len(tokens_a) + len(tokens_b))
            if tokens_a or tokens_b else 0.0)


def _validate_records(frame):
    required = {"record_id", "ProductName", "ProductBrand"}
    if not required <= set(frame):
        raise ValueError(f"missing record columns: {sorted(required - set(frame))}")
    if frame.record_id.astype(str).str.strip().eq("").any() or frame.record_id.duplicated().any():
        raise ValueError("record IDs must be nonempty and unique")


def _prior_endpoints(prior):
    if not set(PAIR_COLUMNS) <= set(prior):
        raise ValueError("prior reviews must contain both record endpoints")
    return set(prior.left_record_id.astype(str)) | set(prior.right_record_id.astype(str))


def assemble_evidence(frame, pairs):
    """Attach full permitted product evidence; never copy target/method columns."""
    _validate_records(frame)
    if not set(PAIR_COLUMNS) <= set(pairs):
        raise ValueError("pairs must contain both endpoints")
    source = frame.set_index("record_id")
    result = pairs.copy()
    for side in ("left", "right"):
        ids = result[f"{side}_record_id"]
        if not set(ids) <= set(source.index):
            raise ValueError("candidate endpoint absent from input")
        for field in EVIDENCE_FIELDS:
            result[f"{side}_{field}"] = ids.map(source[field]) if field in source else ""
    return result


def blind_evidence(pool):
    """Only neutral pair/record IDs and complete product evidence reach reviewers."""
    columns = ["pair_id", *PAIR_COLUMNS,
               *(f"{side}_{field}" for side in ("left", "right") for field in EVIDENCE_FIELDS)]
    if not set(columns) <= set(pool):
        raise ValueError("candidate pool lacks complete evidence")
    return pool[columns].to_dict(orient="records")


def candidate_pool(frame, prior, config=None):
    """Return independent lexical candidates and an auditable retrieval report.

    Large brand/prefix blocks are deterministically capped before comparison;
    no uncapped global all-pairs matrix is built. All prior endpoints, their
    simple exact/short-title families, repeat endpoints, and repeat short-title
    families are excluded. Raw brand spellings are not alias-canonicalized.
    """
    config = config or CandidateConfig()
    if min(config.per_bucket, config.brand_record_cap, config.cross_brand_block_cap,
           config.max_pairs_per_brand_bucket) < 1:
        raise ValueError("candidate caps must be positive")
    _validate_records(frame)
    prior_ids = _prior_endpoints(prior)
    known = frame.set_index("record_id")
    if not prior_ids <= set(known.index):
        raise ValueError("prior review endpoint absent from input")
    excluded_exact, excluded_prefix = set(), set()
    for endpoint in prior_ids:
        row = known.loc[endpoint]
        exact, prefix = family_keys(row.ProductName, row.ProductBrand)
        excluded_exact.add(exact)
        excluded_prefix.add(prefix)
    # Project explicitly before retrieval so target/outcome fields cannot affect it.
    rows = []
    excluded_family_rows = 0
    for row in frame[["record_id", "ProductName", "ProductBrand"]].itertuples(index=False):
        if row.record_id in prior_ids:
            continue
        brand, name = normalize_raw(row.ProductBrand), normalize_raw(row.ProductName)
        if not brand or not name:
            continue
        exact, prefix = family_keys(name, brand)
        if exact in excluded_exact or prefix in excluded_prefix:
            excluded_family_rows += 1
            continue
        words = _name_words(name, brand)
        if not words:
            continue
        rows.append({"id": row.record_id, "brand": brand, "name": name, "exact": exact,
                     "prefix": prefix, "words": words, "grams": _char3(name),
                     "tokens": frozenset(name.split())})
    by_brand, by_prefix = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        by_brand[row["brand"]].append(i)
        # Brand-independent name prefix retrieves possible generic aliases.
        by_prefix[" ".join(row["words"][:3])].append(i)
    seeds = {}
    capped_brand_blocks, capped_prefix_blocks = 0, 0
    comparisons = 0

    def compare(a, b, origin):
        nonlocal comparisons
        left, right = rows[a], rows[b]
        if left["id"] == right["id"]:
            return
        key = tuple(sorted((left["id"], right["id"])))
        if key in seeds:
            return
        comparisons += 1
        union = left["grams"] | right["grams"]
        jaccard = len(left["grams"] & right["grams"]) / len(union) if union else 0.0
        dice = 2 * len(left["tokens"] & right["tokens"]) / (len(left["tokens"]) + len(right["tokens"]))
        same_brand = left["brand"] == right["brand"]
        different_core = left["words"][:2] != right["words"][:2]
        core_delta = len(set(left["words"]) ^ set(right["words"]))
        if same_brand and jaccard >= .68 and dice >= .70:
            bucket = BUCKETS[0]
        elif (same_brand and different_core and core_delta >= 3
              and .28 <= jaccard <= .72 and .25 <= dice <= .78
              and left["prefix"] != right["prefix"]):
            bucket = BUCKETS[1]
        elif ((same_brand and .48 <= jaccard < .85 and dice >= .45)
              or (not same_brand and jaccard >= .60 and dice >= .60)):
            bucket = BUCKETS[2]
        else:
            return
        ordered = (left, right) if left["id"] < right["id"] else (right, left)
        seeds[key] = {"left_record_id": key[0], "right_record_id": key[1],
                      "provisional_stratum": bucket, "retrieval_origin": origin,
                      "name_char3_jaccard": round(jaccard, 6), "name_token_dice": round(dice, 6),
                      "left_family_key": ordered[0]["prefix"], "right_family_key": ordered[1]["prefix"],
                      "left_exact_family_key": ordered[0]["exact"], "right_exact_family_key": ordered[1]["exact"],
                      "raw_brand_block": "|".join(sorted({left["brand"], right["brand"]})),
                      "selection_rank": _rank(config.seed, *key)}

    for brand, group in sorted(by_brand.items()):
        group = sorted(group, key=lambda i: _rank(config.seed, rows[i]["id"]))
        if len(group) > config.brand_record_cap:
            capped_brand_blocks += 1
        group = group[:config.brand_record_cap]
        for a, b in itertools.combinations(group, 2):
            compare(a, b, "same_raw_brand")
    for prefix, group in sorted(by_prefix.items()):
        if len(prefix.split()) < 2 or len({rows[i]["brand"] for i in group}) < 2:
            continue
        group = sorted(group, key=lambda i: _rank(config.seed, rows[i]["id"]))
        if len(group) > config.cross_brand_block_cap:
            capped_prefix_blocks += 1
        group = group[:config.cross_brand_block_cap]
        for a, b in itertools.combinations(group, 2):
            if rows[a]["brand"] != rows[b]["brand"]:
                compare(a, b, "raw_name_prefix_cross_brand")
    chosen, used_ids, used_families = [], set(), set()
    brand_counts = Counter()
    eligible_counts = Counter(row["provisional_stratum"] for row in seeds.values())
    # Round-robin avoids earlier lexical strata exhausting independent families.
    queues = {bucket: iter(sorted((row for row in seeds.values() if row["provisional_stratum"] == bucket),
                                 key=lambda row: row["selection_rank"])) for bucket in BUCKETS}
    exhausted, chosen_counts = set(), Counter()
    while len(exhausted) < len(BUCKETS):
        for bucket in BUCKETS:
            if bucket in exhausted:
                continue
            if chosen_counts[bucket] >= config.per_bucket:
                exhausted.add(bucket)
                continue
            for row in queues[bucket]:
                endpoints = {row[c] for c in PAIR_COLUMNS}
                families = {row["left_family_key"], row["right_family_key"]}
                brand_key = (bucket, row["raw_brand_block"])
                if (endpoints & used_ids or families & used_families
                        or brand_counts[brand_key] >= config.max_pairs_per_brand_bucket):
                    continue
                chosen.append(row)
                used_ids.update(endpoints)
                used_families.update(families)
                chosen_counts[bucket] += 1
                brand_counts[brand_key] += 1
                break
            else:
                exhausted.add(bucket)
    chosen.sort(key=lambda row: _rank(config.seed, "neutral_pair_id", *[row[c] for c in PAIR_COLUMNS]))
    for i, row in enumerate(chosen, 1):
        row["pair_id"] = f"B{i:03d}"
    pool = pd.DataFrame(chosen)
    if pool.empty:
        pool = pd.DataFrame(columns=["pair_id", *PAIR_COLUMNS, "provisional_stratum"])
    pool = assemble_evidence(frame, pool)
    report = {"seed": config.seed, "config": asdict(config), "input_rows": len(frame),
              "prior_review_pairs": len(prior), "prior_endpoints_excluded": len(prior_ids),
              "prior_family_rows_excluded": excluded_family_rows, "eligible_rows": len(rows),
              "pair_comparisons": comparisons, "capped_raw_brand_blocks": capped_brand_blocks,
              "capped_cross_brand_prefix_blocks": capped_prefix_blocks,
              "eligible_provisional_counts": dict(eligible_counts), "candidate_counts": dict(chosen_counts),
              "candidate_pairs": len(pool), "candidate_targets_used": False,
              "candidate_method_outcomes_used": False,
              "retrieval_fields": ["ProductName", "ProductBrand"],
              "gold_labels_inferred": False,
              "exclusion_rule": "all prior185 endpoints and their packaging-insensitive normalized-brand/name and three-word title-prefix families; no repeated endpoints or short-title families in pool",
              "selection_rule": "round-robin lexical strata; seeded SHA256 endpoint rank; brand cap; family/endpoint dedup; provisional strata are not labels",
              "final_selection_rule": "explicit independent human categories and judgments; 30 per category by SHA256(seed + pair_id); no method correctness inputs"}
    return pool, report


def _validate_gold(review):
    required = {"pair_id", *PAIR_COLUMNS, *REVIEW_FIELDS}
    if not required <= set(review):
        raise ValueError(f"missing independent review columns: {sorted(required - set(review))}")
    if review.empty or review.pair_id.astype(str).str.strip().eq("").any() or review.pair_id.duplicated().any():
        raise ValueError("review pair IDs must be nonempty and unique")
    pairs = [tuple(sorted(pair)) for pair in review[PAIR_COLUMNS].itertuples(index=False, name=None)]
    if len(set(pairs)) != len(pairs) or any(not a or not b or a == b for a, b in pairs):
        raise ValueError("review endpoints must be distinct nonempty unique pairs")
    for field, allowed in (("product_identity", {"same_product", "different_product", "uncertain"}),
                           ("identity_detail", {"identical_item", "pack_size_variant", "strength_variant",
                                                "flavor_style_variant", "formulation_variant", "distinct_product", "uncertain"}),
                           ("identity_confidence", {"high", "medium", "low"}),
                           ("leakage_decision", {"keep_together", "keep_separate", "uncertain"}),
                           ("leakage_confidence", {"high", "medium", "low"}),
                           ("review_category", set(CATEGORIES))):
        if not review[field].isin(allowed).all():
            raise ValueError(f"invalid independent {field}")
    for field in ("judgment_uses_Segment", "judgment_uses_method_outcomes"):
        if not review[field].astype(str).str.casefold().eq("false").all():
            raise ValueError(f"independent judgments must exclude {field.removeprefix('judgment_uses_')}")
    for field in ("identity_rationale", "leakage_rationale"):
        if review[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"empty {field}")
    if (review.product_identity.eq("same_product") & review.leakage_decision.eq("keep_separate")).any():
        raise ValueError("same-product judgments contradict keep_separate")
    related = review.review_category.eq("related")
    separate = review.review_category.eq("safe_to_separate")
    borderline = review.review_category.eq("borderline")
    if (related & ~(review.leakage_decision.eq("keep_together") & review.leakage_confidence.eq("high"))).any():
        raise ValueError("related requires independently reviewed high-confidence keep_together")
    if (separate & ~(review.leakage_decision.eq("keep_separate") & review.leakage_confidence.eq("high"))).any():
        raise ValueError("safe_to_separate requires independently reviewed high-confidence keep_separate")
    if (borderline & ~(review.leakage_decision.eq("uncertain") | review.leakage_confidence.isin(["medium", "low"]))).any():
        raise ValueError("borderline requires uncertainty or medium/low leakage confidence")


def select_balanced_review(pool, review, *, per_category=30, seed=SEED):
    """Select exactly balanced independently labeled pairs, never by correctness.

    Every annotation must match a registered endpoint pair. Category and gold
    judgments come exclusively from the explicit review, not provisional seeds.
    The returned selected rows include all registered product evidence and the
    review rationales, but omit candidate retrieval scores and strata.
    """
    if per_category < 1:
        raise ValueError("per-category quota must be positive")
    _validate_gold(review)
    if not {"pair_id", *PAIR_COLUMNS} <= set(pool) or pool.pair_id.duplicated().any():
        raise ValueError("candidate registry pair IDs must be unique")
    if not set(review.pair_id) <= set(pool.pair_id):
        raise ValueError("review pair absent from candidate registry")
    registry = pool.set_index("pair_id")
    for row in review[["pair_id", *PAIR_COLUMNS]].itertuples(index=False):
        expected = registry.loc[row.pair_id]
        if set((row.left_record_id, row.right_record_id)) != set(expected[PAIR_COLUMNS]):
            raise ValueError("review endpoints differ from candidate registry")
    counts = review.review_category.value_counts()
    short = {category: int(counts.get(category, 0)) for category in CATEGORIES
             if counts.get(category, 0) < per_category}
    if short:
        raise ValueError(f"insufficient independently reviewed categories for quota {per_category}: {short}")
    ranked = review.copy()
    ranked["selection_rank"] = ranked.pair_id.map(lambda value: _rank(seed, value))
    ids = []
    for category in CATEGORIES:
        ids.extend(ranked.loc[ranked.review_category.eq(category)].sort_values("selection_rank", kind="stable")
                   .head(per_category).pair_id)
    # Rebuild endpoints in registry orientation so attached evidence is unambiguous.
    evidence_columns = ["pair_id", *PAIR_COLUMNS,
                        *(f"{side}_{field}" for side in ("left", "right") for field in EVIDENCE_FIELDS)]
    result = pool.set_index("pair_id").loc[ids].reset_index()[evidence_columns]
    result = result.merge(ranked[["pair_id", *REVIEW_FIELDS, "selection_rank"]], on="pair_id", validate="one_to_one", sort=False)
    if "left_family_key" in pool and "right_family_key" in pool:
        families = pool.set_index("pair_id").loc[ids, ["left_family_key", "right_family_key"]]
        used = set()
        for left, right in families.itertuples(index=False, name=None):
            current = {left, right}
            if used & current:
                raise ValueError("selected pairs repeat an independent short-title family")
            used.update(current)
    if pd.concat([result.left_record_id, result.right_record_id]).duplicated().any():
        raise ValueError("selected pairs repeat endpoints")
    return result


def generate_candidate_artifacts(input_path, prior_review_path, output_dir, config=None, identifier_path=None):
    """Create new review artifacts without modifying any existing experiment."""
    input_path, prior_review_path, output_dir = map(Path, (input_path, prior_review_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output_dir}")
    frame, metadata = load_input(input_path, identifier_path)
    prior = pd.read_csv(prior_review_path, dtype=str, keep_default_na=False, usecols=PAIR_COLUMNS)
    pool, report = candidate_pool(frame, prior, config)
    output_dir.mkdir(parents=True, exist_ok=False)
    pool.to_csv(output_dir / "candidate_pool.csv", index=False)
    (output_dir / "candidate_blind.json").write_text(
        json.dumps(blind_evidence(pool), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    template = pool[["pair_id", *PAIR_COLUMNS]].copy()
    for field in REVIEW_FIELDS:
        template[field] = "False" if field.startswith("judgment_uses_") else ""
    template.to_csv(output_dir / "independent_review_template.csv", index=False)
    report.update({"input_manifest": metadata, "prior_review_sha256": sha256(prior_review_path),
                   "sampler_code_sha256": sha256(Path(__file__)),
                   "artifact_sha256": {name: sha256(output_dir / name)
                                       for name in ("candidate_pool.csv", "candidate_blind.json", "independent_review_template.csv")}})
    (output_dir / "candidate_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return pool, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prior-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=70)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    _, report = generate_candidate_artifacts(args.input, args.prior_review, args.output_dir,
                                             CandidateConfig(seed=args.seed, per_bucket=args.per_bucket))
    print(json.dumps({key: report[key] for key in ("candidate_pairs", "candidate_counts", "pair_comparisons")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
