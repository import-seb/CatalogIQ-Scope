"""Shared, label-blind duplicate audit for product-group splitting experiments.

The name audit uses character-set Jaccard, independently of either grouping
model. Its prefix and length filters are lossless within the reported scope;
neither the rule blocks nor the TF-IDF graph supplies its candidate pairs.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import combinations
import html
import math
import time
import unicodedata

import numpy as np
import pandas as pd

from .splitting import SPLITS, SplitConfig

PAYLOAD_FIELDS = ("ProductBrand", "ProductName", "ProductDescription", "ProductContents")


def evaluation_name(value) -> str:
    """Independent Unicode name normalization; retain every letter and digit.

    Punctuation and whitespace delimit words. Numbers, units, model suffixes,
    and accents are retained rather than rewritten or removed.
    """
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.casefold() in {"", "null"}:
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def _ceil(value):
    # Roundoff must never make a necessary filter exclude a boundary match.
    return math.ceil(value - 1e-12)


def near_duplicate_pairs(frame, config: SplitConfig, progress=None):
    """Return all qualifying positional row pairs and audit-scope metadata.

    Eligible ProductName values have at least evaluation_min_name_chars after
    normalization and at least one shingle. The similarity is Jaccard on sets
    of consecutive evaluation_shingle_size characters (including spaces).

    A globally consistent rare-first shingle order gives each set a prefix of
    length n-ceil(threshold*n)+1. Any qualifying pair shares a prefix shingle.
    Process sets in ascending size and ignore preceding sets smaller than
    ceil(threshold*n), then verify candidates using the full sets. Identical
    normalized names are collapsed only during comparison and expanded to all
    original row pairs. No comparison block, top-k cap, or sampling is used.
    """
    started = time.perf_counter()
    names = [evaluation_name(v) for v in frame.get("ProductName", pd.Series("", index=frame.index))]
    buckets = defaultdict(list)
    shingle_size = config.evaluation_shingle_size
    for row, name in enumerate(names):
        if len(name) >= max(config.evaluation_min_name_chars, shingle_size):
            buckets[name].append(row)

    sets = {name: {name[i:i + shingle_size] for i in range(len(name) - shingle_size + 1)}
            for name in buckets}
    ordered_names = sorted(sets, key=lambda name: (len(sets[name]), name))
    lengths = [len(sets[name]) for name in ordered_names]
    frequencies = Counter(token for shingles in sets.values() for token in shingles)
    # Frequencies only improve efficiency; any shared global ordering is valid.
    rank = {token: i for i, token in enumerate(sorted(frequencies, key=lambda t: (frequencies[t], t)))}
    indexed = defaultdict(list)
    pairs = []
    identical_pairs = 0
    for members in buckets.values():
        identical_pairs += len(members) * (len(members) - 1) // 2
        pairs.extend((a, b, 1.0) for a, b in combinations(members, 2))

    candidate_hits = candidate_names = candidate_rows = 0
    accepted_name_pairs = accepted_nonidentical_rows = 0
    threshold = config.evaluation_threshold
    for current, name in enumerate(ordered_names):
        shingles = sets[name]
        size = lengths[current]
        prefix_length = size - _ceil(threshold * size) + 1
        prefix = sorted(shingles, key=rank.__getitem__)[:prefix_length]
        minimum_index = bisect_left(lengths, _ceil(threshold * size), 0, current)
        candidates = set()
        for token in prefix:
            posting = indexed.get(token, ())
            start = bisect_left(posting, minimum_index)
            candidate_hits += len(posting) - start
            for offset in range(start, len(posting)):
                candidates.add(posting[offset])
        candidate_names += len(candidates)
        for previous in sorted(candidates):
            previous_name = ordered_names[previous]
            candidate_rows += len(buckets[name]) * len(buckets[previous_name])
            intersection = len(shingles & sets[previous_name])
            required = _ceil(threshold * (size + lengths[previous]) / (1 + threshold))
            if intersection < required:
                continue
            score = intersection / (size + lengths[previous] - intersection)
            if score + 1e-12 < threshold:
                continue
            accepted_name_pairs += 1
            accepted_nonidentical_rows += len(buckets[name]) * len(buckets[previous_name])
            for a in buckets[name]:
                for b in buckets[previous_name]:
                    pairs.append((min(a, b), max(a, b), float(score)))
        for token in prefix:
            indexed[token].append(current)
        if progress and (current == 0 or (current + 1) % 5000 == 0 or current + 1 == len(ordered_names)):
            progress(f"Independent name audit: {current + 1:,}/{len(ordered_names):,} unique eligible names")

    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    eligible = sum(len(members) for members in buckets.values())
    return pairs, {
        "method": "independent_character_shingle_jaccard",
        "features": ["ProductName"],
        "normalization": "HTML decode; NFKC; casefold; alphanumeric words separated by spaces; numbers retained",
        "threshold": threshold,
        "shingle_size": shingle_size,
        "minimum_normalized_name_chars": config.evaluation_min_name_chars,
        "input_rows": len(frame),
        "eligible_rows": eligible,
        "ineligible_rows": len(frame) - eligible,
        "unique_eligible_names": len(buckets),
        "possible_eligible_row_pairs": eligible * (eligible - 1) // 2,
        "prefix_posting_hits": candidate_hits,
        "unique_name_candidates_verified": candidate_names,
        "candidate_row_pairs": candidate_rows + identical_pairs,
        "identical_normalized_name_pairs": identical_pairs,
        "accepted_nonidentical_name_pairs": accepted_name_pairs,
        "accepted_nonidentical_row_pairs": accepted_nonidentical_rows,
        "accepted_pairs": len(pairs),
        "candidate_filter": "lossless rare-first prefix intersection and shingle-set-size bounds",
        "exhaustive_within_scope": True,
        "limitations": [
            "Name-only lexical audit; it does not measure semantic equivalence or description-only similarity.",
            "Missing and short normalized names are excluded; no minimum brand or identifier agreement is required.",
            "Similar names can describe distinct variants; matches are leakage candidates, not verified identities.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }


def _labels(frame):
    return pd.Series(["<MISSING>" if not evaluation_name(v) else str(v)
                      for v in frame.get("Segment", pd.Series("", index=frame.index))], dtype="string")


def _check_alignment(frame, assignments):
    if len(frame) != len(assignments):
        raise ValueError("Assignments must be aligned with the shared input rows")
    if "record_id" in frame and "record_id" in assignments:
        left = frame["record_id"].reset_index(drop=True).astype("string")
        right = assignments["record_id"].reset_index(drop=True).astype("string")
        if not left.equals(right):
            raise ValueError("Assignment record_id order does not match the input rows")


def segment_distribution(frame, assignments):
    """Tabulate counts, within-split class shares, and each class's allocation."""
    _check_alignment(frame, assignments)
    labels = _labels(frame)
    split_values = assignments["split"].reset_index(drop=True)
    result = labels.value_counts().sort_index().rename("total_rows").to_frame()
    result.index.name = "Segment"
    result["global_class_proportion"] = result["total_rows"] / max(len(frame), 1)
    support = pd.DataFrame({"Segment": labels,
                            "group_id": assignments["group_id"].reset_index(drop=True)})
    result["supporting_groups"] = support.groupby("Segment")["group_id"].nunique().reindex(result.index)
    for split in SPLITS:
        counts = labels[split_values.eq(split)].value_counts()
        result[f"{split}_rows"] = counts.reindex(result.index, fill_value=0).astype(int)
        split_total = int(split_values.eq(split).sum())
        result[f"{split}_class_proportion"] = result[f"{split}_rows"] / max(split_total, 1)
        result[f"{split}_delta_from_global"] = result[f"{split}_class_proportion"] - result["global_class_proportion"]
        result[f"{split}_allocation_proportion"] = result[f"{split}_rows"] / result["total_rows"]
    return result.reset_index()


def _split_pair_counts():
    return {f"{a}__{b}": 0 for a, b in combinations(SPLITS, 2)}


def _exact_metrics(keys, splits, scope):
    buckets = defaultdict(list)
    for row, key in enumerate(keys):
        if key is not None:
            buckets[key].append(row)
    duplicate_pairs = crossing_pairs = clusters = crossing_clusters = 0
    crossing_rows = set()
    split_pairs = _split_pair_counts()
    for members in buckets.values():
        if len(members) < 2:
            continue
        clusters += 1
        duplicate_pairs += len(members) * (len(members) - 1) // 2
        counts = Counter(splits[row] for row in members)
        if len(counts) > 1:
            crossing_clusters += 1
            crossing_rows.update(members)
        for a, b in combinations(SPLITS, 2):
            count = counts[a] * counts[b]
            split_pairs[f"{a}__{b}"] += count
            crossing_pairs += count
    return {
        "scope": scope, "eligible_rows": sum(len(v) for v in buckets.values()),
        "duplicate_clusters": clusters, "crossing_clusters": crossing_clusters,
        "duplicate_pairs": duplicate_pairs, "crossing_pairs": crossing_pairs,
        "crossing_pair_rate": crossing_pairs / duplicate_pairs if duplicate_pairs else 0.0,
        "crossing_rows": len(crossing_rows), "crossing_split_pairs": split_pairs,
    }


def leakage_metrics(frame, assignments, pairs, config: SplitConfig):
    """Compute both split summaries against the same independently audited pairs.

    Rows in assignments must have the same order as frame. Exact payload keys
    preserve raw ProductBrand/Name/Description/Contents strings and distinguish
    absent values from empty strings; labels, URLs, ratings, and provenance
    never enter those keys. A normalized brand/name audit is reported separately.
    """
    _check_alignment(frame, assignments)
    splits = assignments["split"].astype(str).tolist()
    if any(split not in SPLITS for split in splits):
        raise ValueError("Unrecognized split")
    groups = assignments["group_id"].reset_index(drop=True)
    group_sizes = groups.value_counts().to_numpy(dtype=int)
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    size_summary = {str(q): float(np.quantile(group_sizes, q)) if len(group_sizes) else 0.0
                    for q in quantiles}
    size_summary["mean"] = float(group_sizes.mean()) if len(group_sizes) else 0.0
    histogram = {
        "1": int((group_sizes == 1).sum()), "2": int((group_sizes == 2).sum()),
        "3-5": int(((group_sizes >= 3) & (group_sizes <= 5)).sum()),
        "6-10": int(((group_sizes >= 6) & (group_sizes <= 10)).sum()),
        "11-50": int(((group_sizes >= 11) & (group_sizes <= 50)).sum()),
        "51-100": int(((group_sizes >= 51) & (group_sizes <= 100)).sum()),
        ">100": int((group_sizes > 100).sum()),
    }
    labels = _labels(frame)
    mixtures = pd.DataFrame({"group": groups, "Segment": labels}).groupby("group")["Segment"].nunique()
    raw = frame.reindex(columns=PAYLOAD_FIELDS)
    raw_keys = []
    for row in raw.itertuples(index=False, name=None):
        key = tuple(None if pd.isna(value) else str(value) for value in row)
        raw_keys.append(key if any(evaluation_name(value) for value in key) else None)
    names = [evaluation_name(v) for v in frame.get("ProductName", pd.Series("", index=frame.index))]
    brands = [evaluation_name(v) for v in frame.get("ProductBrand", pd.Series("", index=frame.index))]
    name_keys = [(brand, name) if name else None for brand, name in zip(brands, names)]
    crossing_rows = set()
    crossing_pairs = nonidentical_crossing = nonidentical_pairs = co_grouped_pairs = 0
    group_values = groups.tolist()
    split_pairs = _split_pair_counts()
    split_order = {split: i for i, split in enumerate(SPLITS)}
    for a, b, _ in pairs:
        co_grouped_pairs += int(group_values[a] == group_values[b])
        nonidentical_pairs += int(names[a] != names[b])
        if splits[a] == splits[b]:
            continue
        crossing_pairs += 1
        crossing_rows.update((a, b))
        nonidentical_crossing += int(names[a] != names[b])
        first, second = sorted((splits[a], splits[b]), key=split_order.__getitem__)
        split_pairs[f"{first}__{second}"] += 1
    distribution = segment_distribution(frame, assignments)
    missing = {split: distribution.loc[distribution[f"{split}_rows"].eq(0), "Segment"].tolist()
               for split in SPLITS}
    rare = distribution["total_rows"].le(config.rare_class_max_rows)
    return {
        "rows": len(frame),
        "groups": {
            "count": len(group_sizes), "singleton_groups": histogram["1"],
            "size_summary": size_summary, "size_histogram": histogram,
            "mixed_segment_groups": int(mixtures.gt(1).sum()),
        },
        "splits": {split: {"rows": splits.count(split),
                           "proportion": splits.count(split) / max(len(frame), 1),
                           "target_proportion": config.ratios[i]}
                   for i, split in enumerate(SPLITS)},
        "exact_payload": _exact_metrics(raw_keys, splits, list(PAYLOAD_FIELDS)),
        "exact_brand_name": _exact_metrics(name_keys, splits, "independently normalized brand and name"),
        "near_duplicate": {
            "pairs": len(pairs), "crossing_pairs": crossing_pairs,
            "co_grouped_pairs": co_grouped_pairs,
            "co_grouped_pair_rate": co_grouped_pairs / len(pairs) if pairs else 0.0,
            "nonidentical_name_pairs": nonidentical_pairs,
            "crossing_pair_rate": crossing_pairs / len(pairs) if pairs else 0.0,
            "crossing_rows": len(crossing_rows),
            "crossing_row_proportion": len(crossing_rows) / max(len(frame), 1),
            "crossing_split_pairs": split_pairs,
            "nonidentical_name_crossing_pairs": nonidentical_crossing,
        },
        "missing_segment_classes": missing,
        "segment_supporting_groups": {row.Segment: int(row.supporting_groups)
                                      for row in distribution.itertuples(index=False)},
        "classes_not_representable_in_all_three_splits": distribution.loc[
            distribution["supporting_groups"].lt(3), "Segment"].tolist(),
        "rare_class_max_rows": config.rare_class_max_rows,
        "rare_segment_classes": distribution.loc[rare, "Segment"].tolist(),
        "rare_segment_class_count": int(rare.sum()),
        "rare_segment_classes_missing_validation": distribution.loc[
            rare & distribution["validation_rows"].eq(0), "Segment"].tolist(),
        "rare_segment_classes_missing_test": distribution.loc[
            rare & distribution["test_rows"].eq(0), "Segment"].tolist(),
    }
