"""Deterministic, label-free candidate discovery for rule-based grouping.

Candidates are proposals, never confirmed product-family relationships. The
matcher owns all evidence safeguards and component validation. Bounded sorted
neighborhoods replace quadratic comparisons in large blocks; their budgets do
not constrain the eventual size of a product group.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import heapq
from itertools import combinations
import math


@dataclass(frozen=True)
class RuleCandidateConfig:
    block_tokens: int = 3
    max_exhaustive_block: int = 96
    large_block_window: int = 12
    sorted_neighbor_window: int = 12
    line_prefix_words: int = 2
    global_rare_token_frequency: int = 32
    char_candidate_threshold: float = 0.45
    char_top_k: int = 12
    max_character_comparisons: int | None = None
    character_cache_size: int = 4096

    def validate(self):
        for name in ("block_tokens", "max_exhaustive_block", "large_block_window",
                     "sorted_neighbor_window", "line_prefix_words",
                     "global_rare_token_frequency", "char_top_k", "character_cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.char_candidate_threshold, (int, float))
                or isinstance(self.char_candidate_threshold, bool)
                or not math.isfinite(self.char_candidate_threshold)
                or not 0 <= self.char_candidate_threshold <= 1):
            raise ValueError("char_candidate_threshold must be in [0, 1]")
        if (self.max_character_comparisons is not None
                and (isinstance(self.max_character_comparisons, bool)
                     or not isinstance(self.max_character_comparisons, int)
                     or self.max_character_comparisons < 1)):
            raise ValueError("max_character_comparisons must be positive or None")


def _chargrams(value):
    value = " " + value + " "
    return frozenset(value[i:i + 3] for i in range(max(0, len(value) - 2)))


def generate_candidates(names, brands, ids, cores, config=None):
    """Return ``(index-pair/channel tuples, statistics)`` from feature arrays.

    ``names`` are already normalized and stripped of packaging and negated
    marketing claims, but retain named-line and brand terms. ``cores`` are
    informative token sets supplied by the label-free evidence extractor.
    Records with empty cores still receive exact-name, named-line/prefix and
    character-neighborhood searches. Character similarity is trigram Jaccard;
    its deliberately broad cutoff only governs candidate discovery.

    Output endpoint orientation, pair order, top-k ties and all traversal order
    depend on immutable record IDs, not input position. No target, annotations,
    group assignments or split memberships enter this API.
    """
    config = config or RuleCandidateConfig()
    config.validate()
    names = ["" if value is None else str(value) for value in names]
    brands = ["" if value is None else str(value) for value in brands]
    ids = [str(value) for value in ids]
    cores = [frozenset(map(str, value)) for value in cores]
    size = len(ids)
    if not (len(names) == len(brands) == len(cores) == size):
        raise ValueError("candidate feature arrays and record IDs must align")
    if any(not value.strip() for value in ids) or len(set(ids)) != size:
        raise ValueError("record IDs must be nonempty and unique")
    stats = Counter()
    proposed = defaultdict(set)
    brand_keys = [value.replace(" ", "") for value in brands]
    frequencies = Counter(token for core in cores for token in core)
    words = [tuple(name.split()) for name in names]
    line_words = [tuple(token for token in tokens
                        if token not in set(brand.split()) and not token.isdecimal())
                  for tokens, brand in zip(words, brands)]

    def ordered_pair(a, b):
        return (a, b) if ids[a] < ids[b] else (b, a)

    def offer(a, b, channel):
        if a == b:
            return
        pair = ordered_pair(a, b)
        proposed[pair].add(channel)

    def neighbors(members, window, ordering):
        ordered = sorted(members, key=ordering)
        for pos, left in enumerate(ordered):
            for right in ordered[pos + 1:pos + 1 + window]:
                yield ordered_pair(left, right)

    def block_pairs(members, channel):
        members = sorted(set(members), key=lambda i: ids[i])
        if len(members) < 2:
            return
        stats[f"{channel}_blocks"] += 1
        theoretical = len(members) * (len(members) - 1) // 2
        stats[f"{channel}_possible_block_comparisons"] += theoretical
        if len(members) <= config.max_exhaustive_block:
            pairs = combinations(members, 2)
        else:
            stats[f"{channel}_large_blocks"] += 1
            # Both views preserve every row in the block. No row sampling or
            # truncation is used, including for exceptionally large exact keys.
            pairs = set(neighbors(members, config.large_block_window,
                                  lambda i: (names[i], ids[i])))
            pairs.update(neighbors(members, config.large_block_window,
                                   lambda i: (" ".join(sorted(cores[i])), names[i], ids[i])))
            pairs = sorted(pairs, key=lambda pair: (ids[pair[0]], ids[pair[1]]))
            stats[f"{channel}_block_comparisons_omitted"] += theoretical - len(pairs)
        for left, right in pairs:
            stats[f"{channel}_proposals"] += 1
            offer(left, right, channel)

    exact, rare, prefix, by_brand = (defaultdict(list) for _ in range(4))
    for i, (name, brand, core, tokens) in enumerate(zip(names, brand_keys, cores, words)):
        if not name:
            continue
        exact[name].append(i)
        if brand:
            by_brand[brand].append(i)
        eligible = sorted((token for token in core
                           if len(token) >= 3 and not token.isdecimal()),
                          key=lambda token: (frequencies[token], token))[:config.block_tokens]
        for token in eligible:
            if brand:
                rare["brand", brand, token].append(i)
            if tokens:
                rare["title_prefix", tokens[0], token].append(i)
            # Low-frequency informative words can recover a copied meaningful
            # family when one listing omits a brand or has a parent/sub-brand.
            if frequencies[token] <= config.global_rare_token_frequency:
                rare["global_rare", token].append(i)
        if line_words[i]:
            first = line_words[i][0]
            if brand:
                prefix["brand_line", brand, first].append(i)
            named_prefix = line_words[i][:config.line_prefix_words]
            if len(named_prefix) >= config.line_prefix_words:
                prefix["unbranded_name_prefix", *named_prefix].append(i)
            # Also preserve the literal prefix where declared and title brands
            # disagree. This is a proposal, not a inferred brand equivalence.
            if len(tokens) >= config.line_prefix_words:
                prefix["literal_name_prefix", *tokens[:config.line_prefix_words]].append(i)

    for key in sorted(exact):
        block_pairs(exact[key], "normalized_exact_name")
    for key in sorted(rare):
        block_pairs(rare[key], "rare_token:" + key[0])
    for key in sorted(prefix):
        block_pairs(prefix[key], "name_prefix:" + key[0])

    grams = lru_cache(maxsize=config.character_cache_size)(lambda i: _chargrams(names[i]))
    char_candidates = defaultdict(list)
    id_ranks = {index: rank for rank, index in enumerate(sorted(range(size), key=lambda i: ids[i]))}
    directed_omitted = 0
    budget_exhausted = False
    for brand in sorted(by_brand):
        members = sorted(by_brand[brand], key=lambda i: ids[i])
        if len(members) < 2:
            continue
        stats["character_brand_blocks"] += 1
        theoretical = len(members) * (len(members) - 1) // 2
        stats["character_possible_brand_comparisons"] += theoretical
        if len(members) <= config.max_exhaustive_block:
            searches = combinations(members, 2)
        else:
            stats["character_large_brand_blocks"] += 1
            local = set()
            for ordering in (
                lambda i: (names[i], ids[i]),
                lambda i: (" ".join(reversed(words[i])), names[i], ids[i]),
                lambda i: (" ".join(sorted(words[i])), names[i], ids[i]),
                lambda i: (" ".join(sorted(cores[i])), names[i], ids[i]),
            ):
                local.update(neighbors(members, config.sorted_neighbor_window, ordering))
            stats["character_brand_comparisons_omitted_by_neighborhood"] += theoretical - len(local)
            searches = sorted(local, key=lambda pair: (ids[pair[0]], ids[pair[1]]))
        for left, right in searches:
            pair = ordered_pair(left, right)
            if (config.max_character_comparisons is not None
                    and stats["character_comparisons"] >= config.max_character_comparisons):
                budget_exhausted = True
                stats["character_proposals_omitted_by_budget"] += 1
                continue
            stats["character_comparisons"] += 1
            a, b = grams(left), grams(right)
            intersection = len(a & b)
            denominator = len(a) + len(b) - intersection
            similarity = intersection / denominator if denominator else 0.0
            if similarity >= config.char_candidate_threshold:
                stats["character_pairs_above_cutoff"] += 1
                for query, neighbor in ((left, right), (right, left)):
                    value = (similarity, -id_ranks[neighbor], neighbor)
                    if len(char_candidates[query]) < config.char_top_k:
                        heapq.heappush(char_candidates[query], value)
                    else:
                        heapq.heappushpop(char_candidates[query], value)
                        directed_omitted += 1
    selected_char_pairs = set()
    for left in sorted(char_candidates, key=lambda i: ids[i]):
        ranked = sorted(char_candidates[left], key=lambda value: (-value[0], ids[value[2]]))
        for _, _, right in ranked:
            selected_char_pairs.add(ordered_pair(left, right))
            offer(left, right, "character_trigram_neighborhood")
    stats["character_pairs_selected"] = len(selected_char_pairs)
    stats["character_directed_neighbors_omitted_by_top_k"] = directed_omitted
    stats["candidate_pairs"] = len(proposed)
    stats["candidate_channel_counts"] = dict(sorted(Counter(
        channel for channels in proposed.values() for channel in channels).items()))
    stats["empty_core_rows"] = sum(not core for core in cores)
    stats["rows_with_candidates"] = len({i for pair in proposed for i in pair})
    result = [(left, right, tuple(sorted(proposed[left, right])))
              for left, right in sorted(proposed, key=lambda pair: (ids[pair[0]], ids[pair[1]]))]
    report = dict(stats)
    report.update({"input_records": size, "config": asdict(config),
                   "character_comparison_budget_exhausted": budget_exhausted,
                   "targets_used": False, "annotations_used": False,
                   "candidate_similarity_is_confirmation": False,
                   "group_size_cap_applied": False,
                   "large_block_strategy": "all rows, deterministic multiple sorted neighborhoods",
                   "character_cache": {"hits": grams.cache_info().hits,
                                       "misses": grams.cache_info().misses}})
    return result, report
