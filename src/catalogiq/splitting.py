"""Label-blind product grouping and deterministic group-stratified splitting.

Groups are connected components: a chain of accepted links stays together even
if its endpoints are dissimilar. No cleaner or source record is modified here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import html
import json
import math
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as normalize_vectors

from .cleaning import PROVENANCE, TARGET_COLUMNS

TEXT_FIELDS = ("ProductName", "ProductDescription", "ProductContents", "ProductBrand")
IDENTIFIER_FIELDS = ("Sku", "Upc", "ProductModelNumber", "MDM_Id", "JoiningKey")
# Category is a supplied label; ProductCategory is a retailer feature, but is
# deliberately omitted from matching to avoid grouping entire taxonomy branches.
GROUP_FIELDS = (*TEXT_FIELDS, "Retailer", "ProductUrl", *IDENTIFIER_FIELDS)
SPLITS = ("train", "validation", "test")
MISSING = {"", "null"}


@dataclass(frozen=True)
class SplitConfig:
    grouping_version: int = 1
    seed: int = 42
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    rule_token_threshold: float = 0.82
    rule_edit_threshold: float = 0.88
    identifier_name_threshold: float = 0.50
    block_tokens: int = 2
    max_block_size: int = 300
    tfidf_threshold: float = 0.88
    tfidf_weights: tuple[float, float, float, float] = (3.0, 1.0, 1.0, 0.5)
    tfidf_max_features: int = 100_000
    tfidf_max_df: float = 0.20
    description_chars: int = 1500
    contents_chars: int = 1000
    cosine_chunk_size: int = 128
    evaluation_threshold: float = 0.85
    evaluation_shingle_size: int = 5
    evaluation_min_name_chars: int = 12
    split_restarts: int = 3
    split_refinement_passes: int = 2
    size_weight: float = 1.0
    rare_class_max_rows: int = 100
    disagreement_examples: int = 40
    rule_family_threshold: float = 0.78
    rule_family_edit_threshold: float = 0.85
    tfidf_name_threshold: float = 0.86
    tfidf_family_threshold: float = 0.90
    tfidf_supported_name_threshold: float = 0.78
    tfidf_support_threshold: float = 0.50
    tfidf_name_max_features: int | None = None
    core_min_overlap: float = 0.35
    brand_override_name_threshold: float = 0.75
    family_strip_package_sizes: bool = True
    brand_alias_suffixes: tuple[str, ...] = ("health products", "organic oils", "products", "health",
        "research", "nutrition", "naturals", "formulas", "inc", "llc", "ltd", "company")

    def __post_init__(self):
        if type(self.grouping_version) is not int or self.grouping_version not in (1, 2):
            raise ValueError("grouping_version must be 1 or 2")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if (len(self.ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.ratios)
                or not math.isclose(sum(self.ratios), 1)):
            raise ValueError("ratios must contain three positive values summing to one")
        for name in ("rule_token_threshold", "rule_edit_threshold", "identifier_name_threshold",
                     "tfidf_threshold", "tfidf_max_df", "evaluation_threshold", "rule_family_threshold",
                     "rule_family_edit_threshold", "tfidf_name_threshold", "tfidf_family_threshold",
                     "tfidf_supported_name_threshold", "tfidf_support_threshold", "core_min_overlap",
                     "brand_override_name_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("block_tokens", "max_block_size", "tfidf_max_features", "description_chars",
                     "contents_chars", "cosine_chunk_size", "evaluation_shingle_size",
                     "evaluation_min_name_chars", "split_restarts", "rare_class_max_rows",
                     "disagreement_examples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.split_refinement_passes) is not int or self.split_refinement_passes < 0:
            raise ValueError("split_refinement_passes must be a nonnegative integer")
        if (len(self.tfidf_weights) != 4 or any(not math.isfinite(w) or w < 0 for w in self.tfidf_weights)
                or sum(self.tfidf_weights) == 0):
            raise ValueError("tfidf_weights requires four nonnegative weights, at least one positive")
        if not math.isfinite(self.size_weight) or self.size_weight < 0:
            raise ValueError("size_weight must be nonnegative and finite")
        if self.tfidf_name_max_features is not None and (type(self.tfidf_name_max_features) is not int
                                                       or self.tfidf_name_max_features < 1):
            raise ValueError("tfidf_name_max_features must be null or a positive integer")
        if type(self.family_strip_package_sizes) is not bool:
            raise ValueError("family_strip_package_sizes must be boolean")
        if not self.brand_alias_suffixes or any(not isinstance(s, str) or not s.strip() for s in self.brand_alias_suffixes):
            raise ValueError("brand_alias_suffixes must contain nonblank strings")
        if self.tfidf_supported_name_threshold > self.tfidf_name_threshold:
            raise ValueError("support candidate threshold cannot exceed strong name threshold")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name in ("ratios", "tfidf_weights", "brand_alias_suffixes"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def text(value) -> str:
    """Normalize a comparison view only, treating the repository's null as missing."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.casefold() in MISSING:
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def grouping_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Allowlisted attributes only: labels and provenance cannot reach matching."""
    return frame.reindex(columns=GROUP_FIELDS, fill_value="").fillna("").astype(str)


def record_ids(frame: pd.DataFrame) -> np.ndarray:
    if not set(PROVENANCE).issubset(frame.columns):
        raise ValueError(f"Input needs source identity columns {PROVENANCE}")
    if frame[PROVENANCE].isna().any().any():
        raise ValueError("Source identities must not be missing")
    keys = frame[PROVENANCE].astype(str)
    if keys.duplicated().any() or keys.apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Source identities must be nonblank and unique")
    return np.array([hashlib.sha256(json.dumps(list(row), ensure_ascii=False,
                     separators=(",", ":")).encode()).hexdigest()
                     for row in keys.itertuples(index=False, name=None)])


class Components:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

    def groups(self, ids, prefix):
        roots = [self.find(i) for i in range(len(self.parent))]
        minimum = {}
        for root, key in zip(roots, ids):
            minimum[root] = min(minimum.get(root, key), key)
        return np.array([f"{prefix}_{minimum[root]}" for root in roots])


def gtin(value):
    """Intact checksum-valid GTIN only; never reconstruct scientific notation."""
    value = str(value).strip()
    if not re.fullmatch(r"(?:\d{8}|\d{12}|\d{13}|\d{14})", value) or int(value) == 0:
        return ""
    digits = [int(c) for c in value]
    check = sum(n * (3 if i % 2 == 0 else 1) for i, n in enumerate(reversed(digits[:-1])))
    return value.zfill(14) if (check + digits[-1]) % 10 == 0 else ""


def listing_url(value):
    """Preserve query parameters and path case; variants can live in the query."""
    try:
        p = urlsplit(str(value).strip())
        if p.scheme.lower() not in {"http", "https"} or not p.netloc:
            return ""
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, p.query, ""))
    except ValueError:
        return ""


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def rule_groups(frame, ids, config: SplitConfig):
    if config.grouping_version == 2:
        from .split_rules_v2 import refine_rule_groups
        return refine_rule_groups(frame, ids, config)
    view = grouping_view(frame)
    normalized = {c: [text(v) for v in view[c]] for c in TEXT_FIELDS}
    names, brands = normalized["ProductName"], normalized["ProductBrand"]
    tokens = [set(n.split()) for n in names]
    uf = Components(len(view))
    edges = []
    stats = Counter()

    def link(a, b, reason, score=1.0):
        uf.union(a, b)
        edges.append((a, b, reason, float(score)))
        stats[reason] += 1

    def by_key(keys, reason, guard=False):
        buckets = defaultdict(list)
        for i, key in enumerate(keys):
            if key:
                buckets[key].append(i)
        for members in buckets.values():
            if guard:
                for pos, a in enumerate(members):
                    for b in members[pos + 1:]:
                        score = jaccard(tokens[a], tokens[b])
                        if score >= config.identifier_name_threshold:
                            link(a, b, reason, score)
            else:
                for b in members[1:]:
                    link(members[0], b, reason)

    retailers = [text(v) for v in view["Retailer"]]
    valid_gtins = [gtin(v) for v in view["Upc"]]
    stats["valid_gtin_rows"] = sum(bool(v) for v in valid_gtins)
    by_key(valid_gtins, "valid_gtin")
    by_key([listing_url(v) for v in view["ProductUrl"]], "listing_url")
    skus = [str(v).strip().casefold() for v in view["Sku"]]
    by_key([(r, s) if r and s not in MISSING else None for r, s in zip(retailers, skus)],
           "retailer_sku_name", guard=True)
    models = [str(v).strip().casefold() for v in view["ProductModelNumber"]]
    by_key([(b, m) if b and re.fullmatch(r"[\w-]{4,}", m) and re.search(r"\d", m)
            and not re.fullmatch(r"\d+(?:\.\d+)?e[+-]?\d+", m) else None
            for b, m in zip(brands, models)], "brand_model_name", guard=True)
    by_key([(b, n) if n and len(n) >= 12 else None for b, n in zip(brands, names)],
           "brand_exact_name")

    # Two infrequent name tokens within a brand provide separate blocking passes.
    # Oversized blocks are skipped and counted, never silently truncated.
    frequencies = Counter(t for row in tokens for t in row if len(t) >= 3 and not t.isdigit())
    blocks = defaultdict(list)
    for i, (brand, row) in enumerate(zip(brands, tokens)):
        if not brand:
            continue
        eligible = sorted((t for t in row if len(t) >= 3 and not t.isdigit()),
                          key=lambda t: (frequencies[t], t))[:config.block_tokens]
        for token in eligible:
            blocks[brand, token].append(i)
    seen = set()
    for members in blocks.values():
        if len(members) > config.max_block_size:
            stats["oversized_blocks_skipped"] += 1
            continue
        for pos, a in enumerate(members):
            for b in members[pos + 1:]:
                pair = (a, b)
                if pair in seen:
                    continue
                seen.add(pair)
                stats["blocked_comparisons"] += 1
                score = jaccard(tokens[a], tokens[b])
                if score >= config.rule_token_threshold:
                    # SequenceMatcher can depend on argument orientation.
                    left, right = sorted((names[a], names[b]))
                    edit = SequenceMatcher(None, left, right, autojunk=False).ratio()
                    if edit >= config.rule_edit_threshold:
                        link(a, b, "blocked_name", score)
    return uf.groups(ids, "rule"), edges, dict(stats)


def tfidf_groups(frame, ids, config: SplitConfig, progress=None):
    """Exact threshold graph, chunked sparse cosine; no top-k or rule blocking.

    IDF is fitted to the shared, unlabeled splitting universe. This vectorizer is
    a grouping tool, not model preprocessing to reuse across evaluation folds.
    """
    if config.grouping_version == 2:
        from .split_tfidf_v2 import refine_tfidf_groups
        return refine_tfidf_groups(frame, ids, config, progress=progress)
    view = grouping_view(frame)
    matrices = []
    vocabulary_sizes = {}
    for field, weight in zip(TEXT_FIELDS, config.tfidf_weights):
        if weight == 0:
            continue
        limit = {"ProductDescription": config.description_chars,
                 "ProductContents": config.contents_chars}.get(field)
        values = [text(v) for v in view[field]]
        if limit:
            values = [v[:limit] for v in values]
        if not any(re.search(r"\w", v) for v in values):
            continue
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b", dtype=np.float64,
                                    max_features=config.tfidf_max_features,
                                    max_df=max(1, int(len(view) * config.tfidf_max_df)))
        try:
            matrix = vectorizer.fit_transform(values)
        except ValueError as error:
            if "After pruning, no terms remain" not in str(error):
                raise
            # Tiny/degenerate fixtures may have only ubiquitous terms.
            vectorizer.set_params(max_df=1.0)
            matrix = vectorizer.fit_transform(values)
        matrices.append(matrix * weight)
        vocabulary_sizes[field] = len(vectorizer.vocabulary_)
    uf = Components(len(view))
    edges = []
    if not matrices:
        return uf.groups(ids, "tfidf"), edges, {"vocabulary_sizes": {}, "zero_vector_rows": len(view)}
    matrix = normalize_vectors(hstack(matrices, format="csr"), copy=False)
    transpose = matrix.T.tocsr()
    for start in range(0, len(view), config.cosine_chunk_size):
        stop = min(start + config.cosine_chunk_size, len(view))
        similarities = (matrix[start:stop] @ transpose).tocsr()
        # Discard low similarities before converting to coordinate arrays.
        similarities.data[similarities.data < config.tfidf_threshold - 1e-12] = 0
        similarities.eliminate_zeros()
        coo = similarities.tocoo()
        for a, b, score in zip(coo.row + start, coo.col, coo.data):
            if a < b:
                uf.union(int(a), int(b))
                edges.append((int(a), int(b), "tfidf_cosine", float(score)))
        if progress and (start == 0 or stop == len(view) or start // 10000 != stop // 10000):
            progress(f"TF-IDF cosine: {stop:,}/{len(view):,} rows")
    return uf.groups(ids, "tfidf"), edges, {
        "vocabulary_sizes": vocabulary_sizes, "matrix_nonzeros": matrix.nnz,
        "zero_vector_rows": int(np.sum(np.diff(matrix.indptr) == 0)), "accepted_edges": len(edges),
    }


def stratified_group_split(groups, labels, config: SplitConfig):
    """Greedy group allocation plus whole-group refinement; mixed labels retained.

    Minimize squared per-class allocation error and total-size allocation error.
    Multiple seeded restarts share the same objective for both grouping methods.
    """
    groups, inverse = np.unique(np.asarray(groups), return_inverse=True)
    labels = np.array(["<MISSING>" if text(v) == "" else str(v) for v in labels])
    classes, codes = np.unique(labels, return_inverse=True)
    counts = np.zeros((len(groups), len(classes)), dtype=np.int64)
    np.add.at(counts, (inverse, codes), 1)
    sizes = counts.sum(axis=1)
    totals = counts.sum(axis=0)
    ratios = np.array(config.ratios)
    target = ratios[:, None] * totals
    size_target = ratios * sizes.sum()
    class_weight = 1.0 / (totals.astype(float) ** 2 * len(classes))
    size_weight = config.size_weight / float(sizes.sum()) ** 2
    priority = np.maximum(sizes / max(sizes.max(), 1), (counts / totals).max(axis=1))
    best = None
    rng = np.random.default_rng(config.seed)
    for _ in range(config.split_restarts):
        order = np.lexsort((rng.random(len(groups)), -priority))
        assignment = np.full(len(groups), -1, dtype=np.int8)
        actual = np.zeros((3, len(classes)), dtype=np.int64)
        split_sizes = np.zeros(3, dtype=np.int64)
        for g in order:
            change = (2 * (actual - target) * counts[g] + counts[g] ** 2) @ class_weight
            change += size_weight * (2 * (split_sizes - size_target) * sizes[g] + sizes[g] ** 2)
            dest = int(np.argmin(change))
            assignment[g] = dest
            actual[dest] += counts[g]
            split_sizes[dest] += sizes[g]
        for _ in range(config.split_refinement_passes):
            moved = 0
            for g in order:
                source = int(assignment[g])
                remove = ((-2 * (actual[source] - target[source]) * counts[g] + counts[g] ** 2)
                          @ class_weight + size_weight *
                          (-2 * (split_sizes[source] - size_target[source]) * sizes[g] + sizes[g] ** 2))
                delta = ((2 * (actual - target) * counts[g] + counts[g] ** 2) @ class_weight
                         + size_weight * (2 * (split_sizes - size_target) * sizes[g] + sizes[g] ** 2)
                         + remove)
                delta[source] = 0
                dest = int(np.argmin(delta))
                if delta[dest] < -1e-15:
                    actual[source] -= counts[g]
                    actual[dest] += counts[g]
                    split_sizes[source] -= sizes[g]
                    split_sizes[dest] += sizes[g]
                    assignment[g] = dest
                    moved += 1
            if moved == 0:
                break
        score = float(((actual - target) ** 2 @ class_weight).sum()
                      + size_weight * ((split_sizes - size_target) ** 2).sum())
        if best is None or score < best[0]:
            best = score, assignment.copy()
    return np.array(SPLITS)[best[1][inverse]], {"objective": best[0], "classes": classes.tolist()}


def check_assignments(assignments, expected_ids):
    if assignments["record_id"].duplicated().any() or set(assignments["record_id"]) != set(expected_ids):
        raise ValueError("Assignments must cover every shared input record exactly once")
    if (not assignments["split"].isin(SPLITS).all() or assignments["group_id"].isna().any()
            or assignments["group_id"].astype(str).str.strip().eq("").any()):
        raise ValueError("Invalid split or group identifier")
    if assignments.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("Product group crosses split boundaries")
