"""Title-first TF-IDF grouping with independent supporting-text scores.

Names and packaging-normalized names each have an exhaustive sparse cosine
threshold graph. All name features survive by default, including uncommon
formula names. Description/contents similarity can support a borderline name
match, but cannot reduce a strong name match or create a link on its own.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as normalize_vectors

from .split_matching import compatible, prepare_products
from .splitting import Components, grouping_view, text


def _fit(values, max_features=None):
    """Keep ubiquitous and rare title features; do not fit any target label."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
        dtype=np.float64, max_features=max_features, max_df=1.0,
    )
    try:
        matrix = vectorizer.fit_transform(values)
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return csr_matrix((len(values), 0), dtype=np.float64), 0
    return matrix, len(vectorizer.vocabulary_)


def _supporting_matrix(frame, config):
    view = grouping_view(frame)
    matrices, sizes = [], {}
    for field, limit, weight in (
        ("ProductDescription", config.description_chars, config.tfidf_weights[1]),
        ("ProductContents", config.contents_chars, config.tfidf_weights[2]),
    ):
        if weight == 0:
            continue
        values = [text(value)[:limit] for value in view[field]]
        matrix, size = _fit(values, config.tfidf_max_features)
        sizes[field] = size
        if size:
            matrices.append(matrix * weight)
    if not matrices:
        return csr_matrix((len(frame), 0), dtype=np.float64), sizes
    return normalize_vectors(hstack(matrices, format="csr"), copy=False), sizes


def _threshold_pairs(matrix, threshold, chunk_size, progress, field):
    """Enumerate every cosine-qualified pair, with no blocking or top-k cap."""
    transpose = matrix.T.tocsr()
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        similarities = (matrix[start:stop] @ transpose).tocsr()
        similarities.data[similarities.data < threshold - 1e-12] = 0
        similarities.eliminate_zeros()
        coo = similarities.tocoo()
        for left, right, score in zip(coo.row + start, coo.col, coo.data):
            if left < right:
                yield int(left), int(right), float(score)
        if progress and (start == 0 or stop == matrix.shape[0]
                         or start // 10000 != stop // 10000):
            progress(f"TF-IDF v2 {field} cosine: {stop:,}/{matrix.shape[0]:,} rows")


def refine_tfidf_groups(frame, ids, config, progress=None):
    """Return stable connected components, accepted links, and diagnostics.

    Exact identifiers never create TF-IDF links. The shared label-blind evidence
    guard rejects conflicting brands, core names, and known formulations. A
    component guard also prevents an unknown-formulation bridge from joining
    components containing mutually conflicting known formulations.
    """
    if len(frame) != len(ids):
        raise ValueError("Record identities must align with the input frame")
    products = prepare_products(frame, config)
    title, title_size = _fit([product.name for product in products],
                            config.tfidf_name_max_features)
    family, family_size = _fit([product.family for product in products],
                              config.tfidf_name_max_features)
    support, support_sizes = _supporting_matrix(frame, config)
    stats = Counter()
    proposals = {}

    def propose(left, right, reason, score):
        key = (left, right)
        candidate = (reason, score)
        previous = proposals.get(key)
        # Strongest evidence wins; reason breaks exact-score ties consistently.
        if previous is None or (-round(score, 12), reason) < (-round(previous[1], 12), previous[0]):
            proposals[key] = candidate

    for left, right, score in _threshold_pairs(
        title, config.tfidf_supported_name_threshold,
        config.cosine_chunk_size, progress, "title",
    ):
        stats["title_candidate_pairs"] += 1
        if not compatible(products[left], products[right], config, allow_brand_override=True):
            stats["title_guard_rejections"] += 1
            continue
        if score >= config.tfidf_name_threshold - 1e-12:
            propose(left, right, "tfidf_title_cosine", score)
        elif support.shape[1]:
            support_score = float(support[left].multiply(support[right]).sum())
            if support_score >= config.tfidf_support_threshold - 1e-12:
                propose(left, right, "tfidf_supported_title_cosine", score)
            else:
                stats["insufficient_support_pairs"] += 1

    for left, right, score in _threshold_pairs(
        family, config.tfidf_family_threshold,
        config.cosine_chunk_size, progress, "family",
    ):
        stats["family_candidate_pairs"] += 1
        if compatible(products[left], products[right], config, allow_brand_override=True):
            propose(left, right, "tfidf_family_cosine", score)
        else:
            stats["family_guard_rejections"] += 1

    uf = Components(len(products))
    signatures = [{product.ingredients} if product.ingredients else set()
                  for product in products]
    edges = []
    # Stable score/identity order matters when a component conflict prevents a
    # later edge. It keeps this safeguard independent of input row/chunk order.
    ordered = sorted(proposals.items(), key=lambda item: (
        -round(item[1][1], 12), *sorted((str(ids[item[0][0]]), str(ids[item[0][1]]))),
        item[1][0],
    ))
    for (left, right), (reason, score) in ordered:
        root_left, root_right = uf.find(left), uf.find(right)
        if root_left != root_right:
            if any(a.isdisjoint(b) for a in signatures[root_left] for b in signatures[root_right]):
                stats["component_formulation_rejections"] += 1
                continue
            merged = signatures[root_left] | signatures[root_right]
            uf.union(root_left, root_right)
            signatures[uf.find(root_left)] = merged
        edges.append((left, right, reason, score))
        stats[reason] += 1

    return uf.groups(ids, "tfidf"), edges, {
        **dict(stats),
        "grouping_version": 2,
        "vocabulary_sizes": {"ProductName": title_size, "ProductNameFamily": family_size,
                             **support_sizes},
        "name_features_uncapped": config.tfidf_name_max_features is None,
        "matrix_nonzeros": int(title.nnz + family.nnz + support.nnz),
        "zero_vector_rows": int(np.sum(np.diff(title.indptr) == 0)),
        "accepted_edges": len(edges),
        "candidate_search": "exhaustive_title_and_family_cosine_threshold_graphs",
    }
