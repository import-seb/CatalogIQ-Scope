"""Reviewed Maria reference definitions; extracted unchanged, not a new evaluator.

Source: import-seb/CatalogIQ-Scope, commit a41608e83dd3f1420ce5cb63a28b022254f879d4
notebooks/04-segment-evaluation.ipynb, Git blob 5670c9d55777f8f9b313b3f1e7835145fd5a6885.
The three calibration definitions are identical to the former 05e52e2 test
fixture. temperature_scale is retained for those historical compatibility tests;
saved-findings replay never calls it. Use maria_harness_compatibility to load
only pinned definitions, never notebook setup, fitting loops or discovery.
"""

def queue_metrics(flagged, correct):
    flagged = np.asarray(flagged, dtype=bool); wrong = ~np.asarray(correct, dtype=bool)
    n = len(flagged)
    return {"review_load_%": round(flagged.mean()*100, 2),
            "flag_precision_%": round((wrong & flagged).sum()/max(flagged.sum(), 1)*100, 2),
            "miss_rate_%": round((wrong & ~flagged).sum()/max(wrong.sum(), 1)*100, 2),
            "wrong_caught": int((wrong & flagged).sum()), "wrong_total": int(wrong.sum())}

def coverage_risk(f, labels, conf_col="top1_prob", points=(1.00, 0.90, 0.75, 0.50, 0.25)):
    s = f.sort_values(conf_col, ascending=False)
    # Average over the same class set as the §2 headline, so the 100% row here
    # and the scoreboard are the same number rather than two near-misses.
    cls = class_set(f, labels)
    # The comparison that matters is against the class mix at full coverage, not
    # against the declared vocabulary: a class with no validation support is
    # absent at 100% too, so testing against the vocabulary would mark every row
    # incomparable on any target that has one.
    present = lambda d: set(d["true_label"].astype("string").str.strip()) | \
                        set(d["predicted_label"].astype("string").str.strip())
    base = present(s)
    rows = []
    for c in points:
        k = max(1, int(round(len(s)*c)))
        head = s.iloc[:k]
        # Taking the most confident rows first does not keep the class mix. On a
        # confident model the top half is almost all majority class, so a macro
        # average over whatever survives is not the same quantity as the macro F1
        # at full coverage and must not be put in the same column. Report it only
        # while every class that was there at full coverage is still there.
        kept = present(head)
        full = base <= kept
        rows.append({"coverage_%": round(c*100, 1), "answered": k,
                     "abstained": len(s)-k,
                     "accuracy_on_answered_%": round(head["correct"].mean()*100, 2),
                     "classes_left": len(kept),
                     "macro_F1_on_answered_%": (round(macro_f1_on(head, cls), 2)
                                                if full else np.nan),
                     "confidence_threshold": round(float(head[conf_col].min()), 4)})
    return pd.DataFrame(rows)

def class_set(frame, labels=None):
    truth = set(frame["true_label"].astype("string").str.strip())
    return sorted(set(labels or []) | truth)

def macro_f1_on(frame, classes, pct=True):
    # Fixed class list. The bootstrap resamples rows, so the denominator must not
    # be recomputed per sample or the interval is not an interval on one metric.
    v = f1_score(frame["true_label"], frame["predicted_label"],
                 labels=classes, average="macro", zero_division=0)
    return v*100 if pct else v

def ece_and_reliability(conf, correct, bins=10):
    conf = np.asarray(conf, dtype=float); correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins+1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any(): continue
        acc, mean_conf, share = correct[m].mean(), conf[m].mean(), m.mean()
        ece += share*abs(acc-mean_conf)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "rows": int(m.sum()),
                     "share_%": round(share*100, 1),
                     "mean_confidence": round(mean_conf, 3),
                     "accuracy": round(acc, 3),
                     "gap": round(acc-mean_conf, 3)})
    return pd.DataFrame(rows), ece

def multiclass_brier(probs, true_label, labels):
    if probs is None: return float("nan")
    P = np.vstack(probs)
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros_like(P)
    for r, t in enumerate(true_label):
        j = idx.get(str(t).strip())
        if j is not None: Y[r, j] = 1.0
    return float(((P-Y)**2).sum(axis=1).mean())

def temperature_scale(probs, true_label, labels, grid=None):
    """One parameter on the logits. Argmax is invariant, so accuracy cannot move."""
    P = np.clip(np.vstack(probs), 1e-12, 1.0)
    logits = np.log(P)
    idx = {l: i for i, l in enumerate(labels)}
    y = np.array([idx.get(str(t).strip(), -1) for t in true_label])
    keep = y >= 0
    logits, y = logits[keep], y[keep]
    best = (1.0, float("inf"))
    for T in (grid if grid is not None else np.linspace(0.5, 5.0, 46)):
        z = logits/T
        z = z - z.max(axis=1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean()
        if nll < best[1]: best = (float(T), float(nll))
    T = best[0]
    z = logits/T; z = z - z.max(axis=1, keepdims=True)
    Q = np.exp(z); Q = Q/Q.sum(axis=1, keepdims=True)
    return T, Q, keep
