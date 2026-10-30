"""Unchanged Maria calibration definitions, commit 05e52e2e0203dee6d58dfc2d0e72d513eed9ad95. No notebook setup or fitting loop."""
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
