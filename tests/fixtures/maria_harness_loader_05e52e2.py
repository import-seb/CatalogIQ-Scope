"""Unchanged loader definitions from Maria's harness; no notebook setup/training.
Commit: 05e52e2e0203dee6d58dfc2d0e72d513eed9ad95; notebook Git blob: 493f68ed0d2df09a291dd37d2dc1b558e6cba0d5.
"""
MANIFESTS = ('training_manifest.json', 'protocol.json', 'model_config.json', 'label_mapping.json', 'manifest.json')

def hash_items(items):
    d = hashlib.sha256()
    for item in items:
        payload = str(item).encode("utf-8")
        d.update(len(payload).to_bytes(8, "big"))
        d.update(payload)
    return d.hexdigest()

def _labels_from(obj):
    # Pull an ordered class list out of any of the shapes these files use.
    if isinstance(obj, list):
        return [str(x) for x in obj] or None
    if not isinstance(obj, dict): return None
    for k in ("label_names", "labels", "classes", "class_names", "label_mapping"):
        v = obj.get(k)
        if isinstance(v, list) and v: return [str(x) for x in v]
        if isinstance(v, dict) and v: obj = v; break
    else:
        if not all(isinstance(x, (int, str)) for x in obj.values()): return None
    # index -> class, or class -> index; decide by which side holds the indices.
    # An index may be bare (3) or carried by a column name ("probability_3"),
    # which is the shape the baseline handoff writes.
    def idx(x):
        if isinstance(x, bool): return None
        if isinstance(x, int): return x
        m = re.fullmatch(r"\D*?(\d+)", str(x).strip())
        return int(m.group(1)) if m else None
    def positions(xs):
        got = [idx(x) for x in xs]
        return got if None not in got and sorted(got) == list(range(len(got))) else None
    keys, vals = list(obj.keys()), list(obj.values())
    pos = positions(keys)
    if pos: return [str(v) for _, v in sorted(zip(pos, obj.values()))]
    pos = positions(vals)
    if pos: return [str(k) for _, k in sorted(zip(pos, obj.keys()))]
    return None

def load_predictions(directory, name=None):
    directory = Path(directory)
    name = name or directory.name
    p = pd.read_csv(directory/"predictions.csv",
                    dtype={"record_id": str, "true_label": str, "predicted_label": str})
    need = {"record_id", "true_label", "predicted_label", "confidence"}
    missing = need - set(p.columns)
    if missing: raise ValueError(f"{name}: predictions.csv missing {sorted(missing)}")

    labels, claimed = None, {}
    for fn in MANIFESTS:
        f = directory/fn
        if not f.is_file(): continue
        obj = json.loads(f.read_text())
        if isinstance(obj, dict) and obj.get("validation_record_ids_sha256"):
            claimed[fn] = obj["validation_record_ids_sha256"]
        if labels is None:
            labels = _labels_from(obj) or None

    # A run that names the split it trained on gets checked against ours.
    for fn, digest in claimed.items():
        hit = [k for k, v in VA_ID_HASHES.items() if v == digest]
        if not hit:
            raise ValueError(
                f"{name}: {fn} claims validation_record_ids_sha256 {digest[:16]}... "
                f"which matches neither this split's labelled validation rows nor its "
                f"whole validation partition. The run was trained against a different "
                f"split, so refusing to score it.")
        print(f"  {name:28} validation ids match this split ({hit[0]})")
    prob_cols = sorted((c for c in p.columns if c.startswith("probability_")),
                       key=lambda c: int(c.split("_")[1]))
    if labels is None and prob_cols:
        raise ValueError(f"{name}: has probability columns but no class order in any of "
                         f"{list(MANIFESTS)}, so they cannot be mapped to classes")
    if prob_cols and len(prob_cols) != len(labels):
        raise ValueError(f"{name}: {len(prob_cols)} probability columns against "
                         f"{len(labels)} classes")
    # Do not take the class order on trust. If probability_i is not class i, the
    # argmax stops agreeing with predicted_label, and every calibration and
    # margin number below is computed on the wrong column. Check it against the
    # data instead of believing the manifest.
    if prob_cols:
        argmax = np.asarray(labels)[p[prob_cols].to_numpy(dtype=float).argmax(axis=1)]
        off = (argmax != p["predicted_label"].str.strip().to_numpy()).mean()
        if off > 0.01:
            raise ValueError(
                f"{name}: the argmax of the probability columns disagrees with "
                f"predicted_label on {off*100:.1f}% of rows. The class order taken "
                f"from the manifest is probably wrong, so refusing to score it.")
        if off:
            print(f"  !! {name}: argmax disagrees with predicted_label on {off*100:.2f}% "
                  f"of rows (ties, or a stale predicted_label column)")

    leaked = set(p["record_id"]) & TEST_IDS
    if leaked:
        raise ValueError(f"{name}: {len(leaked)} prediction rows belong to the RESERVED "
                         f"TEST partition. Refusing to score them.")
    outside = set(p["record_id"]) - VA_IDS
    if outside:
        print(f"  !! {name}: {len(outside):,} rows are not validation rows; dropping them")
        p = p[p["record_id"].isin(VA_IDS)]

    p["group_id"] = p["record_id"].map(GROUP_OF)
    p["correct"] = p["true_label"].str.strip() == p["predicted_label"].str.strip()
    if prob_cols:
        P = p[prob_cols].to_numpy(dtype=float)
        order = np.argsort(-P, axis=1)
        p["top1_prob"] = P[np.arange(len(P)), order[:, 0]]
        p["top2_prob"] = P[np.arange(len(P)), order[:, 1]] if P.shape[1] > 1 else 0.0
        p["margin"] = p["top1_prob"] - p["top2_prob"]
        p["_probs"] = list(P)
    else:
        p["top1_prob"] = pd.to_numeric(p["confidence"], errors="coerce")
        p["margin"] = np.nan
        p["_probs"] = None
    print(f"  {name:28} {len(p):,} rows   accuracy {p['correct'].mean()*100:6.2f}%   "
          f"{'with' if prob_cols else 'no'} probabilities")
    return {"name": name, "frame": p, "labels": labels, "dir": directory,
            "split_checked": bool(claimed)}
