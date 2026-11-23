"""Generate private, retailer-stratified audit inputs for the frozen candidate bands."""
import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from pathlib import Path
from zipfile import ZipFile

from replay_card_risk import score_catalog
from replay_partial_review import ROOT, load_inputs


FIELDS = ("Segment", "Sub-Segment", "Brand", "Mnfr", "TargetAgeGroup")


def evidence_text(text):
    value = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    value = " ".join(value.split())
    return "" if value.casefold() in {"null", "none", "nan"} else value


def draw(args):
    _, lookup, _ = load_inputs(args.handoff, args.validation)
    catalog, scores = score_catalog(args.catalog, args.warnings, lookup)
    rng = random.Random(args.seed)
    strata, selected = [], {}
    for label, mask in [("98%+", scores.ge(.98)), ("95–<98%", scores.ge(.95) & scores.lt(.98))]:
        band = catalog.loc[mask]
        if band.empty or band.Retailer.str.strip().eq("").any():
            raise ValueError("Missing band or retailer")
        for retailer, store in band.groupby("Retailer", sort=True):
            size = len(store)
            n = min(size, max(5, math.ceil(300 * size / len(band))))
            ids = rng.sample(sorted(store.index), n)
            strata.append([label, retailer, size, n])
            selected.update({key: label for key in ids})
    if sum(s[3] for s in strata) != len(selected):
        raise ValueError("Duplicate sample IDs")
    with args.source.open(encoding="utf-8-sig", newline="") as f:
        evidence = {}
        for row in csv.DictReader(f):
            key = row["JoiningKey"]
            if key not in selected:
                continue
            if key in evidence or any(evidence_text(row[t]) for t in FIELDS):
                raise ValueError("Unexpected duplicate or supplied reference label")
            evidence[key] = row
    keys = list(selected)
    rng.shuffle(keys)
    sample = []
    for key in keys:
        row, src = catalog.loc[key], evidence[key]
        if src["Retailer"] != row.Retailer:
            raise ValueError("Source retailer mismatch")
        sample.append({"id": key, "band": selected[key], "retailer": row.Retailer,
                       "product": row.ProductName, "brand": row.ProductBrand,
                       "category": row.ProductCategory, "url": row.ProductUrl,
                       "description": evidence_text(src["ProductDescription"]),
                       "contents": evidence_text(src["ProductContents"]),
                       "labels": [row[f"catalogiq:{t}:original_label"] for t in FIELDS]})
    routes = json.loads((ROOT / "src/catalogiq/resources/inference/frozen_routes.json").read_text())["routes"]
    with ZipFile(args.handoff) as z:
        vocab = []
        for t in FIELDS:
            manifest = routes[t]["manifest"].split("/", 1)[1]
            vocab.append(sorted(json.loads(z.read(manifest))["label_names"]))
    with args.source.open("rb") as f:
        source_hash = hashlib.file_digest(f, "sha256").hexdigest()
    return {"seed": args.seed, "fields": FIELDS, "strata": strata, "sample": sample,
            "vocab": vocab, "source_sha256": source_hash}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "validation", "catalog", "warnings", "source", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    payload = draw(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print(json.dumps(payload["strata"], ensure_ascii=True))
