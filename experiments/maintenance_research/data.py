"""Full-input fingerprints and group-disjoint maintenance/replay selection."""
import hashlib
import json
import re
import unicodedata

from core import FIELDS


def fingerprint(row):
    # Full untruncated feature tuple; no target, model score or validation label.
    values=[re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(row.get(k,'')))).strip().casefold() for k in FIELDS]
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def select_groups(rows,wrong,seed=42,incoming_cap=512,replay_cap=2048):
    """Error-enriched seen-training proxies and disjoint whole-group replay."""
    import random
    rng=random.Random(seed);groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group_id'],[]).append(i)
    order=sorted(groups);rng.shuffle(order)
    priority=sorted(order,key=lambda g:-sum(bool(wrong[i]) for i in groups[g]))
    incoming=[];incoming_groups=set()
    for gid in priority:
        if len(incoming)+len(groups[gid])<=incoming_cap:
            incoming.extend(groups[gid]);incoming_groups.add(gid)
    remaining=[g for g in order if g not in incoming_groups]
    replay=[];replay_groups=set();covered=set()
    # Represent classes where a disjoint replay group is available, then fill randomly.
    for label in sorted({r['y'] for r in rows}):
        candidates=[g for g in remaining if g not in replay_groups and any(rows[i]['y']==label for i in groups[g])]
        for gid in candidates:
            if len(replay)+len(groups[gid])<=replay_cap:
                replay.extend(groups[gid]);replay_groups.add(gid)
                covered.update(rows[i]['y'] for i in groups[gid]);break
    for gid in remaining:
        if gid not in replay_groups and len(replay)+len(groups[gid])<=replay_cap:
            replay.extend(groups[gid]);replay_groups.add(gid)
    if not incoming or not replay or incoming_groups&replay_groups:raise RuntimeError('Invalid disjoint maintenance/replay partition')
    return sorted(incoming),sorted(replay)
