"""Isolated append-only prototype. No inference routing, ingestion server or deployment.

The calling service must authenticate principals and own the eligibility registry.
Hash chains detect accidental corruption; an external signed anchor is still needed
against a malicious database owner. Never accept caller-supplied novelty flags.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from core import FIELDS
from data import fingerprint

TARGETS = ('Segment', 'Mnfr', 'TargetAgeGroup', 'Sub-Segment', 'Brand')
APP_ID = 0x43495131


@contextmanager
def connection(*args, **kwargs):
    db = sqlite3.connect(*args, **kwargs)
    try:
        with db:
            yield db
    finally:
        db.close()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()


class EligibilityIndex:
    """Server-owned development membership, supplied without opening protected data.

    Attestations must come from the authorized data steward, covering reserved-data
    exclusion and verified identity/grouping. Absence from this index proves nothing
    about protected membership. Unattested unknowns therefore remain quarantined.
    """
    def __init__(self, version, populations, attestations=None):
        self.version = version
        self.populations = populations
        self.attestations = attestations or {}

    def classify(self, event):
        p = self.populations.get(event['target'], {})
        rid, gid = event['record_id'], event['group_id']
        fp = fingerprint(event['inputs'])
        # Validation takes precedence even when an identifier claims training/new.
        for role in ('validation', 'train'):
            rows = p.get(role, [])
            if any(rid == r['record_id'] or gid == r['group_id'] or fp == r['fingerprint'] for r in rows):
                return 'validation_excluded' if role == 'validation' else 'seen_training'
        a = self.attestations.get(gid)
        if (a and a.get('reserved_exclusion_verified') is True
                and a.get('training_exclusion_verified') is True
                and a.get('identity_verified') is True and a.get('provenance')
                and event['entity_id'] in a.get('entity_ids', [])):
            return 'unseen_attested'
        return 'unverified_new_candidate'


class Ledger:
    def __init__(self, path, allowed_root, catalog, registry, principals):
        self.path, root = Path(path).resolve(), Path(allowed_root).resolve()
        if root not in self.path.parents:
            raise ValueError('Ledger must be inside its isolated output root')
        if set(catalog) != set(TARGETS):
            raise ValueError('All five target catalogs required')
        self.catalog, self.registry, self.principals = catalog, registry, principals
        for c in catalog.values():
            if not c['version'] or not c['labels'] or len(c['labels']) != len(set(c['labels'])):
                raise ValueError('Invalid ontology')
        binding = canonical(catalog)
        if self.path.exists():
            # Inspect a foreign database read-only before any write/DDL.
            with connection(self.path.as_uri() + '?mode=ro', uri=True) as db:
                if db.execute('PRAGMA application_id').fetchone()[0] != APP_ID:
                    raise ValueError('Foreign database refused')
                if db.execute('SELECT value FROM metadata WHERE name=?', ('catalog',)).fetchone() != (binding,):
                    raise ValueError('Catalog mismatch; explicit ledger migration required')
        else:
            with connection(self.path) as db:
                db.execute(f'PRAGMA application_id={APP_ID}')
                db.execute('CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL)')
                db.execute('INSERT INTO metadata VALUES (?,?)', ('catalog', binding))
                db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, request TEXT NOT NULL, received TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL)')
                for operation in ('UPDATE', 'DELETE'):
                    db.execute(f"CREATE TRIGGER no_{operation.lower()} BEFORE {operation} ON events BEGIN SELECT RAISE(ABORT,'append only'); END")
        self.verify()

    def events(self):
        with connection(self.path.as_uri() + '?mode=ro', uri=True) as db:
            return db.execute('SELECT seq,event_id,request,received,previous,hash FROM events ORDER BY seq').fetchall()

    def verify(self):
        rows = self._verified_rows()
        return rows[-1][-1] if rows else '0' * 64

    def _verified_rows(self):
        rows = self.events()
        previous = '0' * 64
        for expected_seq, (seq, eid, request, received, prior, hashed) in enumerate(rows, 1):
            if seq != expected_seq or prior != previous or digest([seq, eid, request, received, prior]) != hashed:
                raise ValueError('Ledger integrity failure')
            previous = hashed
        return rows

    def append(self, event_id, request, principal):
        if not isinstance(event_id, str) or not event_id or len(event_id) > 200:
            raise ValueError('Stable external event ID required')
        role = self.principals.get(principal)
        if role not in ('reviewer', 'adjudicator'):
            raise PermissionError('Unknown principal')
        request = json.loads(canonical(request))
        kind = request.get('kind')
        if kind == 'review':
            required = ('target', 'entity_id', 'record_id', 'group_id', 'inputs', 'model_version',
                        'ontology_version', 'source_system', 'occurred_at', 'action', 'human_approved', 'synthetic',
                        'identity_verified', 'grouping_verified')
            if any(k not in request for k in required):
                raise ValueError('Missing review provenance')
            if request['target'] not in TARGETS or request['action'] not in ('confirm', 'correct', 'reject', 'defer'):
                raise ValueError('Invalid target/action')
            if any(not isinstance(request[k], str) or not request[k].strip() for k in
                   ('entity_id', 'record_id', 'group_id', 'model_version', 'ontology_version', 'source_system')):
                raise ValueError('Verified identity/provenance references required')
            if set(request['inputs']) != set(FIELDS) or any(not isinstance(v, str) for v in request['inputs'].values()):
                raise ValueError('Exactly five full string input fields required')
            if any(type(request[k]) is not bool for k in ('human_approved','synthetic','identity_verified','grouping_verified')):
                raise ValueError('Approval and synthetic flags must be boolean')
            occurred = datetime.fromisoformat(request['occurred_at'])
            if occurred.tzinfo is None or occurred.utcoffset() is None:
                raise ValueError('Timezone-aware event time required')
            if request['action'] in ('confirm', 'correct') and not isinstance(request.get('label'), str):
                raise ValueError('Confirmation label required')
        elif kind in ('resolve', 'revoke'):
            if role != 'adjudicator' or not request.get('reason'):
                raise PermissionError('Adjudicator and rationale required')
        else:
            raise ValueError('Unsupported event kind')
        request['principal'] = principal  # authenticated server identity overrides payload
        raw = canonical(request)
        if len(raw.encode()) > 100_000:
            raise ValueError('Event exceeds bounded payload')
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            previous = self.verify()
            exists = db.execute('SELECT request,hash FROM events WHERE event_id=?', (event_id,)).fetchone()
            if exists:
                if exists[0] != raw:
                    raise ValueError('Idempotency-key collision')
                return exists[1]
            known = {eid: json.loads(r) for _, eid, r, *_ in self.events()}
            if kind == 'revoke':
                if request.get('event_id') not in known or known[request['event_id']]['kind'] == 'revoke':
                    raise ValueError('Invalid revocation target')
            if kind == 'resolve':
                state = self.snapshot()
                match = [v for v in state['entries'] if v['key'] == request.get('key')]
                if (len(match) != 1 or match[0]['status'] != 'conflict'
                        or sorted(request.get('covers', [])) != sorted(match[0]['review_event_ids'])
                        or request.get('label') not in self.catalog[match[0]['target']]['labels']):
                    raise ValueError('Resolution must cover the entire current conflict')
            seq = db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM events').fetchone()[0]
            received = datetime.now(timezone.utc).isoformat()
            hashed = digest([seq, event_id, raw, received, previous])
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)', (seq, event_id, raw, received, previous, hashed))
        return hashed

    def snapshot(self):
        verified = self._verified_rows()
        head = verified[-1][-1] if verified else '0' * 64
        rows = [(eid, json.loads(raw)) for _, eid, raw, *_ in verified]
        revoked = {r['event_id'] for _, r in rows if r['kind'] == 'revoke'}
        active = [(eid, r) for eid, r in rows if eid not in revoked]
        grouped, resolutions = {}, []
        for eid, r in active:
            if r['kind'] == 'resolve':
                resolutions.append((eid, r)); continue
            if r['kind'] != 'review':
                continue
            key = digest([r['entity_id'], r['target'], digest(r['inputs'])])
            grouped.setdefault(key, []).append((eid, r))
        entries = []
        for key, reviews in grouped.items():
            first = reviews[0][1]; target = first['target']
            approved = [(eid, r) for eid, r in reviews if r['human_approved'] and r['action'] in ('confirm', 'correct')]
            labels = {r['label'] for _, r in approved}
            hold = any(r['action'] == 'reject' for _, r in reviews)
            incompatible = any(r['ontology_version'] != self.catalog[target]['version'] or
                               r.get('label') not in self.catalog[target]['labels'] for _, r in approved)
            status = 'conflict' if len(labels) > 1 or hold else 'active' if labels else 'audit_only'
            label = next(iter(labels)) if len(labels) == 1 else None
            ids = sorted(eid for eid, r in reviews if r['action'] != 'defer')
            applied = None
            for reid, resolution in resolutions:
                if resolution['key'] == key and sorted(resolution['covers']) == ids:
                    status, label, applied = 'active', resolution['label'], reid
            if incompatible:
                status = 'ontology_quarantine'
            if len({r['group_id'] for _, r in reviews}) != 1:
                status = 'identity_conflict'
            if any(not r['identity_verified'] or not r['grouping_verified'] for _, r in reviews):
                status = 'identity_quarantine'
            eligibilities = {self.registry.classify(r) for _, r in reviews}
            eligibility = ('validation_excluded' if 'validation_excluded' in eligibilities else
                           'seen_training' if 'seen_training' in eligibilities else
                           'unseen_attested' if eligibilities == {'unseen_attested'} else 'unverified_new_candidate')
            synthetic = any(r['synthetic'] for _, r in reviews)
            entries.append({'key': key, 'target': target, 'entity_id': first['entity_id'],
                            'group_id': first['group_id'], 'record_id': first['record_id'],
                            'inputs': first['inputs'], 'input_sha256': digest(first['inputs']),
                            'fingerprint': fingerprint(first['inputs']), 'status': status, 'label': label,
                            'eligibility': eligibility, 'synthetic': synthetic, 'review_event_ids': ids,
                            'resolution': applied, 'human_approval_required': True})
        # Detect contradictory aliases/identities for an identical feature tuple.
        by_fp = {}
        for entry in entries:
            if entry['status'] == 'active':
                by_fp.setdefault((entry['target'], entry['fingerprint']), set()).add(entry['label'])
        for entry in entries:
            if len(by_fp.get((entry['target'], entry['fingerprint']), set())) > 1:
                entry['status'] = 'identity_conflict'
        return {'schema': 'catalogiq-ledger-snapshot-v1', 'event_head': head,
                'registry_version': self.registry.version, 'catalog_sha256': digest(self.catalog),
                'registry_sha256': digest([self.registry.populations, self.registry.attestations]),
                'entries': entries, 'human_approval_required': True}

    def suggest(self, entity_id, target, inputs):
        key = digest([entity_id, target, digest(inputs)])
        values = [e for e in self.snapshot()['entries'] if e['key'] == key and e['status'] == 'active'
                  and e['eligibility'] != 'validation_excluded' and not e['synthetic']]
        if len(values) != 1:
            return None
        return {k: v for k, v in values[0].items() if k != 'inputs'}

    def export(self, allow_synthetic=False):
        snapshot = self.snapshot()
        snapshot['entries'] = [e for e in snapshot['entries'] if e['status'] == 'active'
                               and e['eligibility'] in ('seen_training', 'unseen_attested')
                               and (allow_synthetic or not e['synthetic'])]
        snapshot['not_a_deployment_authorization'] = True
        return snapshot
