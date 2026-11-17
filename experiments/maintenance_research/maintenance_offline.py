"""Offline paired validation and reversible Brand counters; no production calls."""
import copy
import hashlib
import numpy as np
from core import hash_items, metrics
from methods import forgetting


def compare_records(baseline, candidate, names, *, baseline_binding, candidate_binding):
    """Explicit identical record/truth contract; never silently intersect populations."""
    if len(names) != len(set(names)) or not names:
        raise ValueError('Ordered vocabulary required')
    for field in ('target', 'baseline_model_sha256', 'input_contract_sha256', 'evaluation_population_sha256', 'label_names'):
        if field not in baseline_binding or baseline_binding[field] != candidate_binding.get(field):
            raise ValueError('Candidate/baseline evaluation binding mismatch: '+field)
    if baseline_binding['label_names'] != names:
        raise ValueError('Vocabulary binding mismatch')
    def index(rows):
        result = {}
        for row in rows:
            if row['record_id'] in result or row['truth'] not in names or row['prediction'] not in names:
                raise ValueError('Duplicate identity or unknown label')
            result[row['record_id']] = row
        return result
    a, b = index(baseline), index(candidate)
    if not a or set(a) != set(b) or any(a[k]['truth'] != b[k]['truth'] for k in a):
        raise ValueError('Paired population/truth mismatch')
    ids = sorted(a)
    if hash_items(ids) != baseline_binding['evaluation_population_sha256']:
        raise ValueError('Evaluation population seal mismatch')
    y = np.array([names.index(a[k]['truth']) for k in ids])
    old = np.array([names.index(a[k]['prediction']) for k in ids])
    new = np.array([names.index(b[k]['prediction']) for k in ids])
    before, after = metrics(y, old, names), metrics(y, new, names)
    f = forgetting(y, old, new, names)
    return {'baseline': before, 'candidate': after, 'forgetting': f,
            'original_retention_gate': f['accuracy_change'] >= -.005 and f['macro_f1_change'] >= -.01
              and f['worst_supported_class_recall_drop'] <= .02,
            'precision_drops': {n: before['per_class'][n]['precision'] - after['per_class'][n]['precision'] for n in names},
            'small_support_classes': [n for n in names if before['per_class'][n]['support'] < 30],
            'offline_screen_only': True, 'human_approval_required': True}


def plan_cohorts(snapshot, holdout_fraction=.2):
    """Deterministic whole-group proposals, shared across targets; never a data split edit.

    Call only on a frozen eligible snapshot before evaluating candidates. Existing
    training corrections may update but never become unseen evaluation evidence.
    Conflicting labels within an identical target/group/input cannot earn extra votes.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError('Holdout fraction must be between zero and one')
    entries=snapshot['entries']
    by_key={}
    group_roles={}
    for e in entries:
        if e['status']!='active' or e['eligibility'] not in ('seen_training','unseen_attested') or e['synthetic']:
            raise ValueError('Cohort contains quarantined or synthetic records')
        k=(e['target'],e['group_id'],e['input_sha256'])
        if k in by_key and by_key[k]['label']!=e['label']:
            raise ValueError('Contradictory group/input labels')
        by_key.setdefault(k,e)
        group_roles.setdefault(e['group_id'],set()).add(e['eligibility'])
    result={'update':[],'incoming_holdout':[], 'event_head':snapshot['event_head'],
            'registry_sha256':snapshot['registry_sha256'], 'proposal_only':True}
    for e in by_key.values():
        # A group seen in any included target cannot be called wholly unseen.
        known='seen_training' in group_roles[e['group_id']]
        fraction=int(hashlib.sha256(('catalogiq-incoming-holdout-v1:'+e['group_id']).encode()).hexdigest()[:16],16)/2**64
        role='incoming_holdout' if not known and fraction<holdout_fraction else 'update'
        result[role].append(e)
    assert not ({e['group_id'] for e in result['update']}&{e['group_id'] for e in result['incoming_holdout']})
    return result


class BrandOverlay:
    """Separate candidate format, anchored prior. Original model remains unchanged.

    Contributions are owned by distinct verified products, not review events.
    Baseline replacement requires exact old contribution binding. New vocabulary
    is rejected. Production must resolve grouping/identity before calling this.
    """
    def __init__(self, base):
        self.base = copy.deepcopy(base)
        self.changes = {}

    def set_contribution(self, product, new, *, original=None):
        def check(pair):
            if pair is not None and (not isinstance(pair[0], str) or not pair[0]
                                     or pair[0] in self.base['missing_normalized_values']
                                     or pair[1] not in self.base['label_names']):
                raise ValueError('Canonical normalized nonmissing key and existing label required')
        check(new); check(original)
        if product in self.changes and self.changes[product][0] != original:
            raise ValueError('Original contribution binding cannot change')
        previous = copy.deepcopy(self.changes)
        self.changes[product] = (original, new)
        try:
            self.materialize()
        except Exception:
            self.changes = previous
            raise

    def remove_change(self, product):
        del self.changes[product]

    def materialize(self, recompute_prior=False):
        counts = {k: dict(v) for k, v in self.base['key_counts'].items()}
        classes = np.array(self.base['training_class_counts'], dtype=np.int64)
        touched = set()
        for old, new in self.changes.values():
            if old == new:
                continue
            for pair, delta in ((old, -1), (new, 1)):
                if pair is None:
                    continue
                key, label = pair; idx = self.base['label_names'].index(label)
                touched.add(key); bucket = counts.setdefault(key, {})
                bucket[idx] = bucket.get(idx, 0) + delta; classes[idx] += delta
                if bucket[idx] < 0:
                    raise ValueError('Cannot subtract nonexistent baseline contribution')
        if (classes <= 0).any():
            raise ValueError('Cannot remove all support for a supported Brand')
        sparse = {k: [[i, n] for i, n in sorted(v.items()) if n] for k, v in counts.items() if sum(v.values())}
        prior = classes / classes.sum() if recompute_prior else np.array(self.base['class_prior'])
        return {'format': 'isolated-brand-count-overlay-v1', 'prior_policy': 'recomputed' if recompute_prior else 'anchored_baseline',
                'class_prior': prior.tolist(), 'training_class_counts': classes.tolist(),
                'key_counts': sparse, 'touched_keys': sorted(touched), 'label_names': self.base['label_names'],
                'human_approval_required': True}


def brand_probabilities(model, normalized_keys):
    prior = np.asarray(model['class_prior'], dtype=float)
    result = np.tile(prior, (len(normalized_keys), 1))
    for row, key in enumerate(normalized_keys):
        pairs = model['key_counts'].get(key)
        if pairs:
            pairs = np.array(pairs, dtype=np.int64)
            result[row, pairs[:, 0]] += pairs[:, 1]
            result[row] /= pairs[:, 1].sum() + 1.
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError('Invalid Brand probabilities')
    return result
