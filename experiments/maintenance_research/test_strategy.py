"""Synthetic safety/behavior tests; all databases live in supervisor TEMP."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np

from core import FIELDS, hash_items
from correction_ledger import Ledger, EligibilityIndex, TARGETS, connection
from data import fingerprint
from maintenance_offline import BrandOverlay, brand_probabilities, compare_records, plan_cohorts


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.catalog = {t: {'version': 'v1', 'labels': ['a', 'b']} for t in TARGETS}
        self.inputs = {f: f for f in FIELDS}
        self.registry = EligibilityIndex('synthetic-v1', {}, {'g': {'reserved_exclusion_verified': True,
            'training_exclusion_verified': True, 'identity_verified': True, 'provenance': 'synthetic steward', 'entity_ids': ['p']}})
        self.ledger = Ledger(self.root/'ledger.sqlite', self.root, self.catalog, self.registry,
                             {'r': 'reviewer', 'a': 'adjudicator'})

    def tearDown(self):
        self.temp.cleanup()

    def event(self, **values):
        return dict(kind='review', target='Segment', entity_id='p', record_id='r1', group_id='g',
                    inputs=self.inputs, model_version='base-sha', ontology_version='v1', source_system='synthetic',
                    occurred_at='2026-11-14T12:00:00+00:00', action='confirm', label='a',
                    human_approved=True, synthetic=True, identity_verified=True, grouping_verified=True, **values)

    def add(self, eid='e1', **values):
        e = self.event(); e.update(values)
        return self.ledger.append(eid, e, 'r')

    def test_idempotency_collision_and_one_vote(self):
        self.assertEqual(self.add(), self.add())
        with self.assertRaises(ValueError): self.add(label='b')
        self.add('e2')
        self.assertEqual(len(self.ledger.export(True)['entries']), 1)

    def test_conflict_adjudication_reopening_revocation(self):
        self.add(); self.add('e2', label='b')
        entry = self.ledger.snapshot()['entries'][0]
        self.assertEqual(entry['status'], 'conflict')
        resolution = {'kind': 'resolve', 'key': entry['key'], 'covers': ['e1','e2'], 'label': 'a', 'reason': 'checked source'}
        with self.assertRaises(PermissionError): self.ledger.append('s1', resolution, 'r')
        self.ledger.append('s1', resolution, 'a')
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'active')
        self.add('e3', label='b')
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'conflict')
        self.ledger.append('v1', {'kind':'revoke','event_id':'e3','reason':'duplicate source'}, 'a')
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'active')
        self.ledger.append('v2', {'kind':'revoke','event_id':'s1','reason':'recheck'}, 'a')
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'conflict')

    def test_no_automatic_approval_and_synthetic_export(self):
        self.add()
        self.assertEqual(self.ledger.export()['entries'], [])
        self.assertIsNone(self.ledger.suggest('p', 'Segment', self.inputs))
        self.assertTrue(self.ledger.export(True)['human_approval_required'])

    def test_all_five_targets(self):
        for i, target in enumerate(TARGETS): self.add(str(i), target=target)
        self.assertEqual(len(self.ledger.export(True)['entries']), 5)

    def test_unknown_label_and_ontology_quarantine(self):
        self.add(label='new')
        self.assertEqual(self.ledger.export(True)['entries'], [])
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'ontology_quarantine')

    def test_validation_match_precedes_novelty(self):
        self.registry.populations = {'Segment': {'validation':[{'record_id':'other','group_id':'other','fingerprint':fingerprint(self.inputs)}]}}
        self.add()
        self.assertEqual(self.ledger.snapshot()['entries'][0]['eligibility'], 'validation_excluded')
        self.assertEqual(self.ledger.export(True)['entries'], [])

    def test_unknown_does_not_mean_new_and_seen_not_new(self):
        self.registry.attestations = {}
        self.add()
        self.assertEqual(self.ledger.snapshot()['entries'][0]['eligibility'], 'unverified_new_candidate')
        self.assertEqual(self.ledger.export(True)['entries'], [])
        self.registry.populations = {'Segment': {'train':[{'record_id':'old','group_id':'g','fingerprint':'old'}]}}
        self.assertEqual(self.ledger.snapshot()['entries'][0]['eligibility'], 'seen_training')

    def test_unapproved_and_rejected_reviews(self):
        self.add(human_approved=False)
        self.assertEqual(self.ledger.export(True)['entries'], [])
        self.add('e2')
        self.add('e3', action='reject', human_approved=False)
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'conflict')

    def test_unverified_identity_cannot_suggest_or_export(self):
        self.add(identity_verified=False, synthetic=False)
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'identity_quarantine')
        self.assertIsNone(self.ledger.suggest('p','Segment',self.inputs))
        self.assertEqual(self.ledger.export()['entries'],[])

    def test_input_version_and_alias_conflict(self):
        self.add()
        changed = dict(self.inputs, ProductName='changed')
        self.add('e2', inputs=changed, label='b')
        self.assertEqual(len(self.ledger.export(True)['entries']), 2)
        self.add('e3', entity_id='alias', label='b')
        self.assertEqual(self.ledger.snapshot()['entries'][0]['status'], 'identity_conflict')

    def test_principal_and_required_provenance(self):
        with self.assertRaises(PermissionError): self.ledger.append('bad', self.event(), 'unknown')
        with self.assertRaises(ValueError): self.add(occurred_at='2026-11-14')
        with self.assertRaises(ValueError): self.add(inputs={'ProductName':'x'})
        with self.assertRaises(ValueError): self.add(human_approved='yes')

    def test_path_foreign_database_catalog_and_append_only(self):
        with self.assertRaises(ValueError): Ledger(self.root.parent/'foreign.sqlite', self.root, self.catalog, self.registry, {})
        foreign = self.root/'foreign.sqlite'; foreign.write_bytes(b'not a database')
        before = foreign.read_bytes()
        with self.assertRaises(sqlite3.DatabaseError): Ledger(foreign, self.root, self.catalog, self.registry, {})
        self.assertEqual(before, foreign.read_bytes())
        self.add()
        with connection(self.ledger.path) as db:
            with self.assertRaises(sqlite3.IntegrityError): db.execute('DELETE FROM events')
            with self.assertRaises(sqlite3.IntegrityError): db.execute("UPDATE events SET hash='bad'")
        altered = copy.deepcopy(self.catalog); altered['Brand']['version']='v2'
        with self.assertRaises(ValueError): Ledger(self.ledger.path, self.root, altered, self.registry, {})

    def test_chain_corruption_detected(self):
        self.add()
        with connection(self.ledger.path) as db:
            db.execute('DROP TRIGGER no_update'); db.execute("UPDATE events SET hash='bad'")
        with self.assertRaises(ValueError): self.ledger.verify()

    def test_brand_idempotence_inverse_invariance_and_rejection(self):
        base = {'label_names':['a','b'], 'training_class_counts':[10,10], 'class_prior':[.5,.5],
                'key_counts':{'x':[[0,2]], 'z':[[1,3]]}, 'missing_normalized_values':['','none']}
        before = copy.deepcopy(base); overlay=BrandOverlay(base)
        overlay.set_contribution('p', ('x','a'), original=('x','a'))
        self.assertEqual(overlay.materialize()['key_counts'], base['key_counts'])
        overlay.set_contribution('p', ('x','b'), original=('x','a'))
        first=overlay.materialize(); overlay.set_contribution('p', ('x','b'), original=('x','a'))
        self.assertEqual(first, overlay.materialize())
        np.testing.assert_array_equal(brand_probabilities(first,['z','unseen','']), brand_probabilities(base,['z','unseen','']))
        self.assertFalse(np.array_equal(brand_probabilities(overlay.materialize(True),['unseen']),brand_probabilities(base,['unseen'])))
        overlay.remove_change('p'); self.assertEqual(overlay.materialize()['key_counts'], base['key_counts'])
        with self.assertRaises(ValueError): overlay.set_contribution('bad', ('x','new'))
        with self.assertRaises(ValueError): overlay.set_contribution('bad', ('x','a'), original=('absent','b'))
        with self.assertRaises(ValueError): overlay.set_contribution('bad', ('','a'))
        self.assertEqual(base, before)

    def test_paired_evaluation_rejects_population_and_label_changes(self):
        a=[{'record_id':'1','truth':'a','prediction':'a'}, {'record_id':'2','truth':'b','prediction':'b'}]
        binding={'target':'Segment','baseline_model_sha256':'base','input_contract_sha256':'input',
                 'evaluation_population_sha256':hash_items(['1','2']),'label_names':['a','b']}
        def compare(x,y,b=binding):
            return compare_records(x,y,['a','b'],baseline_binding=binding,candidate_binding=b)
        self.assertTrue(compare(a,list(reversed(a)))['original_retention_gate'])
        with self.assertRaises(ValueError): compare(a,a[:1])
        with self.assertRaises(ValueError): compare(a,a+a[:1])
        b=copy.deepcopy(a); b[0]['truth']='b'
        with self.assertRaises(ValueError): compare(a,b)
        b=copy.deepcopy(a); b[0]['prediction']='new'
        with self.assertRaises(ValueError): compare(a,b)
        with self.assertRaises(ValueError): compare(a,a,dict(binding,baseline_model_sha256='wrong'))

    def test_group_atomic_cohorts_seen_not_unseen_and_dedup(self):
        self.add('e1',synthetic=False)
        snapshot=self.ledger.export()
        first=plan_cohorts(snapshot)
        alias=copy.deepcopy(snapshot['entries'][0]);alias['entity_id']='alias'
        snapshot['entries'].append(alias)
        self.assertEqual(first,plan_cohorts(snapshot))
        for e in snapshot['entries']:e['eligibility']='seen_training'
        self.assertEqual(plan_cohorts(snapshot)['incoming_holdout'],[])
        snapshot['entries'][0]['synthetic']=True
        with self.assertRaises(ValueError):plan_cohorts(snapshot)


if __name__ == '__main__': unittest.main(verbosity=2)
