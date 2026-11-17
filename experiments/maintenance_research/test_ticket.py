"""Acceptance and failure-path tests; no finished weights or source datasets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core import FIELDS, snapshot
from ticket_demo import setup_demo, features, SyntheticLoader
from ticket_lifecycle import (Store, TARGETS, encoded, byte_sha, register_model, prepare_approved_data,
    update_rows, record_evaluation, decide_candidate, select_models, latest_selection,
    classify_batch, fixed_adapter, save_review, import_frozen_versions)


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store,self.versions,self.ledger,self.dataset=setup_demo(self.root)

    def tearDown(self):self.temp.cleanup()

    def request(self):
        return {'schema_version':'catalogiq-inference-request-v1','products':[{'record_id':'0001',**features(100)}]}

    def infer(self,versions=None):
        from catalogiq.inference_engine import InferenceEngine
        from catalogiq.inference_adapter import infer_request
        return fixed_adapter(InferenceEngine('unused',loader=SyntheticLoader(self.store,versions or self.versions)),infer_request)

    def child(self,target='Segment'):
        parent=self.store.read(self.versions[target],'model')['metadata']
        payload=encoded({'weight':[[-1.,0.],[1.,0.]],'bias':[0.,0.]})
        return register_model(self.store,target=target,dataset_id=self.dataset,parent_id=self.versions[target],
          upstream_model_id='synthetic:bad',artifact_sha256=byte_sha(payload),labels=['A','B'],
          config=parent['config'],backend='synthetic',payload=payload,method='softmax_sgd')

    def test_content_versions_idempotence_and_changed_payload(self):
        a=self.store.put('dataset',{'synthetic':True},{'rows.json':b'A'})
        self.assertEqual(a,self.store.put('dataset',{'synthetic':True},{'rows.json':b'A'}))
        b=self.store.put('dataset',{'synthetic':True},{'rows.json':b'B'})
        self.assertNotEqual(a,b);self.assertEqual(self.store.payload(a,'rows.json'),b'A')

    def test_frozen_metadata_index_binds_all_five_training_populations(self):
        pins={'routes':{}};sources={}
        for target in TARGETS:
            pins['routes'][target]={'model_id':'synthetic-reference:'+target,
                'artifact_sha256':'1'*64,'manifest':target+'/manifest.json',
                'files_sha256':{target+'/manifest.json':'2'*64}}
            sources[target]={'metadata':{'training_rows':2,'train_record_ids_sha256':'3'*64,
                'config':{'synthetic_reference':True},'label_names':['A','B']}}
        brand={**sources['Brand']['metadata'],'training_record_ids_sha256':'3'*64}
        versions=import_frozen_versions(self.store,pins,sources,brand)
        self.assertEqual(set(versions),set(TARGETS))
        for target,version in versions.items():
            model=self.store.read(version,'model')['metadata']
            data=self.store.read(model['dataset_id'],'dataset')['metadata']
            self.assertEqual(model['artifact_sha256'],'1'*64)
            self.assertEqual(data['ordered_training_ids_sha256'],'3'*64)
            self.assertEqual(data['target'],target)
            self.assertEqual(model['vocabulary_sha256'],__import__('correction_ledger').digest(['A','B']))

    def test_missing_optional_memory_sampler_does_not_invent_measurements(self):
        from methods import ProcessSample
        with patch.dict('sys.modules',{'psutil':None}):
            measurement=ProcessSample().finish()
        self.assertFalse(measurement['memory_sampling_available'])
        self.assertGreaterEqual(measurement['update_cpu_seconds'],0)

    def test_external_tampering_is_detected_not_overwritten(self):
        a=self.store.put('dataset',{}, {'rows.json':b'A'})
        (self.store.location(a)/'rows.json').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.store.read(a)
        with self.assertRaises(ValueError):self.store.put('dataset',{}, {'rows.json':b'A'})
        self.assertEqual((self.store.location(a)/'rows.json').read_bytes(),b'corrupt')

    def test_paths_foreign_store_and_manifest_binding(self):
        for value in ('../test.csv','model-../x','dataset-'+'x'*64):
            with self.assertRaises(ValueError):self.store.read(value)
        with self.assertRaises(ValueError):self.store.put('dataset',{}, {'../x':b'x'})
        unrelated=self.root/'unrelated';unrelated.mkdir()
        with self.assertRaises(ValueError):Store(unrelated)
        a=self.store.put('dataset',{}, {'rows.json':b'A'})
        manifest=self.store.location(a)/'manifest.json';value=json.loads(manifest.read_bytes());value['metadata']['forged']=True
        manifest.write_bytes(encoded(value))
        with self.assertRaises(ValueError):self.store.read(a)

    def test_dataset_lineage_dedup_and_holdout_exclusion_from_update(self):
        meta=self.store.read(self.dataset,'dataset')['metadata']
        self.assertEqual(meta['ledger_head'],self.ledger.verify())
        self.assertEqual(meta['parent_models'],self.versions)
        rows=update_rows(self.store,self.dataset,'Segment')
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['role']!='evaluation' for r in rows))
        self.assertEqual(len({(r['group_id'],r['input_sha256']) for r in rows}),12)

    def test_real_export_excludes_synthetic_and_roles_are_exhaustive(self):
        with self.assertRaises(ValueError):prepare_approved_data(self.store,self.ledger,{},self.versions)
        with self.assertRaises(ValueError):prepare_approved_data(self.store,self.ledger,{},self.versions,synthetic=True)

    def test_new_ledger_snapshot_creates_new_dataset_preserves_old(self):
        before=snapshot(self.store.location(self.dataset))
        request=json.loads(self.ledger.events()[0][2]);request.pop('principal')
        self.ledger.append('duplicate-confirmation',request,'synthetic-reviewer')
        roles=self.store.read(self.dataset,'dataset')['metadata']['roles']
        new=prepare_approved_data(self.store,self.ledger,roles,self.versions,synthetic=True)
        self.assertNotEqual(new,self.dataset);self.assertEqual(before,snapshot(self.store.location(self.dataset)))
        self.assertEqual(len(update_rows(self.store,new,'Segment')),12)

    def test_parent_vocab_method_and_payload_integrity(self):
        original=self.store.read(self.versions['Segment'],'model')['metadata']
        args=dict(target='Segment',dataset_id=self.dataset,parent_id=self.versions['Segment'],
             upstream_model_id='bad',artifact_sha256='0'*64,labels=['A','B'],
             config=original['config'],backend='synthetic',method='softmax_sgd')
        with self.assertRaises(ValueError):register_model(self.store,**args,payload=b'bad')
        with self.assertRaises(ValueError):register_model(self.store,**dict(args,labels=['B','A']))
        with self.assertRaises(ValueError):register_model(self.store,**dict(args,method='lora'))
        with self.assertRaises(ValueError):self.child('Sub-Segment')

    def test_rejection_and_failed_retention_cannot_be_selected(self):
        child=self.child()
        rows=[r for r in json.loads(self.store.payload(self.dataset,'approved_rows.json')) if r['target']=='Segment' and r['role']=='evaluation']
        before=[{'record_id':r['record_id'],'truth':r['label'],'prediction':r['label']} for r in rows]
        after=[dict(r,prediction='B' if r['truth']=='A' else 'A') for r in before]
        evaluations=[record_evaluation(self.store,self.versions['Segment'],child,before,after,cohort_id=self.dataset,order=o)
                     for o in ('seeded_shuffle','class_blocks','reverse_class_blocks')]
        with self.assertRaises(ValueError):decide_candidate(self.store,child,evaluations,operator='operator',reason='bad',accept=True)
        rejected=decide_candidate(self.store,child,evaluations,operator='operator',reason='retention failed')
        versions=dict(self.versions,Segment=child)
        with self.assertRaises(ValueError):select_models(self.store,versions,operator='operator',reason='blocked',decisions={'Segment':rejected})
        self.assertIsNone(latest_selection(self.store))

    def test_evaluation_truth_and_population_must_match_frozen_roles(self):
        child=self.child()
        wrong=[{'record_id':'not-held-out','truth':'A','prediction':'A'}]
        with self.assertRaises(ValueError):record_evaluation(self.store,self.versions['Segment'],child,wrong,wrong,cohort_id=self.dataset,order='seeded_shuffle')

    def test_selected_version_bound_provenance_and_exact_ids(self):
        selected=select_models(self.store,self.versions,operator='operator',reason='synthetic baseline')
        result=classify_batch(self.store,self.request(),self.infer())
        meta=self.store.read(result,'result')['metadata']
        self.assertEqual(meta['selection_id'],selected);self.assertEqual(meta['model_versions'],self.versions)
        data=json.loads(self.store.payload(result,'predictions.json'))
        self.assertEqual(data['products'][0]['record_id'],'0001')
        self.assertTrue(data['products'][0]['review_required'])
        for target in TARGETS:
            p=data['products'][0]['predictions'][target]
            self.assertEqual(p['provenance']['model_id'],'synthetic:'+target+':A')
            self.assertTrue(p['review_required'])

    def test_adapter_cannot_silently_serve_the_wrong_version(self):
        child=self.child();versions=dict(self.versions,Segment=child)
        with self.assertRaises(ValueError):classify_batch(self.store,self.request(),self.infer(),versions=versions)
        self.assertEqual(list(self.store.root.glob('result-*')),[])

    def test_bad_requests_and_automatic_acceptance_response_refused(self):
        for request in (dict(self.request(),targets=['Segment']),dict(self.request(),schema_version='wrong'),
                        dict(self.request(),products=self.request()['products']*2)):
            with self.assertRaises(ValueError):classify_batch(self.store,request,self.infer(),versions=self.versions)
        good=self.infer()
        def unsafe(request,versions):
            response=good(request,versions);response['products'][0]['review_required']=False;return response
        with self.assertRaises(ValueError):classify_batch(self.store,self.request(),unsafe,versions=self.versions)
        def stale(request,versions):
            response=good(request,versions);response['products'][0]['predictions']['Segment']['status']='failed';return response
        with self.assertRaises(ValueError):classify_batch(self.store,self.request(),stale,versions=self.versions)

    def test_same_record_id_in_new_version_does_not_replace_prior_results(self):
        old=classify_batch(self.store,self.request(),self.infer(),versions=self.versions)
        before=snapshot(self.store.location(old))
        versions=dict(self.versions,Segment=self.child())
        new=classify_batch(self.store,self.request(),self.infer(versions),versions=versions)
        self.assertNotEqual(old,new);self.assertEqual(before,snapshot(self.store.location(old)))
        self.assertEqual(self.store.read(old,'result')['metadata']['model_versions'],self.versions)
        self.assertEqual(self.store.read(new,'result')['metadata']['model_versions'],versions)

    def test_approved_history_and_original_predictions_never_overwritten(self):
        result=classify_batch(self.store,self.request(),self.infer(),versions=self.versions)
        original=self.store.payload(result,'predictions.json');labels={t:'A' for t in TARGETS}
        first=save_review(self.store,result,'0001',labels,reviewer='reviewer',reason='synthetic')
        before=snapshot(self.store.location(first))
        second=save_review(self.store,result,'0001',dict(labels,Segment='B'),reviewer='reviewer',reason='synthetic correction',previous_review=first)
        self.assertNotEqual(first,second);self.assertEqual(before,snapshot(self.store.location(first)))
        self.assertEqual(original,self.store.payload(result,'predictions.json'))
        self.assertEqual(self.store.read(second,'review')['metadata']['previous_review'],first)

    def test_selection_lock_released_after_failure(self):
        with self.assertRaises(RuntimeError):
            with self.store.selection_lock():raise RuntimeError('synthetic failure')
        with self.store.selection_lock():pass


if __name__=='__main__':unittest.main(verbosity=2)
