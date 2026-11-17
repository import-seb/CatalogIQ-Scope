"""Synthetic engineering demonstration, never a model-quality experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core import FIELDS, EXPERIMENT, snapshot
from correction_ledger import Ledger, EligibilityIndex
from methods import update_head
from ticket_lifecycle import (Store, TARGETS, encoded, byte_sha, register_model,
    prepare_approved_data, update_rows, record_evaluation, decide_candidate,
    select_models, classify_batch, fixed_adapter, save_review)


def features(i):
    sign=1 if i%2==0 else -1
    return dict(zip(FIELDS,('SYNTHETIC category',f'SYNTHETIC product {i}',
                           'synthetic-a' if sign==1 else 'synthetic-b',str(sign),str(sign*.5))))


class SyntheticLoader:
    """Test dependency injected into the unmodified upstream engine, not serving code."""
    def __init__(self, store, versions):
        self.store=store;self.versions=versions

    def describe(self,target):
        m=self.store.read(self.versions[target],'model')['metadata']
        return {'model_id':m['upstream_model_id'],'artifact_sha256':m['artifact_sha256'],
                'synthetic':True,'policy_approved':False,'operational_threshold':None}

    def load(self,target):
        m=self.store.read(self.versions[target],'model')['metadata']
        state=json.loads(self.store.payload(self.versions[target],'candidate.bin'))
        def predict(frame):
            if target=='Brand':
                from maintenance_offline import brand_probabilities
                return brand_probabilities(state,list(frame.ProductBrand)),['seen']*len(frame)
            x=np.array([[float(r.ProductDescription),float(r.ProductContents)] for r in frame.itertuples()],dtype=np.float32)
            logits=x@np.asarray(state['weight'],dtype=np.float32).T+np.asarray(state['bias'],dtype=np.float32)
            logits-=logits.max(1,keepdims=True);probs=np.exp(logits);probs/=probs.sum(1,keepdims=True)
            return probs,[None]*len(frame)
        return SimpleNamespace(labels=tuple(m['labels']),temperature=1.,provenance=self.describe(target),predict=predict)


def setup_demo(root):
    root=Path(root);store=Store(root/'store');versions={};labels=['A','B']
    config={'synthetic_feature_dimension':2,'not_a_minilm_model':True}
    for target in TARGETS:
        dataset=store.put('dataset',{'target':target,'synthetic':True,'role':'synthetic_parent'},
                          {'lineage.json':encoded({'synthetic':True,'training_records':0})})
        state=({'weight':[[.2,.1],[-.1,-.2]],'bias':[.01,-.01]} if target!='Brand' else
               {'label_names':labels,'training_class_counts':[10,10],'class_prior':[.5,.5],
                'missing_normalized_values':[''],'key_counts':{'synthetic-a':[[0,10]],'synthetic-b':[[1,10]]}})
        payload=encoded(state)
        versions[target]=register_model(store,target=target,dataset_id=dataset,parent_id=None,
              upstream_model_id='synthetic:'+target+':A',artifact_sha256=byte_sha(payload),
              labels=labels,config=config,backend='synthetic',payload=payload)
    catalog={t:{'version':'synthetic-ontology-v1','labels':labels} for t in TARGETS}
    attestations={'g'+str(i):{'reserved_exclusion_verified':True,'training_exclusion_verified':True,
                             'identity_verified':True,'provenance':'SYNTHETIC engineering fixture',
                             'entity_ids':['p'+str(i)]} for i in range(16)}
    registry=EligibilityIndex('synthetic-registry-v1',{},attestations)
    ledger=Ledger(root/'confirmations.sqlite',root,catalog,registry,{'synthetic-reviewer':'reviewer'})
    for i in range(16):
        for target in TARGETS:
            ledger.append(f'e-{target}-{i}',{'kind':'review','target':target,'entity_id':'p'+str(i),
                'record_id':f'{i:04d}','group_id':'g'+str(i),'inputs':features(i),
                'model_version':versions[target],'ontology_version':'synthetic-ontology-v1',
                'source_system':'synthetic-engineering-test','occurred_at':'2026-11-14T15:00:00+00:00',
                'action':'confirm','label':labels[i%2],'human_approved':True,'synthetic':True,
                'identity_verified':True,'grouping_verified':True},'synthetic-reviewer')
    roles={'g'+str(i):'update' if i<12 else 'evaluation' for i in range(16)}
    prepared=prepare_approved_data(store,ledger,roles,versions,synthetic=True)
    return store,versions,ledger,prepared


def demonstrate(root):
    from catalogiq.inference_engine import InferenceEngine
    from catalogiq.inference_adapter import infer_request
    store,versions,ledger,prepared=setup_demo(root)
    selection_a=select_models(store,versions,operator='synthetic-operator',reason='Synthetic baseline only')
    engine_a=InferenceEngine('unused',loader=SyntheticLoader(store,versions))
    request_a={'schema_version':'catalogiq-inference-request-v1','products':[{'record_id':'000A',**features(100)}]}
    result_a=classify_batch(store,request_a,fixed_adapter(engine_a,infer_request))
    original=json.loads(store.payload(result_a,'predictions.json'))['products'][0]
    labels={t:original['predictions'][t]['predicted_label'] for t in TARGETS}
    review_a=save_review(store,result_a,'000A',labels,reviewer='synthetic-reviewer',reason='Synthetic approval')
    before=snapshot(store.root)
    rows=update_rows(store,prepared,'Segment')
    x=np.array([[float(r['inputs']['ProductDescription']),float(r['inputs']['ProductContents'])] for r in rows],dtype=np.float32)
    y=np.array([0 if r['label']=='A' else 1 for r in rows],dtype=np.int64)
    base=json.loads(store.payload(versions['Segment'],'candidate.bin'))
    evaluations=[];trials=[];costs=[]
    eval_rows=[r for r in json.loads(store.payload(prepared,'approved_rows.json')) if r['target']=='Segment' and r['role']=='evaluation']
    baseline=[{'record_id':r['record_id'],'truth':r['label'],'prediction':r['label']} for r in eval_rows]
    for order in EXPERIMENT['orders']:
        w,b,cost=update_head('softmax_sgd',x,y,np.array(base['weight'],dtype=np.float32),
                            np.array(base['bias'],dtype=np.float32),order,Path(root)/('synthetic_fit_'+order))
        payload=encoded({'weight':w.tolist(),'bias':b.tolist()})
        assert payload!=store.payload(versions['Segment'],'candidate.bin')
        child=register_model(store,target='Segment',dataset_id=prepared,parent_id=versions['Segment'],
             upstream_model_id='synthetic:Segment:B:'+order,artifact_sha256=byte_sha(payload),labels=['A','B'],
             config=store.read(versions['Segment'],'model')['metadata']['config'],backend='synthetic',
             payload=payload,method='softmax_sgd')
        xv=np.array([[float(r['inputs']['ProductDescription']),float(r['inputs']['ProductContents'])] for r in eval_rows],dtype=np.float32)
        oldpred=(xv@np.array(base['weight'],dtype=np.float32).T+np.array(base['bias'],dtype=np.float32)).argmax(1)
        pred=(xv@w.T+b).argmax(1)
        a=[dict(r,prediction=['A','B'][int(p)]) for r,p in zip(baseline,oldpred)]
        c=[dict(r,prediction=['A','B'][int(p)]) for r,p in zip(baseline,pred)]
        evaluations.append(record_evaluation(store,versions['Segment'],child,a,c,cohort_id=prepared,order=order))
        trials.append(child);costs.append(cost)
    receipt=decide_candidate(store,trials[0],evaluations,operator='synthetic-operator',reason='Engineering gates only',accept=True)
    versions_b=dict(versions,Segment=trials[0])
    selection_b=select_models(store,versions_b,operator='synthetic-operator',reason='Synthetic version B',decisions={'Segment':receipt})
    engine_b=InferenceEngine('unused',loader=SyntheticLoader(store,versions_b))
    result_b=classify_batch(store,{'schema_version':'catalogiq-inference-request-v1',
                         'products':[{'record_id':'000B',**features(101)}]},fixed_adapter(engine_b,infer_request))
    # Old bundles, predictions AND approved decisions must remain exactly unchanged.
    assert all((store.root/p).read_bytes() and snapshot(store.root)[p]==hashed for p,hashed in before.items())
    assert store.read(result_a,'result')['metadata']['model_versions']==versions
    assert store.read(result_b,'result')['metadata']['model_versions']==versions_b
    assert store.read(review_a,'review')['metadata']['result_id']==result_a
    rolled=select_models(store,versions,operator='synthetic-operator',reason='Demonstrate rollback',rollback_to=selection_a)
    replay=classify_batch(store,request_a,fixed_adapter(engine_a,infer_request))
    assert store.payload(replay,'predictions.json')==store.payload(result_a,'predictions.json')
    summary={'synthetic_engineering_only':True,'generalization_claim':False,'finished_models_loaded':False,
        'criterion_1':{'five_target_versions':versions,'dataset_version':prepared},
        'criterion_2':{'ledger_head':ledger.verify(),'fitted_orders':EXPERIMENT['orders'],'candidate_trials':trials,
                       'evaluations':evaluations,'experimental_decision':receipt,'costs':costs},
        'criterion_3':{'actual_upstream_adapter_used':True,'version_a_result':result_a,'version_b_result':result_b},
        'criterion_4':{'preserved_prior_files':len(before),'byte_for_byte_unchanged':True,
                       'approved_result':review_a,'a_selection':selection_a,'b_selection':selection_b,
                       'rollback_selection':rolled,'rollback_predictions_exact':True}}
    return summary
