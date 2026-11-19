"""Offline maintenance lifecycle. No model training, inference implementation or UI.

Objects are content-addressed and published once; selections/reviews are new
objects. Operator identities and eligibility attestations belong to the calling
trusted service. This local store is not authentication, a production registry,
or an OS-level WORM store. Read verification detects external file tampering.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import re
import uuid

from core import FIELDS, hash_items
from correction_ledger import canonical, digest

TARGETS = ('Segment', 'Mnfr', 'TargetAgeGroup', 'Brand', 'Sub-Segment')
KINDS = {'dataset', 'model', 'result', 'review', 'evaluation', 'decision', 'selection'}
STORE_FORMAT = 'catalogiq-offline-maintenance-store-v1'
METHODS = {'Segment': {'softmax_sgd'}, 'Mnfr': {'softmax_sgd','passive_aggressive_ii'},
           'TargetAgeGroup': {'softmax_sgd','head_adamw'}, 'Sub-Segment': set(),
           'Brand': {'anchored_count_overlay'}}


def encoded(value):
    return (canonical(value)+'\n').encode('utf8')


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


class Store:
    def __init__(self, root):
        self.root=Path(root).resolve()
        require(os.name!='nt' or len(str(self.root))+91<260,
                'Choose a shorter Windows store path before publishing objects')
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
            with (self.root/'STORE.json').open('xb') as f:f.write(encoded({'format':STORE_FORMAT}))
        else:
            require((self.root/'STORE.json').is_file(), 'Refusing an unrelated existing directory')
            require(json.loads((self.root/'STORE.json').read_bytes())=={'format':STORE_FORMAT}, 'Foreign store')

    def location(self, object_id):
        require(isinstance(object_id,str) and re.fullmatch(r'(dataset|model|result|review|evaluation|decision|selection)-[0-9a-f]{64}',object_id), 'Invalid object ID')
        path=(self.root/object_id).resolve()
        require(path.parent==self.root,'Object escapes store')
        return path

    def read(self, object_id, kind=None):
        path=self.location(object_id)
        manifest=json.loads((path/'manifest.json').read_bytes())
        require(manifest['id']==object_id and manifest['format']==STORE_FORMAT,'Object identity mismatch')
        require(kind is None or manifest['kind']==kind,'Object kind mismatch')
        body={k:manifest[k] for k in ('format','kind','metadata','files_sha256')}
        require(manifest['id']==manifest['kind']+'-'+digest(body),'Manifest content mismatch')
        require({p.name for p in path.iterdir()}=={'manifest.json',*manifest['files_sha256']},'Unexpected object members')
        for name, expected in manifest['files_sha256'].items():
            require(re.fullmatch(r'[A-Za-z0-9_.-]+',name) and name not in ('.','..','manifest.json'),'Unsafe member name')
            member=(path/name).resolve()
            require(member.parent==path,'Member escapes store')
            require(file_sha(member)==expected,'Object payload checksum mismatch: '+name)
        return manifest

    def payload(self, object_id, name):
        manifest=self.read(object_id)
        require(name in manifest['files_sha256'],'Unknown payload')
        data=(self.location(object_id)/name).read_bytes()
        require(byte_sha(data)==manifest['files_sha256'][name],'Payload changed during read')
        return data

    def put(self, kind, metadata, files):
        require(kind in KINDS and isinstance(metadata,dict) and files,'Invalid object')
        for name,value in files.items():
            require(re.fullmatch(r'[A-Za-z0-9_.-]+',name) and name not in ('.','..','manifest.json'),'Unsafe member name')
            require(isinstance(value,bytes),'Payloads must be immutable bytes')
        body={'format':STORE_FORMAT,'kind':kind,'metadata':deepcopy(metadata),
              'files_sha256':{name:byte_sha(value) for name,value in sorted(files.items())}}
        object_id=kind+'-'+digest(body);destination=self.location(object_id)
        if destination.exists():
            require(self.read(object_id)==dict(body,id=object_id),'Conflicting existing object')
            return object_id
        staging=self.root/('.staging-'+uuid.uuid4().hex);staging.mkdir(exist_ok=False)
        try:
            import os
            for name,data in {**files,'manifest.json':encoded(dict(body,id=object_id))}.items():
                with (staging/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            try:staging.rename(destination)
            except FileExistsError:
                require(self.read(object_id)==dict(body,id=object_id),'Concurrent publication mismatch')
            return object_id
        finally:
            # Only this operation's fresh staging directory; never delete objects.
            if staging.exists():
                for p in staging.iterdir():p.unlink()
                staging.rmdir()

    @contextmanager
    def selection_lock(self):
        path=self.root/'.selection.lock'
        f=path.open('xb')
        try:yield
        finally:
            f.close();path.unlink()


def register_model(store, *, target, dataset_id, parent_id, upstream_model_id,
                   artifact_sha256, labels, config, backend, payload=None, method=None):
    require(target in TARGETS and backend in ('frozen_adapter','synthetic'),'Unsupported model route')
    store.read(dataset_id,'dataset')
    require(len(labels)>=2 and len(labels)==len(set(labels)),'Ordered unique vocabulary required')
    require(re.fullmatch('[0-9a-f]{64}',artifact_sha256),'Artifact hash required')
    if parent_id is not None:
        parent=store.read(parent_id,'model')['metadata']
        require(parent['target']==target and parent['labels']==labels and parent['backend']==backend,'Incompatible parent')
        require(method in METHODS[target],'Target update method is not enabled')
        require(config==parent['config'],'Input/config changes need a new protocol')
    metadata={'target':target,'dataset_id':dataset_id,'parent_id':parent_id,
              'upstream_model_id':upstream_model_id,'artifact_sha256':artifact_sha256,
              'labels':labels,'vocabulary_sha256':digest(labels),'config':config,
              'configuration_sha256':digest(config),'backend':backend,'method':method,
              'scope':'offline_experimental','human_approval_required':True}
    contract=({'fields':FIELDS,'max_tokens':256,'pooling':'attention-mask mean'}
              if backend=='frozen_adapter' and target!='Brand' else
              {'fields':['ProductBrand'],'lookup_config':config} if backend=='frozen_adapter' else
              {'synthetic_contract':config})
    metadata['input_contract']=contract
    metadata['configuration_sha256']=digest({'config':config,'input_contract':contract})
    files={'reference.json':encoded(metadata)}
    if payload is not None:
        require(byte_sha(payload)==artifact_sha256,'Candidate artifact hash mismatch')
        files['candidate.bin']=payload
    return store.put('model',metadata,files)


def import_frozen_versions(store, pins, sources, brand_manifest):
    """Metadata-only import; serving loader remains responsible for verified bytes."""
    versions={}
    for target in TARGETS:
        spec=pins['routes'][target]
        meta=brand_manifest if target=='Brand' else sources[target]['metadata']
        allocation=meta.get('split_identity',{})
        data={'target':target,'role':'historical_training_reference','training_records':meta['training_rows'],
              'ordered_training_ids_sha256':meta.get('train_record_ids_sha256',meta.get('training_record_ids_sha256')),
              'split_identity':allocation,'serving_manifest_sha256':spec['files_sha256'][spec['manifest']],
              'dataset_contents_copied':False}
        from core import SOURCE_SEAL, SOURCE
        data['source_directory']=str(SOURCE)
        data['source_completion_sha256']=SOURCE_SEAL
        data['allocation']=('brand-development-allocation-v1' if target=='Brand' else
                            'sub-segment-development-allocation-v1-r2' if target=='Sub-Segment' else 'segment-final-splits-v1')
        require(data['ordered_training_ids_sha256'],'Original training membership hash absent')
        dataset=store.put('dataset',data,{'lineage.json':encoded(data)})
        config=meta.get('config',{})
        require(target!='Brand' or config,'Brand requires its actual lookup configuration')
        versions[target]=register_model(store,target=target,dataset_id=dataset,parent_id=None,
                upstream_model_id=spec['model_id'],artifact_sha256=spec['artifact_sha256'],labels=meta['label_names'],
                config=config,backend='frozen_adapter',payload=None)
    return versions


def prepare_approved_data(store, ledger, roles, model_versions, *, synthetic=False):
    """Export one frozen ledger head. No data loading, allocation mutation or fitting.

    Roles are explicit steward-owned group assignments, not reviewer flags.
    Synthetic and real records cannot be mixed. Excluded events stay in ledger.
    """
    snapshot=ledger.export(allow_synthetic=synthetic)
    require(snapshot['entries'],'No eligible confirmations')
    require(set(model_versions)==set(TARGETS),'Five explicit parent versions required')
    require(set(roles)=={e['group_id'] for e in snapshot['entries']},'Group roles must cover exactly the eligible export')
    require(set(roles.values())<= {'update','replay','evaluation'},'Invalid role')
    rows={};feature_roles={}
    for e in snapshot['entries']:
        require(e['synthetic'] is synthetic,'Synthetic/real data must not mix')
        require(e['eligibility']!='seen_training' or roles[e['group_id']]!='evaluation','Seen training cannot be new evaluation')
        m=store.read(model_versions[e['target']],'model')['metadata']
        require((m['backend']=='synthetic') is synthetic,'Dataset/model synthetic boundary mismatch')
        require(e['label'] in m['labels'],'Label is outside parent vocabulary')
        k=(e['target'],e['group_id'],e['input_sha256'])
        row={**e,'role':roles[e['group_id']]}
        require(k not in rows or rows[k]['label']==row['label'],'Conflicting product contribution')
        rows.setdefault(k,row)
        fp=(e['target'],e['fingerprint'])
        if fp in feature_roles:require(feature_roles[fp]==row['role'],'Feature duplicate crosses roles')
        feature_roles[fp]=row['role']
    values=sorted(rows.values(),key=lambda r:(r['target'],r['group_id'],r['input_sha256']))
    metadata={'ledger_head':snapshot['event_head'],'registry_sha256':snapshot['registry_sha256'],
              'catalog_sha256':snapshot['catalog_sha256'],'parent_models':model_versions,
              'synthetic':synthetic,'roles':roles,'independent_new_product_claim':False,
              'counts':{role:sum(r['role']==role for r in values) for role in ('update','replay','evaluation')},
              'parent_datasets':{t:store.read(v,'model')['metadata']['dataset_id'] for t,v in model_versions.items()}}
    return store.put('dataset',metadata,{'approved_rows.json':encoded(values)})


def update_rows(store, dataset_id, target):
    store.read(dataset_id,'dataset')
    require(target in TARGETS,'Unknown target')
    return [r for r in json.loads(store.payload(dataset_id,'approved_rows.json'))
            if r['target']==target and r['role'] in ('update','replay')]


def record_evaluation(store, parent_id, candidate_id, baseline, candidate, *, cohort_id, order):
    from maintenance_offline import compare_records
    old=store.read(parent_id,'model')['metadata'];new=store.read(candidate_id,'model')['metadata']
    require(new['parent_id']==parent_id,'Evaluation parent mismatch')
    cohort=store.read(cohort_id,'dataset')
    eligible=[r for r in json.loads(store.payload(cohort_id,'approved_rows.json'))
              if r['target']==new['target'] and r['role']=='evaluation']
    require(eligible,'Evaluation cohort is empty')
    truth={r['record_id']:r['label'] for r in eligible}
    require(len(truth)==len(eligible),'Duplicate evaluation record')
    require({r['record_id']:r['truth'] for r in baseline}==truth,'Baseline differs from frozen evaluation truth')
    binding={'target':new['target'],'baseline_model_sha256':old['artifact_sha256'],
             'input_contract_sha256':old['configuration_sha256'],
             'evaluation_population_sha256':hash_items(sorted(truth)),'label_names':old['labels']}
    result=compare_records(baseline,candidate,old['labels'],baseline_binding=binding,candidate_binding=binding)
    metadata={'parent_id':parent_id,'candidate_id':candidate_id,'cohort_id':cohort_id,'order':order,
              'retention_passed':result['original_retention_gate'],'synthetic':cohort['metadata']['synthetic'],
              'production_qualification':False,'human_approval_required':True}
    return store.put('evaluation',metadata,{'metrics.json':encoded(result)})


def decide_candidate(store, candidate_id, evaluations, *, operator, reason, accept=False):
    candidate=store.read(candidate_id,'model')['metadata']
    require(candidate['parent_id'] and operator.strip() and reason.strip(),'Explicit candidate/operator/reason required')
    require(evaluations,'Evaluation evidence required')
    reports=[store.read(e,'evaluation')['metadata'] for e in evaluations]
    require(candidate_id in {e['candidate_id'] for e in reports},'Candidate itself must be evaluated')
    for report in reports:
        trial=store.read(report['candidate_id'],'model')['metadata']
        require(all(trial[k]==candidate[k] for k in ('parent_id','dataset_id','target','config','method','backend')),
                'Order trial uses a different experiment')
    require(len({e['cohort_id'] for e in reports})==1,'Order comparisons must use identical cohort')
    required={'seeded_shuffle','class_blocks','reverse_class_blocks'}
    require(not accept or {e['order'] for e in reports}==required,'All predeclared orders required')
    require(not accept or all(e['retention_passed'] for e in reports),'Unsafe candidate cannot be accepted')
    metadata={'candidate_id':candidate_id,'evaluations':evaluations,'operator':operator,'reason':reason,
              'decision':'accept_experimental' if accept else 'reject','production_qualification':False}
    return store.put('decision',metadata,{'decision.json':encoded(metadata)})


def latest_selection(store):
    selections=[store.read(p.name,'selection') for p in store.root.glob('selection-*')]
    if not selections:return None
    by_parent={}
    for s in selections:
        parent=s['metadata']['previous_selection']
        require(parent not in by_parent,'Divergent selection journal')
        by_parent[parent]=s['id']
    current=by_parent.get(None);visited=set()
    while current is not None:
        require(current not in visited,'Selection cycle');visited.add(current)
        if current not in by_parent:break
        current=by_parent[current]
    require(len(visited)==len(selections),'Disconnected selection journal')
    return current


def select_models(store, versions, *, operator, reason, decisions=None, rollback_to=None):
    """Manual offline selection journal. Does not alter the application's loader."""
    require(set(versions)==set(TARGETS) and operator.strip() and reason.strip(),'Five routes and explicit operator required')
    decisions=decisions or {}
    for target, version in versions.items():
        m=store.read(version,'model')['metadata'];require(m['target']==target,'Wrong target version')
        if m['parent_id'] and rollback_to is None:
            receipt=store.read(decisions[target],'decision')['metadata']
            require(receipt['candidate_id']==version and receipt['decision']=='accept_experimental','Candidate lacks acceptance')
    if rollback_to is not None:
        require(store.read(rollback_to,'selection')['metadata']['models']==versions,'Rollback version mismatch')
    with store.selection_lock():
        previous=latest_selection(store)
        metadata={'models':versions,'previous_selection':previous,'rollback_to':rollback_to,
                  'operator':operator,'reason':reason,'decisions':decisions,'scope':'offline_only',
                  'production_publication':False}
        return store.put('selection',metadata,{'selection.json':encoded(metadata)})


def classify_batch(store, request, infer, *, selection_id=None, versions=None):
    """Delegate to Max/Sebastian's infer_request; freeze version references once.

    infer(request, model_descriptors) is an injected binding to the existing
    adapter. No model/head/lookup inference is reimplemented here.
    """
    require(not (selection_id and versions),'Choose a selection or explicit versions')
    if versions is None:
        selection_id=selection_id or latest_selection(store)
        require(selection_id,'No current offline selection')
        versions=store.read(selection_id,'selection')['metadata']['models']
    require(set(versions)==set(TARGETS),'Five model references required')
    require(isinstance(request,dict) and set(request)<= {'schema_version','products'},'Unsupported envelope')
    require(request.get('schema_version')=='catalogiq-inference-request-v1','Unsupported request version')
    rows=request.get('products');require(isinstance(rows,list) and 1<=len(rows)<=250,'Batch size 1..250 required')
    ids=[]
    for r in rows:
        require(set(r)<={'record_id','quarantined',*FIELDS} and set(FIELDS)<=set(r),'Unsupported/missing product fields')
        require(isinstance(r.get('record_id'),str) and r['record_id'].strip(),'String record ID required')
        ids.append(r['record_id'])
        require(all(r[f] is None or isinstance(r[f],str) for f in FIELDS),'Invalid feature type')
        require(type(r.get('quarantined',False)) is bool,'Invalid quarantine flag')
    require(len(ids)==len(set(ids)) and len(encoded(request))<=2*1024**2,'Duplicate IDs or oversized request')
    models={t:store.read(v,'model')['metadata'] for t,v in versions.items()}
    require(all(m['target']==t for t,m in models.items()),'Model target mismatch')
    frozen_request=deepcopy(request)
    response=infer(deepcopy(frozen_request),deepcopy(models))
    require(response.get('schema_version')=='catalogiq-inference-request-v1','Wrong adapter schema')
    require([r['record_id'] for r in response['products']]==ids,'Adapter changed record alignment')
    for row in response['products']:
        require(row['review_required'] is True,'Human review cannot be disabled')
        require(set(row['predictions'])=={*TARGETS,'Platform'},'Wrong target population')
        require(all(p['review_required'] is True for p in row['predictions'].values()),'Every target requires review')
        for target,m in models.items():
            p=row['predictions'][target]
            require(p['review_required'] is True and p['status'] in ('suggestion','failed','unavailable'),'Invalid review/status')
            if p['status']=='suggestion':
                require(p['provenance']['model_id']==m['upstream_model_id'] and p['provenance']['artifact_sha256']==m['artifact_sha256'],'Adapter served a different model')
                require(p['label_mapping']=={f'probability_{i}':n for i,n in enumerate(m['labels'])},'Adapter vocabulary mismatch')
                values=[p['probabilities'][f'probability_{i}'] for i in range(len(m['labels']))]
                require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in values)
                        and abs(sum(values)-1)<1e-6,'Invalid adapter probabilities')
                winner=max(range(len(values)),key=values.__getitem__)
                require(p['predicted_label']==m['labels'][winner] and abs(p['confidence']-values[winner])<1e-6,'Adapter decision mismatch')
            else:
                require(all(p.get(k) is None for k in ('predicted_label','confidence','probabilities')),
                        'Failed/unavailable route contains stale suggestions')
    metadata={'model_versions':versions,'dataset_versions':{t:m['dataset_id'] for t,m in models.items()},
              'selection_id':selection_id,'request_sha256':byte_sha(encoded(frozen_request)),
              'human_approval_required':True,'synthetic':any(m['backend']=='synthetic' for m in models.values())}
    return store.put('result',metadata,{'request.json':encoded(frozen_request),'predictions.json':encoded(response)})


def fixed_adapter(engine, infer_request):
    """Existing frozen loader supports only its pins; response bindings fail closed."""
    def infer(request, descriptors):
        for target,expected in descriptors.items():
            actual=engine.loader.describe(target)
            require(actual['model_id']==expected['upstream_model_id'] and actual['artifact_sha256']==expected['artifact_sha256'],
                    'Engine does not serve selected version; coordinate a loader with its owner')
        return infer_request(engine,request)
    return infer


def save_review(store, result_id, record_id, labels, *, reviewer, reason, previous_review=None):
    """Persist approval separately; model results and prior decisions are immutable."""
    store.read(result_id,'result')
    require(reviewer.strip() and reason.strip(),'Reviewer and rationale required')
    rows=json.loads(store.payload(result_id,'predictions.json'))['products']
    matches=[r for r in rows if r['record_id']==record_id];require(len(matches)==1,'Unknown review record')
    predictions=matches[0]['predictions']
    require(set(labels)==set(TARGETS),'Every supported target needs human review')
    for target,label in labels.items():
        p=predictions[target]
        require(p['status']=='suggestion' and label in p['label_mapping'].values(),'Cannot approve unavailable/OOV output')
    if previous_review:
        prev=store.read(previous_review,'review')['metadata']
        require(prev['result_id']==result_id and prev['record_id']==record_id,'Review supersession mismatch')
    metadata={'result_id':result_id,'record_id':record_id,'labels':labels,'reviewer':reviewer,
              'reason':reason,'previous_review':previous_review,'human_approved':True,
              'occurred_at':datetime.now(timezone.utc).isoformat()}
    return store.put('review',metadata,{'approved.json':encoded(metadata)})
