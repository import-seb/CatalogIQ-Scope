"""Development-only maintenance research primitives; no ML imports at module load."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'data/processed/split_integration_reproduced_20261010'
SOURCE_SEAL='53f3ab943e66eefc701b3f2c598a5fcd379084a6f9f851c6f7e7e6251d9e9a81'
ALLOCATION=ROOT/'data/processed/sub_segment_development_allocation_v1_20261029_r2'
ALLOCATION_SEAL='26f52bd5360af96d0271c998f13c846dfeaa325bf0fb90b4c85011df5cabdc6a'
FIELDS=['ProductCategory','ProductName','ProductBrand','ProductDescription','ProductContents']
TARGETS=['Segment','Mnfr','TargetAgeGroup','Sub-Segment']
MISSING={'','null','none','nan'}
GIB=1024**3
LIMITS={'total_seconds':6*3600,'cpu_seconds':6*3600*4,'worker_bytes':4*GIB,
        'monitor_bytes':GIB//4,'allocator_bytes':3*GIB,'used_stop_bytes':int(4.5*GIB),
        'free_stop_bytes':int(1.5*GIB),'start_free_bytes':int(4.5*GIB),
        'stale_seconds':2.,'threads':4,'interop_threads':1}
EXPERIMENT={'seed':42,'incoming_cap':512,'replay_cap':2048,'epochs':2,'batch_size':16,
 'sgd_lr':1e-3,'head_adamw_lr':1e-4,'head_weight_decay':.01,
 'lora_lr':1e-5,'lora_head_lr':1e-4,'rank':8,'alpha':16,'lora_dropout':.05,
 'orders':['seeded_shuffle','class_blocks','reverse_class_blocks'],
 'accuracy_margin':.005,'macro_f1_margin':.01,'recall_margin':.02,'recall_support_min':30,
 'required_proxy_error_reduction':.10,'human_review_required':True,
 'genuine_new_confirmation_data':False,'full_retraining':False}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    def default(v):
        if isinstance(v,(set,frozenset)):return sorted(v,key=repr)
        if hasattr(v,'item'):return v.item()
        raise TypeError(type(v).__name__)
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,default=default)

def hash_items(items):
    digest=hashlib.sha256()
    for item in items:
        value=str(item).encode('utf8');digest.update(len(value).to_bytes(8,'big'));digest.update(value)
    return digest.hexdigest()

def snapshot(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}


def install_guard(output,permitted):
    output=Path(output).resolve();allowed={Path(p).resolve() for p in permitted}
    data=(ROOT/'data').resolve()
    forbidden={'test.csv','protected_final_test.csv','assignments.csv','rule_assignments.csv','model_inputs.csv'}
    access={str(p):0 for p in allowed}
    def inside(path,folder):return path==folder or folder in path.parents
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:
            raise PermissionError('Maintenance worker forbids networking and child processes')
        if event in {'os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.truncate','os.utime'}:
            for name in args[:2] if event in {'os.rename','os.link','os.symlink'} else args[:1]:
                if isinstance(name,(str,bytes,os.PathLike)) and not inside(Path(os.fsdecode(name)).resolve(),output):
                    raise PermissionError('Maintenance writes outside isolated output denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if path.name.casefold() in forbidden:raise PermissionError('Protected or combined data access denied')
        if writing and not inside(path,output):raise PermissionError('Original files are read-only')
        if not writing and inside(path,data) and path not in allowed and not inside(path,output):
            raise PermissionError('Data file outside explicit maintenance allowlist: '+str(path))
        if not writing and str(path) in access:access[str(path)]+=1
    sys.addaudithook(audit)
    return access

def metrics(y,pred,names):
    from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
    labels=list(range(len(names)))
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'per_class':classification_report(y,pred,labels=labels,target_names=names,zero_division=0,output_dict=True),
        'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}

def hash_parameters(named):
    digest=hashlib.sha256()
    for name,tensor in sorted(named):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()

_stages={}
def stage(output,phase,**values):
    key=str(output);number=_stages.get(key,0)+1;_stages[key]=number
    path=Path(output)/f'status_{number:06d}.json'
    write(path.with_suffix('.writing'),{'phase':phase,'monotonic':time.monotonic(),**values})
    path.with_suffix('.writing').rename(path)
    print(json.dumps({'phase':phase,**values}),flush=True)
