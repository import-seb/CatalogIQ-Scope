"""Cheap warm-start updates to the EXISTING head, never a replacement encoder/head."""
import copy
import json
from pathlib import Path
import sqlite3
import time
import threading
import numpy as np

from core import EXPERIMENT,metrics,write


class ProcessSample:
    """Sample native process RSS; report observed increments, not an allocation guarantee."""
    def __init__(self):
        self.cpu=time.process_time();self.process=None
        try:import psutil
        except ImportError:return  # Optional telemetry; OS job limits remain independent.
        self.process=psutil.Process();self.initial=self.process.memory_info().rss
        self.peak=self.initial;self.cpu=time.process_time();self.done=threading.Event()
        def sample():
            while not self.done.wait(.005):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()

    def finish(self):
        if self.process is None:
            return {'update_cpu_seconds':time.process_time()-self.cpu,'updater_device':'cpu',
                    'memory_sampling_available':False}
        cpu=time.process_time()-self.cpu;self.done.set();self.thread.join(timeout=1)
        final=self.process.memory_info().rss;self.peak=max(self.peak,final)
        return {'update_cpu_seconds':cpu,'sampled_worker_rss_before_bytes':self.initial,
                'sampled_worker_rss_after_bytes':final,'sampled_worker_peak_rss_bytes':self.peak,
                'sampled_incremental_rss_bytes':max(0,self.peak-self.initial),
                'memory_sampling_interval_seconds':.005,'updater_device':'cpu'}


def ordered_indices(y,order,epoch,seed=42):
    if order=='seeded_shuffle':return np.random.default_rng(seed+epoch).permutation(len(y))
    values=np.argsort(y,kind='stable')
    return values if order=='class_blocks' else values[::-1].copy()


def predict(x,w,b):return (np.asarray(x)@w.T+b).argmax(1)


def forgetting(y,before,after,names):
    before_m=metrics(y,before,names);after_m=metrics(y,after,names)
    drops={name:before_m['per_class'][name]['recall']-after_m['per_class'][name]['recall'] for name in names}
    watched={name:value for name,value in drops.items() if before_m['per_class'][name]['support']>=EXPERIMENT['recall_support_min']}
    return {'previously_correct_became_wrong':int(((before==y)&(after!=y)).sum()),
        'previously_wrong_became_correct':int(((before!=y)&(after==y)).sum()),
        'correct_to_wrong_fraction_of_population':float(((before==y)&(after!=y)).mean()),
        'per_class_recall_drop':drops,'worst_supported_class_recall_drop':max([0.,*watched.values()]),
        'accuracy_change':after_m['accuracy']-before_m['accuracy'],
        'macro_f1_change':after_m['macro_f1']-before_m['macro_f1']}


def assess(y,before,after,yi,pi_before,pi_after,names):
    m=metrics(y,after,names);f=forgetting(y,before,after,names)
    initial_errors=int((pi_before!=yi).sum());final_errors=int((pi_after!=yi).sum())
    gain=(initial_errors-final_errors)/initial_errors if initial_errors else 0.
    retains=f['accuracy_change']>=-EXPERIMENT['accuracy_margin'] and f['macro_f1_change']>=-EXPERIMENT['macro_f1_margin'] and f['worst_supported_class_recall_drop']<=EXPERIMENT['recall_margin']
    return {'validation':m,'forgetting':f,'seen_training_proxy_update_examples':metrics(yi,pi_after,names),
        'proxy_initial_errors':initial_errors,'proxy_final_errors':final_errors,'proxy_net_error_reduction_fraction':gain,
        'retention_development_gate':retains,'proxy_mechanics_gate':gain>=EXPERIMENT['required_proxy_error_reduction'] if initial_errors else True,
        'genuinely_new_example_performance':None}


def update_head(method,x,y,w,b,order,destination):
    """Fixed settings, original weights; no estimator fit/reset or hyperparameter search."""
    import torch
    from safetensors.torch import save_file,load_file
    destination.mkdir(exist_ok=False)
    w=w.copy();b=b.copy();history=[];sample=ProcessSample();started=time.perf_counter()
    if method=='head_adamw':
        head=torch.nn.Linear(w.shape[1],w.shape[0])
        with torch.no_grad():head.weight.copy_(torch.from_numpy(w));head.bias.copy_(torch.from_numpy(b))
        torch.random.default_generator.manual_seed(42)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01,foreach=False,fused=False)
    step=0
    for epoch in range(2):
        indexes=ordered_indices(y,order,epoch)
        losses=[]
        for offset in range(0,len(y),16):
            ids=indexes[offset:offset+16];xb=np.asarray(x[ids],dtype=np.float32);yb=y[ids]
            if method=='head_adamw':
                optimizer.zero_grad(set_to_none=True)
                inputs=torch.nn.functional.dropout(torch.from_numpy(xb),p=.1,training=True)
                logits=head(inputs);loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(yb))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite head maintenance loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
            elif method=='softmax_sgd':
                scores=xb@w.T+b;scores-=scores.max(1,keepdims=True)
                probs=np.exp(scores);probs/=probs.sum(1,keepdims=True)
                losses.append(float(-np.log(np.maximum(probs[np.arange(len(ids)),yb],1e-30)).mean()))
                probs[np.arange(len(ids)),yb]-=1;probs/=len(ids)
                w-=np.float32(1e-3)*(probs.T@xb+np.float32(.01)*w)
                b-=np.float32(1e-3)*probs.sum(0)
            elif method=='passive_aggressive_ii':
                for xx,yy in zip(xb,yb):
                    scores=w@xx+b;scores[yy]=-np.inf;competitor=int(scores.argmax())
                    hinge=max(0.,1.+float(scores[competitor])-float(w[yy]@xx+b[yy]))
                    tau=hinge/(2.*(float(xx@xx)+1.)+1./(2.*.01))
                    w[yy]+=tau*xx;w[competitor]-=tau*xx;b[yy]+=tau;b[competitor]-=tau
                    losses.append(hinge)
            else:raise ValueError(method)
            step+=1
        history.append({'epoch':epoch+1,'mean_training_objective':float(np.mean(losses))})
    if method=='head_adamw':
        w=head.weight.detach().numpy().copy();b=head.bias.detach().numpy().copy()
    if not np.isfinite(w).all() or not np.isfinite(b).all():raise RuntimeError('Nonfinite updated head')
    elapsed=time.perf_counter()-started
    process_cost=sample.finish()
    began=time.perf_counter()
    save_file({'weight':torch.from_numpy(w),'bias':torch.from_numpy(b)},str(destination/'head.safetensors'))
    restored=load_file(destination/'head.safetensors',device='cpu')
    if not np.array_equal(w,restored['weight'].numpy()) or not np.array_equal(b,restored['bias'].numpy()):
        raise RuntimeError('Head delta failed exact save/reload')
    info={'method':method,'order':order,'update_seconds':elapsed,'save_reload_seconds':time.perf_counter()-began,
        'minibatches':step,'seen_training_proxy_records':len(y),'epochs':2,'history':history,
        'initialized_from_existing_fine_tuned_head':True,'encoder_updates':0,
        'trainable_parameters':int(w.size+b.size),'head_roundtrip_exact':True,
        'additional_storage_bytes':(destination/'head.safetensors').stat().st_size}
    info.update(process_cost)
    if method=='head_adamw':
        torch.save(optimizer.state_dict(),destination/'optimizer.pt')
        loaded=torch.load(destination/'optimizer.pt',map_location='cpu',weights_only=True)
        if set(loaded['state'])!=set(optimizer.state_dict()['state']):raise RuntimeError('Optimizer restore membership mismatch')
        # Compact optimizer state is retained only for this reproducible head delta.
        info['additional_storage_bytes']+=(destination/'optimizer.pt').stat().st_size
    write(destination/'update.json',info)
    return w,b,info


def correction_memory(incoming_rows,val_rows,baseline_pred,names,destination):
    destination.mkdir(exist_ok=False);sample=ProcessSample();start=time.perf_counter()
    grouped={}
    for row in incoming_rows:grouped.setdefault(row['fingerprint'],set()).add(row['y'])
    with sqlite3.connect(destination/'correction_memory.sqlite') as db:
        db.execute('CREATE TABLE corrections (fingerprint TEXT PRIMARY KEY, label_id INTEGER, provenance TEXT NOT NULL, status TEXT NOT NULL)')
        db.executemany('INSERT INTO corrections VALUES (?,?,?,?)',[(key,next(iter(values)) if len(values)==1 else None,
            'proxy: previously trained-on development label; not a new human confirmation','active' if len(values)==1 else 'conflict') for key,values in grouped.items()])
        db.commit()
    update_seconds=time.perf_counter()-start;process_cost=sample.finish();began=time.perf_counter()
    with sqlite3.connect(destination/'correction_memory.sqlite') as db:
        stored={key:value for key,value in db.execute("SELECT fingerprint,label_id FROM corrections WHERE status='active'")}
    after=baseline_pred.copy();hits=0
    for i,row in enumerate(val_rows):
        if row['fingerprint'] in stored:after[i]=stored[row['fingerprint']];hits+=1
    incoming_pred=np.asarray([stored.get(r['fingerprint'],-1) for r in incoming_rows])
    result={'update_seconds':update_seconds,'validation_lookup_seconds':time.perf_counter()-began,
        'entries':len(stored),'conflicting_signatures':sum(len(v)>1 for v in grouped.values()),
        'validation_exact_feature_hits':hits,'validation_coverage':hits/len(val_rows),
        'proxy_update_example_resolved_fraction':float((incoming_pred>=0).mean()),
        'proxy_update_example_lookup_correct_fraction':float((incoming_pred==np.asarray([r['y'] for r in incoming_rows])).mean()),
        'additional_storage_bytes':(destination/'correction_memory.sqlite').stat().st_size,
        'model_parameters_changed':False,'mandatory_human_review':True,
        'limitation':'Full-feature exact match is not a verified business product identifier; no new confirmations are present.'}
    result.update(process_cost)
    write(destination/'update.json',result)
    return after,result
