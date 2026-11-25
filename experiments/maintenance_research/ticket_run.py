"""Fresh, bounded Windows synthetic check; no private sessions or source caches.

Usage: python -B experiments/maintenance_research/ticket_run.py --output SHORT_PATH
Optional: --web-domain PATH_TO_MAXS_UNMODIFIED_DOMAIN_JS (Node must exist).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from core import HERE, ROOT, sha, snapshot, write
from resources import BoundedJob


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--web-domain',type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():parser.error('Use a fresh output directory; previous results are never overwritten')
    if os.name!='nt':parser.error('This OS-bounded runner requires Windows; use the documented synthetic unittest suite on other platforms')
    if len(str(output))+145>=260:parser.error('Choose a shorter Windows output path')
    if args.web_domain and (not args.web_domain.is_file() or not shutil.which('node')):
        parser.error('The optional web check requires an existing domain.js and Node; nothing is downloaded')
    output.mkdir(parents=True,exist_ok=False);temporary=output/'t';temporary.mkdir()
    write(output/'protocol.json',{'wall_seconds':600,'cpu_seconds':2400,'worker_bytes':4*1024**3,
          'affinity_cpus':4,'numerical_threads':4,'interop_threads':1,'synthetic_only':True,'cuda_allowed':False})
    before={str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),*HERE.glob('*.mjs'),
            ROOT/'src/catalogiq/inference_adapter.py',ROOT/'src/catalogiq/inference_engine.py',ROOT/'src/catalogiq/inference_models.py']}
    if args.web_domain:before[str(args.web_domain.resolve())]=sha(args.web_domain)
    write(output/'source_sha256.json',before)
    job=BoundedJob(4*1024**3,2400,4);worker=None;reason=None;peak=0
    began=time.monotonic();own=time.process_time();codes=[]
    try:
        limits=job.enforcement()
        if limits['allowed_cpu_count']!=4 or limits['job_commit_limit_bytes']!=4*1024**3:
            raise RuntimeError('Resource limits could not be enforced')
        env=os.environ.copy()
        env.update({n:'4' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')})
        env.update({'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
             'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
        env['PYTHONPATH']=str(ROOT/'src')
        env.update({n:str(temporary) for n in ('TMP','TEMP','TMPDIR','HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR')})
        commands=[('python',[sys.executable,'-B',str(HERE/'ticket_worker.py'),str(output)])]
        if args.web_domain:
            commands.append(('web',[shutil.which('node'),str(HERE/'ticket_web_test.mjs'),str(args.web_domain.resolve()),str(output/'web')]))
        for label,command in commands:
            if label=='web':(output/'web').mkdir()
            with (output/(label+'.log')).open('x',encoding='utf8') as log:
                worker=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=log,
                    stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|0x4)
                job.assign(worker);job.resume(worker)
                if label=='python':worker.stdin.write(b'1')
                worker.stdin.close()
                while worker.poll() is None:
                    usage=job.usage();peak=max(peak,usage['working_set_bytes'])
                    if time.monotonic()-began>=595:reason='wall limit approaching'
                    elif job.accounting()['cpu_seconds']+time.process_time()-own>=2395:reason='CPU limit approaching'
                    elif usage['working_set_bytes']>=4*1024**3:reason='RSS limit reached'
                    if reason:job.terminate();break
                    time.sleep(.05)
                code=worker.wait(timeout=10);codes.append({'stage':label,'exit_code':code})
                if code:raise RuntimeError(f'{label} checks failed; inspect {output}')
        require_path=output/'result.json'
        if not require_path.is_file():raise RuntimeError('Missing worker result')
        if any(sha(ROOT/p)!=v for p,v in before.items()):raise RuntimeError('Source preservation failed')
        write(output/'verification.json',{'status':'passed','source_preservation':True,
              'private_artifacts_required':False,'production_integration':False,'exit_codes':codes})
        print('PASSED='+str(output),flush=True)
    finally:
        if worker and worker.poll() is None:job.terminate();worker.wait(timeout=10)
        write(output/'resources.json',{'wall_seconds':time.monotonic()-began,
             'cpu_seconds':job.accounting()['cpu_seconds']+time.process_time()-own,
             'peak_sampled_rss_bytes':peak,'limits':job.enforcement(),'accounting':job.accounting(),
             'stop_reason':reason,'exit_codes':codes})
        job.close()


if __name__=='__main__':main()
