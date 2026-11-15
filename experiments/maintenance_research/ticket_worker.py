"""Synthetic acceptance worker; use the repository's existing inference adapter."""
import json
from pathlib import Path
import sys
import unittest

from core import HERE, ROOT, install_guard, snapshot, write


def run(output):
    if sys.stdin.read(1)!='1':raise RuntimeError('Use the bounded ticket_run.py supervisor')
    access=install_guard(output,[])
    import torch
    from threadpoolctl import threadpool_limits,threadpool_info
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def deny_cuda(*args,**kwargs):raise RuntimeError('CUDA initialization is prohibited')
    torch.cuda._lazy_init=deny_cuda
    sys.path.insert(0,str(ROOT/'tests'))
    with threadpool_limits(limits=4):
        suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test*.py')
        import test_inference_adapter
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_inference_adapter))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise RuntimeError('Synthetic acceptance tests failed')
        from ticket_demo import demonstrate
        write(output/'engineering_demo.json',demonstrate(output/'demo'))
        pools=threadpool_info()
        if torch.cuda.is_initialized() or torch.get_num_threads()>4 or torch.get_num_interop_threads()>1 or any(p['num_threads']>4 for p in pools):
            raise RuntimeError('CPU-only safety check failed')
        write(output/'result.json',{'status':'passed','synthetic_tests':result.testsRun,
            'cuda_initialized':False,'finished_models_loaded':False,'protected_data_accessed':False,
            'production_integration':False,'data_access_counts':access,'native_threadpools':pools})


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'src'))
    run(Path(sys.argv[1]).resolve())
