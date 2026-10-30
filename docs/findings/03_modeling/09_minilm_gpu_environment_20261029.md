# Isolated CatalogIQ MiniLM CUDA environment

Created October 29, 2026 for Windows x64 and the NVIDIA GeForce RTX 3060 Laptop
GPU with 6 GiB dedicated VRAM. The existing Python 3.13.7 CPU environment is
preserved. Model code, hyperparameters, frozen artifacts and allocations are
unchanged. No model training or GPU benchmark is authorized by this setup.

A subsequent separately authorized [20-step CUDA benchmark](10_mnfr_minilm_gpu_benchmark_20261029.md)
verified actual training execution. The results below describe environment setup
and its original non-training verification.

## Reproduce the environment

Run from the repository root in PowerShell. Use an explicit interpreter so shell
PATH, activation policy and another Python installation cannot select the wrong
environment. Create a new directory; do not overwrite an existing environment.

```powershell
$cpuPython = 'C:\Users\sebas\AppData\Local\Programs\Python\Python313\python.exe'
& $cpuPython -m venv .venv-gpu
$gpuPython = Join-Path (Get-Location) '.venv-gpu/Scripts/python.exe'
New-Item -ItemType Directory -Force '.cache/gpu_environment_setup_20261029'
& $gpuPython -m pip --isolated install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple --report '.cache/gpu_environment_setup_20261029/pip_install_report.json' --log '.cache/gpu_environment_setup_20261029/pip_install.log' -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw 'Stop: GPU dependency installation failed.' }
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Stop: GPU dependency conflicts.' }
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
```

`requirements-gpu.txt` pins the relevant CPU environment's runtime, transitive
dependencies and build-tool versions. PyTorch alone changes from the CPU build
to `2.11.0+cu128`. Its CPython 3.13 Windows wheel is selected from the official
CUDA 12.8 index and pinned by URL and SHA-256:

`6f367e62fd81b75cdf23ca4b75ced834d2db2cf98d1588ac935bde345de9de23`

The remaining pinned wheels come from PyPI. No source-package builds are allowed.
The environment uses `include-system-site-packages = false`, and `/.venv-gpu/`
is explicitly ignored by Git. Repository source is selected with `PYTHONPATH`;
the setup does not reinstall the project or alter the CPU environment's editable
installation. A separate CUDA Toolkit, cuDNN installation, torchvision and
torchaudio are unnecessary for this trainer.

## Verify without training

```powershell
@'
import torch
print('torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA compiled:', torch.backends.cuda.is_built())
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
'@ | & $gpuPython -
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

The focused suite deliberately selects text checks, preparation and orchestration
tests, and recording benchmark doubles. It excludes actual neural fine-tuning
tests in `test_segment_transformer.py`.

```powershell
@'
import sys, unittest
sys.path.insert(0, 'tests')
from test_segment_transformer import TransformerTextTests
suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(TransformerTextTests)])
for pattern in (
    'test_target_minilm_orchestration.py',
    'test_mnfr_minilm_benchmark.py',
    'test_mnfr_minilm_orchestration.py',
    'test_target_minilm_preparation.py',
    'test_target_development_preparation.py',
    'test_segment_frozen_development.py',
):
    suite.addTests(unittest.defaultTestLoader.discover('tests', pattern=pattern))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
'@ | & $gpuPython -
```

Use cached files only for the real Mnfr dry run. The guard patches cause an error
if a training call or encoder initialization is attempted.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
@'
from unittest.mock import patch
from catalogiq.target_minilm_preparation import prepare_target_minilm, run_mnfr_minilm
prepared = prepare_target_minilm(
    'data/processed/split_integration_reproduced_20261010',
    'Mnfr',
    '.cache/gpu_environment_setup_20261029/planned_mnfr_outputs',
)
with patch('catalogiq.target_minilm_preparation.train_transformer', autospec=True,
           side_effect=AssertionError('Training forbidden')) as trainer, \
     patch('catalogiq.segment_transformer._initialize_torch',
           side_effect=AssertionError('Encoder initialization forbidden')):
    report = run_mnfr_minilm(prepared)
    trainer.assert_not_called()
assert report['dry_run'] and not report['training_authorized'] and not report['model_trained']
assert report['readiness']['ready_for_training'], report['readiness']['blockers']
assert report['readiness']['resolved_device'] == 'cuda'
assert not prepared.output_directory.exists()
print(report['readiness'])
'@ | & $gpuPython -
```

## Verification results

Installation completed without dependency conflicts. `pip check` reported
`No broken requirements found.` All 46 pinned package versions were verified;
the 45 non-PyTorch versions match the existing CPU environment.

| Check | Result |
|---|---|
| Python | 3.13.7, isolated `.venv-gpu/Scripts/python.exe` |
| PyTorch | `2.11.0+cu128` |
| Compiled CUDA runtime | 12.8 |
| CUDA compiled / available | True / True |
| Detected GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Compute capability | 8.6; supported by the wheel's `sm_86` kernels |
| NVIDIA driver | 591.59 |
| Dedicated VRAM total / free via `nvidia-smi` | 6,144 / 5,994 MiB at verification |
| Focused tests | 66 passed; zero failures, errors or skips |
| Mnfr dry-run readiness | Ready; `auto` resolves to `cuda`; no blockers |

The Mnfr dry run retained model identity
`minilm-mnfr-category256-c3458037a670c8f0`, frozen V4 roles, ordered labels
`All others` = 0 and `J&J` = 1, 55,233 supervised training rows and 11,812
validation rows. It retained the five-field/256-token representation, batch
sizes 32/64, FP32, three epochs, learning rate 3e-5, weight decay 0.01, dropout
0.10, warmup fraction 0.10, seed 42 and four CPU threads. The planned schedule
remains 5,181 optimizer steps with 518 warmup steps.

Model identity, configuration, vocabulary/mapping, record/group membership,
class counts, encoded labels, text/token hashes, tokenizer fingerprints and
schedule match the original CPU benchmark. The saved benchmark artifacts are
unchanged. The dry run did not initialize an encoder, download weights, train,
inspect protected test features/labels or create a model output directory.

All 203 original CPU distribution entries retain their versions, locations and
METADATA/RECORD hashes. Its PyTorch remains `2.11.0+cpu`, CUDA runtime `None`,
and CUDA availability False. All 13 frozen algorithm source hashes still match
their authoritative manifest. CPU site-packages are excluded from the GPU
environment's import path.

Local verification evidence is kept in `.cache/gpu_environment_setup_20261029/`:

- `pip_install.log`, `pip_install_report.json`, `torch_wheel_source.json`
- `gpu_environment.json`, `gpu_freeze.txt`
- `focused_tests.txt`, `focused_tests.json`, `mnfr_dry_run.json`
- `cpu_before.json`, `cpu_after.json`, `preservation_checks.json`

Device `auto` may now resolve to CUDA when this environment is used; the same
configuration in the original CPU environment continues to resolve to CPU.
Readiness establishes environment and input compatibility, not GPU training
execution, memory fit, throughput or numerical equivalence to CPU training.
