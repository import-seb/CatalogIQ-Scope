# Mnfr MiniLM: actual 20-step CUDA benchmark

Completed October 29, 2026 in `.venv-gpu` using NVIDIA GeForce RTX 3060 Laptop
GPU, PyTorch `2.11.0+cu128`, CUDA runtime 12.8 and NVIDIA driver 591.59. Exactly
20 actual optimizer steps and 20 scheduler updates completed on CUDA. No full
training, epoch validation, predictions, checkpoint saving or merge occurred.

## Configuration and correctness

The full configuration matches the saved CPU benchmark: pinned
`sentence-transformers/all-MiniLM-L6-v2` revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, ProductCategory-first five-field
representation, 256-token maximum, batch size 32, evaluation batch size 64,
FP32, learning rate 3e-5, weight decay 0.01, dropout 0.10, seed 42, four CPU
threads and three configured epochs. Cross-entropy is unweighted, AMP and TF32
are disabled, and the existing deterministic settings remain unchanged.

The existing `run_mnfr_minilm` orchestration and benchmark instrumentation ran
unchanged. The full dataset and three-epoch schedule were retained: 5,181
optimizer steps and 518 warmup steps. The first scheduled learning rate is
zero, as in the CPU benchmark; all 20 forward/backward/optimizer/scheduler cycles
executed. Every measured learning rate and seeded batch label count matched
the CPU benchmark.

All 640 training labels matched the frozen mapping: `All others` = 0 (627
observed), `J&J` = 1 (13 observed). All losses were finite, ranging from
0.653448 to 0.694942; first loss 0.688162, last loss 0.659030. CPU and CUDA losses
are not expected to be numerically identical, and these losses do not establish
classification quality.

Initial model state, model identity, ordered vocabulary, label encoding,
record/group membership, product text, tokens, tokenizer fingerprints and full
configuration match the CPU baseline. Run protocol and initialization metadata
identify Mnfr and the five-field/256-token representation; initialization records
the actual CUDA device, runtime and PyTorch build.

## Timing and memory

| Measure | GPU benchmark |
|---|---:|
| Actual optimizer / scheduler steps | 20 / 20 |
| Trainer initialization | 1.289 s |
| Total measured training-step time | 4.684 s |
| Mean training step, all 20 | 0.234190 s |
| Median training step | 0.173615 s |
| First training step | 1.381370 s |
| Mean steps 2 through 20 | 0.173812 s |
| Throughput, all-20 mean | 136.64 records/s |
| Speedup versus requested CPU baseline, 7.81 s/step | 33.349x |
| Speedup versus exact saved CPU mean, 7.809490 s/step | 33.347x |
| Peak allocated GPU memory | 1,746,948,096 bytes; 1.627 GiB |
| Peak reserved GPU memory | 1,889,533,952 bytes; 1.760 GiB |
| Whole benchmark, including preparation | 52.240 s |

The existing instrumentation synchronizes CUDA after initialization and after
each optimizer/scheduler completion before stopping its timer. Step time includes
batch preparation, transfers, forward/backward, label checks, optimizer and
scheduler work. Instrumentation printing is excluded from the next interval.
Initialization spans entry into the legacy trainer through the model-ready
callback; tokenization/cache verification precede it. The first training step
includes cold CUDA execution overhead and is included in the headline mean.

CUDA peak statistics were reset before the run, with zero allocated/reserved
allocator bytes at baseline. Peak allocated and reserved bytes were sampled after
the benchmark. They describe the PyTorch allocator, excluding driver/context
memory outside it. No batch-size reduction or mixed precision was used. The
20-step FP32 workload fit comfortably within the available 6 GiB dedicated VRAM.
Full-run validation-batch memory and sustained thermal performance were not
measured.

## Runtime projections

These projections use the complete 20-step mean for consistency with the CPU
comparison, including the cold first step in that mean.

| Scope | Steps | Training-step projection | With initialization |
|---|---:|---:|---:|
| Mnfr | 5,181 | 1,213.339 s; 20 min 13 s | 1,214.627 s; 20 min 15 s |
| All four targets | 20,856 | 4,884.268 s; 1 h 21 min 24 s | 4,889.422 s; 1 h 21 min 29 s |

The four-target estimate assumes the same throughput and one measured-duration
initialization for each target. Only Mnfr throughput was measured. Tokenization,
epoch validation, checkpoint saving and exports are additional; these are not
complete experiment wall-clock measurements. Laptop thermals, power limits and
the targets' input-length distributions may change sustained throughput.

## Artifacts and reproduction

Isolated benchmark directory:

`data/processed/mnfr_minilm_gpu_benchmark_v1_20261029/mnfr/minilm-mnfr-category256-c3458037a670c8f0/`

- `benchmark_report.json`: orchestration metadata, synchronized timing, step losses and allocation peaks.
- `benchmark_steps.csv`: 20 per-step timings, losses, learning rates and label counts.
- `gpu_benchmark_summary.json`: allocated/reserved peaks, CPU comparison, runtime projections and verification results.
- `run_protocol.json`, `initialization.json`, `model_config.json`: exact run identity and configuration.

There is no completed-training manifest, model checkpoint or prediction output.
The CPU benchmark directory and its file hashes remain unchanged. Frozen algorithm
source hashes still match their authoritative manifest. Protected test features
and labels were not inspected, and `test.csv` was not opened or hashed.

The following is the authorized invocation used inside the verification runner.
Future execution requires separate authorization and a new output root; this
existing benchmark directory must not be overwritten.

```python
from catalogiq.target_minilm_preparation import prepare_target_minilm, run_mnfr_minilm

prepared = prepare_target_minilm(
    "data/processed/split_integration_reproduced_20261010",
    "Mnfr",
    "data/processed/mnfr_minilm_gpu_benchmark_v1_20261029",
)
report = run_mnfr_minilm(
    prepared, dry_run=False, training_authorized=True, benchmark_steps=20,
)
```

Use `.venv-gpu/Scripts/python.exe` with repository `src` on `PYTHONPATH`, Hugging
Face offline mode enabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The runner first
asserted exact CPU configuration/identity, CUDA availability and the initial
model hash, and reset CUDA peak statistics. After execution it verified all 20
updates, labels, learning rates, metadata, source hashes and unchanged CPU
artifacts, then recorded peak allocator memory and the projections separately.

No benchmark instrumentation or neural implementation edits were needed, and
no environment packages changed. The existing focused suite had already passed
all 66 tests in this GPU environment during setup. Post-run checks validated the
actual CUDA execution separately; no additional training test was executed.

There were no training errors or out-of-memory failures. The loader emitted the
same nonfatal unexpected `embeddings.position_ids` entry noted during the CPU
benchmark. No learned weight was reported missing. There are no benchmark
blockers; the full experiment remains unrun.
