# Mnfr MiniLM: actual 20-step CPU benchmark

Run October 29, 2026 through `run_mnfr_minilm(prepared, dry_run=False,
benchmark_steps=20)`. Exactly 20 optimizer steps and 20 subsequent scheduler
steps completed. The run stopped before epoch validation, checkpoint saving or
prediction export. The full experiment was not run or merged.

The subsequently generalized orchestration also requires
`training_authorized=True` for actual execution. Its read-only regression checks
preserved this benchmark's configuration and data/token fingerprints; the
benchmark was not rerun during generalization.

## Configuration and step cap

The complete configuration matches the approved dry-run configuration exactly:
seed 42, batch size 32, evaluation batch size 64, FP32, four PyTorch CPU threads,
learning rate 3e-5, weight decay 0.01, dropout 0.10, warmup fraction 0.10, three
configured epochs, pinned MiniLM revision and ProductCategory-first five-field
256-token representation. Cross-entropy remains unweighted.

All 55,233 labeled V4 training records remain in the original seeded shuffle;
the benchmark executes its first 20 batches (640 records). It does not shorten
the dataset or epochs to force an early exit. The full 5,181-step schedule and
518 warmup steps are retained. The first optimizer step uses the approved
schedule's zero learning rate; all 20 forward/backward/optimizer/scheduler
cycles execute. The measured rates match the approved warmup formula exactly.

The benchmark budget is implemented around the unchanged legacy loop with
scoped loss/scheduler instrumentation and an optimizer post-step hook. Neural
operations delegate unchanged, while an expected stop signal fires after the
20th optimizer and scheduler update. Hooks are restored on exit. A benchmark
has its own report and never receives a completed training manifest.

## Results

| Measure | Observed |
|---|---:|
| Device | CPU; PyTorch 2.11.0+cpu; no CUDA available |
| Actual optimizer / scheduler steps | 20 / 20 |
| Trainer initialization | 0.876 s |
| Mean training step | 7.809 s |
| Median training step | 7.838 s |
| First step | 8.943 s |
| Mean steps 2–20 | 7.750 s |
| Total measured training-step time | 156.190 s |
| Initialization + 5,181 projected steps | 40,461.842 s; about 11 h 14 min |
| Orchestration preflight, including tokenization | 30.694 s |
| Whole benchmark, including initial data preparation | 196.910 s |
| Peak process working set | 4.381 GiB |
| Process working set after step 20 | 3.746 GiB |

Initialization is measured from entry into the legacy trainer to its model-ready
callback; framework imports, data preparation, tokenization and cached-file
verification precede it. Step intervals include batch preparation, forward and
backward passes, optimizer and scheduler work. Instrumentation logging is
excluded from the following interval. Peak process memory includes preparation
as well as training; it is not model-only tensor memory.

The runtime projection is mean step time multiplied by 5,181, plus observed
trainer initialization. Input preparation, three epoch validation passes and
final saving/export are additional and were not benchmarked. It is a throughput
estimate from 20 batches, not a complete experiment runtime measurement.

## Correctness and errors

Every loss was finite: range 0.654719–0.695841, with the first loss 0.691755 and
last loss 0.654719. These training losses do not establish classification quality.

All 640 labels matched the unchanged seed-42 training permutation and frozen
Mnfr mapping: All others = 0 (627 observed), J&J = 1 (13 observed). Metadata in
both the run protocol and initialization manifest identifies Mnfr, its ordered
vocabulary, five input fields, 256-token budget, configuration and V4 split.

There were no training errors. Transformers reported one nonfatal unexpected
checkpoint entry, `embeddings.position_ids`. The installed BERT implementation
generates this as a nonpersistent buffer; this was not a missing learned weight.
Training proceeded normally. No configuration was changed in response.

Cached pretrained files were verified unchanged after the run. Frozen algorithm
source hashes still match the authoritative manifest. Protected test features
and labels were not inspected, and `test.csv` was not read or hashed.

## Artifacts and tests

All benchmark artifacts are separate from production and Segment model outputs:

`data/processed/mnfr_minilm_benchmark_v1_20261029/mnfr/minilm-mnfr-category256-c3458037a670c8f0/`

- `benchmark_report.json`: complete metadata, timings, memory, step losses and checks.
- `benchmark_steps.csv`: per-step timing, loss, learning rate and label counts.
- `run_protocol.json`, `initialization.json`, `model_config.json`: verified run configuration and identity.

There is no `training_manifest.json`, checkpoint or prediction output. The
planned production Mnfr output directory remains absent. This partial run is
not a production model or a resumable full-training checkpoint.

All 58 focused tests passed. New tests verify the exact step cap, optimizer and
scheduler alignment, finite loss, unweighted loss, correct labels, hook cleanup
and orchestration output isolation using recording doubles. The actual 20-step
benchmark separately verified real forward/backward execution.
