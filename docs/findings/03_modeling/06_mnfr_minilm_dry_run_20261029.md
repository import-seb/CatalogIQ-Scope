# Mnfr MiniLM orchestration and dry run

Dry run verified October 29, 2026. During that dry run, no actual model training,
encoder initialization or weight downloads were performed. A subsequent,
separately authorized [20-step benchmark](07_mnfr_minilm_benchmark_20261029.md)
is recorded independently. No full experiment has run. Segment source and
frozen V4 artifacts remain unchanged.

## Entry point

`catalogiq.target_minilm_preparation.run_mnfr_minilm` accepts a prepared Mnfr
object and defaults to `dry_run=True`:

It is now a compatibility wrapper around the shared `run_target_minilm` entry
point. See [the four-target orchestration note](08_target_minilm_orchestration_20261029.md).

```python
from catalogiq.target_minilm_preparation import prepare_target_minilm, run_mnfr_minilm

prepared = prepare_target_minilm(
    "data/processed/split_integration_reproduced_20261010",
    "Mnfr",
    "data/processed/minilm_mnfr_runs_v1",
)
report = run_mnfr_minilm(prepared)  # Dry run; no model or output directory created.
```

The dry run uses the saved local tokenizer, validates label encoding and trainer
arguments, checks source/tokenizer stability, and records the full model config,
ordered vocabulary, model identity, five input fields, token budget and frozen
split identity. It checks installed dependencies, resolves the requested device
without model initialization, and hashes the pinned cached encoder configuration
and safetensors using the existing local-only cache helper. Missing cache/runtime
prerequisites appear as readiness blockers; no download is attempted.

Only an explicitly authorized future call with both `dry_run=False` and
`training_authorized=True` reaches the existing
`train_transformer` loop. That path was implemented but not executed against a
real encoder. Tests use a recording stub. A new `run_protocol.json` is written in
the prepared Mnfr directory before delegation. The initialization manifest is
annotated through the legacy progress callback before training batches, and the
completed training manifest receives the same correct Mnfr identity and updated
artifact checksums. Weights, neural architecture, pooling, optimization and
checkpoint selection remain entirely in the unchanged trainer. No new prediction
writer or inference/packaging code was added; future execution inherits the
legacy trainer's existing training artifacts.

## Actual dry-run result

Saved report: `data/processed/mnfr_minilm_dry_run_v1_20261029/dry_run.json`.

| Ordered label | Label ID | Training rows | Validation rows |
|---|---:|---:|---:|
| All others | 0 | 54,496 | 11,574 |
| J&J | 1 | 737 | 238 |
| Total | | 55,233 | 11,812 |

The model identity is `minilm-mnfr-category256-c3458037a670c8f0`. Its future
directory is under `data/processed/minilm_mnfr_runs_v1/mnfr/`, separate from
Segment outputs. This directory was not created.

Configuration: pinned `sentence-transformers/all-MiniLM-L6-v2` revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, ProductCategory-first five-field
representation, 256-token combined budget, unchanged character limits, seed 42,
FP32, batch sizes 32/64, learning rate 3e-5, weight decay 0.01, warmup fraction
0.10 and three fixed epochs. The planned schedule has 5,181 optimizer steps and
518 warmup steps; validation does not select a checkpoint.

The source remains `segment-final-20261009`, `rule-final-v4`, with the original
128-token grouping contract and all 12,160 protected test records unchanged.
`test.csv` was neither opened nor hashed. The report binds source completion,
assignments, ordered record/group membership, encoded labels, product text,
tokens, tokenizer bytes and cached encoder bytes by SHA-256.

All readiness checks passed in the tested Python 3.13.7 environment: NumPy 2.3.3,
pandas 2.3.2, PyTorch 2.11.0, Transformers 5.3.0 and tokenizers 0.22.2. The `auto`
device resolves to CPU. CPU configuration/cache readiness has no blockers; GPU
training is not verified, and explicitly requesting unavailable CUDA is rejected.
This dry run establishes configuration and data readiness, not model quality or
training runtime performance. Training remains subject to explicit authorization.

## Tests

All 53 focused tests passed, including ten new orchestration tests. They cover
target metadata, ordered label encoding/counts, frozen role alignment, output
isolation, default dry-run behavior, local cache/dependency readiness, CUDA
availability checks, source/tokenizer drift, and delegation to a non-neural stub.
The stub verifies both legacy manifests are corrected, state hashes are retained
and artifact checksums are refreshed. No neural fine-tuning tests were run.
