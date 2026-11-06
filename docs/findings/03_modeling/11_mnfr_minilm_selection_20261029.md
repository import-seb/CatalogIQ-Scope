# Mnfr MiniLM: unweighted versus moderate class weighting

Experiment completed October 29, 2026. This experiment compares exactly two final
third-epoch Mnfr checkpoints, using the frozen V4 development allocation. The
protected test population is excluded throughout. No other target is trained.

## Predeclared comparison

- Baseline: ordinary unweighted mean cross-entropy.
- Candidate: mean cross-entropy with ordered training weights `[1.0, 5.0]` for
  `All others`, `J&J`. Validation loss remains unweighted for both candidates.
- Selection: higher validation macro-F1, with an absolute difference of 0.001
  or less treated as an effective tie and resolved in favor of unweighted CE.
- Within each run: three fixed epochs; no early stopping, epoch selection,
  threshold tuning, resampling or additional candidates.

The protocol was saved before either run in
`data/processed/mnfr_minilm_selection_v1_20261029/experiment_protocol.json`.
Training uses `.venv-gpu`, PyTorch `2.11.0+cu128`, CUDA 12.8 and NVIDIA GeForce
RTX 3060 Laptop GPU. The original CPU environment is unchanged.

All other configuration is the approved benchmark configuration: pinned MiniLM
revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, ProductCategory-first
five-field representation, 256-token maximum, FP32, training/evaluation batches
32/64, three epochs, learning rate 3e-5, weight decay 0.01, warmup fraction 0.10,
dropout 0.10, seed 42, four CPU threads and the existing deterministic settings.
Each run completes 5,181 optimizer updates, with 518 warmup updates.

Ordered labels and support:

| Label ID | Label | Training records | Validation records |
|---|---|---:|---:|
| 0 | All others | 54,496 | 11,574 |
| 1 | J&J | 737 | 238 |

## Implementation and isolation

`catalogiq.mnfr_minilm_experiment` delegates to `run_mnfr_minilm` and the frozen
trainer. A scoped CE-forward adapter adds only the class-weight argument during
gradient-enabled training; no-grad validation delegates to the original CE.
The unweighted path delegates to the original CE without changing its math.
The patch is restored on completion or exception, and each real candidate runs
in a separate Python process. The adapter rejects other targets and reversed
label order. No frozen trainer or shared target orchestration code changed.

The weighted candidate receives its own identity derived from the base input
identity and explicit loss configuration. Both candidates retain their own
directories, checkpoints, validation exports, histories, manifests, loss
configuration and timing results. The five-field representation, original
product groups and all target-label eligibility rules are retained.

All 74 focused tests passed before training, including eight new Mnfr-specific
checks. They verify exact weighted CE, unchanged unweighted/validation CE,
exception cleanup, distinct candidate identities with identical inputs,
authorization/CUDA guards, validation membership, tie handling and selected-only
handoff metadata. The new tests do not train an encoder or optimizer.

## Reproduction

Use a new experiment directory for any separately authorized repetition; completed
candidate directories cannot be overwritten. The actual invocations were:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$experiment = 'data/processed/mnfr_minilm_selection_v1_20261029'
& '.venv-gpu/Scripts/python.exe' -u -m catalogiq.mnfr_minilm_experiment train --experiment-directory $experiment --candidate unweighted --training-authorized
& '.venv-gpu/Scripts/python.exe' -u -m catalogiq.mnfr_minilm_experiment train --experiment-directory $experiment --candidate weighted_1_5 --training-authorized
& '.venv-gpu/Scripts/python.exe' -m catalogiq.mnfr_minilm_experiment select --experiment-directory $experiment
```

Training uses cached pretrained weights only. Actual stdout/stderr is retained
in the experiment's `unweighted_run.log` and `weighted_1_5_run.log`. Original
CPU/GPU benchmark outputs, Segment artifacts and approved allocations are not
overwritten.

## Results

Select the **unweighted final third-epoch checkpoint**. It has higher validation
macro-F1, the same J&J recall and one fewer false positive. Neither final model
shows majority-class collapse on these validation records. Weighting did not
improve this fixed comparison and is not used in the designated handoff.

| Candidate | Accuracy | Macro-F1 | J&J precision | J&J recall | J&J F1 | Trainer runtime |
|---|---:|---:|---:|---:|---:|---:|
| Unweighted, selected | 99.9661% | 0.99572949 | 98.7500% | 99.5798% | 99.1632% | 1,322.567 s; 22 min 03 s |
| Weighted `[1, 5]` | 99.9577% | 0.99467278 | 98.3402% | 99.5798% | 98.9562% | 1,745.295 s; 29 min 05 s |

Confusion matrices use rows = true label and columns = predicted label, both in
the fixed order `[All others, J&J]`:

```text
Unweighted:                 Weighted [1, 5]:
[[11571,   3],              [[11570,   4],
 [    1, 237]]               [    1, 237]]
```

Both correctly identify 237 of 238 J&J records. Unweighted makes four total
errors; weighted makes five. A paired validation check confirms that both miss
the same J&J record and their predictions differ on exactly one record, the
additional weighted-model false positive. The unweighted macro-F1 advantage is
0.0010567129. This is greater than the predeclared 0.001 effective-tie threshold,
so the tie preference is not needed. It is still a one-record difference on a
validation set with only 238 positive examples, not a claim of statistically
established superiority on unseen data.

Trainer runtime includes initialization, all training/validation epochs and CPU
state hashing, but excludes final artifact writing. Total candidate wall times,
including preflight/tokenization and saving, were 1,352.054 s (22 min 32 s) and
1,786.287 s (29 min 46 s), respectively. CUDA was synchronized around complete
candidate calls. These are measured runs, not benchmark extrapolations. Laptop
conditions varied: a weighted-run snapshot recorded 88 C, 697 MHz SM clock and
99% GPU utilization. Runtime differences should not be attributed solely to
class weighting.

All training losses passed the trainer's finite-loss guard, and all three saved
epoch losses and final validation probabilities were finite. Each run completed
exactly 5,181 training batches/optimizer updates, processed 165,699 supervised
training examples across three epochs and evaluated the same 11,812 validation
records three times. The initial model hash matched between candidates and the
GPU benchmark. All other training configuration and split identities matched.

The full runs fit in 6 GiB: peak allocated memory was approximately 1.627 GiB for
both candidates, with peak reserved memory approximately 1.949 GiB. There were
no training or out-of-memory errors. Both runs emitted the same nonfatal
`embeddings.position_ids` loader notice seen in the benchmarks.

## Designated local handoff

Only the unweighted checkpoint is designated for Maria:

`data/processed/mnfr_minilm_selection_v1_20261029/mnfr_minilm_selected_for_maria_v1_20261029.zip`

The ZIP contains `maria_handoff/` contents (at the ZIP root): one `checkpoint.pt`,
the saved tokenizer, encoder/model configuration, evaluation, original training
manifest, bundle `manifest.json` and README. The checkpoint itself embeds Mnfr,
ordered labels/mapping, five input fields, token budget, representation, split
identity, full configuration, loss configuration and candidate selection. It
contains the exact selected state tensors; adding metadata changes the serialized
file checksum without changing the model state hash. The copied source training
manifest records original experiment artifacts; the bundle manifest records the
packaged file checksums.

- Selected identity: `minilm-mnfr-category256-c3458037a670c8f0`.
- Weighted archived identity: `minilm-mnfr-category256-9c07ac618e70007a`.
- Selected final model-state SHA-256: `237063f3097fd7de26db0e5b13e4d8c29128a5f7283347e2cc6ad5f6f61dafaf`.
- Packaged checkpoint SHA-256: `13af52155597c62e6c29ce83f229fd57fa20bee8e4184b8b654b79713deff797`.
- ZIP SHA-256: `2ca02a3a68b0c168c49587936d220e83d21c0f9f4e3192dd6d11d87a7c0e424e`.
- ZIP size: 83,613,152 bytes.

Both full experiment directories retain their original checkpoints, histories,
validation predictions, evaluation, candidate results and manifests. The parent
directory retains the predeclared `experiment_protocol.json`, `selection.json`,
`handoff_delivery.json`, logs and exact source/dependency snapshots in
`reproduction_sources/`. Only the selected weights are included in the handoff
ZIP. Packaging used `select_and_package`, followed by `shutil.make_archive` on
the selected bundle; no model was retrained or altered during packaging.

## Final verification and limits

The selected checkpoint was reloaded with `weights_only=True`; its state hash,
target, ordered labels, representation, split identity and configuration matched
the selected result. All candidate artifact hashes, bundle file hashes and ZIP
CRC checks passed. All 13 frozen algorithm source hashes and the five consumed
V4 artifact checksums remain unchanged. All 203 original CPU distribution
metadata entries remain unchanged. No packages or model hyperparameters changed.

No protected-test features, labels or predictions were inspected; `test.csv`
was not opened or hashed. This is development-validation model selection, not
protected-test certification. No other target was trained, no inference code
was implemented, no external delivery to Maria occurred and nothing was merged.
There are no remaining training or packaging blockers.
