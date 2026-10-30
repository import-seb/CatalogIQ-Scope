# Shared target-aware MiniLM orchestration

Implemented October 29, 2026. `run_target_minilm` in
`catalogiq.target_minilm_preparation` now handles Mnfr, TargetAgeGroup, Brand and
Sub-Segment. `run_mnfr_minilm` delegates to it as a Mnfr-only compatibility wrapper.
Platform remains excluded. No neural implementation or allocation logic changed.

## Entry point and safeguards

```python
from catalogiq.target_minilm_preparation import prepare_target_minilm, run_target_minilm

prepared = prepare_target_minilm(
    "data/processed/split_integration_reproduced_20261010",
    "TargetAgeGroup",
    "data/processed/target_minilm_runs_v1",
)
report = run_target_minilm(prepared)  # Read-only dry run by default.
```

Mnfr and TargetAgeGroup retain frozen V4 roles. Brand and Sub-Segment require
the approved directories and explicit completion checksums documented in
[input preparation](05_target_minilm_preparation_20261029.md). Their saved record
order, class order and development assignments remain authoritative. Missing
labels remain in the development population but are excluded from supervision.

Actual execution requires both `dry_run=False` and `training_authorized=True`;
the latter records explicit authorization rather than granting it. A missing
authorization fails before tokenization, directory creation or delegation.
The existing optional benchmark budget remains Mnfr-only.

Dry runs verify label encoding, trainer argument compatibility, frozen input
checksums, approved allocation seals, source/tokenizer stability and local runtime
readiness. They return metadata in memory. They neither initialize an encoder
nor download weights, train, create output directories or save run files.
Protected test identity metadata is used solely for isolation checks; `test.csv`
is not opened or hashed, and test features and labels are not inspected.

The shared path supplies the existing trainer unchanged. Run protocol,
initialization and completed-training manifests identify the selected target,
ordered vocabulary and label IDs, model identity, five input fields, token budget,
configuration and split identity. Brand/Sub-Segment metadata additionally binds
the approved allocation completion, assignments, development membership and
class vocabulary by SHA-256. Each target uses its own directory:
`<output_root>/<target slug>/<model identity>`.

## Common configuration

The pinned model remains `sentence-transformers/all-MiniLM-L6-v2`, revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, with local files only. Input order is
ProductCategory, ProductName, ProductBrand, ProductDescription, ProductContents.
The combined token budget is 256; preprocessing and character limits are unchanged.

All targets retain three fixed epochs, training/evaluation batches 32/64, learning
rate 3e-5, weight decay 0.01, dropout 0.10, warmup fraction 0.10, seed 42, FP32,
four CPU threads and device `auto`. Cross-entropy is unweighted. Validation does
not select a checkpoint. Frozen grouping remains at its original 128-token
contract; this experimental representation is not final-test certification.

## Read-only dry-run verification

All four real-data dry runs passed with the saved local tokenizer and cached
encoder hashes. Device `auto` resolved to CPU, with no readiness blockers.
Training delegation and encoder initialization were explicitly forbidden during
verification. The planned common output root
`data/processed/target_minilm_runs_v1` was not created.

| Target | Development allocation | Classes | Labeled train | Labeled validation | Planned optimizer / warmup steps |
|---|---|---:|---:|---:|---:|
| Mnfr | Frozen V4 | 2 | 55,233 | 11,812 | 5,181 / 518 |
| TargetAgeGroup | Frozen V4 | 3 | 54,282 | 11,624 | 5,091 / 509 |
| Brand | Approved Brand v1 | 143 | 58,888 | 12,619 | 5,523 / 552 |
| Sub-Segment | Approved Sub-Segment v1 r2 | 49 | 53,956 | 11,562 | 5,061 / 506 |

Preserved model identities:

- Mnfr: `minilm-mnfr-category256-c3458037a670c8f0`
- TargetAgeGroup: `minilm-targetagegroup-category256-48256ce8adfceee1`
- Brand: `minilm-brand-category256-aa0e7c04715d5006`
- Sub-Segment: `minilm-sub_segment-category256-8a1e8f3e742eb3a7`

Mnfr's configuration, ordered vocabulary, label mapping, model identity,
record/group membership hashes, class counts, encoded labels, text/token hashes,
tokenizer fingerprints and schedule match the saved 20-step benchmark. Saved
benchmark artifacts remain unchanged. No new model training, predictions,
inference, checkpoint packaging, allocation generation or merge occurred.

## Tests

All 66 focused tests passed, including eight shared-orchestration tests.
They cover all-target dry-run metadata and output isolation, strict
authorization flags, target-aware legacy manifest correction, stale allocation
and membership rejection, Mnfr compatibility and pre-generalization golden
fingerprints. Existing preparation, group-isolation, protected-test, tokenizer,
benchmark and trainer-delegation tests remain in the verification suite. Training
paths use recording doubles; no neural fine-tuning tests are executed.
