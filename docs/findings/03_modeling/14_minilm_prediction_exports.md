# Shared MiniLM validation prediction exports

All five selected MiniLM checkpoints were loaded with their original saved
tokenizers and strictly matched state dictionaries. Validation inference used
the existing five-field, 256-token formatter, FP32 and evaluation batches of 64
on the RTX 3060. No optimizer, training, weight download, calibration or protected
test scoring was performed.

## Implementation

`src/catalogiq/minilm_prediction_export.py` provides the shared preparation,
inference, validation and export workflow. It reuses the frozen classifier,
text formatter, tokenization and target-specific input preparation. Selected
sources are explicit; it never discovers a checkpoint by modification time.

`src/catalogiq/maria_harness_compatibility.py` executes only the reviewed
`hash_items`, `_labels_from` and `load_predictions` definitions from Maria's
actual notebook. Its notebook bytes and loader definitions are pinned. The
unchanged definitions are also retained as an offline regression fixture.

The current harness was inspected on
[`maria/segment-evaluation-harness`](https://github.com/import-seb/CatalogIQ-Scope/blob/05e52e2e0203dee6d58dfc2d0e72d513eed9ad95/notebooks/04-segment-evaluation.ipynb),
commit `05e52e2e0203dee6d58dfc2d0e72d513eed9ad95`, notebook Git blob
`493f68ed0d2df09a291dd37d2dc1b558e6cba0d5`. The GitHub repository now resolves to
`import-seb/CatalogIQ-Scope`. The reviewed notebook is cached locally under
`.cache/maria_harness/<commit>/04-segment-evaluation.ipynb` with provenance.

## Selected models and exports

Output root: `data/processed/minilm_validation_exports_v1/`.

| Target / subdirectory | Approved allocation | Labeled validation rows | Vocabulary / validation classes | Actual loader result |
|---|---|---:|---:|---|
| Segment / `segment` | Frozen V4 | 11,952 | 8 / 8 | Compatible; no rows dropped |
| Mnfr / `mnfr` | Frozen V4 | 11,812 | 2 / 2 | Compatible; no rows dropped |
| TargetAgeGroup / `targetagegroup` | Frozen V4 | 11,624 | 3 / 3 | Compatible; no rows dropped |
| Brand / `brand` | Approved Brand v1 | 12,619 | 143 / 138 | Compatible; no rows dropped |
| Sub-Segment / `sub_segment` | Approved Sub-Segment v1 r2 | 11,562 | 49 / 47 | Compatible; no rows dropped |

Segment uses the original `segment_category_ablation_256_20261010/with_category`
checkpoint. Mnfr uses the selected unweighted model in
`mnfr_minilm_selection_v1_20261029/maria_handoff`. The other three use their
verified handoff checkpoints under
`minilm_remaining_targets_v1_20261029/handoff`. Each export manifest records
the selected checkpoint hash, model state hash, tokenizer files, model identity
and all consumed source hashes.

Every target directory contains:

- `predictions.csv`: exactly `record_id`, `true_label`, `predicted_label`,
  `confidence`, followed by numerically ordered `probability_0` through
  `probability_{K-1}`. Confidence is the uncalibrated softmax maximum.
- `manifest.json`: model/configuration, ordered classes, probability mapping,
  representation, allocation identity, ID fingerprints, source hashes and checks.
- `label_mapping.json`: the probability-column-to-class mapping Maria's loader
  already accepts. JSON uses Unicode escapes so accented labels, including
  `Horbäach`, survive its default Windows text encoding.
- `validation_context.csv`: record/group IDs, split and true target labels for
  the exact eligible validation population. Missing-label records are excluded
  from scoring; whole-partition identity is retained separately in the manifest.
- `harness_compatibility.json` and `README.txt`: actual loader/scoring results,
  scope of verification and target-context instructions.

The root also contains `export_summary.json` and `verification_report.json`.
No checkpoint copies or altered model packages were created.

## Verification results

All 59,569 target-record rows were consumed, with unique IDs within each target,
correct labels, exact approved membership and no protected record/group overlap.
All saved vocabulary columns are retained, including classes with zero validation
support. Training and validation groups remain isolated by the existing guarded
preparation code and sealed assignments.

There were zero prediction-label disagreements with each selected model's
original internal validation predictions. The largest regenerated probability
difference before CSV serialization was below `3e-8`. After CSV round trips,
the largest probability-sum deviation was `3.40e-7`, below the `1e-5` tolerance;
all probabilities and confidences were finite and within their required bounds.

All 89 distinct consumed source files were rehashed unchanged after the exports.
All 13 frozen algorithm sources matched their frozen LF-normalized hashes.
Protected population checks read only the ID/group metadata for the 12,160
reserved records. `test.csv` was neither opened nor hashed, and protected labels
were not read. Frozen artifacts, Segment code and checkpoints were not changed.

| Target | Accuracy | Macro-F1, full saved vocabulary | Macro-F1, harness observed labels |
|---|---:|---:|---:|
| Segment | 0.969461 | 0.944490 | 0.944490 |
| Mnfr | 0.999661 | 0.995729 | 0.995729 |
| TargetAgeGroup | 0.997505 | 0.987426 | 0.987426 |
| Brand | 0.966638 | 0.878344 | 0.910168 |
| Sub-Segment | 0.885833 | 0.657268 | 0.685237 |

The harness headline uses sklearn's observed-label macro-F1; the original
trainer includes the entire ordered vocabulary. Brand has five classes with no
validation support, and Sub-Segment has two. Report the metric scope explicitly;
these are the same predictions under two averaging conventions.

Tests: 49 passed, including 13 export tests plus the target-preparation,
shared-orchestration, Mnfr-orchestration and frozen text regression suites.
Checks include inference from a tiny saved state without optimization, numeric
class ordering through 143 classes, Unicode portability, wrong-split/class-order
rejection, invalid probabilities/IDs, protected-group exclusion, staging cleanup,
immutable sources and zero-support vocabulary retention. No fine-tuning tests
were executed.

## Running and consuming the workflow

From the repository root, with the existing GPU environment, use a new output
directory for each export batch. This command performs inference only:

```powershell
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.venv-gpu/Scripts/python.exe -m catalogiq.minilm_prediction_export `
  --target all --device cuda `
  --output-root data/processed/minilm_validation_exports_v2 `
  --harness-notebook .cache/maria_harness/05e52e2e0203dee6d58dfc2d0e72d513eed9ad95/04-segment-evaluation.ipynb
```

The exporter refuses existing target directories and publishes each directory
only after CSV round-trip, original-result and actual harness checks pass.
An early Brand staging export exposed the loader's Windows encoding behavior;
the class-order guard rejected it, cleanup removed it, and the Unicode-escaped
JSON fix was verified before publishing Brand. Completed earlier targets were
retained without rerunning inference.

Maria must bind a separate target context for each evaluation. The notebook's
original Segment setup and broad recursive run discovery are unsuitable for
mixing these five targets, particularly the reallocated Brand/Sub-Segment
populations. Do not disable its split guard or combine different targets in
one `RUNS` dictionary. A read-only example using the approved source data is:

```python
from pathlib import Path
from catalogiq.minilm_prediction_export import prepare_export, slug
from catalogiq.maria_harness_compatibility import load_with_maria_harness

TARGET = "Brand"
inputs = prepare_export(Path.cwd(), TARGET)
whole_ids = inputs.assignments.loc[
    inputs.assignments.split.eq("validation"), "record_id"
]
run, checks = load_with_maria_harness(
    Path("data/processed/minilm_validation_exports_v1") / slug(TARGET),
    Path(".cache/maria_harness/05e52e2e0203dee6d58dfc2d0e72d513eed9ad95/04-segment-evaluation.ipynb"),
    inputs.validation, inputs.protected.record_id, whole_ids,
)
RUNS = {"minilm_" + slug(TARGET): run}
```

The helper supplies the loader's `VA_IDS`, `GROUP_OF`, `TEST_IDS` and
`VA_ID_HASHES` from independently checked approved inputs. Export rows are sorted
by record ID; the manifest's canonical fingerprint matches that order and the
harness's sorted-ID convention. The original training-order fingerprint is
retained under `original_training_validation_record_ids_sha256`.

Compatibility verification covered the unchanged prediction loader and
headline scoring. Notebook setup, baseline fitting, calibration and report-write
cells were not executed. Brand still uses authorized retailer-supplied
`ProductBrand` for raw-to-curated normalization. No model or split changes,
retraining, protected-test evaluation or automatic merge are required.

As an additional integration check, both actual Brand and Sub-Segment exports
were presented to the unchanged loader with original V4 validation contexts.
Both were rejected by its split-fingerprint guard as expected. The results are
recorded in `verification_report.json`; correct target contexts accepted both.
