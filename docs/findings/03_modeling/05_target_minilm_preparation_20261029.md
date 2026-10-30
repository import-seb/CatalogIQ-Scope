# Target-aware MiniLM input preparation

Implemented October 29, 2026. This document covers the input-preparation API.
All four supported targets now share a dry-run-first orchestration layer,
described in [the orchestration note](08_target_minilm_orchestration_20261029.md).
Input preparation performs no training, downloads, inference, checkpoint
packaging or prediction exports.
Platform is excluded from MiniLM. Existing Segment code and frozen V4 artifacts
remain unchanged.

## API and representation

`catalogiq.target_minilm_preparation.prepare_target_minilm` accepts the frozen
split directory, selected target and a future output root. Brand and Sub-Segment
also require the approved allocation directory and its expected completion
SHA-256. The explicit digest pins the approved membership and saved class order;
the allocation's existing pending-review header does not need to be edited.

The prepared object retains every development row, raw selected labels, frozen
group IDs, original roles and target-specific roles. Target-missing rows are
marked ineligible for supervision. `supervised_frame` filters those rows;
`legacy_frame` copies only record ID, the selected labels renamed to `Segment`,
and the five product fields. The source frame and files are never renamed or
modified.

Mnfr and TargetAgeGroup retain V4 roles and use deterministic development-label
ordering, or an explicitly supplied order over that same vocabulary. Brand and
Sub-Segment retain allocation CSV row order and the exact `class_coverage.csv`
label order. Every vocabulary must have training coverage. Protected test IDs
and groups are checked without reading or hashing `test.csv`.

`tokenize_target_minilm` calls the existing category experiment tokenizer and
unchanged Segment formatter. It preserves ProductCategory-first text, the same
HTML/whitespace cleanup and character limits (category 512, name 1,024, brand
512, description 4,000, contents 2,000), and one combined 256-token budget.
The original grouping contract remains four fields at 128 tokens. The
experimental representation is not a new final-test certification.

The default tokenizer loader reads only the saved local tokenizer with
`local_files_only=True`; it never requests encoder weights. Tests can inject an
offline tokenizer. Preparation itself does not load a tokenizer or require
PyTorch/Transformers. Explicit real tokenization requires the existing optional
Transformers dependency and saved tokenizer files.

Each prepared target gets a model identity derived from its target, ordered
vocabulary, representation, configuration, allocation seal and ordered supervised
membership. Its planned output directory is `<output_root>/<target>/<identity>`.
No directories or run artifacts are created by preparation or tokenization.

## Approved allocation usage

```python
from catalogiq.target_minilm_preparation import prepare_target_minilm, tokenize_target_minilm

prepared = prepare_target_minilm(
    "data/processed/split_integration_reproduced_20261010",
    "Brand",
    "data/processed/minilm_targets",
    allocation_directory="data/processed/brand_development_allocation_v1_20261029",
    allocation_completion_sha256="94c64e5bd80219595f3c8764acd67a5d64496a63164276807d42b14d56624c6e",
)
# Optional, local-tokenizer-only input preparation; this does not train a model.
corpus = tokenize_target_minilm(prepared)
arguments = prepared.trainer_arguments(corpus)
```

Sub-Segment uses `data/processed/sub_segment_development_allocation_v1_20261029_r2`
and completion SHA-256
`26f52bd5360af96d0271c998f13c846dfeaa325bf0fb90b4c85011df5cabdc6a`.
Mnfr and TargetAgeGroup take neither allocation argument.

## Validation and remaining training work

All 43 focused tests passed, including 12 new adapter tests and existing Segment
text, frozen-loader and target-preparation tests. They check selected labels,
ordered encoding, feature allowlisting, source immutability, group isolation,
protected test handling, approval seals, local-only tokenizer loading and binding
prepared arguments to the existing trainer signature. Neural initialization is
explicitly forbidden in the adapter tests; no fine-tuning tests were run.

Read-only preparation against the real inputs produced:

| Target | Classes | Labeled train | Labeled validation |
|---|---:|---:|---:|
| Mnfr | 2 | 55,233 | 11,812 |
| TargetAgeGroup | 3 | 54,282 | 11,624 |
| Brand | 143 | 58,888 | 12,619 |
| Sub-Segment | 49 | 53,956 | 11,562 |

Every target retained all 71,778 development rows and protected all 12,160 test
records. Real-data verification did not tokenize, initialize a model or create
output directories. Frozen algorithm source hashes still match their manifest.

`trainer_arguments` prepares data only. `run_target_minilm` handles the selected
target, model identity and five-field metadata around the unchanged trainer;
its default is a non-training dry run. `run_mnfr_minilm` remains a compatibility
wrapper. Actual training requires both `dry_run=False` and
`training_authorized=True`, following explicit authorization.
Inference, checkpoint packaging and new prediction export workflows remain out
of scope.
