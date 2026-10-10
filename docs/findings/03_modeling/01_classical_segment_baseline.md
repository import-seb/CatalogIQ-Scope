# Classical Segment baseline: implementation and handoff

Updated 2026-10-10, based on main revision `5486ae6`.

Status: word TF-IDF pipeline and ProductCategory comparison implemented.
Synthetic model checks and a real-data preparation run are complete. No real
CatalogIQ model was trained or scored; there is no new accuracy/F1 result.

## Model and feature comparison

- Target: Segment. Sorted class names are learned from training labels, with a
  saved probability-column mapping. Validation classes absent from train stop
  the run. Missing Segment values are excluded from supervision.
- Base text: ProductName, ProductBrand, ProductDescription, ProductContents.
  Reuses the shared builder's HTML/whitespace handling, field markers and
  character limits (1024, 512, 4000, 2000 respectively).
- Category experiment: append ` [ProductCategory] ` and the category to that same
  text. Normalize whitespace around `>`; blank/null categories are empty.
  There is one TF-IDF vectorizer, not a separately weighted category block.
- No Hugging Face tokenization or 128-token truncation. Labels, source keys,
  retailer, ratings, review counts, timestamps and audit fields are excluded
  from model text. No transformer weights or GPU are required.
- Word TF-IDF unigrams/bigrams, lowercase, sublinear TF, min_df=2, at most
  100,000 terms, L2 normalization; vocabulary and IDF fit on train only.
- Logistic regression, C=1, lbfgs, max_iter=1000, no class weighting, seed=42.
  Nonconvergence fails the run. Both variants use identical settings and rows,
  so their difference measures the effect of category within this pipeline.
- Primary metric: macro-F1. Also save accuracy, weighted-F1, macro recall,
  per-class precision/recall/F1/support, confusion counts, log loss and summed
  multiclass Brier score. Compare with a majority class chosen on train.
- Saved confidence values are uncalibrated probabilities. No acceptance or
  human-review threshold has been selected.

This replaces the initial local HF-token-ID baseline. Ordinary TF-IDF does not
need a transformer tokenizer. Maria's research uses LinearSVC, an 80,000-term
cap and balanced class weights; this logistic-regression comparison is not an
exact replication of her reported 80.84 / 91.29 validation macro-F1 scores.
Software versions, row order, text hashes, inputs and settings are recorded.
Cross-platform differences should be investigated rather than attributed to a
guaranteed floating-point tolerance.

## Shared assignments and current preparation

The loader uses an explicit cleaned candidate and assignment CSV, not a random
split or a local reallocation. It checks:

1. Full expected assignment SHA-256, and optionally the expected candidate hash.
2. Unique source keys and canonical record IDs; exact one-to-one coverage of
   dataset + source_sha256 + source_row, regardless of CSV row order.
3. Valid split names, nonempty group IDs and group isolation across all splits.
4. Training/validation label availability and unchanged file hashes after loading.

It ignores Segment values in the assignment file. Reading the combined candidate
CSV necessarily parses all records, including test records; test rows are then
discarded before label normalization, text construction, fitting or scoring.
The separate test export is never opened. Prediction exports contain validation
records only. These are membership/integrity checks, not proof that a holdout
has never been used historically.

The locally reproduced rule-v3 assignment file has SHA-256:

`c2f0e76684630b0394511d77de7331ccb86b6aa5d032f9c4bcb9606854b08b9d`

Its first eight characters match the prefix Maria supplied. She has not supplied
a full hash for comparison. The cleaned candidate hash is:

`1878e3948e8a840b3922780b7304e33d39d0f1e3b5d7a579bf56639d3cdf8fb4`

Preparation verified all 83,938 candidate rows:

| Partition | Assigned rows | Labeled development rows | Missing labels omitted |
|---|---:|---:|---:|
| Train | 58,756 | 55,443 | 3,313 |
| Validation | 12,591 | 11,881 | 710 |
| Test | 12,591 | Not processed for supervision | Not calculated |

Local evidence:

- `data/processed/maria-split-check-20261010.json`
- `data/processed/segment-classical-preflight-20261010/protocol.json`
- `data/processed/segment-classical-preflight-20261010/summary.json`

The current README requires the newer protected final snapshot; the chat/research
commands refer to the older rule-v3 exports. Before actual fitting, confirm with
the split owner which assignment version and development protocol this comparison
should use. Do not train against the older assignments as a workaround: their
training membership can differ from the newer test reservation. No frozen seals,
split assignments or exposure records were changed here. The preparation report
explicitly records that it does not verify the final seal or exposure history.

## Run

Install the normal package: `python -m pip install -e .`.

Prepare the selected inputs, without training:

```bash
python -m scripts.train_segment_baseline --input path/to/training_candidate.csv --assignments path/to/agreed_assignments.csv --assignments-sha256 FULL_64_CHARACTER_SHA256 --features compare --output-dir data/processed/segment_classical_check
```

Paths and the expected hash must identify the team-agreed files. Add
`--input-sha256 FULL_64_CHARACTER_SHA256` to pin the candidate's exact contents.
The installed `catalogiq-segment-baseline` command is equivalent.

For an actual run after selecting the shared protocol, add `--train` and use a
new output directory. `--features compare` runs both variants; `base` or
`category` runs just that variant. There is no test-scoring option.
Existing run directories are never overwritten. A failed fit does not receive
a completed summary.

Optional `--config baseline.json`:

```json
{"c": 1.0, "max_iter": 1000, "min_df": 2, "max_features": 100000,
 "ngram_max": 2, "class_weight": null, "seed": 42}
```

## Outputs for Maria

Preparation writes only `protocol.json` and a summary marked `prepared`.
A trained comparison additionally produces:

| Artifact | Contents |
|---|---|
| `base/`, `category/` | Each variant's model and validation artifacts |
| `predictions.csv` in each variant | Source keys, record_id, group_id, split, true/predicted label, confidence and probability columns |
| `label_mapping.json` | Exact class name corresponding to every probability column |
| `validation_errors.csv` | Incorrect validation predictions |
| `confusion_matrix.csv`, `metrics.json` | Confusion counts, baseline metrics and majority comparator |
| `model.joblib` | Fitted pipeline, class order, configuration and category-mode flag |
| `comparison.csv` | Side-by-side validation accuracy, macro-F1 and weighted-F1 |
| `summary.json` | Completion status, counts, category-minus-base percentage-point changes and artifact hashes |

The saved pipeline consumes strings from
`build_baseline_texts(frame, include_category=saved["include_category"])`.
Use that adapter on future product records; no tokenizer directory is involved.
Source-key columns make prediction joins auditable. Private rows, predictions
and trained artifacts remain ignored under `data/processed/`.

## Verification

Synthetic checks cover train-only vocabulary/IDF, no validation-label influence
on the fitted model, one-vectorizer category inclusion, missing categories,
metadata exclusion, source-key reordering, hash mismatches, missing/duplicate
rows, cross-split group rejection, missing labels, validation-only exports,
model persistence, no HF/network dependency, no test export access, and failure
on nonconvergence. They do not measure real CatalogIQ model quality.

On Python 3.12.14 / scikit-learn 1.9.1, all 13 baseline tests pass. The full
suite discovers 388 tests: 383 pass and five existing optional PyTorch tests
are skipped. The wheel builds, CLI help, dependency and whitespace checks pass.
The real-data preparation above fits no model and scores no partition.
Full-suite results are in `data/processed/baseline-tests-20261010.log`; build
output is in `data/processed/baseline-build-20261010.log`.

Next: settle the shared assignment/protocol version, run both variants, then
give Maria the validation predictions and comparison. Category inclusion remains
an experiment, not a deployed model decision.
