# Classical Segment baseline: implementation and handoff

Date: 2026-10-09. Based on main revision `5486ae6` and the
[final split protocol](../02_splitting/02_finalization_20261009.md).

**Status: implemented; synthetic validation and local cleaning validation only.**
The shared private frozen snapshot and its tokenizer are not present in this
checkout. No real-data model has been trained or scored in this work. There is
no CatalogIQ accuracy/F1 result yet. The final test has not been opened.

## Model and comparison protocol

- First target: `Segment`, using the same eight declared classes as the
  transformer's guarded development loader.
- Product inputs: `ProductName`, `ProductBrand`, `ProductDescription`, and
  `ProductContents`. Other target columns, category, IDs, retailer, timestamps,
  Exclude and audit metadata cannot enter the model.
- Reuse the existing HTML/whitespace handling, field limits, saved tokenizer and
  sequence limit from the sealed model configuration (128 tokens in the adopted
  snapshot). No pretrained neural encoder is loaded or fitted.
- Convert the frozen token IDs into opaque terms, then use TF-IDF unigrams and
  bigrams, sublinear term frequency, minimum document frequency 2, at most
  100,000 features, and L2 normalization. Vocabulary and IDF fit on train only.
- Classifier: logistic regression, C=1, lbfgs, maximum 1,000 iterations, no class
  weighting by default. Fail on nonconvergence rather than reporting a finished
  run. Settings may be changed explicitly in a recorded baseline configuration.
- Primary metric: macro-F1. Also export weighted-F1, macro recall, accuracy,
  per-class precision/recall/F1/support, confusion counts, log loss and summed
  multiclass Brier score (mean over records; not divided by number of classes).
- Compare against the training-derived most-frequent class on exactly the same
  validation records. Missing Segment labels are excluded from supervision by
  the existing loader; original exports are not modified.
- Probability outputs are **uncalibrated**. No operational confidence threshold
  or automatic review routing is claimed. Maria can use the saved probabilities
  for calibration, uncertainty and error analysis.

This is a matched-input classical baseline, not an ordinary full-document word
TF-IDF run: it inherits the transformer's truncation. Its learned bag of ngrams
also discards order beyond the configured ngram width. Frozen product/effective
input groups do not prove an absence of all semantic leakage or collisions in
the fitted sparse representation. The split report's residual risks still apply.
Changing input fields, limits or tokenizer requires a separately agreed protocol;
this command will not silently rebuild or replace the held-out test.

The sparse text approach follows the
[scikit-learn text classification example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).
The project's [evaluation proposal](../../model-evaluation-strategy%20(1).md)
motivates macro-F1, a majority comparator and per-class reporting. Its proposed
category lookup and future feature engineering are not enabled by this change:
the current agreed input contract contains only the four fields above.

## Required private handoff

Use the **existing** `split_finalization_20261009/frozen` snapshot. The older
`python -m scripts.split_data` command in the chat produces rule-v3 research
splits and must not replace it.

Obtain the sealed directory, the effective-input cache and every referenced
verification artifact, plus the corresponding saved tokenizer and
`model_config.json`, from the split owner through the team's private channel.
The current upstream verifier embeds absolute paths in `source_sha256`,
`input_sha256`, and `model_input_cache`; a copied folder may therefore need an
owner-supported portable handoff. Merely copying train/validation CSVs is not
enough. Do not edit checksums, bypass verification, or reserve a new test to get
past that limitation. The baseline reuses the existing verifier unchanged.

```bash
python -m pip install -e ".[baseline]"
python -m scripts.train_segment_baseline --split-dir data/processed/split_finalization_20261009/frozen --tokenizer-dir data/processed/segment_full_development_20261009/random/tokenizer --model-config data/processed/segment_full_development_20261009/random/model_config.json --output-dir data/processed/segment_classical_01
```

The installed `catalogiq-segment-baseline` command has the same arguments.
PyTorch/GPU/model weights are not required. Tokenization is offline. The new
output directory must be outside the immutable inputs. Missing inputs cause an
actionable error before tokenization or output creation; there is no random-split
or unsealed-input fallback. No CLI option scores test.

Optional `--config baseline.json` accepts only these classifier/vectorizer
settings (not model input fields or split settings):

```json
{"c": 1.0, "max_iter": 1000, "min_df": 2, "max_features": 100000,
 "ngram_max": 2, "class_weight": null, "seed": 42}
```

## Outputs for evaluation

| File | Purpose |
|---|---|
| `protocol.json` | Frozen split seal, input/class/configuration hashes, source hashes and package versions |
| `predictions.csv` | Validation `record_id`, `group_id`, `split`, `true_label`, `predicted_label`, `confidence`, `probability_0` through `probability_7` |
| `label_mapping.json` | Exact class name for every probability column; same order as the transformer |
| `validation_errors.csv` | Incorrect validation predictions, with the same columns |
| `confusion_matrix.csv` | True classes as rows, predicted classes as columns, raw counts |
| `metrics.json` | Baseline and majority comparator on the same records |
| `model.joblib` | Fitted TF-IDF/classifier pipeline, class order and input configuration |
| `summary.json` | Completion marker, metrics, counts, timing and artifact hashes |

Join predictions to shared validation rows by `record_id`, retaining `group_id`
for group-aware uncertainty estimates. All generated outputs and private data
stay ignored under `data/processed/`. The saved pipeline consumes the opaque
token documents produced by `token_documents`; it must not be fed raw text.
Keep the frozen tokenizer/configuration with the experiment for later inference.

## Local verification and remaining work

New offline tests exercise train-only vocabulary/IDF, class ordering, probability
exports, persistence, deterministic reruns, metadata exclusion, overlap and
tampering rejection, missing snapshots, and a complete synthetic run through the
existing seal verifier without opening `test.csv`. A real saved-tokenizer test
runs when the baseline extra is installed; it blocks network connections.
Synthetic metrics are not evidence of CatalogIQ model quality.

Executed on Windows with Python 3.12.14, NumPy 2.3.5, pandas 3.0.1,
scikit-learn 1.9.1 and Transformers 5.19.0: **380 tests passed, 5 skipped**
(385 discovered). All 10 new baseline tests passed; the five skips are existing
optional PyTorch transformer-training tests. The wheel build, CLI help,
`pip check`, and diff whitespace check passed. Logs are saved locally as
`data/processed/baseline-tests-20261009.log` and
`data/processed/baseline-build-20261009.log`.

The current integrated cleaner was rerun on both original private files, then
independently reconciled with `catalogiq.integration_validation`:

| Dataset | Input rows | Candidate | Quarantine | Review hold |
|---|---:|---:|---:|---:|
| Training | 84,552 | 83,938 | 614 | 0 |
| Target | 28,082 | 27,870 | 212 | 0 |

All 112,634 source records, provenance keys, protected values, corrections,
partitions, audit decisions and export hashes passed reconciliation. Source
`Exclude` uses the agreed `keep` policy. Seven audited label corrections remain.
The training candidate count matches the documented frozen population; that
count alone does not establish byte-for-byte snapshot equality or permission
to construct a replacement split.

Local artifacts: `data/processed/cleaned-baseline-20261009/` and sibling
`cleaned-baseline-20261009-verification.json`. These are cleaning results, not a
model run. Remaining work is obtaining/verifying the shared private snapshot,
running the first real baseline, and giving Maria the validation artifacts.
No new agreed feature-engineering experiments were found in the repository;
those remain pending the team's decisions.
