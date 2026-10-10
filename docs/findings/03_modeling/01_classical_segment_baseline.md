# Classical Segment baseline: implementation and handoff

Updated 2026-10-10 against main `f4aad6d` (portable split workflow, PR #19).

The classical Segment pipeline uses word TF-IDF and logistic regression. It
supports a controlled validation comparison of four product fields with and
without concatenated ProductCategory. The [first real-run results](02_classical_segment_results_20261010.md)
show validation macro-F1 of 83.90% / 91.00%. Generated private artifacts stay
under `data/processed/`; synthetic checks are separate from model-quality evidence.

## Run on the authoritative shared splits

Install `python -m pip install -e ".[splitting]"` and reproduce the shared splits
using the [portable workflow](../02_splitting/03_portable_workflow_20261010.md).
Cleaning rules are unchanged by PR #19; an existing verified candidate with hash
`1878e3948e8a840b3922780b7304e33d39d0f1e3b5d7a579bf56639d3cdf8fb4`
can be reused.

```bash
python -m scripts.split_data --input data/processed/cleaned/training_candidate.csv --output-dir data/processed/splits
python -m scripts.split_data --verify data/processed/splits
python -m scripts.train_segment_baseline --split-dir data/processed/splits --features compare --output-dir data/processed/segment_classical_check
python -m scripts.train_segment_baseline --split-dir data/processed/splits --features compare --train --output-dir data/processed/segment_classical_run
```

Use new output directories. The baseline command defaults to preparation only;
`--train` explicitly fits and evaluates validation. The installed
`catalogiq-segment-baseline` command has the same arguments. `--features base`
or `--features category` runs one variant; `compare` runs both on identical rows.

The loader calls the team's `load_frozen_development` guard, which verifies
authoritative membership, seals, exposure selectors and record/group isolation.
It then attaches only the required provenance columns and optional category from
sealed train/validation exports. It never opens or hashes `test.csv` or reads
test features/labels. No grouping, seal, test reservation or assignment is changed.

The shared splitter needs the packaged public tokenizer to reproduce its groups.
The TF-IDF classifier does not need a Hugging Face tokenizer, neural weights, GPU
or Transformers package. Its own representation uses ordinary word ngrams.

## Model and experiment

- Target: Segment, missing targets excluded from supervision. Class order is
  learned from train and saved with probabilities. Unseen validation classes stop
  fitting; the official loader separately validates the declared Segment labels.
- Base fields: ProductName, ProductBrand, ProductDescription, ProductContents.
  Uses the existing HTML/whitespace normalization, field markers and character
  limits of 1024/512/4000/2000 respectively, without transformer token truncation.
- Category arm: append ` [ProductCategory] ` and the category to the same text,
  normalizing whitespace around `>` and treating blank/null categories as empty.
  One TF-IDF vectorizer is used; there is no separate category block.
- Labels, source IDs, retailer, ratings, review counts, timestamps and other audit
  columns cannot enter either model's text.
- TF-IDF: lowercase word unigrams/bigrams, sublinear TF, min_df=2,
  max_features=100000, L2 normalization. Vocabulary and IDF fit only on train.
- Logistic regression: C=1, lbfgs, max_iter=1000, no class weighting, seed=42.
  Nonconvergence fails the run. No hyperparameter search is performed.
- Primary metric: macro-F1; also accuracy, weighted-F1, macro recall, per-class
  precision/recall/F1/support, confusion counts, log loss and summed multiclass
  Brier score. The majority comparator is selected on train, evaluated on the
  same validation rows.
- Probabilities are uncalibrated. No production acceptance or review threshold
  has been selected.

Category inclusion is a validation experiment, not a deployed feature decision.
The grouping protocol remains fixed. The word representation differs from the
transformer's token arrays, so zero frozen-input crossings is not proof of zero
collisions or semantic leakage in every alternative representation.

Maria's reported research used LinearSVC with balanced class weights, an 80,000
term cap and the earlier V3 assignments. Her 80.84/91.29 macro-F1 figures are not
directly comparable to this model on the corrected final partitions. Our two
arms isolate category within the same model/settings/rows. Comparing against
the transformer requires its results on these same final validation records.

Optional `--config baseline.json`:

```json
{"c": 1.0, "max_iter": 1000, "min_df": 2, "max_features": 100000,
 "ngram_max": 2, "class_weight": null, "seed": 42}
```

## Outputs for Maria

Preparation writes `protocol.json` and `summary.json` marked `prepared`.
A trained comparison additionally writes:

| Artifact | Contents |
|---|---|
| `base/`, `category/` | Per-variant models and validation artifacts |
| `predictions.csv` | Source keys, record_id, group_id, split, true/predicted labels, confidence, class probabilities |
| `label_mapping.json` | Exact label for every probability column |
| `validation_errors.csv` | Incorrect validation predictions |
| `confusion_matrix.csv`, `metrics.json` | Confusion counts and model/majority scores |
| `model.joblib` | Fitted pipeline, class order, settings and category flag |
| `comparison.csv` | Validation accuracy, macro-F1 and weighted-F1 by variant |
| `summary.json` | Completion, counts, paired percentage-point differences and artifact hashes |

The saved model consumes strings generated by
`build_baseline_texts(frame, include_category=saved["include_category"])`.
Source keys and record/group IDs support joining predictions for evaluation.
Model inputs, source hashes, row/text hashes, package versions and settings are
recorded. A failed run does not receive a completed summary.

## Verification and historical preparation

Sixteen synthetic baseline tests check training-only vocabulary/IDF, validation
labels not influencing fitting, persistence, missing categories, category in one
vectorizer, source-key errors, probability exports and nonconvergence. The official
path is tested through the real portable verifier on a synthetic protocol while
file-access guards prohibit test.csv access and network/HF model use. A changed
sealed export fails before fitting. Full-suite and actual-run evidence are retained
beside the ignored outputs; synthetic scores do not measure CatalogIQ quality.

The earlier V3 preparation (assignment hash starting `c2f0e766`, 55,443 labeled
train and 11,881 validation records) is historical. It trained no model. PR #19
resolves that ambiguity: only the newly generated authoritative assignments are
used for continued development. The CLI retains `--input` and explicit assignment
hashes for historical preparation, but rejects `--train` on that path.
