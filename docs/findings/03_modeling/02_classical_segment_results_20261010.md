# Segment classical baseline: authoritative-split validation results

Date: 2026-10-10. Uses main `f4aad6d` (PR #19), protocol
`segment-final-20261009`, split version `segment-final-splits-v1`.

Both TF-IDF/logistic-regression variants were trained on the same 55,776 labeled
training records and scored on the same 11,952 labeled validation records.
Concatenated ProductCategory improved macro-F1 by **7.09 percentage points**.
These are validation results; no final-test predictions were produced.

## Results

| Model / inputs | Accuracy | Macro-F1 | Incorrect / 11,952 |
|---|---:|---:|---:|
| Training-selected majority class | 61.75% | 9.54% | 4,572 |
| Four agreed product fields | 89.57% | 83.90% | 1,246 |
| Four fields + concatenated ProductCategory | 94.42% | 91.00% | 667 |

Category increases accuracy by 4.84 points and reduces validation errors by 579.
Both models have 100,000 TF-IDF features; logistic regression converged in 103
and 91 iterations. The combined run took about 53 seconds locally, including
verification and exports; this is not a hardware-normalized benchmark.

| Segment | Validation support | Base F1 | Category F1 |
|---|---:|---:|---:|
| Allergy | 413 | 81.68% | 91.44% |
| CCFS | 645 | 81.62% | 91.10% |
| Digestive Health | 1,068 | 75.32% | 87.78% |
| External Analgesics | 859 | 88.18% | 95.27% |
| Internal Analgesics | 727 | 75.06% | 87.54% |
| Lifestyle CHC | 687 | 88.64% | 88.24% |
| Other Self Care | 173 | 86.77% | 89.68% |
| Vitamins, Minerals & Supplements | 7,380 | 93.95% | 96.94% |

The improvement is not uniform: Lifestyle CHC drops about 0.41 points. Maria
should inspect that class and remaining errors before final model selection.
No confidence intervals or cross-retailer holdout experiment were run here.

## Inputs and reproduction

No cleaning or grouping rules changed. Reused the verified integrated candidate:
SHA-256 `1878e3948e8a840b3922780b7304e33d39d0f1e3b5d7a579bf56639d3cdf8fb4`.
The official splitter reproduced all 83,938 assignments and protected IDs against
the committed protocol. Canonical membership digest:
`1620bba4a36f86a3f3d6f3a014cf1df6611228567fbe0e0822bff516c7f98d4a`.

| Partition | All records | Used for this model run |
|---|---:|---:|
| Train | 59,111 | 55,776 labeled records |
| Validation | 12,667 | 11,952 labeled records |
| Protected test | 12,160 | None |

Missing Segment labels (3,335 train and 715 validation) were excluded only from
supervision. Raw files and exported partitions remain unchanged.

```bash
python -m pip install -e ".[splitting]"
python -m scripts.split_data --input data/processed/cleaned-baseline-20261009/training_candidate.csv --output-dir data/processed/splits-final-baseline-20261010
python -m scripts.split_data --verify data/processed/splits-final-baseline-20261010
python -m scripts.train_segment_baseline --split-dir data/processed/splits-final-baseline-20261010 --features compare --train --output-dir data/processed/segment_classical_rerun
```

Use new output directories. Configuration is the default in the
[implementation handoff](01_classical_segment_baseline.md): C=1, no class weights,
min_df=2, word unigrams/bigrams, one text vectorizer per variant. No hyperparameter
search or calibration was performed. Existing field character limits remain,
without transformer token truncation. Category is a validation experiment;
the grouping protocol is unchanged.

Environment: Python 3.12.14, NumPy 2.3.3, pandas 2.3.2, SciPy 1.16.2,
scikit-learn 1.7.2, tokenizers 0.22.2 and joblib 1.6.0. OpenMP/OpenBLAS limits
were four threads. No PyTorch, Transformers package or neural weights were used
for the baseline; the splitter used the committed public tokenizer.

## Verified artifacts for evaluation

Local private artifacts, excluded from Git:

- `data/processed/splits-final-baseline-20261010/`: authoritative regenerated splits.
- `data/processed/segment-classical-final-20261010/`: protocol, both fitted models,
  validation predictions/probabilities, errors, confusion matrices, label mappings,
  metrics, comparison CSV and completion summary.
- `data/processed/segment-classical-final-20261010-access.json`: runtime guards
  recorded zero attempts to open/hash/read test.csv during model work.
- `data/processed/segment-classical-final-20261010-verification.json`: all prediction
  source keys and labels reconciled with sealed validation; protected IDs/groups
  absent; artifact and implementation hashes verified; probability sums, argmax
  and confidence checked; accuracy/per-class F1 independently recomputed; saved
  model predictions reproduced on 64 validation rows.

Each variant has exactly 11,952 predictions in matching record-ID order. Give
Maria both prediction files and their label mappings for paired evaluation.
No product text is included in prediction files. Probabilities are uncalibrated;
no operational confidence threshold is claimed.

The full suite passed: **401 successful, 5 skipped** (optional PyTorch tests),
406 discovered. All 16 baseline tests passed. Wheel build, dependency checks,
CLI help and whitespace checks passed. Logs:
`data/processed/baseline-final-tests-20261010.log` and
`data/processed/baseline-final-build-20261010.log`.

Maria's earlier 80.84/91.29 macro-F1 used LinearSVC and V3 membership. Do not
interpret differences from those figures as a model win/loss or platform rounding.
Compare models on the same authoritative validation IDs. Final test remains for
the agreed final evaluation; the split report's semantic-leakage caveats apply.
