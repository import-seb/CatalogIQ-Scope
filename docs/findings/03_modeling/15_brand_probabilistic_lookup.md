# Probabilistic Brand normalization lookup

The new baseline estimates Brand probabilities from approved training records
only. It remains separate from the original deterministic lookup and MiniLM.
The original mapping, saved comparison, neural checkpoints and existing five
MiniLM validation exports were preserved byte-for-byte.

## Estimation fixed before validation evaluation

The approved Brand allocation supplies 58,888 labeled training records, 12,619
labeled validation records and the original ordered 143-class vocabulary.
The existing `brand_key` function supplies plain-text cleanup, Unicode NFKC,
case folding and stripping; its existing missing-value definitions are reused.

For class c, the global prior is its count divided by all labeled training
records. The 35 training records with missing ProductBrand still contribute to
this prior. For each nonmissing observed key:

    P(c | key) = (training_key_class_count + prior_c) / (training_key_count + 1)

This is a Dirichlet posterior-predictive estimate with unit prior strength.
Alpha=1 was fixed and communicated before validation evaluation; no parameter
search or validation-based adjustment took place. All supported classes have
positive training counts, so every estimated probability is strictly positive
and below one. Record frequencies, rather than independent-group frequencies,
are used; correlated records within training groups can affect reliability.

Missing and unseen values receive the entire global prior. Its argmax remains
`UnItemised brand`, with probability 0.62603586. This replaces the deterministic
fallback decision with a distribution while retaining the same fallback label.
Observed-key training-count ties can change because the prior breaks them;
saved label order breaks any remaining probability ties.

## Validation results

| Baseline | Accuracy | Macro-F1, all 143 classes | Macro-F1, validation-supported classes |
|---|---:|---:|---:|
| Original deterministic lookup | 95.6494175% | 0.93119097 | 0.96492977 |
| New probabilistic lookup | 95.6890403% | 0.93111298 | 0.96484896 |

Coverage remains 10,454 / 12,619 = 82.8433315%. Fallback covers 2,165 records:
2,163 unseen nonmissing values and two missing values. Catch-all labels remain
in the scored population.

| Cohort | Records | Deterministic accuracy | Probabilistic accuracy |
|---|---:|---:|---:|
| Seen normalized keys | 10,454 | 97.5990052% | 97.6468337% |
| Unseen nonmissing keys | 2,163 | 86.2228387% | 86.2228387% |
| Missing values | 2 | 100% | 100% |
| All fallback | 2,165 | 86.2355658% | 86.2355658% |

Seventeen predictions change from `Other Brands` to `UnItemised brand`; all
were independently verified as tied training-frequency cases resolved by the
prior. Eleven changes correct an error and six introduce an error. Other Brands
F1 falls from 0.284375 to 0.27287319; UnItemised brand F1 rises from 0.96600602
to 0.96635549. Every other class F1 remains unchanged. This improves aggregate
accuracy slightly while reducing full-vocabulary macro-F1 slightly. Neither
baseline replaces the other automatically.

## Export and harness checks

Export directory:
`data/processed/minilm_validation_exports_v1/brand_lookup_probabilistic_v1/`.

- `predictions.csv`, `manifest.json`, `label_mapping.json`: Maria's exact
  handoff schema, all probability columns 0–142, original class ordering,
  confidence equal to the argmax probability and complete identity metadata.
- `model.json`: sparse training key/class counts, global counts, priors,
  normalization identity and fixed smoothing specification.
- `per_class_metrics.csv`, `confusion_matrix.csv`, `evaluation.json`: precision,
  recall, F1 and support for all 143 classes, 143-by-143 confusion matrix,
  full/cohort comparisons and both macro-F1 conventions.
- `prediction_differences.csv`: all 17 changed decisions and correctness flags.
- `validation_context.csv`, `harness_compatibility.json`,
  `immutability_verification.json`, `README.md`: verified population, actual
  loader/calibration checks, preservation evidence and reproduction commands.

Every ordered record ID and true label matches the existing Brand MiniLM export.
The agreed 8-byte big-endian length-prefixed UTF-8 SHA-256 fingerprint is
`87cd25d15e37be3487efafa06f26b5a13908513f760377cb8b233327c02a1ce5`.
Maria's unchanged loader retains all 12,619 rows; protected record and product
group overlap is zero. Group isolation and the sealed approved allocation pass.
The cached notebook was checked against the current remote harness Git blob
`493f68ed0d2df09a291dd37d2dc1b558e6cba0d5` before implementation.

Maria's exact ECE, Brier and temperature-scaling definitions also execute with
finite results and unchanged argmaxes. Raw ECE is 0.06871579 and raw multiclass
Brier is 0.08694824. Probability estimates show underconfidence, particularly
for fallback records: confidence 62.60% versus accuracy 86.24%. They are not
calibrated. The diagnostic temperature fit uses the same validation records
as its diagnostic score; it is not independent calibration evidence. No scaled
probabilities were exported and no estimation parameter was changed.

All newly generated text artifacts use LF. CPU frequency estimation took
0.389965 s and batch inference 0.253627 s, excluding source loading, scoring and
file I/O according to the manifest's scopes. Amortized inference cost is
0.020099 ms/record; single-request latency is unmeasured and explicitly null.
CPU/platform/Python/library versions and source hashes are recorded. No GPU
was used for the lookup.

Reloading model.json reproduces exported probabilities within 1.12e-16.
Maximum probability-sum error is below 7.55e-15. All 38 consumed source files
were rehashed unchanged, all 34 pre-existing MiniLM export/audit files were
unchanged and all 13 frozen algorithm hashes matched. No protected-test labels
or test.csv were accessed; no neural models were trained or run.

## Implementation and reproduction

New module: `src/catalogiq/brand_probabilistic_lookup.py`. It reuses guarded
Brand preparation and the original normalizer, missing-value rules, metric
functions, export validators and actual Maria loader. It does not refit the
original deterministic estimator. A pinned fixture retains Maria's unchanged
calibration definitions for offline regression checks.

From the repository root, using a fresh output directory:

```powershell
$env:PYTHONPATH = 'src'
.venv-gpu/Scripts/python.exe -m catalogiq.brand_probabilistic_lookup `
  --output-directory data/processed/minilm_validation_exports_v1/brand_lookup_probabilistic_v2
```

Tests: 66 passed, including ten new probabilistic lookup tests, four original
lookup regression tests and 52 preparation/export/orchestration/text checks.
They cover training-only estimation, ambiguous keys, rare classes, unseen and
missing values, normalization, label ordering, protected groups, exact harness
loading/calibration, LF output, source preservation and saved-model reproduction.
No merge was performed. Maria must bind the approved Brand context and compare
the distinct model identities; the original Segment/V4 context is unsuitable.
