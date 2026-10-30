# Classical multi-target development, October 29, 2026

The common trainer supports all six target names. Eight real models were fitted:
four-field and category variants for Segment, Mnfr, TargetAgeGroup and Platform.
Brand and Sub-Segment were profiled but not fitted because Sebastian's announced
replacement splits were not published in the available remote branches.

These are validation experiments on the existing `segment-final-20261009`
allocation, not new target-balanced splits, final-test evidence or model selection.
The workflow and artifact contract are documented in
[Classical multi-target workflow](../../classical_multitarget.md).

## Results

Macro-F1 is the primary metric. Each target has its own labeled validation view;
missing labels are excluded only for that target. Majority-class baselines and
per-class results are retained in each run's `metrics.json`.

| Target | Variant | Validation rows | Accuracy % | Macro-F1 % | Fit seconds | Warm batch ms/record |
|---|---|---:|---:|---:|---:|---:|
| Segment | four fields | 11,952 | 89.57 | 83.90 | 19.066 | 0.116 |
| Segment | + category | 11,952 | 94.42 | 91.00 | 18.478 | 0.135 |
| Mnfr | four fields | 11,812 | 99.19 | 87.16 | 7.869 | 0.110 |
| Mnfr | + category | 11,812 | 99.05 | 84.37 | 8.447 | 0.121 |
| TargetAgeGroup | four fields | 11,624 | 97.93 | 80.78 | 9.400 | 0.115 |
| TargetAgeGroup | + category | 11,624 | 97.73 | 77.34 | 9.528 | 0.131 |
| Platform | four fields | 247 | 63.16 | 37.61 | 1.000 | 0.089 |
| Platform | + category | 247 | 60.73 | 34.70 | 0.946 | 0.099 |

Segment exactly reproduces the earlier headline metrics. Category improves
Segment by 7.09 percentage points of macro-F1 but lowers Mnfr by 2.79 points,
TargetAgeGroup by 3.43 points and Platform by 2.92 points. It must not be promoted
for every target based on the Segment result. Both variants are retained for
Maria's comparison; no new global feature policy was selected.

Platform has only 797 labeled training rows across 57 classes. Two validation
rows have classes absent from training; they remain in the denominator and count
as errors. Full-set log loss and multiclass Brier are null, and its manifest
requires an explicit unseen-label policy before calibration. This limited result
does not establish deployable Platform quality or reverse the decision to omit
Platform from the neural model. Macro-F1 uses the evaluator's sklearn default
observed-label scope, also recorded explicitly in the metrics.

## Pending split dependency

The last remote check still showed main `35ec970`, `feat/model_splits` `8bbab31`,
and `feat/transformer_model` `c666867`. No announced replacement Brand or
Sub-Segment protocol was present.

| Target on existing allocation | Labeled train | Labeled validation | Train classes | Validation-only classes / rows |
|---|---:|---:|---:|---:|
| Brand | 58,882 | 12,625 | 142 | 1 / 1 |
| Sub-Segment | 53,967 | 11,551 | 48 | 1 / 2 |

Do not move those exceptional rows between partitions or create a separate random
split. Once Sebastian supplies the approved protocol, verify/reconcile it, then
run these targets with the same trainer and artifact format. A different split
format needs its own integration with the verifier before training.

## Timing and environment

CPU run on Windows 11, Intel64 Family 6 Model 151 Stepping 2, 24 logical CPUs,
with numerical libraries limited to two threads. Python 3.12.14, numpy 2.3.3,
pandas 2.3.2, scipy 1.16.2 and scikit-learn 1.7.2. Logistic regression retains
`C=1`, `max_iter=1000`, no class weighting and seed 42. Word TF-IDF uses up to
100,000 features, unigrams/bigrams, min_df=2 and sublinear TF.

Fit time includes TF-IDF and classifier fitting; formatting training text is
separate. Latency is the median of three warm batches after one excluded warm-up,
up to 256 rows per batch (247 for Platform), including text formatting,
vectorization, prediction and constructing the output frame. The displayed
per-record value is amortized batch time, not single-request latency or p95.
Compare with MiniLM only after matching hardware, batching and measurement scope.
The old October 10 combined wall time remains unchanged in historical outputs.

## Verification and private handoff

- Initial full suite: 468 tests discovered, 463 successful and 5 optional PyTorch skips.
- Verified all eight real prediction files against their labeled validation
  identities, source keys, true labels, class order and probability sums.
- Recomputed accuracy and macro-F1; loaded all eight with Maria's exact loader
  functions from `05e52e2`, with target-specific split hash checks passing.
- Reloaded every saved model and reproduced predictions for 300 validation rows
  (247 for Platform), without true labels in the inference input.
- Runtime guards recorded zero final-test/combined-candidate access and zero
  network access during training; verification recorded no model refitting.
- This checks the loader and output contract, not a full rerun of Maria's notebook
  or a new MiniLM comparison. Probabilities remain uncalibrated.

Private outputs:

- `data/processed/classical-preflight-20261029/summary.json`: all six target profiles.
- `data/processed/classical-multitarget-20261029/`: eight fitted models, predictions,
  per-arm manifests, measured cost, `access_audit.json` and `verification.json`.
- `data/processed/max-classical-multitarget-20261029.zip`: private handoff with
  models and evaluation files; 40,771,574 bytes, SHA-256
  `6b7048f367db3d08bad0dcd6d8c2e3ebb885545f34082e0cafd7992954ccd6ca`.

The archive belongs in the private team channel, not Git. Recipient-side model
loading uses the trusted `model.joblib` and `predict_classical` adapter from this
change. The archive has eight model directories; it does not contain Brand or
Sub-Segment models, raw product descriptions or final-test rows.

## Independent follow-up review

Two independent reviews checked the split/training code and the private handoff.
Archive checksums, probability mappings, target-specific validation identities and
reported accuracy/macro-F1 matched. The existing archive was not changed.

The code review found an inference-adapter bug for a reordered mapping: probability
values followed numeric class order but their column names followed dictionary
insertion order. The adapter now uses numeric order for both. A 12-class regression
with JSON-sorted keys failed before the fix and passes after it; all eight classical
tests pass. The reviewer confirmed the fix and regression.

The original models were saved in the correct order, so their predictions and
metrics were unaffected. After the fix, every saved model was reloaded with a
deliberately reordered mapping and matched its existing predictions on 300
validation rows (247 for Platform). This compatibility check did not refit models
or read reserved-test data. The archive's source fingerprints still identify its
original training source at `02ef28d`; they were not rewritten to claim training
with the later adapter fix.
