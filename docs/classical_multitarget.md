# Classical models for the six classification targets

The Segment baseline now has a reusable training workflow for `Segment`,
`Sub-Segment`, `Brand`, `Mnfr`, `Platform`, and `TargetAgeGroup`. It retains
word TF-IDF plus logistic regression, train-only vocabulary/IDF, the four
allowlisted product fields, and the optional ProductCategory text concatenation.
No target, retailer, identifier or timestamp becomes a model feature.

Each target gets its own missing-label mask. A row missing Segment may still be
usable for another target. Labels are never filled in or inferred during training.
This is one classifier per target, not a joint multi-output model.

## Shared splits and the current dependency

Only an explicit, verified portable split directory is accepted. The existing
authoritative verifier checks identities, grouping, frozen membership and hashes.
The trainer reads `train.csv` and `validation.csv`; `test.csv` is neither opened
nor hashed. Protected-test evidence is restricted to IDs and group membership.
The new workflow does not allocate or reshuffle any rows.

At the October 29 check, the remote branches still contain the October 9 Segment
allocation. Sebastian announced replacement splits for Brand and Sub-Segment;
those models must wait for the agreed exports before final comparison. Prepare
their code and profile now, but do not present results from the superseded
allocation as their new benchmark. `--protocol-dir` selects a compatible approved
protocol bundle; an unregistered split format requires integration with its own
verifier, not disabling checks or fabricating a completion manifest.

The existing allocation can support explicitly scoped development experiments on
the other targets. `split_allocation_target` records that it was designed for
Segment; this does not claim a new target-balanced allocation or new test approval.

```bash
python -m scripts.train_classical --split-dir data/processed/splits --targets Segment Sub-Segment Brand Mnfr Platform TargetAgeGroup --output-dir data/processed/classical_preflight
python -m scripts.train_classical --split-dir data/processed/splits --targets Segment Mnfr TargetAgeGroup --features compare --train --output-dir data/processed/classical_run
```

Always select a new output directory. Without `--train`, no model is fitted.
The default configuration matches the previous baseline, including
`class_weight=null`; configurations can be supplied with `--config`.
Numerical libraries use two threads by default (`--threads` changes this).

## Unseen validation classes and sparse targets

The default refuses a target with validation classes absent from training. It
never silently drops the rows or adds validation-only classes to the fitted
classifier. For an explicitly limited experiment, `--allow-unseen-validation`
keeps them in validation and counts their predictions as errors.

Primary macro-F1 follows sklearn's default union of observed validation true and
predicted labels, matching Maria's evaluator. A secondary score covers the union
of all training classes and validation labels. The label scopes are saved in
`metrics.json`; classes absent from validation must not silently be interpreted
as measured failures or successes.

If unseen labels exist, full-set log loss and multiclass Brier are `null`, and
`calibration_requires_unseen_label_policy` is true. The evaluator must explicitly
handle this case before calibration: it cannot treat the unknown true label as
an all-zero one-hot vector or silently discard it. This especially matters for
sparse Platform. A completed experiment is not evidence that Platform is ready
for deployment or that it should be included in Sebastian's neural model.

## Prediction files and model reuse

Every `<target>/<base|category>/` directory contains:

- `predictions.csv`: source key, record/group IDs, `split`, `true_label`,
  `predicted_label`, `confidence`, and `probability_0..n-1`.
- `manifest.json`: target, explicit `label_mapping` keyed by probability column,
  protocol identity, training and validation ID hashes, counts, performance,
  source fingerprints, package versions and artifact checksums.
- `metrics.json`, `validation_errors.csv`, and `confusion_matrix.csv`.
- `model.joblib`: fitted vectorizer/classifier plus target and adapter metadata.

The manifest sits next to predictions. `validation_record_ids_sha256` uses sorted
IDs from that target's labeled validation rows, UTF-8, prefixed by each encoded
ID's unsigned 8-byte big-endian length. Other targets therefore need not have
Segment's `47f60a59...` hash. The evaluator must recompute its target-specific
mask, not compare every target to the Segment hash.

These names and mapping follow the team's October 27 agreement. This document
describes this implementation and does not replace Maria's shared contract doc.
Probabilities remain uncalibrated; Segment's temperature is not applied to other
targets. No operating threshold or production model is selected here.

```python
import joblib
from catalogiq.classical_baseline import predict_classical

# Load trusted team artifacts only (joblib uses pickle).
bundle = joblib.load("data/processed/classical_run/Mnfr/category/model.joblib")
predictions = predict_classical(bundle, products)
```

`products` requires a unique `record_id`, the four product fields, and
ProductCategory if the fitted variant uses it. It does not need a true label.
Prediction formatting is shared with training so Maria does not have to refit
or reproduce a vectorizer independently.

## Cost and provenance

`training_seconds` measures each arm's `pipeline.fit`, including vectorizer fit
and classifier fit. Training text formatting is recorded separately. Validation
prediction wall time includes formatting and output construction.

Inference latency uses a warm, in-memory batch of up to 256 validation records,
one excluded warm-up and three timed repeats. The manifest records all times,
median batch time, amortized time per record, throughput, CPU description,
logical CPU count, numerical thread limit, library versions and measurement
boundaries. This is not a single-request p95 or cold model startup benchmark.
Compare models on the same machine, batch size and measurement scope; two
different computers do not give a controlled latency comparison.

New source and output text files use LF. Source fingerprints contain both raw
byte hashes and CRLF-to-LF normalized hashes, with an explicit normalization
description. Previously generated files and their fingerprints remain intact.
The original October 10 run's 53.38 seconds is a combined wall time and is not
retroactively relabeled as per-model training cost.

## Private handoff

```bash
python -m scripts.package_classical --run-dir data/processed/classical_run --output-zip data/processed/classical_handoff.zip --include-models
```

The packager refuses incomplete runs and modified artifacts. The ZIP includes
explicitly selected run files, a root manifest with checksums and a readme.
`--include-models` supplies the fitted models as requested for integration;
otherwise it includes prediction/evaluation files only. Keep these outputs in
the team's private channel, outside Git. An existing archive is never replaced.
