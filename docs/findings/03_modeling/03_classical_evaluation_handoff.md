# Verified classical baseline handoff

The classical run now has a repeatable packaging step for Maria's evaluation.
It checks saved results rather than retraining or introducing another evaluator.
Run from the repository root with a completed two-arm baseline run and the same
authoritative portable splits:

```bash
python -m scripts.package_segment_baseline --run-dir data/processed/segment_classical_run --split-dir data/processed/splits --output-dir data/processed/classical_evaluation_handoff
```

Use a new output directory outside the run and split directories. The command
writes `evaluation.zip` and `summary.json`. A failed validation creates no output
directory. Existing runs, model files and split exports are never overwritten.
An interrupted file write may leave an incomplete directory without a completed
summary; choose another output directory after resolving the write failure.

## What is checked

- Both `base` and `category` completed under the same authoritative V4 protocol.
- The saved protocol's development export hashes and ordered membership match the
  sealed training/validation files through the existing frozen-development guard.
- Every prediction has exactly one matching labeled validation record, with the
  same source key, group, split and true Segment. Duplicate, missing, foreign and
  non-validation records fail. Joins use identity rather than row position.
- Probability columns follow the verified class mapping; finite probabilities
  lie in [0, 1], sum to one, and agree with predicted labels and confidence.
- Accuracy, macro-F1 and weighted-F1 are recomputed from the saved predictions and
  checked against metrics, comparison CSV and completion summary. Error rows and
  confusion counts must also agree. Other reported measures are retained from
  the original run and are not separately audited by this packaging step.
- Every selected source artifact matches its completed-run checksum. Checksums
  detect corruption; they are not signed proof of who produced the files.

The command uses no model weights, GPU, network or training. The guard reads
training and validation plus protected-test identity metadata; it never opens
`test.csv` or reads final-test labels/features. No split or model setting changes.

## What to send

Send `evaluation.zip` through the team's private channel. Do not commit it to Git.
It contains both arms' `predictions.csv`, `label_mapping.json`, `metrics.json`,
`validation_errors.csv`, `confusion_matrix.csv`, plus `comparison.csv`, a scoped
`manifest.json` and `README.txt`. It excludes model binaries, product descriptions,
training/test CSVs and the original protocol's workstation paths.

The receiver should check `manifest.json` for the protocol ID, validation record
count and file checksums, then join predictions on `record_id` or on
`dataset + source_sha256 + source_row`. Each class mapping specifies the order of
`probability_0` through `probability_7`; no positional joins are needed. This is
the existing baseline prediction schema, not a claim of compatibility with an
unpublished evaluation API.

The ZIP uses stable entry order and timestamps, so the same verified inputs on
the same Python/zlib implementation produce identical bytes. Compression output
across different zlib versions is not guaranteed to be identical. The outer
summary records the archive hash and checks performed.

Probabilities remain uncalibrated. This does not set an acceptance threshold,
select a production model, replace Maria's evaluation, or score the reserved test.
The existing [October 10 results](02_classical_segment_results_20261010.md) remain
the model-quality evidence; packaging is not a new modeling experiment.

## Local verification on October 16, 2026

The command packaged the original completed October 10 run against the unchanged
`segment-final-20261009` exports. Both arms reconciled all **11,952** validation
records. Recomputed macro-F1 stayed **83.90% / 91.00%**, with accuracy
**89.57% / 94.42%**. The ZIP is 3,896,076 bytes with SHA-256
`774614848d2388dd0fcb292f3056963dbce0bd02e041643ac48eb4bcdb829edb`.

The private output is `data/processed/segment-baseline-handoff-20261016/`.
Its runtime guard recorded zero test-file access and zero model load/fit
attempts. Tests verify deterministic packaging, prohibited test/model/network
access, invalid memberships, corrupted probabilities, reversed class mappings,
inconsistent reports, failed runs, and preservation of existing outputs.
The combined suite completed **456 successful tests and 5 optional PyTorch skips**
(461 discovered); package build passed. Logs remain under `data/processed/`
as `handoff-tests-20261016.log` and `handoff-build-20261016.log`.
