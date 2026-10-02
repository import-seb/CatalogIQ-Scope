# Cleaning integration contract

Updated 2026-09-30. Implemented locally in `catalogiq.integration`, using the
target cleaner merged into main at `840951a`. The common identity and candidate
mask are implemented; exclusion policy and model readiness still need team review.
See the [run report](../findings/01_cleaning/02_integration_validation.md).

## Source identity

Use `(dataset, source_sha256, source_row)` for every original record:

- `dataset`: exactly `training` or `target`.
- `source_sha256`: SHA-256 of the unmodified supplied CSV bytes.
- `source_row`: integer starting at **1**, excluding the header. Count parsed CSV
  records, not physical lines; quoted fields can contain newlines.
- Assign identity before sorting, filtering, cleaning, or resetting an index.
- `JoiningKey` is a provenance cross-check, not the unique join key (a duplicate
  business-key group was observed in training).

## Resolved representation differences

Initially inspected the notebook at commit `0c69a4d`; rechecked the new reusable
cleaner at `origin/clean/target_cols`, commit `b434906`, on 2026-09-29:
`src/catalogiq/cleaning.py`, `docs/data_contract.md`, and `docs/target_cleaning.md`.
The package retains the same source-key differences:

| Field | Target cleaner | Feature cleaner |
| --- | --- | --- |
| dataset | Input filename, normally `Selfcare_Training_data (1).csv` / `Selfcare_Target_data (1).csv` | `training` / `target` |
| source_row | `range(len(frame))`, 0-based | 1-based CSV record number |

The integrated command assigns the canonical identity directly to the raw frame
before calling `clean_training`. It does not merge previously filtered files or
guess their index base. Both returned partitions must together contain precisely
the original key set. Duplicate, missing, or foreign keys stop the run.
The legacy target-only command retains its old identity for compatibility.

The new target cleaner exports `training_cleaned.csv`, `training_quarantine.csv`,
`label_changes.csv`, `review_flags.csv`, and a summary. Its `target_unchanged.csv`
is explicitly a parsed pass-through, not a structural screening of the prediction
dataset. Integration must reunite both training partitions by source identity
before comparing recommendations. Target reasons 2/3 are review-only; 4/5/6
recommend quarantine under its current policy. Agree the combined policy before
claiming a final mask.

It reads strings with pandas default NA tokens, whereas the feature cleaner
recognizes only blank/whitespace and case-insensitive `null`. Do not misclassify
NA-parser changes as intentional label edits; reconcile source values and the
explicit label-change log. Shared source hashes alone do not reconcile null rules.

The integrated command uses only blank/whitespace or case-insensitive `null` as
missing for computation. Output cells come from the feature candidate CSV plus
explicit, whitelisted target changes; pandas NA serialization cannot silently
erase other source strings. The two target replacements remain `Cold / Flu` to
`Cold/Flu` and `Other Lifestyle` to `Other Lifestyle CHC` in `Sub-Segment`.

The feature implementation now lives in `src/catalogiq/`. The existing `scripts/`
commands are compatibility wrappers and require the package to be installed.
New runs use `data/processed/`; old `data/derived/` runs remain historical evidence.

## Expected inputs and checks

1. Full feature candidate and full feature decision table, one row per raw key.
2. Target-cleaning output keyed to raw records, containing only the six target
   columns it owns plus identity and a target audit. Missing labels stay missing.
   Treat `Category` as protected until its role is confirmed.
3. Structural recommendations with reasons and coverage information. A sparse
   quarantine-only file is acceptable only with a manifest proving that every
   original record was evaluated; otherwise missing keys mean unknown, not keep.

Before joining: check source hashes, dataset tokens, declared index base, unique
keys, source record coverage, and matching business identifiers where available.
Reject extra/duplicate keys or unexplained missing records. Join by key, never
by current row position. All target columns in the supplied target dataset are
intentionally empty and must not count as structural missingness.

Numeric-anchor checks must examine original feature columns only: newly added
numeric `source_row` values must not make a corrupted row appear numerically valid.
Preserve identifier strings when reading CSV; implicit numeric inference can alter
them before a comparison even starts.

## One candidate mask with explicit pending policy

Keep separate evidence columns for:

- Feature structural recommendation and its reasons.
- Sebastian's structural recommendation and reasons.
- Source `Exclude` marker decision (business policy, not structural corruption).
- Field reviews and approved cell-level remedies.
- Target-specific label eligibility.

`row_decisions.csv` contains every source key once. Status precedence:

1. `quarantine`: feature structural reasons OR target reasons 4/5/6.
2. `review_hold`: an unresolved feature field or source `Exclude` policy.
3. `candidate`: neither of the above, including harmless informative flags.

`keep_candidate` is 1 exactly for `candidate`. It is one reproducible boolean
mask across both cleaners. It is explicitly not a team-approved final training
mask. Target reasons 2/3 are review-only and do not by themselves hold a record.
An entire row is conservatively held for an unresolved feature field; the team
can later approve a field-specific policy without losing the source record.

The default `--exclude-policy hold` does not guess the marker's meaning.
`keep` or `quarantine` are explicit alternatives for a later agreed run in a new
directory. Choosing an option does not mark that policy as approved.
Quarantine takes precedence over a hold, but the audit preserves both reasons.
No files are overwritten and no record is silently deleted. A candidate may be
manually retained only with a recorded reason.
Missing labels make a row ineligible for that specific model, not automatically
invalid for every model or for inference.

Each dataset has three disjoint, complete partitions: `*_candidate.csv`,
`*_quarantine.csv`, and `*_review_hold.csv`. These contain all original columns
plus canonical identity. Quarantined/held outputs may include logged formatting
or approved label corrections; the original CSV remains the untouched authority.
`summary.json` records counts, overlap, label distributions, source/output hashes,
policies and runtime versions. It is written last as the run completion marker.
An interrupted run without it is incomplete; rerun into a new directory.

Prediction data receives feature screening only. Training label and fitted IQR
rules are not fitted to or transferred blindly to the unlabelled prediction file.

Publish aggregate before/after counts and class distributions for Maria's
independent review; source records and derived datasets remain local unless the
team has an approved data-sharing location.

## Decisions requested in this PR

- **Exclude proposal:** keep marked records in `review_hold` until the source
  owner confirms the marker's meaning. This removes them from the candidate
  training mask without deleting them or labeling them structurally corrupt.
  After confirmation, choose an explicit `keep` or `quarantine` policy in a new run.
- **Feature review proposal:** hold the whole row for unresolved feature issues
  for this conservative handoff. Resolve each reason category before promoting
  those rows; any later field-only remedy must be explicit, tested and logged.
- **Structural combination:** use the union of the existing feature and target
  quarantine rules, with every contributing reason retained. Target reasons 2/3
  remain informational for selection and do not by themselves exclude rows.

These are concrete implemented defaults offered for review, not statements of
team approval. Maria can validate their impact separately from code correctness.

## Splits after integration

Build target-specific eligibility on the finalized common row policy. Determine
product grouping from trustworthy identifiers and duplicate checks before using
any random split. Do not expand scientific UPCs to manufacture grouping keys.
Related product records must stay together. Use a fixed seed, persist membership,
record class support and rare-class handling, and flag infeasible stratification
rather than silently dropping classes. The unlabeled target file is for inference,
not a labeled holdout. Fit preprocessing and learned lookup rules on training
partitions only. Final split generation is not implemented by this feature pass.
