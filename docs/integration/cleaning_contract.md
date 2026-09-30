# Proposed cleaning integration contract

Date: 2026-09-28; branch update checked 2026-09-29. Status: proposal;
source key agreed in principle, exact
representation and final exclusion policy still require team alignment.

## Source identity

Use `(dataset, source_sha256, source_row)` for every original record:

- `dataset`: exactly `training` or `target`.
- `source_sha256`: SHA-256 of the unmodified supplied CSV bytes.
- `source_row`: integer starting at **1**, excluding the header. Count parsed CSV
  records, not physical lines; quoted fields can contain newlines.
- Assign identity before sorting, filtering, cleaning, or resetting an index.
- `JoiningKey` is a provenance cross-check, not the unique join key (a duplicate
  business-key group was observed in training).

## Observed mismatch to resolve

Initially inspected the notebook at commit `0c69a4d`; rechecked the new reusable
cleaner at `origin/clean/target_cols`, commit `b434906`, on 2026-09-29:
`src/catalogiq/cleaning.py`, `docs/data_contract.md`, and `docs/target_cleaning.md`.
The package retains the same source-key differences:

| Field | Target cleaner | Feature cleaner |
| --- | --- | --- |
| dataset | Input filename, normally `Selfcare_Training_data (1).csv` / `Selfcare_Target_data (1).csv` | `training` / `target` |
| source_row | `range(len(frame))`, 0-based | 1-based CSV record number |

Proposed adapter: explicit dataset-name mapping and `source_row + 1` for that
cleaner's original 0-based index. Apply only to an export declared as 0-based;
do not infer the base from a filtered file's first row or apply the conversion
twice. Verify both original source hashes first. Filenames alone are insufficient.

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

The new branch also introduces `src/catalogiq/` and `data/processed/` conventions.
This PR retains its existing standalone standard-library `scripts/` entry points
and ignored `data/derived/` runs. Moving the feature pass into the package belongs
to integration after that branch is incorporated; neither cleaner has been run
as part of the other yet.

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

## Combining decisions

Keep these separate until the policy is approved:

- Feature structural recommendation and its reasons.
- Sebastian's structural recommendation and reasons.
- Source `Exclude` marker decision (business policy, not structural corruption).
- Field reviews and approved cell-level remedies.
- Target-specific label eligibility.

Review disagreements and overlap counts, then write one full-row decision table
with explicit `keep`/`quarantine` outcomes, combined reasons, source identity and
policy version. Never mark a final mask approved while any required row decision
is unresolved. A candidate may be manually retained only with a recorded reason.
Missing labels make a row ineligible for that specific model, not automatically
invalid for every model or for inference.

Publish aggregate before/after counts and class distributions for Maria's
independent review; source records and derived datasets remain local unless the
team has an approved data-sharing location.

## Splits after integration

Build target-specific eligibility on the finalized common row policy. Determine
product grouping from trustworthy identifiers and duplicate checks before using
any random split. Do not expand scientific UPCs to manufacture grouping keys.
Related product records must stay together. Use a fixed seed, persist membership,
record class support and rare-class handling, and flag infeasible stratification
rather than silently dropping classes. The unlabeled target file is for inference,
not a labeled holdout. Fit preprocessing and learned lookup rules on training
partitions only. Final split generation is not implemented by this feature pass.
