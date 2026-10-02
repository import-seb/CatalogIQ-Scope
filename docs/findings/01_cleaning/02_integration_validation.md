# Feature and target cleaner integration — local validation

Maksim Pikalov — 2026-09-30. Status: implemented and validated locally; not a
merge approval or a claim that the final modeling dataset is ready.

## Scope and changes

This run combines the feature second pass with Sebastian's reusable target
cleaner from main (`840951a`), brought into the working branch by local merge
`5509789`. Validation was run on the integration changes based on that merge;
the local `integration-20260930-v1-code-manifest.json` records the tested source
file hashes. This report accompanies those changes for review.

- Migrated reusable feature code into `src/catalogiq/`; retained the existing
  `scripts/` interfaces as compatibility wrappers after package installation.
- Added `--mode integrated` to the installed cleaning command. The default
  target-only command remains available with its historical parsing/identity.
- Assigned `(training|target, raw SHA-256, 1-based CSV record)` before processing.
  Joined target decisions by that identity, never by a filtered index or JoiningKey.
- Used a common blank/whitespace/case-insensitive-null computation policy.
  Preserved source strings on export except logged feature corrections and the
  two approved Sub-Segment mappings. No manufacturer reassignment or label fills.
- Created one candidate row mask and three complete partitions. Details and
  precedence are in the [integration contract](../../integration/cleaning_contract.md).
- Kept `MDM_InsertDateTime` unchanged after examining the actual data and source
  information; see the [timestamp audit](03_mdm_datetime_audit.md).

## ProductCategory regression

Direct calls with Python `None` previously failed on string operations. Blank
strings and the literal string `null` already passed the original feature
cleaner; they were not reproduced as failing cases. The missing-value helper
now accepts `None` as well, without inventing a category or rewriting null tokens.

Separate regression tests cover blank, spaces, tab, `None`, `null`, `Null`, and
` NULL `. Validation rejects filling a missing category or replacing one missing
token with a different token. The integration CSV test also covers empty and
literal-null categories while combining the cleaners.

The actual training export has 29 literal-null ProductCategory cells and the
prediction export has 7. Neither supplied CSV contains a blank category cell
under the whitespace-aware definition; those cases are covered synthetically.

## Full-data result

Run directory: `data/processed/integration-20260930-v1/` (ignored, local only).
Sources are the provided files with unchanged bytes:

| Dataset | SHA-256 |
| --- | --- |
| training | `00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452` |
| target | `0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db` |

| Partition | Training | Prediction target |
| --- | ---: | ---: |
| Input | 84,552 | 28,082 |
| Candidate | 78,718 | 26,033 |
| Quarantine | 466 | 171 |
| Review hold | 5,368 | 1,878 |
| Lost/duplicate source records | 0 | 0 |

Training structural overlap: 462 feature-only, 1 target-only, 3 flagged by both.
The union is 466, not the sum of both flag counts. The additional target-only
record was a feature-review case and receives target reason 5 (no numeric content).
Prediction quarantine consists of 171 feature-rule records; the target cleaner
does not assess prediction labels. These are heuristic decisions, not a measured
corruption-detection accuracy score.

Of the holds, training has 5,260 rows with a source `Exclude` marker and 108
additional field-review rows. Target has 1,842 marked rows and 36 additional
field-review rows. Among the marked rows, 43 training and 7 target rows also have
field-review reasons. These overlapping reasons are preserved, not double-counted.

Seven training label cells changed: four `Cold / Flu` to `Cold/Flu`, three
`Other Lifestyle` to `Other Lifestyle CHC`. All other label and identifier strings
remain unchanged. Full before/partition class counts are in `summary.json`.

The narrow parsing adapter and legacy pandas-NA target command quarantined the
same four training source records on these exports. This comparison is executed
in [Notebook 10](../../../notebooks/10_mdm_datetime_audit.ipynb); it is not a guarantee
that both parsing policies are equivalent on future data.

## Checks completed

- 28 synthetic tests passed, including both category missing-value cases,
  preservation of leading-zero IDs and literal `NA`, duplicate business keys,
  embedded CSV newlines, rule precedence, marker policy choices, partition/key
  mismatch rejection, repeatable outputs, and overwrite protection.
- Reran full feature processing and its separate verifier on all 112,634 records.
- Ran an additional local producer-side reconciliation of the integrated output:
  hashes, unique complete keys, disjoint partitions, candidate mask, exact
  feature-output-plus-label-log values, protected cells, summaries and every
  target-label distribution. Evidence:
  `data/processed/integration-20260930-v1-verification.json` (local).
- Ran Notebook 10 from a fresh kernel, top to bottom; validated notebook format.
- Built the package wheel and inspected it for all new package modules; the
  installed CLI help works. Windows console encoding produced logging errors
  during pip's build output, but the build returned success and the wheel archive
  passed integrity/content checks.

Runtime: Python 3.12.14, pandas 3.0.1. Tests use synthetic data; full-file results
above use the hashes shown here. No raw or derived CSV is intended for GitHub.
This is producer-side validation, not Maria's independent quality approval.

## Reproduce

Install the package as described in the root README. Use the actual local source
filenames; the hashes above also apply to the original `(1)` copies.

```bash
python -m unittest discover -s tests -v
python -m catalogiq --mode integrated --train data/provided/Selfcare_Training_data.csv --target data/provided/Selfcare_Target_data.csv --output-dir data/processed/integration-20260930-v1
python -m scripts.validate_feature_decisions --help
```

Use a new output directory if that run already exists. The integrated command
automatically runs feature verification for each dataset. Notebook 10 defaults
to the run directory above; update its `RUN` variable for another run.

## Handoff and limitations

Maria can independently check the full `row_decisions.csv`, the six partition
CSVs, `label_changes.csv`, target flags, and `summary.json`, including class
distribution changes and validity of quarantine reasons. The contract defines
the keys, missingness, decision precedence, and exact coverage to test.

Sebastian can use `keep_candidate` as the current reproducible selection proposal.
Before producing final model data, agree `Exclude` semantics and the whole-row
hold policy, review heuristic exclusions, then define per-target label eligibility
and grouping/splits. A boolean candidate mask now exists; approval is still pending.
Training IQR diagnostics use the full export, so review leakage before splitting
or fit any learned thresholds only inside the training partition. There is no
validation split or model-performance claim in this work.

No missing categories, identifiers or labels were inferred. No broad Aryx Ask
work, Raven CVE/style validation, or board updates are included in this change.
