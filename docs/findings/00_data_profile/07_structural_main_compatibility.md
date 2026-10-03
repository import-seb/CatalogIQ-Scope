# Structural check after Max's main integration

Validated 2026-10-02 on `feat/structural-check` with Python 3.13.11 and pandas 3.0.0.
Fetched and merged `origin/main` at `b01e607` (PR #8, implementation `505d309`)
in local merge `c0b30e8`, preserving the branch's existing sample-script commit.
The previous uncommitted structural implementation was restored and adapted to
the merged package. The feature and integration implementations were inspected
alongside the updated contract and timestamp audit before adapting the check.

## Compatibility changes

- Policy `structural-v2` uses `dataset = training|target` and one-based parsed
  `source_row`, matching Max's feature and integrated audit keys. Raw SHA-256,
  filename and path remain in the source manifest. Source rows stay unchanged.
- CSV record numbering now follows the merged `csv.DictReader` convention:
  empty physical lines are skipped, whitespace-only and quoted-empty records
  count, and quoted multiline cells count once. Parseable wrong-width records
  still receive structural quarantine with full raw evidence. Max's feature
  pass rejects such records, so compatibility verification requires a completed
  feature run rather than guessing keys after a parse failure.
- The shared CLI supports `--mode structural` with the same `--config`, source
  paths and outputs as `python -m catalogiq.check_structure`. Existing target and
  integrated modes retain their behavior. Structural config and Exclude-policy
  options are rejected when passed to the wrong mode.
- `verify_feature_compatibility` validates both complete audits and compares their
  source-keyed decisions. The CLI accepts completed training/target feature runs,
  including the feature subdirectories of an existing integrated run. It rejects
  wrong roles, source hashes, incomplete coverage and invalid keys; it never
  changes either run or produces a combined decision mask.
- `structural-v1` artifacts are historical. The current validator rejects their
  manifests; do not silently reinterpret filename/zero-based keys. Re-run from
  original sources, or explicitly convert only after verifying record conventions.

The structural engine remains independent of both cleaners. No structural result
is fed into Max's `integration.row_decision`. Full-data validation ran each feature
pass separately, retaining all rows, and compared its audit with structural
decisions. The integrated pipeline was exercised on synthetic test data only.
No new combined mask or full-data integrated partition was generated here.

## Full-source results

| Dataset | Verified records | Structural keep | Structural quarantine | Feature quarantine |
| --- | ---: | ---: | ---: | ---: |
| Training | 84,552 | 84,099 | 453 | 465 |
| Target | 28,082 | 27,913 | 169 | 171 |

| Keyed comparison | Training | Target |
| --- | ---: | ---: |
| Both recommend quarantine | 453 | 169 |
| Structural only | 0 | 0 |
| Feature only | 12 | 2 |
| Neither recommends quarantine | 84,087 | 27,911 |

These are comparison counts, not a union or an applied mask. All 112,634 keys,
raw evidence and output hashes reconcile. Every source record has exactly one
structural decision. The independent structural validator and Max's feature
verifier both pass. Neither source was modified.

| Structural quarantine reason | Training | Target |
| --- | ---: | ---: |
| `csv_field_count` | 0 | 0 |
| `text_in_multiple_numeric_anchors` | 305 | 109 |
| `text_numeric_failure_with_corroboration` | 145 | 58 |
| `unexpected_exclude_with_populated_unknown_columns` | 3 | 2 |
| `multiple_suspicious_targets` | 3 | 0 |
| `suspicious_target_with_displacement` | 0 | 0 |

Reasons overlap: the three multi-target training records also trigger metadata
spill. All per-rule, finding and decision counts match the original
[v1 validation tables](07_structural_validation.md#findings-by-dataset-and-decision).
The saved v2 summaries include the complete counts by finding and keep/quarantine,
configuration and limitations. Missingness still uses 12 feature columns, Q1 = 0,
Q3 = 2 and a strict upper fence of 5 in both datasets; it never independently
quarantines a record.

Source hashes are unchanged:

- Training: `00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452`
- Target: `0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db`

## Tests and repeatability

**43 tests passed**, covering both merged cleaners, their existing integration and
15 structural tests. New compatibility checks exercise actual feature/integrated
outputs, duplicate business keys, multiline cells, literal `NA`, matching canonical
identities, all four agreement/disagreement outcomes, wrong-role/source rejection,
read-only comparisons and both structural CLI entry points with configuration.
Malformed-record numbering is checked against the same CSV convention used by
Max's package. Existing target/feature tests remain unchanged.

The shared and standalone structural commands produced **byte-identical copies
of all five artifacts**, including `summary.json`. Both full runs passed the
independent validator. Package wheel construction also passed.

Local ignored run directories:

- `data/processed/structural_v2_main`
- `data/processed/structural_v2_main_repeat`
- `data/processed/structural_v2_feature_training`
- `data/processed/structural_v2_feature_target`

Reproduce from the repository root in the project environment, using new output
directories if these already exist:

```bash
python -m unittest discover -s tests -v
python -m catalogiq --mode structural --output-dir data/processed/structural_v2_main
python -m catalogiq.check_structure --output-dir data/processed/structural_v2_main_repeat
python -m scripts.feature_decisions --input "data/provided/Selfcare_Training_data (1).csv" --output data/processed/structural_v2_feature_training --dataset training
python -m scripts.feature_decisions --input "data/provided/Selfcare_Target_data (1).csv" --output data/processed/structural_v2_feature_target --dataset target
python -m scripts.validate_structural --output-dir data/processed/structural_v2_main --feature-training-run data/processed/structural_v2_feature_training --feature-target-run data/processed/structural_v2_feature_target
```

## Remaining policy decisions

The 12/2 feature-only recommendations use a timestamp-format warning as the only
corroboration of numeric text. The structural pass keeps these as review findings
because timestamp encoding remains unconfirmed in Max's
[new timestamp audit](../01_cleaning/03_mdm_datetime_audit.md). Both policies and
their evidence remain visible; this compatibility change does not decide the
team's final exclusion policy.

Confirm timestamp/schema assumptions, target vocabularies, and Exclude/unknown
metadata semantics before applying decisions to model inputs. Missingness and
isolated field findings remain review-only. Same-type column shifts may still be
missed, and no measured precision/recall is claimed. The shared identity contract
is now implemented; business-policy approval, final masks and models remain
separate work.
