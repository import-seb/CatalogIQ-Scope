# Current integrated cleaning policy validation

2026-10-03. Implements [the decision list](../../decisions.md), including
quarantine (not hold) when name, description and contents are all missing.
Policies: integrated-review-v4, feature-decisions-v3, structural-v6,
target-labels-v2. Group agreement is recorded for Exclude keep and timestamp
exclusion; other policies are Sebastian's explicit implementation decisions.

## Sources and result

Training source SHA-256:
`00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452`

Target source SHA-256:
`0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db`

| Dataset | Input | Candidate | Quarantine | Review hold |
| --- | ---: | ---: | ---: | ---: |
| Training | 84,552 | 83,938 | 614 | 0 |
| Target | 28,082 | 27,870 | 212 | 0 |

Each input record belongs to exactly one partition, under the canonical
(dataset, source_sha256, one-based source_row) identity. All input hashes,
original values, permitted corrections and partition/export values were checked
by the independent integrated verifier. The source files were not changed.

## Rules and overlap

| Structural quarantine reason | Training | Target |
| --- | ---: | ---: |
| missing_product_text | 151 | 42 |
| text_in_multiple_numeric_anchors | 305 | 109 |
| text_numeric_failure_with_corroboration | 157 | 60 |
| url_in_timestamp_with_displacement | 263 | 93 |
| unexpected_exclude_with_populated_unknown_columns | 3 | 2 |
| multiple_suspicious_targets | 3 | 0 |
| missingness_with_unusual_text_length | 0 | 0 |
| suspicious_target_with_displacement | 0 | 0 |
| csv_field_count | 0 | 0 |

Reason counts overlap. Feature and structural passes both quarantine 465
training rows and 171 target rows. Structural alone adds 149 training rows and
41 target rows. Three training rows also have target reason 6; no row is added
solely by target quarantine in this run. Reason prefixes preserve that overlap
instead of silently losing the source of a decision.

Training source row 48936 is quarantined for structural:missing_product_text.
A valid rating, category or brand cannot rescue all three missing descriptive
fields. Timestamp-only anomalies and bad URLs alone do not hold the row.
The IQR joint rule adds no quarantines on these exports; single statistical
outliers remain visible as diagnostics, not automatic exclusions.

## Difference from the earlier conservative handoff

| Transition | Training | Target |
| --- | ---: | ---: |
| Previously cleaned, still candidate | 78,718 | 26,033 |
| Previously held, now candidate | 5,220 | 1,837 |
| Previously held, now quarantined | 148 | 41 |
| Previously quarantined, still quarantined | 466 | 171 |

No previously cleaned row becomes quarantined. The restored rows reflect the
agreed Exclude keep policy and removal of blanket field-review holds. Historical
handoff files are unchanged; they no longer represent the current selection.

## Artifacts and checks

Current run:
`data/processed/integrated_policy_v4_final_20261003/`

- training_candidate.csv / target_candidate.csv: retained records with labels,
  original-schema audit fields, permitted corrections and source keys.
- training_features.csv / target_features.csv: 12 permitted product fields plus
  three join-only provenance keys; no timestamp, Sun1-Sun5, Notes, labels or IDs.
- Complete quarantine and review-hold partitions preserve audit columns.
- row_decisions.csv and the three pass audits preserve field warnings and reasons.
- summary.json includes all reason/decision counts and manifests.
- validation.json records independent full-file reconciliation.
- handoff_comparison.json records every historical-to-current status transition.

Category-level features are deferred. ProductCategory and Retailer remain.
Seven approved label corrections were recorded (four slash-spacing, three
Other Lifestyle replacements); no inferred labels or sentinel values were added.

Tests cover policy outcomes for both datasets, strict IQR boundaries, sparse
sources, identity integrity, raw-value preservation, projection contents,
repeatability and rehashed output tampering. Package build is checked without
private data. See reproducibility.json in the run directory for full-run repeat
hash comparisons and the recorded test/build results.

## Limits and follow-up

Candidate rows may retain flagged bad fields, including URLs or isolated numeric
errors. No guessed repair or downstream field treatment is implied. Profiles are
feature-only and per-source, but small or contaminated populations can distort
IQR baselines. Working manufacturer/age vocabularies and timestamp semantics
remain schema questions; no time-based modeling or date conversion is implied.
Wrong-width records can be audited standalone, while the current feature pass
aborts them in integrated mode. Model eligibility, splits and category feature
engineering remain separate work.
