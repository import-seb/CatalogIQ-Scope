# Dataset-wide structural screening

Current policy: structural-v6 (2026-10-03). The checker reads original training
and target CSVs independently of feature and target cleaning. Integrated mode
now consumes its verified decisions; standalone mode still changes no rows.
See [decisions](decisions.md) and [integration](integration/cleaning_contract.md).

```bash
python -m catalogiq --mode structural --output-dir data/processed/structural_run
python -m scripts.validate_structural --output-dir data/processed/structural_run
```

Use --train/--target for other paths and --config for JSON policy overrides.
Existing output directories cannot be overwritten. Unknown configuration fields
fail explicitly; the retired severe_missingness_min_count is not accepted.

## Row rules

| Quarantine reason | Required evidence |
| --- | --- |
| missing_product_text | ProductName, ProductDescription and ProductContents are all missing, regardless of category, brand or measurements. |
| csv_field_count | Record width differs from the source header. |
| text_in_multiple_numeric_anchors | Nonnumeric text in at least two measurement fields. |
| text_numeric_failure_with_corroboration | Nonnumeric text in one measurement plus another invalid measurement or displaced identifier/invalid URL field. |
| url_in_timestamp_with_displacement | HTTP(S) URL in timestamp plus numeric text or another displaced identifier/invalid URL field. Timestamp alone produces no finding. |
| unexpected_exclude_with_populated_unknown_columns | Unexpected non-marker Exclude content together with populated Sun1-Sun5/Notes. |
| multiple_suspicious_targets | At least two targets contain URLs or fail the working manufacturer/age vocabulary. |
| suspicious_target_with_displacement | A suspicious target plus numeric text or displacement evidence. |
| missingness_with_unusual_text_length | High feature missingness AND mean product-text length strictly outside its IQR bounds. |

Numeric anchors are ProductRating, ProductReviewsCount, ReviewsCount and XRatXRev.
Measurements must be finite and nonnegative; count fields must be integral.
Missing anchors and isolated invalid numeric values remain diagnostics, not
whole-row quarantine. A number in an ID or timestamp is not a measurement.

Working target vocabularies are Mnfr={All others, J&J} and
TargetAgeGroup={Adult, Children, Infant}. Missing targets do not count as failed
vocabulary checks. Case/spacing warnings and rare labels alone do not quarantine.
These working vocabularies are assumptions, not a confirmed general schema.

The missing-product-text rule is a completeness policy, not proof of a shift.
Any one of name/description/contents being present avoids that particular rule.
It deliberately cannot be satisfied by category, brand, a URL or a numeric rating.

## Adaptive profiles

Missingness uses the explicit configured product-feature scope: retailer,
category, brand, name, the four numeric anchors, product/image URLs, description
and contents. Missing means trimmed blank or case-insensitive null. Statistical
comparisons use the missing proportion (equivalently count for a fixed scope).

Text length uses the configured subset of product category, brand, name,
description and contents. It is the average trimmed character length of present
fields. Labels, timestamps and IDs cannot enter either profile. Nonnumeric text
in a measurement column does not change the text scope.

For each source separately, Q1/Q3 use linear quantiles over full-width records.
The default multiplier is the existing 1.5 IQR, configurable separately for
missingness and text length. Quarantine requires missingness strictly ABOVE its
upper fence and mean text length strictly OUTSIDE its lower/upper fences. Low
missingness is not evidence of damage. Equality at a fence is retained.
Each signal alone is a warning. All-empty text has undefined length and is
excluded from the length profile; missing_product_text still covers empty rows.
No fixed 11/12 cutoff remains.

## Interface and provenance

assess_row accepts original string values, canonical identity, StructuralConfig,
missingness_upper_count and text_length_bounds. None for the length bounds means
there was no observed product text to fit the profile. It returns quarantine,
sorted reason_codes, sorted finding_codes and per-finding raw-value evidence.
Disabling a quarantine rule keeps its evidence as a finding.

All decisions use (dataset, source_sha256, source_row), where dataset is training
or target and source_row is one-based parsed data-record position. Header and
empty physical lines are excluded; multiline quoted cells count once. File SHA
and record position distinguish duplicate business IDs. Legacy filename/zero-based
artifacts require explicit conversion, never a guessed positional join.

The structural engine imports neither cleaner. Integration reads its complete
decision tables, verifies hashes and key coverage, and retains structural reason
prefixes alongside Max's feature and target reasons. The optional compatibility
validator still compares audits without changing either pass's decisions.

## Artifacts and limits

- *_decisions.csv: one decision per source record, including clean rows.
- *_evidence.jsonl: affected rows with full ordered raw columns/cells and findings.
- summary.json: source hashes, scopes, fitted bounds, counts by dataset/reason/
  decision, artifact hashes, versions and limitations. Written last.

Standalone screening neither edits nor removes rows. Integrated exports apply
only the already approved corrections and preserve original sources for audit.
Structural keep is not a guarantee of semantic correctness or model eligibility.
IQR may be weak for small, homogeneous or broadly corrupt sources. A negative
lower length fence cannot detect short positive strings. Same-type column shifts
can escape these checks. Ambiguous CSV syntax aborts; wrong-width CSV records
can be audited standalone but the feature cleaner rejects them in integration.
