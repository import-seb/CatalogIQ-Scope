# Dataset-wide structural check

Run independently on the **original, complete** training and prediction CSVs:

```bash
python -m catalogiq.check_structure --output-dir data/processed/structural_run
python -m scripts.validate_structural --output-dir data/processed/structural_run
```

`catalogiq-check-structure` is the installed equivalent. `--train` and `--target`
override source paths. Use a new output directory outside the source directories;
existing runs and source files cannot be overwritten. No row is removed, repaired,
filled, or replaced with a sentinel. A quarantine decision means hold for structural
review, not proven corruption or an approved modeling mask.

After Max's PR #8 was merged into main (`b01e607`), the shared CLI also supports
`python -m catalogiq --mode structural --output-dir data/processed/structural_run`.
It produces the same artifacts as the standalone command. `--config` accepts the
same policy JSON through either entry point; `--exclude-policy` belongs only to
Max's `--mode integrated`, not this independent check.

The implementation is `catalogiq.structural`. It imports neither cleaner and does
not consume cleaner quarantine decisions. The existing target cleaner's behavior
is unchanged. Its reasons 4/5/6 are historical target-cleaning decisions, not aliases
for this check's rules. No model, common mask, or merged dataset is produced.

## Source identity and ingestion

Every row decision and every individual finding carries
`(dataset, source_sha256, source_row)`:

- `dataset` is exactly **`training` or `target`**, matching the feature and
  integrated cleaners. The source filename and path are retained in the summary.
- `source_sha256` is SHA-256 of the original file bytes, including encoding and
  line endings. Changing the source produces a different identity namespace.
- `source_row` is a **one-based parsed CSV record position**, excluding the
  header and empty physical lines. Quoted multiline values count once. Quoted
  empty and whitespace-only records count as records, matching `csv.DictReader`
  in Max's code; wrong-width records are quarantined here. Identity is assigned before any
  profiling or decisions, never from an output index or a business identifier.

This is the merged [integration contract](integration/cleaning_contract.md).
Tests compare structural keys with actual feature and integrated runs, including
multiline records and duplicate business keys. Sequential positions are unique
even when `JoiningKey` or the entire product row repeats. Passing the same path
for both roles is rejected. Records from different roles have separate namespaces.
Runs screen each record exactly once and check fingerprints before and after processing.

`structural-v2` changes the identity/record contract; the row decision rules remain
the same as `structural-v1`. Earlier structural runs and the legacy target-only
loader used filenames and **zero-based** positions, with pandas-style skipping
of whitespace-only physical lines. Those artifacts must not be silently joined
to canonical outputs. Re-run from source for compatibility. For a verified source
without whitespace-only records, an explicit filename-to-role mapping plus
`source_row + 1` converts v1/legacy keys; otherwise re-enumerate the raw source.
The current validator rejects v1 manifests instead of guessing their convention.

Strings are read literally with UTF-8/BOM support. For diagnostics only, surrounding
whitespace is ignored and blank or case-insensitive `null` means missing. This is
the merged feature/integrated policy, **not** the legacy target cleaner's pandas default NA
parsing. For example, literal `NA` stays text and `NaN` is a nonfinite measurement;
both remain available in raw evidence. Do not join on parsed cell equality or
interpret NA-parser differences as changes made by this check.

The default schema requires the configured measurement/missingness fields, six
targets, `Exclude`, `Sun1`–`Sun5`, `Notes`, both URL fields, `MDM_Id`, and
`MDM_InsertDateTime`. Blank target cells are allowed. Other columns are retained in
raw evidence; identifiers are never cast to numbers. Duplicate/missing headers,
reserved cleaner/provenance headers, invalid encoding, and unparseable CSV syntax
abort without a completion summary. A parseable record with the wrong field count
gets its own `csv_field_count` decision and full ordered cells; it is not silently
padded, truncated, or skipped. The target cleaner may fail or parse differently on
such malformed input, so cross-cleaner joins require coverage checks first.

## Numeric-field meaning and sufficiency

The working schema comes from
[the existing profiling guidance](findings/00_data_profile/02_cleaning_direction.md)
and [PR #6](https://github.com/import-seb/CatalogIQ-Scope/pull/6), initially reviewed at head
`879a3261e2ad201ec4cbe0ac093ffcf2017f744f` while open and unmerged. Its feature code
is now in the package, incorporated through Max's PR #8. The original PR found
description fragments in these measurement columns and reviewed 25 structural
candidates. Its results support corroborated review rules, not a measured error
rate or a sponsor-approved schema.

| Field | Working meaning/check | Is nonnumeric text alone sufficient to quarantine? |
| --- | --- | --- |
| `ProductRating` | Finite, nonnegative rating. Fractional values are legitimate; no assumed maximum rating scale. | No. Could be one scrape/field error. |
| `ProductReviewsCount` | Finite, nonnegative whole review count. Decimal/scientific strings representing whole numbers are accepted. | No. A malformed count alone does not locate a row shift. |
| `ReviewsCount` | Finite, nonnegative whole count. No assumed equality with `ProductReviewsCount`. | No, for the same reason. |
| `XRatXRev` | Finite, nonnegative measurement; its name suggests a derived rating/review measure. No product formula or integer constraint is enforced without schema confirmation. | No. The derivation is unconfirmed and one bad field is insufficient. |

Nonnumeric, nonfinite, negative and fractional-count values produce separate
`numeric_*` findings. Missing measurements do not count as failures. Multiple
negative/fractional/nonfinite measurements alone still do not establish textual
displacement. Two or more **distinct fields**, with at least one containing text,
are required by the corroboration rules below. The two-field requirement follows
the PR's observed patterns and expresses independent field evidence, rather than
a tuned statistical cutoff. Related count fields can still share an upstream
error: quarantine remains a review decision.

`Sku`, `Upc`, `JoiningKey`, `ProductModelNumber`, and `MDM_Id` are identifiers,
not numeric anchors. Alphanumeric IDs and scientific-notation UPCs are permitted.
An HTTP(S) URL in `MDM_Id` is a displacement finding. Nonnumeric timestamps are
review-only because their encoding is not established; a valid ISO date must not
corroborate quarantine just because it is not an Excel serial number.
Max's [timestamp audit](findings/01_cleaning/03_mdm_datetime_audit.md) retains that
encoding uncertainty. His feature pass still counts its timestamp warning as
corroboration; the independent structural decision does not. The difference is
preserved for review rather than silently changing either cleaner's policy.

## Explicit decision rules

| Quarantine reason | Evidence required |
| --- | --- |
| `csv_field_count` | Parsed cell count differs from the unique source header's width. Strong direct structural evidence. |
| `text_in_multiple_numeric_anchors` | Nonnumeric text in at least two distinct measurement fields. |
| `text_numeric_failure_with_corroboration` | Text in one measurement plus another invalid measurement, a URL in `MDM_Id`, or an invalid populated product/image URL. Does not also emit the preceding rule. |
| `unexpected_exclude_with_populated_unknown_columns` | Populated `Exclude` other than the literal marker `Exclude`, together with populated `Sun1`–`Sun5` or `Notes`. Observed description fragments spilling into metadata in PR #6; schema remains provisional. |
| `multiple_suspicious_targets` | At least two distinct targets contain HTTP(S) URLs or fail the working `Mnfr`/`TargetAgeGroup` vocabulary. Formatting warnings do not count. |
| `suspicious_target_with_displacement` | A semantic target finding as above, plus numeric text or a displaced identifier/invalid URL field. |

Working vocabularies are `Mnfr = {All others, J&J}` and
`TargetAgeGroup = {Adult, Children, Infant}`. Comparisons trim surrounding spaces
without changing values. Missing labels are allowed. Brand, Platform, Segment,
and Sub-Segment have no enforced vocabulary; URLs are the only semantic check
there. Rare labels and legitimate lowercase names must not become structural
failures. Multiple lowercase/leading-space targets remain review-only. The initial
profile documented three training rows with description/ingredient fragments
across target columns; all three trigger the multi-target rule and also the
independent metadata-spill rule on the supplied data.

Other findings never quarantine on their own:

- `high_feature_missingness`: missing count **strictly greater than**
  `Q3 + 1.5 * (Q3 - Q1)`, with linear quartiles over all full-width records in
  each source separately. The IQR convention already exists in target cleaning;
  this check uses only the high side and treats it as review evidence. Equality
  to the upper fence is not flagged, including when IQR is zero.
- The default 12-column missingness scope is Retailer, ProductCategory,
  ProductBrand, ProductName, the four measurement fields, ProductUrl,
  ProductImageUrl, ProductDescription and ProductContents. All target labels,
  identifiers, metadata and provenance are excluded. Thus intentionally blank
  target labels and sparse training Platform labels cannot inflate this statistic.
- `no_valid_numeric_anchor`: none of the configured measurements is present and
  valid. A numeric `source_row` or identifier cannot satisfy the check. Sparse
  listings can legitimately lack all measurements; this is not a quarantine gate.
- `target_format_review`, individual `suspicious_target_value`,
  `unconfirmed_timestamp_format`, `invalid_url_field`, `url_in_identifier`,
  `unexpected_exclude`, and `populated_unknown_metadata` are field findings.
- Literal `Exclude` is a business-policy marker, not a structural failure and
  does not create a hold or quarantine in this workflow.

No missingness fraction, string-length cutoff, target-frequency threshold, or
arbitrary weighted score automatically quarantines rows. These rules cannot detect
all shifts, especially exchanges between fields with compatible types.

## Configuration and findings interface

`run(train_path, target_path, output_dir, config=StructuralConfig())` profiles and
screens both original files. `--config path.json` accepts validated JSON overrides
for `StructuralConfig`; unknown keys, unsupported numeric fields, duplicate lists,
invalid multipliers, and target/provenance missingness columns are rejected.

Numeric field scope, integer-count fields, missing tokens, missingness scope/IQR
multiplier, working vocabularies and enabled quarantine rules are explicit in the
config and saved in every summary. For example, `{"quarantine_rules": []}` retains
all rule findings as review-only. Disabling rules does not erase their evidence.
Use policy version plus the entire saved config when comparing runs; different
configs need not agree.

The pure `assess_row(raw_string_mapping, identity, config=...,
missingness_upper_count=...)` interface requires full original cells and an
explicit dataset-profile fence. It returns the identity, `quarantine`, sorted
`reason_codes`, sorted `finding_codes`, and individual findings. Each finding
contains its own full identity, `reason_code`, `quarantine_rule` boolean, and
`evidence` with original field/value strings and any relevant numeric cutoff.

Consumers may read this findings schema without importing feature or target
cleaners. Cleaner warnings are not accepted as authoritative structural evidence:
the current implementation recomputes checks from original sources. An eventual
adapter must map a cleaner's findings explicitly to source keys and evidence,
validate complete source coverage/hash, and preserve warning severity. Passing a
cleaner's boolean mask or translating target reason 2 to quarantine is not an
interface supported here.

**Merged feature/integration compatibility:** `structural-v2` writes the same
canonical keys directly. `verify_feature_compatibility(structural_dir,
{"training": feature_training_dir, "target": feature_target_dir})` in the
independent validator verifies both full audits, including raw hashes and coverage,
then compares their keyed decisions. It reports agreements and disagreements only;
it does not generate a union decision or alter either run. The optional validator
uses Max's existing feature verifier; the structural rule engine imports neither
cleaner. Run it against completed feature runs (including `feature_training` and
`feature_target` folders inside an existing integrated run):

```bash
python -m scripts.validate_structural --output-dir data/processed/structural_run --feature-training-run data/processed/integration_run/feature_training --feature-target-run data/processed/integration_run/feature_target
```

Both feature paths are required together. Wrong roles, source hashes, incomplete
coverage, duplicate/renumbered keys and tampered artifacts fail validation.
The structural results are not fed into `integration.row_decision`; the merged
integrated command retains its existing candidate-mask policy.

## Artifacts and validation

| Artifact | Meaning |
| --- | --- |
| `training_decisions.csv`, `target_decisions.csv` | Exactly one row per original record in order; source identity, `quarantine` (0/1), JSON reason and finding code lists. |
| `training_evidence.jsonl`, `target_evidence.jsonl` | Every affected record, including review-only rows; full identity, all findings, original ordered column names and cell strings. Wrong-width records retain extra cells. |
| `summary.json` | Completion marker; policy/config, source paths/hashes, coverage, fitted diagnostic fences, counts per dataset/reason/decision, output hashes, assumptions and limitations. |

Reason totals can overlap. `rows_by_finding` counts distinct affected rows per
code, not cells; `rows_by_finding_and_decision` distinguishes kept warnings from
quarantined rows. `rows_by_quarantine_reason` counts only enabled quarantine
reasons. `keep` means no structural quarantine under this policy, not eligibility
for any target model. A run without `summary.json` is incomplete and must not be
consumed. The original source plus manifest preserves raw values for unflagged
rows; the checker does not duplicate entire private sources.

The independent validator checks every source key/row, evidence values, reason
coverage, summary reconciliation and source/output hashes without recomputing the
decision rules. Tests cover rule behavior; repeated full runs check determinism.
See [current compatibility validation](findings/00_data_profile/08_structural_main_compatibility.md)
and the [original v1 validation](findings/00_data_profile/07_structural_validation.md).
