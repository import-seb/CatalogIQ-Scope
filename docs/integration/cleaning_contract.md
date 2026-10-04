# Cleaning integration contract

Updated 2026-10-03. Current policies: integrated-review-v4,
feature-decisions-v3, structural-v6 and target-labels-v2.
[Current decisions](../decisions.md) distinguish group agreement from Sebastian's
row policies. The default keeps otherwise usable Exclude-marked rows.

## Run and verify

```bash
python -m catalogiq --mode integrated --output-dir data/processed/new_run
python -m catalogiq.integration_validation --output-dir data/processed/new_run
```

Both commands accept --train and --target. Use a new output directory each time;
raw sources and historical runs are never overwritten. The verifier checks raw
hashes, complete partitions, identities, policy decisions, projected values,
allowed corrections and reason counts without rerunning the cleaners.

## Same audit pattern as the feature pass

The integration follows Max's existing pattern: run an independent pass, save
its decisions/evidence and summary, verify its outputs, then consume its keyed
decisions. It adds structural/ alongside feature_training/ and feature_target/.
The target cleaner still returns its two complete partitions and label-change
log. No cleaner imports another cleaner's rules or receives a filtered input.

1. Fingerprint the original training and target sources.
2. Run and verify structural decisions on both raw files.
3. Run and verify Max's feature decisions on both raw files.
4. Run target-label cleaning on the original training records.
5. Validate full source-key coverage and combine quarantine recommendations.
6. Write complete audit partitions plus the candidate feature projections.

The source identity is `(dataset, source_sha256, source_row)`:
- dataset is training or target;
- SHA-256 fingerprints the original file bytes;
- source_row is the one-based parsed CSV data record, excluding header and empty
  physical lines. Quoted multiline fields count once.

Business IDs can repeat and are never the join key. Structural decisions are
joined by key independent of their table order. Duplicates, foreign keys,
missing records, extra records or changed input fingerprints stop the run.
The legacy target-only CLI retains filename/zero-based identity; integration
assigns canonical identity before calling its reusable clean_training function.

## Decision interface and precedence

Every structural decision contains identity, quarantine (0/1), reason_codes and
finding_codes. Its evidence file preserves full raw cells for affected rows.
A finding alone must never be interpreted as a quarantine decision.

row_decisions.csv preserves feature, target and structural findings separately,
and records combined reasons with their owner prefixes:
- feature:<reason> for Max's corroborated feature recommendations;
- target:6 for invalid populated manufacturer labels;
- structural:<reason> for the independent row checks.

Any explicit quarantine recommendation wins. An explicit hold comes next; field
flags alone do not create a hold. Remaining rows are candidate and keep_candidate
is 1. Current default policy has no generic feature-review hold. The optional
Exclude hold/quarantine overrides are recorded in summary.json; keep is default.

Missing all of ProductName, ProductDescription and ProductContents is owned by
structural:missing_product_text and always quarantines under the current policy.
It is not merely a hold and cannot be rescued by a category, brand or rating.
Bad URLs alone remain flagged candidate rows. Bad timestamp format alone produces
no timestamp finding. A URL in a timestamp plus distinct displacement evidence
can quarantine. Target reasons 2/3 do not change row eligibility.

Overlapping rules can produce multiple reasons on one quarantined row. Summary
reason counts therefore need not sum to the number of quarantined rows.

## Outputs and preservation

- *_candidate.csv, *_quarantine.csv, *_review_hold.csv are disjoint, complete
  partitions containing all original columns plus canonical provenance.
- *_features.csv contains candidate rows projected to permitted feature columns
  plus the three join-only provenance columns. These keys are not predictors.
- Feature inputs exclude target labels, protected Category, business identifiers,
  Exclude, MDM_InsertDateTime, Sun1-Sun5 and Notes.
- ProductCategory and Retailer are retained. Category-level features are deferred.
- row_decisions.csv covers every input record. label_changes.csv and individual
  pass evidence record the original value and approved changes where applicable.
- summary.json is written last, with policy versions, scope, counts by dataset
  and reason, overlaps, source/output hashes and runtime versions.

Only logged whitespace/breadcrumb corrections and the two accepted Sub-Segment
replacements may change export values. No label inference, sentinel replacement,
deduplication or reconstruction of shifted rows occurs. Feature projections can
contain flagged original values; selection does not silently fix those values.
Raw sources remain the authority for all original strings. Blank/whitespace or
case-insensitive null is missing; other literal strings remain unchanged.

## Limits

Both datasets receive structural and feature screening; target labels remain
missing. Statistical profiles are computed separately using feature-only scopes.
IQR flags are heuristics and small or homogeneous populations can produce weak
baselines. Individual suspicious values may need later field-specific treatment.
Malformed CSV widths are quarantined by standalone structural mode; the feature
pass currently aborts such inputs, so integrated mode does not produce a complete
handoff for wrong-width sources. It fails rather than reconstructing the row.

The candidate mask implements the stated cleaning policy, not model readiness.
Future feature engineering and train/evaluation split design remain separate.
Historical conservative handoff counts are not the current selection policy.
