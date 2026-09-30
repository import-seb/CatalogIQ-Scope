# CatalogIQ feature decisions — second pass

Date: 2026-09-28. Owner: Maksim Pikalov.

Status: implemented and validated locally; proposed for team review in PR #6.
Feature-side recommendations still require agreement. No final
quarantine mask or modeling-ready dataset is claimed.

## What changed

Added `scripts/feature_decisions.py`, which applies the existing formatting
cleanup and assigns explicit actions to its flags. It preserves every input row,
all seven protected classification columns (including `Category`), and all five
identifier columns. It does not infer labels, reconstruct shifted records, round
counts, or turn invalid numeric values into nulls.

The original output-path check was also corrected: results cannot be
written inside the raw file's containing directory or to an ancestor containing
the input. Existing output directories are refused. Tests cover these cases.

## Results on the supplied files

The following dispositions are mutually exclusive and cover all rows. Priority:
quarantine candidate, source-marker policy hold, field review, repaired, retain.
An earlier disposition can still contain formatting repairs or further reviews;
the detailed audit preserves all of those events.

| Disposition | Training | Target |
| --- | ---: | ---: |
| Retain (including informational flags) | 71,749 | 23,734 |
| Repaired, with no higher-priority issue | 6,969 | 2,299 |
| Field review | 109 | 36 |
| Source `Exclude` policy hold, without structural recommendation | 5,260 | 1,842 |
| Structural quarantine candidate | 465 | 171 |
| **Total** | **84,552** | **28,082** |

- Changed rows across all dispositions: **7,766 / 2,603**.
- Removed rows: **0 / 0**. Raw files unchanged; labels, identifiers, schema and
  record order preserved.
- Exactly repeated complete category paths collapsed: **1 / 1**. These are the
  only additional cell changes relative to the first-pass candidates.
- Source `Exclude` markers: **5,293 / 1,852**; **33 / 10** also have structural
  recommendations. The table does not double-count them.
- Scientific-notation UPC values: **5,006 / 1,690**; preserved literally and
  informational, not corruption or an exclusion rule.
- Remaining field-review rows: training has 105 missing-name-only cases, 3
  invalid-product-URL-only cases, and 1 unexpected-Exclude plus missing-name case.
  Target has 35 missing-name-only and 1 invalid-product-URL-only cases. A source
  policy hold can also carry a field review; inspect its event details.

## Rules and decisions

| Feature issue | Action |
| --- | --- |
| Outer spaces in selected text/URL fields; breadcrumb separators | Normalize formatting; record before/after values. |
| Exact repetition of a complete category path | Collapse only a repeated block with at least two distinct levels, and only without structural evidence. Preserve ambiguous or partial repetition and variable hierarchy depth. |
| Nonnumeric/nonfinite/negative numeric values; fractional counts | Preserve values and require review. No coercion or rounding. Use corroborating evidence for structural recommendations below. |
| `MDM_InsertDateTime` | Preserve as text; exclude metadata from proposed model inputs. Numeric-looking values are not proof of a confirmed Excel date system. Conversion is explicitly deferred until the source format is confirmed. |
| Scientific-notation UPC | Preserve as identifier text; do not expand it or use it as a product grouping key. Leading zeros/precision may already be lost upstream. |
| URL in `MDM_Id` | Preserve the identifier; review row alignment. Do not copy it into a different field by guessing. |
| Invalid product/image URL | Preserve and withhold field pending review. Syntax check only; no remote fetching or URL repair. |
| Literal `Exclude` marker | Propose a separate policy hold until sponsor/team interpretation is agreed. It is not itself a corruption finding. Other text in this column is a review signal, not a boolean. |
| `Sun1`–`Sun5`, `Notes` | Preserve columns; exclude from proposed model inputs. Populated values need alignment review because their intended schema is unknown. |
| Missing product name | Keep missing; if description or contents exists, informational. Otherwise review model eligibility; no invented name or automatic quarantine. |
| Missing `JoiningKey` | Review business identifier, but retain source identity for integration. Do not invent a key. |
| Labels and identifiers | Byte-for-byte equivalent cell strings; no changes, even to rare labels. These columns are excluded from the proposed predictor inputs, not from target outputs or provenance. |

These are proposed downstream feature-use decisions: the full candidate CSV
still contains every original column. The script does not yet create a typed
model-input matrix. Numeric strings will need explicit conversion after the
joint row/field policy is approved; no fitting or imputation is performed here.

### Structural recommendations

The rules use only feature-side evidence, not target labels or their frequency:

| Rule | Training | Target |
| --- | ---: | ---: |
| Text in two or more numeric anchor columns | 305 | 109 |
| Text in one numeric anchor, plus another invalid numeric anchor or displaced metadata signal | 157 | 60 |
| Unexpected `Exclude` text together with populated `Sun`/`Notes` columns | 3 | 2 |

Numeric anchors: `ProductRating`, `ProductReviewsCount`, `ReviewsCount`, `XRatXRev`.
Corroborating metadata: URL in `MDM_Id`, nonnumeric timestamp, invalid product or
image URL. Two fractional counts alone do **not** trigger structural quarantine.

All observed numeric-error flags in these two files fall on the candidate rows
under these combination rules. That is coverage of these observed flags, not a
claim to detect every malformed row.

A reproducible spot review (seed 20260928) examined five examples from each of the
first two rule groups per dataset, and all five records in the third group:
**25 records total**. They show product-name fragments in numeric fields,
misplaced URLs, or ingredient/marketing text in metadata. Full local examples
are in each run's `rule_review.json`. This supports the proposals but is not
independent validation or a measured false-positive rate. Maria's independent
audit and Sebastian's structural checks remain necessary before final removal.

## Validation

**13 automated tests passed:** the three existing tests plus ten second-pass
tests. Coverage includes protected fields, informational UPC, source-marker vs
corruption distinction, isolated numeric errors, fractional-count counterexamples,
corroborated failures, ambiguous category paths, missing names, unknown flags,
multiline CSV records, duplicate business keys, overwrite refusal, malformed
inputs, and deterministic output on a repeated fixture run.

`scripts/validate_feature_decisions.py` separately compared **all 112,634 raw
records** with candidates and audit files without calling the cleaner. It
verified row order, 1-based source keys, protected fields, whitelisted corrections,
before/after logs, disposition totals, source hashes and output hashes. A tampered
0-based key is rejected even when its file hash is recalculated.

Original source hashes:

- Training: `00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452`
- Target: `0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db`

## Reproduction

From the repository root, using Python 3.11+ and the standard library:

```sh
python -m unittest discover -s tests -v
python -m scripts.feature_decisions --input data/provided/Selfcare_Training_data.csv --output data/derived/training-decisions-v1 --dataset training
python -m scripts.feature_decisions --input data/provided/Selfcare_Target_data.csv --output data/derived/target-decisions-v1 --dataset target
python -m scripts.validate_feature_decisions --input data/provided/Selfcare_Training_data.csv --output data/derived/training-decisions-v1 --dataset training
python -m scripts.validate_feature_decisions --input data/provided/Selfcare_Target_data.csv --output data/derived/target-decisions-v1 --dataset target
```

Use a new output directory for a rerun. Each run produces:

- `features_candidate.csv`: every source record, same order and schema.
- `feature_decisions.csv`: one decision row per source record, with the shared key.
- `decision_details.jsonl`: flagged/changed records, actions, reasons and actual
  before/after values. Contains source data; keep local.
- `summary.json`: completion marker, policy version, counts and file hashes.
  Outputs without this marker are incomplete and must not be consumed.

The checked local runs also contain `validation.json` and the spot-review file.
Raw and derived data remain ignored by Git.

## Integration and next steps

See [the cleaning integration contract](../../integration/cleaning_contract.md).
The target-cleaning package inspected on 2026-09-29 uses source filenames as
dataset tokens and 0-based source rows. Normalize this explicitly before joining;
never silently guess. Its default pandas missing-value policy also differs from
this cleaner's blank/null policy; reconcile that separately from label corrections.
Agree the three structural rules, the meaning of `Exclude`, and treatment of the
145 remaining field-review rows. Then combine feature, target and structural
audits into one versioned mask and generate target-specific eligible sets/splits.

The reviewed evaluation research recommends grouped splits and explicit reporting
of rare classes and Platform limitations. Those are inputs to the next stage;
they do not authorize merging or removing target labels here. Ask remains paused.
