# Cleaning Validation & Data Quality Audit

**Status:** complete. Run against raw vs. the v4 integrated run, 2026-10-04.
**Result:** 8 PASS, 2 WARN, 0 FAIL.
**Location:** `docs/validation/cleaning_validation.md`
**Notebook:** `notebooks/02-cleaning-validation-audit.ipynb`
**Owner:** Maria Garcia Sehara

---

## 1. Headline

The v4 integrated run is sound. Every removal is accounted for by a named rule,
no class is lost without a recorded reason, and no structural corruption
survives.

**The finding worth acting on is about the policy it replaced.** The earlier
`conservative_handoff_20261002` destroyed **four entire Sub-Segment classes and
one Platform class** — 676 labelled rows — that exist in the raw data and are
retained by v4. Two of them are substantial categories, not rounding errors.

| | raw | conservative v2 | integrated v4 |
|---|---|---|---|
| Training rows | 84,552 | 78,718 | **83,938** |
| Target rows | 28,082 | 26,033 | **27,870** |
| Sub-Segment classes | 54 | 46 | **50** |

v4 removes 614 training rows (0.73%). v2 removed 5,834 (6.90%).

---

## 2. What was audited

| Input | Path | Rows |
|---|---|---|
| Raw training | `data/provided/Selfcare_Training_data (1).csv` | 84,552 |
| Raw target | `data/provided/Selfcare_Target_data (1).csv` | 28,082 |
| Cleaned training | `data/processed/<run>/training_candidate.csv` | 83,938 |
| Decision table | `data/processed/<run>/row_decisions.csv` | 112,634 |
| Label changes | `data/processed/<run>/label_changes.csv` | 7 |

Reproduced locally with the agreed policy:

```bash
python -m catalogiq --mode integrated --exclude-policy keep \
  --output-dir data/processed/maria_audit_run
python -m catalogiq.integration_validation --output-dir data/processed/maria_audit_run
python scripts/export_training_dataset.py --run-dir data/processed/maria_audit_run
```

Counts match the reference run exactly: **83,938 training candidates and 27,870
target candidates**. The run is reproducible from the same sources and code.

### 2.1 Matching raw rows to cleaned rows

Rows are matched on `source_row`, the pipeline's 1-based provenance index into
the raw file (`summary.json` records `source_row_base: 1`). This matters: the
feature export drops `JoiningKey`, `Sku` and `Upc`, so `source_row` is the only
key that survives end to end.

An earlier pass of this audit fell back to a content hash and reported 7,719
"invented" rows. That was an artifact of the hash, not a finding — missing values
are normalised during cleaning (blank, whitespace and literal `null` all become
empty), so the hashes diverge on rows that were never touched. Recorded here
because the same trap will catch anyone auditing without the provenance key.

---

## 3. Results

| # | Criterion | Status |
|---|---|---|
| 3.1 | Row counts | PASS |
| 3.2 | Quarantine counts and reasons | PASS |
| 3.3 | Missingness | PASS |
| 3.4 | Target distributions | PASS |
| 3.5 | Rare-class retention | PASS |
| 3.6 | Datatype changes | **WARN** |
| 3.7 | Feature normalization | PASS |
| 3.8 | Malformed labels | PASS |
| 3.9 | Structural corruption | PASS |
| 3.10 | Unexpected information loss | **WARN** |

Machine-readable copy: written to `data/interim/cleaning_audit_summary.csv` by
the notebook. `data/` is git-ignored, so regenerate it rather than expecting it
in the repo.

### 3.1 Row counts — PASS

84,552 → 83,938. 614 rows removed (0.73%), no rows added. Every removed row
appears in the decision table.

### 3.2 Quarantine counts and reasons — PASS

All 614 removals carry a named rule, and the counts reconcile exactly:

| Reason | Rows |
|---|---|
| `text_in_multiple_numeric_anchors` (feature + structural) | 199 |
| `url_in_timestamp_with_displacement` + `text_numeric_failure_with_corroboration` | 157 |
| `structural:missing_product_text` | 149 |
| `url_in_timestamp_with_displacement` + `text_in_multiple_numeric_anchors` | 106 |
| `unexpected_exclude_with_populated_unknown_columns` + `target:6` + `missing_product_text` | 2 |
| `unexpected_exclude_with_populated_unknown_columns` + `target:6` | 1 |
| **Total** | **614** |

`hold_reasons` is empty for all 84,552 rows, which is the expected consequence of
decision 6 in the decision log: no automatic whole-row hold.

**Note on where the reasons live.** `training_quarantine.csv` carries the rows
but no reason column. The reasons are in `row_decisions.csv`, which covers both
datasets — filter on `dataset` or the counts will not reconcile.

### 3.3 Missingness — PASS

No column became emptier. Several became *less* empty because quarantined rows
were disproportionately the malformed ones: `ProductUrl` 0.27% → 0.00%,
`ProductName` 0.18% → 0.00%, `ProductImageUrl` 0.89% → 0.59%.

### 3.4 Target distributions — PASS

Largest class-share move is 0.42pp, on Segment. No target was removed unevenly
enough to bias a model trained on the result.

### 3.5 Rare-class retention — PASS, and this is the section that matters

**Under v4, nothing is lost that is not documented.** Four Sub-Segment labels
present in raw are absent from the cleaned file, and all four are accounted for:

| Label | Raw rows | What happened |
|---|---|---|
| `Cold / Flu` | 4 | Relabelled to `Cold/Flu`, rule `slash_spacing` |
| `Other Lifestyle` | 3 | Relabelled to `Other Lifestyle CHC`, rule `reviewed_other_lifestyle` |
| `" gluten-free"` | — | Malformed, removed (§3.8) |
| `" lactose"` | — | Malformed, removed (§3.8) |

Each of the 7 relabels is recorded row by row in `label_changes.csv` with its
source row and rule. That closes the merge-versus-deletion question this audit
could not previously answer.

**Under the conservative v2 handoff, five classes were destroyed.** All exist in
raw and all are retained by v4:

| Target | Class | Raw rows | v2 | v4 |
|---|---|---|---|---|
| Sub-Segment | `Motion Sickness` | 408 | **0** | 405 |
| Sub-Segment | `Hemorrhoid Remedies` | 210 | **0** | 210 |
| Sub-Segment | `Anti-Nausea` | 57 | **0** | 57 |
| Sub-Segment | `Creams & Gels (Rubs)` | 1 | **0** | 1 |
| Platform | `Arthritis` | 2 | **0** | 2 |

676 labelled rows across five classes, two of them substantial. A model trained
on the v2 handoff could not classify a motion-sickness or hemorrhoid product at
all, and both exist in the unlabelled target set awaiting prediction.

This also resolves the third near-duplicate pair left open in the relationships
research. `Creams & Gels (Rubs)` was never relabelled — it does not appear in the
label policies because it was not a label decision. v2 simply dropped it.

**The mechanism.** 5,293 raw rows carry the literal `Exclude` marker. v2's
`review_hold` partition was 5,368 rows. Under `--exclude-policy keep`, 83,938 of
84,552 rows are retained, so the Exclude rows survive. The group decision to keep
otherwise-usable Exclude rows is what recovers these classes.

### 3.6 Datatype changes — WARN

Four columns move from string to numeric: `ProductRating` and `XRatXRev` to
float, `ProductReviewsCount` to float, `ReviewsCount` to int.

Expected for parsed numerics and not a defect. Flagged because it is a schema
change a downstream consumer should know about, and because `ProductReviewsCount`
reading as float rather than int suggests nulls in that column.

No target column reads as float, so the empty-column trap is not present.

### 3.7 Feature normalization — PASS

Zero feature values changed across 76,219 matched rows. The handoff README's
claim that original values are preserved holds: cleaning decides what to keep,
it does not rewrite feature content. The only value changes in the whole run are
the 7 label relabels in §3.5.

### 3.8 Malformed labels — PASS

All 18 malformed values across the six targets were handled — 3 per target, each
beginning with whitespace, exactly as the dataset profile reported. Zero survive.

This unblocks the relationships research, which states in §2.1 that no conclusion
about impossible label combinations is valid on uncleaned data.

### 3.9 Structural corruption — PASS

No cross-column vocabulary leakage and no rows with an unexpected field count.

### 3.10 Unexpected information loss — WARN

Fourteen columns are dropped by the feature export: `Category`, `Exclude`,
`JoiningKey`, `MDM_Id`, `MDM_InsertDateTime`, `Notes`, `ProductModelNumber`,
`Sku`, `Sun1`–`Sun5`, `Upc`.

Most are intended and documented. Three are worth a deliberate confirmation:

- **`JoiningKey` and `Sku`** are the only candidate row identities. Dropping them
  from the modelling export means the split strategy has to be decided before
  the export, not after. See §2.6 of the evaluation strategy.
- **`Exclude`** is dropped from the export, so the modelling artifact cannot tell
  which rows carried the marker. The decision log says the marker "remains
  auditable" — it is auditable in `row_decisions.csv`, not in
  `training_cleaned.csv`.
- **`Category`** carries one value, `Self Care`, in all rows, so nothing is lost.

No labelled row was removed without a recorded reason.

---

## 3.11 Removed rows, examined directly

The ticket asks four specific questions about the 614 removed training rows.
Each is answered against the rows themselves, not against the pipeline's own
report of what it did.

### Were good products removed?

**No.** Every removed row carries concrete displacement evidence.

| Group | Rows | Evidence found in those rows |
|---|---|---|
| No product text at all | 151 | Name, description and contents all blank |
| Text in numeric columns | 199 | 727 non-numeric values in `ProductRating`, `XRatXRev`, `ProductReviewsCount`, `ReviewsCount` |
| URL in the timestamp column | 263 | 263 URLs in `MDM_InsertDateTime`, one per row, plus 369 text-in-numeric values |
| Exclude with populated unknown columns | 3 | 2 of 3 also have no product text |

463 of the 614 have something that looks like a product name, which is why this
needed checking rather than assuming. Reading them explains it: they begin with
a stray quote and contain unescaped inch marks — `Entil Large Heating Pad 24"" x
33""`, `NEWSTYLE Ice Bag, 3 Pack[6"`. The quote broke the CSV parse and shifted
the row's fields. The product name survives because it is the field the
displacement starts in; everything after it is wrong.

### Were real rare classes removed?

**No.** This is the strongest result in the audit, because the rarest classes are
exactly where a removal would be invisible.

Every one-row and two-row class survives intact:

| Target | Class | Raw | v4 |
|---|---|---|---|
| Sub-Segment | `Creams & Gels (Rubs)` | 1 | 1 |
| Sub-Segment | `Daily oral contracception` | 2 | 2 |
| Sub-Segment | `Dermatologicals` | 4 | 4 |
| Platform | `Precise`, `12 Hour Relief`, `Cold Max`, `SmartCheck`, `Effective in 15 Minutes` | 1 each | 1 each |

The labels that do disappear are not classes. They are fragments of marketing
copy that landed in a target column through the same displacement — `" coffee"`,
`" Keto certified"`, `"aids in collagen formation* GRASS-FED COLLAGEN: Super
Collagen + Vitamin C & Biotin is Keto certified"` as an `Mnfr` value. Each has
one or two rows, and each of those rows is quarantined for displacement.

The only two genuine labels that leave are `Cold / Flu` (4 rows) and
`Other Lifestyle` (3), and neither is removed: both are relabelled, with the
rule and the source rows recorded in `label_changes.csv`.

### Were rows quarantined for the correct reason?

**Yes.** Each reason is matched by evidence in the rows it fired on, with no
cross-contamination:

- `url_in_timestamp_with_displacement` — 263 rows, 263 URLs in the timestamp
  column. One per row, exactly.
- `structural:missing_product_text` — 149 rows, 149 with no product text.
- `text_in_multiple_numeric_anchors` — 199 rows, 727 text values across four
  numeric columns, and zero of them have the URL or missing-text signature.

No row is quarantined under a rule whose evidence is absent from it.

### Were review-only rows treated correctly?

**Yes**, and the distinction is enforced rather than merely documented.

| Flag | Rows flagged | Of those, quarantined |
|---|---|---|
| `target_reason_codes` = 2 (formatting) | 242 | **0** |
| `target_reason_codes` = 2;6 | 3 | 3 |
| `feature_flags` | 10,475 | 614 (5.9%) |
| `hold_reasons` | **0** | — |

Target reason 2 is review-only in the decision log, and on its own it removes
nothing: all 242 such rows are retained. The three that are quarantined also
carry reason 6, which the decision log defines as a quarantine rule. 10,475 rows
carry a feature flag and 94.1% of them are kept, so a flag is a note rather than
a verdict.

`hold_reasons` is empty for all 84,552 rows, which is decision 6 working as
written: no automatic whole-row hold. The review evidence is preserved separately
in `target_review_flags.csv` (263 rows) rather than being discarded.

---

## 3.12 Final answer: does the cleaned dataset follow the cleaning rules?

**Yes.** Every policy in `docs/decisions.md` was checked against the data rather
than against the pipeline's own report, and all ten hold.

| # | Policy (decision log) | Verified |
|---|---|---|
| 1 | Keep otherwise usable `Exclude`-marked rows | 5,293 marked, **5,217 kept (98.6%)** |
| 1 | Arbitrary text in `Exclude` is not the marker | 4 such rows; none removed by a marker rule |
| 2 | `MDM_InsertDateTime` out of model inputs, retained in source | Absent from the export, present in the candidate partition |
| 3 | A bad timestamp alone does not quarantine | 263 URL-in-timestamp rows, **263 with independent corroboration** |
| 4 | All three text fields missing → structural quarantine | 151 such rows, **151 quarantined** |
| 5 | A bad URL alone does not hold the row | 3 rows with no other signal, **3 kept** |
| 6 | No automatic whole-row hold | `hold_reasons` empty on all 84,552 rows |
| 6 | A field flag is not a row decision | 10,475 flagged, **94.1% kept** |
| 7 | Shifted columns quarantined on documented combinations | Every reason matched by evidence in the row (§3.11) |
| 9 | `Sun1`–`Sun5` and `Notes` out of exports, kept in source | Absent from the export, present in the candidate partition |
| 10 | No derived category levels; full path preserved | No `retailer_category_level_*`; 84,522 paths intact |

Label policies hold too: `Cold / Flu` → `Cold/Flu` (4 rows) and `Other Lifestyle`
→ `Other Lifestyle CHC` (3 rows), both recorded in `label_changes.csv`; supplied
`Mnfr` preserved with no J&J inferred from Brand; target reason 2 review-only
with 242 rows flagged and none removed; and no fixed parent whitelist enforced,
with `Probiotics` and `Internal Pain` retained under two parents each.

### Three checks that looked like failures and were not

Recorded because the next person to audit this will hit the same thing.

- **Policy 1.** Four rows with arbitrary text in `Exclude` are quarantined, which
  looks like the marker being enforced. Their reasons are `missing_product_text`
  and `multiple_suspicious_targets` — independent rules. No marker rule fired.
- **Policy 3.** All 263 URL-in-timestamp rows are removed, which looks like a bad
  timestamp removing rows on its own. All 263 also carry text in a numeric
  column, so every one has the independent evidence the policy requires.
- **Policy 5.** 241 rows have a `ProductUrl` that fails a naive `startswith("http")`
  test. Only 3 of them have no other displacement signal, and all 3 are kept —
  and those 3 are valid Walgreens URLs whose inch marks (`10"" x 13""`) broke the
  test, not the data.

In all three the pipeline was right and the check was too crude. A policy audit
that stops at the first surprising count will report failures that are not there.

### The qualification

The dataset follows the rules **as the decision log currently states them**. Two
things it does are correct but not written down there, and both appear as WARNs
in §3.6 and §3.10: the dtype change on four columns in the feature export, and
the dropping of fourteen columns including `JoiningKey`, `Sku` and `Exclude`.

Neither is a rule violation. Both are decisions the decision log does not record,
and the second one has a downstream consequence — the split strategy must be
settled before the export, because the identity columns do not survive it.

---

## 3.13 What failed, exactly

**Nothing failed.** Zero FAIL across the ten criteria. What follows is the same
breakdown applied to the two WARNs, so they can be dispositioned rather than
carried as unexplained.

### WARN 1 — dtype change in the feature export

| | |
|---|---|
| **Rule** | None. The decision log does not cover export dtypes. |
| **Which rows** | All 83,938. The change is per-column, not per-row. |
| **Which columns** | `ProductRating`, `XRatXRev`, `ProductReviewsCount` → float; `ReviewsCount` → int |
| **Data lost** | **None.** Every populated value parses as a number: 83,938 / 83,938 / 83,821 / 83,938, zero unparseable. |
| **Fix or review?** | **Review, and it is a confirmation rather than a question.** |

The conversion is representational. No value becomes unreadable and no row is
affected in content. It is listed only because a downstream consumer reading the
export gets a different schema than the one in the candidate partition, and that
should be a known fact rather than a discovery.

### WARN 2 — fourteen columns dropped by the feature export

| | |
|---|---|
| **Rule** | Policies 2 and 9 cover five of the fourteen. The other nine are undocumented. |
| **Which rows** | All 83,938. |
| **Fix or review?** | **Review for eleven. One needs a decision before modelling.** |

Breaking the fourteen down by what is actually lost:

| Column | Rows with data | Distinct values | What dropping it costs |
|---|---|---|---|
| `Notes`, `Sun1`–`Sun5` | **0** | 0 | Nothing. Empty in every row. Policy 9. |
| `Category` | 83,938 | **1** | Nothing. Constant `Self Care`. |
| `MDM_InsertDateTime` | 83,938 | 12 | Nothing for modelling. Policy 2, retained in source. |
| `Exclude` | 5,217 | 1 | The marker is invisible downstream. Auditable in `row_decisions.csv`. |
| `Upc` | 16,613 (19.8%) | 12,280 | A possible feature, 80% missing. |
| `ProductModelNumber` | 11,678 (13.9%) | 10,464 | A possible feature, 86% missing. |
| `MDM_Id` | 83,938 | 83,938 | A row identifier. |
| **`JoiningKey`** | **83,938** | **83,938** | **A row identifier.** |
| **`Sku`** | **83,938** | **83,460** | **A near-unique identifier.** |

**Eleven of the fourteen cost nothing or are already documented.** Seven are
empty or constant, and three are deliberate policy.

**The one that needs a decision is the identity columns.** `JoiningKey`, `Sku`
and `MDM_Id` do not survive into `training_cleaned.csv`. Only `source_row`
remains, and it is a row index, not a grouping key.

The consequence is concrete and it is not an audit finding so much as a
sequencing constraint:

> A grouped train/test split cannot be built from the modelling export. It has to
> be constructed from the candidate partition, before or alongside the export.

This matters because a random row split on a product catalogue puts size and
flavour variants of the same product in both train and test, which inflates every
score invisibly. §2.7 of the evaluation strategy covers the key construction.

### Disposition

| Item | Needs |
|---|---|
| Dtype change | Confirmation that the export schema is intended |
| Eleven of the dropped columns | Nothing |
| `Upc`, `ProductModelNumber` dropped | Confirmation they are not wanted as features |
| Identity columns dropped | **A decision on where the split is built** |

---

## 4. Can the cleaned data be used?

**Yes, with two caveats.**

The v4 run is reproducible, every removal is explained, and no class disappears
without a documented rule. It is suitable for modelling and for the relationship
work.

The caveats are §3.6 and §3.10 — a schema change and fourteen dropped columns —
both of which need a one-line confirmation rather than a fix.

**The conservative v2 handoff should not be used.** Anything already built on it
should be re-run: it is missing 5,220 rows and five classes.

---

## 5. Recommendations

1. **Re-run anything built on `conservative_handoff_20261002`.** It is missing
   five classes that exist in the data.
2. **Carry the recovered classes into the evaluation strategy.** Sub-Segment goes
   from 46 to 50 classes, which changes the macro-F1 denominator and the support
   tiers in §3.5 of that document.
3. **Add a `reason` column to `training_quarantine.csv`**, or document that
   `row_decisions.csv` is the authoritative record. An auditor reading only the
   quarantine partition sees removals with no explanation.
4. **Decide the split strategy before the feature export**, since it drops the
   only candidate identity columns.

---

## 6. Open questions

1. **Is the `ProductReviewsCount` float dtype intended**, or does it indicate
   nulls that should be handled explicitly? (§3.6)
2. **Should `Exclude` survive into the modelling export** so the marker stays
   visible downstream, or is `row_decisions.csv` sufficient? (§3.10)
3. **Should `Category` be dropped from the schema entirely**, given it carries
   one value in all 84,552 rows?
4. **Three `Motion Sickness` rows are quarantined under v4** (408 raw, 405
   retained). Worth confirming they fall under the structural rules rather than
   the label.
