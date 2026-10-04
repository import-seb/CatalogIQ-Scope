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

Machine-readable copy: `data/interim/cleaning_audit_summary.csv`.

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
