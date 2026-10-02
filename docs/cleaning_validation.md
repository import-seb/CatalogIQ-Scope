# Cleaning Validation & Data Quality Audit

**Status:** run against the 2026-10-02 conservative handoff. 4 PASS, 1 FAIL, 5 BLOCKED.
**Location:** `docs/validation/cleaning_validation.md`
**Notebook:** `notebooks/02-cleaning-validation-audit.ipynb`
**Owner:** Maria Garcia Sehara

---

## 1. Headline

The handoff **reconciles exactly against its own manifest** and is internally
clean: no malformed labels, no structural corruption, no empty target columns.

It is **not yet auditable**. Five of the ten criteria compare raw against
cleaned, and the raw CSVs were not shipped with the handoff. Those are recorded
as BLOCKED, not as PASS — a check that could not run is not a check that passed.

One criterion fails outright: 5,834 rows were removed and the files recording
why were not included.

| | Count |
|---|---|
| PASS | 4 |
| FAIL | 1 |
| BLOCKED (needs raw data) | 5 |

---

## 2. What was audited

| Input | Source | Present |
|---|---|---|
| `training_cleaned.csv` | 2026-10-02 handoff | yes — 78,718 rows × 35 cols |
| `target_cleaned.csv` | 2026-10-02 handoff | yes — 26,033 rows |
| `summary.json` | 2026-10-02 handoff | yes |
| Raw `Selfcare_Training_data (1).csv` | PO ZIP | **no** |
| Quarantine / review-hold files | cleaning pipeline | **no** |

Handoff policy: `conservative_handoff_20261002`.
Source fingerprint (training): `00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452`.

The audit runs in **partial mode** when raw is absent: checks that need only the
cleaned file still run, and the rest report BLOCKED with the reason.

---

## 3. Results

### 3.1 Row counts — PASS

| | Training |
|---|---|
| Input | 84,552 |
| Cleaned | 78,718 |
| Removed | 5,834 (6.90%) |
| Quarantined | 466 |
| Held for review | 5,368 |

466 + 5,368 = 5,834. The removal total reconciles exactly, and the file on disk
matches the row count in `summary.json`.

**Note for the team:** 6.90% is a large removal, and most of it — 5,368 rows —
is *held for review* rather than rejected. Those rows are recoverable. Whether
they should be recovered is a question for the PO, not for this audit.

### 3.2 Quarantine counts and reasons — **FAIL**

The counts are known. The reasons are not.

`summary.json` reports 466 quarantined and 5,368 held for review, and the
handoff README states those files are "kept separately by the preparer". Without
them, no individual removal can be checked against the rule that caused it.

This is the one finding that blocks sign-off, and it is a pipeline/handoff gap
rather than a defect in the data.

**Action:** request the quarantine and review-hold files from the preparer.

### 3.3 Missingness — BLOCKED

Cannot compare without raw. The cleaned file's own rates, for reference:

| Column | Missing |
|---|---|
| `Exclude`, `Sun1`–`Sun5`, `Notes` | 100.00% |
| `Platform` | 98.61% |
| `ProductModelNumber` | 86.62% |
| `Upc` | 81.03% |
| `ProductDescription` | 68.58% |
| `ProductContents` | 59.87% |
| `Sub-Segment` | 5.34% |
| `Mnfr` | 5.32% |
| `TargetAgeGroup` | 5.32% |
| `Segment` | 2.28% |
| `Brand` | 0.00% |

`Exclude`, `Sun1`–`Sun5` and `Notes` are 100% empty in the cleaned file. The
handoff README already says to keep them out of model inputs; this confirms
there is nothing in them to keep.

### 3.4 Target distributions — BLOCKED

Class shares cannot be compared against a baseline that is not present.

### 3.5 Rare-class retention — BLOCKED

**The check the ticket singles out, and the strongest argument for obtaining the
raw CSVs.** Two specific things cannot be settled without them:

1. The handoff README says it fixed "the two agreed Sub-Segment label
   spellings". `Cold / Flu` (4 rows in the profile) is absent from the cleaned
   data, and `Cold/Flu` stands at 1,281 against 1,302 in the profile. Whether
   those 4 rows merged into `Cold/Flu` or were removed cannot be determined from
   the cleaned file alone.
2. Platform still has 6 single-row classes and 27 classes under 10 rows. Whether
   any class disappeared entirely during cleaning is unknown.

### 3.6 Datatype changes — PASS

No target column reads as `float64`, so the empty-column trap is not present:
a label join will not fail silently.

`Sun1`–`Sun5`, `Notes` and `MDM_InsertDateTime` read as `float64` because they
are entirely empty or unconverted. That matches the handoff README, which says
timestamps were deliberately not converted.

The raw-versus-cleaned dtype comparison needs the raw file.

### 3.7 Feature normalization — BLOCKED

The handoff README lists the transforms it applied: safe spacing and category
formatting, two Sub-Segment label spellings, identifiers and missing values
preserved, no guessed labels, no sentinel replacement. **Those are claims. This
section exists to verify them, and verification needs the raw file.**

### 3.8 Malformed labels — PASS

Zero values with leading or trailing whitespace across all six targets.

The profile reported 3 malformed values per target, all whitespace-prefixed
(`" coffee"`, `" tea"`, `" lactose"`, and others). None survive. `Mnfr` now
carries exactly 2 values rather than 4.

**This unblocks the relationships work**, which states in §2.1 that no
conclusion about impossible label combinations is valid on uncleaned data.

### 3.9 Structural corruption — PASS

No cross-column vocabulary leakage: no target holds a value belonging to another
target's vocabulary. No rows with an unexpected field count.

### 3.10 Unexpected information loss — BLOCKED

`summary.json` gives totals, not which records left. Dropped columns and the
label composition of removed rows need the raw file.

**Why this one matters more than its status suggests.** Labelled rows are the
scarce resource here. Platform carries 1,092 labelled rows out of 78,718, so a
removal that is negligible against the full dataset can be material against a
single target. 5,834 rows left and nobody can currently say how many carried a
Platform label.

---

## 4. Summary table

| # | Criterion | Status |
|---|---|---|
| 3.1 | Row counts | PASS |
| 3.2 | Quarantine counts and reasons | **FAIL** |
| 3.3 | Missingness | BLOCKED |
| 3.4 | Target distributions | BLOCKED |
| 3.5 | Rare-class retention | BLOCKED |
| 3.6 | Datatype changes | PASS |
| 3.7 | Feature normalization | BLOCKED |
| 3.8 | Malformed labels | PASS |
| 3.9 | Structural corruption | PASS |
| 3.10 | Unexpected information loss | BLOCKED |

Machine-readable copy: `interim/cleaning_audit_summary.csv`.

---

## 5. Can the cleaned data be used?

**For exploratory and relationship work, yes.** It is internally consistent, the
malformed rows are gone, and the structural checks pass. The relationships
notebook was run against it successfully.

**For a final model or any published metric, not yet.** 6.90% of rows were
removed and the project cannot currently state what left or why. A model trained
on a filtered subset, where the filter is undocumented, cannot be defended.

---

## 6. Blocking requests

1. **The quarantine and review-hold files**, with a reason per row. This closes
   §3.2 and is the only outright FAIL.
2. **The raw `Selfcare_Training_data (1).csv`.** This closes the five BLOCKED
   criteria. Nothing else can.
3. **A decision on the 5,368 review-hold rows.** They are held, not rejected,
   which means someone is expected to decide. Until then the usable training set
   is smaller than it needs to be.

---

## 7. Findings that affect the other documents

The relationships notebook was run against the same handoff. Several of its
results change what is written elsewhere, and they are recorded here because
this audit is what made them runnable.

| Finding | Effect |
|---|---|
| `Category` holds one value, `Self Care`, in all 78,718 rows | Closes the §4.7 open question in the relationships doc. Not a target, not a feature — it is a constant. |
| `JoiningKey` is unique per row (78,718 keys / 78,718 rows) | It is a row identifier, not a grouping key. §2.6 of the evaluation strategy named it the highest-priority open question and proposed splitting on it; that is not available. `Sku` has 78,298 unique values, so it is not a grouping key either. |
| Brand → Mnfr is deterministic across all 141 named brands | Rule R2 graduates to a hard rule. |
| Segment → Sub-Segment is **not** strict | Two sub-segments have two parents. Rule R1 stays in warn mode. |
| Platform is 93.4% J&J, not exclusively | The §4.4 scope hypothesis is supported but overstated. |

Details and the corrected parent-child map are in
`notebooks/01-classification-relationships.ipynb` and belong in an update to
`docs/research/classification-relationships-and-hierarchy.md`.

---

## 8. Open questions

1. **Will the quarantine and review-hold files be shared?** (§3.2)
2. **Will the raw CSVs be shared?** (§3.5, §3.10) Five criteria depend on it.
3. **Who decides on the 5,368 review-hold rows, and by when?**
4. **Were the two Sub-Segment spelling fixes merges or removals?** (§3.5) The
   README says "fixed", which implies a merge; the counts cannot confirm it.
5. **Is `Category` dropped from the schema**, given it carries no information?
