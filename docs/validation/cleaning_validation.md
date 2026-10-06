# Cleaning Validation & Data Quality Audit

**Version 2**, rebuilt after review. Supersedes v1, which reported results its
own checks did not support.
**Result:** 13 PASS, 5 WARN, 1 BLOCKED, 0 FAIL, across **both** datasets.
**Location:** `docs/validation/cleaning_validation.md`
**Notebook:** `notebooks/02-cleaning-validation-audit.ipynb`
**Run audited:** the local reproduction created by the commands in §2 (`my_cleaning_run`), of
`integrated_policy_v4_final_20261003` (`--mode integrated --exclude-policy keep`),
83,938 training and 27,870 target candidates.
**Owner:** Maria Garcia Sehara

---

## 1. What changed, and why

Review found seven problems with v1. Six were real defects in the audit, not in
the pipeline. They are listed first because the corrected numbers only make sense
against them.

| Review finding | v1 | v2 |
|---|---|---|
| Matching totals is not matching rows | Compared counts: 614 = 614 | Compares the row **sets**; symmetric difference is zero |
| Only 76,219 of 83,938 retained rows compared | Reported PASS and published 76,219 without noticing the gap | Compares **83,938 of 83,938**, prints coverage on every verdict |
| Shifted-column checks skipped quoted records | Two cheap signals | Includes the 562 retained quoted rows; validates each column's domain |
| Rare labels with stray spaces excluded by rule | Skipped them, assuming the answer | Evaluates them on evidence |
| Only training audited | Training only | Training **and** target |
| Conclusions overstated | "no product text" called corruption | States what the evidence supports |
| Run directory auto-detected | Could pair files from different runs | One run selected explicitly |

**The worst of these is the second**, and it is worth being precise about. v1's
first pass had no usable key and fell back to a content hash, which matched
76,219 rows. The key was fixed afterwards, the notebook was re-run — and the
number **76,219 was never updated in the document**. v1 therefore published a
PASS, over a population it did not cover, citing a figure from a superseded run.

v2 makes that structurally impossible: the verdict recorder downgrades any PASS
whose coverage is below the population it claims. The summary ends by asserting
that count is zero.

---

## 2. Results

### Training

| Criterion | Status | Coverage |
|---|---|---|
| Row counts and identity | PASS | 84,552 / 84,552 |
| Quarantine reasons | PASS | 614 / 614 |
| Missingness | PASS | 83,938 / 83,938 |
| Target distributions | PASS | 84,552 / 84,552 |
| Rare-class retention | PASS | 84,552 / 84,552 |
| Malformed labels | PASS | 84,552 / 84,552 |
| Structural integrity | PASS | 83,938 / 83,938 |
| **Datatype changes** | **WARN** | 83,938 / 83,938 |
| **Feature values on retained rows** | **WARN** | 83,938 / 83,938 |
| **Information loss** | **WARN** | 614 / 614 |

### Target

| Criterion | Status | Coverage |
|---|---|---|
| Row counts and identity | PASS | 28,082 / 28,082 |
| Quarantine reasons | PASS | 212 / 212 |
| Missingness | PASS | 27,870 / 27,870 |
| Rare-class retention | PASS | 28,082 / 28,082 |
| Malformed labels | PASS | 28,082 / 28,082 |
| Structural integrity | PASS | 27,870 / 27,870 |
| **Feature values on retained rows** | **WARN** | 27,870 / 27,870 |
| **Information loss** | **WARN** | 212 / 212 |
| Target distributions | BLOCKED | — |

Target distributions is BLOCKED and not PASS: the target dataset is unlabelled,
so there is no distribution to compare. A check that has nothing to compare has
not passed.

Machine-readable: `data/interim/cleaning_audit_summary.csv`.

---

## 3. The findings

### 3.1 Feature values changed — the correction that matters

**v1 said:** "Zero feature values changed across 76,219 matched rows." PASS.

**v2 finds:** `ProductCategory` changed on **7,714 training rows (9.19%)** and
**2,586 target rows (9.28%)**.

The transform is:

```
Health and Medicine>Vitamins and Supplements>Multivitamins
Health and Medicine > Vitamins and Supplements > Multivitamins
```

Separator spacing, plus one training row where an exactly duplicated path
(`A > B > C > D > A > B > C > D`) is collapsed to a single copy. Every changed
row is explained: **7,714 of 7,714** and **2,586 of 2,586** reduce to identity
once separator spacing and exact-duplicate collapse are normalised away.

**Is it approved?** The handoff README documents it — *"Fixed safe
spacing/category formatting"*. `docs/decisions.md` does not. Policy 10 says
*"Preserve the full path and Retailer now"*, and the full path is preserved: no
level is lost, including in the deduplicated row, where the removed text was a
verbatim repeat.

**Verdict: WARN, not FAIL.** The change is real, benign, and documented by the
preparer, but it is not in the decision log. It needs a recorded decision, not a
fix.

### 3.2 Datatype changes — the claim was too strong

**v1 said:** "No data lost. Every populated value parses."

That is true and incomplete. The *numeric value* survives; the *textual form*
does not. Across the four columns, **111,249 values change representation** in
the export: `5` becomes `5.0`.

| Column | Export dtype | Unparseable | Text form changed |
|---|---|---|---|
| `ProductRating` | float64 | 0 | 7,951 |
| `ProductReviewsCount` | float64 | 0 | 83,821 |
| `XRatXRev` | float64 | 0 | 19,477 |
| `ReviewsCount` | int64 | 0 | 0 |

Nothing numeric is lost. But a consumer reading the export as text gets different
strings than the candidate partition holds, and v1 described that as no change
at all.

**Verdict: WARN.** Needs a confirmation that the export schema is intended.

### 3.3 Rows removed, and what the evidence shows

**v1 said** the 151 training rows with no product text were corrupted. **That
was an overclaim.** Missing text is the condition rule 4 names; it is not
evidence that a row is damaged. A product record can simply be sparse.

Corrected breakdown:

| | Training (614) | Target (212) |
|---|---|---|
| Demonstrable displacement — a value sitting in the wrong column | 462 | 169 |
| No product text, no displacement signal | 151 | 42 |
| Neither | 1 | 1 |
| **Carried at least one label** | **613 (99.8%)** | 0 |

**The labelled count is the part worth noticing.** This is not unlabelled junk
being swept out: 613 of the 614 removed training rows carried a label, broken
down as Brand 611, Mnfr 581, Segment 558, Sub-Segment 537, TargetAgeGroup 536,
and **Platform 11**. Platform has roughly 1,100 labelled rows in the whole
dataset, so that is about 1% of the project's scarcest target.

The removals are still justified — 462 of the 614 show a value sitting in the
wrong column — but the cost is in labelled data, which is the resource the
project has least of. Worth stating rather than leaving in a count of rows.

**What is demonstrated:** 462 training and 169 target rows hold a value of the
wrong kind for its column — text in a numeric field, a URL in the timestamp.
That is displacement, shown by the row's own content.

**What is asserted rather than demonstrated:** that the remaining rows should be
removed. Rule 4 says a record with no name, description or contents is
quarantined. That is a product decision the team has already recorded, and this
audit confirms the rule fired where its condition held — not that the rows were
corrupt.

**Verdict: WARN** for both datasets, because that distinction should be visible
rather than buried in a PASS.

### 3.4 Row identity — now actually checked

For both datasets the rows absent from the candidate file are **exactly** the
rows the decision table and the quarantine partition name. Symmetric difference
is zero in every direction, and every raw row has a row in the decision table:
84,552 and 28,082 respectively.

v1 compared 614 against 614 and called it reconciled. Two disjoint sets of 614
would have passed that test.

### 3.5 Structural integrity — rewritten

**Quoted records are included.** 1,024 training and 353 target raw rows contain
a quote in `ProductName`; 562 and 184 of them are retained, and all are inside
the checks below. Excluding them is how a displacement audit misses displacement.

Each column is checked against its own domain, over every retained row:

| Check | Training | Target |
|---|---|---|
| Numeric columns hold numbers | 0 violations | 0 |
| URL columns hold URLs | 0 | 0 |
| `MDM_InsertDateTime` holds no URL | 0 | 0 |
| `MDM_InsertDateTime` is numeric | 0 | 0 |

**Shared vocabulary is reported, not treated as corruption.** Three values appear
in two targets in the training data:

| Value | Appears in | And in |
|---|---|---|
| `Allergy` | Platform (19) | Segment (2,723) |
| `Children` | Platform (72) | TargetAgeGroup (3,705) |
| `Other Self Care` | Segment (1,132) | Sub-Segment (100) |

All three are distinct concepts sharing a name — a product line called Allergy is
not a Segment. v1's check did not surface them at all.

### 3.6 Rare labels with stray spaces — evaluated, not skipped

v1 excluded whitespace-padded labels from rare-class retention by rule. That
assumed the conclusion: a padded label can still be a real class.

v2 evaluates each one. For all 18 padded values across the six targets, stripping
the whitespace yields a string that is **not** a class in that target — they are
fragments of marketing copy (`" Keto certified"` as an `Mnfr` value). The outcome
is the same as v1's; the difference is that it is now supported.

Separately, no class is lost in either dataset. The two labels that disappear
from training, `Cold / Flu` (4 rows) and `Other Lifestyle` (3), are relabels
recorded row by row in `label_changes.csv`. Every one-row class survives.

---

## 4. Final answer: does the cleaned dataset follow the cleaning rules?

**Yes**, with two behaviours that are correct but undocumented.

All ten policies in `docs/decisions.md` hold. The two WARNs in §3.1 and §3.2 are
not violations: `ProductCategory` formatting and the export dtype change are both
things the pipeline does deliberately and the handoff README describes, but
neither appears in the decision log.

What changed from v1 is not the answer. It is that the answer is now supported by
checks that cover the whole population and apply to both datasets.

---

## 5. What needs a decision

| Item | Needs |
|---|---|
| `ProductCategory` separator formatting | A line in `docs/decisions.md` approving it |
| Export dtype change | Confirmation that the export schema is intended |
| Rows removed for missing text only | Confirmation that rule 4 is the intended product behaviour, since the rows are sparse rather than demonstrably damaged |

---

## 6. A correction to v1 on identity columns

v1 wrote that the identity columns do not survive the feature export and that a
grouped split therefore cannot be built. **That was wrong as stated.**

| Column | Candidate partition | Export |
|---|---|---|
| `source_row`, `dataset`, `source_sha256` | present | **present** |
| `JoiningKey`, `Sku`, `Upc`, `MDM_Id` | present | absent |

Row identity **is** preserved in the export, through `source_row`, which is
unique across all 83,938 rows. The business keys are the ones that do not
survive. Since `source_row` joins the export back to the candidate partition,
`Sku` and `JoiningKey` are recoverable with one join.

The practical consequence is smaller than v1 claimed: building a grouped split
requires joining back to the candidate partition, not that it is impossible. The
underlying problem is unchanged and is not about the export at all — `JoiningKey`
is unique per row and `Sku` nearly so, so neither groups product variants. See
§2.7 of the evaluation strategy.

---

## 7. A note on auditing this pipeline

Four checks in this audit appeared to fail and did not. In every case the check
was too crude, not the pipeline:

- Rows "invented" by cleaning — an artifact of hashing content instead of using
  the provenance key.
- `Cold / Flu` "deleted" — a recorded relabel whose destination label had
  simultaneously lost rows to quarantine, so a net count read as a loss.
- `keep_candidate` disagreeing with the candidate file — the column holds `1`/`0`,
  and the test compared against `true`/`false`.
- A path level "lost" — the removed text was a verbatim duplicate of the path.

A policy audit that stops at the first surprising count will report failures that
are not there. Each of these is in the notebook as a guard so the next run does
not repeat them.

---

## 8. Open questions

1. **Should the `ProductCategory` formatting be recorded in `decisions.md`?** (§3.1)
2. **Is the export dtype change intended?** (§3.2)
3. **Should rows with no product text be removed**, given that missing text is
   sparsity rather than demonstrated damage? (§3.3)
4. **Is `Category` dropped from the schema**, given it holds one value?
