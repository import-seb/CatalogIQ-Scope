# Cleaning Validation & Data Quality Audit

**Version 3**, after the second review round. Supersedes v2.
**Result:** 16 PASS, 4 WARN, 1 BLOCKED, 0 FAIL, plus 4 diagnostics, across **both** datasets.
**Location:** `docs/validation/cleaning_validation.md`
**Notebook:** `notebooks/02-cleaning-validation-audit.ipynb`
**Run audited:** the local reproduction created by the commands in §2 (`my_cleaning_run`), of
`integrated_policy_v4_final_20261003` (`--mode integrated --exclude-policy keep`),
83,938 training and 27,870 target candidates.
**Owner:** Maria Garcia Sehara

---

## 1. What changed, and why

v2 fixed how the checks were **scoped**. This round fixes what two of them
actually **measured**. Two v2 findings are withdrawn.

| Review finding | v2 | v3 |
|---|---|---|
| The category test removed any repeated path element | `dict.fromkeys` on the path parts, so `A>B>A>C` → `A>B>C` would have passed | Collapse allowed only when the **whole path** repeats exactly |
| The dtype finding measured pandas inference, not the files | "111,249 values change representation" | Both files read as text; **0** stored values differ. Finding withdrawn |
| Disappearing labels were excused by a whitespace rule | A padded label was skipped by rule | Every value traced to its **raw rows**; whitespace excuses nothing |
| Quarantine reasons were only checked for presence | "All 614 removed rows name a rule" | Each reason's documented conditions **re-derived from the original row** |
| Isolated bad values weighed the same as policy breaches | A single bad URL could drive a FAIL | FAIL only when the pipeline's **action** breaks a rule; the rest are diagnostics |

### The two withdrawn findings

**§3.2 of v2 is wrong and is withdrawn.** v2 reported that the export changes
the textual form of 111,249 values, `5` becoming `5.0`. That number was produced
by pandas inferring `float64` inside the check and the check then stringifying
the result. Read with `dtype="string"` and `keep_default_na=False` and joined on
`dataset` + `source_sha256` + `source_row`, the candidate partition and the
export store **byte-identical values** in all four measurement columns across
all 83,938 rows. Nothing changed. The raw file already stores `50`, `5`, `46`,
and so does the export.

**§3.1 of v2 was right by accident.** The conclusion holds, but the test did not
support it. Under the strict rule the answer is the same — 0 unexplained rows —
so the number did not move. The check is now sound instead of accidentally
correct, which is the part that matters.

---

## 2. Results

### Training

| Criterion | Status | Coverage |
|---|---|---|
| Row counts and identity | PASS | 84,552 / 84,552 |
| Quarantine reasons recorded | PASS | 614 / 614 |
| **Quarantine conditions re-derived from raw** | **WARN** | 614 / 614 |
| Missingness | PASS | 83,938 / 83,938 |
| Candidate vs export, stored values | PASS | 83,938 / 83,938 |
| Target distributions | PASS | 84,552 / 84,552 |
| Disappearing labels | PASS | 84,552 / 84,552 |
| Malformed labels | PASS | 84,552 / 84,552 |
| Structural integrity | PASS | 83,938 / 83,938 |
| **Feature values on retained rows** | **WARN** | 83,938 / 83,938 |
| Information loss | PASS | 614 / 614 |

### Target

| Criterion | Status | Coverage |
|---|---|---|
| Row counts and identity | PASS | 28,082 / 28,082 |
| Quarantine reasons recorded | PASS | 212 / 212 |
| **Quarantine conditions re-derived from raw** | **WARN** | 212 / 212 |
| Missingness | PASS | 27,870 / 27,870 |
| Disappearing labels | PASS | 28,082 / 28,082 |
| Malformed labels | PASS | 28,082 / 28,082 |
| Structural integrity | PASS | 27,870 / 27,870 |
| **Feature values on retained rows** | **WARN** | 27,870 / 27,870 |
| Information loss | PASS | 212 / 212 |
| Target distributions | BLOCKED | — |

Machine-readable: the notebook writes `cleaning_audit_summary.csv` and
`cleaning_audit_diagnostics.csv` to `data/interim/`. **Neither is in the repo** —
`/data/interim/` is gitignored, like the rest of the generated data — so running
the notebook is what produces them. The tables above and in §3 are the committed
record.

**Reproduced independently.** This run was produced twice, on two machines, from
the same raw CSVs and the same pipeline commands: 16 PASS, 4 WARN, 1 BLOCKED,
0 FAIL, 4 diagnostics, with identical coverage figures, the same 12 and 2 rows
carrying an unsupported reason code, and the same 11 Platform `source_row`
values.

### What the statuses mean now

v2 used FAIL for anything surprising, which put a single bad URL on the same
footing as a broken policy. v3 separates them:

| Status | Means |
|---|---|
| PASS | The action complies with a rule in `decisions.md` or `structural_check.md`, verified over the stated population |
| WARN | The action complies, but it is not in `decisions.md`, or the reason code it recorded is not satisfied by that code's own wording |
| FAIL | The action **violates** a documented rule |
| BLOCKED | Nothing to compare |
| diagnostic | A questionable value or a cost. Not a verdict, and never a FAIL on its own |

---

## 3. The findings

### 3.1 Every quarantine condition re-derived from the original row

§2 of v2 only checked that a reason was *present*. That is not an audit — the
pipeline could record whatever it liked. v3 takes each recorded reason code and
re-evaluates its documented conditions from `docs/structural_check.md` against
the raw CSV row.

| Reason code | Training | Target | Conditions hold |
|---|---|---|---|
| `missing_product_text` | 151 | 42 | 193 / 193 |
| `text_in_multiple_numeric_anchors` | 305 | 109 | 414 / 414 |
| `url_in_timestamp_with_displacement` | 263 | 93 | 356 / 356 |
| `unexpected_exclude_with_populated_unknown_columns` | 3 | 2 | 5 / 5 |
| `multiple_suspicious_targets` | 3 | 0 | 3 / 3 |
| `text_numeric_failure_with_corroboration` | 157 | 60 | **203 / 217** |

**All 826 removals are supported.** Every removed row has at least one recorded
reason whose documented conditions hold on its own raw content.

**14 rows carry one code that its own wording does not cover.** 12 training and
2 target rows record `text_numeric_failure_with_corroboration`, whose documented
corroboration list is *"another invalid measurement or displaced
identifier/invalid URL field"* — a URL in the timestamp is not on that list. On
those 14 rows the only corroborating signal present **is** the URL in the
timestamp.

The same 14 rows also record `url_in_timestamp_with_displacement`, whose
documented condition is *"HTTP(S) URL in timestamp plus numeric text"* — and
numeric text is present. So the removal is justified; it is the reason labelling
that is loose. The two rules corroborate each other in one direction only.

**Verdict: WARN, not FAIL.** The pipeline's action is correct. The wording in
`structural_check.md` should say whether a URL in the timestamp counts as
corroboration for the text-numeric rule, since it already counts the other way.

### 3.2 Candidate against export — the finding that is withdrawn

Read with `dtype="string"`, `keep_default_na=False`, joined on
`dataset` + `source_sha256` + `source_row`, 83,938 of 83,938 export rows matched:

| Column | Stored values that differ |
|---|---|
| `ProductRating` | 0 |
| `ProductReviewsCount` | 0 |
| `XRatXRev` | 0 |
| `ReviewsCount` | 0 |

Zero, and zero unparseable. Pandas infers the **same** type for both files in
every one of those columns.

**Verdict: PASS.** v2's §3.2 WARN is withdrawn, and so is the open question that
went with it. There was nothing to confirm about the export schema, because the
export does not change these values.

**Recorded as a diagnostic, not a finding:** reading any of these files without
an explicit dtype turns a stored `5` into the float `5.0` in memory, so anything
that stringifies a parsed column will see `5.0`. Any consumer that needs the
stored text must pass `dtype="string"` and `keep_default_na=False`. That is a
property of the reader. It is what v2 measured and mistook for a pipeline
change.

### 3.3 The category transform, under the strict rule

| Raw | Candidate | Allowed |
|---|---|---|
| `A > B > C` | `A>B>C` | yes — separator spacing |
| `A > B > A > B` | `A > B` | yes — the **entire** path repeats |
| `A > B > A > C` | `A > B > C` | **no** — a level is lost |

| | Training | Target |
|---|---|---|
| Rows compared | 83,938 / 83,938 | 27,870 / 27,870 |
| `ProductCategory` changed | 7,714 (9.19%) | 2,586 (9.28%) |
| Separator spacing only | 7,713 | 2,585 |
| Entire path repeated, collapsed | 1 | 1 |
| **Unexplained** | **0** | **0** |

The two full-repeat rows are `source_row` 83930 (training) and 20318 (target).
Both are the same path written twice end to end:

```
Health & Household > ... > Licorice Root > Health & Household > ... > Licorice Root
Health & Household > ... > Licorice Root
```

**Verdict: WARN, not FAIL.** No path level is lost, so policy 10 holds. The WARN
is that the transform appears only in the handoff README — *"Fixed safe
spacing/category formatting"* — and not in `docs/decisions.md`. It needs a
recorded decision, not a fix.

### 3.4 Disappearing labels, traced to their rows

v2 skipped a disappearing label if it was whitespace-padded. That explained
nothing about where its rows went. v3 compares **exactly**, strips nothing
first, and requires each original row to be either quarantined under a reason
verified in §3.1 or relabelled with the rewrite recorded per row.

15 values are present in raw and absent from the training candidate:

| Target | Value | Rows | Disposition |
|---|---|---|---|
| Sub-Segment | `Cold / Flu` | 4 | relabelled → `Cold/Flu`, rule `slash_spacing` — **approved in decisions.md** |
| Sub-Segment | `Other Lifestyle` | 3 | relabelled → `Other Lifestyle CHC`, rule `reviewed_other_lifestyle` — **approved in decisions.md** |
| Mnfr | `' coffee'` | 1 | quarantined |
| Mnfr | `' aids in collagen formation* GRASS-FED…'` | 2 | quarantined |
| Brand | `' tea'`, `' gluten-free'` | 1, 2 | quarantined |
| Platform | `' grass fed'`, `' orange juice or a smoothie…'` | 2, 1 | quarantined |
| Segment | `' Keto certified'`, `' IGEN Non-GMO tested.*…'` | 1, 2 | quarantined |
| Sub-Segment | `' gluten-free'`, `' lactose'` | 1, 2 | quarantined |
| TargetAgeGroup | `' Paleo friendly'` and two `' starch or artificial flavors…'` | 1 each | quarantined |

**The 13 junk values come from only three rows:** `source_row` 21528, 43277 and
70021. Those three had their fields displaced, so marketing copy from the product
description landed in the label columns. They are not 13 lost classes; they are
three broken rows seen from six columns.

**Separately, no label was quietly rewritten.** Comparing raw against candidate
on retained rows, exactly: 7 label values differ in the whole training dataset,
and all 7 have a matching row in `label_changes.csv`. Every other target column
is 0 changed. The target dataset is 0 changed.

**Verdict: PASS**, for both datasets. No class is lost, no surviving class loses
more than 20% of its rows, and nothing is excused by whitespace.

**Worth knowing about the mechanism:** the pipeline does **not** strip label
padding. Raw training holds 18 whitespace-padded label values and the candidate
holds 0 — but that is because all 18 belong to those three quarantined rows, not
because anything was cleaned. The target dataset has 0 in both, since it carries
no labels. A padded label on an otherwise healthy row would survive, and under
`decisions.md` that is correct: spacing warnings are review-only. It is still a
modelling hazard and belongs in feature engineering.

### 3.5 Rows removed, and what it cost

§3.1 settles the policy question, so this is only about the cost.

| | Training (614) | Target (212) |
|---|---|---|
| A value sitting in the wrong column | 462 | 169 |
| No product text at all | 151 | 42 |
| Neither of the above | 1 | 1 |
| **Carried at least one label** | **613** | 0 |

Per target, removed against the number labelled in raw:

| Target | Removed | Labelled in raw | Share |
|---|---|---|---|
| Brand | 611 | 84,230 | 0.73% |
| Mnfr | 581 | 78,854 | 0.74% |
| Segment | 558 | 79,763 | 0.70% |
| Sub-Segment | 537 | 77,037 | 0.70% |
| TargetAgeGroup | 536 | 77,475 | 0.69% |
| **Platform** | **11** | **1,105** | **1.00%** |

Five of the six targets lose a consistent 0.69–0.74% of their labelled rows,
which is the removal rate for the dataset as a whole (0.73%). Platform is the
outlier at 1.00%, and it is also the smallest by a wide margin — 1,105 labelled
rows against 77,000–84,000 for the others.

**The 11 Platform rows, individually.** Platform is the scarcest of the six
targets, so they are listed rather than counted:

| `source_row` | Platform | Reason |
|---|---|---|
| 13600 | Eight Hour Arthritis Pain | `missing_product_text` |
| 16558 | Throat | `missing_product_text` |
| 18693 | Eight Hour Arthritis Pain | `missing_product_text` |
| 21528 | `' orange juice or a smoothie…'` | displacement, 4 codes |
| 24303 | Sleep | `missing_product_text` |
| 30769 | 24 Hour Relief | `missing_product_text` |
| 43277 | `' grass fed'` | displacement, 4 codes |
| 45709 | Imodium A-D | `missing_product_text` |
| 62553 | Eight Hour Arthritis Pain | `missing_product_text` |
| 70021 | `' grass fed'` | displacement, 3 codes |
| 74735 | Sleep | `missing_product_text` |

Per class: Eight Hour Arthritis Pain loses 3 of 43 (7.0%), Sleep 2 of 35 (5.7%),
Imodium A-D 1 of 32 (3.1%), Throat 1 of 40 (2.5%), 24 Hour Relief 1 of 46
(2.2%). The three displacement rows are not real Platform classes. **No Platform
class is wiped out**, and the worst single class loses 7%.

**The eight rows removed under `missing_product_text` are the ones worth a
decision.** Each has a genuine Platform label and no name, description or
contents. The rule fired exactly where its condition held, so the pipeline is
correct. Whether a labelled row with no product text is worth keeping for a
Platform model is a product call, not an audit call.

**Verdict: PASS**, with the cost recorded as a diagnostic rather than folded
into the verdict. v1 called these rows corrupted; that stays withdrawn — missing
text is rule 4's condition, and a sparse record is not a damaged one.

### 3.6 Structural integrity, with diagnostics separated

Over all retained rows, including the 562 training and 184 target rows that
contain a quote in `ProductName`:

| Check | Training | Target |
|---|---|---|
| Measurement columns hold numbers | 0 violations | 0 |
| URL columns hold URLs | 0 | 0 |
| `MDM_InsertDateTime` holds no URL | 0 | 0 |
| `MDM_InsertDateTime` is numeric | 0 | 0 |

**Verdict: PASS.** Had any of these been non-zero, only the measurement-column
one would be a FAIL. Policy 3 makes a bad timestamp alone review-only and policy
5 makes a bad URL alone review-only, so finding one of those among retained rows
is the pipeline obeying policy, not breaking it. v2 would have called it a FAIL.

**Diagnostic, not a finding:** three values appear in two targets — `Allergy` in
Platform (19) and Segment (2,723), `Children` in Platform (72) and
TargetAgeGroup (3,705), `Other Self Care` in Segment (1,132) and Sub-Segment
(100). Distinct concepts sharing a name. `decisions.md` forbids enforcing a
parent whitelist, so there is nothing for cleaning to do.

---

## 4. Final answer: does the cleaned dataset follow the cleaning rules?

**Yes.** 0 FAIL across 21 criteria on both datasets, and every PASS covers the
full population it claims.

Stated more strongly than v2 could: all 826 removals were re-derived from the
original CSV rows against the documented conditions, not taken from the
pipeline's own record of why it removed them. Every disappearing label was traced
to its rows. The export was compared to the candidate as stored bytes.

The four WARNs are two issues, each appearing once per dataset, and neither is a
rule violation:

1. `ProductCategory` separator formatting is real, benign and documented by the
   preparer, but absent from `docs/decisions.md`.
2. 14 rows carry a reason code that the code's own wording does not cover, while
   a second code on the same rows does cover the removal.

---

## 5. What needs a decision

| Item | Needs | Changed since v2 |
|---|---|---|
| `ProductCategory` separator formatting | A line in `docs/decisions.md` approving it | unchanged |
| `text_numeric_failure_with_corroboration` wording | State in `structural_check.md` whether a URL in the timestamp corroborates it, since it already corroborates the other direction | **new** |
| 8 labelled rows removed for missing text only | Confirmation that rule 4 is the intended behaviour for a row with a real label but no product text | narrowed from 151 to the 8 that cost a Platform label |
| Export dtype change | — | **withdrawn**, §3.2 |

---

## 6. A note on auditing this pipeline

Across three versions, six of this audit's own checks reported something that was
not there. Not one was a pipeline defect:

- Rows "invented" by cleaning — hashing content instead of using the provenance key.
- `Cold / Flu` "deleted" — a recorded relabel whose destination had simultaneously lost rows.
- `keep_candidate` disagreeing — the column holds `1`/`0`, the test compared `true`/`false`.
- A path level "lost" — the removed text was a verbatim duplicate.
- **"111,249 values changed representation"** — pandas inference inside the check.
- **A PASS over 76,219 of 83,938 rows** — a figure from a superseded run that was never updated.

The last two are the ones that got published. The pattern in all six is the same:
a check that measures something adjacent to the question, and a number that was
never traced back to the file it supposedly came from. Each is now a guard in the
notebook, and the verdict recorder downgrades any PASS whose coverage falls short
of the population it claims.

---

## 7. Open questions

1. **Should the `ProductCategory` formatting be recorded in `decisions.md`?** (§3.3)
2. **Does a URL in the timestamp corroborate `text_numeric_failure_with_corroboration`?** (§3.1)
3. **Should a row with a real label but no product text be removed?** Eight of
   the 11 lost Platform rows are exactly this case. (§3.5)
4. **Is `Category` dropped from the schema**, given it holds one value?
