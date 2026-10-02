# Model Evaluation Strategy — CatalogIQ Classification Targets

**Status:** v3 — figures verified against data, not the profile
**Scope:** how to evaluate the six classification targets
**Location:** `docs/research/`
**Data basis:** `training_cleaned.csv` from the conservative handoff
(`conservative_handoff_20261002`), 78,718 rows of an original 84,552. Supersedes
v2, which was written from the dataset profile. Claims that changed are marked
**CORRECTED**.

---

## 1. Summary

The six targets should not share one scorecard. **Accuracy is not the primary
metric for any of them**, and the measured majority-class baselines show why:

| Target | Labelled rows | Classes | Majority class | Accuracy from predicting it alone |
|---|---|---|---|---|
| Mnfr | 74,532 | 2 | All others | **98.63%** |
| TargetAgeGroup | 74,527 | 3 | Adult | **94.25%** |
| Brand | 78,718 | 144 | UnItemised brand | **66.11%** |
| Segment | 76,927 | 8 | Vitamins, Minerals & Supplements | **63.45%** |
| Sub-Segment | 74,518 | 45 | Supplements | **26.02%** |
| Platform | 1,092 | 60 | Extra Strength | **7.69%** |

A model that always answers "All others" scores 98.63% on Mnfr and finds zero
J&J products. That is the whole argument.

Recommended primary metric per target:

| Target | Shape of the problem | Primary metric |
|---|---|---|
| Mnfr | Binary, 1.37% positive | Recall and precision on J&J; PR-AUC |
| Brand | 141 named classes under a 66% placeholder | Macro-F1 over named brands only |
| Platform | 60 classes, 1,092 labelled rows | Not evaluable as-is — see §3.3 |
| Segment | 8 classes, 63.45% majority | Macro-F1 + full confusion matrix |
| Sub-Segment | 45 classes, support 19,392 to 2 | Macro-F1 by support tier |
| TargetAgeGroup | 3 ordered classes, 94% Adult | Macro-F1 + per-class recall |

---

## 2. Data conditions that affect every metric

### 2.1 Malformed rows — RESOLVED

The profile reported 3 malformed, whitespace-prefixed values per target. **Zero
remain** in the cleaned handoff, verified in
`docs/validation/cleaning_validation.md` §3.8. `Mnfr` now carries exactly 2
values rather than 4.

### 2.2 Placeholder classes are not classes

Three Brand values are not brands:

| Value | Count | Share of labelled |
|---|---|---|
| UnItemised brand | 52,041 | 66.11% |
| Other Brands | 576 | 0.73% |
| Generic | 506 | 0.64% |

**25,595 rows carry a named brand across 141 classes.**

**Recommendation unchanged:** report Brand metrics on named brands only, and
report placeholder performance separately. Including "UnItemised brand" in the
macro average rewards a model for learning to say "I don't know" in the most
common case.

The same framing applies to Mnfr, where "All others" is a catch-all. The model is
not learning what "All others" means; it is learning to detect J&J.

### 2.3 Near-duplicate labels — RESOLVED, provenance unknown

All three pairs flagged in v2 are now single labels: `Cold / Flu`,
`Creams & Gels (Rubs)` and `Other Lifestyle` have 0 rows.

**Whether their rows merged into the surviving label or were deleted is not
known** — in each case the survivor is smaller than in the profile, not larger.
This is BLOCKED in the cleaning audit pending the raw file. For evaluation
purposes the class count is settled at 45 Sub-Segments; if those rows were
deleted rather than merged, nothing in the metrics changes, but the cleaning
record does.

### 2.4 Target dataset dtype — RESOLVED

The cleaned target set loads correctly and no classification column reads as
`float64` in the cleaned training file. The original trap — empty columns
inferring as float and making a label join fail silently — does not apply to the
handoff. Still load identifiers as string to preserve leading zeros, per the
handoff README.

### 2.5 Train and target are the same population — CONFIRMED

Feature missingness is near-identical across the two cleaned files:

| Feature | Training | Target |
|---|---|---|
| ProductModelNumber | 86.62% | 86.42% |
| Upc | 81.03% | 80.89% |
| ProductDescription | 68.58% | 68.84% |
| ProductContents | 59.87% | 59.57% |
| ProductName | 0.00% | 0.00% |
| ProductCategory | 0.01% | 0.01% |

This supports the assumption that the target set is drawn from the same
population, which is what makes training-set evaluation meaningful at all. State
it explicitly in the final report; it is the assumption everything rests on.

### 2.6 Splitting — CORRECTED: there is no grouping key

v2 named this the highest-priority open question and proposed splitting on
`JoiningKey`. **That option does not exist.**

| Candidate | Missing | Unique values | Rows per key |
|---|---|---|---|
| `JoiningKey` | 0.00% | 78,718 | **1.00** |
| `Sku` | 0.00% | 78,298 | 1.01 |
| `Upc` | 81.03% | 10,877 | 1.37 |

`JoiningKey` is a row identifier. `Sku` is effectively one too. Neither groups
product variants, so neither protects against near-duplicate leakage.

**The risk is unchanged and now unmitigated.** Product catalogues carry
near-duplicates — size and flavour variants of the same item — and a random row
split puts near-identical products in both train and test, inflating every score
in a way no metric reveals.

**Revised action.** Build a grouping key before any model is scored. Candidate:
normalised `ProductName` (lowercase, strip punctuation and size/count tokens)
combined with `Brand`. Validate it by inspecting the largest resulting groups by
hand — if they are variants of one product, the key works. This is now a task,
not a question, and it blocks every reported number.

### 2.7 NEW — 5,834 rows are missing and nobody can say which

The cleaned handoff excludes 466 quarantined and 5,368 review-hold rows, 6.90% of
the original. The files recording which rows and why were not shipped.

**Effect on evaluation.** Every figure in this document describes the cleaned
subset. If removal was uneven across classes, the measured baselines and support
tiers are biased, and there is currently no way to check. The 5,368 review-hold
rows are recoverable, which means the usable training set may grow.

Do not publish a final metric until §3.2 of `docs/validation/cleaning_validation.md`
is closed.

---

## 3. Per-target recommendations

### 3.1 Mnfr — binary, 1.37% positive

Labelled 74,532. J&J 1,020 (1.37%). All others 73,512. 4,186 unlabelled.

Not a multiclass problem. Treat J&J as the positive class and report:

- **Precision, recall and F1 on J&J** — the primary numbers
- **PR-AUC** — the right curve for a 1.37% positive rate
- **Confusion matrix in raw counts**, not percentages

**Do not use ROC-AUC.** With 73,512 negatives the false positive rate barely
moves, and ROC-AUC will look strong while the model misses most J&J products.

**NEW — the baseline is now a hard rule, not a heuristic.** Brand → Mnfr is
verified deterministic across all 144 Brand values
(`classification-relationships-and-hierarchy.md` §4.3). **Mnfr is fully derivable
from Brand.** A Brand-to-Mnfr lookup is the baseline, and it should score
perfectly on rows where Brand is known. Any Mnfr model must beat it or justify
its cost; realistically, Mnfr should not be modelled separately at all.

The threshold question — which is worse, a missed J&J product or a false flag —
still belongs to the PO.

### 3.2 Brand — 141 named classes, 66% placeholder

Report:

- **Macro-F1 over named brands only** (primary)
- **Support tiers**, measured:

| Tier | Classes | Rows |
|---|---|---|
| ≥250 | 31 | 13,561 |
| 50–249 | 84 | 11,527 |
| 10–49 | 16 | 471 |
| <10 | 10 | 36 |

Top: NOW (1,239), Nature Made (878), Swanson (809). Bottom: Hers, Health & Her,
Meno, Womaness at 1 row each.

- **Top-3 accuracy.** For a human reviewer choosing from three suggestions this
  is a real product outcome and should be measured deliberately.
- **Placeholder handling reported separately:** how often does the model
  correctly predict "UnItemised brand", and how often does it hide behind it?

**NEW — the near-leakage is confirmed and large.** `ProductBrand` matches `Brand`
exactly **91.79%** of the time on the 25,595 named-brand rows (30.49% across all
rows, dragged down by the placeholder).

This changes what a high Brand score means. **Much of this task is string
normalisation, not classification.** That is not a problem, but it must be stated,
and the lookup must be the baseline:

> Map `ProductBrand` → `Brand` directly. Any model scoring below ~92% macro
> accuracy on named brands is worse than a dictionary.

Expect confusion between brands under the same manufacturer. Pull the top 20
confused pairs and read them by hand before concluding the model is wrong; some
will be label inconsistencies.

### 3.3 Platform — still not evaluable as-is, for a revised reason

- 1,092 labelled rows of 78,718 (**98.61% missing**)
- **60 valid classes**
- Largest class: Extra Strength, 84 rows
- **6 classes have exactly 1 row**
- **27 of 60 classes have fewer than 10 rows**

**CORRECTED — the scope hypothesis is supported but overstated.** v2 said
Platform appeared to be labelled only for one manufacturer's products. Measured:
1,020 of 1,092 Platform rows are J&J, **93.4% — not exclusive**. The remaining 72
are Advil, Excedrin, Kirkland Signature, Visine and others.

What this changes:

- Platform is still overwhelmingly a J&J-labelled field, so **the scope
  conversation with the PO remains the right next step** rather than a request
  for more labels.
- But a Platform prediction on a non-J&J product is **unusual, not meaningless**.
  The abstention argument is weaker than v2 claimed, and rule R4 must not block.

**Stratified k-fold cross-validation remains impossible** on classes with a
single example. Any macro-F1 over all 60 classes is structurally capped well
below 1, and the number says more about the split than the model.

Three options, in order of preference:

1. **Reduce the class count.** Several labels group naturally (`Sinus`,
   `Sinus Plus`, `Sinus Severe`). A coarser taxonomy may make the problem
   tractable. A question for the PO, not a modelling decision.
2. **Evaluate on a support-filtered subset.** Macro-F1 over classes with ≥10
   examples — 33 of 60 — reporting the remaining 27 as out-of-scope with counts.
3. **Ship Platform as review-only** in the first release: the model suggests, a
   human always confirms. Given the evidence this is a defensible outcome, not a
   failure.

Whichever is chosen, include a **learning curve** (train on 25/50/75/100% of the
1,092 labelled rows). If the curve is still climbing at 100%, more labelling is
the highest-value next step and the curve estimates how much. That is a concrete
ask for the PO.

### 3.4 Segment — 8 classes, tractable

Labelled 76,927 across 8 classes, from Vitamins, Minerals & Supplements (48,812)
to Other Self Care (1,117). A 44:1 spread — imbalanced, but every class has
enough data to learn.

Macro-F1 as primary, and **include the full 8×8 confusion matrix**. At this size
it is readable, and it matters more than the score because Segment errors
propagate: a row misassigned at Segment level is wrong at Sub-Segment level
regardless of how good the Sub-Segment model is.

### 3.5 Sub-Segment — 45 classes, extreme support spread

Labelled 74,518. Support runs from Supplements (19,392) to
`Daily oral contracception` (2).

**Measured support tiers:**

| Tier | Classes | Rows |
|---|---|---|
| ≥250 | 31 | 73,298 |
| 50–249 | 7 | 1,106 |
| 10–49 | 5 | 109 |
| <10 | 2 | 5 |

The two classes below 10: `Dermatologicals` (3) and `Daily oral contracception`
(2). **Report these in their own table** rather than averaged into the headline,
where a single prediction swings their F1 between 0 and 1.

**Hierarchical consistency — CORRECTED.** v2 proposed measuring how often the
predicted Sub-Segment is valid under the predicted Segment, treating a rate below
100% as a defect. **The hierarchy is not strict**: `Internal Pain` and
`Probiotics` each have two parents. The measure is still worth reporting, but the
target is "matches the verified parent-child map", not 100%, and the map is in
`interim/segment_subsegment_map.csv`.

Report two views:

- Sub-Segment accuracy overall (end-to-end quality)
- Sub-Segment accuracy given Segment was predicted correctly (isolates whether
  the Sub-Segment model is weak or inheriting upstream errors)

### 3.6 TargetAgeGroup — 3 ordered classes, 94% Adult

Labelled 74,527. Adult 70,242 (94.25%), Children 3,548 (4.76%), Infant 737
(0.99%).

The taxonomy is exactly Adult, Children, Infant — no `All Ages` or `Unknown`
appear in the cleaned data — so the classes are ordered.

With only three classes **adjacent accuracy adds little**, since only one pair is
two steps apart. Use instead:

- **Per-class recall on Children and Infant** — the primary numbers. Infant at
  0.99% is where this target will fail, and where an error matters most in a
  health products catalogue.
- **Macro-F1** as the summary.
- **The full 3×3 confusion matrix**, which shows directly whether Infant is being
  absorbed into Children or into Adult.

---

## 4. Cross-cutting evaluation

### 4.1 Overall performance and baselines

One results table per target: macro-F1, weighted-F1, balanced accuracy, and
accuracy last, labelled as context only.

**Every target needs baselines**, or a score has no meaning:

- **Majority-class baseline** — the floor, measured in §1. Anything at or below
  it is worthless no matter how the number reads.
- **Lookup baseline** — for Brand, `ProductBrand` → `Brand` (~92% on named
  brands, §3.2). For Mnfr, `Brand` → `Mnfr` (deterministic, §3.1). For Segment
  and Sub-Segment, the `ProductCategory` breadcrumb — though that covers only
  9.5% of rows at ≥90% purity, so it is a partial baseline.
- **Simple text baseline** — TF-IDF over `ProductName` plus a linear classifier.
  `ProductName` is 0.00% missing, so it is available everywhere, unlike
  `ProductDescription` (68.58% missing).

**An approach that does not clearly beat the lookup and TF-IDF baselines is not
worth its operating cost.** For Brand and Mnfr the lookups are now measured, and
they are strong. The evaluation must be able to show whether a model beats them.

### 4.2 Minority classes

**No score is reported without its support count**, and classes below the agreed
floor are reported in their own tier rather than averaged into the headline.

### 4.3 Confidence and calibration

A model's output score is not a probability until checked. This matters more than
usual here, because the README makes confidence a product requirement: every
classification must carry a confidence and supporting evidence.

An uncalibrated score cannot drive a review threshold. "Flag everything below
0.80" means nothing if 0.80 does not correspond to being right 80% of the time.

Report per target: **reliability diagram**, **expected calibration error (ECE)**,
**Brier score**. Then calibrate — temperature scaling on a held-out calibration
split is the cheap standard option, and it changes only the confidence, not the
predicted class.

Calibration must be measured per target. A model can be well calibrated on
Segment and badly calibrated on Sub-Segment.

### 4.4 Uncertain predictions

The right frame is **selective prediction**: the model answers when confident and
abstains otherwise. This is exactly what the project principle asks for.

Report a coverage-risk table per target:

| Coverage | Accuracy on answered rows |
|---|---|
| 100% | — |
| 90% | — |
| 75% | — |
| 50% | — |

"70% accurate on everything" and "95% accurate on the 60% it is sure about, rest
to review" are different products. Only the second lets the PO pick an operating
point.

Abstention is not an error. Score it separately from a wrong answer.

### 4.5 Human-review candidates

Routing rules worth evaluating:

1. **Low top-1 confidence**, against a calibrated threshold.
2. **Small margin** between top-1 and top-2. Often catches more real errors than
   raw confidence, because it detects genuine ambiguity.
3. **Rare predicted class** — any prediction of a class with tiny training
   support. Given §3.3 and §3.5, this rule alone covers a lot of ground.
4. **Predicted placeholder** — if the model predicts "UnItemised brand" for a row
   whose `ProductBrand` is populated, something is off. Given the 91.79% match
   rate, this is a strong signal.
5. **Hierarchy violation** — predicted Sub-Segment not in the verified
   parent-child map. Note this is now a *review* trigger, not a block, because
   the hierarchy is not strict.

**The review queue needs its own metrics**, or it becomes an unmeasured cost:

- **Flag precision** — of rows sent to review, what share were actually wrong?
- **Miss rate** — of wrong predictions, what share were *not* flagged?
- **Review load** — what percentage of the 26,033 target rows needs a human?
  This is the number that decides whether the system saves money.

Report these across thresholds, not at one fixed cutoff.

### 4.6 Comparing model approaches

- **Fixed splits, saved and version-controlled**, using a constructed grouping
  key (§2.6) — not `JoiningKey`, which does not group.
- **One primary metric per target, declared before results are seen.**
- **Uncertainty on every comparison.** Bootstrap confidence intervals, or
  McNemar's test for two models on the same rows. On Platform especially, a
  several-point macro-F1 difference across 1,092 rows is probably noise.
- **Cost reported alongside accuracy.** A 2-point gain that triples inference
  cost is a business decision, not an automatic win.

---

## 5. Open questions

1. **Will the raw CSVs and quarantine files be shared?** (§2.7) Until then every
   figure here describes a subset that lost 6.90% of its rows for undocumented
   reasons. Highest priority.
2. **What grouping key should the split use?** (§2.6) `JoiningKey` does not work.
   This blocks the validity of every number and is now a build task.
3. **Can the PO supply or approve a coarser Platform taxonomy?** (§3.3)
4. **Is Platform expected on the target set at all?** (§3.3) 98.61% missing.
5. **For Mnfr, which is worse: a missed J&J product or a false flag?** Sets the
   threshold.
6. **What is the decision on the 5,368 review-hold rows?** They are recoverable
   and would change every support count here.
7. **Should `Category` be dropped?** It holds one value, `Self Care`, in all
   78,718 rows — closed as a target question, open as a schema question.

---

## 6. Done-when checklist

| Criterion | Where |
|---|---|
| Recommended metrics documented | §1, §3 |
| Class imbalance addressed | §1, §2.2, §3.1, §3.2, §3.5, §3.6, §4.2 |
| Platform's limited labelled data addressed | §3.3 |
| Supports comparison between model approaches | §4.6, §4.1 baselines |
