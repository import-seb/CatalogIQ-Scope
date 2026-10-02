# Model Evaluation Strategy — CatalogIQ Classification Targets

**Status:** proposal, v2
**Scope:** how to evaluate the six classification targets
**Location:** `docs/research/`
**Data basis:** CatalogIQ Dataset Profile (84,552 training rows × 32 columns;
28,082 target rows). All counts below come from that profile.

---

## 1. Summary

The six targets should not share one scorecard. The profile shows they differ so
much that a single evaluation table would be misleading.

**Accuracy is not the primary metric for any of the six.** The numbers make this
concrete rather than theoretical:

| Target | Majority class | Share of labeled rows | Accuracy from predicting it alone |
|---|---|---|---|
| Mnfr | All others | 77,821 / 78,851 | 98.7% |
| TargetAgeGroup | Adult | 72,995 / 77,472 | 94.2% |
| Brand | UnItemised brand | 54,059 / 84,227 | 64.2% |
| Segment | Vitamins, Minerals & Supplements | 49,091 / 79,760 | 61.5% |
| Sub-Segment | Supplements | 19,535 / 77,034 | 25.4% |
| Platform | Extra Strength | 84 / 1,102 | 7.6% |

A model that always answers "All others" scores 98.7% accuracy on Mnfr and finds
zero J&J products. That is the entire argument for what follows.

Recommended primary metric per target:

| Target | Shape of the problem | Primary metric |
|---|---|---|
| Mnfr | Binary, 1.3% positive | Recall and precision on J&J; PR-AUC |
| Brand | 143 real classes buried under a 64% placeholder | Macro-F1 over named brands only |
| Platform | 61 classes, 1,102 labeled rows | Not evaluable as-is — see §3.3 |
| Segment | 8 classes, moderate imbalance | Macro-F1 + full confusion matrix |
| Sub-Segment | ~49 classes, support 19,535 to 1 | Macro-F1 by support tier |
| TargetAgeGroup | 3 ordered classes, 94% Adult | Macro-F1 + per-class recall |

---

## 2. Data conditions that affect every metric

These come out of the profile and must be settled before any number is reported.

### 2.1 Malformed rows are few and detectable

Each of the six targets has exactly **3** malformed values, and the profile
reports exactly **3** leading/trailing whitespace values in each of those same
columns. Every malformed label listed in the profile begins with a space:
`" coffee"`, `" tea"`, `" gluten-free"`, `" lactose"`, `" Keto certified"`,
`" Paleo friendly"`.

**Working hypothesis:** these are the same small set of shifted rows appearing
across all six columns, and a whitespace filter finds all of them.

**To verify:** check whether the row indices carrying malformed values in Mnfr
are the same indices as in Segment, Sub-Segment and TargetAgeGroup. If they
match, the corruption is a handful of rows, not a systemic problem, and removing
them costs almost nothing.

This matters for evaluation because these values currently inflate the class
count of every target. Mnfr shows "4 unique labels" but has 2 real ones.

### 2.2 Placeholder classes are not classes

Three values in Brand are not brands:

| Value | Count | Share |
|---|---|---|
| UnItemised brand | 54,059 | 63.9% |
| Other Brands | 3,485 | 4.1% |
| Generic | 557 | 0.7% |

After removing these and the 322 missing, roughly **26,700 rows** carry a named
brand across 143 classes.

**Recommendation:** report Brand metrics on named brands only, and report
performance on the placeholder classes separately. Including "UnItemised brand"
in the macro average rewards a model for learning to say "I don't know" in the
most common case.

A parallel question exists for Mnfr, where "All others" is a catch-all. The model
is not learning what "All others" means; it is learning to detect J&J. Frame the
metrics that way.

### 2.3 Near-duplicate labels inflate the class count

Sub-Segment contains what look like the same category written two ways:

| Label | Count |
|---|---|
| Cold/Flu | 1,302 |
| Cold / Flu | 4 |

Also worth checking: `Creams/Gels & Medicated Patches` (1,009) against
`Creams & Gels (Rubs)` (1), and `Other Lifestyle CHC` (202) against
`Other Lifestyle` (3).

If these are the same class, the 4-row version is not a rare class — it is a
formatting variant, and treating it as a separate class both understates macro-F1
and creates a class that cannot be learned. This is a cleaning decision with a
direct evaluation consequence, so it needs an owner.

### 2.4 Target dataset dtype

In the target dataset the seven classification columns load as `float64` because
they are entirely empty. Force them to string on load, or a join against training
labels will fail silently.

### 2.5 Train and target look like the same population

Feature missingness is nearly identical across the two datasets
(ProductModelNumber 85.8% vs 85.5%; Upc 79.8% vs 79.5%; ProductDescription 67.8%
vs 67.9%; ProductContents 60.1% vs 59.9%).

This supports the assumption that the target set is drawn from the same
population, which is what makes the training-set evaluation meaningful at all.
It is worth stating explicitly in the final report, because it is the assumption
everything else rests on.

### 2.6 Splitting

The profile reports **0 exact duplicate rows**, but exact duplicates are not the
risk. Product catalogs carry near-duplicates (size and flavor variants of the
same item), and a random row split puts near-identical products in both train and
test, inflating every score in a way no metric will reveal.

Candidate grouping keys: `JoiningKey` (0% missing), `Sku` (0.01% missing), `Upc`
(79.8% missing, so unusable alone).

**Action:** determine what `JoiningKey` identifies. If it groups variants of one
product, split on it. This should be decided before any model is scored.

---

## 3. Per-target recommendations

### 3.1 Mnfr — binary, 1.3% positive

Labeled rows: 78,851. J&J: 1,030 (1.31%). All others: 77,821.

This is not a multiclass problem. Treat J&J as the positive class and report:

- **Precision, recall and F1 on J&J** — the primary numbers
- **PR-AUC** — the right curve for a 1.3% positive rate
- **Confusion matrix in raw counts**, not percentages

**Do not use ROC-AUC.** With 77,821 negatives, the false positive rate barely
moves, and ROC-AUC will look strong while the model misses most J&J products.

The business question is which of precision and recall matters more: a missed
J&J product, or a non-J&J product wrongly flagged. That answer sets the threshold,
and it belongs to the PO, not to the modeling team.

### 3.2 Brand — 143 real classes, heavy placeholder

After §2.2, report:

- **Macro-F1 over named brands only** (primary)
- **Support tiers.** The named brands span NOW (1,253) down to Hers, Meno,
  Womaness and Health & Her at 1 each. Report macro-F1 within tiers — say
  ≥250, 50–249, 10–49, <10 — so the headline number is interpretable.
- **Top-3 accuracy.** For a human reviewer choosing from three suggestions, this
  is a real product outcome and should be measured deliberately.
- **Placeholder handling reported separately:** how often does the model correctly
  predict "UnItemised brand", and how often does it hide behind it?

**Check for near-leakage.** `ProductBrand` exists as a feature with only 0.12%
missing, while the target is `Brand`. If `ProductBrand` is the retailer's raw
brand string and `Brand` is the curated version, then much of this task is string
normalization, not classification. That is not a problem, but it changes what a
high score means and it has to be stated. Compare a simple `ProductBrand`-to-
`Brand` lookup against the model as a baseline; if the lookup wins, the model is
not earning its cost.

Expect confusion between brands under the same manufacturer — Tylenol and Motrin
both sit under J&J. Pull the top 20 confused pairs and read them by hand before
concluding the model is wrong; some will be label inconsistencies.

### 3.3 Platform — not evaluable in its current form

This is the finding that needs escalating, not just a metric recommendation.

- 1,105 labeled rows of 84,552 (98.69% missing)
- **61 valid classes** after removing the 3 malformed values
- Largest class: Extra Strength, 84 rows
- **Six classes have exactly 1 row**: Sinus Plus, Precise, 12 Hour Relief,
  Cold Max, SmartCheck, Effective in 15 Minutes
- Roughly **28 of 61 classes have fewer than 10 rows**

**This framing may be wrong, and the companion relationships report explains why.**
The Platform labels are brand-specific product lines (Motrin IB, Tylenol PM,
Imodium A-D), and the 1,105 labeled rows sit close to the 1,030 J&J rows. If
Platform is only labeled for one manufacturer's products, then it is not sparse at
random — it is restricted in scope by design, and the right response is a scope
conversation with the PO rather than a request for more labels. **Run that check
before acting on anything below.**

**Stratified k-fold cross-validation is not possible** on classes with a single
example: that row cannot be in the training fold and the test fold at the same
time. Any macro-F1 over all 61 classes is structurally capped well below 1, and
the number will say more about the split than about the model.

Three options, in order of preference:

1. **Reduce the class count.** Several labels look like they group naturally
   (Sinus, Sinus Plus, Sinus Severe, Sinus + Headache). If the PO can supply or
   approve a coarser Platform taxonomy, the problem may become tractable. This is
   a question for the PO, not a modeling decision.
2. **Evaluate on a support-filtered subset.** Report macro-F1 over classes with
   ≥10 examples, and report the remainder as out-of-scope with counts. Honest,
   and it produces a usable number.
3. **Ship Platform as review-only** in the first release: the model suggests, a
   human always confirms. Given the evidence, this is a defensible outcome, not
   a failure.

Whichever is chosen, include a **learning curve** (train on 25/50/75/100% of the
1,105 labeled rows). If the curve is still climbing at 100%, more labeling is the
highest-value next step, and the curve estimates how much is needed. That is a
concrete ask to bring to the PO.

### 3.4 Segment — 8 classes, tractable

Labeled rows: 79,760 across 8 valid classes, from Vitamins, Minerals &
Supplements (49,091) to Other Self Care (1,135). A 43:1 spread — imbalanced, but
every class has enough data to learn.

Macro-F1 as the primary metric, and **include the full 8×8 confusion matrix** in
the report. At this size it is readable, and it matters more than the score
because Segment errors propagate: a row misassigned at Segment level is wrong at
Sub-Segment level regardless of how good the Sub-Segment model is.

### 3.5 Sub-Segment — ~49 classes, extreme support spread

Labeled rows: 77,034. Support runs from Supplements (19,535) to
Creams & Gels (Rubs) (1).

- **Macro-F1 by support tier**, as with Brand.
- **Set a reporting floor.** Classes below ~10 examples (Cold / Flu 4,
  Dermatologicals 4, Other Lifestyle 3, Daily oral contraception 2,
  Creams & Gels (Rubs) 1) should be reported in their own table rather than
  averaged into the headline, where a single prediction swings their F1 between
  0 and 1.
- **Resolve §2.3 first.** Some of those tiny classes may be formatting variants
  of large ones.

**Hierarchical consistency.** Measure how often the predicted Sub-Segment is valid
under the predicted Segment. This needs no extra labeling and a rate below 100%
is a concrete, fixable defect. The parent-child map itself comes from the
companion relationships ticket, so treat this as planned rather than settled.

Report two views, because they answer different questions:

- Sub-Segment accuracy overall (end-to-end quality)
- Sub-Segment accuracy given Segment was predicted correctly (isolates whether
  the Sub-Segment model is weak or inheriting upstream errors)

### 3.6 TargetAgeGroup — 3 ordered classes, 94% Adult

Labeled rows: 77,472. Adult 72,995 (94.2%), Children 3,718 (4.8%), Infant 759 (1.0%).

The profile settles the earlier open question: the taxonomy is exactly Adult,
Children, Infant. There is no `All Ages` or `Unknown`, so the classes are ordered
(Infant → Children → Adult).

Correcting my earlier draft: with only three classes, **adjacent accuracy adds
little**, since only one pair of classes is two steps apart. Use instead:

- **Per-class recall on Children and Infant** — the primary numbers. Infant at
  1.0% of rows is where this target will fail, and it is also where an error
  matters most in a health products catalog.
- **Macro-F1** as the summary.
- **The full 3×3 confusion matrix.** With three classes it says more than any
  summary statistic, and it shows directly whether Infant is being absorbed into
  Children or into Adult.

---

## 4. Cross-cutting evaluation

### 4.1 Overall performance and baselines

One results table per target: macro-F1, weighted-F1, balanced accuracy, and
accuracy last and labeled as context only.

**Every target needs baselines**, or a score has no meaning:

- **Majority-class baseline** — the floor, computed in §1. Anything at or below
  this is worthless no matter how the number reads.
- **Lookup baseline** — for Brand, map `ProductBrand` to `Brand` directly. For
  Segment and Sub-Segment, map from the `ProductCategory` breadcrumb.
- **Simple text baseline** — TF-IDF over `ProductName` plus a linear classifier.
  `ProductName` is only 0.18% missing, so it is available almost everywhere,
  unlike `ProductDescription` (67.8% missing).

An approach that does not clearly beat the lookup and TF-IDF baselines is not
worth its operating cost, and the evaluation must be able to show that.

### 4.2 Minority classes

The standard, in one line: **no score is reported without its support count**,
and classes below the agreed floor are reported in their own tier rather than
averaged into the headline.

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
to review" are different products. Only the second framing lets the PO pick an
operating point.

Abstention is not an error. Score it separately from a wrong answer.

### 4.5 Human-review candidates

Routing rules worth evaluating:

1. **Low top-1 confidence**, against a calibrated threshold.
2. **Small margin** between top-1 and top-2. Often catches more real errors than
   raw confidence, because it detects genuine ambiguity.
3. **Rare predicted class** — any prediction of a class with tiny training
   support deserves review regardless of confidence. Given §3.3 and §3.5, this
   rule alone covers a lot of ground.
4. **Predicted placeholder** — if the model predicts "UnItemised brand" for a row
   whose `ProductBrand` is populated, something is off.
5. **Hierarchy violation** — predicted Sub-Segment does not belong under predicted
   Segment. Pending the relationships ticket.

**The review queue needs its own metrics**, or it becomes an unmeasured cost:

- **Flag precision** — of rows sent to review, what share were actually wrong?
- **Miss rate** — of wrong predictions, what share were *not* flagged?
- **Review load** — what percentage of the 28,082 target rows needs a human?
  This is the number that decides whether the system saves money.

Report these across thresholds, not at one fixed cutoff.

### 4.6 Comparing model approaches

- **Fixed splits, saved and version-controlled**, using the grouping key from §2.6.
- **One primary metric per target, declared before results are seen.**
- **Uncertainty on every comparison.** Bootstrap confidence intervals, or
  McNemar's test for two models on the same rows. On Platform especially, a
  several-point macro-F1 difference across 1,102 rows is probably noise.
- **Cost reported alongside accuracy.** A 2-point gain that triples inference cost
  is a business decision, not an automatic win.

---

## 5. Open questions

1. **What does `JoiningKey` identify?** Decides the split strategy (§2.6). Highest
   priority — it affects the validity of every number.
2. **Are the malformed rows the same rows across all six targets?** (§2.1)
3. **Is `Cold / Flu` the same class as `Cold/Flu`?** And the other near-duplicate
   pairs in §2.3.
4. **Is `ProductBrand` the raw version of the `Brand` target?** (§3.2)
5. **Can the PO supply or approve a coarser Platform taxonomy?** (§3.3)
6. **For Mnfr, which is worse: a missed J&J product or a false flag?** Sets the
   threshold.
7. **`Category` is a seventh classification column** that is 100% missing in the
   target dataset but almost fully populated in training. It is not in the six
   named targets. Is it a target, a feature, or out of scope?

---

## 6. Done-when checklist

| Criterion | Where |
|---|---|
| Recommended metrics documented | §1, §3 |
| Class imbalance addressed | §1, §2.2, §3.1, §3.2, §3.5, §3.6, §4.2 |
| Platform's limited labeled data addressed | §3.3 |
| Supports comparison between model approaches | §4.6, §4.1 baselines |
