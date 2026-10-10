# Model Evaluation Strategy — CatalogIQ Classification Targets

**Status:** v4 — figures measured against the v4 integrated run
**Scope:** how to evaluate the six classification targets
**Location:** `docs/model-evaluation-strategy.md`
**Data basis:** `training_candidate.csv` from a local `--mode integrated
--exclude-policy keep` run reproducing `integrated_policy_v4_final_20261003`
(83,938 rows of 84,552). Supersedes two earlier versions, one written from the
dataset profile and one measured on the superseded
`conservative_handoff_20261002`. Claims that changed are marked **CORRECTED**.

---

## 1. Summary

The six targets should not share one scorecard. **Accuracy is not the primary
metric for any of them**, and the measured majority-class baselines show why:

| Target | Labelled rows | Classes | Majority class | Accuracy from predicting it alone |
|---|---|---|---|---|
| Mnfr | 78,273 | 2 | All others | **98.69%** |
| TargetAgeGroup | 76,939 | 3 | Adult | **94.20%** |
| Brand | 83,619 | 144 | UnItemised brand | **64.07%** |
| Segment | 79,205 | 8 | Vitamins, Minerals & Supplements | **61.84%** |
| Sub-Segment | 76,500 | 49 | Supplements | **25.45%** |
| Platform | 1,094 | 61 | Extra Strength | **7.68%** |

A model that always answers "All others" scores 98.69% on Mnfr and finds zero
J&J products. That is the whole argument.

Recommended primary metric per target:

| Target | Shape of the problem | Primary metric |
|---|---|---|
| Mnfr | Binary, 1.31% positive | Recall and precision on J&J; PR-AUC |
| Brand | 141 named classes under a 64% placeholder | Macro-F1 over named brands only |
| Platform | 61 classes, 1,094 labelled rows | Not evaluable as-is — see §3.3 |
| Segment | 8 classes, moderate imbalance | Macro-F1 + full confusion matrix |
| Sub-Segment | 49 classes, support 19,467 to 1 | Macro-F1 by support tier |
| TargetAgeGroup | 3 ordered classes, 94% Adult | Macro-F1 + per-class recall |

---

## 2. Data conditions that affect every metric

### 2.1 Which cleaned dataset these figures describe — read this first

Every number here comes from the **v4 integrated run with
`--exclude-policy keep`**, the policy the decision log records as group agreed.

**CORRECTED — the earlier version measured the wrong artifact.** It used
`conservative_handoff_20261002`, which removed 5,834 rows (6.90%) and with them
**five entire classes** that exist in the data:

| Target | Class | Raw rows | Conservative v2 | v4 |
|---|---|---|---|---|
| Sub-Segment | `Motion Sickness` | 408 | **0** | 405 |
| Sub-Segment | `Hemorrhoid Remedies` | 210 | **0** | 210 |
| Sub-Segment | `Anti-Nausea` | 57 | **0** | 57 |
| Sub-Segment | `Creams & Gels (Rubs)` | 1 | **0** | 1 |
| Platform | `Arthritis` | 2 | **0** | 2 |

Sub-Segment therefore has **49 classes, not 46**, and the macro-F1 denominator
and support tiers in §3.2 and §3.5 change accordingly. Details in
`docs/validation/cleaning_validation.md` §3.5.

Anything already evaluated against the v2 handoff should be re-run.

### 2.2 Malformed rows — resolved

The profile reported 3 malformed, whitespace-prefixed values per target. **All 18
were handled** and zero remain, verified in the cleaning audit §3.8. `Mnfr` now
carries exactly 2 values.

### 2.3 Placeholder classes are not classes

Three Brand values are not brands:

| Value | Count | Share of labelled |
|---|---|---|
| UnItemised brand | 53,571 | 64.07% |
| Other Brands | 3,459 | 4.14% |
| Generic | 552 | 0.66% |

**26,037 rows carry a named brand across 141 classes.**

Report Brand metrics on named brands only, and report placeholder performance
separately. Including "UnItemised brand" in the macro average rewards a model
for learning to say "I don't know" in the most common case.

The same framing applies to Mnfr, where "All others" is a catch-all. The model is
not learning what "All others" means; it is learning to detect J&J.

### 2.4 Near-duplicate labels — all three resolved

| Pair | Resolution |
|---|---|
| `Cold / Flu` → `Cold/Flu` | Relabelled, rule `slash_spacing`, 4 rows |
| `Other Lifestyle` → `Other Lifestyle CHC` | Relabelled, rule `reviewed_other_lifestyle`, 3 rows |
| `Creams & Gels (Rubs)` | **Not a duplicate.** A real one-row class, retained |

Each relabel is recorded row by row in `label_changes.csv`. The class count is
settled at 49 Sub-Segments.

### 2.5 Target dataset dtype — resolved

No classification column reads as `float64` in the v4 output, so the
empty-column trap — a silent label-join failure — is not present.

**Corrected.** An earlier version of this line said four feature columns "move
from string to numeric in the export". They do not. The candidate partition and
the export store byte-identical values in `ProductRating`, `XRatXRev`,
`ProductReviewsCount` and `ReviewsCount` — checked across all 83,938 rows with
`dtype="string"` and `keep_default_na=False`. What changes is what **pandas
infers** when you read either file without an explicit dtype, which turns a
stored `5` into the float `5.0` in memory. See cleaning audit §3.2.

The practical rule is unchanged and now applies for the right reason: pass
`dtype="string"` and `keep_default_na=False` whenever the stored text matters,
and always load identifiers as string to preserve leading zeros.

### 2.6 Train and target are the same population — confirmed

| Feature | Training | Target |
|---|---|---|
| ProductModelNumber | 86.09% | 85.82% |
| Upc | 80.21% | 79.99% |
| ProductDescription | 68.01% | 68.18% |
| ProductContents | 60.18% | 59.96% |
| ProductName | 0.00% | 0.00% |
| ProductCategory | 0.02% | 0.01% |

This supports the assumption that the target set is drawn from the same
population, which is what makes training-set evaluation meaningful at all. State
it explicitly in the final report; it is the assumption everything rests on.

### 2.7 Splitting — CORRECTED: there is no grouping key

The first version named this the highest-priority open question and proposed
splitting on `JoiningKey`. **That option does not exist.**

| Candidate | Missing | Unique values | Rows per key |
|---|---|---|---|
| `JoiningKey` | 0.00% | 83,938 | **1.00** |
| `Sku` | 0.00% | 83,460 | 1.01 |
| `Upc` | 80.21% | 12,280 | unusable alone |
| `source_row` | 0.00% | 83,938 | 1.00 (provenance index) |

`JoiningKey` is a row identifier. `Sku` is effectively one too.

**The risk is unchanged and unmitigated.** Product catalogues carry
near-duplicates — size and flavour variants of the same item — and a random row
split puts near-identical products in both train and test, inflating every score
in a way no metric reveals.

**Two things follow, and both are now tasks rather than questions.**

1. **Build a grouping key.** Candidate: normalised `ProductName` (lowercase, strip
   punctuation and size/count tokens) plus `Brand`. Validate by inspecting the
   largest resulting groups by hand — if they are variants of one product, it
   works.
2. **Build it from the candidate partition, not the export.**
   `training_cleaned.csv` drops the business keys `JoiningKey`, `Sku`, `Upc` and
   `MDM_Id`. Row identity itself does survive, through `source_row`, which is
   unique across all 83,938 rows — so the business keys are recoverable with one
   join back to `training_candidate.csv`. The constraint is a join, not a wall.

   That does not solve the problem, because none of those keys groups product
   variants in the first place. The grouping key has to be constructed (item 1)
   and it is easiest to do that where the full schema is still present.

This blocks the validity of every reported number.

---

## 3. Per-target recommendations

### 3.1 Mnfr — binary, 1.31% positive

Labelled 78,273. J&J 1,022 (1.31%). All others 77,251. 5,665 unlabelled.

Not a multiclass problem. Treat J&J as the positive class and report:

- **Precision, recall and F1 on J&J** — the primary numbers
- **PR-AUC** — the right curve for a 1.31% positive rate
- **Confusion matrix in raw counts**, not percentages

**Do not use ROC-AUC.** With 77,251 negatives the false positive rate barely
moves, and ROC-AUC will look strong while the model misses most J&J products.

**CORRECTED — the Brand lookup is a strong baseline, not a substitute.** The
previous version, measuring on the v2 handoff, found Brand → Mnfr deterministic
across all brands and concluded Mnfr need not be modelled. On v4, **four brands
map to two manufacturers** (Tylenol, Benadryl, Sudafed, Bengay — 5 rows total).
The v2 result was an artifact of the conservative policy having removed those
rows.

So the lookup is the baseline any Mnfr model must beat, and it is a very strong
one, but it is not exact.

**A boundary worth stating in the report.** The decision log says *"Preserve
supplied Mnfr... Do not infer J&J from Brand"*, and the repository records a
withdrawn J&J reassignment. The lookup is used here **as an evaluation baseline
and a review signal only**. Nothing in this document proposes filling or
rewriting Mnfr labels from Brand.

The threshold question — which is worse, a missed J&J product or a false flag —
belongs to the PO.

### 3.2 Brand — 141 named classes, 64% placeholder

Report:

- **Macro-F1 over named brands only** (primary)
- **Support tiers**, measured:

| Tier | Classes | Rows |
|---|---|---|
| ≥250 | 32 | 14,055 |
| 50–249 | 83 | 11,469 |
| 10–49 | 16 | 477 |
| <10 | 10 | 36 |

Top: NOW (1,249), Nature Made (881), Swanson (820).

- **Top-3 accuracy.** For a human reviewer choosing from three suggestions this
  is a real product outcome and should be measured deliberately.
- **Placeholder handling reported separately:** how often does the model correctly
  predict "UnItemised brand", and how often does it hide behind it?

**The near-leakage is confirmed and large.** `ProductBrand` matches `Brand`
exactly **91.86%** of the time on the 26,037 named-brand rows.

This changes what a high Brand score means. **Much of this task is string
normalisation, not classification.** That is not a problem, but it must be stated,
and the lookup must be the baseline:

> Map `ProductBrand` → `Brand` directly. Any model scoring below ~92% on named
> brands is worse than a dictionary.

Expect confusion between brands under the same manufacturer. Pull the top 20
confused pairs and read them by hand before concluding the model is wrong; some
will be label inconsistencies — §4.3 of the relationships research found five.

### 3.3 Platform — still not evaluable as-is

- 1,094 labelled rows of 83,938 (**98.70% missing**)
- **61 valid classes**
- Largest class: Extra Strength, 84 rows
- **6 classes have exactly 1 row**
- **28 of 61 classes have fewer than 10 rows**

**The scope hypothesis is supported but not absolute.** 1,022 of 1,094 Platform
rows are J&J — **93.4%, not exclusive**. The remaining 72 are Advil, Excedrin,
Kirkland Signature, Visine and others.

What this changes:

- Platform is still overwhelmingly a J&J-labelled field, so **the scope
  conversation with the PO remains the right next step** rather than a request
  for more labels.
- But a Platform prediction on a non-J&J product is **unusual, not meaningless**.
  The abstention argument is weaker than the first version claimed, and rule R4
  must not block.

**Stratified k-fold cross-validation remains impossible** on classes with a
single example. Any macro-F1 over all 61 classes is structurally capped well
below 1, and the number says more about the split than the model.

Three options, in order of preference:

1. **Reduce the class count.** Several labels group naturally (`Sinus`,
   `Sinus Plus`, `Sinus Severe`). A coarser taxonomy may make the problem
   tractable. A question for the PO, not a modelling decision.
2. **Evaluate on a support-filtered subset.** Macro-F1 over classes with ≥10
   examples — 33 of 61 — reporting the remaining 28 as out-of-scope with counts.
3. **Ship Platform as review-only** in the first release: the model suggests, a
   human always confirms. Given the evidence this is a defensible outcome, not a
   failure.

Whichever is chosen, include a **learning curve** (train on 25/50/75/100% of the
1,094 labelled rows). If the curve is still climbing at 100%, more labelling is
the highest-value next step and the curve estimates how much. That is a concrete
ask for the PO.

### 3.4 Segment — 8 classes, tractable

Labelled 79,205 across 8 classes, from Vitamins, Minerals & Supplements (48,982)
to Other Self Care (1,132). A 43:1 spread — imbalanced, but every class has
enough data to learn.

Macro-F1 as primary, and **include the full 8×8 confusion matrix**. At this size
it is readable, and it matters more than the score because Segment errors
propagate: a row misassigned at Segment level is wrong at Sub-Segment level
regardless of how good the Sub-Segment model is.

### 3.5 Sub-Segment — 49 classes, extreme support spread

Labelled 76,500. Support runs from Supplements (19,467) to
`Creams & Gels (Rubs)` (1).

**Measured support tiers:**

| Tier | Classes | Rows |
|---|---|---|
| ≥250 | 32 | 74,985 |
| 50–249 | 9 | 1,399 |
| 10–49 | 5 | 109 |
| <10 | 3 | 7 |

The three classes below 10: `Dermatologicals` (4), `Daily oral contracception`
(2), `Creams & Gels (Rubs)` (1). **Report these in their own table** rather than
averaged into the headline, where a single prediction swings their F1 between 0
and 1.

**Four of these classes do not exist in the v2 handoff** (§2.1). Any support tier
computed from that artifact is wrong.

**Hierarchical consistency — CORRECTED.** The first version proposed measuring how
often the predicted Sub-Segment is valid under the predicted Segment, treating a
rate below 100% as a defect. **The hierarchy is not strict**: `Internal Pain` and
`Probiotics` each have two parents, and the decision log explicitly says shared
parents can be legitimate. The measure is still worth reporting, but the target
is "matches the verified parent-child map", not 100%, and the map is in
`data/interim/segment_subsegment_map.csv`, regenerated by
`notebooks/01-classification-relationships.ipynb` (`data/` is git-ignored).

Report two views:

- Sub-Segment accuracy overall (end-to-end quality)
- Sub-Segment accuracy given Segment was predicted correctly (isolates whether
  the Sub-Segment model is weak or inheriting upstream errors)

### 3.6 TargetAgeGroup — 3 ordered classes, 94% Adult

Labelled 76,939. Adult 72,475 (94.20%), Children 3,705 (4.82%), Infant 759
(0.99%).

The taxonomy is exactly Adult, Children, Infant, so the classes are ordered.

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
  brands, §3.2). For Mnfr, `Brand` → `Mnfr` (140 of 144 brands exact, §3.1). For
  Segment and Sub-Segment, the `ProductCategory` breadcrumb — though that covers
  only ~9.5% of rows at ≥90% purity, so it is a partial baseline.
- **Simple text baseline** — TF-IDF over `ProductName` plus a linear classifier.
  `ProductName` is 0.00% missing, so it is available everywhere, unlike
  `ProductDescription` (68.01% missing).

**An approach that does not clearly beat the lookup and TF-IDF baselines is not
worth its operating cost.** For Brand and Mnfr the lookups are measured and
strong. The evaluation must be able to show whether a model beats them.

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
   whose `ProductBrand` is populated, something is off. Given the 91.86% match
   rate, this is a strong signal.
5. **Rule violation** — any of R1–R6 from the relationships research firing. All
   six are review triggers, not blocks; none of the relationships is strict
   enough to reject a prediction outright.

**The review queue needs its own metrics**, or it becomes an unmeasured cost:

- **Flag precision** — of rows sent to review, what share were actually wrong?
- **Miss rate** — of wrong predictions, what share were *not* flagged?
- **Review load** — what percentage of the 27,870 target rows needs a human?
  This is the number that decides whether the system saves money.

Report these across thresholds, not at one fixed cutoff.

### 4.6 Comparing model approaches

- **Fixed splits, saved and version-controlled**, using a constructed grouping
  key (§2.7) — not `JoiningKey`, which does not group.
- **One primary metric per target, declared before results are seen.**
- **Uncertainty on every comparison.** Bootstrap confidence intervals, or
  McNemar's test for two models on the same rows. On Platform especially, a
  several-point macro-F1 difference across 1,094 rows is probably noise.
- **Cost reported alongside accuracy.** A 2-point gain that triples inference
  cost is a business decision, not an automatic win.

---

## 5. Open questions

1. **What grouping key should the split use?** (§2.7) `JoiningKey` does not work
   and the feature export drops the alternatives. This blocks the validity of
   every number and is now a build task.
2. **Can the PO supply or approve a coarser Platform taxonomy?** (§3.3)
3. **Is Platform expected on the target set at all?** (§3.3) 98.70% missing.
4. **For Mnfr, which is worse: a missed J&J product or a false flag?** Sets the
   threshold.
5. **Are the five Brand → Mnfr exceptions Brand labelling errors?** (§3.1) If so,
   the lookup baseline is effectively exact.
6. **Should `Category` be dropped?** It holds one value, `Self Care`, in all
   83,938 rows.

---

## 6. Done-when checklist

| Criterion | Where |
|---|---|
| Recommended metrics documented | §1, §3 |
| Class imbalance addressed | §1, §2.3, §3.1, §3.2, §3.5, §3.6, §4.2 |
| Platform's limited labelled data addressed | §3.3 |
| Supports comparison between model approaches | §4.6, §4.1 baselines |
