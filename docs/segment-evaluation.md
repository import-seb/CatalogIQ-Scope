# Segment Evaluation — the harness, and what it says so far

**Ticket:** evaluate both models' Segment predictions, inspect errors, and
compare results before and after feature changes. Make the evaluation reusable
for other targets.
**Notebook:** `notebooks/04-segment-evaluation.ipynb`
**Spec:** §4 of the evaluation strategy. That document is the criteria; the
notebook is the thing that runs them. On `main` it is
`docs/model-evaluation-strategy (1).md`; the renamed and corrected copy,
`docs/model-evaluation-strategy.md`, arrives with the validation-audit PR, so
the two paths refer to the same document at different points in its history.
**Measured on:** the finalized splits, protocol `segment-final-20261009`,
validation only, 11,952 labelled rows. The reserved test is **refused by the
loader**, not merely avoided.
**Owner:** Maria Garcia Sehara

---

## 1. What this is

The harness takes prediction files in the contract the team's trainer already
writes — `record_id`, `true_label`, `predicted_label`, `confidence`,
`probability_0..n-1`, with `label_names` from `training_manifest.json` — and runs
all six parts of §4 on them. Point it at a directory and it scores it.

It is **not** a model selection. No model is picked here, and the team's own
findings say none has been picked.

| Notebook | Strategy §4 |
|---|---|
| §2 Baselines | 4.1 |
| §3 Per class, minority tiers | 4.2 |
| §4 Calibration | 4.3 |
| §5 Selective prediction | 4.4 |
| §6 Review queue | 4.5 |
| §7 Paired model comparison | 4.6 |
| §8 Error inspection | the ticket's "inspect errors" |

**Reusable for the other targets.** Set `TARGET` in §0 and the label column,
baselines, class tiers and per-class tables all follow. The one thing that does
not move: the team's trainer tokenizes a `Segment` column, so another target
needs their trainer pointed at it before there are predictions to score.

### Four things it enforces rather than assumes

1. **The reserved test is refused.** If any `record_id` in a predictions file
   belongs to the test partition, the loader raises instead of scoring it.
2. **Probability columns are mapped by `label_names`** or the run is rejected.
   Guessing that `probability_3` is the fourth class alphabetically is how
   calibration numbers end up meaningless.
3. **The bootstrap resamples product groups, not rows.** Rows inside a group are
   not independent, so a row bootstrap would give intervals that are too narrow.
4. **Temperature scaling cannot change accuracy**, and §4 carries a real
   `assert` on it rather than a comment saying so. Dividing logits by a positive
   scalar leaves the argmax alone; if that ever stops being true the scaling
   implementation is broken and the notebook stops instead of reporting a
   quietly wrong number.

### It runs without a GPU

`data/processed/` is gitignored, so no team checkpoint is in the repo. §1 builds
two reference baselines itself — TF-IDF plus logistic regression, one on four
fields and one on five — in the same contract. That does three jobs: the harness
is runnable by anyone, §4.1 of the strategy wanted a TF-IDF baseline anyway, and
the two arms *are* the before/after feature comparison the ticket asks for.

---

## 2. The baseline that should change how we read every score

| Baseline | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 61.75% | **9.54%** |
| **Lookup on `ProductCategory` alone** | **92.85%** | **88.14%** |
| TF-IDF + logistic regression, five fields | 94.19% | 91.20% |

**A lookup table gets 88.14 macro F1 with no model at all.** 1,209 keys learned
on train, covering 99.2% of validation rows, each mapped to its most common
Segment. Against that:

| | Macro F1 | Over the lookup |
|---|---|---|
| Lookup on `ProductCategory` | 88.14 | — |
| TF-IDF + logreg, five fields | 91.20 | +3.06pp |
| Linear SVM, five fields | 92.46 | +4.32pp |
| **MiniLM, five fields, 256 tokens** | **94.45** | **+6.31pp** |

§4.1 of the strategy says an approach that does not clearly beat the lookup and
TF-IDF baselines is not worth its operating cost. MiniLM does beat them — by
6.31pp over a lookup and 3.25pp over TF-IDF. That is a real margin, and it is a
much smaller one than the headline 94.45 implies on its own.

**This corrects §4.1 of the evaluation strategy.** I wrote there that the
breadcrumb lookup "covers only ~9.5% of rows at ≥90% purity, so it is a partial
baseline". That measured coverage at a purity threshold. Measured as a
most-common-label mapping — which is what a lookup baseline actually is — it
covers 99.2% and scores 88.14 macro F1. It is not a partial baseline. It is the
number every model has to be compared against.

---

## 3. Before and after the feature change (§4.6)

The same classifier, the same splits, the same rows. The only difference is
whether `ProductCategory` is in the text.

| | Accuracy | Macro F1 |
|---|---|---|
| Four fields | 88.62% [87.68, 89.52] | 83.14% [81.29, 84.50] |
| **Five fields** | **94.19%** [93.57, 94.72] | **91.20%** [90.07, 92.04] |
| Difference | +5.56pp | **+8.06pp** |

McNemar on the 11,952 shared rows: 802 rows the five-field model gets right and
the four-field one gets wrong, against 137 the other way. **p ≈ 4e-104.** The
95% group-aware bootstrap intervals are disjoint on both metrics.

That is a **fourth** independent measurement of the category gain, and it lands
in the same band as the other three:

| Model | Gain from `ProductCategory` |
|---|---|
| Logistic regression (Max) | +7.10pp |
| Linear SVM | +8.14pp |
| MiniLM, 256 tokens (Sebastian) | +8.57pp |
| **Logistic regression, this harness** | **+8.06pp** |

---

## 4. Calibration (§4.3)

The README makes confidence a product requirement, so the number has to mean
something. On both baselines it does not, until it is fixed:

| | Four fields | Five fields |
|---|---|---|
| Expected calibration error | 11.09% | **6.84%** |
| Brier score (multiclass) | 0.2003 | **0.0988** |
| Temperature | 0.60 | 0.70 |
| ECE after scaling | 2.38% | **0.31%** |
| Accuracy before / after | 88.62% / 88.62% | 94.19% / 94.19% |

**Adding the category path improved calibration as well as accuracy** — ECE
11.09% down to 6.84% raw, 2.38% down to 0.31% after scaling, and the Brier score
halved. That is not something the accuracy numbers would have told us, and it
matters: the five-field model's confidence is usable after a one-parameter fix,
and the four-field model's is still 2.38% off after the same fix.

**The model is underconfident, not overconfident.** Every reliability bin has
accuracy above its mean confidence — rows at 0.65 confidence are right 80% of
the time, rows at 0.76 are right 88% of the time. A temperature below 1 sharpens
the distribution and closes a 6.84% gap to 0.31%.

That matters operationally: **a naive "flag everything below 0.80" over-flags.**
Before anyone sets a review threshold, scale the confidence. It costs nothing —
the predicted class never moves, which §4 of the notebook asserts rather than
assumes.

Calibration has to be measured per target. This is Segment's temperature, not
the project's.

---

## 5. Selective prediction (§4.4)

The table the PO needs to choose an operating point. Five-field baseline,
calibrated confidence, most confident rows answered first.

| Coverage | Answered | Accuracy on answered | Macro F1 | Confidence cutoff |
|---|---|---|---|---|
| 100% | 11,952 | 94.19% | 91.20% | — |
| **90%** | 10,757 | **97.09%** | 95.43% | 0.684 |
| **75%** | 8,964 | **98.43%** | 97.45% | 0.842 |
| 50% | 5,976 | 99.03% | 98.72% | 0.916 |
| 25% | 2,988 | 99.40% | 99.48% | 0.958 |

Abstaining on 10% of rows buys **+2.90pp accuracy**. Abstaining on 25% buys
+4.24pp. "94% on everything" and "98% on the 75% it is sure about, rest to
review" are different products, and this is the table that lets someone pick.

The same table should be produced for MiniLM before any threshold is shipped.
MiniLM's predictions include probabilities, so the harness can do it as soon as
the directory is available.

---

## 6. The review queue (§4.5)

Scored across thresholds, not at one cutoff, which is what the strategy asks.
Five-field baseline, 695 errors in 11,952 rows.

**Rule 1 — low calibrated confidence**

| Threshold | Review load | Flag precision | Miss rate |
|---|---|---|---|
| 0.70 | 5.59% | 40.57% | 61.01% |
| 0.80 | 9.04% | 35.06% | 45.47% |
| 0.90 | 15.51% | 26.75% | 28.63% |
| 0.95 | 24.63% | 19.29% | 18.27% |

**Rule 2 — small margin between top-1 and top-2**

| Threshold | Review load | Flag precision | Miss rate |
|---|---|---|---|
| 0.10 | 1.83% | 51.60% | 83.74% |
| 0.20 | 3.55% | 44.81% | 72.66% |
| 0.40 | 7.65% | 38.40% | 49.50% |
| 0.60 | 13.86% | 28.68% | 31.65% |

**On margin versus confidence, the strategy's guess was close but not
confirmed.** §4.5 predicted that margin "often catches more real errors than raw
confidence". At matched review load the two are within a couple of points of
each other — margin is slightly ahead in the low-load region (51.6% precision at
1.8% load against 51.6% at 1.1%), and behind at higher loads. **Comparable, not
a clear win.** Either is usable; neither is obviously the one to ship.

**Rule 3 — rare predicted class.** Weak on its own for Segment: 1.71% load,
25.49% flag precision, 96.18% miss rate. Only `Other Self Care` falls under the
cutoff. Worth keeping as a conjunct, not as a rule by itself.

**Rule 4 — predicted placeholder.** Does not apply to Segment. It is a
Brand/Mnfr rule, keyed to classes like "UnItemised brand" with the source field
populated. `Other Self Care` is a real Segment class with 173 validation rows,
and flagging it here would only repeat rule 3. The harness implements the rule
properly for Brand and Mnfr and says plainly that it does not apply here.

**Rule 5 — relationship violations R1–R6.** **Unwired, therefore unmeasured.**
It needs the rules from the relationships research exported in a machine-readable
form. Saying so is better than reporting it as passing.

**The honest summary on the queue:** no single rule gets flag precision above
about 53%, and getting the miss rate under 30% costs 14–16% review load. The
queue is real work. The number that decides whether this saves money is review
load, and nobody has set a target for it.

---

## 7. Where the errors are (§8 of the notebook)

**The top 10 confusion pairs are 77.1% of all errors.** The dominant one is not
subtle:

| Count | True | Predicted |
|---|---|---|
| 200 | Vitamins, Minerals & Supplements | Digestive Health |
| 71 | Vitamins, Minerals & Supplements | Internal Analgesics |
| 61 | Vitamins, Minerals & Supplements | Lifestyle CHC |
| 42 | Digestive Health | Vitamins, Minerals & Supplements |
| 38 | CCFS | Allergy |

Errors concentrate on the boundary between the majority class and its neighbours,
in both directions. A review rule aimed at just `Vitamins ↔ Digestive Health`
would cover 35% of all errors by itself, which is worth testing against the
generic confidence rules above.

The notebook also prints the rows where two runs disagree, with the product name
and category path, which is the shortest route to understanding *why* a feature
helps. One example from the four-field against five-field comparison: "Rose
Pastilles 1.75oz tin by Abbaye de Flavigny", category path
`Cough & Cold > Coughing`. Four fields predicted Vitamins at 0.224 confidence;
five fields predicted CCFS at 0.998. The breadcrumb carried it.

---

## 8. What this does not answer

- **No model is selected.** That needs cost beside accuracy, and cost is not in
  a predictions file. §7 of the notebook says where to get it: `runtime_seconds`
  in each run's `training_manifest.json`, whether a GPU was required, and
  inference latency.
- **MiniLM has not been scored here.** Its prediction directory is not in the
  repo, because `data/processed/` is gitignored. The harness scores it the moment
  the directory is available — nothing needs changing.
- **Review rule 5 is unmeasured**, and the review load target is unset.
- **Only Segment**, and only validation. The reserved test stays closed.
- **Calibration is per target.** Segment's temperature of 0.70 says nothing
  about Sub-Segment.

---

## 9. What I would do next, in order

1. **Score MiniLM through this harness.** One directory path. It gives
   calibration, a coverage-risk table and a paired comparison against both
   baselines, all of which the model-selection decision needs and none of which
   exists yet.
2. **Set a review load target with the PO.** Every queue number above is
   meaningless without one. 15% load for a 29% miss rate may be excellent or
   unaffordable; only they know.
3. **Export R1–R6 machine-readably** so rule 5 stops being a blank.
4. **Put cost in the comparison table.** MiniLM beats the linear SVM by 1.99pp
   macro F1 and needs a GPU. That is a business decision, and it should be
   presented as one rather than settled by whichever number is larger.
