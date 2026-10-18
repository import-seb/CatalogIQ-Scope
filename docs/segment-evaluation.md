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
**Runs scored:** Sebastian's five-field, 256-token MiniLM, plus two TF-IDF
reference arms the harness builds itself.
**Owner:** Maria Garcia Sehara

---

## 1. What this is

The harness takes prediction files in the contract the team's trainer already
writes — `record_id`, `true_label`, `predicted_label`, `confidence`,
`probability_0..n-1`, with the class order from the run's own manifest — and runs
all six parts of §4 on them. Point it at a directory and it scores it.

It is **not** a model selection. No model is picked here.

| Notebook | Strategy §4 |
|---|---|
| §2 Baselines | 4.1 |
| §3 Per class, minority tiers | 4.2 |
| §4 Calibration | 4.3 |
| §5 Selective prediction | 4.4 |
| §5.1 Confidence against margin at matched review load | 4.5 |
| §6 Review queue | 4.5 |
| §7 Paired model comparison | 4.6 |
| §8 Error inspection | the ticket's "inspect errors" |
| §8.1 Source field against the label, on the dominant pair | the ticket's "inspect errors" |

**Reusable for the other targets.** Set `TARGET` in §0 and the label column,
baselines, class tiers and per-class tables all follow. The one thing that does
not move: the team's trainer tokenizes a `Segment` column, so another target
needs their trainer pointed at it before there are predictions to score.

### Five things it enforces rather than assumes

1. **The reserved test is refused.** If any `record_id` in a predictions file
   belongs to the test partition, the loader raises instead of scoring it.
2. **A run that names its split is held to it.** §0.1 recomputes
   `validation_record_ids_sha256` from our own split, and any run whose manifest
   carries that field must match or it is refused rather than scored. §2 covers
   the recipe and both directions of the test.
3. **The class order is read from the run's manifest and then checked against
   the data.** If the argmax of the probability columns disagrees with
   `predicted_label` on more than 1% of rows, the order is wrong and the run is
   rejected. Reading an order out of a file and trusting it is how calibration
   numbers end up meaningless; this is the version that cannot be fooled by a
   mislabelled manifest.
4. **The bootstrap resamples product groups, not rows.** Rows inside a group are
   not independent, so a row bootstrap would give intervals that are too narrow.
5. **Temperature scaling cannot change accuracy**, and the notebook's §4 carries
   a real `assert` on it rather than a comment saying so.

### It accepts however a teammate packages a run

Discovery reads `training_manifest.json`, `protocol.json`, `model_config.json`,
`label_mapping.json` or `manifest.json`, and pulls the class order out of any of
the shapes those files use — a `label_names` list, a bare list, index→class,
class→index, and the `{"probability_0": "Allergy", …}` form keyed by column name
that `scripts.package_segment_baseline` actually writes.

This matters right now, and getting it right took two attempts. Max's baseline
handoff ships `label_mapping.json`; the original loader required
`training_manifest.json` and would have skipped his arms entirely. My first fix
accepted the filename but still could not read his *shape*, because his keys are
column names rather than bare integers — it returned no class order, and the
loader then rejected the run. The reader now pulls a trailing index out of either
side of the mapping.

Verified two ways rather than asserted: an assert block in the notebook exercises
the reader on ten shapes plus three files with no usable order, which must return
nothing rather than a guess; and a directory rebuilt in his exact format — his
column order, his `label_mapping.json`, his `manifest.json` — was discovered,
loaded and scored to the same 94.19% / 91.20% and the same ECE 6.84% → 0.31% as
the arm it was copied from.

### It runs without a GPU

`data/processed/` is gitignored, so no checkpoint is in the repo. §1 builds two
reference arms itself — TF-IDF plus logistic regression, one on four fields and
one on five — in the same contract. The harness is runnable by anyone, §4.1
wanted a TF-IDF baseline anyway, and the two arms *are* the before/after feature
comparison the ticket asks for.

---

## 2. Verifying Sebastian's run before scoring it

A prediction file is worth nothing if it was produced on a different split or
scored against different labels. Checked first, all of it passing:

| Check | Result |
|---|---|
| `predictions.csv` SHA-256 against the `artifacts` block of its own manifest | matches, `394e7415…` |
| Row count against `validation_rows` | 11,952 = 11,952 |
| Duplicate `record_id` | 0 |
| Rows in the **reserved test** partition | **0** |
| Rows outside the validation partition | 0 |
| Row set against the Segment-labelled validation rows | identical — 0 missing, 0 extra |
| Per-class counts against `validation_class_counts` | all 8 match |
| His `true_label` against `splits2/validation.csv` `Segment`, row by row | 0 disagreements |
| Probabilities sum to 1 | max deviation 1.9e-07 |
| Argmax of probabilities equals `predicted_label` | every row |
| `confidence` equals the max probability | exact |
| **Accuracy, macro F1, weighted F1 recomputed from the raw predictions** | **identical to his manifest to 10 decimal places** |
| `validation_record_ids_sha256` recomputed from our own split | matches, `47f60a59…` |

So **94.45% macro F1 reproduces from the raw file**, on our splits, against our
labels, with the test untouched.

### The record-id hash, and why it is now a guard

That last row took two tries. The manifest carries
`validation_record_ids_sha256: 47f60a59…` and my first attempt at it failed
across 24 serialisation recipes — assignment order and sorted, with
newline/comma/concatenated/JSON joins. All 24 were wrong for the same reason:
none of them length-prefixes the ids. The real recipe is `_hash_items()` in
`src/catalogiq/segment_transformer.py` — ids in order, each one's UTF-8 byte
length written as **8 bytes big-endian** before the bytes themselves:

```python
digest = hashlib.sha256()
for record_id in sorted(validation_record_ids):
    payload = record_id.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
```

Computed over the 11,952 **labelled** validation rows it reproduces `47f60a59…`
exactly, from the split export and from his predictions file independently. (The
unlabelled rows must be dropped first; over all 12,667 it does not match. For our
export the sort is a no-op, because `assignments.csv` already lists validation
rows in lexicographic order — but it is the documented recipe, so it stays.)

**§0.1 now recomputes this from our split and checks every run that carries the
field**, so a run trained against a different split is refused instead of
scored. Tested in both directions: Sebastian's run reports
`validation ids match this split (labelled validation rows)`, and a copy of it
with a deliberately wrong hash is rejected with a message naming the mismatch
and left out of the scoring. Two hashing conventions are accepted — over the
labelled rows and over the whole validation partition — in case a teammate
hashes before dropping unlabelled rows; anything else is a different split.

The harness's own reference arms now write the field into their manifests too,
so they are held to the same check rather than exempt from it: the run summary
reports **3 of 3 runs** carrying it and matching. Rebuilding both arms from
scratch to pick up the new manifest reproduced every number in this document
unchanged — accuracy, macro F1, ECE, temperature and Brier to the digit — which
is the check that the arms are actually deterministic rather than just seeded.

### The token-budget exposure, measured

His manifest flags the real risk itself: `budget_caveat` says the category
prefix "can displace product text within the same token budget", the split's
grouping was sealed at **128** tokens, and this model reads **256**. If
truncation at 256 made two rows identical that were different at 128, those rows
could sit in different partitions and the model would see the same input in
train and validation.

I rebuilt both representations with the project's own formatter — the four-field
text at its character limits, the category prefix at 512 characters — and the
sealed tokenizer from `splits2/tokenizer`, pinned to `tokenizers==0.22.2`:

| Representation | Identical inputs appearing in **both** train and validation |
|---|---|
| Four fields, 128 tokens (the sealed contract) | **0** |
| Five fields, 256 tokens (what this model read) | **0** |

The mechanism does not fire on development. That does not certify the final
test, which his manifest correctly leaves at
`final_test_representation_certified: false`, and this harness never opens it.

---

## 3. The baseline that should change how we read every score

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

§4.1 says an approach that does not clearly beat the lookup and TF-IDF baselines
is not worth its operating cost. MiniLM beats them both — by 6.31pp over a
lookup and 3.25pp over TF-IDF. That is a real margin, and a much smaller one
than 94.45 implies on its own.

**This corrects §4.1 of the evaluation strategy.** I wrote there that the
breadcrumb lookup "covers only ~9.5% of rows at ≥90% purity, so it is a partial
baseline". That measured coverage at a purity threshold. Measured as a
most-common-label mapping — which is what a lookup baseline is — it covers 99.2%
and scores 88.14. It is not a partial baseline. It is the number every model has
to be compared against.

---

## 4. Full scoreboard, with cost

| Run | Accuracy | Macro F1 | Training cost |
|---|---|---|---|
| **MiniLM, five fields, 256 tokens** | **96.95%** [96.51, 97.34] | **94.45%** [93.56, 95.19] | 1,356s on an RTX 3060 laptop GPU |
| TF-IDF + logreg, five fields | 94.19% [93.57, 94.72] | 91.20% [90.07, 92.04] | 37s, CPU, no GPU |
| Lookup on `ProductCategory` | 92.85% | 88.14% | seconds, no model |
| TF-IDF + logreg, four fields | 88.62% [87.68, 89.52] | 83.14% [81.29, 84.50] | 37s, CPU, no GPU |
| Majority class | 61.75% | 9.54% | none |

Intervals are 95% group-aware bootstrap, resampled over product groups.

**Cost, which §7 asked for and nobody had filled in.** MiniLM: 1,356 seconds,
3 epochs, 5,229 optimizer steps, 22.7M trainable parameters, fp32, deterministic
algorithms on, on an RTX 3060 laptop GPU. The TF-IDF arm: 37 seconds end to end
on CPU in this container, no GPU. That is roughly **37× the training time plus a
GPU requirement, for +3.25pp macro F1.** Inference latency is still unmeasured
and matters more than training time for a daily pipeline.

### Cross-check against Max's own run

Max's packaged handoff reports 83.90% / 91.00% macro F1 and 89.57% / 94.42%
accuracy for his base and category arms, on these same 11,952 rows. My
independently built TF-IDF arms give 83.14% / 91.20% and 88.62% / 94.19%. Two
people, two codebases, the same splits, within **1pp on every figure** — the
largest gap is 0.95pp on the base arm's accuracy and the smallest is 0.20pp on
the category arm's macro F1. His category gain is +7.10pp and mine is +8.06pp.

That is reassuring about both, and it means my arms are a stand-in rather than a
rival: **his handoff zip should replace them in §1 once it arrives**, and the
loader now reads his packaging without changes. His packaging note says plainly
that his probabilities are uncalibrated, which is exactly what §7 below supplies.

---

## 5. Before and after the feature change (§4.6)

Same classifier, same splits, same rows. The only difference is whether
`ProductCategory` is in the text.

| | Accuracy | Macro F1 |
|---|---|---|
| Four fields | 88.62% | 83.14% |
| **Five fields** | **94.19%** | **91.20%** |
| Difference | +5.56pp | **+8.06pp** |

McNemar on the 11,952 shared rows: 802 rows the five-field model gets right and
the four-field one gets wrong, against 137 the other way. **p ≈ 4e-104.** The
bootstrap intervals are disjoint on both metrics.

That is a **fourth** independent measurement of the category gain, in the same
band as the other three:

| Model | Gain from `ProductCategory` |
|---|---|
| Logistic regression (Max) | +7.10pp |
| Linear SVM | +8.14pp |
| MiniLM, 256 tokens (Sebastian) | +8.57pp |
| **Logistic regression, this harness** | **+8.06pp** |

### MiniLM against both arms

| Comparison | Right/wrong split | McNemar p | Δ accuracy | Δ macro F1 |
|---|---|---|---|---|
| MiniLM vs TF-IDF five fields | 466 / 136 | 5e-41 | +2.76pp | +3.25pp |
| MiniLM vs TF-IDF four fields | 1,146 / 151 | 1e-167 | +8.32pp | +11.31pp |

Intervals disjoint in both. MiniLM's win over the five-field arm is real and
modest; its win over the four-field arm is mostly the category feature, which
the cheap model also gets.

---

## 6. Per class, with minority classes in their own tier (§4.2)

MiniLM, every class:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Vitamins, Minerals & Supplements | 7,380 | 0.982 | 0.989 | 0.986 |
| Digestive Health | 1,068 | 0.951 | 0.929 | 0.940 |
| External Analgesics | 859 | 0.970 | 0.969 | 0.969 |
| Internal Analgesics | 727 | 0.920 | 0.935 | 0.928 |
| Lifestyle CHC | 687 | 0.971 | 0.936 | 0.953 |
| CCFS | 645 | 0.952 | 0.922 | 0.937 |
| Allergy | 413 | 0.919 | 0.935 | 0.927 |
| **Other Self Care** *(minority)* | **173** | 0.914 | 0.919 | **0.916** |

Seven main classes average 94.85 macro F1; the one minority class is 91.64. A
3.2pp tier gap, which is small — **the minority class holds up**. It is also
where the feature change helped most: `Other Self Care` goes 0.806 → 0.895 →
0.916 across four-field TF-IDF, five-field TF-IDF and MiniLM.

---

## 7. Calibration (§4.3) — and a correction to what I wrote before

The README makes confidence a product requirement, so the number has to mean
something.

| | TF-IDF 4 fields | TF-IDF 5 fields | **MiniLM 5 fields** |
|---|---|---|---|
| Expected calibration error | 11.09% | 6.84% | **1.37%** |
| Brier score (multiclass) | 0.2003 | 0.0988 | **0.0492** |
| Temperature | 0.60 | 0.70 | **1.20** |
| ECE after scaling | 2.38% | 0.31% | 0.53% |
| Accuracy before / after | 88.62 / 88.62 | 94.19 / 94.19 | 96.95 / 96.95 |

**The two model families are miscalibrated in opposite directions, and I had
this backwards for MiniLM.** An earlier version of this document said "the model
is underconfident" without naming which model. That was measured on the TF-IDF
five-field arm, where it is true — every bin has accuracy above its mean
confidence, so a temperature below 1 sharpens it. **MiniLM is the reverse:
temperature 1.20, above 1, which softens.** It is overconfident.

Anyone who took "scale confidence down to stop over-flagging" and applied it to
MiniLM would have moved the threshold the wrong way.

MiniLM's reliability bins show where it matters:

| Confidence bin | Rows | Share | Mean confidence | Accuracy | Gap |
|---|---|---|---|---|---|
| 0.7–0.8 | 174 | 1.5% | 0.754 | 0.638 | **−0.116** |
| 0.8–0.9 | 230 | 1.9% | 0.855 | 0.739 | **−0.116** |
| 0.9–1.0 | 11,361 | 95.1% | 0.995 | 0.986 | −0.010 |

95% of rows sit above 0.9 and there the confidence is nearly honest. But in the
0.7–0.9 band it is **about 12 points overconfident** — and that band is exactly
the rows a review queue would pull. **The rows you would send to a human are the
rows whose confidence you can trust least.** Scale before thresholding.

MiniLM's raw ECE of 1.37% is already better than the four-field arm manages
*after* scaling (2.38%), and scaling improves it further to 0.53%.

**But the best-calibrated run here is not the most accurate one.** After scaling,
the TF-IDF five-field arm reaches **0.31% ECE against MiniLM's 0.53%** — the
cheap model's confidence is, once corrected, slightly more trustworthy than the
transformer's. That does not make it the model to pick; MiniLM is 3.25pp better
on macro F1, which is the metric that matters. It does mean **calibration and
accuracy are separate questions and should not be used as evidence for each
other.** Scaling is worth applying to whichever model is chosen, because it costs
nothing: the predicted class never moves, which the notebook asserts rather than
assumes.

Calibration has to be measured per target. These are Segment's temperatures, not
the project's.

---

## 8. Selective prediction (§4.4)

MiniLM, most confident rows answered first:

| Coverage | Answered | Accuracy on answered | Macro F1 | Confidence cutoff |
|---|---|---|---|---|
| 100% | 11,952 | 96.95% | 94.45% | — |
| **90%** | 10,757 | **99.39%** | 98.80% | 0.982 |
| **75%** | 8,964 | **99.88%** | 99.45% | 0.996 |
| 50% | 5,976 | 99.98% | n/a | 0.999 |
| 25% | 2,988 | 100.00% | n/a | 0.999 |

Abstaining on 10% of rows buys **+2.44pp, to 99.39%**. That is a strong
operating point, and the cutoff is the thing to notice: **0.982**, not 0.80.
Because 95% of rows sit above 0.9, any threshold in the range people reach for
by habit abstains on almost nothing.

**Macro F1 is `n/a` below 75% coverage, and that is a fix to this harness, not a
property of the model.** Taking the most confident rows first does not preserve
the class mix. At 50% coverage only **2 of 8** classes survive the cut (5,975
Vitamins rows and a single Allergy row) and at 25% only one does. The earlier
version averaged over whatever classes were left and printed 50.00% at 50%
coverage and 100.00% at 25% — non-monotonic, and read as the model collapsing
when accuracy was 99.98%. The table now reports how many classes remain and
withholds the average when any are missing. Accuracy on the answered rows stays
comparable throughout; the macro average does not.

This defect only surfaces on a model confident enough that its top half is
nearly all one class, which is why neither TF-IDF arm ever triggered it. **It
took Sebastian's run to expose it.**

---

## 9. The review queue (§4.5)

MiniLM, 365 errors in 11,952 rows.

**Rule 1 — low calibrated confidence**

| Threshold | Review load | Flag precision | Miss rate |
|---|---|---|---|
| 0.80 | 3.90% | 36.91% | 52.88% |
| 0.90 | 6.38% | 30.93% | 35.34% |
| 0.95 | 9.06% | 26.50% | 21.37% |

**Rule 2 — small margin between top-1 and top-2**

| Threshold | Review load | Flag precision | Miss rate |
|---|---|---|---|
| 0.10 | 0.25% | 56.67% | 95.34% |
| 0.20 | 0.59% | 47.89% | 90.68% |
| 0.40 | 1.46% | 41.14% | 80.27% |
| 0.60 | 2.86% | 38.60% | 63.84% |

**Correcting how I compared these two rules.** The earlier version put margin
slightly ahead by quoting 51.6% precision at 1.8% load against 51.6% at 1.1% —
which compares them at *different* review loads, and at equal precision the
cheaper rule is the better one. §5.1 of the notebook now fixes the load and asks
which rule finds more errors inside it. MiniLM:

| Review load | Flagged | Confidence precision | Margin precision | Margin − confidence |
|---|---|---|---|---|
| 1% | 120 | 44.17% | 44.17% | 0.00 |
| 2% | 239 | 42.26% | 41.84% | −0.42 |
| 5% | 598 | 33.78% | 33.95% | +0.17 |
| 10% | 1,195 | 25.02% | 24.77% | −0.25 |
| 15% | 1,793 | 18.35% | 18.29% | −0.06 |
| 25% | 2,988 | 11.85% | 11.81% | −0.03 |

**On MiniLM the two rules are indistinguishable** — neither is ahead by more than
0.5pp at any load, and the largest gap in either direction is 0.42pp.

On the linear arms margin does win, but not uniformly: on the five-field arm it
is ahead by 5.8pp and 5.4pp at the two lightest loads, **behind by 1.3pp at 5%**,
and within 2pp above that. On the four-field arm it is ahead at five of six
loads, by up to 7.5pp, and behind by 4.2pp at the lightest.

So §4.5's guess that margin "often catches more real errors than raw confidence"
holds for the linear models, weakens with load, and **disappears entirely for the
transformer**. Neither rule is the one to ship on its own, and the choice is per
model, not per project.

**Rule 3 — rare predicted class.** Weaker on MiniLM than on the baselines: 1.46%
load, 8.62% flag precision, 95.89% miss rate, against 16–25% precision on the
TF-IDF arms. That is not the rule getting worse, it is MiniLM getting good at
`Other Self Care` (F1 0.916), so flagging the class now mostly flags correct
predictions. Keep it as a conjunct, not a rule.

**Rule 4 — predicted placeholder.** Does not apply to Segment. It is a
Brand/Mnfr rule, keyed to classes like "UnItemised brand" with the source field
populated. `Other Self Care` is a real Segment class, and flagging it here would
only repeat rule 3. The harness implements the rule properly for Brand and Mnfr
and says plainly that it does not apply here.

**Rule 5 — relationship violations R1–R6.** **Unwired, therefore unmeasured.**
It needs the rules from the relationships research exported machine-readably.
Saying so is better than reporting it as passing.

**The honest summary.** On MiniLM, 9% review load catches 79% of the errors.
That is a far better trade than on either baseline, because there are only 365
errors to find. No single rule gets flag precision above about 57%, and the
number that decides whether any of this saves money is review load — **nobody
has set a target for it.**

---

## 10. Where the errors are

**MiniLM makes 365 errors, and the top 10 confusion pairs are 75.3% of them.**

| Count | True | Predicted |
|---|---|---|
| **72** | **Digestive Health** | **Vitamins, Minerals & Supplements** |
| **44** | **Vitamins, Minerals & Supplements** | **Digestive Health** |
| 31 | CCFS | Allergy |
| 27 | Lifestyle CHC | Vitamins, Minerals & Supplements |
| 22 | External Analgesics | Internal Analgesics |
| 20 | Internal Analgesics | External Analgesics |

### One pair is a third of all errors — in all three runs

`Vitamins ↔ Digestive Health` in both directions is the dominant pair for every
run scored, at a strikingly stable share: **31.8% of MiniLM's 365 errors, 34.8%
of the five-field arm's 695, 32.7% of the four-field arm's 1,360.**

§8.1 of the notebook reads the retailer breadcrumb on every row of that pair and
asks whether it names the true label or the predicted one. The answer moves the
same way as model quality:

| Run | Breadcrumb names the **predicted** label | Names the **true** label | Neither |
|---|---|---|---|
| TF-IDF, four fields | 43.1% | 50.8% | 6.1% |
| TF-IDF, five fields | 65.3% | 26.0% | 8.7% |
| **MiniLM, five fields** | **80.2%** | 14.7% | 5.2% |

Read down that first column. **The better a model gets at using the breadcrumb,
the more its remaining errors are cases where it followed the breadcrumb and the
gold label disagreed.** The four-field model, which never sees the breadcrumb,
is mostly wrong in ways the breadcrumb would have corrected. MiniLM is mostly
"wrong" in ways the breadcrumb endorses.

For MiniLM that is 93 of 116 rows. Of the 86 whose breadcrumb's second level is
literally "Vitamins, Minerals & Supplements", **59 carry the gold label Digestive
Health** and 27 the reverse — products like digestive enzymes and probiotics,
which a retailer files under supplements and our labels file under digestion.

**That is a taxonomy question for the sponsor, not a modelling problem.** No
amount of training fixes a boundary where the strongest feature and the label
contradict each other by design. It should be asked before anyone tries to tune
it away.

A detail that matters for the review queue: MiniLM is *less* sure on this pair
than on its other errors (0.798 against 0.846), so a confidence rule catches
some of it. Both linear arms are *more* sure on it than on their other errors
(0.683 vs 0.646; 0.628 vs 0.570), so for them a confidence rule works against
the problem.

Note the direction flip: the TF-IDF arms' dominant direction was Vitamins →
Digestive Health (200 and 330 rows); MiniLM's is the reverse, Digestive Health →
Vitamins (72). The boundary moved rather than closed.

### The analgesics pair is the model's own

External ↔ Internal Analgesics, 42 errors in both directions. Running the same
check on it: **in 40 of the 42 the breadcrumb names neither label**, and in 2 it
names the predicted one. No source conflict — that one is a genuine model
limitation and the most likely candidate for a targeted fix.

`CCFS ↔ Allergy` (48 errors) cannot be checked this way at all: "CCFS" carries
no distinctive word for the cue rule to match, so §8.1 reports it as undecidable
rather than counting it as "neither" and quietly inflating that column.

---

## 11. What this does not answer

- **No model is selected.** Cost is now in §4, but inference latency is not, and
  that is the figure a daily pipeline cares about.
- **Review rule 5 is unmeasured**, and the review load target is unset.
- **Max's arms are not scored yet** — my TF-IDF arms stand in for them until his
  handoff zip arrives. The loader now accepts his packaging.
- **Only Segment**, and only validation. The reserved test stays closed.
- **Calibration is per target.** Segment's temperatures say nothing about
  Sub-Segment.

---

## 12. What I would do next, in order

1. **Take the `Vitamins ↔ Digestive Health` boundary to the sponsor**, with the
   93-of-116 breadcrumb finding. It is a third of the best model's errors and it
   looks like a label-definition conflict rather than a model failure.
2. **Set a review load target with the PO.** Every queue number is meaningless
   without one. 9% load for a 21% miss rate may be excellent or unaffordable;
   only they know.
3. **Drop Max's handoff zip into §1** in place of my stand-in arms, and give him
   back the calibrated confidence his packaging note says he does not have.
4. **Measure inference latency** for MiniLM and the TF-IDF arm. Training cost is
   settled; serving cost is the one that recurs.
5. **Export R1–R6 machine-readably** so rule 5 stops being a blank.
