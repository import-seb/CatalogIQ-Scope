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
**Runs scored:** Sebastian's five-field, 256-token MiniLM, and Max's two
packaged TF-IDF arms — four fields and those fields plus `ProductCategory`.
**Owner:** Maria Garcia Sehara

---

## 1. What this is

The harness takes prediction files in the contract the team already writes —
`record_id`, `true_label`, `predicted_label`, `confidence`,
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
   carries that field must match or it is refused rather than scored. §2 of this
   document covers the recipe and both directions of the test. All three runs now carry it.
3. **The class order is read from the run's manifest and then checked against
   the data.** If the argmax of the probability columns disagrees with
   `predicted_label` on more than 1% of rows, the order is wrong and the run is
   rejected. Reading an order out of a file and trusting it is how calibration
   numbers end up meaningless.
4. **The bootstrap resamples product groups, not rows.** Rows inside a group are
   not independent, so a row bootstrap would give intervals that are too narrow.
5. **Temperature scaling cannot change accuracy**, and the notebook's §4 carries
   a real `assert` on it rather than a comment saying so.

### It accepts however a teammate packages a run

Discovery reads `training_manifest.json`, `protocol.json`, `model_config.json`,
`label_mapping.json` or `manifest.json`, and pulls the class order out of any of
the shapes those files use — a `label_names` list, a bare list, index→class,
class→index, and the `{"probability_0": "Allergy", …}` form keyed by column name
that the baseline handoff writes.

Getting that right took two attempts. Max's handoff ships `label_mapping.json`;
the original loader required `training_manifest.json` and would have skipped his
arms entirely. My first fix accepted the filename but still could not read his
*shape*, because his keys are column names rather than bare integers. The reader
now pulls a trailing index out of either side of the mapping, and an assert
block exercises it on ten shapes plus three files with no usable order, which
must return nothing rather than a guess.

### It runs with or without the team's arms

The notebook's §1 prefers the team's packaged arms when
`data/processed/max_tfidf_20261019/{base,category}` is present, and falls back to
building its own TF-IDF arms when it is not, so the harness stays runnable by
anyone without the handoff zip. Discovery skips that directory for the same
reason it skips the harness's own baselines: otherwise the arms load twice and
the notebook's §7 compares a run against itself.

Max's README suggested deleting the auto-discovered duplicates after the fact.
Skipping the directory during discovery reaches the same state without a
clean-up step that someone could forget, and keeps the fallback working.

---

## 2. Verifying the runs before scoring them

A prediction file is worth nothing if it was produced on a different split or
scored against different labels. Every run is checked first.

### Sebastian's MiniLM

| Check | Result |
|---|---|
| `predictions.csv` SHA-256 against the `artifacts` block of its own manifest | matches, `394e7415…` |
| Row count against `validation_rows` | 11,952 = 11,952 |
| Duplicate `record_id` | 0 |
| Rows in the **reserved test** partition | **0** |
| Row set against the Segment-labelled validation rows | identical — 0 missing, 0 extra |
| Per-class counts against `validation_class_counts` | all 8 match |
| His `true_label` against `splits2/validation.csv` `Segment`, row by row | 0 disagreements |
| Probabilities sum to 1 | max deviation 1.9e-07 |
| Argmax of probabilities equals `predicted_label` | every row |
| **Accuracy, macro F1, weighted F1 recomputed from the raw predictions** | **identical to his manifest to 10 decimal places** |
| `validation_record_ids_sha256` recomputed from our own split | matches, `47f60a59…` |

### Max's two arms

| Check | base (4 fields) | category (5 fields) |
|---|---|---|
| Declared file checksums in the root manifest | **15 of 15 match**, nothing undeclared | — |
| Rows / duplicate `record_id` | 11,952 / 0 | 11,952 / 0 |
| Rows in the **reserved test** partition | **0** | **0** |
| Row set against the Segment-labelled validation rows | identical | identical |
| `true_label` against our split, row by row | 0 disagreements | 0 disagreements |
| Probabilities sum to 1 | 8.9e-16 | 8.9e-16 |
| Argmax equals `predicted_label`; `confidence` equals the max | yes, exact | yes, exact |
| **Metrics recomputed from the raw predictions** | **exact to 10 dp** | **exact to 10 dp** |
| `validation_record_ids_sha256` in the arm's manifest | matches `47f60a59…` | matches `47f60a59…` |
| `validation_errors.csv` row count against the real errors | 1,246 = 1,246 | 667 = 667 |

**The zip is not the one his October 16 note describes.** That one was 3,896,076
bytes with SHA-256 `77461484…`; this is 3,899,891 bytes with `bd2b9d15…`. His
README explains why — he added a `manifest.json` beside each arm's
`predictions.csv` so that the split-hash guard in §0.1 runs per arm, where the
earlier archive kept the hash only in the parent manifest. He states the
prediction bytes and scores are unchanged, and the recomputed metrics match his
October 16 figures exactly, so that holds.

His own checks are listed in the manifest (`authoritative_development_seal`,
`class_order_and_probabilities`, `headline_metrics` and four more). Those are his
claim; the table above is the independent check, run here against our own split.

**His run is traceable to merged code.** `run_metadata.json` declares SHA-256 for
the six source modules the run used. All six match the files on `main` — two of
them byte for byte, and the other four once CRLF line endings are accounted for,
which is how those four sit in his working copy. That last part matters: on a raw
byte comparison four of the six appear not to match any version of those files in
the repository's full 54-commit history, which would have read as a run built from
code nobody can locate. It is a line-ending difference and nothing more. Worth
recording because the next person to check these hashes will hit the same thing.

### The record-id hash, and why it is now a guard

That hash took two tries. My first attempt failed across 24 serialisation
recipes, all wrong for the same reason: none of them length-prefixes the ids.
The real recipe is `_hash_items()` in `src/catalogiq/segment_transformer.py` —
ids in order, each one's UTF-8 byte length written as **8 bytes big-endian**
before the bytes themselves:

```python
digest = hashlib.sha256()
for record_id in sorted(validation_record_ids):
    payload = record_id.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
```

Over the 11,952 **labelled** validation rows it reproduces `47f60a59…` exactly.
(The unlabelled rows must be dropped first; over all 12,667 it does not match.
For our export the sort is a no-op, because `assignments.csv` already lists
validation rows in lexicographic order — but it is the documented recipe, so it
stays.)

**§0.1 recomputes this from our split and checks every run that carries the
field**, so a run trained against a different split is refused instead of
scored. Tested in both directions: all three runs report a match, and a copy of
one with a deliberately wrong hash is rejected with a message naming the
mismatch and left out of the scoring while the others continue.

### The token-budget exposure, measured

Sebastian's manifest flags a real risk itself: `budget_caveat` says the category
prefix "can displace product text within the same token budget", the split's
grouping was sealed at **128** tokens, and his model reads **256**. If truncation
at 256 made two rows identical that were different at 128, those rows could sit
in different partitions and the model would see the same input in train and
validation.

I rebuilt both representations with the project's own formatter and the sealed
tokenizer from `splits2/tokenizer`, pinned to `tokenizers==0.22.2`:

| Representation | Identical inputs appearing in **both** train and validation |
|---|---|
| Four fields, 128 tokens (the sealed contract) | **0** |
| Five fields, 256 tokens (what his model read) | **0** |

The mechanism does not fire on development. That does not certify the final
test, which his manifest correctly leaves at
`final_test_representation_certified: false`, and this harness never opens it.

---

## 3. The baseline that should change how we read every score

| Baseline | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 61.75% | **9.54%** |
| **Lookup on `ProductCategory` alone** | **92.85%** | **88.14%** |
| Max's TF-IDF + logreg, five fields | 94.42% | 91.00% |

**A lookup table gets 88.14 macro F1 with no model at all.** 1,209 keys learned
on train, covering 99.2% of validation rows, each mapped to its most common
Segment. Against that:

| | Macro F1 | Over the lookup |
|---|---|---|
| Lookup on `ProductCategory` | 88.14 | — |
| **Max's TF-IDF + logreg, five fields** | **91.00** | **+2.86pp** |
| Linear SVM, five fields | 92.46 | +4.32pp |
| **MiniLM, five fields, 256 tokens** | **94.45** | **+6.31pp** |

§4.1 of the strategy says an approach that does not clearly beat the lookup and
TF-IDF baselines
is not worth its operating cost. MiniLM beats them both — by 6.31pp over a
lookup and 3.45pp over Max's category arm. That is a real margin, and a much
smaller one than 94.45 implies on its own.

**This corrects §4.1 of the evaluation strategy.** I wrote there that the
breadcrumb lookup "covers only ~9.5% of rows at ≥90% purity, so it is a partial
baseline". That measured coverage at a purity threshold. Measured as a
most-common-label mapping — which is what a lookup baseline is — it covers 99.2%
and scores 88.14. It is not a partial baseline. It is the number every model has
to be compared against. Note that **the real TF-IDF arm clears it by only
2.86pp**, narrower than my stand-in arm suggested.

---

## 4. Full scoreboard

| Run | Accuracy | Macro F1 |
|---|---|---|
| **MiniLM, five fields, 256 tokens** | **96.95%** [96.51, 97.34] | **94.45%** [93.56, 95.19] |
| Max's TF-IDF + logreg, five fields | 94.42% [93.88, 94.93] | 91.00% [89.95, 91.80] |
| Lookup on `ProductCategory` | 92.85% | 88.14% |
| Max's TF-IDF + logreg, four fields | 89.57% [88.84, 90.41] | 83.90% [82.32, 85.07] |
| Majority class | 61.75% | 9.54% |

Intervals are 95% group-aware bootstrap, resampled over product groups.

### Cost cannot be compared yet, and should not be faked

MiniLM reports 1,356 seconds of training on a named RTX 3060 laptop GPU, 3
epochs, 5,229 optimizer steps, 22.7M trainable parameters, fp32, deterministic
algorithms on.

Max's run reports **53.38 seconds of combined wall time for both arms**, and his
metadata is explicit about what that is not: it covers data loading, fitting,
validation scoring and artifact writes for the two arms together,
`per_arm_training_seconds` is null, `inference_latency_seconds` is null, hardware
is "not recorded", and his `cost_note` says not to attribute the combined time to
either model or treat it as a per-model benchmark.

So the honest statement is that **the two cost figures are not comparable**, and
the figure that matters most for a daily pipeline — inference latency — is
unmeasured for every run. An earlier version of this document put "37× the
training time" in a table. That compared MiniLM's GPU training against *my own*
stand-in arms on this container's CPU, which is not the team's baseline and not a
controlled comparison. It is withdrawn.

### My stand-in arms, and what they were wrong about

Until the handoff arrived, the notebook's §1 built its own TF-IDF arms so it had
something to run on. They were close to Max's on accuracy — within 1pp on every
headline figure — but they differed in two ways worth recording.

| | My stand-in | Max's run |
|---|---|---|
| Four fields: accuracy / macro F1 | 88.62% / 83.14% | 89.57% / 83.90% |
| Five fields: accuracy / macro F1 | 94.19% / 91.20% | 94.42% / 91.00% |
| Four-field ECE / Brier | 11.09% / 0.2003 | **3.61% / 0.1560** |
| Five-field ECE / Brier | 6.84% / 0.0988 | **2.65% / 0.0858** |
| `class_weight` | `"balanced"` | `None` |
| `max_features` / `max_iter` | 80,000 / 2,000 | 100,000 / 1,000 |

**The whole gap is `class_weight`, and that is measured, not inferred.**
Refitting my own arms with `class_weight=None` and changing nothing else — a
one-off diagnostic run outside the notebook, since the harness's job is to score
runs rather than to re-fit them, repeatable from the settings in the table below:

| My arm | ECE | Brier | Accuracy | Macro F1 |
|---|---|---|---|---|
| Four fields, `"balanced"` | 11.09% | 0.2003 | 88.62% | 83.14% |
| Four fields, `None` | **3.58%** | **0.1552** | 89.69% | 84.00% |
| *Max's four fields, for comparison* | *3.61%* | *0.1560* | *89.57%* | *83.90%* |
| Five fields, `"balanced"` | 6.84% | 0.0988 | 94.19% | 91.20% |
| Five fields, `None` | **2.54%** | **0.0844** | 94.53% | 91.07% |
| *Max's five fields, for comparison* | *2.65%* | *0.0858* | *94.42%* | *91.00%* |

Dropping the one setting moves my arms onto his numbers — **within 0.11pp of ECE,
0.0014 of Brier and 0.15pp of accuracy and macro F1** — even though my vectorizer
settings still differ from his. Reweighting the loss to balance classes skews the
predicted probabilities, and that is the entire difference. My arms were not worse
baselines; they were differently calibrated ones.

An earlier version of this document reported the category path cutting ECE from
11.09% to 6.84% and called it a large calibration win. The honest comparison is
**3.61% to 2.65%** — still an improvement, and a much smaller one. **Most of what
I attributed to the feature was an artifact of my own classifier setting.**

The cross-check is now much tighter than the 1pp I claimed before: two people, two
codebases, the same splits, matched on that one setting, **within 0.15pp on all
four headline numbers.** But the reference arms are his, and the numbers in this
document are his.

---

## 5. Before and after the feature change (§4.6)

Same classifier, the same splits, the same rows. The only difference is whether
`ProductCategory` is in the text.

| | Accuracy | Macro F1 |
|---|---|---|
| Four fields | 89.57% [88.84, 90.41] | 83.90% [82.32, 85.07] |
| **Five fields** | **94.42%** [93.88, 94.93] | **91.00%** [89.95, 91.80] |
| Difference | +4.84pp | **+7.09pp** |

McNemar on the 11,952 shared rows: 706 rows the five-field arm gets right and
the four-field one gets wrong, against 127 the other way. **p ≈ 3e-89.** The
bootstrap intervals are disjoint on both metrics.

Five measurements of the category gain — and they do **not** all mean the same
thing, which an earlier draft of this document got wrong:

| Model | Class weighting | Gain from `ProductCategory` |
|---|---|---|
| **Max's logistic regression, through this harness** | none | **+7.09pp** |
| My independent logreg, matched settings | none | **+7.07pp** |
| My independent logreg, `class_weight="balanced"` | balanced | +8.06pp |
| Linear SVM, `class_weight="balanced"` | balanced | +8.14pp |
| MiniLM, 256 tokens (Sebastian) | none | +8.57pp |

**The two unweighted linear measurements agree to 0.02pp**, from two codebases and
two people. The two weighted ones sit about 1pp higher, so **class weighting
inflates the apparent gain** — the same setting that wrecked my calibration in §4
of this document.

An earlier draft presented these as four independent measurements converging on
7–9pp. They are not independent in that way: they cluster by whether the loss was
reweighted, and the linear SVM measurement carries the same artifact.

MiniLM's +8.57pp is larger than the unweighted linear gain and is **not** a
weighting artifact — its config has no class weighting at all, on training counts
running from 805 rows to 34,448. That is a real difference between model families,
and worth separating from the 1pp the setting was adding.

(Max's own note and my earlier drafts quote his gain as +7.10pp, from subtracting
the rounded 83.90 from 91.00; computed from the unrounded scores it is +7.09pp.
Same measurement.)

### MiniLM against both arms

| Comparison | Right/wrong split | McNemar p | Δ accuracy | Δ macro F1 |
|---|---|---|---|---|
| MiniLM vs five fields | 422 / 120 | 3e-38 | +2.53pp | +3.45pp |
| MiniLM vs four fields | 1,006 / 125 | 6e-151 | +7.37pp | +10.55pp |

Intervals disjoint in both. MiniLM's win over the five-field arm is real and
modest; its win over the four-field arm is mostly the category feature, which the
cheap model also gets.

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

Tier summary across all three runs:

| Run | 7 main classes | 1 minority class | Gap |
|---|---|---|---|
| MiniLM, five fields | 94.85 | 91.64 | −3.21 |
| Max, five fields | 91.19 | 89.68 | −1.51 |
| Max, four fields | 83.49 | **86.77** | **+3.28** |

**On the four-field arm the minority class does better than the main classes**,
which is the opposite of what the tier split is there to catch. `Other Self Care`
scores 0.868 F1 there on 173 rows — high precision (0.928), lower recall (0.815).
It is a small class the model is cautious about rather than a class it is failing.
Worth knowing before anyone reads a tier gap as a fairness problem by default.

`Other Self Care` across the three runs: 0.868 → 0.897 → 0.916. It improves with
the feature and again with the model, so nothing is being traded away for the
headline.

---

## 7. Calibration (§4.3)

The README makes confidence a product requirement, so the number has to mean
something.

| | Max, 4 fields | Max, 5 fields | **MiniLM, 5 fields** |
|---|---|---|---|
| Expected calibration error | 3.61% | 2.65% | **1.37%** |
| Brier score (multiclass) | 0.1560 | 0.0858 | **0.0492** |
| Temperature | 0.80 | 0.80 | **1.20** |
| ECE after scaling | 0.97% | **0.44%** | 0.53% |
| Accuracy before / after | 89.57 / 89.57 | 94.42 / 94.42 | 96.95 / 96.95 |

**The two model families are miscalibrated in opposite directions.** Both of
Max's arms are *under*confident — temperature below 1, which sharpens — while
MiniLM is *over*confident at 1.20, which softens. An earlier version of this
document said "the model is underconfident" without naming which model; applied
to MiniLM that advice moves a review threshold the wrong way.

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

**The best-calibrated run is not the most accurate one.** After scaling, Max's
five-field arm reaches **0.44% ECE against MiniLM's 0.53%**. The cheap model's
confidence is, once corrected, slightly more trustworthy than the transformer's.
That does not make it the model to pick — MiniLM is 3.45pp better on macro F1 —
but calibration and accuracy are separate questions and should not be used as
evidence for each other.

Adding the category path improves calibration as well as accuracy: ECE 3.61% to
2.65%, and the Brier score nearly halves, from 0.1560 to 0.0858. §4 of this
document records why that improvement looked twice as large in an earlier draft.

Calibration has to be measured per target. These are Segment's temperatures, not
the project's. Max's note says plainly that his probabilities ship uncalibrated,
so this is the piece his handoff was missing.

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

Max's five-field arm, for comparison:

| Coverage | Accuracy on answered | Macro F1 | Confidence cutoff |
|---|---|---|---|
| 100% | 94.42% | 91.00% | — |
| 90% | 97.67% | 96.12% | 0.735 |
| 75% | 99.14% | 98.39% | 0.918 |
| 50% | 99.80% | 99.51% | 0.973 |

Abstaining on 10% of rows buys MiniLM **+2.44pp, to 99.39%**, and the cheap arm
**+3.25pp, to 97.67%**. The cutoff is the thing to notice: MiniLM needs **0.982**
to abstain on a tenth of rows, against 0.735 for the TF-IDF arm. Because 95% of
MiniLM's rows sit above 0.9, any threshold in the range people reach for by habit
abstains on almost nothing.

**Macro F1 is `n/a` below 75% coverage for MiniLM, and that is a fix to this
harness rather than a property of the model.** Taking the most confident rows
first does not preserve the class mix. At 50% coverage only **2 of 8** classes
survive the cut (5,975 Vitamins rows and a single Allergy row) and at 25% only
one does. The earlier version averaged over whatever classes were left and
printed 50.00% at 50% coverage and 100.00% at 25% — non-monotonic, and read as
the model collapsing when accuracy was 99.98%. The table now reports how many
classes remain and withholds the average when any are missing.

Both of Max's arms keep all 8 classes at every coverage point, which is why
neither of them — and neither of my stand-ins — ever triggered the defect. **It
took a run as confident as MiniLM to expose it.**

---

## 9. The review queue (§4.5)

Error counts differ enough between runs that review economics are not
transferable: MiniLM makes 365 errors, Max's five-field arm 667, his four-field
arm 1,246.

**Rule 1 — low calibrated confidence**

| Threshold | MiniLM: load / precision / miss | Max 5-field: load / precision / miss |
|---|---|---|
| 0.80 | 3.90% / 36.91% / 52.88% | 9.12% / 36.70% / 40.03% |
| 0.90 | 6.38% / 30.93% / 35.34% | 15.08% / 28.14% / 23.99% |
| 0.95 | 9.06% / 26.50% / 21.37% | 21.10% / 22.60% / 14.54% |

**Rule 2 — small margin between top-1 and top-2** (MiniLM)

| Threshold | Review load | Flag precision | Miss rate |
|---|---|---|---|
| 0.10 | 0.25% | 56.67% | 95.34% |
| 0.20 | 0.59% | 47.89% | 90.68% |
| 0.40 | 1.46% | 41.14% | 80.27% |
| 0.60 | 2.86% | 38.60% | 63.84% |

### Confidence against margin, at matched load (§5.1)

Comparing the two rules at their own thresholds is not a comparison — equal
precision at twice the review load is a worse rule. §5.1 fixes the load instead.
Difference is margin minus confidence, so positive means margin is ahead:

| Review load | MiniLM | Max, 5 fields | Max, 4 fields |
|---|---|---|---|
| 1% | 0.00 | +3.33 | **−8.33** |
| 2% | −0.42 | +4.18 | −5.02 |
| 5% | +0.17 | −0.17 | −0.50 |
| 10% | −0.25 | +0.59 | +0.84 |
| 15% | −0.06 | +0.28 | +1.23 |
| 25% | −0.03 | +0.03 | +0.74 |

**§4.5's guess does not survive this.** The strategy predicted margin "often
catches more real errors than raw confidence". On MiniLM the two are
indistinguishable — never more than 0.42pp apart at any load. On Max's five-field
arm margin wins at light loads by 3–4pp and ties above that. On his four-field
arm **confidence wins by 8.33pp at the lightest load**, and margin only pulls
ahead above 10% load, by around 1pp.

So the answer is per model and per operating point, not a project-wide rule.
Whichever is chosen, it should be chosen at the review load the team actually
intends to staff.

**Rule 3 — rare predicted class.** Weak on every run: 1.3–1.5% load for 7–9% flag
precision and a miss rate above 95%. Keep it as a conjunct, not a rule.

**Rule 4 — predicted placeholder.** Does not apply to Segment. It is a
Brand/Mnfr rule, keyed to classes like "UnItemised brand" with the source field
populated. `Other Self Care` is a real Segment class, and flagging it here would
only repeat rule 3. The harness implements the rule properly for Brand and Mnfr
and says plainly that it does not apply here.

**Rule 5 — relationship violations R1–R6.** **Unwired, therefore unmeasured.**
It needs the rules from the relationships research exported machine-readably.
Saying so is better than reporting it as passing.

**The honest summary.** On MiniLM, 9% review load catches 79% of the errors — a
far better trade than on either TF-IDF arm, because there are only 365 errors to
find. The best flag precision any single rule reaches on any run is **57.72%** —
confidence below 0.50 on the four-field arm, at 3.09% review load, which also
misses 83% of that arm's errors. So roughly two in five flagged rows are correct
predictions wherever the rule is set, and the number that decides whether any of
this saves money is review load. **Nobody has set a target for it.**

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

### §8.1: is the error the model's fault, or the label's?

When one pair dominates, there are two possible causes and they need different
fixes. Either the model is missing something, or the label disagrees with the
strongest feature and no amount of training will close it. §8.1 reads the
retailer breadcrumb on every row of a run's three biggest pairs and asks whether
it names the true label, the predicted one, both or neither. The cue words come
from the class names, so it works for any target.

**MiniLM's three biggest pairs are 56% of its errors, and they split three
different ways:**

| Pair | Errors | Names predicted | Names true | Neither | Verdict |
|---|---|---|---|---|---|
| Digestive Health ↔ Vitamins | 116 (31.8%) | **80.2%** | 14.7% | 5.2% | the label disagrees with the breadcrumb |
| CCFS ↔ Allergy | 48 (13.2%) | — | — | — | undecidable: "CCFS" has no cue word |
| External ↔ Internal Analgesics | 42 (11.5%) | 4.8% | 0% | **95.2%** | the model's own confusion |

That third verdict is the one worth acting on. **In 40 of the 42 analgesics
errors the breadcrumb names neither label**, so nothing in the source explains
the mistake — it is a genuine model limitation and the most tractable target for
a fix. `CCFS ↔ Allergy` is reported as undecidable rather than counted as
"neither", because "CCFS" carries no word the cue rule can match; inflating the
"neither" column with it would have turned a measurement into a guess.

### The dominant pair tracks model quality, and it is not the model's fault

`Digestive Health ↔ Vitamins` is pair 1 for all three runs, at a stable share —
**34.5% of the four-field arm's errors, 32.5% of the five-field arm's, 31.8% of
MiniLM's** — in the same direction every time. What changes is the explanation:

| Run | Macro F1 | Breadcrumb names the **predicted** label | Names the **true** label | Neither |
|---|---|---|---|---|
| Max, four fields | 83.90 | 62.6% | 31.9% | 5.6% |
| Max, five fields | 91.00 | 76.0% | 18.4% | 5.5% |
| **MiniLM, five fields** | **94.45** | **80.2%** | 14.7% | 5.2% |

Read down that column. **The better a model gets, the more its remaining errors
are cases where it followed the breadcrumb and the gold label disagreed.** For
MiniLM that is 93 of 116 rows. Of the 86 whose breadcrumb's second level is
literally "Vitamins, Minerals & Supplements", **59 carry the gold label Digestive
Health** — products like digestive enzymes and probiotics, which a retailer files
under supplements and our labels file under digestion.

**That is a taxonomy question for the sponsor, not a modelling problem.** No
amount of training fixes a boundary where the strongest feature and the label
contradict each other by design.

And it is not one boundary. `Vitamins, Minerals & Supplements` appears in **two
of the three biggest pairs for every run**. On the five-field arm, `Lifestyle CHC
↔ Vitamins` is 107 errors with the breadcrumb naming the predicted label on
**87.9%** of them — a stronger conflict than the headline pair. The pattern is
that the retailer files a lot of things under supplements that our taxonomy files
by what they treat.

The one exception runs the other way: on the four-field arm `Internal Analgesics
↔ Vitamins` is 178 errors with **70.2% naming neither** label, and adding the
category feature cuts that pair to 76 errors. So the feature fixed a real model
weakness there rather than just importing the retailer's opinion.

An earlier draft claimed the dominant direction flipped between the linear models
and MiniLM. That was an artifact of my stand-in arms. On the team's real arms the
direction is the same in all three runs, and the claim is withdrawn.

### For the review queue, the confidence signal points the wrong way

Of MiniLM's three biggest clusters, **the two that are its own fault are the ones
it is most confident about:**

| Pair | Mean confidence on the pair | On its other errors |
|---|---|---|
| Digestive Health ↔ Vitamins (label conflict) | 0.798 | 0.846 |
| CCFS ↔ Allergy (undecidable) | **0.874** | 0.825 |
| External ↔ Internal Analgesics (model's own) | **0.860** | 0.827 |

A confidence rule catches some of the taxonomy pair, where the model hesitates,
and under-catches the two clusters a reviewer could actually resolve. Both of
Max's arms are *more* confident on their dominant pair than on their other errors
too (0.694 vs 0.671; 0.670 vs 0.608). **Confidence is not a proxy for
reviewability**, and the review rules in §9 cannot be read as if it were.

---

## 11. What this does not answer

- **No model is selected.** Cost is the gap: MiniLM's training time and Max's
  combined wall time are not comparable, and inference latency — the figure a
  daily pipeline actually pays — is unmeasured for every run.
- **Review rule 5 is unmeasured**, and the review load target is unset.
- **Only Segment**, and only validation. The reserved test stays closed.
- **Calibration is per target.** Segment's temperatures say nothing about
  Sub-Segment.
- **The lookup baseline is a baseline, not a shipping candidate.** It is a
  most-common-label mapping learned on train; the 0.8% of validation rows whose
  breadcrumb key is unseen fall back to the majority class, and the 88.14 already
  includes that. What it cannot do is abstain or report a confidence, so neither
  §8 nor §9 applies to it — it cannot be put behind a review queue.

---

## 12. What I would do next, in order

1. **Take the `Vitamins ↔ Digestive Health` boundary to the sponsor**, with the
   93-of-116 breadcrumb finding and the fact that it is a third of the errors in
   every run. It looks like a label-definition conflict rather than a model
   failure, and it is the largest single thing standing between the current
   models and a better number.
2. **Set a review load target with the PO.** Every queue number is meaningless
   without one. 9% load for a 21% miss rate may be excellent or unaffordable;
   only they know. It also decides the confidence-versus-margin question, which
   §5.1 shows depends on the operating point.
3. **Measure inference latency** for MiniLM and for one TF-IDF arm, on the same
   hardware. Training cost is recorded for MiniLM and deliberately not
   attributable for the TF-IDF arms, so latency is the only cost number that can
   be compared fairly.
4. **Send Max the calibrated confidence.** His packaging note says his
   probabilities ship uncalibrated; §7 of this document gives both his arms a
   temperature and the
   scaled ECE, and scaling cannot change his accuracy.
5. **Export R1–R6 machine-readably** so rule 5 stops being a blank.
