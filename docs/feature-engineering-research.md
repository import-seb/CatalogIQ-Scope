# Feature Engineering for Segment Classification — Research and Prioritized Experiments

**Ticket:** research feature engineering for retail product classification and
recommend prioritized experiments.
**Target:** Segment, 8 classes. First milestone.
**Measured on:** the **finalized** splits, protocol `segment-final-20261009`,
produced by the portable workflow in
`docs/findings/02_splitting/03_portable_workflow_20261010.md` and confirmed by
`split_data --verify` (all eleven checks `true`). Train + validation only,
67,728 rows. The reserved test is not read anywhere.
**Notebook:** `notebooks/03-feature-engineering-research.ipynb` — every number
below comes from it.
**Owner:** Maria Garcia Sehara

> **Version 3.** Two corrections since the first draft, both from review.
>
> **Max caught a leak.** The exploratory sections and the descriptive counts ran
> on the whole labelled frame, so test rows were fitted and scored. Nothing was
> selected or tuned on them, but reading an aggregate score over a reserved
> split is still reading it. Everything now goes through a `dev()` guard.
>
> **The splits were the wrong ones.** The first draft used the V3 assignments —
> the ones whose `rule_assignments.csv` hashes to `c2f0e766…`. PR 19 states
> plainly that those are superseded. On the finalized V4 assignments **25.30% of
> rows sit in a different split**, no `group_id` survives, and the test shrank
> from 12,591 to 11,477 labelled rows. Every table below was re-measured. §0 of
> the notebook now refuses to run against a V3 directory and checks the
> candidate file's byte hash against the documented value.
>
> **What moved:** the gain from `ProductCategory` is **+8.14pp** macro F1, not
> +10.45pp — closer to what Max measured independently. **What was withdrawn:**
> the retailer-skew feedback in §7; the finalized allocator fixed it. **What did
> not change:** every conclusion and every recommendation.

---

## 1. Summary

**The headline is not a new feature. It is a column we already have and are not
using.** The agreed model input is four fields — `ProductName`, `ProductBrand`,
`ProductDescription`, `ProductContents` (`MODEL_FIELDS` in
`src/catalogiq/segment_transformer.py`). `ProductCategory` is not in that
allowlist.

Trained on `train` (55,776 rows), scored on `validation` (11,952):

| Input | Accuracy | Macro F1 |
|---|---|---|
| Majority class (`Vitamins, Minerals & Supplements`) | 61.75% | — |
| `ProductName` only | 89.03% | 82.92% |
| **The agreed four fields** | **90.00%** | **84.32%** |
| `ProductName` + `ProductCategory` | 94.86% | 91.99% |
| **The agreed four fields + `ProductCategory`** | **95.36%** | **92.46%** |

TF-IDF word 1–2 grams, linear SVM, `class_weight="balanced"`, `C=1.0`,
`random_state=0`. Nothing tuned. A floor, not a prediction of what MiniLM will
do.

### Replicated with a different model

Max built the Segment baseline independently with **TF-IDF + logistic
regression** (PR 20): the agreed four fields reach **83.90%** macro F1, and
adding concatenated `ProductCategory` reaches **91.00%** macro F1 at **94.42%**
accuracy — a **+7.10pp** gain.

**The finding is confirmed by a second model family, which matters more than my
own repeated runs.** Three things to read off it honestly:

- **The ceiling is the same.** 91.00 against my 92.46 macro F1, 94.42% against
  my 95.36% accuracy. Two different linear models land in the same place once
  the category path is in.
- **The two gains are now close.** +7.10pp on his logistic regression, +8.14pp
  on my SVM. On the superseded V3 splits mine read +10.45pp, so part of that gap
  was the allocation, not the model.
- **His run used the V3 assignments.** He reproduced the `c2f0e766` hash, which
  PR 19 identifies as superseded. His numbers will move a little on the
  finalized splits — mine did, upward — but the direction and the size of the
  gain should not.

**The honest range for E1 is +7 to +8pp macro F1.** Still by far the largest
single change available.

**Three things follow.**

1. **Adding `ProductCategory` is worth +7.10pp on a logistic regression and
   +8.14pp on a linear SVM.** More than everything else in this document
   combined, on either model.
2. **Of the four allowlisted fields, one does the work.** `ProductName` alone
   gets 82.92 macro F1; all four get 84.32 — **+1.40pp for three fields**, two
   of which are empty on most rows and one of which is largely redundant.
3. **A linear model on text already reaches 95.36% accuracy and 92.46% macro
   F1.** That is the bar the transformer has to clear, and it means Max's
   classical baseline is not a formality.

### On reproducing these numbers

**The table above was reproduced on a second machine, on macOS, and the largest
disagreement was 0.09pp.** Both runs used the finalized splits, `random_state=0`,
and the library versions the `splitting` extra pins — pandas 2.3.2,
scikit-learn 1.7.2, numpy 2.3.3, scipy 1.16.2. The second machine measured the
`ProductCategory` gain at +8.16pp against +8.14pp here.

It is still not digit for digit, and it will not be: the remaining cause is
platform floating-point, since macOS Accelerate and Linux OpenBLAS sum in
different orders, which moves a handful of the 11,952 validation rows across a
decision boundary. Before the library versions were pinned the gap was up to
0.4pp, so **installing the `splitting` extra is what makes figures comparable
between us.** Treat ±0.1pp as the tolerance, and compare the gaps between rows
rather than the digits. The notebook prints its own versions in §0 so a mismatch
is visible immediately.

---

## 2. What is actually in the data

79,205 of the 83,938 cleaned rows carry a Segment label: 55,776 train, 11,952
validation, 11,477 test. **The 67,728 train + validation rows are what
everything below describes.**

| Column | Populated | Allowlisted | Distinct values |
|---|---|---|---|
| `ProductName` | 100.00% | yes | 66,318 |
| `Retailer` | 100.00% | no | 6 |
| `ProductUrl` | 100.00% | no | 67,728 |
| `ProductRating` / `XRatXRev` / `ReviewsCount` | 100.00% | no | 202 / 5,588 / 1,003 |
| `ProductCategory` | 99.99% | **no** | 1,291 paths |
| `ProductBrand` | 99.95% | yes | 12,191 |
| `ProductReviewsCount` | 99.86% | no | 2,677 |
| `ProductImageUrl` | 99.34% | no | 66,428 |
| **`ProductContents`** | **38.47%** | yes | 23,879 |
| **`ProductDescription`** | **30.61%** | yes | 18,707 |

Distinct counts include the empty value as one of them, which is why §3.1
reports 1,290 category keys and 12,190 brand keys — those are keys over rows
that actually have a value.

**Two of the four allowlisted fields are absent on about two thirds of rows.**
Any experiment that leans on the description is an experiment on 30% of the data.

Note `ProductUrl`: 67,728 distinct values over 67,728 rows. It is a row
identifier, not a feature.

Class support is very uneven, which is why macro F1 is the metric that moves:

| Segment | Rows | Share |
|---|---|---|
| Vitamins, Minerals & Supplements | 41,828 | 61.8% |
| Digestive Health | 6,050 | 8.9% |
| External Analgesics | 4,868 | 7.2% |
| Internal Analgesics | 4,119 | 6.1% |
| Lifestyle CHC | 3,893 | 5.7% |
| CCFS | 3,654 | 5.4% |
| Allergy | 2,338 | 3.5% |
| Other Self Care | 978 | 1.4% |

Always answering "Vitamins" scores 61.76%.

---

## 3. `ProductCategory`

### 3.1 How much signal it holds

Treat each categorical column as a pure lookup table — map every value to its
most common Segment, learn nothing else:

| Feature as a lookup table | Keys | Rows/key | Accuracy | Keys mapping to one Segment |
|---|---|---|---|---|
| `Retailer` | 6 | 11,288.0 | 61.76% | 0.0% |
| `ProductCategory`, first level | 31 | 2,184.5 | 62.14% | 29.0% |
| `ProductCategory`, first 2 levels | 172 | 393.7 | 74.13% | 48.3% |
| `ProductCategory`, first 3 levels | 683 | 99.2 | 82.88% | 65.0% |
| `ProductCategory`, first 4 levels | 1,149 | 58.9 | 91.06% | 66.8% |
| **`ProductCategory`, full path** | **1,290** | **52.5** | **93.37%** | 66.7% |
| `Retailer` + full path | 1,292 | 52.4 | 93.37% | 66.7% |
| `ProductBrand` | 12,190 | 5.6 | 86.08% | 81.2% |
| `ProductUrl` directory path | 9,521 | 7.1 | 68.63% | 99.7% |

Three things to read off it.

- **A 1,290-row lookup table gets 93.37% of Segment right with no model.**
- **`Retailer` carries no signal at all** — exactly the majority baseline, and
  adding it to the path produces 2 extra keys out of 1,290, because the retailer
  taxonomies are already disjoint.
- **`ProductUrl` looks great and is not.** 99.7% of its keys map to a single
  Segment, but there are 9,521 keys over 67,728 rows — 7 rows per key. High
  purity with few rows per key is memorization, and it scores 68.63%, barely
  above baseline. This is the trap when someone proposes "just use the URL".

Accuracy is measured on the rows that have a value. Coverage is 99.3% or better
for every row in the table, so it makes no material difference.

### 3.2 Where the gain lands

Per-class F1 on validation, agreed four fields against the same plus the path:

| Segment | Support | Agreed 4 | + Category | Gain |
|---|---|---|---|---|
| **Internal Analgesics** | 727 | 0.755 | 0.887 | **+13.2pp** |
| **Allergy** | 413 | 0.802 | 0.922 | **+12.0pp** |
| **Digestive Health** | 1,068 | 0.785 | 0.900 | **+11.5pp** |
| **CCFS** | 645 | 0.820 | 0.931 | **+11.1pp** |
| External Analgesics | 859 | 0.875 | 0.953 | +7.8pp |
| Other Self Care | 173 | 0.853 | 0.906 | +5.3pp |
| Vitamins, Minerals & Supplements | 7,380 | 0.946 | 0.976 | +2.9pp |
| Lifestyle CHC | 687 | 0.909 | 0.922 | +1.3pp |

The gain is concentrated in the minority classes, which is where the business
value is — the majority class was already at 0.946.

### 3.3 The cross-retailer risk, and what it actually is

**Of the 969 `ProductCategory` paths that appear in Walmart, Walgreens, Target,
CVS and Kroger, zero appear in Amazon.** Amazon uses 322 paths. The six
taxonomies do not overlap at all, and Amazon is 53,410 of the 67,728 rows — 79%.

Shopify reports the same problem: merchant-supplied product type is unreliable
because "multiple merchants selling the same product can use different values
for Product Type" (Shopify Engineering). We measured the extreme version.

Train on Amazon only, score the other five retailers (14,318 rows):

| Input | Accuracy | Macro F1 |
|---|---|---|
| `ProductName` only | 81.77% | 72.48% |
| The agreed four fields | 83.19% | 74.29% |
| `ProductName` + `ProductCategory` | 82.83% | 74.52% |
| The agreed four fields + `ProductCategory` | **84.61%** | **76.32%** |

**The path still helps on an unseen retailer, by +2.03pp instead of +8.14pp.** It
does not hurt. An earlier exploratory run of mine said it hurt; §3.4 explains
why.

**What does not survive the move is the overall level.** The best cross-retailer
macro F1 is 76.32 against 92.46 in-distribution. **A ~16pp drop is what moving
to a new retailer costs**, whatever features we use, and it belongs next to the
headline number whenever we quote one to the PO.

### 3.4 How the path is represented decides whether it is safe

The path can enter the model two ways: concatenated into the one text field, the
way `build_product_texts` already does it, or as its own TF-IDF block with its
own feature space.

| Evaluation | Name only | Concatenated | Separate block |
|---|---|---|---|
| In-distribution | 82.92 | **91.99** | 91.76 |
| Cross-retailer (Amazon → other five) | 72.48 | **74.52** | **67.29** |

In-distribution they are equivalent, with concatenation now very slightly ahead.
Cross-retailer they are not: a separate block scores **5.19pp below using no
category at all**. Given its own feature space, Amazon's taxonomy vocabulary
dominates and misleads. Concatenated, the path's tokens are diluted among the
name's and the model keeps leaning on the name.

**So the recommendation is specific: include `ProductCategory`, concatenated
into the existing text stream, never as a separate feature block and never
one-hot encoded.** Which is convenient, because concatenating is what the team's
text builder already does, and it is what Max's baseline does.

Supporting evidence that the path works as *language* and not as a key: split by
category path so that **100% of scored rows have a path never seen in
training** (57,660 train, 10,068 scored):

| Input | Accuracy | Macro F1 |
|---|---|---|
| `ProductName` only | 82.70% | 72.44% |
| The agreed four fields | 83.44% | 72.91% |
| `ProductName` + `ProductCategory` | 89.14% | 80.51% |
| The agreed four fields + `ProductCategory` | **89.46%** | **81.46%** |

Every path is new and the path is still worth **+8.55pp macro F1**, because
"Vitamins", "Allergy" and "Digestive" generalize even when the full path does
not. Accuracy holds up better than macro F1, so the minority classes are the
ones that were relying on having seen the exact path.

---

## 4. What the other three allowlisted fields contribute

`ProductName` alone: 82.92 macro F1. The agreed four fields: 84.32. So
`ProductBrand` + `ProductDescription` + `ProductContents` together are worth
**+1.40pp**.

- **`ProductBrand` is largely redundant.** The brand is usually already inside
  the name — "Cora Period Menstrual Cramp Relief Kit" carries "Cora". On a
  random split over the development set, `ProductName` + `ProductCategory`
  scores 91.58 macro F1 and adding all three remaining allowlisted fields gets
  92.47 — **+0.89pp for brand, description and contents combined**. Isolating
  brand on its own was not measured, so this is as far as the evidence goes.
- **`ProductDescription` is present on 30.61% of rows**, and when present it
  starts at median character 141 of the built text, so the model does see it.
- **`ProductContents` is present on 38.47% of rows and the model mostly never
  sees it.** It starts at median character **623**.

That last point needs the token budget spelled out. `sequence_length` is **128
tokens**, roughly 400–550 characters of product text. The built input is
long-tailed:

| | characters |
|---|---|
| median | 230 |
| 75th percentile | 1,242 |
| 90th percentile | 2,209 |
| maximum | 6,287 |

**38.2% of rows exceed 400 characters**, so about a third of the input is cut
off. Meanwhile `field_character_limits` is `(1024, 512, 4000, 2000)` — 7,536
characters of allowance feeding a window that holds about 500.

| Field | Present | Median start character |
|---|---|---|
| `ProductName` | 100.00% | 0 |
| `ProductBrand` | 99.95% | 128 |
| `ProductDescription` | 30.61% | 141 |
| **`ProductContents`** | **38.47%** | **623** |

`ProductContents` is allowlisted, and on most of the rows that have it, it never
reaches the model. **We currently cannot tell whether contents is a weak field
or just a truncated one.** That is experiment E2.

---

## 5. What carries no signal, so we can stop considering it

Trained on `train`, scored on `validation`:

| Input | Accuracy | Macro F1 |
|---|---|---|
| Numerics (`ProductRating`, `ProductReviewsCount`, `XRatXRev`, `ReviewsCount`) + name length | 54.28% | **12.41%** |
| `Retailer` only | 51.80% | **11.51%** |
| Majority class | 61.75% | — |

Both are below the majority baseline on accuracy and near-worthless on macro F1.
That makes sense: a 4.5-star rating is a 4.5-star rating whether it is on an
antacid or a multivitamin.

**Recommendation: drop review and rating features from the Segment experiment
list entirely.** They may matter for a different question later — which
predictions need human review, say — but not for this one.

---

## 6. Prioritized experiments

Ordered by measured payoff per unit of work. Each one states what it is expected
to buy, because that is the part the ticket asks for and the part that usually
gets asserted instead of measured.

### E1 — Add `ProductCategory` to the allowlist, concatenated as text

| | |
|---|---|
| **Change** | Add `ProductCategory` to `MODEL_FIELDS` so `build_product_texts` concatenates it into the same stream. Not a separate block, not one-hot |
| **Expected payoff** | **+7.10pp macro F1** on a logistic regression, **+8.14pp** on a linear SVM, in-distribution. **+2.03pp** cross-retailer and **+8.55pp** on wholly unseen category paths. Biggest gains on Internal Analgesics (+13.2pp), Allergy (+12.0pp), Digestive Health (+11.5pp), CCFS (+11.1pp) |
| **Cost** | One tuple entry plus a character limit. One retrain |
| **Risk** | Low, given §3.4 — concatenated, it helps in all three evaluations. As a separate block it is harmful cross-retailer |
| **Status** | **Already done on the classical baseline**, PR 20. Still open on the transformer |
| **How we know** | Validation macro F1 and per-class F1 against the current four-field run |

This is the whole document in one line. Everything else is a rounding error next
to it. The remaining question is only whether the transformer gets the same
column, since the baseline already has it — until it does, the two models are
not being compared on the same input.

### E2 — Fix the 128-token truncation

| | |
|---|---|
| **Change** | Per-field token budgets instead of truncating the concatenation, or raise `sequence_length` to 256 |
| **Expected payoff** | Not measurable from here, but bounded: it can only affect the 38.2% of rows that overflow, and the field it recovers is present on 38.47% |
| **Cost** | Config change. 256 tokens roughly doubles transformer training time |
| **How we know** | Validation macro F1 **split by whether the row overflows 128 tokens**. If the gain is only on overflow rows, it is real. If it is spread evenly, something else changed |

Do E2 before concluding anything about whether description and contents are
useful. Right now we cannot distinguish a weak field from a truncated one.

One caution: the frozen 128-token input equality is a **grouping constraint** in
the finalized protocol — `model_view_contract` in the protocol manifest pins the
tokenizer and the signature recipe. Changing `sequence_length` changes what
counts as an identical input, so this is not a free config flip. It needs
Sebastian's call on whether the splits would have to be refrozen.

Note for E1: adding `ProductCategory` makes the truncation worse, since the path
is another 60–120 characters. Put the path **early** in the field order, right
after the name — it is short and it is the second strongest signal. That has the
same implication for the frozen contract.

### E3 — Category path as explicit hierarchy levels

| | |
|---|---|
| **Change** | Emit levels 1–4 as separate tagged segments rather than one path string |
| **Rationale** | Lookup accuracy climbs monotonically with depth: 62.14 → 74.13 → 82.88 → 91.06 → 93.37. Shallow levels generalize across retailers; deep ones do not. Making the structure explicit lets the model weight them separately |
| **Expected payoff** | Small in-distribution on top of E1. The value is cross-retailer robustness |
| **Cost** | A text-building change |
| **How we know** | The Amazon-to-others test in §3.3. If E3 beats the flat path there, it is the compromise worth keeping |

### E4 — Character n-grams on `ProductName`, classical baseline only

| | |
|---|---|
| **Change** | Add `char_wb` 3–5 grams to Max's baseline |
| **Expected payoff** | **Not measured in the notebook.** An exploratory run of mine put it around +1pp macro F1, but that run is not in the delivered artifact, so treat the figure as a guess until Max tests it |
| **Cost** | Slow to fit, large feature matrix |
| **How we know** | Validation macro F1, with and without, same splits |

Worth it for the classical baseline, because product names are full of
concatenations and abbreviations — "Anti-Diarrheal", "500mg", "XL". Not worth it
for the transformer, which already does sub-word tokenization.

### E5 — Class weighting, and reporting macro F1

| | |
|---|---|
| **Change** | Class weights inversely proportional to support, in both models |
| **Rationale** | `Other Self Care` has 978 rows against 41,828 for the majority. Shopify's mitigation on a highly imbalanced taxonomy was class weights rather than resampling (Shopify Engineering) |
| **Expected payoff** | Already inside every number here — all of them use `class_weight="balanced"` |
| **How we know** | Report macro F1 and per-class F1, never accuracy alone |

### Not recommended

| Idea | Why not |
|---|---|
| Review and rating features | 12.41 macro F1 on their own, §5 |
| `Retailer` as a feature | 11.51 macro F1, and adds 2 keys over the path alone, §3.1 |
| `ProductUrl` tokens | 68.63% as a lookup across 9,521 keys, and 67,728 distinct values over 67,728 rows. It is a row identifier, §2 and §3.1 |
| `ProductBrand` as its own field | Largely redundant given `ProductName`, §4 |
| `ProductImageUrl` / image features | Shopify gets value from images, at 250M parameters and hundreds of millions of rows. We have 79,205 rows and 8 classes. Out of scope for this milestone |

---

## 7. Notes on the splits

### The retailer-skew finding is withdrawn

An earlier version of this document reported that validation was 10.3pp more
Amazon-heavy than train, and recommended stratifying the allocation by retailer.
**That was measured on the superseded V3 assignments.** On the finalized
protocol:

| Retailer | Train | Validation | Shift |
|---|---|---|---|
| Amazon | 78.8% | 79.0% | **+0.2pp** |
| Walmart | 11.0% | 11.0% | 0.0pp |
| Walgreens | 3.6% | 3.6% | 0.0pp |
| Target | 3.2% | 3.1% | −0.1pp |
| CVS | 2.8% | 2.7% | −0.1pp |
| Kroger | 0.6% | 0.6% | 0.0pp |

And it is not luck. `development_allocation_config.json` sets
`max_retailer_fraction_delta: 0.02`, so the allocator constrains it directly,
alongside `max_segment_fraction_delta: 0.005` and
`max_group_size_fraction_delta: 0.02`. **The finalized allocator already solves
the problem I was about to report.** Worth saying out loud so nobody spends time
on it.

**Group isolation is clean:** 0.0% of validation `group_id`s appear in train.

### A correction to my own earlier claim about grouping

§2.7 of `model-evaluation-strategy.md` said the absence of a grouping key
"blocks the validity of every reported number", and recommended constructing one
from normalized `ProductName` plus `ProductBrand`.

I built that key and measured it over the development set. It produces **62,213
groups over 67,728 rows — 1.09 rows per group**, with 7.1% of groups holding more
than one row and 14.6% of rows in a multi-row group. Random split against
grouped split:

| Input | Random | Grouped | Gap |
|---|---|---|---|
| The agreed four fields | 85.45 | 85.46 | −0.01pp |
| `ProductName` only | 83.41 | 83.62 | −0.22pp |
| `ProductName` + `ProductCategory` | 91.58 | 91.72 | −0.13pp |
| The agreed four fields + `ProductCategory` | 92.47 | 92.86 | −0.39pp |

**Largest absolute gap 0.39pp — noise, and the grouped split is marginally
better, not worse.** Either near-duplicates are rarer in this catalogue than I
assumed, or my key was too strict to find them. The finalized V4 grouping is far
more sophisticated than mine and may well catch what mine missed — but the
measured effect of grouping on Segment scores is the number that justifies the
effort, and this is it.

### Where the real generalization risk is

Not near-duplicate rows. **Unseen taxonomies.**

| | |
|---|---|
| Validation rows whose category path is unseen in train | **0.8%** |
| Non-Amazon rows whose category path is unseen in Amazon | **100.0%** |

The split that would stress this model is a **retailer holdout**, and it is not
one of the three we have. §3.3 is the closest thing to it and it shows a ~16pp
macro F1 drop. That number deserves to be on the record next to the 92.46.

---

## 8. What the literature says, and where it agrees

Short, because the measurements above are the actual argument.

- **Title-first is right.** Title and description are the standard input for
  e-commerce taxonomy classification, with the title weighted most heavily —
  Gupta et al. (2016) repeat the title three times for exactly that reason. Our
  `build_product_texts` already puts the name first.
- **TF-IDF is close to embeddings on this task.** Gupta et al. report
  precision@1 of 84.40% for their cluster-based word-vector method against
  81.10–82.74% for a TF-IDF baseline on non-book products — about two points.
  Consistent with our 92.46 macro F1 from TF-IDF, and a reason to take Max's
  baseline seriously rather than assume the transformer wins.
- **Merchant-supplied category is unreliable across merchants.** Shopify
  Engineering says so directly. We measured zero path overlap between Amazon and
  the other five retailers.
- **Hierarchical taxonomies are handled level by level.** Shopify's model uses
  one output layer per taxonomy level, each feeding the next, with per-level
  confidence thresholds. That is the pattern to reuse when Segment →
  Sub-Segment comes up, since Sub-Segment is the child level.
- **Class weights for imbalance,** not resampling. Same as E5.

---

## 9. What I did not test

Stated so nobody reads a gap as a finding.

- **No transformer.** Everything here is TF-IDF plus a linear SVM. A floor, not
  a prediction for MiniLM.
- **No hyperparameter tuning.** `C=1.0` and defaults throughout. A tuned
  baseline would score higher.
- **The reserved test is not read anywhere.** This was not true of the first
  version; see the note at the top.
- **No image features**, although `ProductImageUrl` is 99.34% populated.
- **Only Segment.** The other five targets may rank features differently, and
  §3.3 of the evaluation strategy already says Platform is not evaluable as-is.
- **The cross-retailer test is one split**, Amazon against the rest. A strong
  signal, not a tuned estimate — and since Amazon is 79% of the development set,
  the in-distribution numbers lean the same way.
- **Two earlier results of mine were wrong and are corrected here.** One
  exploratory run gave `ProductCategory` a *negative* cross-retailer effect; that
  run gave the path its own TF-IDF block while the team's text builder
  concatenates, and §3.4 has both. And the first draft was measured on the
  superseded V3 assignments.
- **Figures are not reproducible digit for digit across machines**, only to about
  ±0.1pp once the `splitting` extra pins the library versions. §1 explains why,
  and reports the second machine's figures.

---

## 10. Open questions

1. **Does `ProductCategory` join `MODEL_FIELDS` for the transformer?** The
   classical baseline already has it (PR 20) and gained +7.10pp macro F1.
   Leaving it out of the transformer means the two models are not being compared
   on the same input. (§3, E1)
2. **Is `sequence_length = 128` deliberate, and can it change?** It cuts
   `ProductContents` off on most of the rows that have it. But the frozen
   128-token input equality is a grouping constraint in the finalized protocol,
   so changing it may mean refreezing the splits. Sebastian's call. (§4, E2)
3. **Will the model score products from these six retailers, or new ones?** It
   does not change E1 any more, but it decides whether 92.46 or 76.32 is the
   number we quote to the PO. (§3.3)
4. **Should there be a retailer-holdout split**, given that it is the only one
   that stresses the actual generalization risk? (§7)
5. **Has anyone measured what the V4 grouping buys on Segment scores?** My own
   key says the gap is noise, which contradicts what I wrote in §2.7 of the
   evaluation strategy. The V4 method is much stronger than mine, so the number
   may differ — but it has not been measured. (§7)
6. **Does the first version of this notebook having fitted and scored test rows
   matter to the reserved test?** Nothing was selected on them and no row was
   inspected. The finalized test is a strict subset of the old one, so all
   11,477 of its labelled rows were in the set that version touched. The leakage
   audit already notes the test is "not an untouched source of product
   evidence". Sebastian's call. (note at top)

---

## Sources

- Gupta, V., Karnick, H., Bansal, A., & Jhala, P. (2016). [Product Classification in E-Commerce using Distributional Semantics](https://aclanthology.org/C16-1052/). *Proceedings of COLING 2016*.
- Shopify Engineering. [Using Rich Image and Text Data to Categorize Products at Scale](https://shopify.engineering/using-rich-image-text-data-categorize-products).
- Shopify Engineering. [Categorizing Products at Scale](https://shopify.engineering/categorizing-products-at-scale).
- Goumy, S. [Ecommerce Product Title Classification](https://ceur-ws.org/Vol-2319/ecom18DC_paper_5.pdf), SIGIR eCom 2018.
