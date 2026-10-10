# Feature Engineering for Segment Classification — Research and Prioritized Experiments

**Ticket:** research feature engineering for retail product classification and
recommend prioritized experiments.
**Target:** Segment, 8 classes. First milestone.
**Measured on:** the official splits from `scripts.split_data` (PR #18), trained
on `train`, scored on **validation only**. The final test was never touched, per
`docs/findings/02_splitting/02_finalization_20261009.md`.
**Notebook:** `notebooks/03-feature-engineering-research.ipynb` — every number
below comes from it.
**Owner:** Maria Garcia Sehara

---

## 1. Summary

**The headline is not a new feature. It is a column we already have and are not
using.** The agreed model input is four fields — `ProductName`, `ProductBrand`,
`ProductDescription`, `ProductContents` (`MODEL_FIELDS` in
`src/catalogiq/segment_transformer.py`). `ProductCategory` is not in that
allowlist.

Run on two machines independently — a Linux container and a macOS laptop — from
the same raw CSVs, the same cleaner and the same split assignments:

| Input | Accuracy | Macro F1 | Second machine | |
|---|---|---|---|---|
| | | | Accuracy | Macro F1 |
| Majority class (`Vitamins, Minerals & Supplements`) | 61.83% | — | 61.83% | — |
| `ProductName` only | 86.74% | 78.73% | 86.89% | 79.11% |
| **The agreed four fields** | **88.13%** | **80.84%** | **88.04%** | **80.66%** |
| `ProductName` + `ProductCategory` | 94.32% | 91.06% | 94.35% | 91.16% |
| **The agreed four fields + `ProductCategory`** | **94.75%** | **91.29%** | **94.79%** | **91.28%** |

TF-IDF word 1–2 grams, linear SVM, `class_weight="balanced"`, `C=1.0`,
`random_state=0`. Nothing tuned. This is a floor, not a prediction of what
MiniLM will do.

**The two runs agree to within 0.4pp and they do not match digit for digit.**
That is worth stating before anyone treats a small difference as a discrepancy.
It is not the random seed and not the library versions — both runs used
`random_state=0`, the solver converged in 213 iterations against a 3,000 cap,
and the Linux run gave identical results under scikit-learn 1.7.2 and 1.8.0. The
remaining cause is platform floating-point arithmetic: macOS Accelerate and
Linux OpenBLAS sum in different orders, which moves a few hundred of the 11,881
validation rows across a decision boundary.

**So compare the gaps between rows, not the digits.** The gap this document
rests on is +10.45pp on one machine and +10.62pp on the other — two orders of
magnitude outside the noise. Every figure below comes from the Linux run unless
it says otherwise.

One detail for honesty: the macOS column was produced before `random_state=0`
was added, so that run and the notebook as shipped differ by that one argument.
It does not affect the comparison — the Linux run gives byte-identical results
with and without the seed, and under two scikit-learn versions — but a re-run on
macOS with the seed may land a tenth of a point away from the column above.

**Three things follow.**

1. **Adding `ProductCategory` is worth +10.45pp macro F1** (+10.62pp on the
   second machine). More than everything else in this document combined.
2. **Of the four allowlisted fields, one does the work.** `ProductName` alone
   gets 78.73 macro F1. Adding `ProductBrand`, `ProductDescription` and
   `ProductContents` gets 80.84 — **+2.11pp for three fields** (+1.55pp on the
   second machine), two of which are empty on most rows and one of which is
   redundant.
3. **A linear model on text already reaches 94.75% accuracy and 91.29% macro
   F1.** That is the bar the transformer has to clear. It also means Max's
   classical baseline is not a formality.

---

## 2. What is actually in the data

79,205 of the 83,938 cleaned training rows carry a Segment label. Under the
official splits that is **55,443 train, 11,881 validation and 11,881 test**, and
the test is reserved.

| Column | Populated | Allowlisted | Distinct values |
|---|---|---|---|
| `ProductName` | 100.00% | yes | 77,782 |
| `Retailer` | 100.00% | no | 6 |
| `ProductUrl` | 100.00% | no | — |
| `ProductRating`, `XRatXRev`, `ReviewsCount` | 100.00% | no | — |
| `ProductCategory` | 99.99% | **no** | 1,343 paths |
| `ProductBrand` | 99.95% | yes | 14,577 |
| `ProductImageUrl` | 99.41% | no | — |
| **`ProductContents`** | **38.41%** | yes | — |
| **`ProductDescription`** | **30.23%** | yes | — |

The distinct counts above include the empty value as one of them, which is why
§3.1 reports 1,342 category paths and 14,576 brands — those are key counts over
the rows that actually have a value.

**Two of the four allowlisted fields are absent on about two thirds of rows.**
Any experiment that leans on the description is an experiment on 30% of the data.

Class support is very uneven, which is why macro F1 is the metric that moves and
accuracy is not a result:

| Segment | Rows | Share |
|---|---|---|
| Vitamins, Minerals & Supplements | 48,982 | 61.8% |
| Digestive Health | 7,072 | 8.9% |
| External Analgesics | 5,696 | 7.2% |
| Internal Analgesics | 4,799 | 6.1% |
| Lifestyle CHC | 4,549 | 5.7% |
| CCFS | 4,252 | 5.4% |
| Allergy | 2,723 | 3.4% |
| Other Self Care | 1,132 | 1.4% |

Always answering "Vitamins" scores 61.8%.

---

## 3. `ProductCategory`

### 3.1 How much signal it holds

Treat each categorical column as a pure lookup table — map every value to its
most common Segment, learn nothing else:

| Feature as a lookup table | Keys | Rows/key | Accuracy | Keys mapping to one Segment |
|---|---|---|---|---|
| `Retailer` | 6 | 13,200.8 | 61.84% | 0.0% |
| `ProductCategory`, first level | 32 | 2,474.9 | 62.23% | 28.1% |
| `ProductCategory`, first 2 levels | 182 | 435.1 | 73.95% | 48.9% |
| `ProductCategory`, first 3 levels | 715 | 110.8 | 82.66% | 65.0% |
| `ProductCategory`, first 4 levels | 1,198 | 66.1 | 90.92% | 66.9% |
| **`ProductCategory`, full path** | **1,342** | **59.0** | **93.31%** | 66.7% |
| `Retailer` + full path | 1,344 | 58.9 | 93.30% | 66.7% |
| `ProductBrand` | 14,576 | 5.4 | 85.64% | 80.9% |
| `ProductUrl` directory path | 10,448 | 7.6 | 68.34% | 99.7% |

Accuracy is measured on the rows that have a value for that column. Coverage is
99.4% or better for every row in the table, so it makes no material difference.

Three things to read off it.

- **A 1,342-row lookup table gets 93.31% of Segment right with no model.**
- **`Retailer` carries no signal at all** — exactly the majority baseline, and
  adding it to the path produces 2 extra keys out of 1,342, because the retailer
  taxonomies are already disjoint.
- **`ProductUrl` looks great and is not.** 99.7% of its keys map to a single
  Segment, but there are 10,448 keys over 79,205 rows — 8 rows per key. High
  purity with few rows per key is memorization, and it scores 68.34%, barely
  above baseline. This is the trap when someone proposes "just use the URL".

### 3.2 Where the gain lands

Per-class F1 on validation, agreed four fields against the same plus the path:

| Segment | Support | Agreed 4 | + Category | Gain |
|---|---|---|---|---|
| **Allergy** | 408 | 0.713 | 0.909 | **+19.6pp** |
| **CCFS** | 638 | 0.767 | 0.919 | **+15.2pp** |
| **Internal Analgesics** | 720 | 0.726 | 0.870 | **+14.5pp** |
| **Digestive Health** | 1,061 | 0.748 | 0.880 | **+13.2pp** |
| External Analgesics | 855 | 0.835 | 0.939 | +10.5pp |
| Other Self Care | 170 | 0.849 | 0.892 | +4.3pp |
| Vitamins, Minerals & Supplements | 7,346 | 0.938 | 0.974 | +3.6pp |
| Lifestyle CHC | 683 | 0.892 | 0.918 | +2.7pp |

The gain is concentrated in the minority classes, which is where the business
value is — the majority class was already at 0.938.

### 3.3 The cross-retailer risk, and what it actually is

**Of the 1,006 `ProductCategory` paths that appear in Walmart, Walgreens,
Target, CVS and Kroger, zero appear in Amazon.** Amazon uses 337 paths. The six
taxonomies do not overlap at all. Amazon is 80% of the data.

Shopify reports the same problem: merchant-supplied product type is unreliable
because "multiple merchants selling the same product can use different values
for Product Type" (Shopify Engineering). We measured the extreme version.

Train on Amazon only, score the other five retailers:

| Input | Accuracy | Macro F1 |
|---|---|---|
| `ProductName` only | 81.65% | 73.04% |
| The agreed four fields | 82.67% | 74.23% |
| `ProductName` + `ProductCategory` | 82.79% | 74.40% |
| The agreed four fields + `ProductCategory` | **84.66%** | **76.44%** |

**The path still helps on unseen retailers, by +2.21pp instead of +10.45pp.** It
does not hurt. That is the corrected result — an earlier exploratory run of mine
said it hurt, and §3.4 explains why.

Everything degrades cross-retailer, which is the real finding: the best
cross-retailer macro F1 is 76.44 against 91.29 in-distribution. **A ~15pp drop
is what moving to a new retailer costs**, whatever features we use.

### 3.4 How the path is represented decides whether it is safe

The path can enter the model two ways: concatenated into the one text field, the
way `build_product_texts` already does it, or as its own TF-IDF block with its
own feature space.

| Evaluation | Name only | Concatenated | Separate block |
|---|---|---|---|
| In-distribution (official splits) | 78.73 | 91.06 | **91.39** |
| Cross-retailer (Amazon → other five) | 73.04 | **74.40** | **67.18** |

In-distribution the two are equivalent. Cross-retailer they are not: a separate
block scores **5.86pp below using no category at all**. Given its own feature
space, Amazon's taxonomy vocabulary dominates and misleads. Concatenated, the
path's tokens are diluted among the name's and the model keeps leaning on the
name.

**So the recommendation is specific: include `ProductCategory`, concatenated
into the existing text stream, never as a separate feature block and never
one-hot encoded.** Which is convenient, because concatenating is what the
team's text builder already does.

Supporting evidence that the path works as *language* and not as a key: split by
category path so that **100% of scored rows have a path never seen in training**:

| Input | Accuracy | Macro F1 |
|---|---|---|
| `ProductName` only | 86.59% | 70.42% |
| The agreed four fields | 87.67% | 71.90% |
| `ProductName` + `ProductCategory` | 93.99% | 84.13% |
| The agreed four fields + `ProductCategory` | **94.82%** | **85.42%** |

Every path is new and the path is still worth +13.52pp macro F1, because
"Vitamins", "Allergy" and "Digestive" generalize even when the full path does
not. Accuracy barely moves; macro F1 is what suffers, so the minority classes
are the ones that were relying on having seen the exact path.

---

## 4. What the other three allowlisted fields contribute

`ProductName` alone: 78.73 macro F1. The agreed four fields: 80.84. So
`ProductBrand` + `ProductDescription` + `ProductContents` together are worth
**+2.11pp**.

- **`ProductBrand` is largely redundant.** The brand is usually already inside
  the name — "Cora Period Menstrual Cramp Relief Kit" carries "Cora". On a
  random split, `ProductName` + `ProductCategory` scores 91.89 macro F1 and
  adding all three remaining allowlisted fields on top gets 92.71 — **+0.82pp
  for brand, description and contents combined**. Isolating brand on its own was
  not measured in the notebook, so this is as far as the evidence goes.
- **`ProductDescription` is present on 30.23% of rows**, and when present it
  starts at median character 142 of the built text, so the model does see it.
- **`ProductContents` is present on 38.41% of rows and the model mostly never
  sees it.** It starts at median character **598**.

That last point needs the token budget spelled out. `sequence_length` is **128
tokens**, roughly 400–550 characters of product text. The built input is
long-tailed:

| | characters |
|---|---|
| median | 230 |
| 75th percentile | 1,232 |
| 90th percentile | 2,200 |
| maximum | 6,288 |

**38.1% of rows exceed 400 characters**, so about a third of the input is cut
off. Meanwhile `field_character_limits` is `(1024, 512, 4000, 2000)` — 7,536
characters of allowance feeding a window that holds about 500.

| Field | Present | Median start character |
|---|---|---|
| `ProductName` | 100.00% | 0 |
| `ProductBrand` | 99.95% | 128 |
| `ProductDescription` | 30.23% | 142 |
| **`ProductContents`** | **38.41%** | **598** |

`ProductContents` is allowlisted, and on most of the rows that have it, it never
reaches the model. **We currently cannot tell whether contents is a weak field
or just a truncated one.** That is experiment E2.

---

## 5. What carries no signal, so we can stop considering it

Official splits, scored on validation:

| Input | Accuracy | Macro F1 |
|---|---|---|
| Numerics (`ProductRating`, `ProductReviewsCount`, `XRatXRev`, `ReviewsCount`) + name length | 54.85% | **11.89%** |
| `Retailer` only | 57.17% | **11.88%** |
| Majority class | 61.83% | — |

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
| **Expected payoff** | **+10.45pp macro F1** in-distribution, **+2.21pp** cross-retailer. Biggest gains on Allergy (+19.6pp), CCFS (+15.2pp), Internal Analgesics (+14.5pp) |
| **Cost** | One tuple entry plus a character limit. One retrain |
| **Risk** | Low, given §3.4 — concatenated, it helps in both evaluations. As a separate block it is harmful cross-retailer |
| **How we know** | Validation macro F1 and per-class F1 against the current four-field run |

This is the whole document in one line. Everything else is a rounding error next
to it.

### E2 — Fix the 128-token truncation

| | |
|---|---|
| **Change** | Per-field token budgets instead of truncating the concatenation, or raise `sequence_length` to 256 |
| **Expected payoff** | Not measurable from here, but bounded: it can only affect the 38.1% of rows that overflow, and the field it recovers is present on 38.41% |
| **Cost** | Config change. 256 tokens roughly doubles transformer training time |
| **How we know** | Validation macro F1 **split by whether the row overflows 128 tokens**. If the gain is only on overflow rows, it is real. If it is spread evenly, something else changed |

Do E2 before concluding anything about whether description and contents are
useful. Right now we cannot distinguish a weak field from a truncated one.

Note for E1: adding `ProductCategory` makes the truncation worse, since the path
is another 60–120 characters. Put the path **early** in the field order, right
after the name — it is short and it is the second strongest signal.

### E3 — Category path as explicit hierarchy levels

| | |
|---|---|
| **Change** | Emit levels 1–4 as separate tagged segments rather than one path string |
| **Rationale** | Lookup accuracy climbs monotonically with depth: 62.23 → 73.95 → 82.66 → 90.92 → 93.31. Shallow levels generalize across retailers; deep ones do not. Making the structure explicit lets the model weight them separately |
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
| **Rationale** | `Other Self Care` has 1,132 rows against 48,982 for the majority. Shopify's mitigation on a highly imbalanced taxonomy was class weights rather than resampling (Shopify Engineering) |
| **Expected payoff** | Already inside every number here — all of them use `class_weight="balanced"` |
| **How we know** | Report macro F1 and per-class F1, never accuracy alone |

### Not recommended

| Idea | Why not |
|---|---|
| Review and rating features | 11.89 macro F1 on their own, §5 |
| `Retailer` as a feature | 11.88 macro F1, and adds 2 keys over the path alone, §3.1 |
| `ProductUrl` tokens | 68.34% as a lookup across 10,448 keys. Memorization, §3.1 |
| `ProductBrand` as its own field | Largely redundant given `ProductName`, §4 |
| `ProductImageUrl` / image features | Shopify gets value from images, at 250M parameters and hundreds of millions of rows. We have 79,205 rows and 8 classes. Out of scope for this milestone |

---

## 7. Feedback on the splitting policy

Sebastian asked for this, and two things came out of using the splits.

### The retailer mix is skewed between train and validation

| Retailer | Train | Validation | Shift |
|---|---|---|---|
| **Amazon** | 77.1% | **87.4%** | **+10.3pp** |
| Walmart | 12.0% | 6.3% | −5.7pp |
| Walgreens | 3.8% | 2.2% | −1.6pp |
| Target | 3.4% | 2.2% | −1.2pp |
| CVS | 3.0% | 1.6% | −1.4pp |
| Kroger | 0.7% | 0.3% | −0.4pp |

Groups are correlated with retailer — products that group together come from the
same retailer — so group-aware allocation concentrated Amazon into validation.
Every validation number is therefore more of an Amazon number than the training
distribution is. Worth stratifying the allocation by retailer as well as
grouping, if that can be done without breaking group isolation.

**Group isolation itself is clean:** 0.0% of validation `group_id`s appear in
train.

### A correction to my own earlier claim about grouping

§2.7 of `model-evaluation-strategy.md` said the absence of a grouping key
"blocks the validity of every reported number", and recommended constructing one
from normalized `ProductName` plus `ProductBrand`.

I built that key and measured it. It produces **73,649 groups over 79,205 rows —
1.08 rows per group**, with only 6.0% of groups holding more than one row and
12.6% of rows in a multi-row group. Random split against grouped split:

| Input | Random | Grouped | Drop |
|---|---|---|---|
| The agreed four fields | 85.27 | 85.34 | −0.07pp |
| `ProductName` only | 83.38 | 83.53 | −0.14pp |
| `ProductName` + `ProductCategory` | 91.89 | 91.69 | +0.20pp |
| The agreed four fields + `ProductCategory` | 92.71 | 92.44 | +0.27pp |

**That is noise.** Either near-duplicates are rarer in this catalogue than I
assumed, or my key was too strict to find them. The team's rule-v3 grouping is
more sophisticated than mine and may well be catching what mine missed — but the
measured effect of grouping on Segment scores is the number that justifies the
effort, and it is worth running once.

### Where the real generalization risk is

Not near-duplicate rows. **Unseen taxonomies.**

| | |
|---|---|
| Validation rows whose category path is unseen in train | **0.6%** |
| Non-Amazon rows whose category path is unseen in Amazon | **100.0%** |

The split that would stress this model is a **retailer holdout**, and it is not
one of the three we have. §3.3 is the closest thing to it and it shows a ~15pp
macro F1 drop. That number deserves to be on the record next to the 91.29.

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
  Consistent with our 91.29 macro F1 from TF-IDF, and a reason to take Max's
  baseline seriously rather than assume the transformer wins.
- **Merchant-supplied category is unreliable across merchants.** Shopify
  Engineering says so directly. We measured zero path overlap between
  Amazon and the other five retailers.
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
- **The final test was never touched.** All scores are validation.
- **No image features**, although `ProductImageUrl` is 99.41% populated.
- **Only Segment.** The other five targets may rank features differently, and
  §3.3 of the evaluation strategy already says Platform is not evaluable as-is.
- **The cross-retailer test is one split**, Amazon against the rest. It is a
  strong signal, not a tuned estimate — and since validation is itself 87.4%
  Amazon, the in-distribution numbers lean the same way.
- **One earlier result of mine was wrong and is corrected here.** An exploratory
  run gave `ProductCategory` a *negative* cross-retailer effect. That run gave
  the path its own TF-IDF block; the team's text builder concatenates. §3.4 has
  both, and the representation turned out to be the whole difference.
- **These numbers are not reproducible digit for digit across machines**, only to
  about ±0.4pp. §1 explains why. Anyone re-running the notebook should check that
  the gaps between rows match, not the individual figures. For the record, the
  reference run used pandas 2.3.3, scikit-learn 1.7.2 and Linux OpenBLAS; the
  notebook prints its own versions in §0 so a mismatch is visible immediately.

---

## 10. Open questions

1. **Does `ProductCategory` join the allowlist?** Worth +10.45pp macro F1
   in-distribution and +2.21pp cross-retailer, concatenated. (§3, E1)
2. **Is `sequence_length = 128` deliberate?** It cuts `ProductContents` off on
   most of the rows that have it, and `field_character_limits` allows 7,536
   characters into a window holding about 500. (§4, E2)
3. **Will the model score products from these six retailers, or new ones?** It
   does not change E1 any more, but it decides whether 91.29 or 76.44 is the
   number we quote to the PO. (§3.3)
4. **Should the split allocation stratify by retailer?** Validation is 10.3pp
   more Amazon than train. (§7)
5. **Has anyone measured what grouping buys on Segment scores?** My own key says
   the gap is noise, which contradicts what I wrote in §2.7 of the evaluation
   strategy. (§7)
6. **Should there be a retailer-holdout split**, given that it is the only one
   that stresses the actual generalization risk? (§7)

---

## Sources

- Gupta, V., Karnick, H., Bansal, A., & Jhala, P. (2016). [Product Classification in E-Commerce using Distributional Semantics](https://aclanthology.org/C16-1052/). *Proceedings of COLING 2016*.
- Shopify Engineering. [Using Rich Image and Text Data to Categorize Products at Scale](https://shopify.engineering/using-rich-image-text-data-categorize-products).
- Shopify Engineering. [Categorizing Products at Scale](https://shopify.engineering/categorizing-products-at-scale).
- Goumy, S. [Ecommerce Product Title Classification](https://ceur-ws.org/Vol-2319/ecom18DC_paper_5.pdf), SIGIR eCom 2018.
