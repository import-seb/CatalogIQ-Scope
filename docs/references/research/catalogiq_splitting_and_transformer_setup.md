# CatalogIQ: Product Grouping and Transformer Setup

This document explains the implementations used in our splitting experiments
and the corrected full-development-pool Segment baseline completed on
**October 9, 2026**.

It accompanies the [split research paper summary](split_research_paper_summary.md).
The research summary describes external studies; this document describes what
we implemented in CatalogIQ.

## Current model comparison: October 10, 2026

The new ProductCategory comparison uses **256 tokens for both models**, the same
**55,776 training / 11,952 validation records**, seed 42, identical initial
weights, and three fixed epochs. It uses the reproduced frozen split protocol;
the October 9 experiments described below remain historical results.

| Model inputs | Validation accuracy | Macro F1 |
| --- | ---: | ---: |
| Name, brand, description, contents | 91.27% | 85.88% |
| Same fields with ProductCategory prefixed | 96.95% | 94.45% |

Adding ProductCategory gained **5.67 percentage points of accuracy** (95% paired
group bootstrap interval: **5.00–6.43**) and **8.57 points of macro F1**
(**7.36–9.86**). All eight Segment classes improved in overall F1. This is a
single-seed validation experiment; the intervals exclude training-seed and
production uncertainty. The prefix can still displace product text within the
shared 256-token budget.

Grouping and split membership remain frozen under their original 128-token
contract. Both experimental input views had zero identical-input crossings
between training and validation. The final test was never opened or scored.

Local checkpoints, predictions, comparison tables, and integrity metadata are in
`data/processed/segment_category_ablation_256_20261010/`. See the
[README command](../../../README.md#compare-productcategory-inputs) to reproduce
the comparison. The new implementation is `src/catalogiq/segment_category_ablation.py`.

### Retailer holdout: ProductCategory only

We trained a fresh ProductCategory model on all **43,969 Amazon training records**,
using 256 tokens, seed 42, three fixed epochs, and the original model's initialization.

| Validation cohort | Training sources | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Five other retailers: 2,511 records | Mixed retailers, original model | 94.46% | 89.92% |
| Same five retailers and records | Amazon only, new model | 87.10% | 77.96% |
| Amazon control: 9,441 records | Mixed retailers, original model | 97.61% | 95.78% |
| Same Amazon control and records | Amazon only, new model | 97.61% | 95.89% |

Held-retailer accuracy dropped **7.37 points** (95% paired group interval: **5.57-9.44**);
macro F1 dropped **11.96 points** (**8.59-16.10**). The compared validation IDs are identical.

All held-retailer breadcrumbs are unseen in Amazon training. This shows a transfer
gap, but training volume, assortment, and class mix also changed; it does not isolate
taxonomy dependence. One training seed was used, and the final test stayed closed.
Local outputs, including the new checkpoint and integrity seals, are in
`data/processed/segment_retailer_holdout_category_20261010/`; original checkpoints were preserved.

## 1. The overall approach

We separated the work into three stages:

```text
Product records -> Product groups -> Split assignments -> Model evaluation
```

| Stage | What it does |
| --- | --- |
| Rule-based grouping, version 3 | Finds relationships using identifiers and explicit product-family evidence. |
| TF-IDF grouping, version 2 | Finds relationships using weighted text similarity and compatibility checks. |
| Split allocation | Assigns each accepted group to one partition while balancing sizes and Segment classes. |
| Segment transformer | Learns to predict Segment and measures performance under different evaluation strategies. |

**Grouping never uses Segment or other target labels.** Segment enters only when
balancing split assignments and training the classifier. Matching uses the
supplied `ProductBrand` feature, rather than the `Brand` classification target.

Both grouping methods used the same **83,938 cleaned records**. Normalization
creates comparison views; it preserves the existing cleaning pipeline and source
datasets.

There were two separate comparisons:

- **Rule-based versus TF-IDF:** compare product groups and potential leakage.
- **Random versus group-aware splitting:** compare transformer validation results.

The latest transformer experiment uses the frozen **rule-v3 groups** for its
group-aware strategy. We have not trained a separate transformer using TF-IDF
groups.

## 2. Rule-based grouping: how it works

### Step 1: Normalize product evidence

The matcher standardizes capitalization, punctuation, brand suffixes, unit
spellings, and packaging terms. Its product-family view removes package counts
and supply quantities while retaining strength information.

It also filters misleading wording. For example, **"No Magnesium Or Rice
Fillers"** should not provide evidence that a product contains magnesium or rice.
Repeated advertising words are removed from the informative name evidence.

### Step 2: Find candidate pairs

Several searches propose records worth comparing:

- Shared checksum-valid UPC/GTIN, retailer-specific SKU, brand/model number, or
  listing URL.
- Identical normalized names or identical classifier text.
- Shared uncommon words and distinctive product-line prefixes.
- Character-level name similarity as a fallback.

**Blocking** means comparing records within plausible subsets instead of testing
every possible pair. Large blocks use multiple deterministic sorted
neighborhoods. These searches can still miss relationships.

The character fallback searches within normalized-brand blocks and deterministic
neighborhoods. It uses three-character shingles and a broad Jaccard cutoff of
**0.45**, retaining up to **12 neighbors per record** among the searched pairs.
This creates candidates; it does not confirm matches or search every global pair.

### Step 3: Confirm product-family relationships

Candidates need substantive evidence, such as:

- Identical classifier text across name, brand, description, and contents.
- A distinctive same-brand product-line name.
- Strong overlap between informative name words.
- Long, copied product-specific wording across declared brands.

**A shared identifier alone is insufficient.** Generic words such as "health,"
"supplement," and "capsule" cannot establish a relationship on their own.

Fuzzy family evidence primarily comes from product names and brands. Descriptions
and contents contribute through exact equality of the four-field classifier
text, rather than a general fuzzy description matcher.

The objective is **evaluation leakage prevention**, so different formulations
can remain together when strong product-line evidence suggests that separating
them would make evaluation artificially easy. Broad ingredient or mineral
wording alone does not justify that connection.

### Step 4: Validate proposed group merges

A good-looking pair can incorrectly connect two otherwise unrelated groups.
Before merging components, rule-v3 requires:

1. A family anchor shared across the proposed combined component.
2. Positive matching evidence between the selected representatives and the
   proposed endpoints from both components.

The frozen configuration retains up to **four representatives per component**
and requires **100% support across the tested representative pairings**.

This reduces weak chains such as `A -> B -> C`, where A and C have little in
common. It is a representative-based safeguard, not exhaustive verification of
every pair in a group.

### Step 5: Create the final groups

Accepted links form **connected components**: records connected by accepted
relationships receive one group ID. Each group stays within one split.

There is **no arbitrary maximum group size**. Large groups are audited for the
quality of their evidence.

## 3. TF-IDF grouping: how it works

### Step 1: Represent product text numerically

TF-IDF gives more weight to distinctive words and phrases and less weight to
terms appearing across many products. The frozen implementation uses individual
words and two-word phrases in three views:

| View | Purpose |
| --- | --- |
| Normalized product name | Identify strong title matches. |
| Packaging-normalized product name | Recover relationships across package quantities. |
| Description and contents | Support borderline title matches. |

Product-name features are uncapped. Supporting text uses up to **1,500 description
characters** and **1,000 contents characters** per record.

### Step 2: Compare pairs using cosine similarity

Cosine similarity measures how closely two TF-IDF vectors align.

| Evidence route | Frozen threshold |
| --- | --- |
| Strong product-name match | Name cosine **>= 0.86** |
| Packaging-normalized name match | Family-name cosine **>= 0.90** |
| Borderline name with supporting text | Name cosine **>= 0.78** and supporting-text cosine **>= 0.50** |

Description similarity cannot create a connection by itself. Shared identifiers
also do not create TF-IDF links.

The implementation computes cosine comparisons in memory-controlled chunks and
enumerates every qualifying title/family pair. It has no top-neighbor cap.

### Step 3: Apply compatibility checks

Candidates must pass brand and informative-name checks. Conflicting declared
brands require supporting evidence in the titles. Disjoint recognized ingredient
or formulation signatures can reject a connection.

A component-level formulation guard also prevents an unknown-formulation record
from bridging groups with conflicting known formulations.

### Step 4: Create groups

Accepted links form connected components, just as in the rule-based approach.
The TF-IDF method uses its formulation guard; it does not use rule-v3's broader
representative-based merge validation.

## 4. Split allocation and the independent audit

The grouping experiments originally generated separate **70% train / 15%
validation / 15% test** assignments using seed **42**. Whole groups are assigned
together, with Segment distributions preserved where practical.

Both methods receive the same independent leakage audit:

| Check | Method |
| --- | --- |
| Exact duplicates | Compare the four product-text fields. |
| Near duplicates | Jaccard similarity over sets of five-character product-name shingles. |
| Near-duplicate threshold | **0.85** |
| Eligible names | At least **12 normalized characters** |

The near-duplicate audit creates its own candidates. It does not use either
grouping method's matching graph. It flags lexical leakage candidates, rather
than proving product identity or semantic equivalence.

For the corrected transformer experiment, we reserved only the existing rule-v3
final test and allocated the full remaining labeled development pool:

| Population | Records |
| --- | ---: |
| Original cleaned input | 83,938 |
| Reserved final test, including unlabeled records | 12,591 |
| Remaining development records without Segment, excluded from supervised fitting | 4,023 |
| Full labeled development pool | **67,324** |
| Training records per strategy | **55,442** |
| Validation records per strategy | **11,882** |

With the final test already reserved, validation takes approximately **3/17
(17.65%)** of the labeled development pool. This retains the original 70/15/15
target across the labeled training, validation, and test populations.

The grouped development allocator balances **Segment, retailer, and group-size
distributions**. The random control selects individual records and exactly
matches the grouped strategy's Segment counts. Related groups can cross the
random train/validation boundary by design.

Grouping implementations remained frozen during this rerun. The development
allocation changed before training; the existing splitting-engine source
remained unchanged.

## 5. Transformer model setup

### Task and input

The model predicts one of **eight Segment classes** from these fields, in order:

```text
[ProductName] ...
[ProductBrand] ...
[ProductDescription] ...
[ProductContents] ...
```

HTML and missing-value placeholders are removed only from the model's text view.
Character caps are **1,024 / 512 / 4,000 / 2,000**, respectively, followed by a
**128-token limit for the combined input**. Padding adjusts to each batch.

Names come first, so later description or contents text may fall outside the
token budget. Categories, retailer, identifiers, URLs, provenance, and target
labels are excluded from the model input.

### Architecture

We load the pinned pretrained **`sentence-transformers/all-MiniLM-L6-v2`** encoder
and fine-tune all its weights together with a new classification head.

```text
Product text, up to 128 tokens
            |
            v
MiniLM encoder: 6 layers, 12 attention heads
            |
            v
Masked mean pooling: 384 values
            |
            v
Dropout: 10%
            |
            v
Linear classifier: 384 inputs -> 8 scores
            |
            v
Softmax -> Segment probabilities -> Highest-probability class
```

Masked mean pooling averages the token representations while excluding padding.
Approximately **22.7 million parameters** are trainable. The pinned model revision
is recorded in the experiment protocol.

### Training configuration

| Setting | Value used |
| --- | --- |
| Epochs | **3 fixed epochs** |
| Training batch size | **32** |
| Validation batch size | **64** |
| Optimizer | AdamW |
| Learning rate | **0.00003** |
| Weight decay | **0.01** |
| Loss | Unweighted cross-entropy |
| Learning-rate schedule | 10% warmup, followed by linear decay |
| Gradient clipping | **1.0** |
| Optimizer steps per full-pool model | **5,199** |
| Random seed | **42** |
| Hardware | NVIDIA RTX 3060 Laptop GPU |
| Precision | FP32 |

Both models start from the **exact same initial weights**, use identical class
counts and training settings, and differ in record membership. Deterministic
algorithms and seeded batch shuffling are enabled. Reproducibility is scoped to
the same hardware, software, inputs, ordering, and configuration.

### Evaluation policy

- Validation is observed after each epoch, but **epoch three is always saved**.
- There is no early stopping, best-checkpoint selection, or threshold search.
- **Macro-F1 is the primary metric**, giving each Segment class equal weight.
- Additional outputs include accuracy, per-class precision/recall/F1, confusion
  matrices, calibration, related-training-neighbor cohorts, and group-bootstrap
  intervals.
- The reserved final test remains **unscored**.

The experiment uses one training seed. Its bootstrap intervals describe
validation-sampling uncertainty, not variation across repeated training runs.

## 6. Where to find the code and saved settings

Paths below are relative to the repository root. Generated datasets and model
artifacts are local and ignored by Git.

| Item | Location |
| --- | --- |
| Rule-v3 matcher | [src/catalogiq/split_rules_v3.py](../../../src/catalogiq/split_rules_v3.py) |
| Rule candidate discovery | [src/catalogiq/split_rule_candidates_v3.py](../../../src/catalogiq/split_rule_candidates_v3.py) |
| Rule evidence normalization | [src/catalogiq/split_rule_evidence_v3.py](../../../src/catalogiq/split_rule_evidence_v3.py) |
| TF-IDF baseline | [src/catalogiq/split_tfidf_v2.py](../../../src/catalogiq/split_tfidf_v2.py) |
| Independent leakage audit | [src/catalogiq/split_evaluation.py](../../../src/catalogiq/split_evaluation.py) |
| Current development allocator | [src/catalogiq/segment_development_allocation.py](../../../src/catalogiq/segment_development_allocation.py) |
| Transformer implementation | [src/catalogiq/segment_transformer.py](../../../src/catalogiq/segment_transformer.py) |
| Full-pool experiment entry point | [scripts/compare_segment_full_development.py](../../../scripts/compare_segment_full_development.py) |
| Frozen rule settings | [rule_refinement_config.json](../../../data/processed/split_rule_v3_20261008/rule_refinement_config.json) |
| Frozen TF-IDF settings | [config.json](../../../data/processed/split_rule_v3_20261008/config.json) |
| Full-pool model and allocation settings | [protocol.json](../../../data/processed/segment_full_development_20261009/protocol.json) |
| Current validation results | [validation_comparison.csv](../../../data/processed/segment_full_development_results_20261009/validation_comparison.csv) |

Within `data/processed/segment_full_development_20261009/`, the `random/` and
`group_aware/` directories each contain the trained `checkpoint.pt`, tokenizer,
configuration, training history, validation predictions, and evaluation metrics.
