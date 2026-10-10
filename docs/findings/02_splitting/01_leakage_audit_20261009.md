# CatalogIQ split leakage audit

Audit date: 9 October 2026. Target: **Segment**.

## What this audit can establish

Group isolation works, but grouping misses some relationships. Identical raw
four-field product payloads stay together in the group-aware splits; independently
normalized names still cross boundaries. There are also questionable connections
within some groups. The actual transformer view has **one train–test pair with
identical token inputs**, despite different raw payloads.

The final test is **unscored in the saved model experiments**, but it is **not an
untouched source of product evidence**: earlier reviews included records that
later became test records. We cannot certify that grouping development was
independent of those records.

This audit changes no grouping rules, split assignments, model settings, or
thresholds. It runs no final-test model inference. Newly reviewed product examples
come from development records only.

## Records and splits covered

| Scope | Records | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Original frozen rule-v3 assignments | 83,938 | 58,756 | 12,591 | 12,591 |
| Original unchanged TF-IDF-v2 assignments | 83,938 | 58,756 | 12,591 | 12,591 |
| Current full-development model experiment, each strategy | 79,915 | 55,442 | 11,882 | 12,591 |

The original allocations meet the 70/15/15 target to rounding. The model experiment
uses all **67,324 labeled development records**, reallocates its train/validation
roles, and retains the original rule-v3 final test. Its audit scope additionally
includes all 12,591 protected test records, including 710 without Segment labels.
The 4,023 unlabeled development records were excluded from model training and from
this model-experiment audit scope. Original TF-IDF test assignments are a
comparison artifact; they are not the current protected final test.

Inputs are the preserved cleaned export and its identifier supplement. Records
are aligned by immutable `record_id`, not CSV row position. The audit checks input
hashes, coverage, role validity, and group isolation before interpreting counts.

## Independent duplicate checks

The same independent checks apply to both grouping methods:

- **Exact complete payload:** identical raw ProductBrand, ProductName,
  ProductDescription, and ProductContents. Labels and provenance are excluded.
- **Normalized brand/name:** HTML decoding, Unicode normalization, lowercase,
  punctuation and whitespace normalization; numbers and units remain.
- **Near names:** Jaccard similarity of character five-shingle sets, at least
  **0.85**, for normalized names at least **12 characters** long. Candidate
  retrieval is exhaustive within this scope, using lossless prefix/length filters;
  it does not use either matcher's blocks, top-k candidates, or accepted edges.
- **Copied descriptions:** identical independently normalized descriptions at
  least **200 characters** long, regardless of name agreement. This adds a
  description channel but also detects shared boilerplate. Those counts are
  suspicious-text reuse, not automatically related products.
- **Exact transformer inputs, current model experiment:** identical effective
  input IDs, attention masks and token-type IDs after the unchanged model text
  preprocessing and fixed **128-token** truncation. This uses the saved local
  tokenizer, with no weights, predictions or Segment labels.

Parameters were recorded before the fresh run. The fresh near-name pair universe
is compared with the saved audit universe. Pair counts and unique affected-record
counts are kept separate: a repeated listing can participate in several pairs.
The probes overlap and must not be added together.

The fresh run reproduced **all 9,362 saved near-name pairs and scores exactly**.
It covers 83,849 eligible records and 82,005 distinct eligible names; 89 records
have missing or short names. Runtime was **65.3 seconds**, including **31.0
seconds** for near-name discovery. Eight focused synthetic audit tests passed.

**Cross-split pair counts, over all three roles:**

| Allocation | Exact complete payload | Normalized brand/name | Near name | Copied description |
|---|---:|---:|---:|---:|
| Original rule-v3, 83,938 records | 0 | 32 | 291 | 2,689 |
| Original TF-IDF-v2, same records | 0 | 0 | 410 | 2,350 |
| Current group-aware, 79,915 records | 0 | 23 | 277 | 2,575 |
| Current random control, same records | 165 | 497 | 2,504 | 2,843 |

These are detected text matches, not a count of confirmed leakage relationships.
In particular, the copied-description column includes templates; its ordering
does not demonstrate that one method prevents leakage better.

**Near-name crossings by boundary:**

| Allocation | Train–validation | Train–test | Validation–test | Unique records in any crossing |
|---|---:|---:|---:|---:|
| Original rule-v3 | 151 | 96 | 44 | 462 |
| Original TF-IDF-v2 | 172 | 207 | 31 | 592 |
| Current group-aware | **143** | 111 | 23 | 437 |
| Current random control | **2,370** | 108 | 26 | 3,905 |

For the actual current experiment, **103 / 11,882 validation records (0.87%)**
have an audited near-name training neighbor under grouping, versus **1,627 /
11,882 (13.69%)** under random splitting. These independently recomputed counts
agree with the saved model diagnostic flags. All 165 random exact-payload
crossings are train–validation; none involve the protected test. Group-aware
normalized brand/name crossings comprise **2 train–validation** and **21
train–test** pairs.

The current test has the same rule-v3 protection in both model strategies. Random
splitting intentionally changes development train/validation roles; it is not a
randomly selected final-test control. Different train–test counts between these
strategies reflect which development role receives the neighboring record.

Group isolation passes for both original methods and current group-aware
allocation: **zero crossing assigned groups**. The intentional random control
has **5,255 groups crossing train/validation**, containing 22,468 records. That
check proves assignment isolation; it cannot detect products assigned different
group IDs incorrectly.

The original full input contains 6,235 identical-description pairs in 970
clusters; the current model audit scope has 6,012 pairs in 890 clusters. Of the
current clusters, **846 contain different normalized names**, a review flag for
possible templates or product-line variants. Descriptions do expose relationships
missed by the name check, but counting all reused copy as leakage would overstate
the findings.

### Actual transformer-input duplicates

Rebuilt development texts and token arrays match the saved hashes for **both
67,324-record training corpora exactly**. This ties the probe to the recorded
experiment rather than assuming that today's tokenization reproduces it.

The 79,915-record current audit scope has **630 pairs with identical token
inputs**: 570 raw-equal pairs and **60 pairs whose raw payloads differ**.

| Current allocation | Train–validation | Train–test | Validation–test |
|---|---:|---:|---:|
| Group-aware | **0** | **1** | 0 |
| Random control | **185** | **1** | 0 |

Thus **one protected test record has exactly the same effective input as one
training record** in both current strategies. The two records have different raw
payloads and belong to a two-record token-signature cluster at the 128-token cap.
Their test product names and labels were not manually inspected. This is a
verified input crossing, not a measured accuracy effect or an externally verified
identity judgment.

Twenty of the random train/validation crossings involve raw-unequal inputs;
the other 165 correspond to the raw-payload probe. In group-aware validation,
neither type crosses. Across the audit scope, 29,260 records reach the sequence
length cap; reaching it alone does not prove that distinguishing text was removed.
The input probe took **84.8 seconds** and the saved-corpus verification **48.9
seconds**, both offline, without inference or changes to model preprocessing.

## Reviewing missed relationships

The new review sample contains **24 suspicious current train/validation pairs**.
Selection mixes name similarity and copied-description evidence, excludes final
test records, and excludes previously reviewed exact pairs. Product evidence is
reviewed without Segment labels or per-pair method outcomes; judgments are saved
before joining those outcomes.

There are two judgments for each pair:

1. **Identity:** same product, different product, or uncertain.
2. **Evaluation:** keep together, safe to separate, or uncertain.

Packaging and formulation variants can be different products yet belong together
for evaluation when they share distinctive product-family evidence. Generic
category words or supplier-wide marketing alone are insufficient evidence.
Uncertain judgments stay uncertain.

This is AI-assisted qualitative review, with no external identity verification or
human adjudication. The reviewer knows that selection targets rule-v3 crossings;
it is not a fully algorithm-blind study. These selected cases cannot estimate
population-wide grouping accuracy, precision, recall, or leakage prevalence.

The sample has 12 name-route and 12 description-route pairs. Selection excludes
507 previously reviewed exact pairs; it does not establish independence from
previously reviewed product families. Description-route name similarities range
from **0.000 to 0.240**, so this review reaches beyond the high-name-similarity
audit.

Judgments: **12 keep together, 11 safe to separate, 1 uncertain**. Identity
judgments are 3 same, 17 different commercial variants/products, and 4 uncertain.
Explicit differences in pack count, strength, formulation, or kit contents count
as different commercial variants for this identity review.

Rule-v3 separates all 12 keep-together pairs; TF-IDF-v2 separates **6** and groups
the other 6. Both separate all 11 safe pairs. Every selected pair crosses the
**current** grouped train/validation boundary. Only two of the 12 related pairs
crossed the original v3 roles: their previous accidental co-allocation hid other
misses until validation was reallocated.

This sample was selected from rule-v3 crossings, so it is biased toward exposing
rule misses. It cannot rank overall methods or measure rule false positives:
same-group pairs were excluded by selection. Incorrect grouping is examined
separately below.

Names are shortened here; the CSV retains the full names, evidence and judgments.

| Reviewed example | Evaluation judgment and evidence | Rule-v3 | TF-IDF-v2 |
|---|---|---|---|
| Bayer Low Dose Aspirin 81 MG Enteric Coated Talblets ↔ Aspirin Regimen Bayer, 81mg Enteric Coated Tablets | Keep together: same CVS SKU 134019, description and contents; retitled listing | Missed | Missed |
| Icy Hot Advanced Pain Relief Cream, 2 oz ↔ same title at another retailer | Keep together: identical branded product/size; one description missing | Missed | Grouped |
| Rugby Chest Congestion Relief PE, 30 ↔ 60 Immediate Release Tablets | Keep together: same named 400 mg / 10 mg formulation; brand alias and pack variation | Missed | Grouped |
| One A Day Teen for Him ↔ Teen for Her Multivitamin Gummies, 60 Count | Keep together: different variants with an almost identical distinctive line/title | Missed | Grouped |
| Rising Mag64 Magnesium Chloride with Calcium Tablets, 60 Count (Pack of 5) ↔ same title | Keep together: identical title across retailers; Rising/Rising Mag64 metadata | Missed | Missed |
| Ear Wax Removal Tool, 1920P HD, 6 LED Lights (Gray) ↔ same feature title without Gray | Keep together: copied specific listing text; actual identity uncertain across brands | Missed | Missed |
| Safrel Senna 8.6 mg Tablets ↔ KOLLAGEN DUO Advanced Collagen Peptide Powder | Safe to separate: identical body is Amazon Compact by Design shipping copy | Separated | Separated |
| Solgar Potassium Magnesium Aspartate ↔ Solgar Ester-C Plus 500 mg | Safe to separate: shared body is disclaimer/glass-recycling copy; ingredients differ | Separated | Separated |
| CONZAY 8 Pcs Ear Pick Kit ↔ JOTIMEI 10 Pcs Ear Pick Kit | Uncertain: generic tool-list title; insufficient evidence of an OEM family | Separated | Separated |

Of the 12 copied-description cases, **one** is a reviewed relationship—the Bayer
listing—and **11** are generic reused copy. This describes this selected sample
only; it is not an estimate that 11/12 of all copied-description crossings are
safe.

The earlier 75-pair regression set and 60-pair post-freeze challenge set provide
additional context:

| Selected review set | Method | Missed relationships | Incorrect groupings |
|---|---|---:|---:|
| 75 regression pairs: 25 related + 25 safe, excluding borderline cases | Rule-v3 | 2 / 25 | 0 / 25 |
| Same regression cases | TF-IDF-v2 | 20 / 25 | 0 / 25 |
| 60 challenge pairs, primary reviewer: 35 related + 20 safe | Rule-v3 | 12 / 35 | 0 / 20 |
| Same primary-review cases | TF-IDF-v2 | 25 / 35 | 0 / 20 |

The regression cases informed development, so they are checks against known
failures, not independent evaluation. The 60 challenge pairs were reserved before
v3 and reviewed after its freeze. Primary results are shown because the secondary
reviewer had algorithm-design context. Their remaining five borderline cases are
excluded from this table. Neither table gives a population error rate. Zero
incorrect groupings in these selected pairs does not clear all resulting groups.

## Reviewing large and suspicious groups

The original partitions have the following size profiles:

| Group statistic | Rule-v3 | TF-IDF-v2 |
|---|---:|---:|
| Groups | 52,142 | 74,182 |
| Singletons | 38,143 | 67,247 |
| Mean size | 1.61 | 1.13 |
| 99th-percentile size | 9 | 3 |
| Largest size | 90 | 24 |
| Groups larger than 50 | 10 | 0 |

The qualitative audit selects the largest development groups and components with
heterogeneous names, representative-support flags, or transitive amplification.
It covers **13 rule groups containing 622 development records** and **four
TF-IDF control groups containing 62 development records**. Group witnesses and
paths were reviewed without Segment labels; no protected-test product content
was reviewed. One diverse witness per group is judged, so this does not certify
every member pair.

Rule witnesses include **six coherent families**, **four unrelated-family
connections**, and **three uncertain broad-line umbrellas**. The four TF-IDF
controls look coherent in the presented evidence. Selection differs between
methods; these counts cannot compare their population accuracy.

| Rule group / witness | Members | Direct / implied pairs | Review finding |
|---|---:|---:|---|
| BulkSupplements Lemon Balm → American Ginseng → Panax Ginseng → Fish Oil | 13 | 25 / 78 | Different ingredient families connected through supplier `.com`, generic words and a 1000 mg strength; safe to separate |
| BulkSupplements Ashwagandha → Valerian → Red Yeast Rice | 17 | 47 / 136 | Supplier-wide herbal/extract wording creates an unrelated-family connection |
| BulkSupplements Cayenne → Butcher's Broom → Bacopa → Hoodia → Senna | 10 | 18 / 45 | Different named ingredients/uses joined by generic extract language |
| Hion motion-sickness glasses → glasses variants → acupressure wristbands | 35 | 301 / 595 | Shared category/use phrase joins different device families |
| Garden of Life Dr. Formulated magnesium ↔ Advanced Omega Fish Oil | 90 | 4,005 / 4,005 | Uncertain broad umbrella; already fully connected, so transitivity is not the cause |
| Swanson Full Spectrum Citrus Bioflavonoid ↔ Milk Thistle | 61 | 1,830 / 1,830 | Uncertain broad umbrella spanning different ingredients |
| Children's Tylenol liquid ↔ Extra Strength coated tablets | 68 | 1,413 / 2,278 | Coherent branded acetaminophen family despite different doses/forms |

“Implied pairs” means all unordered member pairs in a component, not additional
accepted matching edges. The 13-member BulkSupplements component gains **53
transitive-only pairs**. Its saved three-hop path ends in an
`active_or_strength_line` match scored **0.25**; the first two edges are
`substantive_shared_line_head`. These scores are matcher-specific evidence
scores, not calibrated probabilities. Hion's path uses `distinctive_named_line`
edges, with its final edge scored **0.214**.

Potential overgrouping can also arise through broad direct evidence: the uncertain
Dr. Formulated and Full Spectrum umbrellas are complete graphs. By contrast, the reviewed TF-IDF Flare
Calmer group has **16 members, 28 direct edges and 120 implied pairs**, including
a five-hop path between adult/kids variants, yet its distinctive device-line
evidence remains coherent. Chain length or amplification alone cannot decide
correctness.

Across the 17 selected group witnesses, outcome joins identify **four incorrect
rule connections and one rule miss** (Quantum/Lip Clear brand aliases). TF-IDF
misses six related variants and separates the four unrelated rule witnesses.
The three uncertain umbrella witnesses are excluded from error counts. These
are example counts, not group error rates. The remaining coherent rule examples
include Aleve, Centrum Silver, Prevagen, DayQuil and Natrol Melatonin variants;
the Sambucol homeopathic/gummy umbrella remains uncertain.

Size alone does not establish a bad group. Shared branded lines can legitimately
span many formulations for leakage prevention. Conversely, a small component
can be wrong if unrelated products are linked by generic wording. Graph
connectivity proves that accepted edges reconstruct a group; it does not prove
that every implied member pair is meaningfully related.

## Frozen final-test integrity

The ID audit verified that the protected snapshot matches the rule-v3 test:
**12,591 records in 12,399 groups**. In all four saved transformer runs—old/current
experiments, random/group-aware strategies—final-test overlaps with training,
validation, and saved predictions are **zero**. Predictions contain exactly each
run's validation IDs. The recorded three completed epochs and fixed final-epoch
checkpoint policy match the model protocols; checkpoint internals were not decoded
for this audit.

That is evidence against direct training or model-score exposure in the saved
runs. It does not establish that no unrecorded experiment exists. ID disjointness
also does not establish feature novelty: the token-input audit above detects one
train–test crossing between different record IDs.

Earlier review exposure is measurable:

| Review scope, deduplicated across registries | Eventual final-test records reviewed |
|---|---:|
| Selected 121 diagnostic + 64 held-out + 75 balanced + 53 graph cases, before v3 freeze | 24 |
| Expanded pre-v3 scope, including all 210 annotated balanced candidates | 38 |
| Including the 60 post-v3 challenge cases reviewed before model protocols | **44** |

These counts are unions; individual registries overlap. All 44 touched current
test groups are singletons. The 38 pre-v3 records were available during grouping
development; the six additional records were reviewed after the v3 freeze. The
44 directly reviewed records are **0.35% of the final test**, a measured lower
bound on recorded product-review exposure, not a bound on possible decision bias.

Whole-dataset duplicate metrics and group diagnostics also existed during grouping
development. Routine feature-only group construction before splitting is distinct
from interactive review used to improve the matcher. The saved history does not
support the stronger claim that all grouping decisions were made without eventual
test evidence.

The v3 method froze at **02:21:19 UTC**, reserved judgments at **02:33:52 UTC**,
the first model protocol at **03:32:10 UTC**, and the full-development protocol at
**13:23:15 UTC**, all on 9 October. Every recorded `judgment_uses_Segment` flag
denies using Segment during review. Exposure establishes access to product
evidence; it does not prove that each example changed a rule or that its label
influenced development.

Integrity checks verify **86 distinct frozen data/artifact paths** and **11
pretrained snapshot files**. Saved inputs and results are intact. Current bytes
of four source files differ from old seals: three historical versions are
recoverable and later diffs cover pandas compatibility, empty-cohort handling,
and package-root lookup. The earlier frozen CLI bytes are not established by the
available historical commit; its current version also adds split export. These
differences prevent the old strict source-byte verifier from passing against the
current checkout; intact artifacts do not imply an identical historical source
checkout. Detailed hash checks and provenance are saved separately.

Conclusion: retain the current test as **unscored**, but disclose its prior
product-level exposure. A claim of fully untouched confirmatory evaluation needs
a separately reserved source of examples excluded from future grouping and model
development. This audit does not create or score such a test.

## Remaining risks and practical interpretation

- Near-name checks miss semantic relationships, heavily rewritten titles, and
  relationships expressed only in descriptions or identifiers. Copied-description
  checks miss paraphrases and shorter text.
- High text similarity can arise from different variants or generic templates.
  Automatic crossings are candidates; reviewed relationships are qualitative
  judgments, not measured effects on model accuracy.
- Missing or unreliable brands, names, descriptions, and identifiers limit all
  checks. Empty or short names fall outside the near-name scope.
- Selected pair and group reviews deliberately seek difficult cases. They do not
  establish population false-positive or false-negative rates.
- Broad grouping can make evaluation harder by withholding whole product lines.
  This audit does not determine whether that matches CatalogIQ's production mix.
- The existing full-pool validation accuracy gap is about **0.41 percentage
  points**. Different validation membership, one training seed, and residual
  matching errors prevent interpreting that gap as the causal effect of leakage.
- This audit starts at the frozen cleaned export. It does not establish that
  earlier cleaning decisions or unrecorded work were independent of eventual
  test records.

**Zero detected exact-payload crossings does not prove zero leakage.** The findings
support mechanical isolation and substantially fewer lexical crossings than the
random control, while documenting residual misses, questionable mergers, and a
limit on test independence. They do not establish either grouping method as the
default choice.

## Evidence and reproduction

The run directory is
[`data/processed/split_leakage_audit_20261009/`](../../../data/processed/split_leakage_audit_20261009/).
Detailed product evidence remains there with the private dataset artifacts.

| Location within the run | Contents |
|---|---|
| [`automatic/summary.json`](../../../data/processed/split_leakage_audit_20261009/automatic/summary.json) | Scope, parameters, independent audit coverage, counts, runtime, environment and input/source hashes |
| [`automatic/leakage_comparison.csv`](../../../data/processed/split_leakage_audit_20261009/automatic/leakage_comparison.csv) | Side-by-side pair counts and affected records by method and boundary |
| [`automatic/near_crossing_pairs.csv`](../../../data/processed/split_leakage_audit_20261009/automatic/near_crossing_pairs.csv) | All detected crossing name pairs, record IDs, scores and roles |
| [`automatic/description_duplicate_clusters.csv`](../../../data/processed/split_leakage_audit_20261009/automatic/description_duplicate_clusters.csv) | Exact copied-body counts without expanding every pair |
| [`model_inputs/summary.json`](../../../data/processed/split_leakage_audit_20261009/model_inputs/summary.json) | Exact effective-token duplicate counts, fixed preprocessing and scope |
| [`model_inputs/development_corpus_verification.json`](../../../data/processed/split_leakage_audit_20261009/model_inputs/development_corpus_verification.json) | Rebuilt development text/token hashes compared with both training manifests |
| [`model_inputs/raw_unequal_input_pairs.csv`](../../../data/processed/split_leakage_audit_20261009/model_inputs/raw_unequal_input_pairs.csv) | All 60 raw-unequal token collisions, including the test crossing, traced by IDs and differing field names without displaying test content |
| [`review/reviewed_pair_outcomes.csv`](../../../data/processed/split_leakage_audit_20261009/review/reviewed_pair_outcomes.csv) | Full names, separate identity/leakage judgments, rationale and both methods' grouping outcomes |
| [`review/review_freeze.json`](../../../data/processed/split_leakage_audit_20261009/review/review_freeze.json) | Judgment seal before outcome joins; review policy and evidence hashes |
| [`groups/reviewed_group_witnesses.csv`](../../../data/processed/split_leakage_audit_20261009/groups/reviewed_group_witnesses.csv) | Group judgments, names, sizes, rationale and exact saved edge paths |
| [`groups/reviewed_witness_paths.csv`](../../../data/processed/split_leakage_audit_20261009/groups/reviewed_witness_paths.csv) | Each path's actual edge endpoints, reasons and scores |
| [`integrity/integrity.json`](../../../data/processed/split_leakage_audit_20261009/integrity/integrity.json) | Test-ID overlaps, recorded experiment checks, chronology and integrity limitations |
| [`integrity/frozen_hash_verification.csv`](../../../data/processed/split_leakage_audit_20261009/integrity/frozen_hash_verification.csv) | Individual artifact/source/pretrained hash checks |

Recompute automatic counts into a **new** directory from the repository root.
In this PowerShell checkout, expose the source package explicitly:

```powershell
$env:PYTHONPATH = 'src'
python -m scripts.audit_split_leakage --output-dir data/processed/split_leakage_audit_repeat/automatic
python -m unittest discover -s tests -p test_split_leakage_audit.py -v
```

With an installed project environment, the `PYTHONPATH` line is unnecessary.

The additional offline tokenizer checks require the training dependencies in
`requirements-training.txt` and the saved local tokenizer:

```powershell
python -m scripts.audit_segment_model_inputs --output-dir data/processed/split_leakage_audit_repeat/model_inputs
python -m scripts.verify_segment_model_input_corpus --output data/processed/split_leakage_audit_repeat/model_inputs/development_corpus_verification.json
```

The default input/run paths intentionally identify these historical experiments;
`--input`, `--split-dir` and `--development-dir` can be specified explicitly.
Re-running automatic counts does not recreate qualitative judgments. The saved
group collection scripts and `review/score_review.py` reproduce evidence/outcome
joins; `integrity/audit_integrity.py` rechecks recorded IDs and hashes. Existing
research/source artifacts remain preserved.
