# Segment splits: finalized for model development

The confirmed problems are fixed. Grouping is frozen, the replacement final test
is reserved, and the integrity checks pass. Further work should use training and
validation. The final test has not been scored or manually reviewed in this run.

## What changed

- **Identical transformer inputs stay together.** The existing four-field text
  preparation and saved tokenizer are applied at the frozen 128-token limit.
  All 561 identical-input clusters, containing 682 pairs, become mandatory
  grouping constraints. One constraint joins previously separate rule groups.
- **Four confirmed incorrect connections are separated**, including alternative
  transitive paths: BulkSupplements Lemon Balm–Fish Oil, Ashwagandha–Red Yeast
  Rice, Cayenne–Senna, and motion-sickness glasses–acupressure wristbands.
  Supplier-domain residue and generic extract/root/herbal words no longer supply
  ingredient evidence. Device-family compatibility checks the whole proposed
  component, so an untyped intermediary cannot bypass it.
- **No further threshold tuning or group-size cap.** V3 and TF-IDF research
  baselines, cleaning, source records, and the shared allocation engines remain
  unchanged. The versioned V4 grouper and exact-input constraints are separate
  from the Segment allocation adapter.

The 75 retained regression pairs introduce **zero new clear errors**; two known
clear misses remain. All four confirmed unrelated witnesses are disconnected in
the full final graph. These selected examples are regression evidence, not
population-wide accuracy estimates.

## How the final test was reserved

The grouping configuration, code, source inputs, tokenizer files, input cache,
exposure registry, and allocation settings were hashed **before test selection**.
No test labels or outcomes selected membership.

The candidate pool is restricted to the original **unscored** test. The registry
conservatively inventories 2,096 previously presented/reviewed records and
67,324 records used by earlier transformer runs. Historical rule, TF-IDF, and
independent near-name components extend those exclusions to related families.

This leaves 12,198 eligible records. Corrected whole-group and exact-input closure
remove another 38. **All remaining eligible groups are reserved**: 12,160 records
in 11,979 groups. Previously trained records were not used to pad the test to 15%.
The remaining development pool uses the existing seeded allocator, targeting
the original 70:15 train/validation relationship and balancing Segment, retailer,
and group-size margins. Missing targets form a temporary allocation margin;
their original values are preserved and never imputed for model fitting.

| Split | All records | Proportion | With Segment labels |
|---|---:|---:|---:|
| Train | 59,111 | 70.42% | 55,776 |
| Validation | 12,667 | 15.09% | 11,952 |
| Final test | 12,160 | 14.49% | 11,477 |

## Integrity results

| Check | Result |
|---|---|
| Complete, unique coverage | All 83,938 original records |
| Product/effective-input groups crossing splits | 0 |
| Exact four-field product payload crossings | 0 |
| Identical effective transformer-input crossings | 0 of 682 duplicate pairs |
| Reviewed/model-seen/historically exposed families in final test | 0 recorded overlaps |
| Reproducibility | Full grouping identical after row reversal; seeded allocation repeated exactly |
| Graph integrity | Product edges plus exact-input edges reconstruct every saved component |
| Classes missing from validation or test | None of the eight Segment classes |
| Group sizes | 52,393 groups; 38,468 singletons; median 1; 99th percentile 9; maximum 90 |

The full automated suite passes **375 tests**, and the package wheel builds
successfully. Tests cover the confirmed connections, exact-input unions, target
exclusion, row-order stability, exposure-aware reservation, tampering, and a
development loader that cannot open the test CSV.

The independent name audit is unchanged: normalized character 5-gram Jaccard
at least 0.85, names at least 12 characters, with lossless candidate filtering.
Its **9,362 pairs and scores exactly match the previous audit universe**.

It finds **223 suspicious crossings** across all retained records: 221
train–validation and two train–test. Restricting both endpoints to records with
Segment labels leaves **179 train–validation crossings and zero final-test
crossings**. These are automatic lexical candidates, not 223 confirmed errors.
Four independently normalized brand/name duplicates also cross train–validation;
their complete raw payloads and effective model inputs differ.

An additional independent copied-description check finds **2,411 crossing
pairs** with identical normalized descriptions of at least 200 characters:
1,005 train–validation, 1,100 train–test, and 306 validation–test. Repeated supplier
copy can connect different products, so this is a residual-risk signal rather
than an instruction to merge every pair. No new test products were manually
reviewed to resolve it.

All eight classes remain represented:

| Segment | Train | Validation | Final test |
|---|---:|---:|---:|
| Allergy | 1,925 | 413 | 385 |
| CCFS | 3,009 | 645 | 598 |
| Digestive Health | 4,982 | 1,068 | 1,022 |
| External Analgesics | 4,009 | 859 | 828 |
| Internal Analgesics | 3,392 | 727 | 680 |
| Lifestyle CHC | 3,206 | 687 | 656 |
| Other Self Care | 805 | 173 | 154 |
| Vitamins, Minerals & Supplements | 34,448 | 7,380 | 7,154 |
| Missing target; excluded from supervision | 3,335 | 715 | 683 |

The maximum class-share difference from the whole retained pool is 0.08
percentage points in training, 0.09 in validation, and 0.48 in test. These shares
include missing-target records in each denominator. Eligibility, rather than
optimizing test class balance, determines the reserved test.

The finalization run takes about **329 seconds**, including grouping and
allocation repetitions, the independent name audit, exports, and hashing. This
excludes the separately frozen 84-second effective-input cache construction.

## Remaining risks and stopping point

Zero detected exact duplicates does not establish zero semantic leakage. The
name audit misses description-only relationships and short/missing names.
Copied marketing text, known grouping misses, and ambiguous broad umbrellas
remain. A connected component can still be overly broad despite graph integrity;
the previously reviewed Dr. Formulated, Full Spectrum, and Sambucol umbrellas
remain unresolved. The maximum group size of 90 is not itself proof of error.

“Clean” means absent from the recorded manual-review/model-use history under
this conservative policy. Earlier automatic whole-universe audits and
unsupervised feature construction cannot be undone; undocumented review cannot
be ruled out. Some historical review tables included Segment labels, so all
their presented records and related families are excluded conservatively.
Historical false-positive connections can also exclude legitimate test families.

The resulting test represents unexposed products within this catalog. It is not
validated as a future-retailer, chronological, or other production-shift sample.
The old 277-crossing figure used a different population and allocation, so its
difference from 223 is not an isolated estimate of the algorithm improvement.

**Stop grouping refinement here.** Develop the Segment model using training and
validation, record experiments, and keep the reserved final test closed until
the model and evaluation procedure are selected. The guarded preparation and
tokenization run passed without opening `test.csv`. No new transformer was
trained during this finalization task.

## Where everything is

- `src/catalogiq/split_rules_v4.py`: corrected product-only rule evidence.
- `src/catalogiq/split_rule_components.py`: reusable component confirmation.
- `src/catalogiq/model_input_groups.py`: frozen tokenizer contract, equality
  constraints, and cache validation.
- `src/catalogiq/split_finalization.py`: freeze, exposure-aware reservation,
  development allocation, exports, and seal verification.
- `src/catalogiq/segment_frozen_development.py`: protected training/validation
  loader and fresh-model entry point.
- `data/processed/split_finalization_20261009/frozen/`: full-record train,
  validation, and protected test CSVs; assignments, edges, distributions,
  regression results, `grouping_freeze.json`, `summary.json`, and `completion.json`.
- Sibling `exposure/` and `model_view/`: exposure registry and exact-input cache.
- Sibling `post_seal_integrity.json`, `post_seal_description_clusters.csv`, and
  `post_seal_audit.py`: independent automatic corroboration and reproduction.
- Sibling `development_preparation/`: successful guarded development-only
  preparation/tokenization manifests. Private generated artifacts are Git-ignored.

```bash
# Verify the existing snapshot; do not generate another final test.
python -m scripts.finalize_splits --verify data/processed/split_finalization_20261009/frozen

# Start a fresh model experiment; validation is the only evaluation partition.
python -m scripts.train_frozen_segment --train --output-dir data/processed/segment_model_run_01
```

The training command requires the optional training dependencies and cached
pretrained model. Changing text construction, tokenizer, or sequence limits
invalidates the frozen input contract; it requires a separately documented
exposure-aware protocol rather than silently reusing this seal.
