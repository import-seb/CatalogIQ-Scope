# Finalize Model Selection: decision record

November 6, 2026. **Select MiniLM for Segment, Mnfr and TargetAgeGroup, and
lookup for Brand for reviewer-assisted integration. Keep Sub-Segment MiniLM
provisional and defer Platform. No target is approved for automatic acceptance
or production deployment.** These are recommendations for this ticket; team
release authority is unchanged. Max's implementation requirements are in the
[separate handoff](../../integration/model_selection_handoff_20261106.md).

## Evidence and comparison rules

Read the latest remote [main at eb1506c](https://github.com/import-seb/CatalogIQ-Scope/commit/eb1506cf2fe3e774bd0b3909522d15c3e45f1398),
[merged PR #23](https://github.com/import-seb/CatalogIQ-Scope/pull/23)
(head `618010afaf5a6d9a4ee7b9a23fef7c8c996bb22c`, merge
`707ef4b4dd3e5df93c6cc63fd006192ce1c1cb3f`), Maria's current harness/contract,
and the existing MiniLM, calibration and policy findings. Local checkout:
`sprint4/demo` at `aee0d65`; its remote-tracking main was older. Remote main's
Sub-Segment finding and exporter were inspected directly; the exporter Git blob
`a1702885cfd02803f3c5d1b076660b64fd226d2c` matches this checkout. No branch
was merged, pushed or switched for this work. The repository now resolves to
`import-seb/CatalogIQ-Scope` despite the older local directory/remote name.

Maria's [current contract](https://github.com/import-seb/CatalogIQ-Scope/blob/maria/segment-evaluation-harness/docs/predictions-contract.md)
has document blob `0a155cee291f05cdb832c84a438330a343eb8915`; her
[notebook](https://github.com/import-seb/CatalogIQ-Scope/blob/maria/segment-evaluation-harness/notebooks/04-segment-evaluation.ipynb)
has blob `5670c9d55777f8f9b313b3f1e7835145fd5a6885`. These are document
identifiers, not commit SHAs. Existing owned exports were checked against loader
commit `05e52e2e0203dee6d58dfc2d0e72d513eed9ad95`; Max's Sub-Segment comparison
used loader/macro functions at `a41608e83dd3f1420ce5cb63a28b022254f879d4`.
The current notebook was inspected, not executed end to end.

Primary macro-F1 gives every class in **model vocabulary union validation truth**
equal weight, including zero-support classes, with `zero_division=0`. Accuracy
counts correct rows and is secondary. Percentages below are comparable **within
one target and the stated allocation**, never across different populations.
Missing target labels are excluded only for that target. Product groups remain
frozen; Brand and Sub-Segment change development roles, not protected test roles.
Different text formatting means comparisons concern complete pipelines, not an
isolated architecture effect.

## All six target decisions

Each score pair is **accuracy % / union macro-F1 %**. TF-IDF means the existing
word TF-IDF plus logistic regression; `+C` includes ProductCategory.

| Target; population | Available candidates and scores | Decision and reason |
|---|---|---|
| Segment; frozen V4, 11,952 rows, 8 classes | Majority 61.75 / 9.54; TF-IDF 89.57 / 83.90; TF-IDF+C 94.42 / 91.00; four-field 256-token MiniLM 91.2734 / 85.8788; five-field MiniLM **96.9461 / 94.4490** | **Selected: five-field MiniLM.** +3.45 percentage points of macro-F1 over the strongest classical arm on the same records. Category/retailer transfer and taxonomy limitations remain. |
| Mnfr; frozen V4 labeled subset, 11,812 rows, 2 classes | TF-IDF 99.19 / 87.16192; TF-IDF+C 99.05 / 84.37463; unweighted five-field MiniLM **99.9661 / 99.57295**; weighted MiniLM 99.9577 / 99.46728 | **Selected: unweighted MiniLM.** Strong minority J&J evidence and a large macro-F1 advantage over classical. The unweighted/weighted distinction is only one prediction, not established general superiority. |
| TargetAgeGroup; frozen V4 labeled subset, 11,624 rows, 3 classes | TF-IDF 97.93 / 80.77868; TF-IDF+C 97.73 / 77.34484; five-field MiniLM **99.7505 / 98.74256** | **Selected: MiniLM.** Substantial macro-F1 advantage; Children and Infant are not hidden by Adult accuracy. Infant has limited support. |
| Brand; approved Brand v1, 12,619 rows, 143 classes | MiniLM 96.6638 / 87.83435; deterministic lookup **95.6494 / 93.11910**; probabilistic lookup **95.6890 / 93.11130**; classical Brand not fitted | **Selected: lookup family; probabilistic v1 for integration.** Better macro-F1 than MiniLM, one input and saved JSON counts. Prefer its genuine probability output for the integration contract; retain the deterministic baseline and disclose its slightly higher macro-F1. |
| Sub-Segment; approved v1 r2, 11,562 rows, 49 classes | Majority 25.4800 / 0.8288; TF-IDF 71.5015 / 51.4644; TF-IDF+C 81.9235 / 59.1184; MiniLM **88.5833 / 65.7268** | **Provisional: MiniLM.** Best aggregate evidence, but severe rare-class misses, two unsupported validation classes and no approved review-load target prevent unconditional taxonomy coverage claims. Integrate only as a provisional suggestion. |
| Platform; original V4 labeled subset, 247 rows; 57 model classes, 59 union classes | TF-IDF 63.16 / **22.31286**; TF-IDF+C 60.73 / **21.17165**; no MiniLM. Separate target-allocation proposals have no model results here. | **Deferred.** Only 797 original labeled training rows, two OOV truth rows, sparse independent support and weak union macro-F1. No defensible production or integration winner. |

Evidence: [classical results](04_multitarget_baseline_20261029.md),
[corrected union metrics](05_classical_contract_20261031.md),
[Segment comparison](01_minilm_256_reproducibility_20261015.md),
[Mnfr selection](11_mnfr_minilm_selection_20261029.md),
[remaining MiniLM/Brand experiments](13_remaining_minilm_training_20261029.md),
[probabilistic lookup](15_brand_probabilistic_lookup.md), and
[paired Sub-Segment comparison](06_subsegment_comparison_20261103.md).
Training-majority baselines also exist in classical metrics/Maria's harness;
they are reference floors, not additional integration candidates.

Brand's probabilistic version loses **0.00780 percentage points** of macro-F1
against deterministic lookup while gaining 0.03962 points of accuracy. Seventeen
tied-key predictions change (11 fixes, 6 new errors). This does not demonstrate
statistical superiority; the integration choice trades that tiny observed
macro-F1 loss for a usable probability distribution. Lookup's advantage over
Brand MiniLM is about 5.28 macro-F1 points despite lower overall accuracy.

Sub-Segment's MiniLM margin over TF-IDF+C is 6.6084 macro-F1 points; the existing
paired 1,000-draw product-group bootstrap interval is [4.8054, 9.1813]. It uses
the same 6,609 groups and fixed 49-class denominator. This supports the preferred
candidate, not reliable performance for every label or release approval.

Platform's historical **37.61% / 34.70% observed-label macro-F1 values are not
the primary scores**. The union includes the two classes the classifier cannot
emit. Full-population log loss/Brier are null until an explicit OOV treatment
is supplied. No Platform calibration or queue result is established.

## Artifact verification and population identity

The [new read-only audit](model_selection_20261106/artifact_verification.json)
recomputed accuracy/union macro-F1 from six aligned prediction exports, both
Max Sub-Segment exports and the original saved deterministic Brand mapping.
All six owned runs passed the existing strict contract validator and their
48 per-run sealed files matched. Five checkpoint byte hashes also matched.
Max's predictions exactly match MiniLM's approved Sub-Segment IDs, truth and
groups; probability sums, argmax and confidence agree. No models were refitted.

| Target | Labeled validation ID SHA-256 |
|---|---|
| Segment | `47f60a59ac99f8ca3375926505cbb9dcffddb03446cfde024abbacd340e96dea` |
| Mnfr | `4e117e752d1693dcad3d0cda057e9ef9ef94f452f58e0628007fe5a31168b00c` |
| TargetAgeGroup | `ed225324b5490fc85506b86477698d2bf9389a7ec5f3912ca008193b422a6585` |
| Brand, all three methods | `87cd25d15e37be3487efafa06f26b5a13908513f760377cb8b233327c02a1ce5` |
| Sub-Segment, all three models | `39a929e0232fe9c76e12240ffe47f78f26c7b9d67ac909e310cf140a4c8491a8` |

Aligned runs live under `data/processed/predictions_contract_v2_20261031/`.
Their manifests pin checkpoints, tokenizer, class order and allocation. The ID
hash length-prefixes each UTF-8 ID with 8 big-endian bytes in CSV order; these
exports already use sorted IDs. Do not substitute another hash recipe/order.
Classical Segment/Mnfr/age/Platform figures were audited against committed
findings and PR #23, not locally recomputed from missing classical exports.
Their private source references are `classical-contract-20261031/` and
`max-classical-validation-contract-20261031.zip` (hash in the contract finding).
Platform's target-specific validation digest is unavailable locally; obtain its
manifest if reproducing those historical scores. No such scores were generated
for the newer Platform allocation.

## Confidence, calibration and review workload

Raw integration exports have `probabilities_calibrated=false`. Existing
[calibration findings](17_confidence_calibration_review_20261103.md) fitted one
temperature per model on one group-disjoint validation role, selected a policy
on another and audited it on a third. These are reused development records,
not untouched production data. Calibration does not improve classification F1.
On policy-audit rows, ECE in percentage points changed as follows:

| Candidate | ECE raw to calibrated | Experimental cutoff | Hypothetical review per 100 audit inputs | Accepted accuracy |
|---|---:|---|---:|---:|
| Segment MiniLM | 1.1905 to 0.4274 | 93% calibrated | 7.9 | 99.185% (31 errors / 3,803 accepted) |
| Mnfr MiniLM | 0.0199 to 0.0088 | 50% calibrated | 0 | 99.975% (1 / 4,076) |
| TargetAgeGroup MiniLM | 0.0888 to 0.2209 | 50% calibrated | 0 | 99.875% (5 / 4,010) |
| Brand MiniLM | 0.6737 to 0.5954 | 97% raw | 17.1 | 98.778% (44 / 3,600) |
| Probabilistic Brand lookup | 7.1492 to 0.5603 | 95.5% calibrated, seen keys only | 24.1 | 99.333% (22 / 3,298) |
| Sub-Segment MiniLM | 2.6452 to 2.6831 | 97.7% calibrated | 54.0 | 98.793% (22 / 1,822) |

Calibration improved several scores for Segment/Mnfr/lookup, but age ECE/NLL
worsened, Brand MiniLM NLL worsened, and Sub-Segment barely changed and worsened
slightly. Do not promote a temperature simply because one statistic improves.

The [later accuracy-standard reassessment](18_accuracy_standard_reassessment_20261103.md)
supersedes the initial custom maximum-family-size bound: Mnfr and age pass the
preferred revised 99% development check, conditionally (lower bounds 99.525%
and 99.347%; six-model adjustment 99.261% and 99.048%). All tested methods can
fail when errors concentrate in a few large families. Historical data reuse
still prevents prospective certification. Segment/Brand/Sub-Segment do not
pass the primary check at those cutoffs.

[Product-specific rules](19_product_specific_review_rules_20261103.md) reduce
Segment errors from 31 to 7 while raising review from 7.9% to 30.7%; lookup
errors fall from 22 to 3 while review rises from 24.1% to 75.9%. These exploratory
replays do not establish approved rules. **Every saved operational policy is
disabled, with a null operational threshold: review all suggestions.** Per-target
coverage is not full-product coverage. No classical model's review queue was
evaluated in the studies above.

## Label quality, taxonomy and transfer limits

- **Vitamins versus Digestive Health:** the existing development demonstration
  (`notebooks/catalogiq_review_visuals.py`, `taxonomy`) shows Jarrow Formulas
  Fem-Dophilus under a retailer Vitamins, Minerals & Supplements breadcrumb,
  while its curated Segment is Digestive Health. Retailer shopping taxonomy and
  curated CatalogIQ classification answer different questions. This disagreement
  is not sufficient evidence of a wrong label. Preserve both values; ask a
  taxonomy owner to adjudicate disputed cases before changing rules or labels.
- **ProductCategory risk:** it helps Segment/Sub-Segment but worsens classical
  Mnfr, age and Platform. The selected five-field MiniLM checkpoints nevertheless
  used it; no same-model category ablation establishes its benefit for Mnfr/age.
  Keep their saved representation rather than removing a feature at serving.
  Retailer-specific paths can change, be missing, or act as shortcuts. Confirm
  the breadcrumb is available before prediction and not derived from the answer
  being predicted. Category-first truncation can crowd out product text.
- The [retailer-transfer study](../../references/research/catalogiq_modeling_findings.md)
  compares mixed-retailer and Amazon-only models on the same 2,511 other-retailer
  records: accuracy 94.46% versus 87.10%, macro-F1 89.92% versus 77.96%.
  Do not compare 87.10% to the full-population 96.95%. Assortment, wording,
  training volume and category paths all changed; category memorization alone
  is not established as the cause. Earlier V3 full-development/random-split
  experiments have different populations and are not candidates in this table.
- Mnfr's J&J class has only 238 validation rows; the selected model finds 237,
  with 3 false positives. It is a binary J&J/All others classifier, not a general
  manufacturer identifier. Preserve the decision log's prohibition on inferring
  Mnfr by automatically rewriting Brand relationships.
- Age has 138 Infant validation rows. Sub-Segment has eight supported classes
  with zero recall; External Upper Respiratory recall is 2/23. Two training
  classes have no validation support. Stronger aggregate F1 cannot repair that.
- Brand lookup covers 82.8433% of validation keys; 2,165 records use fallback.
  `Other Brands` F1 is only 0.27287 for probabilistic lookup. Its fallback prior
  is not product-specific knowledge. Five Brand classes have no validation
  support, and many others have few independent groups. ProductBrand is
  explicitly authorized retailer-supplied raw-to-curated normalization input;
  this does not authorize feeding predicted curated Brand into that input.
- No single-parent Segment/Sub-Segment whitelist is approved. Rare/missing
  labels, catch-all semantics and genuine multi-parent relationships require
  review, not automatic taxonomy correction. Development reuse, one-seed
  experiments and incomplete future retailer coverage limit generalization.

## Computational evidence and the optional benchmark

Historical measurements, not controlled speed comparisons:

| Model | Recorded training seconds | Recorded amortized batch ms/record |
|---|---:|---:|
| Segment MiniLM | 1,356.146 | 1.46452 |
| Mnfr MiniLM, unweighted | 1,322.567 | 1.47868 |
| TargetAgeGroup MiniLM | 2,548.705 | 1.47430 |
| Brand MiniLM | 1,349.605 | 1.45810 |
| Sub-Segment MiniLM | 1,604.098 | 1.48192 |
| Probabilistic Brand lookup | 0.389965 | 0.02010 |
| Sub-Segment TF-IDF / TF-IDF+C | 168.222 / 136.673 | 0.15302 / 0.16727 |

MiniLM training uses an RTX 3060 Laptop GPU, initialization, three training and
validation epochs and bookkeeping; historical inference is batch 64, excluding
tokenization/model initialization. Age training was partly battery-limited.
Classical fitting includes TF-IDF and logistic regression on preformatted text,
on Max's 24-logical-CPU Intel Model 151 Windows machine with two numerical
threads; warm batch 256 inference includes formatting and output construction.
Lookup CPU frequency estimation/inference exclude source loading, scoring and
file writing. Earlier classical Segment/Mnfr/age/Platform fit times were about
19 / 8 / 9 / 1 seconds per arm, with 0.089–0.135 amortized ms/record; full values
and scopes remain in the October 29 finding. No hardware-normalized ratio follows.
MiniLM training peaks around 1.627–1.628 GiB allocated and 1.945–1.949 GiB
reserved GPU memory for the documented runs; these are not serving RSS limits.

The user supplied an extracted Max handoff at `C:\Users\sebas\Downloads\max handoff`.
All 19 package seals and the PR-documented comparison/script fingerprints match.
Both `.joblib` files are Sub-Segment, not Segment or other targets. Their expected
dependencies are Python 3.12.14, numpy 2.3.3, pandas 2.3.2, scipy 1.16.2,
scikit-learn 1.7.2, joblib 1.6.0 and threadpoolctl 3.7.0. They use the existing
`predict_classical` adapter. Only these explicitly identified, checksum-verified
teammate files were deserialized; packaged scripts were not executed.

The [standalone benchmark](../../../scripts/benchmark_model_selection.py) ran on
the same Windows 11 computer, Intel Core i5-10500H @ 2.50 GHz (12 logical CPUs),
CPU only, two numerical/PyTorch threads, FP32. First 64 sorted approved validation
records, same product inputs; three full-batch warmups; three passes over all 64
single-record requests (192 timings/model); five repeated batches of 64. Both
pipelines include their native preprocessing and full probability/output creation.
MiniLM uses dynamic padding and its original 256-token/character limits.

| Sub-Segment model | Single-record p50 ms | Single-record p95 ms | Batch records/sec | Separate load seconds |
|---|---:|---:|---:|---:|
| TF-IDF, four fields | 17.04 | 21.12 | 2,262.52 | 0.695 |
| TF-IDF + category | 15.28 | 20.66 | 1,593.60 | 0.668 |
| MiniLM, five fields | 16.26 | 48.23 | 24.92 | 2.606 |

The [aggregate result and raw timing samples](model_selection_20261106/subsegment_cpu_benchmark.json)
record exact versions, sample fingerprint and scope. Python 3.13.7, joblib 1.5.2
and threadpoolctl 3.6.0 differ from Max's producer environment; sklearn/numerical
versions match. Both classical models reproduce saved probabilities within 1e-12;
CPU MiniLM matches saved GPU probabilities within 1e-5 with identical labels.
All 42 consumed files were rehashed unchanged after benchmarking.

Classical batch throughput is substantially higher on this CPU sample; median
single-request times are close while MiniLM's tail is slower. Batch padding and
request overhead matter. This is a small deterministic local benchmark, not a
production SLA, a GPU speed claim, an all-target benchmark or cold-start p95.
Load observations use the current filesystem cache. Per-model serving memory
is null because reliable isolated RSS was not measured. No dependency was added.

Reproduce with the same trusted handoff and existing environment, a fresh output:

```powershell
$env:PYTHONPATH = 'src'
.venv-gpu/Scripts/python.exe scripts/benchmark_model_selection.py --handoff-dir 'C:\Users\sebas\Downloads\max handoff' --output data/processed/model_selection_benchmark_repeat.json
```

## Acceptance criteria and remaining release work

| Ticket criterion | Status | Evidence or remaining limitation |
|---|---|---|
| Audit latest main, PR #23, Maria and MiniLM | Satisfied | Pinned sources above; existing artifacts inspected, current harness scope stated. |
| Defensible recommendations for all six targets | Satisfied | Four selected, one provisional, one deferred; release approval separate. |
| Validate available scores and comparable populations | Satisfied for available artifacts; partial for historical classical targets | Nine recomputations; classical Segment/Mnfr/age/Platform rely on published findings because their exports are absent locally. |
| Review compute and inspect Max's two artifacts | Satisfied | Target/dependencies/provenance verified, existing adapter successfully used. |
| Optional controlled benchmark | Satisfied within stated scope | Sub-Segment CPU p50/p95 and batch rates; memory and other target/hardware comparisons unmeasured. |
| Decision/limitations document and Max handoff | Satisfied | This document, evidence JSONs and separate practical specification. |
| Production readiness and autonomous approval | Pending follow-up, not a ticket blocker | Fresh representative validation, taxonomy ownership, approved review policy, serving resource/SLA checks and release decision. |

Before production: confirm feature availability and category provenance by retailer;
resolve catch-all/taxonomy disputes with owners; qualify rare classes and new
retailers on fresh development data; freeze a complete reviewer/relationship policy
and product-level aggregation rules; approve thresholds and review capacity;
verify packaged artifacts/mappings in Max's serving environment and measure its
actual latency, concurrency and memory budget. Maria's full comparative notebook
and any eventual protected-test authorization remain separate team work. None is
silently claimed complete by selecting an integration candidate.

This closes the documentation/recommendation scope locally with explicit
follow-ups. No external ticket was edited. No training/tuning, split or model
changes, protected-test feature/label access, dashboard changes or publication
occurred. Evidence JSONs contain aggregate results/hashes, not product records.
