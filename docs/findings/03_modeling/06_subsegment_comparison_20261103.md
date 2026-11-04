# Sub-Segment classical model and MiniLM comparison — November 3, 2026

The classical model now trains on the same approved Sub-Segment allocation as
Sebastian's MiniLM. Its ProductCategory variant reaches 59.12% union macro-F1,
versus MiniLM's 65.73% on the exact same validation records. This supplies the
previously missing classical candidate; it does not select a production model.

## Population and inputs

- Allocation: `sub_segment_development_allocation_v1_20261029_r2`.
- 53,956 labelled training records; 11,562 labelled validation records in 6,609
  frozen product groups. All 49 training classes stay in the macro denominator;
  47 have validation support and there are no out-of-vocabulary truth rows.
- Validation ID SHA-256: `39a929e0232fe9c76e12240ffe47f78f26c7b9d67ac909e310cf140a4c8491a8`.
- Approved allocation completion SHA-256:
  `26f52bd5360af96d0271c998f13c846dfeaa325bf0fb90b4c85011df5cabdc6a`.
- Hyperparameters are the existing TF-IDF/logistic-regression defaults, with no
  tuning and `class_weight=null`. Vocabulary and weights fit on train only.
  ProductCategory is a separate experimental arm, concatenated into the same text.

The category arm and MiniLM use the same five original product fields, but their
text representations differ. MiniLM puts category first, limits it to 512
characters and uses 256 tokens overall. TF-IDF appends category without those
token/category limits. The comparison concerns complete pipelines, not the
isolated effect of architecture.

## Results

| Run | Accuracy % | Union macro-F1 % | Observed-label macro-F1 % |
|---|---:|---:|---:|
| Training-majority lookup | 25.4800 | 0.8288 | — |
| TF-IDF, four fields | 71.5015 | 51.4644 | 53.6544 |
| TF-IDF + ProductCategory | 81.9235 | 59.1184 | 61.6341 |
| MiniLM, five fields | 88.5833 | 65.7268 | 68.5237 |

The majority class, Supplements, is selected from training labels. Observed-label
F1 is shown only as a secondary statistic; the headline always uses model
vocabulary union validation truth with `zero_division=0`.

| Reference to candidate | Macro-F1 difference, pp | Paired group bootstrap 95% interval, pp | Errors fixed / introduced |
|---|---:|---:|---:|
| TF-IDF base to category | +7.6540 | [+5.9783, +9.4529] | 1,532 / 327 |
| TF-IDF base to MiniLM | +14.2624 | [+12.4775, +16.9570] | 2,338 / 363 |
| TF-IDF category to MiniLM | +6.6084 | [+4.8054, +9.1813] | 1,164 / 394 |

These are descriptive paired percentile intervals from 1,000 draws of the 6,609
validation groups with replacement, seed 42. Each draw uses the same sampled
groups for both models and retains the fixed 49-class denominator, including
zero-support classes. They are not final-test estimates.

ProductCategory helps this target, as it helped Segment. That does not justify
enabling it by default for Mnfr or TargetAgeGroup, where the earlier experiments
showed worse results. MiniLM is the stronger Sub-Segment candidate on these
validation metrics; deployment still needs the team's model decision, measured
single-request latency and the PO's review-load target.

## Portable source reconciliation

The October 31 blocker compared generated completion/grouping-freeze hashes
between machines. The team's November 1 contract section 3.7 clarifies that
timestamp-bearing sidecars are local seals, not cross-machine identity.

The revised adapter keeps the authoritative source verifier and committed
protocol bundle unchanged. It checks stable grouping identity against that
bundle and requires exact sender byte hashes for all four source CSVs: train,
validation, assignments and protected-test identities. Approved handoff and
allocation pins remain exact. It also checks full membership, original labels,
label eligibility, class vocabulary, target-specific hashes and group isolation.
Sender and local sidecar hashes are recorded separately in each run's manifest.
No source, allocation, historical seal or split membership was edited.

Regression checks accept an independently timestamped, correctly sealed local
source, while rejecting a changed feature CSV even after resealing and rejecting
a different protocol identity. They also forbid test/candidate access.

## Verification and integration evidence

- Full suite: 473 discovered, 468 passed, 5 optional PyTorch skips; 12 focused
  classical tests passed. Two independent reviews checked the adapter and outputs.
- Training ran with network and `test.csv`/`training_candidate.csv` access blocked.
- Every validation ID, true label, group, class order, probability sum, argmax,
  confidence, manifest checksum and protected record/group exclusion was checked.
- Independent sklearn recomputation reproduced all metrics. The reviewed loader
  and macro-F1 functions from Maria's notebook at
  `a41608e83dd3f1420ce5cb63a28b022254f879d4` accepted all three runs and agreed.
  Notebook setup, fitting, calibration and review-queue sections were not run.
- Both saved classical models reproduced probabilities on 300 validation records
  after reload, with all target columns absent from inference input (tolerance
  1e-12). Evaluation requires no model unpickling; integration models are optional.
- Classical training wall times: 168.22 seconds (base), 136.67 seconds (category).
  These are CPU fit durations on a shared workstation. MiniLM used a GPU and a
  trainer including epoch validation. Batch timing scopes remain in manifests;
  neither a controlled speed ratio nor single-request latency is claimed.

Private run: `data/processed/classical-subsegment-20261103/`.
Comparison: `data/processed/subsegment-comparison-20261103-portable/`.
The handoff includes both variants' predictions/manifests, saved models, ID-only
registries, the comparison report and its guarded reproduction script. The script
accepts explicit source, allocation, MiniLM ZIP, run and output paths and executes
only reviewed notebook function definitions. It refuses existing output paths.

Evidence fingerprints:

- MiniLM source ZIP: `15170a1b102117719416aebed9c6ebda43afd1efdfe86517f7af45269f450ea7`.
- Comparison JSON: `c78962ddf95c7fb0e052ce5ce2ab2accb146d0a62fa5c7b5c8c6bada9ecb94ff`.
- Reproduction script: `b0ece37e30d3314c0e71a105b973e9935b6adbfe3d5bddca3f4e71d84365a00f`.

Raw data, model binaries and prediction archives remain outside Git. Brand's
classical run is still pending; this delivery covers the requested Sub-Segment
comparison and does not claim all six targets are complete.
