# TargetAgeGroup, Sub-Segment and Brand MiniLM experiments

Started October 29, 2026 after the user resolved the Brand feature contract.
ProductBrand remains an authorized retailer-supplied inference feature, and
Brand explicitly means raw-to-curated brand normalization. Direct overlap with
named-brand labels is therefore expected under this contract. The previous
preflight stop is resolved; its original audit remains unchanged for provenance.

The three targets run sequentially in this order: TargetAgeGroup, Sub-Segment,
Brand. Each uses one unweighted model and the fixed final third-epoch checkpoint.
No epoch selection, threshold tuning, class weighting or extra neural experiment
is performed. A predecessor's checkpoint, metadata and handoff ZIP must pass
verification before the next target can start. Failures stop the batch.

## Fixed configuration and data

Training uses the existing `.venv-gpu` environment, PyTorch `2.11.0+cu128`, CUDA
12.8 and NVIDIA RTX 3060 Laptop GPU. The original CPU environment is preserved.
Neural training delegates to the unchanged `run_target_minilm` orchestration and
frozen Segment trainer; no neural implementation was copied or edited.

All targets retain the pinned MiniLM revision, ProductCategory-first five-field
representation, 256-token budget, three epochs, training/evaluation batches
32/64, learning rate 3e-5, weight decay 0.01, warmup 0.10, dropout 0.10, seed 42,
FP32, four CPU threads and existing deterministic settings.

| Target | Allocation | Labeled train | Labeled validation | Classes | Planned optimizer steps |
|---|---|---:|---:|---:|---:|
| TargetAgeGroup | Frozen V4 | 54,282 | 11,624 | 3 | 5,091 |
| Sub-Segment | Approved Sub-Segment v1 r2 | 53,956 | 11,562 | 49 | 5,061 |
| Brand | Approved Brand v1 | 58,888 | 12,619 | 143 | 5,523 |

The saved label order and all development/product group assignments remain
authoritative. The 12,160 protected test records are excluded; only identity
metadata is read to check isolation, never protected labels or test.csv.

## Brand lookup baseline and scoring rules

The lookup uses ProductBrand only, fitted on the same labeled Brand training
records as MiniLM. Normalization is existing plain-text cleanup, Unicode NFKC,
case folding and surrounding-whitespace removal. Conflicting training values
map to the most frequent training class for that key; ties use the smallest
saved label ID. Missing feature keys are excluded from the mapping.

All labeled validation records, including `UnItemised brand`, `Other Brands`,
missing ProductBrand and unseen feature values, are scored. Missing/unseen keys
fall back to the most frequent labeled training class, with saved-label-order
tie breaking. This rule was fixed before evaluation and never uses validation
labels. Seen-key coverage and accuracy, missing/unseen fallback cohorts and
full-population metrics are recorded separately.

Primary macro-F1 uses the entire ordered training vocabulary, consistent with
the existing trainer. Brand has five training classes with zero validation
support; Sub-Segment has two. Their zero F1 contributions are retained for both
lookup and MiniLM. Macro-F1 over validation-supported labels is also reported
separately. Per-class support and complete confusion matrices accompany every
model; no labels are silently omitted from evaluation.

## Implementation and artifacts

`catalogiq.target_minilm_experiment` adds aggregate evaluation, the Brand lookup,
sequential verification and packaging around the existing shared entry point.
It implements no record-level prediction writer or inference endpoint. The
legacy trainer's existing internal validation artifact remains unchanged and
is used for verification; it is excluded from handoff ZIPs.

The versioned batch directory is:

`data/processed/minilm_remaining_targets_v1_20261029/`

It contains the predeclared `experiment_protocol.json`, exact source/dependency
snapshots in `reproduction_sources/`, separate model directories, run logs,
verified target result JSON files and separate Maria handoff ZIPs. Each ZIP
contains one checkpoint with target/representation/split/configuration metadata,
tokenizer, ordered vocabulary, aggregate metrics and checksums. Brand also
includes its lookup model, aggregate results and fallback rules. No external
delivery or merge is performed.

All 81 focused tests passed before training, including seven new checks for
training-only lookup fitting, deterministic ties, complete-population fallback
scoring, zero-support classes, exact validation membership, authorization and
handoff checkpoint/metadata verification. These tests do not train an encoder.

## Results

All three targets are complete and verified. TargetAgeGroup was retained without retraining. Sub-Segment completed before Brand started, using the corrected predecessor check. No new failure or serious integrity issue occurred during the resumed runs.

| MiniLM target | Accuracy | Macro-F1, full vocabulary | Macro-F1, supported labels | Fit time | Peak allocated / reserved GiB |
| --- | --- | --- | --- | --- | --- |
| TargetAgeGroup | 99.7505% | 0.98742561 | 0.98742561 | 42m 29s | 1.627 / 1.949 |
| Sub-Segment | 88.5833% | 0.65726828 | 0.68523715 | 26m 44s | 1.627 / 1.947 |
| Brand | 96.6638% | 0.87834353 | 0.91016757 | 22m 30s | 1.628 / 1.945 |

Fit time includes initialization, all three training/validation epochs and CPU state hashing, and excludes final artifact writing. Total wall time including preflight and saving was TargetAgeGroup: 42m 54s, Sub-Segment: 27m 30s, Brand: 23m 04s. These are measured wall times. Part of the original TargetAgeGroup run was battery-limited; AC was reconnected near its end. The resumed runs used AC power. No power settings or model configuration were changed, so these timings should not be interpreted as controlled hardware throughput comparisons.

Each model completed exactly three epochs: TargetAgeGroup 5,091 steps, Sub-Segment 5,061 steps, Brand 5,523 steps. These are the only requested unweighted MiniLM candidates; no extra experiments or threshold tuning were performed.

## Brand: full-population normalization comparison

ProductBrand is explicitly authorized at inference, supplied by the retailer, and used for raw-to-curated Brand normalization. Its direct overlap with named Brand labels is expected under this user-approved feature contract. The lookup was fitted on training records only; validation labels never determine its mapping, fallback or tie rules.

| Method; all 12,619 validation records | Accuracy | Macro-F1, all 143 labels | Macro-F1, 138 supported labels |
| --- | --- | --- | --- |
| MiniLM | 96.6638% | 0.87834353 | 0.91016757 |
| Lookup plus fallback | 95.6494% | 0.93119097 | 0.96492977 |

The validation record-ID hashes match exactly. Catch-all classes, missing ProductBrand values and unseen keys are included in both full-population scores. Lookup coverage is not used to filter the headline validation result.

| Lookup cohort | Records | Accuracy |
| --- | --- | --- |
| Seen normalized ProductBrand | 10454 | 97.5990% |
| Unseen nonmissing values | 2163 | 86.2228% |
| Missing values | 2 | 100.0000% |
| All fallback records | 2165 | 86.2356% |

Lookup coverage: 10,454/12,619 (82.8433%). All 2,165 uncovered records receive `UnItemised brand`, the global training-majority class (36,866 training records versus 2,259 for Other Brands). The fallback covers 2,163 unseen nonmissing keys and two missing feature values. It is never chosen using validation labels.

The lookup contains 11,354 normalized training keys, including 470 keys associated with multiple training labels. Its CPU fit took 0.474 seconds and full/cohort evaluation took 0.236 seconds. No GPU was used for fitting the lookup. The full lookup JSON also records cohort macro-F1, but subset macro-F1 over the full 143-label vocabulary has many zero-support terms and is not the headline comparison.

| Catch-all label | Method | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| UnItemised brand | MiniLM | 7820 | 0.975557 | 0.974808 | 0.975182 |
| UnItemised brand | Lookup | 7820 | 0.947932 | 0.984783 | 0.966006 |
| Other Brands | MiniLM | 484 | 0.748780 | 0.634298 | 0.686801 |
| Other Brands | Lookup | 484 | 0.583333 | 0.188017 | 0.284375 |

**Recommendation:** Brand is the target where a simpler baseline may be preferable when balanced class coverage or named-brand normalization is the priority. The lookup has higher macro-F1 (0.93119 versus 0.87834), with negligible fitting cost. MiniLM has higher overall accuracy by 1.0143 percentage points (128 fewer errors), and handles Other Brands substantially better: recall 63.43% versus 18.80%, F1 0.68680 versus 0.28438. These results support a business-dependent choice rather than a universal winner. The requested final MiniLM checkpoint remains packaged; the complete lookup model and results accompany it for Maria. No hybrid, fallback change or additional model was trained after observing these results.

## Per-class results and classification concerns

Every ordered class, including zero-support classes, has precision, recall, F1 and support in the aggregate CSVs below and in each handoff ZIP. The [consolidated per-class table](../../../data/processed/minilm_remaining_targets_v1_20261029/consolidated_per_class_metrics.csv) additionally records frozen training/validation group support. Complete confusion matrices use rows=true labels and columns=predicted labels in the saved vocabulary order.

- TargetAgeGroup: [per-class metrics](../../../data/processed/minilm_remaining_targets_v1_20261029/models/targetagegroup/minilm-targetagegroup-category256-48256ce8adfceee1/per_class_metrics.csv); [complete confusion matrix](../../../data/processed/minilm_remaining_targets_v1_20261029/models/targetagegroup/minilm-targetagegroup-category256-48256ce8adfceee1/confusion_matrix.csv).
- Sub-Segment: [per-class metrics](../../../data/processed/minilm_remaining_targets_v1_20261029/models/sub_segment/minilm-sub_segment-category256-8a1e8f3e742eb3a7/per_class_metrics.csv); [complete confusion matrix](../../../data/processed/minilm_remaining_targets_v1_20261029/models/sub_segment/minilm-sub_segment-category256-8a1e8f3e742eb3a7/confusion_matrix.csv).
- Brand: [per-class metrics](../../../data/processed/minilm_remaining_targets_v1_20261029/models/brand/minilm-brand-category256-aa0e7c04715d5006/per_class_metrics.csv); [complete confusion matrix](../../../data/processed/minilm_remaining_targets_v1_20261029/models/brand/minilm-brand-category256-aa0e7c04715d5006/confusion_matrix.csv).
- Brand lookup: [per-class metrics](../../../data/processed/minilm_remaining_targets_v1_20261029/models/brand/minilm-brand-category256-aa0e7c04715d5006/lookup_per_class_metrics.csv); [complete confusion matrix](../../../data/processed/minilm_remaining_targets_v1_20261029/models/brand/minilm-brand-category256-aa0e7c04715d5006/lookup_confusion_matrix.csv); [full/cohort metrics and fallback rules](../../../data/processed/minilm_remaining_targets_v1_20261029/models/brand/minilm-brand-category256-aa0e7c04715d5006/lookup_baseline.json).

### TargetAgeGroup

| Label | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Adult | 10914 | 0.99871713 | 0.99862562 | 0.99867137 |
| Children | 572 | 0.97565217 | 0.98076923 | 0.97820401 |
| Infant | 138 | 0.99264706 | 0.97826087 | 0.98540146 |

Confusion matrix, ordered `[Adult, Children, Infant]`:

```text
[[10899,  14,   1],
 [   11, 561,   0],
 [    3,   0, 135]]
```

No majority-class collapse is evident: Children F1 is 0.97820 and Infant F1 is 0.98540. Infant has only 138 validation records; this is a validation estimate, not protected-test certification.

### Sub-Segment

Accuracy 88.58% masks substantial class-level weaknesses. External Upper Respiratory has only 8.70% recall (2/23), despite precision 1.0. Other Allergy recall is 51.54% (130 records), and Temperature Packs recall is 50.77% (65 records). These labels need careful review; the model is a baseline, not evidence of uniformly strong taxonomy coverage.

Classes with zero validation support: `Creams & Gels (Rubs)`, `Daily oral contracception`. Their F1 is retained as zero in full-vocabulary macro-F1.

Classes with validation support but zero recall (record support in parentheses): Anti-Nausea (9), Dermatologicals (2), Devices/Refills (1), Other Self Care (15), Remaining Internal Pain (2), Stimulants (2), Topical Itch (6), Urinary products (14). Low-support estimates are unstable; no class is silently omitted.

| True label | Predicted label | Records |
| --- | --- | --- |
| Supplements | Herbal & Natural (H&N) | 120 |
| Herbal & Natural (H&N) | Supplements | 95 |
| Minerals | Supplements | 78 |
| Supplements | Minerals | 66 |
| Other Allergy | Oral Antihistamine | 41 |
| Supplements | Multivitamins | 38 |
| Single Vitamins | Supplements | 38 |
| All Other Digestive Health | Supplements | 34 |

### Brand

MiniLM has no correct validation predictions for Astepro (13 records) or Vitauthority (14), alongside several classes with one or two records. Its 96.66% overall accuracy should not hide this long-tail performance. Raw ProductBrand alone is ambiguous for some curated catch-all decisions; the lookup majority rule and unseen-key fallback particularly under-detect Other Brands. No automatic relabeling or feature removal was performed.

Classes with zero validation support: `Bonafide`, `Health & Her`, `Hers`, `Meno`, `Womaness`. Their F1 is retained as zero in full-vocabulary macro-F1.

Classes with validation support but zero recall (record support in parentheses): Astepro (13), Estroven (1), Love Wellness (2), Nasonex (2), NaturaNectar (1), Vitauthority (14), Voltaren (1), Wile (1). Low-support estimates are unstable; no class is silently omitted.

| True label | Predicted label | Records |
| --- | --- | --- |
| Other Brands | UnItemised brand | 170 |
| UnItemised brand | Other Brands | 101 |
| Vitauthority | Vitamatic | 14 |
| Astepro | UnItemised brand | 11 |
| UnItemised brand | Pepto Bismol | 9 |
| UnItemised brand | CVS Health | 7 |
| UnItemised brand | Vitamatic | 5 |
| UnItemised brand | Metamucil | 4 |

## Verified Maria handoffs

There is one final MiniLM checkpoint in each ZIP. Each checkpoint embeds the correct target, ordered labels/mapping, five-field/256-token representation, split identity, model configuration, feature contract and unweighted loss configuration. Tokenizer, encoder configuration, aggregate metrics and checksum manifest are included. Brand additionally contains the training-only lookup mapping, its full-population/cohort results and fallback rules. No new inference endpoint or record-level prediction exporter was implemented. The frozen trainer's existing internal validation artifact was reused unchanged and excluded from the ZIPs.

- TargetAgeGroup: [Maria handoff ZIP](../../../data/processed/minilm_remaining_targets_v1_20261029/targetagegroup_minilm_maria_v1.zip) (83,624,228 bytes). SHA-256: `c5577b4f32985ce744290bafddb597ef20ef6fddf689dab0f4d6928e94a1d84b`.
- Sub-Segment: [Maria handoff ZIP](../../../data/processed/minilm_remaining_targets_v1_20261029/sub_segment_minilm_maria_v1.zip) (83,706,118 bytes). SHA-256: `01c46b9ba2bcf75c2cf31d7bced63c2661da1bf03e3b3b173cf38173f104261b`.
- Brand: [Maria handoff ZIP](../../../data/processed/minilm_remaining_targets_v1_20261029/brand_minilm_maria_v1.zip) (83,980,751 bytes). SHA-256: `7913f526ac384e708b742192fb874045fcb8c873f5ed58fa73a77e1bf6c589ad`.

All three packaged checkpoints were reloaded with weights_only=True and their tensor-state hashes, classifier dimensions, ordered labels, representation, target and configuration matched the evaluated models. ZIP CRC checks, member checksums, source artifact hashes, source/approved allocation seals, and all 13 frozen algorithm source hashes passed. TargetAgeGroup's 31 tracked artifacts remain byte-for-byte unchanged. Both Brand methods share the identical validation record-ID hash, and the lookup's training record/group hashes match the approved training partition.

The 203 original CPU environment distribution metadata entries remain unchanged. No package, hyperparameter, approved allocation, frozen Segment artifact or frozen trainer code changed during the resumed training. Protected-test labels were not inspected, test.csv was not opened or hashed, no external delivery occurred and nothing was merged.

**Remaining technical handoff blockers: none.** Sub-Segment's weak/unsupported classes and Brand's accuracy-versus-macro-F1 tradeoff remain important evaluation limitations for Maria. No separate simple baseline was fitted for TargetAgeGroup or Sub-Segment in this batch; no comparative claim about simpler methods for those targets is established.

## Resume provenance

The original sequencing failure happened before Sub-Segment training. It was corrected by converting JSON archive path strings to Path in verify_predecessors; all 82 focused tests passed after that correction. On explicit user authorization, the original protocol and stop record were backed up, and maintenance/authorized_resume_revision.json recorded only the runner hash revision. Training configuration, label mappings, allocations and completed TargetAgeGroup artifacts were preserved. Sub-Segment then completed and verified before Brand started. No training retry or retraining of TargetAgeGroup occurred.

Machine-readable records: [completion verification](../../../data/processed/minilm_remaining_targets_v1_20261029/completion.json), [consolidated comparison](../../../data/processed/minilm_remaining_targets_v1_20261029/consolidated_comparison.json), and each target's *_result.json. The original source snapshots, corrected runner snapshot and dependency lock are retained for reproduction.

The resumed runs completed with no training errors. The inherited loader notice about nonpersistent embeddings.position_ids was unchanged. Brand also emitted pandas DataFrame-fragmentation warnings from the frozen trainer's existing internal validation writer; these were nonfatal and all artifact/metric checks passed. The frozen writer was not changed.
