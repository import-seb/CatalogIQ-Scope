# Sub-Segment and Brand maintenance findings

Publication note (November 15, 2026): private `data/processed/...` evidence referenced below is excluded from the PR. These historical findings are unchanged; the [shareable verification summary](docs/maintenance_verification.md) and [portable test instructions](experiments/maintenance_research/README.md) cover the published engineering subset. Further model-quality research remains deferred.

These findings extend [the accepted maintenance report](MAINTENANCE_RESEARCH_REPORT.md). Original results and artifacts were preserved. New experiments used only approved development data and retained finished-model features; no protected test data/identities, pretrained downloads, encoder fitting, full retraining or production changes were involved.

Evidence root: `data/processed/maintenance_research_v1_20261114T060644Z/strategy_v1_20261114T123819Z/`. The predeclared protocol (`data/processed/maintenance_research_v1_20261114T060644Z/strategy_v1_20261114T123819Z/protocol.json`, private local evidence), Sub-Segment analysis (`data/processed/maintenance_research_v1_20261114T060644Z/strategy_v1_20261114T123819Z/run_003_research/subsegment_analysis.json`, private local evidence) and Brand analysis (`data/processed/maintenance_research_v1_20261114T060644Z/strategy_v1_20261114T123819Z/run_003_research/brand_analysis.json`, private local evidence) retain complete per-class results. A final seal records checksums, tests, resources and preservation.

## Sub-Segment: failure mechanisms and safe response

We independently recounted all 12 previous method/order results from saved validation predictions; they exactly reproduced the stored per-class metrics. Original baseline: 53,956 training records, 11,562 validation records, 49 classes, accuracy 88.5833%, macro-F1 0.657268. Updates used 512 seen-training hard-example proxies plus 2,048 disjoint-group replay records. These are not future human confirmations.

### Rare classes matter, but do not explain the common failure

Thirteen classes have fewer than 30 validation examples; two have none. One additional miss costs 3.125 recall points for Hemorrhoid Remedies (32 examples), but those quantization limits are not the explanation for ordinary SGD's recurring failures:

| Original SGD order | Failing supported classes | Net loss of correct predictions |
|---|---|---|
| Shuffle | Nasal Sprays, support 70; Other External Analgesics, support 114 | 2 and 7; recall losses 2.86 and 6.14 points |
| Class blocks | The same two, plus Multivitamins, support 537 | 2, 7 and 17; Multivitamins loss 3.17 points |
| Reverse class blocks | Nasal Sprays; Other External Analgesics | 3 and 7; losses 4.29 and 6.14 points |

AdamW and LoRA repeatedly harmed Other External Analgesics too (roughly 8.77â€“11.40 recall points across their orders). PA-II additionally caused large failures in common classes, with accuracy falling to 61â€“63% in blocked orders. Keeping base weights frozen does not preserve the resulting classification function.

### Replay coverage is present; effective balance and boundaries remain concerns

All 49 classes were represented in replay, so an absent-class explanation is insufficient. Coverage is very uneven: some classes have one replay example; Nasal Sprays has 13, Other External Analgesics 16 and Cough six. The incoming batch is error-enriched, with 125 Multivitamins examples and 27 Cough examples versus 100 and six replay examples respectively. This distribution differs from the historical population. One exemplar and coarse class presence do not ensure coverage of within-class decision boundaries.

The fixed new 2Ã—2 diagnostic varied only replay weighting and post-update delta fraction. It reused the original SGD heads as controls rather than refitting them. Exactly **three new SGD fits** used inverse-square-root replay class-frequency weights, clipped relative to [.25, 4] and normalized to mean one over replay records. Incoming weight remained one. Epochs, LR .001, L2 .01, batch 16, memberships and all three orders stayed fixed. Each original/weighted head was also evaluated at a fixed **one-quarter delta** from the original head. No parameter search or validation-based choice followed.

| Replay / applied delta | Accuracy range | Macro-F1 range | Original retention gate |
|---|---:|---:|---:|
| Original / full (reused control) | 88.5487â€“88.7044% | 0.659032â€“0.659893 | 0/3 orders |
| Original / quarter | 88.5400â€“88.5919% | 0.656845â€“0.659807 | 0/3 |
| Weighted / full | 88.5054â€“88.6006% | 0.659687â€“0.660055 | 0/3 |
| Weighted / quarter | 88.5487â€“88.5746% | 0.659353â€“0.659698 | 0/3 |

Quarter deltas still lost four or five correct Other External Analgesics decisions, or 3.51â€“4.39 recall points. Weighted full deltas still lost seven. Each new fit took 0.028â€“0.035 seconds; no embeddings were regenerated. Shrinking the delta reduced the median original top-two logit margin of regressed records into roughly .016â€“.063, consistent with near-boundary sensitivity. This is observational evidence about the frozen representation/head, not proof that all label errors or causal mechanisms have been identified. Weighting a fixed replay sample is not a full test of alternative replay selection.

### Conflicts are real, but cannot be used to dismiss retention failures

The full approved development population contains ten conflicting normalized five-field fingerprints: eight pairs entirely in training, two entirely in validation. One training pair, labeled Minerals versus Multivitamins, occurs in the selected proxy; neither label directly names the persistent Nasal Sprays/Other External Analgesics victims. Feature equality is not conclusive product identity, so these need evidence-based adjudication. They demonstrate ambiguity/noise but do not establish that removing one pair would fix the observed harms. Validation conflicts were flagged without changing truths or historical scores.

**Safe outcome:** keep the existing Sub-Segment model, use verified correction memory and adjudicate identities/taxonomy. No tested update is eligible. Future investigation should obtain independent affected-class/boundary examples, resolve conflicting labels, and predeclare a group-preserving class/boundary replay or constrained/distilled-head study. Class balance alone and modest delta shrinkage have not solved the problem. Do not weaken the per-class gate, drop rare classes or run repeated shrinkage trials until historical validation passes. An independently evaluated safe improvement remains an open research requirement; a static model plus memory is an acceptable durable policy.

## Brand: preserve the probabilistic lookup

Inspected implementation: `src/catalogiq/brand_probabilistic_lookup.py`, `brand_key` in `target_minilm_experiment.py` and `_plain_text` in `segment_transformer.py`. Only the exact pure normalization/prediction definitions were executed for verification; export/preparation entrypoints that read protected identities were never called.

Reconstructed the sealed Brand-only train/validation allocation using original development source records and `brand_development_allocation_v1_20261029/development_assignments.csv`. Exact original membership hashes, 58,888 training counts, all 11,354 normalized keys and 143 class counts matched. All 12,619 original validation probabilities reproduced with **maximum absolute error 0.0**, including all saved decisions/truths. Baseline accuracy is 95.6890%, macro-F1 0.931113. No original model/result was revised.

The estimator uses alpha=1 and

`P(class | key) = (key_class_count + baseline_prior_class) / (key_total + 1)`.

Missing/unseen keys fall back to the class prior. The original prior is computed from all labeled training counts, including missing ProductBrand records. There are 470 multi-label keys and 2,165 validation fallback records. Consequently, a single hard mapping or fallback-label overwrite is not a safe maintenance representation.

The isolated `BrandOverlay` prototype stores explicit per-product replacements against an unchanged base. Reconfirming a known old contribution is idempotent with zero count delta. Correcting one decrements the exact old contribution and increments the adjudicated replacement; impossible subtraction, missing keys and unknown labels are rejected. Removing the change exactly restores base distributions. Identity/contribution binding must be supplied by a trusted registry; the utility cannot infer it from strings.

One synthetic counter stress selected the first eligible **training** contribution by record ID, not by validation outcome. It arbitrarily switched its label to the next existing vocabulary label; this is deliberately **not a real correction or quality improvement**. The touched key had zero validation rows, so this test does not measure touched-key accuracy/generalization.

| Prior policy after synthetic correction | Untouched validation distributions changed | Fallback maximum probability change | Validation decisions changed |
|---|---:|---:|---:|
| Anchored original prior | 0 / 12,619 | 0 | 0 |
| Recomputed empirical prior | 12,619 / 12,619 | 0.0000169814 | 0 |

Both preserve aggregate accuracy in this stress, but only anchoring makes untouched-key/fallback distributions invariant. Small probability changes may matter to calibration and ranking even when argmax stays unchanged. Do not claim that the recomputed-prior stress caused decision regressions; none occurred here. Synthetic tests also verify touched-key count replacement and reversibility, not empirical new-key performance.

**Recommendation:** local, deduplicated count overlays anchored to the approved prior, with explicit new format/version, touched-key evidence and held-out review. Preserve alpha, normalization and vocabulary. Require separate approval and whole-system evaluation for a global prior refresh, canonical new class or production loader change. Novel ProductBrand aliases of known brands are not novel labels. Ambiguous contribution identity, new classes and weak key evidence remain memory-only until resolved.

## Prototype, resources and preservation

New files reuse `core.py`, membership/fingerprint utilities, checked artifacts and `resources.BoundedJob`; no duplicate model/embedding cache or replacement experiment tree was created. `correction_ledger.py` supplies the versioned five-target event ledger and conservative eligibility projection. `maintenance_offline.py` supplies group-cohort proposals, paired evaluation and reversible Brand counts. Synthetic tests cover conflicts/adjudication/reopening, revocations, deduplication, synthetic/validation/unknown exclusion, identity and ontology quarantine, all five targets, approval requirements, path/foreign-database refusal, hash corruption, append-only enforcement, paired-population/binding failures and Brand inverse/invariance properties.

The first test attempt exposed Windows SQLite connections left open after transaction contexts. It failed during temporary-file cleanup; explicit close-in-finally fixed this only in new prototype code. Failure logs remain. Subsequent tests passed; final counts and source hashes are recorded in the final verification. No research candidate was refit to fix that infrastructure issue.

The research worker ran for **21.51 seconds**, used **20.05 aggregate CPU seconds**, peaked at **1.873 GiB job commit** and **1.263 GiB sampled working set**. It initialized no CUDA. Across the phase, four numerical threads/four-core affinity, one interop thread, hard 4-GiB job commit and sampled 4-GiB RSS stops were enforced using the retained Windows job controls. The persisted phase envelope was one hour / 14,400 CPU seconds; no limit was reached. These are worker/supervisor execution measurements, not total document-writing time or production cost estimates.

Before/after checks cover original tracked source, selected model artifacts, development files, original environment metadata/configuration, all shared-overlay files, Brand artifacts/allocation, the unchanged accepted report, and every sealed prior research source/evidence file. Tracked/staged diffs remain empty. Only isolated utilities, these documents and new bounded evidence were added. No dependencies were installed, original files rewritten, protected files opened, candidate applied, push, merge or deployment performed.
