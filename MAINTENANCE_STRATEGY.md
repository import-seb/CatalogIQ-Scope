# CatalogIQ maintenance strategy — all five targets

Publication note (November 15, 2026): private `data/processed/...` evidence referenced below is excluded from the PR. These historical findings are unchanged; the [shareable verification summary](docs/maintenance_verification.md) and [portable test instructions](experiments/maintenance_research/README.md) cover the published engineering subset. Further model-quality research remains deferred.

Status: isolated development research and prototype; no production behavior or model has changed. Every prediction, including a correction-memory suggestion, requires human approval. This extends [the accepted research report](MAINTENANCE_RESEARCH_REPORT.md); it does not replace its results. Detailed new evidence is in [maintenance findings](MAINTENANCE_FINDINGS.md), and operational steps are in [the workflow](MAINTENANCE_WORKFLOW.md).

## Recommended operating policy

Keep the finished models and Brand lookup. Record every review event, make reliable past confirmations available as suggestions, and update parameters only when independent evidence justifies it. A new confirmation, a newly observed catalog product, and an example unseen by a particular model are different things. Old training products remain old training products even when corrected today.

| Target | Existing validation accuracy / macro-F1 | Default maintenance | Conditional offline update |
|---|---|---|---|
| Segment | 96.9461% / 0.944490 | Verified correction memory | Warm-start softmax SGD on the current accepted head; freeze its existing fine-tuned encoder |
| Mnfr | 99.9661% / 0.995729 | Verified correction memory; inspect both classes | Restricted PA-II pilot versus SGD/no change; no general PA deployment |
| TargetAgeGroup | 99.7505% / 0.987426 | Verified correction memory; otherwise leave model alone | Head-only SGD or previously tested head AdamW only for a demonstrated persistent error pattern |
| Sub-Segment | 88.5833% / 0.657268 | Correction memory plus ontology/identity adjudication | Hold all parameter updates; current retention failures remain unresolved |
| Brand | 95.6890% / 0.931113 | Correction memory plus versioned probabilistic lookup maintenance | Conservative per-key count overlay with an explicitly anchored prior, subject to separate integration approval |

The four neural metrics retain their original populations and definitions. Brand's saved 12,619-row probabilities, decisions and training counts reproduced exactly. These are development results, not new-product estimates or human-review outcomes.

## Common requirements, independent of target

Store an immutable confirmation/correction event immediately after a verified reviewer acts. Retain rejects and deferrals for audit too; they are not training labels. Preserve the original suggestion, reviewer-approved label, product/input version and model/ontology version. Memory activates only for a verified identity and exact input version with one unambiguous approved label. A later contradiction suspends the suggestion; ordinary last-write-wins is prohibited. Every reuse is visibly a suggestion requiring a new human decision.

Never count repeated confirmations as independent examples. Separate original-row label corrections from newly labeled products. Resolve identifier aliases and product groups before deduplication; GTIN checksum validity, retailer/SKU equality or feature equality alone does not establish identity. Keep unresolved and unknown-vocabulary records in quarantine. Do not silently expand output classes or interpret a new label as “All others.”

For any candidate, require unchanged source artifacts; explicit encoder/head or lookup, vocabulary, tokenizer/input and payload hashes; leakage-free group membership; no unresolved update labels; reproducible save/reload; and a frozen evaluation protocol. Compare with both the current accepted version and the immutable original reference to detect cumulative drift. Keep each target's historical validation closed to training and replay. Do not open protected test data or its identity registry; reserved-data exclusion for future records requires an authorized upstream steward's attestation.

Retain the original development gate: accuracy loss no greater than **0.5 percentage points**, macro-F1 loss no greater than **0.01**, and recall loss no greater than **2 percentage points** for each class with historical validation support at least 30. Require every declared update order to pass, not just the best order. Retain all configured labels in macro-F1 with zero-division handling unchanged. These tolerances are screening limits, not permission to consume that much quality every cycle.

Proposed additional safeguards, requiring sponsor approval before release: apply the same 2-point limit to per-class precision when baseline predicted-positive support is at least 30; separately inspect low-support and zero-support classes rather than silently treating them as protected by that gate. On sensitive or tiny cohorts require manual review of every changed decision, with no unexplained new error. Report recall and precision together; human review does not make either kind of error harmless. A class with no evaluation examples cannot be certified.

An update must also improve a predeclared problem on a separately held-out incoming cohort, without unacceptable regressions on unaffected classes. The old 10% seen-proxy error-reduction screen remains descriptive mechanics evidence; it is not a production success criterion. Measure confidence reliability after function changes, but never convert confidence into automatic approval. Do not tune repeatedly against historical validation until something passes.

## Target-specific policies

### Segment

Consider a small frozen-encoder head update only after confirmations show repeated, adjudicated errors across distinct product groups, not merely repeated reviews of one SKU. Existing three-order SGD results preserved all retention gates, with accuracy 96.9963–97.0131% and macro-F1 0.945640–0.946059 on seen-training proxies. They justify a pilot, not a deployed delta.

Start from the current accepted head, using the unchanged five-field, 256-token, attention-mask mean-pooled encoder. Keep the previously tested SGD configuration: LR .001, L2 .01, two epochs, batch 16, with disjoint-group incoming and historical-plus-confirmed replay. Bind cached features to the exact encoder function, tokenizer/text contract and input fingerprint. Do not train a new head from scratch. Begin with the existing 512 incoming / 2,048 replay research caps; larger cohorts require a new resource plan.

Provisional pilot eligibility: at least **100 distinct adjudicated update groups**, including **20 confirmed first-choice error groups**, plus the common incoming evaluation cohort below. Require affected-label coverage and all original retention gates in shuffle, class-block and reverse-class-block orders. These count thresholds require validation/sponsor approval and do not establish statistical power. If the candidate fails, keep memory and the accepted head, record its failure, and investigate that failure before considering another method. PA-II and LoRA are not routine Segment maintenance.

### Mnfr

The original validation contains only four errors. Prior PA-II corrected six seen-proxy errors and two or three validation decisions, with all three order gates passing; this is narrow evidence. Use correction memory while collecting genuine adjudicated examples from both “J&J” and “All others.” Count the two error directions separately, including precision/recall and incorrect first-choice burden. Do not assume J&J recall should trump precision.

Provisional eligibility: **100 update groups**, including **30 independent confirmed error groups**, and the common holdout with adequate J&J coverage. Many months without this evidence is a reason not to update. Test only the previously specified PA-II C=.01 against SGD/no change under the same caps, replay and orders. In addition to common gates, propose **no observed net increase in historical J&J false negatives or false positives**, and inspect every new historical error. This stricter safeguard needs sponsor approval and is not a population guarantee. A biased rare-class challenge cohort must be reported separately from naturally sampled accuracy. On failure, retain the existing head and memory; no automatic PA aggressiveness tuning.

### TargetAgeGroup

Its high-quality baseline and tiny prior changes do not justify scheduled retraining. Trigger review only for sustained, independently confirmed Adult/Children/Infant error patterns, an input-distribution change, or an ontology issue. Resolve ambiguous age evidence first. Store corrections immediately regardless of whether an update is warranted.

Provisional eligibility: **100 update groups**, including **20 adjudicated error groups**, with at least **10 error groups in each class claimed to need repair** and an adequate incoming holdout. Trial SGD first; head AdamW with the already tested LR .0001, decay .01, dropout .1, norm clip 1 and batch 16 is a conditional alternative. Use class-preserving replay and unchanged encoder/input contract. Require no unexplained new Children/Infant errors on the historical changed-decision audit, common retention gates and incoming benefit. If errors are too scarce, ambiguous or unsupported by evaluation data, do nothing beyond memory. Prior LoRA gave no useful advantage over cheap head changes.

### Sub-Segment

Use memory and adjudication; **do not promote any researched head or LoRA delta**. All 12 previous updates failed a supported-class recall safeguard. A new fixed diagnostic reused those results and ran only three additional weighted SGD fits. Neither inverse-square-root replay weighting nor a fixed quarter-sized head delta overcame retention failures in any order.

This is not solely a rare-class quantization problem: the recurring SGD victims had 70 and 114 validation examples; a class with 537 examples also regressed. All 49 classes were represented in replay, but some by only one record. Full-feature conflicts exist, including one conflicting pair in the update proxy, but that pair does not directly label the recurring victim classes. Conflicts cannot explain away the failures. The evidence supports output-boundary sensitivity and order/replay imbalance as contributing mechanisms, without identifying a complete causal decomposition.

Resolve conflicting labels against the original product evidence and versioned Segment/Sub-Segment taxonomy. Keep the original files and validation truths unchanged; append adjudications separately. Send possible validation-label problems to the data steward, and retain historical scores as originally reported. New corrected-truth evaluations must be separately versioned.

A future research candidate needs resolved affected-class labels, genuine new groups, and independent boundary/challenge examples. Preserve whole groups while ensuring class coverage and multiple independent replay groups where available; consider a fixed boundary-aware replay design or a constrained/distilled head objective only in a separately specified study. One historical exemplar cannot certify a rare class. Do not reduce learning rates/delta fractions in a search for a historical-validation pass, drop classes, or weaken safeguards. No minimum total count alone unlocks Sub-Segment training: class-specific evidence and a new controlled protocol are prerequisites. If no candidate is safe, memory plus the unchanged model is the complete maintenance policy.

### Brand

Maintain the **existing 143-label probabilistic lookup**, never a MiniLM Brand model. Its existing normalized ProductBrand keys use plain-text cleanup, Unicode NFKC, casefold and strip. Preserve that normalization and alpha=1 smoothing. The baseline has 11,354 keys, of which 470 contain more than one label; overwriting a key with one confirmed brand would destroy legitimate distributions.

For an already-trained product, reconfirming its existing label adds **zero counts**. Correcting it requires the exact original row/key/class contribution: subtract that contribution once, add the adjudicated replacement once, and retain an inverse delta. If the old contribution cannot be proved, keep memory only. For an attested genuinely new product, add at most one effective contribution per verified product/group; repeated reviews do not add votes. Different input versions of the same product require replacement/supersession, not accumulation. Missing/ambiguous keys stay in memory pending adjudication.

Use a separately versioned sparse count overlay with **baseline prior anchored** for local maintenance. This leaves untouched keys and missing/unseen fallback probabilities exactly unchanged. It differs explicitly from the original empirical-prior estimator; do not pass it off as an unchanged model format or silently insert it into the production loader. Re-estimating the global prior is a separate global-refresh proposal requiring whole-lookup validation and approval. Our synthetic one-count correction moved all 12,619 validation probability distributions under a recomputed prior, although no decisions changed.

Provisional key-update eligibility: **three independent adjudicated product groups** agreeing on an alias, plus **three independently held-out groups** for that changed key. This is a cautionary minimum, not proof of reliability; absence of evidence means memory only. Contradictory keys require an adjudicated distribution, not majority overwrite. Audit every touched key's probabilities/argmax and existing supporting products; require exact invariance for untouched keys/fallback, nonnegative counts, all current labels retained, paired historical metrics, per-class safeguards and incoming benefit. A new alias of an existing brand differs from a new canonical class: the latter requires ontology and lookup-version approval, with new-class evaluation, before any count update.

## Evidence and minimum evaluation data

The above numeric pilot thresholds are **proposals, not measured optima or release authorization**. For each neural pilot, propose a naturally sampled incoming holdout of at least **200 unseen, steward-attested product groups**, with at least **30 independently evaluated groups per affected class**. If natural sampling cannot supply a rare class, collect a separate targeted challenge set and report it separately; do not present its case-control accuracy as traffic accuracy. Precision needs sufficient predicted positives as well as true-class support. Use group-level uncertainty intervals and predeclared sample-size/power analysis before a release claim; 200 groups will not resolve a 0.5-point difference reliably in general. Need more data or an inconclusive decision is an acceptable result.

Primary comparative requirements remain accuracy, macro-F1 and per-class preservation, on both historical and incoming data. Report first-choice error rate (=1−accuracy on complete adjudicated single-label truth), old-correct→new-wrong/new-correct flips, precision, recall and order variability. Top-3 coverage/MRR are secondary only where saved complete rankings/scores and a meaningful label space exist; top-3 is vacuous for two-class Mnfr and three-class TargetAgeGroup. Measure correction rate, review duration, disagreement and reviewer quality prospectively; none was inferred from these proxies. No 25%-faster training requirement is imposed.

## Reject, roll back, and monitor

Any failed integrity, leakage, vocabulary, resource or per-class gate rejects the candidate. Insufficient support or lack of useful incoming improvement means no update. Keep the accepted model/lookup and current nonconflicting memory, retaining candidate logs and rejection reasons. Never automatically retry with a more aggressive method or auto-promote a passing score.

Future approved releases must be immutable bundles: parent/base hashes, accepted head or lookup delta, ontology/input versions, eligible ledger snapshot, source/replay membership hashes, evaluation reports and inverse/rollback pointer. Keep the original reference and last accepted bundle available. Revoking a source confirmation invalidates dependent candidate lineage; disable its memory suggestion immediately and review affected released versions. A model does not unlearn a revoked event by deleting memory.

After separately approved shadow integration, monitor weekly per-target/cohort counts and monthly quality trends: first-choice errors, repeat-product memory coverage, conflicts, correction rate, per-class precision/recall, confidence reliability, unresolved/new-label rates and measured review time. Alert provisionally for a two-window deterioration greater than the chosen development margins, a systematic class failure, an integrity discrepancy, or an invalid source adjudication; these monitoring thresholds need sponsor validation. Roll back immediately for integrity/identity violations; quality alarms pause promotion and trigger paired review/rollback according to impact. Review low-volume cohorts by counts and uncertainty, not unstable percentages. There is no calendar obligation to retrain and no automatic acceptance path.
