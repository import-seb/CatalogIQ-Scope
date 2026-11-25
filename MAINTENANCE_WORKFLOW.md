# CatalogIQ correction-to-maintenance workflow

This is an offline design and tested isolated prototype. Production ingestion, inference routing, automatic updates and deployment have not been implemented. [Target policies](MAINTENANCE_STRATEGY.md) determine whether any update is worth attempting. Human approval remains mandatory for every prediction.

## 1. Capture a review event, never overwrite history

Create an append-only event when a reviewer confirms, corrects, rejects or defers a suggestion. Record each target decision separately, linked to one review session and product/input version; a multi-target session is not five independent products. Include:

| Field group | Required information and purpose |
|---|---|
| Event provenance | Stable external event/idempotency ID, source system, UTC occurrence and server-receipt times, authenticated pseudonymous reviewer, action, explicit human approval, evidence/reason and adjudication/revocation references |
| Product identity | Steward-owned canonical product/entity ID, source record ID, retailer plus SKU/MDM identifiers where available, raw/validated GTIN, grouping/version and alias evidence; verification status and steward provenance |
| Exact input | All five **full, untruncated** fields, raw canonical input SHA and normalized feature fingerprint; input/text/tokenizer contract version; capture changed content as a new product-input version |
| Classification | Target, original displayed first choice, human-approved label or rejection/deferral, ontology/version, original model/checkpoint/head or lookup identity, memory suggestion/version if shown, final displayed suggestion source |
| Optional diagnostic capture | Saved score/ranking vector and ordered label vocabulary, displayed alternatives/confidence, uncertainty/calibration version, active review start/end/pause timestamps and UI actions; explicit missingness when unavailable |
| Eligibility/lineage | Registry snapshot/version, known train/validation matches, prior accepted-update membership, identity/grouping attestations, protected/reserved exclusion attestation from the authorized steward, synthetic/import flags |

Do not invent timestamps, rankings, probabilities or reviewer effort retrospectively. Confirmation date does not prove product novelty. Authentication, role control, privacy/retention policy and audit-anchor storage must be designed before production ingestion. This prototype receives a trusted caller principal; it is not an authentication service. Reviewer text and metadata are stored as data, never executable paths or commands.

## 2. Resolve identity, duplication and contradictions

Compare source identity, canonical group and full-feature fingerprint against the **target-specific** original and all accepted-update training registries. A new row ID for the same old group or inputs is seen training. A product might be new to the business but still require verification; an old catalog product may be unseen by a particular target because its label was previously missing. Track those claims separately.

The prototype's `EligibilityIndex` accepts only development membership and steward-owned attestations. Validation matches take precedence, including feature matches under a new identifier. Unknown membership becomes `unverified_new_candidate`, not “new.” Only an upstream attestation of verified identity/grouping, exclusion from training and reserved data, and provenance can mark `unseen_attested`. No protected test files or protected identity registry are opened. If that attestation is unavailable, keep the event for memory/audit, but exclude it from experiments.

Treat checksum-valid GTIN as supporting evidence only: the prior study found a harmful GTIN-only reuse. Retailer/SKU conflicts, reformulations, pack variants, missing identifiers and identical descriptions require identity checks. Same product, target and exact raw input version plus the same label yields one effective memory/contribution entry regardless of review count. Same context plus different labels creates a conflict. Same normalized full-feature tuple with contradictory active labels across identities creates an identity conflict. A changed input version does not inherit the old suggestion automatically.

Resolve conflicts through an authenticated adjudicator with product evidence and taxonomy version. A resolution must cover all current conflicting event IDs; an additional contradiction reopens the hold. Append revocations and resolutions; never mutate/delete old events. Bad identity aliases need steward reconciliation before a new memory version. Do not settle classification by majority reviewer votes or the latest ordinary review.

## 3. Update correction memory without training

Project the ledger into a versioned snapshot: event-chain head, ontology/catalog hash, eligibility-registry version/hash, product/input keys, approved labels and provenance. Memory suggestions require verified identity **and** matching input version, active nonconflicting approved label and compatible ontology. Reject/defer/unapproved events do not become labels. Synthetic entries never surface as real suggestions. A validation example can be audited but is excluded from memory reuse in development evaluation and from all updates.

Show the prior approved label and source to the reviewer as an assistance feature only. The reviewer must approve every current prediction; a memory hit creates no approval event on its own. On a conflict/revocation, suspend affected suggestions and rebuild the projection. A correction-memory update needs no encoder/head/lookup modification.

The prototype uses an append-only SQLite event table with UPDATE/DELETE rejection, stable idempotency keys, bounded payloads, parameterized SQL and a SHA-256 chain. Same ID/same request is idempotent; same ID/different request is rejected. Reads verify the chain and snapshot one consistent event list. It refuses writes outside its explicit output root and refuses foreign databases/catalog migrations. Connections are explicitly closed on Windows.

Limitations: SQLite/file ownership is trusted, hash chains need externally retained signed anchors/backups to resist an owner rewriting history, and cross-system identity/ontology stewardship is external. The prototype is deliberately not wired into application inference, identity services or real confirmation ingestion.

## 4. Freeze an eligible cohort before model selection

Classify each effective event as seen training, validation excluded, steward-attested unseen, or unresolved. Retain original-row label corrections as such, not “new products.” An old model mistake with an unchanged old training label is a seen-training hard example, not a newly discovered training label. Deduplicate by verified product/group and input version, and retain links to exact original contributions for Brand.

The offline `plan_cohorts` helper proposes a deterministic 20% whole-group holdout for genuinely unseen eligible groups, shared across targets. It deduplicates same target/group/input entries, rejects contradictions/synthetic/quarantined records, and never places a seen-training group in the unseen holdout. Freeze that proposal and its membership hashes **before** candidate fitting or performance inspection. The 20% allocation is a proposal; ensure the target-specific minimum counts in the strategy or collect more data. Do not change the existing validation splits. Entire groups remain on one side. Combine an old-label correction with any aliases/input versions in the same role; no independent votes from duplicates.

Use historical validation only for retention, a frozen prospective/naturally sampled incoming holdout for generalization, and separately named class/boundary challenge cohorts for coverage. Repeated development-validation reuse creates selection bias: do not tune against it indefinitely. Future decisions need fresh confirmations and later time-based holdouts, without rerunning the protected test set.

## 5. Decide whether an offline update is warranted

Generate a per-target evidence card: independent groups, eligible corrections versus reconfirmations, true new groups, class/error-direction counts, unresolved conflicts, recent first-choice errors, memory coverage and holdout availability. A periodic review is permitted; a periodic retrain is unnecessary. Apply the policy for Segment, Mnfr, TargetAgeGroup, Sub-Segment or Brand rather than routing all five through one learner.

Insufficient evidence, no persistent quality problem, or no independent holdout means **keep memory and do no parameter update**. A new class or unresolved taxonomy conflict means adjudication/ontology planning. Sub-Segment remains on hold even if a total-count threshold is reached. All proposed numeric eligibility/precision/monitoring thresholds need sponsor approval and empirical validation; the current prototype does not turn them into an automatic release scheduler.

## 6. Create a bounded isolated candidate

For a separately approved pilot, snapshot the current accepted parent, original reference and eligible ledger head. Verify hashes before loading. Copy no source datasets; store membership hashes and small deltas under a new evidence directory. Keep the original encoder/head, data splits and Brand lookup immutable. Cache features by encoder function (including adapters), tokenizer/text contract and input version. Head-only updates reuse those vectors; changing the encoder function invalidates that cache.

Use the target's fixed method and replay policy. Test shuffle, class-block and reverse-class-block orders and, with real data, the actual chronological arrival order. These are separate stress tests, not permission to choose the highest-scoring order. Future multi-cycle tests must advance from the prior accepted version and compare cumulative retention with the original. Never represent the earlier 512-record seen-training proxies as incoming data.

Proposed initial CPU pilot envelope: at most 512 incoming records, 2,048 replay records, two epochs, four numerical threads/four-CPU affinity, one interop thread, aggregate worker job commit capped at 4 GiB, sampled working-set stop at 4 GiB, no CUDA initialization, ten-minute wall budget and 2,400 CPU-second budget. Exact historical-validation size stays target-specific. Establish an enforceable per-pilot supervisor and a new persisted budget before execution. Larger replay, encoder adaptation, new classes or bigger cohorts need a new plan. Brand count-only candidates should use the same envelope and no neural model. The one-hour strategy diagnostic budget used here is not authorization for future pilots.

For Brand, keep the frozen original lookup separate from a candidate count overlay; idempotently replace exact known product contributions. Anchored prior keeps untouched-key/fallback distributions invariant. Do not create a permanent mapping from a single review, recompute the global prior silently, tune smoothing to validation, or add a label without ontology approval.

## 7. Evaluate and reject unsafe candidates

Require identical ordered vocabulary, records, truth definitions and metric handling for each paired comparison. `compare_records` refuses duplicate/missing records, different truths or unknown labels, requires matching target/base/input/population/vocabulary bindings, verifies the population-ID hash, and reports historical retention, per-class precision drops and correct/wrong flips. It returns an **offline screen**, not a release decision. Small-class precision safeguards and human-review audits remain separate explicit policy steps.

For each order/cohort, report accuracy, macro-F1, per-class support/precision/recall/F1, confusion matrix, incorrect first choices, old-correct→new-wrong and reverse flips. Report range and worst order, never just averages. Preserve the earlier 0.5-point/.01/2-point development screens. Add proposed per-class precision/challenge safeguards without rewriting past results. Quantify uncertainty at group level on genuine new data; support absence is “unmeasured,” not “passed.” Inspect exact changed decisions for Mnfr, Children/Infant, Sub-Segment boundary classes and touched Brand keys.

Use top-3/MRR only with actual saved rankings and enough classes; retain historical primary metrics. Evaluate calibration only with legitimate score distributions; distinguish probabilities from margins. Recalibration is a separate fitted operation requiring its own development split. Measure preprocessing/embedding, head/count update, validation, serialization/reload and whole-cycle time separately; record CPU time, peak RSS/commit, storage and GPU use if later authorized. The present CPU-only diagnostic did not generate embeddings or benchmark inference latency.

If any gate fails, reject and keep the current accepted version plus valid memory. Record the first material failure, harmed classes and data lineage. Ordinary harness debugging may be corrected within the isolated scope and budget; do not make successive performance-driven changes until validation passes. Missing evidence yields an inconclusive result rather than forced promotion.

## 8. Review, release and roll back separately

No candidate from this work is deployed. Future integration needs explicit approval, authenticated reviewer/adjudicator services, steward identity/eligibility interfaces, privacy policy, frozen schemas and an operator-controlled model/lookup version switch. Passing metrics alone cannot change that switch.

An immutable candidate bundle contains current parent/original hashes, small payload with checksum and target/vocabulary/input binding, ledger/registry snapshot hashes, train/replay/holdout memberships, complete evaluation, costs, approval and rollback pointer. Reuse the prior checked neural delta loader; do not infer an encoder from bare adapter metadata. A Brand overlay needs its own versioned loader, not a disguised original empirical-prior JSON. Verify save/reload and the inverse/previous-version path before promotion. Never overwrite a baseline or auto-apply ledger contents to it.

On an integrity or identity incident, disable affected memory entries, revert the version pointer to the last safe accepted bundle and preserve audit evidence. On a quality regression, pause promotion, assess the affected cohorts and use the approved rollback path. Revocations must identify all dependent update bundles; deleting a correction does not reverse its trained effect. Back up the ledger and anchor its hash outside the same writable store, with retention/deletion handling designed for privacy obligations before production.

## 9. Track maintenance usefulness and reviewer outcomes

Record every traffic prediction and its subsequent review outcome, including unchanged approvals, with sampling policy and exposure/source metadata. Distinguish first-choice error rate on independently adjudicated truth from human correction rate: acceptance can include reviewer mistakes and anchoring. Use independently double-reviewed samples to estimate final quality and disagreement. If measuring review time, capture active time and pauses; compare matched/randomized assisted workflows and account for reviewer/product mix. No current review-time measurement exists.

Track per target and class, repeat/new/seen-training cohorts, memory/model suggestion source, version, unresolved labels, revocations, new-key coverage, and elapsed time since update. Report counts and uncertainty alongside rates. Monitor memory coverage/conflict rate and lookup probability drift even when parameters never change. No automatic approval, confidence-based acceptance or hidden bypass is introduced.

## Implementation roadmap

| Stage | Available now | Remaining work / evidence |
|---|---|---|
| Isolated prototype and offline review | Tested five-target ledger; conflict/idempotency/revocation logic; conservative eligibility; cohort proposals; paired evaluator; reversible anchored Brand count candidate; existing checked delta tooling | Review these artifacts; no production traffic or application changes |
| Approved shadow integration | Schema and human-review design are ready to evaluate | Explicit integration approval; authenticate roles; steward identity/grouping and reserved-data attestations; canonical original contribution registry; privacy/retention and externally anchored backups; UI must still request approval |
| Prospective evidence collection | Event/novelty definitions and cohort protocol specified | Genuine confirmations, verified labels and groups, original/prior-update membership, untouched incoming holdout, score snapshots and actual reviewer actions/timing |
| First target pilots | Segment SGD is first candidate; Mnfr PA-II is restricted; age updates conditional; Brand overlay local | Sponsor-approved count/precision gates, adequate class support and power, bounded pilot approval, paired incoming results and rollback checks |
| Sub-Segment recovery | Failure analysis completed; no safe researched update | Adjudicated ontology/conflicts, independent boundary cases, fixed class/boundary replay or constrained-objective study; safe retention across orders before considering promotion |
| Long-term operation | No continuous retraining required | Multi-cycle chronological drift/forgetting tests, live reviewer-outcome study, operator-approved promotion and demonstrated rollback; LoRA only if a new controlled study shows a need |

Prototype source: [correction_ledger.py](experiments/maintenance_research/correction_ledger.py), [maintenance_offline.py](experiments/maintenance_research/maintenance_offline.py), and [synthetic tests with complete API examples](experiments/maintenance_research/test_strategy.py).

The publishable subset excludes local session pointers, environment inventories and historical research runners. Reproduce the isolated engineering checks using the [portable README](experiments/maintenance_research/README.md) and review the [verification summary](docs/maintenance_verification.md). The private study evidence remains archived locally; no new model research is needed to review this workflow. Future quality pilots still require an approved cohort, protocol and budget. These documents authorize no production integration or model promotion.
