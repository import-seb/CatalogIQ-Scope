# Model-selection integration handoff for Max

November 6, 2026. Build reviewer-assisted suggestions using the routing below.
This specification implements the [model-selection recommendation](../findings/03_modeling/20_model_selection_20261106.md),
not production approval. No dashboard code was changed. **Automatic acceptance
stays disabled for every target; operational confidence thresholds are null.**

## Confirmed model routing and artifact identity

All paths below are relative to the repository root, under private/gitignored
`data/processed/`. Copy verified bundles to a configured serving location; an
author's absolute workstation path is provenance, not a deployment requirement.

| Target | Route | Exact artifact / version |
|---|---|---|
| Segment | Selected MiniLM, five fields, 256 tokens | `segment_category_ablation_256_20261010/with_category/checkpoint.pt`; identity `segment_category_ablation_256_20261010/with_category` |
| Mnfr | Selected unweighted MiniLM, final epoch 3 | `mnfr_minilm_selection_v1_20261029/maria_handoff/checkpoint.pt`; `minilm-mnfr-category256-c3458037a670c8f0` |
| TargetAgeGroup | Selected MiniLM, final epoch 3 | `minilm_remaining_targets_v1_20261029/handoff/targetagegroup/checkpoint.pt`; `minilm-targetagegroup-category256-48256ce8adfceee1` |
| Brand | Selected probabilistic lookup v1 | `predictions_contract_v2_20261031/brand_lookup_probabilistic_v1/model.json`; `brand-productbrand-empirical-prior-v1-8cbdb95a5638dbed` |
| Sub-Segment | Provisional MiniLM suggestion, final epoch 3 | `minilm_remaining_targets_v1_20261029/handoff/sub_segment/checkpoint.pt`; `minilm-sub_segment-category256-8a1e8f3e742eb3a7` |
| Platform | Deferred; no model route | No artifact selected. Return unavailable and leave classification to a reviewer. |

Use the adjacent saved tokenizer, `encoder_config.json`, `model_config.json`,
training/bundle manifests and vocabulary together. Encoder base is
`sentence-transformers/all-MiniLM-L6-v2`, revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; base weights alone are not the
fine-tuned classifier. Evaluation manifests are in
`data/processed/predictions_contract_v2_20261031/<segment|mnfr|targetagegroup|sub_segment|brand_lookup_probabilistic_v1>/`.

Verify these SHA-256 values before loading, along with each manifest's member
hashes, target, representation and ordered class mapping:

| Artifact | SHA-256 |
|---|---|
| Segment checkpoint | `7e417d8204dbaeef4cea4e54690594560e3fbd3b8288f4be850ff60f4de32e27` |
| Mnfr checkpoint | `13af52155597c62e6c29ce83f229fd57fa20bee8e4184b8b654b79713deff797` |
| Age checkpoint | `9d864499526a921709272982c055f70305ff98cf5c135ee1125fc33f6b6efda0` |
| Sub-Segment checkpoint | `02f4fbb0f6ee55c80eb9068f0c8be9cc2a56982edd6bfdd8a2436918aa10997f` |
| Probabilistic Brand model.json | `19d0b22672d8e1e4033c4b9a511eee101bf9653da951f1d8c0cd574b909f2fe8` |

The deterministic Brand baseline remains at
`minilm_remaining_targets_v1_20261029/handoff/brand/lookup_model.json`
(hash `654409ad9a1ed4cc675f97b198d8a3a6df6e9828c7606410effaea50ef1759eb`).
It has slightly higher measured macro-F1 but does not provide the full confidence
distribution used by the current review experiment. Never replace it silently
with the probabilistic version or invent 100% confidence for a table match.
Brand MiniLM is retained as a comparison artifact, not an automatic fallback.

Max's supplied folder `C:\Users\sebas\Downloads\max handoff` contains only
Sub-Segment TF-IDF base/category models. They successfully use
`catalogiq.classical_baseline.predict_classical(bundle, product_frame)`, without
true labels. They are comparison alternatives; neither is the default route.
Model hashes are `c239c533f77b1d5fa4a3f8fb4a46fa90896506b508d595fa094d771ecea09b53`
(base) and `cce91a0765fd1f3ab93414da7b6b2b4ac6ff17d0ca039dea1efe75b7335d464d`
(category). Do not infer their target from the `.joblib` extension or use them
for Segment. Joblib loading is restricted to verified trusted teammate artifacts.

## Confirmed inputs, preprocessing and execution

Carry a nonblank, unique `record_id` as a correlation key, separate from model
features. Predictors require no truth columns. Keep the existing cleaning and
structural/quarantine decisions upstream; never use labels, retailer identifiers,
URLs, timestamps or provenance hashes as classifier text.

For **all four routed MiniLM targets**, preserve the saved five-field view:

1. `ProductCategory` first, at most 512 cleaned characters.
2. `ProductName`, at most 1,024 characters.
3. Retailer-supplied `ProductBrand`, at most 512 characters.
4. `ProductDescription`, at most 4,000 characters.
5. `ProductContents`, at most 2,000 characters.

Call the existing `catalogiq.segment_category_ablation.build_category_texts`
with the saved `ModelConfig` and category limit 512. It reuses `_plain_text`:
blank/null/none/nan values become absent text; HTML/entities and whitespace are
cleaned in the model view; populated fields retain their `[FieldName]` markers.
Then use the saved tokenizer, a **single combined 256-token limit**, truncation,
attention masks and dynamic batch padding. Category is a retailer breadcrumb,
not curated Segment. The frozen grouping representation still uses four fields
and 128 tokens; it must not replace the selected inference representation.
Do not omit category for Mnfr/age merely because a classical category arm lost.

Reconstruct the same MiniLM encoder, masked mean pooling, dropout and linear
classification head using `_make_classifier`; strictly load the saved state
dictionary (`torch.load(..., weights_only=True)`), verify its state/label hashes,
call `eval()` and run under `torch.inference_mode()`. Apply softmax to the logits.
Existing reference logic is `minilm_prediction_export.predict_validation`; that
function requires approved validation truth/context and is **not a general
serving endpoint**. Max needs a serving wrapper around the same formatter,
tokenizer and forward path. Do not call training or validation orchestration
from the dashboard. The standalone benchmark demonstrates label-free inference.

For **Brand lookup**, load the trusted JSON counts and call
`catalogiq.brand_probabilistic_lookup.predict_frequency_lookup(model, values)`
with retailer `ProductBrand` values. Reuse its existing `brand_key`: plain-text
cleanup, Unicode NFKC, case folding and trimming, with saved missing tokens.
The return is a probability matrix, cohort flags and normalized keys; associate
rows with input record IDs. Its fixed alpha=1 smoothed frequencies and global
training class prior are already fitted. Do not rebuild counts, tune smoothing,
or learn a mapping from incoming/validation records.

Unseen/missing keys receive the saved prior: its winner is `UnItemised brand`,
raw confidence about 0.6260. This is a fallback suggestion requiring review,
not an unknown-brand detector or evidence that the actual product has that label.
Do not cascade into Brand MiniLM without a separately approved, evaluated policy.

## Confirmed output semantics and proposed dashboard envelope

Confirmed model outputs are a saved-vocabulary predicted label and full ordered
probabilities, with `confidence = max(probabilities)` on a 0–1 scale. MiniLM
scores are raw softmax; lookup scores are raw smoothed training frequencies.
Neither is an approved probability of future correctness. The aligned manifests
declare `probabilities_calibrated=false`. No global temperature is shared across
targets, and no experimental cutoff is approved for serving.

**Proposed interface for Max/team approval** (field names/status enum are not an
already implemented dashboard contract):

```json
{
  "record_id": "<incoming product ID>",
  "target": "Brand",
  "selection_status": "selected",
  "model_id": "brand-productbrand-empirical-prior-v1-8cbdb95a5638dbed",
  "artifact_sha256": "19d0b22672d8e1e4033c4b9a511eee101bf9653da951f1d8c0cd574b909f2fe8",
  "predicted_label": "UnItemised brand",
  "confidence": 0.62603586,
  "raw_confidence": 0.62603586,
  "probabilities_calibrated": false,
  "calibration_id": null,
  "policy_version": "catalogiq-review-proposal-v1-20261106",
  "policy_approved": false,
  "probabilities": {"probability_0": "<numeric value; include every class>"},
  "status": "suggestion",
  "review_required": true,
  "review_reasons": ["automatic_acceptance_disabled", "no_approved_threshold", "lookup_without_matching_training_key"],
  "operational_threshold": null
}
```

The example uses a placeholder for the full probability map; real values must
all be numbers. Bind them to the immutable explicit `probability_i -> label`
mapping or mapping version. An evaluation CSV is a separate format and includes
known validation truth; new incoming products have no `true_label` to invent.

## Confirmed class mapping and failure requirements

- Read `manifest.label_mapping` in **numeric suffix order**, 0, 1, 2, ..., 142;
  dictionary insertion order/lexicographic sorting is not class order. Verify
  contiguous columns, classifier dimensions and exact saved vocabulary.
- Mnfr has `All others`, `J&J` in that order; age has `Adult`, `Children`, `Infant`.
  Segment has 8 classes, Brand 143 and Sub-Segment 49. Use saved files for the
  longer lists; do not rebuild an alphabetic mapping or drop unsupported labels.
  Preserve Unicode labels, including `Horbäach`.
- Unknown requested target, missing bundle, checksum mismatch, invalid input,
  inference failure, invalid/nonfinite probabilities or an out-of-vocabulary
  emitted label must not produce a successful classification. Fail the affected
  target and retain reviewer control. A classifier cannot emit a genuinely new
  class; unfamiliar in-vocabulary inputs may still be confidently wrong.
- Brand missing/unseen keys keep their explicit fallback/cohort flags. A fallback
  prior is not a calibrated product-specific confidence. Sub-Segment retains
  `provisional` status on every result. Platform remains `deferred`.

**Proposed failure envelope for approval:** use `unavailable` for deferred/missing
routes, `failed` for execution/contract errors, null predicted label/confidence/
probabilities for either, and stable reason codes. Keep other successful target
suggestions available; do not let a partial failure mark the whole product
approved. A missing feature column should be reported explicitly instead of
silently substituting a different model/representation. Missing values within
the required fields follow the existing formatter but still need review rules.

## Review policy: existing behavior and decisions still required

Confirmed: every saved policy in `confidence_review_v3_20261103/<run>/policy.json`
has `enabled=false` and `operational_threshold=null`. Keep all predictions as
suggestions for a person to approve. Existing `confidence_calibration` primitives
support disabled/provisional policy review, low confidence, all of name/
description/contents absent, Brand missing/unseen keys, and explicitly supplied
approved relationship maps. Quarantine integration and complete hierarchy rules
are not established by those primitive checks.

The later [99% reassessment](../findings/03_modeling/18_accuracy_standard_reassessment_20261103.md)
gives conditional support to Mnfr/age development policies; it does not enable
them. Experimental 50%, 93%, 95.5% and 97.7% cutoffs must not become defaults.
Calibration artifacts bind to exact source manifests/classes and are research
evidence until the team approves a policy and validates it on fresh data.

Team decisions still required: reviewer workload limits; per-target and whole-
product approval logic; approved threshold/calibration versions; retailer drift
and unsupported-input detection; rare-class policy; review audit trail; and a
versioned relationship registry. Preserve both retailer Vitamins breadcrumbs
and curated Digestive Health labels when they disagree. A fixed single-parent
Segment/Sub-Segment whitelist is explicitly disallowed by the decision log;
do not invent one or auto-rewrite manufacturer from Brand.

## November 6 application review-policy specification

The November 6 review-policy specification below extends the earlier handoff.
It does not change the selected models, fitted scores, existing primitive API or
any saved policy. See the [calibration completion](../findings/03_modeling/17_confidence_calibration_review_20261103.md#practical-cutoffs-observed-workload-escaping-errors-and-uncertainty)
for measured workload and the [Ravi proposal](review_policy_po_proposal_20261106.md)
for business choices.

### Review severity, reason codes and precedence

The following are **proposed application requirements**. Existing primitive codes
are identified explicitly; new app-only codes require Max's wrapper. They do not
imply that the dashboard already implements them or that Ravi has approved them.

| Severity | Condition and machine-readable code | Implementation/evidence |
|---|---|---|
| Mandatory | Disabled/unapproved policy: `automatic_acceptance_disabled`; missing/invalid approved cutoff: `no_approved_threshold` | Existing `review_decision`; active default for every model. Passing an experimental number does not constitute approval. |
| Mandatory | Provisional target: `provisional_model`; confidence below a future approved target cutoff: `below_confidence_threshold` | Existing primitive. Sub-Segment stays provisional. Threshold equality passes only this check. |
| Mandatory | Invalid probabilities: `invalid_probabilities`; unsupported output: `unsupported_classification`; probability/label mismatch: `conflicting_probability_classification` | Existing primitive. Wrapper also checks confidence against max probability, complete mapping and score version. |
| Mandatory | Existing quarantine: `existing_quarantine`; all name/description/contents absent: `missing_essential_product_text` | Existing primitive; upstream quarantine information must be wired in. Brand/category alone do not satisfy the essential-text rule. |
| Mandatory | Missing/unseen lookup key: `lookup_without_matching_training_key` | Existing primitive accepts `lookup_status=missing/unseen`; calibration never removes this requirement. |
| Mandatory | Broken approved relationship: `invalid_approved_classification_relationship`; incomplete related prediction: `missing_relationship_classification` | Existing primitive, only with explicitly approved/evidenced allowed pairs. Multi-parent pairs remain valid. No complete approved map is currently loaded. |
| Mandatory | Unapproved/malformed relationship settings: `unapproved_relationship_rule`, `invalid_approved_relationship_rule` | Existing primitive. Configuration failure, not evidence that a product violates taxonomy. |
| Mandatory | Deferred/unknown target: `target_unavailable`; missing prediction: `prediction_missing`; model exception: `prediction_failed`; missing model/checksum mismatch: `model_unavailable` / `artifact_mismatch`; invalid required schema: `required_input_invalid` | Proposed wrapper codes, not existing primitive outputs. Platform is unavailable; null label/confidence/probabilities rather than fabricated success. |
| Recommended | Known weak/sparsely supported predicted class: `class_reliability_concern`; new retailer/feature distribution: `domain_shift_suspected`; unusually small top-two margin: `ambiguous_prediction` | Proposed prioritization flags. Class list, drift criteria and margin cutoff unresolved; no invented numeric defaults. Rare true-class errors cannot all be detected by checking the predicted label. |
| Recommended | Unresolved curated-label dispute or suspicious catch-all: `taxonomy_adjudication_needed` | Proposed priority for taxonomy owners. `Other Brands`, `UnItemised brand` and Mnfr `All others` are supported classes, not invalid labels merely because they are catch-alls. |
| Informational | Retailer Vitamins breadcrumb versus curated Digestive Health: `retailer_taxonomy_differs` | Existing example crosses different taxonomies. Preserve both; this is not an approved incompatibility rule and must not independently reject a classification. Escalate an actual dispute for adjudication. |
| Informational | Confidence method/provenance: `confidence_not_operationally_qualified` | Explain raw/scaled research status. Positive calibration results alone never establish an approval policy. |

Initial high-priority review examples: J&J emissions, Infant suggestions,
Sub-Segment rare/weak labels and Brand catch-all/fallback cases. The evidence
shows missed true classes too, so these priorities supplement full review;
they are not detectors for every problematic product. Any future rare-class
rule must use saved training support or a versioned reliability registry known
at arrival, not hidden validation truth or audit support counts.

Combine reasons with **OR**, keep every unique code, and never let high confidence
cancel a mandatory reason. For a proposed primary display state, use this order:
`failed/unavailable` (cannot supply a valid model suggestion), then mandatory
review, recommended review, informational warning, then suggestion. Sorting
reasons alphabetically in the existing primitive is not severity precedence;
Max's wrapper selects the display priority from the registry above.

Until a policy is approved, `review_required=true` for all targets. For a
future approved policy, any mandatory condition still requires review.
Recommend treating elevated-risk recommended flags as review-required in the
initial qualified rollout until Ravi explicitly changes their handling. An
informational warning alone would not add a review requirement. All of this
is specification, not enablement; `review_decision` still defaults to disabled.

### Exact information passed between model, policy and UI

Keep three stages separate: model generates suggestions/probabilities; policy
evaluates gates and returns reasons; UI displays/records the person's decision.
Reuse `review_decision` (`review_required = not accept`) rather than rebuilding
its checks in JavaScript. The wrapper handles unavailable outputs/schema and
attaches policy state before calling the primitive. Only explicit release
configuration may supply `enabled=true`; it must bind target, model hash,
calibration identity, approved threshold and policy version together.

The proposed envelope above carries `record_id`, exact target, nullable predicted
label and confidence, raw confidence, calibrated flag/calibration ID, immutable
`model_id`/artifact hash, `policy_version`/approval status, decision status,
review flag and all reason codes. Add a `review_explanations` array or a versioned
code-to-explanation dictionary for readable UI text, e.g.:

- `no_approved_threshold`: "This target's review policy has not been approved."
- `lookup_without_matching_training_key`: "This brand name was absent from the saved lookup; its fallback needs a person."
- `target_unavailable`: "Platform has no selected classifier. Enter or review this label manually."

Unknown reason codes must retain the code and require review; do not silently
hide them. Missing outputs are null, not zero confidence or an empty string
pretending to be a label. Always retain raw model outputs; if a future approved
calibrator is applied, store the calibrated vector and its identity separately,
keeping confidence equal to the maximum of the displayed probability variant.
Human edits never change the original model confidence to 1.0.

### CSV upload, correction and export workflow

1. **Upload/validate:** preserve source columns and product identifiers. Require
   the expected feature columns; do not silently classify under a different
   representation. Propose a unique upload/row correlation `record_id` if source
   product IDs are blank or duplicated, while preserving the original identifier
   separately. This is an app identity proposal, not a change to frozen evaluation
   IDs/hashes. A missing required column blocks classification for affected
   targets; per-row missing values follow the formatter and review rules.
2. **Classify/evaluate policy:** return one outcome per requested row/target,
   including explicit failed/unavailable outcomes. Combine target review flags
   with OR for the product's queue entry and deduplicate products; do not add
   per-target percentages to estimate product workload. Missing a required
   target prevents a complete product from being marked ready.
3. **Review/correct:** show suggestions, source evidence, warnings and explanations.
   Store `final_label` separately from `predicted_label`, with `review_status`
   (`pending`, `confirmed`, `corrected`, or `unresolved`), reviewer identity,
   timestamp, previous/new value and reason/comment. Confirming without changing
   a value still needs an explicit action. Validate edits against the versioned
   approved taxonomy; an unsupported new label stays unresolved pending owner
   approval. A model failure may still receive an explicitly manual classification.
4. **Export:** write final confirmed/corrected labels without silently losing
   input columns/identifiers. Retain suggestions, review status, reason codes and
   model/policy/correction provenance in agreed CSV columns or an audit sidecar.
   Do not substitute an unreviewed suggestion for an unresolved final label.
   Before approval, a completed export requires human confirmation of every
   required target; an explicitly labeled draft export may retain pending/null
   values. Platform stays pending/manual unless Ravi excludes it from required
   export scope. Quarantine resolution needs its existing authorized workflow.

These correction/status/export field names are proposed; Max and Ravi should
settle them before wiring. No automatic retraining, taxonomy mutation or feedback
learning follows from a correction. Log reviewer time, corrections, unresolved
rows and separately sampled final-export accuracy; review itself is not a
measured guarantee of correctness.

### PO approval and application follow-ups

Recommend the conservative reviewer-confirmed pilot in the meeting proposal:
zero automatic acceptance initially; preserve existing experimental recipes
as candidates for fresh qualification. Max can implement the disabled-policy
contract without waiting for arbitrary cutoff choices. Policy activation,
critical-class decisions, required output targets and review service levels need
named approval. No new operational threshold or dashboard behavior is enabled
by publishing this document.

## Maria's evaluation compatibility requirements

Follow the current [prediction contract](https://github.com/import-seb/CatalogIQ-Scope/blob/maria/segment-evaluation-harness/docs/predictions-contract.md)
(reviewed document blob `0a155cee291f05cdb832c84a438330a343eb8915`), including:

- Ship `predictions.csv`, adjacent `manifest.json`, explicit probability mapping,
  and groups either in the CSV or `validation_context.csv` containing
  `record_id,group_id,split,true_label`. Required CSV columns retain relative
  order: `record_id,true_label,predicted_label,confidence,probability_0,...`.
- Bind Segment/Mnfr/age to `segment-final-20261009` and their own labeled subset;
  Brand to `brand_development_allocation_v1_20261029`; Sub-Segment to
  `sub_segment_development_allocation_v1_20261029_r2`. Never compare another
  target's digest to Segment's or silently filter to a shared intersection.
- Hash labeled IDs in prediction CSV order with 8-byte big-endian UTF-8 length
  prefixes. Preserve context truth and frozen group IDs. Reject arriving group
  disagreement rather than overwriting it.
- All 13 core manifest keys are required: `format,target,partition,protocol_id,
  model,config,input_fields,label_mapping,validation_rows,
  validation_record_ids_sha256,probabilities_calibrated,training_seconds,hardware`.
  Also declare union macro-F1, retained-as-errors OOV/count, target allocation,
  protected registry, raw-byte hash normalization and cost scopes. Use null for
  unmeasured single-request cost; batch amortization is a separate measurement.
- Use full vocabulary union truth with `zero_division=0`. Zero-support classes
  remain in the denominator and OOV truth remains an error. Platform's old
  observed-label macro-F1 is not its headline. Probabilistic OOV metrics require
  an explicit treatment; do not discard rows or construct all-zero truth vectors.
- Ship only the ID/group protected registry (12,160 records / 11,979 groups).
  Exclude protected records and groups; never read protected features/labels for
  this integration. Evaluation requires no pickle loading.
- Keep new files LF; preserve existing bytes and declare hash normalization.
  Timestamped split completion/grouping sidecars are local seals; the committed
  portable protocol and exact membership establish cross-machine identity.

Inspection of the current notebook shows legacy behaviors beyond the written
contract: its loader can drop outside-context rows and tolerates up to 1% argmax
disagreement. Do not treat loader acceptance as complete conformance. Perform
strict exact-population/truth/probability checks with the existing owned validator
or equivalent guards, and select target contexts explicitly. Avoid recursive
mixed-target discovery and the notebook's training/setup cells. No full current
harness/calibration/queue rerun is claimed by this handoff.

## Environment and integration completion checks

The existing GPU environment has Python 3.13.7, torch 2.11.0+cu128,
transformers 5.3.0, tokenizers 0.22.2, numpy 2.3.3, pandas 2.3.2, scipy 1.16.2
and sklearn 1.7.2. Lookup needs no GPU. CPU MiniLM is supported by the local
benchmark; deployment GPU requirements/limits still need measurement. Classical
producer versions and the measured recipient differences are in the decision
document. No new dependency is required for this specification.

Before enabling reviewer suggestions, Max should verify artifact hashes and
class order at startup; reproduce saved probabilities on an approved validation
sample without target input columns; check missing/unseen Brand handling,
unavailable Platform, provisional Sub-Segment and partial failures; confirm all
outputs still require review; and measure serving latency/concurrency/memory in
the actual dashboard environment. Select specific artifacts by manifest/hash,
never by newest modification time. Production approval, policy changes and
external publication are separate decisions.
