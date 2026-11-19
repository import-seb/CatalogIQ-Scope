# Versioned maintenance integration for Max

Owner handoff, November 14, 2026. The [ticket audit](../maintenance_ticket.md)
records inspected commits, tests and scope. No changes were made to your web app,
server, inference adapter or model files. This contract is ready for owner review;
no message, push, merge or deployment was performed.

## Reuse the existing application and adapter

Keep the current `InferenceEngine`, `FrozenModelLoader`, `infer_request`,
`recordDecision`, `exportCSV` and `auditExport` implementations. Do not send dataset
labels to `/api/predict`, replace the five-field input contract, or turn a stored
confirmation into automatic approval. Platform remains unavailable and Sub-Segment
provisional. Raw probabilities and disabled operational policies remain unchanged.

The new utilities are isolated in `experiments/maintenance_research/ticket_lifecycle.py`.
They supply storage/lineage checks and delegate classification; they do not supply
another neural implementation, application server or UI.

## Backend integration seam

1. **Initialize a private store.** Use an empty, short configured directory, outside
   public assets and original artifact/dataset paths. `Store(path)` refuses an
   unrelated existing directory. Keep the OS/service account as the only writer.
   The local prototype trusts caller identities; authentication and authorization
   must be supplied by the service before real ingestion or operator actions.
2. **Import the reviewed frozen versions.** `import_frozen_versions` creates
   metadata-only references from the exact `frozen_routes.json`, selected training
   metadata and probabilistic Brand manifest. The private serving members remain
   at your existing configured path and are verified by `FrozenModelLoader`.
   Do not choose the newest checkpoint or edit pins to suppress a mismatch.
3. **Select explicitly.** Record an operator-selected five-target version map with
   `select_models`. This creates an offline selection receipt; it does not change
   your app's process. For a request, capture one selection once and reuse it for
   the entire batch. Concurrent changes affect later batches only. Explicit maps
   are supported for offline experiments; keep that route unavailable to ordinary
   public clients and do not present experimental selection as production approval.
4. **Delegate and archive.** Bind the already-loaded engine with
   `fixed_adapter(engine, infer_request)`, then call `classify_batch`. It preserves
   the existing request/response schema and stores an immutable result bundle.
   It refuses a loader whose model ID/artifact differs from the selected map,
   unexpected IDs/vocabularies, stale suggestions attached to failed statuses,
   or disabled human-review flags. Limits match the current wrapper: 250 records
   and 2 MiB per request.
5. **Return a separate storage reference.** Keep the inference envelope unchanged.
   An application-owned wrapper can return `result_id` and `selection_id` alongside
   `response`, with per-target content-addressed `model_versions` and
   `dataset_versions`. These additions are a proposed transport seam for your
   approval; the original adapter rejects unknown request fields, so do not add
   version IDs inside its current product objects. Preserve original
   `provenance.model_id` and `artifact_sha256` as well.
6. **Store decisions separately.** Continue validating reviews with `recordDecision`.
   Archive approved/corrected labels through `save_review(result_id, record_id,
   labels, reviewer=..., reason=..., previous_review=...)`. This storage seam
   requires a reviewer and rationale, all five supported targets and valid available
   labels. A rejected or deferred review belongs in the existing ledger/audit path,
   not as an approved training label. Never rewrite the archived inference JSON.
   Preserve your original review history and exact JSON export too; CSV export is
   a presentation format and its formula escaping is not a raw training-data source.

Illustrative backend call (after the upstream adapter and store utilities are
available in the integrated checkout):

```python
from catalogiq.inference_adapter import infer_request
# ticket_lifecycle is currently an isolated experiment module, not an installed API.
from ticket_lifecycle import Store, fixed_adapter, classify_batch

store = Store(private_store_path)
result_id = classify_batch(store, request, fixed_adapter(engine, infer_request),
                           selection_id=operator_selected_receipt)
response_bytes = store.payload(result_id, "predictions.json")
# Keep result_id attached to the UI record and every later review/export.
```

`fixed_adapter` supports the existing frozen loader's pinned routes. A future
accepted head/lookup delta needs an explicitly coordinated loader implementation
using the checked base/target/vocabulary/payload bindings. Selecting a B reference
against the unchanged A loader fails closed; there is no hidden fallback or claim
that the fixed loader already serves arbitrary updates. The synthetic B acceptance
uses the engine's existing loader-injection interface to demonstrate that seam.

## Prepare corrections and review candidates

Map an authenticated app decision to per-target ledger events with a stable event
ID, canonical entity/group, exact five inputs, model reference, ontology, reviewer,
timestamps, verification flags and source evidence. The current browser export
does not establish trusted group identity, original training exposure or reserved-
data exclusion. Require the steward's registry/attestation; do not infer novelty
from import time, a new CSV row ID, SKU alone or a confidence score.

Use `prepare_approved_data` to freeze the ledger head and steward-assigned roles.
This exports approved eligible confirmations and preserves old dataset versions.
`update_rows` excludes evaluation roles. Apply only the target-specific method
and approved validation protocol; no automatic trainer or scheduler is added.
The ticket's only actual fitting was synthetic toy-head engineering.

An experimental acceptance receipt requires passing reports for the same
parent/data/method/config and evaluation cohort across all three declared orders.
It is not production qualification. Production promotion needs the team's explicit
approval, appropriate real-data evidence, per-class review and the coordinated
loader. No researched delta is enabled by this handoff.

## Preserve historical results

Use immutable `result_id` plus exact `record_id` as the review reference. Product
IDs may recur across batches; they cannot replace a previous batch's results.
Store the request, full response, five-target model/dataset map and selection
receipt together. A new model/data version creates new objects. Corrections append
new review objects referencing the prior review. Read old results directly; never
recompute them using the current model to render history.

The prototype publishes complete content-addressed directories atomically,
verifies manifest/payload hashes and refuses inconsistent existing objects.
Selection changes are append-only, operator initiated and protected by a local
exclusive lock. A crash can leave a staging directory or stale lock; fail closed
and inspect it, rather than automatically deleting a lock another process may own.
Rollback appends a reference to a previous selection; it does not modify any old
prediction, approval, dataset or model bundle.

For shared deployment, agree on authenticated roles, cross-host transaction/locking
semantics, backups, retention/privacy handling and externally anchored audit hashes.
This filesystem prototype is suitable for isolated engineering acceptance; it is
not a distributed storage service or protection against a malicious file owner.
Hash verification detects accidental external edits when objects are read. These
deployment decisions remain owner work, not undocumented changes to your app.

## Owner acceptance checklist

- Verify the exact upstream revision, 36 serving pins and existing artifact checker.
- Bind the frozen-version index and confirm a request selects one snapshot.
- Send a synthetic single product and CSV batch; preserve IDs, schema and mandatory review.
- Confirm every returned model ID/hash matches the archived version map.
- Archive A results and an approval; introduce an isolated simulated B and new batch.
- Verify byte-for-byte A preservation, including approvals and same-ID repeat cases.
- Attempt a wrong-version loader and confirm rejection with no result publication.
- Append a correction and a rollback; verify previous history remains accessible.
- Keep demo/browser-only behavior labeled; do not claim shared persistence or live
  updated-model inference until the owner integration is actually exercised.

Engineering evidence already demonstrates these storage/adapter boundaries on
synthetic data and Max's unmodified export functions. Production wiring, service
authentication and release acceptance are not claimed complete.
