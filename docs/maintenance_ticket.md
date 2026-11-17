# Model maintenance ticket: audit and engineering acceptance

Original audit: November 14, 2026. Publication update: November 15, 2026. Scope: existing finished models, versioned offline maintenance
engineering and synthetic acceptance tests. No new model research, new labeled
data, finished-model training, protected-test access, deployment or application
changes were performed. This supplements the capstone scope decision in
[decisions.md](decisions.md#2026-11-14-maintenance-scope-and-confirmation-collection).
It does not reopen the deferred model-quality experiments.

## Audit scope and ownership

The working branch is `experiment/frozen-minilm-sgd-20261113`; it predates the
inference integration. Read-only GitHub inspection found:

- Main at `de2f03be61de8f8e87cc141861fa12152c8b9324` contains the inference adapter.
- Adapter branch at `581a57458039fab74ba70f682ee9a7544ee8bde7` has the same inspected
  engine/model/adapter Git blobs as current main. Its 14 offline tests were run
  unmodified here. [Adapter PR #35](https://github.com/import-seb/CatalogIQ-Scope/pull/35)
  documents the prior actual frozen-artifact integration run; that run was not
  repeated for this ticket.
- Max's web branch at `586f6faed5c5aa951be29f1d2a0717a65c7af8fb` supplies upload,
  review/history and exports. [Web PR #36](https://github.com/import-seb/CatalogIQ-Scope/pull/36)
  explicitly distinguishes browser-local demo storage from shared authenticated
  storage and live inference integration.

For the original audit, reviewed source snapshots were retained only in gitignored evidence, with commit paths,
Git blob IDs and SHA-256 hashes. They are test fixtures, not another application
or an installation. That audit did not switch, fetch, merge or push a branch. The publication update
uses a dedicated feature worktree based on current main and imports its adapter
directly; private source snapshots are not required or included. No message was sent to Max; the integration contract below is ready
for his review. His application and the existing adapter were not edited.

## Acceptance audit

Final PASS means the specified capability has an implementation and engineering
demonstration. It does not mean production rollout, scientific qualification of
an updater, or completed shared application integration.

| Criterion | Initial assessment | Final engineering assessment | Evidence |
|---|---|---|---|
| 1. Identifiable dataset and model versions | **PASS** for existing models: sealed source/split identities, checkpoint hashes, configurations and ordered vocabularies already exist. | **PASS**, extended to new ledger datasets and parent-linked maintenance versions. | Existing sealed source/bundle manifests; `minilm_prediction_export.selected_source`; `deltas.validate_binding`; `import_frozen_versions`, `register_model`, `Store.read`; `existing_versions.json`. |
| 2. Approved new data can be prepared and used in a documented update procedure | **PARTIAL**: ledger, target policies and updaters existed, but there was no tested end-to-end lifecycle from a frozen confirmation export to a candidate, decision and rollback. | **PASS** for the controlled offline workflow, demonstrated with synthetic labels/features only. | `prepare_approved_data`, `update_rows`, existing `methods.update_head`, `record_evaluation`, `decide_candidate`, `select_models`; `ticket_demo.demonstrate`; rejection/holdout tests. |
| 3. New batches use the appropriate version and identify it | **PARTIAL** in this checkout. Upstream already implements batch classification against pinned selections; the local branch lacks those modules and had no lifecycle acceptance test binding a selected version to stored results. | **PASS** by delegating to the unchanged upstream adapter and checking its returned identity, artifact, vocabulary and record alignment. | Upstream `InferenceEngine.predict`, `FrozenModelLoader.describe/load`, `infer_request`; new `fixed_adapter`, `classify_batch`; original 14 adapter tests plus version-mismatch tests. |
| 4. Historical stored predictions and approvals survive version changes | **PARTIAL**: Max's `recordDecision` preserves the original result and appends history; exports include model IDs. Browser-local workspace storage alone was not a durable immutable archive or a demonstrated A/B version-change test. | **PASS** in the isolated content-addressed archive. A predictions/approval bundles remain unchanged after new datasets, B results, corrections and rollback. | `Store.put/read`, `save_review`; `ticket_demo.demonstrate`; same-ID/different-version, corruption and history tests; four checks using Max's original domain functions. |

### Existing code and tests supporting the audit

- [`src/catalogiq/resources/segment_final_20261009/manifest.json`](../src/catalogiq/resources/segment_final_20261009/manifest.json): source/protocol/grouping/split identities and hashes.
- [`minilm_prediction_export.selected_source`](../src/catalogiq/minilm_prediction_export.py): explicit reviewed selections, not newest-file discovery. This is an evaluation exporter, not a serving API; its development preparation must not be used for arbitrary new batches.
- [`prediction_contract.ordered_id_hash`, `validate_manifest`](../src/catalogiq/prediction_contract.py): ordered record identities and prediction-package validation. Existing tests: `tests/test_prediction_contract.py`, `tests/test_minilm_prediction_export.py`.
- [`segment_transformer.train_transformer`](../src/catalogiq/segment_transformer.py) and [`target_minilm_experiment.run_experiment`](../src/catalogiq/target_minilm_experiment.py): existing training paths with configuration/checkpoint metadata. They were inspected, not called.
- [`correction_ledger.Ledger`](../experiments/maintenance_research/correction_ledger.py): versioned events, idempotency, conflicts, revocations, eligibility and export; its existing 16 synthetic tests passed again.
- [`methods.update_head`](../experiments/maintenance_research/methods.py), [`maintenance_offline.compare_records`](../experiments/maintenance_research/maintenance_offline.py), [`deltas.validate_binding/load_delta`](../experiments/maintenance_research/deltas.py): reuse fixed head updaters, paired metrics and checked delta binding rather than a new training/inference implementation. The delta loader was not invoked on finished models here.
- [Upstream engine](https://github.com/import-seb/CatalogIQ-Scope/blob/581a57458039fab74ba70f682ee9a7544ee8bde7/src/catalogiq/inference_engine.py), [loader](https://github.com/import-seb/CatalogIQ-Scope/blob/581a57458039fab74ba70f682ee9a7544ee8bde7/src/catalogiq/inference_models.py), [adapter](https://github.com/import-seb/CatalogIQ-Scope/blob/581a57458039fab74ba70f682ee9a7544ee8bde7/src/catalogiq/inference_adapter.py), [original tests](https://github.com/import-seb/CatalogIQ-Scope/blob/581a57458039fab74ba70f682ee9a7544ee8bde7/tests/test_inference_adapter.py).
- [Max's `domain.js`](https://github.com/import-seb/CatalogIQ-Scope/blob/586f6faed5c5aa951be29f1d2a0717a65c7af8fb/web/public/domain.js): `recordDecision`, `exportCSV`, `auditExport`; [server](https://github.com/import-seb/CatalogIQ-Scope/blob/586f6faed5c5aa951be29f1d2a0717a65c7af8fb/web/server.py) delegates `/api/predict` to `infer_request`. It has no durable server-side result archive. The public demo does not classify arbitrary products with live models.

## Version identification and lineage

The five frozen routes are unchanged:

| Target | Original model identity | Labeled training records / allocation |
|---|---|---|
| Segment | `segment_category_ablation_256_20261010/with_category` | 55,776 / frozen V4 |
| Mnfr | `minilm-mnfr-category256-c3458037a670c8f0` | 55,233 / frozen V4 |
| TargetAgeGroup | `minilm-targetagegroup-category256-48256ce8adfceee1` | 54,282 / frozen V4 |
| Sub-Segment | `minilm-sub_segment-category256-8a1e8f3e742eb3a7` | 53,956 / Sub-Segment v1 r2 |
| Brand | `brand-productbrand-empirical-prior-v1-8cbdb95a5638dbed` | 58,888 / Brand v1 |

The metadata-only import binds each to its exact artifact hash, serving-manifest
hash, ordered training-ID hash, source completion hash, allocation, saved config,
input contract and vocabulary. The [existing handoff](integration/model_selection_handoff_20261106.md)
retains full legacy paths/checkpoint hashes. Neural inputs remain five fields,
256 tokens and attention-mask mean pooling. Brand uses its saved lookup config.

New objects use `kind-<full SHA-256>` identities computed from canonical metadata
and payload hashes. Dataset versions bind the ledger head, registry/catalog,
explicit group roles and parent models/datasets. Model versions bind the dataset,
parent model, method, config/input contract, ordered vocabulary and artifact hash.
Results bind the exact model/dataset map, request hash and selection receipt.
Approval/correction objects reference the result and any superseded review.
Changing any content produces a different version; file modification is detected
on read. Legacy source datasets and weights are referenced, never copied into
this registry or modified.

## Controlled maintenance procedure

This is the documented future procedure. Actual new labels, real candidate fitting
and production promotion remain deferred; only its synthetic engineering path was
executed for this ticket.

1. **Record and approve:** use the existing ledger with authenticated caller roles
   supplied by the eventual service. Capture all five raw fields, canonical
   entity/record/group, reviewer, model/ontology/source/time and evidence. Resolve
   conflicts/revocations. Steward-owned membership and reserved-data-exclusion
   attestations are required; absence from a known training list proves no novelty.
2. **Freeze preparation:** call `prepare_approved_data(store, ledger, roles,
   model_versions, synthetic=False)`. Roles map whole groups to update, replay or
   evaluation before fitting. Synthetic/real data cannot mix. Validation,
   unresolved, unverified-new and unknown-vocabulary entries are excluded by the
   ledger. Reconfirmations do not earn extra product votes. Feature duplicates
   crossing roles are rejected, as is seen-training data assigned to new evaluation.
3. **Select the target policy:** Segment SGD; Mnfr cautious PA-II/SGD; age conditional
   SGD/head AdamW; Sub-Segment memory/adjudication with parameter updates disabled;
   Brand anchored count overlays. Preserve the existing strategy's evidence and
   class safeguards. Insufficient evidence means keep the current model. LoRA and
   full-model retraining are not enabled by this ticket.
4. **Fit a separately authorized candidate:** `update_rows` returns only update/replay
   records. For neural heads, obtain verified frozen-encoder features using the
   existing runtime/cache contract, warm-start the current accepted head and call
   the existing `methods.update_head` under a fresh resource supervisor. For Brand,
   use `BrandOverlay` with trusted original contribution bindings. Preserve the
   current artifact and write a new delta. Never pass evaluation-role rows to fitting.
5. **Version and validate:** register each order's candidate with `register_model`;
   preserve parent/data/method/config and saved payload. Use `record_evaluation`
   for paired identical historical and incoming evaluation cohorts; keep truth and
   populations frozen. Evaluate shuffle, class blocks and reverse class blocks.
   `decide_candidate` accepts experimental use only when all three order gates pass
   for the same parent/data/config/method/cohort; any failed gate can be recorded
   as a rejection and cannot qualify the candidate. Archive historical retention,
   per-class audits and independent incoming evidence under the approved protocol.
6. **Review and retain:** an operator explicitly records an offline selection using
   `select_models`. Nothing switches the application's loader. Experimental
   acceptance is not production qualification, PO approval or a release. Existing
   `deltas.validate_binding` remains required before loading actual maintenance
   artifacts. Sub-Segment remains blocked; small synthetic passes cannot waive its
   retention failures or any real-data eligibility requirement.
7. **Rollback:** append a selection referencing the previous accepted selection
   through `rollback_to`; verify replay against the old bundle. Never overwrite
   model files or reassign old predictions to the new version. Revoked confirmations
   require reviewing dependent candidates; deleting memory does not untrain a model.

The original dataset guards and protected-test status remain unchanged. A future
real workflow needs its own authorized input allowlist and budget; no protected
identity registry is needed or opened by this engineering demonstration.

## Synthetic demonstration and verification

New implementation: [`ticket_lifecycle.py`](../experiments/maintenance_research/ticket_lifecycle.py).
Executable example: [`ticket_demo.py`](../experiments/maintenance_research/ticket_demo.py).
Tests: [`test_ticket.py`](../experiments/maintenance_research/test_ticket.py) and
[`ticket_web_test.mjs`](../experiments/maintenance_research/ticket_web_test.mjs).

The demo creates a synthetic five-target ledger with 16 distinct groups, 12 for
updates and four for evaluation. Its toy two-dimensional heads and two-label
vocabularies are explicitly not MiniLM models. It uses the original SGD updater
in three fixed orders; Brand synthetic routing uses the existing probability
counter utility. The actual upstream engine and adapter run with a dependency-
injected synthetic loader, so their request, review and provenance behavior is
tested without loading finished checkpoints. The upstream tiny-checkpoint test
also uses only generated toy weights and performs forward inference without
optimizer updates.

For the required A/B sequence, the demo saves A predictions and an explicit
synthetic human approval, records a new prepared dataset and B candidate, validates
it, selects B offline, and classifies a new batch. It rehashes every prior stored
file, checks both results' version maps, then rolls back and reproduces A's
prediction bytes exactly. Additional tests repeat a product ID across versions,
reject altered payloads/manifests and mismatched served versions, prevent approval
of unavailable outputs, preserve superseded approvals, and exercise rejected
candidates. Max's original JS functions separately preserve A's export bytes and
approval history and retain A/B model references in CSV exports.

Evidence directory: `data/processed/maintenance_ticket_20261114_v1/` (gitignored).
`ticket_session.json` points there. Successful runs and the final verification
identify exact source hashes; failed preliminary attempts remain recorded.

| Final check | Result / saved evidence |
|---|---|
| Python suites | **45 passed**: 15 lifecycle tests, 16 existing ledger/strategy tests, 14 unchanged upstream adapter tests. Final log (`../data/processed/maintenance_ticket_20261114_v1/run_005/worker.log`, private local evidence). |
| Max's unchanged JS domain functions | **Four scenarios passed**: export preservation, approval/history preservation, A/B version references and exclusive output creation. Web result (`../data/processed/maintenance_ticket_20261114_v1/run_004/result.json`, private local evidence). |
| Required A/B demonstration | **35 prior stored files unchanged**, correct A/B model maps and exact prediction replay after rollback. Demo evidence (`../data/processed/maintenance_ticket_20261114_v1/run_005/engineering_demo.json`, private local evidence). |
| Existing-model version index | Five metadata-only model/data references imported. Index (`../data/processed/maintenance_ticket_20261114_v1/run_005/existing_versions.json`, private local evidence). |
| Resources | No limit breach. Peak worker job commit across attempts: **1.163 GiB**. Final Python worker: **44.42 s wall / 32.14 CPU s**. Resource measurements (`../data/processed/maintenance_ticket_20261114_v1/run_005/resources.json`, private local evidence). |
| Preservation | Original tracked files (including the pre-existing decision-log edit), model artifacts, development data, environment metadata/configuration, overlay files and prior research sources/documents match the pre-ticket snapshot. Checks (`../data/processed/maintenance_ticket_20261114_v1/run_005/preservation.json`, private local evidence). |

The final verification seal (`../data/processed/maintenance_ticket_20261114_v1/FINAL_VERIFICATION.json`, private local evidence)
binds code, documents, source fixtures and evidence. Existing full-repository CI
results are not claimed: the focused suites above were run, with protected-data
access denied. The immutable approved reviews in this demonstration are explicitly
synthetic, not real reviewer confirmations.

## Reproduce the packaged checks

The published subset replaces workstation-specific runners with a fresh-output
Windows supervisor and a cross-platform synthetic suite. It uses the current
interpreter and main's existing adapter directly, with no private JSON inventories,
source caches, data, models or session files. Follow the [portable README](../experiments/maintenance_research/README.md).
The [November 15 verification summary](maintenance_verification.md) records the
fresh source-only run in addition to the historical audit evidence above.

```powershell
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s experiments/maintenance_research -p 'test*.py' -v
python -B experiments/maintenance_research/ticket_run.py --output "$env:TEMP/ciqm-check-1"
```

The Windows supervisor enforces 600 seconds, 2,400 CPU seconds, four-CPU affinity,
at most four numerical threads, one interop thread, a hard 4-GiB job commit limit
and sampled 4-GiB RSS stop. CUDA initialization, network/child-process use and
out-of-output writes are denied in the Python worker; its data allowlist is empty.
The optional web check receives Max's existing module via `--web-domain`; it never
downloads or changes his code. Existing output directories are refused. The CI
also runs the maintenance unit suite on Ubuntu and Windows with existing dependencies.
No dependencies were installed for the publication checks.

Routine preliminary issues were confined to the harness: deep Windows evidence
paths exceeded normal file access limits, and Node 24 rejected an obsolete module
flag. Evidence was moved to the verified shorter workspace path, stores now refuse
unsafe path lengths before publication, and Node uses Max's original `type=module`
package metadata. No safety check or resource limit was weakened.

## Max integration contract and remaining work

See [the dedicated integration instructions](integration/maintenance_versions.md).
The shared-storage service, authenticated operator/reviewer endpoints, UI version
selector and live-hosting acceptance remain Max/team integration work. They were
not silently added to his components. All four engineering criteria have offline
implementations and tests; production integration and model-quality qualification
remain explicitly incomplete. No model update was published and no capstone
claim of new-product generalization follows.
