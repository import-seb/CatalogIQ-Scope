# Predictions contract alignment: Sebastian's handoff

Reviewed Maria's remote
[predictions contract](https://github.com/import-seb/CatalogIQ-Scope/blob/maria/segment-evaluation-harness/docs/predictions-contract.md),
particularly Sections 2 and 3, on October 31, 2026. Reviewed Git **blob**:
`662bea5144c3fbf13ecf04f7e42c15eac614df44`. This is the document blob, not a
harness commit. The older local tracking branch did not contain this document.
Maria's harness and Max's TF-IDF code/artifacts were not modified.

## Audit before implementation

| Requirement | Existing owned exports | Necessary change |
|---|---|---|
| Thirteen core manifest keys | All five MiniLM exports complete; probabilistic Brand lookup missing two | Derive lookup `protocol_id` from its actual split identity and `config` from its saved estimator |
| Explicit numeric probability mapping | Present in every manifest and label_mapping.json | Preserve and validate through all 143 Brand classes |
| Unique IDs, ordered validation fingerprint, correct truth and probabilities | Already verified; rechecked against the exact approved validation populations | No prediction, label, order or population changes |
| Product groups in the accepted context format | Every run already has record_id, group_id, split, true_label in validation_context.csv | Preserve this supported format; no extra prediction column needed |
| Macro class set / OOV declarations | Missing declarations; every current export has zero OOV truth rows | Declare model_vocabulary_union_truth and retained_as_errors, with actual row counts |
| Target-specific allocation | Detailed source identity already recorded | Add the explicit split_allocation_target name and sealed assignment identity |
| Protected test identities | Present in frozen and allocation sources, absent from the portable prediction packages | Export two-column ID/group registries and verify against the committed original V4 digests |
| Cost names/scopes and explicit null request latency | Already present in all six; lookup lacked a separate amortized-cost scope | Add only that scope from its recorded inference scope; preserve every measured value |
| Source hash normalization | Raw-byte hashes existed but the declaration was absent | State raw_bytes and distinguish semantic fingerprints / historical LF-normalized algorithm seals |

## Delivered artifacts

The revised, self-contained handoff is
`data/processed/predictions_contract_v2_20261031/`, with a sibling ZIP of the same
name. Original exports remain in `minilm_validation_exports_v1/`, byte-for-byte
unchanged, so their historical seals and presentation evidence remain valid.
The corrected lookup manifest is under
`predictions_contract_v2_20261031/brand_lookup_probabilistic_v1/manifest.json`.

The root and every run contain `protected_test_identity_registry.csv`, with
**exactly `record_id,group_id`**. It holds **12,160 records in 11,979 groups**.
Membership and grouping match both the original finalization registry and the
portable V4 assignments, as well as both target-specific allocation copies.
The committed canonical test fingerprints are:

- IDs: `443b96d4028166b9eb0a254aa42c767f35a86f7448c7d009ad1abe73420547bf`.
- ID/group/test-role assignments:
  `69e046b4813bcd15336dd20165cfc7af3715bf4c2571103df0a1e83458cd6d62`.

Those are canonical identity fingerprints, not the raw CSV file hash. Each
manifest separately records the raw hash of its new two-column registry.
The delivered registry's raw SHA-256 is
`48b578ec7289628a1713625942e0e456d8804c693d397b7656506ebd32070bdf`.
Protected labels/features and test.csv were not read, hashed or exported.
No partition was regenerated or reallocated.

Each run keeps its existing prediction, context, mapping, metric and historical
compatibility files unchanged. Revised manifests declare the union class set,
OOV policy/count, exact allocation, raw-byte hash convention and registry path.
All 13 core fields are present. New metadata/registry files use LF; existing
CRLF data, if supplied, is copied unchanged rather than normalized.

Lookup config comes from model.json: estimator version, saved smoothing
(alpha=1, empirical training class prior), existing normalizer/missing tokens
and the implemented numpy.argmax label-order tie rule. No transformer settings,
new fitted parameter or invented measurement is substituted for lookup config.

## Code and validation

`catalogiq.prediction_contract` provides the read-only audit, metadata alignment,
identity export and validation. The MiniLM and probabilistic lookup exporters
also add the declarations and per-run registry for future exports. The frozen
trainer, model parameters and allocation implementations are unchanged.

The metadata command performs no model fitting or inference:

```powershell
$env:PYTHONPATH = 'src'
.venv-gpu/Scripts/python.exe -m catalogiq.prediction_contract `
  --source-root data/processed/minilm_validation_exports_v1 `
  --output-root data/processed/predictions_contract_v2_20261031
```

Use a fresh output directory for a later repetition. The command refuses an
existing directory/archive and verifies all consumed inputs again before
publishing. It selects the six owned runs explicitly, without recursively
discovering Max's exports. `verification_report.json` records checks and source
hashes; `completion.json` seals the delivered files.

**48 focused tests passed**: 21 new contract/registry/package tests, 17 MiniLM
export tests and 10 probabilistic lookup tests. They cover missing core/policy
fields, actual saved lookup settings, numeric/Unicode class mappings, OOV truth
retention and union F1, row-order fingerprints, wrong contexts, protected IDs
and groups, resealed-registry tampering, two-column projection, cost scopes,
CRLF preservation, staging cleanup and original-source immutability. Tests use
synthetic data; the existing tiny saved-state inference test does not optimize
weights or load any selected project checkpoint.

```powershell
$env:PYTHONPATH = 'src;tests'
.venv-gpu/Scripts/python.exe -m unittest `
  test_prediction_contract test_minilm_prediction_export `
  test_brand_probabilistic_lookup -v
```

Real-artifact validation independently reconciles the five labeled populations:
Segment 11,952; Mnfr 11,812; TargetAgeGroup 11,624; Brand 12,619; Sub-Segment
11,562. The Brand lookup uses the same 12,619 ordered IDs/truth as its original
export. All current OOV counts are zero, so the union convention equals the
existing full-vocabulary convention. Scores, costs and populations were not
changed. Observed-label figures remain separately labeled historical figures.
The real handoff reverified 59 consumed source files unchanged. An independent
check verified all archive seals and the 13 frozen algorithm source hashes.
The ZIP SHA-256 is
`15170a1b102117719416aebed9c6ebda43afd1efdfe86517f7af45269f450ea7`.

## Maria's attention

- Use the revised manifests and their local context/registry together. Brand
  uses `brand_development_allocation_v1_20261029`; Sub-Segment uses
  `sub_segment_development_allocation_v1_20261029_r2`. The other three retain
  `segment-final-20261009` and their own target-specific labeled subsets.
- Headline macro-F1 uses model vocabulary union truth. Keep an arriving
  group_id check rather than silently overwriting it with the registry value.
- OOV truth must remain an error; Brier/calibration code needs an explicit OOV
  treatment rather than class-index lookup failure or silent row dropping.
  This is not exercised by the six current zero-OOV packages.
- Existing compatibility reports concern the previously pinned loader. This
  metadata alignment does not claim a new full harness execution, complete
  comparative evaluation, independent calibration or production selection.
- CPU lookup and GPU MiniLM batch costs have different hardware/scopes;
  single-request latency is still unmeasured. No direct throughput ratio or
  operational confidence threshold is established.
