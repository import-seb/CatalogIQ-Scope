# CatalogIQ inference adapter for Max

The adapter generates suggestions for Segment, Mnfr, TargetAgeGroup,
probabilistic Brand and provisional Sub-Segment. Platform is deferred/unavailable.
Every product and target requires human review.

## Calling it

Use the existing training environment (`requirements-training.txt`; CPU works).
Create one engine per application process and reuse it:

```python
import json
from pathlib import Path
from catalogiq.inference_engine import InferenceEngine
from catalogiq.inference_adapter import infer_request

engine = InferenceEngine(Path("data/processed"))
request = json.loads(Path("docs/integration/inference_adapter_request.json").read_text())
batch_response = infer_request(engine, request)
single_response = infer_request(engine, request["products"][0])  # same batch envelope
```

The [request](inference_adapter_request.json) uses the requested product names
with **synthetic descriptive fields**, not dataset records or ground truth.
The [response](inference_adapter_response.json) is actual CPU inference, including
every class probability. Predictions are observations, not asserted correct
labels. Culturelle's retailer breadcrumb does not create a taxonomy-conflict
reason. Similar listings with different labels establish no rule.

## Provisional schema

Accept a single flat product object or
`{"schema_version":"catalogiq-inference-request-v1","products":[...]}`.
The version may be omitted. Arrays must be nonempty.

| Product field | Contract |
|---|---|
| `record_id` | Required nonblank string, unique within the request; returned exactly in input order. Convert numeric CSV IDs upstream without losing leading zeros. |
| `ProductCategory`, `ProductName`, `ProductBrand`, `ProductDescription`, `ProductContents` | All five keys required. Values are strings or null; blank values use existing preprocessing. Category must be the arriving retailer breadcrumb. |
| `quarantined` | Optional boolean, defaults false; carries existing upstream quarantine. |

Unknown fields are rejected to catch misspellings and exclude curated labels.
Keep upstream source columns separately. Bad envelopes, missing/duplicate IDs
and unknown versions raise `ValueError`; Max maps that to his transport's
validation response. Invalid features fail the affected row explicitly while
valid rows still run.

Responses contain `schema_version`, `engine_version`,
`relationship_registry_version` and ordered `products`. Each product has its
original `record_id`, `review_required=true` and `predictions` keyed by exact
target names. Each target contains:

| Field | Meaning |
|---|---|
| `status`, `selection_status` | `suggestion`/`failed`/`unavailable`; `selected`/`provisional`/`deferred`. |
| `predicted_label`, `confidence`, `raw_confidence` | Saved-vocabulary argmax and maximum raw probability on 0–1 scale. |
| `probabilities`, `label_mapping` | Full `probability_0` through `probability_N` maps; numeric suffix order is class order. |
| `calibrated_probabilities`, `calibrated_confidence` | Unchanged saved-temperature calculation, explicitly research-only. Raw confidence remains primary per the selection handoff. |
| `review_required`, `review_reasons` | Mandatory review and all unique existing primitive/wrapper reason codes. |
| `lookup_status` | Brand `seen`/`missing`/`unseen`; null for MiniLM. Missing/unseen keys use the saved prior and require review. |
| `provenance` | Model/configuration versions, model/manifest/policy hashes, calibration identity/temperature, disabled policy and null operational threshold. |
| `error` | Diagnostic for failures, null on success. Treat as internal diagnostic, not end-user copy. |

Failed/unavailable predictions have null label, confidence and probability values.
Missing dependencies/artifacts produce `model_unavailable`; altered files produce
`artifact_mismatch`; execution/invalid output produces `prediction_failed`.
An execution exception fails that target's valid request rows; other targets
remain available. The existing `review_decision` adds policy, missing-text,
quarantine and applicable approved-relationship reasons.

## Artifacts and safeguards

The [pins](../../src/catalogiq/resources/inference/frozen_routes.json) name every
required file relative to the configured processed directory. Deploy those exact
members preserving relative paths. Validation CSVs, training data, protected-test
files and workstation provenance paths are not consumed. Pins were checked
against the existing selection verification and calibration completion seals.
Models load lazily once per engine; failed loads can retry. Checksums precede
deserialization; state/config/classes are verified again. Inference reuses the
saved five-field formatter, 256-token budget, architecture and tokenizer in eval
and inference mode. Nothing trains, downloads weights or fits calibration.

Calls within one engine are serialized. CPU is default; supply
`FrozenModelLoader(processed_dir, device="cuda")` through the engine's `loader`
argument for GPU use. No deployment throughput/memory SLA is claimed.

Maria's PR #34 registry supplies **only `status == "approved"`** entries to
the primitive; the shipped list is empty. Candidate warn-mode helpers are not
serving gates. Exact `Sub-Segment`, `Segment`, `Brand`, `Mnfr` keys are preserved.
Relationships never rewrite classifications. Policy/temperature files remain
unchanged; experimental thresholds are never used.

`inference_engine.py` owns typed inputs/results and policy orchestration;
`inference_models.py` owns verified loading/forward passes;
`inference_adapter.py` alone owns JSON parsing/formatting. Rename/nest interface
fields there without rewriting inference. Serving preserves the existing
prediction contract's vocabulary, numeric mapping and argmax/confidence
semantics; it does not invent truth labels, groups or evaluation partitions.

## Checks and remaining decisions

```powershell
$env:PYTHONPATH = 'src'
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -v
$env:CATALOGIQ_FROZEN_INFERENCE_TESTS = '1'
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_inference_frozen.py -v
```

Offline tests cover schemas, IDs, alignment, partial failures, review reasons,
approved-only filtering, numeric class order, existing prediction validation and
tiny checkpoint loading/repeated inference. The opt-in test runs all five real
routes on synthetic examples, compares single/batch/repeated probabilities and
rehashes serving members unchanged. It opens no project datasets and does not
rerun the protected assessment.

Awaiting Max: envelope/ID names, transport/error mapping, batch limits, application
lifecycle and deployment paths/resources. Frontend, uploads, corrections, export
and approval persistence belong to his app. No local artifact is missing; Max
must receive the private serving members listed in the pins. Policy activation,
threshold approval and taxonomy approval remain separate team decisions.

See the [validation record](inference_adapter_validation.md) for local results
and the unrelated pre-existing Windows CI failure.
