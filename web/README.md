# CatalogIQ web workspace

Local web interface for cards **#1283** (product input/results), **#1284**
(human review) and **#1285** (export/record tracking). Based on Sebastian's
`feature/int-adpt` commit `9f8c83cd3418aa7cef2fab15c0eebb70f60ad23f`.

## Run the demo

From the repository root, with Python 3.11 or newer:

```powershell
python web/server.py
```

Open **http://127.0.0.1:8765**. No npm install, model download, training data,
API key or GPU is needed for the demo. The static frontend uses native browser
modules. Fonts are bundled, so the application makes no external requests.

The first visit loads the two **synthetic product examples and saved model
responses** in `docs/integration/inference_adapter_{request,response}.json`.
Those model outputs are examples, not verified ground truth. Exact five-field
matches can reuse a saved example; new/modified/quarantined products receive
explicit unavailable predictions with null confidence. No mock random scores
or newly inferred classifications are generated.

## Browser-only online demo on Vercel

Demo: **https://catalogiq-team-b-demo.vercel.app**

The Vercel project is `catalogiq-team-b-demo`. The repository-root `vercel.json`
builds only this static interface. No training data or model artifacts are
included in the public output.

```powershell
npm --prefix web ci
npm --prefix web run build:demo
```

This creates a static `web/dist` with the same interface and an explicit
`static-demo` runtime. `transport.js` handles demo requests in the visitor's
browser; there is no prediction API or upload endpoint in this build. New
products have null predictions, and only exact five-field matches to the two
synthetic examples reuse saved outputs. Each visitor's local storage is separate.

To deploy from the repository root using an authenticated Vercel account with
access to the project:

```powershell
vercel link --project catalogiq-team-b-demo --team <your-team-id>
vercel deploy --prod
```

Use `npm --prefix web run test:static` after building to test the independent
static runtime. Set `CATALOGIQ_DEMO_URL` to test a deployed URL instead. Tests
check that uploaded data stays local and no inference API requests are sent.

The initial deployment is published through the Vercel CLI. Automatic deployments
from GitHub require the repository owner to grant the Vercel GitHub integration
access to the team repository. The CLI deployment does not require that access.

## What works

- Single product entry and UTF-8 CSV upload (2 MB / 250 products per workspace).
- CSV quoted commas/newlines, BOM, CRLF, leading-zero IDs, duplicate-ID and
  malformed-file validation. The downloadable template declares the fields.
- CSV preview shows the first five products before import, with extra columns
  identified. Existing workspace IDs and workspace capacity are checked before
  a prediction request is sent. Invalid replacement files clear the old preview.
- Exact `record_id` alignment; curated labels and extra CSV columns stay in
  the original source and never enter the model request. `quarantined` is
  carried as a boolean when supplied.
- Search/filter, five-target suggestions plus explicit unavailable Platform,
  raw confidence, top alternatives, review reasons and provisional status.
- Approve, correct to a known class, or reject. Corrections/rejections require
  a reason; decisions record reviewer name, timestamp and append-only history
  in the local workspace. A human correction keeps the original model score
  attached to the original prediction, not to the replacement label.
- Unfinished review notes and corrections are saved as per-product drafts in
  this browser. Switching products, filters or refreshing keeps them. Save review
  applies a decision; Discard draft restores the previous decision without
  changing history. JSON exports include drafts separately; CSV final labels
  still come only from saved decisions. Browser storage must be available.
- Local browser persistence, CSV results and JSON audit exports. Original
  source fields, model versions, probabilities, provenance, decisions and
  review history remain separate. Pending/rejected rows have no final labels.
- CSV formula protection. Product-entry source values remain available in
  the JSON audit export. Saved-result imports preserve the source values as
  supplied in that CSV, including any existing formula protection. CSV source
  columns are prefixed `source:` to prevent collisions with generated fields.
- Import saved classification results with their candidate relationship warnings.
  Warnings explain incompatible or missing classifications without changing
  suggestions, review policy or the existing approval/correction controls.
- Desktop and mobile layouts, keyboard focus, native dialogs, labelled
  controls and reduced-motion support.

All current model suggestions require human review. Sub-Segment is provisional;
Platform is deferred. The UI does not introduce confidence thresholds, invent
relationship conflicts, activate candidate rules or modify model policy.
Reviewer names are self-entered attribution, **not authenticated identity**.

## Review Maria's saved classification run

In **Add products → Saved results**, select these three files from the same run:

1. `target_predictions.csv`
2. `relationship_conflicts.csv`
3. The classification `report.json` containing `eligibility_version`,
   `decisions_sha256`, counts and model/registry versions.

The supported export is documented on `evaluation/classify-target` at `0f962ee`,
under `docs/findings/03_modeling/target_classification_20261115/`. The report
declares `target-eligibility-20261116-v2`: 27,870 classified products and 212
quarantined products. Obtain the two CSVs from the run owner; they are not in
Git. The interface does not recreate eligibility or reject an Exclude marker.

Files stay in the browser. The reader checks exact product IDs, supported
model/registry versions, labels, confidence, report counts and candidate warning
completeness. A stale or truncated warning file fails before changing the
workspace. The source files' SHA-256 hashes are recorded in each imported batch.
The cleaner decision hash is retained as declared provenance; the cleaner CSV
is not supplied here, so that hash is not independently recomputed.

Start with **Products with relationship warnings**, or select all classified
products. Choose a batch that fits the remaining workspace capacity (250 total).
Export and clear the workspace before reviewing the next batch. The summary
distinguishes warning rows from distinct products, since a product can have
multiple warnings. The 64 MB per-file limit applies to this saved-run reader;
ordinary product input retains its 2 MB / 250-row limit.

Warnings remain attached to the original model suggestions after a human
correction. Both exports preserve them; JSON also retains the complete review
history. Saved CSVs omit full probability distributions and some source fields;
those values are not reconstructed. These rows are not sent for new inference.

The tests use synthetic fixtures, including a 27,870-product / 651-warning
fixture. **The actual latest CSVs have not been received or verified yet.**
After receiving them, confirm their counts and IDs, inspect examples of each
warning type, and exercise approval, correction, refresh and both exports.

## Connect real models after the handoff

Install the existing training requirements into the project's inference
environment and obtain all files listed in
`src/catalogiq/resources/inference/frozen_routes.json`. Preserve the relative
directory structure. Old prediction-only archives are not sufficient.

```powershell
python web/server.py --live --artifacts "C:\path\to\processed"
```

`--live` checks serving file paths and SHA-256 hashes before starting. The adapter
also verifies their contents when it loads models. A single reused
`InferenceEngine` delegates to the unchanged `infer_request`. The UI sends
the `products` envelope for one or many records. Transport endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | `mode` (`demo` or `live`) and request limits |
| `GET /api/examples` | Synthetic request, saved example response and candidate registry |
| `POST /api/predict` | Adapter JSON request → adapter JSON response |

Do not mix old demo reviews with a real run: export/clear the demo workspace
first. The example loader always marks its records `demo`, even in live mode.
Failures never silently fall back to example predictions.

Use the dependency-free checker before installing or loading the models:

```powershell
python web/check_artifacts.py --list
python web/check_artifacts.py --artifacts "C:\path\to\processed"
```

The current pinned registry requires 36 unique files across five model routes.
The checker reports every missing, changed, unreadable or escaped path, hashes
large checkpoints in chunks, and returns exit code 1 for an incomplete bundle.
`--json` produces a machine-readable report. Artifact hashes use exact bytes;
do not normalize their line endings. A passing file check does not prove that
inference works. Follow [the integration handoff](INTEGRATION_HANDOFF.md) next.

The Python server is a **loopback-only development wrapper**, not a production
deployment. It serves only `web/public`, limits request size, restricts Host and
Origin, and sends a restrictive CSP. Actual inference has not been exercised
on this machine because the serving files have not been received.

## Verification

```powershell
cd web
npm ci
npm test
python -m unittest discover -s tests -p "test_*.py" -v
npx playwright install chromium
npm run test:browser
```

To use an already installed Chrome instead of downloading Chromium:

```powershell
$env:PLAYWRIGHT_CHANNEL = 'chrome'
npm run test:browser
```

Tests cover data alignment, CSV boundary cases and formula handling, review
history, unchanged original predictions, unavailable-model behavior, HTTP
guards, browser imports/reviews/downloads/persistence and mobile layout.
Browser screenshots are generated under the ignored `web/test-results/`.

The suites contain 20 domain/transport/import tests, 12 Python server/artifact tests,
15 local API browser scenarios and 10 static browser scenarios (57 checks).
Browser scenarios include CSV preview and draft recovery as well as
partial model failure and network failure/retry, plus desktop and mobile layouts.
Saved-result scenarios cover informational warnings, review/export persistence,
stale-file rejection, storage failure without data loss and batch selection.
No real-model inference has been run with this interface yet.

## Remaining before live model integration

Receive and verify the pinned serving bundle, exercise live inference and
agree on deployment resources/batch limits. Replace local browser storage with
shared persistence and authenticated reviewers if the team needs shared history.
Policy activation, thresholds and relationship approval belong to the team/PO.
This demo does not close those decisions or mark capstone cards accepted.

## Visual assets

Colors and typography follow the public [Giggso website](https://www.giggso.com/):
navy `#141b29`, cyan `#0aa3c2`, Manrope and Plus Jakarta Sans. The darker button
color `#007f98` provides better text contrast. Font files are local Latin
subsets from Google Fonts; OFL notices are beside them. CatalogIQ's simple
grid/check icon is original. This is a project prototype, not a published
Giggso product.

The Giggso logo is the public navigation asset from
`https://www.giggso.com/assets/Copy%20of%20Full_Logo_3D_Tilted-D6z56a46.png`.
