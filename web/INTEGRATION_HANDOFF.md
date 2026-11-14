# CatalogIQ serving handoff

The UI and local API wrapper are ready to exercise the existing adapter. Actual
model inference remains pending. The Vercel site is a browser-only demo and
cannot load these Python models.

## Required bundle

Use the same commit of `src/catalogiq/resources/inference/frozen_routes.json`
as the web branch. It is the source of truth for paths and SHA-256 hashes.
The current registry `model-selection-20261106-inference-v1` pins 36 files:

| Route | Required files |
| --- | --- |
| Segment | 8: manifest, checkpoint, model/encoder configs, two tokenizer files, policy and temperature |
| Mnfr | 8: same file roles |
| TargetAgeGroup | 8: same file roles |
| Sub-Segment | 8: same file roles; model remains provisional |
| Brand | 4: manifest, lookup model, policy and temperature |

Platform remains unavailable. Do not send training or reserved-test data for UI
integration. A predictions-only export is not a serving bundle. Preserve the
registry's relative paths and original file bytes, including line endings.

Generate the exact inventory without loading any models:

```powershell
python web/check_artifacts.py --list
```

If a serving artifact has changed, coordinate an explicit registry update with
the adapter owner instead of replacing a hash just to get a passing check.

## Receive and verify

1. Put the bundle under a local directory outside public web assets.
2. Run `python web/check_artifacts.py --artifacts "C:\path\to\processed"`.
3. Resolve every missing or mismatched file with the adapter owner. This command
   only reads files and needs the Python standard library.
4. Use the project's inference environment and run
   `python web/server.py --live --artifacts "C:\path\to\processed"`.
5. Use a fresh browser workspace, or export and clear the demo records first.

File verification is one prerequisite. Model imports, compatible dependencies,
runtime memory and prediction behavior must still be checked in the real run.

## Real-model acceptance run

- Enter a new authorized product and upload a small CSV batch. Confirm that each
  response keeps the exact `record_id`, including leading zeros.
- Verify all six target entries and explicit unavailable/failed states. Platform
  stays unavailable; do not invent scores or enable automatic approval.
- Compare the UI's labels, confidence, alternatives and review reasons with the
  actual adapter response. Do not use the saved example labels as ground truth.
- Correct one available classification and save the reviewer and reason. Confirm
  the original model output remains intact. Refresh, then inspect both exports.
- Check a model failure and recovery; no failed or partial run may silently use
  saved demo predictions or replace results belonging to another product.
- Record the commit, registry version, hardware, startup time and single-product
  and small-batch latency. Agree on public inference hosting and batch limits
  before connecting the public site.

Share implementation evidence for cards #1283, #1284 and #1285. Team review and
PO acceptance remain separate from passing technical checks.
