# Shareable maintenance verification

November 15, 2026. This summary covers the source-only publication prepared from
main `de2f03be61de8f8e87cc141861fa12152c8b9324`. It supplements the
[original audit](maintenance_ticket.md); no private dataset, checkpoint, embedding,
model inventory, upstream source cache, log or session pointer is included.

## Fresh checks before push

| Check | Observed result |
|---|---|
| Reusable maintenance unit tests | **39 passed**: six safety/binding cases, 16 ledger/strategy cases and 17 lifecycle cases. |
| Existing main inference-adapter tests | **14 passed**, unchanged. They use generated data/tiny weights, not finished models. |
| Total Python checks | **53 passed** in the isolated source-only worktree. |
| A/B update demonstration | Three fixed-order SGD updates of a toy head; explicit experimental validation/selection; version-B batch; rollback to A. No finished model training. |
| Historical bytes | **35 previously stored files unchanged**, including A's predictions and synthetic approval. Both result sets retain their model/data/selection references. Rollback reproduces A prediction bytes exactly. |
| Max's original JS module | **Four scenarios passed**: original export bytes, approval/history preservation, A/B model references and exclusive export creation. Module was supplied read-only from his inspected branch. |
| CPU/CUDA | Four-CPU affinity, at most four numerical threads, one interop thread; **CUDA remained uninitialized**. |
| Runtime/resources | **55.67 seconds wall**, **22.58 CPU seconds**, peak job commit **1.163 GiB**; limits 600 seconds, 2,400 CPU seconds and 4 GiB. Both Python and Node exited 0; no limit termination or active child remained. |
| Data and application boundaries | Python data allowlist empty; no protected data, original models, application components or dependencies changed. Source hashes matched before/after the run. |

The process used the existing Python inference/training environment and Node
24.21.0; nothing was installed. The optional `psutil` sampler is not a requirement:
a test verifies that its absence yields an explicit unavailable measurement.
Independent Windows job limits still apply. The toy save/reload uses the existing
`torch`/`safetensors` requirements, with no pretrained downloads.

## What the code checks

- `test_ticket.py`: content-derived version IDs and lineage for all five routes;
  frozen metadata references; immutable data snapshots; evaluation-role exclusion;
  rejected/unsafe candidates; exact version/probability/ID alignment; wrong-version
  loader refusal; stale failed-response rejection; same product IDs across model
  versions; preservation of approved history; lock cleanup and external tampering.
- `test_strategy.py`: all five targets, idempotency, conflicts/adjudication/reopening,
  revocations, synthetic/unknown/validation/identity/ontology quarantine,
  mandatory human review, strict paired evaluation, group-atomic cohort planning,
  Brand count reversibility and untouched-key/fallback invariance.
- `tests.py`: full-input fingerprints, whole-group replay separation, ordered-ID
  hashing, explicit resource/scope constants, GTIN/SKU limits and wrong-base/path
  traversal rejection in maintenance-delta bindings.
- `ticket_demo.demonstrate`: the complete requested A-save, B-introduce, B-classify,
  A-byte-preservation and rollback sequence using the actual existing adapter with
  its synthetic loader-injection seam.
- `ticket_web_test.mjs`: Max's original `recordDecision`, `auditExport` and
  `exportCSV` functions retain prior labels/history and model provenance.

## Reproduction and limits

Follow [the portable README](../experiments/maintenance_research/README.md).
The CI adds the 39-test maintenance suite before the existing full-repository
checks on Ubuntu and Windows. The bounded Windows runner executes those tests,
the 14 main adapter tests and the toy update/rollback demonstration. The optional
web check needs an existing copy of Max's module; the runner never fetches it.
Generated verification files go only into a fresh explicitly selected output
directory and are not committed.

The original November 14 investigation additionally verified the five real model
versions from permitted metadata, with 45 Python checks and four web scenarios.
Its private evidence remains archived locally, and its aggregate results are
preserved in the research reports. This new source-only run establishes that
reviewers can test the reusable engineering without that private archive.

The initial publication checks above did not include the full-repository suite.
PR CI then reproduced three Windows failures already recorded on main/adapter
PR #35: the synthetic assessment fixture passed an unresolved temporary root to
`file_pin`, whose production callers require a resolved checkout. The fixture now
resolves its root before creating/pinning files. Regression cases exercise a
noncanonical root and confirm that outside-checkout files remain rejected before
reading. The production assessment module, frozen protocol and evaluation outputs
are unchanged. Actual CI status and the follow-up verification belong to this
repair; they must not be inferred from the earlier focused run.

Local repair verification: **10 assessment tests passed** and **719 full-repository
tests completed with no failures/errors and one expected private-artifact skip**.
The full run used an aggregate 4-GiB job limit, four-CPU affinity and CPU/time
bounds; the temporary harness allowed the child processes required by the CLI
tests. Real project data was blocked, the private-model opt-in was disabled and
CUDA remained uninitialized. No installed dependency or production file changed.

All four criteria pass at the **offline engineering** scope. Shared authenticated
application storage, UI/backend wiring and a coordinated loader for any future
approved maintenance delta remain Max/team work. Experimental acceptance is not
production qualification, and no improvement on genuinely new products is claimed.
Every prediction still requires human approval. No model was published or deployed.
