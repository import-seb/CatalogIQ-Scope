# Isolated CatalogIQ maintenance engineering

Reusable code and synthetic tests for the existing finished models. No production
inference, application components, dependencies or model weights are replaced.
Every prediction remains human-reviewed; no model promotion runs automatically.

Start with [the acceptance audit](../../docs/maintenance_ticket.md),
[Max's integration instructions](../../docs/integration/maintenance_versions.md),
[the scope decision](../../docs/decisions.md#2026-11-14-maintenance-scope-and-confirmation-collection)
and [verification evidence](../../docs/maintenance_verification.md).

## Contents

- `correction_ledger.py`: append-only five-target review events, conflicts,
  adjudications, revocations, conservative eligibility and versioned exports.
- `ticket_lifecycle.py`: content-addressed data/model/result versions, explicit
  experimental acceptance/rejection, version-bound delegation to the existing
  inference adapter, immutable approved reviews and rollback receipts.
- `maintenance_offline.py`: paired validation, whole-group cohort proposals and
  reversible anchored-prior Brand count overlays.
- `methods.py`, `data.py`, `maintenance_keys.py`, `deltas.py`: existing warm-start
  head routines, full-input signatures/grouping, cautious identity helpers and
  checked delta bindings. These are offline primitives, not automatic training.
- `ticket_demo.py`: synthetic A/B engineering demonstration using toy heads,
  generated confirmations and the existing engine's loader-injection interface.
- `test_ticket.py`, `test_strategy.py`, `tests.py`: synthetic acceptance and safety
  tests. `resources.py` supplies the existing Windows job-object limits.

## Reproduce without private data or cached source snapshots

Use the repository's existing inference/training environment. Do not install or
download model weights to run these checks. From a checkout containing main's
inference adapter:

```powershell
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s experiments/maintenance_research -p 'test*.py' -v
```

On POSIX, prefix the command with `PYTHONPATH=src`. The CI runs this suite on both
Windows and Ubuntu using its existing dependencies. The suite creates only toy
data in private temporary directories and loads no finished models.

For the additional OS-bounded test/demo on Windows, use a fresh short output path:

```powershell
python -B experiments/maintenance_research/ticket_run.py --output "$env:TEMP/ciqm-check-1"
```

The runner uses the Python interpreter that launched it, not a workstation-specific
virtualenv or session file. It enforces 600 seconds wall time, 2,400 CPU seconds,
four-CPU affinity, at most four numerical threads, one interop thread and 4 GiB
job commit. It also stops at 4 GiB sampled RSS and denies CUDA initialization,
networking, child processes and writes outside its fresh worker output. Its
Python data allowlist is empty. Existing output directories are refused.

The toy updater's save/reload demonstration uses `torch` and `safetensors` already
listed in `requirements-training.txt`; the optional `psutil` sampler is not
required. When it is absent, memory measurements are explicitly unavailable and
the independent OS resource limits remain enforced.

To additionally check Max's unchanged export/review functions, supply his existing
module from the inspected web branch (Node must already exist):

```powershell
python -B experiments/maintenance_research/ticket_run.py --output "$env:TEMP/ciqm-check-2" --web-domain PATH_TO_MAXS_WEB/public/domain.js
```

No web source is fetched, installed or copied by the runner. The standalone
`ticket_web_test.mjs` checks prior export bytes, approval history, A/B version
references and refusal to overwrite existing exports.

## Deliberately excluded

Private source datasets, embeddings, checkpoints, adapters, logs, session pointers,
environment paths, copied upstream code, model inventories and cleanup inventories
are not part of this change. The historical full-data/LoRA research runners are
also excluded: the capstone decision defers further quality experiments. The
published research reports retain aggregate findings and limitations; any mentioned
`data/processed/...` evidence is private and is not a distributable artifact.

Synthetic success establishes the engineering workflow, not generalization or
production qualification. Shared authenticated storage, UI wiring and a reviewed
loader for any future accepted update remain Max/team integration work.
