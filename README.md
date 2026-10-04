# CatalogIQ

Product classification research and auditable preparation of six CatalogIQ labels:
`Mnfr`, `Brand`, `Platform`, `Segment`, `Sub-Segment`, and `TargetAgeGroup`.

**Current stage: data understanding and preparation.** The repo contains exploratory
audits and a tested cleaning command. A trained production classifier, frozen
model-evaluation split, and deployment are not implemented here yet.

## Start here

Use Python 3.11 or newer; Python 3.12 is the suggested project environment.
From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -e ".[notebooks]"
```

Conda users can use their existing environment instead. `requirements.txt` remains
an equivalent convenience entry point. Dependencies have compatibility ranges;
there is not yet a frozen environment lock.

Place the two private source CSVs in `data/provided/` as described in
[data setup](data/README.md). Do not commit the datasets.

## Run the workflow

```bash
python -m catalogiq --output-dir data/processed/my_run
python -m catalogiq.check_structure --output-dir data/processed/structural_run
python -m unittest discover -s tests -v
python -m jupyterlab
```

`catalogiq-clean` is the equivalent installed cleaning command. Each run requires
a new output directory. The command writes source fingerprints, changes, review
flags, and a summary beside its CSV outputs. See the [cleaning guide](docs/target_cleaning.md).

For combined feature/target/structural processing, use `--mode integrated` and a new run directory:

```bash
python -m catalogiq --mode integrated --output-dir data/processed/integration_run
python -m catalogiq.integration_validation --output-dir data/processed/integration_run
```

Pass `--train` and `--target` if your local filenames differ from the defaults.
This writes one candidate row mask plus candidate, quarantine and review-hold
partitions. It preserves source strings except audited corrections and runs the
feature and structural verifiers on both inputs. The agreed default source
`Exclude` policy is `keep`. Field flags alone do not hold a row; missing name,
description and contents triggers structural quarantine. Candidate feature
exports omit timestamps and Sun1-Sun5/Notes while complete audit partitions
preserve them. Category-level features remain deferred to feature engineering. See the
[integration contract](docs/integration/cleaning_contract.md) and
[full-data validation](docs/findings/01_cleaning/02_integration_validation.md).

The separate [structural check](docs/structural_check.md) screens both original
datasets and records row decisions and raw evidence without changing rows or
applying either cleaner's quarantine mask. Integrated mode consumes this audit
and combines the three explicit quarantine recommendations by source key.
It uses the same role-named, one-based identities as integrated mode. You can
also run it with `python -m catalogiq --mode structural --output-dir data/processed/structural_run`.

**Current manufacturer policy:** preserve all supplied `Mnfr` values, including
missing values. No Brand-to-J&J reassignment. Read [current decisions](docs/decisions.md)
before promoting any notebook experiment into reusable code.

## Repository map

```text
src/catalogiq/          Installable Python package
  cleaning.py           Reusable cleaning, validation, and audit output
  features.py           Conservative feature formatting and flags
  feature_decisions.py  Second-pass feature decisions and audit
  feature_validation.py Separate full-file feature reconciliation
  integration.py        Combined source-keyed candidate mask and partitions
  structural.py         Independent dataset-wide structural decisions and evidence
  check_structure.py    Enables python -m catalogiq.check_structure
  cli.py                Command-line arguments and invocation
  paths.py              Project path discovery
  __main__.py           Enables python -m catalogiq
  __init__.py           Package exports
notebooks/             Ordered exploratory studies; see their status index
tests/                Behavioral tests using synthetic, nonprivate fixtures
data/provided/        Private source CSVs (ignored)
data/processed/       Generated run directories (ignored)
docs/decisions.md     Current rule decisions and withdrawn policies
docs/findings/        Written analysis findings
docs/references/      External partner and tooling background
.github/workflows/     Automated checks; no private data required
```

- [Notebook index](notebooks/README.md): reading order and stale-analysis warnings.
- [Documentation index](docs/README.md): evidence, policy, and background.
- [Contribution workflow](CONTRIBUTING.md): how to make reproducible changes.

Keep scientific conclusions separate from exploratory hypotheses. Model-generated
labels will remain recommendations until validated; review flags are not proof
that a supplied label is wrong.
