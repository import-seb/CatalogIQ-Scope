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
python -m unittest discover -s tests -v
python -m jupyterlab
```

`catalogiq-clean` is the equivalent installed cleaning command. Each run requires
a new output directory. The command writes source fingerprints, changes, review
flags, and a summary beside its CSV outputs. See the [cleaning guide](docs/target_cleaning.md).

**Current manufacturer policy:** preserve all supplied `Mnfr` values, including
missing values. No Brand-to-J&J reassignment. Read [current decisions](docs/decisions.md)
before promoting any notebook experiment into reusable code.

## Repository map

```text
src/catalogiq/          Installable Python package
  cleaning.py           Reusable cleaning, validation, and audit output
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
