# CatalogIQ

Product classification research and auditable preparation of six CatalogIQ labels:
`Mnfr`, `Brand`, `Platform`, `Segment`, `Sub-Segment`, and `TargetAgeGroup`.

**Current stage: Segment model development.** The repo contains audited cleaning,
grouping experiments, a first transformer baseline, and a frozen evaluation
protocol. Generate the authoritative splits with the commands below for model work.
A production classifier and deployment remain future work.

## Start here

Use Python 3.11 or newer; Python 3.12 is the suggested project environment.
From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -e ".[splitting]"
```

Conda users can use their existing environment instead. The `splitting` extra pins
the tested numerical dependencies for exact reproduction. The core installation
includes the pinned tokenizer dependency; splitting does not require model weights.
Install `python -m pip install -e ".[splitting,notebooks]"` if you need notebooks.

Place the two private source CSVs in `data/provided/` as described in
[data setup](data/README.md). Do not commit the datasets.

## Run the workflow

```bash
python -m catalogiq --mode integrated --output-dir data/processed/cleaned
python -m scripts.split_data --input data/processed/cleaned/training_candidate.csv --output-dir data/processed/splits
python -m scripts.split_data --verify data/processed/splits
```

Run from the repository root and use new output directories. Verification checks
complete source and assignment hashes against the committed October 9 protocol;
an incompatible dataset or assignment fails rather than creating a replacement test.
No private split CSVs, exposure archive, model weights, or local cache need to be shared.

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

### Create Segment model splits

`scripts.split_data` is the authoritative team splitter. It reproduces the
finalized October 9 product groups and split membership, using corrected V4 rules,
identical transformer-input constraints, fixed seed **42**, and the frozen
exposure exclusions. The target ratio remains **70/15/15**; the protected test
contains **12,160 records (14.49%)** because previously exposed records cannot
pad it. Train has 59,111 records and validation has 12,667.

The source of truth is the packaged
[`segment-final-20261009` manifest](src/catalogiq/resources/segment_final_20261009/manifest.json):
grouping version `rule-final-v4`, split version `segment-final-splits-v1`, full
source and membership hashes, and the recorded exposure selectors. Commands
regenerate assignments locally and verify every membership against that protocol.

Use these generated files:

| File in `data/processed/splits/` | Purpose |
|---|---|
| `train.csv`, `validation.csv` | Full cleaned records ready for model development; no assignment join |
| `test.csv` | Reserved final evaluation; keep closed during development |
| `assignments.csv` | Authoritative provenance, record ID, product group, and split mapping |
| `rule_assignments.csv` | Identical alias of the final assignments, retained for compatibility |
| `protected_final_test.csv` | Test IDs and groups only, used by development guards |
| `grouping_freeze.json`, `completion.json`, `summary.json` | Protocol identity, integrity seals, and diagnostics |
| `model_config.json`, `tokenizer/`, `model_inputs.csv` | Frozen text/tokenizer contract and input-equality evidence |

The guarded development command checks the authoritative seal and opens only
training and validation:

```bash
python -m scripts.train_frozen_segment --output-dir data/processed/segment_development_preparation
```

This prepares supervised development records without training a model. Blank or
`null` Segment labels remain in the CSVs and are excluded from supervision.
For Python model code, use the same guard:

```python
from catalogiq.segment_frozen_development import load_frozen_development

development = load_frozen_development("data/processed/splits")
train = development.frame.iloc[development.train_indices]
validation = development.frame.iloc[development.validation_indices]
X_train = train[["ProductName", "ProductBrand", "ProductDescription", "ProductContents"]]
y_train = train["Segment"]
```

For future transformer runs, install `python -m pip install -r requirements-training.txt`
and supply the pinned pretrained encoder through the existing model-cache process.
`--tokenize` verifies development tokenization; `--train` explicitly trains a fresh
baseline. Model features, tokenizer, and text limits must match the sealed contract;
optimizer settings can change using validation. Choose a new output directory.
Pass an experiment configuration with `--model-config path/to/model_config.json`;
leave the generated split artifacts unchanged.

IDs, grouping metadata, identifiers, provenance, and other target columns are
audit data, never model inputs. Do not select grouping rules or model settings
using the reserved test. The [portable workflow note](docs/findings/02_splitting/03_portable_workflow_20261010.md)
identifies the committed protocol and verification evidence; the
[October 9 finalization](docs/findings/02_splitting/02_finalization_20261009.md)
records its remaining leakage risks.

### Classical Segment baseline

The word TF-IDF + logistic-regression baseline uses the same sealed training and
validation records. Its model does not need Hugging Face weights or tokenization;
the shared splitter still uses its packaged tokenizer to reproduce product groups.

```bash
python -m scripts.train_segment_baseline --split-dir data/processed/splits --features compare --output-dir data/processed/segment_classical_check
python -m scripts.train_segment_baseline --split-dir data/processed/splits --features compare --train --output-dir data/processed/segment_classical_run
```

The first command verifies and prepares only. The second fits both the four-field
baseline and the experiment with ProductCategory concatenated into the same text.
Vocabulary and IDF fit only on train; predictions and comparisons cover validation.
Both commands reuse the authoritative seal verifier and never open or hash test.csv.
The category arm is a validation experiment, with the split protocol unchanged.
See the [baseline handoff](docs/findings/03_modeling/01_classical_segment_baseline.md)
for outputs and limitations.

### Historical research

Earlier comparison/refinement `rule_assignments.csv` files and
`data/processed/split_finalization_20261009/frozen/` are historical artifacts.
The filename alone does not identify a protocol; verify the generated directory
with `scripts.split_data --verify`. The existing
`scripts.finalize_splits` entry point preserves the local historical protocol;
it is not the portable team workflow. Use the files generated by the official
`scripts.split_data` command above for continued development.

Rule/TF-IDF comparison remains available for historical research on the separately
exported `training_cleaned.csv`:

```bash
python -m scripts.compare_splits --output-dir data/processed/split_comparison
python -m scripts.compare_splits --show data/processed/split_comparison
python -m scripts.compare_splits --verify data/processed/split_comparison
```

These commands produce experimental assignments and never replace the protected
team test. Grouping refinement has stopped; model development uses training and
validation only.

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
