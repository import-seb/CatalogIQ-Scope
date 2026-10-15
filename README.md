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
baseline. For this guarded baseline, model features, tokenizer, and text limits
must match the sealed contract; optimizer settings can change using validation.
Choose a new output directory.
Pass an experiment configuration with `--model-config path/to/model_config.json`;
leave the generated split artifacts unchanged.

IDs, grouping metadata, identifiers, provenance, and other target columns are
audit data, never model inputs. Do not select grouping rules or model settings
using the reserved test. The [portable workflow note](docs/findings/02_splitting/03_portable_workflow_20261010.md)
identifies the committed protocol and verification evidence; the
[October 9 finalization](docs/findings/02_splitting/02_finalization_20261009.md)
records its remaining leakage risks.

### Compare ProductCategory inputs

This separate development experiment registers ProductCategory outside the frozen
four-field model contract. After installing the training dependencies and caching
the pinned encoder, run:

```bash
python -m scripts.compare_segment_category --split-dir data/processed/splits --output-dir data/processed/segment_category_ablation --sequence-length 256 --device cuda
```

This trains two fresh models on the same full frozen development pool: **55,776
training records and 11,952 validation records**. `baseline` uses ProductName,
ProductBrand, ProductDescription, and ProductContents; `with_category` prefixes
the retailer's ProductCategory breadcrumb. Defaults share the same initial weights,
seed **42**, training order, optimizer, and fixed three-epoch schedule. The category
representation is audited on development records only. The final test stays
closed, and the split artifacts remain unchanged.

Both experimental models use **256 tokens by default**, controlled by
`--sequence-length`. The original splitting protocol keeps its **128-token**
grouping contract and unchanged test membership. The category prefix shares the
experimental token budget with product text and can still displace it. The default
category limit is 512 characters; `--category-character-limit` configures it.

Read `validation_comparison.csv` for overall metrics, `per_class_comparison.csv`
for Segment results, and `validation_slices.csv` for retailer and category coverage
results. `paired_differences.csv` and `paired_comparison.json` compare predictions
on the same validation records with group bootstrap intervals. Each model's
checkpoint and predictions are in `baseline/` or `with_category/`; `protocol.json`
and `summary.json` record settings and integrity checks.

Use a new output directory. Add `--prepare-only` to tokenize and audit inputs
without training, then repeat the same command with `--resume` to continue.
Resume validates the original configuration and reuses completed model runs;
it does not resume an interrupted optimizer step. `--model-config` selects shared
settings and preserves its configured token limit unless `--sequence-length`
overrides it. `--bootstrap-samples` controls the paired comparison's sampling.

The five-field MiniLM remains experimental; the team has not selected an official
model. See the [verified settings and checkpoint-loading instructions](docs/findings/03_modeling/01_minilm_256_reproducibility_20261015.md)
and [256-token equality audit](docs/findings/03_modeling/02_minilm_256_equality_audit_20261015.md).
The frozen V4 four-field, 128-token contract and project defaults are unchanged.

### Compare transfer to other retailers

This separate development experiment trains on **all 43,969 Amazon records** in
the frozen training split, without downsampling. It evaluates the whole original
validation pool: **2,511 Walmart, Walgreens, CVS, Target, and Kroger records** form
the retailer holdout endpoint; **9,441 Amazon records** provide an in-domain control.

```bash
python -m scripts.compare_segment_retailers --split-dir data/processed/split_integration_reproduced_20261010 --output-dir data/processed/segment_retailer_holdout_category_20261010 --reference-dir data/processed/segment_category_ablation_256_20261010 --device cuda --sequence-length 256
```

By default, this trains **only `with_category`**: the four product fields with
the ProductCategory prefix, using **256 tokens**, seed **42**, and a fixed
three-epoch schedule. `--reference-dir` optionally reuses the completed mixed-retailer
ProductCategory model's predictions on the identical validation records. Settings
and initial weights must match that reference. Omit it to run without the
reference. Use your locally generated split and experiment paths.
The frozen grouping and split artifacts remain unchanged; the final test stays closed.

Read `validation_metrics.csv` for overall, holdout, control, and retailer metrics,
`per_class_comparison.csv` for class results, and `paired_differences.csv` plus
`paired_comparisons.json` for paired group bootstrap comparisons. `category_coverage.csv`
and `category_slices.csv` show breadcrumb coverage. `with_category/` saves the
new checkpoint and predictions; `protocol.json` and `summary.json` record settings
and checks. Existing checkpoints are preserved.
Use a new output directory; `--prepare-only` and `--resume` work as above.
Retailer transfer also changes training volume, assortment, and class mix, so this
is a diagnostic of transfer rather than a causal test of taxonomy dependence.

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
