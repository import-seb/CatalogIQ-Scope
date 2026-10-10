# CatalogIQ

Product classification research and auditable preparation of six CatalogIQ labels:
`Mnfr`, `Brand`, `Platform`, `Segment`, `Sub-Segment`, and `TargetAgeGroup`.

**Current stage: Segment model development.** The repo contains audited cleaning,
grouping experiments, a first transformer baseline, and a frozen evaluation
protocol. Use the protected snapshot described below for further model work.
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

### Create Segment model splits

For continued Segment model development, use the corrected, frozen snapshot in
`data/processed/split_finalization_20261009/frozen/`. It includes full-record
`train.csv` and `validation.csv`. `test.csv` is reserved for the final evaluation;
keep it closed during development. The guarded command checks the seal and
loads training and validation only:

```bash
python -m scripts.train_frozen_segment --tokenize --output-dir data/processed/segment_development_preparation
```

Add `--train` and choose a new output directory when starting a fresh transformer
run. Model features, tokenizer, and text limits must match the frozen input
contract; optimizer settings can change during validation-based development.
Install `requirements-training.txt` for tokenization/training. The private
snapshot and saved tokenizer must be available locally.

The [split finalization note](docs/findings/02_splitting/02_finalization_20261009.md)
records fixes, integrity results, residual risks, and the test reservation policy.
The policy reserves all eligible groups from the earlier unscored test after
excluding recorded reviews, prior model use, and their historical related
families. It accepts a smaller test instead of using exposed records to reach 15%.
Grouping is frozen before reserving the test. Do not select new grouping rules
or model settings using that test.

Verify the saved snapshot without creating another split:

```bash
python -m scripts.finalize_splits --verify data/processed/split_finalization_20261009/frozen
```

`scripts.finalize_splits` creates a new sealed snapshot only when given the
archived exposure registry, effective-input cache, original assignments, and
review fixtures. Its `--config` accepts `split`, `rule`, `model`, and `allocation`
JSON sections; resolved configuration and source/input hashes are sealed before
test selection. Preserve the adopted snapshot during model development.

The earlier general splitting command below remains a **rule-v3 research
export**, without the corrected grouping, transformer equality constraints, or
historical exposure exclusions. Its outputs should not replace the frozen test.

Create train, validation, and test datasets from the integrated cleaner's
`training_candidate.csv` export:

```bash
python -m scripts.split_data --input data/processed/integration_run/training_candidate.csv --output-dir data/processed/my_splits
```

Use a new output directory for each run. This command uses rule-v3 product groups,
seed **42**, and a **70/15/15** target ratio, balancing `Segment` where practical.
Groups stay within one split, so actual proportions may differ from the targets.
This command preserves the earlier experimental grouping behavior.

The output contains `train.csv`, `validation.csv`, and `test.csv`, with all input
columns plus `record_id`, `group_id`, and `split`; no assignment join is needed.
`rule_assignments.csv` remains available for tracing records. Matching edges,
group sizes, and Segment distributions accompany `summary.json`, which records
diagnostics, isolation checks, runtime, package versions, and output hashes.
`input_manifest.json`, `split_config.json`, and `rule_refinement_config.json`
capture input fingerprints and resolved settings for reproduction.

Use `--seed`, `--config path/to/split_config.json`, or
`--grouping-config path/to/rule_config.json` to change settings.
`--identifiers path/to/identifiers.csv` supplies source-keyed identifiers;
otherwise they are recovered from the cleaned-export manifest when available.

Read the saved datasets as strings to preserve original values. Missing Segment
values are preserved; exclude blank/`null` labels when fitting a supervised model:

```python
import pandas as pd

train = pd.read_csv("data/processed/my_splits/train.csv",
                    dtype=str, keep_default_na=False, encoding="utf-8")
train = train.loc[~train["Segment"].str.strip().str.casefold().isin(["", "null"])]
X_train = train[["ProductName", "ProductBrand", "ProductDescription", "ProductContents"]]
y_train = train["Segment"]
```

Load validation the same way; reserve test loading for final evaluation.
IDs, group/split columns, provenance,
identifiers, and other target columns are audit data, not model inputs.

Compare rule-based and TF-IDF product groups on the retained cleaned training export:

```bash
python -m scripts.compare_splits --output-dir data/processed/split_comparison
python -m scripts.compare_splits --show data/processed/split_comparison
python -m scripts.compare_splits --verify data/processed/split_comparison
```

The experiment saves separate 70/15/15 assignments, matching links, Segment
distributions, disagreements, and a shared independent character-shingle leakage
audit. It repeats both grouping/splitting computations by default. `--config`
accepts JSON parameter overrides; resolved settings, input hashes and package
versions are saved with each new run. `--input` and `--identifiers` select another
cleaned export and its source-keyed identifier artifact. No cleaning rules or
source records change. The missing Segment stratum stays visible as `<MISSING>`.

Set `"grouping_version": 2` in the configuration for the refined matchers. V2
uses title/core/formulation guards and packaging-family normalization. Its rules
use `rule_family_threshold` and `rule_family_edit_threshold`; TF-IDF uses
`tfidf_name_threshold`, `tfidf_family_threshold`, and the two supporting-text
thresholds. Title features are uncapped by default. V2 uses only the description
and contents entries of `tfidf_weights`; title/brand weights, `tfidf_threshold`,
and `tfidf_max_df` retain their v1 meaning and do not control v2 title matching.
The independent leakage evaluator and split allocator are shared across versions.

For diagnostic refinement, reserve identifier-only pairs with
`catalogiq.split_review_validation.reserve_holdout` before changing matchers.
It excludes whole baseline components touched by the diagnostic annotations.
Call `seal_refinement` after the completed run and before inspecting held-out
product details. Then compare saved runs and weak pair judgments:

```bash
python -m scripts.compare_refinement --baseline data/processed/split_comparison_20261008 --refined data/processed/split_refinement_20261008_final --diagnostic data/processed/split_pair_review_20261008/annotated_pair_review.csv --holdout-dir data/processed/split_refinement_holdout_20261008 --output data/processed/split_refinement_comparison_20261008
```

Add `--heldout-annotations path/to/annotated_pairs.csv` after the blind review.
The comparison verifies identical input records, evaluator pairs/scores, and
the frozen run; it reports diagnostic and held-out errors separately. These
selected weak judgments are not population accuracy estimates. Keep held-out
judgments out of subsequent tuning of the same experiment.

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
