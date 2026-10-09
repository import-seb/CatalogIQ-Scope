# CatalogIQ

Product classification research and auditable preparation of six CatalogIQ labels:
`Mnfr`, `Brand`, `Platform`, `Segment`, `Sub-Segment`, and `TargetAgeGroup`.

**Current stage: data understanding and preparation.** The repo contains exploratory
audits, a tested cleaning command, and experimental grouped evaluation splits.
A splitting policy has not yet been adopted; a trained production classifier,
frozen model-evaluation split, and deployment are not implemented here yet.

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
