# Working on CatalogIQ

## Where work belongs

- Put reusable, tested behavior in `src/catalogiq/`. New code imports `catalogiq`,
  not `src`. Command-line argument handling belongs in `catalogiq/cli.py`.
- Put exploration in numbered notebooks. Keep existing numbers stable so findings
  and decisions can continue to cite them. Add new notebooks to the index.
- Put conclusions in `docs/findings/`, with the notebook, input scope, denominators,
  and assumptions they depend on. External background goes in `docs/references/`.
- Put run outputs under a new `data/processed/<run_name>/`. Retain `summary.json`
  beside its artifacts. Do not overwrite raw inputs or an earlier run.
- Add model, feature, or serving modules when those workflows exist. Empty folders
  do not establish a workflow.

## Promote an analysis into a rule

1. Describe the observation and counterexamples in an exploratory notebook.
2. Distinguish formatting corrections, taxonomy decisions, and uncertain heuristics.
3. Record the accepted decision and its limitations in `docs/decisions.md`.
4. Implement it in the package with tests for valid exceptions and missing values.
5. Run on a new output directory; reconcile rows and the change log.
6. Update findings that depend on the previous policy, or mark them historical.

Do not learn cleaning mappings from the unlabelled prediction dataset. Before
model development, decide which fitted preprocessing must be trained inside each
training fold. Label-dependent preprocessing can otherwise leak evaluation data.

## Local checks

```bash
python -m pip install -e ".[notebooks]"
python -m unittest discover -s tests -v
```

CI runs synthetic tests and a package build without private
CSVs. It does not execute the research notebooks or validate model quality.

## Notebook evidence

Run changed analysis from a fresh kernel, top to bottom, before relying on its
outputs. State the data view (full export or screened subset), missing-value policy,
thresholds, and denominator. Preserve useful outputs only after reviewing them for
private data. Clearing outputs is not a substitute for checking reproducibility.

Old notebooks may contain private product examples and superseded assumptions.
Consult the notebook index before reusing findings, and review data-sharing
permission and scientific validity before publishing results.

## Dependencies and reproducibility

Maintain direct dependencies in `pyproject.toml`; avoid machine-specific paths or
freezing an unrelated global environment. A portable, tested lock remains follow-up
work. Record the environment and package revision with important experimental runs;
current cleaning summaries capture input hashes and policy, not a full environment.
