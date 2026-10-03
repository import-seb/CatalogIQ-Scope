# Target-label cleaning

Dataset-wide structural screening is available as a
[separate command](structural_check.md) for both original datasets. It does not
apply or replace this cleaner's quarantine rules. This command's target output
remains an unscreened pass-through.

Run from an installed project environment:

```bash
python -m catalogiq --output-dir data/processed/my_run
```

The default output directory is `data/processed/target_cleaning`. Existing run
directories are never overwritten. Use `--train` and `--target` for other CSVs.
The source notebook is [02_clean_target_cols.ipynb](../notebooks/02_clean_target_cols.ipynb).

## Current behavior

**Reason 2 is review-only.** Rows with reason 2 (leading space or lowercase label)
remain in the cleaned data unless they also trigger a quarantine reason. Reason 3
(label rarity) remains review-only, as before. Reasons **4, 5, and 6** quarantine
rows. For example, `[2]` is retained and `[2, 5]` is quarantined. Reasons accumulate;
quarantine is for review, not proof of invalidity.

The cleaner applies only `Cold / Flu` -> `Cold/Flu` and `Other Lifestyle` ->
`Other Lifestyle CHC` in Sub-Segment. It preserves every supplied Mnfr value,
including missing and out-of-vocabulary values. Invalid populated Mnfr values are
flagged and quarantined, never reassigned. There is no Brand, Platform, Segment, or TargetAgeGroup
inference, and no deduplication.

Checks record label formatting, rare-label frequency, unusual source-text
missingness/length, absent numeric content, and invalid manufacturer vocabulary.
Source metadata is excluded from those checks. Multiple reasons accumulate.
Lowercase brands may be legitimate; these are prompts for review.

## Outputs

| File | Contents |
| --- | --- |
| `training_cleaned.csv` | Retained records, including reason-2-only flags, corrections and provenance. |
| `training_quarantine.csv` | Records triggering reasons 4, 5, or 6, with all accumulated reasons. |
| `target_unchanged.csv` | Parsed prediction data with provenance; not screened or filled. |
| `label_changes.csv` | Source identity, old/new values, and rule for each correction. |
| `review_flags.csv` | Detailed flag events for both retained and quarantined records. |
| `hierarchy_violations.csv` | Single-parent count diagnostics, without automatic changes. |
| `summary.json` | Row counts, flagged-row count, source hashes, changes, distributions, reasons. |

`rows.training_flagged` counts flagged rows retained in the cleaned output.
`rows.training_quarantine` counts excluded rows. These output partitions together
contain every input row exactly once. `flag_events_by_reason` counts events across
both partitions, so it can exceed either row count.

`dataset`, `source_sha256`, and zero-based parsed `source_row` identify each record.
Identifiers are read as strings with pandas default NA parsing, so exported CSVs
are parsed-data exports rather than byte-identical source copies.

## Target distributions

All six target columns have `before` and `after` maps in `target_distributions`.
Before covers every input row; after covers retained cleaned rows, including
review-only flags. Differences reflect corrections and quarantine exclusions. The prediction dataset is separate.
`<missing>` counts parsed null values, including zero counts; labels appearing in
either stage appear in both maps. A literal `<missing>` label is rejected to avoid
conflating it with nulls. `target_distributions_scope` records the definitions.

The notebook uses the shared package checks for Q_REASON, while retaining its
exploratory sequence of label corrections. Previously saved notebook outputs from
the exclusion-based policy must be rerun before reuse. Older generated output runs are historical; the current policy is reason-2-only
review behavior with the existing reason-3 exception.

Run checks with `python -m unittest discover -s tests -v`.
