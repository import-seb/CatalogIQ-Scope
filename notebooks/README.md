# Analysis index

Install from the repository root with `python -m pip install -e ".[notebooks]"`,
then launch `python -m jupyterlab` using that environment. Run each notebook in a
fresh kernel from top to bottom. Numbering is reading order, not a pipeline DAG.

| Notebook | Question | Status / limitation |
| --- | --- | --- |
| [01](01_inspect_datasets.ipynb) | What is in the source exports? | Initial profiling; saved counts reflect its data view. |
| [02](02_clean_target_cols.ipynb) | Which label corrections are justified? | Decision exploration; reusable implementation is `catalogiq.cleaning`. No manufacturer reassignment. |
| [03](03_platform_purpose.ipynb) | What does Platform represent? | Historical: analysis includes withdrawn J&J normalization. Revalidate before using its results. |
| [04](04_productcategory_claim_audit.ipynb) | Do category paths support classification claims? | Exploratory claim audit, not an approved cleaning rule. |
| [05](05_targetagegroup_attribute_audit.ipynb) | How does TargetAgeGroup behave? | Exploratory attribute audit. |
| [06](06_age_separating_signals.ipynb) | Which signals distinguish age groups? | Exploratory rule evaluation; not the final model benchmark. |
| [07](07_mdm_id_identity_audit.ipynb) | Does MDM ID establish identity? | Provisional screening assumptions; no canonical product key established. |
| [08](08_dataset_grain_and_uniqueness.ipynb) | What makes a row unique? | Includes a historical normalized-label comparison using withdrawn J&J assignments. |
| [09](09_productbrand_brand_relationship.ipynb) | Is ProductBrand -> Brand stable? | Candidate evidence only; no automatic fills or relabeling approved. |
| [10](10_mdm_datetime_audit.ipynb) | What does MDM_InsertDateTime encode? | Full-export and integrated-partition audit; Excel serials plausible, conversion deferred. Requires a local integrated run. |
| [11: capstone sprint review](11_catalogiq_capstone_sprint_review.ipynb) | What has CatalogIQ delivered from Sprint 1 through 30 October 2026? | Nine visual slides with a 13-minute speaking plan. Original charts, calculations and technical detail are retained in the appendix. Maria's final cross-target comparative evaluation remains ongoing. |

**Authority:** [current decisions](../docs/decisions.md) and
[cleaning guide](../docs/target_cleaning.md). Historical notes are prominently marked
inside affected notebooks. Their saved evidence has not been silently recomputed
under a different policy.

Use `docs/findings/` for concise conclusions and `src/catalogiq/` for reusable code.
Keep source-row identity and define every denominator. Notebook outputs may contain
private product examples: inspect them before publishing. CI checks notebook format,
not execution or scientific validity.

The sprint review also has a [static HTML presentation](11_catalogiq_capstone_sprint_review.html)
and an [execution/evidence report](11_catalogiq_capstone_sprint_review_validation.json).
Its saved outputs are ready to present. Re-execution needs the existing private
development and prediction artifacts, the adjacent `catalogiq_review_support.py`
and `catalogiq_review_visuals.py`,
and the notebook dependencies plus Matplotlib, nbformat, nbclient and nbconvert.
It works from the repository root or `notebooks/`, uses CPU-only calculations,
and does not train models, run neural inference, or open/hash `test.csv`.
Model selection remains open. The intended business workflow reduces manual
classification through validated acceptance of routine predictions and human
review of uncertain or flagged cases; routing and measured savings remain future work.

The offline HTML uses arrow keys to navigate, **N** for optional presenter notes,
**A** for the appendix and **F** for full screen. Notes and the appendix are hidden
during the presentation by default. Notebook notes are collapsed Markdown details;
the original detailed narrative remains below the nine-slide core.
