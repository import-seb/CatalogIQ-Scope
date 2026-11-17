# Target classification output format

Ticket #1287. Owner: Maria. Eligibility version `target-eligibility-20261116-v2`.

This is the documented format the fourth acceptance criterion asks for. It
describes what `scripts/classify_target.py` writes and how those columns line up
with the dashboard's own export, so "compatible with the dashboard" can be
checked rather than asserted.

## 1. What runs

```
python -m scripts.classify_target "data/provided/Selfcare_Target_data (1).csv" \
    --processed-dir data/processed \
    --decisions data/processed/integrated_20261116/row_decisions.csv \
    --output-root data/processed/target_classification_20261116
```

The script chooses no labels and loads no model of its own. It reads the target
CSV, applies the eligibility rule below, hands the eligible rows to the frozen
`InferenceEngine` in batches, and writes the results. Outputs go under
`data/processed/`, which is gitignored; only the report and the ineligible list
are small enough to keep as evidence.

## 2. Eligibility

Eligibility is not decided in this module. `docs/decisions.md` settles it and
`integration.row_decision` implements it, so the classifier reads that verdict
from `row_decisions.csv` instead of keeping a second rule that can drift.

A product is eligible when the integrated cleaner's status is `candidate`, run
with the agreed `--exclude-policy keep`.

An earlier version of this file carried a rule written here instead, and it
disagreed with the policy on four counts. It disqualified the `Exclude` marker
on its own, against policy 1. It disqualified a missing `ProductName` without
looking at description or contents, against policy 4. It quarantined on a single
unexpected `Retailer`, against policies 6 and 7, which require documented
combinations of evidence. And it treated only blanks as missing, while the
policy also counts a case-insensitive `null`. That rule excluded 1,678 products
the agreed policy keeps.

Of 28,082 target rows, **27,870 are eligible** and 212 are quarantined:

| Reason code | Rows |
|---|---|
| `text_in_multiple_numeric_anchors` | 109 |
| `url_in_timestamp_with_displacement` | 93 |
| `text_numeric_failure_with_corroboration` | 60 |
| `missing_product_text` | 42 |
| `unexpected_exclude_with_populated_unknown_columns` | 2 |

Each code appears under both a `feature:` and a `structural:` prefix, since the
two streams reached it independently. A row carries every contributing reason.

The two rows with product description text in `Retailer` are quarantined here as
`unexpected_exclude_with_populated_unknown_columns`, which is the policy's own
route: the `Exclude` marker combined with populated unknown columns, not one
suspicious field on its own.

The models were fitted under the same policy. The pinned split protocol records
`source.input_rows: 83938`, which is exactly the integrated cleaner's training
candidate count, so classifying `Exclude`-marked target products is inside the
training distribution rather than an extrapolation.

## 3. Files written

| File | What it is |
|---|---|
| `target_predictions.csv` | One row per eligible product. The deliverable. |
| `target_ineligible.csv` | One row per skipped product, with its reason. Nothing vanishes. |
| `target_predictions_sample.json` | The first 50 products in the adapter's own envelope, as the format example and the audit copy. |
| `target_classification_report.json` | Counts, versions and per-target status totals. |

`--full-json` writes every product as one envelope. It is off by default: the
file is large, and the adapter's `error` text embeds the absolute path of the
machine that ran it, which does not belong in a shared artifact.

## 4. Column contract

Source columns come first, copied from the input: `JoiningKey`, `Retailer`,
`Category`, `ProductCategory`, `ProductBrand`, `ProductName`, `Sku`, `Upc`,
`ProductUrl`. `JoiningKey` is the `record_id` and is never regenerated.

Then the dashboard's own column names, from `web/public/domain.js`:

| Column | Source |
|---|---|
| `catalogiq:mode` | `batch` for this script |
| `catalogiq:review_status`, `:reviewer`, `:reviewed_at`, `:review_note` | empty; the dashboard fills them during review |
| `catalogiq:engine_version` | the adapter's engine version |
| `catalogiq:registry_version` | the relationship registry version the engine loaded |
| `catalogiq:<target>:original_label` | the model's prediction |
| `catalogiq:<target>:final_label` | empty; the reviewer's decision |
| `catalogiq:<target>:raw_confidence` | the model's own score |
| `catalogiq:<target>:prediction_status` | `suggestion`, `unavailable` or `failed` |
| `catalogiq:<target>:review_reasons` | joined with `; ` |
| `catalogiq:<target>:model_version` | `provenance.model_id` |

Every column the dashboard's `exportCSV` generates is present. The file also
carries `catalogiq:<target>:calibrated_confidence`, `:selection_status`,
`:review_required`, plus `catalogiq:eligibility`, `:eligibility_version`,
`:schema_version` and `:review_required`. Those are extra and the dashboard
ignores them.

Targets are `Segment`, `Sub-Segment`, `Brand`, `Mnfr`, `TargetAgeGroup` and
`Platform`. Platform has no route and always reports `unavailable`.

### Spreadsheet formulas

A value that starts with `=`, `+`, `-` or `@` gets a leading apostrophe, the
same rule as the dashboard's `csvCell`. Three cells in this file need it, one of
them the real brand `+1HEROLABS`. The JSON sample keeps the untouched value and
is the audit copy, which is the split the dashboard already uses.

Five source values already began with an apostrophe of their own and are left
alone. Everything else is byte identical to the input: 112,328 cell comparisons
across both files found no other difference.

## 5. Run of 2026-11-16

All 27,870 eligible rows were classified. Five routes returned suggestions and
Platform returned `unavailable`, which is correct: it has no selected route.
Every policy ships `enabled=false` with `operational_threshold=null`, so nothing
is auto-accepted and every product still needs a human.

The report records the decisions file by content hash
(`791b84911f442de3821a50ddebe43064d3aea0dd32df7d82a7ee5e3bd72242d2`) rather than
by path. An earlier version wrote the path, which leaked the machine the run
happened on and broke the byte-for-byte comparison below.

### Relationship rules

| Relation | On validation | On the target set |
|---|---|---|
| R1 Sub-Segment under Segment | 1.83% | 2.29% (638 of 27,870) |
| R2 Brand under Mnfr | 0.05% | 0.047% (13) |
| Pairs that could not be checked | 7,285 | 0 |

651 products carry a relationship conflict. The relations are still `candidate`
and nothing is enforced, so this is a warn-mode queue rather than a decision.

The zero confirms what section 3 could only assert: the 7,285 unchecked pairs on
validation were an artifact of each target having its own validation set. In
serving, every available model predicts for every product.

### The `Exclude` marker predicts disagreement

R1's 2.29% is higher than validation's 1.83%, and the gap comes from one place:

| Population | Products | With a conflict |
|---|---|---|
| Marked `Exclude` | 1,836 | 219 (11.93%) |
| Not marked | 26,034 | 431 (1.66%) |

Products carrying the marker are **7.2 times** more likely to produce a
relationship conflict. The 1.66% on the rest sits close to validation's 1.83%,
so the rule generalizes; the marker is what moves the overall rate.

Both things are true at once. These are real products and the agreed policy to
keep them is right, since the models were trained on a population that includes
them. They also deserve priority in the review queue, and the marker already in
the source data is a usable signal for ordering it.

An earlier version of this file reported 1.61% and said the rates held. That
comparison was against the 26,192 rows the superseded eligibility rule had
selected, not against the population the policy defines.

### Predictions look like the training population

`J&J` appears on 1.4% of rows, against 1.4% in training. Segment is dominated by
`Vitamins, Minerals & Supplements`. Nothing shifted.

### Serving artifacts

Delivered 2026-11-15 and verified before use: the outer archive's SHA-256, all
18 files, the 37 members inside the three model ZIPs with their CRCs, and all 36
pins in `frozen_routes.json`. The delivery's copy of `frozen_routes.json` is
byte identical to the one in the repository.

Running inference also needs `torch` and `transformers`, and `pyproject.toml`
declares neither as a runtime dependency. Without them every MiniLM route
reports `model_unavailable`, which reads as missing files rather than a missing
package. `transformers>=4.50,<5` resolves against the pinned
`tokenizers==0.22.2`; the 5.x series forces an upgrade that breaks the pin.
Tokenizing the same strings under both versions gives identical token ids, so
the pin is about reproducibility discipline, not a change in output. With both
installed the full suite runs green.
