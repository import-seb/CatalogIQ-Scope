# Target classification output format

Ticket #1287. Owner: Maria. Eligibility version `target-eligibility-20261115-v1`.

This is the documented format the fourth acceptance criterion asks for. It
describes what `scripts/classify_target.py` writes and how those columns line up
with the dashboard's own export, so "compatible with the dashboard" can be
checked rather than asserted.

## 1. What runs

```
python -m scripts.classify_target data/raw/Selfcare_Target_data.csv \
    --processed-dir data/processed --output-root data/processed/target_classification_20261115
```

The script chooses no labels and loads no model of its own. It reads the target
CSV, applies the eligibility rule below, hands the eligible rows to the frozen
`InferenceEngine` in batches, and writes the results. Outputs go under
`data/processed/`, which is gitignored; only the report and the ineligible list
are small enough to keep as evidence.

## 2. Eligibility

The criterion says "every **eligible** target product" and nothing in the repo
defined eligible, so the rule lives in `target_classification.py` with a version
string of its own. A row gets at most one reason, the first that applies.

| Reason | Meaning | Rows, 2026-11-15 |
|---|---|---|
| `marked_exclude` | The source row carries the `Exclude` marker. | 1,852 |
| `missing_identifier` | No `JoiningKey`, so the result could not be traced back. | 0 |
| `duplicate_identifier` | `JoiningKey` is not unique within the file. | 0 |
| `missing_product_name` | No `ProductName`, so there is nothing to classify. | 37 |
| `unknown_retailer` | `Retailer` is not one of the six known retailers, which indicates a shifted row rather than a product. | 1 |

Of 28,082 rows, **26,192 are eligible** and 1,890 are not. The two counts always
add back to the row count, and a test enforces that nothing is dropped silently.

The `unknown_retailer` rule catches real corruption. Two rows in this file have
product description text in `Retailer` and in `JoiningKey`, from a shifted row.

The five fields the models read are present among eligible rows at
`ProductCategory` 26,191, `ProductName` 26,192, `ProductBrand` 26,185,
`ProductDescription` 8,249 and `ProductContents` 10,624. The last two look thin,
but training is 31.7% and 40.2% on the same fields against 31.5% and 40.6% here,
so the target data is no poorer than what the models were fitted on.

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

## 5. Current state

The run on 2026-11-15 processed all 26,192 eligible rows. **All six targets
report `unavailable`**, because the frozen serving members are not present. The
engine reports `model_unavailable` per route instead of failing, so the file has
its final shape already and the same command produces real predictions once the
artifacts arrive.

Missing: 30 of the 36 files `frozen_routes.json` pins. The Brand lookup's
`model.json` and `manifest.json` and the four `predictions_contract_v2_20261031`
manifests are present and hash-verified. The four MiniLM routes and all ten
`confidence_review_v3_20261103` policy and temperature files are not.
`13_remaining_minilm_training_20261029.md` records that the handoff ZIPs were
built and verified but that "no external delivery occurred", which is consistent
with none of them being on this machine.

Nothing needs to be trusted in that transfer. The loader rehashes every pinned
file and raises `artifact_mismatch` if one does not match.
