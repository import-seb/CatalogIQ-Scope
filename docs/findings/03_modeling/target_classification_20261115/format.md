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

## 5. Run of 2026-11-15

All 26,192 eligible rows were classified. Five routes returned suggestions and
Platform returned `unavailable`, which is correct: it has no selected route.

| Target | Status | Notes |
|---|---|---|
| Segment | suggestion | 26,192 |
| Sub-Segment | suggestion | 26,192, provisional per the handoff |
| Brand | suggestion | 26,192 |
| Mnfr | suggestion | 26,192 |
| TargetAgeGroup | suggestion | 26,192 |
| Platform | unavailable | no selected route |

Every policy ships `enabled=false` with `operational_threshold=null`, so nothing
is auto-accepted and every product still needs a human.

### The same bytes on two machines

The run was executed twice, on an Apple Silicon Mac and on an x86 Linux
container, against the same frozen artifacts. `report.json` and
`ineligible.csv` came out byte identical:

```
828f95272517dc813559c06bb949f06905beeb713c8dc5490273796f573cc0a4  report.json
a4b2438ffcc3c876399cbf4eafdaadd25a6ebb5ab7733d511c7a8dab58439eca  ineligible.csv
```

### Predictions look like the training population

`J&J` appears on 361 of 26,192 rows, 1.4%, against 1.4% in training.
`UnItemised brand` lands on 66.0% against 70% in training. Segment is dominated
by `Vitamins, Minerals & Supplements` at 62.7%. Nothing shifted.

### Relationship rules on unseen data

The warn-mode checker ran over the predictions for the first time on data no
model had seen:

| Relation | On validation | On the target set |
|---|---|---|
| R1 Sub-Segment under Segment | 1.83% | 1.61% (421 of 26,192) |
| R2 Brand under Mnfr | 0.05% | 0.042% (11) |
| Pairs that could not be checked | 7,285 | **0** |

The rates hold on unseen data, which is the first evidence that the rules
generalize rather than describing the validation split.

The zero confirms what section 3 could only assert: the 7,285 unchecked pairs on
validation were an artifact of each target having its own validation set. In
serving, every available model predicts for every product, so the pair is always
complete.

The relations are still `candidate` and nothing is enforced. These 432 products
are a warn-mode queue, not a decision.

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
Tokenizing the same strings under 0.22.2 and 0.23.3 gives identical token ids,
so the pin is about reproducibility discipline, not a change in output.
