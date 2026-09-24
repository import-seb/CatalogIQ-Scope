# Feature cleaning and tooling validation — technical details

Status: initial conservative pass, 19 September 2026.
Companion to [the short validation report](03_feature_cleaning_validation.md). Sebastian owns target-label cleaning; this work covers non-target feature checks.

## Inputs and reproducibility

| Input | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| Selfcare_Training_data.csv | 84,552 | 32 | `00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452` |
| Selfcare_Target_data.csv | 28,082 | 32 | `0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db` |

The local copies match the supplied Downloads files byte for byte. Both parse as UTF-8 CSV with 32 fields per record. Embedded line breaks mean a record number is not a physical file line number.
Raw inputs live in ignored `data/provided/`; derived data and audit files live in ignored `data/derived/`. No product records are included in this report.

Run from the repository root with Python 3.12+; no extra packages are required:

```text
python scripts/clean_features.py --input data/provided/Selfcare_Training_data.csv --output data/derived/training-v1 --dataset training
python scripts/clean_features.py --input data/provided/Selfcare_Target_data.csv --output data/derived/target-v1 --dataset target
python -m unittest discover -s tests -v
```

Each output directory must be new. A successful run contains `summary.json`; an interrupted/failed run without that file is incomplete and must not be consumed.

## Scope and results

Only surrounding whitespace in selected feature strings and spacing around `>` in `ProductCategory` are normalized. Repeated category paths are flagged, not collapsed. Descriptions and contents are preserved.

| Result | Training | Target |
|---|---:|---:|
| Records retained | 84,552 | 28,082 |
| Records with formatting changes | 7,765 | 2,602 |
| Records with one or more review flags | 10,475 | 3,601 |
| Records removed | 0 | 0 |
| ProductCategory formatting changes | 7,745 | 2,594 |
| Explicit Exclude markers | 5,293 | 1,852 |
| Non-numeric ProductRating | 462 | 169 |
| Non-numeric ProductReviewsCount | 305 | 109 |
| Non-numeric ReviewsCount | 131 | 54 |
| Non-numeric XRatXRev | 198 | 76 |
| Fractional ProductReviewsCount | 145 | 57 |
| Fractional ReviewsCount | 180 | 67 |
| URL in MDM_Id | 174 | 55 |
| Non-numeric MDM_InsertDateTime | 284 | 108 |
| Missing ProductName | 151 | 43 |
| UPC scientific notation, preserved | 5,006 | 1,690 |

Flags overlap. They are review prompts, **not confirmed corruption counts or automatic exclusion decisions**. For example, scientific notation and Exclude have different meanings from malformed numeric fields. URL checks validate syntax only; no product URL was fetched.

Missing means an empty/whitespace-only value or case-insensitive `null`. This intentionally differs from pandas' larger default NA vocabulary, explaining small differences from the initial profile. Source null strings are retained. No imputation, statistical fitting, target-label inference, numeric casting, date conversion, row reconstruction or deduplication occurs.

The candidate files are **not model-ready**. They still contain flagged records, original missing tokens and original target values. One repeated JoiningKey group was found in training; JoiningKey alone is therefore not a safe merge key. No cross-dataset duplicate/leakage analysis has been completed.

## Integration contract for parallel scripts

- `features_candidate.csv` preserves all 32 column names, row count and input record order.
- `Category`, `Mnfr`, `Brand`, `Platform`, `Segment`, `Sub-Segment`, `TargetAgeGroup` are unchanged, including missing values and suspicious strings. Category is protected until its role is confirmed.
- `JoiningKey`, `Sku`, `Upc`, `ProductModelNumber`, `MDM_Id` remain exact source strings. Scientific notation is not expanded; lost leading digits cannot safely be reconstructed.
- `row_audit.csv` contains every record, keyed by `(dataset, source_sha256, source_row)`. `source_row` is a one-based CSV data-record index, excluding the header.
- Sebastian's script should use the same original input hashes and retain the same source-row identity before any filtering or sorting. Merge label corrections one-to-one on that identity; do not merge on row positions after filtering or only on JoiningKey.
- Numeric casting, date interpretation, final exclusions and label corrections remain explicit integration decisions. Apply an agreed row mask only after both audit outputs have been combined. Evaluation splits and model fitting remain Sebastian's work.

## Checks performed

Three automated tests passed: protected values and idempotent formatting; malformed numeric/URL values and repeated paths; multiline CSV round-trip, raw-file preservation and overwrite refusal.
An independent full-file comparison checked all 112,634 records for equal target-label/identifier values and maintained record order. Local raw-copy hashes were checked against the supplied originals. The script also verifies the source hash after processing.

Raven's installed secret scanner was run on the new cleaner and its tests; no blocking findings. It warned about missing ignore patterns for key/certificate files. The scanner was attached to this CatalogIQ checkout's local pre-commit hook. Full Raven routing/plugin integration, CVE/style enforcement and data-quality reasoning were **not** validated here. A secret scanner does not verify CSV cleaning correctness. Prior synthetic-secret/clean-commit tests were conducted in a separate demo repository, not on sponsor data.

## Aryx status

The earlier public-data demo passed entity/link/provenance checks, but its default Ask model returned an unfinished answer. Those results are not validation on CatalogIQ.
After Docker was started, tested Aryx 1.9.0 in a separate local workspace (ID 3) with Ollama (`qwen3.5:0.8b`, `lfm2.5-thinking:latest`). No cloud model was used.

Sample: the first five records with no current feature flags and first five with a non-numeric-field or URL-in-identifier flag from each input, in original record order (20 total). This is a deliberately selected smoke-test sample, not representative random sampling. Only eight fields were uploaded: explicit source identity/name, dataset, row index, ProductName, ProductBrand, ProductCategory, Retailer and JoiningKey. Structural issues in omitted fields were not tested inside Aryx. Target labels were not uploaded.

- Ingestion completed: 20 expected names became 20 entities; no unintended entity merges observed.
- All 20 entities retained source provenance. All source-row identities and JoiningKey strings matched the sample.
- Compared all 160 attribute values: 159 exact matches. One ProductName lost a trailing space. Thus exact source fidelity did not pass, although no other sampled field mismatch was found.
- Zero relationships were created. We pinned the type to CatalogProduct and supplied explicit record keys. Automatic ontology/type discovery and product-brand/category relationship correctness are untested.
- Ask failed the functional test: a question requesting the retailer of the first sampled record in one short cited sentence produced 2,097 characters of unfinished text and did not provide the expected retailer. The retailer exists in Data Explorer attributes. Ask returned one citation, yet its own grounding score was 0.2; citations/grounding metadata are not proof of a correct answer.
- Full-dataset performance, classification quality, malformed-row repair and protection from instructions inside product descriptions remain untested.

Reproduce using `python scripts/validate_aryx_sample.py prepare`, then `start`, `check` and `ask`. Wait for the ingest job to finish before interpreting checks or running Ask. The script uses only localhost and requires a ready Ollama provider before upload. `prepare` and `start` refuse to replace an existing validation run. The initial sample was prepared with equivalent local selection code before this helper was added. Local record-level evidence lives under ignored `data/derived/aryx-sample/`; do not commit it. Ask responses require human review, not simply checking for a citation.

Integration needed: decide whether source whitespace normalization is acceptable; expose relevant product attributes to the Ask context and resolve unfinished answers; define and separately test intended relationships. Aryx is not ready to act as a trusted CatalogIQ answering or classification component on the strength of this test.

## Remaining decisions

Confirm the role of Category, numeric-field semantics/ranges, timestamp format, how to treat Exclude, permitted missing-value tokens and a shared quarantine policy. Confirm the sample and runtime results before extending Aryx ingestion. No real dataset or derived row-level output should be committed without redistribution approval.
