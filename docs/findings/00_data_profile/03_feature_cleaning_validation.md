# CatalogIQ — initial tooling validation

Maksim Pikalov · 19 September 2026

**Status: preliminary validation, not completed integration or model-ready cleaning.**
Sebastian handles target-label cleaning; this work covers non-target feature checks.

## What was tested and found

| Test | Result |
|---|---|
| Feature cleanup on both supplied CSVs | Processed 84,552 training and 28,082 target records. Normalized selected surrounding whitespace and category separators; removed no rows. |
| Data preservation | Full-file comparison confirmed unchanged target labels, identifiers and record order. Raw-file hashes matched the originals. Three automated tests passed. |
| Review flags | Flagged 10,475 training and 3,601 target records. These are review candidates, not confirmed bad rows: flags include existing Exclude markers and scientific-notation UPCs. |
| Raven secret scanner | No blocking findings in the cleaner, its tests or the Aryx validation script. Reported missing ignore patterns for key/certificate files. Local pre-commit scanner connected to CatalogIQ. |
| Aryx ingestion and provenance | Loaded a deliberately selected 20-record, eight-field sample into local Aryx 1.9.0 with Ollama. All 20 entities, source references and record identifiers were retained. |
| Aryx attribute preservation | 159 of 160 values matched exactly. One product name lost a trailing space. |
| Aryx Ask | Failed: a simple retailer question produced unfinished text rather than the expected answer, despite the retailer being present in stored attributes. |

## Limitations

- Raven checked code for secret patterns; it did not validate cleaning logic. Full routing/plugin integration and CVE/style checks remain unverified.
- Aryx used a small, non-random feature-only sample, with an explicit entity type and record keys. Target labels were not uploaded. No relationships were created, and relationship discovery, classification accuracy and full-dataset performance were not tested.
- The cleanup preserves suspicious values for review. It does not repair shifted rows, infer labels, cast numeric/date fields, impute values or produce a final exclusion mask. The candidate files are not ready for modeling.

## What still needs to be integrated

1. Agree with Sebastian on the merge key: `(dataset, source_sha256, source_row)`, where source_row is the one-based original CSV record number. Training contains a repeated JoiningKey group, so JoiningKey alone is unsafe.
2. Agree on Exclude handling, numeric/date interpretation, missing-value tokens, Category's role and which flagged records to quarantine. Combine both scripts' audit results before filtering or splitting.
3. Resolve Aryx's unfinished answers and ensure Ask receives the relevant product attributes. Define and test the intended product-brand/category relationships before adopting that integration.

Raw and derived CSVs remain local and Git-ignored. Only code, tests and documentation are intended for review.

[Detailed counts, source hashes, reproduction commands and integration contract](04_feature_cleaning_reproduction.md).
