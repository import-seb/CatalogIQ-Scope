# Working data contract

This documents current code assumptions, **not a Product Owner-approved schema**.
Do not confuse a successful CSV parse with a structurally valid product record.

## Inputs and grain

| Item | Current behavior |
| --- | --- |
| Training | `data/provided/Selfcare_Training_data (1).csv`; labeled listings, some labels missing. |
| Prediction input | `data/provided/Selfcare_Target_data (1).csv`; target labels expected to be unpopulated, but not forcibly erased. |
| Required cleaning columns | Mnfr, Brand, Platform, Segment, Sub-Segment, TargetAgeGroup. |
| Other columns | Retained; complete source schema and column-level types are not yet enforced. |
| Parsing | UTF-8 CSV, pandas string dtype and default NA tokens. Leading-zero strings are preserved. |
| Missingness | pandas defaults for the cleaner. Notebook 09 uses a narrower explicit blank/null policy, so its counts can differ. |
| Record identity | Source filename + SHA-256 + zero-based parsed record position. This is provenance, not product identity. |
| Business grain | Listing/source record is the working unit. No validated canonical product key. |

Reserved output columns `dataset`, `source_sha256`, `source_row`, and `Q_REASON`
are rejected on input. The cleaner expects raw exports, not a previous output.
An existing run directory is rejected to prevent accidental overwrite.

## Label policy

- Preserve supplied Mnfr, including missingness. Flag populated values outside
  `J&J` and `All others`; do not infer it from Brand.
- Apply only the two accepted Sub-Segment replacements in the decision log.
- Preserve Brand, Platform, Segment, and TargetAgeGroup after parsing.
- Keep `UnItemised brand` and `Other Brands`; their taxonomy intent needs clarification.
- Do not transfer labels based on UPC, SKU, product name, or MDM ID.

## Outputs and invariants

Cleaned and quarantined outputs together retain every input row exactly once.
Reasons 2 and 3 are review-only. Reasons 4, 5, and 6 quarantine rows, including
rows that also have reason 2. Every label change needs old/new values, a rule, and source identity.
The target export is a parsed pass-through with provenance, not a claim that its
records passed structural screening. See [artifact definitions](target_cleaning.md).

## Unresolved contract questions

Confirm with the Product Owner: row grain; canonical entity identity; Exclude field
semantics; timestamp encoding; permitted label vocabularies and applicability;
meaning of missing versus not-applicable values; handling of malformed CSV records;
retailer-specific taxonomy rules; redistribution and retention terms.

Current heuristic flagging is not schema enforcement. In particular, lowercase
names and records without numeric content may be legitimate. A review is needed
before any exclusions are considered for model training.
