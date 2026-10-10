# Reproduce the finalized Segment splits

The team workflow is `scripts.split_data`. It reproduces the October 9 decisions;
this integration introduces no new grouping rules, threshold changes, or final
test. Historical experiment directories remain unchanged.

The previous public command used V3 and reallocated all three partitions. The
finalized workflow instead used V4, exact-input constraints, historical exposure
exclusions, and a development-only allocator. Its local paths and ignored archives
were the portability break; the packaged protocol now preserves those decisions.

## Commands

From the repository root, after obtaining the private CSVs in `data/provided/`:

```bash
python -m pip install -e ".[splitting]"
python -m catalogiq --mode integrated --output-dir data/processed/cleaned
python -m scripts.split_data --input data/processed/cleaned/training_candidate.csv --output-dir data/processed/splits
python -m scripts.split_data --verify data/processed/splits
python -m scripts.train_frozen_segment --output-dir data/processed/segment_development_preparation
```

Use new output directories. The last command only verifies and prepares training
and validation; it does not train or score a model. Core dependencies include
the frozen tokenizer implementation; the `splitting` extra pins the tested
numerical dependency versions. No local research archive or pretrained
model weights are required to clean, split, or prepare development records.

## Source of truth

| Identity | Meaning |
|---|---|
| `dataset`, `source_sha256`, `source_row` | Original CSV-record identity; row numbers are one-based parsed records |
| `src/catalogiq/resources/segment_final_20261009/manifest.json` | Full source hashes, grouping/allocation settings, tokenizer contract, exposure selector, and expected assignment membership hashes |
| Sibling `eligibility.json` and `tokenizer/` | Non-sensitive source-row exposure selectors and public frozen tokenizer files |
| Generated `grouping_freeze.json` | The algorithm/configuration and protocol used for this run |
| Generated `assignments.csv` | Final source identity, record ID, product group, and split membership |
| Generated `rule_assignments.csv` | Exact alias of `assignments.csv`, retained for compatibility |
| Generated `completion.json` | Hashes of generated files using paths relative to the split directory |

Required source SHA-256 values:

```text
Training: 00f1fcd0d45bc4196ea8c0e9fffa63897c8e24d12a7639468928fbd6afff4452
Target:   0122a49d6e1ca5f4f0fb7a2ab381418bf0e63d9c5dc3e915d64e259a973951db
```

The older `c2f0e766` prefix identifies a V3 assignment CSV, not the training source
or the finalized assignments. Use the complete identities in the protocol manifest.

Fresh integrated cleaning reproduces the historical `training_candidate.csv`
byte hash:

```text
1878e3948e8a840b3922780b7304e33d39d0f1e3b5d7a579bf56639d3cdf8fb4
```

All 83,938 retained source identities match the October 9 universe. The older
`training_cleaned.csv` was a projection of this candidate file, with identifiers
joined back afterward. Direct candidate input has identical matching attributes
and avoids that extra export/join step. Cleaning rules are unchanged.

## Membership and protection

Grouping uses the existing corrected V4 method and frozen 128-token input-equality
constraints. The committed exposure selector reproduces the historical eligible
pool; corrected whole-group closure reserves the same final test. Development
allocation uses the existing seeded Segment/retailer/group-size allocator.

| Partition | Records | With Segment labels |
|---|---:|---:|
| Train | 59,111 | 55,776 |
| Validation | 12,667 | 11,952 |
| Final test | 12,160 | 11,477 |

Verification binds stable identities, group membership, every split assignment,
and the exact protected-test membership to complete authoritative hashes. It also
checks group isolation, exposure exclusions, full exports, and identical-input
constraints. Matching row counts or class distributions alone is insufficient.
An incompatible input or changed assignment fails verification.

The authoritative assignment digest is
`1620bba4a36f86a3f3d6f3a014cf1df6611228567fbe0e0822bff516c7f98d4a`.
The final-test ID digest is
`443b96d4028166b9eb0a254aa42c767f35a86f7448c7d009ad1abe73420547bf`.
These are full canonical membership hashes, not source-file prefixes. The manifest
specifies UTF-8 compact JSON rows sorted by immutable record ID, each followed by
LF. This verifies logical equivalence despite input row order or CSV serialization.

## Actual verification on October 10

Fresh official cleaning and splitting reproduced all **83,938** October 9
provenance/record/group/split mappings with **zero differences** in every role.
Final-test membership and exposure exclusions match exactly. All 17 archived
finalization artifact hashes remain unchanged.

The Segment distribution table and all **9,362** independent audit pairs,
including scores and split roles, match the historical results. There are
52,393 groups, maximum size 90, with zero exact-payload or identical-transformer
input crossings. The regenerated CSV omits Segment from assignment metadata,
so its byte hash differs from the historical assignment CSV; the complete
canonical membership digest above is identical.

**390 automated tests pass**, and the package wheel builds. Local generated
evidence is in `data/processed/split_pipeline_integration_20261010/`:
`actual_frozen_comparison.json` records the complete comparison, and
`guarded_development.json` records an actual prepare-only run with file-access
guards. That run protected all 12,160 test IDs, prepared 55,776 labeled training
and 11,952 labeled validation records, and made **zero `test.csv` open calls**.
No model was trained or evaluated.

An isolated checkout containing repository files and only the two source CSVs
also ran the official workflow using the built wheel, with Python source files
normalized to LF and tokenizer bytes preserved. It passed all 390 tests and public
verification, matched every October 9 assignment, and produced byte-identical
assignments and train/validation/test CSVs. No historical research files were
copied. Proofs are in `.cache/split_integration_20261010/fresh_*.json`.

## Use the development partitions

Use generated `train.csv` and `validation.csv` for fitting models; no assignment
join is required. The guarded development loader verifies the portable seal,
excludes missing targets from supervision, checks record/group isolation against
the protected ID mapping, and never opens or hashes `test.csv`.

```python
from catalogiq.segment_frozen_development import load_frozen_development

development = load_frozen_development("data/processed/splits")
train = development.frame.iloc[development.train_indices]
validation = development.frame.iloc[development.validation_indices]
```

Model features remain ProductName, ProductBrand, ProductDescription, and
ProductContents. The generated model configuration and tokenizer identify the
fixed representation. Additional target fields, identifiers, provenance, and
split metadata are excluded from model features. Training requires the optional
training dependencies and pinned pretrained encoder, then an explicit `--train`.

## Historical artifacts and residual risk

Older `rule_assignments.csv` files describe their respective experiments. The
local `scripts.finalize_splits` workflow remains available for historical
reconciliation; it is not required for team reproduction. Only assignments from
the current authoritative `scripts.split_data` workflow should be used now.

The [October 9 findings](02_finalization_20261009.md) remain applicable: zero exact
crossings does not establish zero semantic leakage. Suspicious lexical matches,
copied descriptions, and ambiguous broad groups remain documented risks. No new
test examples or model performance are used for this integration. Keep the final
test closed until the model and evaluation procedure are selected.
