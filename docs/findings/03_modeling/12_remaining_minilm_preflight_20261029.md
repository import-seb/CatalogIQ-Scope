# Remaining MiniLM targets: preflight stopped at the Brand leakage gate

The requested three-target training batch was stopped before GPU training.
The user's explicit safeguard requires a stop when ProductBrand directly
duplicates the Brand target or another serious leakage concern is found.
Preflight established extensive direct target echo for named brands. No model
was initialized or trained, and no checkpoints, model output directories,
prediction exports or handoff ZIPs were created for this batch.

No Python training process remained from the interrupted preceding request.
The requested execution order was TargetAgeGroup, Brand, Sub-Segment. Read-only
Brand auditing was performed before the first training run so the stop condition
could be checked before spending GPU time.

## Verified development populations

| Target | Allocation | Labeled train | Labeled validation | Training / validation classes present | Training status |
|---|---|---:|---:|---:|---|
| TargetAgeGroup | Frozen V4 | 54,282 | 11,624 | 3 / 3 | Held at batch preflight gate |
| Brand | Approved Brand v1 | 58,888 | 12,619 | 143 / 138 | Stopped: direct target echo |
| Sub-Segment | Approved Sub-Segment v1 r2 | 53,956 | 11,562 | 49 / 47 | Held at batch preflight gate |

`prepare_target_minilm` verified each target's ordered vocabulary, approved
allocation seal where applicable, record membership, frozen product group
assignments and train/validation/test isolation. All declared classes have
training support. The protected population remains 12,160 records. Its identity
metadata was used solely for isolation checks; `test.csv` was not opened or
hashed, and no protected-test features or labels were inspected.

Full ordered vocabularies, per-class record/group support, configuration and
checks are saved in:

`data/processed/minilm_remaining_targets_preflight_v1_20261029/audit.json`

## Brand: stop condition

Named-brand figures below exclude only `UnItemised brand` and `Other Brands`
for the purpose of this diagnostic. Training eligibility and allocations were
not changed. Those two classes account for 8,304 of 12,619 validation records,
which obscures the extent of direct copying among named brands when looking
only at the entire validation population.

| ProductBrand versus Brand | Training | Validation |
|---|---:|---:|
| All labeled records | 58,888 | 12,619 |
| Raw string equality, all records | 16,455 (27.94%) | 3,584 (28.40%) |
| Normalized equality, all records | 18,238 (30.97%) | 3,961 (31.39%) |
| Named-brand records | 19,763 | 4,315 |
| Raw string equality, named brands | 16,455 (83.26%) | 3,584 (83.06%) |
| Normalized equality, named brands | 18,238 (92.28%) | 3,961 (91.80%) |

Normalization uses the existing plain-text cleanup, Unicode NFKC, case folding
and surrounding-whitespace removal. Raw equality requires no normalization at
all. Examples of exact field/target pairs include `Nature Made`, `Nature's
Bounty`, `CVS Health` and `Walgreens`.

Across all labeled development records, 137 of 141 named-brand classes have
some exact normalized copies, 119 have at least 90% exact copies, and 66 have
exact copies on every record. The full columns are not identical; the detected
problem is extensive direct echo within the named-brand classification task.

A diagnostic lookup fitted on training ProductBrand values only reinforces
the concern. It covers 10,454 validation records (82.84%) and correctly maps
10,203 of those records (97.60% accuracy on seen feature values). Among named
brands, it covers 4,283 of 4,315 validation records (99.26%) and correctly maps
4,278 of those (99.88% on seen values). No validation labels were used to fit
the lookup, there is no unseen-value fallback, and no record-level lookup
predictions were exported. These are feature-redundancy diagnostics, not MiniLM
performance results.

Direct input/target redundancy is established. Whether this is impermissible
label leakage depends on the business task and feature provenance: raw-to-curated
brand canonicalization can legitimately use a raw retailer brand available at
inference. The existing evaluation strategy already flags that distinction and
the raw-to-curated relationship remains an open question in the relationship
notes. The audit does not establish that ProductBrand was populated from curated
labels. Under the user's explicit stop condition, this ambiguity and the extent
of direct copying require resolving the Brand feature contract before training.

## Other target feature checks

**TargetAgeGroup:** ProductCategory never equals the target, either as a whole
field or as its final breadcrumb component, in the labeled development data.
Age-label words occur in some approved product text, including 334 validation
ProductName fields and 78 ProductCategory fields. Those can be legitimate
product attributes; these string checks found no direct column-copy pattern.
They do not prove the absence of all proxy leakage or establish serving-time
provenance. Validation support is Adult 10,914; Children 572; Infant 138, so
overall accuracy would be dominated by Adult and macro-F1 remains important.

**Sub-Segment:** ProductCategory never equals the complete target field. The
last breadcrumb component equals the target on 432 of 11,562 validation records
(3.74%); the target label occurs as a word/phrase within the breadcrumb on 5,173
records (44.74%). A train-only full-breadcrumb lookup covers 99.26% of validation
records and is correct on 76.59% of those seen values. This is a strong semantic
feature, with some direct label phrases, rather than a complete target copy.
Repository documentation describes ProductCategory as the retailer's taxonomy,
distinct from CatalogIQ's curated taxonomy. No comparable direct-column-copy
condition was established by this audit; feature provenance still matters.

## Outputs and next decision

All three targets have **no new MiniLM validation metrics, confusion matrices,
training runtimes, peak training GPU-memory measurements or handoff artifacts**
from this request because the batch was stopped before training. Existing Mnfr
experiments and handoff artifacts are unaffected.

Resolve whether Brand is intentionally evaluated as raw-to-curated normalization
with ProductBrand available, or approve a revised feature/evaluation contract.
ProductBrand was not silently removed or masked; the five-field representation
and all hyperparameters remain unchanged. No retraining, extra experiment,
prediction-export implementation, frozen trainer edit, artifact modification or
merge occurred.

After the Brand gate is resolved, the requested sequential training can resume
under an explicit agreed contract. No model-selection or inference workflow was
added during this stopped preflight.
