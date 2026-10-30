# Brand and Sub-Segment development allocation proposals

Prepared October 29, 2026. These are allocation proposals for review, not model
results. No models were trained and no predictions were inspected.

## Modeling scope decision

Platform is excluded from MiniLM training because its labeled population and
independent class support are sparse. Simpler classification methods will be
evaluated separately later. Its existing allocation and tests are preserved.
The remaining MiniLM targets are Segment, Mnfr, Brand, Sub-Segment and
TargetAgeGroup. This change implements allocation only, with no training or
inference workflow.

## Shared implementation and protections

`catalogiq.target_development_allocation` provides `TargetAllocationConfig`,
`allocate_target_development(prepared, config=None)` and
`write_target_proposal(split_directory, output_directory, target_column, config=None)`.
The target is selected during preparation and may be Brand, Sub-Segment or
Platform. The Platform module retains its public compatibility API and uses the
same engine. Its small-population algorithm and configuration remain unchanged.

Every proposal reuses frozen final V4 `group_id` values, including exact-input
must-links. All 71,778 development records are retained, including target-missing
rows. Original labels and roles remain auditable; missing labels are excluded
from class counts and supervised eligibility. Unlabeled-only groups retain their
original role; all members of other groups move together.

All 12,160 protected test records in 11,979 groups remain fixed. Their identity
file is copied byte-for-byte. Test features, labels and predictions are not read.
No Segment V4 source or frozen artifact is changed.

The allocation order is:

1. Require training examples from every development-observed class. Pin singleton
   class groups to training and respect every shared-group consequence.
2. Prove maximum feasible validation class coverage with integer constraints.
3. Prove the nearest feasible labeled validation count to the existing 3/17
   development fraction. For large problems, an integer rounding lower bound
   avoids searching for a fractional record count.
4. Improve the existing quadratic class, labeled-retailer and group-size margin
   objective using seed 42 and whole-group exchanges. Large populations start at
   a minimum-change V4 solution satisfying coverage/count constraints and use
   bounded, seeded exchanges between groups of equal labeled size. Composition
   is a local-search result, with no global optimality claim.

## Original V4 versus proposed development

Distribution differences below are maximum absolute deviations of validation
fractions from the labeled development pool, in percentage points.

| Measure | Brand: V4 → proposed | Sub-Segment: V4 → proposed |
|---|---:|---:|
| Training classes | 142 → 143 | 48 → 49 |
| Validation classes | 133 → 138 | 45 → 47 |
| Labeled training records | 58,882 → 58,888 | 53,967 → 53,956 |
| Labeled validation records | 12,625 → 12,619 | 11,551 → 11,562 |
| Labeled validation share | 17.6556% → 17.6472% | 17.6303% → 17.6471% |
| Maximum class difference | 0.590794 → 0.521739 pp | 1.266856 → 0.021368 pp |
| Maximum retailer difference | 0.015832 → 0.003700 pp | 0.098888 → 0.004070 pp |
| Validation classes with 1–4 records | 7 → 8 | 3 → 4 |
| Validation classes with 1–4 groups | 30 → 39 | 4 → 5 |

Brand has 33,280 proposed training groups and 7,134 validation groups; Sub-Segment
has 33,282 and 7,132. Brand changes 2,642 development group roles (4,256 records);
Sub-Segment changes 1,728 (2,241 records). Coverage and counts refer only to
development-observed classes, not the complete taxonomy including test-only labels.

## Validation support and limitations

| Proposed support | Brand classes by records / groups | Sub-Segment classes by records / groups |
|---|---:|---:|
| Exactly 1 | 4 / 18 | 1 / 2 |
| Exactly 2 | 2 / 7 | 3 / 3 |
| Exactly 3 | 1 / 8 | 0 / 0 |
| 4–9 | 14 / 34 | 3 / 4 |
| 10+ | 117 / 71 | 40 / 38 |

Brand's eight classes below five validation records are Dimetapp (4 records,
1 group), Estroven (1, 1), Love Wellness (2, 2), Nasonex (2, 2), NaturaNectar
(1, 1), Rolaids (3, 1), Voltaren (1, 1) and Wile (1, 1). They account for
15/12,619 validation records (0.119%). Five singleton classes remain training-only:
Bonafide, Health & Her, Hers, Meno and Womaness.

Sub-Segment's four classes below five validation records are Dermatologicals
(2 records, 1 group), Devices/Refills (1, 1), Remaining Internal Pain (2, 2) and
Stimulants (2, 2). They account for 7/11,562 records (0.061%). Creams & Gels (Rubs)
and Daily oral contracception remain training-only because each has one
development group.

These proposals improve coverage without supplying additional rare-class data.
Per-class scores for the listed classes remain unstable. Brand also has 39
represented classes below five independent validation groups, up from 30 in V4;
better class coverage does not imply uniformly stronger support. Preserve the
actual class/group denominators in future evaluation and state which classes
are absent from validation. Class-balanced metrics and aggregate distribution
balance are different quantities; no model-quality improvement is established.

## Artifacts and verification

Source splits: `data/processed/split_integration_reproduced_20261010/`.

- Brand: `data/processed/brand_development_allocation_v1_20261029/`.
- Sub-Segment: `data/processed/sub_segment_development_allocation_v1_20261029_r2/`.
- Preserved Platform: `data/processed/platform_development_allocation_v1_20261029_r2/`.

Each new proposal contains assignments, class coverage, distribution differences,
validation-support summaries, source/configuration evidence and an output seal.
Independent CSV checks confirmed all 83,938 record identities and group mappings,
exact test preservation, complete development membership, training coverage and
reported support. All source labels remain unchanged. Existing Platform CSV
hashes and its full allocation summary match recomputation through the shared
engine. Existing Platform artifact checksums and frozen Segment source hashes
remain unchanged.

All 47 focused tests passed. They exercise deterministic preparation/allocation, group/test isolation,
shared singleton constraints, exact small-case coverage, large-population integer
counts, Platform regression, and proposal seals. No allocation was promoted or
merged automatically.
