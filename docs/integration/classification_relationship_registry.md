# Classification relationship registry

Version `relationships-20261107-candidate-v2`. Owner: Maria.

This closes the "versioned relationship registry" item in
`docs/integration/model_selection_handoff_20261106.md`, which says
"No complete approved map is currently loaded."

Files:

- `src/catalogiq/resources/relationships/classification_relationship_registry.json`
- `src/catalogiq/resources/relationships/segment_subsegment_map.csv` (the 51-pair
  curated map R1 is built from, so the registry is reproducible from the repo)
- `tests/test_relationship_registry.py` (9 tests)

Three relations: **R1** Sub-Segment under Segment, **R2** Brand under Mnfr, and
**R3** Brand to Platform, which is recorded as `rejected` so nobody builds it.
R1 and R2 are `candidate`. The file holds **zero approved relations**.

The registry carries no timestamps and no absolute paths, like the pinned
protocol manifest, so it is byte-reproducible: rebuilding from
`Selfcare_Training_data.csv` and the shipped map gives sha256
`c4703dac177a386ae7749e511442305cd7a5f69bfa91d8b777e8425d71858d78`. R2 and R3 are
derived from the training file; R1 comes from the map, which ships beside it.

## 1. What v2 corrects from v1

I built v1 from co-occurrence in validation truth. Two written rules exclude that
basis, and I found them only after measuring.

`docs/decisions.md`:

> Do not enforce a fixed Segment/Sub-Segment parent whitelist: shared parents
> can be legitimate. Single-parent count diagnostics do not change labels.

> Preserve supplied Mnfr, including missing and invalid values. Do not infer
> J&J from Brand.

The primitive's own docstring in `src/catalogiq/confidence_calibration.py`:

> No relation is inferred from label names, validation truth, or co-occurrence.

And the handoff, on any future rule:

> must use saved training support or a versioned reliability registry known at
> arrival, not hidden validation truth or audit support counts.

v2 is built from training support instead. What that changed:

| | v1 | v2 |
|---|---|---|
| R1 pairs | 45 | **51** |
| R1 children with two parents | 1 | **2** |
| R2 pairs | 132 | **144** |
| R2 verdict | unstable, do not load | **deterministic in training** |
| Brand/Platform | "cannot build it" | **recorded as not a relation** |
| Basis | validation co-occurrence | curated map plus training support |

### The v1 precision figure was not a result

I reported 194/194 = 100% flag precision. That was forced by construction: if the
predicted pair is absent from truth, both predictions cannot be correct, because
if they were, the pair would be in truth. The allowed set also came from the same
rows it was scored on. I should not have offered it as evidence.

In v2 the allowed set comes from training, independent of the rows being scored,
so a false alarm is now possible. **None occurred**, in either relation. That is a
real result, and it says every true pair in validation is present in the training
map.

## 2. The schema is the primitive's

`review_decision(..., approved_relations=...)` already reads this shape:

| Key | Read by the primitive as |
|---|---|
| `status` | anything but `"approved"` → `unapproved_relationship_rule` |
| `evidence` | empty or missing → `unapproved_relationship_rule` |
| `child_target`, `parent_target` | keys into the `classifications` dict |
| `allowed_pairs` | list of `[child_class, parent_class]`, **child first** |

Target names are `"Segment"`, `"Sub-Segment"`, `"Brand"`, `"Mnfr"`, `"Platform"`,
`"TargetAgeGroup"`, matching `structural.py`, `classical_data.py` and
`minilm_prediction_export.py`. Sebastian should confirm his adapter uses the same
keys, because they are dictionary lookups and a different spelling fails silently
as `missing_relationship_classification`.

Unknown keys are never read, so support counts, provenance and measurements
travel inside each relation without breaking it.

### The loading rule matters

A relation whose status is not `approved` appends `unapproved_relationship_rule`
to **every single product**. Confirmed against the real function:

```
review_decision(..., approved_relations=[R1], classifications={"Sub-Segment": "Allergy", "Segment": "Allergy"})
  -> ['unapproved_relationship_rule']
```

So loading this file unfiltered sends the whole catalog to review. The adapter
must filter to `status == "approved"` first. That filter yields an empty list
today, which is the correct behaviour: nothing is enforced until someone approves
it. A test locks this.

### Read it as UTF-8 explicitly

The files are UTF-8 with LF endings and no BOM. The registry contains one brand
with a non-ASCII character, `Horbäach`. Python on Windows picks the locale
encoding when `open()` is called without `encoding=`, which is cp1252 on most
machines, and that name fails to decode. Pass `encoding="utf-8"`. The test does,
and CI runs it on both ubuntu-latest and windows-latest.

## 3. R1, Sub-Segment under Segment

**51 pairs over 49 Sub-Segments and 8 Segments**, from the curated map covering
76,500 training rows. I verified the curated map is exactly the full training
pair inventory once two corrupted pairs are removed, and the build asserts that
equality so it fails if it stops holding.

The two rejected pairs are column-shift corruption, product attribute text landing
in the Segment and Sub-Segment columns:

| Rejected pair | Rows |
|---|---|
| `gluten-free` under `Keto certified` | 1 |
| `lactose` under `IGEN Non-GMO tested.* No soy wheat` | 2 |

Measured on Sebastian's `predictions_contract_v2_20261031` validation predictions:

| | R1 | R2 |
|---|---|---|
| Rows scored | 10,497 | 9,833 |
| Fires | 192 (1.83%) | 5 (0.05%) |
| Rows with at least one wrong prediction | 1,243 (11.8%) | 202 (2.1%) |
| **Recall of those errors** | **15.5%** | **2.5%** |
| False alarms (both labels right) | 0 | 0 |
| Errors a 0.70 both-confident gate would skip | 53 | 2 |

Recall is the informative half and it is modest. R1 is a cheap extra signal on top
of confidence, not a detector. Its value is the 53 errors that a confidence gate
would let through.

### Scope limit on these rates

Every target has a different `validation_record_ids_sha256`:

| Target | validation set |
|---|---|
| Segment | `47f60a59…` |
| Sub-Segment | `39a929e0…` |
| Brand / Brand lookup | `87cd25d1…` |
| Mnfr | `4e117e75…` |
| TargetAgeGroup | `ed225324…` |

So the 10,497 rows are the ones that are validation for **both** models, not a
representative product sample. Both predictions are out of sample, so there is no
leakage, but the firing rate is measured on an intersection and the per-product
rate will differ. The review policy proposal still says "No full-product workload
has been measured," and this does not close that.

### The Exclude column is dirty

5,293 training rows are marked `Exclude`, and 4 more rows have product description
text sitting in that column instead of a marker. It is also a nullable string
column, so `frame["Exclude"].eq("Exclude")` returns `pd.NA` on nulls and negating
that mask silently drops 79,255 rows. I hit that exact bug while building this.
Use `fillna("")` first.

**Four real taxonomy pairs appear only in excluded rows**, so they stay in
`allowed_pairs`:

| Pair | Rows, all in Exclude |
|---|---|
| `Motion Sickness` under `Digestive Health` | 408 |
| `Hemorrhoid Remedies` under `Digestive Health` | 210 |
| `Anti-Nausea` under `Digestive Health` | 57 |
| `Creams & Gels (Rubs)` under `External Analgesics` | 1 |

A rule built from kept rows alone would flag all four as violations.

## 4. R2, Brand under Mnfr

**144 pairs, and every one of the 144 brands carries exactly one manufacturer**
across 79,259 training rows with Exclude rows dropped. The relation is
deterministic in training support.

**I withdraw the v1 finding.** v1 said 78% of half-splits produced a false alarm
and that R2 should not be loaded. That was a coverage artifact of deriving the map
from a 9,833-row validation slice, where rare brands drop out and a correct
prediction then looks like a violation. It was not a property of the relation.

Two details that do hold:

- The contradictions are all in excluded rows. `Benadryl`, `Bengay`, `Sudafed`
  and `Tylenol` each show a second manufacturer, but only in rows marked
  `Exclude`.
- `Mnfr` has two values outside the `J&J` / `All others` vocabulary, both column
  shift: `coffee`, and a collagen description. Those fall under target reason 6
  quarantine, which `decisions.md` says does not change the source label. They
  are excluded from `allowed_pairs`.

R2 may raise a review reason and nothing else. Per `decisions.md` it must never
rewrite Mnfr from Brand.

## 5. R3, Brand to Platform is not a relation

The ticket asks for incompatible Brand and Platform combinations. The data says
that question does not have a hierarchical answer.

Only **1,103 of 79,259 training rows carry a Platform label (1.4%)**, across 62
classes. Of the 17 brands that carry one, **12 carry more than one**:

| Brand | Distinct platforms |
|---|---|
| Tylenol | 22 |
| Benadryl | 8 |
| Motrin | 7 |
| Zyrtec | 6 |
| Excedrin | 6 |
| Zarbee's | 5 |
| Sudafed | 4 |

Tylenol spanning 22 platforms is normal, not a conflict. So an unseen
Brand/Platform pair is evidence of thin coverage, not of a taxonomy violation,
and a compatibility rule here would generate false flags by design. Platform is
also unavailable per the handoff: 247 labelled validation rows against a 57-class
vocabulary, 27 of those classes at zero support, and only 32 appearing in truth.

I recorded this as `rejected` with the counts rather than leaving it open, so the
next person does not spend the sprint on it. If the sponsor wants Brand/Platform
checked, it has to be a coverage or drift signal, not a parent-child rule.

## 6. Why only these two relations, out of thirty

The ticket asks which relationships between the targets are valid. I screened all
30 ordered pairs of the six targets over the training rows, counting how many
children carry more than one parent. A relation is only usable as a parent-child
rule if nearly every child has one parent.

| Child to parent | Rows | Children | Children with 2+ parents | Most parents on one child |
|---|---|---|---|---|
| **Brand → Mnfr** | 75,053 | 146 | **0** | 1 |
| **Sub-Segment → Segment** | 75,039 | 47 | **2** | 2 |
| Platform → Mnfr | 1,103 | 62 | 4 | 2 |
| Platform → Brand | 1,103 | 62 | 11 | 3 |
| Mnfr → Segment | 75,045 | 4 | 2 | 8 |
| TargetAgeGroup → Segment | 75,041 | 6 | 3 | 8 |
| Sub-Segment → Mnfr | 75,039 | 47 | 24 | 2 |
| Brand → Segment | 77,449 | 146 | 116 | 8 |
| Sub-Segment → TargetAgeGroup | 75,035 | 47 | 41 | 3 |
| Brand → Sub-Segment | 75,039 | 146 | 131 | 45 |

The two I built are the only hierarchical ones in the data. Everything else is
many-to-many, and three results are worth stating because they rule out rules
that sound reasonable:

- **Brand does not determine Segment.** 116 of 146 brands span more than one
  Segment, up to 8. A Brand/Segment consistency check would fire constantly.
- **Sub-Segment does not determine TargetAgeGroup.** 41 of 47 children span
  several age groups.
- `Platform → Mnfr` is the third most hierarchical and the only remaining
  candidate, but it rests on 1,103 rows and Platform is unavailable. Parked, not
  rejected on evidence.

So the registry is not a sample of the relationships. It is the complete set of
the ones the data supports.

## 7. What approval needs

Two decisions and an approver. The basis is no longer the blocker.

**Multi-parent children.** Both are valid under `decisions.md` and neither can be
whitelisted away without a decision:

| Child | Parents | Training rows |
|---|---|---|
| `Probiotics` | `Vitamins, Minerals & Supplements` | 634 |
| | `Digestive Health` | 22 |
| `Internal Pain` | `Internal Analgesics` | 1,975 |
| | `External Analgesics` | 1 |

`Probiotics` is Max's demo case. Culturelle Daily Probiotic is the listing he
proposed as a review example, and he is right that similar Culturelle listings
carry different Segment labels. The handoff already treats a retailer `Vitamins`
breadcrumb disagreeing with a curated `Digestive Health` label as
`retailer_taxonomy_differs`, which is **Informational**, "Preserve both", and
explicitly "must not independently reject a classification." So this is a real
shared parent unless the sponsor says the 22 rows are errors.

`Internal Pain` under `External Analgesics` is a single row and looks like a data
error, but the same rule applies: it needs a decision, not a whitelist.

Be honest about scale when presenting. These are 22 rows and 1 row. The larger
Vitamins and Digestive Health confusion that accounts for roughly a third of
Segment errors in every model is a Segment-level problem, not this pair.

**An approver**, and a record of what source they approved. The curated map is the
candidate source.

Until then R1 and R2 run in warn mode in evaluation only, and the adapter enforces
nothing.
