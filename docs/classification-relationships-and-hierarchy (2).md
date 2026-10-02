# Classification Relationships and Hierarchy — CatalogIQ

**Status:** verified against data. Row-level results from the 2026-10-02 handoff.
**Scope:** relationships among the CatalogIQ classification targets, and whether
they can serve as validation rules or supporting evidence
**Location:** `docs/research/`
**Notebook:** `notebooks/01-classification-relationships.ipynb`
**Data basis:** `training_cleaned.csv` from the conservative handoff
(`conservative_handoff_20261002`), 78,718 rows. Supersedes the earlier version of
this document, which was written from the dataset profile without row-level
access. Claims that changed are marked **CORRECTED**.

---

## 1. Summary

The goal of this ticket is to find out whether the relationships between the
classification targets are regular enough to be used as validation rules.

**One is. Two are not. One was overstated.**

| Relationship | Earlier claim | Verified result | Rule status |
|---|---|---|---|
| Brand → Mnfr | Strong lookup | **Deterministic**, 141 named brands, 0 violations | **R2 → hard rule** |
| Segment → Sub-Segment | Strict hierarchy | **CORRECTED — not strict.** 2 sub-segments have 2 parents | R1 stays warn mode |
| Platform → Brand | Strong, hard rule | **CORRECTED — not nested.** 11 Platform values span brands | R3 stays warn mode |
| Platform scope | Labeled only for one manufacturer | **CORRECTED — 93.4%, not exclusive** | R4 as a soft rule |
| ProductCategory → Segment | Soft mapping | Confirmed soft. 12 strong prefixes at depth 2 | R5 review-only |
| TargetAgeGroup | Attribute, not hierarchy | Confirmed | Soft check only |

**Two open questions are now closed, and both close badly:**

- **`Category` carries one value**, `Self Care`, in all 78,718 rows. It is a
  constant. Not a target, not a feature.
- **`JoiningKey` is unique per row.** It is a row identifier, not a grouping key,
  so it cannot be used to prevent near-duplicate leakage in the train/test split.

---

## 2. Method

The earlier version of this document was written from the dataset profile —
counts and missingness per column, with no access to the rows. Counts are enough
to rule a relationship out (a child cannot have more rows than its parent) but
not enough to confirm one, because confirming requires a crosstab.

Every claim in that version was marked as demonstrated or assumed, with the query
that would settle it. `notebooks/01-classification-relationships.ipynb` runs
those queries. This document reports what came back.

**Important caveat on the data used.** These results come from the *cleaned*
handoff, which excludes 5,834 of the original 84,552 rows (466 quarantined,
5,368 held for review). A relationship that holds in the cleaned data might not
hold in the full data, and a class that is absent here might exist there. See
`docs/validation/cleaning_validation.md`, which records five audit criteria as
BLOCKED for want of the raw file.

---

## 3. Three kinds of relationship

Unchanged from the earlier version, and the distinction held up.

**Taxonomy — a tree, "is a type of".** Segment → Sub-Segment. Acid Relief *is a
type of* Digestive Health. Yields a hard rule, if strict. It is not strict (§4.1).

**Ontology relationship — an edge between different kinds of thing.** Brand →
Mnfr. Tylenol is not *a type of* Johnson & Johnson; it is *made by* it. Yields a
consistency rule. Verified deterministic (§4.3).

**Attribute — a property of the product.** TargetAgeGroup. Owns no children,
yields at most a soft plausibility check (§4.6).

---

## 4. Relationship by relationship

### 4.1 Segment → Sub-Segment — CORRECTED: not a strict hierarchy

**Earlier claim:** "the hierarchy exists and is probably strict."
**Verified:** 8 Segments, 45 Sub-Segments, 47 parent-child pairs. **Two
Sub-Segments have two parents**, so the taxonomy is a graph, not a tree.

| Sub-Segment | Parents | Reading |
|---|---|---|
| `Internal Pain` | Internal Analgesics 1,941 · External Analgesics 1 | The single row is almost certainly a labelling error |
| `Probiotics` | Vitamins, Minerals & Supplements 634 · Digestive Health 19 | Genuine ambiguity — probiotics legitimately sit in both |

These are different problems. The first is one bad row and should be corrected.
The second is a real taxonomy decision and belongs to the PO.

**Rule R1 stays in warn mode** until both are resolved. A hard rule today would
block 20 legitimate rows.

**The CCFS group — confirmed, with a correction.** The earlier version's
strongest evidence was that seven Sub-Segments summed exactly to the CCFS count
of 4,267. Six of those seven are confirmed children. The seventh, `Cold / Flu`
(4 rows), no longer exists in the cleaned data.

| Sub-Segment | Count |
|---|---|
| Sore Throat / Cough Drops | 1,317 |
| Cold/Flu | 1,281 |
| Cough | 577 |
| Nasal Sprays | 463 |
| Sinus excl Sprays | 424 |
| External Upper Respiratory | 147 |
| **Total** | **4,209** |

**The substantive claim survives: `Nasal Sprays` belongs under CCFS, not under
Allergy**, even though nasal sprays are an allergy product in ordinary use. That
had to come from the data rather than from domain intuition, and it did.

**The Internal Analgesics coincidence — confirmed.** The earlier version noticed
a 1,065-row gap under Internal Analgesics that happened to equal the Sleep Aids
count exactly, and said: "That is interesting, and it is also the kind of
coincidence that method B produces and cannot resolve. Do not write it down as a
finding. Test it." Tested: **Sleep Aids (1,060) is a child of Internal
Analgesics.** The restraint was correct and so was the hunch.

**Ear Care — elimination confirmed, parent identified.** The earlier version
proved Ear Care (3,095) could not sit under Other Self Care (1,135) by counts
alone. It sits under **Lifestyle CHC** (Ear Care 3,090 of that Segment's 4,540).

The full map is in `interim/segment_subsegment_map.csv`.

**Rows with a Sub-Segment but no Segment: 0.** The earlier version flagged 4,789
such rows as the caveat that kept the upper-bound elimination from being airtight.
In the cleaned data there are none, so that caveat no longer applies here —
though it may still apply to the raw file.

### 4.2 Near-duplicate labels — resolved, but how is unknown

All three pairs flagged earlier are now single labels:

| Pair | Earlier (profile) | Cleaned |
|---|---|---|
| `Cold/Flu` / `Cold / Flu` | 1,302 / 4 | 1,281 / **0** |
| `Creams/Gels & Medicated Patches` / `Creams & Gels (Rubs)` | 1,009 / 1 | 979 / **0** |
| `Other Lifestyle CHC` / `Other Lifestyle` | 202 / 3 | 200 / **0** |

The handoff README says it fixed "the two agreed Sub-Segment label spellings".
**Whether the small variants merged into the large ones or were removed cannot
be determined from the cleaned file alone** — in each case the surviving label is
*smaller* than before, not larger, which is consistent with overall row removal
but does not demonstrate a merge. This is recorded as BLOCKED in the cleaning
audit (§3.5 there) and needs the raw file.

`Other Self Care` still appears as both a Segment (1,117) and a Sub-Segment (92).
Normal for an "other" bucket; confirmed present rather than assumed.

### 4.3 Brand → Mnfr — CONFIRMED deterministic

**The one relationship that graduates to a hard rule.**

Mnfr has exactly two values: `All others` (73,512) and `J&J` (1,020), with 4,186
unlabelled. Grouping by Brand and counting distinct Mnfr values returns **zero
brands with more than one manufacturer**, across all 144 Brand values.

**Rule R2 becomes a hard rule:** a predicted Brand implies its Mnfr, and a
prediction that contradicts the lookup is wrong.

**Consequence for modelling, now demonstrated rather than suspected.** Mnfr is
fully derivable from Brand. A Mnfr model that does not use Brand is solving a
harder problem than it needs to, and the lookup is the baseline any Mnfr model
has to beat.

### 4.4 Platform → Brand, and the scope question — CORRECTED on both counts

**Earlier claim:** Platform is a product-line variant *within* a brand, populated
only for one manufacturer's products. Both halves were overstated.

**Scope: 93.4%, not exclusive.** Of 1,092 rows carrying a Platform label, 1,020
are J&J. The remaining **72 are not**: Advil, Excedrin, Kirkland Signature,
Visine, and some under `UnItemised brand`.

The direction of the finding holds — Platform is overwhelmingly a J&J-labelled
field, and 98.61% of the catalogue has no Platform at all — so the scope
conversation with the PO is still the right next step. But "labeled only for one
manufacturer" is not what the data says, and **rule R4 cannot block**. A
prediction of Platform on a non-J&J product is unusual, not impossible.

**Nesting: not nested.** 11 of 60 Platform values span more than one Brand:

| Platform value | Brands |
|---|---|
| Extra Strength | Benadryl, Excedrin, Tylenol |
| Chewables | Benadryl, Motrin, Tylenol |
| Liquid Gels | Benadryl, Motrin, Zyrtec |
| Children | Benadryl, Zyrtec |
| Allergy | Kirkland Signature, UnItemised brand, Visine |
| Oral Suspension | Motrin, Tylenol |
| Original Strength | Benadryl, Pepcid |
| Complete | Claritin, Pepcid |
| Cold + Cough + Runny Nose | Sudafed, Tylenol |
| Cream | Bengay, UnItemised brand |
| Sinus | Sudafed, Zarbee's |

The earlier reading — that Platform values are brand-specific product lines —
is true for some (`Motrin IB`, `Tylenol PM`, `Imodium A-D`) and false for the
generic strength-and-format descriptors that make up most of the list. That is
exactly the failure mode the earlier version warned about when it rated
label-name reading as the weakest method: *"it is easy to fool yourself with it."*

**Rule R3 stays in warn mode.**

### 4.5 ProductCategory → Segment — confirmed soft

`ProductCategory` is a retailer breadcrumb, 99.99% populated, depth 2 to 7.

At depth 2, **12 prefixes map to one Segment at least 90% of the time with at
least 50 rows**, covering 7,343 rows (9.5%). At depth 3, 74 such prefixes.

| Prefix (depth 2) | Segment | n | share |
|---|---|---|---|
| Health and Medicine > Vitamins and Supplements | VMS | 3,684 | 98.0% |
| Shop > Vitamins & Supplements | VMS | 1,065 | 93.3% |
| Health > Vitamins & Supplements | VMS | 1,051 | 94.4% |
| Health and Medicine > Cough Cold and Flu medicine | CCFS | 481 | 90.1% |
| Vitamins > Supplements | VMS | 317 | 100% |
| Health and Medicine > Eye Care | Other Self Care | 180 | 90.5% |
| Health and Medicine > Ear Care | Lifestyle CHC | 76 | 100% |
| Health and Medicine > Digestive Health | Digestive Health | 51 | 100% |

**9.5% coverage is the honest headline.** This is useful as supporting evidence
for individual predictions — which the README requires alongside every
classification — and as a soft review trigger. It is not a usable standalone
baseline for the whole catalogue.

**Rule R5 remains review-only and must never block.**

### 4.6 TargetAgeGroup — confirmed an attribute

Adult 70,242 (94.25%), Children 3,548, Infant 737, with 4,191 unlabelled.

No hierarchical relationship, as expected. Any co-occurrence rule learned from
737 Infant rows would encode this dataset rather than what is possible, so the
earlier recommendation stands: compute the crosstab, report combinations that
never occur, and label them **unobserved, not impossible**.

### 4.7 `Category` — CLOSED: it is a constant

The earlier version flagged this as a hole: a seventh column, almost fully
populated in training, 100% missing in the target set, not named among the six
targets, with its values absent from the profile.

**It holds one value, `Self Care`, in all 78,718 rows.** Zero variance, zero
information. It is neither a target nor a feature, and the relationship map is
complete without it.

---

## 5. Demonstrated vs. assumed

### Demonstrated by the data

- Segment → Sub-Segment is a two-level classification with 8 and 45 valid classes
  and 47 parent-child pairs, and it is **not strict**: `Internal Pain` and
  `Probiotics` each have two parents.
- The CCFS group has six children totalling 4,209 rows; `Nasal Sprays` is one of
  them.
- `Sleep Aids` (1,060) is a child of Internal Analgesics.
- `Ear Care` (3,090) is a child of Lifestyle CHC.
- Brand → Mnfr is deterministic across all 144 Brand values.
- Platform is labelled on 1,092 rows, 1,020 of them J&J (93.4%).
- 11 of 60 Platform values span more than one Brand.
- `ProductCategory` has 12 depth-2 prefixes mapping to one Segment at ≥90% with
  ≥50 rows, covering 9.5% of rows.
- `Category` holds exactly one value.
- `JoiningKey` is unique per row; `Sku` has 78,298 unique values over 78,718 rows.
- Zero malformed (whitespace-prefixed) values remain in any target.

### Still assumed or unknown

- **Whether the near-duplicate labels merged or were deleted** (§4.2). Needs raw.
- **Whether the relationships above hold in the full 84,552-row dataset.** All of
  this is measured on the cleaned subset.
- **Whether the single `Internal Pain` row under External Analgesics is an error
  or a real case.** It reads as an error; nobody has confirmed it.
- **Which brands belong to J&J beyond what the Mnfr column states.** Deliberately
  not inferred from outside knowledge.

### Cannot be established from this dataset at all

- That a label combination is **impossible**. Absence in 78,718 rows is evidence
  of rarity, not impossibility. Only the PO or a taxonomy document can declare a
  combination invalid. Every rule below marked *hard* depends on that.

---

## 6. Validation rules — updated status

| ID | Rule | Status | Action when violated |
|---|---|---|---|
| R1 | Predicted Sub-Segment must belong to predicted Segment | **Warn mode** — hierarchy not strict (§4.1) | Review |
| R2 | Predicted Brand implies its Mnfr | **HARD** — verified deterministic (§4.3) | Block |
| R3 | Platform must be consistent with predicted Brand | **Warn mode** — not nested (§4.4) | Review |
| R4 | Platform absent outside its supported scope | **Soft** — 93.4%, not exclusive (§4.4) | Review, never block |
| R5 | Prediction contradicts a >90% breadcrumb pattern | Soft, 9.5% coverage (§4.5) | Review only |
| R6 | Predicted combination never occurs in training | Soft | Review only |

Only **R2** is ready to block a prediction. The earlier version expected three
hard rules; the data supports one.

R1 and R3 should run in warn mode with their firing rate counted. A rule firing
on a large share of predictions is describing a broken assumption, not catching
errors.

---

## 7. How this supports Aryx integration

The structure of this section is unchanged; the content is now verified.

**1. This document is the review input, not the ontology.** Aryx proposes a model
from the data and asks a human to correct it. A reviewer walking in now knows
that Brand → Mnfr is a deterministic lookup, that Segment → Sub-Segment is a
near-tree with two known exceptions, and that Platform is **not** cleanly nested
under Brand. Without that last point a reviewer would approve a wrong nesting.

**2. R2 is ready to become a standing rule in Aryx.** R1 and R3 are not, and
feeding them in as hard constraints would reject valid records.

**3. Entity types.** Brand, Mnfr, Platform and Product are entity types. Segment
and Sub-Segment are a classification hierarchy over Product. TargetAgeGroup is an
attribute. `Category` is a constant and should not be modelled at all.

**A caution, now with a number.** Aryx does entity resolution, and
`ProductBrand` (the retailer's raw string) matches `Brand` (the curated label)
exactly **91.79%** of the time on named brands. They are close enough that an
entity resolver will try to merge them, and they must not be merged: one is input
and the other is a target. Flag this before the first build.

**Open question:** whether CatalogIQ is expected to integrate with Aryx at all.
What is settled is that Aryx exposes an MCP server locally at
`http://localhost:8765/sse`, so the path exists if it is wanted.

---

## 8. Open questions

1. **Is the single `Internal Pain` row under External Analgesics an error?**
   (§4.1) One row, and fixing it makes R1 nearly strict.
2. **Should `Probiotics` have one parent or two?** (§4.1) A taxonomy decision for
   the PO. R1 cannot become a hard rule until it is answered.
3. **Did the near-duplicate labels merge or get deleted?** (§4.2) Needs the raw
   file.
4. **Is Platform expected on the target set at all, and for which products?**
   (§4.4) 98.61% of the catalogue has no Platform label.
5. **Does a documented taxonomy exist from the PO?** Without one, no combination
   can be called impossible (§5).
6. **Should `Category` be dropped from the schema?** (§4.7) It carries nothing.
7. **Is Aryx integration in scope, or forward-looking?** (§7)

---

## 9. Done-when checklist

| Criterion | Where |
|---|---|
| Important target relationships documented | §3, §4 |
| Potential validation rules identified | §6 |
| Unsupported assumptions separated from demonstrated relationships | §2, §5 |
| Findings explain support for prediction validation / Aryx integration | §6, §7 |
