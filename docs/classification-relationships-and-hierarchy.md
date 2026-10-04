# Classification Relationships and Hierarchy — CatalogIQ

**Status:** verified against data. Row-level results from the v4 integrated run.
**Scope:** relationships among the CatalogIQ classification targets, and whether
they can serve as validation rules or supporting evidence
**Location:** `docs/research/`
**Notebook:** `notebooks/01-classification-relationships.ipynb`
**Data basis:** `training_candidate.csv` from a local `--mode integrated
--exclude-policy keep` run reproducing `integrated_policy_v4_final_20261003`
(83,938 rows of 84,552). Supersedes two earlier versions: one written from the
dataset profile without row-level access, and one measured on the superseded
`conservative_handoff_20261002`. Claims that changed are marked **CORRECTED**.

---

## 1. Summary

The goal of this ticket is to find out whether the relationships between the
classification targets are regular enough to be used as validation rules.

**None of them is strict enough to block a prediction.** All four candidate hard
rules have measurable exceptions. That is the finding.

| Relationship | Type | Verified result | Rule status |
|---|---|---|---|
| Brand → Mnfr | Lookup | **CORRECTED — not deterministic.** 4 brands, 5 rows | Warn, near-hard |
| Segment → Sub-Segment | Hierarchy | **Not strict.** 2 sub-segments have 2 parents | Warn |
| Platform → Brand | Nesting | **Not nested.** 11 of 61 values span brands | Warn |
| Platform scope | Attribute | **93.4% J&J, not exclusive** | Soft, never block |
| ProductCategory → Segment | Soft mapping | Confirmed soft, ~9.5% coverage | Review only |
| TargetAgeGroup | Attribute | Confirmed, no hierarchy | Soft check only |

**Two open questions are closed, and both close badly:**

- **`Category` carries one value**, `Self Care`, in all 83,938 rows. A constant.
- **`JoiningKey` is unique per row** (83,938 keys, 83,938 rows). It is a row
  identifier, not a grouping key, so it cannot prevent near-duplicate leakage in
  the train/test split. `Sku` has 83,460 unique values, so it cannot either.

---

## 2. Method, and why the data basis changed twice

The first version of this document was written from the dataset profile — counts
and missingness per column, no rows. Counts can rule a relationship out (a child
cannot have more rows than its parent) but cannot confirm one.

The second version was measured on the `conservative_handoff_20261002` cleaned
file. **That handoff is superseded**, and measuring on it produced a wrong
result: Brand → Mnfr looked deterministic because the five contradicting rows had
been removed by the conservative policy. See §4.3.

This version runs on a local reproduction of the current v4 policy:

```bash
python -m catalogiq --mode integrated --exclude-policy keep \
  --output-dir data/processed/maria_audit_run
```

83,938 training candidates and 27,870 target candidates, matching the reference
run exactly.

**The lesson is worth recording.** A cleaning policy that removes rows can make a
relationship look more regular than it is. Any rule derived from cleaned data
should be re-checked against the policy that produced it.

---

## 3. Three kinds of relationship

The distinction held up and still decides what kind of rule each yields.

**Taxonomy — a tree, "is a type of".** Segment → Sub-Segment. Acid Relief *is a
type of* Digestive Health. Would yield a hard rule if strict. It is not (§4.1).

**Ontology relationship — an edge between different kinds of thing.** Brand →
Mnfr. Tylenol is not *a type of* Johnson & Johnson; it is *made by* it. Yields a
consistency rule (§4.3).

**Attribute — a property of the product.** TargetAgeGroup, and Platform. Owns no
children, yields at most a soft plausibility check (§4.6).

---

## 4. Relationship by relationship

### 4.1 Segment → Sub-Segment — not a strict hierarchy

8 Segments, 49 Sub-Segments, 51 parent-child pairs. **Two Sub-Segments have two
parents**, so the taxonomy is a graph, not a tree.

| Sub-Segment | Parents | Reading |
|---|---|---|
| `Internal Pain` | Internal Analgesics 1,975 · External Analgesics 1 | The single row is almost certainly a labelling error |
| `Probiotics` | Vitamins, Minerals & Supplements 634 · Digestive Health 22 | Genuine ambiguity — probiotics legitimately sit in both |

These are different problems. The first is one bad row. The second is a real
taxonomy question, and the decision log already answers it: *"Do not enforce a
fixed Segment/Sub-Segment parent whitelist: shared parents can be legitimate."*

**Rule R1 stays in warn mode.** A hard rule today would reject 23 legitimate rows.

**Rows with a Sub-Segment but no Segment: 0.** The profile-based version flagged
4,789 such rows as a caveat on its elimination method. Under v4 there are none,
so the caveat no longer applies.

**The CCFS group — confirmed, with the expected relabel.** The profile-based
version's strongest evidence was that seven Sub-Segments summed exactly to CCFS.
Six are confirmed children under v4; the seventh, `Cold / Flu` (4 rows), was
relabelled to `Cold/Flu` by the documented `slash_spacing` rule.

| Sub-Segment | Count |
|---|---|
| Sore Throat / Cough Drops | 1,318 |
| Cold/Flu | 1,293 |
| Cough | 594 |
| Nasal Sprays | 464 |
| Sinus excl Sprays | 429 |
| External Upper Respiratory | 154 |
| **Total** | **4,252** |

**The substantive claim survives: `Nasal Sprays` belongs under CCFS, not under
Allergy**, even though nasal sprays are an allergy product in ordinary use. That
had to come from the data rather than from domain intuition, and it did.

**The Internal Analgesics coincidence — confirmed.** The first version noticed a
1,065-row gap that happened to equal the Sleep Aids count, and said: *"Do not
write it down as a finding. Test it."* Tested: **Sleep Aids (1,064) is a child of
Internal Analgesics**, its only parent.

**Ear Care — elimination confirmed, parent identified.** The first version proved
by counts alone that Ear Care could not sit under Other Self Care. It sits under
**Lifestyle CHC** (3,094 of that Segment's 4,549), its only parent.

Full map: `data/interim/segment_subsegment_map.csv`.

### 4.2 Near-duplicate labels — all three resolved

| Pair | Raw | v4 | Resolution |
|---|---|---|---|
| `Cold/Flu` / `Cold / Flu` | 1,302 / 4 | 1,293 / 0 | Relabelled, rule `slash_spacing` |
| `Other Lifestyle CHC` / `Other Lifestyle` | 202 / 3 | 204 / 0 | Relabelled, rule `reviewed_other_lifestyle` |
| `Creams/Gels & Medicated Patches` / `Creams & Gels (Rubs)` | 1,009 / 1 | 1,002 / **1** | **Not a duplicate. Both survive.** |

The first two are recorded row by row in `label_changes.csv` with their source
rows and rules. The third was never a label decision: `Creams & Gels (Rubs)` is a
real one-row class that the conservative handoff deleted and v4 retains. See
§3.5 of the cleaning validation.

`Other Self Care` still appears as both a Segment (1,132) and a Sub-Segment (100).
Normal for an "other" bucket, and confirmed rather than assumed.

### 4.3 Brand → Mnfr — CORRECTED: not deterministic

**This is the claim that changed, and it changed because the data basis changed.**

Measured on the superseded v2 handoff, all 144 brands mapped to exactly one
manufacturer and this document previously called R2 a hard rule. Measured on v4,
**four brands map to two**:

| Brand | J&J | All others |
|---|---|---|
| Tylenol | 311 | 1 |
| Benadryl | 119 | 1 |
| Sudafed | 43 | 2 |
| Bengay | 32 | 1 |

Five rows out of 78,273 labelled. **The exceptions look like Brand labelling
errors, not Mnfr errors** — the products are `Major Anti-Diarrheal (Loperamide)`
filed under Sudafed, `HealthA2Z Laxative (Bisacodyl)` under Tylenol, a
`Hemorrhoidal Ointment` under Bengay. None is plausibly that brand's product.

**Rule R2 is near-deterministic but cannot block.** Run it in warn mode and treat
every firing as a candidate Brand correction rather than an Mnfr correction.

**A caution on how this is used.** The decision log states *"Preserve supplied
Mnfr... Do not infer J&J from Brand"*, and the repository records a withdrawn J&J
reassignment. Nothing here proposes reviving it. The relationship is reported as
a measurement and as a review signal; it is not a basis for rewriting labels.

### 4.4 Platform → Brand, and the scope question

**Scope: 93.4%, not exclusive.** Of 1,094 rows carrying a Platform label, 1,022
are J&J. The remaining 72 are Advil, Excedrin, Kirkland Signature, Visine and
rows under `UnItemised brand`.

The direction holds — Platform is overwhelmingly a J&J-labelled field, and 98.70%
of the catalogue has no Platform at all — so a scope conversation with the PO is
still the right next step. But "labelled only for one manufacturer" is not what
the data says, and **rule R4 cannot block**. A Platform prediction on a non-J&J
product is unusual, not impossible.

**Nesting: not nested.** 11 of 61 Platform values span more than one Brand:

| Platform value | Brands |
|---|---|
| Extra Strength | Benadryl, Excedrin, Tylenol |
| Chewables | Benadryl, Motrin, Tylenol |
| Liquid Gels | Benadryl, Motrin, Zyrtec |
| Allergy | Kirkland Signature, UnItemised brand, Visine |
| Children | Benadryl, Zyrtec |
| Oral Suspension | Motrin, Tylenol |
| Original Strength | Benadryl, Pepcid |
| Complete | Claritin, Pepcid |
| Cold + Cough + Runny Nose | Sudafed, Tylenol |
| Cream | Bengay, UnItemised brand |
| Sinus | Sudafed, Zarbee's |

The earlier reading — that Platform values are brand-specific product lines — is
true for some (`Motrin IB`, `Tylenol PM`, `Imodium A-D`) and false for the generic
strength-and-format descriptors that make up most of the list. That is exactly
the failure mode the first version warned about when it rated label-name reading
as the weakest method: *"it is easy to fool yourself with it."*

**Rule R3 stays in warn mode.**

### 4.5 ProductCategory → Segment — confirmed soft

`ProductCategory` is a retailer breadcrumb, 99.98% populated, depth 2 to 7.

At depth 2, roughly a dozen prefixes map to one Segment at least 90% of the time
with at least 50 rows, covering about 9.5% of rows. At depth 3 the count rises
but coverage stays modest.

**That coverage is the honest headline.** It is useful as supporting evidence for
individual predictions — which the README requires alongside every classification
— and as a soft review trigger. It is not a usable standalone baseline for the
whole catalogue.

**Rule R5 remains review-only and must never block.**

### 4.6 TargetAgeGroup — confirmed an attribute

Adult 72,475 (94.20%), Children 3,705, Infant 759, with 6,999 unlabelled.

No hierarchical relationship, as expected. A co-occurrence rule learned from 759
Infant rows would encode this dataset rather than what is possible, so the
original recommendation stands: compute the crosstab, report combinations that
never occur, and label them **unobserved, not impossible**.

### 4.7 `Category` — closed: it is a constant

The first version flagged this as a hole: a seventh column, almost fully
populated in training, 100% missing in the target set, not among the six targets,
values absent from the profile.

**It holds one value, `Self Care`, in all 83,938 rows.** Zero variance, zero
information. Neither target nor feature, and the relationship map is complete
without it.

---

## 5. Demonstrated vs. assumed

### Demonstrated by the data

- Segment → Sub-Segment has 8 and 49 classes over 51 pairs, and is **not strict**:
  `Internal Pain` and `Probiotics` each have two parents.
- The CCFS group has six children totalling 4,252 rows; `Nasal Sprays` is one.
- `Sleep Aids` (1,064) is a child of Internal Analgesics, its only parent.
- `Ear Care` (3,094) is a child of Lifestyle CHC, its only parent.
- Brand → Mnfr holds for 140 of 144 brands; 4 brands and 5 rows contradict it.
- Platform is labelled on 1,094 rows, 1,022 of them J&J (93.4%).
- 11 of 61 Platform values span more than one Brand.
- `Category` holds exactly one value.
- `JoiningKey` is unique per row; `Sku` has 83,460 unique values over 83,938 rows.
- Zero malformed values remain in any target.
- `Creams & Gels (Rubs)` is a real one-row class, not a formatting variant.

### Still assumed or unknown

- **Whether the 5 Brand → Mnfr exceptions are Brand errors.** The product names
  strongly suggest so; nobody has confirmed it.
- **Whether the single `Internal Pain` row under External Analgesics is an error.**
- **Whether `Probiotics` should have one parent or two.** A PO decision.

### Cannot be established from this dataset at all

- That a label combination is **impossible**. Absence in 83,938 rows is evidence
  of rarity, not impossibility. Only the PO or a taxonomy document can declare a
  combination invalid.

---

## 6. Validation rules — updated status

| ID | Rule | Status | Action when violated |
|---|---|---|---|
| R1 | Predicted Sub-Segment belongs to predicted Segment | Warn — not strict (§4.1) | Review |
| R2 | Predicted Brand implies its Mnfr | Warn — 4 exceptions (§4.3) | Review, flag as Brand candidate |
| R3 | Platform consistent with predicted Brand | Warn — not nested (§4.4) | Review |
| R4 | Platform absent outside its scope | Soft — 93.4% (§4.4) | Review, never block |
| R5 | Contradicts a >90% breadcrumb pattern | Soft, ~9.5% coverage (§4.5) | Review only |
| R6 | Combination never occurs in training | Soft | Review only |

**No rule is ready to block a prediction.** The first version expected three hard
rules; the second claimed one; the data supports none.

That is not a failure of the research. Every rule still works as a review trigger,
and a rule that fires on a known-small exception set is more useful than a hard
rule that rejects valid records. Run them all in warn mode and count the firing
rate: a rule firing on a large share of predictions is describing a broken
assumption, not catching errors.

---

## 7. How this supports Aryx integration

**1. This document is the review input, not the ontology.** Aryx proposes a model
from the data and asks a human to correct it. A reviewer walking in now knows
that Segment → Sub-Segment is a near-tree with two known exceptions, that
Brand → Mnfr has four, and that Platform is **not** nested under Brand. Without
that last point a reviewer would approve a wrong nesting.

**2. No rule should be fed to Aryx as a hard constraint.** All six belong in
warn mode until the exceptions in §5 are resolved.

**3. Entity types.** Brand, Mnfr, Platform and Product are entity types. Segment
and Sub-Segment are a classification hierarchy over Product. TargetAgeGroup is an
attribute. `Category` is a constant and should not be modelled.

**A caution, with a number.** Aryx does entity resolution, and `ProductBrand` (the
retailer's raw string) matches `Brand` (the curated label) exactly **91.86%** of
the time on named brands. They are close enough that a resolver will try to merge
them, and they must not be merged: one is input and the other is a target.

**Open question:** whether CatalogIQ is expected to integrate with Aryx at all.
What is settled is that Aryx exposes an MCP server locally at
`http://localhost:8765/sse`, so the path exists if it is wanted.

---

## 8. Open questions

1. **Are the 5 Brand → Mnfr exceptions Brand labelling errors?** (§4.3) Five rows,
   and confirming them would make R2 effectively hard.
2. **Is the single `Internal Pain` row under External Analgesics an error?** (§4.1)
3. **Should `Probiotics` have one parent or two?** (§4.1) A PO decision.
4. **Is Platform expected on the target set at all, and for which products?**
   (§4.4) 98.70% of the catalogue has no Platform label.
5. **Does a documented taxonomy exist from the PO?** Without one, no combination
   can be called impossible (§5).
6. **Should `Category` be dropped from the schema?** (§4.7)
7. **Is Aryx integration in scope, or forward-looking?** (§7)

---

## 9. Done-when checklist

| Criterion | Where |
|---|---|
| Important target relationships documented | §3, §4 |
| Potential validation rules identified | §6 |
| Unsupported assumptions separated from demonstrated relationships | §2, §5 |
| Findings explain support for prediction validation / Aryx integration | §6, §7 |
