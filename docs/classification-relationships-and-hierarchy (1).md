# Classification Relationships and Hierarchy — CatalogIQ

**Status:** research findings, pending row-level confirmation
**Scope:** relationships among the CatalogIQ classification targets, and whether
they can later serve as validation rules or supporting evidence
**Location:** `docs/research/`
**Data basis:** CatalogIQ Dataset Profile (84,552 training rows, 28,082 target
rows) and CatalogIQ Cleaning Guidance. No row-level crosstabs were available when
this was written, which shapes what can and cannot be claimed — see §2.

---

## 1. Summary

The goal of this ticket is not to build a model. It is to find out whether the
relationships between the classification targets are regular enough to be used
later as validation rules.

Short answer: **yes, at least three of them look usable**, and one of them may
change how the project treats a target.

| Relationship | Type | Confidence | Potential use |
|---|---|---|---|
| Segment → Sub-Segment | Strict hierarchy | Strong | Hard validation rule |
| Brand → Mnfr | Lookup / many-to-one | Strong | Hard validation rule |
| Platform → Brand | Product line within a brand | Strong | Hard validation rule + scope question |
| ProductCategory → Segment | Soft mapping | Moderate | Supporting evidence, not a rule |
| TargetAgeGroup | Attribute, not hierarchy | N/A | Soft consistency check only |

**The finding most worth escalating:** Platform appears to be labeled only for
one manufacturer's products (§4.4). If confirmed, Platform is not a sparse target
that needs more labeling — it is a target with a restricted scope, and that is a
different conversation with the PO.

---

## 2. Method, and what it can and cannot prove

This is the part the ticket's fourth criterion asks for, so it comes before the
findings rather than after.

**What I had:** the dataset profile, which gives the full label list and row count
for every target, and the cleaning guidance.

**What I did not have:** the rows themselves. Without them I cannot produce a
crosstab, so I cannot state that a given Sub-Segment *does* occur under a given
Segment. Every co-occurrence claim below is a hypothesis with a stated test.

Three techniques were available, and they differ in how much weight they carry:

**A. Upper-bound elimination (strongest).** A child class cannot have more rows
than its parent. Ear Care has 3,095 rows; the Segment "Other Self Care" has 1,135.
Ear Care therefore cannot sit under Other Self Care.

One caveat keeps this from being airtight: 4,789 rows have a Sub-Segment but a
missing Segment, so in principle a child's rows could be concentrated among rows
whose parent is blank. For a child larger than 4,789 the elimination is certain;
below that it is strong but not proven.

**B. Support-sum matching (suggestive).** If a set of Sub-Segments sums exactly to
a Segment's count, those Sub-Segments are probably that Segment's children.
Suggestive, not proof — sums can coincide. It is strongest when a second,
independent signal agrees, as in §4.1.

**C. Label-name reading (weakest on its own).** "Motrin IB" as a Platform value
plainly references a brand. This generates hypotheses; it does not confirm them,
and it is easy to fool yourself with it.

**What I deliberately did not do:** fill gaps with outside product knowledge. I
know which consumer brands belong to Johnson & Johnson, and using that to assign
brands to the Mnfr label would have produced a confident-looking mapping built on
nothing in the data. Where outside knowledge informs a hypothesis, §5 says so.

### 2.1 Clean before you crosstab

Every target carries exactly **3** malformed values, and each of those values
begins with a space (`" tea"`, `" lactose"`, `" Keto certified"`). These are
shifted rows, per the cleaning guidance.

If the relationship analysis runs on uncleaned data, it will "discover"
combinations such as Segment = `" Keto certified"` with Sub-Segment = `" lactose"`
and report them as rare relationships. They are parsing damage, not structure.

**Any conclusion about which combinations are impossible must be drawn after
cleaning, or it is wrong.** This is the single largest risk to this ticket.

---

## 3. Three different kinds of relationship

These are not all the same shape, and treating them the same produces the wrong
validation rule. The distinction follows the taxonomy/ontology split in the team's
study notes.

**Taxonomy — a tree, "is a type of".**
Segment → Sub-Segment. Acid Relief *is a type of* Digestive Health. The rule this
yields is hard: a Sub-Segment either belongs under a Segment or it does not.

**Ontology relationship — a graph edge between different kinds of things.**
Brand → Mnfr. Tylenol is not *a type of* Johnson & Johnson; it is *made by* it.
The rule this yields is a consistency rule: this brand always maps to this
manufacturer, so a prediction that contradicts it is wrong.

**Attribute — a property of the product.**
TargetAgeGroup. Infant is not a type of anything in this taxonomy and does not
own children. It yields at most a soft plausibility check.

Keeping these apart matters because a hard rule can block a prediction, while a
soft check can only flag it for review.

---

## 4. Relationship by relationship

### 4.1 Segment → Sub-Segment (strict hierarchy)

**Status: strong. One group confirmed by arithmetic, full map pending crosstab.**

Eight valid Segments cover 79,760 labeled rows; roughly 49 valid Sub-Segments
cover 77,034. The 2,726-row difference is rows with a Segment but no Sub-Segment.

**The CCFS group sums exactly.** Seven Sub-Segments add up to the CCFS count with
no remainder:

| Sub-Segment | Count |
|---|---|
| Sore Throat / Cough Drops | 1,319 |
| Cold/Flu | 1,302 |
| Cough | 594 |
| Nasal Sprays | 464 |
| Sinus excl Sprays | 430 |
| External Upper Respiratory | 154 |
| Cold / Flu | 4 |
| **Total** | **4,267** |
| **Segment CCFS** | **4,267** |

Two independent signals agree here: the arithmetic is exact, and the acronym CCFS
reads as Cough, Cold, Flu, Sinus, which matches the members. Taken together this
is the strongest evidence in this document.

It also settles a question the label names alone leave ambiguous: **Nasal Sprays
belongs under CCFS, not under Allergy**, even though nasal sprays are an allergy
product in ordinary use. That is exactly the kind of thing that has to come from
the data rather than from domain intuition.

**Eliminations that follow from counts** (method A):

- Ear Care (3,095) cannot sit under Other Self Care (1,135).
- Supplements (19,535), Herbal & Natural (10,565) and Minerals (7,556) exceed
  every Segment except Vitamins, Minerals & Supplements (49,091), so they can
  only sit there.
- Cold & Heat Wraps (3,818) exceeds Other Self Care, Allergy and CCFS, and
  Multivitamins (3,426) does the same.

**Where the arithmetic does not close.** Allergy (2,737) against its obvious
children — Other Allergy (882), Oral Antihistamine (823), Allergy INS (581) —
leaves a 451-row gap. Internal Analgesics (4,808) against Internal Pain (1,977),
Speciality Pain (1,755) and Remaining Internal Pain (11) leaves 1,065, which
happens to equal the Sleep Aids count exactly. That is interesting, and it is
also the kind of coincidence that method B produces and cannot resolve. Do not
write it down as a finding. Test it.

**Confidence:** the hierarchy exists and is probably strict. One group is
established. The full parent-child map needs one crosstab.

**Candidate rule (§6, R1):** a predicted Sub-Segment must belong to the predicted
Segment.

### 4.2 Near-duplicate labels that distort the hierarchy

`Cold/Flu` (1,302) and `Cold / Flu` (4) are almost certainly the same category
written two ways. Both fall inside the CCFS sum above, which is consistent with
them being one class split by a spacing difference.

Also worth checking:

| Pair | Counts |
|---|---|
| Creams/Gels & Medicated Patches / Creams & Gels (Rubs) | 1,009 / 1 |
| Other Lifestyle CHC / Other Lifestyle | 202 / 3 |

These matter to this ticket, not only to cleaning. A 4-row class looks like a
rare category worth protecting; a spacing variant of a 1,302-row class is noise
that inflates the taxonomy and creates a class no model can learn. **Resolve
these before the parent-child map is written down**, or the map will contain
classes that should not exist.

Note also that `Other Self Care` appears as both a Segment (1,135) and a
Sub-Segment (102). That is normal for an "other" bucket in a two-level taxonomy
and is probably not an error, but it should be confirmed rather than assumed.

### 4.3 Brand → Mnfr (many-to-one lookup)

**Status: strong on structure, unverified on membership.**

Mnfr has only two valid values: `J&J` (1,030) and `All others` (77,821). That
makes the relationship a lookup: each brand should map to exactly one
manufacturer, and a brand belonging to J&J should carry `J&J` on every one of its
rows.

**This is testable with a single query and gives a clean yes/no.** Group by Brand,
count distinct Mnfr values. If every brand returns 1, the relationship is
deterministic and becomes a hard validation rule. Any brand returning 2 is either
a labeling inconsistency or a genuine exception, and both are worth knowing.

**What I am not claiming.** I can see brands in the list that I know are J&J
consumer brands, and their row counts are in the right neighborhood to account
for the 1,030 J&J rows. I am not writing that mapping here, because it would come
from outside knowledge rather than from the supplied data, and the ticket asks
for exactly that separation. The crosstab produces the real answer in one step.

**One consequence for modeling.** Because `Mnfr` is derivable from `Brand` if the
lookup holds, a Mnfr model that ignores Brand is solving a harder problem than it
needs to. Worth flagging to whoever builds it.

**Candidate rule (§6, R2):** a predicted Brand implies its Mnfr; a prediction that
contradicts the lookup is wrong.

### 4.4 Platform → Brand, and the scope question

**Status: strong hypothesis, high impact, needs one query.**

The Platform label list is not a list of generic product formats. It contains
brand-specific product lines:

| Platform value | Count | Reads as |
|---|---|---|
| Motrin IB | 47 | Motrin |
| Imodium A-D | 32 | Imodium |
| Tylenol PM | 12 | Tylenol |
| Motrin PM | 8 | Motrin |
| Simply Sleep | 5 | Tylenol line |
| Ultratabs | 20 | antihistamine line |
| Red Eye / Dry Eye | 28 / 12 | eye care line |

**The hypothesis:** Platform is a product-line variant *within* a brand, and it is
populated only for the products of one manufacturer.

**The supporting number:** Platform has **1,105** labeled rows. Mnfr = J&J has
**1,030**. Those are close enough to be worth checking immediately.

**Why this matters more than the other relationships.** The evaluation strategy
document currently treats Platform as a 61-class problem over 1,102 rows and
concludes it is not evaluable as-is. If this hypothesis holds, the framing is
wrong in a useful way:

- Platform is not sparse at random. It is **out of scope for most of the catalog**
  by design, and the 98.69% missing rate is not a data quality problem.
- Predicting Platform for a non-J&J product may be meaningless rather than merely
  uncertain, which is a much stronger basis for abstention.
- The real question for the PO stops being "can we get more Platform labels" and
  becomes "is Platform expected on the target set at all, and for which products".

**The test:** crosstab Platform against Mnfr and against Brand. Two lines of code,
and it either confirms or kills the hypothesis. This should be run first.

**Candidate rules (§6, R3 and R4).**

### 4.5 ProductCategory → Segment / Sub-Segment (soft mapping)

**Status: moderate. Useful as evidence, not as a hard rule.**

`ProductCategory` already contains a hierarchy, supplied by the retailer as a
breadcrumb path with `>` separators and a depth of 2 to 7:

```
Health & Household > Health Care > OTC Medications & Treatments > Pain Relievers
```

It is also 99.97% populated in training and 99.98% in the target dataset, which
makes it one of the few near-complete fields available.

**Why this is not a hard rule.** It is a *retailer's* taxonomy, not CatalogIQ's.
Different retailers use different breadcrumbs for the same product, the depth
varies, and the cleaning guidance documents duplicated and malformed paths. A
one-to-one mapping to Segment should not be expected.

**What it is good for.** Two things, both valuable:

1. **Supporting evidence.** The README requires that each classification comes
   with evidence. A breadcrumb ending in "Pain Relievers" is human-readable
   justification for a Segment of Internal Analgesics, and it is the most natural
   evidence field in the dataset.
2. **A baseline and a soft check.** If a breadcrumb prefix maps to one Segment
   90%+ of the time, that mapping is both a cheap baseline model and a soft
   validation rule: a prediction that contradicts a strong breadcrumb pattern is
   a review candidate, not an automatic error.

**Test:** for each distinct breadcrumb prefix at depth 2 and 3, count the Segment
distribution. Report the prefixes where one Segment exceeds 90%.

### 4.6 TargetAgeGroup (attribute, not hierarchy)

**Status: no hierarchical relationship expected.**

Three valid values: Adult (72,995), Children (3,718), Infant (759).

This is a property of the product, not a level of the taxonomy. It does not own
children and is not owned by Segment.

**What might exist is a soft co-occurrence pattern.** Pregnancy Vitamins (359)
should skew Adult. Some Sub-Segments may never appear with Infant. These are
plausibility checks at best, and they carry a real risk: a rule learned from 759
Infant rows will mostly encode what happens to be in this dataset, not what is
possible in the world.

**Recommendation:** do not build hard rules from TargetAgeGroup. Compute the
crosstab, report combinations that never occur, and label them as **unobserved,
not impossible**. The distinction is the whole point of the ticket's third
criterion.

One genuine use: an Infant prediction on a product whose Sub-Segment never
co-occurs with Infant is a good review candidate, precisely because Infant is the
rarest and highest-consequence class in a health products catalog.

### 4.7 `Category` — a seventh column nobody named

`Category` behaves exactly like the six targets: almost fully populated in
training (1 missing) and 100% missing in the target dataset. But it is not in the
six targets named in the ticket.

The profile does not list its label values, so I cannot say what it contains or
how it relates to Segment.

**This needs an answer before the relationship map is called complete.** If
`Category` is a seventh target, or the parent of Segment, this document has a
hole in it that is not the analyst's fault but is still a hole.

---

## 5. Demonstrated vs. assumed

The ticket asks for this separation explicitly, so it gets its own section.

### Demonstrated by the supplied data

- Segment and Sub-Segment form a two-level taxonomy with 8 and ~49 valid classes.
- The seven Sub-Segments listed in §4.1 sum exactly to the CCFS Segment count.
- Ear Care (3,095) cannot sit under Other Self Care (1,135); the other
  eliminations in §4.1 follow the same logic.
- Mnfr has exactly two valid values, making Brand → Mnfr a two-way lookup.
- Platform values include brand-specific product-line names.
- Platform labeled rows (1,105) and J&J rows (1,030) are close in magnitude.
- ProductCategory contains a retailer breadcrumb hierarchy, 2 to 7 levels deep,
  populated on 99.97% of training rows.
- Each target carries exactly 3 malformed values, all beginning with whitespace.

### Assumed, hypothesized, or from outside the data

- That the Segment → Sub-Segment hierarchy is strict (one parent per child). Very
  likely, not yet tested.
- The full parent-child map beyond the CCFS group.
- That Platform is populated only for J&J products. The strongest hypothesis in
  this document, and still a hypothesis.
- That `Cold / Flu` and `Cold/Flu` are the same class.
- Which specific brands belong to J&J. Deliberately excluded; this is outside
  knowledge, not data.
- That `Other Self Care` appearing as both Segment and Sub-Segment is intentional.
- Anything about `Category` (§4.7).

### Cannot be established from this dataset at all

- That a label combination is **impossible**. Absence in 84,552 rows is evidence
  of rarity, not of impossibility. Only the PO or a taxonomy document can declare
  a combination invalid. Every rule in §6 marked *hard* depends on that
  confirmation, and until it arrives they should run in warn mode.

---

## 6. Candidate validation rules

Each rule states what it checks, what it depends on, and what should happen when
it fires.

| ID | Rule | Depends on | Action when violated |
|---|---|---|---|
| R1 | Predicted Sub-Segment must belong to predicted Segment | Full parent-child map (§4.1) | Block or review |
| R2 | Predicted Brand implies its Mnfr | Brand → Mnfr lookup being 1:1 (§4.3) | Block or review |
| R3 | Platform must be consistent with predicted Brand | Platform → Brand map (§4.4) | Block or review |
| R4 | Platform should be absent outside its supported scope | §4.4 hypothesis confirmed | Abstain |
| R5 | Prediction contradicts a >90% breadcrumb pattern | §4.5 mapping | Review only, never block |
| R6 | Predicted combination never occurs in training | Cleaned crosstabs | Review only, never block |

**Two notes on using these.**

First, R1 to R4 are only *hard* if the underlying relationship turns out to be
deterministic after cleaning. Until then, run every rule in warn mode and count
how often it fires. A rule that fires on 30% of predictions is describing a
broken assumption, not catching 30% errors.

Second, R5 and R6 must never block. They are pattern-based, and a rule built from
observed frequencies will reject legitimate new products — which is exactly the
failure the "unobserved is not impossible" distinction exists to prevent.

These rules also compose well with the review-routing signals in the evaluation
strategy document (§4.5 there): a hierarchy violation is a strong review trigger
independent of model confidence, because it can be true even when the model is
certain.

---

## 7. How this supports Aryx integration

Aryx builds a knowledge graph by discovering entities and relationships,
resolving duplicates, and keeping provenance. Three concrete connections:

**1. This document is the review input, not the ontology.** Aryx proposes a model
from the data and asks a human to correct it before building. The value of this
research is that the reviewer walks in already knowing that Segment → Sub-Segment
is a hierarchy, Brand → Mnfr is a lookup, and Platform belongs under Brand — so a
wrong proposal gets caught instead of approved.

**2. The rules in §6 map onto a mechanism Aryx already has.** Corrections in Aryx
can be retained as standing rules for later ingests. R1 through R4 are candidates
for exactly that. This is the concrete answer to how these findings support later
prediction validation.

**3. Entity types fall out of the relationship types in §3.** Brand, Mnfr,
Platform and Product are entity types. Segment and Sub-Segment are a
classification hierarchy over Product. TargetAgeGroup is an attribute of Product,
not an entity. Feeding Aryx a flat list of seven equal columns would produce a
worse model than feeding it these distinctions.

**A caution worth writing down.** Aryx does entity resolution, and near-duplicate
brand strings are exactly what it will try to merge. Two of the pairs in §4.2 are
merge candidates that should be merged; but `ProductBrand` (the retailer's raw
string) and `Brand` (the curated label) are *not* the same field and should not be
collapsed. Flag this before the first build rather than after.

**Open question:** whether CatalogIQ is expected to integrate with Aryx at all, or
whether this is a forward-looking note. That is a question for the PO. What is
settled is that Aryx exposes an MCP server locally at
`http://localhost:8765/sse`, so the connection path exists if it is wanted.

---

## 8. Queries to run when the cleaned data lands

In priority order. The first two change project decisions, not just documentation.

```sql
-- 1. Is Platform scoped to one manufacturer?  (§4.4 — run this first)
SELECT Mnfr, COUNT(*) AS rows_with_platform
FROM training
WHERE Platform IS NOT NULL
GROUP BY Mnfr;

-- 2. Is Brand -> Mnfr deterministic?  (§4.3)
SELECT Brand, COUNT(DISTINCT Mnfr) AS n_mnfr
FROM training
WHERE Brand IS NOT NULL AND Mnfr IS NOT NULL
GROUP BY Brand
HAVING COUNT(DISTINCT Mnfr) > 1;
-- empty result = deterministic = R2 becomes a hard rule

-- 3. Is Segment -> Sub-Segment strict?  (§4.1)
SELECT "Sub-Segment", COUNT(DISTINCT Segment) AS n_parents
FROM training
WHERE Segment IS NOT NULL AND "Sub-Segment" IS NOT NULL
GROUP BY "Sub-Segment"
HAVING COUNT(DISTINCT Segment) > 1;
-- empty result = strict hierarchy = R1 becomes a hard rule

-- 4. The full parent-child map
SELECT Segment, "Sub-Segment", COUNT(*) AS n
FROM training
WHERE Segment IS NOT NULL AND "Sub-Segment" IS NOT NULL
GROUP BY Segment, "Sub-Segment"
ORDER BY Segment, n DESC;

-- 5. Platform -> Brand
SELECT Brand, Platform, COUNT(*) AS n
FROM training
WHERE Platform IS NOT NULL
GROUP BY Brand, Platform
ORDER BY Brand, n DESC;

-- 6. Are the malformed rows the same rows across targets?  (§2.1)
SELECT *
FROM training
WHERE Mnfr LIKE ' %' OR Brand LIKE ' %' OR Segment LIKE ' %'
   OR "Sub-Segment" LIKE ' %' OR TargetAgeGroup LIKE ' %'
   OR Platform LIKE ' %';

-- 7. Breadcrumb -> Segment strength  (§4.5)
-- split ProductCategory on '>', take the depth-2 prefix,
-- then report the top Segment share per prefix; keep prefixes above 90%.
```

Two reminders that apply to all of them: run on **cleaned** data (§2.1), and treat
a zero count as *unobserved*, never as *impossible* (§5).

---

## 9. Open questions

1. **Is Platform populated only for J&J products?** (§4.4) Highest priority — it
   changes the scope of a target.
2. **Is `Category` a seventh classification target?** (§4.7) Cannot complete the
   map without this.
3. **Does a documented taxonomy exist from the PO?** Without one, no combination
   can be called impossible (§5).
4. **Are the near-duplicate labels in §4.2 the same class?**
5. **Is `ProductBrand` the raw version of the `Brand` target?** Affects both the
   Brand→Mnfr work and the Aryx entity-resolution caution in §7.
6. **Is Aryx integration in scope for CatalogIQ, or forward-looking?**

---

## 10. Done-when checklist

| Criterion | Where |
|---|---|
| Important target relationships documented | §3, §4 |
| Potential validation rules identified | §6 |
| Unsupported assumptions separated from demonstrated relationships | §2, §5 |
| Findings explain support for prediction validation / Aryx integration | §6, §7 |
