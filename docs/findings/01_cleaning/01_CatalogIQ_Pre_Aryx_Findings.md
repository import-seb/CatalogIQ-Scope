# CatalogIQ: What We Know Before Aryx

> **Policy update:** The manufacturer-whitelist discussion below predates the withdrawal of Brand-to-J&J reassignment. Preserve supplied Mnfr values, including blanks. Treat that discussion as historical; [current decisions](../../decisions.md) and the package define current cleaning behavior.

This is the short version of what we learned from profiling and cleaning the CatalogIQ data.

The rule we have followed so far is simple:

> **Fix what we can prove is wrong. Flag what looks suspicious. Leave uncertain taxonomy alone.**

## The big picture

The training data has **84,552 rows** and the target data has **28,082 rows**.

The six prediction targets are:

- `Mnfr`
- `Brand`
- `Platform`
- `Segment`
- `Sub-Segment`
- `TargetAgeGroup`

We found real data-quality problems, but we also found several cases where something that first looked "wrong" was probably intentional taxonomy.

That distinction matters.

---

## 1. Some rows are genuinely broken

There are rows where values appear shifted into the wrong columns.

Examples include:

- description text inside target fields,
- text inside numeric fields,
- URLs inside `MDM_Id`,
- obvious mismatches between `ProductBrand` and `ProductName`.

These are the best candidates for **whole-row quarantine**.

We are not quarantining rows just because a label is rare or unusual.

---

## 2. We do not have a trustworthy "same product" key

This was one of the biggest findings.

We tested UPC, SKU, MDM_Id, ProductName, URLs, and repeated combinations.

None of them safely defines a canonical product.

Examples:

- the same UPC can appear on different products,
- UPCs can be damaged by scientific notation,
- some SKU values contain descriptive text,
- repeated SKU groups can contain different listings,
- `MDM_Id` does not behave like a product ID,
- the same ProductName can still represent different variants.

So for now:

> **Do not merge rows or transfer labels just because two records appear to describe the same product.**

This is why we dropped automatic Platform backfilling.

---

## 3. Segment and Sub-Segment are related, but not a strict tree

We initially expected each Sub-Segment to belong to one Segment.

That is not always true.

Some Sub-Segments appear under multiple Segments, so we are **not** enforcing a global parent-child whitelist.

We only made obvious formatting/taxonomy fixes where the evidence was strong:

- `Cold / Flu` -> `Cold/Flu`
- `Other Lifestyle` -> `Other Lifestyle CHC`

---

## 4. Manufacturer is one of the easier targets

`Mnfr` is basically a two-value controlled field:

- `J&J`
- `All others`

A few rows contained obvious junk values.

We also found a strong relationship between known J&J brands and `Mnfr`, so this is one area where a small validated whitelist makes sense.

---

## 5. Platform is sparse and should stay sparse for now

Platform is missing on most training rows.

At first it looked almost J&J-specific, but that was not true. Some non-J&J brands, especially Excedrin, also use Platform labels heavily.

So Platform appears to depend more on **brand/product-line structure** than manufacturer alone.

We tested automatic backfilling, then removed it.

Why?

Because we cannot reliably prove that two rows are the same underlying product.

Decision:

> **Missing Platform stays missing. No automatic backfill.**

---

## 6. ProductCategory is useful, but not a cleaning rule

Retailer breadcrumbs in `ProductCategory` contain useful classification signal.

But they are not consistent enough to force Segment labels.

So:

> **Keep ProductCategory as a modeling/evidence feature, not a hard cleaner.**

---

## 7. TargetAgeGroup has useful clues, not hard rules

The main labels are:

- Adult
- Children
- Infant

Text cues such as `baby`, `infant`, `children`, and `kids` are useful signals.

But we are not using them to rewrite labels automatically.

They are better suited for modeling and review.

---

# The interesting part: Brand vs ProductBrand

This is the biggest unresolved relationship.

`ProductBrand` looks like the literal brand from the retailer.

`Brand` looks like CatalogIQ's classification of that brand.

The relationship is strong:

- about **95.5%** of ProductBrands with labels map to one Brand value,
- those mappings cover about **83% of training rows**.

But many mappings only have one example, so "deterministic" does not always mean "well supported."

We are using strong ProductBrand -> Brand relationships as **review evidence**, not automatic relabeling.

---

# `UnItemised brand` and `Other Brands`

These looked suspicious at first because they dominate the Brand column.

They now look intentional.

We found roughly:

- **53.6k `UnItemised brand` rows**
- **3.5k `Other Brands` rows**
- **640 ProductBrands appear in both buckets**
- some ProductBrands also switch between a bucket and their own named Brand

That means `UnItemised brand` is probably not just "missing brand."

A reasonable working hypothesis is:

```text
ProductBrand
   ↓
CatalogIQ classification decision
   ↓
Named Brand
or
UnItemised brand
or
Other Brands
```

We do **not** yet know the rule that decides between those outcomes.

For example, a ProductBrand might map:

```text
103 times -> UnItemised brand
1 time    -> Other Brands
```

That one row is worth reviewing.

But the 103-to-1 ratio does **not** prove the one row is wrong.

So for now:

> **Keep both `UnItemised brand` and `Other Brands` exactly as valid labels.**

---

# What we are deliberately NOT doing

For now we are avoiding:

- Platform backfilling
- Brand backfilling
- majority-vote relabeling
- merging `UnItemised brand` and `Other Brands`
- replacing bucket labels with ProductBrand
- forcing every Sub-Segment under one Segment
- trusting UPC/SKU/MDM_Id as product identity
- treating rare labels as bad labels

---

# Why Aryx is next

The cleaner should remove known garbage first.

Then Aryx can focus on the relationships we still do not understand.

The biggest question is:

> **Why can the same ProductBrand become a named Brand, `UnItemised brand`, or `Other Brands`?**

Useful context for Aryx to investigate:

- Retailer
- ProductCategory
- Segment
- Sub-Segment
- ProductName / product-line wording

Aryx should discover patterns and return provenance, counts, and exceptions.

Then we verify the important claims with pandas before changing the cleaning pipeline.

---

## Current position

We have already learned enough to avoid several bad cleaning decisions.

The next problem is less about obvious dirty data and more about understanding **CatalogIQ's hidden taxonomy logic**.

That is what Aryx is being used for.
