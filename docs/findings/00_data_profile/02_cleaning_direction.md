# CatalogIQ Cleaning Guidance

## Purpose

This document summarizes the main data-quality findings from the initial profiling work and provides guidance for the cleaning stage.

The goal is to preserve the original information where possible, remove obvious structural corruption, and avoid making assumptions that belong in later modeling work.

## 1. Preserve Raw Data

The original supplied CSV files should remain unchanged.

Cleaning should create new cleaned outputs rather than modifying the raw files in place.

## 2. Malformed or Shifted Records

Several rows appear to have values shifted across columns.

Examples observed include:

* `ProductRating` containing product-description text
* `ProductReviewsCount` containing descriptive text
* `ReviewsCount` containing descriptive text
* `XRatXRev` containing descriptive text
* `MDM_Id` containing image URLs, Amazon ASINs, or product URLs
* target columns containing ingredient or product-description fragments
* `Retailer` containing descriptive text instead of a retailer name
* `ProductCategory` containing values such as `cure`

These patterns suggest that some source rows may have been malformed during export or parsing.

### Cleaning recommendation

Do not fix only the individual bad cell.

When an obviously misplaced value is found:

1. flag the entire row,
2. inspect neighboring columns,
3. determine whether the row can be reliably reconstructed,
4. otherwise quarantine or exclude the row from modeling.

The cleaner should record how many rows were flagged and how many were ultimately retained or removed.

## 3. Numeric-Looking Fields

Several fields appear to represent numeric measurements but are currently loaded as strings.

Examples include:

* `ProductRating`
* `ProductReviewsCount`
* `ReviewsCount`
* `XRatXRev`

Before conversion, inspect non-numeric values and remove or handle malformed rows.

Do not blindly coerce all invalid values to missing because some failures appear to result from shifted rows rather than ordinary dirty values.

## 4. Identifier Fields

The following fields should be treated as identifiers rather than numeric measurements:

* `JoiningKey`
* `Sku`
* `Upc`
* `ProductModelNumber`
* `MDM_Id`

`Upc` may appear in scientific notation, such as:

`3.00054E+11`

UPC values should be preserved as strings where possible because numeric conversion may remove leading zeros or alter formatting.

## 5. `MDM_InsertDateTime`

Some `MDM_InsertDateTime` values appear as decimal values such as:

`45117.83254`

These may represent Excel serial date/time values.

The cleaning stage should determine whether:

* all values use Excel serial dates,
* multiple datetime formats are present,
* malformed records are contributing invalid values.

Convert to a standard datetime format only after confirming the source format.

## 6. `Sun1`–`Sun5`

`Sun1` through `Sun5` are nearly 100% missing.

The few populated values contain fragments of:

* ingredients
* nutrition information
* product descriptions
* marketing text

Some text continues sequentially across `Sun1` through `Sun5`, suggesting malformed or shifted source records.

The intended meaning of these columns is unknown.

### Suggested handling

* Preserve the columns in raw data.
* Exclude them from the current modeling feature set.
* Do not permanently remove them from the ingestion pipeline.
* Monitor their population rate in future datasets.
* If any `Sun` column becomes more than approximately 1% populated, flag it for review before modeling.
* Rows containing unexpected `Sun` values should also be considered candidates for malformed-row inspection.

## 7. `Notes`

`Notes` is effectively 100% missing in the current datasets.

It should not be used as a modeling feature for the current dataset.

Preserve the field in the source schema in case future data releases begin populating it.

## 8. High-Missingness Feature Columns

Several non-target fields have substantial missingness in both training and target data.

Approximate missingness includes:

* `Exclude`: ~93%
* `ProductModelNumber`: ~86%
* `Upc`: ~80%
* `ProductDescription`: ~68%
* `ProductContents`: ~60%

The similar missingness patterns between training and target datasets suggest that this is largely structural rather than unique to one split.

Do not automatically drop these columns solely based on missing percentage.

Feature usefulness should be evaluated later during modeling.

## 9. Target Dataset

The following classification columns are completely missing in the target dataset:

* `Category`
* `Mnfr`
* `Brand`
* `Platform`
* `Segment`
* `Sub-Segment`
* `TargetAgeGroup`

These columns should be excluded when checking whether a target row has excessive missingness.

They are intentionally blank and should not cause a row to be flagged as incomplete.

## 10. Training Target Labels

Target columns contain a mixture of:

* legitimate labels
* missing labels
* obvious malformed values

Rare labels should not automatically be removed.

Rarity alone is not evidence of corruption.

### `Mnfr`

The main observed valid labels appear to be:

* `All others`
* `J&J`

Other observed values such as product-description fragments appear malformed.

Rows containing those malformed values should be inspected as possible shifted records.

### `Brand`

Most Brand values appear legitimate.

Examples of suspicious values include:

* `gluten-free`
* `tea`

Because Brand contains many legitimate low-frequency classes, do not use frequency alone to determine whether a label is invalid.

### `Platform`

`Platform` is extremely sparse:

* 84,552 total training rows
* 1,105 labeled rows
* 83,447 missing
* 98.69% missing

Some non-missing values also appear malformed.

### Suggested handling

* Preserve missing Platform values as missing.
* Do not convert missing values to `Unknown`, `None`, or `Not Applicable`.
* Remove or quarantine clearly malformed rows.
* Preserve legitimate rare Platform labels.
* Leave the question of whether 1,105 labeled rows are sufficient to the modeling stage.

`Platform` should remain a prediction target unless the project scope changes.

### `Segment`

Most Segment labels appear valid.

Clearly suspicious values include product-description fragments such as:

* `IGEN Non-GMO tested...`
* `Keto certified`

These rows should be flagged for structural review.

### `Sub-Segment`

Most Sub-Segment labels appear valid.

Suspicious values include:

* `lactose`
* `gluten-free`

Again, rare but semantically valid categories should not be removed simply because they occur infrequently.

### `TargetAgeGroup`

The apparent valid taxonomy is:

* `Adult`
* `Children`
* `Infant`

Other observed values contain product-description or ingredient text and appear malformed.

Rows containing those values should be reviewed for column shifting.

## 11. Missing Target Labels

Missing labels in the training data should remain missing.

Do not create artificial classes such as:

* `Unknown`
* `Missing`
* `Not Applicable`

unless the dataset documentation explicitly defines such a category.

Different models may later use different subsets of training rows depending on which target is being predicted.

For example, a Platform model would train only on rows with a valid Platform label.

## 12. `ProductCategory`

`ProductCategory` appears to contain hierarchical retailer taxonomy paths separated by `>`.

Examples:

`Health & Household > Health Care > OTC Medications & Treatments > Pain Relievers`

Hierarchy depth varies across products.

Observed depths range mainly from 2 to 7 levels.

This variable depth is not inherently an error.

### Known issues

Some malformed records were identified:

* a one-level value of `cure`
* duplicated category paths
* repeated breadcrumb structures
* inconsistent spacing around `>`

Examples of duplication include a category path repeated twice inside the same value.

### Cleaning recommendation

Normalize formatting where safe:

* trim whitespace
* standardize spacing around `>`
* detect exact repeated breadcrumb sequences
* collapse obvious duplicated paths when the correction is unambiguous

Do not force all category paths to the same depth.

The hierarchy may later be useful as:

* the complete category path
* separate category levels
* parent category features

## 13. Do Not Over-Clean

The cleaner should avoid:

* dropping rare labels simply because they are rare
* replacing missing targets with invented classes
* converting identifier columns to numeric types
* forcing all category hierarchies to equal depth
* deleting rows based only on one suspicious value without inspecting the full row
* permanently removing unknown source columns solely because they are currently sparse

## 14. Recommended Cleaning Outputs

The cleaning process should produce:

* a cleaned dataset
* a list of removed or quarantined row indices
* reasons each row was flagged
* summary counts of malformed rows
* summary of datatype conversions
* summary of label corrections or removals
* summary of category normalization
* remaining missingness after cleaning

The cleaning stage should also preserve enough information to trace any cleaned row back to the original source record.

## 15. Guiding Principle

When the intended meaning of a value is clear and the correction is deterministic, clean it.

When the intended meaning is uncertain, flag it rather than guessing.

The raw data should always remain recoverable.