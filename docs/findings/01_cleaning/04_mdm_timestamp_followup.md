# Timestamp conversions and the 14 disputed rows

Audit date: 2026-10-02. Full original training (84,552 records) and target
(28,082 records) files; original strings unchanged. Source hashes match the
[earlier timestamp audit](03_mdm_datetime_audit.md). Both feature and structural
audits were independently validated before selecting their 12 training / 2 target
disagreements by `(dataset, source_sha256, source_row)`. Record numbers below are
canonical **one-based parsed records**, not physical lines.

## Raw timestamp patterns

Counts below partition every source record. Numeric categories describe literal
formatting, not validity as a timestamp.

| Pattern in `MDM_InsertDateTime` | Training | Target |
| --- | ---: | ---: |
| Integer numeric literal | 45,419 | 15,034 |
| Decimal numeric literal | 38,730 | 12,902 |
| Scientific-notation numeric literal | 1 | 1 |
| Missing (`null`) | 118 | 37 |
| Single image URL | 10 | 3 |
| Pipe-separated image-URL list | 147 | 57 |
| Product-page URL | 106 | 33 |
| ASIN-shaped identifier | 14 | 13 |
| Other identifier/model-code text | 6 | 2 |
| Other descriptive text | 1 | 0 |
| **Total** | **84,552** | **28,082** |

There are no nonfinite numeric tokens and no recognizable textual calendar dates.
All nonnumeric tokens were inspected, not only attempted with a permissive date
parser. Identifiers include codes such as `TK-HP2412`, `MAV17199`, `PM02` and
`R-U005`; ASIN shape is a lexical observation, not externally verified identity.
Images include Amazon, Walmart and one CVS JPEG. The remaining text is a
product-description fragment. Missingness diagnostics trim whitespace and recognize
case-insensitive `null`; raw tokens remain preserved in local artifacts.

## Conversion trials

Every distinct finite numeric token was tried as Excel 1900 days, Excel 1904 days,
Unix seconds and Unix milliseconds. Results are weighted by source frequency.
Decimal arithmetic preserves fractional-day precision before rounding to Python
datetime microseconds. Excel 1900 serials below 60 use their correct early-date
offset, and fictitious serial day 60 is rejected; none of the supplied tokens
lands on that day. Conversion checks include Microsoft's known example
40729 (1900) / 39267 (1904) = 2011-07-05, and explicit day 59/60/61 boundaries.

Microsoft documents the two Excel systems and their 1,462-day offset:
[Date systems in Excel](https://support.microsoft.com/en-us/excel/date-systems-in-excel).

The same 12 dominant serials occur in both datasets:

| Raw serial | Excel 1900 date/time (timezone unknown) | Training | Target |
| --- | --- | ---: | ---: |
| 44936 | 2023-01-10 00:00:00 | 36,118 | 11,992 |
| 44967 | 2023-02-10 00:00:00 | 9,259 | 3,026 |
| 44977.33627 | 2023-02-20 08:04:13.728 | 87 | 48 |
| 44977.3512 | 2023-02-20 08:25:43.680 | 10 | 3 |
| 44998.35338 | 2023-03-13 08:28:52.032 | 5,624 | 1,797 |
| 45027.6581 | 2023-04-11 15:47:39.840 | 5,413 | 1,821 |
| 45055.70023 | 2023-05-09 16:48:19.872 | 4,194 | 1,384 |
| 45058.43501 | 2023-05-12 10:26:24.864 | 151 | 61 |
| 45089.7987 | 2023-06-12 19:10:07.680 | 4,594 | 1,469 |
| 45117.83254 | 2023-07-10 19:58:51.456 | 7,991 | 2,704 |
| 45145.83061 | 2023-08-07 19:56:04.704 | 6,083 | 1,966 |
| 45177.76383 | 2023-09-08 18:19:54.912 | 4,562 | 1,640 |
| **Total** | | **84,086** | **27,911** |

| Interpretation | Observed result | Assessment |
| --- | --- | --- |
| Excel 1900 | The dominant 111,997 records become 2023-01-10 through 2023-09-08. | Strongly consistent with past batch ingestion dates, though source epoch/timezone still need confirmation. |
| Excel 1904 | Those same records become 2027-01-11 through 2027-09-09. | Future relative to the files already available on 2026-10-02; implausible as actual past insertion dates. |
| Unix seconds | 84,145 / 27,935 numeric records become times on 1970-01-01; 5 / 2 overflow Python's calendar range. | Does not support a plausible insertion-date interpretation. |
| Unix milliseconds | 84,145 / 27,935 numeric records become times on 1970-01-01; the remaining 5 / 2 become 1990–1997 dates. | Arithmetic conversion succeeds, but the distribution is inconsistent with the strong Excel batch-date pattern. |

There are **64 training and 26 target numeric outliers** beyond those 12 serials.
Under Excel 1900, 58 / 24 become 1900–1904 dates, one training value (`48717`)
becomes 2033-05-18, and 5 / 2 overflow. Small values resemble ratings/counts;
large values resemble identifier numbers. These are hypotheses about displaced
fields, not proof of their intended columns. Do not convert every numeric value
blindly, or infer timestamp validity merely from successful conversion.

No arbitrary acceptable-year threshold was used: the audit reports years,
conversion failures, ranges and dates later than the explicit audit date. The
2023 dates are plausible, not confirmed. No timezone or date-based split is inferred.

## Every disputed row

**All 14 timestamps contain image URLs, not recognizable dates.** Twelve are URL
lists; two are single URLs. Nine rows contain Amazon image URLs and five contain
Walmart image URLs. Every row also has all four corroborating observations:

1. Product-description text in `ProductRating`.
2. Missing `ProductUrl`.
3. A product-page URL in `ProductImageUrl`.
4. A value from the 12 common timestamp serials in the adjacent `MDM_Id` column.

This repeated arrangement strongly supports row displacement. It does not prove
that shifting the entire row one column would repair it: surrounding values and
embedded quote/comma patterns need source-owner review before reconstruction.

| Dataset | source_row | Image URLs in timestamp | Raw `MDM_Id` | Excel 1900 interpretation of `MDM_Id` |
| --- | ---: | ---: | --- | --- |
| training | 12663 | 5 | 45055.70023 | 2023-05-09 16:48:19.872 |
| training | 13520 | 7 | 45117.83254 | 2023-07-10 19:58:51.456 |
| training | 16231 | 4 | 45145.83061 | 2023-08-07 19:56:04.704 |
| training | 35791 | 9 | 44936 | 2023-01-10 00:00:00 |
| training | 36169 | 1 | 45117.83254 | 2023-07-10 19:58:51.456 |
| training | 36606 | 7 | 45177.76383 | 2023-09-08 18:19:54.912 |
| training | 39066 | 1 | 45145.83061 | 2023-08-07 19:56:04.704 |
| training | 40625 | 5 | 45145.83061 | 2023-08-07 19:56:04.704 |
| training | 48663 | 3 | 44936 | 2023-01-10 00:00:00 |
| training | 51791 | 7 | 44936 | 2023-01-10 00:00:00 |
| training | 59598 | 19 | 45117.83254 | 2023-07-10 19:58:51.456 |
| training | 69839 | 7 | 44936 | 2023-01-10 00:00:00 |
| target | 9231 | 8 | 45089.7987 | 2023-06-12 19:10:07.680 |
| target | 26999 | 4 | 45055.70023 | 2023-05-09 16:48:19.872 |

Full hashes and original values for every disputed record are retained locally
in `data/processed/timestamp_followup_20261002/disputed_rows.jsonl` and the focused
`disputed_rows.csv`. `timestamp_values.csv` records every raw timestamp token and
frequency; `numeric_conversion_trials.jsonl` records every numeric conversion
attempt. `summary.json` records source hashes, aggregate counts and output hashes.
These private artifacts remain ignored. Source files and decision code were not
changed, and no masks were combined or applied.

## Revised assessment

The previous caution that these 14 differences might be legitimate alternative
timestamp formats is resolved for the observed records: **none is such a date**.
Max's quarantine recommendations for these rows are supported by multiple concrete
signs of displacement. A future narrow rule should recognize URLs in the timestamp
field together with independent numeric/column evidence, while keeping unfamiliar
but plausible date formats review-only. This audit does not implement that policy
change or authorize rewriting/reconstructing source records.
