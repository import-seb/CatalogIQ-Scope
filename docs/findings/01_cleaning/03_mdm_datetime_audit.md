# MDM_InsertDateTime — evidence and deferred conversion

Maksim Pikalov — 2026-09-30. Reproducible evidence:
[Notebook 10](../../../notebooks/10_mdm_datetime_audit.ipynb), executed from a fresh
kernel against the same source hashes as the [integration run](02_integration_validation.md).

## What was checked

Follow-up: the [2026-10-02 conversion and disputed-row audit](04_mdm_timestamp_followup.md)
lists all timestamp patterns and tests four encodings. All 14 feature/structural
disagreements contain image URLs in the timestamp field plus corroborating column
displacement; none is an alternative date format. Source values and rules remain unchanged.

Inspected every supplied CSV record, not a sample. Compared the full export with
the subset outside integrated quarantine, which still includes review holds.
Reviewed the working data contract, initial profiling, cleaning direction and
current decisions. They do not specify a confirmed date system, timezone, or
source transformation for this field. The supplied CSVs carry no Excel workbook
date-system setting. The earlier cleaning direction calls serial dates a hypothesis.

Missing means blank/whitespace or case-insensitive `null`, without modifying the
source cells. Numeric means a finite Decimal; counts are record counts.

| Value class | Training (84,552) | Target (28,082) |
| --- | ---: | ---: |
| Finite numeric | 84,150 | 27,937 |
| Missing | 118 | 37 |
| URL text | 263 | 93 |
| Other text | 21 | 15 |
| Nonfinite numeric | 0 | 0 |
| Fractional numeric (subset of finite numeric) | 38,730 | 12,902 |

Full numeric ranges are 1 to 877006774521 (training), and 3.9 to 7.62042E+11
(target). Blindly treating every numeric cell as a date would misinterpret
displaced or otherwise implausible values. Missingness alone is not evidence
that a row is corrupt; these rows were partitioned using the existing combined
rules, not a new timestamp rule.

Outside quarantine, all 84,086 training and 27,911 target values are finite
numbers. Both subsets have exactly 12 distinct numeric values, spanning `44936`
to `45177.76383`. All observed missing/text values occur in quarantine; that
association does not prove that future missing timestamps should be excluded.

## Supported hypothesis and remaining ambiguity

The common numeric values and their fractional parts are consistent with Excel
day serials. Under the 1900-system interpretation, `44936` becomes
`2023-01-10T00:00:00`, and `45117.83254` becomes
`2023-07-10T19:58:51.456000`. These are examples of a hypothesis, not validated
source timestamps. The 1904 interpretation shifts the same serials by 1,462 days,
to 2027 in these examples. Microsoft documents both systems and their offset in
[Date systems in Excel](https://support.microsoft.com/en-us/excel/date-systems-in-excel).

The 2023 interpretation is more plausible for insertion times in files received
in 2026, but the field's meaning and export encoding are not confirmed. A plausible
year does not establish the source timezone or justify an automatic conversion.

## Decision and follow-up

Preserve the raw field unchanged and keep this metadata outside the proposed
model inputs. No conversion, missing-value fill, timezone assignment, or
timestamp-based split is applied.

Confirm with the source owner:

1. Does this field contain Excel serials, and which workbook date system was used?
2. What timezone, if any, applies to the fractional time?
3. Does it mean MDM insertion time, export time, or a product event time?

If confirmed, create a separate derived ISO 8601 field:
`YYYY-MM-DDTHH:MM:SS[.fraction]`, adding an offset only when known. Preserve the
original token, explicitly handle Excel's early-date behavior, and log failed or
implausible conversions. The notebook's rendered fractional seconds demonstrate
arithmetic precision, not a claim about the source's measurement precision.
