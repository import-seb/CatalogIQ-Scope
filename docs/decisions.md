# Current cleaning decisions

These decisions govern `catalogiq.cleaning`. Historical notebooks may retain withdrawn assumptions; see the [analysis index](../notebooks/README.md).

1. We avoided enforcing a fixed Segment–Sub-Segment whitelist because some Sub-Segments legitimately appear under multiple Segments; instead, we apply incremental structural checks, such as validating count relationships only for Sub-Segments that have exactly one observed parent. (Sebastian, 9/27/2026)
2. Decision Note — Normalize `Other Lifestyle` to `Other Lifestyle CHC`

The three `Other Lifestyle` records were normalized to `Other Lifestyle CHC`. Both labels occur under the same parent Segment, `Lifestyle CHC`, and the smaller class contains only three products, including blister lip balm and an `ORGANIC VITALITY Premium Berberine HCL Supplement - 1200mg of Berberine Per...`. These products do not demonstrate a clear, distinct taxonomy from the 195-record `Other Lifestyle CHC` class. Given the shared parent, near-identical label meaning, and very small support for `Other Lifestyle`, the smaller label was treated as an inconsistent naming variant rather than a separate Sub-Segment.

3. Decision Note - Preserve supplied manufacturer labels

Do not infer or overwrite `Mnfr` from `Brand`. The earlier J&J reassignment decision is withdrawn: a dominant association does not establish that minority labels or missing values should become `J&J`. Preserve supplied values, including missing values.

Keep vocabulary validation only: populated values outside `J&J` and `All others` are flagged for review, not replaced.

4. Decision Note - Reason 2 is review-only

Reason 2 (label starts with a space or lowercase character) adds a flag but does
not quarantine a row. Reason 3 remains review-only as before. Reasons 4, 5, and 6
still quarantine records; having reason 2 as well does not override them.
Manufacturer values remain unchanged. The earlier all-reasons-flag-only change
was a misunderstanding and is withdrawn.
