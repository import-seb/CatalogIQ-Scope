# Current cleaning decisions

Updated 2026-10-03. These decisions govern feature, target-label, structural,
and integrated cleaning. The authority column distinguishes group agreement
reported by Sebastian from Sebastian's implementation decisions. They supersede
the earlier conservative handoff's blanket feature-review and Exclude holds.

## Cleaning philosophy

Preserve usable products. Distinguish a questionable field from an unusable or
misaligned row. Record field warnings without automatically excluding the row;
quarantine only when an explicit row rule applies. Preserve original source
values and identity for audit. Never invent labels, repair a suspected column
shift by guessing, or replace suspicious numeric values with sentinel values.

| Policy | Authority/status | Current implementation |
| --- | --- | --- |
| 1. Literal `Exclude` marker | Group agreed | Keep otherwise usable rows. Marker remains auditable; arbitrary text in Exclude is not treated as the marker. |
| 2. `MDM_InsertDateTime` as model input | Group agreed | Omit from feature exports/model inputs; retain the original value in source and audit partitions. No guessed date conversion. |
| 3. Bad timestamp alone | Sebastian decision | No timestamp flag, hold or quarantine on its own. A URL in the timestamp contributes to quarantine only with independent displacement evidence. |
| 4. Name, description and contents all missing | Sebastian decision | Structural `missing_product_text` quarantine, even if category, brand or numeric measurements are present. Any one of the three being present avoids this particular rule. |
| 5. Bad URL alone | Sebastian decision | Keep the URL flag and original value; do not hold the row. Corroborated displacement can still quarantine it. |
| 6. One feature issue generally | Sebastian decision | No automatic whole-row hold. Apply the explicit rule for that issue; a field-review flag is not a row decision. |
| 7. Evidence of shifted columns | Sebastian decision | Quarantine using documented combinations of distinct-field evidence; retain every contributing reason. |
| 8. Structural checker integration | Sebastian decision | Run independently on raw sources, verify its complete decision tables, and combine keyed quarantine decisions with feature and target-label decisions in integrated mode. |
| 9. `Sun1`-`Sun5` and `Notes` | Sebastian decision | Omit from feature exports/model inputs; retain in raw sources, complete partitions and structural evidence. Do not infer their intended meaning. |
| 10. Category-level features | Deferred to feature engineering | Later derive `retailer_category_level_N` from ProductCategory. Preserve the full path and Retailer now. No derived levels, root removal or new ampersand normalization in cleaning. |

Missing means blank/whitespace or case-insensitive `null` in the integrated and
structural passes. No fixed 11/12 missing-field cutoff remains. Statistical
missingness and text-length checks use explicit product-feature scopes and
per-source IQR profiles; both unusual missingness and unusual text length are
required for that quarantine rule. Labels, IDs and timestamps cannot distort
those profiles. Missing measurements alone are not grounds for quarantine.

## Label policies retained

- Normalize `Cold / Flu` to `Cold/Flu` in Sub-Segment.
- Normalize `Other Lifestyle` to `Other Lifestyle CHC`. The three reviewed
  records share Lifestyle CHC as their parent and do not support a separate class.
- Preserve supplied Mnfr, including missing and invalid values. Do not infer
  J&J from Brand. Populated values outside J&J / All others are quarantined under
  target reason 6 without changing the source label.
- Target reasons 2 (leading-space/lowercase formatting) and 3 (rarity) remain
  review-only. Missing labels are eligibility concerns for the corresponding
  future model, not automatic whole-row failures.
- Do not enforce a fixed Segment/Sub-Segment parent whitelist: shared parents
  can be legitimate. Single-parent count diagnostics do not change labels.

## Ownership and superseded behavior

Target reasons 4/5 (row-content checks) have moved out of target cleaning. Their
reinforced equivalents live in the structural module and run on both datasets.
The structural module does not import either cleaner or consume their masks.
Integration alone combines the three decision streams, preserving reason prefixes
`feature:`, `target:`, and `structural:`. See the
[integration contract](integration/cleaning_contract.md).

Historical runs retain their original policy versions. In particular, v3's
11/12 cutoff, the earlier standalone no-content rule, automatic feature holds,
and default Exclude holds are superseded. The prior decision to preserve raw
timestamps remains; only corroborated displacement is now actionable. Explicit
`--exclude-policy hold|quarantine` overrides remain available for comparison,
are recorded in the run summary, and are not the agreed default.

No model, feature-engineered category levels, evaluation split or inferred target
labels are introduced by these decisions. Working manufacturer/age vocabularies,
ambiguous fields and individual flagged values remain visible for team review.
