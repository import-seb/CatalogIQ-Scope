# How much of a product still needs review?

This offline replay examined 3,056 jointly labeled validation records from 1,580
product families. On 843 records a reviewer could inspect fewer than five modeled
fields. Across the 2,076 field checks this would skip, every saved prediction
matched the reference label. Every record still receives human review.

| Fields shown to reviewer | Records | Field checks |
| --- | ---: | ---: |
| Segment, Sub-Segment, Brand, Mnfr, TargetAgeGroup | 2,213 | 11,065 |
| Segment, Sub-Segment, Brand | 648 | 1,944 |
| Sub-Segment | 195 | 195 |
| Total | 3,056 | 13,204 |

The five-field baseline entails 15,280 field checks. The combined rules reduce
this count by 13.6%. This is a workload proxy. Opening a card, reading the
product and confirming the decision still take time. Reviewer time and final
export accuracy require a timed trial.

## Population and decision rules

The five selected runs cover 15,687 distinct validation IDs in total. Only
8,768 have predictions and reference labels in **all five** runs because Brand
and Sub-Segment have their own allocations. The 3,056 above belong to the
existing group-based `policy_audit` role within that intersection. Incomplete
records were excluded from the evaluation denominator. Platform is unavailable
and falls outside these five fields. Manual Platform work, if required, is extra.

The replay checks each selected manifest against the frozen route hash and
verifies predictions, context, and the ID-only protected registry against the
manifest. It checks group consistency across models and source features. It
uses the existing calibration temperatures, cutoffs and Segment class list
without fitting a model or selecting a new cutoff on these rows.

- Keep full review for missing essential text, candidate relationship warnings,
  missing or unseen input brand keys, keys with multiple training labels, and
  predicted catch-all or generic brands. Warnings are computed from these
  validation predictions. The separate target catalog's warning counts do not
  enter this replay.
- Once those checks pass, hide Mnfr and TargetAgeGroup together only when both
  calibrated scores reach their existing 0.50 cutoffs.
- Also hide Brand and Segment only when the input has a description, Brand
  reaches 0.955, Segment reaches 0.93, and the predicted Segment is one of the
  three classes in the team's guarded policy. Sub-Segment is always reviewed.

As standalone comparisons, showing just Sub-Segment on 195 records and five
fields elsewhere would skip 780 checks. Hiding only Mnfr and age on 843 records
would skip 1,686. The 195 are also in the 843, so the combined table counts
each field once.

## What to test with reviewers

The validation families and existing rules have already informed development.
Zero observed hidden errors on this subset does not establish that new uploads
are error-free. The 27,870 target predictions have no independently reviewed
reference labels in the supplied export. Avoid projecting these figures onto
that catalog.

Time two comparable batches of fresh product families in the current and
shortened review workflows. Alternate which workflow each reviewer sees first,
and give each reviewer different products in each workflow. Record minutes per
100 completed products and independently check the final labels, including
fields skipped by the pilot. Report unfamiliar brands separately. Keep the
showcase workflow on its existing human-confirmation policy during evaluation.

## Reproduce

From the repository root with the project environment active and this checkout's
`src` on the Python path, run:

```text
python scripts/replay_partial_review.py <predictions_contract_v2_20261031.zip> <frozen-split-dir>/validation.csv
```

Keep the split's existing `completion.json` beside `validation.csv`. The
script reads the two local inputs and prints aggregate counts. The shared
branch contains no product rows.
