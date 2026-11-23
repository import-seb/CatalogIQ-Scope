# Can 1,000 manual reviews catch the faulty products?

The target catalog contains 27,870 products. A queue of 1,000 would send
3.59% to review. The quality measure is the share of all incorrectly classified
products routed to that queue. A product counts once if any of its five modeled
labels is wrong. Platform is unavailable and outside the calculation.

## Replay on labeled products

The shared policy-audit cohort described in [the partial review experiment](partial_review_pilot.md)
has 3,056 products. Of these, 408 have at least one wrong label, including
343 with a wrong Sub-Segment. The nearest equivalent review budget is 110
products. Even with perfect knowledge, 110 reviews can cover at most 110 of
the 408 faulty products (27.0%), if each review handles one card.

| Queue ranking | Faulty products sent to review | Faulty products outside | Share caught |
| --- | ---: | ---: | ---: |
| Lowest calibrated field score first | 60 | 348 | 14.7% |
| Relationship warnings first, then lowest score | 83 | 325 | 20.3% |

Ranking uses only model output and candidate warnings, with stable record ID
tie breaking. Reference labels are used for scoring. A card sent for review
is counted as reachable, without assuming the reviewer actually corrects it.

## What the target catalog can tell us

The 651 warnings map to 650 unique products. Another routing signal is the
input brand: 3,841 products have a missing key or a key unseen by the training
lookup. The union of these two sets has 4,405 products. Neither flag alone
establishes that a label is wrong. The target export lacks independent truth,
so it gives queue sizes, not target error recall. Do not extrapolate the audit
error rate directly to the target catalog.

The results point to Sub-Segment as the first place to investigate repeated
errors and upstream fixes. A reusable correction across genuinely equivalent
products could affect more than one card per human decision. Such grouping
would need its own validation for conflicting labels within a group.

Reproduce with the same environment and split as the partial review replay:

```text
python scripts/replay_card_budget.py <handoff.zip> <split-dir>/validation.csv <target_predictions.csv> <relationship_conflicts.csv> --budget 1000
```
