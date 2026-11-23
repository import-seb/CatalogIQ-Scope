# Whole-card review without a queue quota

For each hypothetical skip, all five modeled fields must reach the indicated
calibrated score. The product must have a name, a nonmissing input brand with
a single observed training label, a named Brand suggestion, and no candidate
relationship warnings. The target replay also requires an eligible row and all
five suggestions from the selected models. Platform is outside this scope.
The lowest field score is a filter, not a probability that the entire card
is correct.

| Minimum score | Validation cards skipped | Distinct families skipped | Skipped cards with any wrong label | Target cards skipped | Target cards to review |
| --- | ---: | ---: | ---: | ---: | ---: |
| .95 | 527 | 239 | 7 | 4,537 | 23,333 |
| .98 | 380 | 172 | 0 | 3,078 | 24,792 |
| .99 | 198 | 102 | 0 | 1,522 | 26,348 |
| .995 | 0 | 0 | 0 | 0 | 27,870 |

Of 3,056 jointly labeled audit cards, 408 had at least one wrong label.
At .98, all 408 would have remained visible for review. The audit has been
consulted in previous experiments. Products within a family may share errors,
and the target catalog has no independent reference labels. The target totals
measure queue size only. They provide no measured count of escaping errors.

Freeze the .98 candidate before collecting fresh human labels. Randomly sample
from its 3,078 target candidates with known sampling probabilities and examine
family and retailer coverage. Have reviewers independently check all five labels.
Define an acceptable whole-card error rate and critical-error criteria with the
Product Owner before using those results to change review policy. If revised
after the check, qualify the revision on separate data.

Reproduce with the project environment and this checkout's `src` on the Python path:

```text
python scripts/replay_card_risk.py <handoff.zip> <split-dir>/validation.csv <target_predictions.csv> <relationship_conflicts.csv>
```
