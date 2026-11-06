# Review-policy proposal for Ravi

November 6, 2026. **Recommend a reviewer-confirmed launch pilot: show model
suggestions, but require a person to confirm every required classification
before a completed CSV export.** Use confidence and known problems to prioritize
review. Keep automatic acceptance disabled while we collect fresh evidence and
measure reviewer workload. This proposal requires Ravi/team approval; it does
not change any deployed policy or model-selection decision.

## Meeting-ready explanation

"CatalogIQ is already a useful sorting assistant. Our manufacturer and age
models get more than 99% right on existing validation data. Segment gets about
97%, Brand lookup 96%, and detailed Sub-Segment 89%. Those averages hide weaker
small categories, and a very confident answer can still be wrong.

"We recommend launching with suggestions that people confirm or correct. This
lets us measure time saved and the quality of corrected exports without claiming
that the model's confidence is a guarantee. The strongest future opportunity
to reduce checking is manufacturer and age. Segment and Brand need more careful
gates; detailed Sub-Segment and Platform should stay with reviewers.

"Our proposed future quality target is at least 99% correct among each target's
answers used without review, with a defensible uncertainty check and explicit
review/rare-class coverage. That is different from saying the whole product or
the final corrected spreadsheet is 99% correct. We want your help validating
which mistakes matter most, how much review the team can handle, which fields
must be completed, and what proof is needed before we reduce checking."

## What the evidence supports

The [completed calibration assessment](../findings/03_modeling/17_confidence_calibration_review_20261103.md)
and [99% uncertainty reassessment](../findings/03_modeling/18_accuracy_standard_reassessment_20261103.md)
use frozen product groups, saved probabilities and separate calibration-fit,
threshold-selection and policy-audit roles. The audit roles contain roughly
4,000 labeled records per target, not new production products. The validation
data had already influenced development, so these are opportunities to qualify,
not a fresh production guarantee.

| Existing experimental recipe, not an operating default | Review per 100 audit inputs | Errors escaping review / unflagged rows | Observed unflagged accuracy | Preferred one-sided 95% lower bound |
|---|---:|---:|---:|---:|
| Mnfr: calibrated .50 floor | 0 | 1 / 4,076 | 99.975% | 99.525% |
| Age: calibrated .50 floor | 0 | 5 / 4,010 | 99.875% | 99.347% |
| Segment: calibrated .93 | 7.9 | 31 / 3,803 | 99.185% | 98.369% |
| Brand lookup: calibrated .955, seen keys | 24.1 | 22 / 3,298 | 99.333% | 98.354% |
| Sub-Segment: calibrated .977, provisional | 54.0 | 22 / 1,822 | 98.793% | 97.359% |

Mnfr and age clear the revised check only under its stated family-sampling
assumptions, including limits on how family size relates to error risk. With a
six-model confidence adjustment, their lower bounds remain 99.261% and 99.048%.
The .50 floor does not mean their suggestions are coin flips: all audit records
happen to exceed it. Raising Mnfr's cutoff to .99 still misses its one mistake
while flagging two correct records. More confidence filtering is not always
more useful review.

The five target percentages cannot be added to get a product-level queue: targets
share products and have different validation allocations. A complete product
can require review because of any target. No full-product workload, accuracy
after human correction or reviewer throughput has been measured yet.

## Three operating approaches

These are choices for discussion. **Only A is recommended as the starting
operating proposal; B and C are historical replays/future candidates requiring
fresh qualification and approval.** The table quantifies per-target eligibility,
not permission to automate, and does not assume humans correct every error.

| Approach | Segment | Brand lookup | Mnfr / age | Sub-Segment / Platform | Benefit and cost |
|---|---|---|---|---|---|
| **A. Conservative: recommended launch pilot** | Review 100% | Review 100% | Review 100% | Review 100%; Platform manual/unavailable | Every required label gets a human decision. Highest review burden; actual corrected accuracy and time savings must be measured. No model output bypasses review. |
| **B. Balanced: guarded future qualification** | Existing .93 + three predicted-class gate: review 30.7%, 7 escaping errors / 2,864 eligible, 99.756% observed accuracy | Existing .955 + seen-key/description gate: review 75.9%, 3 / 1,049, 99.714% | Existing recipes above: 0% hypothetical review and 1 / 5 audit errors respectively | Keep 100% review/manual handling | Far fewer escaping Segment/Brand errors than C, but more review. Gates remain exploratory and do not clear the adjusted qualification checks. |
| **C. Efficiency-focused: confidence-only future candidate** | Existing .93: review 7.9%, 31 / 3,803 escaping errors | Existing .955: review 24.1%, 22 / 3,298 | Same conditional recipes as B | Sub-Segment .977 replay: review 54.0%, 22 / 1,822; Platform manual | Lowest measured target review load among these candidates, with more escaping errors and weak minority coverage. Not recommended for unattended use. Sub-Segment remains provisional. |

For B versus C, Segment routes about 23 more products per 100 to review and
avoids 24 escaping errors in 4,131 audit inputs. Brand routes about 52 more per
100 to review and avoids 19 errors in 4,345 inputs. These are sample counts,
not forecasts of live savings. B's Segment gate allows predicted External
Analgesics, Lifestyle CHC and Vitamins/Minerals/Supplements; it is not a newly
asserted taxonomy hierarchy. Brand's description gate is an observational
association, not an input to the lookup or proof of causation.

The [product-rule analysis](../findings/03_modeling/19_product_specific_review_rules_20261103.md)
gives B's Segment a two-policy-adjusted lower bound of 98.864% and Brand
98.650%, both below 99%. C's lookup and Segment also fail the preferred check.
A newly observed good result at another audit cutoff cannot be promoted
without new confirmation. B is a frozen candidate to test, not an approved plan.

Proposed progression: run A, measure actual reviewer time and corrections,
collect fresh representative development families and freeze the future policy
before scoring that data. Qualify Mnfr/age first; test B's Segment gate next;
keep Brand's costly description gate secondary and Sub-Segment manual. Require
coverage by retailer and uncommon label, and deliberately test large families.
Do not retrain models or open the protected test for this ticket.

## Proposed meaning of the 99% goal

Recommend **at least 99% correctness per target among decisions used without
human review**, assessed on a fresh representative collection under a fixed,
versioned policy. Require a predeclared one-sided 95% lower-confidence check
that respects product families and the number of simultaneous target claims;
document its sampling assumptions. Pair it with review coverage, escaping errors
per 100 inputs, rare-class coverage and a separate critical-error measure.

This is useful when the business wants to reduce review. It can be gamed by
answering only easy/common labels, so accuracy alone is not sufficient. Propose
reporting coverage separately for every relevant class/retailer, declining
class-specific claims with inadequate support, and setting workload/critical-
error guardrails with Ravi before qualification.

If the business primarily values the delivered spreadsheet, audited accuracy
of **final corrected exports** plus reviewer minutes per product may be the
better launch KPI. Measure it separately; do not give reviewers assumed 100%
accuracy. If every required field must be correct, use an additional whole-
product correctness KPI. Neither can be inferred from today's per-target
validation figures, averaged target accuracies or multiplied probabilities.

## Five decisions to discuss with Ravi, in priority order

1. **Which quality outcome matters first?**
   **Finding:** manufacturer/age exceed 99% overall validation accuracy, while
   Segment/Brand can exceed 99% only on filtered subsets and detailed labels
   remain weak. Whole-product/corrected-export quality is unmeasured.
   **Proposal:** use reviewer-confirmed exports and measured reviewer time for
   launch; use per-target unreviewed accuracy of at least 99%, with uncertainty
   and coverage, as the later automation qualification goal.
   **Tradeoff:** measurable quality goals require separate samples and prevent
   a simple but misleading single accuracy headline.
   **Question:** "Is the first business promise a correct delivered spreadsheet,
   reduced checking of certain fields, or both—and which products/retailers must
   it cover?"
   **Decision impact:** determines required KPIs, sampling population and release
   conditions; it does not choose an arbitrary confidence cutoff.

2. **Which mistakes are most costly?**
   **Finding:** Infant, J&J, Brand catch-alls and rare Sub-Segment classes have
   limited/uneven support; confidence can miss their errors.
   **Proposal:** keep all launch decisions reviewed and prioritize these cases;
   define a separate critical-error list and qualification gate before reducing
   review, instead of treating every mistake as equally costly.
   **Tradeoff:** more expert attention and potentially lower automation coverage.
   **Question:** "Which wrong classifications cause the most rework or business
   damage, and are any unacceptable even when overall accuracy is above 99%?"
   **Decision impact:** changes queue priority and future class-specific release
   restrictions; no current model/label is automatically rewritten.

3. **What review capacity and turnaround are realistic?**
   **Finding:** B adds about 23 Segment and 52 Brand reviews per 100 target inputs
   versus C; no actual reviewer speed or product-level union workload is known.
   **Proposal:** time a representative pilot under A, measure product-level
   queue union and correction rate, then set daily capacity and turnaround limits.
   **Tradeoff:** A delays unattended throughput but avoids planning around
   speculative minutes saved or summed target percentages.
   **Question:** "Who will review, how many products arrive per day, and what
   turnaround can the team actually support during a pilot?"
   **Decision impact:** sets pilot size, staffing and sequencing of qualification;
   insufficient capacity means narrowing scope, not silently lowering quality.

4. **Who owns disputed labels and required export fields?**
   **Finding:** retailer Vitamins versus curated Digestive Health can both be
   legitimate. Maria found the breadcrumb names the model's prediction on 93 of
   116 Segment errors across that boundary; that observation does not prove the
   curated truth is wrong. Strict single-parent rules are unreliable, and
   Platform is deferred.
   **Proposal:** preserve source taxonomy and curated suggestions, designate a
   taxonomy owner, and explicitly decide whether Platform needs manual completion
   or may remain unavailable in the first release. Unknown new labels stay pending.
   **Tradeoff:** some exports may wait for adjudication instead of receiving a
   plausible but unsupported answer.
   **Question:** "Who can settle catch-all/taxonomy disputes, and which target
   fields must be complete before a CSV counts as finished?"
   **Decision impact:** determines escalation authority, draft/final export gates
   and whether Platform blocks completion; it creates no fixed hierarchy by fiat.

5. **What evidence and authority permit less review later?**
   **Finding:** today's validation was reused; conditional bounds fail under some
   large-family error patterns. All stored operational thresholds are null.
   **Proposal:** approve a frozen fresh-validation plan, named policy approvers
   and monitoring/reversion criteria; start qualification with Mnfr/age and keep
   automatic acceptance disabled until the agreed checks pass.
   **Tradeoff:** collecting independent labels costs time; it prevents historical
   good results from becoming an unsupported production guarantee.
   **Question:** "Can we allocate people to a fresh representative label sample,
   who signs off on activation, and what observed problems should send us back
   to full review?"
   **Decision impact:** establishes a release gate and owner; a meeting agreement
   alone cannot supply missing statistical evidence.

## What Max can implement now

The [existing handoff](model_selection_handoff_20261106.md) now specifies nullable
predictions, raw/calibrated confidence identity, mandatory/recommended/informational
codes, OR/precedence rules, model and policy versions, human explanations,
confirmation/correction records and draft/final CSV behavior. Implement that
disabled-policy workflow while business choices are reviewed. No competing
adapter or dashboard was implemented for this task.

The assessment ticket is ready to close after deliverable review. Ravi's policy
approval, fresh qualification, actual dashboard wiring and measurement of human
correction quality remain tracked follow-up responsibilities, not completed
production capabilities. No message was sent to Ravi and no ticket was changed.
