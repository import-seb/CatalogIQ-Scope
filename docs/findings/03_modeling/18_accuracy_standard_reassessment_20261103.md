# Reevaluate the 99% accepted-accuracy standard

November 3, 2026. The goal remains **at least 99% accuracy among automatically
accepted classifications**. No model, score variant, temperature, cutoff,
allocation, product group or previous result was changed. No category slices
were investigated. Automatic acceptance remains disabled.

## Decision in plain English

**Mnfr and TargetAgeGroup have good development evidence for the 99% goal under
the stated family-sampling assumptions.** Both clear all three numerical
checks, including the preferred conservative adjusted check. This updates the
earlier conclusion that no candidate cleared the custom safety calculation.

Segment, Brand MiniLM and provisional Sub-Segment do not have enough evidence
at their existing cutoffs. Brand lookup is a meaningful disagreement: the
bootstrap clears 99%, while both adjusted checks do not. We do not select the
bootstrap simply because that answer is more favorable.

The adjusted checks behaved more reliably than the plain percentile bootstrap
in the reference simulations. However, **all three failed when errors were
concentrated only in a small number of large families**. The passing results
are therefore conditional development evidence, not a guarantee against every
possible failure pattern or approval for production.

The practical recommendation is to use the conservative effective-sample beta
check for primary development reporting, the adjusted Wilson check as a
comparison, and the bootstrap as a diagnostic. Before operational approval,
obtain a fresh evaluation whose sampling design addresses family size and
previous development-data reuse. Do not resurrect the original maximum-size
bound, and do not replace it with an unqualified bootstrap pass.

## Fixed-policy comparison

All confidence bounds here are **one-sided 95% lower bounds, per model**.
The numerator/denominator is always product rows, not an unweighted average
of family accuracy. A lower bound of at least 99% clears the numerical check.

| Model | Accepted rows / families | Mistakes | Observed accuracy | A: adjusted Wilson | B: group bootstrap | Preferred beta check | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Segment MiniLM | 3,803 / 2,179 | 31 | 99.185% | 98.405% | 98.904% | 98.369% | Not supported at this cutoff |
| Mnfr MiniLM | 4,076 / 2,266 | 1 | 99.975% | 99.566% | 99.926% | 99.525% | All three clear 99%, conditionally |
| TargetAgeGroup MiniLM | 4,010 / 2,239 | 5 | 99.875% | 99.386% | 99.776% | 99.347% | All three clear 99%, conditionally |
| Brand MiniLM | 3,600 / 2,071 | 44 | 98.778% | 97.836% | 98.424% | 97.797% | Not supported |
| Brand lookup | 3,298 / 1,727 | 22 | 99.333% | 98.406% | 99.018% | 98.354% | Methods disagree; not supported by primary check |
| Sub-Segment MiniLM, provisional | 1,822 / 1,154 | 22 | 98.793% | 97.430% | 98.309% | 97.359% | Not supported; remains provisional |

The unchanged cutoffs are Segment 93% calibrated, Mnfr 50% calibrated,
TargetAgeGroup 50% calibrated, Brand MiniLM 97% raw, Brand lookup 95.5%
calibrated with seen-key gating, and Sub-Segment 97.7% calibrated.
No alternative cutoff was scored or selected in this reassessment.

The requested per-model intervals do not jointly guarantee six successful
claims. A separate Bonferroni sensitivity calculation at alpha=.05/6 gives
preferred-check lower bounds of **99.261% for Mnfr and 99.048% for age**; both
still clear 99% under the same assumptions. The other models do not. Thus the
change in conclusion is not solely a switch from simultaneous to per-model
confidence. Historical reuse prevents interpreting even this sensitivity
calculation as a new, prospective validation exercise.

## Why the old check returned 90.56% for Mnfr

The original calculation used the largest accepted family, 79 rows, as the
error-contribution limit for every one of 2,266 accepted families. Its bound
contained the multiplier `G*M/N`, approximately **43.92** for Mnfr. That
discarded information about the many much smaller families. Coupled with a
family-wide confidence correction, it yielded a very loose lower bound of
90.56% despite just one error in 4,076 rows.

That was a pessimistic bound, not an accuracy estimate. The new adjustment
uses **all actual family sizes**. For Mnfr, its size-based design effect is
5.837 and its adjusted effective sample size is 697.78. Those numbers describe
the amount of information under a stated sampling model; they are neither the
raw 4,076 rows nor simply the 2,266 families. The preferred lower bound is now
99.525%. None of these calculations changes the one observed mistake.

## Grouping, data scope and separation

Reused the frozen `rule-final-v4` group IDs and approved target-specific
allocations. No grouping was rebuilt. The previous deterministic SHA-256
group-role assignment is checked against the sealed saved assignments.
The audit populations remain 4,131 Segment rows, 4,076 Mnfr, 4,010 age,
4,345 Brand for each Brand model, and 3,957 Sub-Segment rows.

Only previously designated policy-audit rows enter the model confidence
calculations. The original validation predictions and contexts are validated
using the existing contract loader and independent approved contexts.
The same saved temperature is applied only to reconstruct the old acceptance
mask; there is no temperature estimation or recalibration fit. Saved
essential-information/lookup eligibility gates are reused. Accepted counts,
mistakes and accepted-family counts must exactly reconcile with the old audit.

Simulations read **no catalog data**, including no audit outcomes, category
labels or family sizes. Method definitions, sources, scenarios, seeds and the
preferred reporting method were saved before simulation execution. A sealed
method decision was required before the new audit calculation could start.
An independent high-repetition simulation confirmation was subsequently run
without changing that decision or any method.

The earlier audit results were already visible in the preceding investigation;
this cannot be described as blinded method selection. The procedural freeze
prevents further optimization against them, but it cannot remove historical
exposure. The original validation data also influenced model development.
Fresh, representative development evidence is needed before release. The
protected test was neither opened nor hashed; only existing identity-only
registries were used for exclusion checks.

## Statistical methods and assumptions

### A: unequal-family-size adjusted Wilson

Let m_g be the accepted row count in family g, N=sum(m_g), and p the observed
accepted accuracy. The working adjustment uses within-family correlation
rho=1, so records in the same family are allowed to be perfectly dependent.
This avoids estimating correlation from a handful of observed mistakes.

The design effect is `sum(m_g^2)/N`, and
`n_eff=N^2/sum(m_g^2)`. Large families therefore count more heavily in the
uncertainty calculation. This is not the substitution `n_eff=number_of_groups`
except when every accepted family has the same size.

Under independent families with a common expected error probability, this
uses the maximum within-family covariance to obtain a conservative working
variance. It does **not** bound variance for every possible systematic
association between family size and error risk. Wilson then inverts a score
test using the adjusted information, rather than treating all N rows as
independent. [Statistics Canada's survey Wilson extension](https://www150.statcan.gc.ca/n1/pub/12-001-x/2017002/article/54872/02-eng.htm)
explains the effective-sample approach.

The implementation also uses the published degrees-of-freedom adjustment
`n_eff_df=n_eff*(z(.95)/t(G-1,.95))^2`, with G the number of families containing
accepted rows. This is the one-sided analogue of the Dean–Pagano adjustment
discussed in [Franco et al.'s comparative study](https://www.math.umd.edu/~slud/s770/SurveyConfidenceIntervals/JSSAM-2017-065-FINAL.pdf).
It makes a negligible difference at these family counts but avoids assuming
infinite design degrees of freedom. No acceptance or one accepted family
cannot produce a positive analytic lower-confidence claim.

### B: whole-family percentile bootstrap

Resample the complete set of audit families with replacement, including
families containing zero accepted rows. For each resample, divide the total
correct accepted rows by the total accepted rows. Use the fifth percentile
as the one-sided 95% lower bound. This preserves dependence within families
and the row-weighted estimand.

The implementation compresses identical `(accepted rows, errors)` family
pairs and samples their multiplicities multinomially. That is exactly the
same resampling distribution as drawing individual family IDs; it is not
a row bootstrap, a Poisson approximation or equal weighting of family accuracy.

Main audit results use **49,999 resamples**, seed 20261104. Stability is checked
at 1,999, 9,999 and 49,999 draws with three seeds. At 49,999, the largest
across-seed change in a lower bound is about **0.004 percentage points**.
For lookup the range is 99.0177%–99.0217%; its bootstrap pass is numerically
stable but methodologically disputed. More repetitions reduce numerical
noise; they cannot make a miscalibrated confidence procedure reliable.

With zero observed errors, a nonparametric bootstrap cannot manufacture a
previously unseen failure and returns a lower bound of 100%. With Mnfr's one
error, about **36.75% of audit bootstrap resamples contain no errors**.
This is a warning about rare-event inference, not a reason to remove that
model or alter its cutoff.

### Established alternative: effective-sample beta bound

For extreme proportions, published clustered-survey comparisons motivate
checking a more conservative beta/Clopper–Pearson-type interval. We calculate
`BetaQuantile(.05, p*n_eff_df, (1-p)*n_eff_df+1)` for the lower accuracy bound.
Fractional effective counts are intentional. The underlying approach is
[Korn and Graubard's effective-sample interval](https://www150.statcan.gc.ca/n1/pub/12-001-x/1998002/article/4356-eng.pdf),
with the same explicit size-only working design effect as Method A.

This is described as **Korn–Graubard-type**, not an exact clustered binomial
interval or the full NHANES procedure. It does not use NHANES sampling weights,
and does not estimate an outcome-based survey design effect. The conservative
rho=1 design effect is a working-model choice that the simulations examine.
In particular, zero errors still leaves room for future errors; the formula
does not return a perfect-accuracy lower bound.

[NCHS's standards](https://www.cdc.gov/nchs/media/pdfs/2024/09/sr02_175.pdf)
describe using effective sample size and design information for proportions
from complex samples, and identify zero-event and uncertain-variance cases as
requiring care. CatalogIQ is not a probability-sampled health survey, so those
standards do not certify our data. We use the established interval family and
state its working assumptions explicitly.

The beta check was selected as the primary **reporting** method before running
the simulations or recalculating audit intervals. It is not a newly invented
safety inequality, a maximum/minimum of favorable bounds or a method selected
to make particular classifiers pass. Wilson remains a required comparison;
bootstrap remains a diagnostic. Neither is substituted when beta fails.

## Simulation evidence, beyond passing unit tests

Ran **26 scenarios × 2,000 independent synthetic catalogs = 52,000 trials**.
Each catalog has 2,000 independent families. Sizes include singletons, equal
two-row families, an unequal 1–30-row mixture, and a long-tailed 1–80-row
mixture. Depending on the mixture, each catalog has 2,000–4,150 rows.

Error processes include independent row mistakes; beta-binomial family risk
with within-family correlation .2; perfectly correlated all-or-none family
mistakes; and mistakes confined to the 20 largest families. True accuracies
include 98.9%, 98.99%, exactly 99%, 99.01%, 99.1%, 99.5% and 99.975%.
All are known from the simulation-generating probabilities, not estimated from
the simulated sample. No real audit errors determine a scenario parameter.

The following table asks how often a method incorrectly clears 99% when the
true accuracy is **98.99%**. A nominal one-sided 95% test should keep this
probability around or below 5%, subject to approximation/discreteness.

| Synthetic sampling situation | Adjusted Wilson false pass | Bootstrap false pass | Preferred beta false pass |
|---|---:|---:|---:|
| Independent single products | 3.60% | 8.05% | 3.60% |
| Two-row families, shared mistakes | 3.50% | 7.95% | 3.50% |
| Unequal families, moderate dependence | 0.15% | 7.50% | 0.05% |
| Long-tailed sizes, shared family mistakes | 0.00% | 23.90% | 0.00% |
| Mistakes only in a few large families | 29.70% | 29.70% | 29.70% |

Full results include exact binomial Monte Carlo intervals for these trial
rates. For the first row, the bootstrap false-pass interval is approximately
6.89%–9.33%; the beta interval is 2.83%–4.51%. For the final row, the common
false-pass interval is approximately 27.70%–31.76%. These are uncertainty
intervals for independent simulations, not product-row binomial intervals.
Undercoverage flags are diagnostic, not multiplicity-adjusted tests over the
scenario grid, and absence of a flag does not prove validity.

In the **22 size-independent-risk reference scenarios**, neither adjusted
method demonstrated coverage failure significantly above 5% at the recorded
Monte Carlo interval level. The bootstrap had clear undercoverage in 20 of
those reference scenarios. This does not mean adjusted intervals are tight:
the rho=1 assumption makes them deliberately conservative when dependence
is smaller. For a true 99.5% classifier with unequal moderately dependent
families, beta cleared 99% in only 43.3% of trials. A non-pass can mean limited
information rather than an inadequate classifier.

### Rare mistakes and zero-error samples

At true 99.975% accuracy with 4,000 independent row outcomes arranged in
two-row families, there is roughly one expected error. The bootstrap lower
bound incorrectly exceeds the **true accuracy** in 36.65% of trials, because
that many samples contain no observed errors and the bootstrap returns 100%.
This is **coverage failure**, not a false 99% approval: the true accuracy in
this particular example already exceeds 99%. Both adjusted bounds retain
room for unobserved mistakes.

Base simulations used 1,999 bootstrap draws per trial. To check that finite
bootstrap repetitions did not drive the conclusions, ran another **4,000
independent synthetic catalogs**, across four fixed confirmation scenarios,
with **19,999 resamples each** and different seeds. Bootstrap false passes at
true 98.99% were 9.9% for singletons, 7.5% for unequal moderately dependent
families, and 24.8% for the long-tail shared-error case. Their Monte Carlo
intervals overlap the base findings. Near-perfect zero-error coverage failure
was 37.6%, consistent with the earlier 36.65%. The method decision was unchanged.

### Why no method receives a universal safety certificate

In the large-family-only stress scenario, 20 families contain 700 of the
4,150 rows. Give those families all-or-none mistakes with probability chosen
so expected row accuracy is 98.99%; other families have no mistakes. The
expected risk now depends on family size. About 30% of samples happen to see
no failing family. All three methods can then clear 99% even though the
generating accuracy is below the goal.

This is an explicit counterexample to unconditional certification by any of
these methods. The adjusted Wilson and beta working model assumes no such
systematic size/risk association; the small-tail bootstrap cannot recover
unseen failures either. The beta alternative improves rare-event handling
under the reference assumptions but does not solve informative family size.
We have not established that this stress process describes CatalogIQ. We also
cannot dismiss it merely because errors are rare. No category or size slice
was investigated to tune away this result.

## Limits on what the real-model passes establish

Mnfr and age have the strongest evidence under the tested reference model.
Their accepted mistakes occur in one and five families respectively. This is
still limited evidence about how future failures might cluster. Their numerical
passes must retain that qualification rather than becoming unconditional
claims of at least 99% on future products.

Segment is above 99% observed accuracy, but its lower bounds remain below the
goal. Brand MiniLM and Sub-Segment are already below 99% in the observed audit.
Lookup's observed 99.333% is promising, but the method disagreement is material:
bootstrap 99.018% versus adjusted Wilson 98.406% and beta 98.354%. Keep it
unconfirmed at the current policy rather than declaring success from bootstrap.

The actual incoming population may differ from cleaned, labeled development
records. Historical model selection and repeated inspection of validation
remain unresolved. Frozen groups can miss relationships between products,
so independence across families is an assumption, not a proven property.
There is no new production-readiness claim and no enablement change.

## Deliverables and reproducibility

Reusable code is in `catalogiq.clustered_accuracy`,
`catalogiq.accuracy_simulations`, `catalogiq.accuracy_reassessment`, and
`catalogiq.accuracy_simulation_stability`. Command handling is in `catalogiq.cli`.
No new dependencies were required.

```powershell
$env:PYTHONPATH = 'src'
# Use fresh output directories when repeating.
.venv-gpu/Scripts/python.exe scripts/reevaluate_accuracy_standard.py simulate `
  --output-root data/processed/accuracy_method_validation_repetition
.venv-gpu/Scripts/python.exe scripts/reevaluate_accuracy_standard.py audit `
  --method-validation data/processed/accuracy_method_validation_repetition `
  --output-root data/processed/accuracy_reassessment_repetition
# The confirmation script also refuses an existing output directory.
.venv-gpu/Scripts/python.exe scripts/check_accuracy_bootstrap_stability.py
```

The completed run directories are:

- `data/processed/accuracy_method_validation_v1_20261103/`: frozen protocol,
  all 52,000 simulation trials, comparison table, decision and checksum seal.
- `data/processed/accuracy_reassessment_v1_20261103/`: frozen audit protocol,
  six-model comparison, bootstrap convergence, family-count arithmetic,
  source/code hashes, environment and checksum seal.
- `data/processed/accuracy_bootstrap_stability_v1_20261103/`: independent
  high-repetition confirmation, protocol and checksum seal.

Python 3.13.7; NumPy 2.3.3; pandas 2.3.2; SciPy 1.16.2. Seeds and exact
repetition counts are in each protocol. Output directories are never
overwritten. The new audit run reverified **144 consumed evidence files**,
including the old result package and all 53 original consumed sources,
unchanged. The new command adapter extends `cli.py`; the old mathematical
calibration/evaluation code remains unchanged. Old result bytes and their
historical code fingerprints are retained.

Compact aggregate tables and protocols are checked in beside this report in
`accuracy_standard_reassessment_20261103/`. They contain no raw product text,
record IDs or family IDs. Large local outputs remain gitignored.

**88 relevant automated tests passed**, including 17 new tests for effective
sample calculations, score inversion, beta reference values, whole-family
bootstrap behavior, zero errors, unequal sizes, simulation publication,
protected-test guards, and refusal to evaluate audit before a sealed method
decision. Existing calibration/contract/export/lookup regressions also passed.

```powershell
$env:PYTHONPATH = 'src;tests'
.venv-gpu/Scripts/python.exe -m unittest test_clustered_accuracy `
  test_confidence_calibration test_prediction_contract `
  test_minilm_prediction_export test_brand_probabilistic_lookup -q
git diff --check
```

Tests establish implementation behavior; the operating-characteristic
simulations provide the separate evidence about statistical suitability.
No classifier was trained or run, no protected-test data was evaluated, and
Maria's harness and Max's artifacts were untouched.

## What I recommend we do next.

Use the adjusted beta check for cautious development reporting, keep Wilson
and bootstrap comparisons visible, and label Mnfr/age as conditional 99%
candidates. Treat lookup as unresolved and the remaining models as needing
improvement or more evidence at their current cutoffs.

Yes, category investigation can be the next development task. It can help
explain mistakes and uneven performance; it does not retroactively validate
changes against this same audit. This task has not performed that analysis.
Freeze any resulting revised policies before collecting fresh representative
development evaluation labels, with deliberate attention to independent
families and a defensible sampling design. Keep the protected test closed and
automatic classification disabled until that separate qualification is complete.
