# Protected-test assessment: single authorized execution

November 6, 2026. The five frozen models were assessed once against the original
protected population of **12,160 records in 11,979 groups**. This records model
performance against supplied labels; automatic acceptance remains disabled.
Sub-Segment remains provisional and Platform remains deferred.

## Execution and integrity

The user's instruction was: "Run the authorized test once using the frozen
protocol. Record results, checksums, environment details, and limitations."
Execution started at `2026-11-06T19:32:41.910307+00:00` and completed at
`2026-11-06T20:13:55.641011+00:00` (2473.73 seconds between receipt timestamps). There was one
assessment invocation, one completed attempt receipt and no assessment retry.
Preparation-only checksum checks did not expose the protected data.

The [original protocol](protected_test_assessment_20261106/protocol.json) was
unchanged: SHA-256 `f51fcbaa6db6aef775b7b3a5b48fb94bbf3fc0008a504c2938ecfe0daaa9ed2e`. All **75 source/artifact pins** matched before
execution, after execution, and during saved-output verification. The assessment
verified protected-test bytes before and after the run against
`0664b82be6b23b0752ab310b01394868be4de6c73cc1d76559efd10e1fa824c4`.

The evaluation branch began at `cebfe9d5df392a6b49487909e4bd9179634adcb4`,
before the freeze commit. The required frozen files were restored from
`988a6b752df4226cddf538b9a98c9b0b369b78b7` before test access. JSON line endings
were restored to their original LF bytes to match the existing protocol checksum.
This was recovery of the frozen implementation, with no model, calibration,
threshold or scoring changes. Executed working-tree file hashes identify the
code more precisely than the evaluation branch's base commit.

The remaining reporting helpers, shared calibration resource consumers, tests
and documentation from the same freeze commit were recovered after execution.
Existing experiment function bodies were verified unchanged; all 75 frozen
execution pins remained unchanged. This also preserves the saved-report CLI and
removes its superseded duplicate fixture.

## Scores

Accuracy is the fraction of correctly labeled products. Macro-F1 gives every
category equal weight, so small categories matter as much as large categories.
The primary score uses the fixed model vocabulary union observed test truth,
including unsupported truth classes and zero-support model classes. No row was
removed for low confidence, class frequency, input overlap or an incorrect label.
Only missing truth is excluded from that target's score.

Values and 95% intervals are percentages. Intervals use the prespecified 2,000
whole-group bootstrap resamples, seed 20261106, fixed class union and percentile
limits. They describe sampling uncertainty conditional on these frozen groups,
labels and fitted models; they are not production guarantees.

| Target | Status | Labeled records | Accuracy (95% interval) | Primary macro-F1 (95% interval) | Unsupported truth rows | Zero-support / union classes |
|---|---|---:|---|---|---:|---:|
| Segment | selected | 11,477 | 96.70% (96.39%–97.01%) | 94.07% (93.31%–94.77%) | 0 | 0 / 8 |
| Mnfr | selected | 11,228 | 99.95% (99.90%–99.98%) | 96.99% (94.42%–99.09%) | 0 | 0 / 2 |
| TargetAgeGroup | selected | 11,033 | 99.78% (99.68%–99.86%) | 97.93% (96.77%–98.86%) | 0 | 0 / 3 |
| Brand | selected | 12,112 | 93.82% (93.36%–94.23%) | 88.83% (83.73%–88.46%) | 2 | 10 / 144 |
| Sub-Segment | provisional | 10,982 | 88.69% (88.13%–89.27%) | 65.75% (64.04%–67.02%) | 0 | 3 / 49 |

**Brand interval limitation:** its observed macro-F1 (88.83%) is above the frozen
percentile interval's upper limit (88.46%). Brand has 123 classes with 1–49 truth
rows and ten zero-support classes in the 144-class union. In a fixed-union
bootstrap draw, omitted rare classes receive zero F1, which can pull resampled
macro-F1 below the observed point. The original interval is retained verbatim;
it is a descriptive result with a sparse-class limitation, not a calibrated
coverage guarantee. No new interval method was selected after seeing this result.

Every available model produced suggestions for all 12,160 records. Missing-truth
counts are recorded in the [verification record](protected_test_assessment_20261106/assessment_verification.json).
All 12,160 suggestions per model remain review-required. Platform has zero model
prediction coverage and null quality metrics; it was not assigned a forced label.

Mnfr retains its current binary vocabulary, **J&J versus All others**. Its score
does not measure identification of every manufacturer. Brand is the frozen
feature-key lookup, so familiar-key performance must be distinguished from
unseen-key behavior; the familiarity audit below records that distinction.

Sub-Segment's 88.69% row accuracy accompanies 65.75% macro-F1: performance is
uneven across categories. Eighteen classes have only 1-49 labeled examples and
three have zero test support. The model's provisional status is unchanged.

Exact metrics, weighted/macro precision and recall, runtime and coverage are in
[results_summary.json](protected_test_assessment_20261106/results_summary.json).
Unrounded intervals are in [metric_intervals.csv](protected_test_assessment_20261106/metric_intervals.csv).
Per-class tables, confusion matrices and raw probabilities remain in the private
sealed output described below.

## Confidence and diagnostic queues

Both raw probabilities and the existing saved temperatures were assessed without
fitting or selecting a new calibration variant. The
[calibration table](protected_test_assessment_20261106/calibration_comparison.csv)
records ten-bin ECE, multiclass-sum Brier and NLL. Infinite NLL is represented by
a null value plus `nll_is_infinite=true`; a clipped sensitivity statistic is
separately labeled and does not turn an infinite result into a finite guarantee.

Each pair below is **raw / saved temperature**, not a newly selected variant.
Lower values indicate better calibration under that metric, conditional on the
supplied test labels. Temperature scaling preserves the class ranking.

| Target | ECE, ten bins | Multiclass Brier sum | Unclipped NLL |
|---|---:|---:|---:|
| Segment | 0.015144 / 0.007462 | 0.053203 / 0.052066 | 0.111843 / 0.106726 |
| Mnfr | 0.000414 / 0.000331 | 0.000792 / 0.000776 | 0.001953 / 0.001787 |
| TargetAgeGroup | 0.001560 / 0.001884 | 0.003354 / 0.003165 | 0.008764 / 0.008796 |
| Brand | 0.084806 / 0.017716 | 0.122258 / 0.109449 | infinite / infinite |
| Sub-Segment | 0.024301 / 0.024683 | 0.176476 / 0.176536 | 0.385726 / 0.385777 |

Saved temperatures reduced ECE for Segment, Mnfr and Brand; ECE increased for
TargetAgeGroup and Sub-Segment. Both variants are retained without reselection.
Brand's NLL is infinite in both variants because its two unsupported truth rows
receive zero probability for their true labels. They remain scored errors.

The table below reports **only the candidates already fixed in development**.
It does not select a winner or operational threshold from the test. "Candidate
rows" means records that would clear the confidence and essential-information
gate, with familiar keys also required for lookup. Its denominator is the target's
labeled test cohort, not all products or a live retailer batch. Actual automated
acceptance is zero and actual required review remains 100%.

| Target | Fixed cutoff | Probability variant | Candidate rows | Errors among candidates | Cohort coverage | Candidate accuracy |
|---|---:|---|---:|---:|---:|---:|
| Segment | 0.930 | saved_temperature | 10,609 | 122 | 92.44% | 98.85% |
| Mnfr | 0.500 | saved_temperature | 11,228 | 6 | 100.00% | 99.95% |
| TargetAgeGroup | 0.500 | saved_temperature | 11,032 | 23 | 99.99% | 99.79% |
| Brand | 0.955 | saved_temperature | 8,383 | 113 | 69.21% | 98.65% |
| Sub-Segment | 0.977 | saved_temperature | 5,237 | 97 | 47.69% | 98.15% |

The [complete diagnostic table](protected_test_assessment_20261106/threshold_diagnostics.csv)
also includes the frozen .80/.90/.95/.99 cutoffs for both variants. It is
descriptive evidence, with no post-test cutoff tuning, confidence certification,
new 99% release rule or automatic-approval policy.

## Input-overlap audit

The assessment compared the selected neural models' exact effective five-field,
256-token inputs with original train/validation inputs. All protected rows and
groups were retained regardless of the audit result.

| Target | Audit result |
|---|---|
| Segment | 0 / 12,160 exact effective-input matches |
| Mnfr | 0 / 12,160 exact effective-input matches |
| TargetAgeGroup | 0 / 12,160 exact effective-input matches |
| Brand | 9,099 / 12,160 familiar lookup keys (expected lookup behavior) |
| Sub-Segment | 0 / 12,160 exact effective-input matches |

Frozen V4 grouping used four fields and 128 tokens. Exact-input crossings limit
independence claims; no regrouping or selective rescoring follows this test.
Even zero exact matches would not establish absence of semantic family overlap.
Familiar Brand lookup keys are expected and are not the same audit as neural
input identity.

## Environment and time

Inference used CPU FP32, two numerical/PyTorch intra-op threads, batch 64 and
seed 42, with deterministic PyTorch algorithms requested and local/offline
model loading. Python was 3.13.7; the exact frozen library
versions, operating system, CPU details, installed package versions and Git
provenance are in [execution_environment.json](protected_test_assessment_20261106/execution_environment.json).
The CUDA-enabled PyTorch package was used for CPU inference, as frozen.

| Target | Model-load + overlap-audit + prediction seconds |
|---|---:|
| Segment | 539.97 |
| Mnfr | 565.62 |
| TargetAgeGroup | 655.13 |
| Brand | 0.27 |
| Sub-Segment | 640.69 |

These per-target times include audit/model-loading overhead and exclude scoring
and bootstrap work. Receipt duration includes test access, inference and scoring,
but excludes command startup, final checksum-seal creation and shutdown. Neither is
a controlled request-latency or GPU benchmark, and neither estimates reviewer
time or an application's complete end-to-end processing cost.

## Saved artifacts and checks

The private run directory is
`data/processed/protected_test_assessment_v1_20261106/`. It contains per-target
predictions, manifests, per-class scores, confusion matrices, uncertainty,
calibration and diagnostic tables, plus a summary, authorization receipt and
completion seal. Product-level truth and probabilities stay out of Git.

All **42 sealed output files** were checked against
their hashes. Saved-output checks verified unique protected membership/groups,
12,160 predictions per model, valid probability vectors and numeric class order,
all review flags, labeled/missing/OOV counts, accuracy reconciliation, confusion
and support totals, and bootstrap point estimates. This recording reads saved
predictions; it does not invoke inference or reopen `test.csv`.

[assessment_artifact_checksums.json](protected_test_assessment_20261106/assessment_artifact_checksums.json)
records the completion-seal hash and every sealed output hash.
[assessment_verification.json](protected_test_assessment_20261106/assessment_verification.json)
contains the completed receipt, reconciliation and limitations. A separate
[evidence manifest](protected_test_assessment_20261106/assessment_evidence_manifest.json)
fingerprints the compact published result files and these findings, using raw
file hashes and LF bytes preserved by Git attributes.

The 21 assessment/bootstrap unit checks passed before execution. After recovery
of the remaining supporting freeze files, the full suite passed **681 tests**
in 90.95 seconds. The saved-report CLI help/import check also passed. All checks
use development or synthetic fixtures; the protected inference was not repeated.

## Limits and next use

- Supplied labels have not been certified as independent ground truth. Agreement
  with those labels is not proven catalog correctness.
- Rare and zero-support classes, grouping misses, retailer/taxonomy ambiguity,
  and historical development reuse remain. The class union and audit above
  make those limits visible without changing the frozen population.
- Sub-Segment remains provisional. Platform quality is unmeasured. No approved
  complete relationship registry exists, so valid-pair rates remain null and
  relationship checks are not marked as passing.
- Human-corrected export accuracy, reviewer effort, workflow speed and live
  retailer quality require separate evidence. This assessment does not measure
  any of them or approve publication into production master data or CRM.
- The protected set is now exposed. It must not become a new tuning/selection
  set followed by a second claim of independent final evaluation. Any future
  model, calibration, threshold or taxonomy changes require a new evaluation
  plan and fresh independent assessment evidence.

The frozen preparation history is in the
[protocol record](21_protected_test_assessment_protocol_20261106.md). No model,
threshold or release policy was changed in response to these results.
