# Frozen model evaluation protocol and protected-test preparation

**Execution update:** the separately authorized assessment completed once on
November 6. See the [results and limitations](22_protected_test_assessment_20261106.md).
This document preserves the earlier preparation record; the frozen JSON is unchanged.

November 6, 2026. **Freeze the selected artifacts and assessment method; keep
the protected test closed until a separately authorized assessment.** This is a
technical model assessment for a reviewer-assisted product, not approval for
automatic publication or a newly invented 99% release criterion.

The machine-readable freeze is
[`protected_test_assessment_20261106/protocol.json`](protected_test_assessment_20261106/protocol.json).
Its exact file SHA-256 is recorded below. Any change
to the selected artifacts, code, vocabulary, features or scoring procedure
requires a new documented protocol before test access. Never edit this freeze
in response to protected-test outcomes.

## Fixed decisions

| Item | Frozen choice |
|---|---|
| Model routing | Segment: five-field MiniLM; Mnfr: selected unweighted MiniLM; TargetAgeGroup: MiniLM; Brand: probabilistic lookup v1; Sub-Segment: provisional MiniLM. No challenger/model-family selection on the test. |
| Platform | Deferred/unavailable. Zero model prediction coverage; quality metrics null. No forced label or new Platform experiment. |
| Test membership | Original `segment-final-20261009` V4 protected population: 12,160 records in 11,979 frozen product groups. Brand/Sub-Segment retain this same protected population despite their different development allocations. |
| Truth eligibility | Score each target's nonmissing supplied labels. Blank/null/none/nan follow the existing target missingness convention. Missing labels are reported, never filled or used to drop the record from other targets. |
| Unsupported truth labels | Retain as errors and include in the primary macro denominator. Emit only the saved model vocabulary; never expand/retrain a head from test truth. |
| Primary score | Macro-F1 over full model vocabulary union test truth, `zero_division=0`; zero-support model classes remain included. |
| Other model metrics | Accuracy; macro/weighted precision, recall and F1; per-class precision/recall/F1/support; confusion matrices. Preserve row/target counts and prediction coverage. |
| Uncertainty | 2,000 whole-frozen-group percentile bootstrap resamples, seed 20261106, two-sided 95% intervals for accuracy and union macro-F1. Fixed union class set within every resample. Descriptive per-target intervals, not simultaneous production guarantees. |
| Confidence | Report raw and existing saved-temperature ECE, Brier and NLL. No calibration fitting or new score-variant selection. Infinite NLL for OOV/zero true probability is explicitly flagged, with the existing clipped sensitivity statistic separately labeled. |
| Review cutoffs | Prespecified .80/.90/.95/.99 plus each target's already selected development candidate. Diagnostic workload/error tables only; all operational policies remain disabled. No post-test threshold tuning. |
| Inference | CPU, FP32, numerical/PyTorch intra-op threads 2, batch 64, seed 42; pinned Python/library versions. Saved MiniLM formatter, tokenizer, head and weights; five-field/256-token/category-first view. Lookup uses its original normalization/counts/prior. |
| Relationship validity | Null/unmeasured: no complete approved machine-readable registry is available. Never infer a whitelist from test co-occurrences or mark missing checks as passing. |
| Human-corrected quality | Unmeasured by this assessment. Reviewer confirmations, final-export accuracy and reviewer time need a separate application study. |

The [model-selection record](20_model_selection_20261106.md) and
[integration handoff](../../integration/model_selection_handoff_20261106.md)
identify the original artifacts and their rationale. The JSON freeze copies the
exact selected model identities, label mappings, checkpoint/count JSON hashes,
tokenizer/encoder/config hashes, development-manifest/allocation evidence,
saved temperatures and research-policy identities. Files stay in their existing
private locations; no model or raw data is copied into Git.

## What preparation reads, and what it does not

`freeze` and `preflight` inspect existing model bytes without deserializing them,
development metadata/data as needed for source verification, and protected
**identity-only** registries. The expected `test.csv` byte hash is copied from
the existing split seal; neither preparation phase opens or hashes `test.csv`.
Original splits, model files and prediction packages remain unchanged.

The protocol pins the actual working-tree source bytes (CRLF-to-LF explicitly
normalized for code), rather than claiming that an older Git HEAD contains the
uncommitted assessment implementation. Metadata/data/model hashes use raw bytes.
Once frozen, changing a pinned executable source causes preflight to fail.
The JSON freeze itself uses LF and an external expected file checksum supplied
to the assessment command.

The complete assessment boundary lives in
`src/catalogiq/protected_test_assessment.py`; the thin command is
`scripts/assess_protected_test.py`, with arguments in the existing CLI module.
Existing development/validation loaders still reject final-test use. The new
boundary reuses the saved classifier/formatter and existing scoring/calibration
functions; it does not fit a competing inference model or evaluation framework.

## Required input-representation check during authorized execution

Frozen V4 grouping used four fields and 128 tokens; selected neural models use
five fields and 256 tokens. Earlier development equality audits do not alone
certify final-test independence. Before reporting test scores, the assessment
checks exact effective 256-token input signatures against the original train
and validation inputs, using the pinned tokenizer and formatter.

Any crossing is reported, with every original test row retained. It limits the
independence interpretation and is not repaired by regrouping, reallocating,
dropping rows or choosing another model. Zero exact crossings would still not
prove absence of semantic family leakage. Brand's expected seen-key lookup
behavior is reported separately and is not treated as neural-input leakage.

## Commands and authorization boundary

Preparation is authorized by this task; protected-test scoring is not performed.
Use the existing environment with `PYTHONPATH=src`. The first command has already
been run for the versioned JSON file; it refuses to overwrite an existing freeze:

```powershell
$env:PYTHONPATH = 'src'
.venv-gpu/Scripts/python.exe scripts/assess_protected_test.py freeze --protocol docs/findings/03_modeling/protected_test_assessment_20261106/protocol.json
.venv-gpu/Scripts/python.exe scripts/assess_protected_test.py preflight --protocol docs/findings/03_modeling/protected_test_assessment_20261106/protocol.json --protocol-sha256 f51fcbaa6db6aef775b7b3a5b48fb94bbf3fc0008a504c2938ecfe0daaa9ed2e --output-root data/processed/protected_test_preparation_v1_20261106
```

Choose a new preflight output directory for a repeat. The readiness report is
`summary.json`, explicitly marked prepared/not executed. Missing artifacts or
different runtime versions are preparation failures, not poor model performance.
Resolve them before test access; do not silently change the frozen contract.

Only after an explicit instruction authorizes opening/scoring the protected test,
the operator may run the following **separate** command. The authorization
reference records that actual instruction; the command-line flag itself is not
a substitute for approval:

```powershell
.venv-gpu/Scripts/python.exe scripts/assess_protected_test.py assess --protocol docs/findings/03_modeling/protected_test_assessment_20261106/protocol.json --protocol-sha256 f51fcbaa6db6aef775b7b3a5b48fb94bbf3fc0008a504c2938ecfe0daaa9ed2e --execute --authorization-reference ACTUAL_AUTHORIZATION_REFERENCE --output-root data/processed/protected_test_assessment_v1_20261106
```

The command validates pins and writes an attempt receipt **before** first test
access. It requires a fresh, private `data/processed/` child directory. Failed
attempts retain their receipts and partial outputs; never delete an exposed
attempt to present a later result as the first. A technical retry must use the
same frozen pins, preserve original results and disclose earlier access.

## Prepared outputs and reporting limits

Authorized execution would produce per-target predictions and manifests,
per-class tables, confusion matrices, bootstrap intervals, confidence and
diagnostic queue tables, input-equality checks, a reconciled summary and a
checksum seal. All 12,160 records receive each available model's suggestion,
including rows without evaluation truth; missing truth only changes that
target's metric denominator. Every suggestion remains review-required.

Protected truth, record-level predictions and product inputs stay in private
outputs outside Git. Summaries must distinguish raw model evaluation from
reviewed-label accuracy, target-batch inference and production performance.
Do not compare a raw protected-test score directly with a differently filtered
validation audit score as if the populations matched.

Remaining limitations are frozen and disclosed: supplied labels are not yet
certified independent ground truth; rare classes may be unsupported; grouping
misses and retailer/taxonomy ambiguity remain; Sub-Segment is provisional;
Platform and relationship-validity rates remain unassessed. These are not
permission to adapt models or label rules after opening the protected labels.

The capstone brief's final held-out metrics remain distinct from its classified
unlabelled-target deliverable and assisted-workflow study. Preparing this
assessment does not satisfy all final demonstration requirements or approve
automatic publication to production master-data/CRM systems.

## Preparation verification

Synthetic tests exercise the no-test-access preparation boundary, explicit
execution authorization, immutable output/attempt behavior, out-of-vocabulary
truth, missing labels, numeric class mapping, and sealed result publication.
An eight-record-per-target validation-only smoke check reproduced all five
models' saved labels; maximum probability differences stayed below 1e-5.
Those were previously evaluated validation records, not protected test records.

The existing group-bootstrap implementation now uses per-group support,
predicted support and true positives instead of allocating a group-by-class-by-
class cube (roughly 2 GiB for Brand). Its draw sequence, fixed denominator and
metrics are unchanged; a regression test reproduces the original dense
implementation's intervals exactly. No model/training algorithm was changed.

The frozen protocol SHA-256 is
`f51fcbaa6db6aef775b7b3a5b48fb94bbf3fc0008a504c2938ecfe0daaa9ed2e`.
Preparation verified **75 artifact/code pins**, **12,160 protected identities**
and **11,979 frozen groups**, with status `prepared_not_executed`. Both phases
ran under file-open guards that reject `test.csv`; there were **zero attempted
opens**. No protected-test features, labels, predictions or scores were read or
generated. The preflight report is in the ignored private directory
`data/processed/protected_test_preparation_v1_20261106/summary.json`.

The full unit suite passed **681 tests**, including eight new assessment tests.
The package wheel built successfully without adding dependencies; its contents
include the assessment module and shared calibration resource. Existing pandas
future warnings do not affect the passing result.

Compact reproducibility evidence, including the validation-only smoke deltas,
runtime, readiness-report hash and build hash, is in
[`preparation_verification.json`](protected_test_assessment_20261106/preparation_verification.json).
The separately authorized execution is now recorded in the
[assessment findings](22_protected_test_assessment_20261106.md). Relationship
rates remain unmeasured without an approved map; the completed test-input audit
is reported with those findings.
