# Saved relationship results verification — November 18, 2026

Card #1647. Verified implementation: `216c923`, plus the deployment allowlist
fix accompanying this record. Source CSVs were received privately from the run
owner; neither they nor the browser's test review exports belong in Git or the
public demo.

## Inputs and counts

| Input | SHA-256 |
| --- | --- |
| target_predictions.csv | `6bc3dd3dded45438a28e091b4663aec80a32f3f9db195bef439a613dfc81bca5` |
| relationship_conflicts.csv | `249033969311e7b16199f226c70c100725d09680496107f8ecdaf858f87abe61` |

The matching report comes from `evaluation/classify-target`, commit `0f962ee`,
`docs/findings/03_modeling/target_classification_20261115/report.json`.
Its contents are unchanged at `4289b44`.

- 27,870 prediction rows with 27,870 unique `JoiningKey` values.
- 651 warning rows across **650 products**; one product has two warnings.
- 638 Sub-Segment/Segment warnings and 13 Brand/Mnfr warnings.
- All 651 rows have `invalid_approved_classification_relationship`; none in this
  run has `missing_relationship_classification`. The missing-classification path
  remains covered by the separate synthetic regression cases.
- All relations remain `candidate`, and all `is_approved_decision` values are false.
- Five targets have suggestions for every row; Platform is unavailable for every
  row. One eligible product has no name and is accepted by the importer.
- The matching report declares 28,082 input rows and 212 quarantined rows.
  Cleaner eligibility was not rerun, and its declared decision hash was not
  recomputed because `row_decisions.csv` was not supplied.

## Verification performed

1. Ran `parseSavedResults` from the production interface on the complete files:
   exact ID matching, model/registry versions, vocabulary, confidence, status
   counts and warning completeness all passed. No unknown, duplicate or missing
   warning keys were found.
2. Independently rebuilt all relationship observations with
   `catalogiq.relationship_conflicts.flag_all` and the checked-in registry.
   The complete warning key/label sets match the supplied 651 rows.
3. Exercised approval and correction for every one of the 650 affected products
   through the domain functions. Original predictions, candidate warnings and
   two-event review histories survived JSON export; CSV export retained warnings.
   These were isolated verification operations, not accepted catalog decisions.
4. Imported the real files in the built static UI using the in-app Chromium
   browser. Native Chrome's file picker could not be controlled reliably, so the
   interactive import check used this browser instead. The UI displayed the
   complete 27,870 / 651 / 650 counts and the report's 212 quarantined products.
5. Added a batch of 248 real products alongside the two synthetic demo examples
   (250 total), including 249 warning rows. Approved one flagged product, reopened
   it, saved a correction with a test note and reloaded the page. Both original
   suggestions and the warnings remained visible; the correction and history
   persisted. The original score was explicitly identified as the original
   prediction's score. No candidate rule blocked either action.
6. Downloaded both CSV and JSON through the UI and inspected the resulting files:
   250 records, 249 warning rows, distinct original/final labels, original input
   file hashes and two review history events were preserved. No browser console
   warnings or errors were captured during this flow.

The earlier 57 synthetic checks (20 JS, 12 Python, 15 local API browser and
10 static browser scenarios) cover stale/truncated imports, the missing-label
warning, storage errors, batch selection and mobile layout. The real-file checks
above supplement those tests; the real dataset is not committed as a fixture.
The 20 JavaScript and 12 Python checks were rerun on November 18 and passed.

## Limits and delivery

This verifies saved-result review. The public static interface does not perform
new model inference, activate candidate rules or approve thresholds. Live
inference still requires Sebastian's pinned serving bundle and a separate run.
Workspaces remain browser-local with a 250-product capacity and self-entered
reviewer attribution. Public deployment includes the interface, synthetic
examples and candidate registry, never the private CSVs.
