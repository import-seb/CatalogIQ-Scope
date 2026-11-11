# Adapter validation — November 11, 2026

Implemented on `feature/int-adpt`, based on main `2df400e` (includes Maria's
merged PR #34). Existing frozen models, calibration and review policy are unchanged.

## Completed checks

- Full local discovery: **717 tests, zero errors/failures, one skip**. The skip
  is the explicitly opt-in private-artifact inference test. This includes all
  702 existing tests and 14 new offline adapter tests.
- Private-artifact integration: **one test passed separately**, covering all
  five real routes, single/batch/repeated inference, ID/probability alignment,
  empty approved relationships and unchanged serving-member hashes.
- Package wheel built; verified it includes all three inference modules, frozen
  route pins and Maria's registry. Existing CLI help checks and diff checks pass.
- [Synthetic request](inference_adapter_request.json) and
  [actual CPU response](inference_adapter_response.json) demonstrate both
  requested products. These are suggestions, not ground-truth assertions.

Primary raw Segment suggestions on those exact synthetic inputs were Allergy
for Claritin RediTabs (0.99485886) and Digestive Health for Culturelle Daily
Probiotic (0.91897619). Both require review because automatic acceptance is
disabled and no operational threshold is approved. Neither receives an invented
taxonomy-relationship reason. Sub-Segment additionally remains provisional.

The engine applies the unchanged saved temperatures through the existing helper
and returns those scores separately as research-only. It uses no experimental
cutoff and never enables acceptance. The frozen integration test rehashes all
36 unique serving members after inference. No project datasets or protected
assessment outputs are opened by this integration test; the full unit suite's
assessment tests use temporary synthetic fixtures.

## Unrelated existing CI failure

[Main's existing run 43054059506](https://github.com/import-seb/CatalogIQ-Scope/actions/runs/43054059506)
passes on Ubuntu and fails on Windows before this adapter change. The three
errors are in `test_protected_test_assessment`: failed-authorized-attempt,
preflight-identity and artifact/mapping-change synthetic tests. Each fails in
`file_pin` with `ValueError: Pinned files must be inside the checkout`.
These pass in the local Python 3.13.7 environment. CI uses Python 3.12.10.
The environment-dependent temporary-path cause has not been changed or claimed
resolved. Protected-assessment code/artifacts/results remain untouched.

## Pending application integration

Max's final envelope/ID naming, transport validation responses, request limits,
application lifecycle and deployment resources remain open. They do not block
the reusable engine. All required artifacts exist locally; Max's deployment must
receive the exact private serving members listed by the pins. See the
[integration guide](inference_adapter.md) for the callable API and schema boundary.
