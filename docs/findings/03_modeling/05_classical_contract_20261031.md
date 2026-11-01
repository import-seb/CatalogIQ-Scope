# Classical prediction contract revision, October 31, 2026

Maria's contract at `2345648` declares macro-F1 over model vocabulary union
validation truth, explicit OOV handling, allocation identity, protected IDs/groups
and cost scopes. The classical producer now exports those fields. A separate
metadata revision applies the policy to the eight existing validation runs without
training, inference, unpickling models or changing their predictions.

## Identical predictions, declared class set

| Target / variant | Previous observed-label macro-F1 % | New union macro-F1 % |
|---|---:|---:|
| Segment / base | 83.90222 | 83.90222 |
| Segment / category | 90.99651 | 90.99651 |
| Mnfr / base | 87.16192 | 87.16192 |
| Mnfr / category | 84.37463 | 84.37463 |
| TargetAgeGroup / base | 80.77868 | 80.77868 |
| TargetAgeGroup / category | 77.34484 | 77.34484 |
| Platform / base | 37.61310 | 22.31286 |
| Platform / category | 34.69798 | 21.17165 |

Both Platform runs retain two OOV truth rows as errors. The union contains 59
classes (57 model classes plus two OOV classes). Unsupported full-population log
loss/Brier remain null; no calibration or threshold was selected. Observed scores
remain explicitly available alongside the new headline. Majority-class scores use
the same declared union denominator. All accuracies are unchanged.

Per-run manifests retain the original training source fingerprints, versions and
cost measurements. New metadata records its own producer hash and previous
manifest checksum. `split_allocation_target` is `segment-final-20261009` for these
four targets, with our actual source completion and assignments hashes. Single-
request latency is explicitly null; measured warm-batch time has a separate scope.

## Approved Brand/Sub-Segment allocations received; source identity still blocked

The reviewed allocation handoff contains Brand v1 and Sub-Segment v1 r2.
`classical_allocations.load_approved_targets` consumes sealed CSV/JSON evidence;
it does not execute `verify_handoff.py` or import neural training code. The CLI
requires the handoff and its manifest pin for training either target. An attempt
to train them on old V4 roles, even with `--allow-unseen-validation`, is rejected.

Expected populations and validation ID hashes:

- Brand: 58,888 train / 12,619 validation, 143 training classes, hash
  `87cd25d15e37be3487efafa06f26b5a13908513f760377cb8b233327c02a1ce5`.
- Sub-Segment: 53,956 / 11,562, 49 training classes, hash
  `39a929e0232fe9c76e12240ffe47f78f26c7b9d67ac909e310cf140a4c8491a8`.

Independent inspection confirmed development IDs, raw target labels, eligibility,
original roles, whole groups and protected identities match our data. Train,
validation, assignments and protected-registry source files match the handoff's
raw byte hashes. However, two source-sidecar pins differ:

| File | Our source SHA-256 | Required source SHA-256 |
|---|---|---|
| completion.json | `a00c9ef9a9fcec6a51aa3ef633f6fa3165bd4ce339e9046cac628317ea23f285` | `53f3ab943e66eefc701b3f2c598a5fcd379084a6f9f851c6f7e7e6251d9e9a81` |
| grouping_freeze.json | `9e83972b157a5691d15e7a2d2edbcf8afc09978b8e132c2c1b50242e9d52eb8c` | `2c595b61fff4f1fa3f6ff51c8ff7aabd4bcd98397ece8b74608e0011b854bfa5` |

CRLF normalization does not reconcile the grouping-freeze hash. The real adapter
preflight correctly stops at the source completion check. No Brand/Sub-Segment
model was trained. Obtain the sender's original completion/grouping sidecars,
then inspect all hashes in that completion and obtain any other differing non-test
sidecars. Alternatively review a documented source reconciliation. Do not modify
pins or copy another producer's source seal into our manifests. Keep the existing
source directory and original allocations unchanged.

## Verification and delivery

Two independent audits checked the allocation evidence and prediction handoffs.
The sender's six v2 exports preserve v1 predictions, context and mappings exactly.
Their registry matches our 12,160 protected IDs / 11,979 groups; no test labels
were read. Our revised eight arms independently pass checks of checksums, class
order, probability sums/argmax/confidence, target truth, source keys, validation
hashes, group membership, metrics and protected-test exclusion.

Synthetic tests cover the approved allocation path, refusal of a mismatched
source pin, refusal of training without a handoff, zero-support/OOV union metrics,
immutable metadata revision and prediction-only ZIP contents. The revision run
guards forbid fitting, inference, unpickling and test/candidate access.

Private revision: `data/processed/classical-contract-20261031/`.
Evaluation ZIP: `data/processed/max-classical-validation-contract-20261031.zip`,
12,917,005 bytes, SHA-256
`5c1818719349003ece8f4077e4094af4e0463b90acbbfc65e3865317726ca24b`.
It contains eight validation runs for four targets and ID-only registries, with
no model binaries. The October 29 model ZIP and all previous files remain intact.
No files were sent to teammates automatically. Maria's full comparative harness,
probability diagnostics and group-aware evaluation remain her integration work.
