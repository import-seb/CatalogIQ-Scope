# MiniLM five-field, 256-token equality audit

Verified on October 15, 2026, on branch `feat/transformer_model`.
MiniLM remains an **experimental comparison candidate**. Maria and the team have
not selected an official model, and no default workflow was changed.

## Result

**No identical effective tokenized inputs cross the frozen V4 partitions under
the successful experiment's five-field, 256-token representation.**

| Partition comparison | Identical-input pairs | Affected records |
| --- | ---: | ---: |
| Train / validation | 0 | 0 |
| Train / test | 0 | 0 |
| Validation / test | 0 | 0 |

The audit covers **all 83,938 frozen records**: 59,111 training, 12,667 validation,
and 12,160 test records. Development records without Segment labels are included
in this conservative, label-free input audit. Final-test label availability and
eligibility were not inspected.

There are 83,544 distinct effective input signatures, 362 duplicate signature
classes, and 432 duplicate pairs. All duplicate pairs remain within a single
partition; the largest class contains five records. Existing product groups
also remain isolated across partitions.

**Affected record IDs: none. Affected existing group IDs: none.**
`cross_split_pairs.csv` and `affected_records.csv` contain headers and no rows.
`record_signatures.csv` records every audited ID, original group assignment,
partition, signature, and effective token count.

## What was reconstructed

The verified checkpoint is `with_category/checkpoint.pt` in
`data/processed/segment_category_ablation_256_20261010/`, not the later Amazon-only
model. Its inputs are, in order:

```text
ProductCategory, ProductName, ProductBrand, ProductDescription, ProductContents
```

The auditor calls the experiment's unchanged `build_category_texts` formatter:
the original HTML/placeholder/whitespace handling, character limits, field
markers, and 512-character category prefix are preserved. It loads the saved
tokenizer locally and uses the same explicit tokenizer arguments:

```python
truncation=True, max_length=256, padding=False, return_attention_mask=True
```

Each signature hashes **all** effective unpadded `input_ids`, `attention_mask`,
and `token_type_ids`: field names, four-byte big-endian lengths, and little-endian
int32 array bytes, followed by SHA-256. Equal hashes are checked against their
full serialized payloads to reject a hash collision. Dynamic batch padding is
excluded because it does not change the effective product input.

The reconstructed ordered supervised development text and token-array digests
exactly match both the original protocol and trained model manifest:

| Development identity | SHA-256 |
| --- | --- |
| Text sequence | `072421f04779f02f70eaf43262547cc7587403efbed13fd3741ae3573560b8d4` |
| Effective token arrays | `ea7830cdb4b281efe190dc56ebdbc804fb4e80f0d7b7a1260d996c7e73fe09dd` |

This verifies the exact inputs used for **55,776 supervised training records and
11,952 validation records**, including their original ordered membership hashes.
An independent pass also compared the original frozen tokenizer with the copy
saved beside the checkpoint on **all 83,938 records**. All three effective arrays
matched exactly, and every per-record signature matched the audit output.
The original experiment only measured development equality; its generic zero
test-comparison entries did not constitute a final-test audit. This audit adds
the actual test-input comparisons.

## Final-test access boundary

The equality audit parsed only these eight final-test columns:

```text
record_id, group_id, split,
ProductCategory, ProductName, ProductBrand, ProductDescription, ProductContents
```

An independent `pandas.read_csv` guard rejected unrestricted test reads or any
target column. Protected-test metadata was restricted to identity columns.
No final-test labels, predictions, or performance were inspected or computed.
No raw product text is exported in the audit results.

The normal V4 verification separately checked sealed file bytes and parsed only
test identity/provenance fields. Its historical summary output was suppressed.
Reading input features for this authorized integrity check does not make the
test available for selecting thresholds, choosing models, or tuning training.

## Experiment and compatibility checks

- Recomputed saved validation accuracy: **96.9461%**; macro-F1: **94.4490%**.
- Input fields/order, tokenizer, pinned encoder revision, seed 42, three fixed
  epochs, FP32, batch sizes, optimizer/schedule, and checkpoint identity match
  the saved experiment. See the [reproducibility document](01_minilm_256_reproducibility_20261015.md).
- All **26 original experiment artifacts and 14 pinned source artifacts** match
  their original seals.
- All **eight pre-existing checkpoints** match their original training-manifest
  hashes. The successful checkpoint loads strictly on CPU and passes synthetic
  inference with its saved mean-pooling classification head.
- The official `python -m scripts.split_data --verify` workflow passes against
  the finalized V4 exports. Its four-field, **128-token** contract and verification
  hashes are unchanged; the default `ModelConfig` remains 128 tokens.
- No split membership, group assignment, source dataset, cleaning pipeline,
  frozen contract, model default, or checkpoint was modified. No model was
  retrained, no commit was created, and no PR was opened.
- **440 automated tests passed**, including 15 new synthetic equality-audit
  tests. The offline package wheel build passed.

Artifact/input reproducibility and checkpoint loading were verified without
rerunning training. Bitwise retraining on a new machine remains untested; the
original run did not capture a complete environment lockfile or optimizer state.
The historical source seals also retain absolute paths; relocation needs a
read-only path mapping or access to the recorded layout, as documented in the
reproducibility notes.

## Reproduce and inspect

From the repository root, in the environment described in the reproducibility
document:

```bash
python -m scripts.audit_segment_representation --split-dir data/processed/split_integration_reproduced_20261010 --experiment-dir data/processed/segment_category_ablation_256_20261010 --output-dir data/processed/minilm_representation_audit_repeat
python -m unittest discover -s tests -p test_segment_representation_audit.py -v
```

Use a new output directory: the command refuses to overwrite an existing one.
The audit computes signatures and validation metrics; it does not train or run
final-test inference.

| Evidence | Location relative to repository root |
| --- | --- |
| Full machine-readable audit, input/source hashes, settings | `data/processed/minilm_representation_audit_20261015/audit.json` |
| Every record's signature and original group | `data/processed/minilm_representation_audit_20261015/record_signatures.csv` |
| Cross-partition pair evidence; empty | `data/processed/minilm_representation_audit_20261015/cross_split_pairs.csv` |
| Affected IDs/groups; empty | `data/processed/minilm_representation_audit_20261015/affected_records.csv` |
| Audit output checksums | `data/processed/minilm_representation_audit_20261015/completion.json` |
| Feature-only access guard | `.cache/minilm_validation_20261015/equality_access_guard.json` |
| Official V4 verification and identity-only guard | `.cache/minilm_validation_20261015/v4_verification.json` |
| Original checkpoint/artifact preservation inventory | `.cache/minilm_validation_20261015/checkpoint_preservation.json` |
| Post-audit preservation: eight checkpoints, 61 source seals, four output seals | `.cache/minilm_validation_20261015/final_preservation.json` |
| Strict checkpoint loading proof | `.cache/minilm_validation_20261015/checkpoint_loading.json` |
| Independent membership and metric checks | `.cache/minilm_validation_20261015/development_membership_metrics.json` |
| Original/saved tokenizer equivalence on every record | `.cache/minilm_validation_20261015/tokenizer_equivalence.json` |
| Test and build logs | `.cache/minilm_representation_validation_20261015/unittest.log`, `build.log` |

Generated data and cache evidence remain local and ignored by Git. The auditor,
synthetic tests, and documentation are repository files; saved source/output
hashes bind the local verification to the preserved experiment.

## Remaining risk and conflict policy

Zero exact-input collisions does **not** prove zero leakage: similar products,
semantic duplicates, label quality, and production behavior remain separate
questions. This is an equality-integrity check, not official model selection.

No membership resolution is required for this representation's exact equality.
If a future audit finds conflicts, preserve V4 and document the conflicting IDs
and groups. Any required regrouping/reallocation should be a separately versioned
protocol with a fresh test-integrity decision, rather than a silent repair to
the current frozen test or tuning to its examples.
