# Predictions contract

What a model run has to hand over so the evaluation harness can score it without
anyone editing the harness or guessing. Six targets and three people write
against this, so it is worth being exact.

**Status:** the five decisions in §1 are settled. §2 is the required core, and
both existing packages already satisfy it. §3 is the part the five decisions do
not cover, and it is where the real disagreements have been.
**Harness:** `notebooks/04-segment-evaluation.ipynb` on
`maria/segment-evaluation-harness`.
**Owner:** Maria Garcia Sehara

---

## 1. The five settled decisions

1. **Manifest filename.** One file, `manifest.json`, next to `predictions.csv`,
   for every run.
2. **Class mapping.** The explicit `probability_0 → label` mapping, not an
   ordered `label_names` list. It reads better and the relationship between a
   probability column and its class is impossible to misread.
3. **Validation hash.** `validation_record_ids_sha256` goes in each run's own
   manifest, beside its `predictions.csv` — not only in a parent manifest.
4. **Performance metrics.** Per-model training time, hardware and inference
   latency. `null` for anything not measured yet, rather than blocking.
5. **Line endings.** New files are LF. Existing CRLF artifacts stay as they are
   and get documented instead of rewritten.

Each of those came out of something that cost real time, so the reasons are
recorded in §4.

---

## 2. The required core

### `predictions.csv`

These columns must all be present, in this relative order:

```
record_id, true_label, predicted_label, confidence, probability_0 … probability_{K-1}
```

**Extra columns are allowed and a scorer must ignore them.** One package ships
exactly this set; the other adds `dataset`, `source_sha256`, `source_row`,
`group_id` and `split` ahead of them. Those five are useful and there is no
reason to make anyone strip them, so the contract requires a minimum rather than
forbidding extras.

The harness used to derive `group_id` from the split registry and assign it over
whatever arrived in the file, which would have hidden a disagreement. It now
compares and raises instead; a shipped `group_id` is checked, not overwritten.

Rules the loader checks and will refuse a run over:

- `record_id` unique, and every row in the declared validation population.
- **No row from the reserved test partition.** The loader raises rather than
  scoring it.
- Probabilities sum to 1 and none negative.
- `argmax` of the probability columns equals `predicted_label` on at least 99%
  of rows. This is the check that catches a wrong class order, which a manifest
  cannot catch on its own.
- `confidence` equals the row's maximum probability.
- `true_label` agrees with the declared validation context row by row.

### `manifest.json`

These thirteen keys are required, and **all fourteen current runs write all of
them**. When this was first written the probabilistic Brand lookup was missing
`protocol_id` and `config`; both packages have since been reissued against this
document and nothing is outstanding.

| Key | What it is |
|---|---|
| `format` | contract identifier, e.g. `catalogiq-prediction-run-v1` |
| `target` | `Segment`, `Brand`, `Mnfr`, `Sub-Segment`, `TargetAgeGroup`, `Platform` |
| `partition` | `validation`. Anything else is refused. |
| `protocol_id` | the split protocol, e.g. `segment-final-20261009` |
| `model` | short human description of what produced the run |
| `config` | the settings that would be needed to reproduce it |
| `input_fields` | the fields the model actually read |
| `label_mapping` | `{"probability_0": "<class>", …}`, per decision 2 |
| `validation_rows` | row count; must equal `len(predictions.csv)` |
| `validation_record_ids_sha256` | per decision 3, recipe in §2.1 |
| `probabilities_calibrated` | `true` / `false`. Never omit it. |
| `training_seconds` | per decision 4, `null` if unmeasured |
| `hardware` | per decision 4, `null` or `"not recorded"` if unknown |

### 2.1 The record-id hash

```python
digest = hashlib.sha256()
for record_id in record_ids:              # in predictions.csv order
    payload = record_id.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))   # 8-byte big-endian length
    digest.update(payload)
```

The 8-byte length prefix is the part that is easy to miss — 24 plausible
recipes without it all produce the wrong digest. No separator, no count prefix.
Hash the **labelled** rows only, the same population `predictions.csv` holds.

The harness recomputes this from its own copy of the split and refuses any run
whose manifest value does not match, so a run trained against a different split
cannot be scored by accident.

---

## 3. What the five decisions do not cover

These caused more trouble than anything in §1 and need deciding too.

### 3.1 Which class set the macro average runs over

The same predictions give very different macro-F1 depending on the denominator.
Platform, base arm:

| Averaged over | Macro-F1 |
|---|---|
| Only the classes observed in validation (35) | **37.61** |
| The model's own vocabulary (57) | 23.10 |
| **Model vocabulary ∪ classes present in the truth (59)** | **22.31** |

**15 points of spread on one run.** Two people can report honestly and disagree
by that much, which is exactly what a contract is for.

**Settled: the union.** Every class the model can emit, plus every class that
appears in the truth. Both packages now declare
`macro_f1_class_set: model_vocabulary_union_truth` on every run, and their
figures reconcile with an independent recomputation to 1e-12, label lists
included. It is the most conservative reading and the only one that
does not hide a class the model has no chance on. State it in the manifest:

```json
"macro_f1_class_set": "model_vocabulary_union_truth"
```

Report the observed-label figure too if it is useful, but always labelled as
such and never as the headline. For targets where every class has validation
support — Segment, Mnfr, TargetAgeGroup — all three definitions agree, which is
why this went unnoticed until Brand, Sub-Segment and Platform arrived.

### 3.2 True labels outside the model's vocabulary

Platform has two validation rows whose true label the model cannot predict at
all. Required:

```json
"out_of_vocabulary_true_labels": 2,
"out_of_vocabulary_policy": "retained_as_errors"
```

They stay in, they count as errors, and they count in the macro denominator per
§3.1. **Nobody drops them silently.** Worth flagging for whoever implements
scoring: a Brier score that builds its one-hot target by looking the true label
up in the class list raises `KeyError` on these rows rather than failing
visibly.

### 3.3 Which validation population

Brand and Sub-Segment use target-specific allocations, not the Segment V4
validation membership — 12,619 and 11,562 rows against Segment's 11,952, with
only 79% and 91% of their rows inside the Segment validation set. The hash in
§2.1 proves *a* population but does not say *which*. Required:

```json
"split_allocation_target": "brand_development_allocation_v1_20261029"
```

All allocations reuse the same frozen V4 product groups, and that has been
checked: every record shared between two targets carries an identical
`group_id`. That is what keeps the group-aware bootstrap valid across targets,
so it must stay true.

### 3.4 Where `group_id` travels

Currently it arrives two ways: as a column in `predictions.csv`, or in a
separate `validation_context.csv` with `record_id, group_id, split, true_label`.
Both work and the harness reads both.

**Settled: either is fine, and it must be one of the two.** Both are in use —
one package ships `group_id` as a column in `predictions.csv`, the other in a
separate `validation_context.csv` — and the harness reads both. Since §2 permits
extra columns there is no reason to make anyone move a column that already
works, and a separate context file is equally good — it also gives a second,
independent copy of `true_label` to check the predictions against, which is how
the row-by-row label agreement in §2 gets verified.

What is **not** acceptable is neither. `group_id` is required, because without
it the bootstrap has to resample rows, and rows inside a product group are not
independent, so the intervals come out too narrow and every "the difference is
established" claim gets weaker than it looks.

If a producer prefers the context file, shipping the provenance key — `dataset`,
`source_sha256`, `source_row` — in it as well is welcome and never required.

### 3.5 The protected test identity registry

The harness refuses test rows by checking against a registry of protected
record IDs. For Segment that registry is available, so the check genuinely
runs. When this was written the Brand and Sub-Segment allocations had no such
registry, so their test exclusion rested on the producer's word.

Required for any new allocation: **the protected test record IDs and group IDs,
labels excluded.** IDs only. Both packages now ship a 12,160-row
`protected_test_identity_registry.csv` with exactly `record_id, group_id` and no
label column, byte-identical between the two. Checked here rather than taken on
trust: zero protected records and **zero protected groups** across all fourteen
runs. The group-level check is the stronger one, since the splitter is
group-aware.

### 3.6 Cost field names

Decision 4 settles that cost is included; it does not settle the names, and the
two packages use different ones. Required:

```json
"training_seconds": 1356.14,
"training_seconds_scope": "3 epochs including per-epoch validation",
"inference_latency_ms": null,
"inference_latency_scope": "single-request latency, unmeasured",
"hardware": "NVIDIA GeForce RTX 3060 Laptop GPU"
```

Amortised batch time divided by record count is a useful number and is **not**
single-request latency; if both are reported they get separate keys. Both of the
current packages now do this correctly, each with its own scope string, and
between them the cost column can finally be filled — per-model training seconds
and amortised milliseconds per record for every run.

Two things still have to be read off the scopes rather than the numbers. The
amortised figures were measured on different devices — one model's inference ran
on a GPU and the other's on CPU — so the ratio between them flatters whichever
got the accelerator.

Neither package has benchmarked single-request latency, and both now say so the
same way: `inference_latency_ms: null` with a scope string. That is decision 4
working as intended — an absent key cannot be told apart from an oversight, a
null one can. Amortised per-record figures are reported separately under their
own keys, as they should be, since they are a different quantity.

### 3.7 Which hashes are comparable between machines, and which are not

This one cost a round trip, so it is worth writing down. Four files the splitter
writes carry a wall-clock timestamp, so their SHA-256 can never match across two
machines, or across two runs on one machine:

| File | Varying field |
|---|---|
| `grouping_freeze.json` | `reproduced_utc` |
| `test_reservation.json` | `reserved_utc` |
| `summary.json` | `created_utc` |
| `completion.json` | `completed_utc`, plus the hashes of the three above |

`completion.json` is a seal **internal to one split directory**: it records the
hashes of that directory's own files, and `--verify` checks them against that
same directory. It was never meant to be compared between machines, and
comparing it is what produces a mismatch that looks alarming and means nothing.

The cross-machine anchors are the ones `split_protocol.py` actually checks:

- `completion["protocol_manifest_sha256"]` against the committed bundle's
  `manifest.json`
- `protocol_manifest.json`, `eligibility.json` and the four config JSONs, byte
  for byte against that bundle
- `expected.splits`, which carries per-partition `records`, `groups`,
  `record_ids_sha256` and `assignment_sha256`

All of that lives in `src/catalogiq/resources/segment_final_20261009/manifest.json`,
which **is in the repo** and contains no timestamps and no absolute paths. That
is the pinned source. `/data/processed/` is gitignored, so there is no pinned
copy of the generated split directory and asking for one does not help.

Note which code wrote your split. `scripts.split_data` is the team workflow and
goes through `split_protocol.py`, where the only varying field in
`grouping_freeze.json` is `reproduced_utc`. The older `scripts.finalize_splits`
path writes a different freeze that also embeds absolute machine paths, and the
README says plainly it is not the portable workflow.

### 3.8 Line-ending normalisation, stated

Decision 5 covers the files. The manifest should also say how its own hashes
were taken, because that is where it bit:

```json
"source_hash_normalization": "raw_bytes"
```

Four of six source hashes in one package matched no version of those files in
the repository's full history on a raw comparison, and all four matched under
CRLF. Nothing was wrong; the normalisation just was not stated.

---

## 4. Why each decision exists

Short version, so nobody has to relitigate them.

- **One manifest filename.** The loader originally required
  `training_manifest.json` and would have skipped a package shipping
  `label_mapping.json` plus `manifest.json` entirely.
- **Explicit class mapping.** Two packages used two shapes — an ordered
  `label_names` list and `{"probability_0": "Allergy"}`. The loader now reads
  either and verifies the result against the argmax, but one shape is better
  than a reader that guesses.
- **Hash beside the predictions.** One package had to be repackaged because the
  hash sat only in the parent manifest, so the per-run guard never fired.
- **Cost with scope.** An earlier package reported 53.38 seconds of combined
  wall time for two arms on unrecorded hardware, with an explicit note not to
  treat it as a per-model benchmark — honest, and impossible to put in the same
  column as a named GPU's 1,356 training seconds. The current packages both ship
  per-model seconds and amortised per-record latency with scope strings, and the
  comparison works.
- **LF, and state the normalisation.** See §3.8.

---

## 5. What not to ship

- **Reserved test rows or test labels.** Ever.
- **Pickled models as a requirement.** `model.joblib` and friends execute code
  when loaded, so scoring must never depend on unpickling one. Ship them
  separately if they are useful; the evaluation only needs
  `predictions.csv`, `manifest.json` and the validation context.
- **Raw product text** beyond what an error-inspection sample needs.
- **Absolute workstation paths** as anything other than provenance. They do not
  need to resolve anywhere else, and the manifests already say so.

---

## 6. Checklist for a new run

- [ ] `predictions.csv` containing the §2 columns in order; extras are fine
- [ ] `group_id` present, either as a column there or in a
      `validation_context.csv` with `record_id, group_id, split, true_label` (§3.4)
- [ ] `manifest.json` with the thirteen required keys
- [ ] `macro_f1_class_set` declared (§3.1)
- [ ] `out_of_vocabulary_true_labels` and its policy (§3.2)
- [ ] `split_allocation_target` (§3.3)
- [ ] protected test IDs for any new allocation (§3.5)
- [ ] cost keys with their scopes, `null` where unmeasured (§3.6)
- [ ] `source_hash_normalization` (§3.8)
- [ ] LF line endings on everything new
