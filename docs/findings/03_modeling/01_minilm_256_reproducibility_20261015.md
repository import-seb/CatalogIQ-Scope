# MiniLM five-field, 256-token experiment: reproducibility

Verified on October 15, 2026. This is an **experimental model for comparison**.
Maria and the team have not selected an official model. It does not replace the
default transformer configuration or the frozen V4 splitting protocol.

## Which experiment this describes

The successful mixed-retailer experiment is stored in:

```text
data/processed/segment_category_ablation_256_20261010/
├── protocol.json                 # Inputs, source hashes, settings, representation
├── protocol_freeze.json          # Protocol sealed before training
├── completion.json               # Hashes of the completed outputs
├── validation_comparison.csv
├── baseline/                     # Four fields, 256 tokens; comparison arm
└── with_category/                # Five fields, 256 tokens; model discussed here
    ├── checkpoint.pt
    ├── encoder_config.json
    ├── model_config.json
    ├── initialization.json
    ├── training_manifest.json
    ├── training_history.csv
    ├── predictions.csv           # Validation predictions only
    └── tokenizer/
```

| Validation result | Four-field comparison | Five-field MiniLM |
| --- | ---: | ---: |
| Records | 11,952 | 11,952 |
| Accuracy | 91.2734% | **96.9461%** |
| Macro-F1 | 85.8788% | **94.4490%** |

The five-field model is the `with_category` checkpoint above. It is **not** the
later Amazon-only retailer-holdout checkpoint. The original 96.95% accuracy
demonstrates excellent performance on a familiar retailer mix; it does not
establish that level of performance on entirely new retailers.

## Frozen data and separate experimental representation

The experiment consumed the finalized `rule-final-v4` exports in
`data/processed/split_integration_reproduced_20261010/`, using the
`segment-final-20261009` protocol. The loader verifies the original split seal,
record identities, group isolation, and protected final-test identities before
constructing a development view.

| Population | Training | Validation |
| --- | ---: | ---: |
| Frozen exported records | 59,111 | 12,667 |
| Missing Segment, excluded from supervised view | 3,335 | 715 |
| Used by this experiment | **55,776** | **11,952** |

The full labeled development pool was used without downsampling or imputing
missing targets. Records are stably sorted by `record_id`; each role retains its
frozen membership. The supplied `Segment` is the target, not an input feature.

| Contract | Fields | Combined limit |
| --- | --- | ---: |
| Unchanged V4 exact-input grouping contract | ProductName, ProductBrand, ProductDescription, ProductContents | **128 tokens** |
| Separate MiniLM experimental view | ProductCategory, then those same four fields | **256 tokens** |

The experiment verifies V4 using its original 128-token contract before building
the separate 256-token inputs. It does not rewrite grouping, memberships, model
defaults, or any split verification hashes.

### Exact text construction

The implementation is
[`segment_category_ablation.py`](../../../src/catalogiq/segment_category_ablation.py),
which calls the unchanged four-field formatter in
[`segment_transformer.py`](../../../src/catalogiq/segment_transformer.py).

1. For each field, treat missing values and case-insensitive `null`, `none`, and
   `nan` placeholders as empty. Strip surrounding whitespace, unescape HTML,
   remove script/style blocks and HTML tags, then collapse whitespace.
2. Apply these **character** caps after that cleaning:

   | Field, in actual order | Character cap |
   | --- | ---: |
   | ProductCategory | 512 |
   | ProductName | 1,024 |
   | ProductBrand | 512 |
   | ProductDescription | 4,000 |
   | ProductContents | 2,000 |

3. Omit empty fields and their markers. Join nonempty fields with single spaces:

   ```text
   [ProductCategory] ... [ProductName] ... [ProductBrand] ... [ProductDescription] ... [ProductContents] ...
   ```

4. Tokenize that **one combined string**, with special tokens enabled by the
   tokenizer, `truncation=True`, explicit `max_length=256`, `padding=False`, and
   `return_attention_mask=True`. Right-side truncation applies to the combined
   input. There are no per-field token allocations or new truncation refinements
   in this experiment. The trainer pads each batch to its longest input.

`ProductCategory` is the existing retailer breadcrumb. `Category`, retailer name,
identifiers, URLs, provenance, and other target columns are excluded. Adding the
breadcrumb can displace later product text inside the same 256-token budget.
Exactly 16,404 training and 3,510 validation inputs reach the limit; reaching it
does not by itself prove truncation of useful information.

### Tokenizer identity

The experiment loads the saved V4 tokenizer locally, rather than resolving a
moving upstream model name. It is the uncased BERT WordPiece tokenizer for the
pinned MiniLM revision, with a 30,522-token vocabulary, right padding/truncation,
and `[CLS]`, `[SEP]`, `[PAD]`, and `[UNK]` tokens. The emitted effective inputs are
`input_ids`, `attention_mask`, and `token_type_ids`.

The saved tokenizer configuration contains a historical `max_length: 128` entry.
The experiment's explicit `max_length=256` controls this run; the configuration
entry is not the run's sequence limit. Its `model_max_length` is 512.

The tokenizer saved with the trained checkpoint has different file hashes from
the original V4 tokenizer because `save_pretrained` serializes its configuration
again. The experiment protocol binds the original tokenizer files; the completed
output seal binds the saved copies. Verification compares their actual encoded
arrays, rather than inferring equivalence from their filenames.

## Model and training settings

| Setting | Verified value |
| --- | --- |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Pinned revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| Architecture | Six BERT encoder layers, 12 attention heads, hidden size 384 |
| Head | Attention-mask mean pooling → dropout 0.10 → linear 384-to-8 classifier |
| Trainable parameters | 22,716,296; encoder and head both fine-tuned |
| Target | Segment; eight fixed classes in the saved `label_names` order |
| Epochs/checkpoint | Three fixed epochs; always save epoch three |
| Batch sizes | Training 32; validation 64 |
| Optimizer/loss | AdamW; unweighted cross-entropy |
| Learning rate/weight decay | `3e-5` / `0.01` |
| Schedule | 522 warmup steps, then linear decay to zero; 5,229 total steps |
| Gradient clipping | Maximum norm 1.0 |
| Precision | FP32; TF32 disabled |
| Seed | 42 for Python, NumPy, PyTorch, CUDA, and CPU batch shuffling |
| Determinism | Deterministic algorithms; cuDNN benchmark off; `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Threads | Four PyTorch CPU threads |
| Recorded GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Recorded software | PyTorch `2.11.0+cu128`; CUDA `12.8`; Transformers `5.3.0`; NumPy `2.3.3` |
| Repository tokenizer dependency | `tokenizers==0.22.2` |

The comparison arm began from the same initial model state. Validation was
observed after every epoch but did not choose the checkpoint. There was no early
stopping, validation-driven epoch selection, or final-test scoring.

Reproducibility means the same software, hardware, input bytes, ordering, and
configuration. One training seed was run. Existing group-bootstrap intervals
describe validation sampling uncertainty, not variation across training seeds.
The original manifest does not record a complete environment lockfile, Python
patch version, GPU driver version, or every transitive dependency; identical
results on a different stack are not guaranteed.

The historical protocol also records absolute source/cache paths. The archived
source-seal check currently requires those recorded paths to remain available;
relocating the archive may require a separate read-only path mapping. Do not edit
the original manifests or hashes to accommodate a move. The checkpoint-loading
example below uses its local folder and does not require the original cache paths.

## Identities to verify

The hexadecimal values below are SHA-256. Checkpoint-file and model-state hashes
are distinct: the state hash covers ordered tensor names, types, shapes, and
values; the file hash covers the serialized checkpoint bytes.

| Item | Expected SHA-256 |
| --- | --- |
| Five-field `checkpoint.pt` | `7e417d8204dbaeef4cea4e54690594560e3fbd3b8288f4be850ff60f4de32e27` |
| Five-field final tensor state | `08d6f4d337dc35f858515ebd5be54200d41c5ca77e2f862fff5ad549aad842bd` |
| Common initial tensor state | `8c4b98ca24f4a797fd0e498684d4bcec78c7c027ba8761799de822174ebc6207` |
| Four-field 256-token comparison checkpoint | `0938bb03b6e49e0aef058d16592d7f194a9731d340b514eefc13e79129207ece` |
| Successful experiment `protocol.json` | `e31853f0453612aeed725d0659b84d7fb8093b71bea24d244811c66b69ec72bd` |
| Ordered supervised training IDs | `df9cdf4753a7fec526129baaa38f62ed8590a1f5df6c9dd0005571c92b311c2f` |
| Ordered supervised validation IDs | `47f60a59ac99f8ca3375926505cbb9dcffddb03446cfde024abbacd340e96dea` |
| Five-field full development text sequence | `072421f04779f02f70eaf43262547cc7587403efbed13fd3741ae3573560b8d4` |
| Five-field full development token arrays | `ea7830cdb4b281efe190dc56ebdbc804fb4e80f0d7b7a1260d996c7e73fe09dd` |
| Original V4 `grouping_freeze.json` | `2c595b61fff4f1fa3f6ff51c8ff7aabd4bcd98397ece8b74608e0011b854bfa5` |
| Original V4 four-field, 128-token contract | `47f4de902ee95fd840ea847a3ea48defb1e14b376178ffabc26cd2369cb7b3e0` |
| Original V4 `model_config.json` | `97989422ebbc7246b46e49588038a92b298893d5a097cdfe23ac728d08d76def` |

`protocol.json` also seals the original source modules, frozen train/validation
exports, tokenizer, and cached pretrained encoder. `completion.json` seals all
26 experiment artifacts. The verification audit checks these without rewriting
them. See the companion audit for the verification output and checkpoint
preservation inventory.

## Reproduce without touching existing outputs

Run commands from the repository root. The original local interpreter was
`C:/Users/sebas/AppData/Local/Programs/Python/Python313/python.exe`, with the CUDA
PyTorch installation supplied from `.cache/segment_cuda`. Teammates should use an
environment with the recorded versions and adapt only these environment paths.
The cached pinned encoder and saved tokenizer must already be available; these
commands deliberately prohibit network downloads.

Preparation and hashing only, **without training**:

```powershell
$env:PYTHONPATH = '.cache/segment_cuda;src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$experimentPython = 'C:/Users/sebas/AppData/Local/Programs/Python/Python313/python.exe'

& $experimentPython -m scripts.compare_segment_category `
  --split-dir data/processed/split_integration_reproduced_20261010 `
  --model-config data/processed/segment_category_ablation_256_20261010/with_category/model_config.json `
  --output-dir data/processed/minilm_256_reproduction_preparation_20261015 `
  --sequence-length 256 --category-character-limit 512 `
  --bootstrap-samples 1000 --device cuda --prepare-only
```

For an explicitly authorized future retraining, use the same command with a
**different, nonexistent output directory** and omit `--prepare-only`. This
reproduces the original two-arm experiment, including its four-field control.
It does not resume training from the successful checkpoint. The command refuses
an existing output directory unless `--resume` is supplied; `--resume` reuses
verified completed arms rather than resuming optimizer state. No retraining was
needed for this verification task.

Representation-specific equality and reproducibility verification:

```powershell
& $experimentPython -m scripts.audit_segment_representation `
  --split-dir data/processed/split_integration_reproduced_20261010 `
  --experiment-dir data/processed/segment_category_ablation_256_20261010 `
  --output-dir data/processed/minilm_representation_audit_20261015
```

Use a new output directory to repeat the audit if that directory already exists.
This audit reads only allowlisted final-test input features and identities; it
does not read test targets, run test inference, or calculate test performance.

## Load the saved checkpoint for inference

The checkpoint contains the fine-tuned encoder and custom mean-pooling head.
Reconstruct that architecture, not a different classification head. The example
uses local configuration and tokenizer files and loads every saved parameter
strictly; no pretrained encoder download or training is involved.

Run with the environment above. Supply a DataFrame with the five input columns;
the example is synthetic and does not use a final-test record.

```python
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from catalogiq.segment_category_ablation import build_category_texts
from catalogiq.segment_transformer import ModelConfig, _make_classifier

folder = Path("data/processed/segment_category_ablation_256_20261010/with_category")
manifest = json.loads((folder / "training_manifest.json").read_text(encoding="utf8"))
actual_sha = hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()
assert actual_sha == manifest["artifacts"]["checkpoint.pt"]

checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
config = ModelConfig.from_dict(checkpoint["model_config"])
encoder_settings = json.loads((folder / "encoder_config.json").read_text(encoding="utf8"))
encoder_config = AutoConfig.for_model(encoder_settings.pop("model_type"), **encoder_settings)
model = _make_classifier(
    torch, AutoModel.from_config(encoder_config),
    len(checkpoint["label_names"]), checkpoint["dropout"],
)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"), local_files_only=True)

products = pd.DataFrame([{
    "ProductCategory": "Health & Household > Vitamins, Minerals & Supplements",
    "ProductName": "Example Vitamin D3 1000 IU Softgels",
    "ProductBrand": "Example",
    "ProductDescription": "Vitamin D3 dietary supplement.",
    "ProductContents": "Cholecalciferol (vitamin D3) 25 mcg",
}])
texts = build_category_texts(
    products, config,
    category_character_limit=manifest["experimental_representation"]["category_character_limit"],
)
inputs = tokenizer(
    texts, truncation=True, max_length=config.sequence_length,
    padding=True, return_attention_mask=True, return_tensors="pt",
)
with torch.inference_mode():
    probabilities = model(**inputs).softmax(dim=-1)
predicted_segments = [checkpoint["label_names"][i] for i in probabilities.argmax(-1).tolist()]
print(predicted_segments)
```

The checkpoint stores model weights, model configuration, class ordering, and
state hashes. Category prefix settings live in the annotated manifest/protocol,
so preserve those files alongside the weights. Optimizer, scheduler, and RNG
states are not saved; exact continuation of an interrupted training run is not
supported. The weights remain available for inference, comparisons, and a future
explicitly configured fine-tuning run.

## Equality audit and limits

The original experiment's zero identical-input crossings applied to supervised
**training versus validation only**. Its generic diagnostic object also printed
zero train/test and validation/test entries, but the original run never read
`test.csv`; those entries were not measurements of the final test.

The new [five-field equality audit](02_minilm_256_equality_audit_20261015.md)
covers all 83,938 frozen records, including unlabeled records, using only the
five feature columns and identities. It reports every partition comparison,
affected IDs, and existing group assignments. It is an input-equality integrity
check, not a test-set evaluation or certification of model quality. Exact-input
equality does not detect all semantic or near-duplicate leakage.

No official model selection, default promotion, split reassignment, checkpoint
overwrite, final-test prediction, or retraining is part of this verification.
