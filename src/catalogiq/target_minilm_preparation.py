"""Target-aware inputs and dry-run-first MiniLM orchestration.

Preparation and dry runs do not train, download weights or create output files.
Explicitly authorized training delegates to the unchanged legacy loop. Brand/Sub-Segment
require a pinned approved allocation seal. The experimental
representation stays ProductCategory-first, five fields and 256 tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib import metadata as package_metadata
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Sequence

import numpy as np
import pandas as pd

from .features import sha256
from .segment_category_ablation import CATEGORY, load_tokenizer, pretrained_artifacts, tokenize_category_corpus
from .segment_transformer import MODEL_FIELDS, ModelConfig, TokenizedCorpus, _hash_items, _validate_indices, train_transformer
from .split_protocol import verify_protocol_splits
from .splitting import check_assignments
from .target_development_preparation import MISSING_TARGETS, _support, _validate_identity


SUPPORTED_TARGETS = ("Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")
REALLOCATED_TARGETS = ("Brand", "Sub-Segment")
INPUT_FIELDS = (CATEGORY, *MODEL_FIELDS)
IDENTITY_FIELDS = ["record_id", "group_id", "split"]


@dataclass(frozen=True)
class PreparedTargetMiniLM:
    target_column: str
    frame: pd.DataFrame
    label_names: tuple[str, ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    protected_test: pd.DataFrame
    config: ModelConfig
    tokenizer_directory: Path
    model_identity: str
    output_directory: Path
    audit: dict

    @property
    def supervised_frame(self) -> pd.DataFrame:
        return self.frame.loc[self.frame.is_labeled].reset_index(drop=True).copy()

    @property
    def legacy_frame(self) -> pd.DataFrame:
        """Copy only the selected labels and five allowlisted product fields."""
        return self.supervised_frame[["record_id", self.target_column, *INPUT_FIELDS]].rename(
            columns={self.target_column: "Segment"})

    def trainer_arguments(self, corpus: TokenizedCorpus) -> dict:
        """Return compatible arguments without calling or wrapping training.

        The legacy trainer has Segment/four-field manifest defaults. Use the
        shared orchestration entry point to register its actual target and view.
        """
        expected = self.supervised_frame
        labels = expected[self.target_column].map({name: i for i, name in enumerate(self.label_names)})
        if (corpus.record_ids != tuple(expected.record_id) or corpus.label_names != self.label_names
                or corpus.sequence_length != self.config.sequence_length or self.config.sequence_length != 256
                or not np.array_equal(corpus.label_ids, labels.to_numpy())
                or _membership_hash(expected) != self.audit["ordered_supervised_membership_sha256"]
                or _hash_items(expected[self.target_column].tolist()) != self.audit["ordered_supervised_label_sha256"]
                or not np.array_equal(self.train_indices, np.flatnonzero(expected.split.eq("train").to_numpy()))
                or not np.array_equal(self.validation_indices, np.flatnonzero(expected.split.eq("validation").to_numpy()))):
            raise ValueError("Corpus does not match the prepared target, label order or record order")
        _validate_indices(corpus, self.train_indices, self.validation_indices)
        return {"corpus": corpus, "train_indices": self.train_indices.copy(),
                "validation_indices": self.validation_indices.copy(), "label_names": self.label_names,
                "config": self.config, "output_dir": self.output_directory}


def _json(path):
    return json.loads(path.read_text(encoding="utf8"))


def _read(directory, name, columns):
    return pd.read_csv(directory / name, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", usecols=columns)[columns]


def _membership_hash(frame):
    return _hash_items([json.dumps(list(row), ensure_ascii=False)
        for row in frame[["record_id", "group_id", "split"]].itertuples(index=False, name=None)])


def _snapshot(directory, seal, names):
    result = {}
    for name in names:
        expected = seal.get("artifact_sha256", {}).get(name)
        if not expected or sha256(directory / name) != expected:
            raise ValueError(f"Sealed input changed or checksum missing: {name}")
        result[name] = expected
    return result


def _allocation(directory, expected_sha256, target, source_frame, protected, source_seal_sha, source_hashes):
    seal_sha = sha256(directory / "completion.json")
    if seal_sha != expected_sha256:
        raise ValueError("Allocation completion checksum differs from the approved allocation")
    seal = _json(directory / "completion.json")
    consumed = _snapshot(directory, seal, ("protocol.json", "assignments.csv", "development_assignments.csv",
                                          "class_coverage.csv", "protected_final_test.csv"))
    protocol = _json(directory / "protocol.json")
    preparation = protocol.get("preparation", {})
    if (protocol.get("version") != f"{target.casefold()}-development-allocation-v1"
            or preparation.get("target_column") != target
            or preparation.get("completion_sha256") != source_seal_sha
            or any(preparation.get("consumed_artifact_sha256", {}).get(name) != digest
                   for name, digest in source_hashes.items()
                   if name in ("train.csv", "validation.csv", "grouping_freeze.json", "protected_final_test.csv"))):
        raise ValueError("Allocation target or frozen source identity differs")
    assignments = _read(directory, "assignments.csv", IDENTITY_FIELDS)
    reserved = _read(directory, "protected_final_test.csv", IDENTITY_FIELDS)
    columns = ["record_id", "group_id", "original_split", target, "is_labeled", "split"]
    metadata = _read(directory, "development_assignments.csv", columns)
    _validate_identity(metadata, "allocated development")
    check_assignments(assignments, [*source_frame.record_id, *protected.record_id])
    canonical = lambda frame: frame[IDENTITY_FIELDS].sort_values("record_id").reset_index(drop=True)
    if (not canonical(reserved).equals(canonical(protected))
            or not canonical(assignments.loc[assignments.split.eq("test")]).equals(canonical(protected))):
        raise ValueError("Protected test population differs from frozen V4")
    if not metadata.split.isin(("train", "validation")).all() or set(metadata.record_id) != set(source_frame.record_id):
        raise ValueError("Allocation must preserve every development record and development-only role")
    original = source_frame.set_index("record_id").loc[metadata.record_id]
    for column in ("group_id", "original_split", target):
        if metadata[column].tolist() != original[column].tolist():
            raise ValueError(f"Allocation changed frozen development {column}")
    if metadata.is_labeled.tolist() != original.is_labeled.map({True: "True", False: "False"}).tolist():
        raise ValueError("Allocation changed target supervision eligibility")
    assigned = assignments.set_index("record_id").loc[metadata.record_id]
    if (assigned.group_id.tolist() != metadata.group_id.tolist()
            or assigned.split.tolist() != metadata.split.tolist()):
        raise ValueError("Allocation assignment exports disagree")
    # Allocation CSV order, including its class order, is authoritative here.
    frame = source_frame.set_index("record_id").loc[metadata.record_id].reset_index()
    frame["split"] = metadata.split.to_numpy()
    names = tuple(_read(directory, "class_coverage.csv", ["label"]).label)
    return frame, names, {**consumed, "completion.json": seal_sha}


def prepare_target_minilm(split_directory, target_column: str, output_root, *,
                          allocation_directory=None, allocation_completion_sha256=None,
                          label_names: Sequence[str] | None = None,
                          config: ModelConfig | None = None) -> PreparedTargetMiniLM:
    """Prepare identities, raw product fields and ordered labels without writes.

    Selecting an approved Brand/Sub-Segment directory plus its expected
    completion SHA-256 pins the exact allocation and class ordering. Existing
    proposal status text need not be rewritten to record the caller's approval.
    V4 targets use deterministic class order unless explicitly supplied.
    """
    if target_column not in SUPPORTED_TARGETS:
        raise ValueError(f"MiniLM target must be one of {SUPPORTED_TARGETS}; Platform is excluded")
    reallocated = target_column in REALLOCATED_TARGETS
    if reallocated and (allocation_directory is None or not allocation_completion_sha256):
        raise ValueError("An approved allocation directory and expected completion SHA-256 are required")
    if not reallocated and (allocation_directory is not None or allocation_completion_sha256 is not None):
        raise ValueError("Mnfr and TargetAgeGroup must retain frozen V4 allocation")
    source = Path(split_directory).resolve()
    source_seal_sha = sha256(source / "completion.json")
    seal = _json(source / "completion.json")
    verify_protocol_splits(source, verify_exports=False)
    consumed = _snapshot(source, seal, ("train.csv", "validation.csv", "protected_final_test.csv",
                                        "model_config.json", "grouping_freeze.json"))
    sealed_config = ModelConfig.from_dict(_json(source / "model_config.json"))
    config = config or replace(sealed_config, sequence_length=256, local_files_only=True)
    if (config.sequence_length != 256 or not config.local_files_only
            or config.model_name != sealed_config.model_name or config.revision != sealed_config.revision
            or config.field_character_limits != sealed_config.field_character_limits):
        raise ValueError("MiniLM preparation requires the pinned encoder, existing character limits, local files and 256 tokens")
    columns = [*IDENTITY_FIELDS, target_column, *INPUT_FIELDS]
    pieces = []
    for role in ("train", "validation"):
        frame = _read(source, f"{role}.csv", columns)
        _validate_identity(frame, role)
        if not frame.split.eq(role).all():
            raise ValueError("Frozen development export contains another split role")
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).sort_values("record_id", kind="stable").reset_index(drop=True)
    frame["original_split"] = frame.split
    frame["is_labeled"] = ~frame[target_column].str.strip().str.casefold().isin(MISSING_TARGETS)
    protected = _read(source, "protected_final_test.csv", IDENTITY_FIELDS)
    _validate_identity(protected, "protected test")
    protected = protected.sort_values("record_id").reset_index(drop=True)
    if not protected.split.eq("test").all():
        raise ValueError("Protected population contains another split role")
    allocation_path, allocation_hashes, saved_names = None, {}, None
    if reallocated:
        allocation_path = Path(allocation_directory).resolve()
        frame, saved_names, allocation_hashes = _allocation(allocation_path, allocation_completion_sha256,
            target_column, frame, protected, source_seal_sha, consumed)
    if (frame.record_id.duplicated().any() or frame.groupby("group_id").split.nunique().gt(1).any()
            or set(frame.record_id) & set(protected.record_id) or set(frame.group_id) & set(protected.group_id)):
        raise ValueError("Development record/group isolation or protected-test isolation failed")
    support, _, pinned = _support(frame, target_column)
    names = tuple(label_names) if label_names is not None else saved_names if saved_names is not None else tuple(support.label)
    if (len(names) < 2 or any(not isinstance(name, str) or name.strip().casefold() in MISSING_TARGETS for name in names)
            or len(set(names)) != len(names) or set(names) != set(support.label)
            or saved_names is not None and names != saved_names):
        raise ValueError("Label vocabulary must preserve the complete target class ordering")
    supervised = frame.loc[frame.is_labeled].reset_index(drop=True)
    train = np.flatnonzero(supervised.split.eq("train").to_numpy())
    validation = np.flatnonzero(supervised.split.eq("validation").to_numpy())
    if not len(train) or not len(validation) or set(supervised.iloc[train][target_column]) != set(names):
        raise ValueError("Both supervised roles and training coverage of every declared class are required")
    if reallocated and not frame.loc[frame.group_id.isin(pinned), "split"].eq("train").all():
        raise ValueError("Allocation moved a required singleton group out of training")
    identity = {"target_column": target_column, "label_names": list(names), "model_config": config.to_dict(),
        "input_fields": list(INPUT_FIELDS), "category_character_limit": 512,
        "source_sha256": consumed, "allocation_completion_sha256": allocation_completion_sha256,
        "ordered_supervised_membership_sha256": _membership_hash(supervised)}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf8")).hexdigest()
    slug = target_column.casefold().replace("-", "_")
    model_identity = f"minilm-{slug}-category256-{identity_sha[:16]}"
    output = Path(output_root).resolve() / slug / model_identity
    if output.exists():
        raise FileExistsError("Use a new target model output directory")
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (source, allocation_path) if path is not None):
        raise ValueError("Model output must be outside frozen splits and approved allocations")
    if (sha256(source / "completion.json") != source_seal_sha
            or any(sha256(source / name) != digest for name, digest in consumed.items())
            or allocation_path is not None and any(sha256(allocation_path / name) != digest for name, digest in allocation_hashes.items())):
        raise ValueError("Preparation inputs changed while being read")
    audit = {**identity, "model_identity": model_identity, "model_identity_sha256": identity_sha,
        "source_completion_sha256": source_seal_sha, "allocation_sha256": allocation_hashes,
        "allocation_directory": str(allocation_path) if allocation_path is not None else None,
        "ordered_development_membership_sha256": _membership_hash(frame),
        "development_records": len(frame), "supervised_records": {"train": len(train), "validation": len(validation)},
        "missing_target_records": int((~frame.is_labeled).sum()), "label_names": list(names),
        "protected_test_records": len(protected), "test_csv_read_or_hashed": False,
        "ordered_supervised_label_sha256": _hash_items(supervised[target_column].tolist()),
        "legacy_target_alias": "Segment", "experimental_sequence_length": 256,
        "frozen_grouping_sequence_length": sealed_config.sequence_length,
        "final_test_representation_certified": False, "model_trained": False}
    return PreparedTargetMiniLM(target_column, frame, names, train, validation, protected,
        config, source / "tokenizer", model_identity, output, audit)


def tokenize_target_minilm(prepared: PreparedTargetMiniLM, *, tokenizer=None) -> TokenizedCorpus:
    """Reuse the experimental tokenizer path; load only a saved local tokenizer."""
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(prepared.tokenizer_directory)
    corpus = tokenize_category_corpus(prepared.legacy_frame, prepared.label_names,
        prepared.config, tokenizer=tokenizer, category_character_limit=512)
    prepared.trainer_arguments(corpus)  # Validate compatibility, without training.
    return corpus


def _verify_run_inputs(prepared):
    if not isinstance(prepared, PreparedTargetMiniLM) or prepared.target_column not in SUPPORTED_TARGETS:
        raise ValueError(f"Training target must be one of {SUPPORTED_TARGETS}; Platform is excluded")
    target = prepared.target_column
    reallocated = target in REALLOCATED_TARGETS
    if (prepared.audit["target_column"] != target
            or _membership_hash(prepared.frame) != prepared.audit["ordered_development_membership_sha256"]
            or list(prepared.label_names) != prepared.audit["label_names"]
            or prepared.config.to_dict() != prepared.audit["model_config"]
            or prepared.model_identity != prepared.audit["model_identity"]):
        raise ValueError(f"Prepared {target} target, vocabulary, membership or configuration changed")
    allocation_path = prepared.audit["allocation_directory"]
    if reallocated:
        if (not allocation_path or not prepared.audit["allocation_completion_sha256"]
                or prepared.audit["allocation_completion_sha256"] != prepared.audit["allocation_sha256"].get("completion.json")):
            raise ValueError("Target requires its pinned approved allocation")
        allocation_path = Path(allocation_path).resolve()
        if any(sha256(allocation_path / name) != digest for name, digest in prepared.audit["allocation_sha256"].items()):
            raise ValueError("Approved allocation changed after preparation")
    elif (allocation_path is not None or prepared.audit["allocation_completion_sha256"] is not None
            or not prepared.frame.split.equals(prepared.frame.original_split)):
        raise ValueError(f"{target} must retain frozen V4 membership")
    if (prepared.config.sequence_length != 256 or not prepared.config.local_files_only
            or prepared.config.precision == "fp16" and prepared.config.device == "cpu"):
        raise ValueError("MiniLM requires offline 256-token configuration; fp16 requires CUDA")
    source = prepared.tokenizer_directory.parent.resolve()
    output = prepared.output_directory.resolve()
    slug = target.casefold().replace("-", "_")
    expected_identity = f"minilm-{slug}-category256-{prepared.audit['model_identity_sha256'][:16]}"
    if (output.name != prepared.model_identity or output.parent.name != slug or prepared.model_identity != expected_identity
            or any(output.is_relative_to(path) or path.is_relative_to(output)
                   for path in (source, allocation_path) if path is not None)):
        raise ValueError("Target outputs must use their own prepared directory outside frozen splits and allocations")
    if output.exists():
        raise FileExistsError("Use a new target model output directory")
    if (sha256(source / "completion.json") != prepared.audit["source_completion_sha256"]
            or any(sha256(source / name) != digest for name, digest in prepared.audit["source_sha256"].items())):
        raise ValueError("Frozen target inputs changed after preparation")
    # Bind every saved tokenizer file before tokenization. No encoder is loaded.
    tokenizer_hashes = {path.name: sha256(path) for path in sorted(prepared.tokenizer_directory.iterdir()) if path.is_file()}
    freeze = _json(source / "grouping_freeze.json")
    expected = freeze.get("model_view_contract", {}).get("tokenizer_fingerprint")
    if expected is not None and tokenizer_hashes != expected:
        raise ValueError("Saved tokenizer differs from the frozen model contract")
    return source, output, freeze, tokenizer_hashes


def _training_readiness(config):
    """Check installed packages and byte-pinned cached encoder files offline."""
    versions, blockers = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, []
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
            blockers.append(f"Missing training dependency: {name}")
    resolved_device = None
    if versions["torch"] is not None:
        try:
            import torch
            available = torch.cuda.is_available()
            requested = ("cuda" if available else "cpu") if config.device == "auto" else config.device
            device = torch.device(requested)
            resolved_device = str(device)
            if device.type == "cuda" and (not available or device.index is not None and device.index >= torch.cuda.device_count()):
                blockers.append(f"Requested CUDA device is unavailable: {requested}")
            if config.precision == "fp16" and device.type != "cuda":
                blockers.append("fp16 requires a CUDA device")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            blockers.append(f"PyTorch runtime/device unavailable: {error}")
    try:
        # Existing helper calls snapshot_download(..., local_files_only=True)
        # and hashes config.json/model.safetensors. It never initializes a model.
        cached = pretrained_artifacts(config)
    except (ImportError, OSError, ValueError) as error:
        cached = {}
        blockers.append(f"Pinned encoder cache unavailable: {error}")
    return {"ready_for_training": not blockers, "blockers": blockers,
            "software_versions": versions, "pretrained_artifact_sha256": cached,
            "resolved_device": resolved_device,
            "encoder_initialized": False, "weight_downloads_allowed": False}


def _write_run_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def _annotate_target_manifest(path, metadata):
    value = _json(path)
    if (tuple(value["label_names"]) != tuple(metadata["label_names"])
            or value["train_record_ids_sha256"] != metadata["train_record_ids_sha256"]
            or value["validation_record_ids_sha256"] != metadata["validation_record_ids_sha256"]
            or json.dumps(value["config"], sort_keys=True) != json.dumps(metadata["config"], sort_keys=True)):
        raise ValueError("Legacy trainer metadata differs from the verified target inputs")
    for name in ("target", "label_names", "label_to_id", "model_identity", "model_identity_sha256",
                 "input_fields", "sequence_length", "representation", "split_identity",
                 "ordered_supervised_membership_sha256", "label_ids_sha256"):
        value[name] = metadata[name]
    value["legacy_trainer_target_alias"] = "Segment"
    value["legacy_trainer_default_input_fields"] = list(MODEL_FIELDS)
    value["final_test_representation_certified"] = False
    _write_run_json(path, value)
    return value


def run_target_minilm(prepared: PreparedTargetMiniLM, *, dry_run: bool = True,
                      training_authorized: bool = False, tokenizer=None, progress=print,
                      benchmark_steps: int | None = None) -> dict:
    """Verify any supported target; training requires explicit authorization.

    The default dry run tokenizes and checks metadata/cache readiness without
    initializing an encoder, downloading weights or writing output files. Actual
    training delegates once to train_transformer; no inference or new prediction
    export logic is implemented. Its existing training artifacts stay in the
    prepared target directory, and both legacy manifests receive correct metadata.
    Training requires both dry_run=False and training_authorized=True. The
    existing optional step-budget benchmark remains Mnfr-only.
    """
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    if type(training_authorized) is not bool:
        raise ValueError("training_authorized must be boolean")
    if not dry_run and not training_authorized:
        raise PermissionError("Actual training requires explicit authorization and training_authorized=True")
    if benchmark_steps is not None and (type(benchmark_steps) is not int or benchmark_steps < 1 or dry_run):
        raise ValueError("benchmark_steps requires a positive integer and dry_run=False")
    run_started = time.perf_counter()
    source, output, freeze, tokenizer_hashes = _verify_run_inputs(prepared)
    target = prepared.target_column
    if benchmark_steps is not None and target != "Mnfr":
        raise ValueError("Benchmark mode currently supports Mnfr only")
    corpus = tokenize_target_minilm(prepared, tokenizer=tokenizer)
    arguments = prepared.trainer_arguments(corpus)
    # Validate keyword compatibility without invoking the neural implementation.
    inspect.signature(train_transformer).bind(**arguments, progress=None)
    readiness = _training_readiness(prepared.config)
    frame = prepared.supervised_frame
    steps = math.ceil(len(prepared.train_indices) / prepared.config.batch_size) * prepared.config.epochs
    if benchmark_steps is not None and benchmark_steps > math.ceil(len(prepared.train_indices) / prepared.config.batch_size):
        raise ValueError("Benchmark must stop within the first epoch before validation")
    metadata = {
        "version": f"{target.casefold()}-minilm-orchestration-v1", "target": target, "dry_run": dry_run,
        "training_authorized": training_authorized,
        "status": "verified" if dry_run else "prepared", "model_trained": False,
        "model_identity": prepared.model_identity, "model_identity_sha256": prepared.audit["model_identity_sha256"],
        "output_directory": str(output), "label_names": list(prepared.label_names),
        "label_to_id": {name: i for i, name in enumerate(prepared.label_names)},
        "input_fields": list(INPUT_FIELDS), "sequence_length": 256, "config": prepared.config.to_dict(),
        "representation": {"name": "ProductCategory-first five-field experiment", "category_character_limit": 512,
            "field_character_limits": list(prepared.config.field_character_limits), "combined_token_budget": 256,
            "frozen_grouping_token_budget": prepared.audit["frozen_grouping_sequence_length"],
            "final_test_representation_certified": False},
        "split_identity": {"protocol_id": freeze.get("protocol_id"), "grouping_version": freeze.get("grouping_version"),
            "split_version": freeze.get("split_version"), "source_directory": str(source),
            "source_completion_sha256": prepared.audit["source_completion_sha256"],
            "grouping_freeze_sha256": prepared.audit["source_sha256"]["grouping_freeze.json"],
            "assignments_sha256": sha256(source / "assignments.csv"), "allocation": "frozen V4; no reallocation"},
        "ordered_supervised_membership_sha256": prepared.audit["ordered_supervised_membership_sha256"],
        "train_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in prepared.train_indices]),
        "validation_record_ids_sha256": _hash_items([corpus.record_ids[i] for i in prepared.validation_indices]),
        "train_group_ids_sha256": _hash_items(frame.iloc[prepared.train_indices].group_id.tolist()),
        "validation_group_ids_sha256": _hash_items(frame.iloc[prepared.validation_indices].group_id.tolist()),
        "training_rows": len(prepared.train_indices), "validation_rows": len(prepared.validation_indices),
        "train_class_counts": np.bincount(corpus.label_ids[prepared.train_indices], minlength=len(prepared.label_names)).tolist(),
        "validation_class_counts": np.bincount(corpus.label_ids[prepared.validation_indices], minlength=len(prepared.label_names)).tolist(),
        "label_ids_sha256": hashlib.sha256(corpus.label_ids.astype("<i8").tobytes()).hexdigest(),
        "text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
        "tokenizer_source": "injected" if tokenizer is not None else "saved local tokenizer",
        "tokenizer_sha256": tokenizer_hashes, "readiness": readiness,
        "schedule": {"fixed_final_epoch": prepared.config.epochs, "optimizer_steps": steps,
                     "warmup_steps": int(steps * prepared.config.warmup_fraction), "validation_selects_checkpoint": False},
        "checks": {"target_and_label_alignment": True, "trainer_signature_compatible": True,
            "frozen_group_isolation": True, "protected_test_records": len(prepared.protected_test),
            "test_csv_read_or_hashed": False, "weight_downloads_performed": False},
        "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in (
            "target_minilm_preparation.py", "segment_transformer.py", "segment_category_ablation.py")},
    }
    if target in REALLOCATED_TARGETS:
        metadata["split_identity"].update(
            allocation="approved target-specific development allocation",
            allocation_directory=prepared.audit["allocation_directory"],
            allocation_completion_sha256=prepared.audit["allocation_completion_sha256"],
            allocation_assignments_sha256=prepared.audit["allocation_sha256"]["assignments.csv"],
            development_assignments_sha256=prepared.audit["allocation_sha256"]["development_assignments.csv"],
            label_vocabulary_sha256=prepared.audit["allocation_sha256"]["class_coverage.csv"])
    # Catch source/tokenizer edits while tokenization or cache checks were running.
    _, _, _, after_tokenizer = _verify_run_inputs(prepared)
    if after_tokenizer != tokenizer_hashes:
        raise ValueError("Saved tokenizer changed during target run verification")
    if dry_run:
        return metadata
    if not readiness["ready_for_training"]:
        raise ValueError(f"{target} training prerequisites failed: " + "; ".join(readiness["blockers"]))
    if benchmark_steps is not None:
        metadata["run_kind"] = "benchmark"
        metadata["benchmark_requested_steps"] = benchmark_steps
        metadata["full_training_completed"] = False
        metadata["implementation_sha256"]["mnfr_minilm_benchmark.py"] = sha256(Path(__file__).with_name("mnfr_minilm_benchmark.py"))
    output.mkdir(parents=True, exist_ok=False)
    _write_run_json(output / "run_protocol.json", metadata)
    annotated = False
    benchmark = None

    def training_progress(message):
        nonlocal annotated
        path = output / "initialization.json"
        if not annotated and path.exists():
            _annotate_target_manifest(path, metadata)
            annotated = True
        if progress:
            progress(message)
        if benchmark is not None and message.startswith("Transformer initialized:"):
            benchmark.mark_initialized()

    # The only neural execution path. Dry runs return before reaching this call.
    if benchmark_steps is not None:
        from .mnfr_minilm_benchmark import BenchmarkComplete, MnfrBenchmark
        benchmark = MnfrBenchmark(corpus, prepared.train_indices, prepared.config, benchmark_steps, steps, progress)
        preflight_seconds = time.perf_counter() - run_started
        try:
            with benchmark:
                train_transformer(**arguments, progress=training_progress)
            raise RuntimeError("Trainer returned without reaching the benchmark step limit")
        except BenchmarkComplete:
            status, error = "benchmark_completed", None
        except Exception as failure:
            status, error = "benchmark_failed", {"type": type(failure).__name__, "message": str(failure)}
        report = {**metadata, "status": status, "model_trained": benchmark.optimizer_steps > 0,
            "full_training_completed": False, "benchmark": benchmark.report(),
            "orchestration_preflight_seconds": preflight_seconds, "error": error}
        if status == "benchmark_completed" and (benchmark.optimizer_steps != benchmark_steps
                or len(benchmark.rows) != benchmark_steps or not report["benchmark"]["all_losses_finite"]
                or not report["benchmark"]["all_batch_labels_verified"]):
            report.update(status="benchmark_failed", error={"type": "BenchmarkValidationError", "message": "Step/loss/label checks failed"})
            status, error = report["status"], report["error"]
        _write_run_json(output / "benchmark_report.json", report)
        pd.DataFrame(benchmark.rows).to_csv(output / "benchmark_steps.csv", index=False)
        if status != "benchmark_completed":
            raise RuntimeError(f"Mnfr benchmark failed; see {output / 'benchmark_report.json'}: {error}")
        return report
    _, manifest = train_transformer(**arguments, progress=training_progress)
    _annotate_target_manifest(output / "initialization.json", metadata)
    manifest = _annotate_target_manifest(output / "training_manifest.json", metadata)
    manifest.update(orchestration_status="completed", model_trained=True)
    manifest["artifacts"] = {path.name: sha256(path) for path in output.iterdir()
        if path.is_file() and path.name != "training_manifest.json"}
    _write_run_json(output / "training_manifest.json", manifest)
    return {**metadata, "status": "completed", "model_trained": True}


def run_mnfr_minilm(prepared: PreparedTargetMiniLM, *, dry_run: bool = True,
                    training_authorized: bool = False, tokenizer=None, progress=print,
                    benchmark_steps: int | None = None) -> dict:
    """Compatibility entry point; all orchestration lives in run_target_minilm."""
    if not isinstance(prepared, PreparedTargetMiniLM) or prepared.target_column != "Mnfr":
        raise ValueError("This compatibility entry point supports Mnfr only")
    return run_target_minilm(prepared, dry_run=dry_run, training_authorized=training_authorized,
        tokenizer=tokenizer, progress=progress, benchmark_steps=benchmark_steps)
