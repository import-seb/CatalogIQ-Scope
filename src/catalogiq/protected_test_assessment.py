"""Freeze/preflight without test access; an explicitly authorized assessment only.

Development loaders remain validation-only. This separate boundary never trains,
fits calibration, selects models/thresholds, or publishes classifications.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .confidence_calibration import calibration_metrics, essential_information_reasons, temperature_scale, threshold_table
from .features import sha256
from .prediction_contract import Evidence, MISSING, _safe_member, frozen_identity_context, read_json, write_json
from .segment_validation_analysis import bootstrap_predictions

FORMAT = "catalogiq-protected-test-assessment-v1"
SOURCE = "data/processed/split_integration_reproduced_20261010"
EXPORTS = "data/processed/predictions_contract_v2_20261031"
CALIBRATION = "data/processed/confidence_review_v3_20261103"
RUNS = {"segment": ("Segment", "selected"), "mnfr": ("Mnfr", "selected"),
        "targetagegroup": ("TargetAgeGroup", "selected"),
        "brand_lookup_probabilistic_v1": ("Brand", "selected"),
        "sub_segment": ("Sub-Segment", "provisional")}
FIELDS = ["ProductCategory", "ProductName", "ProductBrand", "ProductDescription", "ProductContents"]
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "tokenizers", "torch", "transformers", "threadpoolctl")
CODE = ("protected_test_assessment.py", "confidence_calibration.py", "clustered_accuracy.py",
        "prediction_contract.py", "split_protocol.py", "segment_validation_analysis.py",
        "segment_transformer.py", "segment_category_ablation.py", "brand_probabilistic_lookup.py",
        "target_minilm_experiment.py", "minilm_prediction_export.py", "features.py", "structural.py",
        "accuracy_reassessment.py", "cli.py", "paths.py", "__init__.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_pin(root, path, *, normalization="raw_bytes"):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "Pinned files must be inside the checkout")
    require(path.name.casefold() != "test.csv", "Freezing/preflight cannot open or hash test.csv")
    require(normalization in ("raw_bytes", "crlf_to_lf"), "Unknown source hash normalization")
    content = path.read_bytes()
    if normalization == "crlf_to_lf":
        content = content.replace(b"\r\n", b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(content).hexdigest(),
            "normalization": normalization}


def freeze_protocol(root, output):
    """Pin existing evidence and artifacts, reading only test identity metadata."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Refusing to overwrite a frozen assessment protocol")
    evidence = Evidence()
    split_protocol, protected, _, _, _ = frozen_identity_context(root, evidence)
    seal = evidence.json(root/SOURCE/"completion.json")
    # This hash is copied from the existing split seal. test.csv is not opened.
    test_hash = seal["artifact_sha256"]["test.csv"]
    for name in ("train.csv", "validation.csv"):
        path = evidence.capture(root/SOURCE/name)
        require(evidence.hashes[path] == seal["artifact_sha256"][name], "Development source seal mismatch")
    specs, pins = [], {}

    def capture(path, expected=None):
        pin = file_pin(root, path)
        require(expected is None or pin["sha256"] == expected, f"Artifact changed: {path}")
        pins[pin["path"]] = pin
        return pin["path"]

    handoff = evidence.json(root/EXPORTS/"completion.json")
    for run, (target, status) in RUNS.items():
        directory = root/EXPORTS/run
        manifest = evidence.json(directory/"manifest.json")
        require(manifest["target"] == target, "Wrong target in selected manifest")
        capture(directory/"manifest.json", handoff["files_sha256"][f"{run}/manifest.json"])
        labels = [manifest["label_mapping"][f"probability_{i}"] for i in range(len(manifest["label_mapping"]))]
        temperature_path = root/CALIBRATION/run/"temperature.json"
        policy_path = root/CALIBRATION/run/"policy.json"
        cal_seal = evidence.json(root/CALIBRATION/"completion.json")
        capture(temperature_path, cal_seal["files_sha256"][f"{run}/temperature.json"])
        capture(policy_path, cal_seal["files_sha256"][f"{run}/policy.json"])
        temperature, policy = read_json(temperature_path), read_json(policy_path)
        require(not policy["enabled"] and policy["operational_threshold"] is None, "Operational policy must stay disabled")
        require(policy["source_manifest_sha256"] == sha256(directory/"manifest.json"), "Calibration/model identity mismatch")
        spec = {"run": run, "target": target, "selection_status": status,
                "model_identity": manifest["model_identity"], "label_mapping": manifest["label_mapping"],
                "input_fields": manifest["input_fields"], "training_allocation": manifest["split_allocation_target"],
                "temperature": temperature["temperature"], "diagnostic_candidate": policy["selected_candidate"]}
        if run == "brand_lookup_probabilistic_v1":
            spec.update(kind="lookup", model=capture(directory/"model.json", manifest["files_sha256"]["model.json"]))
            require(read_json(directory/"model.json")["label_names"] == labels, "Lookup vocabulary mismatch")
        else:
            checkpoint = Path(manifest["checkpoint"]["source"])
            spec.update(kind="minilm", checkpoint=capture(checkpoint, manifest["checkpoint"]["sha256"]),
                        state_sha256=manifest["checkpoint"]["state_sha256"], config=manifest["config"])
            parent = checkpoint.parent
            spec["model_directory"] = parent.relative_to(root).as_posix()
            for name in ("model_config.json", "encoder_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
                path = parent/name
                expected = manifest["source_input_sha256"].get(str(path))
                require(expected is not None, f"Selected metadata does not seal {name}")
                capture(path, expected)
            require(spec["config"]["sequence_length"] == 256 and spec["config"]["precision"] == "fp32", "Unexpected neural representation")
        specs.append(spec)
    for path, expected in evidence.hashes.items():
        capture(path, expected)
    evidence.reverify()
    protocol = {
        "format": FORMAT, "status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Final held-out model assessment against supplied labels; reviewer-assisted product, no automatic production publication",
        "test": {"path": f"{SOURCE}/test.csv", "expected_sha256_from_existing_seal": test_hash,
                 "identity_registry": f"{SOURCE}/protected_final_test.csv",
                 "assignments": f"{SOURCE}/assignments.csv", "records": len(protected),
                 "groups": int(protected.group_id.nunique()), "identity": split_protocol["expected"]["splits"]["test"]},
        "models": specs, "artifacts": list(pins.values()),
        "code": [file_pin(root, path, normalization="crlf_to_lf") for path in
                 [*[Path(__file__).parent/name for name in CODE], root/"scripts/assess_protected_test.py", root/"pyproject.toml"]],
        "runtime": {"device": "cpu", "precision": "fp32", "threads": 2, "batch_size": 64, "seed": 42,
                    "python": platform.python_version(), "libraries": {name: version(name) for name in LIBRARIES}},
        "metrics": {"primary": "macro_f1_model_vocabulary_union_truth", "zero_division": 0,
                    "secondary": ["accuracy", "macro_precision", "macro_recall", "weighted_precision", "weighted_recall", "weighted_f1", "per_class_support", "confusion_matrix"],
                    "population": "All protected records; score each target's nonmissing supplied labels; no confidence/class/outlier row exclusions",
                    "missing_truth": sorted(MISSING), "oov_truth": "Retained as errors and included in union denominator",
                    "intervals": {"method": "whole_frozen_group_percentile_bootstrap", "samples": 2000, "seed": 20261106, "confidence": .95,
                                  "scope": "Descriptive two-sided per-target accuracy and union macro-F1; not simultaneous production guarantees"},
                    "calibration": "Raw and fixed saved-temperature ECE/Brier/NLL; no fitting; OOV/zero true probability gives infinite NLL flag",
                    "thresholds": [.8, .9, .95, .99], "threshold_role": "Descriptive review prioritization only, never retune or enable approval",
                    "relationship_rates": {"status": "unmeasured_no_approved_machine_readable_registry", "valid_pair_rate": None},
                    "platform": {"status": "deferred", "model": None, "metrics": None, "prediction_coverage": 0},
                    "human_corrected_accuracy": None, "numerical_99_percent_release_requirement": None},
        "representation_check": "Before reporting, audit exact 256-token input equality against train/validation without changing groups or dropping test rows; report any crossings as a limitation",
        "controls": {"test_opened_or_hashed_during_freeze": False, "execution_authorized_by_freeze": False,
                     "new_models_or_threshold_selection": False, "production_approval": False,
                     "attempt_policy": "One planned assessment. Create a durable receipt before test access; preserve failed attempts. Technical retries use identical pins and disclose prior exposure."},
        "limitations": ["Supplied labels have not been certified as independent ground truth",
                        "Frozen grouping used four fields/128 tokens; selected neural inputs use five fields/256 tokens",
                        "Known grouping misses, taxonomy ambiguity and historical development reuse remain",
                        "No approved complete relationship map; relationship-validity requirements remain unresolved",
                        "No final corrected-export, reviewer-time or live retailer quality claim follows from model test metrics"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(protocol, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"protocol": str(output), "protocol_sha256": sha256(output), "test_opened_or_hashed": False}


def preflight(root, protocol_path, expected_sha256):
    """No test access or model deserialization, including on malformed protocols."""
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    require(sha256(protocol_path) == expected_sha256, "Frozen protocol checksum mismatch")
    protocol = read_json(protocol_path)
    require(protocol["format"] == FORMAT and protocol["status"] == "frozen", "Unsupported assessment protocol")
    require(protocol["test"]["path"] == f"{SOURCE}/test.csv", "Unexpected protected-test path")
    require({key: protocol["runtime"][key] for key in ("device", "precision", "threads", "batch_size", "seed")}
            == {"device": "cpu", "precision": "fp32", "threads": 2, "batch_size": 64, "seed": 42}, "Frozen inference runtime changed")
    require([(s["run"], s["target"], s["selection_status"]) for s in protocol["models"]]
            == [(run, *values) for run, values in RUNS.items()], "Model routing changed")
    for pin in [*protocol["artifacts"], *protocol["code"]]:
        path = _safe_member(root, pin["path"])
        actual = file_pin(root, path, normalization=pin["normalization"])
        require(actual == pin, f"Pinned input/code changed: {pin['path']}")
    registry = pd.read_csv(_safe_member(root, protocol["test"]["identity_registry"]), dtype=str, usecols=["record_id", "group_id"])
    require(len(registry) == protocol["test"]["records"] and registry.group_id.nunique() == protocol["test"]["groups"], "Protected registry count mismatch")
    from .split_protocol import canonical_digest
    require(canonical_digest(registry, ["record_id"]) == protocol["test"]["identity"]["record_ids_sha256"], "Protected identities changed")
    require(canonical_digest(registry.assign(split="test"), ["record_id", "group_id", "split"])
            == protocol["test"]["identity"]["assignment_sha256"], "Protected groups changed")
    require(platform.python_version() == protocol["runtime"]["python"], "Python version differs from frozen runtime")
    require({name: version(name) for name in LIBRARIES} == protocol["runtime"]["libraries"], "Library versions differ from frozen runtime")
    return protocol, registry, {"status": "prepared_not_executed", "protocol_sha256": expected_sha256,
        "artifact_and_code_pins_verified": len(protocol["artifacts"]) + len(protocol["code"]),
        "protected_records": len(registry), "protected_groups": int(registry.group_id.nunique()),
        "test_opened_or_hashed": False, "models_loaded": False,
        "pending": ["Explicit execution authorization", "256-token test input-equality audit during authorized assessment",
                    "Approved relationship registry remains unavailable; relationship rates will be null"]}


def score_target(truth, predicted, probabilities, labels, groups, *, intervals, temperature, eligible, thresholds=None):
    """Metrics reuse existing evaluators; true OOV labels remain errors."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(len(truth) == len(predicted) == len(probabilities) == len(groups), "Scoring rows do not align")
    matrix = np.asarray(probabilities, float)
    calibration, reliability = calibration_metrics(matrix, truth, labels)
    require(np.array_equal(np.asarray(labels)[matrix.argmax(axis=1)], predicted), "Probability class order mismatch")
    classes = [*labels, *sorted(set(truth) - set(labels))]
    frame = pd.DataFrame({"record_id": [str(i) for i in range(len(truth))], "group_id": groups,
                          "true_label": truth, "predicted_label": predicted})
    report = classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"label": label, **report[label]} for label in classes])
    bootstrap = bootstrap_predictions(frame, classes, n_bootstrap=intervals["samples"], seed=intervals["seed"], confidence=intervals["confidence"])
    correct = truth == predicted
    summaries, queues = [], []
    for name, p in (("raw", matrix), ("saved_temperature", temperature_scale(matrix, temperature))):
        metric, _ = calibration_metrics(p, truth, labels)
        summaries.append({"variant": name, **metric})
        table = threshold_table(p.max(axis=1), correct, groups, eligible, thresholds=thresholds if thresholds is not None else [.8, .9, .95, .99])
        # Do not revive the superseded maximum-family-size certification bound
        # or the row-independent reference interval as final qualification.
        table = table.drop(columns=["cluster_accuracy_lower", "iid_row_accuracy_lower_reference_only", "alpha"])
        table["variant"] = name
        queues.append(table)
    return {"metrics": {"records": len(truth), "accuracy": float(correct.mean()),
                        "macro_f1": report["macro avg"]["f1-score"],
                        "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
                        "weighted": report["weighted avg"], "macro_class_set": "model_vocabulary_union_truth",
                        "oov_truth_rows": int((~np.isin(truth, labels)).sum()), "calibration_raw": calibration},
            "per_class": per_class, "confusion": pd.DataFrame(confusion_matrix(truth, predicted, labels=classes), index=classes, columns=classes),
            "intervals": bootstrap["intervals"], "reliability_raw": pd.DataFrame(reliability),
            "calibration": pd.DataFrame(summaries), "thresholds": pd.concat(queues, ignore_index=True)}


def predict_frozen(root, spec, features, runtime, development):
    """Narrow protected-assessment adapter; original formatter/classifier math."""
    if spec["kind"] == "lookup":
        from .brand_probabilistic_lookup import predict_frequency_lookup
        matrix, cohorts, _ = predict_frequency_lookup(read_json(root/spec["model"]), features.ProductBrand)
        return matrix, cohorts["seen_feature_values"], {"method": "lookup_key_familiarity", "seen_keys": int(cohorts["seen_feature_values"].sum())}
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .segment_category_ablation import build_category_texts
    from .segment_transformer import ModelConfig, _make_classifier, _state_sha256
    directory = root/spec["model_directory"]
    config = ModelConfig.from_dict(spec["config"])
    tokenizer = AutoTokenizer.from_pretrained(directory/"tokenizer", local_files_only=True)
    state = torch.load(root/spec["checkpoint"], map_location="cpu", weights_only=True)
    labels = [spec["label_mapping"][f"probability_{i}"] for i in range(len(spec["label_mapping"]))]
    require(list(state["label_names"]) == labels, "Checkpoint label order differs from protocol")
    require(_state_sha256(state["state_dict"]) == spec["state_sha256"], "Checkpoint state differs")
    values = read_json(directory/"encoder_config.json")
    encoder = AutoModel.from_config(AutoConfig.for_model(values.pop("model_type"), **values))
    model = _make_classifier(torch, encoder, len(labels), config.dropout)
    model.load_state_dict(state["state_dict"], strict=True)
    require(_state_sha256(model.state_dict()) == spec["state_sha256"], "Reconstructed classifier differs from pinned state")
    model.to("cpu").eval()
    del state

    def signatures(frame):
        encoded = tokenizer(build_category_texts(frame, config, category_character_limit=512), truncation=True, max_length=256, padding=False)
        return [hashlib.sha256(json.dumps([encoded[name][i] for name in sorted(encoded)], separators=(",", ":")).encode()).hexdigest()
                for i in range(len(frame))]

    development_signatures = set(signatures(development))
    crossing = sum(value in development_signatures for value in signatures(features))
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(features), runtime["batch_size"]):
            texts = build_category_texts(features.iloc[start:start+runtime["batch_size"]], config, category_character_limit=512)
            batch = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            probabilities.append(torch.softmax(model(**batch).float(), dim=1).numpy())
    return np.concatenate(probabilities), np.ones(len(features), bool), {"method": "exact_256_token_inputs", "test_rows_matching_development_input": crossing,
         "interpretation": "No rows removed; any crossing limits independence claims and is not repaired using test results"}


def assess(root, protocol_path, expected_sha256, output, *, authorized=False, authorization_id=None):
    """The only test-opening boundary. Never called by freeze or preflight."""
    if not authorized or not isinstance(authorization_id, str) or not authorization_id.strip():
        raise PermissionError("Explicit protected-test assessment authorization and its reference are required")
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Preserve previous assessment attempts; choose a new output directory")
    require(output.parent == root/"data/processed", "Assessment output must be a new direct data/processed child")
    protocol, registry, checks = preflight(root, protocol_path, expected_sha256)
    output.mkdir()
    receipt = {"protocol_sha256": expected_sha256, "authorization_reference": authorization_id,
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": "started_before_test_access"}
    write_json(output/"assessment_receipt.json", receipt)
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(protocol["runtime"]["threads"])
        torch.manual_seed(protocol["runtime"]["seed"])
        torch.use_deterministic_algorithms(True)
        path = root/protocol["test"]["path"]
        require(sha256(path) == protocol["test"]["expected_sha256_from_existing_seal"], "Protected test bytes differ from sealed export")
        columns = ["record_id", "group_id", "split", *FIELDS, *[s["target"] for s in protocol["models"]]]
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
        require(not frame.record_id.duplicated().any() and frame.split.eq("test").all(), "Invalid protected test rows")
        actual = frame[["record_id", "group_id"]].sort_values("record_id").reset_index(drop=True)
        require(actual.equals(registry.sort_values("record_id").reset_index(drop=True)), "Protected test membership/group mismatch")
        frame = frame.sort_values("record_id").reset_index(drop=True)
        features = frame[["record_id", *FIELDS]]
        development = pd.concat([pd.read_csv(root/SOURCE/f"{role}.csv", dtype=str, keep_default_na=False, usecols=FIELDS)
                                 for role in ("train", "validation")], ignore_index=True)
        report = {"protocol_sha256": expected_sha256, "test_records": len(frame), "models": [],
                  "runtime": {**protocol["runtime"], "platform": platform.platform(), "cpu": platform.processor()},
                  "platform": protocol["metrics"]["platform"], "relationship_rates": protocol["metrics"]["relationship_rates"],
                  "human_corrected_accuracy": None, "automatic_acceptance_enabled": False}
        with threadpool_limits(limits=protocol["runtime"]["threads"]):
            for spec in protocol["models"]:
                directory = output/spec["run"]
                directory.mkdir()
                started = time.perf_counter()
                matrix, matching_key, representation = predict_frozen(root, spec, features, protocol["runtime"], development)
                elapsed = time.perf_counter()-started
                labels = [spec["label_mapping"][f"probability_{i}"] for i in range(len(spec["label_mapping"]))]
                require(matrix.shape == (len(frame), len(labels)), "Incomplete predictions")
                # Validate all probabilities, even for rows with no evaluation truth.
                from .confidence_calibration import validate_probabilities
                matrix = validate_probabilities(matrix)
                predicted = np.asarray(labels)[matrix.argmax(axis=1)]
                eligible = matching_key & np.asarray([not r for r in essential_information_reasons(features)])
                truth = frame[spec["target"]]
                labeled = ~truth.str.strip().str.casefold().isin(protocol["metrics"]["missing_truth"]).to_numpy()
                predictions = frame[["record_id", "group_id", "split"]].copy()
                predictions["true_label"] = truth
                predictions["predicted_label"], predictions["confidence"] = predicted, matrix.max(axis=1)
                predictions["review_required"] = True
                predictions = pd.concat([predictions, pd.DataFrame(matrix, columns=[f"probability_{i}" for i in range(len(labels))])], axis=1)
                predictions.to_csv(directory/"predictions.csv", index=False, lineterminator="\n")
                entry = {"target": spec["target"], "model_identity": spec["model_identity"], "selection_status": spec["selection_status"],
                         "predicted_rows": len(frame), "labeled_rows": int(labeled.sum()), "missing_truth_rows": int((~labeled).sum()),
                         "prediction_coverage": 1., "load_audit_and_prediction_seconds": elapsed, "representation_audit": representation,
                         "timing_scope": "Model load, exact-input audit and batch inference; not request latency"}
                if labeled.any():
                    cutoffs = protocol["metrics"].get("thresholds", [.8, .9, .95, .99])
                    saved_cutoff = spec.get("diagnostic_candidate", {}).get("threshold")
                    if saved_cutoff is not None:
                        cutoffs = sorted(set([*cutoffs, saved_cutoff]))
                    scored = score_target(truth.to_numpy()[labeled], predicted[labeled], matrix[labeled], labels, frame.group_id.to_numpy()[labeled],
                                          intervals=protocol["metrics"]["intervals"], temperature=spec["temperature"], eligible=eligible[labeled], thresholds=cutoffs)
                    entry["metrics"] = scored.pop("metrics")
                    for name, table in scored.items():
                        table.to_csv(directory/f"{name}.csv", index=name == "confusion", lineterminator="\n")
                else:
                    entry["metrics"] = None
                write_json(directory/"manifest.json", {**entry, "partition": "test", "protocol_sha256": expected_sha256,
                    "label_mapping": spec["label_mapping"], "probabilities_calibrated": False,
                    "operational_threshold": None, "training_allocation": spec["training_allocation"]})
                report["models"].append(entry)
        require(sha256(path) == protocol["test"]["expected_sha256_from_existing_seal"], "Protected source changed during assessment")
        preflight(root, protocol_path, expected_sha256)
        write_json(output/"summary.json", report)
        receipt["status"] = "completed"
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"assessment_receipt.json", receipt)
        from .accuracy_reassessment import seal_output
        seal_output(output)
    except BaseException as error:
        receipt.update(status="failed_preserve_attempt", error_type=type(error).__name__)
        write_json(output/"assessment_receipt.json", receipt)
        raise
    return report
