"""Amazon-only MiniLM training with frozen retailer-held-out validation.

Uses every eligible Amazon training row and the entire frozen validation pool.
Amazon validation is an in-domain control; five other retailers are held out
from training. The original groups, split exports and final test stay sealed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .features import sha256
from .segment_category_ablation import (
    ARMS, CATEGORY, DEFAULT_SEQUENCE_LENGTH, _annotate_training, _check_sources,
    _json, _read_completed_arm, _source_hashes as _ablation_source_hashes,
    _write_json, input_audit, load_category_development, load_tokenizer,
    pretrained_artifacts, tokenize_category_corpus,
)
from .segment_transformer import MODEL_FIELDS, ModelConfig, _hash_items, _plain_text, tokenize_corpus, train_transformer
from .segment_validation_analysis import bootstrap_predictions, compare_paired_predictions, evaluate_predictions

VERSION = "segment-retailer-holdout-v2"
DEFAULT_ARMS = ("with_category",)
DEFAULT_HELD_OUT = ("Walmart", "Walgreens", "CVS", "Target", "Kroger")


def select_retailer_development(development, train_retailer="Amazon", held_out_retailers=DEFAULT_HELD_OUT):
    """Filter training by source retailer without resampling or moving groups."""
    names = (str(train_retailer).strip(), *(str(value).strip() for value in held_out_retailers))
    keys = [name.casefold() for name in names]
    if any(not key for key in keys) or len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("Training and held-out retailers must be nonblank and distinct")
    frame = development.frame.copy()
    if not {"record_id", "group_id", "split", "Segment", "Retailer", CATEGORY}.issubset(frame):
        raise ValueError("Retailer selection requires the guarded development columns")
    if (frame.record_id.isna().any() or frame.record_id.duplicated().any()
            or frame.record_id.astype(str).str.strip().eq("").any()
            or frame.group_id.isna().any() or frame.group_id.astype(str).str.strip().eq("").any()
            or not frame.split.isin(("train", "validation")).all()):
        raise ValueError("Development identities or split roles are invalid")
    if (set(frame.record_id) & development.protected_record_ids
            or set(frame.group_id) & development.protected_group_ids):
        raise ValueError("Protected final-test identity or group appears in development")
    source_train = frame.loc[frame.split.eq("train")]
    source_validation = frame.loc[frame.split.eq("validation")]
    if set(source_train.group_id) & set(source_validation.group_id):
        raise ValueError("Training and validation group overlap")
    if frame.Retailer.isna().any():
        raise ValueError("Missing retailer cannot enter a retailer holdout")
    retailer_keys = frame.Retailer.astype(str).str.strip().str.casefold()
    if not retailer_keys.isin(keys).all():
        raise ValueError("Missing or unexpected retailer in development")
    frame["Retailer"] = retailer_keys.map(dict(zip(keys, names)))
    mask = frame.split.eq("train") & frame.Retailer.eq(names[0])
    train = frame.loc[mask]
    validation = frame.loc[frame.split.eq("validation")]
    if train.empty or train.Segment.nunique() < 2:
        raise ValueError("Amazon-only training needs records from at least two Segment classes")
    if not set(frame.Segment) <= set(development.label_names):
        raise ValueError("Segment outside the fixed declared label space")
    if not set(names[1:]) <= set(validation.Retailer):
        raise ValueError("A held-out retailer has no validation records")
    selected = frame.loc[mask | frame.split.eq("validation")].sort_values("record_id", kind="stable").reset_index(drop=True)
    audit = {"training_retailer": names[0], "held_out_retailers": list(names[1:]),
        "source_training_records": len(source_train), "selected_training_records": len(train),
        "excluded_training_records": len(source_train) - len(train), "validation_records": len(validation),
        "training_class_counts": train.Segment.value_counts().sort_index().to_dict(),
        "missing_training_classes": [label for label in development.label_names if label not in set(train.Segment)],
        "validation_retailer_counts": validation.Retailer.value_counts().sort_index().to_dict(),
        "no_downsampling": True, "record_and_group_isolation_passed": True,
        "selection_policy": "All Amazon frozen-training records; all frozen-validation records. Other training retailers excluded."}
    return replace(development, frame=selected,
        train_indices=np.flatnonzero(selected.split.eq("train").to_numpy()),
        validation_indices=np.flatnonzero(selected.split.eq("validation").to_numpy()),
        audit={**development.audit, "retailer_selection": audit})


def _source_hashes(split_dir, reference_dir=None, arms=DEFAULT_ARMS):
    sources = _ablation_source_hashes(split_dir)
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    near = split_dir / "independent_near_pairs.csv"
    if near.exists():
        sources[str(near)] = sha256(near)
    if reference_dir is not None:
        for name in ("completion.json", "protocol.json", "protocol_freeze.json"):
            sources[str(reference_dir / name)] = sha256(reference_dir / name)
        for arm in arms:
            for name in ("training_manifest.json", "predictions.csv"):
                sources[str(reference_dir / arm / name)] = sha256(reference_dir / arm / name)
    return sources


def _validate_predictions(predicted, manifest, development, protocol, arm):
    validation = development.frame.iloc[development.validation_indices]
    if set(predicted.record_id) & development.protected_record_ids:
        raise ValueError("Protected final-test record was predicted")
    if predicted.record_id.duplicated().any() or set(predicted.record_id) != set(validation.record_id):
        raise ValueError("Prediction population differs from the whole frozen validation pool")
    if predicted.set_index("record_id").true_label.to_dict() != validation.set_index("record_id").Segment.to_dict():
        raise ValueError("Prediction truth differs from frozen validation targets")
    if (manifest["train_record_ids_sha256"] != protocol["train_record_ids_sha256"]
            or manifest["validation_record_ids_sha256"] != protocol["validation_record_ids_sha256"]):
        raise ValueError("Trainer changed controlled experiment membership")
    expected_fields = list(MODEL_FIELDS) if arm == "baseline" else [CATEGORY, *MODEL_FIELDS]
    if (ModelConfig.from_dict(manifest["config"]) != ModelConfig.from_dict(protocol["model_config"])
            or manifest["label_names"] != protocol["label_names"]
            or manifest["input_fields"] != expected_fields
            or any(manifest[key] != protocol["tokenization"][arm][key] for key in ("text_sha256", "token_sha256"))):
        raise ValueError("Completed arm configuration or representation differs from the frozen protocol")


def _load_reference(reference_dir, config, development, category_character_limit=512, arms=DEFAULT_ARMS):
    if reference_dir is None:
        return {}, {}
    protocol, completion = _json(reference_dir / "protocol.json"), _json(reference_dir / "completion.json")
    if (completion["protocol_sha256"] != sha256(reference_dir / "protocol.json")
            or _json(reference_dir / "protocol_freeze.json")["protocol_sha256"] != completion["protocol_sha256"]):
        raise ValueError("Mixed-retailer reference protocol seal changed")
    if ModelConfig.from_dict(protocol["model_config"]) != config or tuple(protocol["label_names"]) != development.label_names:
        raise ValueError("Mixed-retailer reference model settings or label order differ")
    representation = protocol["representation"]
    if (representation["category_position"] != "prefix"
            or representation["category_character_limit"] != category_character_limit
            or representation["combined_token_limit"] != config.sequence_length
            or representation["baseline_fields"] != list(MODEL_FIELDS)
            or representation["category_fields"] != [CATEGORY, *MODEL_FIELDS]
            or protocol["development_audit"]["grouping_freeze_sha256"] != development.audit["grouping_freeze_sha256"]):
        raise ValueError("Mixed-retailer reference representation or frozen groups differ")
    if any(Path(name).name == "test.csv" for name in protocol["source_sha256"]):
        raise ValueError("Reference source checks cannot consume final-test records")
    _check_sources(protocol["source_sha256"])
    validation = development.frame.iloc[development.validation_indices]
    predictions, manifests = {}, {}
    for arm in arms:
        for name in ("training_manifest.json", "predictions.csv"):
            key = f"{arm}/{name}"
            if sha256(reference_dir / key) != completion["artifact_sha256"][key]:
                raise ValueError("Mixed-retailer reference artifact changed")
        predicted, manifest = _read_completed_arm(reference_dir / arm)
        if (predicted.record_id.duplicated().any() or set(predicted.record_id) != set(validation.record_id)
                or predicted.set_index("record_id").true_label.to_dict() != validation.set_index("record_id").Segment.to_dict()):
            raise ValueError("Mixed-retailer reference validation membership or truth differs")
        predictions[arm], manifests[arm] = predicted, manifest
    return predictions, manifests


def _near_audit(split_dir, development):
    path = split_dir / "independent_near_pairs.csv"
    if not path.exists():
        return {"available": False, "scope": "No frozen independent candidate pool available"}, set()
    if sha256(path) != _json(split_dir / "completion.json")["artifact_sha256"][path.name]:
        raise ValueError("Frozen independent audit changed")
    # Identity-only audit evidence; never inspect product names or test features.
    pairs = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["left_record_id", "right_record_id"])
    train = set(development.frame.iloc[development.train_indices].record_id)
    validation = development.frame.iloc[development.validation_indices]
    val = set(validation.record_id)
    crossing = set()
    for left, right in pairs.itertuples(index=False, name=None):
        if left in train and right in val:
            crossing.add((left, right))
        elif right in train and left in val:
            crossing.add((right, left))
    neighbors = {right for _, right in crossing}
    external = set(validation.loc[validation.Retailer.isin(development.audit["retailer_selection"]["held_out_retailers"]), "record_id"])
    return {"available": True, "candidate_crossings_all_validation": len(crossing),
        "candidate_crossings_held_out_retailers": sum(right in external for _, right in crossing),
        "validation_rows_with_candidate_training_neighbor": len(neighbors),
        "scope": "Existing independent name-similarity candidate pool only; suspicious pairs are not all confirmed leakage."}, neighbors


def _cohorts(validation, selection):
    return {"all_validation": set(validation.record_id),
        "amazon_control": set(validation.loc[validation.Retailer.eq(selection["training_retailer"]), "record_id"]),
        "retailer_holdout": set(validation.loc[validation.Retailer.isin(selection["held_out_retailers"]), "record_id"]),
        **{f"retailer:{name}": set(validation.loc[validation.Retailer.eq(name), "record_id"])
           for name in selection["held_out_retailers"]}}


def run_experiment(split_dir, output_dir, config=None, *, reference_dir=None, bootstrap_samples=1000,
                   category_character_limit=512, progress=print, resume=False, prepare_only=False, arms=DEFAULT_ARMS):
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    reference_dir = Path(reference_dir).resolve() if reference_dir is not None else None
    if output_dir.exists() and not resume:
        raise FileExistsError("Use a new retailer-holdout output directory")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("Select one or both distinct supported input arms")
    arms = tuple(arm for arm in ARMS if arm in arms)
    config = config or replace(ModelConfig.from_dict(_json(split_dir / "model_config.json")), sequence_length=DEFAULT_SEQUENCE_LENGTH)
    started = perf_counter()
    development = select_retailer_development(load_category_development(split_dir, config))
    frame = development.frame
    validation = frame.iloc[development.validation_indices].copy()
    selection = development.audit["retailer_selection"]
    hashes = _source_hashes(split_dir, reference_dir, arms)
    hashes.update(pretrained_artifacts(config))
    reference, reference_manifests = _load_reference(reference_dir, config, development, category_character_limit, arms)
    tokenizer = load_tokenizer(split_dir / "tokenizer")
    corpora = {arm: (tokenize_corpus(frame, development.label_names, config, tokenizer=tokenizer)
        if arm == "baseline" else tokenize_category_corpus(frame, development.label_names, config,
            tokenizer=tokenizer, category_character_limit=category_character_limit)) for arm in arms}
    if (any(corpus.record_ids != tuple(frame.record_id) for corpus in corpora.values())
            or any(not np.array_equal(corpus.label_ids, corpora[arms[0]].label_ids) for corpus in corpora.values())):
        raise ValueError("Representations changed controlled record or label ordering")
    near, neighbors = _near_audit(split_dir, development)
    categories = frame.ProductCategory.map(_plain_text)
    training_categories = set(categories.iloc[development.train_indices]) - {""}
    category_by_id = dict(zip(frame.record_id, categories))
    cohorts = _cohorts(validation, selection)
    coverage = [{"cohort": name, "records": len(ids),
        "seen_in_amazon_training": sum(category_by_id[record] in training_categories for record in ids),
        "unseen_in_amazon_training": sum(bool(category_by_id[record]) and category_by_id[record] not in training_categories for record in ids),
        "missing": sum(not category_by_id[record] for record in ids)} for name, ids in cohorts.items()]
    representation = {"baseline_fields": list(MODEL_FIELDS), "category_fields": [CATEGORY, *MODEL_FIELDS],
        "category_position": "prefix", "category_character_limit": category_character_limit,
        "combined_token_limit": config.sequence_length, "sealed_grouping_token_limit": development.audit["sealed_sequence_length"],
        "budget_caveat": "Category prefix can displace product text; comparison measures the net representation effect.",
        "final_test_representation_certified": False}
    protocol = {"version": VERSION, "arms": list(arms), "model_config": config.to_dict(), "label_names": list(development.label_names),
        "representation": representation, "source_sha256": hashes, "development_audit": development.audit,
        "training_rows": len(development.train_indices), "validation_rows": len(development.validation_indices),
        "train_record_ids_sha256": _hash_items(frame.iloc[development.train_indices].record_id.tolist()),
        "validation_record_ids_sha256": _hash_items(validation.record_id.tolist()),
        "bootstrap_samples": bootstrap_samples, "reference_directory": str(reference_dir) if reference_dir is not None else None,
        "category_coverage": coverage, "independent_near_audit": near,
        "checkpoint_policy": "Fixed final epoch; validation is observational only.",
        "selection_policy": selection["selection_policy"],
        "controlled": "Full Amazon training subset and whole frozen validation; reference model settings and initial weights match.",
        "tokenization": {arm: {"text_sha256": corpus.text_sha256, "token_sha256": corpus.token_sha256,
            "input_audit": input_audit(corpus, frame)} for arm, corpus in corpora.items()},
        "caveats": ["Retailer shift also changes assortment, wording, class mix and training volume; not a pure causal taxonomy experiment.",
            "Whole-breadcrumb novelty does not imply novel category words or concepts.",
            "One training seed; bootstrap intervals cover validation-group sampling only.",
            "Held-out validation was previously used in development; this side experiment is not a new final test."],
        "final_test_read_or_scored": False}
    protocol = json.loads(json.dumps(protocol, ensure_ascii=False, allow_nan=False))
    _check_sources(hashes)
    if resume:
        existing = _json(output_dir / "protocol.json")
        if {key: value for key, value in existing.items() if key != "created_utc"} != protocol:
            raise ValueError("Resume corpus, configuration or source differs")
        freeze = _json(output_dir / "protocol_freeze.json")
        if freeze["protocol_sha256"] != sha256(output_dir / "protocol.json"):
            raise ValueError("Experiment protocol seal changed")
        if any(sha256(output_dir / name) != expected for name, expected in freeze["prepared_artifact_sha256"].items()):
            raise ValueError("Prepared retailer assignment or category coverage changed")
        protocol = existing
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        protocol["created_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "protocol.json", protocol)
        assignments = frame[["record_id", "group_id", "split", "Retailer"]].rename(columns={"split": "original_split", "Retailer": "retailer"})
        assignments["experiment_role"] = np.where(frame.split.eq("train"), "train",
            np.where(frame.Retailer.eq(selection["training_retailer"]), "amazon_control", "retailer_holdout"))
        assignments.to_csv(output_dir / "retailer_assignments.csv", index=False, lineterminator="\n")
        pd.DataFrame(coverage).to_csv(output_dir / "category_coverage.csv", index=False, lineterminator="\n")
        _write_json(output_dir / "protocol_freeze.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
            "frozen_before_training": True, "prepared_artifact_sha256": {name: sha256(output_dir / name)
                for name in ("retailer_assignments.csv", "category_coverage.csv")}})
    progress(f"Prepared retailer holdout: {len(development.train_indices):,} Amazon train / {len(cohorts['amazon_control']):,} Amazon control / {len(cohorts['retailer_holdout']):,} external validation")
    for arm in arms:
        progress(f"{arm} identical-input crossings: {protocol['tokenization'][arm]['input_audit']['crossing_pairs']}")
    if prepare_only:
        return protocol
    predictions, manifests = {}, {}
    for arm in arms:
        _check_sources(hashes)
        directory = output_dir / arm
        if resume and (directory / "training_manifest.json").exists():
            predicted, manifest = _read_completed_arm(directory)
        else:
            def arm_progress(message):
                initialization = directory / "initialization.json"
                if initialization.exists():
                    initial = _json(initialization)["initial_state_sha256"]
                    if manifests and initial != manifests[arms[0]]["initial_state_sha256"]:
                        raise ValueError("Both arms must start from identical weights")
                    if reference_manifests and initial != reference_manifests[arm]["initial_state_sha256"]:
                        raise ValueError("Reference and holdout initial weights differ")
                progress(message)
            progress(f"Training {arm} on Amazon only")
            predicted, manifest = train_transformer(corpora[arm], development.train_indices,
                development.validation_indices, development.label_names, config, directory, progress=arm_progress)
            _annotate_training(directory, manifest, MODEL_FIELDS if arm == "baseline" else (CATEGORY, *MODEL_FIELDS), representation)
            predicted, manifest = _read_completed_arm(directory)
        _validate_predictions(predicted, manifest, development, protocol, arm)
        if manifests and manifest["initial_state_sha256"] != manifests[arms[0]]["initial_state_sha256"]:
            raise ValueError("Both arms must start from identical weights")
        if reference_manifests and manifest["initial_state_sha256"] != reference_manifests[arm]["initial_state_sha256"]:
            raise ValueError("Reference and holdout initial weights differ")
        predictions[arm], manifests[arm] = predicted, manifest
    metrics, classes, slices, intervals, uncertainty, comparisons = [], [], [], [], [], []
    regimes = {"amazon_only": predictions, **({"mixed_retailer_reference": reference} if reference else {})}
    groups = validation[["record_id", "group_id"]]
    for regime, regime_arms in regimes.items():
        for arm, predicted in regime_arms.items():
            for name, ids in cohorts.items():
                subset = predicted.loc[predicted.record_id.isin(ids)]
                evaluated = evaluate_predictions(subset, development.label_names)
                tags = {"training_regime": regime, "arm": arm, "cohort": name}
                metrics.append({**tags, **evaluated["metrics"]})
                classes.append(evaluated["per_class"].assign(**tags))
                if name in ("all_validation", "amazon_control", "retailer_holdout"):
                    clustered = subset.merge(groups, on="record_id", validate="one_to_one")
                    boot = bootstrap_predictions(clustered, development.label_names, n_bootstrap=bootstrap_samples, seed=config.seed)
                    uncertainty.append(boot["intervals"].assign(**tags))
                    for bucket in ("seen_in_amazon_training", "unseen_in_amazon_training", "missing"):
                        eligible = {record for record in ids if (category_by_id[record] in training_categories if bucket == "seen_in_amazon_training"
                            else bool(category_by_id[record]) and category_by_id[record] not in training_categories if bucket == "unseen_in_amazon_training"
                            else not category_by_id[record])}
                        slices.append({**tags, "dimension": "breadcrumb", "value": bucket,
                            **evaluate_predictions(subset.loc[subset.record_id.isin(eligible)], development.label_names)["metrics"]})
                    if near["available"]:
                        for present in (False, True):
                            subset_ids = {record for record in ids if (record in neighbors) == present}
                            slices.append({**tags, "dimension": "independent_near_neighbor", "value": present,
                                **evaluate_predictions(subset.loc[subset.record_id.isin(subset_ids)], development.label_names)["metrics"]})
    for name in ("all_validation", "amazon_control", "retailer_holdout"):
        ids = cohorts[name]
        pairings = ([("category_effect_amazon_only", predictions["with_category"], predictions["baseline"])]
                    if set(ARMS) <= set(predictions) else [])
        if reference:
            pairings += [(f"holdout_training_vs_reference:{arm}", predictions[arm], reference[arm]) for arm in arms]
        for label, left, right in pairings:
            paired = compare_paired_predictions(left.loc[left.record_id.isin(ids)], right.loc[right.record_id.isin(ids)],
                development.label_names, groups, n_bootstrap=bootstrap_samples, seed=config.seed)
            intervals.append(paired["intervals"].assign(comparison=label, cohort=name))
            comparisons.append({**{key: value for key, value in paired.items()
                if key not in ("intervals", "record_ids", "comparison")},
                "comparison_method": paired["comparison"], "comparison": label, "cohort": name})
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output_dir / "validation_metrics.csv", index=False, lineterminator="\n")
    pd.concat(classes, ignore_index=True).to_csv(output_dir / "per_class_comparison.csv", index=False, lineterminator="\n")
    pd.DataFrame(slices).to_csv(output_dir / "category_slices.csv", index=False, lineterminator="\n")
    pd.concat(uncertainty, ignore_index=True).to_csv(output_dir / "metric_intervals.csv", index=False, lineterminator="\n")
    interval_table = (pd.concat(intervals, ignore_index=True) if intervals else
        pd.DataFrame(columns=["metric", "estimate", "ci_lower", "ci_upper", "comparison", "cohort"]))
    interval_table.to_csv(output_dir / "paired_differences.csv", index=False, lineterminator="\n")
    _write_json(output_dir / "paired_comparisons.json", comparisons)
    _check_sources(hashes)
    summary = {"version": VERSION, "protocol_sha256": sha256(output_dir / "protocol.json"),
        "selection": selection, "metrics": metrics,
        "paired_differences": interval_table.astype(object).where(pd.notna(interval_table), None).to_dict("records"),
        "input_audit": {arm: protocol['tokenization'][arm]['input_audit'] for arm in arms},
        "independent_near_audit": near, "same_initial_weights": True,
        "initial_state_sha256": manifests[arms[0]]["initial_state_sha256"], "arms": list(arms),
        "same_training_validation_membership": True, "amazon_only_training": True,
        "no_downsampling": True, "final_test_read_or_scored": False,
        "split_artifacts_and_previous_implementations_unchanged": True,
        "category_coverage": coverage, "representation": representation, "caveats": protocol['caveats'],
        "runtime_seconds": perf_counter() - started}
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "completion.json", {"protocol_sha256": sha256(output_dir / "protocol.json"),
        "artifact_sha256": {path.relative_to(output_dir).as_posix(): sha256(path)
            for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "completion.json"}})
    progress(metric_table.loc[metric_table.cohort.isin(("amazon_control", "retailer_holdout")),
        ["training_regime", "arm", "cohort", "records", "accuracy", "macro_f1", "balanced_accuracy"]].to_string(index=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Completed mixed-retailer 256-token category ablation")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--category-character-limit", type=int, default=512)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS),
                        help="Input representations to train (default: with_category only)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = ModelConfig.from_dict(_json(args.model_config or args.split_dir / "model_config.json"))
    if args.sequence_length is not None:
        config = replace(config, sequence_length=args.sequence_length)
    elif args.model_config is None:
        config = replace(config, sequence_length=DEFAULT_SEQUENCE_LENGTH)
    if args.device:
        config = replace(config, device=args.device)
    run_experiment(args.split_dir, args.output_dir, config, reference_dir=args.reference_dir,
        bootstrap_samples=args.bootstrap_samples, category_character_limit=args.category_character_limit,
        resume=args.resume, prepare_only=args.prepare_only, arms=args.arms,
        progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
