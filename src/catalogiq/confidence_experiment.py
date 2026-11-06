"""Reproducible confidence experiment on explicitly selected validation exports."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd

from .confidence_calibration import (
    GROUP_SALT, ROLES, accepted_summary, bootstrap_accuracy, calibration_metrics,
    essential_information_reasons, fit_temperature, partition_context, per_class_table,
    temperature_scale, threshold_grid, threshold_table,
)
from .features import sha256
from .prediction_contract import (
    CONTRACT_BLOB, Evidence, IDENTITY, REGISTRY_NAME, RUNS, frozen_identity_context,
    read_json, validate_manifest, write_json,
)


DEFAULT_SOURCE = "data/processed/predictions_contract_v2_20261031"
DEFAULT_OUTPUT = "data/processed/confidence_review_v3_20261103"
TARGET_ACCURACY = .99
ALPHA = .05


def csv_output(frame, path):
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def load_development_features(root, evidence, protected):
    """Only the two existing development files; never discover partitions by glob."""
    source = root / "data/processed/split_integration_reproduced_20261010"
    seal = evidence.json(source / "completion.json")
    fields = ["ProductCategory", "ProductName", "ProductBrand", "ProductDescription", "ProductContents"]
    frames = []
    for role in ("train", "validation"):
        path = evidence.capture(source / f"{role}.csv")
        if evidence.hashes[path] != seal["artifact_sha256"][path.name]:
            raise ValueError("Development features differ from their existing seal")
        part = evidence.csv(path, [*IDENTITY, *fields])
        if not part.split.eq(role).all():
            raise ValueError("Development feature split changed")
        frames.append(part)
    features = pd.concat(frames, ignore_index=True)
    if (features.record_id.duplicated().any()
            or set(features.record_id) & set(protected.record_id)
            or set(features.group_id) & set(protected.group_id)):
        raise ValueError("Protected identity/group or duplicate in development features")
    return features.set_index("record_id")


def load_run(directory, evidence, contexts, identities, protected, protocol,
             *, handoff_seal=None, expected_manifest_sha256=None):
    directory = Path(directory).resolve()
    manifest_path = evidence.capture(directory / "manifest.json")
    if expected_manifest_sha256 and evidence.hashes[manifest_path] != expected_manifest_sha256:
        raise ValueError("Candidate manifest differs from the supplied SHA-256")
    if handoff_seal is not None:
        expected = handoff_seal["files_sha256"][f"{directory.name}/manifest.json"]
        if evidence.hashes[manifest_path] != expected:
            raise ValueError("Aligned handoff manifest seal mismatch")
    manifest = evidence.json(manifest_path)
    target = manifest["target"]
    if target not in contexts:
        raise ValueError("Target has no approved development context")
    if manifest["partition"] != "validation" or manifest["probabilities_calibrated"]:
        raise ValueError("Uncalibrated validation probabilities required")
    if manifest["split_allocation_identity"] != identities[target]:
        raise ValueError("Run allocation identity differs from approved target allocation")
    # Read only the minimum approved handoff artifacts. Do not load checkpoints.
    for name in ("predictions.csv", "validation_context.csv", "label_mapping.json", REGISTRY_NAME):
        path = evidence.capture(directory / name)
        if evidence.hashes[path] != manifest["files_sha256"][name]:
            raise ValueError(f"Run file seal mismatch: {name}")
        if handoff_seal is not None and evidence.hashes[path] != handoff_seal["files_sha256"][f"{directory.name}/{name}"]:
            raise ValueError(f"Aligned handoff file seal mismatch: {name}")
    columns = [f"probability_{i}" for i in range(len(manifest["label_mapping"]))]
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"record_id": str, "true_label": str,
                              "predicted_label": str}, keep_default_na=False, encoding="utf-8-sig")
    context = evidence.csv(directory / "validation_context.csv", [*IDENTITY, "true_label"])
    expected = contexts[target].sort_values("record_id").reset_index(drop=True)
    if not context.sort_values("record_id").reset_index(drop=True).equals(expected):
        raise ValueError("Run context differs from independently verified approved population")
    if evidence.json(directory / "label_mapping.json") != manifest["label_mapping"]:
        raise ValueError("Mapping artifact disagrees with manifest")
    registry = evidence.csv(directory / REGISTRY_NAME, IDENTITY[:2])
    # protected was independently bound to the full original assignments and
    # committed V4 digests by frozen_identity_context, before loading any run.
    if not registry.sort_values("record_id").reset_index(drop=True).equals(
            protected[IDENTITY[:2]].sort_values("record_id").reset_index(drop=True)):
        raise ValueError("Run protected registry differs from independently verified original V4")
    checks = validate_manifest(manifest, predictions, context, protected)
    frame = predictions.merge(context[["record_id", "group_id", "split"]], on="record_id",
                              how="left", validate="one_to_one", suffixes=("", "_context"))
    # validate_manifest checks arriving extra group/split columns before this merge.
    labels = [manifest["label_mapping"][column] for column in columns]
    return manifest, frame, labels, columns, checks


def model_gates(run_directory, frame, features, evidence):
    """Measured gates use product features, never validation target labels."""
    part = features.loc[frame.record_id]
    if part.group_id.tolist() != frame.group_id.tolist():
        raise ValueError("Development features disagree with frozen prediction groups")
    reasons = essential_information_reasons(part)
    cohorts = np.repeat("neural", len(frame)).astype(object)
    if (Path(run_directory) / "model.json").is_file():
        # The explicit lookup JSON is counts/normalization metadata, not an
        # executable model. No estimator is fitted or used for new inference.
        from .target_minilm_experiment import brand_key
        model = evidence.json(Path(run_directory) / "model.json")
        manifest = evidence.json(Path(run_directory) / "manifest.json")
        if evidence.hashes[(Path(run_directory) / "model.json").resolve()] != manifest["files_sha256"]["model.json"]:
            raise ValueError("Saved lookup metadata seal mismatch")
        keys = part.ProductBrand.map(brand_key)
        missing = keys.isin(model["missing_normalized_values"]).to_numpy()
        seen = keys.isin(model["key_counts"]).to_numpy() & ~missing
        cohorts = np.where(missing, "missing", np.where(seen, "seen", "unseen"))
        for i, cohort in enumerate(cohorts):
            if cohort != "seen":
                reasons[i].append("lookup_without_matching_training_key")
    eligible = np.asarray([not r for r in reasons])
    return eligible, reasons, cohorts


def choose_policy(selection_table):
    """Choose using selection only. Never fall back using the policy-audit labels."""
    supported = selection_table.loc[selection_table.cluster_accuracy_lower.ge(TARGET_ACCURACY)]
    status = "selection_supported" if len(supported) else "exploratory_only"
    pool = supported
    if pool.empty:
        pool = selection_table.loc[selection_table.accepted_accuracy.ge(TARGET_ACCURACY)]
    if pool.empty:
        pool = selection_table.loc[selection_table.accepted_rows.gt(0)]
        if pool.empty:
            return None, "no_nonempty_candidate"
        pool = pool.sort_values(["accepted_accuracy", "coverage", "selection_nll", "threshold"],
                                ascending=[False, False, True, True], kind="stable")
    else:
        pool = pool.sort_values(["coverage", "selection_nll", "threshold", "variant"],
                                ascending=[False, True, True, True], kind="stable")
    chosen = pool.iloc[0]
    return {"variant": chosen.variant, "threshold": float(chosen.threshold)}, status


def evaluate_model(frame, probabilities, labels, eligible, cohorts, directory, *,
                   family_size=6, provisional=False, bootstrap_samples=2000):
    """Reusable evaluator; expects independently validated uncalibrated exports."""
    directory = Path(directory)
    directory.mkdir()
    context = partition_context(frame[[*IDENTITY, "true_label"]])
    roles = context.experiment_role.to_numpy()
    fit = roles == ROLES[0]
    select = roles == ROLES[1]
    audit = roles == ROLES[2]
    fit_result = fit_temperature(probabilities[fit], frame.true_label[fit], labels)
    scaled = temperature_scale(probabilities, fit_result["temperature"])
    raw = temperature_scale(probabilities, 1.)
    if not np.array_equal(raw.argmax(axis=1), scaled.argmax(axis=1)):
        raise ValueError("Temperature scaling changed a predicted class")
    variants = {"raw": raw, "temperature": scaled}
    metrics, reliabilities, tables = [], [], []
    grid = threshold_grid()
    # Simultaneous one-sided bounds over every searched policy in all models.
    selection_alpha = ALPHA / (family_size * len(variants) * len(grid))
    audit_alpha = ALPHA / family_size
    selection_tables = []
    for variant, p in variants.items():
        selection_nll = None
        for role, mask in zip(ROLES, (fit, select, audit)):
            metric, bins = calibration_metrics(p[mask], frame.true_label[mask], labels)
            metrics.append({"variant": variant, "subset": role, **metric})
            reliabilities.extend({"variant": variant, "subset": role, **row} for row in bins)
            if role == ROLES[1]:
                selection_nll = metric["nll_clipped_1e15"]
        for role, mask, alpha in ((ROLES[1], select, selection_alpha), (ROLES[2], audit, audit_alpha)):
            correct = np.asarray(labels)[p[mask].argmax(axis=1)] == frame.true_label[mask].to_numpy()
            table = threshold_table(p[mask].max(axis=1), correct, frame.group_id[mask],
                                    eligible[mask], grid, alpha)
            table["variant"], table["subset"] = variant, role
            table["selection_nll"] = selection_nll
            tables.append(table)
            if role == ROLES[1]:
                selection_tables.append(table)
    # Audit tables are descriptive. choose_policy receives selection rows only.
    chosen, selection_status = choose_policy(pd.concat(selection_tables, ignore_index=True))
    audit_result, selected_bootstrap = None, None
    accepted = np.zeros(len(frame), bool)
    if chosen:
        p = variants[chosen["variant"]]
        correct = np.asarray(labels)[p.argmax(axis=1)] == frame.true_label.to_numpy()
        accepted = eligible & (p.max(axis=1) >= chosen["threshold"])
        audit_result = accepted_summary(accepted[audit], correct[audit], frame.group_id[audit], audit_alpha)
        selected_bootstrap = bootstrap_accuracy(accepted[audit], correct[audit], frame.group_id[audit],
                                                samples=bootstrap_samples)
        csv_output(per_class_table(frame.true_label[audit], frame.predicted_label[audit], accepted[audit], labels),
                   directory / "selected_policy_per_class.csv")
    else:
        p = raw
    supported = bool(chosen and selection_status == "selection_supported"
                     and audit_result["cluster_accuracy_lower"] is not None
                     and audit_result["cluster_accuracy_lower"] >= TARGET_ACCURACY)
    policy = {"selected_candidate": chosen, "selection_status": selection_status,
              "audit": audit_result, "audit_group_bootstrap_descriptive": selected_bootstrap,
              "development_supports_99_percent": supported,
              "provisional": provisional, "enabled": False,
              "recommendation": "manual_review_until_fresh_validation" if not supported or provisional
                                else "candidate_for_fresh_validation",
              "operational_threshold": None,
              "operational_auto_per_100": 0, "operational_review_per_100": 100,
              "mandatory_review_conditions": ["disabled_or_unapproved_policy", "provisional_model",
                  "below_selected_threshold", "missing_essential_product_text", "existing_quarantine",
                  "invalid_probabilities", "unsupported_classification", "conflicting_probability_classification",
                  "lookup_missing_or_unseen_key", "invalid_or_incomplete_approved_relationship"],
              "relationships": {"approved_rules_loaded": 0, "status": "pending_owner_approved_mapping",
                  "fixed_segment_subsegment_whitelist": "forbidden_by_docs_decisions"},
              "selection_alpha": selection_alpha, "audit_alpha": audit_alpha}
    csv_output(pd.DataFrame(metrics), directory / "calibration_metrics.csv")
    csv_output(pd.DataFrame(reliabilities), directory / "reliability.csv")
    all_tables = pd.concat(tables, ignore_index=True)
    csv_output(all_tables, directory / "thresholds.csv")
    # Full per-class coverage/errors at every threshold, including absent/OOV classes.
    class_tables = []
    for variant, candidate_p in variants.items():
        for role, mask in ((ROLES[1], select), (ROLES[2], audit)):
            for t in grid:
                a = eligible[mask] & (candidate_p[mask].max(axis=1) >= t)
                table = per_class_table(frame.true_label[mask], frame.predicted_label[mask], a, labels)
                table["variant"], table["subset"], table["threshold"] = variant, role, t
                class_tables.append(table)
    csv_output(pd.DataFrame.from_records([row for table in class_tables for row in table.to_dict("records")]),
               directory / "thresholds_per_class.csv")
    assignments = context[["record_id", "group_id", "split", "experiment_role"]].copy()
    assignments["eligible_measured_gates"] = eligible
    assignments["cohort"] = cohorts
    # Contains validation identities only; no product text.
    csv_output(assignments, directory / "experiment_assignments.csv")
    cohort_rows = []
    for cohort in sorted(set(cohorts)):
        mask = audit & (cohorts == cohort)
        if mask.any():
            for variant, candidate_p in variants.items():
                metric, _ = calibration_metrics(candidate_p[mask], frame.true_label[mask], labels)
                cohort_rows.append({"cohort": cohort, "variant": variant, **metric})
    csv_output(pd.DataFrame(cohort_rows), directory / "audit_cohorts.csv")
    write_json(directory / "temperature.json", fit_result)
    write_json(directory / "policy.json", policy)
    write_json(directory / "subset_counts.json", {role: {"rows": int((roles == role).sum()),
               "groups": int(context.loc[roles == role, "group_id"].nunique())} for role in ROLES})
    return policy, metrics


def run_experiment(root, source_root, output_root, *, bootstrap_samples=2000, plot_cache=None,
                   candidates=()):
    root, source_root, output_root = Path(root).resolve(), Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing experiment directory: {output_root}")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Outputs and source handoff must be separate")
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples required")
    evidence = Evidence()
    protocol, protected, contexts, identities, registry_checks = frozen_identity_context(root, evidence)
    handoff_seal = evidence.json(source_root / "completion.json")
    if handoff_seal["contract_git_blob"] != CONTRACT_BLOB:
        raise ValueError("Unexpected aligned predictions contract")
    contract = root / ".cache/predictions-contract-maria.md"
    if contract.is_file():
        evidence.capture(contract)
    for name in ("docs/decisions.md", "docs/findings/03_modeling/16_prediction_contract_alignment_20261031.md"):
        evidence.capture(root / name)
    features = load_development_features(root, evidence, protected)
    specs = [(name, source_root/name, None) for name in RUNS]
    # Explicit opt-in future candidate; never discover Max's files recursively.
    for path, digest in candidates:
        path = Path(path).resolve()
        if path.name in {name for name, _, _ in specs}:
            raise ValueError("Additional candidate name must be unique")
        specs.append((path.name, path, digest))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # A normal mkdir inherits workspace access on Windows. tempfile.mkdtemp's
    # owner-only ACL can make published artifacts unreadable by the user/UI.
    staging = output_root.parent / f"confidence-staging-{uuid4().hex}"
    staging.mkdir()
    results, metric_rows, policies = [], [], {}
    try:
        # Freeze the protocol to disk before evaluating any model.
        experiment_protocol = {
            "format": "catalogiq-confidence-review-v1", "development_only": True,
            "group_salt": GROUP_SALT, "roles": list(ROLES), "target_accuracy": TARGET_ACCURACY,
            "family_alpha": ALPHA, "models": [name for name, _, _ in specs],
            "thresholds": threshold_grid().tolist(), "temperature_bounds": [.05, 20.],
            "selection": "Maximum coverage with simultaneous bounded-cluster lower accuracy >= .99; otherwise exploratory candidate only",
            "tie_break": "Lower selection NLL, then lower threshold, then variant name",
            "audit": "One fixed candidate/model, Bonferroni alpha=.05/model_count; no reselection from audit",
            "interval": "Independent bounded-group KL concentration; M=max accepted group size; conditional on observed accepted volumes",
            "bootstrap": "2000 whole-group resamples (or explicit CLI count), descriptive two-sided 95%; never used to authorize acceptance",
            "essential_fields": "Existing policy: at least one of name, description, contents; blank/null missing",
            "lookup_gate": "Missing/unseen normalized ProductBrand requires review",
            "hierarchy": "No approved hard maps found; fixed Segment/Sub-Segment whitelist explicitly prohibited",
            "future_guarantee": False,
        }
        write_json(staging / "protocol.json", experiment_protocol)
        for name, directory, digest in specs:
            manifest, frame, labels, columns, checks = load_run(directory, evidence, contexts, identities,
                protected, protocol, handoff_seal=handoff_seal if digest is None else None,
                expected_manifest_sha256=digest)
            eligible, reasons, cohorts = model_gates(directory, frame, features, evidence)
            policy, metrics = evaluate_model(frame, frame[columns].to_numpy(float), labels, eligible, cohorts,
                staging/name, family_size=len(specs), provisional=manifest["target"] == "Sub-Segment",
                bootstrap_samples=bootstrap_samples)
            policy.update({"target": manifest["target"], "model": manifest["model"],
                           "source_manifest_sha256": evidence.hashes[(directory/"manifest.json").resolve()],
                           "split_allocation_target": manifest["split_allocation_target"]})
            write_json(staging/name/"policy.json", policy)
            policies[name] = policy
            metric_rows.extend({"run": name, **m} for m in metrics)
            audit = policy["audit"] or {}
            candidate = policy["selected_candidate"] or {}
            results.append({"run": name, "target": manifest["target"], "provisional": policy["provisional"],
                "candidate_variant": candidate.get("variant"), "candidate_threshold": candidate.get("threshold"),
                "temperature": read_json(staging/name/"temperature.json")["temperature"],
                "audit_rows": audit.get("rows"), "audit_accepted": audit.get("accepted_rows"),
                "candidate_auto_per_100": 100*audit.get("coverage", 0),
                "candidate_review_per_100": 100*audit.get("review_fraction", 1),
                "candidate_wrong_per_100_incoming": 100*audit.get("accepted_errors", 0)/audit["rows"] if audit else None,
                "audit_accepted_accuracy": audit.get("accepted_accuracy"),
                "audit_accepted_errors": audit.get("accepted_errors"),
                "audit_cluster_lower_accuracy": audit.get("cluster_accuracy_lower"),
                "selection_status": policy["selection_status"],
                "development_supports_99_percent": policy["development_supports_99_percent"],
                "operational_auto_per_100": 0, "operational_review_per_100": 100})
            csv_output(pd.DataFrame({"record_id": frame.record_id, "review_reasons": [";".join(r) for r in reasons]}),
                       staging/name/"measured_gate_reasons.csv")
            print(f"Evaluated {name}: {len(frame):,} validation rows; {policy['selection_status']}", flush=True)
        csv_output(pd.DataFrame(results), staging / "results.csv")
        csv_output(pd.DataFrame(metric_rows), staging / "calibration_comparison.csv")
        from .confidence_plots import plot_experiment
        plot_experiment(staging, [name for name, _, _ in specs], plot_cache=plot_cache)
        evidence.reverify()
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        sources = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): h for p, h in evidence.hashes.items()}
        code_sources = [Path(__file__), Path(__file__).with_name("confidence_calibration.py"),
                        Path(__file__).with_name("confidence_plots.py"), Path(__file__).with_name("cli.py")]
        write_json(staging / "summary.json", {
            "format": "catalogiq-confidence-review-v1", "git_head": revision,
            "python": platform.python_version(), "libraries": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
            "matplotlib": __import__("matplotlib").__version__,
            "bootstrap_samples": bootstrap_samples, "source_hash_normalization": "raw_bytes",
            "source_sha256": sources, "experiment_code_sha256": {p.name: sha256(p) for p in code_sources},
            "source_files_reverified_unchanged": len(sources), "protected_registry_checks": registry_checks,
            "test_csv_opened_or_hashed": False, "protected_labels_accessed": False,
            "classifiers_fitted_or_modified": False, "predictions_modified": False,
            "audit_tables_are_descriptive_not_selection": True,
            "policies": policies,
        })
        write_json(staging / "completion.json", {"files_sha256": {
            str(p.relative_to(staging)).replace("\\", "/"): sha256(p)
            for p in sorted(staging.rglob("*")) if p.is_file()}})
        staging.rename(output_root)
    except BaseException:
        # This freshly created staging directory is the only deletion target.
        if not staging.resolve().is_relative_to(output_root.parent) or not staging.name.startswith("confidence-staging-"):
            raise RuntimeError("Refusing cleanup outside experiment staging directory")
        shutil.rmtree(staging)
        raise
    return {"runs": len(results), "output_root": str(output_root), "results": results}
