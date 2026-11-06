"""Discover and freeze product-specific rules before a historical audit replay."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .accuracy_reassessment import (
    OLD_RUN, SOURCE_RUN, fresh_staging, cleanup_staging, verify_seal, seal_output,
)
from .confidence_calibration import group_role, temperature_scale
from .confidence_experiment import load_development_features, load_run
from .features import sha256
from .prediction_contract import Evidence, IDENTITY, MISSING, frozen_identity_context, write_json
from .product_review_rules import (
    COMMON_TRAINING_ROWS, GATES, CLASS_MODES, MIN_DISCOVERY_CLASS_GROUPS, MIN_POLICY_GROUPS,
    MIN_POLICY_EFFECTIVE_ROWS, SEARCHED_RULES, FIXED_THRESHOLDS, apply_rule, check_phase_isolation, choose_rule, diagnostic_tables,
    discover_rules, require_role, score_registry, summarize_policy, uncommon_class_table,
)
from .structural import StructuralConfig, missing
from .target_minilm_experiment import MISSING_FEATURE, brand_key


DISCOVERY_RUN = "data/processed/product_rule_discovery_v1_20261103"
AUDIT_RUN = "data/processed/product_rule_audit_v1_20261103"
FLOORS = FIXED_THRESHOLDS
FEATURE_DEFINITIONS = {
    "description_present": "Existing structural blank/null missingness; presence is not verified completeness",
    "category_present": "Retailer ProductCategory supplied and nonmissing; no inferred hierarchy rule",
    "brand_seen": "Normalized ProductBrand occurs in this target's approved labeled training records",
    "brand_training_groups": "Distinct frozen groups containing that normalized key in this target's labeled training",
    "brand_key_unambiguous": "Training key has exactly one observed target label; not a taxonomy assertion",
    "common_class": f"Predicted class has at least {COMMON_TRAINING_ROWS} approved labeled training rows",
    "class_whitelist": "Class screen learned only from the old calibration-fit subset; no true label required at arrival",
}


def code_hashes():
    directory = Path(__file__).parent
    return {name: sha256(directory/name) for name in (
        "product_review_rules.py", "product_rule_experiment.py", "product_rule_plots.py", "clustered_accuracy.py")}


def training_reference(root, model, labels, features, evidence):
    """Describe approved training inputs; do not fit/refit any classifier or lookup."""
    source = root/"data/processed/split_integration_reproduced_20261010"
    if model == "segment":
        metadata = evidence.csv(source/"train.csv", [*IDENTITY, "Segment"])
        target = "Segment"
    else:
        directory = root/"data/processed/brand_development_allocation_v1_20261029"
        metadata = evidence.csv(directory/"development_assignments.csv", [*IDENTITY, "Brand"])
        metadata = metadata.loc[metadata.split.eq("train")]
        target = "Brand"
    metadata = metadata.loc[metadata[target].isin(labels)].copy()
    if metadata.empty or not metadata.split.eq("train").all():
        raise ValueError("Only approved labeled training records may define arrival-time reference counts")
    part = features.loc[metadata.record_id]
    if part.group_id.tolist() != metadata.group_id.tolist():
        raise ValueError("Training reference group mismatch")
    keys = part.ProductBrand.map(brand_key).to_numpy()
    table = pd.DataFrame({"key": keys, "group_id": metadata.group_id.to_numpy(), "label": metadata[target].to_numpy()})
    valid = ~table.key.isin(MISSING_FEATURE)
    counts = metadata[target].value_counts().reindex(labels, fill_value=0)
    key_groups = table.loc[valid].groupby("key").group_id.nunique().to_dict()
    key_labels = table.loc[valid].groupby("key").label.nunique().to_dict()
    result = {"target": target, "training_rows": len(metadata), "training_groups": int(metadata.group_id.nunique()),
              "class_training_rows": {str(k): int(v) for k, v in counts.items()},
              "brand_key_training_groups": {str(k): int(v) for k, v in key_groups.items()},
              "brand_key_unambiguous": {str(k): bool(v == 1) for k, v in key_labels.items()},
              "source": "Approved labeled target training membership only; no validation key or truth fitting"}
    if model == "brand_lookup_probabilistic_v1":
        directory = root/SOURCE_RUN/model
        saved = evidence.json(directory/"model.json")
        manifest = evidence.json(directory/"manifest.json")
        if evidence.hashes[(directory/"model.json").resolve()] != manifest["files_sha256"]["model.json"]:
            raise ValueError("Saved lookup seal mismatch")
        if saved["training_class_counts"] != counts.tolist() or set(saved["key_counts"]) != set(key_groups):
            raise ValueError("Training reference differs from the saved lookup's actual fitted training population")
        if any(result["brand_key_unambiguous"][k] != (len(v) == 1) for k, v in saved["key_counts"].items()):
            raise ValueError("Saved lookup ambiguity disagrees with approved training reference")
    return result


def load_inputs(root, evidence):
    """Validate originals; only the two explicitly requested prediction runs load."""
    old = root/OLD_RUN
    verify_seal(old, evidence)
    previous = evidence.json(old/"summary.json")
    for name, digest in previous["source_sha256"].items():
        path = evidence.capture(root/Path(name))
        if evidence.hashes[path] != digest:
            raise ValueError("Original experiment source has changed")
    protocol, protected, contexts, identities, checks = frozen_identity_context(root, evidence)
    features = load_development_features(root, evidence, protected)
    handoff = evidence.json(root/SOURCE_RUN/"completion.json")
    frames, references, originals = {}, {}, {}
    config = StructuralConfig()
    for model, expected_floor in FLOORS.items():
        source = root/SOURCE_RUN/model
        policy = evidence.json(old/model/"policy.json")
        if policy != previous["policies"][model] or policy["selected_candidate"] != {"variant": "temperature", "threshold": expected_floor}:
            raise ValueError("Original target score variant or confidence floor changed")
        manifest, predictions, labels, columns, _ = load_run(source, evidence, contexts, identities, protected,
                                                           protocol, handoff_seal=handoff)
        assignments = evidence.csv(old/model/"experiment_assignments.csv", [*IDENTITY, "experiment_role", "eligible_measured_gates"])
        if (assignments.record_id.duplicated().any() or set(assignments.record_id) != set(predictions.record_id)
                or not assignments.split.eq("validation").all()
                or not assignments.eligible_measured_gates.isin(["True", "False"]).all()
                or assignments.experiment_role.tolist() != assignments.group_id.map(group_role).tolist()):
            raise ValueError("Frozen validation role/gate membership mismatch")
        assignment = assignments.set_index("record_id").loc[predictions.record_id]
        if assignment.group_id.tolist() != predictions.group_id.tolist():
            raise ValueError("Original frozen groups changed")
        temperature = evidence.json(old/model/"temperature.json")["temperature"]
        p = temperature_scale(predictions[columns].to_numpy(float), temperature)
        reference = training_reference(root, model, labels, features, evidence)
        part = features.loc[predictions.record_id]
        if part.group_id.tolist() != predictions.group_id.tolist():
            raise ValueError("Product feature groups differ from approved prediction groups")
        key = part.ProductBrand.map(brand_key)
        known = key.isin(reference["brand_key_training_groups"])
        key_missing = key.isin(MISSING_FEATURE)
        frame = predictions[["record_id", "group_id", "split", "predicted_label", "true_label"]].copy()
        frame["confidence"] = p.max(axis=1)
        frame["base_eligible"] = assignment.eligible_measured_gates.eq("True").to_numpy()
        frame["experiment_role"] = assignment.experiment_role.to_numpy()
        frame["description_present"] = (~part.ProductDescription.map(lambda v: missing(str(v), config))).to_numpy()
        frame["category_present"] = (~part.ProductCategory.map(lambda v: missing(str(v), config))).to_numpy()
        frame["brand_seen"] = (known & ~key_missing).to_numpy()
        frame["brand_familiarity"] = np.where(key_missing, "missing", np.where(known, "seen", "unfamiliar"))
        frame["brand_training_groups"] = key.map(reference["brand_key_training_groups"]).fillna(0).astype(int).to_numpy()
        frame["brand_key_unambiguous"] = key.map(reference["brand_key_unambiguous"]).eq(True).to_numpy()
        frame["predicted_class_training_rows"] = frame.predicted_label.map(reference["class_training_rows"]).fillna(0).astype(int)
        frame["true_class_training_rows"] = frame.true_label.map(reference["class_training_rows"]).fillna(0).astype(int)
        if (set(frame.record_id) & set(protected.record_id) or set(frame.group_id) & set(protected.group_id)
                or frame.groupby("group_id").experiment_role.nunique().gt(1).any()):
            raise ValueError("Protected identity or product-group role overlap")
        frames[model], references[model], originals[model] = frame, reference, policy
    return frames, references, originals, checks


def discover_product_rules(root, output, *, plot_cache=None):
    root = Path(root).resolve()
    requested = Path(output).resolve()
    if any(requested.is_relative_to(p) or p.is_relative_to(requested)
           for p in (root/OLD_RUN, root/SOURCE_RUN)):
        raise ValueError("Discovery output must be separate from previous source/result directories")
    destination, staging = fresh_staging(requested)
    evidence = Evidence()
    try:
        protocol = {"format": "catalogiq-product-rule-discovery-v1", "models": list(FLOORS),
            "fixed_confidence_floors": FLOORS, "class_screens": list(CLASS_MODES), "gates": GATES,
            "searched_rules": SEARCHED_RULES, "selection_alpha": .05/SEARCHED_RULES,
            "discovery_role": "calibration_fit", "selection_role": "threshold_selection",
            "audit_role_forbidden_for_rule_design": "policy_audit",
            "min_discovery_class_groups": MIN_DISCOVERY_CLASS_GROUPS,
            "min_recommended_policy_groups": MIN_POLICY_GROUPS,
            "min_recommended_effective_rows": MIN_POLICY_EFFECTIVE_ROWS,
            "common_training_rows": COMMON_TRAINING_ROWS, "feature_definitions": FEATURE_DEFINITIONS,
            "confidence_ranges": "Diagnostics only; no confidence threshold or score variant changes",
            "selection": "Maximum coverage with search-adjusted beta >=.99; else nominal >=.99; else best lower bound among sufficiently large observed-99% improvements; never audit fallback",
            "untouched_independent_confirmation_available": False,
            "evidence_status": "Historical development exploration; role separation does not undo previous exposure",
            "code_sha256": code_hashes(), "automatic_acceptance_enabled": False}
        write_json(staging/"protocol.json", protocol)
        frames, references, _, checks = load_inputs(root, evidence)
        choices, selections = {}, []
        all_diagnostics = []
        for model in FLOORS:
            directory = staging/model
            directory.mkdir()
            frame = frames[model]
            discovery = frame.loc[frame.experiment_role.eq("calibration_fit")].copy()
            selection = frame.loc[frame.experiment_role.eq("threshold_selection")].copy()
            check_phase_isolation(discovery, selection)
            vocabulary = list(references[model]["class_training_rows"])
            registry, classes = discover_rules(discovery, model, vocabulary, FLOORS[model])
            scores = score_registry(selection, registry)
            choice = choose_rule(scores)
            choice["rule"] = next((r for r in registry if r["id"] == choice["rule_id"]), None)
            choice["model"] = model
            choice["fixed_threshold"] = FLOORS[model]
            choice["feature_definitions"] = FEATURE_DEFINITIONS
            choices[model] = choice
            write_json(directory/"registry.json", registry)
            write_json(directory/"selected_experimental_rule.json", choice)
            write_json(directory/"training_reference.json", references[model])
            classes.to_csv(directory/"discovery_predicted_classes.csv", index=False, lineterminator="\n")
            scores.to_csv(directory/"selection_policies.csv", index=False, lineterminator="\n")
            selections.append(scores)
            for role, part in (("calibration_fit", discovery), ("threshold_selection", selection)):
                diagnostics = diagnostic_tables(part, registry[0], role=role)
                diagnostics["model"] = model
                all_diagnostics.append(diagnostics)
            # Only non-audit rows are exported for discovery reproducibility.
            pd.concat([discovery, selection]).to_csv(directory/"non_audit_features.csv", index=False, lineterminator="\n")
            print(f"Discovered {model}: {len(registry)} fixed-floor candidates; selected {choice['rule_id']} ({choice['status']})", flush=True)
        pd.concat(selections).to_csv(staging/"selection_policy_comparison.csv", index=False, lineterminator="\n")
        pd.concat(all_diagnostics).to_csv(staging/"characteristic_associations.csv", index=False, lineterminator="\n")
        write_json(staging/"selected_rules.json", choices)
        from .product_rule_plots import plot_discovery
        plot_discovery(staging, plot_cache=plot_cache)
        evidence.reverify()
        write_json(staging/"summary.json", {"choices": choices, "protected_registry_checks": checks,
            "source_sha256": {str(p): h for p, h in evidence.hashes.items()},
            "source_files_verified_unchanged": len(evidence.hashes), "code_sha256": code_hashes(),
            "python": platform.python_version(), "libraries": {p: version(p) for p in ("numpy", "pandas", "scipy")},
            "no_classifier_training_or_calibration_fit": True, "no_audit_results_used_for_rule_selection": True,
            "untouched_independent_confirmation_available": False, "automatic_acceptance_enabled": False})
        seal_output(staging)
        staging.rename(destination)
    except BaseException:
        cleanup_staging(staging, destination)
        raise
    return destination


def audit_product_rules(root, discovery_run, output, *, plot_cache=None):
    root, discovery_run = Path(root).resolve(), Path(discovery_run).resolve()
    requested = Path(output).resolve()
    if any(requested.is_relative_to(p) or p.is_relative_to(requested)
           for p in (discovery_run, root/OLD_RUN, root/SOURCE_RUN)):
        raise ValueError("Audit output must be separate from previous source/result directories")
    destination, staging = fresh_staging(requested)
    evidence = Evidence()
    try:
        verify_seal(discovery_run, evidence)
        protocol = evidence.json(discovery_run/"protocol.json")
        if protocol["code_sha256"] != code_hashes():
            raise ValueError("Rule code changed after discovery/selection was frozen")
        choices = evidence.json(discovery_run/"selected_rules.json")
        write_json(staging/"protocol.json", {"format": "catalogiq-product-rule-historical-audit-v1",
            "discovery_completion_sha256": sha256(discovery_run/"completion.json"),
            "chosen_rules_frozen_before_audit": choices, "selection_on_audit_forbidden": True,
            "fixed_confidence_floors": FLOORS, "untouched_independent_confirmation": False,
            "claim_correction": "alpha=.05/2 for the two selected aggregate policies; class diagnostics are not simultaneous guarantees",
            "automatic_acceptance_enabled": False})
        frames, references, originals, checks = load_inputs(root, evidence)
        rows, class_tables, frequency_tables = [], [], []
        for model in FLOORS:
            frame = frames[model]
            audit = frame.loc[frame.experiment_role.eq("policy_audit")].copy()
            require_role(audit, "policy_audit")
            registry = evidence.json(discovery_run/model/"registry.json")
            baseline = registry[0]
            chosen = choices[model]["rule"]
            for policy_name, rule in (("fixed_threshold_baseline", baseline), ("selected_experimental_rule", chosen)):
                accepted = apply_rule(audit, rule, expected_threshold=FLOORS[model]) if rule else np.zeros(len(audit), bool)
                summary = summarize_policy(audit, accepted)
                corrected = summarize_policy(audit, accepted, alpha=.05/len(FLOORS))
                if policy_name == "fixed_threshold_baseline":
                    old = originals[model]["audit"]
                    if summary["accepted_rows"] != old["accepted_rows"] or summary["errors"] != old["accepted_errors"]:
                        raise ValueError("Fixed-threshold baseline replay changed")
                rows.append({"model": model, "policy": policy_name, "rule_id": rule["id"] if rule else None,
                    **summary, "beta_lower_two_policy_adjusted": corrected["beta_lower"],
                    "nominal_lower_clears_99": bool(summary["beta_lower"] is not None and summary["beta_lower"] >= .99),
                    "two_policy_lower_clears_99": bool(corrected["beta_lower"] is not None and corrected["beta_lower"] >= .99),
                    "selection_status": choices[model]["status"], "independently_confirmed": False,
                    "automatic_acceptance_enabled": False})
                per_class = uncommon_class_table(audit, accepted, baseline["vocabulary"], policy_name=policy_name,
                                                 training_counts=references[model]["class_training_rows"])
                per_class["model"] = model
                class_tables.append(per_class)
                for definition, field, arrival in (("true_class_frequency", "true_class_training_rows", False),
                                                    ("predicted_class_frequency", "predicted_class_training_rows", True)):
                    for label, condition in (("common", audit[field].ge(COMMON_TRAINING_ROWS)),
                                              ("uncommon", audit[field].lt(COMMON_TRAINING_ROWS))):
                        if condition.any():
                            s = summarize_policy(audit.loc[condition], accepted[condition.to_numpy()])
                            frequency_tables.append({"model": model, "policy": policy_name, "definition": definition,
                                "frequency": label, "available_at_arrival": arrival, **s})
            write_json(staging/f"{model}_experimental_policy.json", {**choices[model],
                "enabled": False, "independently_confirmed": False,
                "audit_used_to_change_rule": False, "pending": ["Fresh independent development evaluation",
                    "Previously documented approved relationship and serving-review integration"],
                "unchanged_baseline_mandatory_review_rules": True})
            print(f"Historical audit replayed {model}: baseline and frozen selected rule only", flush=True)
        pd.DataFrame(rows).to_csv(staging/"audit_policy_comparison.csv", index=False, lineterminator="\n")
        pd.concat(class_tables).to_csv(staging/"audit_per_class.csv", index=False, lineterminator="\n")
        pd.DataFrame(frequency_tables).to_csv(staging/"audit_class_frequency.csv", index=False, lineterminator="\n")
        from .product_rule_plots import plot_audit
        plot_audit(staging, plot_cache=plot_cache)
        evidence.reverify()
        write_json(staging/"summary.json", {"results": rows, "protected_registry_checks": checks,
            "source_files_verified_unchanged": len(evidence.hashes),
            "source_sha256": {str(p): h for p, h in evidence.hashes.items()}, "code_sha256": code_hashes(),
            "independently_confirmed": False, "no_threshold_changes_or_classifier_training": True,
            "other_target_policies_preserved": True, "protected_test_evaluated": False,
            "automatic_acceptance_enabled": False})
        seal_output(staging)
        staging.rename(destination)
    except BaseException:
        cleanup_staging(staging, destination)
        raise
    return destination
