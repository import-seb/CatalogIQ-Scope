"""Simulation-first reassessment of frozen confidence policies; never fits models."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from .accuracy_simulations import run_simulations, scenarios, SIMULATION_SEED
from .clustered_accuracy import analytic_bounds, aggregate_groups, group_bootstrap
from .confidence_calibration import group_role, temperature_scale
from .confidence_experiment import load_run
from .features import sha256
from .prediction_contract import Evidence, RUNS, frozen_identity_context, write_json


OLD_RUN = "data/processed/confidence_review_v3_20261103"
SOURCE_RUN = "data/processed/predictions_contract_v2_20261031"
VALIDATION_RUN = "data/processed/accuracy_method_validation_v1_20261103"
AUDIT_RUN = "data/processed/accuracy_reassessment_v1_20261103"
PRIMARY = "beta_effective_sample"
SOURCES = {
    "wilson_survey_extension": "https://www150.statcan.gc.ca/n1/pub/12-001-x/2017002/article/54872/02-eng.htm",
    "franco_et_al_2018": "https://www.math.umd.edu/~slud/s770/SurveyConfidenceIntervals/JSSAM-2017-065-FINAL.pdf",
    "korn_graubard_1998": "https://www150.statcan.gc.ca/n1/pub/12-001-x/1998002/article/4356-eng.pdf",
    "dean_pagano_2015": "https://academic.oup.com/jssam/article-abstract/3/4/484/950501",
    "nchs_standards": "https://www.cdc.gov/nchs/media/pdfs/2024/09/sr02_175.pdf",
}


def fresh_staging(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing existing result directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent/f"accuracy-staging-{uuid4().hex}"
    staging.mkdir()
    return destination, staging


def cleanup_staging(staging, destination):
    if (staging.resolve().parent != destination.resolve().parent
            or not staging.name.startswith("accuracy-staging-")):
        raise RuntimeError("Refusing cleanup outside the new staging directory")
    shutil.rmtree(staging)


def code_hashes():
    directory = Path(__file__).parent
    return {name: sha256(directory/name) for name in (
        "clustered_accuracy.py", "accuracy_simulations.py", "accuracy_reassessment.py")}


def seal_output(directory):
    write_json(directory/"completion.json", {"files_sha256": {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*")) if p.is_file()}})


def verify_seal(directory, evidence):
    directory = Path(directory).resolve()
    seal = evidence.json(directory/"completion.json")
    for name, expected in seal["files_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or path.name.casefold() == "test.csv":
            raise ValueError("Unsafe result seal member")
        evidence.capture(path)
        if evidence.hashes[path] != expected:
            raise ValueError(f"Saved result changed: {name}")
    return seal


def run_method_validation(output, *, trials=2000, bootstrap_repetitions=1999):
    """No real catalog data is read by this entry point, including audit sizes."""
    destination, staging = fresh_staging(output)
    try:
        protocol = {
            "format": "catalogiq-accuracy-method-validation-v1", "seed": SIMULATION_SEED,
            "alpha": .05, "goal": .99, "trials_per_case": trials,
            "bootstrap_repetitions_per_trial": bootstrap_repetitions,
            "scenarios": scenarios(), "sources": SOURCES, "code_sha256": code_hashes(),
            "catalog_data_read": False, "audit_results_used_for_method_choice": False,
            "method_a": "One-sided Wilson with unequal-size rho=1 design effect and published df adjustment",
            "method_b": "One-sided percentile group-ratio bootstrap, fifth percentile",
            "alternative": "Korn-Graubard-type effective-sample beta interval with the same size-only working design effect",
            "primary_reporting_method_fixed_before_simulations": PRIMARY,
            "primary_reason": "Published comparisons favor conservative beta/Clopper-Pearson-type intervals for extreme proportions and clustering; bootstrap is a cross-check, not a chooser",
            "working_model": "Independent families; no systematic association of family size with expected error probability; arbitrary within-family correlation up to one",
            "veto": "Simulation undercoverage must be reported and limits claims; do not switch to a method because it passes a real model",
            "simulation_intervals": "Exact binomial 95% Monte Carlo uncertainty on independent trial rates; diagnostic comparisons, not multiplicity-adjusted tests",
            "per_model_not_simultaneous": True,
        }
        write_json(staging/"protocol.json", protocol)
        results = run_simulations(staging, trials=trials, bootstrap_repetitions=bootstrap_repetitions)
        flags = results.loc[results.demonstrated_undercoverage]
        primary_flags = flags.loc[flags.method.eq(PRIMARY)]
        common_flags = primary_flags.loc[primary_flags.size_independent_risk]
        decision = {
            "primary_reporting_method": PRIMARY,
            "method_preference_based_on": "Published rare-proportion guidance; frozen before simulations and new audit calculation",
            "general_99_percent_certification_validated": False,
            "all_tested_primary_coverage_checks_without_demonstrated_undercoverage": not len(primary_flags),
            "common_family_risk_working_model_demonstrated_undercoverage": bool(len(common_flags)),
            "primary_undercoverage_cases": primary_flags.case.tolist(),
            "all_method_undercoverage_flags": flags[["case", "method", "coverage_failure_rate",
                                                        "coverage_failure_mc_lower", "coverage_failure_mc_upper"]].to_dict("records"),
            "production_enabled": False,
            "interpretation": "Report conditional development bounds. If stress cases fail, no unconditional 99% certification is justified, even for a real model whose numerical bound clears .99.",
            "sources": SOURCES,
        }
        write_json(staging/"method_decision.json", decision)
        write_json(staging/"summary.json", {"trials": trials, "scenarios": len(scenarios()),
                   "total_independent_catalogs": trials*len(scenarios()), "python": platform.python_version(),
                   "libraries": {p: version(p) for p in ("numpy", "pandas", "scipy")},
                   "code_sha256": code_hashes(), "catalog_data_read": False})
        seal_output(staging)
        staging.rename(destination)
    except BaseException:
        cleanup_staging(staging, destination)
        raise
    return destination


def evaluate_frozen_audit(root, previous, method_validation, output, *, bootstrap_repetitions=49999):
    """Consume a sealed simulation decision before opening real audit predictions."""
    root, previous, method_validation = Path(root).resolve(), Path(previous).resolve(), Path(method_validation).resolve()
    requested = Path(output).resolve()
    if any(requested.is_relative_to(p) or p.is_relative_to(requested)
           for p in (previous, method_validation, root/SOURCE_RUN)):
        raise ValueError("New evaluation output must be separate from existing evidence")
    destination, staging = fresh_staging(requested)
    evidence = Evidence()
    try:
        verify_seal(method_validation, evidence)
        decision = evidence.json(method_validation/"method_decision.json")
        simulation_protocol = evidence.json(method_validation/"protocol.json")
        if (decision["primary_reporting_method"] != PRIMARY
                or simulation_protocol["primary_reporting_method_fixed_before_simulations"] != PRIMARY
                or simulation_protocol["code_sha256"] != code_hashes()):
            raise ValueError("Method decision/code changed after simulation validation")
        # Record this frozen choice before any catalog/audit content is loaded.
        write_json(staging/"protocol.json", {"format": "catalogiq-frozen-policy-reassessment-v1",
            "primary_method": PRIMARY, "alpha": .05, "goal": .99,
            "simulations_completion_sha256": sha256(method_validation/"completion.json"),
            "bootstrap_repetitions": bootstrap_repetitions, "bootstrap_seed": 20261104,
            "thresholds_temperatures_and_gates": "Exactly as sealed in the previous experiment",
            "reselection_recalibration_or_category_slices": False,
            "simultaneous_sensitivity": "Also report primary bound at alpha=.05/6; primary requested comparison remains one-sided 95% per model",
            "production_enabled": False})
        previous_seal = verify_seal(previous, evidence)
        previous_summary = evidence.json(previous/"summary.json")
        # Bind to every previously consumed source, without reading model weights.
        for name, expected in previous_summary["source_sha256"].items():
            path = evidence.capture(root/Path(name))
            if evidence.hashes[path] != expected:
                raise ValueError(f"Original experiment source changed: {name}")
        protocol, protected, contexts, identities, registry_checks = frozen_identity_context(root, evidence)
        source = root/SOURCE_RUN
        handoff = evidence.json(source/"completion.json")
        rows, stability, all_counts = [], [], []
        for name in RUNS:
            old_policy = evidence.json(previous/name/"policy.json")
            if old_policy != previous_summary["policies"][name]:
                raise ValueError("Old per-run policy differs from saved summary")
            candidate = old_policy["selected_candidate"]
            temperature = evidence.json(previous/name/"temperature.json")["temperature"]
            assignments = evidence.csv(previous/name/"experiment_assignments.csv",
                                       ["record_id", "group_id", "split", "experiment_role", "eligible_measured_gates"])
            if (assignments.record_id.duplicated().any() or not assignments.split.eq("validation").all()
                    or assignments.experiment_role.tolist() != assignments.group_id.map(group_role).tolist()
                    or not assignments.eligible_measured_gates.isin(["True", "False"]).all()):
                raise ValueError("Saved validation experiment roles/gates changed")
            manifest, frame, labels, columns, checks = load_run(source/name, evidence, contexts, identities,
                                                               protected, protocol, handoff_seal=handoff)
            if set(assignments.record_id) != set(frame.record_id):
                raise ValueError("Saved experiment membership differs from predictions")
            part = assignments.set_index("record_id").loc[frame.record_id]
            if part.group_id.tolist() != frame.group_id.tolist():
                raise ValueError("Saved frozen groups differ from prediction context")
            audit_mask = part.experiment_role.eq("policy_audit").to_numpy()
            audit = frame.loc[audit_mask].reset_index(drop=True)
            # Apply the saved scalar; never estimate a temperature or change a label.
            p = temperature_scale(audit[columns].to_numpy(float),
                                  temperature if candidate["variant"] == "temperature" else 1.)
            accepted = part.eligible_measured_gates.eq("True").to_numpy()[audit_mask] & (p.max(axis=1) >= candidate["threshold"])
            correct = audit.predicted_label.eq(audit.true_label).to_numpy()
            counts = aggregate_groups(accepted, correct, audit.group_id)
            analytic = analytic_bounds(counts.accepted_rows, counts.errors)
            prior = old_policy["audit"]
            for current, old in ((len(audit), prior["rows"]), (analytic["rows"], prior["accepted_rows"]),
                                 (analytic["errors"], prior["accepted_errors"]), (analytic["groups"], prior["accepted_groups"])):
                if current != old:
                    raise ValueError("Frozen-policy replay differs from original audit counts")
            boot = group_bootstrap(counts.accepted_rows, counts.errors, repetitions=bootstrap_repetitions)
            familywise = analytic_bounds(counts.accepted_rows, counts.errors, alpha=.05/len(RUNS))
            row = {"run": name, "target": manifest["target"], "provisional": old_policy["provisional"],
                "threshold": candidate["threshold"], "score_variant": candidate["variant"],
                "audit_rows": len(audit), "audit_groups": len(counts),
                "accepted_predictions": analytic["rows"], "accepted_groups": analytic["groups"],
                "mistakes": analytic["errors"], "error_groups": analytic["error_groups"],
                "observed_accuracy": analytic["accuracy"], "max_accepted_group_rows": analytic["max_group_rows"],
                "design_effect_rho_one": analytic["design_effect"], "n_eff": analytic["n_eff"],
                "n_eff_df": analytic["n_eff_df"],
                "wilson_lower_95": analytic["wilson_size_adjusted"], "bootstrap_lower_95": boot["lower"],
                "beta_effective_lower_95": analytic["beta_effective_sample"],
                "wilson_clears_99": analytic["wilson_size_adjusted"] >= .99,
                "bootstrap_clears_99": boot["lower"] >= .99,
                "primary_beta_clears_99": analytic["beta_effective_sample"] >= .99,
                "beta_familywise_lower": familywise["beta_effective_sample"],
                "bootstrap_zero_error_resample_fraction": boot["zero_error_resample_fraction"],
                "general_99_certification_validated_by_simulations": decision["general_99_percent_certification_validated"],
                "historical_development_only": True, "automatic_acceptance_enabled": False}
            rows.append(row)
            counts.insert(0, "run", name)
            all_counts.append(counts)
            for repetitions in (1999, 9999, 49999):
                for seed in (20261104, 20261105, 20261106):
                    check = group_bootstrap(counts.accepted_rows, counts.errors, repetitions=repetitions, seed=seed)
                    stability.append({"run": name, "seed": seed, "repetitions": repetitions, "lower": check["lower"]})
            print(f"Replayed fixed policy {name}: {analytic['errors']} errors / {analytic['rows']:,} accepted", flush=True)
        pd.DataFrame(rows).to_csv(staging/"comparison.csv", index=False, lineterminator="\n")
        pd.DataFrame(stability).to_csv(staging/"bootstrap_stability.csv", index=False, lineterminator="\n")
        pd.concat(all_counts).to_csv(staging/"audit_group_counts.csv", index=False, lineterminator="\n")
        evidence.reverify()
        write_json(staging/"summary.json", {"primary_method": PRIMARY, "method_decision": decision,
            "code_sha256": code_hashes(), "source_sha256": {str(p): h for p, h in evidence.hashes.items()},
            "source_files_verified_unchanged": len(evidence.hashes), "rows": rows,
            "protected_registry_checks": registry_checks, "python": platform.python_version(),
            "libraries": {p: version(p) for p in ("numpy", "pandas", "scipy")},
            "no_retraining_recalibration_threshold_reselection_or_category_analysis": True,
            "test_csv_opened_or_hashed": False, "automatic_acceptance_enabled": False})
        seal_output(staging)
        staging.rename(destination)
    except BaseException:
        cleanup_staging(staging, destination)
        raise
    return destination
