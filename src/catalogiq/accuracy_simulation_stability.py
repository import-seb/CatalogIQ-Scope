"""Independent high-repetition check of bootstrap simulation conclusions.

This reads no CatalogIQ data and never changes the frozen method decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .accuracy_reassessment import cleanup_staging, code_hashes, fresh_staging, seal_output
from .accuracy_simulations import family_sizes, scenarios, simulate_errors
from .clustered_accuracy import analytic_bounds, group_bootstrap, monte_carlo_interval
from .prediction_contract import write_json


CASES = (
    "singletons_row_independent_0.01010",
    "unequal_beta_binomial_rho_0.2_0.01010",
    "long_tail_whole_family_0.01010",
    "equal_two_row_independent_0.00025",
)


def run_stability(output, *, trials=1000, repetitions=19999):
    destination, staging = fresh_staging(output)
    snapshots = code_hashes()
    rows = []
    try:
        write_json(staging/"protocol.json", {"cases": list(CASES), "trials": trials,
                   "bootstrap_repetitions": repetitions, "seed": 30361103,
                   "catalog_data_read": False, "frozen_method_decision_changed": False,
                   "purpose": "Sensitivity to finite bootstrap repetition count; independent simulation seeds",
                   "code_sha256": snapshots})
        for index, case in enumerate(c for c in scenarios() if c["case"] in CASES):
            sizes = family_sizes(case["size_pattern"])
            rng = np.random.default_rng(np.random.SeedSequence([30361103, index]))
            values = {"wilson_size_adjusted": [], "group_bootstrap": [], "beta_effective_sample": []}
            for trial in range(trials):
                errors = simulate_errors(sizes, case, rng)
                analytic = analytic_bounds(sizes, errors)
                values["wilson_size_adjusted"].append(analytic["wilson_size_adjusted"])
                values["beta_effective_sample"].append(analytic["beta_effective_sample"])
                values["group_bootstrap"].append(group_bootstrap(sizes, errors,
                    repetitions=repetitions, seed=40361103+100000*index+trial)["lower"])
            for method, lower in values.items():
                lower = np.asarray(lower)
                failures = int((lower > case["true_accuracy"]+1e-12).sum())
                supports = int((lower >= .99).sum())
                lo, hi = monte_carlo_interval(failures, trials)
                support_lo, support_hi = monte_carlo_interval(supports, trials)
                rows.append({**case, "method": method, "trials": trials,
                    "bootstrap_repetitions": repetitions, "coverage_failure_rate": failures/trials,
                    "coverage_failure_mc_lower": lo, "coverage_failure_mc_upper": hi,
                    "false_support_99_rate": supports/trials if case["true_accuracy"] < .99 else None,
                    "support_99_mc_lower": support_lo, "support_99_mc_upper": support_hi})
            print(f"High-repetition confirmation {case['case']}: {trials:,} trials x {repetitions:,} resamples", flush=True)
        if snapshots != code_hashes():
            raise ValueError("Statistical code changed during stability confirmation")
        pd.DataFrame(rows).to_csv(staging/"simulation_confirmation.csv", index=False, lineterminator="\n")
        seal_output(staging)
        staging.rename(destination)
    except BaseException:
        cleanup_staging(staging, destination)
        raise
    return destination
