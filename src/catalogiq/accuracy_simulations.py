"""Data-free operating-characteristic checks for clustered accuracy intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clustered_accuracy import analytic_bounds, group_bootstrap, monte_carlo_interval


SIMULATION_SEED = 20261103


def family_sizes(pattern):
    """Fixed synthetic populations, chosen without consulting audit outcomes."""
    definitions = {
        "singletons": {1: 2000},
        "equal_two": {2: 2000},
        "unequal": {1: 1300, 2: 450, 3: 150, 5: 70, 10: 28, 30: 2},
        "long_tail": {1: 1300, 2: 400, 3: 150, 5: 80, 10: 50, 30: 18, 80: 2},
    }
    return np.concatenate([np.repeat(size, count) for size, count in definitions[pattern].items()])


def scenarios():
    cases = []
    def add(pattern, process, rates):
        for rate in rates:
            cases.append({"case": f"{pattern}_{process}_{rate:.5f}", "size_pattern": pattern,
                          "error_process": process, "true_error_rate": rate,
                          "true_accuracy": 1-rate,
                          "size_independent_risk": process != "large_only"})
    add("singletons", "row_independent", [.0101, .01, .0099])
    add("equal_two", "row_independent", [.0101, .01, .0099, .00025])
    add("equal_two", "whole_family", [.0101, .01, .0099, .005])
    add("unequal", "row_independent", [.0101, .0099])
    add("unequal", "beta_binomial_rho_0.2", [.0101, .01, .0099, .005, .00025])
    add("long_tail", "whole_family", [.0101, .01, .0099, .00025])
    add("long_tail", "large_only", [.011, .0101, .01, .009])
    return cases


def simulate_errors(sizes, case, rng):
    q, process = case["true_error_rate"], case["error_process"]
    if process == "row_independent":
        return rng.binomial(sizes, q)
    if process == "whole_family":
        return sizes * rng.binomial(1, q, size=len(sizes))
    if process == "beta_binomial_rho_0.2":
        rho = .2
        latent = rng.beta(q*(1-rho)/rho, (1-q)*(1-rho)/rho, size=len(sizes))
        return rng.binomial(sizes, latent)
    if process == "large_only":
        large = sizes >= 30
        # Expected row error remains exactly q; every error is in a large family.
        chance = q*sizes.sum()/sizes[large].sum()
        if chance > 1:
            raise ValueError("Infeasible large-family error probability")
        return sizes * large * rng.binomial(1, chance, size=len(sizes))
    raise ValueError("Unknown synthetic error process")


def run_simulations(output, *, trials=2000, bootstrap_repetitions=1999, seed=SIMULATION_SEED):
    """Generate intervals before any new real-audit evaluation is permitted."""
    if trials < 100 or bootstrap_repetitions < 99:
        raise ValueError("Simulation requires >=100 trials and >=99 bootstrap draws")
    details, results = [], []
    for index, case in enumerate(scenarios()):
        sizes = family_sizes(case["size_pattern"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        local = []
        for trial in range(trials):
            errors = simulate_errors(sizes, case, rng)
            bounds = analytic_bounds(sizes, errors)
            boot = group_bootstrap(sizes, errors, repetitions=bootstrap_repetitions,
                                   seed=seed + 100000*(index+1) + trial)
            row = {"case": case["case"], "trial": trial, "errors": int(errors.sum()),
                   "error_groups": int((errors > 0).sum()),
                   "observed_accuracy": bounds["accuracy"],
                   "wilson_size_adjusted": bounds["wilson_size_adjusted"],
                   "beta_effective_sample": bounds["beta_effective_sample"],
                   "group_bootstrap": boot["lower"]}
            local.append(row)
        frame = pd.DataFrame(local)
        details.extend(local)
        for method in ("wilson_size_adjusted", "group_bootstrap", "beta_effective_sample"):
            lower = frame[method].to_numpy()
            misses = int((lower > case["true_accuracy"]+1e-12).sum())
            support = int((lower >= .99).sum())
            miss_lo, miss_hi = monte_carlo_interval(misses, trials)
            support_lo, support_hi = monte_carlo_interval(support, trials)
            results.append({**case, "method": method, "trials": trials,
                "bootstrap_repetitions": bootstrap_repetitions,
                "groups": len(sizes), "rows": int(sizes.sum()), "max_group_rows": int(sizes.max()),
                "coverage": 1-misses/trials, "coverage_failure_rate": misses/trials,
                "coverage_failure_mc_lower": miss_lo, "coverage_failure_mc_upper": miss_hi,
                "support_99_rate": support/trials, "support_99_mc_lower": support_lo,
                "support_99_mc_upper": support_hi,
                "false_support_99_rate": support/trials if case["true_accuracy"] < .99 else None,
                "zero_observed_error_fraction": float(frame.errors.eq(0).mean()),
                "bound_equals_one_fraction": float((lower == 1).mean()),
                "mean_lower_bound": float(lower.mean()),
                "demonstrated_undercoverage": miss_lo > .05})
        print(f"Simulated {case['case']}: {trials:,} independent trials", flush=True)
    result = pd.DataFrame(results)
    pd.DataFrame(details).to_csv(output/"simulation_trials.csv", index=False, lineterminator="\n")
    result.to_csv(output/"simulation_comparison.csv", index=False, lineterminator="\n")
    return result
