"""Established approximate intervals for row accuracy with independent families.

These are working-model/sampling intervals, not distribution-free guarantees.
No classifier fitting, probability calibration, threshold choice or file access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, t


METHODS = ("wilson_size_adjusted", "group_bootstrap", "beta_effective_sample")


def validate_group_counts(accepted_rows, errors):
    sizes, errors = np.asarray(accepted_rows), np.asarray(errors)
    if (sizes.ndim != 1 or errors.shape != sizes.shape or not len(sizes)
            or not np.isfinite(sizes).all() or not np.isfinite(errors).all()
            or (sizes < 0).any() or (errors < 0).any() or (errors > sizes).any()
            or (sizes != np.floor(sizes)).any() or (errors != np.floor(errors)).any()):
        raise ValueError("Matching nonnegative integer family counts with errors <= rows required")
    return sizes.astype(np.int64), errors.astype(np.int64)


def aggregate_groups(accepted, correct, group_ids):
    accepted, correct, groups = np.asarray(accepted), np.asarray(correct), np.asarray(group_ids)
    if (accepted.ndim != 1 or accepted.shape != correct.shape or accepted.shape != groups.shape
            or not len(accepted) or not np.isin(accepted, [False, True]).all()
            or not np.isin(correct, [False, True]).all()
            or pd.isna(groups).any() or any(not str(g).strip() for g in groups)):
        raise ValueError("Matching nonempty boolean masks and nonblank family IDs required")
    frame = pd.DataFrame({"group_id": groups, "accepted_rows": accepted.astype(int),
                          "errors": accepted.astype(bool) & ~correct.astype(bool)})
    # Retain groups with zero accepted rows for the ratio bootstrap.
    return frame.groupby("group_id", sort=True)[["accepted_rows", "errors"]].sum().reset_index()


def effective_sample_size(accepted_rows, alpha=.05):
    """Unequal-size design effect with rho=1; Dean/Pagano df adjustment.

    Under a common family error probability, Var(row fraction) is bounded
    using perfectly correlated records within each family:
        deff=sum(m_g^2)/sum(m_g), n_eff=sum(m_g)^2/sum(m_g^2).
    This is NOT valid under every association between family size and risk.
    No intraclass correlation is estimated from one or two observed errors.
    """
    if not 0 < alpha < .5:
        raise ValueError("One-sided alpha must be between zero and .5")
    sizes = np.asarray(accepted_rows, float)
    if sizes.ndim != 1 or not np.isfinite(sizes).all() or (sizes < 0).any():
        raise ValueError("Finite nonnegative family sizes required")
    positive = sizes[sizes > 0]
    n, g = float(positive.sum()), len(positive)
    if not g:
        return {"rows": 0, "groups": 0, "design_effect": None, "n_eff": None, "n_eff_df": None}
    squared_sum = float(np.square(positive).sum())
    neff = n*n/squared_sum
    df_factor = float((norm.ppf(1-alpha)/t.ppf(1-alpha, g-1))**2) if g > 1 else 0.
    return {"rows": int(n), "groups": g, "design_effect": squared_sum/n,
            "n_eff": neff, "n_eff_df": neff*df_factor}


def wilson_lower(accuracy, n_eff, alpha=.05):
    if not 0 < alpha < .5 or not 0 <= accuracy <= 1 or not np.isfinite(n_eff) or n_eff < 0:
        raise ValueError("Valid accuracy, effective size and alpha required")
    if n_eff == 0:
        return 0.
    z2 = float(norm.ppf(1-alpha)**2)
    center = accuracy + z2/(2*n_eff)
    radius = np.sqrt(accuracy*(1-accuracy)/n_eff + z2/(4*n_eff*n_eff)) * np.sqrt(z2)
    return float(np.clip((center-radius)/(1+z2/n_eff), 0, 1))


def beta_lower(accuracy, n_eff, alpha=.05):
    """Korn-Graubard-type beta interval with specified working effective size.

    Fractional effective counts are intentional. This is not an exact binomial
    interval for clustered data, and is not the complete NHANES procedure.
    """
    if not 0 < alpha < .5 or not 0 <= accuracy <= 1 or not np.isfinite(n_eff) or n_eff < 0:
        raise ValueError("Valid accuracy, effective size and alpha required")
    if n_eff == 0 or accuracy == 0:
        return 0.
    return float(beta.ppf(alpha, accuracy*n_eff, (1-accuracy)*n_eff+1))


def analytic_bounds(accepted_rows, errors, alpha=.05):
    sizes, errors = validate_group_counts(accepted_rows, errors)
    effective = effective_sample_size(sizes, alpha)
    n, e = int(sizes.sum()), int(errors.sum())
    if n == 0:
        return {**effective, "errors": 0, "accuracy": None,
                "wilson_size_adjusted": None, "beta_effective_sample": None,
                "error_groups": 0, "max_group_rows": 0}
    accuracy = 1-e/n
    return {**effective, "errors": e, "accuracy": accuracy,
            "wilson_size_adjusted": wilson_lower(accuracy, effective["n_eff_df"], alpha),
            "beta_effective_sample": beta_lower(accuracy, effective["n_eff_df"], alpha),
            "error_groups": int((errors > 0).sum()), "max_group_rows": int(sizes.max())}


def group_bootstrap(accepted_rows, errors, *, repetitions=49999, seed=20261104, alpha=.05):
    """Percentile one-sided lower bound for the accepted-row ratio.

    Sampling the histogram of identical (size,error) pairs multinomially is
    exactly the same resampling distribution as drawing whole groups with
    replacement. It is much faster than materializing repeated product rows.
    """
    sizes, errors = validate_group_counts(accepted_rows, errors)
    if not 0 < alpha < .5 or repetitions < 99:
        raise ValueError("Valid alpha and at least 99 bootstrap repetitions required")
    if not sizes.sum():
        return {"lower": None, "repetitions": repetitions, "valid_repetitions": 0,
                "zero_observed_errors": True, "zero_error_resample_fraction": None}
    pairs, counts = np.unique(np.column_stack([sizes, errors]), axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    values, zero_errors = [], 0
    for start in range(0, repetitions, 2048):
        draw = rng.multinomial(len(sizes), counts/len(sizes), size=min(2048, repetitions-start))
        n, e = draw @ pairs[:, 0], draw @ pairs[:, 1]
        valid = n > 0
        values.append(1-e[valid]/n[valid])
        zero_errors += int((e[valid] == 0).sum())
    values = np.concatenate(values)
    return {"lower": float(np.quantile(values, alpha)) if len(values) else None,
            "repetitions": repetitions, "valid_repetitions": len(values),
            "zero_observed_errors": bool(errors.sum() == 0),
            "zero_error_resample_fraction": zero_errors/len(values) if len(values) else None}


def monte_carlo_interval(events, trials, alpha=.05):
    """Exact binomial interval for independent simulation trials, not products."""
    if not 0 <= events <= trials or trials < 1:
        raise ValueError("Invalid independent simulation trial counts")
    return (float(beta.ppf(alpha/2, events, trials-events+1)) if events else 0.,
            float(beta.ppf(1-alpha/2, events+1, trials-events)) if events < trials else 1.)
