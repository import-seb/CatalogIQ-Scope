"""Target-specific allocation over frozen V4 development groups.

Training coverage is mandatory. Integer constraints establish maximum validation
coverage and the nearest feasible labeled validation count, in that order. The
existing Segment allocator's margin matrix and seeded move/exchange objective
then improve composition without relaxing either result. Segment is only an
internal column alias for those unchanged matrix utilities.

Unlabeled-only groups retain their original role. All members of a labeled group
move together. Test identities are copied exactly; test.csv is never consumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack

from .features import sha256
from .segment_development_allocation import AllocationConfig, _development, _matrix, _size_buckets
from .splitting import check_assignments, text
from .target_development_preparation import (
    MISSING_TARGETS, SUPPORTED_TARGETS, PreparedTargetDevelopment, _support, prepare_target_development,
)


VERSION = "target-development-allocation-v1"


def _proposal_version(target):
    return f"{target.casefold()}-development-allocation-v1"


@dataclass(frozen=True)
class TargetAllocationConfig:
    seed: int = 42
    validation_fraction: float = 3 / 17
    restarts: int = 5
    refinement_passes: int = 6
    exchange_passes: int = 2
    margin_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.25, 1.0)
    solver_time_limit: float = 60.0

    def __post_init__(self):
        self.margin_config()  # Reuse the frozen allocator's configuration validation.
        if (isinstance(self.solver_time_limit, bool) or not math.isfinite(self.solver_time_limit)
                or self.solver_time_limit <= 0):
            raise ValueError("solver_time_limit must be positive and finite")

    def margin_config(self):
        return AllocationConfig(seed=self.seed, validation_fraction=self.validation_fraction,
            restarts=self.restarts, refinement_passes=self.refinement_passes,
            exchange_passes=self.exchange_passes, margin_weights=self.margin_weights)


@dataclass(frozen=True)
class TargetAllocation:
    development_assignments: pd.DataFrame
    assignments: pd.DataFrame
    class_coverage: pd.DataFrame
    distribution_differences: pd.DataFrame
    summary: dict


def _inputs(prepared):
    if not isinstance(prepared, PreparedTargetDevelopment) or prepared.target_column not in SUPPORTED_TARGETS:
        raise ValueError(f"Prepared target must be one of {SUPPORTED_TARGETS}")
    target = prepared.target_column
    frame = prepared.frame.sort_values("record_id", kind="stable").reset_index(drop=True).copy()
    required = {"record_id", "group_id", "original_split", "Retailer", target, "is_labeled"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("Prepared target development columns are missing or empty")
    if (frame.record_id.duplicated().any() or frame[["record_id", "group_id"]].isna().any().any()
            or frame.record_id.str.strip().eq("").any() or frame.group_id.str.strip().eq("").any()
            or not frame.original_split.isin(("train", "validation")).all()
            or frame.groupby("group_id").original_split.nunique().gt(1).any()):
        raise ValueError("Prepared development identities or group isolation are invalid")
    if (set(frame.record_id) & set(prepared.protected_test.record_id)
            or set(frame.group_id) & set(prepared.protected_test.group_id)):
        raise ValueError("Protected test identity or group appears in development")
    labeled = ~frame[target].fillna("").str.strip().str.casefold().isin(MISSING_TARGETS)
    if not frame.is_labeled.equals(labeled):
        raise ValueError(f"Prepared supervision mask differs from {target} labels")
    support, pairs, pinned = _support(frame, target)
    if pinned != prepared.required_training_group_ids or not support.equals(prepared.class_support):
        raise ValueError("Prepared class support changed; prepare frozen development again")
    if support.empty:
        raise ValueError(f"{target} development has no labeled records")
    selected = _development(frame.loc[labeled].rename(columns={target: "Segment"}))
    full_sizes = frame.groupby("group_id").size()
    selected["group_size_bucket"] = _size_buckets(selected.group_id.map(full_sizes).to_numpy())
    return frame, selected, support, pinned


def _coverage_problem(matrix, class_count, pinned, config, solver=None):
    """x[group]=validation; y[class]=represented in validation; d=ratio error."""
    group_count = len(matrix)
    incidence = (matrix[:, :class_count] > 0).astype(float).T
    sizes = matrix[:, -1]
    width = group_count + class_count + 1
    # At least one independent group for each class must remain in training.
    training = hstack([csr_matrix(incidence), csr_matrix((class_count, class_count + 1))])
    presence = hstack([-csr_matrix(incidence), csr_matrix(np.eye(class_count)), csr_matrix((class_count, 1))])
    constraints = [LinearConstraint(training, -np.inf, incidence.sum(axis=1) - 1),
                   LinearConstraint(presence, -np.inf, 0)]
    upper = np.ones(width)
    upper[-1] = np.inf
    upper[np.flatnonzero(pinned)] = 0
    lower = np.zeros(width)
    if group_count > 1000:
        # A labeled record count is an integer. Exposing its unavoidable rounding
        # error prevents the large ratio problem from spending its time proving
        # that a fractional record count cannot be attained. Coverage constraints
        # may require a larger error; this remains a valid lower bound then too.
        desired_count = float(sizes.sum() * config.validation_fraction)
        lower[-1] = abs(round(desired_count) - desired_count)
    bounds = Bounds(lower, upper)
    integrality = np.ones(width, dtype=int)
    integrality[-1] = 0

    def solve(cost, extra=()):
        result = (solver or milp)(cost, integrality=integrality, bounds=bounds,
            constraints=[*constraints, *extra], options={"time_limit": config.solver_time_limit,
                                                       "mip_rel_gap": 0.0})
        if not result.success or result.status != 0:
            raise ValueError(f"Target allocation did not prove optimal coverage/constraints: {result.message}")
        return result.x

    cost = np.zeros(width)
    cost[group_count:group_count + class_count] = -1
    solution = solve(cost)
    maximum = int(round(solution[group_count:group_count + class_count].sum()))
    coverage_row = np.zeros(width)
    coverage_row[group_count:group_count + class_count] = 1
    coverage = LinearConstraint(csr_matrix(coverage_row[None, :]), maximum, maximum)
    desired = float(sizes.sum() * config.validation_fraction)
    ratio_rows = np.zeros((2, width))
    ratio_rows[0, :group_count], ratio_rows[0, -1] = sizes, -1
    ratio_rows[1, :group_count], ratio_rows[1, -1] = -sizes, -1
    ratio = LinearConstraint(csr_matrix(ratio_rows), -np.inf, [desired, -desired])
    cost[:] = 0
    cost[-1] = 1
    solution = solve(cost, (coverage, ratio))
    nearest_count = int(round(sizes @ np.rint(solution[:group_count])))
    error = abs(nearest_count - desired)
    # Keep all equally nearest counts available for composition refinement.
    row = np.zeros(width)
    row[:group_count] = sizes
    count_constraint = LinearConstraint(csr_matrix(row[None, :]),
        math.ceil(desired - error - 1e-7), math.floor(desired + error + 1e-7))
    return solve, coverage, count_constraint, maximum, error, solution


def _refine(start, matrix, weights, pinned, class_count, maximum, count_error, config, rng):
    """The frozen allocator's quadratic move/exchange deltas, with hard guards."""
    validation = start.copy()
    actual = matrix[validation].sum(axis=0)
    totals = matrix.sum(axis=0)
    target = totals * config.validation_fraction
    incidence = (matrix[:, :class_count] > 0).astype(int)
    class_groups = incidence.sum(axis=0)
    validation_groups = incidence[validation].sum(axis=0)
    square_cost = matrix ** 2 @ weights
    accepted_moves = accepted_exchanges = 0

    def feasible(group_change, record_change):
        proposed = validation_groups + group_change
        return (np.all(proposed < class_groups) and np.count_nonzero(proposed) == maximum
                and abs(actual[-1] + record_change - target[-1]) <= count_error + 1e-7)

    if len(matrix) > 1000:
        # With a unique nearest integer validation count, a one-group move is
        # infeasible. Pair whole groups with equal labeled counts, as in the
        # frozen allocator's bounded seeded exchange strategy. This avoids an
        # exhaustive O(groups**2) search for Brand and Sub-Segment.
        for _ in range(config.exchange_passes):
            train = np.flatnonzero(~validation & ~pinned)
            valid = np.flatnonzero(validation)
            improved = False
            for size in np.unique(matrix[:, -1]):
                lefts = rng.permutation(train[matrix[train, -1] == size])
                rights = rng.permutation(valid[matrix[valid, -1] == size])
                for left, right in zip(lefts, rights):
                    difference = matrix[left] - matrix[right]
                    delta = 2 * np.dot(difference, (actual - target) * weights) + np.dot(difference ** 2, weights)
                    if delta < -1e-15 and feasible(incidence[left] - incidence[right], difference[-1]):
                        actual += difference
                        validation_groups += incidence[left] - incidence[right]
                        validation[left], validation[right] = True, False
                        accepted_exchanges += 1
                        improved = True
            if not improved:
                break
    else:
        for _ in range(config.refinement_passes):
            improved = False
            for index in rng.permutation(len(matrix)):
                if pinned[index]:
                    continue
                sign = -1 if validation[index] else 1
                delta = sign * 2 * np.dot(matrix[index], (actual - target) * weights) + square_cost[index]
                if delta < -1e-15 and feasible(sign * incidence[index], sign * matrix[index, -1]):
                    actual += sign * matrix[index]
                    validation_groups += sign * incidence[index]
                    validation[index] = not validation[index]
                    accepted_moves += 1
                    improved = True
            if not improved:
                break
        # Platform has few labeled groups: inspect all train/validation group pairs,
        # rather than the frozen allocator's sampled, size-bucket exchange pairs.
        for _ in range(config.exchange_passes):
            improved = False
            for left in rng.permutation(np.flatnonzero(~validation & ~pinned)):
                for right in rng.permutation(np.flatnonzero(validation)):
                    difference = matrix[left] - matrix[right]
                    delta = 2 * np.dot(difference, (actual - target) * weights) + np.dot(difference ** 2, weights)
                    if delta < -1e-15 and feasible(incidence[left] - incidence[right], difference[-1]):
                        actual += difference
                        validation_groups += incidence[left] - incidence[right]
                        validation[left], validation[right] = True, False
                        accepted_exchanges += 1
                        improved = True
                        break
            if not improved:
                break
    score = float(np.dot((actual - target) ** 2, weights))
    return validation, score, accepted_moves, accepted_exchanges


def _reports(support, assignments, config, target):
    coverage = support.copy()
    for role in ("train", "validation"):
        labeled = assignments.loc[assignments.is_labeled & assignments.split.eq(role)]
        coverage[f"{role}_records"] = coverage.label.map(labeled[target].value_counts()).fillna(0).astype(int)
        coverage[f"{role}_groups"] = coverage.label.map(labeled.groupby(target).group_id.nunique()).fillna(0).astype(int)
    coverage["validation_exclusion_reason"] = np.select([
        coverage.validation_records.gt(0), coverage.single_group_training_only,
        coverage.blocked_by_training_only_groups],
        ["", "single_development_group", "shares_required_training_groups"],
        default="joint_class_coverage_constraints")
    differences = []
    labeled = assignments.loc[assignments.is_labeled].copy()
    labeled["retailer"] = labeled.Retailer.map(text).replace("", "<MISSING>")
    for dimension, column in (("class", target), ("retailer", "retailer")):
        pool = labeled[column].value_counts().sort_index()
        for role in ("train", "validation"):
            old = labeled.loc[labeled.original_split.eq(role), column].value_counts()
            new = labeled.loc[labeled.split.eq(role), column].value_counts()
            for name, count in pool.items():
                original = int(old.get(name, 0))
                proposed = int(new.get(name, 0))
                pool_fraction = count / len(labeled)
                old_fraction = original / old.sum() if old.sum() else 0.0
                new_fraction = proposed / new.sum() if new.sum() else 0.0
                differences.append({"dimension": dimension, "value": name, "role": role,
                    "pool_records": int(count), "pool_fraction": float(pool_fraction),
                    "original_records": original, "proposed_records": proposed,
                    "original_fraction": float(old_fraction), "proposed_fraction": float(new_fraction),
                    "original_delta_pp": float(100 * (old_fraction - pool_fraction)),
                    "proposed_delta_pp": float(100 * (new_fraction - pool_fraction))})
    differences = pd.DataFrame(differences)
    summary = {"version": _proposal_version(target), "target_column": target, "config": asdict(config),
        "class_scope": "Development-observed labels only; test labels are not read",
        "development_classes": len(coverage), "training_unsupported_classes": coverage.loc[coverage.train_records.eq(0), "label"].tolist(),
        "validation_unsupported_classes": coverage.loc[coverage.validation_records.eq(0), ["label", "validation_exclusion_reason"]].to_dict("records")}
    for version, split_column, prefix in (("original", "original_split", "original"), ("proposed", "split", "proposed")):
        data = {}
        for role in ("train", "validation"):
            selected = assignments.loc[assignments[split_column].eq(role)]
            supervised = selected.loc[selected.is_labeled]
            data[role] = {"records": len(selected), "groups": int(selected.group_id.nunique()),
                "labeled_records": len(supervised), "labeled_groups": int(supervised.group_id.nunique()),
                "classes": int(supervised[target].nunique()), "missing_target_records": int((~selected.is_labeled).sum())}
            for dimension in ("class", "retailer"):
                values = differences.loc[differences.role.eq(role) & differences.dimension.eq(dimension), f"{prefix}_delta_pp"]
                data[role][f"max_{dimension}_fraction_delta_pp"] = float(values.abs().max())
        data["labeled_validation_fraction"] = data["validation"]["labeled_records"] / len(labeled)
        data["labeled_validation_fraction_delta_pp"] = 100 * (data["labeled_validation_fraction"] - config.validation_fraction)
        summary[version] = data
    summary["changed_development_records"] = int(assignments.split.ne(assignments.original_split).sum())
    summary["changed_development_groups"] = int(assignments.loc[assignments.split.ne(assignments.original_split), "group_id"].nunique())
    if target != "Platform":
        summary["validation_support"] = {
            "original": _validation_support(coverage, "original_validation_records", "original_validation_groups"),
            "proposed": _validation_support(coverage, "validation_records", "validation_groups"),
        }
    return coverage, differences, summary


def _validation_support(coverage, record_column, group_column):
    present = coverage.loc[coverage[record_column].gt(0)]

    def bins(column):
        values = present[column]
        return {"1": int(values.eq(1).sum()), "2": int(values.eq(2).sum()),
                "3": int(values.eq(3).sum()), "4-9": int(values.between(4, 9).sum()),
                "10+": int(values.ge(10).sum())}

    small = present.loc[present[record_column].lt(5)]
    return {"record_support_bins": bins(record_column), "group_support_bins": bins(group_column),
        "classes_below_five_records": small.label.tolist(),
        "classes_below_five_groups": present.loc[present[group_column].lt(5), "label"].tolist(),
        "records_in_classes_below_five": int(small[record_column].sum()),
        "pct_records_in_classes_below_five": float(100 * small[record_column].sum() / present[record_column].sum()) if len(present) else 0.0}


def allocate_target_development(prepared: PreparedTargetDevelopment,
                                config: TargetAllocationConfig | None = None, *, _solver=None) -> TargetAllocation:
    """Allocate the prepared target; neither read nor write any files.

    ``_solver`` preserves the Platform compatibility wrapper's testing seam.
    Coverage and nearest labeled validation count are exact integer stages;
    composition is a seeded heuristic. Large populations start near V4 and
    use bounded exchanges rather than exhaustive group pairs.
    """
    config = config or TargetAllocationConfig()
    if not isinstance(config, TargetAllocationConfig):
        raise TypeError("config must be TargetAllocationConfig")
    frame, selected, support, required = _inputs(prepared)
    target_column = prepared.target_column
    groups, _, _, matrix, weights, families = _matrix(selected, config.margin_config())
    class_count = len(families[0]["categories"])
    # Report the actual target rather than the legacy utility's column alias.
    families = [{**family, "name": target_column if family["name"] == "Segment" else family["name"]}
                for family in families]
    pinned = np.isin(groups, sorted(required))
    solve, coverage_constraint, count_constraint, maximum, error, initial = _coverage_problem(matrix, class_count, pinned, config, _solver)
    if len(groups) > 1000:
        # Find a coverage/count-optimal starting point with minimal V4 group
        # changes. Keeping this same point across seeded restarts bounds solver
        # work; exchange order still varies deterministically by seed.
        original = frame.groupby("group_id").original_split.first().reindex(groups)
        cost = np.zeros(len(initial))
        cost[:len(groups)] = np.where(original.eq("validation"), -1.0, 1.0)
        initial = solve(cost, (coverage_constraint, count_constraint))
    rng = np.random.default_rng(config.seed)
    best = None
    for restart in range(config.restarts):
        if restart == 0 or len(groups) > 1000:
            solution = initial
        else:
            cost = np.zeros(len(initial))
            cost[:len(groups)] = rng.uniform(-1, 1, len(groups))
            solution = solve(cost, (coverage_constraint, count_constraint))
        candidate = _refine(np.rint(solution[:len(groups)]).astype(bool), matrix, weights,
                            pinned, class_count, maximum, error, config, rng)
        if best is None or candidate[1] < best[0][1] - 1e-15:
            best = candidate, restart
    (validation, objective, moves, exchanges), restart = best
    roles = dict(zip(groups, np.where(validation, "validation", "train")))
    assignments = frame.copy()
    assignments["split"] = assignments.group_id.map(roles).fillna(assignments.original_split)
    combined = pd.concat([assignments[["record_id", "group_id", "split"]],
                          prepared.protected_test[["record_id", "group_id", "split"]]], ignore_index=True)
    combined = combined.sort_values("record_id", kind="stable").reset_index(drop=True)
    check_assignments(combined, [*frame.record_id, *prepared.protected_test.record_id])
    if not combined.loc[combined.split.eq("test")].reset_index(drop=True).equals(prepared.protected_test.sort_values("record_id").reset_index(drop=True)):
        raise AssertionError("Protected test membership changed")
    coverage, differences, summary = _reports(support, assignments, config, target_column)
    if coverage.train_records.eq(0).any() or int(coverage.validation_records.gt(0).sum()) != maximum:
        raise AssertionError("Allocation violated optimal class coverage")
    if not assignments.loc[assignments.group_id.isin(required), "split"].eq("train").all():
        raise AssertionError("Required training group moved to validation")
    if set(assignments.split) != {"train", "validation"}:
        raise ValueError("The constraints leave an empty development partition")
    summary["optimization"] = {"maximum_validation_classes": maximum,
        "individual_validation_eligibility_upper_bound": int(support.eligible_for_both.sum()),
        "maximum_coverage_proven": True, "nearest_labeled_validation_count_proven": True,
        "labeled_validation_record_error": float(error), "composition_objective": objective,
        "composition_optimality": "seeded whole-group local search; no global optimality claim",
        "chosen_restart": restart, "accepted_moves": moves, "accepted_exchanges": exchanges,
        "unlabeled_only_groups": "retain original V4 role", "margin_families": families}
    summary["checks"] = {"all_development_records_assigned": True, "group_isolation": True,
        "all_development_classes_in_training": True, "required_training_groups_preserved": True,
        "protected_test_records_unchanged": len(prepared.protected_test),
        "protected_test_groups_unchanged": int(prepared.protected_test.group_id.nunique()),
        "test_csv_read_or_hashed": False, "models_or_predictions_used": False}
    if target_column != "Platform":
        summary["optimization"]["exchange_search"] = (
            "seeded equal-labeled-size group pairs" if len(groups) > 1000 else "exhaustive group pairs")
        summary["optimization"]["starting_point"] = (
            "minimum V4 group changes under optimal coverage/count constraints" if len(groups) > 1000
            else "integer coverage/count solution")
    return TargetAllocation(assignments, combined, coverage, differences, summary)


def write_target_proposal(split_directory, output_directory, target_column: str,
                          config: TargetAllocationConfig | None = None, *, _allocator=None) -> TargetAllocation:
    """Verify frozen inputs, allocate, then seal a new review-only proposal."""
    if target_column not in SUPPORTED_TARGETS:
        raise ValueError(f"target_column must be one of {SUPPORTED_TARGETS}")
    source, output = Path(split_directory).resolve(), Path(output_directory).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a new proposal directory outside the frozen split directory")
    prepared = prepare_target_development(source, target_column)
    result = (_allocator or allocate_target_development)(prepared, config)
    expected = {**prepared.audit["consumed_artifact_sha256"], "completion.json": prepared.audit["completion_sha256"]}

    def verify_inputs():
        if any(sha256(source / name) != digest for name, digest in expected.items()):
            raise ValueError("Frozen inputs changed during target allocation")

    verify_inputs()
    output.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")

    result.development_assignments.to_csv(output / "development_assignments.csv", index=False, lineterminator="\n")
    result.assignments.to_csv(output / "assignments.csv", index=False, lineterminator="\n")
    result.class_coverage.to_csv(output / "class_coverage.csv", index=False, lineterminator="\n")
    result.distribution_differences.to_csv(output / "distribution_differences.csv", index=False, lineterminator="\n")
    result.assignments.groupby(["group_id", "split"], sort=True).size().rename("records").reset_index().to_csv(output / "group_assignments.csv", index=False, lineterminator="\n")
    shutil.copyfile(source / "protected_final_test.csv", output / "protected_final_test.csv")
    if sha256(output / "protected_final_test.csv") != expected["protected_final_test.csv"]:
        raise ValueError("Protected test identity file changed while copying")
    write_json("summary.json", result.summary)
    sources = [Path(__file__), Path(__file__).with_name("target_development_preparation.py"),
               Path(__file__).with_name("segment_development_allocation.py")]
    write_json("protocol.json", {"version": result.summary["version"], "status": "proposed; pending review",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_split_directory": str(source),
        "preparation": prepared.audit, "config": result.summary["config"],
        "implementation_sha256": {path.name: sha256(path) for path in sources},
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "policy": f"{target_column} development roles only; frozen groups and test IDs unchanged; no test labels or predictions used"})
    (output / "README.md").write_text(
        f"# Proposed {target_column} development allocation\n\n"
        f"Review-only {result.summary['version']}. This is not a replacement Segment V4 protocol.\n"
        "assignments.csv contains all development and protected-test identities.\n"
        f"development_assignments.csv retains original_split, raw {target_column} labels and supervision eligibility.\n"
        "Test features/labels remain in the unchanged source directory; no test.csv was read or copied.\n"
        "See summary.json, class_coverage.csv and distribution_differences.csv before adopting.\n",
        encoding="utf8")
    verify_inputs()
    write_json("completion.json", {"version": result.summary["version"], "status": "proposed; pending review",
        "artifact_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    return result
