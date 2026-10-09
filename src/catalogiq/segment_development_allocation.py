"""Experiment-local allocation of an already frozen development population.

This adapter never creates product groups or selects development records. It
balances record-level population margins while moving complete supplied groups.
The final test population is not an input to any function in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .splitting import text

ASSIGNMENT_COLUMNS = ("record_id", "group_id", "Segment", "split")
SIZE_BUCKETS = ("1", "2", "3-5", "6-10", "11+")


@dataclass(frozen=True)
class AllocationConfig:
    seed: int = 42
    validation_fraction: float = 3 / 17
    restarts: int = 5
    refinement_passes: int = 6
    exchange_passes: int = 2
    # Each independent margin family has equal aggregate weight by default.
    # Within a family, relative squared errors are averaged over its categories.
    margin_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    max_retailer_fraction_delta: float = .02
    max_group_size_fraction_delta: float = .02
    max_segment_fraction_delta: float = .005
    max_validation_fraction_delta: float = .001

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if (isinstance(self.validation_fraction, bool)
                or not math.isfinite(self.validation_fraction)
                or not 0 < self.validation_fraction < 1):
            raise ValueError("validation_fraction must lie strictly between zero and one")
        for field in ("restarts", "refinement_passes", "exchange_passes"):
            value = getattr(self, field)
            if type(value) is not int or value < (1 if field == "restarts" else 0):
                raise ValueError(f"{field} must be an appropriate nonnegative integer")
        if (len(self.margin_weights) != 4
                or any(isinstance(w, bool) or not math.isfinite(w) or w <= 0
                       for w in self.margin_weights)):
            raise ValueError("four positive finite margin weights are required")
        for field in ("max_retailer_fraction_delta", "max_group_size_fraction_delta",
                      "max_segment_fraction_delta", "max_validation_fraction_delta"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{field} must be a finite fraction in [0, 1)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "margin_weights" in values:
            values["margin_weights"] = tuple(values["margin_weights"])
        return cls(**values)


def _development(frame):
    required = {"record_id", "group_id", "Segment", "Retailer"}
    if not required.issubset(frame.columns):
        raise ValueError(f"development records require {sorted(required)}")
    selected = frame.loc[:, ["record_id", "group_id", "Segment", "Retailer"]].copy()
    if selected.empty or selected.isna().any().any():
        raise ValueError("development records must be nonempty and have no null values")
    selected = selected.astype(str).sort_values("record_id", kind="stable").reset_index(drop=True)
    if selected.record_id.duplicated().any():
        raise ValueError("record IDs must be unique")
    for field in ("record_id", "group_id", "Segment"):
        if selected[field].str.strip().str.casefold().isin(("", "null", "<missing>")).any():
            raise ValueError(f"development {field} must be present; missing labels are not a class")
    selected["retailer"] = selected.Retailer.map(text).replace("", "<MISSING>")
    sizes = selected.groupby("group_id", sort=True).size()
    selected["development_group_size"] = selected.group_id.map(sizes).astype(int)
    selected["group_size_bucket"] = _size_buckets(selected.development_group_size.to_numpy())
    return selected


def _size_buckets(sizes):
    return np.select([sizes == 1, sizes == 2, sizes <= 5, sizes <= 10],
                     SIZE_BUCKETS[:4], default=SIZE_BUCKETS[4])


def _matrix(frame, config):
    groups, inverse = np.unique(frame.group_id.to_numpy(), return_inverse=True)
    pieces, weights, families = [], [], []
    for field, family_weight in zip(("Segment", "retailer", "group_size_bucket"),
                                    config.margin_weights[:3]):
        categories, codes = np.unique(frame[field].to_numpy(), return_inverse=True)
        counts = np.zeros((len(groups), len(categories)), dtype=np.int64)
        np.add.at(counts, (inverse, codes), 1)
        totals = counts.sum(axis=0).astype(float)
        pieces.append(counts)
        weights.extend(family_weight / (len(categories) * totals ** 2))
        families.append({"name": field, "categories": categories.tolist()})
    sizes = np.bincount(inverse, minlength=len(groups)).astype(np.int64)
    pieces.append(sizes[:, None])
    weights.append(config.margin_weights[3] / float(len(frame)) ** 2)
    matrix = np.concatenate(pieces, axis=1).astype(float)
    return groups, inverse, sizes, matrix, np.asarray(weights), families


def _assignment(frame, inverse, validation):
    result = frame.loc[:, ["record_id", "group_id", "Segment"]].copy()
    result["split"] = np.where(validation[inverse], "validation", "train")
    return result


def _population_diagnostics(frame, assignment, config):
    aligned = assignment.set_index("record_id").loc[frame.record_id]
    validation = aligned.split.eq("validation").to_numpy()
    counts = {"train": int((~validation).sum()), "validation": int(validation.sum())}
    if not all(counts.values()):
        raise ValueError("both development partitions must contain records")
    diagnostics = {"records": len(frame), "groups": int(frame.group_id.nunique()),
        "train_records": counts["train"], "validation_records": counts["validation"],
        "validation_fraction": counts["validation"] / len(frame),
        "validation_fraction_delta": abs(counts["validation"] / len(frame)
                                           - config.validation_fraction),
        "margins": {}, "failed_checks": []}
    if diagnostics["validation_fraction_delta"] > config.max_validation_fraction_delta:
        diagnostics["failed_checks"].append("validation_record_fraction")
    limits = {"Segment": config.max_segment_fraction_delta,
              "retailer": config.max_retailer_fraction_delta,
              "group_size_bucket": config.max_group_size_fraction_delta}
    for field, limit in limits.items():
        totals = frame[field].value_counts().sort_index()
        fractions = totals / len(frame)
        partition_data = {}
        for role, mask in (("train", ~validation), ("validation", validation)):
            actual = frame.loc[mask, field].value_counts().reindex(totals.index, fill_value=0)
            actual_fraction = actual / counts[role]
            deltas = abs(actual_fraction - fractions)
            partition_data[role] = {
                "counts": {str(k): int(v) for k, v in actual.items()},
                "fractions": {str(k): float(v) for k, v in actual_fraction.items()},
                "maximum_fraction_delta": float(deltas.max())}
            if float(deltas.max()) > limit:
                diagnostics["failed_checks"].append(f"{role}_{field}_composition")
            if field == "Segment" and actual.eq(0).any():
                diagnostics["failed_checks"].append(f"{role}_missing_Segment")
        diagnostics["margins"][field] = {"pool_counts": {
            str(k): int(v) for k, v in totals.items()},
            "pool_fractions": {str(k): float(v) for k, v in fractions.items()},
            **partition_data}
    return diagnostics


def validate_development_population(frame, assignments, config=None, *,
                                    require_group_isolation=True):
    """Check shared population identity and pre-training composition safeguards.

    Random-control assignments intentionally may cross supplied product groups;
    callers must explicitly disable isolation for that strategy only.
    """
    config = AllocationConfig() if config is None else config
    if not isinstance(config, AllocationConfig):
        raise TypeError("config must be AllocationConfig")
    if type(require_group_isolation) is not bool:
        raise ValueError("require_group_isolation must be a boolean")
    selected = _development(frame)
    if not set(ASSIGNMENT_COLUMNS).issubset(assignments.columns):
        raise ValueError("assignments require record_id, group_id, Segment, split")
    if (assignments.record_id.duplicated().any()
            or set(assignments.record_id) != set(selected.record_id)):
        raise ValueError("assignments must cover exactly the development population")
    if not assignments.split.isin(("train", "validation")).all():
        raise ValueError("development assignments permit training and validation only")
    aligned = assignments.set_index("record_id").loc[selected.record_id]
    if (aligned.group_id.tolist() != selected.group_id.tolist()
            or aligned.Segment.tolist() != selected.Segment.tolist()):
        raise ValueError("supplied product groups and labels must remain unchanged")
    crossings = int(assignments.groupby("group_id").split.nunique().gt(1).sum())
    if require_group_isolation and crossings:
        raise ValueError("a supplied product group crosses development partitions")
    diagnostics = _population_diagnostics(selected, assignments, config)
    diagnostics["groups_crossing_train_validation"] = crossings
    if diagnostics["failed_checks"]:
        raise ValueError("development population safeguards failed: "
                         + ", ".join(diagnostics["failed_checks"]))
    return diagnostics


def allocate_development_groups(frame, config=None):
    """Assign all supplied records using randomized whole-group allocation.

    Seeded starts precede deterministic objective-improving whole-group moves and
    exchanges. Independent Segment, retailer, size-bucket and record-count
    margins each have configurable aggregate weight. No size-order packing or
    similarity comparison is used. Infeasible safeguards fail before training.
    """
    config = AllocationConfig() if config is None else config
    if not isinstance(config, AllocationConfig):
        raise TypeError("config must be AllocationConfig")
    selected = _development(frame)
    if selected.groupby("Segment").group_id.nunique().lt(2).any():
        raise ValueError("each Segment needs at least two supplied groups for isolated validation")
    groups, inverse, sizes, matrix, weights, families = _matrix(selected, config)
    target = matrix.sum(axis=0) * config.validation_fraction
    square_cost = matrix ** 2 @ weights
    buckets = _size_buckets(sizes)
    rng = np.random.default_rng(config.seed)
    best = None
    best_failed = None
    for restart in range(config.restarts):
        validation = rng.random(len(groups)) < config.validation_fraction
        actual = matrix[validation].sum(axis=0)
        moves = exchanges = 0
        for _ in range(config.refinement_passes):
            improved = False
            for index in rng.permutation(len(groups)):
                sign = -1 if validation[index] else 1
                delta = sign * 2 * np.dot(matrix[index], (actual - target) * weights) + square_cost[index]
                if delta < -1e-15:
                    actual += sign * matrix[index]
                    validation[index] = not validation[index]
                    moves += 1
                    improved = True
            if not improved:
                break
        # Exchanges can improve class/retailer balance at a one-move local
        # optimum. Seeded pairings within size buckets bound computation without
        # bounding product-group size or altering membership.
        for _ in range(config.exchange_passes):
            for bucket in SIZE_BUCKETS:
                train = rng.permutation(np.flatnonzero((buckets == bucket) & ~validation))
                valid = rng.permutation(np.flatnonzero((buckets == bucket) & validation))
                for left, right in zip(train, valid):
                    difference = matrix[left] - matrix[right]
                    delta = (2 * np.dot(difference, (actual - target) * weights)
                             + np.dot(difference ** 2, weights))
                    if delta < -1e-15:
                        actual += difference
                        validation[left], validation[right] = True, False
                        exchanges += 1
        result = _assignment(selected, inverse, validation)
        score = float(np.dot((actual - target) ** 2, weights))
        try:
            diagnostics = _population_diagnostics(selected, result, config)
        except ValueError:
            continue
        candidate = (score, result, diagnostics, restart, moves, exchanges)
        if diagnostics["failed_checks"]:
            if best_failed is None or score < best_failed[0]:
                best_failed = candidate
        elif best is None or score < best[0]:
            best = candidate
    if best is None:
        failed = "no nonempty two-partition allocation"
        if best_failed is not None:
            failed = ", ".join(best_failed[2]["failed_checks"])
        raise ValueError("no development allocation satisfied pre-training safeguards: " + failed)
    score, result, diagnostics, restart, moves, exchanges = best
    stats = validate_development_population(selected, result, config)
    stats.update({"objective": score, "chosen_restart": restart,
                  "accepted_whole_group_moves": moves,
                  "accepted_whole_group_exchanges": exchanges,
                  "margin_families": families,
                  "config": config.to_dict(), "grouping_unchanged": True,
                  "all_input_records_assigned": True})
    return result, stats


def random_control(allocated, seed=42):
    """Seeded row-random control matching every Segment's role counts exactly."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if (not set(ASSIGNMENT_COLUMNS).issubset(allocated.columns)
            or allocated.empty or allocated.record_id.duplicated().any()
            or allocated.isna().any().any()
            or not allocated.split.isin(("train", "validation")).all()):
        raise ValueError("a complete valid development allocation is required")
    result = allocated.loc[:, list(ASSIGNMENT_COLUMNS)].sort_values(
        "record_id", kind="stable").reset_index(drop=True).copy()
    if result.Segment.astype(str).str.strip().str.casefold().isin(("", "null", "<missing>")).any():
        raise ValueError("missing Segment cannot become a class")
    rng = np.random.default_rng(seed)
    for _, positions in result.groupby("Segment", sort=True).groups.items():
        positions = np.asarray(positions, dtype=int)
        count = int(result.loc[positions, "split"].eq("validation").sum())
        if count == 0 or count == len(positions):
            raise ValueError("each Segment must appear in both development partitions")
        selected = rng.permutation(positions)[:count]
        result.loc[positions, "split"] = "train"
        result.loc[selected, "split"] = "validation"
    return result
