"""F1-F4 feature overlap."""

__all__ = ["compute_f_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.complexity.common import summarize

_EPS = 1e-8
_F_KEYS = ("f1", "f2", "f3", "f4")


def _f1_pair(own_rows: np.ndarray, rival_rows: np.ndarray) -> float:
    """Return F1, 1/(1 + max Fisher ratio); higher is harder."""
    mean_own = own_rows.mean(axis=0)
    mean_rival = rival_rows.mean(axis=0)
    var_own = own_rows.var(axis=0)
    var_rival = rival_rows.var(axis=0)
    fisher = (mean_own - mean_rival) ** 2 / (var_own + var_rival + _EPS)
    return float(1.0 / (1.0 + np.max(fisher)))


def _f2_pair(own_rows: np.ndarray, rival_rows: np.ndarray) -> float:
    """Return F2, the mean per-feature box overlap; higher is harder."""
    min_own, max_own = own_rows.min(axis=0), own_rows.max(axis=0)
    min_rival, max_rival = rival_rows.min(axis=0), rival_rows.max(axis=0)
    overlap = np.maximum(
        0.0, np.minimum(max_own, max_rival) - np.maximum(min_own, min_rival)
    )
    total_range = np.maximum(max_own, max_rival) - np.minimum(min_own, min_rival) + _EPS
    return float(np.mean(overlap / total_range))


def _f3_pair(own_rows: np.ndarray, rival_rows: np.ndarray) -> float:
    """Return F3, least own share in the overlap; higher is harder."""
    min_own, max_own = own_rows.min(axis=0), own_rows.max(axis=0)
    min_rival, max_rival = rival_rows.min(axis=0), rival_rows.max(axis=0)
    lo = np.maximum(min_own, min_rival)
    hi = np.minimum(max_own, max_rival)
    fractions = []
    for feature in range(own_rows.shape[1]):
        if hi[feature] >= lo[feature]:
            fraction = float(
                np.mean(
                    (own_rows[:, feature] >= lo[feature])
                    & (own_rows[:, feature] <= hi[feature])
                )
            )
        else:
            fraction = 0.0
        fractions.append(fraction)
    return float(np.min(fractions))


def _f4_pair(own_rows: np.ndarray, rival_rows: np.ndarray) -> float:
    """Return F4, the own share in every overlap; higher is harder."""
    min_own, max_own = own_rows.min(axis=0), own_rows.max(axis=0)
    min_rival, max_rival = rival_rows.min(axis=0), rival_rows.max(axis=0)
    lo = np.maximum(min_own, min_rival)
    hi = np.minimum(max_own, max_rival)
    if np.any(hi < lo):
        return 0.0
    in_all = np.ones(len(own_rows), dtype=bool)
    for feature in range(own_rows.shape[1]):
        in_all &= (own_rows[:, feature] >= lo[feature]) & (
            own_rows[:, feature] <= hi[feature]
        )
    return float(np.mean(in_all))


def _pair_measures(
    own_rows: np.ndarray, rivals_rows: Sequence[np.ndarray]
) -> dict[str, list[float]]:
    """Return F1-F4 of a population against each of its rivals."""
    values_by_key: dict[str, list[float]] = {key: [] for key in _F_KEYS}
    for rival_rows in rivals_rows:
        if len(rival_rows) < 2:
            continue
        values_by_key["f1"].append(_f1_pair(own_rows, rival_rows))
        values_by_key["f2"].append(_f2_pair(own_rows, rival_rows))
        values_by_key["f3"].append(_f3_pair(own_rows, rival_rows))
        values_by_key["f4"].append(_f4_pair(own_rows, rival_rows))
    return values_by_key


def compute_f_measures(
    points: np.ndarray,
    population: np.ndarray,
    rivals: Mapping[int, Sequence[int]],
) -> dict[int, dict[str, float | None]]:
    """Return F1-F4 per population against its nearest rivals.

    Args:
        points: The sampled rows, shape (n, d).
        population: The population id of every row of `points`.
        rivals: The nearest rival ids of every population.

    Returns:
        By population id, the min, mean and max of each measure over
        its rivals; None each when the population has fewer than two
        rows or none of its rivals does.
    """
    rows_by_population = {
        int(population_id): points[population == population_id]
        for population_id in np.unique(population)
    }

    result: dict[int, dict[str, float | None]] = {}
    for population_id, own_rows in tqdm(
        rows_by_population.items(), desc="F measures", unit="pop", leave=False
    ):
        if len(own_rows) < 2 or own_rows.shape[1] == 0:
            pair_values = {key: [] for key in _F_KEYS}
        else:
            pair_values = _pair_measures(
                own_rows, [rows_by_population[rival] for rival in rivals[population_id]]
            )
        row: dict[str, float | None] = {}
        for key in _F_KEYS:
            row |= summarize(key, pair_values[key])
        result[population_id] = row

    return result
