"""The sample and aggregation helpers of the complexity families."""

__all__ = [
    "MeasuredSample",
    "compute_centroids",
    "summarize",
    "weighted_group_measures",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasuredSample:
    """The rows sampled from each population, with their graph nodes.

    Attributes:
        X: The sampled rows, embedded by `Space`, shape (n, d).
        population: The population id of every sampled row.
        neighbors: The nearest graph nodes of every row, as indices into
            the graph, shape (n, k).
        neighbor_dist: The distances to those nodes, shape (n, k).
    """

    X: np.ndarray
    population: np.ndarray
    neighbors: np.ndarray
    neighbor_dist: np.ndarray


def summarize(key: str, values: Sequence[float]) -> dict[str, float | None]:
    """Return the min, mean and max of `values`, as `<key>_min` etc.

    Args:
        key: The descriptor name, e.g. "f1".
        values: One value per rival; may be empty.

    Returns:
        The three entries, in that order; None each when `values` is
        empty.
    """
    names = (f"{key}_min", f"{key}_mean", f"{key}_max")
    if not values:
        return dict.fromkeys(names)
    array = np.asarray(values, dtype=np.float64)
    stats = (float(array.min()), float(array.mean()), float(array.max()))
    return dict(zip(names, stats))


def weighted_group_measures(
    measures: Mapping[int, Mapping[str, float | None]],
    group: Mapping[int, int],
    weight: Mapping[int, int],
) -> dict[int, dict[str, float | None]]:
    """Average each group's population measures, weighted by rows.

    Args:
        measures: The measures of every population, by population id.
        group: The group of every population in `measures`.
        weight: The rows of every population in `measures`.

    Returns:
        The weighted mean of each measure by group, over the populations
        where it is defined; None where none defines it.
    """
    members: dict[int, list[int]] = {}
    for population in measures:
        members.setdefault(group[population], []).append(population)
    grouped = {}
    for group_id, populations in members.items():
        group_measures = {}
        for name in next(iter(measures.values())):
            defined = [
                population
                for population in populations
                if measures[population][name] is not None
            ]
            total = sum(weight[population] for population in defined)
            weighted_sum = sum(
                weight[population] * measures[population][name]
                for population in defined
            )
            group_measures[name] = weighted_sum / total if total else None
        grouped[group_id] = group_measures
    return grouped


def compute_centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return the mean point of every label.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row.

    Returns:
        The centroid by label.
    """
    return {int(label): X[labels == label].mean(axis=0) for label in np.unique(labels)}
