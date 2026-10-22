"""The sample and aggregation helpers of the complexity families."""

__all__ = ["MeasuredSample", "summarize"]

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasuredSample:
    """The rows sampled from each population, with their graph nodes.

    Attributes:
        X: The sampled rows in the clustering space, shape (n, d).
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
