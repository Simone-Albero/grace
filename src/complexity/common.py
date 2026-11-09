"""The sample the complexity families read, and their summary."""

from __future__ import annotations

__all__ = ["MeasuredSample", "summarize", "summary_keys"]

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


def summary_keys(key: str) -> tuple[str, str, str]:
    """Return the names summarize gives the min, mean and max of key.

    Args:
        key: The descriptor name, e.g. "f1".

    Returns:
        `<key>_min`, `<key>_mean` and `<key>_max`.
    """
    return (f"{key}_min", f"{key}_mean", f"{key}_max")


def summarize(key: str, values: Sequence[float]) -> dict[str, float | None]:
    """Return the min, mean and max of `values`, as `<key>_min` etc.

    Args:
        key: The descriptor name, e.g. "f1".
        values: One value per rival; may be empty.

    Returns:
        The three entries, in that order; None each when `values` is
        empty.
    """
    names = summary_keys(key)
    if not values:
        return dict.fromkeys(names)
    array = np.asarray(values, dtype=np.float64)
    stats = (float(array.min()), float(array.mean()), float(array.max()))
    return dict(zip(names, stats))
