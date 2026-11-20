"""The points of the space and uniform samples of rows.

space_points turns the prepared rows into the points every distance,
euclidean, is measured on. sample_rows draws a uniform sample of rows.
"""

from __future__ import annotations

__all__ = [
    "sample_rows",
    "space_points",
]

import numpy as np


def space_points(numbers: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Return the points every distance of the method is measured on.

    The prepared numbers, then the categorical slots, as they are: the
    distance of two rows is the euclidean one of their points.

    Args:
        numbers: The prepared numeric columns, shape (n, d_num).
        slots: The prepared one-hot slots, shape (n, d_cat); may have
            no column.

    Returns:
        Float64 points, shape (n, d_num + d_cat).
    """
    return np.hstack(
        [np.asarray(numbers, dtype=np.float64), np.asarray(slots, dtype=np.float64)]
    )


def sample_rows(n: int, max_samples: int, seed: int) -> np.ndarray:
    """Return a uniform sample of range(n), ascending.

    Args:
        n: The rows to sample from.
        max_samples: The most rows kept; every row when n <=
            max_samples.
        seed: The seed of the draw (without replacement).

    Returns:
        The kept positions, int64, ascending.
    """
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False)).astype(np.int64)
