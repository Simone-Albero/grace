"""The points of the space and uniform samples of rows.

space_points turns the prepared rows into the points every distance is
measured on; past it every distance is euclidean. sample_rows draws a
uniform sample of rows.
"""

from __future__ import annotations

__all__ = [
    "sample_rows",
    "space_points",
]

import numpy as np


def space_points(numbers: np.ndarray, slots: np.ndarray, metric: str) -> np.ndarray:
    """Return the points every distance of the method is measured on.

    The metric applies to the numbers only: under "cosine" each row of
    numbers is divided by max(its norm, 1e-8), so that the euclidean
    distance of two rows is sqrt(2 (1 - cos)); under "euclidean" they
    stay as they are. The categorical slots follow unchanged, so a
    mismatch costs the same squared distance under either metric.

    Args:
        numbers: The prepared numeric columns, shape (n, d_num).
        slots: The prepared one-hot slots, shape (n, d_cat); may have
            no column.
        metric: "euclidean" or "cosine".

    Returns:
        Float64 points, shape (n, d_num + d_cat).
    """
    numbers = np.asarray(numbers, dtype=np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(numbers, axis=1, keepdims=True)
        numbers = numbers / np.maximum(norms, 1e-8)
    return np.hstack([numbers, np.asarray(slots, dtype=np.float64)])


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
