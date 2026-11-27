"""Uniform samples of rows: sample_rows draws the rows a class is read on."""

from __future__ import annotations

__all__ = ["sample_rows"]

import numpy as np


def sample_rows(n: int, max_samples: int | None, seed: int) -> np.ndarray:
    """Return a uniform sample of range(n), ascending.

    The sample is the first max_samples rows of one random order of
    range(n), fixed by seed: a larger max_samples keeps the same rows
    and more (nested samples).

    Args:
        n: The rows to sample from.
        max_samples: The most rows kept; every row when None or when
            n <= max_samples.
        seed: The seed of the order.

    Returns:
        The kept positions, int64, ascending.
    """
    if max_samples is None or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[:max_samples]).astype(np.int64)
