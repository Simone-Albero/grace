"""The euclidean space every distance of the method is measured in.

The prepared numeric columns as they are, plus a one-hot block per
categorical column for its top_k most frequent codes of train and one
shared "other" slot. A slot is worth sqrt(cat_cost / 2), so two rows
that differ on one categorical column are cat_cost apart in squared
distance. Under the cosine distance every row is then scaled to unit
length.
"""

from __future__ import annotations

__all__ = ["Space", "embed", "fit_space", "scale_for_metric"]

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Space:
    """The kept codes of each categorical column and the slot weight.

    Attributes:
        kept: The kept codes of each categorical column, ascending.
        weight: The value of an active slot, sqrt(cat_cost / 2).
    """

    kept: tuple[np.ndarray, ...]
    weight: float


def fit_space(cat: np.ndarray, top_k: int, cat_cost: float) -> Space:
    """Learn the top_k most frequent codes of each categorical column.

    Args:
        cat: The categorical codes of train, shape (n, n_cat); n_cat may
            be 0.
        top_k: The codes per column with their own slot (ties: the
            smaller code first); 0 leaves only the "other" slot.
        cat_cost: The squared distance a mismatch adds, > 0.

    Returns:
        The fitted space.
    """
    kept = []
    for column in cat.T:
        codes, counts = np.unique(column, return_counts=True)
        order = np.lexsort((codes, -counts))
        kept.append(np.sort(codes[order[:top_k]]))
    return Space(kept=tuple(kept), weight=math.sqrt(cat_cost / 2))


def embed(space: Space, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Place rows in the space.

    Args:
        space: The fitted space.
        num: The prepared numeric columns, shape (n, n_num).
        cat: The categorical codes, shape (n, n_cat).

    Returns:
        Float64 points of shape (n, n_num + Σ (len(kept) + 1)): num,
        then per column one slot per kept code and a last "other" slot,
        exactly one of them set to space.weight.
    """
    blocks = [np.asarray(num, dtype=np.float64)]
    for kept, column in zip(space.kept, cat.T):
        position = np.searchsorted(kept, column)
        found = position < len(kept)
        found[found] = kept[position[found]] == column[found]
        slot = np.where(found, position, len(kept))
        block = np.zeros((len(column), len(kept) + 1))
        block[np.arange(len(column)), slot] = space.weight
        blocks.append(block)
    return np.hstack(blocks)


def scale_for_metric(points: np.ndarray, metric: str) -> np.ndarray:
    """Scale points so that euclidean distances serve the metric.

    Args:
        points: Shape (n, d).
        metric: "euclidean" (unchanged) or "cosine" (each row divided by
            max(its norm, 1e-8)).

    Returns:
        A new array of the same shape.
    """
    if metric == "cosine":
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-8)
    return np.array(points, dtype=np.float64)
