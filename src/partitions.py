"""Group failure-rate estimation and equal-mass confidence binning."""

__all__ = [
    "assign_bins",
    "assign_class_bins",
    "class_confidence_edges",
    "confidence_edges",
    "estimate_group_rates",
]

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def estimate_group_rates(
    group: np.ndarray, observed: np.ndarray, group_ids: Sequence[int]
) -> pd.Series:
    """Estimate each group's failure rate from its own rows.

    A group with no row takes the global rate instead.

    Args:
        group: Each scored row's group id.
        observed: Each scored row's failure, 0 or 1.
        group_ids: Every group to return a rate for.

    Returns:
        The estimated rate of every id in `group_ids`, indexed by it.
    """
    global_rate = float(np.mean(observed))
    rates = (
        pd.Series(np.asarray(observed, dtype=float)).groupby(np.asarray(group)).mean()
    )
    return rates.reindex(group_ids).fillna(global_rate)


def confidence_edges(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    """Return equal-mass bin edges of `confidence`, ties merged.

    Args:
        confidence: The confidence of every row.
        n_bins: The bins asked for; fewer survive where ties collapse
            edges.

    Returns:
        The ascending, deduplicated edges, `n_bins + 1` at most.
    """
    probabilities = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(confidence, probabilities, method="inverted_cdf")
    return np.unique(edges)


def assign_bins(confidence: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each row to its equal-mass confidence bin.

    Args:
        confidence: The confidence of every row.
        edges: The ascending bin edges, as `confidence_edges` returns.

    Returns:
        Each row's bin index, in `[0, max(len(edges) - 2, 0)]`: a single
        edge (every confidence tied) is one bin, index 0.
    """
    bins = np.searchsorted(edges, confidence, side="right") - 1
    return np.clip(bins, 0, max(len(edges) - 2, 0))


def _apportion(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` whole units across `weights`, each getting one+.

    The largest-remainder method rounds each share down, then hands the
    units the floor dropped to the shares with the largest remainder.
    Forcing every share to at least one can overshoot `total`; when it
    does, units are taken back from the largest shares instead, never
    below one. Either way the shares sum to exactly `total`.

    Args:
        total: The units to split; at least `len(weights)`.
        weights: Each share's weight, not necessarily normalized.

    Returns:
        Each share's whole units.
    """
    raw = total * weights / weights.sum()
    allocated = np.maximum(1, np.floor(raw).astype(int))
    surplus = int(allocated.sum()) - total
    if surplus > 0:
        order = np.argsort(-allocated)
        i = 0
        while surplus > 0:
            candidate = order[i % len(order)]
            if allocated[candidate] > 1:
                allocated[candidate] -= 1
                surplus -= 1
            i += 1
    elif surplus < 0:
        remainder = raw - np.floor(raw)
        for candidate in np.argsort(-remainder)[:-surplus]:
            allocated[candidate] += 1
    return allocated


def class_confidence_edges(
    confidence: np.ndarray, predicted_class: np.ndarray, n_bins: int
) -> dict[int, np.ndarray]:
    """Return equal-mass confidence edges inside every predicted class.

    Each class's bin count is its largest-remainder share of `n_bins`
    by row count, at least one, so the shares sum to exactly `n_bins`.

    Args:
        confidence: The confidence of every row.
        predicted_class: The predicted class of every row.
        n_bins: The bins to share out across classes.

    Returns:
        The edges of every class seen in `predicted_class`, by class.

    Raises:
        ValueError: Fewer bins than classes: one bin per class is not
            possible.
    """
    predicted_class = np.asarray(predicted_class)
    classes, counts = np.unique(predicted_class, return_counts=True)
    if n_bins < len(classes):
        raise ValueError(
            f"n_bins={n_bins} cannot give each of {len(classes)} classes a bin."
        )
    n_bins_by_class = _apportion(n_bins, counts)
    return {
        int(class_id): confidence_edges(
            confidence[predicted_class == class_id], int(n_class_bins)
        )
        for class_id, n_class_bins in zip(classes, n_bins_by_class)
    }


def assign_class_bins(
    confidence: np.ndarray,
    predicted_class: np.ndarray,
    edges: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Assign each row a group id unique across every predicted class.

    A class absent from `edges` (seen here but not where `edges` was
    built, such as a class predicted on test but never on val) gets a
    single id of its own instead of being split by confidence, so a
    caller grouping by this id can still fall back to a global rate for
    it rather than raising.

    Args:
        confidence: The confidence of every row.
        predicted_class: The predicted class of every row.
        edges: Each class's bin edges, as `class_confidence_edges`
            returns.

    Returns:
        Each row's group id. Ids never repeat across classes: unlike a
        plain bin index, they can be grouped on directly without
        pooling two classes' same-numbered bins together.
    """
    predicted_class = np.asarray(predicted_class)
    group = np.empty(len(confidence), dtype=np.int64)
    offset = 0
    for class_id in np.unique(predicted_class):
        in_class = predicted_class == class_id
        if class_id in edges:
            group[in_class] = offset + assign_bins(
                confidence[in_class], edges[class_id]
            )
            offset += max(len(edges[class_id]) - 1, 1)
        else:
            group[in_class] = offset
            offset += 1
    return group
