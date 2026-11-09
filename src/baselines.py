"""Failure rates per region: observed, and from confidence alone.

MCP and ATC read only the classifier's maximum probability. The observed
rate is the target of every predictor; it needs the test labels.
"""

from __future__ import annotations

__all__ = ["atc_rate", "atc_threshold", "failure_rates", "mcp_rate"]

from collections.abc import Callable

import numpy as np


def failure_rates(
    region: np.ndarray, error: np.ndarray, regions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count the rows and errors of every region.

    Args:
        region: The region of every row.
        error: Whether the classifier got every row wrong.
        regions: The regions to report, in output order.

    Returns:
        n_eval (int64), n_error (int64) and n_error / n_eval (NaN when
        n_eval is 0), one value per region of regions.
    """
    n_eval = _by_region(np.ones(len(region)), region, regions, np.sum).astype(np.int64)
    n_error = _by_region(error.astype(np.float64), region, regions, np.sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n_eval > 0, n_error / n_eval, np.nan)
    return n_eval, n_error.astype(np.int64), rate


def mcp_rate(
    confidence: np.ndarray, region: np.ndarray, regions: np.ndarray
) -> np.ndarray:
    """Return the mean of 1 - confidence over the rows of every region.

    Args:
        confidence: The maximum class probability of every row.
        region: The region of every row.
        regions: The regions to report, in output order.

    Returns:
        One value per region; NaN when it has no row.
    """
    return _by_region(1 - confidence, region, regions, np.mean)


def atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Fit the ATC threshold on labelled rows (val).

    Among the distinct confidences and +inf, the cut whose count of rows
    with confidence < cut is closest to the count of wrong rows; ties:
    the smallest cut.

    Args:
        confidence: The maximum class probability of every row.
        correct: Whether the classifier got every row right.

    Returns:
        The threshold.
    """
    cuts = np.append(np.unique(confidence), np.inf)
    n_below = np.searchsorted(np.sort(confidence), cuts, side="left")
    n_wrong = int((~np.asarray(correct, dtype=bool)).sum())
    return float(cuts[np.argmin(np.abs(n_below - n_wrong))])


def atc_rate(
    confidence: np.ndarray, region: np.ndarray, regions: np.ndarray, threshold: float
) -> np.ndarray:
    """Return the share of rows under the threshold in every region.

    Args:
        confidence: The maximum class probability of every row.
        region: The region of every row.
        regions: The regions to report, in output order.
        threshold: From atc_threshold.

    Returns:
        One value per region; NaN when it has no row.
    """
    below = (confidence < threshold).astype(np.float64)
    return _by_region(below, region, regions, np.mean)


def _by_region(
    values: np.ndarray,
    region: np.ndarray,
    regions: np.ndarray,
    reduce: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Reduce values per region; a mean over no row is NaN."""
    order = np.argsort(region, kind="stable")
    present, starts = np.unique(region[order], return_index=True)
    groups = dict(zip(present.tolist(), np.split(values[order], starts[1:])))
    empty = 0.0 if reduce is np.sum else np.nan
    return np.array(
        [reduce(groups[r]) if r in groups else empty for r in regions.tolist()],
        dtype=np.float64,
    )
