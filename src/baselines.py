"""MCP and ATC per-region rates and the ATC threshold."""

__all__ = ["atc_rate", "atc_threshold", "mcp_rate"]

import numpy as np
import pandas as pd

from src.grouping import RowsBy


def _by_region(values: np.ndarray, region: np.ndarray) -> pd.Series:
    """Return the mean of `values` over each region, by region id."""
    groups = RowsBy(np.asarray(region))
    return pd.Series(
        groups.mean(np.asarray(values, dtype=float)),
        index=pd.Index(groups.ids, name="region_id"),
    )


def mcp_rate(confidence: np.ndarray, region: np.ndarray) -> pd.Series:
    """Return each region's mean MCP risk, 1 - max class probability.

    Args:
        confidence: The maximum class probability of every row.
        region: The region id of every row.

    Returns:
        The mean risk, indexed by region id.
    """
    return _by_region(1.0 - np.asarray(confidence, dtype=float), region)


def atc_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Return the ATC confidence cut.

    Args:
        confidence: The maximum class probability of every row.
        correct: Whether the prediction of every row was right.

    Returns:
        The confidence below which the row count is closest to the
        number of errors.
    """
    sorted_confidence = np.sort(np.asarray(confidence, dtype=float))
    n_errors = int((~np.asarray(correct).astype(bool)).sum())
    # Only distinct values are cuts: `confidence < cut` cannot split a
    # run of ties, so a cut placed inside one would flag fewer samples
    # than it was chosen for.
    cuts = np.append(np.unique(sorted_confidence), np.inf)
    n_below = np.searchsorted(sorted_confidence, cuts, side="left")
    return float(cuts[np.argmin(np.abs(n_below - n_errors))])


def atc_rate(
    confidence: np.ndarray, region: np.ndarray, *, threshold: float
) -> pd.Series:
    """Return each region's share of rows with confidence < `threshold`.

    Args:
        confidence: The maximum class probability of every row.
        region: The region id of every row.
        threshold: The ATC cut from `atc_threshold`.

    Returns:
        The share of rows flagged, indexed by region id.
    """
    return _by_region(np.asarray(confidence, dtype=float) < threshold, region)
