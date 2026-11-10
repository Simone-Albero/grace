"""Does the geometry add to the classifier's signals? The measures.

The ablation fits the failure regressor on sets of columns of one X:
the geometric descriptors, the classifier's signals (the baselines'
estimates) and both, and every signal alone and with the geometry. All
fits share their outer folds, so their out-of-fold estimates compare
region by region: a paired bootstrap over the regions gives each gain
its interval, and the correlation of two residuals says whether two
estimates fail on the same regions.
"""

from __future__ import annotations

__all__ = [
    "GEOMETRY",
    "SIGNALS",
    "feature_sets",
    "paired_bootstrap",
    "residual_corr",
    "within_class_spearman",
]

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

GEOMETRY = "geometry"  # the feature set of the descriptors alone
SIGNALS = "signals"  # the feature set of every signal


def feature_sets(n_geometry: int, signals: Sequence[str]) -> dict[str, np.ndarray]:
    """Return the columns of every feature set of the ablation.

    Args:
        n_geometry: The geometric descriptors, the first columns of X.
        signals: The signals, the columns after them, in order.

    Returns:
        Set name to column indices: GEOMETRY, SIGNALS,
        "geometry+signals", then for every signal s, s alone and
        "geometry+s".
    """
    geometry = np.arange(n_geometry)
    signal_columns = n_geometry + np.arange(len(signals))
    sets = {
        GEOMETRY: geometry,
        SIGNALS: signal_columns,
        f"{GEOMETRY}+{SIGNALS}": np.r_[geometry, signal_columns],
    }
    for column, name in zip(signal_columns, signals):
        sets[name] = np.array([column])
        sets[f"{GEOMETRY}+{name}"] = np.r_[geometry, column]
    return sets


def within_class_spearman(
    predicted: np.ndarray,
    observed: np.ndarray,
    group: np.ndarray,
    min_regions: int = 10,
) -> float:
    """Return the mean Spearman correlation inside the groups.

    Args:
        predicted: The predicted rates.
        observed: The observed rates.
        group: The class of every region.
        min_regions: The fewest regions of a group that counts.

    Returns:
        The mean over the groups with min_regions regions and neither
        side constant; NaN when no group qualifies.
    """
    values = []
    for name in np.unique(group):
        inside = group == name
        a, b = predicted[inside], observed[inside]
        if inside.sum() >= min_regions and a.std() > 1e-12 and b.std() > 1e-12:
            values.append(spearmanr(a, b).statistic)
    return float(np.mean(values)) if values else np.nan


def residual_corr(
    predicted_a: np.ndarray, predicted_b: np.ndarray, observed: np.ndarray
) -> float:
    """Return the Pearson correlation of the residuals of two estimates.

    Args:
        predicted_a: One estimate per region.
        predicted_b: Another estimate per region.
        observed: The observed rates.

    Returns:
        The correlation of observed - predicted_a and observed -
        predicted_b; NaN when either residual is constant.
    """
    a, b = observed - predicted_a, observed - predicted_b
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def paired_bootstrap(
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    observed: np.ndarray,
    weight: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Compare two estimates on the same regions, with 95% intervals.

    Every draw resamples the regions with replacement and scores both
    estimates on the same draw.

    Args:
        predicted_a: One estimate per region.
        predicted_b: Another estimate per region.
        observed: The observed rates.
        weight: The weight of every region in wmae.
        n_bootstrap: The draws.
        seed: The seed of the draws.

    Returns:
        d_mae, d_wmae and d_spearman, a's score minus b's on all the
        regions, each with _lo and _hi, the 2.5 and 97.5 percentiles
        over the draws.
    """
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(observed), (n_bootstrap, len(observed)))
    error_a = np.abs(predicted_a - observed)
    error_b = np.abs(predicted_b - observed)
    # Row-wise Spearman: the Pearson correlation of the ranks.
    ranks = {
        side: rankdata(values[drawn], axis=1)
        for side, values in (("a", predicted_a), ("b", predicted_b), ("y", observed))
    }
    draws = {
        "mae": error_a[drawn].mean(axis=1) - error_b[drawn].mean(axis=1),
        "wmae": ((error_a - error_b)[drawn] * weight[drawn]).sum(axis=1)
        / weight[drawn].sum(axis=1),
        "spearman": _row_corr(ranks["a"], ranks["y"])
        - _row_corr(ranks["b"], ranks["y"]),
    }
    full = {
        "mae": error_a.mean() - error_b.mean(),
        "wmae": np.average(error_a - error_b, weights=weight),
        "spearman": _spearman(predicted_a, observed) - _spearman(predicted_b, observed),
    }
    result = {}
    for key, values in draws.items():
        low, high = np.nanpercentile(values, [2.5, 97.5])
        result[f"d_{key}"] = float(full[key])
        result[f"d_{key}_lo"] = float(low)
        result[f"d_{key}_hi"] = float(high)
    return result


def _spearman(predicted: np.ndarray, observed: np.ndarray) -> float:
    if predicted.std() <= 1e-12 or observed.std() <= 1e-12:
        return np.nan
    return float(spearmanr(predicted, observed).statistic)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row of a with its row of b."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=1) / denominator, np.nan)
