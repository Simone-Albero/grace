"""T2-T4 and anisotropy: how many features there are relative to rows.

With d features, n the population's true size and n95 the
principal components that reach 95% of the variance of its sampled,
scaled rows: t2 = d / n, t3 = n95 / n, t4 = n95 / d, anisotropy = share
of variance of the first component. With one feature: t3 = 1 / n, t4 =
1, anisotropy = 1; with fewer than 2 rows or no feature, t3, t4 and
anisotropy are None.
"""

from __future__ import annotations

__all__ = ["compute_t_measures"]

from collections.abc import Mapping

import numpy as np
from sklearn.decomposition import PCA

from src.space import scale_for_metric


def _pca_measures(
    rows: np.ndarray, n_rows: int
) -> tuple[float | None, float | None, float | None]:
    """Return T3, T4 and the anisotropy from one PCA fit.

    The anisotropy is the variance share of the first component, None
    when the rows do not vary. All three are None when the sample is
    degenerate.
    """
    n_sampled, n_features = rows.shape
    if n_sampled < 2 or n_features < 1:
        return None, None, None
    if n_features == 1:
        return 1.0 / n_rows, 1.0, 1.0
    max_components = min(n_sampled, n_features)
    variance_ratio = (
        PCA(n_components=max_components).fit(rows).explained_variance_ratio_
    )
    n_components_95 = min(
        int(np.searchsorted(np.cumsum(variance_ratio), 0.95)) + 1, max_components
    )
    # Rows that never vary give 0/0 ratios.
    anisotropy = float(variance_ratio[0])
    return (
        n_components_95 / n_rows,
        n_components_95 / n_features,
        anisotropy if np.isfinite(anisotropy) else None,
    )


def compute_t_measures(
    X: np.ndarray, population: np.ndarray, sizes: Mapping[int, int], *, metric: str
) -> dict[int, dict[str, float | None]]:
    """Return T2, T3, T4 and the anisotropy per population.

    Args:
        X: The sampled rows, shape (n, d); the PCA components come from
            these rows, a sample of each population.
        population: The population id of every row of `X`.
        sizes: The rows each population really holds; the ratios are
            over these.
        metric: The distance, which fixes how `X` is scaled.

    Returns:
        By population id, "t2", "t3", "t4" and "anisotropy"; the last
        three are None when the sample is degenerate.
    """
    X = scale_for_metric(X, metric)
    result: dict[int, dict[str, float | None]] = {}
    n_features = X.shape[1]
    for population_id in np.unique(population):
        n_rows = sizes[int(population_id)]
        t3, t4, anisotropy = _pca_measures(X[population == population_id], n_rows)
        result[int(population_id)] = {
            "t2": n_features / n_rows,
            "t3": t3,
            "t4": t4,
            "anisotropy": anisotropy,
        }
    return result
