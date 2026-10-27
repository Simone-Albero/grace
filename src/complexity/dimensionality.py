"""T2-T4 dimensionality measures."""

__all__ = ["compute_t_measures"]

from collections.abc import Mapping

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def _t3_t4(rows: np.ndarray, n_rows: int) -> tuple[float | None, float | None]:
    """Return T3 and T4 from one PCA fit, None when degenerate."""
    n_sampled, n_features = rows.shape
    if n_sampled < 2 or n_features < 1:
        return None, None
    if n_features == 1:
        return 1.0 / n_rows, 1.0
    max_components = min(n_sampled, n_features)
    cumulative_variance = np.cumsum(
        PCA(n_components=max_components).fit(rows).explained_variance_ratio_
    )
    n_components_95 = min(
        int(np.searchsorted(cumulative_variance, 0.95)) + 1, max_components
    )
    return n_components_95 / n_rows, n_components_95 / n_features


def compute_t_measures(
    X: np.ndarray, population: np.ndarray, sizes: Mapping[int, int]
) -> dict[int, dict[str, float | None]]:
    """Return the dimensionality measures T2, T3 and T4 per population.

    Args:
        X: The sampled rows, shape (n, d); the PCA components come from
            these rows, a sample of each population.
        population: The population id of every row of `X`.
        sizes: The rows each population really holds; the ratios are
            over these.

    Returns:
        By population id, "t2", "t3" and "t4"; the last two are None
        when the sample is degenerate.
    """
    result: dict[int, dict[str, float | None]] = {}
    n_features = X.shape[1]
    for population_id in tqdm(
        np.unique(population), desc="T measures", unit="pop", leave=False
    ):
        n_rows = sizes[int(population_id)]
        t3, t4 = _t3_t4(X[population == population_id], n_rows)
        result[int(population_id)] = {"t2": n_features / n_rows, "t3": t3, "t4": t4}
    return result
