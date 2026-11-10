"""L1-L3: how well a linear SVM separates a population from a rival.

Per rival with 2+ rows each side, a LinearSVC(class_weight="balanced")
fits own = 1 against rival = 0 on the scaled rows:
- l1 = 1 - 1 / (1 + (Σ |margin of the misclassified rows| / (‖coef‖ +
  1e-8)) / N);
- l2 = 1 - balanced accuracy on the rows;
- l3 = 1 - balanced accuracy on as many points interpolated between
  random pairs of rows of the same side.
Each is the largest over the rivals.
"""

from __future__ import annotations

__all__ = ["L_KEYS", "compute_l_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

_EPS = 1e-8
L_KEYS = ("l1", "l2", "l3")


def _interpolate(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return as many points as `rows`, each between two of them."""
    first = rng.integers(len(rows), size=len(rows))
    second = rng.integers(len(rows), size=len(rows))
    weight = rng.random((len(rows), 1))
    return rows[first] + weight * (rows[second] - rows[first])


def _rival_linearity(
    own_rows: np.ndarray, rival_rows: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return L1, L2 and L3 of a linear SVM against one rival.

    The errors are balanced, so a small region weighs as much as a
    large rival.
    """
    X = np.vstack([own_rows, rival_rows])
    y = np.concatenate([np.ones(len(own_rows)), np.zeros(len(rival_rows))])
    # The default 1000 iterations leave about 5% of the CIC pairs
    # unconverged; 10000 leave a handful, at a third more time.
    svm = LinearSVC(class_weight="balanced", max_iter=10_000).fit(X, y)

    margins = svm.decision_function(X)
    is_misclassified = (margins > 0) != (y == 1)
    total_error_distance = np.abs(margins[is_misclassified]).sum() / (
        np.linalg.norm(svm.coef_) + _EPS
    )
    l1 = 1.0 - 1.0 / (1.0 + total_error_distance / len(X))
    l2 = 1.0 - balanced_accuracy_score(y, svm.predict(X))

    X_synthetic = np.vstack(
        [_interpolate(own_rows, rng), _interpolate(rival_rows, rng)]
    )
    l3 = 1.0 - balanced_accuracy_score(y, svm.predict(X_synthetic))
    return float(l1), float(l2), float(l3)


def compute_l_measures(
    points: np.ndarray,
    population: np.ndarray,
    rivals: Mapping[int, Sequence[int]],
    *,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Return L1-L3 per population, the worst over its nearest rivals.

    A linear SVM separates the population from each rival: L2 is its
    balanced training error, L1 the mean distance of its errors from
    the hyperplane, mapped into [0, 1), and L3 its balanced error on
    points interpolated between rows of the same side, which grows
    when the boundary is curved rather than overlapped. Rows that
    coincide across the two sides leave the SVM no direction: its
    errors then lie on the hyperplane, so L1 reads 0 while L2 and L3
    read chance.

    Args:
        points: The sampled rows, shape (n, d).
        population: The population id of every row of `points`.
        rivals: The nearest rival ids of every population.
        random_state: The seed of the interpolated points.

    Returns:
        By population id, the most of each measure over its rivals;
        None each when the population has fewer than two rows or none
        of its rivals does.
    """
    rows_by_population = {
        int(population_id): points[population == population_id]
        for population_id in np.unique(population)
    }
    rng = np.random.default_rng(random_state)

    result: dict[int, dict[str, float | None]] = {}
    for population_id, own_rows in rows_by_population.items():
        rival_measures = [
            _rival_linearity(own_rows, rows_by_population[rival], rng)
            for rival in rivals[population_id]
            if len(own_rows) >= 2 and len(rows_by_population[rival]) >= 2
        ]
        result[population_id] = (
            dict(zip(L_KEYS, np.max(rival_measures, axis=0).tolist()))
            if rival_measures
            else dict.fromkeys(L_KEYS)
        )
    return result
