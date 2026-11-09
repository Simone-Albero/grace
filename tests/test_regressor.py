"""src.regressor: descriptors and the nested-CV forest."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from src.regressor import fit_failure_regressor, region_descriptors

GRID = {"n_estimators": [30], "min_samples_leaf": [1, 3]}


def _long() -> pl.DataFrame:
    rows = [
        ("region", 1, "m1", 0.1),
        ("region", 1, "m2", 0.2),
        ("region", 2, "m1", 0.3),
        ("region", 2, "m2", None),
        ("class", 0, "m1", 1.0),
        ("class", 0, "m2", 2.0),
    ]
    return pl.DataFrame(rows, schema=["scope", "id", "measure", "value"], orient="row")


def test_region_descriptors():
    X, names = region_descriptors(_long(), np.array([2, 1]), np.array([0, 0]), None)
    assert names == ["region_m1", "region_m2", "class_m1", "class_m2"]
    np.testing.assert_allclose(X, [[0.3, np.nan, 1.0, 2.0], [0.1, 0.2, 1.0, 2.0]])


def test_region_descriptors_selects_measures():
    X, names = region_descriptors(_long(), np.array([1]), np.array([0]), ["m2"])
    assert names == ["region_m2", "class_m2"]
    np.testing.assert_allclose(X, [[0.2, 2.0]])
    with pytest.raises(ValueError, match="m9"):
        region_descriptors(_long(), np.array([1]), np.array([0]), ["m9"])


def _fit(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 3))
    y = np.clip(0.8 * X[:, 0] + rng.normal(0, 0.05, n), 0, 1)
    fitted = fit_failure_regressor(
        X,
        y,
        params={},
        param_grid=GRID,
        n_outer_splits=4,
        n_inner_splits=3,
        n_iter=2,
        seed=0,
    )
    return X, y, fitted


def test_fit_failure_regressor_out_of_fold():
    _, y, fitted = _fit(60)
    assert fitted.predicted.shape == fitted.fold.shape == fitted.base.shape == (60,)
    assert fitted.contributions.shape == (60, 3)
    assert fitted.predicted.min() >= 0 and fitted.predicted.max() <= 1
    unclipped = fitted.base + fitted.contributions.sum(axis=1)
    np.testing.assert_allclose(np.clip(unclipped, 0, 1), fitted.predicted, atol=1e-6)
    assert set(fitted.fold.tolist()) == {0, 1, 2, 3}
    assert fitted.importances.shape == (3,)
    assert fitted.importances.argmax() == 0
    assert spearmanr(fitted.predicted, y).statistic > 0.7
    assert sorted(f["fold"] for f in fitted.folds) == [-1, 0, 1, 2, 3]


def test_fit_failure_regressor_with_few_regions():
    _, _, fitted = _fit(3)
    assert fitted.predicted.shape == (3,)


def test_fit_failure_regressor_rejects_degenerate_targets():
    with pytest.raises(ValueError):
        _fit(1)
    with pytest.raises(ValueError):
        fit_failure_regressor(
            np.zeros((5, 2)),
            np.full(5, 0.2),
            params={},
            param_grid=GRID,
            n_outer_splits=2,
            n_inner_splits=2,
            n_iter=1,
            seed=0,
        )
