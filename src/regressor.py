"""The failure regressor: region descriptors to failure rate.

Nested cross-validation: each outer fold refits a RandomizedSearchCV
over the other regions and predicts the held-out ones, so every region's
estimate comes from a forest that never saw it, and TreeSHAP of that
same forest explains it. A final refit on every region is the model
kept.
"""

from __future__ import annotations

__all__ = ["FailureRegressor", "fit_failure_regressor", "region_descriptors"]

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.evaluation import rate_scores


def region_descriptors(
    complexity: pl.DataFrame,
    regions: np.ndarray,
    region_class: np.ndarray,
    measures: Sequence[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Return the descriptors of regions: own measures and class's.

    Args:
        complexity: Long rows scope ("region" or "class"), id, measure,
            value (null allowed), as the complexity stage writes them.
        regions: The region ids, in output row order.
        region_class: The class of every region of regions.
        measures: The measures to use, None for all.

    Returns:
        X of shape (len(regions), 2 · m), NaN where a value is missing,
        and its column names: region_<m> for every measure in order of
        first appearance in complexity, then class_<m> in the same
        order.

    Raises:
        ValueError: When measures names a measure complexity does not
            hold.
    """
    # First appearance, not alphabetical: the forest draws max_features
    # columns by position, so another order changes its results.
    available = complexity["measure"].unique(maintain_order=True).to_list()
    if measures is None:
        measures = available
    unknown = sorted(set(measures) - set(available))
    if unknown:
        raise ValueError(f"regressor.measures not in the complexity table: {unknown}")
    order = [m for m in available if m in set(measures)]
    values = {
        (scope, population, measure): np.nan if value is None else value
        for scope, population, measure, value in complexity.select(
            "scope", "id", "measure", "value"
        ).iter_rows()
    }
    X = np.array(
        [
            [values.get(("region", r, m), np.nan) for m in order]
            + [values.get(("class", c, m), np.nan) for m in order]
            for r, c in zip(regions.tolist(), region_class.tolist())
        ],
        dtype=np.float64,
    ).reshape(len(regions), 2 * len(order))
    names = [f"region_{m}" for m in order] + [f"class_{m}" for m in order]
    return X, names


@dataclass(frozen=True)
class FailureRegressor:
    """The fitted regressor and its out-of-fold results per region.

    Attributes:
        model: The forest refit on every region.
        predicted: The out-of-fold estimates, shape (n,), clipped to
            [0, 1].
        fold: The outer fold that held each region out, shape (n,).
        base: The TreeSHAP expected value of that fold's forest, shape
            (n,).
        contributions: The TreeSHAP values, shape (n, f); base plus
            their sum is the unclipped estimate.
        importances: The mean impurity importance over the outer
            folds, shape (f,).
        folds: One row per fold, -1 for the refit: fold, n_regions,
            spearman, r2, mae (NaN for the refit), best_score and
            params (JSON text).
    """

    model: RandomForestRegressor
    predicted: np.ndarray
    fold: np.ndarray
    base: np.ndarray
    contributions: np.ndarray
    importances: np.ndarray
    folds: list[dict[str, Any]]


def fit_failure_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Mapping[str, Any],
    param_grid: Mapping[str, Sequence[Any]],
    n_outer_splits: int,
    n_inner_splits: int,
    n_iter: int,
    seed: int,
) -> FailureRegressor:
    """Fit the regressor under nested cross-validation.

    Outer: StratifiedKFold on the quantile bins of y (shuffled, seed)
    with min(n_outer_splits, smallest bin) folds;
    KFold(min(n_outer_splits, n)) when that is below 2. Inner:
    KFold(min(n_inner_splits, n - ceil(n / n_outer))), and no search
    when that is below 2. Each search is a RandomizedSearchCV over
    param_grid with min(n_iter, grid size) draws, scored by the r2 of
    the clipped prediction, random_state seed + fold; the refit uses
    seed + number of outer folds.

    Args:
        X: The descriptors, shape (n, f); NaN allowed.
        y: The observed failure rates, shape (n,).
        params: Fixed RandomForestRegressor params (random_state is
            seed).
        param_grid: The searched params.
        n_outer_splits: The most outer folds.
        n_inner_splits: The most inner folds.
        n_iter: The most search draws.
        seed: The seed of the folds, the searches and the forests.

    Returns:
        The fitted regressor.

    Raises:
        ValueError: With fewer than 2 regions or a constant y (std <
            1e-9).
    """
    n = len(y)
    if n < 2 or float(np.std(y)) < 1e-9:
        raise ValueError(
            f"the failure regressor needs 2+ regions whose rates differ, got {n} "
            f"region(s) with rate std {float(np.std(y)) if n else 0.0:.3g}"
        )
    outer, strata = _outer_cv(y, n_outer_splits, seed)
    n_outer = outer.get_n_splits()
    n_inner = _usable_splits(n - math.ceil(n / n_outer), n_inner_splits)
    inner = KFold(n_inner, shuffle=True, random_state=seed) if n_inner else None
    search_settings = dict(
        params=params, param_grid=param_grid, inner=inner, n_iter=n_iter, seed=seed
    )

    predicted, base = np.empty(n), np.empty(n)
    fold_of = np.empty(n, dtype=np.int64)
    contributions = np.empty(X.shape)
    folds, importances = [], []
    for fold, (fit_rows, held_out) in enumerate(outer.split(X, strata)):
        forest, best_params, best_score = _fit_forest(
            X[fit_rows], y[fit_rows], search_seed=seed + fold, **search_settings
        )
        # Clipped as the inner scorer clips, so the fold scores match.
        predicted[held_out] = np.clip(forest.predict(X[held_out]), 0.0, 1.0)
        fold_of[held_out] = fold
        explainer = shap.TreeExplainer(forest)
        base[held_out] = float(np.ravel(explainer.expected_value)[0])
        contributions[held_out] = explainer.shap_values(X[held_out])
        importances.append(forest.feature_importances_)
        scores = rate_scores(predicted[held_out], y[held_out])
        folds.append(
            {
                "fold": fold,
                "n_regions": len(held_out),
                **{k: scores[k] for k in ("spearman", "r2", "mae")},
                "best_score": best_score,
                "params": json.dumps(best_params),
            }
        )
    model, refit_params, refit_score = _fit_forest(
        X, y, search_seed=seed + n_outer, **search_settings
    )
    folds.append(
        {
            "fold": -1,
            "n_regions": n,
            "spearman": np.nan,
            "r2": np.nan,
            "mae": np.nan,
            "best_score": refit_score,
            "params": json.dumps(refit_params),
        }
    )
    return FailureRegressor(
        model=model,
        predicted=predicted,
        fold=fold_of,
        base=base,
        contributions=contributions,
        importances=np.mean(importances, axis=0),
        folds=folds,
    )


def _usable_splits(n_smallest: int, n_splits: int) -> int:
    """Return the most splits the smallest group fills, 0 below 2."""
    usable = min(n_splits, n_smallest)
    return usable if usable >= 2 else 0


def _outer_cv(
    y: np.ndarray, n_splits: int, seed: int
) -> tuple[StratifiedKFold | KFold, np.ndarray | None]:
    """Return the outer CV, on quantile strata of y when it can be."""
    bins = pd.qcut(y, q=min(n_splits, len(y)), duplicates="drop")
    strata = np.asarray(bins.codes) if bins.categories.size >= 2 else None
    if strata is not None:
        n_outer = _usable_splits(int(np.bincount(strata).min()), n_splits)
        if n_outer:
            return StratifiedKFold(n_outer, shuffle=True, random_state=seed), strata
    n_outer = _usable_splits(len(y), n_splits)
    return KFold(n_outer, shuffle=True, random_state=seed), None


def _clipped_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, np.clip(y_pred, 0.0, 1.0))


def _fit_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: Mapping[str, Any],
    param_grid: Mapping[str, Sequence[Any]],
    inner: KFold | None,
    n_iter: int,
    seed: int,
    search_seed: int,
) -> tuple[RandomForestRegressor, dict[str, Any], float | None]:
    """Fit one forest; without an inner CV, on its params, no search."""
    forest = RandomForestRegressor(**{**params, "random_state": seed})
    if inner is None:
        forest.fit(X, y)
        fitted = forest.get_params()
        return forest, {k: fitted[k] for k in param_grid}, None
    search = RandomizedSearchCV(
        forest,
        param_distributions=dict(param_grid),
        # A grid smaller than the budget is searched whole.
        n_iter=min(n_iter, len(ParameterGrid(dict(param_grid)))),
        cv=inner,
        # Scored clipped, as reported: tuned as the predictor it is
        # judged as.
        scoring=make_scorer(_clipped_r2),
        # Each fit draws its own combinations: at one seed every search
        # would draw the same ones.
        random_state=search_seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)
