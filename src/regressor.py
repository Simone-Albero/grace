"""Leaf descriptors and the random-forest failure-rate regressor."""

__all__ = ["FailureRegressor", "fit_failure_regressor", "leaf_descriptors"]

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureRegressor:
    """Out-of-fold failure rates of every leaf, and a forest on all.

    Attributes:
        predicted: The out-of-fold rate by leaf, indexed like `X`.
        fold: The outer fold holding each leaf out, indexed like
            `X`.
        folds: One dict per outer fold, with "fold", "n_leaves",
            "best_score" and "params".
        importances: The mean `feature_importances_` over the outer
            folds, by descriptor.
        model: The forest refit on every leaf.
        refit_params: The searched parameters of the refit.
        refit_score: The best inner-CV score of the refit; None when it
            was not searched.
        n_outer: The outer split count, after adapting to the number of
            leaves.
        n_inner: The inner split count, after the same adaptation.
    """

    predicted: pd.Series
    fold: pd.Series
    folds: list[dict[str, object]]
    importances: pd.Series
    model: RandomForestRegressor
    refit_params: dict[str, object]
    refit_score: float | None
    n_outer: int
    n_inner: int


def leaf_descriptors(complexity: pd.DataFrame, leaf_class: pd.Series) -> pd.DataFrame:
    """Build one row per leaf: its descriptors, then its class's.

    Args:
        complexity: The long complexity table, with "scope", "scope_id",
            "measure" and "value".
        leaf_class: The class id of every leaf, by node id: the class
            its root predicts.

    Returns:
        The `leaf_*` and `class_*` descriptors, indexed by node id, in
        measure order; NaN for a leaf no population of which was
        measured.
    """
    # Measures keep their order of first appearance: the forest draws
    # max_features columns by position, so another order changes its
    # results.
    measures = complexity["measure"].unique()

    def wide(scope: str, ids: pd.Index, prefix: str) -> pd.DataFrame:
        table = complexity[complexity["scope"] == scope].pivot(
            index="scope_id", columns="measure", values="value"
        )
        return table.reindex(columns=measures, index=ids).add_prefix(prefix)

    return pd.concat(
        [
            wide("leaf", leaf_class.index, "leaf_"),
            wide("class", pd.Index(leaf_class), "class_").set_index(leaf_class.index),
        ],
        axis=1,
    )


def _usable_n_splits(n_smallest: int, n_splits: int) -> int:
    """Return the most splits the smallest group can fill.

    It is at most `n_splits`, and 0 when fewer than two fit.
    """
    usable = min(n_splits, n_smallest)
    return usable if usable >= 2 else 0


def _quantile_strata(y: pd.Series, n_bins: int) -> pd.Series | None:
    """Return the target's quantile-bin codes, None below two bins."""
    bins = pd.qcut(y, q=min(n_bins, len(y)), duplicates="drop")
    if bins.nunique() < 2:
        return None
    return bins.cat.codes


def _fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series,
    params: Mapping[str, object],
    param_grid: Mapping[str, Sequence[object]],
    inner_cv: KFold | None,
    n_iter: int,
    n_jobs: int,
    random_state: int,
    search_seed: int,
) -> tuple[RandomForestRegressor, dict[str, object], float | None]:
    """Fit one forest; without `inner_cv`, keep defaults, no search."""
    forest = RandomForestRegressor(**{**params, "random_state": random_state})
    if inner_cv is None:
        forest.fit(X, y, sample_weight=sample_weight)
        fitted = forest.get_params()
        return forest, {key: fitted[key] for key in param_grid}, None

    search = RandomizedSearchCV(
        estimator=forest,
        param_distributions=param_grid,
        # A grid smaller than the budget is searched whole: asking for
        # more draws than it holds makes sklearn warn on every fold and
        # run the whole grid anyway.
        n_iter=min(n_iter, len(ParameterGrid(param_grid))),
        cv=inner_cv,
        # Scored as reported, clipped: a model is tuned for the
        # predictor it is judged as.
        scoring=make_scorer(
            lambda y_true, y_pred: r2_score(y_true, np.clip(y_pred, 0.0, 1.0))
        ),
        n_jobs=n_jobs,
        # At a fixed random_state RandomizedSearchCV draws the same
        # combinations whatever the data, so each fit gets its own seed
        # to search its own slice.
        random_state=search_seed,
    )
    search.fit(X, y, sample_weight=sample_weight)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def _outer_cv(
    y: pd.Series, n_splits: int, random_state: int
) -> tuple[BaseCrossValidator, int, pd.Series | None]:
    """Build the outer CV, stratified by target quantile when it can be.

    Returns:
        The splitter, its effective split count and the strata (None
        when it is a plain KFold).
    """
    strata = _quantile_strata(y, n_splits)
    n_outer = (
        _usable_n_splits(int(strata.value_counts().min()), n_splits)
        if strata is not None
        else 0
    )
    if n_outer >= 2:
        outer_cv = StratifiedKFold(
            n_splits=n_outer, shuffle=True, random_state=random_state
        )
    else:
        n_outer = _usable_n_splits(len(y), n_splits)
        outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
        strata = None
    return outer_cv, n_outer, strata


def _inner_cv(
    n_used: int,
    n_outer: int,
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
) -> tuple[KFold | None, int]:
    """Build the inner CV the smallest outer fit can fill.

    It warns when a split count had to be lowered.

    Returns:
        The splitter, None when no search is possible, and its
        effective split count.
    """
    n_fit_worst = n_used - math.ceil(n_used / n_outer)
    n_inner = _usable_n_splits(n_fit_worst, n_inner_splits)
    if n_outer < n_outer_splits or n_inner < n_inner_splits:
        logger.warning(
            "Adapting CV to %d leaves: outer %d -> %d, inner %d -> %d%s",
            n_used,
            n_outer_splits,
            n_outer,
            n_inner_splits,
            n_inner,
            " (no search, forest defaults)" if n_inner == 0 else "",
        )
    inner_cv = (
        KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        if n_inner > 0
        else None
    )
    return inner_cv, n_inner


def fit_failure_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series,
    params: Mapping[str, object],
    param_grid: Mapping[str, Sequence[object]],
    n_outer_splits: int,
    n_inner_splits: int,
    n_iter: int,
    n_jobs: int,
    random_state: int,
) -> FailureRegressor:
    """Fit a forest on leaf descriptors by nested CV, then refit.

    Args:
        X: The descriptors, one row per leaf.
        y: The observed failure rate of each leaf, indexed like `X`.
        sample_weight: Each leaf's weight, indexed like `X`; passed
            to the forest's fit, not to the inner search's own scorer.
        params: The forest's own parameters.
        param_grid: The values the inner search draws from, by forest
            parameter.
        n_outer_splits: The outer folds asked for; fewer when the
            leaves cannot fill them.
        n_inner_splits: The inner folds asked for, adapted likewise.
        n_iter: The most parameter draws of the inner search.
        n_jobs: The parallel jobs of the inner search.
        random_state: The seed of the splits, the forests and the
            searches.

    Returns:
        The out-of-fold rates, each outer fold's search result, the mean
        importances and the forest refit on every leaf.

    Raises:
        ValueError: If fewer than two leaves have rates that differ.
    """
    n_used = len(X)
    if n_used < 2 or float(y.std()) < 1e-9:
        raise ValueError(
            f"{n_used} usable leaf(s), failure-rate std {float(y.std()):.3g}: the "
            "failure regressor needs two leaves whose rates differ. Raise "
            "data.val_frac for more val-labelled leaves; regressor=false "
            "draws the classifier's figures without it."
        )

    outer_cv, n_outer, strata = _outer_cv(y, n_outer_splits, random_state)
    inner_cv, n_inner = _inner_cv(
        n_used,
        n_outer,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
    )
    fit_args = dict(
        params=params,
        param_grid=param_grid,
        inner_cv=inner_cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    predicted = np.empty(n_used)
    fold_of = np.empty(n_used, dtype=int)
    folds, importances = [], []
    for fold, (fit_idx, held_idx) in enumerate(outer_cv.split(X, strata)):
        forest, best_params, best_score = _fit_model(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            sample_weight=sample_weight.iloc[fit_idx],
            search_seed=random_state + fold,
            **fit_args,
        )
        # Clip as the inner scorer does, so held-out predictions match
        # the tuned ones.
        predicted[held_idx] = np.clip(forest.predict(X.iloc[held_idx]), 0.0, 1.0)
        fold_of[held_idx] = fold
        importances.append(forest.feature_importances_)
        folds.append(
            {
                "fold": fold,
                "n_leaves": len(held_idx),
                "best_score": best_score,
                "params": best_params,
            }
        )
        logger.info("outer fold %d/%d: n_held_out=%d", fold + 1, n_outer, len(held_idx))

    logger.info("Refitting the forest on the %d usable leaves", n_used)
    model, refit_params, refit_score = _fit_model(
        X,
        y,
        sample_weight=sample_weight,
        # The seed after the outer folds' own, so the refit searches a
        # slice of its own.
        search_seed=random_state + n_outer,
        **fit_args,
    )
    return FailureRegressor(
        predicted=pd.Series(predicted, index=X.index),
        fold=pd.Series(fold_of, index=X.index),
        folds=folds,
        importances=pd.Series(np.mean(importances, axis=0), index=X.columns),
        model=model,
        refit_params=refit_params,
        refit_score=refit_score,
        n_outer=n_outer,
        n_inner=n_inner,
    )
