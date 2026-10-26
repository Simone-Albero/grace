"""Random Forest failure regressor on the complexity descriptors.

Each region with a val row has its val failure rate predicted out of
fold from its descriptors (and its class's), weighted by its val rows.
A region baselines scored (`used`) but without a val row is predicted
by the refit instead, which never trained on it. Every predictor is
scored against test, on the regions baselines scored. Artifact: the
forest refit on every region with a val row. DB rows written:
`estimates` (every region; `regressor` NULL without a val row and
unused), `regressor_folds` (the outer folds, scored against test, then
the refit as fold -1), `importances` and the regressor's scores in
metrics.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import json

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.regressor import FailureRegressor, fit_failure_regressor, region_descriptors
from src.scores import log_scores, rate_scores
from stages import Context

CONFIG = ("failure_regressor", "seed")
INPUTS = ("regions", "complexity", "failures", "baselines")
FILES = ()
OUTPUTS = ("model.joblib",)


def _fold_score(
    fold_of: pd.Series, predicted: pd.Series, test_rate: pd.Series, fold: int
) -> dict[str, float]:
    """Score one outer fold's held-out, used regions against test.

    Returns:
        `spearman`, `r2` and `mae`, NaN when none of the fold's
        held-out regions are used.
    """
    names = ("spearman", "r2", "mae")
    held_out = fold_of.index[fold_of == fold].intersection(test_rate.index)
    if held_out.empty:
        return dict.fromkeys(names, float("nan"))
    score = rate_scores(predicted.loc[held_out], test_rate.loc[held_out])
    return {name: score[name] for name in names}


def _predicted_rates(
    fitted: FailureRegressor,
    descriptors: pd.DataFrame,
    columns: pd.Index,
    refit_only: pd.Index,
) -> pd.Series:
    """Return every trained region's out-of-fold rate, refit-filled.

    Args:
        fitted: The fitted regressor.
        descriptors: Every region's descriptors.
        columns: The descriptor columns the model trained on.
        refit_only: Regions the refit predicts but never trained on.

    Returns:
        The predicted rate of every trained region, plus `refit_only`.
    """
    if refit_only.empty:
        refit_predicted = pd.Series(dtype=float)
    else:
        refit_predicted = pd.Series(
            np.clip(
                fitted.model.predict(descriptors.loc[refit_only, columns]), 0.0, 1.0
            ),
            index=refit_only,
        )
    return pd.concat([fitted.predicted, refit_predicted])


def _folds_table(
    fitted: FailureRegressor, predicted: pd.Series, test_rate: pd.Series, n_trained: int
) -> pd.DataFrame:
    """Return the outer folds' test scores, the refit as fold -1."""
    rows = [
        {**fold, **_fold_score(fitted.fold, predicted, test_rate, fold["fold"])}
        for fold in fitted.folds
    ]
    refit = {
        "fold": -1,
        "n_regions": n_trained,
        "best_score": fitted.refit_score,
        "params": fitted.refit_params,
    }
    folds = pd.DataFrame([*rows, refit])
    folds["params"] = [json.dumps(params, sort_keys=True) for params in folds["params"]]
    return folds


def run(ctx: Context) -> None:
    """Fit the failure regressor out of fold on val-labelled regions.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: Fewer than two regions have a val row, or their val
            rates do not differ.
    """
    regressor_cfg = ctx.cfg.failure_regressor
    regions = ctx.read("regions", "regions").set_index("region_id")
    descriptors = region_descriptors(
        ctx.read("complexity", "complexity"), regions["class_id"]
    )
    failures = ctx.read("failures", "failures").set_index("region_id")
    baselines = ctx.read("baselines", "baselines").set_index("region_id")
    used = baselines.index[baselines["used"] == 1]
    # Kept in failures order, which fixes the CV folds.
    trained = failures.index[failures["n_val"] > 0]

    # A descriptor no region has a value for has nothing to learn from.
    X = descriptors.loc[trained].dropna(axis=1, how="all")
    y = failures.loc[trained, "val_rate"].astype(float)
    weight = failures.loc[trained, "n_val"].astype(float)

    fitted = fit_failure_regressor(
        X,
        y,
        sample_weight=weight,
        params=OmegaConf.to_container(regressor_cfg.params),
        param_grid=OmegaConf.to_container(regressor_cfg.param_grid),
        n_outer_splits=regressor_cfg.n_outer_splits,
        n_inner_splits=regressor_cfg.n_inner_splits,
        n_iter=regressor_cfg.n_iter,
        n_jobs=ctx.runtime.n_jobs,
        random_state=ctx.cfg.seed,
    )

    # Used regions without a val row: the refit never saw them either.
    refit_only = used.difference(trained)
    predicted = _predicted_rates(fitted, descriptors, X.columns, refit_only)

    estimates = pd.DataFrame(
        {
            "fold": fitted.fold.reindex(failures.index).astype("Int64"),
            "regressor": predicted.reindex(failures.index),
        },
        index=failures.index,
    )
    ctx.write("estimates", estimates.reset_index())

    test_rate = failures.loc[used, "test_rate"].astype(float)
    score = rate_scores(predicted.loc[used], test_rate)
    baseline_metrics = ctx.read("metrics", "baselines").set_index("name")["value"]
    log_scores("regressor", score)
    for name in ("val", "mcp", "atc"):
        log_scores(name, {key: baseline_metrics[f"{name}_{key}"] for key in score})

    ctx.write("regressor_folds", _folds_table(fitted, predicted, test_rate, len(X)))
    ctx.write(
        "importances",
        fitted.importances.rename_axis("feature").rename("importance").reset_index(),
    )

    ctx.save_metrics(
        {
            **{f"regressor_{name}": value for name, value in score.items()},
            "n_features": X.shape[1],
            "n_outer_splits": fitted.n_outer,
            "n_inner_splits": fitted.n_inner,
        }
    )
    joblib.dump(fitted.model, ctx.out / "model.joblib")
