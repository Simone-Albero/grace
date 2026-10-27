"""Random Forest failure regressor on the complexity descriptors.

Each leaf with a val row has its val failure rate predicted out of fold
from its descriptors (and its predicted class's), weighted by its val
rows. A leaf without one is predicted by the refit instead, which never
trained on it. Artifact: the forest refit on every leaf with a val row.
DB rows written: `estimates` (every leaf), `regressor_folds` (the outer
folds, then the refit as fold -1), `importances` and metrics: the
out-of-fold scores against the leaves' val rates, n_features, the splits
and n_leaves_trained.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import json

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.regressor import FailureRegressor, fit_failure_regressor, leaf_descriptors
from src.scores import log_scores, rate_scores
from stages import Context

CONFIG = ("failure_regressor", "seed")
INPUTS = ("tree", "complexity")
FILES = ()
OUTPUTS = ("model.joblib",)


def _predicted_rates(
    fitted: FailureRegressor,
    descriptors: pd.DataFrame,
    columns: pd.Index,
    refit_only: pd.Index,
) -> pd.Series:
    """Return each leaf's out-of-fold rate, else the refit's.

    Args:
        fitted: The fitted regressor.
        descriptors: Every leaf's descriptors.
        columns: The descriptor columns the model trained on.
        refit_only: Leaves the refit predicts but never trained on.

    Returns:
        The predicted rate of every trained leaf, plus `refit_only`.
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


def _folds_table(fitted: FailureRegressor, n_trained: int) -> pd.DataFrame:
    """Return the outer folds, then the refit as fold -1."""
    refit = {
        "fold": -1,
        "n_leaves": n_trained,
        "best_score": fitted.refit_score,
        "params": fitted.refit_params,
    }
    folds = pd.DataFrame([*fitted.folds, refit])
    folds["params"] = [json.dumps(params, sort_keys=True) for params in folds["params"]]
    return folds


def run(ctx: Context) -> None:
    """Fit the failure regressor out of fold on val-labelled leaves.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: Fewer than two leaves have a val row, or their val
            rates do not differ.
    """
    regressor_cfg = ctx.cfg.failure_regressor
    nodes = ctx.read("tree_nodes", "tree")
    leaves = nodes[nodes["status"] == "leaf"].set_index("node_id")
    descriptors = leaf_descriptors(
        ctx.read("complexity", "complexity"), leaves["class_id"]
    )
    # Kept in node order, which fixes the CV folds.
    trained = leaves.index[leaves["n_val"] > 0]

    # A descriptor no leaf has a value for has nothing to learn from.
    X = descriptors.loc[trained].dropna(axis=1, how="all")
    n_val = leaves.loc[trained, "n_val"].astype(float)
    y = (leaves.loc[trained, "n_val_errors"] / n_val).astype(float)

    fitted = fit_failure_regressor(
        X,
        y,
        sample_weight=n_val,
        params=OmegaConf.to_container(regressor_cfg.params),
        param_grid=OmegaConf.to_container(regressor_cfg.param_grid),
        n_outer_splits=regressor_cfg.n_outer_splits,
        n_inner_splits=regressor_cfg.n_inner_splits,
        n_iter=regressor_cfg.n_iter,
        n_jobs=ctx.runtime.n_jobs,
        random_state=ctx.cfg.seed,
    )

    # A leaf without a val row: the refit never saw it either.
    predicted = _predicted_rates(
        fitted, descriptors, X.columns, leaves.index.difference(trained)
    )
    estimates = pd.DataFrame(
        {
            "fold": fitted.fold.reindex(leaves.index).astype("Int64"),
            "regressor": predicted.reindex(leaves.index),
        },
        index=leaves.index,
    )
    ctx.write("estimates", estimates.rename_axis("node_id").reset_index())

    score = rate_scores(fitted.predicted, y)
    log_scores("regressor (out of fold)", score)

    ctx.write("regressor_folds", _folds_table(fitted, len(X)))
    ctx.write(
        "importances",
        fitted.importances.rename_axis("feature").rename("importance").reset_index(),
    )
    ctx.save_metrics(
        {
            **{f"oof_{name}": value for name, value in score.items()},
            "n_features": X.shape[1],
            "n_outer_splits": fitted.n_outer,
            "n_inner_splits": fitted.n_inner,
            "n_leaves_trained": len(X),
        }
    )
    joblib.dump(fitted.model, ctx.out / "model.joblib")
