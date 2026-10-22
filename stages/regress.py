"""Random Forest failure regressor on the complexity descriptors.

Each region's test failure rate is predicted out of fold from its
descriptors (and its class's), on the regions that baselines scored. The
log compares it with the MCP and ATC scores of baselines. Artifact: the
forest refit on every used region. DB rows written: `estimates` (every
region), `regressor_folds` (the outer folds, then the refit as fold -1),
`importances` and the regressor's scores in metrics.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import json

import joblib
import pandas as pd
from omegaconf import OmegaConf

from src.regressor import FailureRegressor, fit_failure_regressor, region_descriptors
from src.scores import log_scores, rate_scores
from stages import Context

CONFIG = ("failure_regressor", "seed")
INPUTS = ("regions", "complexity", "failures", "baselines")
FILES = ()
OUTPUTS = ("model.joblib",)


def _folds_table(fitted: FailureRegressor, n_regions: int) -> pd.DataFrame:
    """Return the outer folds, then the refit as fold -1."""
    refit = {
        "fold": -1,
        "n_regions": n_regions,
        "best_score": fitted.refit_score,
        "params": fitted.refit_params,
    }
    folds = pd.DataFrame([*fitted.folds, refit])
    folds["params"] = [json.dumps(params, sort_keys=True) for params in folds["params"]]
    return folds


def run(ctx: Context) -> None:
    """Fit the failure regressor out of fold on the scored regions.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: Fewer than two regions are scored, or their
            failure rates do not differ.
    """
    regressor_cfg = ctx.cfg.failure_regressor
    regions = ctx.read("regions", "regions").set_index("region_id")
    descriptors = region_descriptors(
        ctx.read("complexity", "complexity"), regions["class_id"]
    )
    failures = ctx.read("failures", "failures").set_index("region_id")
    baselines = ctx.read("baselines", "baselines").set_index("region_id")
    # Kept in failures order, which fixes the CV folds.
    used = baselines.index[baselines["used"] == 1]

    # A descriptor no region has a value for has nothing to learn from.
    X = descriptors.loc[used].dropna(axis=1, how="all")
    y = failures.loc[used, "failure_rate"].astype(float)

    fitted = fit_failure_regressor(
        X,
        y,
        params=OmegaConf.to_container(regressor_cfg.params),
        param_grid=OmegaConf.to_container(regressor_cfg.param_grid),
        n_outer_splits=regressor_cfg.n_outer_splits,
        n_inner_splits=regressor_cfg.n_inner_splits,
        n_iter=regressor_cfg.n_iter,
        n_jobs=ctx.runtime.n_jobs,
        random_state=ctx.cfg.seed,
    )

    estimates = pd.DataFrame(
        {
            "fold": fitted.fold.reindex(failures.index).astype("Int64"),
            "regressor": fitted.predicted.reindex(failures.index),
        },
        index=failures.index,
    )
    ctx.write("estimates", estimates.reset_index())

    score = rate_scores(fitted.predicted.loc[used], y)
    baseline_metrics = ctx.read("metrics", "baselines").set_index("name")["value"]
    log_scores("regressor", score)
    for name in ("mcp", "atc"):
        log_scores(name, {key: baseline_metrics[f"{name}_{key}"] for key in score})

    ctx.write("regressor_folds", _folds_table(fitted, len(X)))
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
