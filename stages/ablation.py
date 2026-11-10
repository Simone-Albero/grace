"""Stage ablation: does the geometry add to the classifier's signals?"""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import numpy as np
import polars as pl

from src import db
from src.ablation import (
    GEOMETRY,
    SIGNALS,
    feature_sets,
    paired_bootstrap,
    residual_corr,
    within_class_spearman,
)
from src.config import DatasetConfig
from src.evaluation import rate_scores
from src.regressor import fit_failure_regressor, region_signals
from stages.paths import ablation_file
from stages.regress import read_inputs

# The feature sets refitted with the outer folds grouped by class.
_GROUPED = (GEOMETRY, SIGNALS, f"{GEOMETRY}+{SIGNALS}")


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Fit the regressor on every feature set and compare the fits.

    X = regress.read_inputs (the geometry), then
    regressor.region_signals of regressor.signals; the feature sets are
    ablation.feature_sets of them. Every set is fitted as stage regress
    fits (regressor.fit_failure_regressor, the regressor config and
    seed, each region weighted by sqrt(n_eval)), so all share the outer
    folds ("strata"); GEOMETRY, SIGNALS and both are fitted again with
    the outer folds grouped by class ("class"). Gains: geometry+s
    against s, for every signal s and for s = SIGNALS, under each cv
    that fitted both (ablation.paired_bootstrap, ablation.n_bootstrap,
    seed), with the residual correlation of GEOMETRY and s.

    Writes to output_dir: ablation_<model>_estimates.parquet (region,
    class, n_eval, observed, then "<cv>:<set>" for every fit).
    Logs: ablation_scores (feature_set, cv, n_features, spearman, r2,
    mae, mse, wmae weighted by n_eval, within_spearman: the mean
    Spearman inside the classes of 10+ regions), ablation_gains (cv,
    signal, with_geometry, without, residual_corr, d_mae, d_wmae,
    d_spearman, each with _lo and _hi), ablation_stats (one row:
    n_regions, n_classes, n_geometry, n_signals, geometry_shap_share:
    the share of the mean |TreeSHAP| of geometry+signals that goes to
    the geometry).

    Raises:
        ValueError: When cfg.model is None, regressor.signals is empty,
            or read_inputs raises.
    """
    if cfg.model is None:
        raise ValueError(f"ablation of {cfg.name} needs a model, got none")
    settings = cfg.method.regressor
    if not settings.signals:
        raise ValueError("ablation needs regressor.signals, got none")
    failures, geometry, _ = read_inputs(cfg)
    signals, _ = region_signals(failures, settings.signals)
    X = np.hstack([geometry, signals])
    observed = failures["failure_rate"].to_numpy()
    n_eval = failures["n_eval"].to_numpy()
    region_class = failures["class"].to_numpy()
    sets = feature_sets(geometry.shape[1], settings.signals)
    fits = [(name, "strata") for name in sets]
    fits += [(name, "class") for name in _GROUPED]

    estimates, scores, shap_share = {}, [], np.nan
    for number, (name, cv) in enumerate(fits, start=1):
        print(f"[ablation] {number}/{len(fits)}: {name} ({cv})", flush=True)
        fitted = fit_failure_regressor(
            X[:, sets[name]],
            observed,
            params=settings.params,
            param_grid=settings.param_grid,
            n_outer_splits=settings.n_outer_splits,
            n_inner_splits=settings.n_inner_splits,
            n_iter=settings.n_iter,
            seed=cfg.seed,
            weight=np.sqrt(n_eval),
            groups=region_class if cv == "class" else None,
        )
        estimates[(cv, name)] = fitted.predicted
        scores.append(
            {
                "feature_set": name,
                "cv": cv,
                "n_features": len(sets[name]),
                **rate_scores(fitted.predicted, observed, n_eval),
                "within_spearman": within_class_spearman(
                    fitted.predicted, observed, region_class
                ),
            }
        )
        if (name, cv) == (f"{GEOMETRY}+{SIGNALS}", "strata"):
            mean_abs = np.abs(fitted.contributions).mean(axis=0)
            shap_share = float(mean_abs[: geometry.shape[1]].sum() / mean_abs.sum())

    comparisons = [("strata", signal) for signal in (*settings.signals, SIGNALS)]
    comparisons.append(("class", SIGNALS))
    gains = []
    for cv, signal in comparisons:
        with_geometry = estimates[(cv, f"{GEOMETRY}+{signal}")]
        without = estimates[(cv, signal)]
        gains.append(
            {
                "cv": cv,
                "signal": signal,
                "with_geometry": f"{GEOMETRY}+{signal}",
                "without": signal,
                "residual_corr": residual_corr(
                    estimates[(cv, GEOMETRY)], without, observed
                ),
                **paired_bootstrap(
                    with_geometry,
                    without,
                    observed,
                    n_eval.astype(np.float64),
                    n_bootstrap=cfg.method.ablation.n_bootstrap,
                    seed=cfg.seed,
                ),
            }
        )
    for gain in gains:
        adds = gain["d_mae_hi"] < 0
        print(
            f"[ablation] {gain['with_geometry']} vs {gain['without']} ({gain['cv']}): "
            f"dMAE {gain['d_mae']:+.4f} [{gain['d_mae_lo']:+.4f}, "
            f"{gain['d_mae_hi']:+.4f}], residual corr {gain['residual_corr']:.2f}: "
            f"the geometry {'adds' if adds else 'does not add'}",
            flush=True,
        )

    pl.DataFrame(
        {
            "region": failures["region"],
            "class": failures["class"],
            "n_eval": n_eval,
            "observed": observed,
            **{f"{cv}:{name}": values for (cv, name), values in estimates.items()},
        }
    ).write_parquet(ablation_file(cfg))
    db.log_table(conn, run_id, "ablation_scores", scores)
    db.log_table(conn, run_id, "ablation_gains", gains)
    db.log_table(
        conn,
        run_id,
        "ablation_stats",
        [
            {
                "n_regions": len(observed),
                "n_classes": len(np.unique(region_class)),
                "n_geometry": geometry.shape[1],
                "n_signals": len(settings.signals),
                "geometry_shap_share": shap_share,
            }
        ],
    )
