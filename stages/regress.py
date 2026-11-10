"""Stage regress: the forest that predicts region failure rates."""

from __future__ import annotations

__all__ = ["read_inputs", "run"]

import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.baselines import BASELINES
from src.config import DatasetConfig
from src.evaluation import rate_scores
from src.regressor import fit_failure_regressor, region_descriptors, region_signals
from stages.paths import (
    complexity_file,
    contributions_file,
    estimates_file,
    failures_file,
    need,
    read_classes,
    regressor_file,
)


def read_inputs(cfg: DatasetConfig) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """Read the used regions and their geometric descriptors.

    Args:
        cfg: The dataset, with its model.

    Returns:
        The used rows of failures_<model>.parquet; X, the
        regressor.region_descriptors of their regressor.measures without
        the all-NaN columns; and the names of its columns.

    Raises:
        ValueError: When complexity.parquet and failures_<model>.parquet
            describe other regions, or regressor.measures names an
            unknown measure.
    """
    complexity = pl.read_parquet(need(complexity_file(cfg), "complexity"))
    failures = pl.read_parquet(need(failures_file(cfg), "failures"))
    described = complexity.filter(pl.col("scope") == "region")["id"].unique()
    if set(described.to_list()) != set(failures["region"].to_list()):
        raise ValueError(
            "complexity.parquet describes other regions than "
            f"{failures_file(cfg).name}: run complexity and failures again"
        )
    failures = failures.filter(pl.col("used"))
    X, names = region_descriptors(
        complexity,
        failures["region"].to_numpy(),
        failures["class"].to_numpy(),
        cfg.method.regressor.measures,
    )
    present = ~np.isnan(X).all(axis=0)
    return failures, X[:, present], [n for n, kept in zip(names, present) if kept]


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Fit the failure regressor and explain its estimates.

    Reads complexity.parquet, failures_<model>.parquet and classes.json.
    X = read_inputs, then regressor.region_signals of
    regressor.signals; y = failure_rate.
    regressor.fit_failure_regressor with the regressor config and seed,
    each region weighted by sqrt(n_eval). The estimates are scored
    against the observed rates (wmae weighted by n_eval), next to each
    of baselines.BASELINES on the same regions and to the mean of the
    estimate and each baseline ("rf+<baseline>").

    Writes to output_dir: regress_<model>_estimates.parquet (region,
    class, observed, predicted, fold, base, then each baseline, then
    each rf+<baseline>),
    regress_<model>_contributions.parquet (region, descriptor, value,
    contribution) and regress_<model>.joblib (the refit forest and its
    descriptor names).
    Logs: estimates (region, class name, observed, predicted, fold,
    base), contributions (region, descriptor, value, contribution),
    regressor_folds (FailureRegressor.folds), importances (descriptor,
    mdi, mean_abs_shap), regress_scores (predictor "rf", a baseline or
    "rf+<baseline>", spearman, r2, mae, mse, wmae).

    Raises:
        ValueError: When cfg.model is None, or read_inputs raises.
    """
    if cfg.model is None:
        raise ValueError(f"regress of {cfg.name} needs a model, got none")
    settings = cfg.method.regressor
    failures, geometry, names = read_inputs(cfg)
    signals, signal_names = region_signals(failures, settings.signals)
    X = np.hstack([geometry, signals])
    names = [*names, *signal_names]
    regions = failures["region"].to_numpy()
    observed = failures["failure_rate"].to_numpy()
    n_eval = failures["n_eval"].to_numpy()
    print(f"[regress] {len(regions)} regions, {len(names)} descriptors", flush=True)
    fitted = fit_failure_regressor(
        X,
        observed,
        params=settings.params,
        param_grid=settings.param_grid,
        n_outer_splits=settings.n_outer_splits,
        n_inner_splits=settings.n_inner_splits,
        n_iter=settings.n_iter,
        seed=cfg.seed,
        weight=np.sqrt(n_eval),
    )
    joblib.dump({"model": fitted.model, "descriptors": names}, regressor_file(cfg))
    estimates = pl.DataFrame(
        {
            "region": regions,
            "class": failures["class"],
            "observed": observed,
            "predicted": fitted.predicted,
            "fold": fitted.fold,
            "base": fitted.base,
            **{name: failures[name] for name in BASELINES},
            **{
                f"rf+{name}": (fitted.predicted + failures[name].to_numpy()) / 2
                for name in BASELINES
            },
        }
    )
    estimates.write_parquet(estimates_file(cfg))
    contributions = pl.DataFrame(
        {
            "region": np.repeat(regions, len(names)),
            "descriptor": names * len(regions),
            "value": X.ravel(),
            "contribution": fitted.contributions.ravel(),
        }
    ).with_columns(pl.col("value").fill_nan(None))
    contributions.write_parquet(contributions_file(cfg))
    columns = {"rf": "predicted"} | {
        name: name for name in (*BASELINES, *(f"rf+{b}" for b in BASELINES))
    }
    scores = [
        {
            "predictor": name,
            **rate_scores(estimates[column].to_numpy(), observed, n_eval),
        }
        for name, column in columns.items()
    ]
    print(
        "[regress] spearman "
        + ", ".join(
            f"{score['predictor']} {score['spearman']:.3f}" for score in scores
        ),
        flush=True,
    )

    classes = read_classes(cfg)
    db.log_table(
        conn,
        run_id,
        "estimates",
        [
            {
                key: row[key]
                for key in ("region", "observed", "predicted", "fold", "base")
            }
            | {"class": classes[row["class"]]}
            for row in estimates.to_dicts()
        ],
    )
    db.log_table(conn, run_id, "contributions", contributions.to_dicts())
    db.log_table(conn, run_id, "regressor_folds", fitted.folds)
    mean_abs_shap = np.abs(fitted.contributions).mean(axis=0)
    db.log_table(
        conn,
        run_id,
        "importances",
        [
            {"descriptor": name, "mdi": float(mdi), "mean_abs_shap": float(abs_shap)}
            for name, mdi, abs_shap in zip(names, fitted.importances, mean_abs_shap)
        ],
    )
    db.log_table(conn, run_id, "regress_scores", scores)
