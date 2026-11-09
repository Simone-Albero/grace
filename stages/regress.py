"""Stage regress: the forest that predicts region failure rates."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.baselines import BASELINES
from src.config import DatasetConfig
from src.evaluation import rate_scores
from src.regressor import fit_failure_regressor, region_descriptors
from stages.paths import (
    complexity_file,
    contributions_file,
    estimates_file,
    failures_file,
    need,
    read_classes,
    regressor_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Fit the failure regressor and explain its estimates.

    Reads complexity.parquet, failures_<model>.parquet and classes.json.
    X = regressor.region_descriptors of the used regions with
    regressor.measures, without its all-NaN columns; y = failure_rate.
    regressor.fit_failure_regressor with the regressor config and seed.
    The estimates are scored against the observed rates, next to each
    of baselines.BASELINES on the same regions and to the mean of the
    estimate and each baseline ("rf+<baseline>").

    Writes to output_dir: regress_<model>_estimates.parquet (region,
    class, observed, predicted, fold, base, mcp, atc, meta, rf+mcp,
    rf+atc, rf+meta),
    regress_<model>_contributions.parquet (region, descriptor, value,
    contribution) and regress_<model>.joblib (the refit forest and its
    descriptor names).
    Logs: estimates (region, class name, observed, predicted, fold,
    base), contributions (region, descriptor, value, contribution),
    regressor_folds (FailureRegressor.folds), importances (descriptor,
    mdi, mean_abs_shap), regress_scores (predictor "rf", a baseline or
    "rf+<baseline>", spearman, r2, mae, mse).

    Raises:
        ValueError: When cfg.model is None, or regressor.measures names
            an unknown measure.
    """
    if cfg.model is None:
        raise ValueError(f"regress of {cfg.name} needs a model, got none")
    settings = cfg.method.regressor
    complexity = pl.read_parquet(need(complexity_file(cfg), "complexity"))
    failures = pl.read_parquet(need(failures_file(cfg), "failures"))
    described = complexity.filter(pl.col("scope") == "region")["id"].unique()
    if set(described.to_list()) != set(failures["region"].to_list()):
        raise ValueError(
            "complexity.parquet describes other regions than "
            f"{failures_file(cfg).name}: run complexity and failures again"
        )
    failures = failures.filter(pl.col("used"))
    regions = failures["region"].to_numpy()
    X, names = region_descriptors(
        complexity, regions, failures["class"].to_numpy(), settings.measures
    )
    present = ~np.isnan(X).all(axis=0)
    X = X[:, present]
    names = [name for name, is_present in zip(names, present) if is_present]
    observed = failures["failure_rate"].to_numpy()
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
        {"predictor": name, **rate_scores(estimates[column].to_numpy(), observed)}
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
