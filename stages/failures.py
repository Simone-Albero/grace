"""Stage failures: observed failure rates and baselines per region."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import polars as pl

from src import db
from src.baselines import (
    BASELINES,
    atc_rate,
    atc_threshold,
    bcts_confidence,
    error_inputs,
    error_probability,
    failure_rates,
    fit_bcts,
    mcp_rate,
    meta_rate,
)
from src.config import DatasetConfig
from src.data import LABEL
from src.evaluation import rate_scores
from src.preprocessing import feature_cols
from stages.paths import (
    assignments_file,
    failures_file,
    need,
    predictions_file,
    read_classes,
    read_split,
    regions_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Count the test errors per region and score the baselines.

    Reads regions.parquet, the val and test rows of assignments.parquet
    and pred_<model>_{val,test}.parquet (rows must line up by position),
    and the val and test split files. The failure rate is counted on
    test only. Fitted on val: the ATC threshold on the confidence, the
    BCTS calibration (baselines.fit_bcts) and a second ATC threshold on
    the calibrated confidence, and meta (baselines.error_probability on
    baselines.error_inputs, error_model, top_n, seed). A region is used
    when it has at least regressor.min_eval_support test rows; each of
    baselines.BASELINES is scored (evaluation.rate_scores) against the
    observed rates of the used regions.

    Writes to output_dir: failures_<model>.parquet (region, class,
    n_eval, n_error, failure_rate, used, then one column per baseline),
    one row per region in the order of regions.parquet.
    Logs: failures (the same rows, class as name), baseline_scores
    (predictor: a baseline, spearman, r2, mae, mse), failure_stats (one
    row: n_regions, n_used, n_empty, n_small, error_rate,
    error_rate_used, atc_threshold, atc_cal_threshold, temperature),
    calibration (class, bias: the BCTS bias, minus its mean).

    Raises:
        ValueError: When cfg.model is None, the predictions do not line
            up with the assignments, or no region is used.
    """
    if cfg.model is None:
        raise ValueError(f"failures of {cfg.name} needs a model, got none")
    regions = pl.read_parquet(need(regions_file(cfg), "regions"))
    assignments = pl.read_parquet(need(assignments_file(cfg), "regions"))
    predictions, region, rows = {}, {}, {}
    for split in ("val", "test"):
        predictions[split] = pl.read_parquet(
            need(predictions_file(cfg, split), "train")
        )
        assigned = assignments.filter(pl.col("split") == split).sort("row")
        if assigned["row"].to_list() != predictions[split]["row"].to_list():
            raise ValueError(
                f"{split}: the predictions ({predictions[split].height} rows) do not "
                f"line up with the region assignments ({assigned.height} rows)"
            )
        rows[split] = read_split(cfg, split)
        if not predictions[split]["label"].equals(rows[split][LABEL]):
            raise ValueError(
                f"{split}: the predictions were made on another split "
                "(their labels differ from the split's): run train again"
            )
        region[split] = assigned["region"].to_numpy()

    ids = regions["region"].to_numpy()
    test, val = predictions["test"], predictions["val"]
    error = (test["label"] != test["pred"]).to_numpy()
    n_eval, n_error, rate = failure_rates(region["test"], error, ids)
    confidence = test["confidence"].to_numpy()
    val_error = (val["label"] != val["pred"]).to_numpy()
    threshold = atc_threshold(val["confidence"].to_numpy(), ~val_error)
    used = n_eval >= cfg.method.regressor.min_eval_support
    if not used.any():
        raise ValueError(
            f"no region has min_eval_support={cfg.method.regressor.min_eval_support} "
            f"test rows (most: {n_eval.max()})"
        )
    proba_cols = [column for column in val.columns if column.startswith("proba_")]
    temperature, bias = fit_bcts(
        val.select(proba_cols).to_numpy(), val["label"].to_numpy()
    )
    calibrated = {
        split: bcts_confidence(
            frame.select(proba_cols).to_numpy(),
            frame["pred"].to_numpy(),
            temperature,
            bias,
        )
        for split, frame in predictions.items()
    }
    cal_threshold = atc_threshold(calibrated["val"], ~val_error)
    features = feature_cols(rows["val"].columns, cfg.num_cols, cfg.cat_cols)
    val_inputs, inputs = error_inputs(rows["val"], val, features)
    test_inputs, _ = error_inputs(rows["test"], test, features)
    n_fit = min(val.height, cfg.method.error_model.max_samples)
    print(f"[failures] fitting meta on {n_fit} of {val.height} val rows", flush=True)
    error_probability_test = error_probability(
        val_inputs,
        val_error,
        test_inputs,
        inputs,
        params=cfg.method.error_model.params,
        max_samples=cfg.method.error_model.max_samples,
        top_n=cfg.top_n,
        seed=cfg.seed,
    )
    failures = pl.DataFrame(
        {
            "region": ids,
            "class": regions["class"],
            "n_eval": n_eval,
            "n_error": n_error,
            "failure_rate": rate,
            "used": used,
            "mcp": mcp_rate(confidence, region["test"], ids),
            "atc": atc_rate(confidence, region["test"], ids, threshold),
            "mcp_cal": mcp_rate(calibrated["test"], region["test"], ids),
            "atc_cal": atc_rate(calibrated["test"], region["test"], ids, cal_threshold),
            "meta": meta_rate(error_probability_test, region["test"], ids),
        }
    )
    failures.write_parquet(failures_file(cfg))
    observed = rate[used]
    scores = [
        {"predictor": name, **rate_scores(failures[name].to_numpy()[used], observed)}
        for name in BASELINES
    ]
    print(
        f"[failures] {used.sum()} of {len(ids)} regions used; "
        + ", ".join(
            f"{score['predictor']} spearman {score['spearman']:.3f}" for score in scores
        ),
        flush=True,
    )

    classes = read_classes(cfg)
    db.log_table(
        conn,
        run_id,
        "failures",
        [
            {**row, "class": classes[row["class"]], "used": bool(row["used"])}
            for row in failures.to_dicts()
        ],
    )
    db.log_table(conn, run_id, "baseline_scores", scores)
    db.log_table(
        conn,
        run_id,
        "failure_stats",
        [
            {
                "n_regions": len(ids),
                "n_used": int(used.sum()),
                "n_empty": int((n_eval == 0).sum()),
                "n_small": int(((n_eval > 0) & ~used).sum()),
                "error_rate": float(error.mean()),
                "error_rate_used": float(n_error[used].sum() / n_eval[used].sum()),
                "atc_threshold": threshold,
                "atc_cal_threshold": cal_threshold,
                "temperature": temperature,
            }
        ],
    )
    db.log_table(
        conn,
        run_id,
        "calibration",
        [
            {"class": name, "bias": float(value)}
            for name, value in zip(classes, bias - bias.mean())
        ],
    )
