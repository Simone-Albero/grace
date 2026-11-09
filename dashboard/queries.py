"""Read-only queries: db tables and parquet files to JSON-able dicts.

Every function raises LookupError("run <stage> first") when the stage
output is missing (no db, table, run or file). The db is opened
read-only (file:...?mode=ro). The run shown is the latest one with
status "ok" for the stage and dataset (and model). Every "run" value is
{id, started, finished}. Non-finite numbers become None (JSON has no
NaN).
"""

from __future__ import annotations

__all__ = [
    "complexity",
    "failures",
    "region",
    "regions",
    "regress",
    "split",
    "train",
]

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import DatasetConfig, ModelConfig
from stages.paths import estimates_file, predictions_file, read_classes

CONFIDENCE_BINS = 20


def split(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last split run.

    Args:
        cfg: The dataset.

    Returns:
        run; summary {n_raw, n_incomplete, n_rare, n_kept, n_classes};
        classes [{label, train, val, test}] sorted by label;
        dropped_classes [{label, n_rows}]; features {num: [...], cat:
        [...]}.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "split", cfg)
        summary = _rows(conn, "split", "split_summary", run)[0]
        features = _rows(conn, "split", "features", run)
        return {
            "run": _public(run),
            "summary": summary,
            "classes": sorted(
                _rows(conn, "split", "split_class_counts", run),
                key=lambda r: r["label"],
            ),
            "dropped_classes": _rows(conn, "split", "dropped_classes", run),
            "features": {
                kind: [f["column"] for f in features if f["kind"] == kind]
                for kind in ("num", "cat")
            },
        }


def regions(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last regions run.

    Args:
        cfg: The dataset.

    Returns:
        run; stats {n_regions, algorithm, distance, n_graph_nodes,
        eval_ratio}; classes [{class, n_train, n_regions, loss,
        params}]; candidates [{class, params, n_clusters, loss}];
        regions [{region, class, n_train, n_val, n_test}].
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "regions", cfg)
        classes = _rows(conn, "regions", "class_clusterings", run)
        candidates = _rows(conn, "regions", "granularity_candidates", run)
        for row in classes + candidates:
            row["params"] = json.loads(row["params"])
        return {
            "run": _public(run),
            "stats": _rows(conn, "regions", "region_stats", run)[0],
            "classes": classes,
            "candidates": candidates,
            "regions": _rows(conn, "regions", "regions", run),
        }


def complexity(cfg: DatasetConfig, measure: str | None) -> dict[str, Any]:
    """Return the last complexity run, focused on one measure.

    Args:
        cfg: The dataset.
        measure: A measure of the run; None picks the first.

    Returns:
        run; measures [names, table order]; measure (the one shown);
        regions [{region, class, value}]; classes [{class, value}];
        summary [{measure, mean, min, max, n_null}] over the regions.

    Raises:
        LookupError: Also when measure is not a measure of the run.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "complexity", cfg)
        rows = _rows(conn, "complexity", "complexity", run)
        region_class = _region_classes(conn, cfg)
    measures = list(dict.fromkeys(r["measure"] for r in rows))
    measure = measures[0] if measure is None else measure
    if measure not in measures:
        raise LookupError(f"unknown measure: {measure}")
    class_names = _classes(cfg)
    by_region = defaultdict(list)
    for r in rows:
        if r["scope"] == "region":
            by_region[r["measure"]].append(r["value"])
    summary = []
    for m in measures:
        values = np.array([np.nan if v is None else v for v in by_region[m]])
        known = values[np.isfinite(values)]
        summary.append(
            {
                "measure": m,
                "mean": _finite(known.mean()) if known.size else None,
                "min": _finite(known.min()) if known.size else None,
                "max": _finite(known.max()) if known.size else None,
                "n_null": int(values.size - known.size),
            }
        )
    chosen = [r for r in rows if r["measure"] == measure]
    return {
        "run": _public(run),
        "measures": measures,
        "measure": measure,
        "regions": [
            {"region": r["id"], "class": region_class.get(r["id"]), "value": r["value"]}
            for r in chosen
            if r["scope"] == "region"
        ],
        "classes": [
            {"class": class_names[r["id"]], "value": r["value"]}
            for r in chosen
            if r["scope"] == "class"
        ],
        "summary": summary,
    }


def train(cfg: DatasetConfig, model: str, split: str) -> dict[str, Any]:
    """Return the last train run of model, on val or test.

    Args:
        cfg: The dataset.
        model: A model name.
        split: "val" or "test".

    Returns:
        run; models [models with an ok train run]; overall {accuracy,
        macro_f1, weighted_f1}; classes [{label, precision, recall, f1,
        support}]; confusion {labels, counts (rows true, columns pred)};
        confidence {edges (CONFIDENCE_BINS + 1), correct, wrong} (counts
        per bin, from the predictions file).
    """
    model_cfg = _with_model(cfg, model)
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "train", cfg, model)
        models = _models_with_run(conn, "train", cfg)
        metrics = [
            r for r in _rows(conn, "train", "model_metrics", run) if r["split"] == split
        ]
        cells = [
            r for r in _rows(conn, "train", "confusion", run) if r["split"] == split
        ]
    class_names = _classes(cfg)
    overall = {
        r["metric"]: r["value"]
        for r in metrics
        if r["label"] == "all"
        and r["metric"] in ("accuracy", "macro_f1", "weighted_f1")
    }
    per_class = defaultdict(dict)
    for r in metrics:
        if r["label"] != "all":
            per_class[r["label"]][r["metric"]] = r["value"]
    position = {name: i for i, name in enumerate(class_names)}
    counts = np.zeros((len(class_names), len(class_names)), dtype=int)
    for cell in cells:
        counts[position[cell["true"]], position[cell["pred"]]] = cell["count"]
    predictions = _parquet(predictions_file(model_cfg, split), "train")
    correct = (predictions["label"] == predictions["pred"]).to_numpy()
    confidence = predictions["confidence"].to_numpy()
    edges = np.linspace(0.0, 1.0, CONFIDENCE_BINS + 1)
    return {
        "run": _public(run),
        "models": models,
        "overall": overall,
        "classes": [{"label": name, **per_class[name]} for name in class_names],
        "confusion": {"labels": class_names, "counts": counts.tolist()},
        "confidence": {
            "edges": edges.tolist(),
            "correct": np.histogram(confidence[correct], edges)[0].tolist(),
            "wrong": np.histogram(confidence[~correct], edges)[0].tolist(),
        },
    }


def failures(cfg: DatasetConfig, model: str) -> dict[str, Any]:
    """Return the last failures run of model.

    Args:
        cfg: The dataset.
        model: A model name.

    Returns:
        run; models [models with an ok failures run]; stats {n_regions,
        n_used, n_empty, n_small, error_rate, error_rate_used,
        atc_threshold}; scores [{predictor, spearman, r2, mae, mse}];
        regions [{region, class, n_eval, n_error, failure_rate, used,
        mcp, atc}].
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "failures", cfg, model)
        return {
            "run": _public(run),
            "models": _models_with_run(conn, "failures", cfg),
            "stats": _rows(conn, "failures", "failure_stats", run)[0],
            "scores": _rows(conn, "failures", "baseline_scores", run),
            "regions": [
                {**r, "used": bool(r["used"])}
                for r in _rows(conn, "failures", "failures", run)
            ],
        }


def regress(cfg: DatasetConfig, model: str) -> dict[str, Any]:
    """Return the last regress run of model.

    Args:
        cfg: The dataset.
        model: A model name.

    Returns:
        run; models [models with an ok regress run]; scores [{predictor,
        spearman, r2, mae, mse}] (rf, mcp, atc); estimates [{region,
        class, observed, predicted, fold, base, mcp, atc}]; importances
        [{descriptor, mdi, mean_abs_shap}] by mdi descending; folds
        [{fold, n_regions, spearman, r2, mae, best_score, params}].
    """
    model_cfg = _with_model(cfg, model)
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "regress", cfg, model)
        models = _models_with_run(conn, "regress", cfg)
        scores = _rows(conn, "regress", "regress_scores", run)
        importances = _rows(conn, "regress", "importances", run)
        folds = _rows(conn, "regress", "regressor_folds", run)
    class_names = _classes(cfg)
    estimates = _parquet(estimates_file(model_cfg), "regress").to_dicts()
    return {
        "run": _public(run),
        "models": models,
        "scores": scores,
        "estimates": [
            {
                k: _json_value(v)
                for k, v in {**r, "class": class_names[r["class"]]}.items()
            }
            for r in estimates
        ],
        "importances": sorted(importances, key=lambda r: -r["mdi"]),
        "folds": [{**f, "params": json.loads(f["params"])} for f in folds],
    }


def region(cfg: DatasetConfig, model: str, region_id: int) -> dict[str, Any]:
    """Return the explanation of one region of the last regress run.

    Args:
        cfg: The dataset.
        model: A model name.
        region_id: A region id.

    Returns:
        run; region {region, class, observed, predicted, fold, base};
        contributions [{descriptor, value, contribution}] by
        |contribution| descending; descriptors [{measure, region, class,
        class_mean, dataset_mean}] for every measure of the complexity
        run (class_mean and dataset_mean: means over the regions of its
        class and over all regions).

    Raises:
        LookupError: Also when region_id is not a used region of the
            run.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "regress", cfg, model)
        estimate = _rows(conn, "regress", "estimates", run, region=region_id)
        if not estimate:
            raise LookupError(f"region {region_id} has no estimate in run {run['id']}")
        contributions = _rows(conn, "regress", "contributions", run, region=region_id)
        complexity_run = _latest_run(conn, "complexity", cfg)
        rows = _rows(conn, "complexity", "complexity", complexity_run)
        region_class = _region_classes(conn, cfg)
    class_names = _classes(cfg)
    own_class = estimate[0]["class"]
    values = defaultdict(dict)
    for r in rows:
        values[(r["scope"], r["id"])][r["measure"]] = r["value"]
    measures = list(dict.fromkeys(r["measure"] for r in rows))
    same_class = [i for i, c in region_class.items() if c == own_class]

    def mean(ids: list[int], measure: str) -> float | None:
        of_regions = [values[("region", i)].get(measure) for i in ids]
        known = [v for v in of_regions if v is not None]
        return _finite(float(np.mean(known))) if known else None

    return {
        "run": _public(run),
        "region": estimate[0],
        "contributions": sorted(contributions, key=lambda r: -abs(r["contribution"])),
        "descriptors": [
            {
                "measure": m,
                "region": values[("region", region_id)].get(m),
                "class": values[("class", class_names.index(own_class))].get(m),
                "class_mean": mean(same_class, m),
                "dataset_mean": mean(list(region_class), m),
            }
            for m in measures
        ],
    }


def _finite(value: Any) -> float | None:
    """Return a finite float, or None (JSON has no NaN)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _json_value(value: Any) -> Any:
    """Return floats through _finite, other values unchanged."""
    return _finite(value) if isinstance(value, float) else value


def _connect(cfg: DatasetConfig) -> sqlite3.Connection:
    if not cfg.db_path.exists():
        raise LookupError("run split first")
    conn = sqlite3.connect(f"file:{cfg.db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_run(
    conn: sqlite3.Connection, stage: str, cfg: DatasetConfig, model: str | None = None
) -> dict[str, Any]:
    """Return the last ok run of a stage on the dataset (and model)."""
    row = conn.execute(
        "SELECT id, started, finished FROM runs WHERE stage = ? AND dataset = ? "
        "AND status = 'ok' AND (? IS NULL OR model = ?) ORDER BY id DESC LIMIT 1",
        (stage, cfg.name, model, model),
    ).fetchone()
    if row is None:
        raise LookupError(f"run {stage} first")
    return dict(row)


def _public(run: Mapping[str, Any]) -> dict[str, Any]:
    return {k: run[k] for k in ("id", "started", "finished")}


def _rows(
    conn: sqlite3.Connection,
    stage: str,
    table: str,
    run: Mapping[str, Any],
    **where: Any,
) -> list[dict[str, Any]]:
    """Return the rows of a table logged by a run, without run_id.

    A missing table means the stage never logged it: LookupError.
    """
    filters = "".join(f' AND "{k}" = ?' for k in where)
    try:
        rows = conn.execute(
            f'SELECT * FROM "{table}" WHERE run_id = ?{filters}',
            (run["id"], *where.values()),
        ).fetchall()
    except sqlite3.OperationalError as e:
        raise LookupError(f"run {stage} first") from e
    return [{k: _json_value(r[k]) for k in r.keys() if k != "run_id"} for r in rows]


def _models_with_run(
    conn: sqlite3.Connection, stage: str, cfg: DatasetConfig
) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT model FROM runs WHERE stage = ? AND dataset = ? "
        "AND status = 'ok' AND model IS NOT NULL ORDER BY model",
        (stage, cfg.name),
    )
    return [model for (model,) in rows]


def _region_classes(conn: sqlite3.Connection, cfg: DatasetConfig) -> dict[int, str]:
    """Return the class name of every region of the last regions run."""
    run = _latest_run(conn, "regions", cfg)
    return {r["region"]: r["class"] for r in _rows(conn, "regions", "regions", run)}


def _classes(cfg: DatasetConfig) -> list[str]:
    try:
        return read_classes(cfg)
    except FileNotFoundError as e:
        raise LookupError(str(e)) from e


def _parquet(path: Path, stage: str) -> pl.DataFrame:
    if not path.exists():
        raise LookupError(f"run {stage} first")
    return pl.read_parquet(path)


def _with_model(cfg: DatasetConfig, model: str) -> DatasetConfig:
    return replace(cfg, model=ModelConfig(name=model, params={}))
