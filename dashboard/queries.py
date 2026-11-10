"""Read-only queries: db tables and parquet files to JSON-able dicts.

Every function raises LookupError("run <stage> first") when the stage
output is missing (no db, table, run or file). The db is opened
read-only (file:...?mode=ro). The run shown is the latest one with
status "ok" for the stage and dataset. Every "run" value is
{id, started, finished}. Non-finite numbers become None (JSON has no
NaN).
"""

from __future__ import annotations

__all__ = ["clean", "complexity", "regions"]

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from typing import Any

import numpy as np

from src.config import DatasetConfig
from stages.paths import read_classes


def clean(cfg: DatasetConfig) -> dict[str, Any]:
    """Return the last clean run.

    Args:
        cfg: The dataset.

    Returns:
        run; summary {n_raw, n_duplicate, n_rare, n_kept, n_classes,
        n_constant}; classes [{label, n_read, n_rows}] sorted by label;
        dropped_classes [{label, n_rows}]; dropped_columns [column];
        features {num: [...], cat: [...]}.
    """
    with closing(_connect(cfg)) as conn:
        run = _latest_run(conn, "clean", cfg)
        summary = _rows(conn, "clean", "clean_summary", run)[0]
        features = _rows(conn, "clean", "features", run)
        return {
            "run": _public(run),
            "summary": summary,
            "classes": sorted(
                _rows(conn, "clean", "class_counts", run),
                key=lambda r: r["label"],
            ),
            "dropped_classes": _rows(
                conn, "clean", "dropped_classes", run, missing_ok=True
            ),
            "dropped_columns": [
                row["column"]
                for row in _rows(conn, "clean", "dropped_columns", run, missing_ok=True)
            ],
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
        run; stats {n_regions, algorithm, distance, n_graph_nodes};
        classes [{class, n_rows, n_regions, loss, params}]; candidates
        [{class, params, n_clusters, loss}]; regions [{region, class,
        n_rows}].
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
        raise LookupError("run clean first")
    conn = sqlite3.connect(f"file:{cfg.db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_run(
    conn: sqlite3.Connection, stage: str, cfg: DatasetConfig
) -> dict[str, Any]:
    """Return the last ok run of a stage on the dataset."""
    row = conn.execute(
        "SELECT id, started, finished FROM runs WHERE stage = ? AND dataset = ? "
        "AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (stage, cfg.name),
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
    missing_ok: bool = False,
) -> list[dict[str, Any]]:
    """Return the rows of a table logged by a run, without run_id.

    A missing table means the stage never logged it: LookupError, or no
    rows when missing_ok, for a table that db.log_table skips when a
    run has nothing to log in it.
    """
    try:
        rows = conn.execute(
            f'SELECT * FROM "{table}" WHERE run_id = ?', (run["id"],)
        ).fetchall()
    except sqlite3.OperationalError as e:
        if missing_ok and "no such table" in str(e):
            return []
        raise LookupError(f"run {stage} first") from e
    return [{k: _json_value(r[k]) for k in r.keys() if k != "run_id"} for r in rows]


def _region_classes(conn: sqlite3.Connection, cfg: DatasetConfig) -> dict[int, str]:
    """Return the class name of every region of the last regions run."""
    run = _latest_run(conn, "regions", cfg)
    return {r["region"]: r["class"] for r in _rows(conn, "regions", "regions", run)}


def _classes(cfg: DatasetConfig) -> list[str]:
    try:
        return read_classes(cfg)
    except FileNotFoundError as e:
        raise LookupError(str(e)) from e
