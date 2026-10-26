"""Read-only queries of the dashboard on grace.db."""

__all__ = [
    "CONFIG_KEYS",
    "METRICS",
    "PREDICTORS",
    "REGION_COLUMNS",
    "SCORES",
    "class_metrics",
    "connect",
    "descriptors",
    "figure_file",
    "figures",
    "job",
    "job_runs",
    "jobs",
    "preset",
    "presets",
    "regions",
    "regressor_folds",
    "results",
    "run",
    "run_stages",
    "runs",
    "stage_ids",
    "stage_metrics",
]

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

# Each config key is a results column, read from this path of the
# JSON in runs.config.
CONFIG_KEYS = {
    "data": "$.groups.data",
    "classifier": "$.groups.classifier",
    "clustering": "$.groups.clustering",
    "distance": "$.distance",
    "seed": "$.seed",
}
# The failure-rate predictors, each with its label and the stage whose
# execution holds its scores; the `regions` columns carry its rates.
PREDICTORS = {
    "regressor": ("RF regressor", "regress"),
    "val": ("Val rate", "baselines"),
    "mcp": ("MCP", "baselines"),
    "atc": ("ATC", "baselines"),
}
SCORES = ("spearman", "r2", "mae")
# The results metrics, each with the stage whose execution holds it.
METRICS = {
    "accuracy": "classify",
    "f1_macro": "classify",
    **{
        f"{key}_{score}": stage
        for key, (_, stage) in PREDICTORS.items()
        for score in SCORES
    },
    "n_regions_used": "baselines",
}
# The numeric columns of `regions` other than the complexity measures;
# they follow the SELECT there and its abs_error.
REGION_COLUMNS = (
    "n_train n_val n_test n_val_error val_rate n_test_error test_rate used fold "
    "regressor val mcp atc abs_error"
).split()
_CONFIG_COLUMNS = ", ".join(
    f"json_extract(r.config, '{path}') AS {name}" for name, path in CONFIG_KEYS.items()
)
_RUN_COLUMNS = (
    "r.id, r.targets, r.status, r.error, "
    f"r.started_at, r.finished_at, {_CONFIG_COLUMNS}"
)
_METRIC_COLUMNS = ", ".join(
    f"(SELECT value FROM metrics WHERE name = '{name}' "
    f"AND execution_id = d.{stage}_id) AS {name}"
    for name, stage in METRICS.items()
)


def connect(path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the DB at `path`.

    Args:
        path: The grace.db file.

    Returns:
        The connection; its rows are `sqlite3.Row`.
    """
    # FastAPI runs the dependency and the endpoint in different threads,
    # but the connection still serves a single request.
    db = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
    )
    db.row_factory = sqlite3.Row
    return db


def _all(
    db: sqlite3.Connection,
    sql: str,
    params: Sequence[object] | Mapping[str, object] = (),
) -> list[dict[str, object]]:
    """Return the rows of a query as dicts."""
    return [dict(row) for row in db.execute(sql, params)]


def results(db: sqlite3.Connection) -> list[dict[str, object]]:
    """List one row per done failures execution, newest run first.

    A row shows the run that linked the execution and also a done
    regress, else a done baselines, else the latest one, with the
    baselines and regress executions of that run.

    Args:
        db: A connection to grace.db.

    Returns:
        The rows: run, execution ids, `n_runs` (runs that linked the
        failures execution), config values and metrics (None when its
        stage did not run).
    """
    return _all(
        db,
        f"""
        WITH links AS (
            SELECT f.run_id, f.execution_id AS failures_id,
                   json_extract(e.inputs, '$.classify') AS classify_id,
                   b.id AS baselines_id, g.id AS regress_id
            FROM run_stages f
            JOIN executions e ON e.id = f.execution_id AND e.status = 'done'
            LEFT JOIN run_stages sb
              ON sb.run_id = f.run_id AND sb.stage = 'baselines'
            LEFT JOIN executions b ON b.id = sb.execution_id AND b.status = 'done'
            LEFT JOIN run_stages sg
              ON sg.run_id = f.run_id AND sg.stage = 'regress'
            LEFT JOIN executions g ON g.id = sg.execution_id AND g.status = 'done'
            WHERE f.stage = 'failures'
        ), d AS (
            SELECT *, COUNT(*) OVER (PARTITION BY failures_id) AS n_runs,
                   ROW_NUMBER() OVER (
                       PARTITION BY failures_id
                       ORDER BY regress_id IS NULL, baselines_id IS NULL, run_id DESC
                   ) AS pick
            FROM links
        )
        SELECT d.run_id, d.failures_id, d.baselines_id, d.regress_id, d.n_runs,
               {_CONFIG_COLUMNS}, {_METRIC_COLUMNS}
        FROM d JOIN runs r ON r.id = d.run_id
        WHERE d.pick = 1
        ORDER BY d.run_id DESC
        """,
    )


def _decode(row: dict[str, object], *keys: str) -> dict[str, object]:
    """Return `row` with the JSON text of `keys` decoded."""
    return {**row, **{key: json.loads(row[key]) for key in keys}}


def runs(db: sqlite3.Connection) -> list[dict[str, object]]:
    """List every run, newest first.

    Args:
        db: A connection to grace.db.

    Returns:
        The runs, with decoded `targets` and the config values.
    """
    rows = _all(db, f"SELECT {_RUN_COLUMNS} FROM runs r ORDER BY r.id DESC")
    return [_decode(row, "targets") for row in rows]


def run(db: sqlite3.Connection, run_id: int) -> dict[str, object] | None:
    """Return a run with its parsed config.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        The run, with decoded `targets` and `config`, or None.
    """
    rows = _all(
        db, f"SELECT {_RUN_COLUMNS}, r.config FROM runs r WHERE r.id = ?", (run_id,)
    )
    return _decode(rows[0], "targets", "config") if rows else None


def run_stages(db: sqlite3.Connection, run_id: int) -> list[dict[str, object]]:
    """List the stage executions of a run, upstream first.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        Rows with the stage, execution id, cached flag, status, start,
        duration and error.
    """
    # An execution id is larger than those of its inputs: a valid DAG
    # order.
    return _all(
        db,
        """
        SELECT rs.stage, rs.execution_id, rs.cached, e.status, e.started_at,
               e.duration_s, e.error
        FROM run_stages rs JOIN executions e ON e.id = rs.execution_id
        WHERE rs.run_id = ? ORDER BY e.id
        """,
        (run_id,),
    )


def stage_ids(db: sqlite3.Connection, run_id: int) -> dict[str, int]:
    """Map every stage of a run to its execution id.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        The execution id by stage name.
    """
    return {row["stage"]: row["execution_id"] for row in run_stages(db, run_id)}


def stage_metrics(
    db: sqlite3.Connection, run_id: int
) -> dict[str, list[tuple[str, float | None]]]:
    """List the metrics of each stage of a run, in insertion order.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        The `(name, value)` pairs by stage name.
    """
    grouped: dict[str, list[tuple[str, float | None]]] = {}
    rows = db.execute(
        """
        SELECT e.stage, m.name, m.value
        FROM run_stages rs
        JOIN executions e ON e.id = rs.execution_id
        JOIN metrics m ON m.execution_id = e.id
        WHERE rs.run_id = ? ORDER BY e.id, m.rowid
        """,
        (run_id,),
    )
    for stage, name, value in rows:
        grouped.setdefault(stage, []).append((name, value))
    return grouped


def class_metrics(
    db: sqlite3.Connection, prepare_id: int, classify_id: int
) -> list[dict[str, object]]:
    """List the test size and classifier scores of every class.

    Args:
        db: A connection to grace.db.
        prepare_id: The execution id of prepare.
        classify_id: The execution id of classify.

    Returns:
        One row per class: id, name, `n_test`, precision, recall, f1.
    """
    return _all(
        db,
        """
        SELECT k.class_id, k.name, k.n_test, m.precision, m.recall, m.f1
        FROM classes k LEFT JOIN class_metrics m
          ON m.execution_id = ? AND m.class_id = k.class_id
        WHERE k.execution_id = ? ORDER BY k.class_id
        """,
        (classify_id, prepare_id),
    )


def regressor_folds(db: sqlite3.Connection, regress_id: int) -> list[dict[str, object]]:
    """List the outer folds of the regressor, the refit (-1) last.

    Args:
        db: A connection to grace.db.
        regress_id: The execution id of regress.

    Returns:
        One row per fold, with its sizes, scores and parameters.
    """
    return _all(
        db,
        """
        SELECT fold, n_regions, spearman, r2, mae, best_score, params
        FROM regressor_folds WHERE execution_id = ? ORDER BY fold = -1, fold
        """,
        (regress_id,),
    )


def figures(db: sqlite3.Connection, run_id: int) -> list[dict[str, object]]:
    """List the figures of every stage execution of a run.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        Rows with `execution_id`, `name` and `file`, ordered by
        execution and name.
    """
    return _all(
        db,
        """
        SELECT f.execution_id, f.name, f.file
        FROM run_stages rs JOIN figures f ON f.execution_id = rs.execution_id
        WHERE rs.run_id = ? ORDER BY f.execution_id, f.name
        """,
        (run_id,),
    )


def figure_file(
    db: sqlite3.Connection, execution_id: int, name: str
) -> tuple[str, str] | None:
    """Find a registered figure.

    Args:
        db: A connection to grace.db.
        execution_id: The execution that saved the figure.
        name: The figure name, e.g. "regress/scores".

    Returns:
        The stage and the file, relative to the execution folder; None
        if the execution registered no such figure.
    """
    row = db.execute(
        """
        SELECT e.stage, f.file FROM figures f
        JOIN executions e ON e.id = f.execution_id
        WHERE f.execution_id = ? AND f.name = ?
        """,
        (execution_id, name),
    ).fetchone()
    return None if row is None else tuple(row)


def regions(
    db: sqlite3.Connection, stages: Mapping[str, int]
) -> tuple[list[str], list[dict[str, object]]]:
    """List the complexity measures and the regions of a run.

    Args:
        db: A connection to grace.db.
        stages: The execution id of each stage of the run.

    Returns:
        The measure names and one row per region, with None where a
        stage did not run.
    """
    rows = _all(
        db,
        """
        SELECT g.region_id, g.class_id, k.name AS class,
               g.n_train, g.n_val, g.n_test,
               f.n_val_error, f.val_rate, f.n_test_error, f.test_rate,
               b.used, e.fold, e.regressor, b.val, b.mcp, b.atc
        FROM regions g
        JOIN classes k ON k.execution_id = :prepare AND k.class_id = g.class_id
        LEFT JOIN failures f
          ON f.execution_id = :failures AND f.region_id = g.region_id
        LEFT JOIN baselines b
          ON b.execution_id = :baselines AND b.region_id = g.region_id
        LEFT JOIN estimates e
          ON e.execution_id = :regress AND e.region_id = g.region_id
        WHERE g.execution_id = :regions ORDER BY g.region_id
        """,
        {
            key: stages.get(key)
            for key in ("prepare", "failures", "baselines", "regress", "regions")
        },
    )
    by_id = {row["region_id"]: row for row in rows}
    measures: list[str] = []
    for region_id, measure, value in db.execute(
        "SELECT scope_id, measure, value FROM complexity "
        "WHERE execution_id = ? AND scope = 'region' ORDER BY rowid",
        (stages.get("complexity"),),
    ):
        if measure not in measures:
            measures.append(measure)
        if region_id in by_id:
            by_id[region_id][measure] = value
    for row in rows:
        scored = (
            row["used"] == 1
            and row["regressor"] is not None
            and row["test_rate"] is not None
        )
        row["abs_error"] = abs(row["regressor"] - row["test_rate"]) if scored else None
    return measures, rows


def descriptors(
    db: sqlite3.Connection, complexity_id: int, region_id: int, class_id: int
) -> list[dict[str, object]]:
    """List the region and class value of every complexity measure.

    Args:
        db: A connection to grace.db.
        complexity_id: The execution id of complexity.
        region_id: The region id.
        class_id: The class of the region.

    Returns:
        One row per measure, with its `region` and `class` values.
    """
    return _all(
        db,
        """
        SELECT r.measure, r.value AS region, c.value AS class
        FROM complexity r LEFT JOIN complexity c
          ON c.execution_id = r.execution_id AND c.scope = 'class'
         AND c.measure = r.measure AND c.scope_id = ?
        WHERE r.execution_id = ? AND r.scope = 'region' AND r.scope_id = ?
        ORDER BY r.rowid
        """,
        (class_id, complexity_id, region_id),
    )


def presets(db: sqlite3.Connection) -> list[dict[str, object]]:
    """List every preset by name, its lists decoded.

    Args:
        db: A connection to grace.db.

    Returns:
        The presets.
    """
    rows = _all(db, "SELECT * FROM presets ORDER BY name")
    return [_decode(row, "targets", "overrides", "sweeps") for row in rows]


def preset(db: sqlite3.Connection, name: str) -> dict[str, object] | None:
    """Return a preset, its lists decoded.

    Args:
        db: A connection to grace.db.
        name: The preset name.

    Returns:
        The preset, or None.
    """
    rows = _all(db, "SELECT * FROM presets WHERE name = ?", (name,))
    return _decode(rows[0], "targets", "overrides", "sweeps") if rows else None


def jobs(db: sqlite3.Connection) -> list[dict[str, object]]:
    """List every job, newest first, its argv decoded.

    Args:
        db: A connection to grace.db.

    Returns:
        The jobs.
    """
    rows = _all(db, "SELECT * FROM jobs ORDER BY id DESC")
    return [_decode(row, "argv") for row in rows]


def job(db: sqlite3.Connection, job_id: int) -> dict[str, object] | None:
    """Return a job, its argv decoded.

    Args:
        db: A connection to grace.db.
        job_id: The job id.

    Returns:
        The job, or None.
    """
    rows = _all(db, "SELECT * FROM jobs WHERE id = ?", (job_id,))
    return _decode(rows[0], "argv") if rows else None


def job_runs(
    db: sqlite3.Connection, started_at: str, finished_at: str | None
) -> list[dict[str, object]]:
    """List the runs started while a job ran.

    A concurrent CLI run shows up too.

    Args:
        db: A connection to grace.db.
        started_at: When the job started.
        finished_at: When it ended; None while it runs.

    Returns:
        The runs, oldest first, with decoded `targets`.
    """
    rows = _all(
        db,
        f"SELECT {_RUN_COLUMNS} FROM runs r WHERE r.started_at >= ? "
        "AND (? IS NULL OR r.started_at <= ?) ORDER BY r.id",
        (started_at, finished_at, finished_at),
    )
    return [_decode(row, "targets") for row in rows]
