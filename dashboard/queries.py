"""Read-only queries of the dashboard on grace.db."""

__all__ = [
    "PREDICTORS",
    "SCORES",
    "STAGE_ORDER",
    "class_clusterings",
    "class_metrics",
    "classes",
    "classifier_grid",
    "connect",
    "execution_configs",
    "execution_dirs",
    "features",
    "history",
    "importances",
    "metrics_by_stage",
    "regions",
    "regressor_folds",
    "run",
    "run_metrics",
    "runs",
    "stage_ids",
]

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

# The stages in execution order, as `pipeline.runner.STAGES` lists them;
# a test keeps the two equal, since importing the runner here would load
# every stage's dependencies.
STAGE_ORDER = (
    "split prepare space regions classify crossfit tree failures baselines "
    "complexity regress render"
).split()
# The failure-rate predictors, each with its label; the `regions`
# columns carry its rates and `<key>_<score>` metrics its scores.
PREDICTORS = {
    "val": "Val rate",
    "mcp": "MCP",
    "atc": "ATC",
}
SCORES = ("spearman", "mse", "r2", "mae")


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


def _decode(row: dict[str, object], *keys: str) -> dict[str, object]:
    """Return `row` with the JSON text of `keys` decoded."""
    return {**row, **{key: json.loads(row[key]) for key in keys}}


def _stages_by_run(
    db: sqlite3.Connection, run_id: int | None = None
) -> dict[int, list[dict[str, object]]]:
    """Group the stage executions of the runs by run, upstream first."""
    # An execution id is larger than those of its inputs: a valid DAG
    # order.
    rows = _all(
        db,
        """
        SELECT rs.run_id, rs.stage, rs.execution_id, rs.cached, e.status,
               e.started_at, e.duration_s, e.error, e.dir
        FROM run_stages rs JOIN executions e ON e.id = rs.execution_id
        WHERE ? IS NULL OR rs.run_id = ? ORDER BY e.id
        """,
        (run_id, run_id),
    )
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row.pop("run_id"), []).append(row)
    return grouped


def runs(db: sqlite3.Connection) -> list[dict[str, object]]:
    """List every run, newest first, with its stage executions.

    Args:
        db: A connection to grace.db.

    Returns:
        The runs, with decoded `targets` and `config` and a `stages`
        list of the executions the run used, whatever their status.
    """
    stages = _stages_by_run(db)
    rows = _all(
        db,
        "SELECT id, targets, config, status, error, started_at, finished_at "
        "FROM runs ORDER BY id DESC",
    )
    return [
        {**_decode(row, "targets", "config"), "stages": stages.get(row["id"], [])}
        for row in rows
    ]


def run(db: sqlite3.Connection, run_id: int) -> dict[str, object] | None:
    """Return a run with its parsed config and stage executions.

    Args:
        db: A connection to grace.db.
        run_id: The run id.

    Returns:
        The run as `runs` lists it, or None.
    """
    rows = _all(
        db,
        "SELECT id, targets, config, status, error, started_at, finished_at "
        "FROM runs WHERE id = ?",
        (run_id,),
    )
    if not rows:
        return None
    stages = _stages_by_run(db, run_id).get(run_id, [])
    return {**_decode(rows[0], "targets", "config"), "stages": stages}


def stage_ids(run_row: Mapping[str, object]) -> dict[str, int]:
    """Map every done stage of a run to its execution id.

    Args:
        run_row: A run as `run` returns it.

    Returns:
        The execution id by stage name; a failed or running execution
        has no results to show and is left out.
    """
    return {
        row["stage"]: row["execution_id"]
        for row in run_row["stages"]
        if row["status"] == "done"
    }


def execution_dirs(run_row: Mapping[str, object]) -> dict[str, str]:
    """Map every done stage of a run to its folder under the root."""
    return {
        row["stage"]: row["dir"] for row in run_row["stages"] if row["status"] == "done"
    }


def run_metrics(db: sqlite3.Connection) -> list[tuple[int, str, int, str, float]]:
    """List the metrics of every done execution each run used.

    Args:
        db: A connection to grace.db.

    Returns:
        `(run_id, stage, execution_id, name, value)` tuples; `value` is
        None for a metric that was NaN.
    """
    rows = db.execute("""
        SELECT rs.run_id, e.stage, e.id, m.name, m.value
        FROM run_stages rs
        JOIN executions e ON e.id = rs.execution_id AND e.status = 'done'
        JOIN metrics m ON m.execution_id = e.id
        """)
    return [tuple(row) for row in rows]


def metrics_by_stage(
    db: sqlite3.Connection, ids: Mapping[str, int]
) -> dict[str, dict[str, float | None]]:
    """Read the metrics of some executions.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each stage.

    Returns:
        The metric values by stage and name, in insertion order.
    """
    return {
        stage: dict(
            db.execute(
                "SELECT name, value FROM metrics WHERE execution_id = ? "
                "ORDER BY rowid",
                (execution_id,),
            )
        )
        for stage, execution_id in ids.items()
    }


def execution_configs(
    db: sqlite3.Connection, ids: Mapping[str, int]
) -> dict[str, dict[str, object]]:
    """Read the config slice each of some executions ran with.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each stage.

    Returns:
        The decoded config by stage.
    """
    configs = {}
    for stage, execution_id in ids.items():
        row = db.execute(
            "SELECT config FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        configs[stage] = json.loads(row[0])
    return configs


def features(db: sqlite3.Connection, split_id: int) -> list[dict[str, object]]:
    """List the features of a split execution, in column order."""
    return _all(
        db,
        "SELECT position, name, kind FROM features WHERE execution_id = ? "
        "ORDER BY position",
        (split_id,),
    )


def classes(db: sqlite3.Connection, prepare_id: int) -> list[dict[str, object]]:
    """List the classes of a prepare execution, rows per split."""
    return _all(
        db,
        "SELECT class_id, name, n_train, n_val, n_test FROM classes "
        "WHERE execution_id = ? ORDER BY class_id",
        (prepare_id,),
    )


def class_clusterings(
    db: sqlite3.Connection, regions_id: int, prepare_id: int
) -> list[dict[str, object]]:
    """List the clustering of every class of a regions execution.

    Args:
        db: A connection to grace.db.
        regions_id: The execution id of regions.
        prepare_id: The execution id of prepare, for the class names.

    Returns:
        One row per class: id, name, its clustering columns and the
        number of regions it holds.
    """
    return _all(
        db,
        """
        SELECT c.class_id, k.name, c.n_train, c.n_clusters, c.loss,
               c.size_balance, c.n_merged_clusters, c.n_merged
        FROM class_clusterings c
        JOIN classes k ON k.execution_id = ? AND k.class_id = c.class_id
        WHERE c.execution_id = ? ORDER BY c.class_id
        """,
        (prepare_id, regions_id),
    )


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


def classifier_grid(
    db: sqlite3.Connection, classify_id: int
) -> list[dict[str, object]]:
    """List the grid-search candidates of a classify execution.

    Returns:
        The candidates, empty without a search, with decoded `params`.
    """
    rows = _all(
        db,
        "SELECT candidate, params, mean_score, std_score, best "
        "FROM classifier_grid WHERE execution_id = ? ORDER BY candidate",
        (classify_id,),
    )
    return [_decode(row, "params") for row in rows]


def history(db: sqlite3.Connection, classify_id: int) -> list[dict[str, object]]:
    """List the training loss of a DL classify execution by step."""
    return _all(
        db,
        "SELECT step, loss FROM history WHERE execution_id = ? ORDER BY step",
        (classify_id,),
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
        SELECT fold, n_leaves, best_score, params
        FROM regressor_folds WHERE execution_id = ? ORDER BY fold = -1, fold
        """,
        (regress_id,),
    )


def importances(db: sqlite3.Connection, regress_id: int) -> list[dict[str, object]]:
    """List the regressor's mean feature importances, largest first."""
    return _all(
        db,
        "SELECT feature, importance FROM importances WHERE execution_id = ? "
        "ORDER BY importance DESC",
        (regress_id,),
    )


def regions(db: sqlite3.Connection, ids: Mapping[str, int]) -> list[dict[str, object]]:
    """List the regions of a run with their rates and predictions.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each done stage of the run.

    Returns:
        One row per region, with None where a stage did not run.
    """
    return _all(
        db,
        """
        SELECT g.region_id, g.class_id, k.name AS class,
               g.n_train, g.n_val, g.n_test,
               f.n_val_error, f.val_rate, f.n_test_error, f.test_rate,
               b.used, b.val, b.mcp, b.atc
        FROM regions g
        JOIN classes k ON k.execution_id = :prepare AND k.class_id = g.class_id
        LEFT JOIN failures f
          ON f.execution_id = :failures AND f.region_id = g.region_id
        LEFT JOIN baselines b
          ON b.execution_id = :baselines AND b.region_id = g.region_id
        WHERE g.execution_id = :regions ORDER BY g.region_id
        """,
        {key: ids.get(key) for key in ("prepare", "failures", "baselines", "regions")},
    )
