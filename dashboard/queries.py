"""Read-only queries of the dashboard on grace.db."""

__all__ = [
    "OPERATING_COLUMNS",
    "PARTITION_AXES",
    "PARTITION_METRICS",
    "PREDICTORS",
    "REGION_COLUMNS",
    "SCORES",
    "STAGE_ORDER",
    "class_clusterings",
    "class_metrics",
    "classes",
    "classifier_grid",
    "connect",
    "descriptors",
    "execution_configs",
    "execution_dirs",
    "features",
    "history",
    "importances",
    "metrics_by_stage",
    "operating_points",
    "partition_class_levels",
    "partition_groups",
    "partitions",
    "predictor_stage",
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
    "split prepare regions classify failures baselines complexity regress "
    "render partitions"
).split()
# The failure-rate predictors, each with its label; the `regions`
# columns carry its rates and `<key>_<score>` metrics its scores.
PREDICTORS = {
    "regressor": "RF regressor",
    "val": "Val rate",
    "mcp": "MCP",
    "atc": "ATC",
}
SCORES = ("spearman", "mse", "r2", "mae")
# The numeric columns of `regions` other than the complexity measures;
# they follow the SELECT there and its abs_error.
REGION_COLUMNS = (
    "n_train n_val n_test n_val_error val_rate n_test_error test_rate used fold "
    "regressor val mcp atc abs_error"
).split()
# The partitions columns the panel plots: its metrics and its x axes.
PARTITION_METRICS = ("skill", "brier", "auroc", "kdn_spearman", "share_no_val")
PARTITION_AXES = ("n_groups", "val_median")
# The partitions columns averaged into an operating point.
OPERATING_COLUMNS = (
    "n_groups val_median share_no_val brier skill auroc kdn_spearman"
).split()


def predictor_stage(key: str) -> str:
    """Return the stage whose metrics hold a predictor's scores.

    Args:
        key: A key of `PREDICTORS`.

    Returns:
        "regress" for the regressor, "baselines" for the others.
    """
    return "regress" if key == "regressor" else "baselines"


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


def operating_points(
    db: sqlite3.Connection, run_id: int | None = None
) -> list[dict[str, object]]:
    """List the operating points of the done partitions executions.

    Args:
        db: A connection to grace.db.
        run_id: Only this run's; None lists every run.

    Returns:
        One row per run, arm and rule other than `level`, with the
        partitions columns of `OPERATING_COLUMNS` averaged over the
        draws of `null`.
    """
    averages = ", ".join(f"AVG(p.{name}) AS {name}" for name in OPERATING_COLUMNS)
    return _all(
        db,
        f"""
        SELECT rs.run_id, rs.execution_id, p.arm, p.rule, {averages}
        FROM run_stages rs
        JOIN executions e ON e.id = rs.execution_id AND e.status = 'done'
        JOIN partitions p ON p.execution_id = e.id
        WHERE rs.stage = 'partitions' AND p.rule != 'level'
          AND (? IS NULL OR rs.run_id = ?)
        GROUP BY rs.run_id, p.arm, p.rule ORDER BY rs.run_id, p.arm, p.rule
        """,
        (run_id, run_id),
    )


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
        SELECT fold, n_regions, spearman, r2, mae, best_score, params
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


def regions(
    db: sqlite3.Connection, ids: Mapping[str, int], *, measures: bool = True
) -> tuple[list[str], list[dict[str, object]]]:
    """List the complexity measures and the regions of a run.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each done stage of the run.
        measures: Whether to read the complexity measures too; they are
            most of the rows.

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
            key: ids.get(key)
            for key in ("prepare", "failures", "baselines", "regress", "regions")
        },
    )
    by_id = {row["region_id"]: row for row in rows}
    names: list[str] = []
    if measures and "complexity" in ids:
        for region_id, measure, value in db.execute(
            "SELECT scope_id, measure, value FROM complexity "
            "WHERE execution_id = ? AND scope = 'region' ORDER BY rowid",
            (ids["complexity"],),
        ):
            if measure not in names:
                names.append(measure)
            if region_id in by_id:
                by_id[region_id][measure] = value
    for row in rows:
        scored = (
            row["used"] == 1
            and row["regressor"] is not None
            and row["test_rate"] is not None
        )
        row["abs_error"] = abs(row["regressor"] - row["test_rate"]) if scored else None
    return names, rows


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


def partitions(db: sqlite3.Connection, partitions_id: int) -> list[dict[str, object]]:
    """List the rows of a partitions execution.

    Args:
        db: A connection to grace.db.
        partitions_id: The execution id of partitions.

    Returns:
        One row per arm, rule, level and draw, in that order.
    """
    return _all(
        db,
        "SELECT * FROM partitions WHERE execution_id = ? "
        "ORDER BY arm, rule, level, draw",
        (partitions_id,),
    )


def partition_class_levels(
    db: sqlite3.Connection, partitions_id: int
) -> list[dict[str, object]]:
    """List the intra arm's per-class loss curve.

    Args:
        db: A connection to grace.db.
        partitions_id: The execution id of partitions.

    Returns:
        The rows, ordered by class and level.
    """
    return _all(
        db,
        "SELECT * FROM partition_class_levels WHERE execution_id = ? "
        "ORDER BY class_id, level",
        (partitions_id,),
    )


def partition_groups(
    db: sqlite3.Connection, partitions_id: int
) -> list[dict[str, object]]:
    """List the groups of a partitions execution's operating points.

    Args:
        db: A connection to grace.db.
        partitions_id: The execution id of partitions.

    Returns:
        The rows, ordered by arm, rule and group.
    """
    return _all(
        db,
        "SELECT * FROM partition_groups WHERE execution_id = ? "
        "ORDER BY arm, rule, group_id",
        (partitions_id,),
    )
