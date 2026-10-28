"""Read-only queries of the dashboard on grace.db."""

__all__ = [
    "PREDICTORS",
    "SCORES",
    "STAGE_ORDER",
    "class_metrics",
    "class_tree",
    "classes",
    "classifier_grid",
    "connect",
    "execution_configs",
    "execution_dirs",
    "features",
    "history",
    "leaf_descriptors",
    "leaf_path",
    "leaves",
    "metrics_by_stage",
    "run",
    "run_metrics",
    "runs",
    "stage_ids",
    "tree_roots",
]

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

# The stages in execution order, as `pipeline.runner.STAGES` lists them;
# a test keeps the two equal, since importing the runner here would load
# every stage's dependencies.
STAGE_ORDER = (
    "split prepare space classify crossfit tree complexity baselines evaluate render"
).split()
# The failure predictors, each with its label; its scores are the
# `evaluate` metrics `<key>_<score>`.
PREDICTORS = {
    "tree": "Tree",
    "tree_leaf": "Tree leaf",
    "val_rate": "Val rate",
    "confidence": "Confidence",
    "binned": "Binned",
    "gbm": "GBM",
}
SCORES = ("mse", "mse_ratio", "spearman")


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


def class_metrics(
    db: sqlite3.Connection, prepare_id: int, classify_id: int
) -> list[dict[str, object]]:
    """List the split sizes and classifier scores of every class.

    Args:
        db: A connection to grace.db.
        prepare_id: The execution id of prepare.
        classify_id: The execution id of classify.

    Returns:
        One row per class: id, name, `n_train`, `n_test`, precision,
        recall, f1.
    """
    return _all(
        db,
        """
        SELECT k.class_id, k.name, k.n_train, k.n_test, m.precision, m.recall, m.f1
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


def tree_roots(
    db: sqlite3.Connection, tree_id: int, prepare_id: int
) -> list[dict[str, object]]:
    """List the root of every predicted class of a tree execution.

    Args:
        db: A connection to grace.db.
        tree_id: The execution id of tree.
        prepare_id: The execution id of prepare, for the class names.

    Returns:
        One row per class: id, name, the train rows and out-of-fold
        errors at its root, its leaves and the depth of its cut.
    """
    return _all(
        db,
        """
        SELECT k.class_id, k.name,
               SUM(CASE WHEN t.parent_id IS NULL THEN t.n_train END) AS n_train,
               SUM(CASE WHEN t.parent_id IS NULL THEN t.n_train_errors END)
                 AS n_train_errors,
               SUM(t.status = 'leaf') AS n_leaves,
               MAX(CASE WHEN t.status != 'pruned' THEN t.depth END) AS depth
        FROM tree_nodes t
        JOIN classes k ON k.execution_id = :prepare AND k.class_id = t.class_id
        WHERE t.execution_id = :tree
        GROUP BY k.class_id ORDER BY k.class_id
        """,
        {"prepare": prepare_id, "tree": tree_id},
    )


def leaves(db: sqlite3.Connection, ids: Mapping[str, int]) -> list[dict[str, object]]:
    """List the leaves of a run with their rates and estimates.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each done stage of the run.

    Returns:
        One row per leaf, with None where a stage did not run or a
        count is 0: the out-of-fold, val and test rates, the tree's
        rate `q` and the mean corrected probability of the leaf's test
        rows.
    """
    rows = _all(
        db,
        """
        SELECT t.node_id, t.class_id, k.name AS class, t.depth,
               t.n_train, t.n_train_errors, t.n_val, t.n_val_errors, t.q, t.gamma,
               l.n_test, l.n_test_errors, l.mean_q
        FROM tree_nodes t
        JOIN classes k ON k.execution_id = :prepare AND k.class_id = t.class_id
        LEFT JOIN leaf_tests l
          ON l.execution_id = :evaluate AND l.node_id = t.node_id
        WHERE t.execution_id = :tree AND t.status = 'leaf'
        ORDER BY t.node_id
        """,
        {key: ids.get(key) for key in ("prepare", "tree", "evaluate")},
    )
    _add_rates(rows)
    return rows


def _add_rates(rows: Sequence[dict[str, object]]) -> None:
    """Add the out-of-fold, val and test rates, None for 0 rows."""
    for row in rows:
        for rate, errors, count in (
            ("oof_rate", "n_train_errors", "n_train"),
            ("val_rate", "n_val_errors", "n_val"),
            ("test_rate", "n_test_errors", "n_test"),
        ):
            row[rate] = row[errors] / row[count] if row[count] else None


def class_tree(
    db: sqlite3.Connection, ids: Mapping[str, int], class_id: int
) -> list[dict[str, object]]:
    """List the kept nodes of a predicted class's tree, with rates.

    Args:
        db: A connection to grace.db.
        ids: The execution id of each done stage of the run.
        class_id: The predicted class of the root.

    Returns:
        One row per node that was not pruned, parents first, with the
        `purity` of its train rows: the share of the class's own. A node's
        test counts are the sums over the leaves below it, and None
        when evaluate did not run.
    """
    rows = _all(
        db,
        """
        SELECT t.node_id, t.parent_id, k.name AS class, t.depth, t.status,
               t.n_train, t.n_train_errors, t.n_val, t.n_val_errors, t.q,
               l.n_test, l.n_test_errors
        FROM tree_nodes t
        JOIN classes k ON k.execution_id = :prepare AND k.class_id = t.class_id
        LEFT JOIN leaf_tests l
          ON l.execution_id = :evaluate AND l.node_id = t.node_id
        WHERE t.execution_id = :tree AND t.class_id = :class_id
          AND t.status != 'pruned'
        ORDER BY t.depth, t.node_id
        """,
        {
            "prepare": ids["prepare"],
            "tree": ids["tree"],
            "evaluate": ids.get("evaluate"),
            "class_id": class_id,
        },
    )
    by_id = {row["node_id"]: row for row in rows}
    # Deepest first, so a node is complete before it feeds its parent.
    for row in reversed(rows):
        parent = by_id.get(row["parent_id"])
        if parent is None or row["n_test"] is None:
            continue
        for key in ("n_test", "n_test_errors"):
            parent[key] = (parent[key] or 0) + row[key]
    _add_rates(rows)
    # Every train row of the tree has the class as its prediction, so the
    # out-of-fold errors are the rows of another class.
    for row in rows:
        row["purity"] = None if row["oof_rate"] is None else 1 - row["oof_rate"]
    return rows


def leaf_path(
    db: sqlite3.Connection, tree_id: int, node_id: int
) -> list[dict[str, object]]:
    """List a leaf and its ancestors, root first.

    Args:
        db: A connection to grace.db.
        tree_id: The execution id of tree.
        node_id: The leaf.

    Returns:
        The nodes from the root of its predicted class down to the leaf.
    """
    return _all(
        db,
        """
        WITH RECURSIVE path(node_id, parent_id) AS (
          SELECT node_id, parent_id FROM tree_nodes
          WHERE execution_id = :tree AND node_id = :node
          UNION ALL
          SELECT t.node_id, t.parent_id FROM tree_nodes t
          JOIN path p ON t.node_id = p.parent_id WHERE t.execution_id = :tree
        )
        SELECT t.node_id, t.depth, t.status, t.n_train, t.n_train_errors,
               t.n_val, t.n_val_errors
        FROM tree_nodes t JOIN path USING (node_id)
        WHERE t.execution_id = :tree ORDER BY t.depth
        """,
        {"tree": tree_id, "node": node_id},
    )


def leaf_descriptors(
    db: sqlite3.Connection, complexity_id: int, node_id: int, class_id: int
) -> list[dict[str, object]]:
    """List the leaf and class value of every complexity measure.

    Args:
        db: A connection to grace.db.
        complexity_id: The execution id of complexity.
        node_id: The leaf.
        class_id: The class its root predicts.

    Returns:
        One row per measure, with its `leaf` and `class` values.
    """
    return _all(
        db,
        """
        SELECT r.measure, r.value AS leaf, c.value AS class
        FROM complexity r LEFT JOIN complexity c
          ON c.execution_id = r.execution_id AND c.scope = 'class'
         AND c.measure = r.measure AND c.scope_id = ?
        WHERE r.execution_id = ? AND r.scope = 'leaf' AND r.scope_id = ?
        ORDER BY r.rowid
        """,
        (class_id, complexity_id, node_id),
    )
