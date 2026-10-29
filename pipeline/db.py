"""The SQLite schema of grace.db and its connection."""

__all__ = [
    "DB_NAME",
    "KILLED",
    "RUNS_LOCK",
    "SCHEMA",
    "connect",
    "explain",
    "now",
    "reap",
]

import fcntl
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DB_NAME = "grace.db"
# Held shared, in the root, by every live run.
RUNS_LOCK = "runs.lock"
KILLED = "its process ended before it finished (killed?)"
_CORRUPTED = (
    "it looks corrupted; restore a copy, or move it away together with "
    "artifacts/ to start a new one"
)
# What to do about an sqlite3 error, by a word of its message. An error
# matching none gets no advice: only these name a certain remedy.
_REMEDIES = {
    "locked": "another process holds it; retry when it ends",
    "readonly": "the root is read-only; make it writable",
    "full": "the disk is full; free some space",
    "not a database": _CORRUPTED,
    "malformed": _CORRUPTED,
}

# Result tables are keyed by execution_id and a natural key.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    targets TEXT NOT NULL,
    config TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    key TEXT NOT NULL,
    config TEXT NOT NULL,
    inputs TEXT NOT NULL,
    dir TEXT NOT NULL,  -- relative to the root: artifacts/<stage>/<id>
    status TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    error TEXT,
    started_at TEXT NOT NULL,
    duration_s REAL
);
CREATE INDEX IF NOT EXISTS executions_cache ON executions (stage, key, status);
CREATE TABLE IF NOT EXISTS run_stages (
    run_id INTEGER NOT NULL REFERENCES runs (id),
    stage TEXT NOT NULL,
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    cached INTEGER NOT NULL CHECK (cached IN (0, 1)),
    PRIMARY KEY (run_id, stage)
);
CREATE TABLE IF NOT EXISTS metrics (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    name TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (execution_id, name)
);
CREATE TABLE IF NOT EXISTS features (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('num', 'cat')),
    PRIMARY KEY (execution_id, position)
);
CREATE TABLE IF NOT EXISTS classes (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    class_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    n_train INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    PRIMARY KEY (execution_id, class_id)
);
CREATE TABLE IF NOT EXISTS complexity (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    scope TEXT NOT NULL CHECK (scope IN ('leaf', 'class')),
    scope_id INTEGER NOT NULL,  -- a tree node_id or a class_id
    measure TEXT NOT NULL,
    value REAL,  -- NULL: undefined measure
    PRIMARY KEY (execution_id, scope, scope_id, measure)
);
CREATE TABLE IF NOT EXISTS class_metrics (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    class_id INTEGER NOT NULL,
    precision REAL,
    recall REAL,
    f1 REAL,
    PRIMARY KEY (execution_id, class_id)
);
CREATE TABLE IF NOT EXISTS classifier_grid (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    candidate INTEGER NOT NULL,
    params TEXT NOT NULL,  -- JSON
    mean_score REAL,
    std_score REAL,
    best INTEGER NOT NULL CHECK (best IN (0, 1)),
    PRIMARY KEY (execution_id, candidate)
);
CREATE TABLE IF NOT EXISTS history (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    step INTEGER NOT NULL,
    loss REAL NOT NULL,
    PRIMARY KEY (execution_id, step)
);
CREATE TABLE IF NOT EXISTS tree_nodes (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    node_id INTEGER NOT NULL,
    parent_id INTEGER,  -- NULL at a root, the root of a predicted class
    class_id INTEGER NOT NULL,  -- the predicted class of the root above it
    depth INTEGER NOT NULL,
    n_grow INTEGER NOT NULL,  -- grown rows (val's first half) at or below it
    n_grow_errors INTEGER NOT NULL,  -- the classifier's errors among them
    cost REAL NOT NULL,  -- its expected Brier cost as a leaf
    best REAL NOT NULL,  -- the least cost of its subtree
    status TEXT NOT NULL CHECK (status IN ('internal', 'leaf', 'pruned')),
    n_estimate INTEGER,  -- the same for val's other half; NULL when pruned
    n_estimate_errors INTEGER,  -- NULL when pruned
    q REAL,  -- the leaf's failure rate; NULL unless a leaf
    gamma REAL,  -- the weight of the leaf's own estimate rate; NULL without rows
    PRIMARY KEY (execution_id, node_id)
);
CREATE TABLE IF NOT EXISTS leaf_tests (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    node_id INTEGER NOT NULL,  -- a leaf of the tree
    n_test INTEGER NOT NULL,  -- test rows routed to it
    n_test_errors INTEGER NOT NULL,
    mean_q REAL,  -- the mean probability the tree gave them; NULL without rows
    test_rate REAL,  -- NULL without test rows
    PRIMARY KEY (execution_id, node_id)
);
CREATE TABLE IF NOT EXISTS figures (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    name TEXT NOT NULL,  -- e.g. predictors/mse_ratio
    file TEXT NOT NULL,  -- relative to the execution dir
    PRIMARY KEY (execution_id, name)
);
"""


def now() -> str:
    """Return the current UTC time as ISO-8601 text.

    Returns:
        The time, to the second.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    """Open the DB at `path`, creating it with the schema if new.

    Args:
        path: The grace.db file; its folder is created if missing.

    Returns:
        The connection, with foreign keys on.

    Raises:
        sqlite3.Error: The DB cannot be opened or created; `explain`
            says what to do about it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA foreign_keys = ON")
    # The dashboard reads while a run writes.
    db.execute("PRAGMA journal_mode = WAL")
    db.executescript(SCHEMA)
    return db


def explain(error: sqlite3.Error, path: Path) -> str:
    """Return a message naming the DB, the error and the fix if known.

    Args:
        error: The sqlite3 error.
        path: The grace.db it concerns.

    Returns:
        The message.
    """
    remedy = next((text for word, text in _REMEDIES.items() if word in str(error)), "")
    advice = f" ({remedy})." if remedy else "."
    return f"Cannot use the database {path}: {error}{advice}"


def reap(root: Path) -> None:
    """Fail the runs and executions that no live process is running.

    A live run holds a shared lock on `<root>/runs.lock`, so taking it
    exclusively proves that every row still `running` was left by a
    killed process.

    Args:
        root: The experiment root, with its grace.db.
    """
    with (root / RUNS_LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        db = connect(root / DB_NAME)
        try:
            with db:
                db.execute(
                    "UPDATE executions SET status = 'failed', error = ? "
                    "WHERE status = 'running'",
                    (KILLED,),
                )
                db.execute(
                    "UPDATE runs SET status = 'failed', error = ?, finished_at = ? "
                    "WHERE status = 'running'",
                    (KILLED, now()),
                )
        finally:
            db.close()
