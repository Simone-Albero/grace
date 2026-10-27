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
CREATE TABLE IF NOT EXISTS regions (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    region_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    n_train INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    PRIMARY KEY (execution_id, region_id)
);
CREATE TABLE IF NOT EXISTS class_clusterings (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    class_id INTEGER NOT NULL,
    n_train INTEGER NOT NULL,
    n_clusters INTEGER NOT NULL,
    loss REAL,
    size_balance REAL,
    n_merged_clusters INTEGER NOT NULL,
    n_merged INTEGER NOT NULL,
    PRIMARY KEY (execution_id, class_id)
);
CREATE TABLE IF NOT EXISTS complexity (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    scope TEXT NOT NULL CHECK (scope IN ('region', 'class')),
    scope_id INTEGER NOT NULL,  -- region_id or class_id
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
CREATE TABLE IF NOT EXISTS failures (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    region_id INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    n_val_error INTEGER NOT NULL,
    val_rate REAL,  -- NULL when n_val = 0
    n_test INTEGER NOT NULL,
    n_test_error INTEGER NOT NULL,
    test_rate REAL,  -- NULL when n_test = 0
    PRIMARY KEY (execution_id, region_id)
);
CREATE TABLE IF NOT EXISTS baselines (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    region_id INTEGER NOT NULL,
    used INTEGER NOT NULL CHECK (used IN (0, 1)),  -- scored: enough test rows
    val REAL,  -- the region's own val rate; the global val rate without one
    mcp REAL,  -- mean 1 - max probability of its test rows; NULL without any
    atc REAL,  -- share of its test rows under the ATC threshold; NULL without any
    PRIMARY KEY (execution_id, region_id)
);
CREATE TABLE IF NOT EXISTS estimates (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    region_id INTEGER NOT NULL,
    fold INTEGER,  -- outer fold that held it out; NULL for the refit or unpredicted
    regressor REAL,  -- predicted rate; NULL without a val row and unused
    PRIMARY KEY (execution_id, region_id)
);
CREATE TABLE IF NOT EXISTS regressor_folds (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    fold INTEGER NOT NULL,  -- -1: the refit on every val-labelled region
    n_regions INTEGER NOT NULL,  -- held out (fold) or trained on (refit)
    spearman REAL,  -- scored on its held-out, used regions; NULL for the refit
    r2 REAL,
    mae REAL,
    best_score REAL,  -- inner-CV clipped R²; NULL without a search
    params TEXT NOT NULL,  -- JSON of the chosen forest parameters
    PRIMARY KEY (execution_id, fold)
);
CREATE TABLE IF NOT EXISTS importances (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    feature TEXT NOT NULL,
    importance REAL NOT NULL,  -- mean over the outer folds
    PRIMARY KEY (execution_id, feature)
);
CREATE TABLE IF NOT EXISTS figures (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    name TEXT NOT NULL,  -- e.g. regress/predicted_vs_observed
    file TEXT NOT NULL,  -- relative to the execution dir
    PRIMARY KEY (execution_id, name)
);
CREATE TABLE IF NOT EXISTS partitions (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    arm TEXT NOT NULL
        CHECK (arm IN ('intra', 'inter', 'null', 'confidence', 'predicted_class')),
    rule TEXT NOT NULL
        CHECK (rule IN ('level', 'selected', 'fixed15', 'cbrt', 'per150')),
    level INTEGER NOT NULL,  -- -1: the rule has no level (selected or a bin rule)
    draw INTEGER NOT NULL,  -- the null arm's draw; 0 elsewhere
    n_target INTEGER NOT NULL,  -- groups asked for, before merging or binning ties
    n_groups INTEGER NOT NULL,  -- groups realized; at most n_target
    loss REAL,  -- NULL for a bin arm, which has no geometric loss
    routed_loss REAL,  -- NULL for a bin arm
    val_p10 REAL NOT NULL,
    val_median REAL NOT NULL,
    val_p90 REAL NOT NULL,
    share_no_val REAL NOT NULL,  -- test rows in a group with no val row
    brier REAL NOT NULL,
    skill REAL,  -- NULL when the reference model already scores zero Brier
    uncertainty REAL NOT NULL,
    resolution REAL NOT NULL,
    reliability REAL NOT NULL,
    auroc REAL,  -- NULL when every scored test row failed, or none did
    kdn_spearman REAL,  -- always NULL for a bin arm; a geometric arm can
        -- also be NULL, with too few groups holding a test row or a
        -- constant mean kDN or test rate among them
    PRIMARY KEY (execution_id, arm, rule, level, draw)
);
CREATE TABLE IF NOT EXISTS partition_class_levels (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    class_id INTEGER NOT NULL,
    level INTEGER NOT NULL,
    n_clusters INTEGER NOT NULL,
    loss REAL NOT NULL,
    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
    PRIMARY KEY (execution_id, class_id, level)
);
CREATE TABLE IF NOT EXISTS partition_groups (
    execution_id INTEGER NOT NULL REFERENCES executions (id),
    arm TEXT NOT NULL
        CHECK (arm IN ('intra', 'inter', 'null', 'confidence', 'predicted_class')),
    rule TEXT NOT NULL
        CHECK (rule IN ('selected', 'fixed15', 'cbrt', 'per150')),  -- no 'level' row
    group_id INTEGER NOT NULL,
    n_train INTEGER NOT NULL,  -- 0 for a bin arm, which routes no train row
    n_val INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    val_rate REAL,  -- NULL when n_val = 0
    test_rate REAL,  -- NULL when n_test = 0
    mean_kdn REAL,  -- NULL for a bin arm
    PRIMARY KEY (execution_id, arm, rule, group_id)
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
