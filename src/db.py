"""SQLite logging: runs and result tables keyed by run_id."""

from __future__ import annotations

__all__ = ["connect", "finish_run", "log_table", "start_run"]

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, stage TEXT, dataset TEXT, model TEXT, config TEXT,
    status TEXT, started TEXT, finished TEXT);
"""


def _sql_type(values: Sequence[Any]) -> str:
    """Return the SQL type of a column from its first non-None value.

    A column of None only is REAL: the undefined measures are its case.
    """
    first = next((v for v in values if v is not None), 0.0)
    if isinstance(first, (bool, int, np.integer)):
        return "INTEGER"
    if isinstance(first, (float, np.floating)):
        return "REAL"
    return "TEXT"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the db, creating its folder and the runs table when missing.

    Args:
        db_path: The SQLite file.

    Returns:
        The open connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def start_run(
    conn: sqlite3.Connection,
    stage: str,
    dataset: str,
    model: str | None,
    config: Mapping[str, Any],
) -> int:
    """Log a new run with status "running".

    Args:
        conn: The open db.
        stage: The stage name.
        dataset: The dataset name.
        model: The model name, None for the stages without a model.
        config: The whole config of the run, stored as JSON.

    Returns:
        The id of the new run.
    """
    with conn:
        cursor = conn.execute(
            "INSERT INTO runs (stage, dataset, model, config, status, started) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (stage, dataset, model, json.dumps(config, default=str), "running", _now()),
        )
    return cursor.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    """Set the final status of a run: "ok" or "failed".

    Args:
        conn: The open db.
        run_id: The run.
        status: Its final status.
    """
    with conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished = ? WHERE id = ?",
            (status, _now(), run_id),
        )


def log_table(
    conn: sqlite3.Connection,
    run_id: int,
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Insert rows into a table, adding a run_id column.

    Creates the table when missing, with columns and types taken from
    the first row (int and bool -> INTEGER, float -> REAL, else TEXT);
    columns missing from an existing table are added, NULL on the older
    rows.

    Args:
        conn: The open db.
        run_id: The run the rows belong to.
        table: The table name, a valid identifier.
        rows: The rows, all with the keys of the first one; none is a
            no-op.

    Raises:
        ValueError: When a table or column name is not a valid
            identifier, or a column is called run_id.
    """
    if not rows:
        return
    columns = list(rows[0])
    for name in (table, *columns):
        if name == "run_id" or not _IDENT.fullmatch(name):
            raise ValueError(f"invalid table or column name: {name!r}")
    types = [_sql_type([row[column] for row in rows]) for column in columns]
    definitions = ", ".join(
        f'"{column}" {sql_type}' for column, sql_type in zip(columns, types)
    )
    names = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" * (len(columns) + 1))
    with conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            f"(run_id INTEGER REFERENCES runs(id), {definitions})"
        )
        existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        for column, sql_type in zip(columns, types):
            if column not in existing:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {sql_type}')
        conn.executemany(
            f'INSERT INTO "{table}" (run_id, {names}) VALUES ({placeholders})',
            [(run_id, *(row[column] for column in columns)) for row in rows],
        )
