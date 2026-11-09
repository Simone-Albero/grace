"""src.db: column types of the dynamic tables."""

import sqlite3

import numpy as np

from src import db


def test_column_types_come_from_the_first_value_that_is_not_none(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    run_id = db.start_run(conn, "s", "d", None, {})
    db.log_table(
        conn,
        run_id,
        "t",
        [
            {"a": None, "b": np.float64(0.5), "c": None, "d": np.int64(3)},
            {"a": 0.25, "b": 1.0, "c": None, "d": 4},
        ],
    )
    types = {row[1]: row[2] for row in conn.execute('PRAGMA table_info("t")')}
    assert types == {
        "run_id": "INTEGER",
        "a": "REAL",
        "b": "REAL",
        "c": "REAL",
        "d": "INTEGER",
    }
    assert conn.execute("SELECT typeof(a) FROM t WHERE a IS NOT NULL").fetchone() == (
        "real",
    )
    conn.close()


def test_runs_record_their_status(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    run_id = db.start_run(conn, "split", "toy", None, {"seed": 1})
    db.finish_run(conn, run_id, "ok")
    assert conn.execute("SELECT stage, status FROM runs").fetchall() == [
        ("split", "ok")
    ]
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
