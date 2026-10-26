"""Shared test helpers: the fast synthetic config and DB queries."""

__all__ = ["BASE_URL", "fast_cfg", "query"]

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from omegaconf import DictConfig

from pipeline.config import load_config

# The dashboard answers only to its own Host.
BASE_URL = "http://127.0.0.1:8000"
# Same keys in every config: the shared fixture's runs stay cached.
_FAST = (
    "failure_regressor.n_iter=2",
    "failure_regressor.n_outer_splits=2",
    "failure_regressor.n_inner_splits=2",
    "failure_regressor.param_grid.n_estimators=[20,50]",
    "partitions.null_draws=3",
)


def fast_cfg(root: Path, *overrides: str) -> DictConfig:
    """Return the fast synthetic config of `root`.

    Args:
        root: The experiment root.
        *overrides: `key=value` items applied on top of the fast keys.

    Returns:
        The validated config.
    """
    return load_config(
        [
            "data=synthetic_test",
            "classifier=naive_bayes",
            f"root={root}",
            *_FAST,
            *overrides,
        ]
    )


def query(db_path: Path, sql: str, params: Sequence[object] = ()) -> list[tuple]:
    """Run `sql` on the DB at `db_path`, opened and closed here.

    Args:
        db_path: The grace.db file.
        sql: The statement.
        params: The values of its placeholders.

    Returns:
        The rows.
    """
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()
