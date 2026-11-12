"""The files of every stage, shared with the dashboard.

Every rerun of a stage overwrites its files, so the files always belong
to the latest run of that stage.
"""

from __future__ import annotations

__all__ = [
    "assignments_file",
    "classes_file",
    "data_file",
    "need",
    "prepared_columns",
    "prepared_file",
    "read_classes",
    "read_data",
    "read_points",
    "read_unparsable",
    "read_weights",
    "regions_file",
    "unparsable_file",
]

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from src.config import DatasetConfig
from src.data import DUPLICATES, LABEL, feature_cols
from src.graph import space_points
from src.prepare import MISSING


def need(path: Path, stage: str) -> Path:
    """Return path, or fail with the stage that writes it.

    Args:
        path: A file some stage writes.
        stage: That stage.

    Returns:
        path.

    Raises:
        FileNotFoundError: "run <stage> first, missing: <path>" when it
            does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"run {stage} first, missing: {path}")
    return path


def data_file(cfg: DatasetConfig) -> Path:
    """Return the file of the cleaned rows (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "data.parquet"


def unparsable_file(cfg: DatasetConfig) -> Path:
    """Return the file of the numeric cells that were not numbers (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "unparsable.parquet"


def classes_file(cfg: DatasetConfig) -> Path:
    """Return the file of the class names (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The JSON path; the file is a list of names indexed by code.
    """
    return cfg.output_dir / "classes.json"


def prepared_file(cfg: DatasetConfig) -> Path:
    """Return the file of the prepared rows (stage prepare).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "prepared.parquet"


def regions_file(cfg: DatasetConfig) -> Path:
    """Return the file of one row per region (stage regions).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "regions.parquet"


def assignments_file(cfg: DatasetConfig) -> Path:
    """Return the file of the region of every row (stage regions).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path; it covers every row of the data.
    """
    return cfg.output_dir / "assignments.parquet"


def read_data(cfg: DatasetConfig) -> pl.DataFrame:
    """Return the cleaned rows.

    Args:
        cfg: The dataset config.

    Returns:
        The rows, as stage clean wrote them.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    return pl.read_parquet(need(data_file(cfg), "clean"))


def read_points(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the prepared rows as points of the space, and their classes.

    The numeric columns of the prepared file (data.feature_cols) and its
    one-hot slots (every other column but MISSING and LABEL) go through
    graph.space_points with method.distance. MISSING is a diagnostic:
    no distance sees it.

    Args:
        cfg: The dataset config.

    Returns:
        The Float64 points, shape (n, d_num + d_cat), and the class
        codes, shape (n,). Row i is row i of the data file. Column j is
        a column of prepared_columns(cfg)[j].

    Raises:
        FileNotFoundError: When stage prepare has not run.
    """
    prepared = pl.read_parquet(need(prepared_file(cfg), "prepare"))
    numeric, slots, _ = _space_columns(cfg, prepared.columns)
    points = space_points(
        prepared.select(numeric).to_numpy(),
        prepared.select(slots).to_numpy(),
        cfg.method.distance,
    )
    return points, prepared[LABEL].to_numpy()


def prepared_columns(cfg: DatasetConfig) -> list[str]:
    """Return the configured column behind every column of read_points.

    Args:
        cfg: The dataset config.

    Returns:
        One name per column of the points: a numeric column's own, the
        categorical column of a one-hot slot.

    Raises:
        FileNotFoundError: When stage prepare has not run.
    """
    names = pl.read_parquet_schema(need(prepared_file(cfg), "prepare")).names()
    numeric, _, owners = _space_columns(cfg, names)
    return [*numeric, *owners]


def _space_columns(
    cfg: DatasetConfig, columns: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """Return the numeric columns, the slots and the column of each slot.

    A slot f"{column}={value}" belongs to the longest categorical column
    it starts with; MISSING and LABEL are neither.
    """
    present = feature_cols(columns, cfg.num_cols, cfg.cat_cols).num
    by_length = sorted(cfg.cat_cols, key=len, reverse=True)
    numeric = list(present)
    slots, owners = [], []
    for name in columns:
        if name in present or name in (MISSING, LABEL):
            continue
        owner = next(c for c in by_length if name.startswith(f"{c}="))
        slots.append(name)
        owners.append(owner)
    return numeric, slots, owners


def read_unparsable(cfg: DatasetConfig) -> pl.DataFrame:
    """Return the numeric cells whose text was not a number.

    Args:
        cfg: The dataset config.

    Returns:
        Columns row (the row of the data file) and column, as stage
        clean wrote them (data.unparsable_cells).

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    return pl.read_parquet(need(unparsable_file(cfg), "clean"))


def read_weights(cfg: DatasetConfig) -> np.ndarray:
    """Return the weight of every row: its copies in the CSV.

    Args:
        cfg: The dataset config.

    Returns:
        DUPLICATES of the data file as Float64, shape (n,). Row i is
        row i of the data file.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    rows = pl.read_parquet(need(data_file(cfg), "clean"), columns=[DUPLICATES])
    return rows[DUPLICATES].cast(pl.Float64).to_numpy()


def read_classes(cfg: DatasetConfig) -> list[str]:
    """Return the class names.

    Args:
        cfg: The dataset config.

    Returns:
        The names, indexed by class code.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    return json.loads(need(classes_file(cfg), "clean").read_text())
