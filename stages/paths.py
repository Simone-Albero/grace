"""The files of every stage, shared with the dashboard.

Every rerun of a stage overwrites its files, so the files always belong
to the latest run of that stage.
"""

from __future__ import annotations

__all__ = [
    "assignments_file",
    "cell_tree_file",
    "cells_file",
    "classes_file",
    "conflicts_file",
    "data_blocks",
    "data_file",
    "need",
    "points_columns_file",
    "points_file",
    "prepared_columns",
    "provenance_file",
    "read_classes",
    "read_conflicts",
    "read_data",
    "read_labels",
    "read_points",
    "read_provenance",
    "read_rows",
    "read_unparsable",
    "read_weights",
    "rivals_file",
    "unparsable_file",
]

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.config import DatasetConfig
from src.data import DUPLICATES, LABEL

_BLOCK_ROWS = 500_000  # rows of the data file read at once by data_blocks


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


def provenance_file(cfg: DatasetConfig) -> Path:
    """Return the file of where every row of the CSV comes from (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path; stage clean writes it only when the dataset
        names provenance columns.
    """
    return cfg.output_dir / "provenance.parquet"


def conflicts_file(cfg: DatasetConfig) -> Path:
    """Return the file of the rows equal on every feature under different labels (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path: columns row (of the data file) and group
        (data.conflict_groups), one row per row in a group.
    """
    return cfg.output_dir / "conflicts.parquet"


def points_file(cfg: DatasetConfig) -> Path:
    """Return the file of the points (stage prepare).

    Args:
        cfg: The dataset config.

    Returns:
        The .npy path: float32, shape (n, d), row i the point of row i
        of the data file (prepare.transform).
    """
    return cfg.output_dir / "points.npy"


def points_columns_file(cfg: DatasetConfig) -> Path:
    """Return the file of the names of the columns of the points (stage prepare).

    Args:
        cfg: The dataset config.

    Returns:
        The JSON path: {"columns": prepare.column_names, "sources":
        prepare.column_sources}.
    """
    return cfg.output_dir / "points.json"


def cells_file(cfg: DatasetConfig) -> Path:
    """Return the file of one row per cell (stage cells).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "cells.parquet"


def cell_tree_file(cfg: DatasetConfig) -> Path:
    """Return the file of the cut of every class down to its cells (stage cells).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / "cell_tree.parquet"


def assignments_file(cfg: DatasetConfig) -> Path:
    """Return the file of the cell of every row (stage cells).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path; it covers every row of the data.
    """
    return cfg.output_dir / "assignments.parquet"


def rivals_file(cfg: DatasetConfig) -> Path:
    """Return the file of the nearest rows of every row read (stage quality_report).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path: columns row (of the data file, the rows read,
        ascending), rival_row (its nearest row of another class, -1 for
        none), confused (1, 0.5 or 0) and own_row (its nearest other
        row of its class, -1 for none), as quality.nearest_rivals
        returns them.
    """
    return cfg.output_dir / "rivals.parquet"


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
    """Return the points of every row, on disk, and their classes.

    Args:
        cfg: The dataset config.

    Returns:
        The points, a read-only float32 memmap of shape (n, d) (every
        distance from stage cells on is euclidean on them), and the
        class codes, shape (n,) int64. Row i is row i of the data file;
        column j is column j of points_columns_file.

    Raises:
        FileNotFoundError: When stage prepare has not run.
    """
    points = np.load(need(points_file(cfg), "prepare"), mmap_mode="r")
    return points, read_labels(cfg)


def read_labels(cfg: DatasetConfig) -> np.ndarray:
    """Return the class code of every row of the data file.

    Args:
        cfg: The dataset config.

    Returns:
        LABEL of the data file, shape (n,) int64.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    rows = pl.read_parquet(need(data_file(cfg), "clean"), columns=[LABEL])
    return rows[LABEL].cast(pl.Int64).to_numpy()


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
    names = json.loads(need(points_columns_file(cfg), "prepare").read_text())
    return list(names["sources"])


def read_rows(
    cfg: DatasetConfig, rows: np.ndarray, columns: Sequence[str] | None = None
) -> pl.DataFrame:
    """Return some rows of the data file, read in streaming.

    Args:
        cfg: The dataset config.
        rows: Their positions in the data file, ascending, distinct.
        columns: The columns read; None for all.

    Returns:
        The rows, in the order of rows.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    at = "__row__"
    wanted = pl.Series(np.asarray(rows, dtype=np.int64)).implode()
    scan = pl.scan_parquet(need(data_file(cfg), "clean")).with_row_index(at)
    if columns is not None:
        scan = scan.select(at, *columns)
    found = scan.filter(pl.col(at).cast(pl.Int64).is_in(wanted))
    return found.collect(engine="streaming").sort(at).drop(at)


def data_blocks(
    cfg: DatasetConfig, columns: Sequence[str] | None = None
) -> Iterator[pl.DataFrame]:
    """Yield the rows of the data file, in order, a block at a time.

    Args:
        cfg: The dataset config.
        columns: The columns read; None for all.

    Yields:
        Consecutive blocks of at most _BLOCK_ROWS rows.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    source = pq.ParquetFile(need(data_file(cfg), "clean"))
    read = None if columns is None else list(columns)
    for batch in source.iter_batches(batch_size=_BLOCK_ROWS, columns=read):
        yield pl.from_arrow(batch)


def read_conflicts(cfg: DatasetConfig, n: int) -> np.ndarray:
    """Return the conflict group of every row of the data file.

    Args:
        cfg: The dataset config.
        n: The rows of the data file.

    Returns:
        Shape (n,) int64: the group of conflicts_file, -1 for a row in
        none.

    Raises:
        FileNotFoundError: When stage clean has not run.
    """
    found = pl.read_parquet(need(conflicts_file(cfg), "clean"))
    groups = np.full(n, -1, dtype=np.int64)
    groups[found["row"].to_numpy()] = found["group"].to_numpy()
    return groups


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


def read_provenance(cfg: DatasetConfig) -> pl.DataFrame:
    """Return where every row of the CSV comes from.

    Args:
        cfg: The dataset config.

    Returns:
        The rows of provenance.parquet, as stage clean wrote them: row
        (the row of the data file it is a copy of), then the configured
        keys of provenance.KEYS and provenance.TIME.

    Raises:
        FileNotFoundError: When stage clean has not written it.
    """
    return pl.read_parquet(need(provenance_file(cfg), "clean"))


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
