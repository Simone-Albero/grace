"""The files of every stage, shared with the dashboard.

Every rerun of a stage overwrites its files, so the files always belong
to the latest run of that stage.
"""

from __future__ import annotations

__all__ = [
    "assignments_file",
    "classes_file",
    "complexity_file",
    "data_file",
    "graph_file",
    "need",
    "read_classes",
    "read_data",
    "regions_file",
    "space_file",
]

import json
from pathlib import Path

import polars as pl

from src.config import DatasetConfig


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


def classes_file(cfg: DatasetConfig) -> Path:
    """Return the file of the class names (stage clean).

    Args:
        cfg: The dataset config.

    Returns:
        The JSON path; the file is a list of names indexed by code.
    """
    return cfg.output_dir / "classes.json"


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


def space_file(cfg: DatasetConfig) -> Path:
    """Return the file of the fitted Space (stage regions).

    Args:
        cfg: The dataset config.

    Returns:
        The joblib path of the src.space.Space.
    """
    return cfg.output_dir / "space.joblib"


def graph_file(cfg: DatasetConfig) -> Path:
    """Return the file of the Graph arrays (stage regions).

    Args:
        cfg: The dataset config.

    Returns:
        The npz path of the src.graph.Graph fields.
    """
    return cfg.output_dir / "graph.npz"


def complexity_file(cfg: DatasetConfig) -> Path:
    """Return the file of the descriptors (stage complexity).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path of the long table.
    """
    return cfg.output_dir / "complexity.parquet"


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
