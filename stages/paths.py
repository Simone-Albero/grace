"""The files of every stage, shared with the dashboard.

Every rerun of a stage overwrites its files, so the files always belong
to the latest run of that stage (and model).
"""

from __future__ import annotations

__all__ = [
    "assignments_file",
    "classes_file",
    "complexity_file",
    "contributions_file",
    "estimates_file",
    "failures_file",
    "graph_file",
    "model_file",
    "need",
    "predictions_file",
    "preprocessor_file",
    "read_classes",
    "read_split",
    "regions_file",
    "regressor_file",
    "space_file",
    "split_file",
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


def split_file(cfg: DatasetConfig, name: str) -> Path:
    """Return the file of the prepared rows of one split (stage split).

    Args:
        cfg: The dataset config.
        name: The split name.

    Returns:
        The parquet path.
    """
    return cfg.output_dir / f"{name}.parquet"


def classes_file(cfg: DatasetConfig) -> Path:
    """Return the file of the class names (stage split).

    Args:
        cfg: The dataset config.

    Returns:
        The JSON path; the file is a list of names indexed by code.
    """
    return cfg.output_dir / "classes.json"


def preprocessor_file(cfg: DatasetConfig) -> Path:
    """Return the file of the fitted Preprocessor (stage split).

    Args:
        cfg: The dataset config.

    Returns:
        The joblib path of the src.prepare.Preprocessor.
    """
    return cfg.output_dir / "preprocessor.joblib"


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
        The parquet path; it covers every row of every split.
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


def model_file(cfg: DatasetConfig) -> Path:
    """Return the file of the fitted classifier (stage train).

    Args:
        cfg: The dataset config.

    Returns:
        The joblib path of the pipeline of cfg.model.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / "models" / f"{_model(cfg)}.joblib"


def predictions_file(cfg: DatasetConfig, split: str) -> Path:
    """Return the file of the predictions on a split (stage train).

    Args:
        cfg: The dataset config.
        split: "val" or "test".

    Returns:
        The parquet path of the predictions of cfg.model.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / f"pred_{_model(cfg)}_{split}.parquet"


def failures_file(cfg: DatasetConfig) -> Path:
    """Return the file of the failure rates (stage failures).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path of the observed and baseline rates per region.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / f"failures_{_model(cfg)}.parquet"


def estimates_file(cfg: DatasetConfig) -> Path:
    """Return the file of the estimates (stage regress).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path of the out-of-fold estimate per used region.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / f"regress_{_model(cfg)}_estimates.parquet"


def contributions_file(cfg: DatasetConfig) -> Path:
    """Return the file of the contributions (stage regress).

    Args:
        cfg: The dataset config.

    Returns:
        The parquet path of the TreeSHAP contributions per region.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / f"regress_{_model(cfg)}_contributions.parquet"


def regressor_file(cfg: DatasetConfig) -> Path:
    """Return the file of the refit forest (stage regress).

    Args:
        cfg: The dataset config.

    Returns:
        The joblib path of the Random Forest of cfg.model.

    Raises:
        ValueError: When cfg.model is None.
    """
    return cfg.output_dir / f"regress_{_model(cfg)}.joblib"


def _model(cfg: DatasetConfig) -> str:
    if cfg.model is None:
        raise ValueError(f"dataset {cfg.name}: this file needs a model, got none")
    return cfg.model.name


def read_split(cfg: DatasetConfig, name: str) -> pl.DataFrame:
    """Return the prepared rows of one split.

    Args:
        cfg: The dataset config.
        name: The split name.

    Returns:
        The rows, as stage split wrote them.

    Raises:
        FileNotFoundError: When stage split has not run.
    """
    return pl.read_parquet(need(split_file(cfg, name), "split"))


def read_classes(cfg: DatasetConfig) -> list[str]:
    """Return the class names.

    Args:
        cfg: The dataset config.

    Returns:
        The names, indexed by class code.

    Raises:
        FileNotFoundError: When stage split has not run.
    """
    return json.loads(need(classes_file(cfg), "split").read_text())
