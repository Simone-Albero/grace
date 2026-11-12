"""Configs: dataset and method, from the yamls of configs/.

Every stage receives one `DatasetConfig`, which carries the method.
`load_config` assembles and validates it, so a bad value fails before
any stage reads data.
"""

from __future__ import annotations

__all__ = [
    "DATASET_DIR",
    "DatasetConfig",
    "dataset_names",
    "load_config",
]

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.data import DUPLICATES
from src.prepare import MISSING

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DATASET_DIR = CONFIG_DIR / "datasets"
METHOD_DIR = CONFIG_DIR / "method"

DISTANCES = ("euclidean", "cosine")


@dataclass(frozen=True)
class PrepareConfig:
    """The prepared features distances are measured on (`src.prepare`).

    Attributes:
        cat_cost: The average squared distance a categorical column
            adds between two random rows, in units of a standardized
            number's.
    """

    cat_cost: float


@dataclass(frozen=True)
class CellsConfig:
    """The cells every defect is measured on (`src.cells`).

    Attributes:
        max_fit_samples: The most distinct rows of a class its Ward tree
            is built on; every other row joins its nearest sampled row.
        max_rows: The most distinct rows of a cell, the same for every
            class: the resolution of the report (a defect touching tau
            of a cell's rows is seen), and what makes two cells compared
            sampled alike.
        min_rows: The fewest distinct rows of a cell: a smaller branch
            of the tree joins the cells around it.
    """

    max_fit_samples: int
    max_rows: int
    min_rows: int


@dataclass(frozen=True)
class QualityConfig:
    """The defects of the cells and of their regions (`src.quality`).

    Attributes:
        tau: The share of a cell's rows that may be confused with a
            rival's, in (0, 0.5]: the one threshold of class_overlap and
            shortcut.
        max_rows_per_cell: The distinct rows of a cell its pairs are
            measured on, a uniform sample above this.
        rival_rows: The distinct rows of a cell its rivals are searched
            from, a uniform sample above this: a budget, as the search
            compares every row with every other.
        max_group: The most columns removed from a pair to find a
            shortcut.
    """

    tau: float
    max_rows_per_cell: int
    rival_rows: int
    max_group: int


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        distance: One of DISTANCES: how the numeric columns are
            compared (graph.space_points).
        prepare: The prepared features.
        cells: The cells of every class.
        quality: The quality report.
    """

    distance: str
    prepare: PrepareConfig
    cells: CellsConfig
    quality: QualityConfig


@dataclass(frozen=True)
class DatasetConfig:
    """A dataset and its method.

    Attributes:
        name: The dataset name.
        path: The raw CSV.
        label_col: The class column.
        num_cols: The numeric columns.
        cat_cols: The categorical columns.
        min_class_count: Classes with fewer rows are dropped.
        top_n: The most frequent values of a categorical column with
            their own one-hot slot; the others share one.
        seed: The seed of every stage.
        output_dir: The folder of every file of every stage.
        db_path: The SQLite file.
        method: The method.
    """

    name: str
    path: Path
    label_col: str
    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]
    min_class_count: int
    top_n: int
    seed: int
    output_dir: Path
    db_path: Path
    method: MethodConfig


def _yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _path(name_or_path: str, directory: Path) -> Path:
    """Return name_or_path if a file, else <directory>/<name>.yaml."""
    path = Path(name_or_path)
    if path.is_file() or path.suffix == ".yaml":
        return path
    return directory / f"{name_or_path}.yaml"


def _names(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yaml"))


def dataset_names() -> list[str]:
    """Return the names of the dataset configs, sorted."""
    return _names(DATASET_DIR)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_keys(
    raw: Mapping[str, Any], config: type, path: Path, prefix: str = ""
) -> None:
    known = {field.name for field in fields(config)} - {"method"}
    unknown = sorted(f"{prefix}{key}" for key in set(raw) - known)
    _check(not unknown, f"unknown keys in {path}: {unknown}")


def _validate_method(method: MethodConfig) -> None:
    _check(
        method.distance in DISTANCES,
        f"distance must be one of {DISTANCES}, got {method.distance!r}",
    )
    cells, quality = method.cells, method.quality
    positive = {
        "cells.max_fit_samples": cells.max_fit_samples,
        "cells.min_rows": cells.min_rows,
        "quality.max_rows_per_cell": quality.max_rows_per_cell,
        "quality.rival_rows": quality.rival_rows,
        "quality.max_group": quality.max_group,
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
    # A node over max_rows always has a child of min_rows or more.
    _check(
        cells.max_rows >= 2 * cells.min_rows,
        f"cells.max_rows must be >= 2 * cells.min_rows, got {cells.max_rows}",
    )
    _check(
        0 < quality.tau <= 0.5, f"quality.tau must be in (0, 0.5], got {quality.tau}"
    )
    cat_cost = method.prepare.cat_cost
    _check(cat_cost > 0, f"prepare.cat_cost must be > 0, got {cat_cost}")


def load_method_config(name_or_path: str = "default") -> MethodConfig:
    """Load and validate a method config.

    Args:
        name_or_path: A name of `configs/method` or the path of a yaml.

    Returns:
        The method config.

    Raises:
        ValueError: When a key is unknown, or a value unknown or out of
            its range.
    """
    path = _path(name_or_path, METHOD_DIR)
    raw = _yaml(path)
    _check_keys(raw, MethodConfig, path)
    sections = {
        "prepare": PrepareConfig,
        "cells": CellsConfig,
        "quality": QualityConfig,
    }
    for key, section in sections.items():
        _check_keys(raw[key], section, path, prefix=f"{key}.")
    method = MethodConfig(
        distance=raw["distance"],
        **{key: section(**raw[key]) for key, section in sections.items()},
    )
    _validate_method(method)
    return method


def load_config(dataset: str, method: str = "default") -> DatasetConfig:
    """Load a dataset config with its method.

    Args:
        dataset: A name of `configs/datasets` or the path of a yaml.
        method: A name of `configs/method` or the path of a yaml.

    Returns:
        The validated config.

    Raises:
        FileNotFoundError: When a yaml does not exist.
        ValueError: When a key is unknown, or a value unknown or out of
            its range.
    """
    path = _path(dataset, DATASET_DIR)
    raw = _yaml(path)
    _check_keys(raw, DatasetConfig, path)
    columns = [*(raw["num_cols"] or ()), *(raw["cat_cols"] or ()), raw["label_col"]]
    for added, stage in ((DUPLICATES, "clean"), (MISSING, "prepare")):
        _check(
            added not in columns,
            f"{added} is the column stage {stage} adds, not a configured one",
        )
    _check(raw["top_n"] >= 1, f"top_n must be >= 1, got {raw['top_n']}")
    return DatasetConfig(
        name=raw["name"],
        path=Path(raw["path"]),
        label_col=raw["label_col"],
        num_cols=tuple(raw["num_cols"] or ()),
        cat_cols=tuple(raw["cat_cols"] or ()),
        min_class_count=raw["min_class_count"],
        top_n=raw["top_n"],
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        db_path=Path(raw["db_path"]),
        method=load_method_config(method),
    )
