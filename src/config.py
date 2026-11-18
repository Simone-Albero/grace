"""Configs: dataset and method, from the yamls of configs/.

Every stage receives one `DatasetConfig`, which carries the method.
`load_config` assembles and validates it, so a bad value fails before
any stage reads data.
"""

from __future__ import annotations

__all__ = [
    "DATASET_DIR",
    "DatasetConfig",
    "ProvenanceConfig",
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
from src.provenance import KEYS, TIME
from src.quality import decisive_rows

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
    """

    max_fit_samples: int


@dataclass(frozen=True)
class QualityConfig:
    """The defects of the cells and of their regions (`src.quality`).

    Attributes:
        tau: The share of a cell's rows that may be confused with a
            rival class, in (0, 0.5]: the one threshold of the report.
        max_rows_per_cell: The most distinct rows of the rival side
            when a cell is measured against a rival class (its rival
            cells of that class pooled); the cell itself is always
            measured whole.
        rival_rows: The distinct rows of a cell its rivals are searched
            from, a uniform sample above this: a budget, as the search
            compares every row with every other.
        max_group: The most columns removed from a pair to find a
            shortcut.
        time_share: The share of a cell's rows its time window holds,
            in (0, 1]: the window of the central rows, so that a few
            stray ones do not widen it. Description only, no defect
            depends on it.
        confidence: The level of the interval around every measure of a
            cell, in (0.5, 1): a cell has a defect when the measure is
            past its threshold beyond the error of sampling, has none
            when it is short of it beyond that error, and is undecided
            otherwise.
    """

    tau: float
    max_rows_per_cell: int
    rival_rows: int
    max_group: int
    time_share: float
    confidence: float


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        distance: One of DISTANCES: how the numeric columns are
            compared (graph.space_points).
        prepare: The prepared features.
        cells: How the cells of every class are built.
        quality: The quality report.
    """

    distance: str
    prepare: PrepareConfig
    cells: CellsConfig
    quality: QualityConfig

    @property
    def min_rows(self) -> int:
        """The fewest distinct rows of a cell, the one limit of the cut.

        Twice the rows a share must be measured on to be able to fall
        short of tau (`quality.decisive_rows`), since half of a cell's
        rows choose and half are measured: under that a cell could
        never be told to lack a rival defect, whatever it holds. It is
        so no choice of its own, but what tau and the confidence ask
        for (50 rows at tau 0.1 and 0.95).
        """
        return 2 * decisive_rows(self.quality.tau, self.quality.confidence)


@dataclass(frozen=True)
class ProvenanceConfig:
    """The columns that say where each row of the CSV comes from (`src.provenance`).

    Never features: stage clean reads them for every row of the CSV and
    stage quality_report measures them per cell. Each is optional.

    Attributes:
        src_ip: The column of the source address.
        src_port: The column of the source port.
        dst_ip: The column of the destination address.
        dst_port: The column of the destination port.
        time: The column of the time of the row.
        time_format: The format of time's text (chrono's strftime, as
            polars reads it, e.g. "%Y-%m-%d %H:%M:%S%.f"), set with time
            and only with it.
    """

    src_ip: str | None = None
    src_port: str | None = None
    dst_ip: str | None = None
    dst_port: str | None = None
    time: str | None = None
    time_format: str | None = None

    def columns(self) -> dict[str, str]:
        """Return the configured keys, in the order of KEYS then TIME, with their columns."""
        keys = (*KEYS, TIME)
        return {key: getattr(self, key) for key in keys if getattr(self, key)}


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
        provenance: The columns of where each row comes from, or None
            when the dataset names none.
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
    provenance: ProvenanceConfig | None
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
        "quality.max_rows_per_cell": quality.max_rows_per_cell,
        "quality.rival_rows": quality.rival_rows,
        "quality.max_group": quality.max_group,
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
    _check(
        0 < quality.tau <= 0.5, f"quality.tau must be in (0, 0.5], got {quality.tau}"
    )
    _check(
        0 < quality.time_share <= 1,
        f"quality.time_share must be in (0, 1], got {quality.time_share}",
    )
    _check(
        0.5 < quality.confidence < 1,
        f"quality.confidence must be in (0.5, 1), got {quality.confidence}",
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


def _provenance(raw: Mapping[str, Any] | None, path: Path) -> ProvenanceConfig | None:
    """Return the provenance block of a dataset yaml, None when absent or empty."""
    if not raw:
        return None
    _check_keys(raw, ProvenanceConfig, path, prefix="provenance.")
    provenance = ProvenanceConfig(**raw)
    _check(
        bool(provenance.time) == bool(provenance.time_format),
        f"provenance.time and provenance.time_format go together in {path}",
    )
    _check(bool(provenance.columns()), f"provenance names no column in {path}: {raw}")
    return provenance


def load_config(dataset: str, method: str = "default") -> DatasetConfig:
    """Load a dataset config with its method.

    Args:
        dataset: A name of `configs/datasets` or the path of a yaml.
        method: A name of `configs/method` or the path of a yaml.

    Returns:
        The validated config.

    Raises:
        FileNotFoundError: When a yaml does not exist.
        ValueError: When a key is unknown, a value unknown or out of
            its range, or a provenance column is also a feature.
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
    provenance = _provenance(raw.get("provenance"), path)
    if provenance is not None:
        features = {*(raw["num_cols"] or ()), *(raw["cat_cols"] or ())}
        shared = sorted(set(provenance.columns().values()) & features)
        _check(not shared, f"provenance columns are never features, got {shared}")
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
        provenance=provenance,
        method=load_method_config(method),
    )
