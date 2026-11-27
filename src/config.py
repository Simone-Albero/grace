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
    "load_config",
]

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.data import DUPLICATES
from src.provenance import KEYS, TIME
from src.quality import resolution_rows

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DATASET_DIR = CONFIG_DIR / "datasets"
METHOD_DIR = CONFIG_DIR / "method"


@dataclass(frozen=True)
class CellsConfig:
    """The cells every defect is measured on (`src.cells`).

    Attributes:
        max_sample_rows: The most distinct rows of a class the method
            reads, a uniform sample of them (graph.sample_rows): its Ward
            tree is built on them and every share is read on them; every
            other row is a candidate nearest row of the rows read, and
            joins the cell of its nearest sampled row to be counted.
            None reads every row: the exact method.
    """

    max_sample_rows: int | None


@dataclass(frozen=True)
class QualityConfig:
    """The defects of the cells and of their regions (`src.quality`).

    Attributes:
        tau: The share of a cell's rows that may be confused with a
            rival class, in (0, 0.5]: the one threshold of the report.
        confidence: The level of the interval around every measure of a
            cell, in (0.5, 1): a cell has a defect when the measure is
            past its threshold beyond the error of sampling, has none
            when it is short of it beyond that error, and is undecided
            otherwise.
    """

    tau: float
    confidence: float


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        cells: How the cells of every class are built.
        quality: The quality report.
    """

    cells: CellsConfig
    quality: QualityConfig

    @property
    def min_rows(self) -> int:
        """The fewest distinct rows of a cell, the one limit of the cut.

        The rows on which a share tau away from a threshold of the
        report, tau or one half, is decided against it
        (`quality.resolution_rows`): a cell is then undecided only
        within tau of a threshold. It is so no choice of its own, but
        what tau and the confidence ask for, about z² / (4 tau²) (68
        rows at tau 0.1 and 0.95).
        """
        return resolution_rows(self.quality.tau, self.quality.confidence)


@dataclass(frozen=True)
class ProvenanceConfig:
    """The columns that say where each row of the CSV comes from (`src.provenance`).

    Stage clean reads them for every row of the CSV and stage
    quality_report measures them per cell, as support of the verdicts:
    they never decide one. Each is optional. The
    source and the time (ORIGIN) are never features: a classifier that
    learned them would learn the capture. The destination may be one
    too: its port names the service, which the flows of an attack target.

    Attributes:
        src_ip: The column of the source address.
        src_port: The column of the source port.
        dst_ip: The column of the destination address.
        dst_port: The column of the destination port.
        time: The column of the time of the row.
        time_format: The format of time's text (chrono's strftime, as
            polars reads it, e.g. "%Y-%m-%d %H:%M:%S%.f"), set with time
            and only with it.
        time_afternoon_before: For a time written on 12 hours without AM
            or PM, the first hour of the morning the capture holds, in
            [1, 12]: an hour under it is of the afternoon and gets 12
            hours (provenance.read_provenance). None, the default, reads
            the hours as written; set only with time.
    """

    src_ip: str | None = None
    src_port: str | None = None
    dst_ip: str | None = None
    dst_port: str | None = None
    time: str | None = None
    time_format: str | None = None
    time_afternoon_before: int | None = None

    def columns(self) -> dict[str, str]:
        """Return the configured keys, in the order of KEYS then TIME, with their columns."""
        keys = (*KEYS, TIME)
        return {key: getattr(self, key) for key in keys if getattr(self, key)}


# the provenance keys that are never features: where and when the capture happened
ORIGIN = ("src_ip", "src_port", TIME)


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
        significant_digits: The significant digits every number is
            rounded to as it is read (data.round_significant), so that
            files writing the same value with more or fewer digits give
            the same row; None, the default, keeps every digit.
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
    significant_digits: int | None
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
    cells, quality = method.cells, method.quality
    rows = cells.max_sample_rows
    _check(
        rows is None or rows >= 1,
        f"cells.max_sample_rows must be >= 1 or null, got {rows}",
    )
    _check(
        0 < quality.tau <= 0.5, f"quality.tau must be in (0, 0.5], got {quality.tau}"
    )
    _check(
        0.5 < quality.confidence < 1,
        f"quality.confidence must be in (0.5, 1), got {quality.confidence}",
    )


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
        "cells": CellsConfig,
        "quality": QualityConfig,
    }
    for key, section in sections.items():
        _check_keys(raw[key], section, path, prefix=f"{key}.")
    method = MethodConfig(
        **{key: section(**raw[key]) for key, section in sections.items()}
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
    hour = provenance.time_afternoon_before
    _check(
        hour is None
        or (bool(provenance.time) and isinstance(hour, int) and 1 <= hour <= 12),
        f"provenance.time_afternoon_before must be an hour in [1, 12], with time, in {path}",
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
            its range, or a column of the source or the time is also a
            feature.
    """
    path = _path(dataset, DATASET_DIR)
    raw = _yaml(path)
    _check_keys(raw, DatasetConfig, path)
    columns = [*(raw["num_cols"] or ()), *(raw["cat_cols"] or ()), raw["label_col"]]
    _check(
        DUPLICATES not in columns,
        f"{DUPLICATES} is the column stage clean adds, not a configured one",
    )
    _check(raw["top_n"] >= 1, f"top_n must be >= 1, got {raw['top_n']}")
    digits = raw.get("significant_digits")
    _check(
        digits is None or (isinstance(digits, int) and 1 <= digits <= 17),
        f"significant_digits must be null or in [1, 17], got {digits}",
    )
    provenance = _provenance(raw.get("provenance"), path)
    if provenance is not None:
        features = {*(raw["num_cols"] or ()), *(raw["cat_cols"] or ())}
        origin = {c for k, c in provenance.columns().items() if k in ORIGIN}
        shared = sorted(origin & features)
        _check(not shared, f"the source and the time are never features, got {shared}")
    return DatasetConfig(
        name=raw["name"],
        path=Path(raw["path"]),
        label_col=raw["label_col"],
        num_cols=tuple(raw["num_cols"] or ()),
        cat_cols=tuple(raw["cat_cols"] or ()),
        min_class_count=raw["min_class_count"],
        top_n=raw["top_n"],
        significant_digits=digits,
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        db_path=Path(raw["db_path"]),
        provenance=provenance,
        method=load_method_config(method),
    )
