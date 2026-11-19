"""Configs: dataset and method, from the yamls of configs/.

Every stage receives one `DatasetConfig`, which carries the method.
`load_config` assembles and validates it, so a bad value fails before
any stage reads data.
"""

from __future__ import annotations

__all__ = [
    "DATASET_DIR",
    "DatasetConfig",
    "AttackScenario",
    "ProvenanceConfig",
    "ScenarioConfig",
    "dataset_names",
    "load_config",
]

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
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
            from, a uniform sample above this: a budget for the nearest
            row search.
        confidence: The level of the interval around every measure of a
            cell, in (0.5, 1): a cell has a defect when the measure is
            past its threshold beyond the error of sampling, has none
            when it is short of it beyond that error, and is undecided
            otherwise.
    """

    tau: float
    max_rows_per_cell: int
    rival_rows: int
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

    Stage clean reads them for every row of the CSV and stage
    quality_report describes them per cell, and judges them against the
    scenario when the dataset declares one. Each is optional. The
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


# the provenance keys that are never features: where and when the capture happened
ORIGIN = ("src_ip", "src_port", TIME)


@dataclass(frozen=True)
class AttackScenario:
    """Where and when the documentation says an attack class was generated.

    Attributes:
        src_ip: The addresses of its attacking hosts; empty when not
            declared.
        time: Its windows of time, each (start, end), closed, in the
            clock of the CSV's time column; empty when not declared.
    """

    src_ip: tuple[str, ...] = ()
    time: tuple[tuple[dt.datetime, dt.datetime], ...] = ()


@dataclass(frozen=True)
class ScenarioConfig:
    """What the dataset's documentation says of where and when its classes were generated (`src.provenance`).

    Optional: a dataset without it does not measure off_scenario.

    Attributes:
        benign: The class of the background traffic, bound by no rule.
        background_ports: The destination ports of background services
            (DNS, NTP, DHCP and the like): a row of another class sent
            to one of them is no attack traffic. Empty when not
            declared.
        attacks: Per attack class, its hosts and windows.
    """

    benign: str
    background_ports: tuple[int, ...] = ()
    attacks: Mapping[str, AttackScenario] = field(default_factory=dict)


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
        max_class_rows: The most distinct rows a class keeps, a uniform
            sample of them past it (data.cap_classes); None keeps every
            row.
        top_n: The most frequent values of a categorical column with
            their own one-hot slot; the others share one.
        seed: The seed of every stage.
        output_dir: The folder of every file of every stage.
        db_path: The SQLite file.
        provenance: The columns of where each row comes from, or None
            when the dataset names none.
        scenario: What the documentation says of the classes, or None
            when the dataset declares none.
        method: The method.
    """

    name: str
    path: Path
    label_col: str
    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]
    min_class_count: int
    max_class_rows: int | None
    top_n: int
    seed: int
    output_dir: Path
    db_path: Path
    provenance: ProvenanceConfig | None
    scenario: ScenarioConfig | None
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
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
    _check(
        0 < quality.tau <= 0.5, f"quality.tau must be in (0, 0.5], got {quality.tau}"
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


def _scenario(
    raw: Mapping[str, Any] | None,
    provenance: ProvenanceConfig | None,
    path: Path,
) -> ScenarioConfig | None:
    """Return the scenario block of a dataset yaml, None when absent or empty.

    The keys are benign (required), background_ports and attacks, each
    attack with src_ip and time (a list of [start, end], ISO
    "YYYY-MM-DD HH:MM:SS", in the clock of the CSV's time column). A
    rule needs the provenance column it reads: background_ports
    dst_port, src_ip src_ip, time time.

    Raises:
        ValueError: When a key is unknown, benign is missing, a rule
            lacks its provenance column, a port is no integer, or a
            window does not start before it ends.
    """
    if not raw:
        return None
    _check_keys(raw, ScenarioConfig, path, prefix="scenario.")
    _check(bool(raw.get("benign")), f"scenario.benign is missing in {path}")
    named = provenance.columns() if provenance is not None else {}

    def needs(key: str, rule: str) -> None:
        _check(
            key in named,
            f"scenario {rule} reads provenance.{key}, which {path} does not name",
        )

    ports = tuple(raw.get("background_ports") or ())
    _check(
        all(isinstance(port, int) and not isinstance(port, bool) for port in ports),
        f"scenario.background_ports must be integers, got {list(ports)}",
    )
    if ports:
        needs("dst_port", "background_ports")
    attacks = {}
    for name, attack in (raw.get("attacks") or {}).items():
        attack = attack or {}
        _check_keys(attack, AttackScenario, path, prefix=f"scenario.attacks.{name}.")
        hosts = tuple(str(host) for host in attack.get("src_ip") or ())
        windows = tuple(
            (_time(start, path), _time(end, path))
            for start, end in attack.get("time") or ()
        )
        for start, end in windows:
            _check(
                start < end,
                f"scenario.attacks.{name}.time: {start} must be before {end}",
            )
        if hosts:
            needs("src_ip", f"attacks.{name}.src_ip")
        if windows:
            needs(TIME, f"attacks.{name}.time")
        attacks[str(name)] = AttackScenario(src_ip=hosts, time=windows)
    return ScenarioConfig(
        benign=str(raw["benign"]), background_ports=ports, attacks=attacks
    )


def _time(value: Any, path: Path) -> dt.datetime:
    """Return a time of a scenario window: a yaml timestamp, or ISO text."""
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    try:
        return dt.datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"scenario window time {value!r} in {path} is no ISO time")


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
            its range, a column of the source or the time is also a
            feature, or the scenario is invalid (_scenario).
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
    max_class_rows = raw.get("max_class_rows")
    _check(
        max_class_rows is None
        or (
            isinstance(max_class_rows, int) and max_class_rows >= raw["min_class_count"]
        ),
        f"max_class_rows must be null or >= min_class_count, got {max_class_rows}",
    )
    provenance = _provenance(raw.get("provenance"), path)
    if provenance is not None:
        features = {*(raw["num_cols"] or ()), *(raw["cat_cols"] or ())}
        origin = {c for k, c in provenance.columns().items() if k in ORIGIN}
        shared = sorted(origin & features)
        _check(not shared, f"the source and the time are never features, got {shared}")
    scenario = _scenario(raw.get("scenario"), provenance, path)
    return DatasetConfig(
        name=raw["name"],
        path=Path(raw["path"]),
        label_col=raw["label_col"],
        num_cols=tuple(raw["num_cols"] or ()),
        cat_cols=tuple(raw["cat_cols"] or ()),
        min_class_count=raw["min_class_count"],
        max_class_rows=max_class_rows,
        top_n=raw["top_n"],
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        db_path=Path(raw["db_path"]),
        provenance=provenance,
        scenario=scenario,
        method=load_method_config(method),
    )
