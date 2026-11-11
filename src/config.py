"""Configs: dataset and method, from the yamls of configs/.

Every stage receives one `DatasetConfig`, which carries the method.
`load_config` assembles and validates it, so a bad value fails before
any stage reads data.
"""

from __future__ import annotations

__all__ = [
    "DATASET_DIR",
    "HIGHLIGHTS",
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
class ClusteringConfig:
    """The per-class weighted Ward regions (`src.clustering`).

    Attributes:
        max_fit_samples: The most distinct rows of each half of a
            class: the tree is built on half A, its splits judged on
            half B; the other rows are labelled by vote.
        max_regions: The regions over all classes, shared equally
            between them.
        split_separation: The separation (S_low, `clustering.
            pair_separation`) two regions of a class must keep: a
            split is cut, and two pieces stay apart, at it or over.
        min_cluster_floor: The fewest rows of half B of each side of a
            split judged; a smaller branch is pruned and its rows
            labelled by vote.
        vote_k: The labelled neighbours that vote the region of an
            unlabelled row.
        stability_repeats: The times the regions of a class are made
            again on other halves to measure the stability of each
            region; 0 measures none.
        overlap_separation: Two regions of different classes overlap
            when their centroids lie closer than this many standard
            deviations of their rows along the line between them
            (`clustering.separation`).
    """

    max_fit_samples: int
    max_regions: int
    split_separation: float
    min_cluster_floor: int
    vote_k: int
    stability_repeats: int
    overlap_separation: float


HIGHLIGHTS = ("conflict", "kdn", "enn", "f3", "l2", "pdd", "tc", "dup_share")


@dataclass(frozen=True)
class QualityConfig:
    """The quality report on pairs of regions (`src.quality`).

    Attributes:
        max_rows_per_region: The distinct rows of a region its pairs
            are measured on, a uniform sample above this.
        k: The neighbours inside a pair behind kDN and ENN.
        candidate_separation: Two regions of different classes are
            measured when their separation (`clustering.separation`)
            is under this, in the full space or without the shortcut
            columns.
        diversity_pairs: The pairs of rows drawn per region for its
            diversity.
        highlight: Per key of HIGHLIGHTS, the value from which the
            dashboard marks a measure; the values are kept as they
            are.
    """

    max_rows_per_region: int
    k: int
    candidate_separation: float
    diversity_pairs: int
    highlight: dict[str, float]


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        distance: One of DISTANCES: how the numeric columns are
            compared (graph.space_points).
        prepare: The prepared features.
        clustering: The per-class clustering.
        quality: The quality report.
    """

    distance: str
    prepare: PrepareConfig
    clustering: ClusteringConfig
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
        shortcut_cols: The columns suspected of shortcuts: the quality
            report measures the pairs of regions again without them.
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
    shortcut_cols: tuple[str, ...]
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
    clustering = method.clustering
    positive = {
        "clustering.max_fit_samples": clustering.max_fit_samples,
        "clustering.max_regions": clustering.max_regions,
        "clustering.min_cluster_floor": clustering.min_cluster_floor,
        "clustering.vote_k": clustering.vote_k,
        "quality.max_rows_per_region": method.quality.max_rows_per_region,
        "quality.k": method.quality.k,
        "quality.diversity_pairs": method.quality.diversity_pairs,
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
    split = clustering.split_separation
    _check(split > 0, f"clustering.split_separation must be > 0, got {split}")
    repeats = clustering.stability_repeats
    _check(repeats >= 0, f"clustering.stability_repeats must be >= 0, got {repeats}")
    _check(
        clustering.min_cluster_floor >= 2,
        f"clustering.min_cluster_floor must be >= 2, got {clustering.min_cluster_floor}",
    )
    separation = clustering.overlap_separation
    _check(
        separation > 0,
        f"clustering.overlap_separation must be > 0, got {separation}",
    )
    candidate = method.quality.candidate_separation
    _check(
        candidate > 0,
        f"quality.candidate_separation must be > 0, got {candidate}",
    )
    highlight = method.quality.highlight
    _check(
        set(highlight) == set(HIGHLIGHTS),
        f"quality.highlight must hold {HIGHLIGHTS}, got {sorted(highlight)}",
    )
    for key, value in highlight.items():
        _check(0 <= value <= 1, f"quality.highlight.{key} must be in [0, 1]")
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
        "clustering": ClusteringConfig,
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
    shortcut_cols = tuple(raw.get("shortcut_cols") or ())
    unknown = sorted(set(shortcut_cols) - set(columns[:-1]))
    _check(not unknown, f"shortcut_cols must be configured columns, got {unknown}")
    return DatasetConfig(
        name=raw["name"],
        path=Path(raw["path"]),
        label_col=raw["label_col"],
        num_cols=tuple(raw["num_cols"] or ()),
        cat_cols=tuple(raw["cat_cols"] or ()),
        min_class_count=raw["min_class_count"],
        top_n=raw["top_n"],
        shortcut_cols=shortcut_cols,
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        db_path=Path(raw["db_path"]),
        method=load_method_config(method),
    )
