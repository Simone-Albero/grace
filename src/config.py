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
class GraphConfig:
    """The k-nearest-neighbour graph of the rows (`src.graph`).

    Attributes:
        max_samples: The rows in the graph, a uniform sample above
            this.
        k: The neighbours of each row.
    """

    max_samples: int
    k: int


@dataclass(frozen=True)
class ClusteringConfig:
    """The per-class weighted Ward regions (`src.clustering`).

    Attributes:
        max_fit_samples: The distinct rows of a class the tree is built
            and the cut scored on; the others are labelled by vote.
        max_regions: The regions over all classes, shared equally
            between them.
        min_cluster_floor: The fewest scored rows (of the
            max_fit_samples) of a region; a smaller branch is pruned
            and its rows labelled by vote.
        hardness_k: The neighbours behind the kDN hardness.
        vote_k: The labelled neighbours that vote the region of an
            unlabelled row.
    """

    max_fit_samples: int
    max_regions: int
    min_cluster_floor: int
    hardness_k: int
    vote_k: int


@dataclass(frozen=True)
class ComplexityConfig:
    """The descriptors of every region and class (`src.complexity`).

    Attributes:
        top_k_clusters: The nearest rivals each population is compared
            with.
        max_sample_per_region: The rows measured per region when the
            graph samples the rows.
        max_sample_per_class: The rows measured per class when the
            graph samples the rows.
        silhouette_max_samples: The subsample behind the silhouette
            tail.
        silhouette_min_per_cluster: The rows per population kept in
            that subsample.
    """

    top_k_clusters: int
    max_sample_per_region: int
    max_sample_per_class: int
    silhouette_max_samples: int
    silhouette_min_per_cluster: int


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        distance: One of DISTANCES: how the numeric columns are
            compared (graph.space_points).
        prepare: The prepared features.
        graph: The k-NN graph of the rows.
        clustering: The per-class clustering.
        complexity: The descriptors.
    """

    distance: str
    prepare: PrepareConfig
    graph: GraphConfig
    clustering: ClusteringConfig
    complexity: ComplexityConfig


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
    clustering = method.clustering
    positive = {
        "graph.k": method.graph.k,
        "graph.max_samples": method.graph.max_samples,
        "clustering.max_fit_samples": clustering.max_fit_samples,
        "clustering.max_regions": clustering.max_regions,
        "clustering.min_cluster_floor": clustering.min_cluster_floor,
        "clustering.hardness_k": clustering.hardness_k,
        "clustering.vote_k": clustering.vote_k,
        "complexity.top_k_clusters": method.complexity.top_k_clusters,
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
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
        "graph": GraphConfig,
        "clustering": ClusteringConfig,
        "complexity": ComplexityConfig,
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
