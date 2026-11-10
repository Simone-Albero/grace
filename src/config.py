"""Configs: dataset, model and method, from the yamls of configs/.

Every stage receives one `DatasetConfig`, which carries the method and,
for the stages of `stages.MODEL_STAGES`, the model. `load_config`
assembles and validates it, so a bad value fails before any stage reads
data.
"""

from __future__ import annotations

__all__ = [
    "DATASET_DIR",
    "DatasetConfig",
    "ModelConfig",
    "dataset_names",
    "load_config",
    "model_names",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.baselines import BASELINES
from src.clustering import ALGORITHMS, tunable_params
from src.models import MODELS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DATASET_DIR = CONFIG_DIR / "datasets"
MODEL_DIR = CONFIG_DIR / "models"
METHOD_DIR = CONFIG_DIR / "method"

DISTANCES = ("euclidean", "cosine")


@dataclass(frozen=True)
class SplitConfig:
    """Shares of the rows that go to each split, stratified by class.

    Attributes:
        train: The share of train.
        val: The share of val.
        test: The share of test.
    """

    train: float
    val: float
    test: float


@dataclass(frozen=True)
class ModelConfig:
    """A classifier: a key of `src.models.MODELS` and its kwargs.

    Attributes:
        name: The key of `src.models.MODELS`.
        params: The kwargs of the classifier.
    """

    name: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class SpaceConfig:
    """The euclidean space regions live in (`src.space`).

    Attributes:
        top_k: The most frequent codes per categorical column with
            their own slot.
        cat_cost: The squared distance a categorical mismatch adds.
    """

    top_k: int
    cat_cost: float


@dataclass(frozen=True)
class GraphConfig:
    """The k-nearest-neighbour graph of train (`src.graph`).

    Attributes:
        max_samples: The train rows in the graph, a uniform sample
            above this.
        k: The neighbours of each row.
    """

    max_samples: int
    k: int


@dataclass(frozen=True)
class ClusteringConfig:
    """The per-class clustering of the regions (`src.clustering`).

    Attributes:
        name: The key of `src.clustering.ALGORITHMS`.
        params: The keyword-only params of the fit function; a list is
            swept.
        max_fit_samples: The rows a candidate is fitted and scored on.
        max_regions: The regions over all classes, shared equally
            between them.
        min_cluster_floor: The finest granularity and the merge floor.
        hardness_k: The neighbours behind the kDN hardness.
    """

    name: str
    params: Mapping[str, Any]
    max_fit_samples: int
    max_regions: int
    min_cluster_floor: int
    hardness_k: int


@dataclass(frozen=True)
class ComplexityConfig:
    """The descriptors of every region and class (`src.complexity`).

    Attributes:
        top_k_clusters: The nearest rivals each population is compared
            with.
        max_sample_per_region: The rows measured per region when the
            graph samples train.
        max_sample_per_class: The rows measured per class when the
            graph samples train.
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
class RegressorConfig:
    """The failure regressor (`src.regressor`).

    Attributes:
        min_eval_support: The fewest test rows of a region scored by
            any predictor.
        n_outer_splits: The outer CV folds.
        n_inner_splits: The inner CV folds of the hyper-parameter
            search.
        n_iter: The RandomizedSearchCV draws per fit.
        params: The fixed RandomForestRegressor params.
        param_grid: The searched RandomForestRegressor params.
        measures: The complexity measures used; None uses all.
        signals: The baselines whose estimates the forest reads next
            to the measures; empty for the measures alone.
    """

    min_eval_support: int
    n_outer_splits: int
    n_inner_splits: int
    n_iter: int
    params: Mapping[str, Any]
    param_grid: Mapping[str, list[Any]]
    measures: tuple[str, ...] | None
    signals: tuple[str, ...]


@dataclass(frozen=True)
class ErrorModelConfig:
    """The meta baseline, a model of the errors (`src.baselines`).

    Attributes:
        params: The kwargs of its classifier, baselines.ERROR_MODEL.
        max_samples: The val rows it is fitted on, a uniform sample
            above this.
    """

    params: Mapping[str, Any]
    max_samples: int


@dataclass(frozen=True)
class AblationConfig:
    """The ablation of geometry and signals (`src.ablation`).

    Attributes:
        n_bootstrap: The draws of each paired bootstrap.
    """

    n_bootstrap: int


@dataclass(frozen=True)
class MethodConfig:
    """Parameters of the method, shared by every dataset.

    Attributes:
        distance: One of DISTANCES.
        space: The region space.
        graph: The k-NN graph of train.
        clustering: The per-class clustering.
        complexity: The descriptors.
        regressor: The failure regressor.
        error_model: The meta baseline.
        ablation: The ablation of geometry and signals.
    """

    distance: str
    space: SpaceConfig
    graph: GraphConfig
    clustering: ClusteringConfig
    complexity: ComplexityConfig
    regressor: RegressorConfig
    error_model: ErrorModelConfig
    ablation: AblationConfig


@dataclass(frozen=True)
class DatasetConfig:
    """A dataset, its method and, for the model stages, its model.

    Attributes:
        name: The dataset name.
        path: The raw CSV.
        label_col: The class column.
        num_cols: The numeric columns.
        cat_cols: The categorical columns.
        split: The split shares.
        min_class_count: Classes with fewer rows are dropped.
        max_per_class: The most train rows per class the classifier is
            fitted on; None fits it on all of train.
        top_n: The most frequent values of a categorical column with
            their own code in the space and their own one-hot column in
            the classifier.
        hash_buckets: The codes the other categorical values are
            hashed into in the space.
        seed: The seed of every stage.
        output_dir: The folder of every file of every stage.
        db_path: The SQLite file.
        method: The method.
        model: The classifier; None for the stages without one.
    """

    name: str
    path: Path
    label_col: str
    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]
    split: SplitConfig
    min_class_count: int
    max_per_class: int | None
    top_n: int
    hash_buckets: int
    seed: int
    output_dir: Path
    db_path: Path
    method: MethodConfig
    model: ModelConfig | None = None


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


def model_names() -> list[str]:
    """Return the names of the model configs, sorted."""
    return _names(MODEL_DIR)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_keys(raw: Mapping[str, Any], config: type, path: Path) -> None:
    known = {field.name for field in fields(config)} - {"method", "model"}
    unknown = sorted(set(raw) - known)
    _check(not unknown, f"unknown keys in {path}: {unknown}")


def _validate_split(split: SplitConfig) -> None:
    shares = {"train": split.train, "val": split.val, "test": split.test}
    for name, share in shares.items():
        _check(0 < share < 1, f"split.{name} must be in (0, 1), got {share}")
    total = sum(shares.values())
    _check(math.isclose(total, 1.0), f"split shares must sum to 1, got {total}")


def _validate_method(method: MethodConfig) -> None:
    _check(
        method.distance in DISTANCES,
        f"distance must be one of {DISTANCES}, got {method.distance!r}",
    )
    clustering = method.clustering
    _check(
        clustering.name in ALGORITHMS,
        f"clustering.name must be one of {sorted(ALGORITHMS)}, "
        f"got {clustering.name!r}",
    )
    allowed = tunable_params(ALGORITHMS[clustering.name])
    unknown = sorted(set(clustering.params) - set(allowed))
    _check(
        not unknown,
        f"clustering.params of {clustering.name} must be among {list(allowed)}, "
        f"got {unknown}",
    )
    measures = method.regressor.measures
    _check(
        measures is None or len(measures) > 0,
        "regressor.measures must be null or a non-empty list, got []",
    )
    unknown_signals = sorted(set(method.regressor.signals) - set(BASELINES))
    _check(
        not unknown_signals,
        f"regressor.signals must be among {list(BASELINES)}, got {unknown_signals}",
    )
    positive = {
        "graph.k": method.graph.k,
        "graph.max_samples": method.graph.max_samples,
        "clustering.max_fit_samples": clustering.max_fit_samples,
        "clustering.max_regions": clustering.max_regions,
        "clustering.min_cluster_floor": clustering.min_cluster_floor,
        "clustering.hardness_k": clustering.hardness_k,
        "complexity.top_k_clusters": method.complexity.top_k_clusters,
        "regressor.n_iter": method.regressor.n_iter,
        "error_model.max_samples": method.error_model.max_samples,
        "ablation.n_bootstrap": method.ablation.n_bootstrap,
    }
    for key, value in positive.items():
        _check(value >= 1, f"{key} must be >= 1, got {value}")
    for key in ("n_outer_splits", "n_inner_splits"):
        value = getattr(method.regressor, key)
        _check(value >= 2, f"regressor.{key} must be >= 2, got {value}")
    _check(
        method.regressor.min_eval_support >= 1,
        "regressor.min_eval_support must be >= 1, "
        f"got {method.regressor.min_eval_support}",
    )


def load_model_config(name_or_path: str) -> ModelConfig:
    """Load a model config.

    Args:
        name_or_path: A name of `configs/models` or the path of a yaml.

    Returns:
        The model config.

    Raises:
        ValueError: When its name is not a key of `src.models.MODELS`.
    """
    raw = _yaml(_path(name_or_path, MODEL_DIR))
    _check(
        raw["name"] in MODELS,
        f"model name must be one of {sorted(MODELS)}, got {raw['name']!r}",
    )
    return ModelConfig(name=raw["name"], params=raw.get("params") or {})


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
    regressor = dict(raw["regressor"])
    if regressor["measures"] is not None:
        regressor["measures"] = tuple(regressor["measures"])
    regressor["signals"] = tuple(regressor["signals"] or ())
    clustering = dict(raw["clustering"])
    clustering["params"] = clustering.get("params") or {}
    method = MethodConfig(
        distance=raw["distance"],
        space=SpaceConfig(**raw["space"]),
        graph=GraphConfig(**raw["graph"]),
        clustering=ClusteringConfig(**clustering),
        complexity=ComplexityConfig(**raw["complexity"]),
        regressor=RegressorConfig(**regressor),
        error_model=ErrorModelConfig(
            params=raw["error_model"]["params"] or {},
            max_samples=raw["error_model"]["max_samples"],
        ),
        ablation=AblationConfig(**raw["ablation"]),
    )
    _validate_method(method)
    return method


def load_config(
    dataset: str, model: str | None = None, method: str = "default"
) -> DatasetConfig:
    """Load a dataset config with its method and, optionally, its model.

    Args:
        dataset: A name of `configs/datasets` or the path of a yaml.
        model: A name of `configs/models` or the path of a yaml; None
            for the stages that read no classifier.
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
    split = SplitConfig(**raw["split"])
    _validate_split(split)
    max_per_class = raw["max_per_class"]
    _check(
        max_per_class is None or max_per_class >= 1,
        f"max_per_class must be null or >= 1, got {max_per_class}",
    )
    return DatasetConfig(
        name=raw["name"],
        path=Path(raw["path"]),
        label_col=raw["label_col"],
        num_cols=tuple(raw["num_cols"] or ()),
        cat_cols=tuple(raw["cat_cols"] or ()),
        split=split,
        min_class_count=raw["min_class_count"],
        max_per_class=max_per_class,
        top_n=raw["top_n"],
        hash_buckets=raw["hash_buckets"],
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        db_path=Path(raw["db_path"]),
        method=load_method_config(method),
        model=None if model is None else load_model_config(model),
    )
