"""F/N/ND/T/G complexity descriptors of each region and class, on train.

DB rows written: `complexity` (one row per scope, id and measure) and
metric n_measures.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.clustering import compute_centroids
from src.complexity.common import MeasuredSample
from src.complexity.measure import compute_population_complexity
from src.graph import TrainGraph, nearest_neighbors
from src.grouping import RowsBy
from src.space import Space, scale_for_metric
from stages import LABEL, Context

logger = logging.getLogger(__name__)

CONFIG = ("complexity", "graph.k", "distance", "seed")
INPUTS = ("prepare", "regions")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()


def _draw_sample(
    cfg: DictConfig,
    *,
    space: Space,
    train: pd.DataFrame,
    population: np.ndarray,
    train_graph: TrainGraph,
    graph_rows: np.ndarray,
    cap: int | None,
    n_jobs: int,
) -> MeasuredSample:
    """Draw the rows to measure and find their nearest graph nodes.

    Returns:
        Up to `cap` rows of every population, in train order, with
        their nearest graph nodes.
    """
    by_population = RowsBy(population)
    rng = np.random.default_rng(cfg.seed)
    picked = []
    for i in range(len(by_population.ids)):
        members = by_population.members(i)
        if cap is not None and len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picked.append(members)
    rows = np.sort(np.concatenate(picked))
    if np.array_equal(rows, graph_rows):
        return MeasuredSample(
            train_graph.points,
            population[rows],
            train_graph.knn_idx,
            train_graph.knn_dist,
        )
    points = space.embed(train.iloc[rows])
    neighbors, neighbor_dist = nearest_neighbors(
        train_graph.points,
        graph_rows,
        points,
        rows,
        k=cfg.graph.k,
        metric=cfg.distance,
        n_jobs=n_jobs,
    )
    return MeasuredSample(points, population[rows], neighbors, neighbor_dist)


def _measure(
    cfg: DictConfig,
    *,
    train_graph: TrainGraph,
    graph_rows: np.ndarray,
    population: np.ndarray,
    sample: MeasuredSample,
    population_class: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
) -> dict[int, dict[str, float | None]]:
    """Return the complexity measures of every population, by id."""
    ids, counts = np.unique(population, return_counts=True)
    return compute_population_complexity(
        train_graph,
        population[graph_rows],
        sample,
        population_class=population_class,
        centroids=centroids,
        sizes=dict(zip(ids.tolist(), counts.tolist())),
        top_k_clusters=cfg.complexity.top_k_clusters,
        metric=cfg.distance,
        silhouette_max_samples=cfg.complexity.silhouette_max_samples,
        silhouette_min_per_cluster=cfg.complexity.silhouette_min_per_cluster,
        random_state=cfg.seed,
    )


def _long_rows(
    scope: str, measures: Mapping[int, Mapping[str, float | None]]
) -> pd.DataFrame:
    """Return one row per (scope id, measure).

    An undefined measure gives a NULL value.
    """
    rows = []
    for scope_id, row in measures.items():
        for measure, value in row.items():
            rows.append(
                {
                    "scope": scope,
                    "scope_id": scope_id,
                    "measure": measure,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _load_train_graph(
    regions_dir: Path, space: Space, train: pd.DataFrame
) -> tuple[TrainGraph, np.ndarray]:
    """Load the train graph of regions and the train rows it covers."""
    with np.load(regions_dir / "graph.npz") as stored:
        graph_arrays = {name: stored[name] for name in stored.files}
    graph_rows = graph_arrays["rows"]
    train_graph = TrainGraph(
        space.embed(train.iloc[graph_rows]),
        graph_arrays["knn_idx"],
        graph_arrays["knn_dist"],
        graph_arrays["mst"],
    )
    return train_graph, graph_rows


def run(ctx: Context) -> None:
    """Measure every region and class of the train split.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    regions_dir = ctx.inputs["regions"].dir
    space = joblib.load(regions_dir / "space.joblib")
    train = pd.read_parquet(ctx.inputs["prepare"].dir / "train.parquet")
    train_graph, graph_rows = _load_train_graph(regions_dir, space, train)
    assigned = ctx.split_rows("regions", "assignments.parquet", "train", ["region_id"])
    region = assigned["region_id"].to_numpy(dtype=np.int64)
    label = train[LABEL].to_numpy(dtype=np.int64)

    graph_is_sample = len(graph_rows) < len(train)
    centroids = pd.read_parquet(regions_dir / "centroids.parquet")
    region_class = dict(zip(centroids["region_id"], centroids["class_id"]))
    coordinates = centroids.drop(columns=["region_id", "class_id"]).to_numpy()

    region_sample = _draw_sample(
        cfg,
        space=space,
        train=train,
        population=region,
        train_graph=train_graph,
        graph_rows=graph_rows,
        cap=cfg.complexity.max_sample_per_region if graph_is_sample else None,
        n_jobs=ctx.runtime.n_jobs,
    )
    region_measures = _measure(
        cfg,
        train_graph=train_graph,
        graph_rows=graph_rows,
        population=region,
        sample=region_sample,
        population_class=region_class,
        # Exact centroids: the ones regions routes by, not a sample's.
        centroids=dict(zip(centroids["region_id"], coordinates)),
    )

    label_sample = _draw_sample(
        cfg,
        space=space,
        train=train,
        population=label,
        train_graph=train_graph,
        graph_rows=graph_rows,
        cap=cfg.complexity.max_sample_per_class if graph_is_sample else None,
        n_jobs=ctx.runtime.n_jobs,
    )
    class_measures = _measure(
        cfg,
        train_graph=train_graph,
        graph_rows=graph_rows,
        population=label,
        sample=label_sample,
        population_class={
            int(class_id): int(class_id) for class_id in np.unique(label)
        },
        centroids=compute_centroids(
            scale_for_metric(label_sample.X, cfg.distance),
            label_sample.population,
            metric=cfg.distance,
        ),
    )
    rows = pd.concat(
        [_long_rows("region", region_measures), _long_rows("class", class_measures)],
        ignore_index=True,
    )
    ctx.write("complexity", rows)
    ctx.save_metrics({"n_measures": rows["measure"].nunique()})
