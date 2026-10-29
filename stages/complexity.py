"""F/N/ND/T/G complexity descriptors of each leaf and class of the tree.

The rows are the val rows the tree grew on, with their true labels. A
leaf holds rows of several classes, so it is measured as its
(leaf, class) populations, each against the nearest populations of
another class, and a leaf's descriptors are the means of its
populations' measures, weighted by their rows. A population under
`complexity.min_population` rows is left out and is nobody's rival.
DB rows written: `complexity` (one row per scope, id and measure) and
metrics n_measures, n_populations, n_unmeasured_rows and n_graph_rows.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.complexity.common import (
    MeasuredSample,
    compute_centroids,
    weighted_group_measures,
)
from src.complexity.measure import compute_population_complexity
from src.graph import RowGraph, build_row_graph, nearest_neighbors, subsample_indices
from src.grouping import RowsBy
from src.space import Space
from stages import LABEL, Context

logger = logging.getLogger(__name__)

CONFIG = ("complexity", "graph", "seed")
INPUTS = ("prepare", "space", "tree")
FILES = ()
# The stage writes DB rows only.
OUTPUTS = ()
_CENTROID_ROWS = 500_000


def _draw_sample(
    cfg: DictConfig,
    *,
    space: Space,
    grown: pd.DataFrame,
    population: np.ndarray,
    graph: RowGraph,
    graph_rows: np.ndarray,
    cap: int | None,
    n_jobs: int,
) -> MeasuredSample:
    """Draw the rows to measure and find their nearest graph nodes.

    Returns:
        Up to `cap` rows of every population but -1 (the unmeasured),
        in row order, with their nearest graph nodes.
    """
    by_population = RowsBy(population)
    rng = np.random.default_rng(cfg.seed)
    picked = []
    for i, population_id in enumerate(by_population.ids):
        if population_id < 0:
            continue
        members = by_population.members(i)
        if cap is not None and len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picked.append(members)
    rows = np.sort(np.concatenate(picked))
    is_measured_node = population[graph_rows] >= 0
    if np.array_equal(rows, graph_rows[is_measured_node]):
        return MeasuredSample(
            graph.points[is_measured_node],
            population[rows],
            graph.knn_idx[is_measured_node],
            graph.knn_dist[is_measured_node],
        )
    points = space.embed(grown.iloc[rows])
    neighbors, neighbor_dist = nearest_neighbors(
        graph.points,
        graph_rows,
        points,
        rows,
        k=cfg.graph.k,
        n_jobs=n_jobs,
    )
    return MeasuredSample(points, population[rows], neighbors, neighbor_dist)


def _measure(
    cfg: DictConfig,
    *,
    graph: RowGraph,
    graph_rows: np.ndarray,
    population: np.ndarray,
    sample: MeasuredSample,
    population_class: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
) -> dict[int, dict[str, float | None]]:
    """Return the complexity measures of every population, by id."""
    ids, counts = np.unique(population[population >= 0], return_counts=True)
    return compute_population_complexity(
        graph,
        population[graph_rows],
        sample,
        population_class=population_class,
        centroids=centroids,
        sizes=dict(zip(ids.tolist(), counts.tolist())),
        top_k_clusters=cfg.complexity.top_k_clusters,
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


def _build_graph(
    cfg: DictConfig, space: Space, grown: pd.DataFrame, *, n_jobs: int
) -> tuple[RowGraph, np.ndarray]:
    """Return a graph of grown rows, sampled uniformly, and its rows."""
    graph_rows = np.sort(
        subsample_indices(
            len(grown), max_samples=cfg.graph.max_samples, random_state=cfg.seed
        )
    )
    graph = build_row_graph(
        space.embed(grown.iloc[graph_rows]), k=cfg.graph.k, n_jobs=n_jobs
    )
    return graph, graph_rows


def _compute_population_centroids(
    space: Space, grown: pd.DataFrame, population: np.ndarray
) -> dict[int, np.ndarray]:
    """Return the exact centroid of every measured population.

    The rows are embedded in chunks, as a whole they may not fit in
    memory.
    """
    sums = None
    for start in range(0, len(grown), _CENTROID_ROWS):
        chunk_population = population[start : start + _CENTROID_ROWS]
        is_measured = chunk_population >= 0
        points = space.embed(
            grown.iloc[start : start + _CENTROID_ROWS][is_measured]
        ).astype(np.float64)
        chunk_sums = pd.DataFrame(points).groupby(chunk_population[is_measured]).sum()
        sums = chunk_sums if sums is None else sums.add(chunk_sums, fill_value=0.0)
    ids, counts = np.unique(population[population >= 0], return_counts=True)
    means = sums.loc[ids].to_numpy() / counts[:, None]
    return dict(zip(ids.tolist(), means))


def run(ctx: Context) -> None:
    """Measure every leaf and class of the grown half of val.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    space = joblib.load(ctx.inputs["space"].dir / "space.joblib")
    val = pd.read_parquet(ctx.inputs["prepare"].dir / "val.parquet")
    tree_rows = ctx.split_rows("tree", "rows.parquet", "val", ["grow", "leaf"])
    is_grow = tree_rows["grow"].to_numpy()
    grown = val[is_grow].reset_index(drop=True)
    graph, graph_rows = _build_graph(cfg, space, grown, n_jobs=ctx.runtime.n_jobs)
    leaf = tree_rows["leaf"].to_numpy()[is_grow].astype(np.int64)
    label = grown[LABEL].to_numpy(dtype=np.int64)
    n_classes = int(label.max()) + 1

    # One id per (leaf, class); a thin population is -1: its rows stay
    # nodes of the graph but are neither measured nor anyone's rival.
    code = leaf * n_classes + label
    code_ids, code_sizes = np.unique(code, return_counts=True)
    thin_ids = code_ids[code_sizes < cfg.complexity.min_population]
    population = np.where(np.isin(code, thin_ids), -1, code)
    population_class = {
        int(population_id): int(population_id) % n_classes
        for population_id in code_ids
        if population_id not in thin_ids
    }
    if not population_class:
        raise ValueError(
            "No (leaf, class) population has complexity.min_population="
            f"{cfg.complexity.min_population} rows: lower it or grow "
            "bigger leaves with a smaller tree.n_min."
        )

    graph_is_sample = len(graph_rows) < len(grown)
    population_sample = _draw_sample(
        cfg,
        space=space,
        grown=grown,
        population=population,
        graph=graph,
        graph_rows=graph_rows,
        cap=cfg.complexity.max_sample_per_population if graph_is_sample else None,
        n_jobs=ctx.runtime.n_jobs,
    )
    population_measures = _measure(
        cfg,
        graph=graph,
        graph_rows=graph_rows,
        population=population,
        sample=population_sample,
        population_class=population_class,
        centroids=_compute_population_centroids(space, grown, population),
    )
    leaf_measures = weighted_group_measures(
        population_measures,
        group={
            population_id: population_id // n_classes
            for population_id in population_class
        },
        weight=dict(zip(code_ids.tolist(), code_sizes.tolist())),
    )

    label_sample = _draw_sample(
        cfg,
        space=space,
        grown=grown,
        population=label,
        graph=graph,
        graph_rows=graph_rows,
        cap=cfg.complexity.max_sample_per_class if graph_is_sample else None,
        n_jobs=ctx.runtime.n_jobs,
    )
    class_measures = _measure(
        cfg,
        graph=graph,
        graph_rows=graph_rows,
        population=label,
        sample=label_sample,
        population_class={
            int(class_id): int(class_id) for class_id in np.unique(label)
        },
        centroids=compute_centroids(label_sample.X, label_sample.population),
    )
    rows = pd.concat(
        [_long_rows("leaf", leaf_measures), _long_rows("class", class_measures)],
        ignore_index=True,
    )
    ctx.write("complexity", rows)
    ctx.save_metrics(
        {
            "n_measures": rows["measure"].nunique(),
            "n_populations": len(population_class),
            "n_unmeasured_rows": int((population < 0).sum()),
            "n_graph_rows": len(graph_rows),
        }
    )
