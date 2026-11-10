"""Stage complexity: descriptors of every region and class."""

from __future__ import annotations

__all__ = ["run"]

import math
import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.clustering import centroids
from src.complexity.common import MeasuredSample
from src.complexity.measure import compute_population_complexity
from src.config import DatasetConfig
from src.data import LABEL
from src.graph import Graph, nearest_neighbors
from src.space import embed
from stages.paths import (
    assignments_file,
    complexity_file,
    graph_file,
    need,
    read_data,
    regions_file,
    space_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Describe every region and every class on its rows.

    Reads the data file, regions.parquet, assignments.parquet,
    space.joblib and graph.npz. When the graph samples the rows (fewer
    nodes than rows), a region is measured on at most
    max_sample_per_region of its rows and a class on
    max_sample_per_class (one generator seeded by seed per scope,
    populations in ascending id); otherwise on every row, reusing the
    graph's own neighbours. The measured rows get their graph.k nearest
    graph nodes, never themselves. Region centroids are those of
    regions.parquet; class centroids the mean of the measured rows (unit
    length under cosine). Then measure.compute_population_complexity per
    scope.

    Writes to output_dir: complexity.parquet (scope "region" | "class",
    id, measure, value; null when undefined), one row per population and
    measure of MEASURES.
    Logs: complexity (the same rows).
    """
    method = cfg.method
    data = read_data(cfg)
    space = joblib.load(need(space_file(cfg), "regions"))
    with np.load(need(graph_file(cfg), "regions")) as stored:
        graph = Graph(**{name: stored[name] for name in stored.files})
    regions = pl.read_parquet(need(regions_file(cfg), "regions"))
    assignments = pl.read_parquet(need(assignments_file(cfg), "regions"))
    region = assignments.sort("row")["region"].to_numpy()
    label = data[LABEL].to_numpy()
    region_class = dict(zip(regions["region"].to_list(), regions["class"].to_list()))
    if len(region) != len(label) or any(
        region_class[r] != c for r, c in zip(region.tolist(), label.tolist())
    ):
        raise ValueError("the regions were built on other data: run regions again")
    points = embed(space, data)
    sampled = len(graph.rows) < data.height
    caps = {
        "region": method.complexity.max_sample_per_region if sampled else None,
        "class": method.complexity.max_sample_per_class if sampled else None,
    }
    region_centroids = dict(
        zip(regions["region"].to_list(), np.array(regions["centroid"].to_list()))
    )
    measure_rows = []
    for scope, population in (("region", region), ("class", label)):
        print(f"[complexity] measuring every {scope}", flush=True)
        sample = _draw_sample(cfg, points, population, graph, caps[scope])
        if scope == "region":
            population_class = dict(
                zip(regions["region"].to_list(), regions["class"].to_list())
            )
            centers = region_centroids
        else:
            population_class = {int(code): int(code) for code in np.unique(label)}
            class_ids, class_centroids = centroids(
                sample.X, sample.population, method.distance
            )
            centers = dict(zip(class_ids.tolist(), class_centroids))
        ids, counts = np.unique(population, return_counts=True)
        measures = compute_population_complexity(
            graph,
            population[graph.rows],
            sample,
            population_class=population_class,
            centroids=centers,
            sizes=dict(zip(ids.tolist(), counts.tolist())),
            top_k_clusters=method.complexity.top_k_clusters,
            metric=method.distance,
            silhouette_max_samples=method.complexity.silhouette_max_samples,
            silhouette_min_per_cluster=method.complexity.silhouette_min_per_cluster,
            random_state=cfg.seed,
        )
        for population_id, values in measures.items():
            measure_rows += [
                {
                    "scope": scope,
                    "id": int(population_id),
                    "measure": measure,
                    "value": _number(value),
                }
                for measure, value in values.items()
            ]
    table = pl.DataFrame(
        measure_rows,
        schema={
            "scope": pl.String,
            "id": pl.Int64,
            "measure": pl.String,
            "value": pl.Float64,
        },
    )
    table.write_parquet(complexity_file(cfg))
    print(f"[complexity] saved {table.height} values", flush=True)
    db.log_table(conn, run_id, "complexity", measure_rows)


def _number(value: float | None) -> float | None:
    """Return value as a float, None when undefined or not finite."""
    return None if value is None or not math.isfinite(value) else float(value)


def _draw_sample(
    cfg: DatasetConfig,
    points: np.ndarray,
    population: np.ndarray,
    graph: Graph,
    cap: int | None,
) -> MeasuredSample:
    """Draw at most cap rows per population, with their graph nodes."""
    rng = np.random.default_rng(cfg.seed)
    picked = []
    for population_id in np.unique(population):
        members = np.flatnonzero(population == population_id)
        if cap is not None and len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picked.append(members)
    rows = np.sort(np.concatenate(picked))
    if np.array_equal(rows, graph.rows):
        return MeasuredSample(
            graph.points, population[rows], graph.knn_idx, graph.knn_dist
        )
    neighbors, distances = nearest_neighbors(
        graph.points, graph.rows, points[rows], rows, cfg.method.graph.k
    )
    return MeasuredSample(points[rows], population[rows], neighbors, distances)
