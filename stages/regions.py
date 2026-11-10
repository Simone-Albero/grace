"""Stage regions: each class of the data clustered into regions."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.clustering import ALGORITHMS, centroids, cluster_class, kdn
from src.config import DatasetConfig
from src.data import LABEL, feature_cols
from src.graph import build_graph, nearest_neighbors, sample_rows
from src.space import embed, fit_space
from stages.paths import (
    assignments_file,
    graph_file,
    read_classes,
    read_data,
    regions_file,
    space_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Cluster each class into regions.

    Reads the data file. Order: space.fit_space on the data (the columns
    the config names, top_n, hash_buckets, method.space, distance) ->
    points (space.embed) -> graph.build_graph on
    graph.sample_rows(n_rows, graph.max_samples, seed) -> per class,
    ascending: its kDN on its graph.sample_rows(n_class,
    max_fit_samples, seed) rows (hardness_k nearest graph nodes, never
    itself) -> clustering.cluster_class with max_clusters = max(2,
    max_regions // n_classes) -> region id = label + offset, offset +=
    max label + 1 -> centroids of the class's points.

    Writes to output_dir: regions.parquet (region, class, n_rows,
    centroid as a list of floats), assignments.parquet (row, region for
    every row), space.joblib, graph.npz (rows, points, knn_idx,
    knn_dist, mst).
    Logs: regions (region, class name, n_rows), class_clusterings
    (class, n_rows, n_regions, loss, params as JSON),
    granularity_candidates (class, params as JSON, n_clusters, loss),
    region_stats (one row: n_regions, algorithm, distance,
    n_graph_nodes).
    """
    method, clustering = cfg.method, cfg.method.clustering
    data = read_data(cfg)
    labels = data[LABEL].to_numpy()
    print("[regions] fitting the space", flush=True)
    features = feature_cols(data.columns, cfg.num_cols, cfg.cat_cols)
    space = fit_space(
        data,
        features.num,
        features.cat,
        top_n=cfg.top_n,
        hash_buckets=cfg.hash_buckets,
        top_k=method.space.top_k,
        cat_cost=method.space.cat_cost,
        metric=method.distance,
    )
    points = embed(space, data)
    n_rows = len(labels)
    graph_rows = sample_rows(n_rows, method.graph.max_samples, cfg.seed)
    print(f"[regions] graph of {len(graph_rows)} rows", flush=True)
    graph = build_graph(points[graph_rows], graph_rows, method.graph.k)
    graph_labels = labels[graph_rows]

    classes = read_classes(cfg)
    algorithm = ALGORITHMS[clustering.name]
    max_clusters = max(2, clustering.max_regions // len(classes))
    region = np.empty(n_rows, dtype=np.int64)
    class_rows, candidate_rows = [], []
    offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        X = points[rows]
        scored = sample_rows(len(rows), clustering.max_fit_samples, cfg.seed)
        neighbors, _ = nearest_neighbors(
            graph.points, graph.rows, X[scored], rows[scored], clustering.hardness_k
        )
        chosen = cluster_class(
            X,
            scored,
            kdn(graph_labels[neighbors], code),
            algorithm=algorithm,
            params=clustering.params,
            max_fit_samples=clustering.max_fit_samples,
            min_cluster_floor=clustering.min_cluster_floor,
            max_clusters=max_clusters,
            metric=method.distance,
            seed=cfg.seed,
        )
        n_regions = len(np.unique(chosen.labels))
        if n_regions > max_clusters:
            raise ValueError(
                f"class {name}: {n_regions} regions after the refit, "
                f"more than max_clusters={max_clusters}"
            )
        region[rows] = chosen.labels + offset
        offset += int(chosen.labels.max()) + 1
        print(
            f"[regions] {name}: {n_regions} regions, loss {chosen.loss:.4g}", flush=True
        )
        class_rows.append(
            {
                "class": name,
                "n_rows": len(rows),
                "n_regions": n_regions,
                "loss": chosen.loss,
                "params": json.dumps(chosen.params),
            }
        )
        candidate_rows += [
            {
                "class": name,
                "params": json.dumps(candidate.params),
                "n_clusters": candidate.n_clusters,
                "loss": candidate.loss,
            }
            for candidate in chosen.candidates
        ]

    ids, centers = centroids(points, region, method.distance)
    region_class = labels[[np.flatnonzero(region == i)[0] for i in ids]]
    sizes = np.array([(region == i).sum() for i in ids])
    regions = pl.DataFrame(
        {
            "region": ids,
            "class": region_class,
            "n_rows": sizes,
            "centroid": centers.tolist(),
        }
    )
    assignments = pl.DataFrame({"row": np.arange(n_rows), "region": region})
    regions.write_parquet(regions_file(cfg))
    assignments.write_parquet(assignments_file(cfg))
    joblib.dump(space, space_file(cfg))
    np.savez(
        graph_file(cfg),
        rows=graph.rows,
        points=graph.points,
        knn_idx=graph.knn_idx,
        knn_dist=graph.knn_dist,
        mst=graph.mst,
    )
    print(f"[regions] saved {len(ids)} regions", flush=True)

    db.log_table(
        conn,
        run_id,
        "regions",
        [
            {
                "region": int(region_id),
                "class": classes[code],
                "n_rows": int(size),
            }
            for region_id, code, size in zip(ids, region_class, sizes)
        ],
    )
    db.log_table(conn, run_id, "class_clusterings", class_rows)
    db.log_table(conn, run_id, "granularity_candidates", candidate_rows)
    db.log_table(
        conn,
        run_id,
        "region_stats",
        [
            {
                "n_regions": len(ids),
                "algorithm": clustering.name,
                "distance": method.distance,
                "n_graph_nodes": len(graph_rows),
            }
        ],
    )
