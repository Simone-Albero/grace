"""Stage regions: classes of train clustered into regions."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3

import joblib
import numpy as np
import polars as pl

from src import db
from src.clustering import (
    ALGORITHMS,
    assign_nearest,
    centroids,
    cluster_class,
    kdn,
)
from src.config import DatasetConfig
from src.data import LABEL, SPLITS
from src.graph import build_graph, nearest_neighbors, sample_rows
from src.preprocessing import feature_cols
from src.space import embed, fit_space
from stages.paths import (
    assignments_file,
    graph_file,
    read_classes,
    read_split,
    regions_file,
    space_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Cluster each class into regions and route val and test to them.

    Reads the split files; the labels of val and test are never read.
    Order: space.fit_space on train (the columns of train the config
    names, top_n, hash_buckets, method.space, distance) -> points of
    every split (space.embed) -> graph.build_graph on
    graph.sample_rows(n_train, graph.max_samples, seed) -> per class,
    ascending: its kDN on its graph.sample_rows(n_class,
    max_fit_samples, seed) rows (hardness_k nearest graph nodes, never
    itself) -> clustering.cluster_class with max_clusters = max(2,
    max_regions // n_classes) and eval_ratio = n_test / n_train ->
    region id = label + offset, offset += max label + 1 -> centroids of
    the class's points -> val and test rows to the nearest region
    centroid of any class (clustering.assign_nearest); train rows keep
    their own region.

    Writes to output_dir: regions.parquet (region, class, n_train,
    n_val, n_test, centroid as a list of floats), assignments.parquet
    (split, row, region for every row of every split), space.joblib,
    graph.npz (rows, points, knn_idx, knn_dist, mst).
    Logs: regions (region, class name, n_train, n_val, n_test),
    class_clusterings (class, n_train, n_regions, loss, params as JSON),
    granularity_candidates (class, params as JSON, n_clusters, loss),
    region_stats (one row: n_regions, algorithm, distance,
    n_graph_nodes, eval_ratio).
    """
    method, clustering = cfg.method, cfg.method.clustering
    splits = {name: read_split(cfg, name) for name in SPLITS}
    labels = splits["train"][LABEL].to_numpy()
    print("[regions] fitting the space on train", flush=True)
    features = feature_cols(splits["train"].columns, cfg.num_cols, cfg.cat_cols)
    space = fit_space(
        splits["train"],
        features.num,
        features.cat,
        top_n=cfg.top_n,
        hash_buckets=cfg.hash_buckets,
        top_k=method.space.top_k,
        cat_cost=method.space.cat_cost,
        metric=method.distance,
    )
    points = {name: embed(space, frame) for name, frame in splits.items()}
    n_train = len(labels)
    graph_rows = sample_rows(n_train, method.graph.max_samples, cfg.seed)
    print(f"[regions] graph of {len(graph_rows)} train rows", flush=True)
    graph = build_graph(points["train"][graph_rows], graph_rows, method.graph.k)
    graph_labels = labels[graph_rows]

    classes = read_classes(cfg)
    algorithm = ALGORITHMS[clustering.name]
    max_clusters = max(2, clustering.max_regions // len(classes))
    eval_ratio = splits["test"].height / n_train
    region = np.empty(n_train, dtype=np.int64)
    class_rows, candidate_rows = [], []
    offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        X = points["train"][rows]
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
            eval_ratio=eval_ratio,
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
                "n_train": len(rows),
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

    ids, centers = centroids(points["train"], region, method.distance)
    region_class = labels[[np.flatnonzero(region == i)[0] for i in ids]]
    assigned = {"train": region}
    for name in ("val", "test"):
        assigned[name] = ids[assign_nearest(points[name], centers, method.distance)]
    sizes = {
        name: np.array([(assigned[name] == i).sum() for i in ids]) for name in SPLITS
    }
    regions = pl.DataFrame(
        {
            "region": ids,
            "class": region_class,
            **{f"n_{name}": sizes[name] for name in SPLITS},
            "centroid": centers.tolist(),
        }
    )
    assignments = pl.concat(
        [
            pl.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(assigned[name])),
                    "region": assigned[name],
                }
            )
            for name in SPLITS
        ]
    )
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
                **{f"n_{name}": int(sizes[name][k]) for name in SPLITS},
            }
            for k, (region_id, code) in enumerate(zip(ids, region_class))
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
                "eval_ratio": eval_ratio,
            }
        ],
    )
