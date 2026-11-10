"""Stage regions: each class of the data split into regions."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import numpy as np
import polars as pl

from src import db
from src.clustering import RegionTree, centroids, cluster_class, kdn
from src.config import DatasetConfig
from src.graph import build_graph, nearest_neighbors, sample_rows
from stages.paths import (
    assignments_file,
    graph_file,
    read_classes,
    read_points,
    read_weights,
    regions_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Split each class into regions.

    Reads the points of the prepared file (paths.read_points: the
    numbers scaled for distance, the categorical slots, not MISSING;
    every distance from here on is euclidean on them) and the weight of
    every row (paths.read_weights: its copies in the CSV). Order:
    graph.build_graph on graph.sample_rows(n_rows, graph.max_samples,
    seed) -> per class, ascending: its kDN on its
    graph.sample_rows(n_class, max_fit_samples, seed) rows (hardness_k
    nearest graph nodes, never itself) -> clustering.cluster_class with
    max_clusters = max(1, max_regions // n_classes) -> region id =
    label + offset, offset += the class's regions; tree node id = node +
    node offset -> weighted centroids of the regions.

    Writes to output_dir: regions.parquet (region, class, n_rows,
    n_read, centroid as a list of floats), assignments.parquet (row,
    region for every row), graph.npz (rows, points, knn_idx, knn_dist,
    mst).
    Logs: regions (region, class name, n_rows, n_read, hardness_mean and
    hardness_std: the mean and standard deviation of the kDN of its
    scored rows, weighted by their copies), region_tree
    (node, class, parent (-1 at a root), depth (0 at a root), region
    (null at an internal node), n_rows and n_read of the regions below;
    nodes in pre-order), class_clusterings (class, n_rows, n_read,
    n_regions, n_voted, penalty, loss), granularity_candidates (class,
    penalty, n_clusters, loss), region_stats (one row: n_regions,
    distance, n_graph_nodes). n_rows counts distinct rows, n_read rows
    of the CSV (the summed weights).
    """
    method, clustering = cfg.method, cfg.method.clustering
    points, labels = read_points(cfg)
    weights = read_weights(cfg)
    n_rows = len(labels)
    graph_rows = sample_rows(n_rows, method.graph.max_samples, cfg.seed)
    print(f"[regions] graph of {len(graph_rows)} rows", flush=True)
    graph = build_graph(points[graph_rows], graph_rows, method.graph.k)
    graph_labels = labels[graph_rows]

    classes = read_classes(cfg)
    max_clusters = max(1, clustering.max_regions // len(classes))
    region = np.empty(n_rows, dtype=np.int64)
    class_rows, candidate_rows, tree_rows = [], [], []
    hardness_mean, hardness_std = [], []
    offset = node_offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        X, w = points[rows], weights[rows]
        scored = sample_rows(len(rows), clustering.max_fit_samples, cfg.seed)
        neighbors, _ = nearest_neighbors(
            graph.points, graph.rows, X[scored], rows[scored], clustering.hardness_k
        )
        hardness = kdn(graph_labels[neighbors], code)
        chosen = cluster_class(
            X,
            w,
            scored,
            hardness,
            min_cluster_floor=clustering.min_cluster_floor,
            max_clusters=max_clusters,
            vote_k=clustering.vote_k,
        )
        n_regions = int(chosen.labels.max()) + 1
        region[rows] = chosen.labels + offset
        mean, std = _hardness_stats(chosen.labels[scored], hardness, w[scored])
        hardness_mean += mean.tolist()
        hardness_std += std.tolist()
        tree_rows += _tree_rows(
            chosen.tree, chosen.labels, w, name, offset, node_offset
        )
        offset += n_regions
        node_offset += len(chosen.tree.parent)
        print(
            f"[regions] {name}: {n_regions} regions, {chosen.n_voted} rows voted, "
            f"loss {chosen.loss:.4g}",
            flush=True,
        )
        class_rows.append(
            {
                "class": name,
                "n_rows": len(rows),
                "n_read": float(w.sum()),
                "n_regions": n_regions,
                "n_voted": chosen.n_voted,
                "penalty": chosen.penalty,
                "loss": chosen.loss,
            }
        )
        candidate_rows += [
            {
                "class": name,
                "penalty": candidate.penalty,
                "n_clusters": candidate.n_clusters,
                "loss": candidate.loss,
            }
            for candidate in chosen.candidates
        ]

    ids, centers = centroids(points, region, weights)
    region_class = labels[[np.flatnonzero(region == i)[0] for i in ids]]
    sizes = np.bincount(region, minlength=len(ids))
    reads = np.bincount(region, weights=weights, minlength=len(ids))
    regions = pl.DataFrame(
        {
            "region": ids,
            "class": region_class,
            "n_rows": sizes,
            "n_read": reads,
            "centroid": centers.tolist(),
        }
    )
    assignments = pl.DataFrame({"row": np.arange(n_rows), "region": region})
    regions.write_parquet(regions_file(cfg))
    assignments.write_parquet(assignments_file(cfg))
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
                "n_read": float(read),
                "hardness_mean": mean,
                "hardness_std": std,
            }
            for region_id, code, size, read, mean, std in zip(
                ids, region_class, sizes, reads, hardness_mean, hardness_std
            )
        ],
    )
    db.log_table(conn, run_id, "region_tree", tree_rows)
    db.log_table(conn, run_id, "class_clusterings", class_rows)
    db.log_table(conn, run_id, "granularity_candidates", candidate_rows)
    db.log_table(
        conn,
        run_id,
        "region_stats",
        [
            {
                "n_regions": len(ids),
                "distance": method.distance,
                "n_graph_nodes": len(graph_rows),
            }
        ],
    )


def _hardness_stats(
    labels: np.ndarray, hardness: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted mean and standard deviation of the hardness per label.

    Every label 0..max holds a row: the regions are leaves of the tree
    of these rows.
    """
    mass = np.bincount(labels, weights=weights)
    mean = np.bincount(labels, weights=weights * hardness) / mass
    second = np.bincount(labels, weights=weights * hardness**2) / mass
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def _tree_rows(
    tree: RegionTree,
    labels: np.ndarray,
    weights: np.ndarray,
    name: str,
    region_offset: int,
    node_offset: int,
) -> list[dict]:
    """Return the region_tree rows of one class.

    Each node counts the rows and the weights of the regions below it;
    node, parent and region ids are shifted by the offsets.
    """
    n_regions = int(labels.max()) + 1
    n_rows = np.zeros(len(tree.parent))
    n_read = np.zeros(len(tree.parent))
    leaf = tree.region >= 0
    n_rows[leaf] = np.bincount(labels, minlength=n_regions)[tree.region[leaf]]
    n_read[leaf] = np.bincount(labels, weights=weights, minlength=n_regions)[
        tree.region[leaf]
    ]
    # Pre-order: every child comes after its parent, so walking back
    # adds each finished node into its parent.
    for node in range(len(tree.parent) - 1, 0, -1):
        n_rows[tree.parent[node]] += n_rows[node]
        n_read[tree.parent[node]] += n_read[node]
    depth = np.zeros(len(tree.parent), dtype=np.int64)
    for node in range(1, len(tree.parent)):
        depth[node] = depth[tree.parent[node]] + 1
    return [
        {
            "node": node_offset + node,
            "class": name,
            "parent": node_offset + int(parent) if parent >= 0 else -1,
            "depth": int(depth[node]),
            "region": region_offset + int(region) if region >= 0 else None,
            "n_rows": int(n_rows[node]),
            "n_read": float(n_read[node]),
        }
        for node, (parent, region) in enumerate(zip(tree.parent, tree.region))
    ]
