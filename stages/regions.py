"""Stage regions: each class of the data split into regions."""

from __future__ import annotations

__all__ = ["run"]

import sqlite3

import numpy as np
import polars as pl

from src import db
from src.clustering import (
    ClassClustering,
    OverlapZones,
    RegionTree,
    centroids,
    cluster_class,
    line_spread,
    overlap_pairs,
    overlap_zones,
    region_stability,
    regions_tree,
    separation,
)
from src.config import DatasetConfig
from stages.paths import (
    assignments_file,
    prepared_columns,
    read_classes,
    read_points,
    read_weights,
    regions_file,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Split each class into regions.

    Reads the points of the prepared file (paths.read_points: the
    numbers scaled for distance, the categorical slots, not MISSING;
    every distance from here on is euclidean on them), which of their
    columns are numeric (paths.prepared_columns against num_cols) and
    the weight of every row (paths.read_weights: its copies in the CSV).
    Per class, ascending: clustering.cluster_class with seed, max_clusters
    = max(1, max_regions // n_classes) and split_separation ->
    clustering.regions_tree (the regions renumbered in its pre-order) ->
    stability_repeats times cluster_class again with seed + i on the
    rows of the first halves (sample), clustering.region_stability of
    each region against each repeat, averaged (1 when there is no
    repeat) -> region id = label + offset, offset += the class's
    regions; tree node id = node + node offset. Then, over every
    region: weighted centroids -> clustering.line_spread (weighted by
    its CSV rows) -> clustering.separation of every pair ->
    clustering.overlap_zones and clustering.overlap_pairs under
    overlap_separation.

    Writes to output_dir: regions.parquet (region, class, n_rows,
    n_read, centroid as a list of floats), assignments.parquet (row,
    region for every row).
    Logs: regions (region, class name, n_rows, n_read, stability; null
    when stability_repeats is 0), region_tree (node, class, parent (-1
    at a root), depth (0 at a root), region (null at an internal node),
    n_rows and n_read of the regions below; nodes in pre-order),
    class_clusterings (class, n_rows, n_read, n_regions, n_pieces,
    n_voted, split_separation: the threshold the regions were sewn at,
    stability: the mean of its regions' weighted by their rows),
    granularity_candidates (class, threshold, n_regions: the curve),
    overlap_zones (node, zone, parent (-1 at the root of a zone),
    depth, separation and region_a, region_b of the pair that made an
    internal node (null at a leaf), region (null at an internal node),
    class (null at an internal node), n_classes, n_regions, n_rows,
    n_read of the regions below; zone by zone, nodes in pre-order; no
    row without an overlap), region_overlaps (zone, region_a, class_a,
    region_b, class_b, distance between the centroids, spread_a and
    spread_b along the line between them, separation; one row per
    clustering.overlap_pairs), region_stats (one row: n_regions,
    distance, split_separation, overlap_separation, n_zones). n_rows
    counts distinct rows, n_read rows of the CSV (the summed weights).
    """
    method, clustering = cfg.method, cfg.method.clustering
    points, labels = read_points(cfg)
    numeric = np.isin(prepared_columns(cfg), cfg.num_cols)
    weights = read_weights(cfg)
    n_rows = len(labels)

    classes = read_classes(cfg)
    max_clusters = max(1, clustering.max_regions // len(classes))
    region = np.empty(n_rows, dtype=np.int64)
    class_rows, candidate_rows, tree_rows, stability = [], [], [], []
    offset = node_offset = 0
    for code, name in enumerate(classes):
        rows = np.flatnonzero(labels == code)
        X, w = points[rows], weights[rows]

        def regions_of(seed: int, only: np.ndarray | None = None) -> ClassClustering:
            return cluster_class(
                X,
                numeric,
                w,
                max_fit_samples=clustering.max_fit_samples,
                min_cluster_floor=clustering.min_cluster_floor,
                max_clusters=max_clusters,
                split_separation=clustering.split_separation,
                vote_k=clustering.vote_k,
                seed=seed,
                only=only,
            )

        chosen = regions_of(cfg.seed)
        class_labels, tree = regions_tree(X, w, chosen.labels)
        n_regions = int(class_labels.max()) + 1
        own = class_labels[chosen.sample]
        repeats = [
            region_stability(own, regions_of(cfg.seed + i, chosen.sample).labels)
            for i in range(1, clustering.stability_repeats + 1)
        ]
        held = np.mean(repeats, axis=0) if repeats else np.full(n_regions, np.nan)
        stability += held.tolist()
        region[rows] = class_labels + offset
        tree_rows += _tree_rows(tree, class_labels, w, name, offset, node_offset)
        offset += n_regions
        node_offset += len(tree.parent)
        sizes = np.bincount(class_labels, minlength=n_regions)
        mean_stability = float(np.average(held, weights=sizes)) if repeats else None
        print(
            f"[regions] {name}: {n_regions} regions from {chosen.n_pieces} pieces, "
            f"{chosen.n_voted} rows voted, stability {mean_stability}",
            flush=True,
        )
        class_rows.append(
            {
                "class": name,
                "n_rows": len(rows),
                "n_read": float(w.sum()),
                "n_regions": n_regions,
                "n_pieces": chosen.n_pieces,
                "n_voted": chosen.n_voted,
                "split_separation": chosen.threshold,
                "stability": mean_stability,
            }
        )
        candidate_rows += [
            {"class": name, "threshold": float(t), "n_regions": int(k)}
            for t, k in zip(*chosen.curve)
        ]

    ids, centers = centroids(points, region, weights)
    region_class = labels[[np.flatnonzero(region == i)[0] for i in ids]]
    sizes = np.bincount(region, minlength=len(ids))
    reads = np.bincount(region, weights=weights, minlength=len(ids))
    spread = line_spread(points, region, weights, centers)
    apart = separation(centers, spread)
    threshold = clustering.overlap_separation
    zones = overlap_zones(apart, region_class, threshold)
    pairs = overlap_pairs(apart, region_class, threshold)
    n_zones = int(zones.zone.max()) + 1 if len(zones.zone) else 0
    print(
        f"[regions] {len(pairs)} pairs of regions overlap, in {n_zones} zones",
        flush=True,
    )
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
                "stability": None if np.isnan(held) else float(held),
            }
            for region_id, code, size, read, held in zip(
                ids, region_class, sizes, reads, stability
            )
        ],
    )
    db.log_table(conn, run_id, "region_tree", tree_rows)
    db.log_table(conn, run_id, "class_clusterings", class_rows)
    db.log_table(conn, run_id, "granularity_candidates", candidate_rows)
    db.log_table(
        conn,
        run_id,
        "overlap_zones",
        _zone_rows(zones, region_class, classes, sizes, reads),
    )
    db.log_table(
        conn,
        run_id,
        "region_overlaps",
        _pair_rows(pairs, zones, centers, spread, apart, region_class, classes),
    )
    db.log_table(
        conn,
        run_id,
        "region_stats",
        [
            {
                "n_regions": len(ids),
                "distance": method.distance,
                "split_separation": clustering.split_separation,
                "overlap_separation": threshold,
                "n_zones": n_zones,
            }
        ],
    )


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


def _zone_rows(
    zones: OverlapZones,
    region_class: np.ndarray,
    classes: list[str],
    sizes: np.ndarray,
    reads: np.ndarray,
) -> list[dict]:
    """Return the overlap_zones rows: each node counts the regions below it."""
    k = len(zones.parent)
    leaf = np.flatnonzero(zones.region >= 0)
    n_rows, n_read, n_regions = np.zeros(k), np.zeros(k), np.zeros(k)
    n_rows[leaf] = sizes[zones.region[leaf]]
    n_read[leaf] = reads[zones.region[leaf]]
    n_regions[leaf] = 1
    held: list[set[int]] = [set() for _ in range(k)]
    for v in leaf:
        held[v].add(int(region_class[zones.region[v]]))
    # Pre-order: walking back adds each finished node into its parent.
    for v in range(k - 1, -1, -1):
        up = zones.parent[v]
        if up >= 0:
            n_rows[up] += n_rows[v]
            n_read[up] += n_read[v]
            n_regions[up] += n_regions[v]
            held[up] |= held[v]
    depth = np.zeros(k, dtype=np.int64)
    for v in range(k):
        if zones.parent[v] >= 0:
            depth[v] = depth[zones.parent[v]] + 1
    rows = []
    for v in range(k):
        leaf_region = int(zones.region[v])
        merge = leaf_region < 0
        rows.append(
            {
                "node": v,
                "zone": int(zones.zone[v]),
                "parent": int(zones.parent[v]),
                "depth": int(depth[v]),
                "separation": float(zones.separation[v]) if merge else None,
                "region_a": int(zones.link[v, 0]) if merge else None,
                "region_b": int(zones.link[v, 1]) if merge else None,
                "region": None if merge else leaf_region,
                "class": None if merge else classes[region_class[leaf_region]],
                "n_classes": len(held[v]),
                "n_regions": int(n_regions[v]),
                "n_rows": int(n_rows[v]),
                "n_read": float(n_read[v]),
            }
        )
    return rows


def _pair_rows(
    pairs: np.ndarray,
    zones: OverlapZones,
    centers: np.ndarray,
    spread: np.ndarray,
    apart: np.ndarray,
    region_class: np.ndarray,
    classes: list[str],
) -> list[dict]:
    """Return the region_overlaps rows, one per pair, with the zone holding it."""
    leaf = np.flatnonzero(zones.region >= 0)
    zone_of = dict(zip(zones.region[leaf].tolist(), zones.zone[leaf].tolist()))
    return [
        {
            "zone": zone_of[a],
            "region_a": a,
            "class_a": classes[region_class[a]],
            "region_b": b,
            "class_b": classes[region_class[b]],
            "distance": float(np.linalg.norm(centers[b] - centers[a])),
            "spread_a": float(spread[a, b]),
            "spread_b": float(spread[b, a]),
            "separation": float(apart[a, b]),
        }
        for a, b in pairs.tolist()
    ]
