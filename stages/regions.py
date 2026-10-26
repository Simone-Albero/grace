"""Regions per class in a fitted space, with val and test routed.

Artifacts: the fitted space, the graph arrays, region centroids and
assignments. DB rows written: `regions`, `class_clusterings` and
metrics n_regions, n_graph_rows and eval_rows_per_train_row.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import logging
from collections.abc import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.clustering import (
    KdnNodes,
    assign_nearest_centroid,
    cluster_class,
    cluster_size_balance,
    compute_centroids,
    subsample_indices,
)
from src.graph import build_train_graph
from src.space import Space, scale_for_metric
from stages import LABEL, SPLITS, Context

logger = logging.getLogger(__name__)

CONFIG = ("space", "graph", "clustering", "distance", "seed")
# Split gives the feature kinds; prepare, the splits.
INPUTS = ("split", "prepare")
FILES = ()
OUTPUTS = ("space.joblib", "graph.npz", "centroids.parquet", "assignments.parquet")
_ROUTE_ROWS = 500_000


def _points(space: Space, rows: pd.DataFrame, distance: str) -> np.ndarray:
    """Return the rows in the space the regions are drawn in.

    Under cosine these are unit vectors.
    """
    return scale_for_metric(space.embed(rows), distance)


def _cluster_per_class(
    cfg: DictConfig,
    space: Space,
    train: pd.DataFrame,
    *,
    graph_rows: np.ndarray,
    graph_points: np.ndarray,
    eval_rows_per_train_row: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict[str, object]]]:
    """Cluster every class of train and number its regions.

    Each class gets the granularity that least misreads its rows'
    hardness from their region's error rate; any undersized cluster is
    merged into a survivor.

    Returns:
        The region id of every train row, the centroid of each region
        and one report row per class.
    """
    distance, seed = cfg.distance, cfg.seed
    clustering = cfg.clustering
    train_labels = train[LABEL].to_numpy()
    classes = sorted(np.unique(train_labels).tolist())
    params = OmegaConf.to_container(clustering.params, resolve=True)
    max_clusters_per_class = max(2, clustering.max_regions // len(classes))
    labels = np.full(len(train), -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0
    report: list[dict[str, object]] = []

    nodes = KdnNodes(
        X=scale_for_metric(graph_points, distance),
        y=train_labels[graph_rows],
        row_ids=graph_rows,
    )

    for class_id in tqdm(classes, desc="Clustering classes"):
        in_class = train_labels == class_id
        class_rows = np.flatnonzero(in_class)
        X_class = _points(space, train.iloc[class_rows], distance)

        raw_labels, n_merged_clusters, n_merged, loss = cluster_class(
            X_class,
            row_ids=class_rows,
            label=class_id,
            algorithm=clustering.name,
            params=params,
            nodes=nodes,
            max_fit_samples=clustering.max_fit_samples,
            max_clusters=max_clusters_per_class,
            min_cluster_floor=clustering.min_cluster_floor,
            hardness_k=clustering.hardness_k,
            eval_rows_per_train_row=eval_rows_per_train_row,
            metric=distance,
            random_state=seed,
        )
        cluster_ids = np.unique(raw_labels)
        n_class_clusters = int(cluster_ids.size)

        report.append(
            {
                "class_id": int(class_id),
                "n_train": int(raw_labels.shape[0]),
                "n_clusters": n_class_clusters,
                "loss": loss,
                "size_balance": cluster_size_balance(raw_labels),
                "n_merged_clusters": n_merged_clusters,
                "n_merged": n_merged,
            }
        )

        labels[in_class] = raw_labels + offset
        class_centroids = compute_centroids(X_class, raw_labels, metric=distance)
        centroids.update(
            {
                int(cluster) + offset: centroid
                for cluster, centroid in class_centroids.items()
            }
        )
        offset += int(cluster_ids.max()) + 1

    return labels, centroids, report


def _route(
    space: Space,
    rows: pd.DataFrame,
    centroids: Mapping[int, np.ndarray],
    distance: str,
) -> np.ndarray:
    """Return each row's nearest centroid id, embedded in chunks."""
    region = np.empty(len(rows), dtype=np.int64)
    # A whole split embedded at once peaks at ~28 GB on bot_iot_v2. A
    # multiple of the 50 000-row batches keeps the labels equal.
    for start in range(0, len(rows), _ROUTE_ROWS):
        chunk = rows.iloc[start : start + _ROUTE_ROWS]
        region[start : start + len(chunk)] = assign_nearest_centroid(
            space.embed(chunk), centroids, metric=distance
        )
    return region


def _centroid_table(
    region_ids: Sequence[int],
    centroids: Mapping[int, np.ndarray],
    region_class: pd.Series,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return the region centroids with their ids and classes.

    Raises:
        ValueError: A centroid is not finite.
    """
    table = pd.DataFrame(
        np.stack([centroids[region] for region in region_ids]).astype(np.float32),
        columns=columns,
    )
    finite = np.isfinite(table.to_numpy()).all(axis=1)
    if not finite.all():
        bad = np.asarray(region_ids)[~finite].tolist()
        raise ValueError(
            f"The centroids of regions {bad} are not finite: a column of the "
            "space is."
        )
    table.insert(0, "class_id", region_class.loc[region_ids].to_numpy())
    table.insert(0, "region_id", region_ids)
    return table


def _assignments(assigned: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return the region id of every row of every split."""
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": name,
                    "row": np.arange(len(assigned[name]), dtype=np.int32),
                    "region_id": assigned[name].astype(np.int32),
                }
            )
            for name in SPLITS
        ],
        ignore_index=True,
    )
    assignments["split"] = pd.Categorical(assignments["split"], categories=SPLITS)
    return assignments


def run(ctx: Context) -> None:
    """Write the space, graph, centroids and every row's region.

    Args:
        ctx: The stage context.

    Raises:
        ValueError: A train row has no region or a centroid is not
            finite.
    """
    cfg = ctx.cfg
    num_cols, cat_cols = ctx.features()

    prepared = ctx.inputs["prepare"].dir
    train = pd.read_parquet(prepared / "train.parquet")
    # Features only: routing must not see a row's class (no leakage).
    eval_rows = {
        name: pd.read_parquet(prepared / f"{name}.parquet", columns=num_cols + cat_cols)
        for name in SPLITS[1:]
    }
    eval_rows_per_train_row = len(eval_rows["val"]) / len(train)

    graph_rows = np.sort(
        subsample_indices(
            len(train), max_samples=cfg.graph.max_samples, random_state=cfg.seed
        )
    )
    space = Space.fit(
        train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        top_k=cfg.space.top_k,
        cat_cost=cfg.space.cat_cost,
    )
    graph_points = space.embed(train.iloc[graph_rows])
    train_graph = build_train_graph(
        graph_points,
        k=cfg.graph.k,
        metric=cfg.distance,
        n_jobs=ctx.runtime.n_jobs,
    )

    logger.info("Running per-class clustering on train (n=%d)...", len(train))
    labels, centroids, report = _cluster_per_class(
        cfg,
        space,
        train,
        graph_rows=graph_rows,
        graph_points=graph_points,
        eval_rows_per_train_row=eval_rows_per_train_row,
    )
    if (labels < 0).any():
        raise ValueError(f"{int((labels < 0).sum())} train rows were given no region.")
    region_ids = sorted(centroids)
    region_class = pd.Series(train[LABEL].to_numpy()).groupby(labels).first()
    centroid_table = _centroid_table(
        region_ids, centroids, region_class, space.columns()
    )
    assigned = {"train": labels} | {
        name: _route(space, rows, centroids, cfg.distance)
        for name, rows in eval_rows.items()
    }
    assignments = _assignments(assigned)

    joblib.dump(space, ctx.out / "space.joblib")
    np.savez(
        ctx.out / "graph.npz",
        rows=graph_rows,
        knn_idx=train_graph.knn_idx,
        knn_dist=train_graph.knn_dist,
        mst=train_graph.mst_edges,
    )
    centroid_table.to_parquet(ctx.out / "centroids.parquet", index=False)
    assignments.to_parquet(ctx.out / "assignments.parquet", index=False)

    sizes = (
        assignments.groupby(["region_id", "split"], observed=False)
        .size()
        .unstack("split")
        .add_prefix("n_")
    )
    ctx.write(
        "regions",
        pd.DataFrame(
            {"region_id": region_ids, "class_id": centroid_table["class_id"]}
        ).join(sizes, on="region_id"),
    )
    ctx.write("class_clusterings", pd.DataFrame(report))
    ctx.save_metrics(
        {
            "n_regions": len(region_ids),
            "n_graph_rows": len(graph_rows),
            "eval_rows_per_train_row": eval_rows_per_train_row,
        }
    )
    logger.info("Clustering complete: %d regions", len(region_ids))
