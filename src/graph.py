"""k-NN search, k-NN graph and approximate MST of the train sample."""

__all__ = ["TrainGraph", "build_train_graph", "nearest_neighbors"]

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.space import scale_for_metric

logger = logging.getLogger(__name__)

_KNN_BATCH_ROWS = 256


def _thread_budget() -> int:
    """Return the cores this process may use.

    The affinity mask follows the cgroup or cpuset quota, not the host.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def nearest_neighbors(
    nodes: np.ndarray,
    node_rows: np.ndarray,
    points: np.ndarray,
    point_rows: np.ndarray,
    *,
    k: int,
    metric: str,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k nearest graph nodes of every point.

    Args:
        nodes: The graph nodes, shape (n_nodes, d).
        node_rows: The row id of every node.
        points: The points to search from, shape (n_points, d).
        point_rows: The row id of every point; a point that is a node,
            told by its row id, is not its own neighbour.
        k: The neighbours per point, capped at n_nodes - 1.
        metric: `euclidean` or `cosine`, which fixes how the points are
            scaled before the euclidean search.
        n_jobs: The threads, in joblib's meaning: -1 is all cores, -2
            all but one.

    Returns:
        The indices into `nodes` and the distances, both of shape
        (n_points, min(k, n_nodes - 1)), nearest first.
    """
    n_points = points.shape[0]
    effective_k = min(k, nodes.shape[0] - 1)
    scaled_nodes = scale_for_metric(nodes, metric)
    scaled_points = scale_for_metric(points, metric)

    indices = np.empty((n_points, effective_k), dtype=np.int64)
    distances = np.empty((n_points, effective_k), dtype=np.float64)

    def fill_batch(start: int) -> None:
        end = min(start + _KNN_BATCH_ROWS, n_points)
        distances_to_nodes = cdist(
            scaled_points[start:end], scaled_nodes, metric="euclidean"
        )
        # A point that is a node is not its own neighbour.
        distances_to_nodes[node_rows[None, :] == point_rows[start:end, None]] = np.inf

        candidates = np.argpartition(distances_to_nodes, effective_k, axis=1)[
            :, :effective_k
        ]
        candidate_dist = np.take_along_axis(distances_to_nodes, candidates, axis=1)
        order = np.argsort(candidate_dist, axis=1)
        indices[start:end] = np.take_along_axis(candidates, order, axis=1)
        distances[start:end] = np.take_along_axis(candidate_dist, order, axis=1)

    starts = list(range(0, n_points, _KNN_BATCH_ROWS))
    # cdist and argpartition release the GIL, so threads overlap real
    # work; each fills its own slice of `indices` and `distances`, with
    # nothing to lock between them.
    workers = n_jobs if n_jobs > 0 else max(1, _thread_budget() + 1 + n_jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fill_batch, start) for start in starts]
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="k-NN",
                unit="batch",
                leave=False,
            ):
                future.result()
        except BaseException:
            # Batches not yet started can still be dropped; ones already
            # running finish regardless, since they only write to their
            # own slice.
            for pending in futures:
                pending.cancel()
            raise

    return indices, distances


def _build_knn_graph(
    points: np.ndarray, *, k: int, metric: str, n_jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return every point's k nearest other points: the k-NN graph."""
    rows = np.arange(points.shape[0])
    return nearest_neighbors(
        points, rows, points, rows, k=k, metric=metric, n_jobs=n_jobs
    )


def _symmetrize_knn_graph(
    indices: np.ndarray, distances: np.ndarray, n_points: int
) -> scipy.sparse.csr_matrix:
    """Symmetrize the k-NN graph, keeping the least edge distance."""
    k = indices.shape[1]
    rows = np.repeat(np.arange(n_points), k)
    cols = indices.ravel()
    data = distances.ravel()

    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])

    order = np.lexsort((sym_data, sym_cols, sym_rows))
    sym_rows = sym_rows[order]
    sym_cols = sym_cols[order]
    sym_data = sym_data[order]

    first_of_edge = np.ones(len(sym_rows), dtype=bool)
    first_of_edge[1:] = (sym_rows[1:] != sym_rows[:-1]) | (
        sym_cols[1:] != sym_cols[:-1]
    )

    adjacency = scipy.sparse.csr_matrix(
        (sym_data[first_of_edge], (sym_rows[first_of_edge], sym_cols[first_of_edge])),
        shape=(n_points, n_points),
    )
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def _bridge_disconnected(
    adjacency: scipy.sparse.csr_matrix, points: np.ndarray, *, metric: str
) -> scipy.sparse.csr_matrix:
    """Join each disconnected component to the first by one edge."""
    n_components, component_of = scipy.sparse.csgraph.connected_components(
        adjacency, directed=False
    )
    if n_components == 1:
        return adjacency

    adjacency = adjacency.tolil()
    anchor = int(np.where(component_of == 0)[0][0])
    scaled = scale_for_metric(points, metric)
    anchor_dist = cdist(scaled[anchor : anchor + 1], scaled, metric="euclidean")[0]

    for component in range(1, n_components):
        component_nodes = np.where(component_of == component)[0]
        bridge_node = int(component_nodes[anchor_dist[component_nodes].argmin()])
        bridge_dist = max(float(anchor_dist[bridge_node]), 1e-10)
        adjacency[anchor, bridge_node] = bridge_dist
        adjacency[bridge_node, anchor] = bridge_dist
    return adjacency.tocsr()


def _build_approx_mst(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    points: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Return the MST edges of the k-NN graph, bridged if split."""
    graph = _symmetrize_knn_graph(knn_indices, knn_distances, points.shape[0])
    graph = _bridge_disconnected(graph, points, metric=metric)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph).tocoo()
    if mst.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((mst.row, mst.col)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class TrainGraph:
    """A uniform sample of train with its k-NN graph and MST.

    Every population is measured against it.

    Attributes:
        points: The sampled rows in the clustering space, shape (n, d).
        knn_idx: The k nearest others of every point, shape (n, k).
        knn_dist: The distances to them, shape (n, k).
        mst_edges: The approximate MST as point pairs, shape (m, 2).
    """

    points: np.ndarray
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    mst_edges: np.ndarray


def build_train_graph(
    points: np.ndarray, *, k: int, metric: str, n_jobs: int
) -> TrainGraph:
    """Return the k-NN graph and the approximate MST of a point cloud.

    Args:
        points: The point cloud, shape (n, d).
        k: The neighbours per point.
        metric: `euclidean` or `cosine`.
        n_jobs: The threads of the search, as in `nearest_neighbors`.

    Returns:
        The graph, whose `points` are the given ones.

    Raises:
        ValueError: If there are fewer than two points.
    """
    if points.shape[0] < 2:
        raise ValueError(f"A graph of {points.shape[0]} row(s) has no neighbours.")
    logger.info("Building the %s k-NN graph (k=%d)...", metric, k)
    knn_idx, knn_dist = _build_knn_graph(points, k=k, metric=metric, n_jobs=n_jobs)
    logger.info("Building approximate MST over the k-NN graph...")
    mst_edges = _build_approx_mst(knn_idx, knn_dist, points, metric=metric)
    return TrainGraph(points, knn_idx, knn_dist, mst_edges)
