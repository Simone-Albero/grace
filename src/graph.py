"""The points of the space and their nearest neighbours.

space_points turns the prepared rows into the points every distance is
measured on; past it every distance is euclidean. sample_rows draws the
uniform sample of reference rows the kDN hardness looks up, and
nearest_neighbors finds the neighbours of rows among them.
"""

from __future__ import annotations

__all__ = [
    "nearest_neighbors",
    "sample_rows",
    "space_points",
]

import numpy as np
from sklearn.neighbors import NearestNeighbors

_CHUNK_ROWS = 20_000  # rows of queries whose neighbour offsets are held at once


def space_points(numbers: np.ndarray, slots: np.ndarray, metric: str) -> np.ndarray:
    """Return the points every distance of the method is measured on.

    The metric applies to the numbers only: under "cosine" each row of
    numbers is divided by max(its norm, 1e-8), so that the euclidean
    distance of two rows is sqrt(2 (1 - cos)); under "euclidean" they
    stay as they are. The categorical slots follow unchanged, so a
    mismatch costs the same squared distance under either metric.

    Args:
        numbers: The prepared numeric columns, shape (n, d_num).
        slots: The prepared one-hot slots, shape (n, d_cat); may have
            no column.
        metric: "euclidean" or "cosine".

    Returns:
        Float64 points, shape (n, d_num + d_cat).
    """
    numbers = np.asarray(numbers, dtype=np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(numbers, axis=1, keepdims=True)
        numbers = numbers / np.maximum(norms, 1e-8)
    return np.hstack([numbers, np.asarray(slots, dtype=np.float64)])


def sample_rows(n: int, max_samples: int, seed: int) -> np.ndarray:
    """Return a uniform sample of range(n), ascending.

    Args:
        n: The rows to sample from.
        max_samples: The most rows kept; every row when n <=
            max_samples.
        seed: The seed of the draw (without replacement).

    Returns:
        The kept positions, int64, ascending.
    """
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False)).astype(np.int64)


def nearest_neighbors(
    nodes: np.ndarray,
    node_rows: np.ndarray,
    queries: np.ndarray,
    query_rows: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k nearest nodes of every query, never itself.

    A query whose row id is also a node's row id never gets that node,
    even at distance 0; a duplicate point under another row id is a
    neighbour.

    Args:
        nodes: The node points, shape (n, d), n >= 2.
        node_rows: The row id of every node.
        queries: The query points, shape (m, d).
        query_rows: The row id of every query.
        k: The neighbours wanted; min(k, n - 1) are returned.

    Returns:
        Node indices and distances, both of shape (m, min(k, n - 1)),
        nearest first.
    """
    node_rows, query_rows = np.asarray(node_rows), np.asarray(query_rows)
    n_returned = min(k, len(nodes) - 1)
    search = NearestNeighbors(n_neighbors=n_returned + 1, algorithm="brute", n_jobs=-1)
    distances, indices = search.fit(nodes).kneighbors(queries)
    by_row = np.argsort(node_rows)
    position = np.searchsorted(node_rows, query_rows, sorter=by_row)
    own = by_row[np.minimum(position, len(node_rows) - 1)]
    own = np.where(node_rows[own] == query_rows, own, -1)
    keep = indices != own[:, None]
    # A query that is no node, or whose node fell past the k + 1 nearest
    # among ties at the same distance, drops its farthest neighbour.
    keep[keep.all(axis=1), -1] = False
    shape = (len(queries), n_returned)
    indices = indices[keep].reshape(shape)
    # sklearn's brute search computes distances through dot products, so
    # duplicates come out a little above 0; recompute them exactly.
    distances = np.empty(shape)
    for start in range(0, len(queries), _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        offsets = nodes[indices[rows]] - queries[rows, None, :]
        distances[rows] = np.sqrt(np.einsum("ijk,ijk->ij", offsets, offsets))
    order = np.argsort(distances, axis=1, kind="stable")
    return (
        np.take_along_axis(indices, order, axis=1),
        np.take_along_axis(distances, order, axis=1),
    )
