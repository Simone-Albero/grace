"""Cells: each class cut into compact groups of rows of about the same size.

The cells are the units every defect is measured on (src.quality). Their
size is the same for every class, so that two cells compared are sampled
alike, and bounded, so that a defect touching a share of a cell's rows is
not diluted in a larger group: the size sets the resolution of the report.

The rows of a class are deduplicated, each weighted by its copies in the
CSV, so the weighted Ward tree of the distinct rows is the Ward tree of
the rows as read. The tree is built on a uniform sample of the class,
and every row joins its nearest sampled row, so each node of the tree
holds the rows of the sampled rows below it. The tree is cut from the
root: a node with more rows than a cell may hold is split into its two
children, until every node left is a cell; a branch with too few rows to
be measured is left out, its rows joining the nearest cell.

The cut is the tree of the cells (CellTree): sibling cells are the most
alike, so the report merges cells up this tree where their defects agree
(merge_cells).
"""

from __future__ import annotations

__all__ = [
    "CellTree",
    "WardTree",
    "class_cells",
    "merge_cells",
    "ward_tree",
]

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.graph import sample_rows

_NEAREST_ROWS = 100_000  # rows whose nearest sampled row is searched at once
_MIXED = object()  # the profile of a node whose cells differ


@dataclass(frozen=True)
class WardTree:
    """The merges of a Ward tree, in scipy's numbering.

    Node i < n is row i; node n + j is the cluster made by merge j.

    Attributes:
        children: The two nodes each merge joins, shape (n - 1, 2),
            int64; a merge comes after the merges of its children.
        heights: The height of each merge, sqrt(2 Δ) with Δ the
            increase of the weighted within-cluster sum of squares
            (scipy's convention), shape (n - 1,), ascending.
    """

    children: np.ndarray
    heights: np.ndarray


@dataclass(frozen=True)
class CellTree:
    """The cut of a class's Ward tree: its nodes down to the cells.

    Nodes are in pre-order, the left child first, so a parent comes
    before its children and the cells come in order 0, 1, ...; every
    internal node has two children.

    Attributes:
        parent: The parent of each node, -1 at the root, shape (k,),
            int64.
        cell: The cell of each leaf, 0.. in pre-order; -1 at an
            internal node. Shape (k,), int64.
    """

    parent: np.ndarray
    cell: np.ndarray


def ward_tree(X: np.ndarray, weights: np.ndarray) -> WardTree:
    """Build the weighted Ward tree of the rows.

    Merging clusters a and b costs Δ = w_a w_b / (w_a + w_b) ·
    ||c_a - c_b||², with w the summed weight and c the weighted centroid
    of a cluster: with integer weights, the tree of the rows repeated
    as many times. Built by the nearest-neighbour chain on the centroids
    (memory O(n d)); Ward is reducible, so sorting the merges by height
    (stably: the chain keeps a child's merge before its parent's) gives
    a valid order.

    Args:
        X: The points, shape (n, d), n >= 1.
        weights: The weight of every row, > 0.

    Returns:
        The tree; no merge when n = 1.
    """
    centers = np.array(X, dtype=np.float64)
    mass = np.array(weights, dtype=np.float64)
    n = len(centers)
    if n < 2:
        return WardTree(np.empty((0, 2), dtype=np.int64), np.empty(0))
    norms = np.einsum("ij,ij->i", centers, centers)
    slot = np.arange(n)  # the row whose position a cluster holds
    alive = np.ones(n, dtype=bool)
    made = np.zeros(n)  # the height of the merge that made each row's cluster
    pairs, heights, chain = [], [], []
    remaining = n
    while remaining > 1:
        if 2 * remaining < len(centers) and len(centers) > 256:
            # Keep only the live clusters: the scans below cost their count.
            keep = np.flatnonzero(alive)
            position = np.full(len(centers), -1)
            position[keep] = np.arange(len(keep))
            centers, mass, norms, slot = (
                centers[keep],
                mass[keep],
                norms[keep],
                slot[keep],
            )
            alive = np.ones(len(keep), dtype=bool)
            chain = [int(position[c]) for c in chain]
        if not chain:
            chain.append(int(np.flatnonzero(alive)[0]))
        a = chain[-1]
        squared = np.maximum(norms + norms[a] - 2.0 * (centers @ centers[a]), 0.0)
        cost = squared * (mass * mass[a] / (mass + mass[a]))
        cost[~alive] = np.inf
        cost[a] = np.inf
        b = int(np.argmin(cost))
        # The previous link of the chain wins ties, so the chain ends.
        if len(chain) > 1 and cost[chain[-2]] <= cost[b]:
            b = chain[-2]
        if len(chain) == 1 or b != chain[-2]:
            chain.append(b)
            continue
        del chain[-2:]
        total = mass[a] + mass[b]
        centers[a] = (mass[a] * centers[a] + mass[b] * centers[b]) / total
        mass[a] = total
        norms[a] = centers[a] @ centers[a]
        alive[b] = False
        # Rounding can put a merge a hair under its children's.
        height = max(np.sqrt(2.0 * cost[b]), made[slot[a]], made[slot[b]])
        made[slot[a]] = height
        pairs.append((slot[a], slot[b]))
        heights.append(height)
        remaining -= 1
    order = np.argsort(heights, kind="stable")
    node = np.arange(n)  # the tree node each row's cluster is now
    children = np.empty((n - 1, 2), dtype=np.int64)
    for j, i in enumerate(order):
        a, b = pairs[i]
        children[j] = sorted((node[a], node[b]))
        node[a] = n + j
    return WardTree(children, np.asarray(heights)[order])


def class_cells(
    X: np.ndarray,
    weights: np.ndarray,
    max_rows: int,
    min_rows: int,
    max_fit_samples: int,
    seed: int,
) -> tuple[np.ndarray, CellTree]:
    """Cut one class into cells of at most max_rows distinct rows.

    A weighted Ward tree (ward_tree) is built on a uniform sample of at
    most max_fit_samples rows (graph.sample_rows); every row joins its
    nearest sampled row (euclidean), so a node holds
    the rows of its sampled rows. From the root, a node of more than
    max_rows rows is split into its two children; a child of fewer than
    min_rows rows is left out, its rows joining the cell of their nearest
    sampled row in a cell, and the node goes on as its other child. Every
    node left is a cell. A cell so holds at most max_rows rows, and the
    rows of the small branches left out beside it.

    Args:
        X: The points of the class, shape (n, d), n >= 1.
        weights: The weight of every row (its copies in the CSV), > 0.
        max_rows: The most distinct rows of a cell, >= 2 * min_rows.
        min_rows: The fewest distinct rows of a cell, >= 1.
        max_fit_samples: The most rows the tree is built on.
        seed: The seed of the sample.

    Returns:
        The cell of every row, 0.. with no gap, numbered in the
        pre-order of the cut; and the cut.
    """
    n = len(X)
    sampled = sample_rows(n, max_fit_samples, seed)
    m = len(sampled)
    tree = ward_tree(X[sampled], weights[sampled])
    nearest = _nearest(X, X[sampled])
    leaves, start, stop = _leaf_ranges(tree.children, m)
    held = np.r_[0, np.cumsum(np.bincount(nearest, minlength=m)[leaves])]
    size = held[stop] - held[start]  # the rows of the class below each node
    sample_cell = np.full(m, -1, dtype=np.int64)
    parent, cell = [], []
    stack = [(2 * m - 2, -1)]
    while stack:
        v, up = stack.pop()
        while v >= m and size[v] > max_rows:
            a, b = tree.children[v - m]
            small_a, small_b = size[a] < min_rows, size[b] < min_rows
            if small_a and small_b:  # never when max_rows >= 2 * min_rows
                break
            if small_a or small_b:
                v = b if small_a else a  # the small child is left out
                continue
            parent.append(up)
            cell.append(-1)
            stack += [(b, len(parent) - 1), (a, len(parent) - 1)]
            v = -1
            break
        if v < 0:
            continue
        parent.append(up)
        cell.append(int(sample_cell.max()) + 1)
        sample_cell[leaves[start[v] : stop[v]]] = cell[-1]
    labels = sample_cell[nearest]
    left = np.flatnonzero(labels < 0)
    if len(left):
        kept = np.flatnonzero(sample_cell >= 0)
        labels[left] = sample_cell[kept[_nearest(X[left], X[sampled[kept]])]]
    return labels, CellTree(
        np.asarray(parent, dtype=np.int64), np.asarray(cell, dtype=np.int64)
    )


def merge_cells(tree: CellTree, profile: Sequence[Hashable]) -> np.ndarray:
    """Merge the cells of a class into regions up its cut.

    A node is uniform when it is a cell, or when both its children are
    uniform with the same profile; a region is a uniform node whose
    parent is not uniform, or the root when it is uniform. A region is so
    the largest branch of the cut whose cells all share a profile.

    Args:
        tree: The cut of the class (class_cells).
        profile: The profile of every cell, indexed by cell; two equal
            profiles compare equal.

    Returns:
        The region of every cell, 0.. with no gap, numbered in the
        pre-order of the cut.
    """
    parent = np.asarray(tree.parent, dtype=np.int64)
    k = len(parent)
    children: list[list[int]] = [[] for _ in range(k)]
    for v in range(1, k):
        children[parent[v]].append(v)
    # The profile a node's cells share, _MIXED when they differ; children
    # come after their parent, so walking back sees them first.
    shared: list[object] = [_MIXED] * k
    for v in range(k - 1, -1, -1):
        if tree.cell[v] >= 0:
            shared[v] = profile[tree.cell[v]]
            continue
        a, b = (shared[x] for x in children[v])
        if a is not _MIXED and b is not _MIXED and a == b:
            shared[v] = a
    region_of_node = np.full(k, -1, dtype=np.int64)
    n_regions = 0
    for v in range(k):
        if shared[v] is _MIXED:
            continue
        up = parent[v]
        if up >= 0 and shared[up] is not _MIXED:
            region_of_node[v] = region_of_node[up]
        else:
            region_of_node[v] = n_regions
            n_regions += 1
    leaf = tree.cell >= 0
    region = np.empty(int(leaf.sum()), dtype=np.int64)
    region[tree.cell[leaf]] = region_of_node[leaf]
    return region


def _nearest(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return, for every row of X, the index of its nearest row of Y (euclidean)."""
    search = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Y)
    return np.concatenate(
        [
            search.kneighbors(X[start : start + _NEAREST_ROWS], return_distance=False)[
                :, 0
            ]
            for start in range(0, len(X), _NEAREST_ROWS)
        ]
    ).astype(np.int64)


def _leaf_ranges(
    children: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the leaves in pre-order, and where each node's leaves start and stop."""
    n_nodes = 2 * n - 1
    start, stop = np.zeros(n_nodes, dtype=np.int64), np.zeros(n_nodes, dtype=np.int64)
    leaves = np.empty(n, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)
    for j, (a, b) in enumerate(children):
        size[n + j] = size[a] + size[b]
    stack = [n_nodes - 1]
    while stack:
        v = stack.pop()
        if v < n:
            leaves[start[v]] = v
            stop[v] = start[v] + 1
            continue
        a, b = children[v - n]
        start[a], start[b] = start[v], start[v] + size[a]
        stop[v] = start[v] + size[v]
        stack += [b, a]
    return leaves, start, stop
