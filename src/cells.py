"""Cells: each class cut into the smallest compact groups of rows a measure makes sense on.

The cells are the units every defect is measured on (src.quality). They
are as small as min_rows allows, the fewest rows a measure makes sense
on, so that a defect touching a share of a cell's rows is not diluted in
a larger group; the regions the report merges them into set its
granularity, large where a class is uniform, small where it is not.

The rows of a class are its distinct rows, each counted once whatever
its copies in the CSV: the copies are a defect of their own
(duplicates), and no measure of a cell weighs them. A class is read on
a uniform sample of its rows (graph.sample_rows, stage cells): the Ward
tree is built on every sampled row, and the cut counts sampled rows.
The tree is cut from the root: a node is split into its two children
while it holds rows enough for two cells, until every node left is a
cell; a branch with too few rows to be measured is left out, its rows
joining the nearest cell. Every row that is not sampled joins the cell
of its nearest sampled row (nearest_rows), to be counted only.

The cut is the tree of the cells (CellTree): sibling cells are the most
alike, so the report merges cells up this tree where no verdict they
decide changes on the rows merged (merge_nodes).
"""

from __future__ import annotations

__all__ = [
    "CellTree",
    "WardTree",
    "class_cells",
    "merge_nodes",
    "nearest_rows",
    "ward_tree",
]

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

_NEAREST_ROWS = 100_000  # rows whose nearest sampled row is searched at once


@dataclass(frozen=True)
class WardTree:
    """The merges of a Ward tree, in scipy's numbering.

    Node i < n is row i; node n + j is the cluster made by merge j.

    Attributes:
        children: The two nodes each merge joins, shape (n - 1, 2),
            int64; a merge comes after the merges of its children.
        heights: The height of each merge, sqrt(2 Δ) with Δ the
            increase of the within-cluster sum of squares
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


def ward_tree(X: np.ndarray) -> WardTree:
    """Build the Ward tree of the rows.

    Merging clusters a and b costs Δ = n_a n_b / (n_a + n_b) ·
    ||c_a - c_b||², with n the rows and c the centroid of a cluster.
    Built by the nearest-neighbour chain on the centroids
    (memory O(n d)); Ward is reducible, so sorting the merges by height
    (stably: the chain keeps a child's merge before its parent's) gives
    a valid order.

    Args:
        X: The points, shape (n, d), n >= 1.

    Returns:
        The tree; no merge when n = 1.
    """
    centers = np.array(X, dtype=np.float64)
    n = len(centers)
    mass = np.ones(n)  # the rows of each cluster
    if n < 2:
        return WardTree(np.empty((0, 2), dtype=np.int64), np.empty(0))
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
            centers, mass, slot = centers[keep], mass[keep], slot[keep]
            alive = np.ones(len(keep), dtype=bool)
            chain = [int(position[c]) for c in chain]
        if not chain:
            chain.append(int(np.flatnonzero(alive)[0]))
        a = chain[-1]
        cost = _squared_distances(centers, a) * (mass * mass[a] / (mass + mass[a]))
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


def class_cells(X: np.ndarray, min_rows: int) -> tuple[np.ndarray, CellTree]:
    """Cut the sampled rows of one class into cells of min_rows to 2 · min_rows - 1 rows.

    A Ward tree (ward_tree) is built on every row of X; each row joins
    its nearest row of X (itself, or an equal row before it), so a node
    holds the rows that join its leaves. From the root, a node of at
    least 2 · min_rows rows is split into its two children; a child of
    fewer than min_rows rows is left out, its rows joining the cell of
    their nearest row in a cell, and the node goes on as its other
    child. Every node left is a cell. A cell so holds fewer than
    2 · min_rows rows, and the rows of the small branches left out
    beside it; fewer than 2 · min_rows rows are one cell.

    Args:
        X: The points of the sampled rows of the class (distinct rows),
            float64, shape (m, d), m >= 1.
        min_rows: The fewest rows of a cell, >= 1.

    Returns:
        The cell of every row of X, 0.. with no gap, numbered in the
        pre-order of the cut; and the cut.
    """
    m = len(X)
    tree = ward_tree(X)
    nearest = nearest_rows(X, X)
    leaves, start, stop = _leaf_ranges(tree.children, m)
    held = np.r_[0, np.cumsum(np.bincount(nearest, minlength=m)[leaves])]
    size = held[stop] - held[start]  # the rows below each node
    sample_cell = np.full(m, -1, dtype=np.int64)
    parent, cell = [], []
    stack = [(2 * m - 2, -1)]
    while stack:
        v, up = stack.pop()
        while v >= m and size[v] >= 2 * min_rows:
            a, b = tree.children[v - m]
            small_a, small_b = size[a] < min_rows, size[b] < min_rows
            if small_a and small_b:  # never: one child holds half the rows
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
        labels[left] = sample_cell[kept[nearest_rows(X[left], X[kept])]]
    return labels, CellTree(
        np.asarray(parent, dtype=np.int64), np.asarray(cell, dtype=np.int64)
    )


def merge_nodes(
    tree: CellTree, entries: Sequence[Mapping[Hashable, bool | None]]
) -> np.ndarray:
    """Merge the cells of a class into regions up its cut, where no decided verdict changes.

    Every node of the cut has a verdict on each entry (a defect against
    a target): True, False or None (undecided), read on its own rows.
    A cell is merged. An internal node is merged when its two children
    are, and every verdict a child decides (True or False) is the same
    on the node: a decided verdict that the node's rows turn undecided
    or opposite stops the merge, for merging would dilute the very
    share the child measures. A region is a merged node whose parent is
    not merged (or the root, when merged): the largest branch whose
    rows keep every verdict of its parts, named by its top node, and
    its verdicts are the node's.

    Args:
        tree: The cut of the class (class_cells).
        entries: Per node of the cut, in its order, the verdict of every
            entry (the raw ones: no defect hides another).

    Returns:
        The region of every cell, indexed by cell: the index in the cut
        of the top node of its region (so not numbered from 0, nor
        without gaps).
    """
    parent = np.asarray(tree.parent, dtype=np.int64)
    node_cell = np.asarray(tree.cell, dtype=np.int64)
    k = len(parent)
    children: list[list[int]] = [[] for _ in range(k)]
    for v in range(k):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    merged = np.zeros(k, dtype=bool)
    for v in range(k - 1, -1, -1):  # children after their parent
        if node_cell[v] >= 0:
            merged[v] = True
            continue
        merged[v] = all(merged[c] for c in children[v]) and all(
            entries[v].get(entry) == value
            for c in children[v]
            for entry, value in entries[c].items()
            if value is not None
        )
    region_of_node = np.full(k, -1, dtype=np.int64)
    for v in range(k):  # a parent before its children
        if merged[v]:
            up = parent[v]
            region_of_node[v] = region_of_node[up] if up >= 0 and merged[up] else v
    leaf = node_cell >= 0
    region = np.empty(int(leaf.sum()), dtype=np.int64)
    region[node_cell[leaf]] = region_of_node[leaf]
    return region


def _squared_distances(X: np.ndarray, a: int) -> np.ndarray:
    """Squared euclidean distance of every row of X to row a, exactly symmetric.

    Summed over the differences, not as ||x||² + ||y||² - 2 x·y: the
    distance of row a to row b is then the same float as that of b to a,
    which the nearest-neighbour chain of ward_tree needs to end (with
    rows a hair apart, the rounding of the dot product made a to b
    differ from b to a, and the chain went round three rows forever).

    Args:
        X: The points, shape (n, d), float64.
        a: The row the distances are to, 0 <= a < n.

    Returns:
        The squared distances, shape (n,), 0 at row a.
    """
    diff = X - X[a]
    return np.einsum("ij,ij->i", diff, diff)


def nearest_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return, for every row of X, the index of its nearest row of Y.

    Euclidean, exact (scikit-learn's brute force), _NEAREST_ROWS rows of
    X at a time; of rows of Y at the same distance, the first.

    Args:
        X: The rows, shape (n, d).
        Y: The rows searched, shape (m, d), m >= 1.

    Returns:
        Shape (n,) int64, in [0, m).
    """
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
