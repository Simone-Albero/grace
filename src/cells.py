"""Cells: each class cut into the smallest compact groups of rows a measure makes sense on.

The cells are the units every defect is measured on (src.quality). They
are as small as min_rows allows, the fewest rows a measure makes sense
on, so that a defect touching a share of a cell's rows is not diluted in
a larger group; the regions the report merges them into set its
granularity, large where a class is uniform, small where it is not.

The rows of a class are its distinct rows, each counted once whatever
its copies in the CSV: the copies are a defect of their own
(duplicates), and no measure of a cell weighs them. The Ward tree is
built on a uniform sample of the class, and every row joins its nearest
sampled row, so each node of the tree
holds the rows of the sampled rows below it. The tree is cut from the
root: a node is split into its two children while it holds rows enough
for two cells, until every node left is a cell; a branch with too few
rows to be measured is left out, its rows joining the nearest cell.

The cut is the tree of the cells (CellTree): sibling cells are the most
alike, so the report merges cells up this tree where their defects do
not contradict each other (merge_cells).
"""

from __future__ import annotations

__all__ = [
    "CellTree",
    "WardTree",
    "class_cells",
    "merge_cells",
    "ward_tree",
]

from collections.abc import Hashable, Sequence, Set
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.graph import sample_rows

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


def class_cells(
    X: np.ndarray,
    min_rows: int,
    max_fit_samples: int,
    seed: int,
) -> tuple[np.ndarray, CellTree]:
    """Cut one class into cells of min_rows to 2 · min_rows - 1 distinct rows.

    A Ward tree (ward_tree) is built on a uniform sample of at
    most max_fit_samples rows (graph.sample_rows); every row joins its
    nearest sampled row (euclidean), so a node holds the rows of its
    sampled rows. From the root, a node of at least 2 · min_rows rows is
    split into its two children; a child of fewer than min_rows rows is
    left out, its rows joining the cell of their nearest sampled row in
    a cell, and the node goes on as its other child. Every node left is
    a cell. A cell so holds fewer than 2 · min_rows rows, and the rows
    of the small branches left out beside it; a class of fewer than
    2 · min_rows rows is one cell. A node is never smaller than its
    sampled rows: with more rows than max_fit_samples · 2 · min_rows a
    class is cut at most into its sampled rows.

    Args:
        X: The points of the class (its distinct rows), shape (n, d),
            n >= 1.
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
    tree = ward_tree(X[sampled])
    nearest = _nearest(X, X[sampled])
    leaves, start, stop = _leaf_ranges(tree.children, m)
    held = np.r_[0, np.cumsum(np.bincount(nearest, minlength=m)[leaves])]
    size = held[stop] - held[start]  # the rows of the class below each node
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
        labels[left] = sample_cell[kept[_nearest(X[left], X[sampled[kept]])]]
    return labels, CellTree(
        np.asarray(parent, dtype=np.int64), np.asarray(cell, dtype=np.int64)
    )


def merge_cells(
    tree: CellTree, yes: Sequence[Set[Hashable]], no: Sequence[Set[Hashable]]
) -> np.ndarray:
    """Merge the cells of a class into regions up its cut, where they do not contradict each other.

    Every cell has a verdict on some entries (a defect, against a class
    or on a key): the entries it has (yes) and those it has not (no);
    on the others it is undecided. A node is uniform when no entry is
    in the yes of one of its cells and in the no of another; a region
    is a uniform node whose parent is not uniform, or the root when it
    is uniform. A region is so the largest branch of the cut whose
    cells do not contradict each other (an undecided cell never splits
    a branch), and it is named by its top node: the same branch has the
    same name whatever the cells outside it.

    Args:
        tree: The cut of the class (class_cells).
        yes: Per cell, indexed by cell, the entries it has.
        no: Per cell, the entries it has not; disjoint from its yes.

    Returns:
        The region of every cell: the index in the cut of the top node
        of its region (so not numbered from 0, nor without gaps).
    """
    parent = np.asarray(tree.parent, dtype=np.int64)
    k = len(parent)
    # The entries a node's cells have and have not; children come after
    # their parent, so walking back adds each into its parent.
    has: list[set] = [set() for _ in range(k)]
    lacks: list[set] = [set() for _ in range(k)]
    for v in range(k - 1, -1, -1):
        if tree.cell[v] >= 0:
            has[v] |= yes[tree.cell[v]]
            lacks[v] |= no[tree.cell[v]]
        if parent[v] >= 0:
            has[parent[v]] |= has[v]
            lacks[parent[v]] |= lacks[v]
    uniform = [not (has[v] & lacks[v]) for v in range(k)]
    region_of_node = np.full(k, -1, dtype=np.int64)
    for v in range(k):
        if not uniform[v]:
            continue
        up = parent[v]
        region_of_node[v] = region_of_node[up] if up >= 0 and uniform[up] else v
    leaf = tree.cell >= 0
    region = np.empty(int(leaf.sum()), dtype=np.int64)
    region[tree.cell[leaf]] = region_of_node[leaf]
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
