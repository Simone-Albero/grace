"""Regions: each class split where a weighted Ward tree finds separated groups.

The rows of a class are deduplicated, each weighted by its copies in
the CSV, so the weighted Ward tree of the distinct rows is the Ward
tree of the rows as read. The rows are split in two halves: the tree
is built on half A, and half B, each row attached to the node of its
nearest row of A, judges its splits. Ward always splits where a split
looks best on the rows it was built on, so a split is never judged on
them.

A split is judged by the separation S of its two sides: the distance
between their centroids (on A, which picks the line) in standard
deviations of the rows of B along that line, Ashman's D over sqrt(2).
It is a measure of how far apart two groups are, not a test: the rows
of B are themselves sorted by the split (Gao, Bien and Witten, JASA
2024), so S is not 0 on a single group, and the threshold has to lie
above what a split of one group gives. S is measured on the numeric
columns only (a categorical value is a constant block of slots, so two
values always look apart) and counts each distinct row once (a row of
many copies is not a tight group), less twice its standard error
(delta method): S_low.

The tree is cut from the leaves up: a node is split when a judgeable
split below it has S_low at the threshold or over. The pieces are then
sewn back, the least separated pair of neighbouring pieces first, while
their S_low on the whole pieces is under the threshold (Hennig 2010):
what is left are the regions, a union of leaves of the cut each. A
branch with fewer rows of B than the floor is pruned. Its rows, and the
rows of the class outside the two halves, take the region of their
nearest labelled rows by vote.

Where the regions of different classes overlap is judged on the
regions alone: the separation of two regions is the distance between
their centroids in standard deviations of their rows along the line
between them. Single linkage on the pairs of different classes under
a threshold groups the regions into overlap zones; a region with no
such pair belongs to none.
"""

from __future__ import annotations

__all__ = [
    "ClassClustering",
    "Members",
    "OverlapZones",
    "RegionTree",
    "WardTree",
    "centroids",
    "cluster_class",
    "line_spread",
    "merge_pieces",
    "node_members",
    "node_separation",
    "overlap_pairs",
    "overlap_zones",
    "pair_separation",
    "region_stability",
    "regions_tree",
    "separated_cut",
    "separation",
    "split_halves",
    "vote",
    "ward_tree",
]

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

_CHUNK = 65536  # rows per block of line_spread


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
class RegionTree:
    """The tree of the regions of a class, its leaves the regions.

    The weighted Ward tree of the regions' centroids (regions_tree): it
    shows how the regions group, the regions themselves come from the
    cut and its sewing. Nodes are in pre-order, the left child first, so
    a parent comes before its children and the leaves come in region
    order.

    Attributes:
        parent: The parent of each node, -1 at the root, shape (m,),
            int64.
        region: The region of each leaf, 0.. in pre-order; -1 at an
            internal node. Shape (m,), int64.
    """

    parent: np.ndarray
    region: np.ndarray


@dataclass(frozen=True)
class OverlapZones:
    """The overlap zones: a forest, one tree per zone.

    Single linkage on the regions, linked only across classes: the pair
    of least separation joins first, and nothing joins at or past the
    threshold. A zone is so a connected component of the pairs of
    different classes under it, its tree the order in which they join;
    a region in no such pair is in no zone. Zones come by their least
    separation, ascending; the nodes of a zone in pre-order, the left
    child first, after those of the zones before it.

    Attributes:
        parent: The parent of each node, -1 at the root of a zone,
            shape (k,), int64.
        zone: The zone of each node, 0.., shape (k,), int64.
        region: The region of a leaf; -1 at an internal node. Shape
            (k,), int64.
        separation: The separation of the pair that made an internal
            node; nan at a leaf. Shape (k,), float64; never below that
            of a child.
        link: The two regions of that pair, the one under the left
            child first; -1 at a leaf. Shape (k, 2), int64.
    """

    parent: np.ndarray
    zone: np.ndarray
    region: np.ndarray
    separation: np.ndarray
    link: np.ndarray


@dataclass(frozen=True)
class Members:
    """The rows of B below every node of a tree built on A.

    Each row of B is attached to the leaf of its nearest row of A; the
    rows of B below a node are then a range of order, as its leaves are
    a range of the leaves in pre-order.

    Attributes:
        assign: The nearest row of A of every row of B, shape (n_B,),
            int64.
        order: The rows of B by the pre-order position of their row of
            A, shape (n_B,), int64.
        start: Where the rows of B of every node start in order, shape
            (2 n_A - 1,), int64.
        stop: Where they stop, shape (2 n_A - 1,), int64.
    """

    assign: np.ndarray
    order: np.ndarray
    start: np.ndarray
    stop: np.ndarray


@dataclass(frozen=True)
class ClassClustering:
    """The regions of one class.

    Attributes:
        labels: The region of every row of the class (of the rows of
            only when given), 0.. with no gap.
        sample: The rows of A and B, ascending: the rows the regions
            were judged on.
        n_pieces: The pieces of the cut before they were sewn.
        threshold: The separation the regions were sewn at:
            split_separation, raised when the class had to be capped.
        n_voted: The rows labelled by vote: the pruned rows of A and
            every row outside A.
        curve: The regions at each threshold of _CURVE and at
            split_separation, as (thresholds ascending, regions),
            from the same tree; empty when only was given.
    """

    labels: np.ndarray
    sample: np.ndarray
    n_pieces: int
    threshold: float
    n_voted: int
    curve: tuple[np.ndarray, np.ndarray]


def centroids(
    X: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted mean point of every label.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row; -1 rows are ignored.
        weights: The weight of every row, > 0; None weighs every row 1.

    Returns:
        The labels, ascending, and their centroids, shape (m, d).
    """
    X = np.asarray(X, dtype=np.float64)
    weights = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    kept = labels != -1
    ids, index = np.unique(labels[kept], return_inverse=True)
    mass = np.bincount(index, weights=weights[kept], minlength=len(ids))
    sums = np.zeros((len(ids), X.shape[1]))
    np.add.at(sums, index, X[kept] * weights[kept, None])
    return ids, sums / mass[:, None]


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


def split_halves(
    n: int, max_fit_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return two disjoint uniform samples of range(n): the halves A and B.

    A holds min(max_fit_samples, ceil(n / 2)) rows, B min(max_fit_samples,
    n - |A|) of the others.

    Args:
        n: The rows, >= 1.
        max_fit_samples: The most rows of each half, >= 1.
        seed: The seed of the draw.

    Returns:
        The rows of A and of B, each ascending, int64.
    """
    order = np.random.default_rng(seed).permutation(n)
    n_tree = min(max_fit_samples, (n + 1) // 2)
    n_measure = min(max_fit_samples, n - n_tree)
    tree_rows = np.sort(order[:n_tree]).astype(np.int64)
    measure_rows = np.sort(order[n_tree : n_tree + n_measure]).astype(np.int64)
    return tree_rows, measure_rows


def node_members(X_tree: np.ndarray, X_measure: np.ndarray, tree: WardTree) -> Members:
    """Attach every row of B to the nodes of the tree of A.

    Args:
        X_tree: The rows of A the tree was built on, shape (n_A, d).
        X_measure: The rows of B, shape (n_B, d).
        tree: ward_tree of X_tree.

    Returns:
        The members: each row of B under every ancestor of the leaf of
        its nearest row of A (brute-force euclidean search).
    """
    n = len(X_tree)
    leaves, start, stop = _leaf_ranges(tree.children, n)
    if len(X_measure):
        search = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1)
        assign = search.fit(X_tree).kneighbors(X_measure, return_distance=False)[:, 0]
    else:
        assign = np.empty(0, dtype=np.int64)
    position = np.empty(n, dtype=np.int64)
    position[leaves] = np.arange(n)
    placed = position[assign]
    order = np.argsort(placed, kind="stable")
    ranked = placed[order]
    return Members(
        assign=assign.astype(np.int64),
        order=order.astype(np.int64),
        start=np.searchsorted(ranked, start, "left").astype(np.int64),
        stop=np.searchsorted(ranked, stop, "left").astype(np.int64),
    )


def pair_separation(
    tree_a: np.ndarray,
    tree_b: np.ndarray,
    measure_a: np.ndarray,
    measure_b: np.ndarray,
) -> tuple[float, float]:
    """Return the separation S of two groups, and S_low = S - _Z SE.

    The line runs from the centroid of tree_a to that of tree_b; the
    rows of measure_a and measure_b, projected on it, give the means
    m_a, m_b and the variances V_a, V_b, each row counted once. S =
    (m_b - m_a) / sqrt(V_a + V_b), 0 when m_b <= m_a; its variance, by
    the delta method on the means and the variances (with the fourth
    central moments), gives SE.

    Args:
        tree_a: The rows of A of the first group, shape (k_a, d), k_a
            >= 1.
        tree_b: The rows of A of the second group, shape (k_b, d).
        measure_a: The rows of B of the first group, shape (n_a, d),
            n_a >= 2.
        measure_b: The rows of B of the second group, shape (n_b, d),
            n_b >= 2.

    Returns:
        S and S_low: (0, -inf) when the centroids of A coincide; (inf,
        inf) when the rows of B do not spread along the line and m_b >
        m_a.
    """
    line = np.mean(tree_b, axis=0) - np.mean(tree_a, axis=0)
    length = np.linalg.norm(line)
    if length == 0:
        return 0.0, -np.inf
    unit = line / length
    along_a, along_b = measure_a @ unit, measure_b @ unit
    mean_a, mean_b = along_a.mean(), along_b.mean()
    var_a = np.mean((along_a - mean_a) ** 2)
    var_b = np.mean((along_b - mean_b) ** 2)
    gap, total = mean_b - mean_a, var_a + var_b
    if total <= 0:
        return (np.inf, np.inf) if gap > 0 else (0.0, -np.inf)
    # Var of a variance: (mu4 - V^2) / n; of a mean: V / n.
    fourth_a = max(np.mean((along_a - mean_a) ** 4) - var_a**2, 0.0)
    fourth_b = max(np.mean((along_b - mean_b) ** 4) - var_b**2, 0.0)
    n_a, n_b = len(along_a), len(along_b)
    variance = (var_a / n_a + var_b / n_b) / total + gap**2 / (4 * total**3) * (
        fourth_a / n_a + fourth_b / n_b
    )
    value = max(gap / np.sqrt(total), 0.0)
    return float(value), float(value - _Z * np.sqrt(variance))


def node_separation(
    X_tree: np.ndarray,
    X_measure: np.ndarray,
    tree: WardTree,
    members: Members,
    floor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return S and S_low of the split of every merge of the tree.

    A merge is judgeable when each of its two children holds at least
    floor rows of B; pair_separation of the two children then, their
    rows of A and of B.

    Args:
        X_tree: The rows of A, shape (n_A, d): the columns S is
            measured on.
        X_measure: The rows of B, the same columns.
        tree: The tree of A.
        members: node_members of the tree.
        floor: The fewest rows of B of each child, >= 2.

    Returns:
        S and S_low, shape (n_A - 1,) each, nan where the merge is not
        judgeable.
    """
    n = len(X_tree)
    leaves, start, stop = _leaf_ranges(tree.children, n)
    count = members.stop - members.start
    S = np.full(len(tree.children), np.nan)
    low = np.full(len(tree.children), np.nan)
    for j, (a, b) in enumerate(tree.children):
        if count[a] < floor or count[b] < floor:
            continue
        S[j], low[j] = pair_separation(
            X_tree[leaves[start[a] : stop[a]]],
            X_tree[leaves[start[b] : stop[b]]],
            X_measure[members.order[members.start[a] : members.stop[a]]],
            X_measure[members.order[members.start[b] : members.stop[b]]],
        )
    return S, low


def separated_cut(
    tree: WardTree,
    members: Members,
    floor: int,
    separation_low: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Cut the tree from the leaves up at every separated split.

    A node is split when a judgeable merge of its subtree (itself
    included) has S_low >= threshold; a child with fewer than floor rows
    of B is pruned, and its node goes on as its other child does.

    Args:
        tree: The tree of A.
        members: node_members of the tree.
        floor: The fewest rows of B of a child.
        separation_low: S_low of every merge (node_separation).
        threshold: The separation a split must reach.

    Returns:
        The piece of every row of A, 0.. in the leaf order of the
        tree; -1 for a pruned row.
    """
    n = len(tree.children) + 1
    small = (members.stop - members.start) < floor
    # Whether a judgeable merge of each node's subtree reaches threshold,
    # its pruned branches left out; children come before their parents.
    separated = np.zeros(2 * n - 1, dtype=bool)
    for j, (a, b) in enumerate(tree.children):
        v = n + j
        if small[a] and small[b]:
            continue
        if small[a] or small[b]:
            separated[v] = separated[b if small[a] else a]
        else:
            separated[v] = (
                separated[a] or separated[b] or separation_low[j] >= threshold
            )
    leaves, start, stop = _leaf_ranges(tree.children, n)
    pieces = np.full(n, -1, dtype=np.int64)
    n_pieces = 0
    stack = [2 * n - 2]
    while stack:
        v = stack.pop()
        if v >= n:
            a, b = tree.children[v - n]
            if small[a] != small[b]:
                stack.append(b if small[a] else a)  # the small child is pruned
                continue
            if not small[a] and separated[v]:
                stack += [b, a]
                continue
        pieces[leaves[start[v] : stop[v]]] = n_pieces
        n_pieces += 1
    return pieces


def merge_pieces(
    X_tree: np.ndarray,
    X_measure: np.ndarray,
    pieces: np.ndarray,
    assign: np.ndarray,
    threshold: float,
    max_regions: int,
) -> tuple[np.ndarray, float]:
    """Sew the pieces whose sides are not separated.

    Each piece is compared with its _NEIGHBOURS nearest pieces (by the
    centroids of their rows of A); the pair of least S_low
    (pair_separation, the rows of B by the piece of their nearest row
    of A) merges and its comparisons are made again, while that least
    S_low is under threshold, then past it while more than max_regions
    pieces are left, and no further once past it.

    Args:
        X_tree: The rows of A, the columns S is measured on.
        X_measure: The rows of B, the same columns.
        pieces: The piece of every row of A, -1 for a pruned one
            (separated_cut).
        assign: The nearest row of A of every row of B (Members.assign).
        threshold: The separation the regions must keep.
        max_regions: The most regions, >= 1.

    Returns:
        The region of every row of A (-1 where the piece was -1), 0..
        with no gap, and the threshold the sewing reached: threshold,
        or the S_low of the last merge when the cap forced it past.
    """
    labels = np.asarray(pieces, dtype=np.int64)
    of_measure = np.where(labels[assign] >= 0, labels[assign], -1)
    ids = [int(p) for p in np.unique(labels) if p >= 0]
    rows = {p: np.flatnonzero(labels == p) for p in ids}
    measured = {p: np.flatnonzero(of_measure == p) for p in ids}
    center = {p: X_tree[rows[p]].mean(axis=0) for p in ids}
    known: dict[tuple[int, int], float] = {}

    def low(p: int, q: int) -> float:
        key = (min(p, q), max(p, q))
        if key not in known:
            if len(measured[p]) < 2 or len(measured[q]) < 2:
                known[key] = -np.inf  # too few rows to tell apart
            else:
                known[key] = pair_separation(
                    X_tree[rows[p]],
                    X_tree[rows[q]],
                    X_measure[measured[p]],
                    X_measure[measured[q]],
                )[1]
        return known[key]

    def nearest(p: int) -> list[int]:
        others = [q for q in ids if q != p]
        if not others:
            return []
        distance = np.linalg.norm(
            np.array([center[q] for q in others]) - center[p], axis=1
        )
        return [others[i] for i in np.argsort(distance, kind="stable")[:_NEIGHBOURS]]

    near = {p: nearest(p) for p in ids}
    reached, forced = threshold, False
    while len(ids) > 1:
        best, pair = np.inf, None
        for p in ids:
            for q in near[p]:
                value = low(p, q)
                if value < best:
                    best, pair = value, (p, q)
        # Once the cap forced a merge, a union of separated pieces looks
        # close to the rest: stop as soon as the cap holds.
        if pair is None or (len(ids) <= max_regions and (best >= threshold or forced)):
            break
        if best >= threshold:
            reached, forced = max(reached, float(best)), True
        p, q = pair
        rows[p] = np.concatenate([rows[p], rows.pop(q)])
        measured[p] = np.concatenate([measured[p], measured.pop(q)])
        center[p] = X_tree[rows[p]].mean(axis=0)
        ids.remove(q)
        del near[q]
        known = {key: v for key, v in known.items() if p not in key and q not in key}
        for r in ids:
            if (
                r == p
                or p in near[r]
                or q in near[r]
                or len(near[r]) < min(_NEIGHBOURS, len(ids) - 1)
            ):
                near[r] = nearest(r)
    regions = np.full(len(labels), -1, dtype=np.int64)
    for region, p in enumerate(ids):
        regions[rows[p]] = region
    return regions, reached


def vote(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Label the unlabelled rows by the majority of their nearest labelled rows.

    Ties go to the label of the nearest among the tied.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row, -1 to vote; at least one >= 0.
        k: The labelled neighbours that vote.

    Returns:
        The labels, every one >= 0; labelled rows keep theirs.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    known = np.flatnonzero(labels >= 0)
    unknown = np.flatnonzero(labels < 0)
    if len(unknown) == 0:
        return labels
    if len(np.unique(labels[known])) == 1:
        labels[unknown] = labels[known[0]]
        return labels
    search = NearestNeighbors(
        n_neighbors=min(k, len(known)), algorithm="brute", n_jobs=-1
    ).fit(X[known])
    for start in range(0, len(unknown), _VOTE_ROWS):
        rows = unknown[start : start + _VOTE_ROWS]
        votes = labels[known][search.kneighbors(X[rows], return_distance=False)]
        # Neighbours come nearest first: the first label of the most
        # votes is the nearest of the tied.
        support = (votes[:, :, None] == votes[:, None, :]).sum(axis=2)
        winner = np.argmax(support == support.max(axis=1, keepdims=True), axis=1)
        labels[rows] = votes[np.arange(len(rows)), winner]
    return labels


_VOTE_ROWS = 100_000  # rows voted at once


def regions_tree(
    X: np.ndarray, weights: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, RegionTree]:
    """Return the regions in pre-order of their tree, and that tree.

    The tree is ward_tree of the weighted centroids of the regions,
    each weighted by the summed weights of its rows; its leaves are the
    regions, renumbered 0.. in pre-order.

    Args:
        X: The points, shape (n, d).
        weights: The weight of every row, > 0.
        labels: The region of every row, 0.. with no gap.

    Returns:
        The labels renumbered and the tree.
    """
    labels = np.asarray(labels, dtype=np.int64)
    ids, centers = centroids(X, labels, weights)
    m = len(ids)
    if m == 1:
        return np.zeros(len(labels), dtype=np.int64), RegionTree(
            np.array([-1], dtype=np.int64), np.array([0], dtype=np.int64)
        )
    mass = np.bincount(labels, weights=weights, minlength=m)
    tree = ward_tree(centers, mass)
    renamed = np.empty(m, dtype=np.int64)
    parent, region = [], []
    n_leaves = 0
    stack = [(2 * m - 2, -1)]
    while stack:
        v, up = stack.pop()
        parent.append(up)
        if v < m:
            renamed[v] = n_leaves
            region.append(n_leaves)
            n_leaves += 1
            continue
        here = len(parent) - 1
        region.append(-1)
        a, b = tree.children[v - m]
        stack += [(b, here), (a, here)]
    return renamed[labels], RegionTree(
        np.asarray(parent, dtype=np.int64), np.asarray(region, dtype=np.int64)
    )


def region_stability(labels: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Return how well each region is found again in another partition.

    The stability of region r is max over the regions s of other of
    |r ∩ s| / |r ∪ s| (Jaccard) on the same rows (Hennig 2007): 0.75 or
    more stable, under 0.6 not to be trusted.

    Args:
        labels: The region of every row, 0.. with no gap.
        other: The region of the same rows in the other partition.

    Returns:
        The stability of every region of labels, in [0, 1].
    """
    labels = np.asarray(labels, dtype=np.int64)
    other = np.asarray(other, dtype=np.int64)
    both = np.zeros((labels.max() + 1, other.max() + 1))
    np.add.at(both, (labels, other), 1.0)
    union = both.sum(axis=1)[:, None] + both.sum(axis=0)[None, :] - both
    return (both / union).max(axis=1)


def cluster_class(
    X: np.ndarray,
    numeric: np.ndarray,
    weights: np.ndarray,
    *,
    max_fit_samples: int,
    min_cluster_floor: int,
    max_clusters: int,
    split_separation: float,
    vote_k: int,
    seed: int,
    only: np.ndarray | None = None,
) -> ClassClustering:
    """Split one class into its regions.

    Order: split_halves -> ward_tree of X[A], weighted -> node_members
    (every column) -> node_separation (the numeric columns) ->
    separated_cut at split_separation -> merge_pieces with max_clusters
    -> vote of every row without a region among the labelled rows of A
    (every column). The curve repeats the cut and the sewing at each
    threshold of _CURVE on the same tree.

    Args:
        X: The points of the class, shape (n, d).
        numeric: Which columns of X are numeric, shape (d,), bool: S is
            measured on them only.
        weights: The weight of every row, > 0 (its copies in the CSV).
        max_fit_samples: The most rows of each half.
        min_cluster_floor: The fewest rows of B of a child of a split.
        max_clusters: The most regions, >= 1.
        split_separation: The separation two regions must keep, > 0.
        vote_k: The labelled neighbours that vote.
        seed: The seed of the halves.
        only: The rows to label, ascending; None labels every row.

    Returns:
        The regions.
    """
    n = len(X)
    tree_rows, measure_rows = split_halves(n, max_fit_samples, seed)
    X_tree, X_measure = X[tree_rows], X[measure_rows]
    tree = ward_tree(X_tree, weights[tree_rows])
    members = node_members(X_tree, X_measure, tree)
    numbers_tree, numbers_measure = X_tree[:, numeric], X_measure[:, numeric]
    _, low = node_separation(
        numbers_tree, numbers_measure, tree, members, min_cluster_floor
    )

    def regions_at(threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
        pieces = separated_cut(tree, members, min_cluster_floor, low, threshold)
        regions, reached = merge_pieces(
            numbers_tree,
            numbers_measure,
            pieces,
            members.assign,
            threshold,
            max_clusters,
        )
        return pieces, regions, reached

    pieces, regions, reached = regions_at(split_separation)
    every = np.full(n, -1, dtype=np.int64)
    every[tree_rows] = regions
    if only is None:
        labels = vote(X, every, vote_k)
        curve_at = sorted({*_CURVE, split_separation})
        counts = [
            int((regions if t == split_separation else regions_at(t)[1]).max()) + 1
            for t in curve_at
        ]
        curve = (
            np.asarray(curve_at, dtype=np.float64),
            np.asarray(counts, dtype=np.int64),
        )
    else:
        only = np.asarray(only, dtype=np.int64)
        rows = np.union1d(tree_rows[regions >= 0], only)
        labels = vote(X[rows], every[rows], vote_k)[np.searchsorted(rows, only)]
        curve = (np.empty(0), np.empty(0, dtype=np.int64))
    return ClassClustering(
        labels=labels,
        sample=np.union1d(tree_rows, measure_rows),
        n_pieces=int(len(np.unique(pieces[pieces >= 0]))),
        threshold=float(reached),
        n_voted=int(n - np.sum(regions >= 0)),
        curve=curve,
    )


_Z = 2.0  # standard errors taken off S
_NEIGHBOURS = 10  # nearest pieces each piece is compared with when sewing
_CURVE = (2.0, 3.0, 4.0, 5.0, 6.0)  # thresholds of the curve of a class


def line_spread(
    X: np.ndarray, labels: np.ndarray, weights: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Return the spread of each label's rows along the line to every other centroid.

    spread[a, b] is the weighted standard deviation of the rows of a
    projected on the unit vector from c_a to c_b: sqrt(u' Σ_a u), Σ_a
    the weighted covariance of a's rows, summed over them in blocks.
    One covariance is held at a time, so memory is O(d² + m²).

    Args:
        X: The points, shape (n, d).
        labels: The label of every row, 0..m - 1, each held by a row.
        weights: The weight of every row, > 0.
        centers: The weighted centroid of every label, shape (m, d).

    Returns:
        spread, shape (m, m), >= 0; 0 on the diagonal and where two
        centroids coincide.
    """
    centers = np.asarray(centers, dtype=np.float64)
    m = len(centers)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(m + 1))
    spread = np.zeros((m, m))
    for a in range(m):
        rows = order[bounds[a] : bounds[a + 1]]
        cov = np.zeros((centers.shape[1], centers.shape[1]))
        for start in range(0, len(rows), _CHUNK):
            part = rows[start : start + _CHUNK]
            diff = np.asarray(X[part], dtype=np.float64) - centers[a]
            cov += diff.T @ (diff * weights[part, None])
        cov /= weights[rows].sum()
        line = centers - centers[a]
        length = np.linalg.norm(line, axis=1)
        apart = length > 0
        unit = line[apart] / length[apart, None]
        spread[a, apart] = np.sqrt(np.maximum(np.sum((unit @ cov) * unit, 1), 0.0))
    return spread


def separation(centers: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """Return the separation of every pair of labels.

    S[a, b] = ||c_a - c_b|| / sqrt(spread[a, b]² + spread[b, a]²): the
    distance between the centroids in standard deviations of the two
    labels' rows along the line between them (the square root of
    Fisher's discriminant ratio on that line).

    Args:
        centers: The centroid of every label, shape (m, d).
        spread: line_spread of the labels, shape (m, m).

    Returns:
        S, shape (m, m), symmetric: 0 where the centroids coincide (the
        diagonal too), inf where they differ and neither label spreads
        along the line.
    """
    centers = np.asarray(centers, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    # Row by row, not by the Gram matrix: coinciding centroids must be
    # exactly 0 apart.
    distance = np.array([np.linalg.norm(centers - c, axis=1) for c in centers])
    scale = np.sqrt(spread**2 + spread.T**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(distance == 0, 0.0, distance / scale)


def overlap_pairs(
    separation: np.ndarray, classes: np.ndarray, threshold: float
) -> np.ndarray:
    """Return the pairs of regions of different classes that overlap.

    Args:
        separation: The separation of every pair of regions, shape
            (m, m), symmetric.
        classes: The class of every region, shape (m,).
        threshold: The separation a pair must lie under, > 0.

    Returns:
        The pairs (a, b), a < b, with separation[a, b] < threshold and
        classes[a] != classes[b], by separation then a then b; shape
        (p, 2), int64.
    """
    classes = np.asarray(classes)
    under = (separation < threshold) & (classes[:, None] != classes[None, :])
    a, b = np.nonzero(np.triu(under, 1))
    order = np.lexsort((b, a, separation[a, b]))
    return np.stack([a[order], b[order]], axis=1).astype(np.int64).reshape(-1, 2)


def overlap_zones(
    separation: np.ndarray, classes: np.ndarray, threshold: float
) -> OverlapZones:
    """Group the regions into overlap zones by single linkage.

    The pairs of overlap_pairs join in their order (Kruskal): each that
    links two groups makes a node over them, the group of a on the left.

    Args:
        separation: The separation of every pair of regions, shape
            (m, m), symmetric.
        classes: The class of every region, shape (m,).
        threshold: The separation a pair must lie under, > 0.

    Returns:
        The zones; empty arrays when no pair overlaps.
    """
    m = len(classes)
    group = np.arange(m)  # union-find over the regions

    def find(x: int) -> int:
        while group[x] != x:
            group[x] = group[group[x]]
            x = group[x]
        return x

    node = list(range(m))  # the top node of each group, by its representative
    first = [np.inf] * m  # the first merge of each group
    merges = []  # (left node, right node, separation, a, b); merge j is node m + j
    for a, b in overlap_pairs(separation, classes, threshold):
        top_a, top_b = find(a), find(b)
        if top_a == top_b:
            continue
        merges.append((node[top_a], node[top_b], float(separation[a, b]), a, b))
        group[top_b] = top_a
        node[top_a] = m + len(merges) - 1
        first[top_a] = min(first[top_a], first[top_b], len(merges) - 1)
    tops = sorted(
        (first[r], node[r]) for r in range(m) if find(r) == r and node[r] >= m
    )
    parent, zone, region, apart, link = [], [], [], [], []
    for z, (_, top) in enumerate(tops):
        stack = [(top, -1)]
        while stack:
            v, up = stack.pop()
            parent.append(up)
            zone.append(z)
            if v < m:
                region.append(v)
                apart.append(np.nan)
                link.append((-1, -1))
                continue
            left, right, value, a, b = merges[v - m]
            stack += [(right, len(parent) - 1), (left, len(parent) - 1)]
            region.append(-1)
            apart.append(value)
            link.append((a, b))
    return OverlapZones(
        parent=np.asarray(parent, dtype=np.int64),
        zone=np.asarray(zone, dtype=np.int64),
        region=np.asarray(region, dtype=np.int64),
        separation=np.asarray(apart, dtype=np.float64),
        link=np.asarray(link, dtype=np.int64).reshape(-1, 2),
    )


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
