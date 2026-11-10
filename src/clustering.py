"""Regions: each class split by a weighted Ward tree, cut at least loss.

The rows of a class are deduplicated, each weighted by its copies in
the CSV, so the weighted Ward tree of the distinct rows is the Ward
tree of the rows as read. The tree is cut where the granularity loss
is least, branch by branch: a finer region hides less of the spread of
its rows' hardness inside its mean, but leaves fewer rows to estimate
that mean on. Branches with fewer distinct rows than the floor are
pruned; their rows, and the rows of the class outside the sample the
tree was built on, take the region of their nearest labelled rows by
vote. Hardness is kDN: the share of a row's nearest graph nodes of
another class.

The regions of every class then grow one more Ward tree, the class
ignored: the overlap tree, where the classes meet.
"""

from __future__ import annotations

__all__ = [
    "Candidate",
    "ClassClustering",
    "Contact",
    "OverlapTree",
    "RegionTree",
    "WardTree",
    "best_cut",
    "centroids",
    "cluster_class",
    "contacts",
    "kdn",
    "overlap_tree",
    "region_loss",
    "vote",
    "ward_tree",
    "within_ss",
]

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

_CHUNK = 65536  # rows per block of within_ss


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
    """The tree a cut keeps: the root, the splits it makes, its regions.

    Nodes are in pre-order, the left child first, so a parent comes
    before its children and the leaves come in region order.

    Attributes:
        parent: The parent of each node, -1 at the root, shape (m,),
            int64.
        region: The region of each leaf, 0.. in pre-order; -1 at an
            internal node. Shape (m,), int64.
    """

    parent: np.ndarray
    region: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One penalty of the granularity curve of a class.

    Attributes:
        penalty: The penalty per region of the cut.
        n_clusters: The regions of the cut.
        loss: region_loss of the scored rows after the vote.
    """

    penalty: float
    n_clusters: int
    loss: float


@dataclass(frozen=True)
class ClassClustering:
    """The regions of one class.

    Attributes:
        labels: The region of every row of the class, 0.. in the leaf
            order of tree, no gap.
        tree: The tree of the cut.
        penalty: The penalty per region of the cut (0 unless the
            regions had to be capped).
        loss: region_loss of the scored rows after the vote.
        n_voted: The rows labelled by vote: the pruned scored rows and
            the rows outside the sample.
        candidates: The granularity curve, by descending penalty.
    """

    labels: np.ndarray
    tree: RegionTree
    penalty: float
    loss: float
    n_voted: int
    candidates: list[Candidate]


@dataclass(frozen=True)
class OverlapTree:
    """The Ward tree of the regions of every class, the class ignored.

    Ward needs only the weighted centroid and the mass of each cluster,
    so the tree grown from the regions is the Ward tree their rows would
    grow from there: the class trees continued across the classes.
    Nodes are in pre-order, the left child first, so a parent comes
    before its children and the leaves come in the order of the tree.

    Attributes:
        parent: The parent of each node, -1 at the root, shape (m,),
            int64.
        region: The region of a leaf (its row in the centroids); -1 at
            an internal node. Shape (m,), int64.
        height: The merge height of an internal node, sqrt(2 Δ) as in
            WardTree; 0 at a leaf. Shape (m,), float64.
        distance: The distance between the centroids of an internal
            node's two children; nan at a leaf. Shape (m,), float64.
        ratio: distance / (r_left + r_right), r the radius of a child:
            the root mean square distance of its rows (weighted) from
            its centroid. Under 1 the two spreads overlap. nan at a
            leaf; with both radii 0, 0 on one point and inf apart.
            Shape (m,), float64.
    """

    parent: np.ndarray
    region: np.ndarray
    height: np.ndarray
    distance: np.ndarray
    ratio: np.ndarray


@dataclass(frozen=True)
class Contact:
    """A node of the overlap tree where two classes meet.

    One class lies only below the node's left child, the other only
    below its right child: neither child holds both.

    Attributes:
        node: The overlap tree node.
        classes: The two classes, the left child's first.
        regions: The regions of each class below node, ascending.
        distance: The distance between the centroids of the two
            classes' rows below node.
        ratio: distance / (r_0 + r_1), r the radius of each class's
            rows below node (as in OverlapTree; with both radii 0, 0
            on one point and inf apart).
    """

    node: int
    classes: tuple[int, int]
    regions: tuple[np.ndarray, np.ndarray]
    distance: float
    ratio: float


def kdn(neighbor_labels: np.ndarray, label: int) -> np.ndarray:
    """Return the kDN hardness of rows of one class.

    Args:
        neighbor_labels: The class of each row's nearest graph nodes,
            shape (n, k), k >= 1.
        label: The class of the rows.

    Returns:
        The share of each row's neighbours of another class, in [0, 1].
    """
    return (neighbor_labels != label).mean(axis=1)


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


def within_ss(
    X: np.ndarray, labels: np.ndarray, weights: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Return the weighted sum of squared distances of each label's rows from its centroid.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row, 0..m - 1.
        weights: The weight of every row, > 0.
        centers: The weighted centroid of every label, shape (m, d).

    Returns:
        Σ w ||x - c||² per label, shape (m,).
    """
    ss = np.zeros(len(centers))
    for start in range(0, len(X), _CHUNK):
        part = slice(start, start + _CHUNK)
        diff = np.asarray(X[part], dtype=np.float64) - centers[labels[part]]
        ss += np.bincount(
            labels[part],
            weights=weights[part] * np.einsum("ij,ij->i", diff, diff),
            minlength=len(centers),
        )
    return ss


def overlap_tree(
    centers: np.ndarray, masses: np.ndarray, scatter: np.ndarray
) -> OverlapTree:
    """Grow the Ward tree of the regions, each a cluster of its rows.

    Args:
        centers: The weighted centroid of every region, shape (m, d),
            m >= 1.
        masses: The summed weight of every region's rows, > 0.
        scatter: within_ss of every region.

    Returns:
        The tree; a single leaf when m = 1.
    """
    centers = np.asarray(centers, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    m = len(centers)
    ward = ward_tree(centers, masses)
    n_nodes = 2 * m - 1
    center = np.empty((n_nodes, centers.shape[1]))
    mass, ss = np.empty(n_nodes), np.empty(n_nodes)
    center[:m], mass[:m], ss[:m] = centers, masses, scatter
    distance, ratio = np.full(n_nodes, np.nan), np.full(n_nodes, np.nan)
    for j, (a, b) in enumerate(ward.children):
        v = m + j
        mass[v] = mass[a] + mass[b]
        center[v] = (mass[a] * center[a] + mass[b] * center[b]) / mass[v]
        squared = float(np.sum((center[a] - center[b]) ** 2))
        ss[v] = ss[a] + ss[b] + mass[a] * mass[b] / mass[v] * squared
        distance[v] = np.sqrt(squared)
        ratio[v] = _ratio(
            distance[v], np.sqrt(ss[a] / mass[a]) + np.sqrt(ss[b] / mass[b])
        )
    height = np.concatenate([np.zeros(m), ward.heights])
    order, parent = [], []
    stack = [(n_nodes - 1, -1)]
    while stack:
        v, up = stack.pop()
        parent.append(up)
        if v >= m:
            a, b = ward.children[v - m]
            stack += [(b, len(order)), (a, len(order))]
        order.append(v)
    order = np.asarray(order, dtype=np.int64)
    return OverlapTree(
        parent=np.asarray(parent, dtype=np.int64),
        region=np.where(order < m, order, -1),
        height=height[order],
        distance=distance[order],
        ratio=ratio[order],
    )


def contacts(
    tree: OverlapTree,
    classes: np.ndarray,
    centers: np.ndarray,
    masses: np.ndarray,
    scatter: np.ndarray,
) -> list[Contact]:
    """Return every node where two classes meet, in the node order of tree.

    A node with k classes on one side and l on the other is a contact
    of each pair (one class from each side) that neither child holds
    both of.

    Args:
        tree: overlap_tree of the regions.
        classes: The class of every region.
        centers: The weighted centroid of every region, shape (m, d).
        masses: The summed weight of every region's rows, > 0.
        scatter: within_ss of every region.

    Returns:
        The contacts, by node, then by the pair of classes.
    """
    n_nodes = len(tree.parent)
    kids: list[list[int]] = [[] for _ in range(n_nodes)]
    for v in range(1, n_nodes):
        kids[tree.parent[v]].append(v)  # ascending: the left child first
    members: list[dict[int, list[int]]] = [{} for _ in range(n_nodes)]
    for v in range(n_nodes - 1, -1, -1):
        if tree.region[v] >= 0:
            members[v] = {int(classes[tree.region[v]]): [int(tree.region[v])]}
            continue
        for kid in kids[v]:
            for code, regions in members[kid].items():
                members[v].setdefault(code, []).extend(regions)
    found = []
    for v in np.flatnonzero(tree.region < 0):
        left, right = (members[kid] for kid in kids[v])
        for a in sorted(left):
            for b in sorted(right):
                if a == b or b in left or a in right:
                    continue
                parts = (np.sort(left[a]), np.sort(right[b]))
                (c_a, r_a), (c_b, r_b) = (
                    _spread(part, centers, masses, scatter) for part in parts
                )
                distance = float(np.linalg.norm(c_a - c_b))
                found.append(
                    Contact(
                        int(v), (a, b), parts, distance, _ratio(distance, r_a + r_b)
                    )
                )
    return found


def best_cut(
    tree: WardTree,
    weights: np.ndarray,
    hardness: np.ndarray,
    *,
    floor: float,
    penalty: float = 0.0,
) -> tuple[np.ndarray, RegionTree]:
    """Return the cut of the tree of least loss plus penalty per region.

    A node v taken as a region costs (W_v / W) · (var_v + max(h_v (1 -
    h_v), 1e-3) / n_v) + penalty: W the summed weights, h_v and var_v
    the weighted mean and variance (times n_v / (n_v - 1)) of the
    hardness of its n_v rows; the variance of all rows when n_v = 1. The
    noise counts the rows, not their weights: copies share one hardness
    and add nothing to its estimate.
    A child with fewer than floor rows is no split: v divides as its
    other child does and the rows of the small one get no region. The
    least sum over every cut is found from the leaves up.

    Args:
        tree: The Ward tree of the rows.
        weights: The weight of every row, > 0.
        hardness: The hardness of every row, in [0, 1].
        floor: The fewest rows of a region.
        penalty: The cost added per region, >= 0.

    Returns:
        The region of every row (-1 for a pruned one; 0.. in the leaf
        order of the tree) and the tree of the cut.
    """
    weights = np.asarray(weights, dtype=np.float64)
    hardness = np.asarray(hardness, dtype=np.float64)
    n = len(weights)
    n_nodes = 2 * n - 1
    count, mass = np.ones(n_nodes), np.empty(n_nodes)
    first, second = np.empty(n_nodes), np.empty(n_nodes)
    mass[:n], first[:n] = weights, weights * hardness
    second[:n] = weights * hardness**2
    for j, (a, b) in enumerate(tree.children):
        v = n + j
        count[v] = count[a] + count[b]
        mass[v] = mass[a] + mass[b]
        first[v] = first[a] + first[b]
        second[v] = second[a] + second[b]
    cost = _costs(count, mass, first, second) + penalty
    best, split = cost.copy(), np.full(n_nodes, np.inf)
    via = np.full(n_nodes, -1)  # the node whose two children a split keeps
    small = count < floor
    for j, (a, b) in enumerate(tree.children):
        v = n + j
        if small[a] and small[b]:
            continue
        if small[a] or small[b]:
            big = b if small[a] else a
            split[v], via[v] = split[big], via[big]
        else:
            split[v], via[v] = best[a] + best[b], v
        best[v] = min(cost[v], split[v])
    # The rows below each node are a range of the leaves in pre-order.
    leaves, start, stop = _leaf_ranges(tree.children, n)
    labels = np.full(n, -1, dtype=np.int64)
    parent, region = [], []
    n_regions = 0
    stack = [(n_nodes - 1, -1)]
    while stack:
        v, up = stack.pop()
        parent.append(up)
        if split[v] < cost[v]:
            region.append(-1)
            here = len(parent) - 1
            a, b = tree.children[via[v] - n]
            stack += [(b, here), (a, here)]
        else:
            labels[leaves[start[v] : stop[v]]] = n_regions
            region.append(n_regions)
            n_regions += 1
    return labels, RegionTree(
        np.asarray(parent, dtype=np.int64), np.asarray(region, dtype=np.int64)
    )


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


def region_loss(labels: np.ndarray, hardness: np.ndarray, weights: np.ndarray) -> float:
    """Return the granularity loss of a partition.

    It is the cost of best_cut summed over the regions, without penalty.

    Args:
        labels: The region of every row, all >= 0.
        hardness: The hardness of every row.
        weights: The weight of every row, > 0.

    Returns:
        The loss, >= 0.
    """
    weights = np.asarray(weights, dtype=np.float64)
    hardness = np.asarray(hardness, dtype=np.float64)
    count = np.bincount(labels).astype(np.float64)
    mass = np.bincount(labels, weights=weights)
    first = np.bincount(labels, weights=weights * hardness)
    second = np.bincount(labels, weights=weights * hardness**2)
    used = count > 0
    whole = (len(labels), weights.sum(), first.sum(), second.sum())
    costs = _costs(count[used], mass[used], first[used], second[used], whole=whole)
    return float(costs.sum())


def cluster_class(
    X: np.ndarray,
    weights: np.ndarray,
    scored: np.ndarray,
    hardness: np.ndarray,
    *,
    min_cluster_floor: int,
    max_clusters: int,
    vote_k: int,
) -> ClassClustering:
    """Split one class into the regions of least loss.

    Order: ward_tree of X[scored] -> best_cut with floor =
    min_cluster_floor and penalty 0, raised by bisection when it gives
    more than max_clusters regions -> vote of every row without a region
    among the labelled scored rows -> region_loss of the scored rows.
    The floor counts scored rows, whose hardness is measured, and the
    vote only adds rows, so every region ends with at least
    min_cluster_floor rows unless the class has fewer. The candidates are the cuts
    at penalty L · 2^-j, j = 0..15, and 0, L the loss of the whole class
    as one region; of the cuts with the same count of regions the one of
    least penalty, each scored after its vote.

    Args:
        X: The points of the class, shape (n, d).
        weights: The weight of every row, > 0.
        scored: The rows the tree is built on (graph.sample_rows).
        hardness: The kDN of every scored row.
        min_cluster_floor: The fewest scored rows of a region.
        max_clusters: The most regions, >= 1.
        vote_k: The labelled neighbours that vote.

    Returns:
        The regions.
    """
    n = len(X)
    sample, sample_weights = X[scored], weights[scored]
    tree = ward_tree(sample, sample_weights)

    def cut(penalty: float) -> tuple[np.ndarray, RegionTree]:
        return best_cut(
            tree, sample_weights, hardness, floor=min_cluster_floor, penalty=penalty
        )

    def n_regions(region_tree: RegionTree) -> int:
        return int((region_tree.region >= 0).sum())

    whole = region_loss(np.zeros(len(scored), dtype=np.int64), hardness, sample_weights)
    penalty = 0.0
    labels, region_tree = cut(penalty)
    if n_regions(region_tree) > max_clusters:
        # A penalty of the whole loss leaves one region; the count of
        # regions never grows with the penalty.
        low, high = 0.0, whole
        for _ in range(_BISECTIONS):
            middle = (low + high) / 2
            if n_regions(cut(middle)[1]) > max_clusters:
                low = middle
            else:
                high = middle
        penalty = high
        labels, region_tree = cut(penalty)
    every = np.full(n, -1, dtype=np.int64)
    every[scored] = labels
    voted = vote(X, every, vote_k)
    loss = region_loss(voted[scored], hardness, sample_weights)

    by_count: dict[int, Candidate] = {}
    for tried in [*(whole * 2.0**-j for j in range(_CURVE_STEPS)), 0.0]:
        tried_labels, tried_tree = cut(tried)
        tried_labels = vote(sample, tried_labels, vote_k)
        by_count[n_regions(tried_tree)] = Candidate(
            tried,
            n_regions(tried_tree),
            region_loss(tried_labels, hardness, sample_weights),
        )
    return ClassClustering(
        labels=voted,
        tree=region_tree,
        penalty=penalty,
        loss=loss,
        n_voted=int((every < 0).sum()),
        candidates=sorted(by_count.values(), key=lambda c: -c.penalty),
    )


_VOTE_ROWS = 100_000  # rows voted at once
_BISECTIONS = 50  # halvings of the penalty that caps the regions
_CURVE_STEPS = 16  # penalties of the granularity curve, halving from the whole loss


def _costs(
    count: np.ndarray,
    mass: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    whole: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Return the cost of best_cut of every node from its sums.

    count, mass, first and second are the rows, the weights, and the
    weighted sums of the hardness and of its square of each node; whole
    those of all rows, by default the last node (the root of a tree).
    """
    whole_count, whole_mass, whole_first, whole_second = (
        whole if whole is not None else (count[-1], mass[-1], first[-1], second[-1])
    )

    def variance(n, w, s1, s2):
        spread = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
        return spread * n / np.maximum(n - 1, 1)

    overall = variance(whole_count, whole_mass, whole_first, whole_second)
    mean = first / mass
    var = np.where(count > 1, variance(count, mass, first, second), overall)
    noise = np.maximum(mean * (1 - mean), 1e-3) / count
    return mass / whole_mass * (var + noise)


def _spread(
    regions: np.ndarray, centers: np.ndarray, masses: np.ndarray, scatter: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the weighted centroid and the radius of the rows of some regions."""
    weight = masses[regions]
    mass = weight.sum()
    center = weight @ centers[regions] / mass
    ss = scatter[regions].sum() + weight @ np.sum((centers[regions] - center) ** 2, 1)
    return center, float(np.sqrt(max(ss, 0.0) / mass))


def _ratio(distance: float, radii: float) -> float:
    """Return distance / radii: 0 on one point, inf apart when radii is 0."""
    if radii > 0:
        return float(distance / radii)
    return 0.0 if distance == 0 else float("inf")


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
