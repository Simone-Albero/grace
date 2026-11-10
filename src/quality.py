"""Data-quality measures on regions and on pairs of regions of different classes.

After Flood, Engelen, Aspinall & Desmet, "Bad Design Smells in
Benchmark NIDS Datasets" (EuroS&P 2024), measured on the regions of
`src.clustering` rather than on whole classes:
- label conflicts: rows equal on every feature under different
  classes (label_conflicts, pair_conflicts);
- overlap: how much two regions of different classes mix, from the
  neighbours inside the pair (neighbour_overlap: kDN, ENN, N1);
- single-feature separation: the share of the pair's rows one feature
  alone tells apart (feature_separation, F3 read the other way round);
- complicated separation: the error of a linear boundary between the
  two regions (linear_error, L2);
- poor data diversity and traffic collapse: how alike the rows of a
  region are (diversity).
Which pairs are measured is candidate_pairs' choice. Every measure is
a share in [0, 1] and is computed on distinct rows, unweighted: the
copies of a row are their own diagnosis (n_read - n_rows).
"""

from __future__ import annotations

__all__ = [
    "FeatureSeparation",
    "NeighbourOverlap",
    "Pair",
    "candidate_pairs",
    "diversity",
    "feature_separation",
    "label_conflicts",
    "linear_error",
    "neighbour_overlap",
    "pair_conflicts",
    "region_samples",
]

import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

from src.graph import sample_rows

REASONS = ("near", "conflict", "nearest")
_TINY = 1e-300  # the spanning tree's weight of two copies, which lie 0 apart


@dataclass(frozen=True)
class Pair:
    """Two regions of different classes worth measuring, a < b.

    Attributes:
        a: The first region.
        b: The second region.
        reasons: Why the pair is measured, a subset of REASONS in that
            order: "near" (separation under the threshold in some
            space), "conflict" (they share rows equal on the features),
            "nearest" (one is the nearest rival of the other).
    """

    a: int
    b: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSeparation:
    """How well single columns tell two regions apart.

    Attributes:
        share: The most, over the columns, of the share of the pair's
            rows outside the interval where the two regions' ranges
            overlap; 1 when one column separates them.
        column: The index (into the column names) of the column that
            reaches share; -1 when no column has any row outside.
        n_separating: The columns whose share is 1.
    """

    share: float
    column: int
    n_separating: int


@dataclass(frozen=True)
class NeighbourOverlap:
    """How much two regions mix among their nearest neighbours.

    Attributes:
        kdn: The mean, over the two regions, of the mean share of
            neighbours of the other region (kDN, k neighbours inside
            the pair).
        enn: The share of the pair's rows whose neighbours are at
            least half of the other region (edited nearest neighbour,
            ties count against the row).
        n1: The share of the pair's rows on an edge of the minimum
            spanning tree of the pair that joins the two regions.
    """

    kdn: float
    enn: float
    n1: float


def label_conflicts(features: pl.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Group the rows equal on every column under different labels.

    Null equals null. On every column a group holds one row per label
    (stage clean keeps one copy of the rows equal on every column and
    the label); on fewer columns it may hold several rows of a label.

    Args:
        features: The columns compared, one row per row of the data.
        labels: The class code of every row, shape (n,).

    Returns:
        The group of every row, shape (n,) int64: groups numbered from 0
        in the order of their first row, -1 for a row that no row of
        another label equals.
    """
    groups = np.full(features.height, -1, dtype=np.int64)
    if features.height == 0 or features.width == 0:
        return groups
    row, label = "__quality_row__", "__quality_label__"
    conflicting = (
        features.with_columns(
            pl.Series(label, labels), pl.int_range(features.height).alias(row)
        )
        .group_by(features.columns, maintain_order=True)
        .agg(pl.col(row), pl.col(label).n_unique().alias(label))
        .filter(pl.col(label) > 1)
        .select(pl.col(row))
        .with_row_index("group")
        .explode(row)
    )
    groups[conflicting[row].to_numpy()] = conflicting["group"].to_numpy()
    return groups


def pair_conflicts(
    groups: np.ndarray, labels: np.ndarray, region: np.ndarray, weights: np.ndarray
) -> dict[tuple[int, int], tuple[int, float]]:
    """Count the conflicting rows that every pair of regions shares.

    A group joins the regions of its rows; two regions of different
    classes in a group share its rows that lie in either, counted once
    per group. Two regions of one class share no conflict.

    Args:
        groups: The group of every row (label_conflicts), -1 for none.
        labels: The class code of every row.
        region: The region of every row.
        weights: The copies of every row in the CSV.

    Returns:
        By pair (a, b) with a < b: the conflicting rows of the two
        regions, summed over their groups, and their summed weights.
    """
    rows = np.flatnonzero(groups >= 0)
    rows = rows[np.argsort(groups[rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(groups[rows])) + 1
    counts: dict[tuple[int, int], tuple[int, float]] = {}
    for members in np.split(rows, bounds) if len(rows) else []:
        by_region: dict[int, list[int]] = defaultdict(list)
        for row in members.tolist():
            by_region[int(region[row])].append(row)
        regions = sorted(by_region)
        for i, a in enumerate(regions):
            for b in regions[i + 1 :]:
                if labels[by_region[a][0]] == labels[by_region[b][0]]:
                    continue
                shared = by_region[a] + by_region[b]
                n, read = counts.get((a, b), (0, 0.0))
                counts[(a, b)] = (n + len(shared), read + float(weights[shared].sum()))
    return counts


def candidate_pairs(
    separations: Iterable[np.ndarray],
    conflicts: Iterable[tuple[int, int]],
    classes: np.ndarray,
    threshold: float,
) -> list[Pair]:
    """Choose the pairs of regions of different classes worth measuring.

    A pair is "near" when its separation is under threshold in any of
    separations, "conflict" when it is among conflicts, "nearest" when
    one region is the nearest rival of the other (least separation in
    the first of separations; ties to the lower region id).

    Args:
        separations: The (m, m) separations (clustering.separation) of
            the regions in each space, the full space first.
        conflicts: The pairs (a, b) that share conflicting rows.
        classes: The class of every region, shape (m,).
        threshold: The separation under which a pair is near.

    Returns:
        The pairs sorted by (a, b), each with its reasons.
    """
    separations = [np.asarray(s) for s in separations]
    classes = np.asarray(classes)
    rival = classes[:, None] != classes[None, :]
    reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    for apart in separations:
        for a, b in np.argwhere(np.triu(rival & (apart < threshold), 1)):
            reasons[(int(a), int(b))].add("near")
    for a, b in conflicts:
        if rival[a, b]:
            reasons[(min(a, b), max(a, b))].add("conflict")
    first = separations[0]
    for own in range(len(classes)):
        others = np.flatnonzero(rival[own])
        if len(others):
            other = int(others[np.argmin(first[own, others])])
            reasons[(min(own, other), max(own, other))].add("nearest")
    return [
        Pair(a, b, tuple(r for r in REASONS if r in found))
        for (a, b), found in sorted(reasons.items())
    ]


def region_samples(region: np.ndarray, cap: int, seed: int) -> dict[int, np.ndarray]:
    """Draw the rows each region is measured on.

    Args:
        region: The region of every row.
        cap: The most rows per region.
        seed: The seed of graph.sample_rows, the same for every region.

    Returns:
        By region, its row positions (at most cap, ascending).
    """
    order = np.argsort(region, kind="stable")
    ids, starts = np.unique(region[order], return_index=True)
    samples = {}
    for own, rows in zip(ids.tolist(), np.split(order, starts[1:])):
        samples[own] = rows[sample_rows(len(rows), cap, seed)]
    return samples


def feature_separation(
    Xa: np.ndarray, Xb: np.ndarray, column_of: np.ndarray
) -> FeatureSeparation:
    """Return the share of the rows of a pair one column tells apart.

    Per column of the points, the rows outside [max(min_a, min_b),
    min(max_a, max_b)] are told apart by that column alone (F3 of
    Lorena et al., read as separation instead of overlap). The points
    of one original column (the one-hot slots of a categorical) count
    as one: a row is told apart by the column when any of them tells
    it apart.

    Args:
        Xa: The rows of the first region, shape (n_a, d).
        Xb: The rows of the second region, shape (n_b, d).
        column_of: The original column of each of the d points'
            columns, as an index, shape (d,).

    Returns:
        The best column's share, the column, and how many columns
        separate every row.
    """
    X = np.vstack([Xa, Xb])
    lo = np.maximum(Xa.min(axis=0), Xb.min(axis=0))
    hi = np.minimum(Xa.max(axis=0), Xb.max(axis=0))
    outside = (X < lo) | (X > hi)
    columns, owner = np.unique(column_of, return_inverse=True)
    told = np.zeros((len(X), len(columns)), dtype=bool)
    for j, column in enumerate(owner):
        told[:, column] |= outside[:, j]
    shares = told.mean(axis=0)
    best = int(np.argmax(shares))
    if shares[best] == 0:
        return FeatureSeparation(0.0, -1, 0)
    return FeatureSeparation(
        float(shares[best]), int(columns[best]), int((shares == 1).sum())
    )


def neighbour_overlap(Xa: np.ndarray, Xb: np.ndarray, k: int) -> NeighbourOverlap:
    """Measure how much two regions mix, from neighbours inside the pair.

    The neighbours of a row are its k nearest other rows of the two
    regions (fewer when the pair has fewer rows).

    Args:
        Xa: The rows of the first region, shape (n_a, d), n_a >= 1.
        Xb: The rows of the second region, shape (n_b, d), n_b >= 1.
        k: The neighbours of each row.

    Returns:
        kDN, ENN and N1 of the pair.
    """
    X = np.vstack([Xa, Xb])
    side = np.r_[np.zeros(len(Xa), dtype=bool), np.ones(len(Xb), dtype=bool)]
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    k = min(k, len(X) - 1)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :k]
    rival_share = (side[neighbours] != side[:, None]).mean(axis=1)
    kdn = (rival_share[~side].mean() + rival_share[side].mean()) / 2
    enn = (rival_share >= 0.5).mean()
    # A zero weight is no edge to the spanning tree: copies keep a tiny one.
    np.fill_diagonal(distances, 0.0)
    weights = np.where(distances == 0, _TINY, distances)
    np.fill_diagonal(weights, 0.0)
    tree = minimum_spanning_tree(weights).tocoo()
    border = side[tree.row] != side[tree.col]
    on_border = np.unique(np.r_[tree.row[border], tree.col[border]])
    return NeighbourOverlap(float(kdn), float(enn), len(on_border) / len(X))


def linear_error(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Return the balanced training error of a linear SVM between two regions.

    LinearSVC(class_weight="balanced") fit on the two regions (L2 of
    Lorena et al.). High with a low neighbour overlap, the boundary is
    there but is not a hyperplane.

    Args:
        Xa: The rows of the first region, shape (n_a, d), n_a >= 2.
        Xb: The rows of the second region, shape (n_b, d), n_b >= 2.

    Returns:
        1 - balanced accuracy on the rows, in [0, 1].
    """
    X = np.vstack([Xa, Xb])
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        svm = LinearSVC(class_weight="balanced", max_iter=10_000, random_state=0)
        svm.fit(X, y)
    return float(1.0 - balanced_accuracy_score(y, svm.predict(X)))


def _scaled(X: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Return rows of X scaled into [0, 1] by column; a constant column is 0."""
    return np.divide(X - lo, span, out=np.zeros(X.shape), where=span > 0)


def diversity(
    X: np.ndarray,
    region: np.ndarray,
    n_pairs: int,
    seed: int,
    collapse: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return how alike the rows of every region are (PDD and TC of Flood).

    The cosine similarity is that of the points scaled into [0, 1] by
    column over every row of X (a constant column is 0), so that it
    lies in [0, 1] as in Flood et al.; only the rows drawn are scaled.
    n_pairs pairs of distinct rows of the region are drawn at random
    (with replacement; every pair when the region has no more); two
    rows of norm 0 have similarity 1, one of norm 0 and one not 0.

    Args:
        X: The points, shape (n, d).
        region: The region of every row, ids 0..m-1.
        n_pairs: The pairs drawn per region.
        seed: The seed of the draws.
        collapse: The cosine similarity over which two rows are
            functionally identical.

    Returns:
        Per region, shape (m,): the mean cosine similarity of its pairs
        (pdd) and the share of pairs over collapse (tc); 1 and 1 for a
        region of a single row.
    """
    lo, span = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
    rng = np.random.default_rng(seed)
    m = int(region.max()) + 1
    pdd, tc = np.ones(m), np.ones(m)
    order = np.argsort(region, kind="stable")
    ids, starts = np.unique(region[order], return_index=True)
    for own, rows in zip(ids.tolist(), np.split(order, starts[1:])):
        n = len(rows)
        if n < 2:
            continue
        if n * (n - 1) // 2 <= n_pairs:
            first, second = np.triu_indices(n, 1)
        else:
            first = rng.integers(n, size=n_pairs)
            second = rng.integers(n - 1, size=n_pairs)
            second += second >= first
        drawn, back = np.unique(np.r_[first, second], return_inverse=True)
        points = _scaled(X[rows[drawn]], lo, span)
        a, b = points[back[: len(first)]], points[back[len(first) :]]
        norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        both = norm_a * norm_b
        cosine = np.where(
            both > 0,
            np.einsum("ij,ij->i", a, b) / np.where(both > 0, both, 1.0),
            (norm_a == 0) & (norm_b == 0),
        )
        pdd[own], tc[own] = cosine.mean(), (cosine > collapse).mean()
    return pdd, tc
