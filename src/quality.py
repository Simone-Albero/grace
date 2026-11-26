"""Data-quality defects of the regions of a class, after Flood, Engelen, Aspinall & Desmet.

"Bad Design Smells in Benchmark NIDS Datasets" (EuroS&P 2024). The
cells of `src.cells` are the smallest groups a share is read on; every
node of a class's cut holds the rows of the cells below it, and the
report (src.report) judges every node and merges the cells up the cut
into regions where no verdict changes. The defects (DEFECTS), in three
families:
- STRUCTURAL, facts of single rows, read as shares of a node's rows:
  invalid_values (a null or invalid value, invalid_values),
  duplicates (CSV rows that repeat a row: copies can fall on both
  sides of a train/test split) and label_conflict (a row equal on every
  feature to a row of another class: no classifier tells them apart,
  label_conflicts);
- OVERLAP, against a rival class: a row is confused with a class when
  its nearest row in the dataset, itself left out, is of that class
  (nearest_rivals, the leave-one-out 1-NN on every row, no sample and
  no model; a local N3 of Ho & Basu, kDN of Smith et al. with one
  neighbour). minority_overlap: at least one half of the node's rows
  are (rival_shares): its class is the minority where it lies, as rows
  under a wrong label, or a true minority, would be. majority_overlap:
  that share reaches tau, against a class it has no minority_overlap
  against;
- GENERALIZATION: single_feature_cut, one column alone separates the
  node from its rival class, the most frequent class of its rows'
  nearest rival rows: under tau of that class's rows fall inside the
  node's values on the column (cut_shares, a local F3); low_variability,
  the node lies in a branch of its class's cut that holds at least tau
  of the class's CSV rows and is no more spread than the median cell of
  the dataset (branch_spread, low_variability_verdicts): many rows in
  the room of a few, near copies of one flow.

Every share of a node's rows is read with an interval at a confidence
(share_bounds), and judged against a threshold beyond the error of
sampling (verdict): the node has the defect when the interval lies
past it, has none when it lies short of it, and is undecided otherwise.
The copies of a row are no independent trials: the share of duplicates
is of the CSV rows, its interval on the distinct rows. low_variability
alone is no share: a large branch holds many rows, and its variance is
compared as it is. tau is the one threshold chosen; one half, the
majority, is the other. Every measure but the copies and the size of a
branch is computed on distinct rows, unweighted.
"""

from __future__ import annotations

__all__ = [
    "DEFECTS",
    "GENERALIZATION",
    "OVERLAP",
    "STRUCTURAL",
    "branch_spread",
    "column_values",
    "cut_shares",
    "invalid_values",
    "label_conflicts",
    "low_variability_reference",
    "low_variability_verdicts",
    "nearest_rivals",
    "node_ranges",
    "node_sums",
    "node_verdicts",
    "pair_conflicts",
    "resolution_rows",
    "rival_shares",
    "share_bounds",
    "verdict",
]

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

STRUCTURAL = ("invalid_values", "duplicates", "label_conflict")
OVERLAP = ("minority_overlap", "majority_overlap")
GENERALIZATION = ("single_feature_cut", "low_variability")
DEFECTS = STRUCTURAL + OVERLAP + GENERALIZATION


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
    groups: np.ndarray, labels: np.ndarray, cell: np.ndarray, weights: np.ndarray
) -> dict[tuple[int, int], tuple[int, float]]:
    """Count the conflicting rows that every pair of cells shares.

    A group joins the cells of its rows; two cells of different classes
    in a group share its rows that lie in either, counted once per
    group. Two cells of one class share no conflict.

    Args:
        groups: The group of every row (label_conflicts), -1 for none.
        labels: The class code of every row.
        cell: The cell of every row.
        weights: The copies of every row in the CSV.

    Returns:
        By pair (a, b) with a < b: the conflicting rows of the two
        cells, summed over their groups, and their summed weights.
    """
    rows = np.flatnonzero(groups >= 0)
    rows = rows[np.argsort(groups[rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(groups[rows])) + 1
    counts: dict[tuple[int, int], tuple[int, float]] = {}
    for members in np.split(rows, bounds) if len(rows) else []:
        by_cell: dict[int, list[int]] = defaultdict(list)
        for row in members.tolist():
            by_cell[int(cell[row])].append(row)
        cells = sorted(by_cell)
        for i, a in enumerate(cells):
            for b in cells[i + 1 :]:
                if labels[by_cell[a][0]] == labels[by_cell[b][0]]:
                    continue
                shared = by_cell[a] + by_cell[b]
                n, read = counts.get((a, b), (0, 0.0))
                counts[(a, b)] = (n + len(shared), read + float(weights[shared].sum()))
    return counts


def nearest_rivals(
    points: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, for every row, its nearest row of another class, whether it is confused with it, and its nearest own row.

    Euclidean and exact, on every row: per class a kd-tree on the rows
    of the other classes gives the nearest rival row, and one on the
    rows of the class the nearest row of its own, itself left out. A
    row is confused with the class of its nearest rival row when that
    row is nearer than its nearest own row (the leave-one-out 1-NN on
    the whole dataset); at the same distance (relative tolerance 1e-9)
    it counts one half. Of rows at the same distance, one, the same on
    every run. The nearest own row is the control of the provenance
    measures (src.provenance).

    Args:
        points: The rows, shape (n, d).
        labels: The class code of every row, shape (n,).

    Returns:
        The position of every row's nearest row of another class, shape
        (n,) int64, -1 when the rows are of one class; how much each row
        is confused with it, shape (n,) float: 1, 0.5 or 0 (1 for the
        only row of its class, 0 without a rival row); and the position
        of its nearest other row of its class, shape (n,) int64, -1 for
        the only row of its class.
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    rival_row = np.full(n, -1, dtype=np.int64)
    own_row = np.full(n, -1, dtype=np.int64)
    confused = np.zeros(n)
    for k in np.unique(labels):
        own = np.flatnonzero(labels == k)
        others = np.flatnonzero(labels != k)
        if len(others):
            search = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
            rival, at = search.fit(points[others]).kneighbors(points[own])
            rival_row[own] = others[at[:, 0]]
        if len(own) == 1:  # no row of its own: its nearest row is a rival's
            confused[own] = 1.0 if len(others) else 0.0
            continue
        search = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", n_jobs=-1)
        near, at = search.fit(points[own]).kneighbors(points[own])
        # itself left out; a copy of it at distance 0 may come first
        first = at[:, 0] == np.arange(len(own))
        mine = np.where(first, near[:, 1], near[:, 0])
        own_row[own] = own[np.where(first, at[:, 1], at[:, 0])]
        if not len(others):
            continue
        rival = rival[:, 0]
        tie = np.abs(rival - mine) <= _TIE * np.maximum(mine, 1.0)
        confused[own] = np.where(tie, 0.5, (rival < mine).astype(np.float64))
    return rival_row, confused, own_row


def rival_shares(
    cell: np.ndarray,
    rival_row: np.ndarray,
    confused: np.ndarray,
    cell_class: np.ndarray,
) -> list[tuple[int, int, list[int], float]]:
    """Return, per cell and rival class, its rival cells and the share of its rows confused with that class.

    The rival classes of a cell are the classes of its rows' nearest
    rival rows (nearest_rivals), its rival cells of a class the cells
    those rows lie in. The share is over all the cell's rows: the sum
    of how much each row whose nearest rival row is of that class is
    confused with it, over the cell's rows.

    Args:
        cell: The cell of every row, shape (n,).
        rival_row: The nearest row of another class of every row
            (nearest_rivals), shape (n,), -1 for none.
        confused: How much every row is confused with it, shape (n,).
        cell_class: The class of every cell.

    Returns:
        (cell, rival class, its rival cells of that class sorted, the
        share), sorted by cell then class; none for rows without a
        rival row.
    """
    cell, rival_row = np.asarray(cell), np.asarray(rival_row)
    confused, cell_class = np.asarray(confused, dtype=np.float64), np.asarray(
        cell_class
    )
    size = np.bincount(cell, minlength=len(cell_class))
    found: dict[tuple[int, int], list] = {}
    for row in np.flatnonzero(rival_row >= 0).tolist():
        other = int(cell[rival_row[row]])
        entry = found.setdefault((int(cell[row]), int(cell_class[other])), [set(), 0.0])
        entry[0].add(other)
        entry[1] += float(confused[row])
    return [
        (c, k, sorted(cells), total / size[c])
        for (c, k), (cells, total) in sorted(found.items())
    ]


def invalid_values(features: pl.DataFrame, unparsable: pl.DataFrame) -> pl.DataFrame:
    """List every null or invalid cell, with its kind.

    Kinds (data.KINDS): "unparsable" for a cell of unparsable (a
    numeric cell whose text was not a number, null in features),
    "empty" for any other null, "nan" and "inf" (either sign) for a
    float that is not finite.

    Args:
        features: The columns of the data file but DUPLICATES and
            LABEL, one row per row.
        unparsable: The cells whose text was not a number (paths.
            read_unparsable): columns row and column.

    Returns:
        Columns row (Int64), column (String), kind (String), one row
        per null or invalid cell, sorted by row then column order.
    """
    order = {column: i for i, column in enumerate(features.columns)}
    bad = {(row, column) for row, column in unparsable.iter_rows()}
    cells = []
    for column in features.columns:
        values = features[column]
        empty = values.is_null().to_numpy()
        for row in np.flatnonzero(empty).tolist():
            kind = "unparsable" if (row, column) in bad else "empty"
            cells.append((row, column, kind))
        if values.dtype.is_float():
            x = values.to_numpy()
            for row in np.flatnonzero(np.isnan(x) & ~empty).tolist():
                cells.append((row, column, "nan"))
            for row in np.flatnonzero(np.isinf(x)).tolist():
                cells.append((row, column, "inf"))
    cells.sort(key=lambda cell: (cell[0], order[cell[1]]))
    return pl.DataFrame(
        cells,
        schema={"row": pl.Int64, "column": pl.String, "kind": pl.String},
        orient="row",
    )


def share_bounds(share: float, n: float, confidence: float) -> tuple[float, float]:
    """Return the interval of a share of rows: Wilson's score interval.

    Each bound is one-sided at confidence: the share is under the upper
    bound, or over the lower, with that probability.

    Args:
        share: The share observed, in [0, 1].
        n: The rows it is a share of; (0, 1) when none.
        confidence: The level, in (0.5, 1).

    Returns:
        The lower and the upper bound, in [0, 1].
    """
    if n <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(confidence)
    z2 = z * z
    centre = (share + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(share * (1 - share) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    # it holds the share, but for rounding at 0 and 1
    return min(max(0.0, centre - half), share), max(min(1.0, centre + half), share)


def resolution_rows(tau: float, confidence: float) -> int:
    """Return the fewest rows on which a share tau away from a threshold of the report is decided.

    The report reads a share of a cell's rows against two thresholds,
    tau and one half. On n rows, a share at a threshold t plus tau must
    have its lower bound (share_bounds) at t or over, and one at t
    minus tau its upper bound under t: a cell of n rows is then
    undecided only within tau of a threshold. With the normal interval
    that is n >= z² s (1 - s) / tau² for a share s, with z the quantile
    of confidence, at most z² / (4 tau²) near one half (68 rows at tau
    0.1 and 0.95, 31 at tau 0.15); this returns the first such n on
    the interval the method uses. Against tau alone, a share of 0 is
    decided from z² (1 - tau) / tau rows on, far fewer: one half is
    where a share is the least certain.

    Args:
        tau: The threshold, in (0, 0.5].
        confidence: The level of the interval, in (0.5, 1).

    Returns:
        The rows, >= 1.
    """

    def decided(n: int) -> bool:
        return all(
            share_bounds(t + tau, n, confidence)[0] >= t
            and share_bounds(t - tau, n, confidence)[1] < t
            for t in (tau, 0.5)
        )

    n = 1
    while not decided(n):
        n += 1
    return n


def verdict(low: float, high: float, threshold: float) -> bool | None:
    """Return whether a measure reaches a threshold beyond the error of sampling.

    Args:
        low: The lower bound of the measure.
        high: The upper bound.
        threshold: The threshold.

    Returns:
        True when low >= threshold, False when high < threshold, None
        (undecided) otherwise.
    """
    if low >= threshold:
        return True
    if high < threshold:
        return False
    return None


def branch_spread(
    points: np.ndarray, cell: np.ndarray, parent: np.ndarray, node_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far the rows of every branch of a tree lie from their centre.

    Exact, from the sums of every cell (its rows, their sum, the sum of
    their squared norms), added up the tree: the variance of a branch
    is the mean squared distance of its rows to its centroid.

    Args:
        points: The rows, shape (n, d) (distinct, unweighted).
        cell: The cell of every row.
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for a node that is not a
            cell; the cells are the leaves.

    Returns:
        Per node, shape (k,): the variance of its rows, and its rows.
    """
    points = np.asarray(points, dtype=np.float64)
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    centre = points.mean(axis=0)  # sums about it: no cancellation far away
    order = np.argsort(np.asarray(cell), kind="stable")
    ids, starts = np.unique(np.asarray(cell)[order], return_index=True)
    rows_of = dict(zip(ids.tolist(), np.split(order, starts[1:])))
    k = len(parent)
    variance, n_rows = np.zeros(k), np.zeros(k, dtype=np.int64)
    pending: dict[int, list] = {}  # the sums of the children seen of a node
    for v in range(k - 1, -1, -1):  # children after their parent
        if node_cell[v] >= 0:
            Z = points[rows_of.get(int(node_cell[v]), np.empty(0, int))] - centre
            sums = [len(Z), Z.sum(0), float(np.einsum("ij,ij->", Z, Z))]
        else:
            sums = pending.pop(v)
        n, s1, s2 = sums
        if n:
            c = s1 / n
            variance[v] = max(s2 / n - c @ c, 0.0)
            n_rows[v] = n
        if parent[v] >= 0:
            if parent[v] in pending:
                pending[parent[v]] = [a + b for a, b in zip(pending[parent[v]], sums)]
            else:
                pending[parent[v]] = list(sums)
    return variance, n_rows


def node_sums(
    parent: np.ndarray, node_cell: np.ndarray, per_cell: np.ndarray
) -> np.ndarray:
    """Return, per node of a cut, the sum of a measure of the cells below it.

    Args:
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for a node that is not a
            cell; the cells are the leaves.
        per_cell: The measure of every cell, indexed by cell, shape
            (m,) or (m, c).

    Returns:
        Per node, shape (k,) or (k, c): the sum over its cells, in the
        dtype of per_cell.
    """
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    per_cell = np.asarray(per_cell)
    out = np.zeros((len(parent),) + per_cell.shape[1:], dtype=per_cell.dtype)
    leaf = node_cell >= 0
    out[leaf] = per_cell[node_cell[leaf]]
    for v in range(len(parent) - 1, -1, -1):  # children after their parent
        if parent[v] >= 0:
            out[parent[v]] += out[v]
    return out


def node_ranges(
    values: np.ndarray,
    cell: np.ndarray,
    parent: np.ndarray,
    node_cell: np.ndarray,
    numeric: bool,
) -> np.ndarray:
    """Return, per node of a cut, the values its rows take on one column.

    Args:
        values: The column's value of every row, shape (n,)
            (column_values: the number, or the index of the active
            one-hot slot, -1 for none).
        cell: The cell of every row, shape (n,).
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for a node that is not a
            cell.
        numeric: Whether the column is numeric.

    Returns:
        Numeric: shape (k, 2) float, the least and the most value of
        each node's rows (NaN for a node without rows). Categorical:
        shape (k, s + 1) bool, whether each slot is taken by a row of
        the node, s the slots, the last for no slot (-1).
    """
    values, cell = np.asarray(values, dtype=np.float64), np.asarray(cell)
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    m = int(node_cell.max()) + 1
    leaf = node_cell >= 0
    if numeric:
        least, most = np.full(m, np.inf), np.full(m, -np.inf)
        np.minimum.at(least, cell, values)
        np.maximum.at(most, cell, values)
        out = np.full((len(parent), 2), np.nan)
        out[leaf] = np.c_[least, most][node_cell[leaf]]
        for v in range(len(parent) - 1, -1, -1):  # children after their parent
            up = parent[v]
            if up >= 0:
                out[up, 0] = np.fmin(out[up, 0], out[v, 0])
                out[up, 1] = np.fmax(out[up, 1], out[v, 1])
        return out
    slot = _slots(values)
    taken = np.zeros((m, int(slot.max()) + 1 if len(slot) else 1), dtype=bool)
    taken[cell, slot] = True
    out = np.zeros((len(parent), taken.shape[1]), dtype=bool)
    out[leaf] = taken[node_cell[leaf]]
    for v in range(len(parent) - 1, -1, -1):
        if parent[v] >= 0:
            out[parent[v]] |= out[v]
    return out


def cut_shares(
    ranges: np.ndarray,
    rival_class: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    numeric: bool,
) -> np.ndarray:
    """Return, per node, the share of its rival class's rows inside its values on one column.

    A numeric column: the share of the distinct rows of the rival class
    whose value lies in [min, max] of the node's rows. A categorical
    one: of those whose slot is one the node's rows take. A low share
    means the column alone separates the node from its rival class: a
    local F3 of Ho & Basu, with its known limit (one extreme row widens
    [min, max], so it reports less, never more).

    Args:
        ranges: Per node, what node_ranges returns for the column.
        rival_class: The rival class of every node (the most frequent
            class of its rows' nearest rival rows, ties to the lower
            code), -1 for none.
        values: The column's value of every row, shape (n,)
            (column_values).
        labels: The class code of every row, shape (n,).
        numeric: Whether the column is numeric.

    Returns:
        Per node, shape (k,) float in [0, 1]; NaN for a node without a
        rival class.
    """
    values, labels = np.asarray(values, dtype=np.float64), np.asarray(labels)
    rival_class = np.asarray(rival_class)
    shares = np.full(len(rival_class), np.nan)
    slot = None if numeric else _slots(values)
    for k in np.unique(rival_class[rival_class >= 0]):
        nodes = np.flatnonzero(rival_class == k)
        of_k = labels == k
        if numeric:
            sorted_k = np.sort(values[of_k])
            low, high = ranges[nodes, 0], ranges[nodes, 1]
            inside = np.searchsorted(sorted_k, high, side="right") - np.searchsorted(
                sorted_k, low, side="left"
            )
            shares[nodes] = np.maximum(inside, 0) / len(sorted_k)
        else:
            width = ranges.shape[1]
            share_of_slot = (
                np.bincount(slot[of_k], minlength=width)[:width] / of_k.sum()
            )
            shares[nodes] = ranges[nodes].astype(np.float64) @ share_of_slot
    return shares


def low_variability_reference(variance: np.ndarray, node_cell: np.ndarray) -> float:
    """Return the reference of low_variability: the variance of the median cell of the dataset.

    Every cell holds about as many distinct rows (one to two times
    min_rows), so its variance is the room a typical group of
    neighbouring rows of that size takes in the dataset. It depends on
    tau, which sets the size of the cells, and on the classes of the
    dataset, whose cells it is the median of.

    Args:
        variance: The variance of every node of the cuts of every class
            (branch_spread).
        node_cell: The cell of every node, -1 for a node that is not a
            cell.

    Returns:
        The median of the variances of the cells, >= 0.
    """
    variance, node_cell = np.asarray(variance), np.asarray(node_cell)
    return float(np.median(variance[node_cell >= 0]))


def low_variability_verdicts(
    parent: np.ndarray,
    n_read: np.ndarray,
    variance: np.ndarray,
    reference: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return, per node of the cuts, whether it lies in a large compact branch of its class.

    A branch is large when it holds at least tau of the CSV rows of its
    root (its class), and compact when its variance is at most the
    reference (low_variability_reference): many rows in the room of a
    typical cell, near copies of one flow. A node has low_variability
    when it or a node above it is large and compact. No interval: a
    large branch holds many rows, and its variance is stable.

    Args:
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        n_read: The CSV rows below every node.
        variance: The variance of every node's distinct rows
            (branch_spread).
        reference: The variance of the median cell
            (low_variability_reference).
        tau: The share of its class a branch must hold, in (0, 0.5].

    Returns:
        Per node, shape (k,): whether it has low_variability (bool), and
        the highest large compact node on its path from the root (int64,
        -1 for none).
    """
    parent = np.asarray(parent)
    n_read, variance = np.asarray(n_read, dtype=np.float64), np.asarray(variance)
    k = len(parent)
    root = np.arange(k)
    top = np.full(k, -1, dtype=np.int64)  # the highest large compact node on the path
    for v in range(k):  # a parent before its children
        up = parent[v]
        if up >= 0:
            root[v], top[v] = root[up], top[up]
        large = n_read[v] >= tau * n_read[root[v]]
        if top[v] < 0 and large and variance[v] <= reference:
            top[v] = v
    return top >= 0, top


def node_verdicts(
    own: int,
    n_classes: int,
    n_rows: float,
    n_read: float,
    missing_rows: float,
    conflict_rows: float,
    confused: dict[int, float],
    cut_share: float | None,
    n_cut_rows: int,
    low_variability: bool,
    tau: float,
    confidence: float,
) -> dict[tuple[str, int | None], bool | None]:
    """Return the verdict of every entry of a node: a defect against a target.

    Every share is read with its interval (share_bounds) on the node's
    distinct rows, against its threshold (verdict):
    - invalid_values and label_conflict: the share of its distinct rows
      with a null or invalid value, or equal to a row of another class,
      against tau;
    - duplicates: the share of its CSV rows that repeat a row, against
      tau, its interval on the distinct rows (the copies of a row are no
      independent trials: the rows are the clusters);
    - against every other class k: the share of its rows confused with
      k (0 when none is), minority_overlap against one half,
      majority_overlap against tau;
    - single_feature_cut, against no class (the rival class is its
      description): True when the upper bound of the cut share, on the
      rival class's distinct rows, lies under tau, False when its lower
      bound reaches tau; the least share over the columns decides, the
      bounds growing with the share;
    - low_variability, against no class: as low_variability_verdicts
      found it, never undecided.

    Args:
        own: The class code of the node.
        n_classes: The number of classes.
        n_rows: Its distinct rows, > 0.
        n_read: Its CSV rows.
        missing_rows: Its distinct rows with a null or invalid value.
        conflict_rows: Its distinct rows equal to a row of another
            class.
        confused: By rival class, its rows confused with it (the sum of
            how much each is, a tie one half).
        cut_share: Its least cut share over the columns (cut_shares),
            None without a rival class.
        n_cut_rows: The distinct rows of its rival class.
        low_variability: Whether it has low_variability.
        tau: The threshold, in (0, 0.5].
        confidence: The level of the intervals, in (0.5, 1).

    Returns:
        By (defect, class code or None), True, False or None
        (undecided).
    """

    def judged(share: float, n: float, threshold: float) -> bool | None:
        return verdict(*share_bounds(share, n, confidence), threshold)

    copies = (n_read - n_rows) / n_read if n_read > 0 else 0.0
    found: dict[tuple[str, int | None], bool | None] = {
        ("invalid_values", None): judged(missing_rows / n_rows, n_rows, tau),
        ("duplicates", None): judged(copies, n_rows, tau),
        ("label_conflict", None): judged(conflict_rows / n_rows, n_rows, tau),
    }
    for k in range(n_classes):
        if k == own:
            continue
        bounds = share_bounds(confused.get(k, 0.0) / n_rows, n_rows, confidence)
        found[("minority_overlap", k)] = verdict(*bounds, 0.5)
        found[("majority_overlap", k)] = verdict(*bounds, tau)
    if cut_share is not None:
        reached = judged(cut_share, n_cut_rows, tau)  # the share reaches tau: no cut
        found[("single_feature_cut", None)] = None if reached is None else not reached
    found[("low_variability", None)] = bool(low_variability)
    return found


_TIE = 1e-9  # relative tolerance under which two distances tie


def column_values(Z: np.ndarray, numeric: bool) -> np.ndarray:
    """Return the values of one original column from its point columns.

    Args:
        Z: Its point columns, shape (n, s): one for a number, its
            one-hot slots for a categorical.
        numeric: Whether it is numeric.

    Returns:
        Shape (n,) float: the number, or the index of the active slot
        (-1 for none).
    """
    if numeric:
        return Z[:, 0]
    return np.where(Z.max(axis=1) > 0, Z.argmax(axis=1), -1).astype(float)


def _slots(values: np.ndarray) -> np.ndarray:
    """Return the slot of every value of a categorical column (column_values), the last one for no slot (-1)."""
    codes = np.asarray(values).astype(np.int64)
    width = int(codes.max()) + 2 if len(codes) else 1
    return np.where(codes < 0, width - 1, codes)
