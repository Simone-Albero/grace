"""Data-quality defects of the cells, after Flood, Engelen, Aspinall & Desmet.

"Bad Design Smells in Benchmark NIDS Datasets" (EuroS&P 2024), judged
on the cells of `src.cells`, each by its relation to the other cells
(DEFECTS, defects_of):
- invalid_values: a row holds a null or invalid cell (invalid_values);
- duplicates: a row comes more than once in the CSV, so copies can
  fall on both sides of a train/test split;
- label_conflict: a row equals, on every feature, a row of another
  class: no classifier can tell them apart (label_conflicts);
- suspect_label: against a rival class, at least half of its rows lie
  nearer a row of that class than any row of their own (rival_shares):
  the cell lies inside the other class, as rows under a wrong label
  would;
- class_overlap: against a rival class, that share reaches tau, and it
  is no suspect_label;
- shortcut: against a rival class, the best rule on a single column
  between its class and the rival class, both whole (class_rule),
  leaves under tau of its rows and under tau of the rival class's rows
  on the wrong side (rule_errors): one column alone tells the two
  classes apart where the cell lies;
- repetition: it lies in a branch of its class's tree that holds at
  least tau of the class's CSV rows and is at most tau times as
  dispersed as the dataset (branch_spread, dataset_variance,
  repetition_verdicts): a large part of its class lies in little
  space, the class or one of its modes repeated;
- off_scenario: at least tau of its CSV rows break a rule of the
  scenario the dataset declares for its class (src.provenance.
  scenario_breaches): traffic the documentation does not describe
  under that label.
The first three are facts of single rows (ROW_DEFECTS): counted in any
group of rows, they are never diluted. The next three are measured
between a cell and a rival class (RIVAL_DEFECTS, rival_defects): the
rival class whole, for the rival share and for the shortcut. The last
two between a cell and a reference of its own (CLASS_DEFECTS): its
class's tree against the dataset, the scenario. A row is confused with a class when
its nearest row in the dataset, itself left out, is of that class
(nearest_rivals): the leave-one-out 1-NN on every column and every
row, no sample and no model.

Every share of a cell's rows is an estimate on few rows, so it is read
with an interval at a confidence level (share_bounds), and a cell is
judged against a threshold beyond the error of sampling
(verdict): it has the defect when the interval lies past the
threshold, has none when it lies short of it, and is undecided
otherwise. repetition alone is no share: a branch holds many rows, and
its variance is compared as it is. tau is the one threshold chosen; one
half, the majority, is the other. The rival classes of a cell are the classes of the
nearest rows of other classes to its rows, its rival cells the cells
of those rows (nearest_rivals): no threshold picks them. A cell is
measured on all its rows. Every measure but the copies, the size of a branch and the scenario
is computed on distinct rows, unweighted, and read on the cell's side.
"""

from __future__ import annotations

__all__ = [
    "CLASS_DEFECTS",
    "DEFECTS",
    "RIVAL_DEFECTS",
    "ROW_DEFECTS",
    "RivalMeasure",
    "Rule",
    "class_rule",
    "column_values",
    "branch_spread",
    "dataset_variance",
    "defects_of",
    "invalid_values",
    "label_conflicts",
    "nearest_rivals",
    "pair_conflicts",
    "repetition_verdicts",
    "resolution_rows",
    "rival_defects",
    "rival_shares",
    "rule_errors",
    "share_bounds",
    "verdict",
]

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

ROW_DEFECTS = ("invalid_values", "duplicates", "label_conflict")
RIVAL_DEFECTS = ("suspect_label", "class_overlap", "shortcut")
CLASS_DEFECTS = ("repetition", "off_scenario")
DEFECTS = ROW_DEFECTS + RIVAL_DEFECTS + CLASS_DEFECTS


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


@dataclass(frozen=True)
class Rule:
    """A rule on one column that sends each value to one of two sides.

    Attributes:
        kind: How a value goes to the first side: "le" (at most at),
            "lt" (under at), "ge" (at least at), "gt" (over at), or
            "values" (not among at; those go to the second side).
        at: The threshold, a float; for "values", the values sent to
            the second side, a sorted array.
    """

    kind: str
    at: float | np.ndarray

    def first(self, values: np.ndarray) -> np.ndarray:
        """Return whether each value goes to the first side.

        Args:
            values: The values of the column, shape (n,).

        Returns:
            Bool, shape (n,).
        """
        v = np.asarray(values)
        if self.kind == "le":
            return v <= self.at
        if self.kind == "lt":
            return v < self.at
        if self.kind == "ge":
            return v >= self.at
        if self.kind == "gt":
            return v > self.at
        return ~np.isin(v, self.at)


class RivalMeasure(NamedTuple):
    """What a cell was measured on against one rival class, tau-free.

    Attributes:
        rival_class: The code of the rival class.
        share: The share of the cell's rows confused with the rival
            class (rival_shares), in [0, 1].
        n_rows: The cell's distinct rows, what the share and the first
            error of a rule are shares of.
        rules: Per column whose rule leaves under half of each side on
            the wrong side, the error of class_rule on the cell's rows
            and on the rival class's, shape (c, 2); the other columns
            can never be a shortcut.
        n_rival: The rival class's rows the second error is a share of.
    """

    rival_class: int
    share: float
    n_rows: int
    rules: np.ndarray
    n_rival: int


def nearest_rivals(
    points: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return, for every row, its nearest row of another class and whether it is confused with it.

    Euclidean and exact, on every row: per class a kd-tree on the rows
    of the other classes gives the nearest rival row, and one on the
    rows of the class the nearest row of its own, itself left out. A
    row is confused with the class of its nearest rival row when that
    row is nearer than its nearest own row (the leave-one-out 1-NN on
    the whole dataset); at the same distance (relative tolerance 1e-9)
    it counts one half. Of rival rows at the same distance, one, the
    same on every run.

    Args:
        points: The rows, shape (n, d).
        labels: The class code of every row, shape (n,).

    Returns:
        The position of every row's nearest row of another class, shape
        (n,) int64, -1 when the rows are of one class; and how much each
        row is confused with it, shape (n,) float: 1, 0.5 or 0 (1 for
        the only row of its class, 0 without a rival row).
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    rival_row = np.full(n, -1, dtype=np.int64)
    confused = np.zeros(n)
    if len(np.unique(labels)) < 2:
        return rival_row, confused
    for k in np.unique(labels):
        own = np.flatnonzero(labels == k)
        others = np.flatnonzero(labels != k)
        search = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
        rival, at = search.fit(points[others]).kneighbors(points[own])
        rival_row[own] = others[at[:, 0]]
        if len(own) == 1:  # no row of its own: its nearest row is a rival's
            confused[own] = 1.0
            continue
        search = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", n_jobs=-1)
        near, at = search.fit(points[own]).kneighbors(points[own])
        # itself left out; a copy of it at distance 0 may come first
        mine = np.where(at[:, 0] == np.arange(len(own)), near[:, 1], near[:, 0])
        rival = rival[:, 0]
        tie = np.abs(rival - mine) <= _TIE * np.maximum(mine, 1.0)
        confused[own] = np.where(tie, 0.5, (rival < mine).astype(np.float64))
    return rival_row, confused


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


def class_rule(va: np.ndarray, vb: np.ndarray, numeric: bool) -> Rule:
    """Return the rule on one column of highest balanced accuracy between two samples.

    A numeric column gets a threshold, in either direction: the
    balanced accuracy F_a(t) - F_b(t) + 1/2 (or the other way) changes
    only at the values of either sample, and its extremes are reached at
    a value of the smaller one or just under it, so the search is exact
    over those. A categorical column sends each value to the side where
    it is the larger share of the rows (a tie to the first). Ties to
    the first threshold found.

    Args:
        va: The values of the first sample, shape (n_a,), n_a >= 1.
        vb: The values of the second, shape (n_b,), n_b >= 1.
        numeric: Whether the column is numeric; else its values are the
            codes of its categories.

    Returns:
        The Rule; Rule("values", empty) (every row to the first side)
        when no rule beats chance.
    """
    va, vb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
    if numeric:
        return _threshold(np.sort(va), np.sort(vb))
    return _split(*(_shares(v, np.unique(np.r_[va, vb])) for v in (va, vb)))


def rule_errors(
    values: np.ndarray,
    labels: np.ndarray,
    cell: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    cell_class: np.ndarray,
    numeric: bool,
) -> np.ndarray:
    """Return, per cell and rival class, the errors of their classes' rule on one column.

    For every pair of the cell's class and the rival class, class_rule
    on all the distinct rows of the two classes, chosen once; then the
    share of the cell's rows it sends to the rival's side and the share
    of the rival class's rows it sends to the cell's. The rule is that
    of the two classes, not of the cell: a cell is small, and some
    column puts it in a tail of any class, so a rule chosen for it alone
    would nearly always separate it.

    Args:
        values: The column's value of every row, shape (n,).
        labels: The class code of every row, shape (n,).
        cell: The cell of every row, shape (n,).
        pairs: The (cell, rival class) pairs.
        cell_class: The class of every cell.
        numeric: Whether the column is numeric.

    Returns:
        Shape (len(pairs), 2): the error on the cell's rows and on the
        rival class's rows, each in [0, 1].
    """
    values = np.asarray(values, dtype=np.float64)
    labels, cell = np.asarray(labels), np.asarray(cell)
    cell_class = np.asarray(cell_class)
    m = len(cell_class)
    size = np.bincount(cell, minlength=m)
    errors = np.zeros((len(pairs), 2))
    by_classes = defaultdict(list)  # (class, rival class): the pairs' positions
    for i, (c, k) in enumerate(pairs):
        by_classes[(int(cell_class[c]), int(k))].append(i)
    own = {
        k: np.flatnonzero(labels == k) for k in {k for pair in by_classes for k in pair}
    }
    sorted_of, shares_of = {}, {}
    categories = np.unique(values) if not numeric else None
    for (a, k), positions in sorted(by_classes.items()):
        for c in (a, k):
            if c in sorted_of or c in shares_of:
                continue
            if numeric:
                sorted_of[c] = np.sort(values[own[c]])
            else:
                shares_of[c] = _shares(values[own[c]], categories)
        if numeric:
            rule = _threshold(sorted_of[a], sorted_of[k])
        else:
            rule = _split(shares_of[a], shares_of[k])
        rival = float(rule.first(values[own[k]]).mean())
        wrong = np.bincount(
            cell[own[a]], weights=~rule.first(values[own[a]]), minlength=m
        )
        for i in positions:
            c = pairs[i][0]
            errors[i] = (wrong[c] / size[c], rival)
    return errors


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


def rival_defects(
    rivals: Iterable[RivalMeasure], tau: float, confidence: float
) -> dict[tuple[str, int], bool | None]:
    """Return the verdict of every rival defect of a cell against each rival class.

    Against a rival class, each measure read with its interval
    (share_bounds, verdict): suspect_label is the rival share against
    1/2, class_overlap against tau. A column is a shortcut when the
    upper bounds of both errors of its rule (rule_errors) lie under
    tau, none when the lower bound of either reaches it. shortcut is
    True when some column is a shortcut, False when no column is, None
    otherwise: it is read on its own, whatever the rival share.

    Args:
        rivals: Per rival class, what the cell was measured on.
        tau: The threshold of class_overlap and shortcut.
        confidence: The level of the intervals.

    Returns:
        By (defect, class), its verdict: True, False or None.
    """
    found: dict[tuple[str, int], bool | None] = {}
    for measure in rivals:
        k = int(measure.rival_class)
        bounds = share_bounds(measure.share, measure.n_rows, confidence)
        found[("suspect_label", k)] = verdict(*bounds, 0.5)
        found[("class_overlap", k)] = verdict(*bounds, tau)
        columns = []
        for cell_error, rival_error in measure.rules:
            sides = (
                share_bounds(cell_error, measure.n_rows, confidence),
                share_bounds(rival_error, measure.n_rival, confidence),
            )
            low, high = max(b[0] for b in sides), max(b[1] for b in sides)
            # the worse side of the rule against tau: True when it is over it
            columns.append(verdict(low, high, tau))
        if any(v is False for v in columns):
            found[("shortcut", k)] = True
        elif all(v is True for v in columns):
            found[("shortcut", k)] = False
        else:
            found[("shortcut", k)] = None
    return found


def dataset_variance(points: np.ndarray) -> float:
    """Return the variance of the dataset: the mean squared distance of its rows to their centroid.

    Args:
        points: Every distinct row, shape (n, d), n >= 1.

    Returns:
        The variance, >= 0; about d for standardized columns.
    """
    points = np.asarray(points, dtype=np.float64)
    centre = points.mean(axis=0)
    total = 0.0
    for start in range(0, len(points), _CHUNK):  # no copy of every row at once
        part = points[start : start + _CHUNK] - centre
        total += float(np.einsum("ij,ij->", part, part))
    return total / len(points)


def branch_spread(
    points: np.ndarray, cell: np.ndarray, parent: np.ndarray, node_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far the rows of every branch of a tree lie from their centre.

    Exact, from the sums of every cell (its rows, their sum, the sum of
    their squared norms), added up the tree: the variance of a branch
    is the mean squared distance of its rows to its centroid, as
    dataset_variance's of the dataset.

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


def repetition_verdicts(
    parent: np.ndarray,
    node_cell: np.ndarray,
    n_read: np.ndarray,
    variance: np.ndarray,
    reference: float,
    tau: float,
) -> tuple[list[bool], np.ndarray]:
    """Return whether every cell lies in a large branch of its class as tight as tau of the dataset.

    A branch is large when it holds at least tau of the CSV rows of its
    root (its class), and tight when its variance is at most tau times
    the dataset's. A cell has repetition when a large branch above it
    (or itself) is tight. No interval: a large branch holds many rows.

    Args:
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for none.
        n_read: The CSV rows of every node.
        variance: The variance of the rows of every node
            (branch_spread).
        reference: The variance of the dataset (dataset_variance).
        tau: The threshold of both the size and the ratio.

    Returns:
        Per cell (indexed by cell, 0 to the most cell): whether it has
        repetition, and the node of the largest tight branch above it,
        -1 for none.
    """
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    n_read = np.asarray(n_read, dtype=np.float64)
    k = len(parent)
    root = np.arange(k)
    top = np.full(k, -1)  # the highest tight large branch on the path
    for v in range(k):  # a parent before its children
        p = parent[v]
        if p >= 0:
            root[v], top[v] = root[p], top[p]
        large = n_read[v] >= tau * n_read[root[v]]
        if large and top[v] < 0 and variance[v] <= tau * reference:
            top[v] = v
    m = int(node_cell.max()) + 1 if len(node_cell) else 0
    found = [False] * m
    of_cell = np.full(m, -1)
    for v in np.flatnonzero(node_cell >= 0):
        c = int(node_cell[v])
        of_cell[c] = top[v]
        found[c] = bool(top[v] >= 0)
    return found, of_cell


def defects_of(
    missing_rows: int,
    n_rows: int,
    n_read: float,
    conflict_rows: int,
    established: Iterable[str],
) -> tuple[str, ...]:
    """Return the defects of a cell or of a region of cells, in DEFECTS order.

    invalid_values when a row has a null or invalid cell, duplicates
    when it has copies (n_read > n_rows), label_conflict when a row
    equals a row of another class; the others when they are
    established (a True verdict against some class, key or its own
    class).

    Args:
        missing_rows: Its rows with a null or invalid cell.
        n_rows: Its distinct rows.
        n_read: Its rows in the CSV.
        conflict_rows: Its rows equal to a row of another class.
        established: The defects of RIVAL_DEFECTS and CLASS_DEFECTS it
            has.

    Returns:
        The defects; empty when it has none.
    """
    found = {
        "invalid_values": missing_rows > 0,
        "duplicates": n_read > n_rows,
        "label_conflict": conflict_rows > 0,
        **dict.fromkeys(RIVAL_DEFECTS + CLASS_DEFECTS, False),
    }
    found.update(dict.fromkeys(established, True))
    return tuple(defect for defect in DEFECTS if found[defect])


_TIE = 1e-9  # relative tolerance under which two distances tie
_CHUNK = 1 << 20  # rows taken at once from a large array


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


def _threshold(sa: np.ndarray, sb: np.ndarray) -> Rule:
    """Return the threshold of highest balanced accuracy between two sorted samples (class_rule).

    The difference of their distribution functions changes only at their
    values; between two values of the smaller sample it is monotone, so
    its extremes lie at a value of it or just under one: the search
    reads both limits there, with "le"/"gt" at a value and "lt"/"ge"
    just under it.
    """
    small = np.unique(sa if len(sa) <= len(sb) else sb)
    na, nb = len(sa), len(sb)
    fa_r, fb_r = (
        np.searchsorted(x, small, side="right") / n for x, n in ((sa, na), (sb, nb))
    )
    fa_l, fb_l = (
        np.searchsorted(x, small, side="left") / n for x, n in ((sa, na), (sb, nb))
    )
    candidates = (  # (kind, accuracy on a + accuracy on b)
        ("le", fa_r + 1 - fb_r),
        ("lt", fa_l + 1 - fb_l),
        ("gt", 1 - fa_r + fb_r),
        ("ge", 1 - fa_l + fb_l),
    )
    best = max(range(4), key=lambda i: candidates[i][1].max())
    kind, score = candidates[best]
    if score.max() <= 1.0:  # no rule beats chance
        return Rule("values", np.empty(0))
    return Rule(kind, float(small[np.argmax(score)]))


def _shares(
    values: np.ndarray, categories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the categories and the share of a sample's rows at each, shape (len(categories),)."""
    at = np.searchsorted(categories, values)
    return categories, np.bincount(at, minlength=len(categories)) / max(len(values), 1)


def _split(a: tuple, b: tuple) -> Rule:
    """Return the rule that sends each category to the side where it is the larger share (a tie to a)."""
    categories, share_a = a
    _, share_b = b
    return Rule("values", categories[share_b > share_a])
