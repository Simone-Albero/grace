"""Data-quality defects of the cells, after Flood, Engelen, Aspinall & Desmet.

"Bad Design Smells in Benchmark NIDS Datasets" (EuroS&P 2024), judged
on the cells of `src.cells`, each by its relation to the other cells
(DEFECTS, defects_of):
- invalid_values: a row holds a null or invalid cell (invalid_values);
- duplicates: a row comes more than once in the CSV, so copies can
  fall on both sides of a train/test split;
- label_conflict: a row equals, on every feature, a row of another
  class: no classifier can tell them apart (label_conflicts);
- suspect_label: against a rival class, at least half of its rows are
  confused with that class's rows: the cell lies inside the other
  class, as rows under a wrong label would;
- class_overlap: against a rival class, the share of its rows confused
  with that class's rows reaches tau, and it is no suspect_label;
- shortcut: against a rival class it does not overlap, the best rule
  on a single column between its class and the rival class, both
  whole (class_rule), leaves under tau of its rows and under tau of
  the rival class's rows on the wrong side (rule_errors): one column
  alone tells the two classes apart where the cell lies;
- redundancy: its rows lie at most tau times as far from their centre
  as those of the typical cell of its class (cell_variance): near
  copies of one another, against the rest of its class;
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
between a cell and a rival class (RIVAL_DEFECTS, rival_defects): its
rival cells of that class pooled for the confusion, the rival class
whole for the shortcut. The last three between a cell and a reference
of its own (CLASS_DEFECTS): its class, the dataset, the scenario. The
rows confused are those that the better of two simple classifiers gets
wrong (pair_confusion): the leave-one-out 1-NN on every column
(nearest_error) and the best rule on a single column (single_column,
OneR of Holte 1993).

Every measure of a cell is an estimate on few rows, so it is read with
an interval at a confidence level (share_bounds, mean_bounds), and a
cell is judged against a threshold beyond the error of sampling
(verdict): it has the defect when the interval lies past the
threshold, has none when it lies short of it, and is undecided
otherwise. tau is the one threshold chosen; one half, the majority,
is the other. The rival cells of a cell are the cells of the nearest
rows of other classes to its rows (rival_pairs): no threshold picks
them. A cell is measured on all its rows, so that min_rows holds what
it promises; its rival cells of a class are pooled and sampled down to
quality.max_rows_per_cell, which in many dimensions barely moves the
nearest rival row (thinning by r multiplies its distance by r^(-1/d)).
Every measure but the copies, the size of a branch and the scenario
is computed on distinct rows, unweighted, and read on the cell's side.
"""

from __future__ import annotations

__all__ = [
    "CLASS_DEFECTS",
    "DEFECTS",
    "RIVAL_DEFECTS",
    "ROW_DEFECTS",
    "Confusion",
    "Rule",
    "SingleColumn",
    "cell_samples",
    "cell_variance",
    "class_rule",
    "column_values",
    "branch_spread",
    "dataset_variance",
    "decisive_rows",
    "defects_of",
    "invalid_values",
    "label_conflicts",
    "mean_bounds",
    "nearest_error",
    "pair_confusion",
    "pair_conflicts",
    "repetition_verdicts",
    "rival_defects",
    "rival_pairs",
    "rule_errors",
    "share_bounds",
    "single_column",
    "variance_verdict",
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

from src.graph import sample_rows

ROW_DEFECTS = ("invalid_values", "duplicates", "label_conflict")
RIVAL_DEFECTS = ("suspect_label", "class_overlap", "shortcut")
CLASS_DEFECTS = ("redundancy", "repetition", "off_scenario")
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


def cell_samples(cell: np.ndarray, cap: int, seed: int) -> dict[int, np.ndarray]:
    """Draw the rows each cell is measured on.

    Args:
        cell: The cell of every row.
        cap: The most rows per cell.
        seed: The seed of graph.sample_rows, the same for every cell.

    Returns:
        By cell, its row positions (at most cap, ascending).
    """
    order = np.argsort(cell, kind="stable")
    ids, starts = np.unique(cell[order], return_index=True)
    samples = {}
    for own, rows in zip(ids.tolist(), np.split(order, starts[1:])):
        samples[own] = rows[sample_rows(len(rows), cap, seed)]
    return samples


@dataclass(frozen=True)
class SingleColumn:
    """The best rule on a single column between two cells.

    Attributes:
        side_a: The share of the first cell's rows the rule puts on
            their side.
        side_b: The same for the second cell.
        column: The index (into the column names) of the rule's
            column; -1 when no column has two values.
    """

    side_a: float
    side_b: float
    column: int

    @property
    def share(self) -> float:
        """The balanced accuracy of the rule: the mean of the two sides."""
        return (self.side_a + self.side_b) / 2


@dataclass(frozen=True)
class Confusion:
    """How the better of two classifiers confuses a cell with its rival cells of a class.

    Attributes:
        nn_error: The leave-one-out 1-NN error of each side, the cell's
            then the rival's, on the measured half, shape (2,).
        single: The best rule on a single column, chosen on the other
            half and measured on the measured half.
        confusion: The error of the better of the two on each side,
            shape (2,): the one whose worse side errs less, ties to the
            1-NN.
        measure: Whether each row, the cell's then the rival's, is in
            the measured half, shape (n_a + n_b,).
    """

    nn_error: np.ndarray
    single: SingleColumn
    confusion: np.ndarray
    measure: np.ndarray


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
        confusion: The confusion of the cell's rows (Confusion), None
            when a side had too few rows.
        n_scored: The cell's rows it is a share of (its measured half).
        rules: Per column whose rule leaves under half of each side on
            the wrong side, the error of class_rule on the cell's rows
            and on the rival class's, shape (c, 2); the other columns
            can never be a shortcut.
        n_cell: The cell's rows the first error is a share of.
        n_rival: The rival class's rows the second is a share of.
    """

    rival_class: int
    confusion: float | None
    n_scored: int | None
    rules: np.ndarray
    n_cell: int
    n_rival: int


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


def nearest_error(sq_dist: np.ndarray, side: np.ndarray) -> tuple[float, float]:
    """Return the leave-one-out 1-NN error of each side of a pair.

    A row votes for the side of its nearest other row; rows tied at the
    least distance (relative tolerance 1e-9) share its vote, so the
    error does not depend on the order of the rows. The error of a side
    is the mean share of its rows' votes that go to the other side.

    Args:
        sq_dist: The squared distances between the rows, shape (n, n),
            n >= 2.
        side: True for a row of the second cell, shape (n,); each
            side holds at least one row.

    Returns:
        The error of the first cell's rows and of the second's, each
        in [0, 1].
    """
    rival = _rival_share(sq_dist, side)
    return float(rival[~side].mean()), float(rival[side].mean())


def single_column(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    numeric: np.ndarray,
    skip: Iterable[int] = (),
    Ya: np.ndarray | None = None,
    Yb: np.ndarray | None = None,
) -> SingleColumn:
    """Find the best rule on a single column between two cells (OneR).

    The rule is chosen on Xa and Xb and its accuracy measured on Ya and
    Yb, so that choosing it does not inflate it. A numeric column (one
    point column) gets the threshold, in either direction, of highest
    balanced accuracy; ties of value stay on one side. A categorical
    column (its one-hot slots, read back as the active slot or none)
    sends each value to the side where it is the larger share of the
    rows, a tie and a value it was not chosen on to the first side.
    The column of highest balanced accuracy on Xa and Xb wins (ties to
    the lower index).

    Args:
        Xa: The rows of the first cell the rule is chosen on, shape
            (n_a, d), n_a >= 1.
        Xb: The rows of the second cell it is chosen on, shape
            (n_b, d), n_b >= 1.
        column_of: The original column of each point column, as an
            index into numeric, shape (d,).
        numeric: Whether each original column is numeric, shape (c,).
        skip: The original columns left out.
        Ya: The rows of the first cell its accuracy is measured on,
            at least one; None: Xa (the accuracy is then the one it
            was chosen for).
        Yb: The same for the second cell; None: Xb.

    Returns:
        The rule's accuracy on each side of Ya and Yb and its column;
        (0.5, 0.5, -1) when no column of Xa and Xb has two values.
    """
    Ya = Xa if Ya is None else Ya
    Yb = Xb if Yb is None else Yb
    best, chosen, column = -1.0, None, -1
    skip = set(skip)
    for c in range(len(numeric)):
        slots = column_of == c
        if c in skip or not slots.any():
            continue
        va, vb = column_values(Xa[:, slots], numeric[c]), column_values(
            Xb[:, slots], numeric[c]
        )
        if len(np.unique(np.r_[va, vb])) < 2:
            continue
        rule = _stump(va, vb) if numeric[c] else _majority(va, vb)
        share = sum(_accuracy(rule, va, vb)) / 2
        if share > best:
            best, chosen, column = share, rule, c
    if chosen is None:
        return SingleColumn(0.5, 0.5, -1)
    slots = column_of == column
    ya, yb = (column_values(Y[:, slots], numeric[column]) for Y in (Ya, Yb))
    side_a, side_b = _accuracy(chosen, ya, yb)
    return SingleColumn(side_a, side_b, column)


def pair_confusion(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    numeric: np.ndarray,
    seed: int | tuple[int, ...],
) -> Confusion | None:
    """Return the share of each side's rows the better of two classifiers confuses.

    Each side is split at random into halves: the single column is
    chosen on the first and both classifiers are measured on the
    second, so choosing does not inflate them. The classifiers are the
    leave-one-out 1-NN on every column (nearest_error) and the best rule
    on a single column (single_column); the better is the one whose
    worse side errs less (ties to the 1-NN). Each holds the error of the
    best classifier above it: the 1-NN alone overstates it when columns
    that do not separate swamp the distance, and a single column misses
    a separation that needs several. The worse side, not the mean of
    the two, so that a rule that sends nearly every row to one side,
    right on it and wrong on the other, is no better than the 1-NN.

    Args:
        Xa: The rows of the cell, shape (n_a, d).
        Xb: The rows of its rival cells of a class, shape (n_b, d).
        column_of: The original column of each point column, as an
            index into numeric, shape (d,).
        numeric: Whether each original column is numeric.
        seed: The seed of the halves (np.random.default_rng's).

    Returns:
        The Confusion; None when a side has fewer than 4 rows.
    """
    n_a, n_b = len(Xa), len(Xb)
    if min(n_a, n_b) < 4:
        return None
    rng = np.random.default_rng(seed)
    measure = np.zeros(n_a + n_b, dtype=bool)
    for start, n in ((0, n_a), (n_a, n_b)):
        measure[start + rng.permutation(n)[: n // 2]] = True
    on_a, on_b = measure[:n_a], measure[n_a:]
    X = np.vstack([Xa[on_a], Xb[on_b]]).astype(np.float64)
    side = np.r_[np.zeros(on_a.sum(), dtype=bool), np.ones(on_b.sum(), dtype=bool)]
    norms = np.einsum("ij,ij->i", X, X)
    sq = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
    nn_error = np.array(nearest_error(sq, side))
    single = single_column(
        Xa[~on_a], Xb[~on_b], column_of, numeric, Ya=Xa[on_a], Yb=Xb[on_b]
    )
    one = np.array([1 - single.side_a, 1 - single.side_b])
    confusion = nn_error if nn_error.max() <= one.max() else one
    return Confusion(nn_error, single, confusion, measure)


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


def rival_pairs(
    points: np.ndarray, cell: np.ndarray, cell_class: np.ndarray
) -> list[tuple[int, int]]:
    """Return the pairs of a cell and the cells of its rows' nearest rivals.

    For every row, its nearest row of another class (euclidean, exact:
    a kd-tree on the rows of the other classes, one per class, so the
    memory is linear in the rows; of rows at the same distance, one,
    the same on every run); each cell is paired with the cells those
    rows lie in.

    Args:
        points: The rows, shape (n, d) (the sampled rows of every cell).
        cell: The cell of every row, shape (n,).
        cell_class: The class of every cell.

    Returns:
        The pairs (a, b), a < b, sorted; none when the rows are of one
        class.
    """
    points = np.asarray(points, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.int64)
    classes = np.asarray(cell_class)[cell]
    if len(np.unique(classes)) < 2:
        return []
    found: set[tuple[int, int]] = set()
    for k in np.unique(classes):
        own = classes == k
        search = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
        search.fit(points[~own])
        nearest = search.kneighbors(points[own], return_distance=False)[:, 0]
        a, b = cell[own], cell[~own][nearest]
        found.update(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))
    return sorted(found)


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


def mean_bounds(
    mean: float, sd: float, n: float, confidence: float
) -> tuple[float, float]:
    """Return the interval of a mean of n values: mean ∓ z · sd / sqrt(n).

    Each bound is one-sided at confidence, the normal approximation of
    the mean.

    Args:
        mean: The mean observed.
        sd: The standard deviation of the values.
        n: The values, >= 1; (-inf, inf) when none.
        confidence: The level, in (0.5, 1).

    Returns:
        The lower and the upper bound.
    """
    if n <= 0:
        return -math.inf, math.inf
    half = NormalDist().inv_cdf(confidence) * sd / math.sqrt(n)
    return mean - half, mean + half


def decisive_rows(tau: float, confidence: float) -> int:
    """Return the fewest rows a share must be measured on to be able to fall short of tau.

    With no row of n confused, the upper bound of the share
    (share_bounds) is still over tau when n is small: under that many
    rows a cell can never be told to lack the defect, whatever it
    holds. The bound is under tau from n > z² (1 - tau) / tau, with z
    the quantile of confidence; this returns the first such n.

    Args:
        tau: The threshold, in (0, 0.5].
        confidence: The level of the interval, in (0.5, 1).

    Returns:
        The rows, >= 1.
    """
    z = NormalDist().inv_cdf(confidence)
    n = math.floor(z * z * (1 - tau) / tau) + 1
    while share_bounds(0.0, n, confidence)[1] >= tau:  # rounding at the edge
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
    (share_bounds, verdict): suspect_label is the confusion against
    1/2, class_overlap against tau. A column is a shortcut when the
    upper bounds of both errors of its rule (rule_errors) lie under
    tau, none when the lower bound of either reaches it. shortcut is
    True when the cell is apart (class_overlap False) and some column
    is a shortcut, False when it is not apart or no column is, None
    otherwise. Without a confusion (a side too small) the three are
    None.

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
        if measure.confusion is None:
            found.update({(d, k): None for d in RIVAL_DEFECTS})
            continue
        bounds = share_bounds(measure.confusion, measure.n_scored, confidence)
        overlap = verdict(*bounds, tau)
        found[("suspect_label", k)] = verdict(*bounds, 0.5)
        found[("class_overlap", k)] = overlap
        if overlap is not False:  # overlapping, or maybe
            found[("shortcut", k)] = False if overlap else None
            continue
        columns = []
        for cell_error, rival_error in measure.rules:
            sides = (
                share_bounds(cell_error, measure.n_cell, confidence),
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


def cell_variance(
    points: np.ndarray, cell: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far the rows of each cell lie from their centre.

    Args:
        points: The rows, shape (n, d) (distinct, unweighted).
        cell: The cell of every row, in [0, m).
        m: The number of cells.

    Returns:
        Per cell, shape (m,): the mean of its rows' squared distances
        to its centroid (its variance), and their standard deviation; 0
        for a cell of one row.
    """
    points = np.asarray(points, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.int64)
    count = np.bincount(cell, minlength=m).astype(np.float64)
    centre = np.zeros((m, points.shape[1]))
    np.add.at(centre, cell, points)
    centre /= np.maximum(count, 1)[:, None]
    sq = ((points - centre[cell]) ** 2).sum(axis=1)
    mean = np.bincount(cell, weights=sq, minlength=m) / np.maximum(count, 1)
    second = np.bincount(cell, weights=sq * sq, minlength=m) / np.maximum(count, 1)
    return mean, np.sqrt(np.maximum(second - mean * mean, 0.0))


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return how far the rows of every branch of a tree lie from their centre.

    Exact, from the sums of every cell (its rows, their sum, the sum of
    their squared norms, of their outer products, of their squared norm
    times themselves, of their fourth powers), added up the tree: the
    variance of a branch is the mean squared distance of its rows to its
    centroid, as cell_variance's of a cell, and its standard deviation
    that of those squared distances, what mean_bounds needs.

    Args:
        points: The rows, shape (n, d) (distinct, unweighted).
        cell: The cell of every row.
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for a node that is not a
            cell; the cells are the leaves.

    Returns:
        Per node, shape (k,): the variance of its rows, its standard
        deviation, and its rows.
    """
    points = np.asarray(points, dtype=np.float64)
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    centre = points.mean(axis=0)  # moments about it: no cancellation far away
    order = np.argsort(np.asarray(cell), kind="stable")
    ids, starts = np.unique(np.asarray(cell)[order], return_index=True)
    rows_of = dict(zip(ids.tolist(), np.split(order, starts[1:])))
    k = len(parent)
    variance, sd, n_rows = np.zeros(k), np.zeros(k), np.zeros(k, dtype=np.int64)
    pending: dict[int, list] = {}  # the sums of the children seen of a node
    for v in range(k - 1, -1, -1):  # children after their parent
        if node_cell[v] >= 0:
            Z = points[rows_of.get(int(node_cell[v]), np.empty(0, int))] - centre
            norm = np.einsum("ij,ij->i", Z, Z)
            sums = [len(Z), Z.sum(0), norm.sum(), Z.T @ Z, norm @ Z, (norm**2).sum()]
        else:
            sums = pending.pop(v)
        n, s1, s2, outer, t, q = sums
        if n:
            c = s1 / n
            cc = c @ c
            mean = s2 / n - cc
            fourth = q - 4 * c @ t + 4 * c @ outer @ c + 2 * cc * s2 - 3 * n * cc * cc
            variance[v] = max(mean, 0.0)
            sd[v] = math.sqrt(max(fourth / n - mean * mean, 0.0))
            n_rows[v] = n
        if parent[v] >= 0:
            if parent[v] in pending:
                pending[parent[v]] = [a + b for a, b in zip(pending[parent[v]], sums)]
            else:
                pending[parent[v]] = list(sums)
    return variance, sd, n_rows


def repetition_verdicts(
    parent: np.ndarray,
    node_cell: np.ndarray,
    n_read: np.ndarray,
    spread: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference: float,
    tau: float,
    confidence: float,
) -> tuple[list[bool | None], np.ndarray]:
    """Return whether every cell lies in a large branch of its class as tight as tau of the dataset.

    A branch is large when it holds at least tau of the CSV rows of its
    root (its class); a large branch is tight when its variance is at
    most tau times the dataset's (variance_verdict, with its interval).
    A cell has repetition when a large branch above it (or itself) is
    tight, has it not when every large branch above it is not, and is
    undecided otherwise.

    Args:
        parent: The parent of every node, -1 for a root; a parent comes
            before its children.
        node_cell: The cell of every node, -1 for none.
        n_read: The CSV rows of every node.
        spread: The variance, its standard deviation and the rows of
            every node (branch_spread).
        reference: The variance of the dataset (dataset_variance).
        tau: The threshold of both the size and the ratio.
        confidence: The level of the interval.

    Returns:
        Per cell (indexed by cell, 0 to the most cell): its verdict, and
        the node of the largest tight branch above it, -1 for none.
    """
    parent, node_cell = np.asarray(parent), np.asarray(node_cell)
    n_read = np.asarray(n_read, dtype=np.float64)
    variance, sd, n_rows = spread
    k = len(parent)
    root = np.arange(k)
    top = np.full(k, -1)  # the highest tight large branch on the path
    open_ = np.zeros(k, dtype=bool)  # an undecided large branch on the path
    for v in range(k):  # a parent before its children
        p = parent[v]
        if p >= 0:
            root[v], top[v], open_[v] = root[p], top[p], open_[p]
        if n_read[v] < tau * n_read[root[v]]:  # too small a part of its class
            continue
        held = variance_verdict(
            variance[v], sd[v], n_rows[v], reference, tau, confidence
        )
        if held is True and top[v] < 0:
            top[v] = v
        elif held is None:
            open_[v] = True
    m = int(node_cell.max()) + 1 if len(node_cell) else 0
    found: list[bool | None] = [None] * m
    of_cell = np.full(m, -1)
    for v in np.flatnonzero(node_cell >= 0):
        c = int(node_cell[v])
        of_cell[c] = top[v]
        found[c] = True if top[v] >= 0 else (None if open_[v] else False)
    return found, of_cell


def variance_verdict(
    variance: float,
    sd: float,
    n: int,
    typical: float,
    tau: float,
    confidence: float,
) -> bool | None:
    """Return whether a variance is at most tau times a typical one beyond the error of sampling.

    Args:
        variance: The cell's variance (cell_variance).
        sd: The standard deviation of its rows' squared distances.
        n: Its rows.
        typical: The variance it is compared with: of the typical cell
            of its class (redundancy), or of the dataset for a branch
            (repetition).
        tau: The threshold of the ratio.
        confidence: The level of the interval (mean_bounds).

    Returns:
        True when the upper bound of the variance is at most tau ·
        typical, False when the lower bound is over it, None otherwise
        (or when typical is 0).
    """
    if not typical:
        return None
    low, high = mean_bounds(variance, sd, n, confidence)
    if high <= tau * typical:
        return True
    if low > tau * typical:
        return False
    return None


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


def _rival_share(sq_dist: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Return, per row, the share of its nearest other rows (ties together) of the other side."""
    sq = sq_dist.copy()
    np.fill_diagonal(sq, np.inf)
    least = sq.min(axis=1, keepdims=True)
    near = sq <= least + _TIE * np.maximum(least, 1.0)
    other = side[None, :] != side[:, None]
    return (near & other).sum(axis=1) / near.sum(axis=1)


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


def _stump(va: np.ndarray, vb: np.ndarray) -> tuple[str, float]:
    """Return the threshold on a number of highest balanced accuracy: ("below", t) puts v <= t on a, ("above", t) v > t."""
    cuts = np.unique(np.r_[va, vb])
    below_a = np.searchsorted(np.sort(va), cuts, side="right") / len(va)
    below_b = np.searchsorted(np.sort(vb), cuts, side="right") / len(vb)
    balanced = np.r_[[below_a + 1 - below_b], [1 - below_a + below_b]]  # (way, cut)
    way, cut = np.unravel_index(np.argmax(balanced), balanced.shape)
    return ("below", "above")[way], float(cuts[cut])


def _majority(va: np.ndarray, vb: np.ndarray) -> tuple[str, np.ndarray]:
    """Return ("to_b", the values whose share is larger on b): every other value, unseen ones too, goes to a."""
    values = np.unique(np.r_[va, vb])
    share_a = np.array([(va == v).mean() for v in values])
    share_b = np.array([(vb == v).mean() for v in values])
    return "to_b", values[share_b > share_a]


def _accuracy(rule: tuple, va: np.ndarray, vb: np.ndarray) -> tuple[float, float]:
    """Return the share of each side's values a rule of _stump or _majority puts on its side."""
    kind, at = rule
    if kind == "to_b":
        on_a = lambda v: ~np.isin(v, at)  # noqa: E731
    elif kind == "below":
        on_a = lambda v: v <= at  # noqa: E731
    else:
        on_a = lambda v: v > at  # noqa: E731
    return float(on_a(va).mean()), float(1 - on_a(vb).mean())
