"""Data-quality defects of the cells, after Flood, Engelen, Aspinall & Desmet.

"Bad Design Smells in Benchmark NIDS Datasets" (EuroS&P 2024), judged
on the cells of `src.cells`. A cell, or a region of cells, has a defect
when (DEFECTS, defects_of):
- invalid_values: a row holds a null or invalid cell (invalid_values);
- duplicates: a row comes more than once in the CSV, so copies can
  fall on both sides of a train/test split;
- label_conflict: a row equals, on every feature, a row of another
  class: no classifier can tell them apart (label_conflicts);
- class_overlap: against a rival cell, the share of its rows confused
  with the rival's reaches tau;
- shortcut: against a rival that share is under tau, but removing at
  most max_group columns, chosen on half of the rows and measured on
  the other half (feature_dependence), brings it to tau: the separation
  rests on those few columns.
The first three are facts of single rows (ROW_DEFECTS): counted in any
group of rows, they are never diluted. The last two are measured between
two cells (RIVAL_DEFECTS, rival_defects), so the cells are small enough
that a defect touching tau of a cell's rows is seen. The rows confused
are those that the better of two simple classifiers gets wrong
(pair_confusion): the leave-one-out 1-NN on every column (nearest_error;
by Cover & Hart its error lies between the least error any classifier
can reach and twice it, cover_hart_low) and the best rule on a single
column (single_column, OneR of Holte 1993). Any classifier errs at least
as much as the best one can, so when the better of the two confuses tau
of a cell's rows, the cell overlaps its rival; the 1-NN alone would
overstate it when columns that do not separate swamp the distance. tau
is the one threshold: the share of a cell's rows that may be confused.
The rivals of a cell are the cells of the nearest rows of other classes
to its rows (rival_pairs): no threshold picks them. Every measure but
the copies is computed on distinct rows, unweighted, and per side of a
pair.
"""

from __future__ import annotations

__all__ = [
    "DEFECTS",
    "RIVAL_DEFECTS",
    "ROW_DEFECTS",
    "Dependence",
    "SingleColumn",
    "cell_samples",
    "cover_hart_low",
    "defects_of",
    "feature_dependence",
    "invalid_values",
    "label_conflicts",
    "nearest_error",
    "pair_confusion",
    "pair_conflicts",
    "rival_defects",
    "rival_pairs",
    "single_column",
]

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.graph import sample_rows

ROW_DEFECTS = ("invalid_values", "duplicates", "label_conflict")
RIVAL_DEFECTS = ("class_overlap", "shortcut")
DEFECTS = ROW_DEFECTS + RIVAL_DEFECTS


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
class Dependence:
    """How a pair's 1-NN error grows as columns are removed.

    Attributes:
        group: The columns removed, in the order chosen (indices into
            the column names), at most max_group.
        errors: Shape (len(group) + 1, 2): the 1-NN error of each side
            (nearest_error) on the measured half, with every column
            (row 0) and after removing the first i columns of group
            (row i).
    """

    group: tuple[int, ...]
    errors: np.ndarray


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


def cover_hart_low(error: float) -> float:
    """Return Cover & Hart's lower bound of the Bayes error under a 1-NN error.

    With two classes, R* >= (1 - sqrt(1 - 2 R)) / 2 for the asymptotic
    1-NN error R (R clipped to 0.5). On a finite sample it is a guide,
    not a guarantee.

    Args:
        error: A 1-NN error in [0, 1].

    Returns:
        The bound, in [0, 0.5].
    """
    return (1 - math.sqrt(max(0.0, 1 - 2 * min(error, 0.5)))) / 2


def single_column(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    numeric: np.ndarray,
    skip: Iterable[int] = (),
) -> SingleColumn:
    """Find the best rule on a single column between two cells (OneR).

    A numeric column (one point column) gets the threshold, in either
    direction, of highest balanced accuracy; ties of value stay on one
    side. A categorical column (its one-hot slots, read back as the
    active slot or none) sends each value to the side where it is the
    larger share of the rows. The column of highest balanced accuracy
    wins (ties to the lower index).

    Args:
        Xa: The rows of the first cell, shape (n_a, d), n_a >= 1.
        Xb: The rows of the second cell, shape (n_b, d), n_b >= 1.
        column_of: The original column of each point column, as an
            index into numeric, shape (d,).
        numeric: Whether each original column is numeric, shape (c,).
        skip: The original columns left out.

    Returns:
        The rule's accuracy on each side and its column; (0.5, 0.5, -1)
        when no column is left with two values.
    """
    best, sides, column = -1.0, (0.5, 0.5), -1
    skip = set(skip)
    for c in range(len(numeric)):
        slots = column_of == c
        if c in skip or not slots.any():
            continue
        va, vb = _values(Xa[:, slots], numeric[c]), _values(Xb[:, slots], numeric[c])
        if len(np.unique(np.r_[va, vb])) < 2:
            continue
        rule = _stump(va, vb) if numeric[c] else _majority(va, vb)
        share = (rule[0] + rule[1]) / 2
        if share > best:
            best, sides, column = share, rule, c
    return SingleColumn(float(sides[0]), float(sides[1]), column)


def feature_dependence(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    max_group: int,
    seed: int | tuple[int, ...],
) -> Dependence | None:
    """Find the columns whose removal raises a pair's 1-NN error the most.

    Each side is split at random into halves: the columns are chosen
    on the first, the errors measured on the second, so the search does
    not inflate them. Greedy: max_group times, the column (all the
    slots of an original column together) whose removal gives the
    highest mean of the two sides' errors on the choosing half, ties
    broken by the highest mean margin d_own / (d_own + d_other)
    (nearest row of the same side, of the other). The squared distance
    is the sum of the columns' parts, so a removal subtracts one part.

    Args:
        Xa: The rows of the first cell, shape (n_a, d).
        Xb: The rows of the second cell, shape (n_b, d).
        column_of: The original column of each point column, shape (d,).
        max_group: The most columns removed.
        seed: The seed of the halves (np.random.default_rng's).

    Returns:
        The group and the errors on the measured half; None when a
        side has fewer than 4 rows.
    """
    n_a, n_b = len(Xa), len(Xb)
    if min(n_a, n_b) < 4:
        return None
    rng = np.random.default_rng(seed)
    measure = np.zeros(n_a + n_b, dtype=bool)
    for start, n in ((0, n_a), (n_a, n_b)):
        measure[start + rng.permutation(n)[: n // 2]] = True
    X = np.vstack([Xa, Xb])
    side = np.r_[np.zeros(n_a, dtype=bool), np.ones(n_b, dtype=bool)]
    columns = np.unique(column_of)
    choose_parts = _column_parts(X[~measure], column_of, columns)
    measure_parts = _column_parts(X[measure], column_of, columns)
    choose_side, measure_side = side[~measure], side[measure]
    choose_total, measure_total = choose_parts.sum(0), measure_parts.sum(0)
    group: list[int] = []
    errors = [nearest_error(measure_total, measure_side)]
    for _ in range(min(max_group, len(columns))):
        best, best_score = -1, (-1.0, -1.0)
        for i in range(len(columns)):
            if i in group:
                continue
            sq = choose_total - choose_parts[i]
            rival = _rival_share(sq, choose_side)
            error = (rival[~choose_side].mean() + rival[choose_side].mean()) / 2
            score = (error, _margin(sq, choose_side))
            if score > best_score:
                best, best_score = i, score
        group.append(best)
        choose_total = choose_total - choose_parts[best]
        measure_total = measure_total - measure_parts[best]
        errors.append(nearest_error(measure_total, measure_side))
    return Dependence(tuple(int(columns[i]) for i in group), np.array(errors))


def pair_confusion(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    numeric: np.ndarray,
    dependence: Dependence,
) -> np.ndarray:
    """Return the share of each side's rows the better of two classifiers confuses.

    The classifiers are the 1-NN on every column (dependence.errors)
    and the best rule on a single column (single_column); the better is
    the one of the lower mean error over the two sides. Each holds the
    error of the best classifier above it: the 1-NN alone overstates it
    when columns that do not separate swamp the distance, and a single
    column misses a separation that needs several. After removing the
    first i columns of dependence.group, the single column is sought
    among the others.

    Args:
        Xa: The rows of the first cell, shape (n_a, d).
        Xb: The rows of the second cell, shape (n_b, d).
        column_of: The original column of each point column.
        numeric: Whether each original column is numeric.
        dependence: feature_dependence of the two cells.

    Returns:
        Shape (len(dependence.group) + 1, 2): the confusion of each side
        with every column (row 0) and after removing the first i
        columns of the group (row i).
    """
    confusion = np.empty_like(dependence.errors)
    for i, nearest in enumerate(dependence.errors):
        rule = single_column(Xa, Xb, column_of, numeric, dependence.group[:i])
        one = np.array([1 - rule.side_a, 1 - rule.side_b])
        confusion[i] = nearest if nearest.mean() <= one.mean() else one
    return confusion


def rival_pairs(
    points: np.ndarray, cell: np.ndarray, cell_class: np.ndarray
) -> list[tuple[int, int]]:
    """Return the pairs of a cell and the cells of its rows' nearest rivals.

    For every row, its nearest row of another class (euclidean, ties
    to the first); each cell is paired with the cells those rows lie in.

    Args:
        points: The rows, shape (n, d) (the sampled rows of every cell).
        cell: The cell of every row, shape (n,).
        cell_class: The class of every cell.

    Returns:
        The pairs (a, b), a < b, sorted.
    """
    points = np.asarray(points, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.int64)
    classes = np.asarray(cell_class)[cell]
    if len(np.unique(classes)) < 2:
        return []
    norms = np.einsum("ij,ij->i", points, points)
    found: set[tuple[int, int]] = set()
    for start in range(0, len(points), _RIVAL_ROWS):
        part = slice(start, start + _RIVAL_ROWS)
        sq = norms[part, None] + norms[None, :] - 2.0 * points[part] @ points.T
        sq[classes[part, None] == classes[None, :]] = np.inf
        rival = cell[np.argmin(sq, axis=1)]
        for a, b in zip(cell[part].tolist(), rival.tolist()):
            found.add((min(a, b), max(a, b)))
    return sorted(found)


def rival_defects(
    rivals: Iterable[tuple[int, np.ndarray]], tau: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the classes a cell has class_overlap and shortcut against.

    Against a rival, class_overlap when the confusion of the cell's rows
    with every column reaches tau; shortcut when it is under tau but
    reaches it once the first 1, 2, ... key columns are removed.

    Args:
        rivals: Per rival cell, its class and the confusion of this
            cell's rows (its side of pair_confusion): with every column,
            then after removing the first 1, 2, ... columns of
            feature_dependence, shape (k + 1,).
        tau: The share of its rows that may be confused.

    Returns:
        The classes of class_overlap and the classes of shortcut, each
        sorted with no repeat.
    """
    overlap: set[int] = set()
    shortcut: set[int] = set()
    for rival_class, confusion in rivals:
        steps = np.asarray(confusion, dtype=np.float64)
        if steps[0] >= tau:
            overlap.add(int(rival_class))
        elif (steps[1:] >= tau).any():
            shortcut.add(int(rival_class))
    return tuple(sorted(overlap)), tuple(sorted(shortcut))


def defects_of(
    missing_rows: int,
    n_rows: int,
    n_read: float,
    conflict_rows: int,
    overlap: Iterable[int],
    shortcut: Iterable[int],
) -> tuple[str, ...]:
    """Return the defects of a cell or of a region of cells, in DEFECTS order.

    invalid_values when a row has a null or invalid cell, duplicates
    when it has copies (n_read > n_rows), label_conflict when a row
    equals a row of another class; class_overlap and shortcut when
    there is a class against which it has them (rival_defects).

    Args:
        missing_rows: Its rows with a null or invalid cell.
        n_rows: Its distinct rows.
        n_read: Its rows in the CSV.
        conflict_rows: Its rows equal to a row of another class.
        overlap: The classes it has class_overlap against.
        shortcut: The classes it has shortcut against.

    Returns:
        The defects; empty when it has none.
    """
    found = {
        "invalid_values": missing_rows > 0,
        "duplicates": n_read > n_rows,
        "label_conflict": conflict_rows > 0,
        "class_overlap": bool(tuple(overlap)),
        "shortcut": bool(tuple(shortcut)),
    }
    return tuple(defect for defect in DEFECTS if found[defect])


_TIE = 1e-9  # relative tolerance under which two distances tie
_RIVAL_ROWS = 256  # rows whose nearest rival is searched at once


def _rival_share(sq_dist: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Return, per row, the share of its nearest other rows (ties together) of the other side."""
    sq = sq_dist.copy()
    np.fill_diagonal(sq, np.inf)
    least = sq.min(axis=1, keepdims=True)
    near = sq <= least + _TIE * np.maximum(least, 1.0)
    other = side[None, :] != side[:, None]
    return (near & other).sum(axis=1) / near.sum(axis=1)


def _margin(sq_dist: np.ndarray, side: np.ndarray) -> float:
    """Return the mean over the two sides of d_own / (d_own + d_other), 0.5 at d = 0.

    The squared distances may come from subtracted parts, a hair under 0.
    """
    sq = np.maximum(sq_dist, 0.0)
    np.fill_diagonal(sq, np.inf)
    same = side[None, :] == side[:, None]
    own = np.sqrt(np.where(same, sq, np.inf).min(axis=1))
    other = np.sqrt(np.where(~same, sq, np.inf).min(axis=1))
    both = own + other
    ratio = np.full(len(own), 0.5)
    np.divide(own, both, out=ratio, where=np.isfinite(both) & (both > 0))
    return float((ratio[~side].mean() + ratio[side].mean()) / 2)


def _column_parts(
    X: np.ndarray, column_of: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Return the squared distances each original column adds, shape (c, n, n)."""
    parts = np.empty((len(columns), len(X), len(X)))
    for i, c in enumerate(columns):
        Z = X[:, column_of == c]
        sq = (Z * Z).sum(axis=1)
        parts[i] = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    return parts


def _values(Z: np.ndarray, numeric: bool) -> np.ndarray:
    """Return a column's values: the number, or the active slot (-1 for none)."""
    if numeric:
        return Z[:, 0]
    return np.where(Z.max(axis=1) > 0, Z.argmax(axis=1), -1).astype(float)


def _stump(va: np.ndarray, vb: np.ndarray) -> tuple[float, float]:
    """Return the per-side accuracy of the best threshold on a number, either way."""
    cuts = np.unique(np.r_[va, vb])
    below_a = np.searchsorted(np.sort(va), cuts, side="right") / len(va)
    below_b = np.searchsorted(np.sort(vb), cuts, side="right") / len(vb)
    rules = np.r_[[below_a, 1 - below_b], [1 - below_a, below_b]].reshape(2, 2, -1)
    balanced = rules.mean(axis=1)  # (direction, cut)
    way, cut = np.unravel_index(np.argmax(balanced), balanced.shape)
    return float(rules[way, 0, cut]), float(rules[way, 1, cut])


def _majority(va: np.ndarray, vb: np.ndarray) -> tuple[float, float]:
    """Return the per-side accuracy of sending each value where it is the larger share."""
    values = np.unique(np.r_[va, vb])
    share_a = np.array([(va == v).mean() for v in values])
    share_b = np.array([(vb == v).mean() for v in values])
    to_a = share_a >= share_b
    return float(share_a[to_a].sum()), float(share_b[~to_a].sum())
