"""Data-quality measures on regions, on pairs of regions of different classes and on classes.

After Flood, Engelen, Aspinall & Desmet, "Bad Design Smells in
Benchmark NIDS Datasets" (EuroS&P 2024), measured on the regions of
`src.clustering`:
- null and invalid values, cell by cell (invalid_values);
- label conflicts: rows equal on every feature under different
  classes (label_conflicts, pair_conflicts);
- how much two regions mix: the leave-one-out 1-NN error of each side
  (nearest_error), bounded below by Cover & Hart (cover_hart_low);
- single-feature separation: the per-side accuracy of the best rule on
  one column, a threshold or a value-to-side map (one_feature, OneR of
  Holte 1993; it is Ho & Basu's F3 made robust, 1 - F3 of Lorena et
  al.);
- on how few columns the separation rests (feature_dependence) and the
  pair's verdict (verdict): conflict, overlap, shortcut or robust;
- poor data diversity and traffic collapse: how alike the rows of a
  class are (diversity).
Which pairs are measured is candidate_pairs' choice; region_issues
turns a region's counts into its problems. Every measure but the
counts of copies is computed on distinct rows, unweighted. Every pair
measure is per side (a, b), its summary the mean of the two: a large
region does not hide a small one.
"""

from __future__ import annotations

__all__ = [
    "ISSUES",
    "VERDICTS",
    "Dependence",
    "OneFeature",
    "Pair",
    "candidate_pairs",
    "cover_hart_low",
    "diversity",
    "feature_dependence",
    "invalid_values",
    "label_conflicts",
    "nearest_error",
    "one_feature",
    "pair_conflicts",
    "region_issues",
    "region_samples",
    "verdict",
]

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.graph import sample_rows

REASONS = ("near", "conflict", "nearest")
VERDICTS = ("conflict", "overlap", "shortcut", "robust")
ISSUES = ("missing", "duplicates", "conflict", "overlap", "shortcut")


@dataclass(frozen=True)
class Pair:
    """Two regions of different classes worth measuring, a < b.

    Attributes:
        a: The first region.
        b: The second region.
        reasons: Why the pair is measured, a subset of REASONS in that
            order: "near" (separation under the threshold), "conflict" (they share rows equal on the features),
            "nearest" (one is the nearest rival of the other).
    """

    a: int
    b: int
    reasons: tuple[str, ...]


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
    separation: np.ndarray,
    conflicts: Iterable[tuple[int, int]],
    classes: np.ndarray,
    threshold: float,
) -> list[Pair]:
    """Choose the pairs of regions of different classes worth measuring.

    A pair is "near" when its separation is under threshold, "conflict"
    when it is among conflicts, "nearest" when one region is the nearest
    rival of the other (least separation; ties to the lower region id).

    Args:
        separation: The (m, m) separations of the regions
            (clustering.separation).
        conflicts: The pairs (a, b) that share conflicting rows.
        classes: The class of every region, shape (m,).
        threshold: The separation under which a pair is near.

    Returns:
        The pairs sorted by (a, b), each with its reasons.
    """
    separation = np.asarray(separation)
    classes = np.asarray(classes)
    rival = classes[:, None] != classes[None, :]
    reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    for a, b in np.argwhere(np.triu(rival & (separation < threshold), 1)):
        reasons[(int(a), int(b))].add("near")
    for a, b in conflicts:
        if rival[a, b]:
            reasons[(min(a, b), max(a, b))].add("conflict")
    for own in range(len(classes)):
        others = np.flatnonzero(rival[own])
        if len(others):
            other = int(others[np.argmin(separation[own, others])])
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


@dataclass(frozen=True)
class OneFeature:
    """The best rule on a single column between two regions.

    Attributes:
        side_a: The share of the first region's rows the rule puts on
            their side.
        side_b: The same for the second region.
        column: The index (into the column names) of the rule's
            column; -1 when no column has two values.
        n_separating: The columns whose rule reaches the threshold, as
            the mean of the two sides.
    """

    side_a: float
    side_b: float
    column: int
    n_separating: int

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
        side: True for a row of the second region, shape (n,); each
            side holds at least one row.

    Returns:
        The error of the first region's rows and of the second's, each
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


def one_feature(
    Xa: np.ndarray,
    Xb: np.ndarray,
    column_of: np.ndarray,
    numeric: np.ndarray,
    threshold: float,
) -> OneFeature:
    """Find the best rule on a single column between two regions (OneR).

    A numeric column (one point column) gets the threshold, in either
    direction, of highest balanced accuracy; ties of value stay on one
    side. A categorical column (its one-hot slots, read back as the
    active slot or none) sends each value to the side where it is the
    larger share of the rows. The column of highest balanced accuracy
    wins (ties to the lower index).

    Args:
        Xa: The rows of the first region, shape (n_a, d), n_a >= 1.
        Xb: The rows of the second region, shape (n_b, d), n_b >= 1.
        column_of: The original column of each point column, as an
            index into numeric, shape (d,).
        numeric: Whether each original column is numeric, shape (c,).
        threshold: The balanced accuracy from which a column counts in
            n_separating.

    Returns:
        The rule's accuracy on each side, its column, n_separating.
    """
    best, sides, n_separating = -1.0, (0.5, 0.5), 0
    column = -1
    for c in range(len(numeric)):
        slots = column_of == c
        if not slots.any():
            continue
        va, vb = _values(Xa[:, slots], numeric[c]), _values(Xb[:, slots], numeric[c])
        if len(np.unique(np.r_[va, vb])) < 2:
            continue
        rule = _stump(va, vb) if numeric[c] else _majority(va, vb)
        share = (rule[0] + rule[1]) / 2
        n_separating += share >= threshold
        if share > best:
            best, sides, column = share, rule, c
    return OneFeature(float(sides[0]), float(sides[1]), column, int(n_separating))


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
        Xa: The rows of the first region, shape (n_a, d).
        Xb: The rows of the second region, shape (n_b, d).
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


def verdict(
    errors: np.ndarray, n_conflicts: int, one_feature: float, threshold: float
) -> tuple[str, int]:
    """Judge a pair from its Dependence errors.

    With err the mean of the two sides (row 0) and tau = threshold:
    "conflict" when err >= tau and the pair shares conflicting rows;
    "overlap" when err >= tau and one_feature < 1 - tau (a single
    column often separates what the 1-NN, swamped by the others,
    mixes); "shortcut" when, for the fewest first columns of the group,
    the error rises by at least tau and reaches tau; "robust" otherwise.

    Args:
        errors: Dependence.errors.
        n_conflicts: The conflicting rows the two regions share.
        one_feature: OneFeature.share of the pair.
        threshold: tau.

    Returns:
        One of VERDICTS and how many columns of the group the verdict
        rests on (0 but for "shortcut").
    """
    err = errors.mean(axis=1)
    if err[0] >= threshold:
        if n_conflicts > 0:
            return "conflict", 0
        if one_feature < 1 - threshold:
            return "overlap", 0
    for k in range(1, len(err)):
        if err[k] - err[0] >= threshold and err[k] >= threshold:
            return "shortcut", k
    return "robust", 0


def region_issues(
    missing_rows: int,
    dup_share: float,
    conflict_rows: int,
    verdicts: Iterable[str],
    dup_threshold: float,
) -> tuple[str, ...]:
    """Return the problems of a region, in ISSUES order.

    "missing": a null or invalid value in any row; "duplicates":
    dup_share >= dup_threshold; "conflict": a row equal to a row of
    another class; "overlap" and "shortcut": a pair of that verdict.
    None of them: the region is solid.

    Args:
        missing_rows: The rows with a null or invalid value.
        dup_share: 1 - n_rows / n_read.
        conflict_rows: The rows in a label conflict.
        verdicts: The verdicts of the region's pairs.
        dup_threshold: quality.thresholds.dup_share.

    Returns:
        The problems, empty when the region is solid.
    """
    verdicts = set(verdicts)
    found = {
        "missing": missing_rows > 0,
        "duplicates": dup_share >= dup_threshold,
        "conflict": conflict_rows > 0,
        "overlap": "overlap" in verdicts,
        "shortcut": "shortcut" in verdicts,
    }
    return tuple(issue for issue in ISSUES if found[issue])


_TIE = 1e-9  # relative tolerance under which two distances tie


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
