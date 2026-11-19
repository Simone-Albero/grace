"""Stage quality_report: the defects of every cell, then of the regions the cells are merged into."""

from __future__ import annotations

__all__ = ["run"]

import json
import sqlite3
from collections import defaultdict

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from src import db
from src.config import DatasetConfig
from src.data import DUPLICATES, KINDS, LABEL
from src.graph import sample_rows
from src.provenance import (
    KEYS,
    breach_shares,
    concentration,
    scenario_breaches,
)
from src.quality import (
    DEFECTS,
    branch_spread,
    cell_samples,
    cell_variance,
    column_values,
    dataset_variance,
    invalid_values,
    label_conflicts,
    pair_conflicts,
    pair_confusion,
    rival_pairs,
    rule_errors,
    single_column,
)
from src.report import COUNTS, judge
from stages.paths import (
    assignments_file,
    cell_tree_file,
    cells_file,
    need,
    prepared_columns,
    read_classes,
    read_data,
    read_points,
    read_provenance,
    read_unparsable,
    read_weights,
)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Find the defects of every cell, and merge the cells into regions.

    Reads the data file (the features as read, DUPLICATES), the cells
    stage clean could not read as numbers (paths.read_unparsable), the
    points (paths.read_points, as stage cells), cells.parquet,
    assignments.parquet and cell_tree.parquet, which must be of the same
    rows ("run cells again" otherwise). With q = method.quality. Order:
    quality.cell_samples (q.rival_rows, seed) -> quality.rival_pairs on
    those rows: the rival cells of every cell -> per cell and class of
    its rival cells (a rival class), the cell's rows against the rows
    of those rival cells pooled, the cell whole and the rival side
    sampled to q.max_rows_per_cell (graph.sample_rows, seed):
    quality.pair_confusion (seed (seed, cell, class)) -> per original
    column (quality.column_values of its point columns),
    quality.rule_errors of every cell and rival class ->
    quality.cell_variance of every cell and the median of its class's,
    quality.dataset_variance and quality.branch_spread of every node of
    the cells' tree -> quality.invalid_values ->
    quality.label_conflicts on every feature -> quality.pair_conflicts;
    with cfg.provenance, on provenance.parquet (paths.read_provenance:
    every row of the CSV, with the row of the data file it is a copy
    of; "run clean again" when they disagree), over the CSV rows of each
    cell, provenance.concentration of each of its keys of
    provenance.KEYS, and with cfg.scenario provenance.scenario_breaches
    and provenance.breach_shares -> src.report.judge at q.tau and
    q.confidence (the one place they are applied, so the dashboard can
    apply any other tau): the verdict of every defect of every cell, the
    cells merged into regions where they do not contradict each other,
    the shares of the CSV rows with each defect and with none, per class
    and over the dataset.

    Logs: quality_rivals (one row per cell and rival class: cell,
    region, class, rival_class, rival_cells and rival_regions (JSON
    lists: its rival cells of that class and their regions),
    n_measured and n_rival_measured (the distinct rows of each side
    measured), then of the cell's side: n_scored (its rows in the
    measured half, what the confusion is a share of), nn_error,
    single_column_acc, single_column, confusion (n_scored, nn_error and
    confusion null when a side has fewer than 4 rows); n_rival_rows (the
    distinct rows of the rival class) and rules (JSON list of [column,
    error on the cell's rows, error on the rival class's rows] of the
    rule of the two classes on each column whose errors are both under
    one half, the lowest worse error first)), quality_cells (cell,
    region, class, n_rows, n_read, n_duplicates = n_read - n_rows,
    dup_share = 1 - n_rows / n_read, conflict_rows, conflict_read,
    conflict_with (the classes its conflicting rows are shared with,
    joined by ","), missing_rows and missing_read (rows with a null or
    invalid value), then per kind of data.KINDS <kind>_rows, clear_rows
    (its distinct rows with no null or invalid value and in no label
    conflict, which the share of rows with no defect needs); n_rivals
    (its rival cells); rival_class: the rival class its rows are most
    confused with (ties to the lower code), its confusion with
    confusion_low and confusion_high (its interval), n_scored,
    nn_error, single_column_acc and single_column; label_with,
    overlap_with and shortcut_with (the classes it has suspect_label,
    class_overlap and shortcut against, joined by ","), scenario_with
    (the keys of its off_scenario), variance_ratio (variance over
    variance_typical), repetition_branch (the largest branch of its
    class above it that is a repetition, null for none),
    repetition_share (its share of the class's CSV rows) and
    repetition_ratio (its variance over variance_dataset); shortcut_class (the rival class of its shortcut
    whose best rule errs least), shortcut_columns (the columns whose
    rule is a shortcut against it, the best first) and shortcut_error
    (the worse error of the best); defects and undecided (the defects
    it neither has nor lacks against some target) joined by ","; null
    for an empty list; variance, variance_sd (of its rows' squared
    distances to its centre), variance_typical (the median variance of
    the cells of its class) and variance_dataset (of every distinct row
    of the dataset, the same in every cell); with cfg.provenance,
    per configured key of provenance.KEYS <key>_top, <key>_top_share,
    <key>_n_values and <key>_eff_n; with cfg.scenario, per key of
    provenance.SCENARIO_KEYS it rules on, scenario_<key> (the share of
    its judged CSV rows that break it, null when none is judged) and
    scenario_<key>_n (its judged CSV rows); what depends on tau, the
    region, the class and key lists, shortcut_*, defects and undecided,
    is judged by src.report.judge, the rest measured), quality_regions
    (region, class, n_cells, then the counts of quality_cells summed
    over its cells, conflict_with, label_with, overlap_with,
    shortcut_with and scenario_with: the union of its cells', confusion
    with confusion_low, confusion_high and rival_class: the highest of
    its cells', variance_ratio: the lowest of its cells',
    repetition_share and repetition_ratio: of its cell's widest branch
    of repetition, and defects), quality_branches (node, class, parent,
    cell as in cell_tree.parquet, then n_rows and n_read below the node,
    variance and variance_sd of its rows), quality_classes (class, n_rows,
    n_read, n_cells, n_regions, share_<defect> and share_no_defect: the
    share of its CSV rows with that defect, a row defect of the row
    itself, a defect of its cell for the others, or with none;
    share_off_scenario null when the dataset has no scenario),
    cell_missing (cell, region, column, kind, n_rows, n_read: the null
    and invalid cells; no row without one), label_conflict_groups
    (group, row, class, cell, region, n_read), quality_stats (one row:
    tau, confidence, max_rows_per_cell, rival_rows, n_pairs (the pairs
    of rival cells), n_rivals (the rows of quality_rivals), n_reduced
    (of them, those measured on a sample of the cell's rows), n_groups,
    n_cells, n_regions, regions_<defect>, regions_no_defect,
    undecided_<defect> (the cells undecided on it), share_<defect> and
    share_no_defect of all the CSV rows; those of off_scenario null when
    the dataset has no scenario). n_rows counts distinct rows, n_read
    rows of the CSV.
    """
    q = cfg.method.quality
    classes = read_classes(cfg)
    cells = pl.read_parquet(need(cells_file(cfg), "cells")).sort("cell")
    assignments = pl.read_parquet(need(assignments_file(cfg), "cells"))
    tree = pl.read_parquet(need(cell_tree_file(cfg), "cells")).sort("node")
    cell = assignments.sort("row")["cell"].to_numpy()
    cell_class = cells["class"].to_numpy()
    weights = read_weights(cfg)
    points, labels = read_points(cfg)
    if len(cell) != len(labels) or not np.array_equal(cell_class[cell], labels):
        raise ValueError("the cells were cut on other data: run cells again")
    names = prepared_columns(cfg)
    columns = list(dict.fromkeys(names))
    column_of = np.array([columns.index(name) for name in names], dtype=np.int64)
    numeric = np.isin(columns, cfg.num_cols)
    rows_of = cell_samples(cell, len(cell), cfg.seed)  # every row of every cell
    searched = cell_samples(cell, q.rival_rows, cfg.seed)
    rows = np.concatenate([searched[c] for c in sorted(searched)])
    pairs = rival_pairs(points[rows], cell[rows], cell_class)
    rivals = _rival_classes(pairs, cell_class)
    sides = [
        _sides(rows_of, own, others, q.max_rows_per_cell, cfg.seed)
        for own, _, others in rivals
    ]
    reduced = sum(
        len(a) < len(rows_of[own]) for (own, _, _), (a, _) in zip(rivals, sides)
    )
    print(
        f"[quality_report] measuring {len(rivals)} cells against a rival class "
        f"({len(pairs)} pairs of rival cells; {reduced} on a sample of the cell)",
        flush=True,
    )
    variance, variance_sd = cell_variance(points, cell, len(cell_class))
    median = {k: np.median(variance[cell_class == k]) for k in set(cell_class)}
    typical = np.array([median[k] for k in cell_class], dtype=np.float64)
    n_read = np.bincount(cell, weights=weights, minlength=len(cell_class))
    of_dataset = dataset_variance(points)
    branches = _branches(tree, points, cell, n_read)
    measures = Parallel(n_jobs=-1, batch_size=8)(
        delayed(_measure)(points[a], points[b], column_of, numeric, (cfg.seed, own, k))
        for (own, k, _), (a, b) in zip(rivals, sides)
    )
    pairs_of = [(own, k) for own, k, _ in rivals]
    errors = np.stack(
        [
            rule_errors(
                column_values(points[:, column_of == c], numeric[c]),
                labels,
                cell,
                pairs_of,
                cell_class,
                bool(numeric[c]),
            )
            for c in range(len(columns))
        ],
        axis=1,
    )  # (pairs, columns, 2)
    print(f"[quality_report] rules of {len(columns)} columns measured", flush=True)
    del points
    n_class = np.bincount(labels, minlength=len(classes))

    data = read_data(cfg)
    features = data.drop(DUPLICATES, LABEL)
    invalid = invalid_values(features, read_unparsable(cfg))
    groups = label_conflicts(features, labels)
    conflicts = pair_conflicts(groups, labels, cell, weights)
    counts = _counts(cell, len(cell_class), weights, groups, invalid)
    conflict_with = defaultdict(set)
    for a, b in conflicts:
        conflict_with[a].add(int(cell_class[b]))
        conflict_with[b].add(int(cell_class[a]))

    origins = (
        _provenance(cfg, cell, cell_class, labels, weights, classes)
        if cfg.provenance is not None
        else [{} for _ in range(len(cell_class))]
    )
    flawed = groups >= 0
    flawed[invalid["row"].to_numpy()] = True
    clear = np.bincount(cell[~flawed], minlength=len(cell_class))
    rival_rows = [
        _rival_row(
            rival,
            (len(a), len(b)),
            measure,
            rules,
            n_class,
            cell_class,
            classes,
            columns,
        )
        for rival, (a, b), measure, rules in zip(rivals, sides, measures, errors)
    ]
    spread = {
        "variance": variance,
        "variance_sd": variance_sd,
        "typical": typical,
        "dataset": np.full(len(cell_class), of_dataset),
    }
    cell_rows = [
        _cell_row(c, cell_class, classes, counts, conflict_with, clear, spread, origin)
        for c, origin in enumerate(origins)
    ]
    judged = judge(cell_rows, rival_rows, branches, classes, q.tau, q.confidence)
    region = np.array([row["region"] for row in judged.cells], dtype=np.int64)
    found = {d: judged.stats[f"regions_{d}"] for d in DEFECTS}
    print(
        f"[quality_report] {len(rivals)} cells against a rival class, "
        f"{int(groups.max()) + 1} label conflicts; "
        f"{len(judged.cells)} cells merged into {len(judged.regions)} regions; "
        f"regions by defect: {found}",
        flush=True,
    )
    db.log_table(conn, run_id, "quality_rivals", judged.rivals)
    db.log_table(conn, run_id, "quality_branches", branches.to_dicts())
    db.log_table(conn, run_id, "quality_cells", judged.cells)
    db.log_table(conn, run_id, "quality_regions", judged.regions)
    db.log_table(conn, run_id, "quality_classes", judged.classes)
    db.log_table(
        conn, run_id, "cell_missing", _missing_rows(invalid, cell, region, weights)
    )
    db.log_table(
        conn,
        run_id,
        "label_conflict_groups",
        [
            {
                "group": int(groups[row]),
                "row": int(row),
                "class": classes[labels[row]],
                "cell": int(cell[row]),
                "region": int(region[cell[row]]),
                "n_read": float(weights[row]),
            }
            for row in np.flatnonzero(groups >= 0)
        ],
    )
    db.log_table(
        conn,
        run_id,
        "quality_stats",
        [
            {
                "tau": q.tau,
                "confidence": q.confidence,
                "max_rows_per_cell": q.max_rows_per_cell,
                "rival_rows": q.rival_rows,
                "n_pairs": len(pairs),
                "n_rivals": len(rivals),
                "n_reduced": reduced,
                "n_groups": int(groups.max()) + 1,
                "n_cells": len(judged.cells),
                **{k: v for k, v in judged.stats.items() if k != "tau"},
            }
        ],
    )


def _branches(tree: pl.DataFrame, points, cell, n_read) -> pl.DataFrame:
    """Return the quality_branches rows: the cells' tree with the rows and the spread of every node (quality.branch_spread)."""
    at = {v: i for i, v in enumerate(tree["node"].to_list())}
    parent = np.array([at.get(v, -1) for v in tree["parent"].to_list()])
    node_cell = tree["cell"].fill_null(-1).to_numpy()
    variance, sd, n_rows = branch_spread(points, cell, parent, node_cell)
    read = np.where(node_cell >= 0, n_read[np.maximum(node_cell, 0)], 0.0)
    for i in range(len(parent) - 1, -1, -1):  # children after their parent
        if parent[i] >= 0:
            read[parent[i]] += read[i]
    return tree.select("node", "class", "parent", "cell").with_columns(
        n_rows=pl.Series(n_rows.astype(np.int64)),
        n_read=pl.Series(read),
        variance=pl.Series(variance),
        variance_sd=pl.Series(sd),
    )


def _rival_classes(
    pairs: list[tuple[int, int]], cell_class: np.ndarray
) -> list[tuple[int, int, list[int]]]:
    """Return, per cell and class of its rival cells, those rival cells: (cell, class, cells), in that order."""
    by: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for a, b in pairs:
        by[a][int(cell_class[b])].append(b)
        by[b][int(cell_class[a])].append(a)
    return [(c, k, sorted(by[c][k])) for c in sorted(by) for k in sorted(by[c])]


def _sides(rows_of, own, others, cap, seed) -> tuple[np.ndarray, np.ndarray]:
    """The rows a cell and its rival cells of one class are measured on: the cell whole, the rivals pooled and capped.

    The cell is never sampled: it holds under 2 · min_rows rows, and
    half of them are scored, which min_rows makes enough to decide a
    defect. The rival side is sampled to cap (graph.sample_rows), which
    in many dimensions barely moves the nearest rival row.
    """
    mine = rows_of[own]
    theirs = np.concatenate([rows_of[c] for c in others])
    return mine, theirs[sample_rows(len(theirs), cap, seed)]


def _measure(Xa, Xb, column_of, numeric, seed):
    """The confusion of a cell and a rival class, and its single column on every row when a side is too small."""
    confusion = pair_confusion(Xa, Xb, column_of, numeric, seed)
    if confusion is None:
        return single_column(Xa, Xb, column_of, numeric), None
    return confusion.single, confusion


def _counts(cell, m, weights, groups, invalid) -> dict[str, np.ndarray]:
    """Return per cell its distinct and CSV rows, conflicting rows and null or invalid rows."""
    conflicting = groups >= 0
    hit = np.unique(invalid["row"].to_numpy())
    counts = {
        "n_rows": np.bincount(cell, minlength=m),
        "n_read": np.bincount(cell, weights=weights, minlength=m),
        "conflict_rows": np.bincount(cell[conflicting], minlength=m),
        "conflict_read": np.bincount(
            cell[conflicting], weights=weights[conflicting], minlength=m
        ),
        "missing_rows": np.bincount(cell[hit], minlength=m),
        "missing_read": np.bincount(cell[hit], weights=weights[hit], minlength=m),
    }
    for kind in KINDS:
        rows = np.unique(invalid.filter(pl.col("kind") == kind)["row"].to_numpy())
        counts[f"{kind}_rows"] = np.bincount(cell[rows], minlength=m)
    return counts


def _rival_row(
    rival, sizes, measure, rules, n_class, cell_class, classes, columns
) -> dict:
    """Return the quality_rivals row of a cell and a rival class, without the regions (src.report.judge)."""
    (own, k, others), (n_a, n_b) = rival, sizes
    one, confusion = measure
    worse = rules.max(axis=1)
    kept = sorted((float(worse[c]), columns[c], c) for c in np.flatnonzero(worse < 0.5))
    row = {
        "cell": own,
        "class": classes[cell_class[own]],
        "rival_class": classes[k],
        "rival_cells": json.dumps(others),
        "n_measured": n_a,
        "n_rival_measured": n_b,
        **dict.fromkeys(("n_scored", "nn_error")),
        "single_column_acc": one.side_a,
        "single_column": columns[one.column] if one.column >= 0 else None,
        "confusion": None,
        "n_rival_rows": int(n_class[k]),
        "rules": json.dumps(
            [[name, float(rules[c, 0]), float(rules[c, 1])] for _, name, c in kept]
        ),
    }
    if confusion is not None:
        row.update(
            n_scored=int(confusion.measure[:n_a].sum()),
            nn_error=float(confusion.nn_error[0]),
            confusion=float(confusion.confusion[0]),
        )
    return row


def _cell_row(
    c, cell_class, classes, counts, conflict_with, clear, spread, origin
) -> dict:
    """Return the quality_cells row of a cell without what tau decides (src.report.judge)."""
    own = {key: counts[key][c] for key in COUNTS}
    return {
        "cell": c,
        "class": classes[cell_class[c]],
        "n_rows": int(own["n_rows"]),
        "n_read": float(own["n_read"]),
        "n_duplicates": float(own["n_read"] - own["n_rows"]),
        "dup_share": float(1 - own["n_rows"] / own["n_read"]),
        "conflict_rows": int(own["conflict_rows"]),
        "conflict_read": float(own["conflict_read"]),
        "conflict_with": _names(conflict_with[c], classes),
        "missing_rows": int(own["missing_rows"]),
        "missing_read": float(own["missing_read"]),
        **{f"{kind}_rows": int(own[f"{kind}_rows"]) for kind in KINDS},
        "clear_rows": int(clear[c]),
        "variance": float(spread["variance"][c]),
        "variance_sd": float(spread["variance_sd"][c]),
        "variance_typical": float(spread["typical"][c]),
        "variance_dataset": float(spread["dataset"][c]),
        **origin,
    }


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes in code order joined by ",", None for none."""
    return ",".join(classes[c] for c in sorted(codes)) or None


def _provenance(
    cfg: DatasetConfig,
    cell: np.ndarray,
    cell_class: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    classes: list[str],
) -> list[dict]:
    """Return, per cell, the provenance columns of its quality_cells row.

    Args:
        cfg: The dataset config, with a provenance.
        cell: The cell of every row of the data file.
        cell_class: The class of every cell.
        labels: The class code of every row of the data file.
        weights: The copies of every row of the data file in the CSV.
        classes: The class names, by code.

    Returns:
        One dict per cell, in cell order: per key of provenance.KEYS in
        the file, <key>_top, <key>_top_share, <key>_n_values and
        <key>_eff_n (provenance.concentration); with cfg.scenario, per
        key it rules on, scenario_<key> and scenario_<key>_n
        (provenance.breach_shares, the share null for a cell with no
        judged row).

    Raises:
        ValueError: When provenance.parquet is not of the rows of the
            data file ("run clean again").
    """
    origin = read_provenance(cfg)
    row = origin["row"].to_numpy()
    if not np.array_equal(np.bincount(row, minlength=len(cell)), weights):
        raise ValueError("the provenance is of other rows: run clean again")
    group, m = cell[row], len(cell_class)
    parts = []
    for key in KEYS:
        if key not in origin.columns:
            continue
        rename = lambda c, k=key: f"{k}_{c}"  # noqa: E731
        parts.append(concentration(group, origin[key], m).rename(rename))
    if cfg.scenario is not None:
        breaches = scenario_breaches(origin, labels[row], classes, cfg.scenario)
        for key, breach in breaches.items():
            share, n = breach_shares(group, breach, m)
            parts.append(
                pl.DataFrame(
                    {
                        f"scenario_{key}": pl.Series(share).fill_nan(None),
                        f"scenario_{key}_n": n.astype(np.int64),
                    }
                )
            )
    print(f"[quality_report] provenance of {len(row)} CSV rows measured", flush=True)
    if not parts:
        return [{} for _ in range(m)]
    return pl.concat(parts, how="horizontal").to_dicts()


def _missing_rows(
    invalid: pl.DataFrame, cell: np.ndarray, region: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Return the cell_missing rows: the null and invalid cells per cell, column and kind."""
    rows = invalid["row"].to_numpy()
    found = invalid.with_columns(
        cell=pl.Series(cell[rows]),
        region=pl.Series(region[cell[rows]]),
        weight=pl.Series(weights[rows]),
    )
    return (
        found.group_by("cell", "region", "column", "kind")
        .agg(n_rows=pl.len(), n_read=pl.col("weight").sum())
        .sort("cell", "column", "kind")
        .to_dicts()
    )
