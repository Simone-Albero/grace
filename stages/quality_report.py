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
from src.cells import CellTree, merge_cells
from src.config import DatasetConfig
from src.data import DUPLICATES, KINDS, LABEL
from src.quality import (
    DEFECTS,
    RIVAL_DEFECTS,
    cell_samples,
    cover_hart_low,
    defects_of,
    feature_dependence,
    invalid_values,
    label_conflicts,
    pair_conflicts,
    pair_confusion,
    rival_defects,
    rival_pairs,
    single_column,
)
from stages.paths import (
    assignments_file,
    cell_tree_file,
    cells_file,
    need,
    prepared_columns,
    read_classes,
    read_data,
    read_points,
    read_unparsable,
    read_weights,
)

COUNTS = ("n_rows", "n_read", "conflict_rows", "conflict_read", "missing_rows")
COUNTS += ("missing_read",) + tuple(f"{kind}_rows" for kind in KINDS)


def run(cfg: DatasetConfig, conn: sqlite3.Connection, run_id: int) -> None:
    """Find the defects of every cell, and merge the cells into regions.

    Reads the data file (the features as read, DUPLICATES), the cells
    stage clean could not read as numbers (paths.read_unparsable), the
    points (paths.read_points, as stage cells), cells.parquet,
    assignments.parquet and cell_tree.parquet, which must be of the same
    rows ("run cells again" otherwise). With q = method.quality. Order:
    quality.cell_samples (q.max_rows_per_cell, seed): the rows each cell
    is measured on -> quality.cell_samples (q.rival_rows, seed) ->
    quality.rival_pairs on those rows -> quality.invalid_values ->
    quality.label_conflicts on every feature -> quality.pair_conflicts
    -> per pair, on the measured rows: quality.single_column,
    quality.feature_dependence (q.max_group, seed (seed, a, b)) and
    quality.pair_confusion -> per cell: quality.rival_defects (q.tau) on
    its side of each pair, quality.defects_of on its rows -> per class:
    cells.merge_cells of its cut, a cell's profile its classes of
    class_overlap and of shortcut; region id = region + offset, by class
    -> per region: quality.defects_of on its rows and the classes its
    cells share -> per class and over the dataset: the share of the CSV
    rows with each defect and with none (_shares).

    Logs: quality_pairs (cell_a, class_a, region_a, cell_b, class_b,
    region_b, n_conflicts, n_conflicts_read; nn_error_a, nn_error_b,
    nn_error (their mean), bayes_error_low (quality.cover_hart_low of
    nn_error), single_column_acc_a, single_column_acc_b,
    single_column_acc (their mean), single_column, key_columns (the
    columns feature_dependence removed, in order, joined by ","),
    confusion_a and confusion_b (pair_confusion with every column),
    confusion_without_a and confusion_without_b (JSON lists: after
    removing the first 1, 2, ... key columns); the 1-NN, key column and
    confusion columns are null when a side has fewer than 4 measured
    rows), quality_cells (cell, region, class, n_rows, n_read,
    n_duplicates = n_read - n_rows, dup_share = 1 - n_rows / n_read,
    conflict_rows, conflict_read, conflict_with (the classes its
    conflicting rows are shared with, joined by ","), missing_rows and
    missing_read (rows with a null or invalid value), then per kind of
    data.KINDS <kind>_rows; n_rivals; rival: the rival cell its rows are
    most confused with (ties to the lower id), rival_class, its side's
    confusion, nn_error and single_column_acc, and single_column;
    overlap_with and shortcut_with (the classes of rival_defects,
    joined by ","); shortcut_rival: among the rivals under tau, the one
    whose removal of key columns raises its side's confusion the most,
    shortcut_columns (the first key columns that bring it to tau, all
    of them otherwise) and confusion_without (its confusion without
    them); defects joined by ","; null for an empty list),
    quality_regions (region, class, n_cells, then the counts of
    quality_cells summed over its cells, conflict_with, overlap_with
    and shortcut_with as its cells', confusion: the highest of its
    cells', with rival_class, and defects), quality_classes (class,
    n_rows, n_read, n_cells, n_regions, share_<defect> and
    share_no_defect: the share of its CSV rows with that defect, a row
    defect of the row itself, a rival defect of its cell, or with none),
    cell_missing (cell, region, column, kind, n_rows, n_read: the null
    and invalid cells; no row without one), label_conflict_groups
    (group, row, class, cell, region, n_read), quality_stats (one row:
    tau, max_rows_per_cell, rival_rows, max_group, n_pairs, n_groups,
    n_cells, n_regions, regions_<defect>, regions_no_defect,
    share_<defect> and share_no_defect of all the CSV rows). n_rows counts
    distinct rows, n_read rows of the CSV.
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
    measured = cell_samples(cell, q.max_rows_per_cell, cfg.seed)
    searched = cell_samples(cell, q.rival_rows, cfg.seed)
    rows = np.concatenate([searched[c] for c in sorted(searched)])
    pairs = rival_pairs(points[rows], cell[rows], cell_class)
    print(f"[quality_report] measuring {len(pairs)} pairs of cells", flush=True)
    measures = Parallel(n_jobs=-1, batch_size=8)(
        delayed(_measure)(
            points[measured[a]],
            points[measured[b]],
            column_of,
            numeric,
            q.max_group,
            (cfg.seed, a, b),
        )
        for a, b in pairs
    )
    del points

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

    sides = defaultdict(
        list
    )  # per cell: (rival, its confusion, the pair index, its side)
    n_rivals = np.zeros(len(cell_class), dtype=np.int64)
    for i, ((a, b), (_, _, confusion)) in enumerate(zip(pairs, measures)):
        n_rivals[a] += 1
        n_rivals[b] += 1
        if confusion is not None:
            sides[a].append((b, confusion[:, 0], i, "a"))
            sides[b].append((a, confusion[:, 1], i, "b"))
    profile = [
        rival_defects([(cell_class[r], e) for r, e, _, _ in sides[c]], q.tau)
        for c in range(len(cell_class))
    ]
    region = _merge(tree, profile)
    flawed = groups >= 0
    flawed[invalid["row"].to_numpy()] = True
    rival = np.array([bool(overlap or shortcut) for overlap, shortcut in profile])
    clean = np.bincount(cell[~flawed & ~rival[cell]], minlength=len(cell_class))
    pair_rows = [
        _pair_row(a, b, measure, conflicts, cell_class, region, classes, columns)
        for (a, b), measure in zip(pairs, measures)
    ]
    cell_rows = [
        _cell_row(
            c,
            region,
            cell_class,
            classes,
            counts,
            conflict_with,
            n_rivals,
            sides,
            profile,
            pair_rows,
            q.tau,
        )
        for c in range(len(cell_class))
    ]
    region_rows = _region_rows(cell_rows, region, classes)
    class_rows = _class_rows(cell_rows, region_rows, clean, classes)
    found = {d: sum(d in _listed(r["defects"]) for r in region_rows) for d in DEFECTS}
    print(
        f"[quality_report] {len(pairs)} pairs, {int(groups.max()) + 1} label conflicts; "
        f"{len(cell_rows)} cells merged into {len(region_rows)} regions; "
        f"regions by defect: {found}",
        flush=True,
    )
    db.log_table(conn, run_id, "quality_pairs", pair_rows)
    db.log_table(conn, run_id, "quality_cells", cell_rows)
    db.log_table(conn, run_id, "quality_regions", region_rows)
    db.log_table(conn, run_id, "quality_classes", class_rows)
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
                "max_rows_per_cell": q.max_rows_per_cell,
                "rival_rows": q.rival_rows,
                "max_group": q.max_group,
                "n_pairs": len(pairs),
                "n_groups": int(groups.max()) + 1,
                "n_cells": len(cell_rows),
                "n_regions": len(region_rows),
                **{f"regions_{d}": found[d] for d in DEFECTS},
                "regions_no_defect": sum(not r["defects"] for r in region_rows),
                **_shares(cell_rows, clean),
            }
        ],
    )


def _measure(Xa, Xb, column_of, numeric, max_group, seed):
    """The single column, dependence and confusion of a pair; the last two None when a side is too small."""
    one = single_column(Xa, Xb, column_of, numeric)
    dependence = feature_dependence(Xa, Xb, column_of, max_group, seed)
    if dependence is None:
        return one, None, None
    return one, dependence, pair_confusion(Xa, Xb, column_of, numeric, dependence)


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


def _merge(tree: pl.DataFrame, profile: list) -> np.ndarray:
    """Return the region of every cell: cells.merge_cells class by class, ids offset by class."""
    region = np.empty(len(profile), dtype=np.int64)
    offset = 0
    for _, part in tree.group_by("class", maintain_order=True):
        first = int(part["node"][0])
        own = part["cell"].drop_nulls().to_numpy()
        base = int(own.min())
        cut = CellTree(
            parent=np.where(part["parent"] >= 0, part["parent"] - first, -1),
            cell=np.where(part["cell"].is_null(), -1, part["cell"].fill_null(0) - base),
        )
        merged = merge_cells(cut, profile[base : base + len(own)])
        region[base : base + len(own)] = merged + offset
        offset += int(merged.max()) + 1
    return region


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes in code order joined by ",", None for none."""
    return ",".join(classes[c] for c in sorted(codes)) or None


def _listed(value: str | None) -> list[str]:
    return value.split(",") if value else []


def _pair_row(a, b, measure, conflicts, cell_class, region, classes, columns) -> dict:
    """Return the quality_pairs row of a pair of cells."""
    one, dependence, confusion = measure
    n, read = conflicts.get((a, b), (0, 0.0))
    row = {
        "cell_a": a,
        "class_a": classes[cell_class[a]],
        "region_a": int(region[a]),
        "cell_b": b,
        "class_b": classes[cell_class[b]],
        "region_b": int(region[b]),
        "n_conflicts": n,
        "n_conflicts_read": read,
        **dict.fromkeys(("nn_error_a", "nn_error_b", "nn_error", "bayes_error_low")),
        "single_column_acc_a": one.side_a,
        "single_column_acc_b": one.side_b,
        "single_column_acc": one.share,
        "single_column": columns[one.column] if one.column >= 0 else None,
        **dict.fromkeys(("key_columns", "confusion_a", "confusion_b")),
        **dict.fromkeys(("confusion_without_a", "confusion_without_b")),
    }
    if dependence is not None:
        errors = dependence.errors
        nn_error = float(errors[0].mean())
        row.update(
            nn_error_a=float(errors[0, 0]),
            nn_error_b=float(errors[0, 1]),
            nn_error=nn_error,
            bayes_error_low=cover_hart_low(nn_error),
            key_columns=",".join(columns[c] for c in dependence.group),
            confusion_a=float(confusion[0, 0]),
            confusion_b=float(confusion[0, 1]),
            confusion_without_a=json.dumps(confusion[1:, 0].tolist()),
            confusion_without_b=json.dumps(confusion[1:, 1].tolist()),
        )
    return row


def _cell_row(
    c,
    region,
    cell_class,
    classes,
    counts,
    conflict_with,
    n_rivals,
    sides,
    profile,
    pair_rows,
    tau,
) -> dict:
    """Return the quality_cells row of a cell."""
    own = {key: counts[key][c] for key in COUNTS}
    overlap, shortcut = profile[c]
    mine = sides[c]
    rival = max(mine, key=lambda s: (s[1][0], -s[0]), default=None)
    under = [s for s in mine if s[1][0] < tau]
    cut = max(under, key=lambda s: (s[1][1:].max(), -s[0]), default=None)
    row = {
        "cell": c,
        "region": int(region[c]),
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
        "n_rivals": int(n_rivals[c]),
        **dict.fromkeys(("rival", "rival_class", "confusion", "nn_error")),
        **dict.fromkeys(("single_column_acc", "single_column")),
        "overlap_with": _names(overlap, classes),
        "shortcut_with": _names(shortcut, classes),
        **dict.fromkeys(("shortcut_rival", "shortcut_columns", "confusion_without")),
    }
    if rival is not None:
        other, steps, i, side = rival
        pair = pair_rows[i]
        row.update(
            rival=int(other),
            rival_class=classes[cell_class[other]],
            confusion=float(steps[0]),
            nn_error=pair[f"nn_error_{side}"],
            single_column_acc=pair[f"single_column_acc_{side}"],
            single_column=pair["single_column"],
        )
    if cut is not None:
        other, steps, i, _ = cut
        reached = np.flatnonzero(steps[1:] >= tau)
        k = int(reached[0]) + 1 if len(reached) else len(steps) - 1
        row.update(
            shortcut_rival=int(other),
            shortcut_columns=",".join(pair_rows[i]["key_columns"].split(",")[:k]),
            confusion_without=float(steps[k]),
        )
    row["defects"] = (
        ",".join(
            defects_of(
                own["missing_rows"],
                own["n_rows"],
                own["n_read"],
                own["conflict_rows"],
                overlap,
                shortcut,
            )
        )
        or None
    )
    return row


def _region_rows(
    cell_rows: list[dict], region: np.ndarray, classes: list[str]
) -> list[dict]:
    """Return the quality_regions rows: the counts of their cells summed, the profile they share."""
    rows = []
    for r in range(int(region.max()) + 1):
        own = [cell_rows[c] for c in np.flatnonzero(region == r)]
        first = own[0]
        summed = {key: sum(row[key] for row in own) for key in COUNTS}
        worst = max(
            (row for row in own if row["confusion"] is not None),
            key=lambda row: row["confusion"],
            default=None,
        )
        conflict = {
            classes.index(n) for row in own for n in _listed(row["conflict_with"])
        }
        overlap = [classes.index(n) for n in _listed(first["overlap_with"])]
        shortcut = [classes.index(n) for n in _listed(first["shortcut_with"])]
        defects = defects_of(
            summed["missing_rows"],
            summed["n_rows"],
            summed["n_read"],
            summed["conflict_rows"],
            overlap,
            shortcut,
        )
        rows.append(
            {
                "region": r,
                "class": first["class"],
                "n_cells": len(own),
                "n_rows": int(summed["n_rows"]),
                "n_read": float(summed["n_read"]),
                "n_duplicates": float(summed["n_read"] - summed["n_rows"]),
                "dup_share": float(1 - summed["n_rows"] / summed["n_read"]),
                "conflict_rows": int(summed["conflict_rows"]),
                "conflict_read": float(summed["conflict_read"]),
                "conflict_with": _names(conflict, classes),
                "missing_rows": int(summed["missing_rows"]),
                "missing_read": float(summed["missing_read"]),
                **{f"{kind}_rows": int(summed[f"{kind}_rows"]) for kind in KINDS},
                "overlap_with": first["overlap_with"],
                "shortcut_with": first["shortcut_with"],
                "confusion": None if worst is None else worst["confusion"],
                "rival_class": None if worst is None else worst["rival_class"],
                "defects": ",".join(defects) or None,
            }
        )
    return rows


def _class_rows(
    cell_rows: list[dict],
    region_rows: list[dict],
    clean: np.ndarray,
    classes: list[str],
) -> list[dict]:
    """Return the quality_classes rows: per class its counts and _shares."""
    rows = []
    for name in classes:
        own = [r for r in cell_rows if r["class"] == name]
        if not own:
            continue
        rows.append(
            {
                "class": name,
                "n_rows": sum(r["n_rows"] for r in own),
                "n_read": sum(r["n_read"] for r in own),
                "n_cells": len(own),
                "n_regions": sum(r["class"] == name for r in region_rows),
                **_shares(own, clean),
            }
        )
    return rows


def _shares(cell_rows: list[dict], clean: np.ndarray) -> dict[str, float]:
    """Return share_<defect> and share_no_defect over the CSV rows of these cells.

    A row of the CSV has invalid_values when it holds a null or invalid
    value, duplicates when it repeats a row read before it (the first
    copy of a row has not), label_conflict when it is identical to a
    row of another class, class_overlap and shortcut when its cell has
    them; it has no defect when it has none of the five.

    Args:
        cell_rows: quality_cells rows.
        clean: Per cell, its distinct rows without a null or invalid
            value and in no label conflict, 0 for a cell with
            class_overlap or shortcut: the first copies with no defect.
    """
    n_read = sum(r["n_read"] for r in cell_rows)
    found = {
        "invalid_values": sum(r["missing_read"] for r in cell_rows),
        "duplicates": sum(r["n_duplicates"] for r in cell_rows),
        "label_conflict": sum(r["conflict_read"] for r in cell_rows),
    }
    for d in RIVAL_DEFECTS:
        found[d] = sum(r["n_read"] for r in cell_rows if d in _listed(r["defects"]))
    return {
        **{f"share_{d}": found[d] / n_read for d in DEFECTS},
        "share_no_defect": sum(int(clean[r["cell"]]) for r in cell_rows) / n_read,
    }


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
