"""The verdicts of quality_report at a value of tau.

What quality_report measures does not depend on tau: the confusion of
every pair of cells with every column and without the key ones, the
excess of the source address of every cell, the counts of rows. tau
only compares them. judge turns these measures into the verdicts (the
defects of the cells, the regions they merge into, the shares of rows),
so that the stage, at quality.tau, and the dashboard, at any tau, run
the same code.
"""

from __future__ import annotations

__all__ = ["COUNTS", "Verdicts", "judge"]

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.cells import CellTree, merge_cells
from src.data import KINDS
from src.quality import CLASS_DEFECTS, DEFECTS, RIVAL_DEFECTS, defects_of, rival_defects

COUNTS = ("n_rows", "n_read", "conflict_rows", "conflict_read", "missing_rows")
COUNTS += ("missing_read",) + tuple(f"{kind}_rows" for kind in KINDS)
_OWN = (
    "cell",
    "class",
    "n_rows",
    "n_read",
    "n_duplicates",
    "dup_share",
)  # the tau-free
_OWN += ("conflict_rows", "conflict_read", "conflict_with")  # columns of a cell
_OWN += ("missing_rows", "missing_read")
_OWN += tuple(f"{kind}_rows" for kind in KINDS) + ("clear_rows",)


@dataclass(frozen=True)
class Verdicts:
    """The tables quality_report logs that depend on tau.

    Attributes:
        pairs: The quality_pairs rows, with region_a and region_b.
        cells: The quality_cells rows, with region, n_rivals, the rival
            columns, overlap_with, shortcut_with, shortcut_rival,
            shortcut_columns, confusion_without and defects.
        regions: The quality_regions rows.
        classes: The quality_classes rows.
        stats: tau, n_regions, regions_<defect>, regions_no_defect,
            share_<defect> and share_no_defect of quality_stats.
    """

    pairs: list[dict[str, Any]]
    cells: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    classes: list[dict[str, Any]]
    stats: dict[str, Any]


def judge(
    cells: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    tree: pl.DataFrame,
    classes: list[str],
    tau: float,
) -> Verdicts:
    """Return the verdicts of quality_report at tau.

    Per cell: the rival it is most confused with, the classes of
    class_overlap and of shortcut (src.quality.rival_defects on its side
    of every pair), the host of atypical_source (the excess of src_ip
    reaches tau, never without it) and its defects
    (src.quality.defects_of). Then the cells merge into regions
    (src.cells.merge_cells, class by class, a region named by the node
    of the cell tree at its top), whose defects come from the counts of
    their cells and the profile they share; then the shares of the CSV
    rows with each defect and with none, per class and over the
    dataset: a row has invalid_values when its data row holds a null or
    invalid value, duplicates when it repeats a row read before it,
    label_conflict when it is identical to a row of another class,
    class_overlap, shortcut and atypical_source when its cell has them,
    and none when it has none of the six (its first copy: clear_rows of
    the cells without a defect of cell).

    Args:
        cells: The tau-free quality_cells rows (those the stage logs
            are the judged ones too: what depends on tau is ignored),
            sorted by cell, one per cell 0 to m - 1: cell, class,
            COUNTS, n_duplicates, dup_share, conflict_with, clear_rows
            (its distinct rows without a null or invalid value and in no
            label conflict), the provenance columns when the dataset has
            them (src_ip_excess and src_ip_excess_value among them).
        pairs: The tau-free quality_pairs rows: cell_a, class_a, cell_b,
            class_b, n_conflicts, n_conflicts_read, nn_error_a,
            nn_error_b, nn_error, bayes_error_low, single_column_acc_a,
            single_column_acc_b, single_column_acc, single_column,
            key_columns, confusion_a, confusion_b, confusion_without_a
            and confusion_without_b (JSON lists); region_a and region_b
            are ignored. The confusion columns are None for a pair whose
            side had too few rows.
        tree: The cells' tree (cell_tree.parquet): node, class (the
            code), parent (-1 for a root), cell (null for a node that is
            not a cell), in pre-order per class.
        classes: The class names, by code.
        tau: The threshold, in (0, 0.5].

    Returns:
        The Verdicts. Same rows in the same order as the stage logs
        them, in the columns of quality_report's docstring.
    """
    m = len(cells)
    cell_class = np.array([classes.index(c["class"]) for c in cells], dtype=np.int64)
    sides = defaultdict(list)  # per cell: (rival, its confusion, the pair, its side)
    n_rivals = np.zeros(m, dtype=np.int64)
    for i, pair in enumerate(pairs):
        a, b = pair["cell_a"], pair["cell_b"]
        n_rivals[a] += 1
        n_rivals[b] += 1
        if pair["confusion_a"] is not None:
            for me, other, side in ((a, b, "a"), (b, a, "b")):
                steps = [pair[f"confusion_{side}"]]
                steps += json.loads(pair[f"confusion_without_{side}"])
                sides[me].append((other, np.array(steps, dtype=np.float64), i, side))
    # A cell's profile: its classes of class_overlap and of shortcut, and
    # the host of its atypical_source (None without it).
    profile = [
        (
            *rival_defects([(cell_class[r], e) for r, e, _, _ in sides[c]], tau),
            _atypical_host(cells[c], tau),
        )
        for c in range(m)
    ]
    region = _merge(tree, profile)
    pair_rows = [_pair_row(p, region, cells) for p in pairs]
    cell_rows = [
        _cell_row(
            c, cells[c], region, classes, n_rivals, sides, profile, pair_rows, tau
        )
        for c in range(m)
    ]
    region_rows = _region_rows(cell_rows, region, classes)
    # The defects the dataset cannot measure: atypical_source without the
    # source address in its provenance.
    unmeasured = () if "src_ip_excess" in cells[0] else CLASS_DEFECTS
    class_rows = _class_rows(cell_rows, region_rows, classes, unmeasured)
    stats = {
        "tau": tau,
        "n_regions": len(region_rows),
        **{
            f"regions_{d}": (
                None
                if d in unmeasured
                else sum(d in _listed(r["defects"]) for r in region_rows)
            )
            for d in DEFECTS
        },
        "regions_no_defect": sum(not r["defects"] for r in region_rows),
        **_shares(cell_rows, unmeasured),
    }
    return Verdicts(pair_rows, cell_rows, region_rows, class_rows, stats)


def _merge(tree: pl.DataFrame, profile: list) -> np.ndarray:
    """Return the region of every cell: cells.merge_cells class by class, the node at the top of its branch."""
    region = np.empty(len(profile), dtype=np.int64)
    for _, part in tree.group_by("class", maintain_order=True):
        first = int(part["node"][0])
        own = part["cell"].drop_nulls().to_numpy()
        base = int(own.min())
        cut = CellTree(
            parent=np.where(part["parent"] >= 0, part["parent"] - first, -1),
            cell=np.where(part["cell"].is_null(), -1, part["cell"].fill_null(0) - base),
        )
        region[base : base + len(own)] = (
            merge_cells(cut, profile[base : base + len(own)]) + first
        )
    return region


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes in code order joined by ",", None for none."""
    return ",".join(classes[c] for c in sorted(codes)) or None


def _listed(value: str | None) -> list[str]:
    return value.split(",") if value else []


def _atypical_host(row: dict, tau: float) -> str | None:
    """Return the host of a cell's atypical_source: its src_ip_excess_value when its src_ip_excess reaches tau."""
    measured = row.get("src_ip_excess")
    return (
        row["src_ip_excess_value"] if measured is not None and measured >= tau else None
    )


def _pair_row(pair: dict, region: np.ndarray, cells: list[dict]) -> dict:
    """Return a quality_pairs row with the region of each side, after its class."""
    row: dict[str, Any] = {}
    for key, value in pair.items():
        if key not in ("region_a", "region_b"):
            row[key] = value
        if key in ("class_a", "class_b"):
            side = key[-1]
            row[f"region_{side}"] = int(region[pair[f"cell_{side}"]])
    return row


def _cell_row(
    c, base, region, classes, n_rivals, sides, profile, pair_rows, tau
) -> dict:
    """Return the quality_cells row of a cell: the tau-free columns of base, then what tau decides."""
    overlap, shortcut, host = profile[c]
    mine = sides[c]
    rival = max(mine, key=lambda s: (s[1][0], -s[0]), default=None)
    under = [s for s in mine if s[1][0] < tau]
    cut = max(under, key=lambda s: (s[1][1:].max(), -s[0]), default=None)
    row = {k: base[k] for k in _OWN}
    row = {"cell": row.pop("cell"), "region": int(region[c]), **row}
    row.update(
        n_rivals=int(n_rivals[c]),
        **dict.fromkeys(("rival", "rival_class", "confusion", "nn_error")),
        **dict.fromkeys(("single_column_acc", "single_column")),
        overlap_with=_names(overlap, classes),
        shortcut_with=_names(shortcut, classes),
        **dict.fromkeys(("shortcut_rival", "shortcut_columns", "confusion_without")),
    )
    if rival is not None:
        other, steps, i, side = rival
        pair = pair_rows[i]
        row.update(
            rival=int(other),
            rival_class=pair[f"class_{'b' if side == 'a' else 'a'}"],
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
                row["missing_rows"],
                row["n_rows"],
                row["n_read"],
                row["conflict_rows"],
                overlap,
                shortcut,
                host is not None,
            )
        )
        or None
    )
    row.update({k: v for k, v in base.items() if k not in row and k != "region"})
    return row


def _region_rows(
    cell_rows: list[dict], region: np.ndarray, classes: list[str]
) -> list[dict]:
    """Return the quality_regions rows: the counts of their cells summed, the profile they share."""
    rows = []
    for r in np.unique(region):
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
        atypical = "atypical_source" in _listed(first["defects"])
        excesses = [
            r["src_ip_excess"] for r in own if r.get("src_ip_excess") is not None
        ]
        defects = defects_of(
            summed["missing_rows"],
            summed["n_rows"],
            summed["n_read"],
            summed["conflict_rows"],
            overlap,
            shortcut,
            atypical,
        )
        rows.append(
            {
                "region": int(r),
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
                "src_ip_excess_value": (
                    first["src_ip_excess_value"] if atypical else None
                ),
                "src_ip_excess": max(excesses, default=None),
                "defects": ",".join(defects) or None,
            }
        )
    return rows


def _class_rows(
    cell_rows: list[dict],
    region_rows: list[dict],
    classes: list[str],
    unmeasured: tuple[str, ...],
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
                **_shares(own, unmeasured),
            }
        )
    return rows


def _shares(
    cell_rows: list[dict], unmeasured: tuple[str, ...]
) -> dict[str, float | None]:
    """Return share_<defect> and share_no_defect over the CSV rows of these cells.

    Args:
        cell_rows: quality_cells rows, judged.
        unmeasured: The defects the dataset cannot measure: their
            share is None.
    """
    n_read = sum(r["n_read"] for r in cell_rows)
    found = {
        "invalid_values": sum(r["missing_read"] for r in cell_rows),
        "duplicates": sum(r["n_duplicates"] for r in cell_rows),
        "label_conflict": sum(r["conflict_read"] for r in cell_rows),
    }
    for d in (*RIVAL_DEFECTS, *CLASS_DEFECTS):
        found[d] = sum(r["n_read"] for r in cell_rows if d in _listed(r["defects"]))
    of_cell = set(RIVAL_DEFECTS) | set(CLASS_DEFECTS)
    clear = sum(
        r["clear_rows"] for r in cell_rows if not of_cell & set(_listed(r["defects"]))
    )
    return {
        **{
            f"share_{d}": None if d in unmeasured else found[d] / n_read
            for d in DEFECTS
        },
        "share_no_defect": clear / n_read,
    }
