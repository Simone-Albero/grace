"""The verdicts of quality_report at a value of tau.

What quality_report measures does not depend on tau: the confusion of
every cell with each rival class with every column and without the key
ones, with the rows it was measured on, the variance of every cell and
of the typical cell of its class, the excess of every provenance value
of every cell, the counts of rows. tau and the confidence only compare
them: every measure is read with its interval (src.quality), and a cell
has a defect, has none, or is undecided. judge turns these measures
into the verdicts (the defects of the cells, the regions they merge
into, the shares of rows), so that the stage, at quality.tau, and the
dashboard, at any tau, run the same code.
"""

from __future__ import annotations

__all__ = ["COUNTS", "PROVENANCE", "Verdicts", "judge"]

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.cells import CellTree, merge_cells
from src.data import KINDS
from src.provenance import KEYS, ORIGIN, TIME
from src.quality import (
    CLASS_DEFECTS,
    DEFECTS,
    RIVAL_DEFECTS,
    defects_of,
    provenance_verdict,
    redundancy_verdict,
    rival_defects,
    share_bounds,
)

COUNTS = ("n_rows", "n_read", "conflict_rows", "conflict_read", "missing_rows")
COUNTS += ("missing_read",) + tuple(f"{kind}_rows" for kind in KINDS)
MEASURED = (*KEYS, TIME)  # the provenance keys measured per cell, in order
PROVENANCE = tuple(
    k for k in MEASURED if k in ORIGIN
)  # those atypical_provenance is judged on
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
_JUDGED = RIVAL_DEFECTS + CLASS_DEFECTS  # the defects of a cell, not of its rows


@dataclass(frozen=True)
class Verdicts:
    """The tables quality_report logs that depend on tau.

    Attributes:
        rivals: The quality_rivals rows, with region and rival_regions.
        cells: The quality_cells rows, with region, n_rivals, the rival
            columns, the classes and keys of each defect, the shortcut
            columns, variance_ratio, defects and undecided.
        regions: The quality_regions rows.
        classes: The quality_classes rows.
        stats: tau, confidence, n_regions, regions_<defect>,
            regions_no_defect, undecided_<defect>, share_<defect> and
            share_no_defect of quality_stats.
    """

    rivals: list[dict[str, Any]]
    cells: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    classes: list[dict[str, Any]]
    stats: dict[str, Any]


def judge(
    cells: list[dict[str, Any]],
    rivals: list[dict[str, Any]],
    tree: pl.DataFrame,
    classes: list[str],
    tau: float,
    confidence: float,
) -> Verdicts:
    """Return the verdicts of quality_report at tau.

    Per cell, the verdict of every entry, a defect against a target
    (True: it has it beyond the error of sampling, False: it has it not,
    None: undecided): src.quality.rival_defects against each rival class
    it was measured against (no rival defect against a class it has no
    rival cell of), redundancy_verdict of its variance against the
    typical one of its class, provenance_verdict of the excess of each
    provenance key. A cell has a defect when an entry of it is True;
    class_overlap only against a class it is no suspect_label of. Then
    the cells merge into regions (src.cells.merge_cells, class by class:
    the largest branches whose cells do not contradict each other, each
    named by the node of the cell tree at its top), which have the
    defects their cells have; then the shares of the CSV rows with each
    defect and with none, per class and over the dataset: a row has
    invalid_values when its data row holds a null or invalid value,
    duplicates when it repeats a row read before it, label_conflict when
    it is identical to a row of another class, the others when its cell
    has them, and none when it has none (its first copy: clear_rows of
    the cells without a defect of cell).

    Args:
        cells: The tau-free quality_cells rows (those the stage logs
            are the judged ones too: what depends on tau is ignored),
            sorted by cell, one per cell 0 to m - 1: cell, class,
            COUNTS, n_duplicates, dup_share, conflict_with, clear_rows
            (its distinct rows without a null or invalid value and in no
            label conflict), variance, variance_sd, variance_typical,
            the provenance columns when the dataset has them
            (<key>_excess, <key>_excess_share and <key>_n among them);
            atypical_provenance is not measured when no cell has an
            excess.
        rivals: The tau-free quality_rivals rows, one per cell and
            rival class: cell, class, rival_class, rival_cells (JSON
            list), n_measured, n_rival_measured, n_scored, nn_error,
            bayes_error_low, single_column_acc, single_column,
            key_columns, confusion and confusion_without (JSON list),
            all of the cell's side; region and rival_regions are
            ignored. n_scored, the 1-NN, key column and confusion
            columns are None when a side had too few rows.
        tree: The cells' tree (cell_tree.parquet): node, class (the
            code), parent (-1 for a root), cell (null for a node that is
            not a cell), in pre-order per class.
        classes: The class names, by code.
        tau: The threshold, in (0, 0.5].
        confidence: The level of the intervals, in (0.5, 1).

    Returns:
        The Verdicts. Same rows in the same order as the stage logs
        them, in the columns of quality_report's docstring.
    """
    m = len(cells)
    sides = defaultdict(list)  # per cell: (rival class, its confusion, rows, row)
    rival_of = defaultdict(set)  # per cell: the classes of its rival cells
    n_rivals = np.zeros(m, dtype=np.int64)
    for i, rival in enumerate(rivals):
        c, k = rival["cell"], classes.index(rival["rival_class"])
        n_rivals[c] += len(json.loads(rival["rival_cells"]))
        rival_of[c].add(k)
        if rival["confusion"] is not None:
            steps = [rival["confusion"], *json.loads(rival["confusion_without"])]
            sides[c].append((k, np.array(steps), rival["n_scored"], i))
    keys = [
        key
        for key in PROVENANCE
        if any(c.get(f"{key}_excess") is not None for c in cells)
    ]
    # The defects the dataset cannot measure: atypical_provenance without
    # a provenance (a table shared with other datasets may hold its
    # columns, null).
    unmeasured = () if keys else ("atypical_provenance",)
    entries = [
        _entries(cells[c], sides[c], rival_of[c], classes, keys, tau, confidence)
        for c in range(m)
    ]
    yes = [{e for e, v in found.items() if v is True} for found in entries]
    no = [{e for e, v in found.items() if v is False} for found in entries]
    region = _merge(tree, yes, no)
    rival_rows = [_rival_row(r, region) for r in rivals]
    judged = (region, classes, n_rivals, sides, entries, rivals, tau, confidence)
    cell_rows = [_cell_row(c, cells[c], *judged) for c in range(m)]
    region_rows = _region_rows(cell_rows, region, classes)
    class_rows = _class_rows(cell_rows, region_rows, classes, unmeasured)
    stats = {
        "tau": tau,
        "confidence": confidence,
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
        **{
            f"undecided_{d}": (
                None
                if d in unmeasured
                else sum(d in _listed(r["undecided"]) for r in cell_rows)
            )
            for d in _JUDGED
        },
        **_shares(cell_rows, unmeasured),
    }
    return Verdicts(rival_rows, cell_rows, region_rows, class_rows, stats)


def _entries(row, sides, rival_of, classes, keys, tau, confidence) -> dict:
    """Return the verdict of every entry of a cell: (defect, class code, key or None) to True, False or None."""
    found = {
        (d, k): v
        for (d, k), v in rival_defects(
            [(k, steps, n) for k, steps, n, _ in sides], tau, confidence
        ).items()
    }
    own = classes.index(row["class"])
    for k in range(len(classes)):
        if k != own and k not in rival_of:  # no rival cell: none of them
            found.update({(d, k): False for d in RIVAL_DEFECTS})
    found[("redundancy", None)] = redundancy_verdict(
        row["variance"],
        row["variance_sd"],
        row["n_rows"],
        row["variance_typical"],
        tau,
        confidence,
    )
    for key in keys:
        if row.get(f"{key}_excess") is not None:
            found[("atypical_provenance", key)] = provenance_verdict(
                row[f"{key}_excess_share"],
                row[f"{key}_excess"],
                row[f"{key}_n"],
                tau,
                confidence,
            )
    return found


def _established(found: dict) -> dict[str, set]:
    """Return, per defect of _JUDGED, the targets a cell has it against: class_overlap not where suspect_label."""
    held = {d: set() for d in _JUDGED}
    for (d, target), v in found.items():
        if v is not True:
            continue
        if d == "class_overlap" and found.get(("suspect_label", target)) is True:
            continue
        held[d].add(target)
    return held


def _merge(tree: pl.DataFrame, yes: list[set], no: list[set]) -> np.ndarray:
    """Return the region of every cell: cells.merge_cells class by class, the node at the top of its branch."""
    region = np.empty(len(yes), dtype=np.int64)
    for _, part in tree.group_by("class", maintain_order=True):
        first = int(part["node"][0])
        own = part["cell"].drop_nulls().to_numpy()
        base = int(own.min())
        cut = CellTree(
            parent=np.where(part["parent"] >= 0, part["parent"] - first, -1),
            cell=np.where(part["cell"].is_null(), -1, part["cell"].fill_null(0) - base),
        )
        span = slice(base, base + len(own))
        region[span] = merge_cells(cut, yes[span], no[span]) + first
    return region


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes in code order joined by ",", None for none."""
    return ",".join(classes[c] for c in sorted(codes)) or None


def _keys(keys) -> str | None:
    """Return provenance keys in PROVENANCE order joined by ",", None for none."""
    return ",".join(k for k in PROVENANCE if k in keys) or None


def _listed(value: str | None) -> list[str]:
    return value.split(",") if value else []


def _rival_row(rival: dict, region: np.ndarray) -> dict:
    """Return a quality_rivals row with the region of its cell, after the cell, and of its rival cells, after them."""
    row: dict[str, Any] = {}
    for key, value in rival.items():
        if key in ("region", "rival_regions"):
            continue
        row[key] = value
        if key == "cell":
            row["region"] = int(region[value])
        if key == "rival_cells":
            others = json.loads(value)
            row["rival_regions"] = json.dumps(sorted({int(region[o]) for o in others}))
    return row


def _cell_row(
    c, base, region, classes, n_rivals, sides, entries, rivals, tau, confidence
) -> dict:
    """Return the quality_cells row of a cell: the tau-free columns of base, then what tau decides."""
    found = entries[c]
    held = _established(found)
    mine = sides[c]
    rival = max(mine, key=lambda s: (s[1][0], -s[0]), default=None)
    apart = [s for s in mine if found.get(("class_overlap", s[0])) is False]
    cut = max(apart, key=lambda s: (s[1][1:].max() - s[1][0], -s[0]), default=None)
    row = {k: base[k] for k in _OWN}
    row = {"cell": row.pop("cell"), "region": int(region[c]), **row}
    typical = base["variance_typical"]
    row.update(
        n_rivals=int(n_rivals[c]),
        **dict.fromkeys(("rival_class", "confusion", "confusion_low")),
        **dict.fromkeys(("confusion_high", "n_scored", "nn_error")),
        **dict.fromkeys(("single_column_acc", "single_column")),
        label_with=_names(held["suspect_label"], classes),
        overlap_with=_names(held["class_overlap"], classes),
        shortcut_with=_names(held["shortcut"], classes),
        provenance_with=_keys(held["atypical_provenance"]),
        variance_ratio=base["variance"] / typical if typical else None,
        **dict.fromkeys(("shortcut_class", "shortcut_columns", "confusion_without")),
    )
    if rival is not None:
        k, steps, n, i = rival
        low, high = share_bounds(float(steps[0]), n, confidence)
        row.update(
            rival_class=classes[k],
            confusion=float(steps[0]),
            confusion_low=low,
            confusion_high=high,
            n_scored=int(n),
            nn_error=rivals[i]["nn_error"],
            single_column_acc=rivals[i]["single_column_acc"],
            single_column=rivals[i]["single_column"],
        )
    if cut is not None:
        k, steps, _, i = cut
        # the first key columns whose removal adds tau of the rows
        reached = np.flatnonzero(steps[1:] - steps[0] >= tau)
        last = int(reached[0]) + 1 if len(reached) else len(steps) - 1
        row.update(
            shortcut_class=classes[k],
            shortcut_columns=",".join(rivals[i]["key_columns"].split(",")[:last]),
            confusion_without=float(steps[last]),
        )
    established = {d for d, targets in held.items() if targets}
    row["defects"] = (
        ",".join(
            defects_of(
                row["missing_rows"],
                row["n_rows"],
                row["n_read"],
                row["conflict_rows"],
                established,
            )
        )
        or None
    )
    maybe = {d for (d, _), v in found.items() if v is None} - established
    row["undecided"] = ",".join(d for d in _JUDGED if d in maybe) or None
    row.update({k: v for k, v in base.items() if k not in row and k != "region"})
    return row


def _region_rows(
    cell_rows: list[dict], region: np.ndarray, classes: list[str]
) -> list[dict]:
    """Return the quality_regions rows: the counts of their cells summed, the defects they have."""
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

        def union(key: str) -> set[str]:
            return {n for row in own for n in _listed(row[key])}

        held = {
            "suspect_label": union("label_with"),
            "class_overlap": union("overlap_with"),
            "shortcut": union("shortcut_with"),
            "redundancy": (
                {"cell"}
                if any("redundancy" in _listed(row["defects"]) for row in own)
                else set()
            ),
            "atypical_provenance": union("provenance_with"),
        }
        defects = defects_of(
            summed["missing_rows"],
            summed["n_rows"],
            summed["n_read"],
            summed["conflict_rows"],
            {d for d, targets in held.items() if targets},
        )
        codes = lambda names: {classes.index(n) for n in names}  # noqa: E731
        ratios = [row["variance_ratio"] for row in own if row["variance_ratio"]]
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
                "conflict_with": _names(codes(union("conflict_with")), classes),
                "missing_rows": int(summed["missing_rows"]),
                "missing_read": float(summed["missing_read"]),
                **{f"{kind}_rows": int(summed[f"{kind}_rows"]) for kind in KINDS},
                "label_with": _names(codes(held["suspect_label"]), classes),
                "overlap_with": _names(codes(held["class_overlap"]), classes),
                "shortcut_with": _names(codes(held["shortcut"]), classes),
                "provenance_with": _keys(held["atypical_provenance"]),
                **{
                    key: None if worst is None else worst[key]
                    for key in (
                        "confusion",
                        "confusion_low",
                        "confusion_high",
                        "rival_class",
                    )
                },
                "variance_ratio": min(ratios, default=None),
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
    for d in _JUDGED:
        found[d] = sum(r["n_read"] for r in cell_rows if d in _listed(r["defects"]))
    clear = sum(
        r["clear_rows"]
        for r in cell_rows
        if not set(_JUDGED) & set(_listed(r["defects"]))
    )
    return {
        **{
            f"share_{d}": None if d in unmeasured else found[d] / n_read
            for d in DEFECTS
        },
        "share_no_defect": clear / n_read,
    }
