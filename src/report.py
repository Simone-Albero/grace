"""The verdicts of quality_report at a value of tau.

What quality_report measures does not depend on tau: the share of the
rows of every cell confused with each rival class, the errors of the
one-column rules between its class and each rival class, the variance
of every branch of every class's tree and of the dataset, the share of
its CSV rows that break the scenario, the counts of rows. tau and the
confidence only compare them: every share is read with its interval
(src.quality), and a cell has a defect, has none, or is undecided. judge turns these measures into the verdicts
(the defects of the cells, the regions they merge into, the shares of
rows), so that the stage, at quality.tau, and the dashboard, at any
tau, run the same code.
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
from src.quality import (
    CLASS_DEFECTS,
    DEFECTS,
    RIVAL_DEFECTS,
    RivalMeasure,
    defects_of,
    repetition_verdicts,
    rival_defects,
    share_bounds,
    verdict,
)

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
_JUDGED = RIVAL_DEFECTS + CLASS_DEFECTS  # the defects of a cell, not of its rows


@dataclass(frozen=True)
class Verdicts:
    """The tables quality_report logs that depend on tau.

    Attributes:
        rivals: The quality_rivals rows, with region and rival_regions.
        cells: The quality_cells rows, with region, n_rivals, the rival
            columns, the classes of each rival defect, the shortcut
            columns, the branch of repetition, defects and undecided.
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
    branches: pl.DataFrame,
    classes: list[str],
    tau: float,
    confidence: float,
) -> Verdicts:
    """Return the verdicts of quality_report at tau.

    Per cell, the verdict of every entry, a defect against a target
    (True: it has it beyond the error of sampling, False: it has it not,
    None: undecided): src.quality.rival_defects against each rival class
    it was measured against (no rival defect against a class it has no
    rival cell of), src.quality.repetition_verdicts of the branches
    above it (repetition, which is never undecided), and
    the share of its CSV rows that break the scenario, on any key,
    against tau (off_scenario). A cell has a defect when an entry of it
    is True; class_overlap only against a class it is no suspect_label
    of. Then the cells merge into regions (src.cells.merge_cells, class
    by class: the largest branches whose cells do not contradict each
    other, each named by the node of the cell tree at its top), which
    have the defects their cells have; then the shares of the CSV rows
    with each defect and with none, per class and over the dataset: a
    row has invalid_values when its data row holds a null or invalid
    value, duplicates when it repeats a row read before it,
    label_conflict when it is identical to a row of another class, the
    others when its cell has them, and none when it has none (its first
    copy: clear_rows of the cells without a defect of cell).

    Args:
        cells: The tau-free quality_cells rows (those the stage logs
            are the judged ones too: what depends on tau is ignored),
            sorted by cell, one per cell 0 to m - 1: cell, class,
            COUNTS, n_duplicates, dup_share, conflict_with, clear_rows
            (its distinct rows without a null or invalid value and in no
            label conflict), variance_dataset (the variance of the
            dataset, the same in every cell),
            and with a scenario scenario and scenario_n (the share of
            its judged CSV rows that break it on any key, and those
            rows); off_scenario is not measured when no cell has one.
        rivals: The tau-free quality_rivals rows, one per cell and
            rival class: cell, class, rival_class, rival_cells (JSON
            list), rival_share (of the cell's rows, over its n_rows),
            n_rival_rows and rules (JSON list of [column, error on the
            cell, error on the rival class]); region and rival_regions
            are ignored.
        branches: The quality_branches rows, the cells' tree with the
            spread of every node: node, class (the code), parent (-1 for
            a root), cell (null for a node that is not a cell), n_rows,
            n_read, variance, sorted by node, in pre-order per class.
        classes: The class names, by code.
        tau: The threshold, in (0, 0.5].
        confidence: The level of the intervals, in (0.5, 1).

    Returns:
        The Verdicts. Same rows in the same order as the stage logs
        them, in the columns of quality_report's docstring.
    """
    m = len(cells)
    sides = defaultdict(list)  # per cell: (its RivalMeasure, the column names, row)
    rival_of = defaultdict(set)  # per cell: the classes of its rival cells
    n_rivals = np.zeros(m, dtype=np.int64)
    for i, rival in enumerate(rivals):
        c, k = rival["cell"], classes.index(rival["rival_class"])
        n_rivals[c] += len(json.loads(rival["rival_cells"]))
        rival_of[c].add(k)
        rules = json.loads(rival["rules"]) if rival["rules"] else []
        measure = RivalMeasure(
            rival_class=k,
            share=rival["rival_share"],
            n_rows=cells[c]["n_rows"],
            rules=np.array([r[1:] for r in rules], dtype=np.float64).reshape(-1, 2),
            n_rival=rival["n_rival_rows"],
        )
        sides[c].append((measure, [r[0] for r in rules], i))
    # The defects the dataset cannot measure: off_scenario without a
    # scenario (a table shared with other datasets may hold its column,
    # null).
    scenario = any(c.get("scenario") is not None for c in cells)
    unmeasured = () if scenario else ("off_scenario",)
    entries = [
        _entries(cells[c], sides[c], rival_of[c], classes, tau, confidence)
        for c in range(m)
    ]
    repeated, branch = _repetition(branches, cells, tau)
    for c in range(m):
        entries[c][("repetition", None)] = repeated[c]
    yes = [{e for e, v in found.items() if v is True} for found in entries]
    no = [{e for e, v in found.items() if v is False} for found in entries]
    region = _merge(branches, yes, no)
    rival_rows = [_rival_row(r, region) for r in rivals]
    judged = (region, classes, n_rivals, sides, entries, tau, confidence)
    judged += (_branch_of(branches, branch),)
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


def _entries(row, sides, rival_of, classes, tau, confidence) -> dict:
    """Return the verdict of every entry of a cell: (defect, class code or None) to True, False or None."""
    found = dict(rival_defects([s[0] for s in sides], tau, confidence))
    own = classes.index(row["class"])
    for k in range(len(classes)):
        if k != own and k not in rival_of:  # no rival cell: none of them
            found.update({(d, k): False for d in RIVAL_DEFECTS})
    share, n = row.get("scenario"), row.get("scenario_n")
    if share is not None and n:
        found[("off_scenario", None)] = verdict(
            *share_bounds(share, n, confidence), tau
        )
    return found


def _repetition(branches: pl.DataFrame, cells, tau):
    """Return repetition_verdicts over every class's tree, positions in place of node ids."""
    at = {v: i for i, v in enumerate(branches["node"].to_list())}
    parent = np.array([at.get(p, -1) for p in branches["parent"].to_list()])
    node_cell = branches["cell"].fill_null(-1).to_numpy()
    reference = cells[0]["variance_dataset"] if cells else 0.0
    return repetition_verdicts(
        parent,
        node_cell,
        branches["n_read"].to_numpy().astype(np.float64),
        branches["variance"].to_numpy().astype(np.float64),
        reference,
        tau,
    )


def _branch_of(branches: pl.DataFrame, top: np.ndarray) -> list[dict]:
    """Per cell, repetition_branch, repetition_share and repetition_ratio of the branch top (nulls for -1)."""
    node = branches["node"].to_numpy()
    n_read = branches["n_read"].to_numpy().astype(np.float64)
    variance = branches["variance"].to_numpy().astype(np.float64)
    root = np.arange(len(node))
    parent = branches["parent"].to_numpy()
    at = {v: i for i, v in enumerate(node.tolist())}
    for i in range(len(node)):  # a parent before its children: its root is set
        if parent[i] >= 0:
            root[i] = root[at[parent[i]]]
    rows = []
    for t in top.tolist():
        if t < 0:
            rows.append(dict.fromkeys(_BRANCH))
            continue
        rows.append(
            {
                "repetition_branch": int(node[t]),
                "repetition_share": float(n_read[t] / n_read[root[t]]),
                "repetition_variance": float(variance[t]),
            }
        )
    return rows


_BRANCH = ("repetition_branch", "repetition_share", "repetition_variance")
_RIVAL = ("rival_class", "rival_share", "rival_share_low", "rival_share_high")


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
    c, base, region, classes, n_rivals, sides, entries, tau, confidence, branch
) -> dict:
    """Return the quality_cells row of a cell: the tau-free columns of base, then what tau decides."""
    found = entries[c]
    held = _established(found)
    rival = max(sides[c], key=lambda s: (s[0].share, -s[0].rival_class), default=None)
    row = {k: base[k] for k in _OWN}
    row = {"cell": row.pop("cell"), "region": int(region[c]), **row}
    dataset = base["variance_dataset"]
    top = branch[c]
    row.update(
        n_rivals=int(n_rivals[c]),
        **dict.fromkeys(_RIVAL),
        label_with=_names(held["suspect_label"], classes),
        overlap_with=_names(held["class_overlap"], classes),
        shortcut_with=_names(held["shortcut"], classes),
        repetition_branch=top["repetition_branch"],
        repetition_share=top["repetition_share"],
        repetition_ratio=(
            top["repetition_variance"] / dataset
            if dataset and top["repetition_variance"] is not None
            else None
        ),
        **_shortcut(sides[c], held["shortcut"], classes, tau, confidence),
    )
    if rival is not None:
        measure = rival[0]
        low, high = share_bounds(measure.share, measure.n_rows, confidence)
        row.update(
            rival_class=classes[measure.rival_class],
            rival_share=measure.share,
            rival_share_low=low,
            rival_share_high=high,
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


def _shortcut(sides, against, classes, tau, confidence) -> dict:
    """Return shortcut_class, shortcut_columns and shortcut_error of a cell.

    Among the rival classes it has shortcut against, the one whose best
    rule errs least on its worse side; its columns that are a shortcut
    (both upper bounds under tau), the best first, and that error. None
    without a shortcut.
    """
    best = None
    for measure, columns, _ in sides:
        if measure.rival_class not in against:
            continue
        n = (measure.n_rows, measure.n_rival)
        cut = [
            (float(max(errors)), name)
            for name, errors in zip(columns, measure.rules)
            if all(share_bounds(e, k, confidence)[1] < tau for e, k in zip(errors, n))
        ]
        if cut and (best is None or min(cut)[0] < best[0][0][0]):
            best = (sorted(cut), measure.rival_class)
    if best is None:
        return dict.fromkeys(("shortcut_class", "shortcut_columns", "shortcut_error"))
    cut, k = best
    return {
        "shortcut_class": classes[k],
        "shortcut_columns": ",".join(name for _, name in cut),
        "shortcut_error": cut[0][0],
    }


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
            (row for row in own if row["rival_share"] is not None),
            key=lambda row: row["rival_share"],
            default=None,
        )

        def union(key: str) -> set[str]:
            return {n for row in own for n in _listed(row[key])}

        held = {
            "suspect_label": union("label_with"),
            "class_overlap": union("overlap_with"),
            "shortcut": union("shortcut_with"),
            **{
                d: (
                    {"cell"}
                    if any(d in _listed(row["defects"]) for row in own)
                    else set()
                )
                for d in ("repetition", "off_scenario")
            },
        }
        defects = defects_of(
            summed["missing_rows"],
            summed["n_rows"],
            summed["n_read"],
            summed["conflict_rows"],
            {d for d, targets in held.items() if targets},
        )
        codes = lambda names: {classes.index(n) for n in names}  # noqa: E731

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
                **{key: None if worst is None else worst[key] for key in _RIVAL},
                **_widest(own),
                "defects": ",".join(defects) or None,
            }
        )
    return rows


def _widest(own: list[dict]) -> dict:
    """The repetition_share and repetition_ratio of the widest branch of repetition among some cells, nulls without one."""
    held = [row for row in own if row["repetition_share"] is not None]
    if not held:
        return {"repetition_share": None, "repetition_ratio": None}
    top = max(held, key=lambda row: (row["repetition_share"], -row["repetition_ratio"]))
    return {k: top[k] for k in ("repetition_share", "repetition_ratio")}


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
