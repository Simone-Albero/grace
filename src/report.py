"""The verdicts of quality_report, at the dataset's tau.

What quality_report measures does not depend on tau: per cell its rows,
its CSV rows, its rows with a null or invalid value or in a label
conflict, and the share of its rows confused with each rival class; per
node of each class's cut, the variance of its rows and, on each column,
the share of its rival class's rows inside its values. judge reads them
at tau and the confidence, once, in the stage: it sums the measures of
the cells up the cut, gives every node a verdict on every entry
(src.quality.node_verdicts), merges the cells into regions where no
verdict a part decides changes (src.cells.merge_nodes), and writes the
rows of the regions, of the classes and of the dataset. A region is a
node of the cut, and its verdicts are read on its own rows.
"""

from __future__ import annotations

__all__ = ["COUNTS", "Verdicts", "judge"]

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.cells import CellTree, merge_nodes
from src.data import KINDS
from src.quality import (
    DEFECTS,
    STRUCTURAL,
    low_variability_reference,
    low_variability_verdicts,
    node_sums,
    node_verdicts,
    share_bounds,
)

COUNTS = ("n_rows", "n_read", "conflict_rows", "conflict_read", "missing_rows")
COUNTS += ("missing_read",) + tuple(f"{kind}_rows" for kind in KINDS)


@dataclass(frozen=True)
class Verdicts:
    """The tables quality_report logs that depend on tau.

    Attributes:
        rivals: The quality_rivals rows, with region and rival_regions.
        cells: The quality_cells rows, with region.
        regions: The quality_regions rows.
        classes: The quality_classes rows.
        stats: tau, confidence, n_regions, regions_<defect>,
            regions_no_defect, undecided_<defect>, share_<defect>,
            share_no_defect and low_variability_reference of
            quality_stats.
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

    Per node of every class's cut, the measures of its cells summed
    (COUNTS, the rows confused with each rival class as the sum of the
    rival share times the rows, the provenance sums) and those the
    stage measured on the node (variance, cut_*); then the verdict of
    every entry (src.quality.node_verdicts, low_variability from
    src.quality.low_variability_verdicts against the median cell);
    then the regions (src.cells.merge_nodes on the raw verdicts). A
    region has a defect when an entry of it is True; majority_overlap
    only against a class it has no minority_overlap against. The share
    of the CSV rows with a defect is that of the rows of the regions
    with it, per class and over the dataset.

    Args:
        cells: The tau-free quality_cells rows, sorted by cell, one per
            cell 0 to m - 1: cell, class, COUNTS, n_duplicates,
            conflict_with and the provenance columns of the cell
            (<key>_same, <key>_own, <key>_chance, <key>_n, time_near,
            time_near_own, time_n; none without a provenance).
        rivals: The tau-free quality_rivals rows, one per cell and
            rival class: cell, class, rival_class, rival_cells (JSON
            list) and rival_share (of the cell's distinct rows).
        branches: The quality_branches rows, the cells' tree with the
            measures of every node: node, class (the code), parent (-1
            for a root), cell (null for a node that is not a cell),
            n_rows, n_read, variance, cut_class (null for none),
            cut_share, n_cut_rows, cut_columns (JSON list of [column,
            share]), sorted by node, in pre-order per class.
        classes: The class names, by code.
        tau: The threshold, in (0, 0.5].
        confidence: The level of the intervals, in (0.5, 1).

    Returns:
        The Verdicts: rivals and cells as given, with region and
        rival_regions; regions (one row per region: region, class,
        n_cells, the sums of its cells, conflict_with,
        <defect>_share/_low/_high of the three structural defects,
        rival_class with rival_share/_low/_high (its highest),
        minority_overlap_with and majority_overlap_with, cut_class,
        cut_share/_low/_high and cut_columns, low_variability_branch,
        low_variability_share and low_variability_ratio, defects and
        undecided joined by ",", the provenance sums); classes (class,
        n_rows, n_read, n_cells, n_regions, share_<defect>,
        share_no_defect); stats.
    """
    code = {name: k for k, name in enumerate(classes)}
    nodes = branches["node"].to_list()
    at = {v: i for i, v in enumerate(nodes)}
    parent = np.array([at.get(p, -1) for p in branches["parent"].to_list()])
    node_cell = branches["cell"].fill_null(-1).to_numpy().astype(np.int64)
    node_class = branches["class"].to_numpy().astype(np.int64)
    m, k = len(cells), len(nodes)

    def summed(key: str) -> np.ndarray:
        values = np.array([c.get(key) or 0 for c in cells], dtype=np.float64)
        return node_sums(parent, node_cell, values)

    count = {key: summed(key) for key in COUNTS}
    extra = [key for key in (cells[0] if cells else {}) if key not in _MEASURED]
    origin = {key: summed(key) for key in extra}  # the provenance sums
    confused_cell = np.zeros((m, len(classes)))
    present_cell = np.zeros((m, len(classes)))
    for r in rivals:
        c, j = r["cell"], code[r["rival_class"]]
        confused_cell[c, j] += r["rival_share"] * cells[c]["n_rows"]
        present_cell[c, j] = 1.0
    confused = node_sums(parent, node_cell, confused_cell)
    present = node_sums(parent, node_cell, present_cell) > 0
    conflict = _node_sets(
        parent, node_cell, [set(_listed(c["conflict_with"])) for c in cells]
    )
    n_rows, n_read = count["n_rows"], count["n_read"]
    variance = branches["variance"].to_numpy().astype(np.float64)
    reference = low_variability_reference(variance, node_cell)
    compact, top = low_variability_verdicts(parent, n_read, variance, reference, tau)
    root = np.arange(k)
    for v in range(k):  # a parent before its children
        if parent[v] >= 0:
            root[v] = root[parent[v]]
    cut_share = branches["cut_share"].to_list()
    n_cut = branches["n_cut_rows"].to_list()
    entries = [
        node_verdicts(
            own=int(node_class[v]),
            n_classes=len(classes),
            n_rows=float(n_rows[v]),
            n_read=float(n_read[v]),
            missing_rows=float(count["missing_rows"][v]),
            conflict_rows=float(count["conflict_rows"][v]),
            confused={j: float(x) for j, x in enumerate(confused[v]) if x > 0},
            cut_share=cut_share[v],
            n_cut_rows=int(n_cut[v] or 0),
            low_variability=bool(compact[v]),
            tau=tau,
            confidence=confidence,
        )
        for v in range(k)
    ]
    region = _merge(parent, node_cell, node_class, entries)  # its top node, per cell
    held = [_held(found) for found in entries]
    regions = []
    for t in sorted(set(region.tolist())):
        found, own = entries[t], held[t]
        rows = float(n_rows[t])
        shares = {
            "invalid_values": count["missing_rows"][t] / rows,
            "duplicates": 1 - rows / n_read[t] if n_read[t] > 0 else 0.0,
            "label_conflict": count["conflict_rows"][t] / rows,
        }
        row: dict[str, Any] = {
            "region": int(nodes[t]),
            "class": classes[node_class[t]],
            "n_cells": int((region == t).sum()),
            **{
                key: (
                    float(count[key][t])
                    if key.endswith("_read")
                    else int(count[key][t])
                )
                for key in COUNTS
            },
            "n_duplicates": float(n_read[t] - n_rows[t]),
            "conflict_with": _names(conflict[t], classes),
        }
        for d in STRUCTURAL:
            low, high = share_bounds(shares[d], rows, confidence)
            row.update(
                {f"{d}_share": float(shares[d]), f"{d}_low": low, f"{d}_high": high}
            )
        row.update(_worst_rival(confused[t], present[t], rows, classes, confidence))
        row.update(
            minority_overlap_with=_names(own["minority_overlap"], classes),
            majority_overlap_with=_names(own["majority_overlap"], classes),
            **_cut(branches, t, confidence),
        )
        if compact[t]:
            b = top[t]
            row.update(
                low_variability_branch=int(nodes[b]),
                low_variability_share=float(n_read[b] / n_read[root[b]]),
                low_variability_ratio=(
                    float(variance[b] / reference) if reference else None
                ),
            )
        else:
            row.update(
                low_variability_branch=None,
                low_variability_share=None,
                low_variability_ratio=None,
            )
        defects = [d for d in DEFECTS if own[d]]
        undecided = [
            d
            for d in DEFECTS
            if d not in defects
            and any(v is None for (e, _), v in found.items() if e == d)
        ]
        row["defects"] = ",".join(defects) or None
        row["undecided"] = ",".join(undecided) or None
        for key in extra:
            value = origin[key][t]
            row[key] = int(value) if key.endswith("_n") else float(value)
        regions.append(row)
    cell_rows = [
        {"cell": c["cell"], "region": int(nodes[region[c["cell"]]]), **_without(c)}
        for c in cells
    ]
    rival_rows = [_rival_row(r, region, nodes) for r in rivals]
    class_rows = _class_rows(cell_rows, regions, classes)
    stats = {
        "tau": tau,
        "confidence": confidence,
        "n_regions": len(regions),
        **{
            f"regions_{d}": sum(d in _listed(r["defects"]) for r in regions)
            for d in DEFECTS
        },
        "regions_no_defect": sum(not r["defects"] for r in regions),
        **{
            f"undecided_{d}": sum(d in _listed(r["undecided"]) for r in regions)
            for d in DEFECTS
        },
        **_shares(regions),
        "low_variability_reference": reference,
    }
    return Verdicts(rival_rows, cell_rows, regions, class_rows, stats)


_MEASURED = ("cell", "region", "class", *COUNTS, "n_duplicates", "conflict_with")
# the columns of a quality_cells row that are no provenance sum


def _held(found: dict) -> dict[str, set]:
    """Return, per defect, the targets an entry has it against: majority_overlap not where minority_overlap."""
    held: dict[str, set] = {d: set() for d in DEFECTS}
    for (d, target), value in found.items():
        if value is not True:
            continue
        if d == "majority_overlap" and found.get(("minority_overlap", target)) is True:
            continue
        held[d].add(target)
    return held


def _merge(
    parent: np.ndarray, node_cell: np.ndarray, node_class: np.ndarray, entries: list
) -> np.ndarray:
    """Return the top node of the region of every cell: src.cells.merge_nodes, class by class."""
    region = np.empty(int((node_cell >= 0).sum()), dtype=np.int64)
    for c in dict.fromkeys(
        node_class.tolist()
    ):  # the classes in the order of their trees
        span = np.flatnonzero(node_class == c)
        first = int(span[0])
        own = node_cell[span]
        base = int(own[own >= 0].min())
        cut = CellTree(
            parent=np.where(parent[span] >= 0, parent[span] - first, -1),
            cell=np.where(own >= 0, own - base, -1),
        )
        found = merge_nodes(cut, [entries[i] for i in span])
        region[base : base + len(found)] = found + first
    return region


def _node_sets(
    parent: np.ndarray, node_cell: np.ndarray, per_cell: list[set]
) -> list[set]:
    """Return, per node, the union of the sets of the cells below it."""
    out: list[set] = [set() for _ in range(len(parent))]
    for v in range(len(parent) - 1, -1, -1):  # children after their parent
        if node_cell[v] >= 0:
            out[v] |= per_cell[node_cell[v]]
        if parent[v] >= 0:
            out[parent[v]] |= out[v]
    return out


def _worst_rival(
    confused: np.ndarray,
    present: np.ndarray,
    rows: float,
    classes: list[str],
    confidence: float,
) -> dict[str, Any]:
    """Return rival_class, rival_share, rival_share_low and rival_share_high: the highest of a node's rival classes (ties to the lower code), nulls without one."""
    if not present.any():
        return dict.fromkeys(_RIVAL)
    share = np.where(present, confused / rows, -1.0)
    j = int(np.argmax(share))
    low, high = share_bounds(float(share[j]), rows, confidence)
    return dict(zip(_RIVAL, (classes[j], float(share[j]), low, high)))


_RIVAL = ("rival_class", "rival_share", "rival_share_low", "rival_share_high")


def _cut(branches: pl.DataFrame, t: int, confidence: float) -> dict[str, Any]:
    """Return cut_class, cut_share, cut_low, cut_high and cut_columns of a node (nulls without a rival class)."""
    node = branches.row(t, named=True)
    if node["cut_share"] is None:
        return dict.fromkeys(
            ("cut_class", "cut_share", "cut_low", "cut_high", "cut_columns")
        )
    low, high = share_bounds(node["cut_share"], node["n_cut_rows"], confidence)
    return {
        "cut_class": node["cut_class"],
        "cut_share": float(node["cut_share"]),
        "cut_low": low,
        "cut_high": high,
        "cut_columns": node["cut_columns"],
    }


def _class_rows(
    cell_rows: list[dict], regions: list[dict], classes: list[str]
) -> list[dict]:
    """Return the quality_classes rows: per class its counts and _shares over its regions."""
    rows = []
    for name in classes:
        own = [r for r in cell_rows if r["class"] == name]
        if not own:
            continue
        mine = [r for r in regions if r["class"] == name]
        rows.append(
            {
                "class": name,
                "n_rows": sum(r["n_rows"] for r in own),
                "n_read": sum(r["n_read"] for r in own),
                "n_cells": len(own),
                "n_regions": len(mine),
                **_shares(mine),
            }
        )
    return rows


def _shares(regions: list[dict]) -> dict[str, float]:
    """Return share_<defect> and share_no_defect: the share of the CSV rows of these regions in regions with it, or with none."""
    total = sum(r["n_read"] for r in regions)

    def share(rows: float) -> float:
        return rows / total if total else 0.0

    return {
        **{
            f"share_{d}": share(
                sum(r["n_read"] for r in regions if d in _listed(r["defects"]))
            )
            for d in DEFECTS
        },
        "share_no_defect": share(sum(r["n_read"] for r in regions if not r["defects"])),
    }


def _rival_row(rival: dict, region: np.ndarray, nodes: list) -> dict:
    """Return a quality_rivals row with the region of its cell, after the cell, and of its rival cells, after them."""
    row: dict[str, Any] = {}
    for key, value in rival.items():
        if key in ("region", "rival_regions"):
            continue
        row[key] = value
        if key == "cell":
            row["region"] = int(nodes[region[value]])
        if key == "rival_cells":
            others = json.loads(value)
            row["rival_regions"] = json.dumps(
                sorted({int(nodes[region[o]]) for o in others})
            )
    return row


def _without(row: dict) -> dict:
    """Return a quality_cells row without cell and region (they come first)."""
    return {k: v for k, v in row.items() if k not in ("cell", "region")}


def _names(codes, classes: list[str]) -> str | None:
    """Return the names of class codes (or names) in code order joined by ",", None for none."""
    ks = sorted(c if isinstance(c, int) else classes.index(c) for c in codes)
    return ",".join(classes[c] for c in ks) or None


def _listed(value: str | None) -> list[str]:
    return value.split(",") if value else []
