"""src.report: the verdicts of quality_report at any tau, from the measures it logs."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest

from src.config import load_config
from src.quality import rival_defects
from src.report import judge

TAUS = (0.02, 0.05, 0.1, 0.3, 0.5)


def _logged(workspace, table: str) -> list[dict]:
    """The rows of the last quality_report run in a table, by cell then pair."""
    with sqlite3.connect(workspace.db_path) as conn:
        conn.row_factory = sqlite3.Row
        (run,) = conn.execute(
            "SELECT max(id) FROM runs WHERE stage = 'quality_report' AND status = 'ok'"
        ).fetchone()
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ?", (run,)
        ).fetchall()
    return [{k: r[k] for k in r.keys() if k != "run_id"} for r in rows]


def _measures(workspace):
    cells = sorted(_logged(workspace, "quality_cells"), key=lambda r: r["cell"])
    pairs = _logged(workspace, "quality_pairs")
    tree = pl.read_parquet(workspace.output_dir / "cell_tree.parquet").sort("node")
    classes = json.loads((workspace.output_dir / "classes.json").read_text())
    return cells, pairs, tree, classes


def _tau(workspace) -> float:
    return load_config(
        str(workspace.dataset), method=str(workspace.method)
    ).method.quality.tau


def test_judge_at_the_logged_tau_gives_the_logged_tables(reported):
    cells, pairs, tree, classes = _measures(reported)
    got = judge(cells, pairs, tree, classes, _tau(reported))
    assert got.cells == cells
    assert got.pairs == pairs
    assert got.regions == sorted(
        _logged(reported, "quality_regions"), key=lambda r: r["region"]
    )
    assert got.classes == _logged(reported, "quality_classes")
    (stats,) = _logged(reported, "quality_stats")
    assert {k: stats[k] for k in got.stats} == got.stats


def test_judge_applies_tau_to_the_logged_pairs(reported):
    cells, pairs, tree, classes = _measures(reported)
    sides = {c["cell"]: [] for c in cells}
    for p in pairs:
        for me, other, k in (
            (p["cell_a"], p["class_b"], "a"),
            (p["cell_b"], p["class_a"], "b"),
        ):
            if p[f"confusion_{k}"] is not None:
                steps = [p[f"confusion_{k}"], *json.loads(p[f"confusion_without_{k}"])]
                sides[me].append((classes.index(other), np.array(steps)))
    for tau in TAUS:
        got = judge(cells, pairs, tree, classes, tau)
        for row in got.cells:
            overlap, shortcut = rival_defects(sides[row["cell"]], tau)
            assert row["overlap_with"] == (
                ",".join(classes[c] for c in overlap) or None
            )
            assert row["shortcut_with"] == (
                ",".join(classes[c] for c in shortcut) or None
            )
            has = "atypical_source" in (row["defects"] or "")
            assert has == (
                row["src_ip_excess"] is not None and row["src_ip_excess"] >= tau
            )


def test_a_higher_tau_finds_fewer_overlaps_and_hosts(reported):
    cells, pairs, tree, classes = _measures(reported)

    def found(tau: float, defect: str) -> set[int]:
        return {
            r["cell"]
            for r in judge(cells, pairs, tree, classes, tau).cells
            if defect in (r["defects"] or "")
        }

    for defect in ("class_overlap", "atypical_source"):
        sets = [found(tau, defect) for tau in TAUS]
        assert all(b <= a for a, b in zip(sets, sets[1:]))
    # the synthetic classes overlap, and the tau range shows it
    assert found(TAUS[0], "class_overlap") != found(TAUS[-1], "class_overlap")


def test_the_defects_of_row_do_not_depend_on_tau(reported):
    cells, pairs, tree, classes = _measures(reported)
    rows = ("invalid_values", "duplicates", "label_conflict")
    low, high = (judge(cells, pairs, tree, classes, t).cells for t in (0.02, 0.5))
    for a, b in zip(low, high):
        keep = lambda r: {
            d for d in (r["defects"] or "").split(",") if d in rows
        }  # noqa: E731
        assert keep(a) == keep(b)
    shares = [judge(cells, pairs, tree, classes, t).stats for t in (0.02, 0.5)]
    for d in rows:
        assert shares[0][f"share_{d}"] == shares[1][f"share_{d}"]


def test_a_region_is_named_by_the_node_at_its_top(reported):
    cells, pairs, tree, classes = _measures(reported)
    parent = dict(zip(tree["node"].to_list(), tree["parent"].to_list()))
    node_of_cell = dict(zip(tree["cell"].to_list(), tree["node"].to_list()))
    kind = dict(zip(tree["node"].to_list(), tree["class"].to_list()))
    for tau in TAUS:
        got = judge(cells, pairs, tree, classes, tau)
        assert [r["region"] for r in got.regions] == sorted(
            {c["region"] for c in got.cells}
        )
        for row in got.cells:
            v = node_of_cell[row["cell"]]
            while v != row["region"]:  # the region is above the cell, in its class
                v = parent[v]
                assert v >= 0
            assert classes[kind[v]] == row["class"]
    # a class whose cells share one profile is one region: its root
    got = judge(cells, pairs, tree, classes, 0.5)
    roots = set(tree.filter(pl.col("parent") < 0)["node"].to_list())
    assert roots & {r["region"] for r in got.regions}


def test_the_shares_of_a_class_sum_to_the_dataset(reported):
    cells, pairs, tree, classes = _measures(reported)
    for tau in (0.05, 0.3):
        got = judge(cells, pairs, tree, classes, tau)
        total = sum(r["n_read"] for r in got.classes)
        for name in ("class_overlap", "shortcut", "atypical_source", "no_defect"):
            by_class = (
                sum(r[f"share_{name}"] * r["n_read"] for r in got.classes) / total
            )
            assert got.stats[f"share_{name}"] == pytest.approx(by_class)
        assert (
            got.stats["n_regions"]
            == len(got.regions)
            == sum(r["n_regions"] for r in got.classes)
        )
