"""src.report: the verdicts of quality_report at any tau, from the measures it logs."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest

from src.config import load_config
from src.quality import CLASS_DEFECTS, RIVAL_DEFECTS, redundancy_verdict, rival_defects
from src.report import PROVENANCE, judge

TAUS = (0.02, 0.05, 0.1, 0.3, 0.5)


def _logged(workspace, table: str) -> list[dict]:
    """The rows of the last quality_report run in a table, in the order logged."""
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
    rivals = _logged(workspace, "quality_rivals")
    tree = pl.read_parquet(workspace.output_dir / "cell_tree.parquet").sort("node")
    classes = json.loads((workspace.output_dir / "classes.json").read_text())
    return cells, rivals, tree, classes


def _quality(workspace):
    return load_config(
        str(workspace.dataset), method=str(workspace.method)
    ).method.quality


def _tau(workspace) -> float:
    return _quality(workspace).tau


CONFIDENCE = 0.95  # the run's, the method's default


def test_judge_at_the_logged_tau_gives_the_logged_tables(reported):
    cells, rivals, tree, classes = _measures(reported)
    got = judge(cells, rivals, tree, classes, _tau(reported), CONFIDENCE)
    assert got.cells == cells
    assert got.rivals == rivals
    assert got.regions == sorted(
        _logged(reported, "quality_regions"), key=lambda r: r["region"]
    )
    assert got.classes == _logged(reported, "quality_classes")
    (stats,) = _logged(reported, "quality_stats")
    assert {k: stats[k] for k in got.stats} == got.stats


def test_judge_applies_tau_to_the_logged_rivals(reported):
    cells, rivals, tree, classes = _measures(reported)
    sides = {c["cell"]: [] for c in cells}
    for r in rivals:
        if r["confusion"] is not None:
            steps = [r["confusion"], *json.loads(r["confusion_without"])]
            k = classes.index(r["rival_class"])
            sides[r["cell"]].append((k, np.array(steps), r["n_scored"]))
    for tau in TAUS:
        got = judge(cells, rivals, tree, classes, tau, CONFIDENCE)
        for row in got.cells:
            found = rival_defects(sides[row["cell"]], tau, CONFIDENCE)
            names = lambda d: ",".join(  # noqa: E731
                classes[k]
                for (e, k), v in sorted(found.items(), key=str)
                if e == d and v
            )
            assert set((row["shortcut_with"] or "").split(",")) - {""} == set(
                names("shortcut").split(",")
            ) - {""}
            assert set((row["label_with"] or "").split(",")) - {""} == set(
                names("suspect_label").split(",")
            ) - {""}
            redundant = redundancy_verdict(
                row["variance"],
                row["variance_sd"],
                row["n_rows"],
                row["variance_typical"],
                tau,
                CONFIDENCE,
            )
            assert ("redundancy" in (row["defects"] or "")) == (redundant is True)


def test_a_higher_tau_finds_fewer_overlaps_and_more_redundancy(reported):
    cells, rivals, tree, classes = _measures(reported)

    def found(tau: float, defect: str) -> set[int]:
        return {
            r["cell"]
            for r in judge(cells, rivals, tree, classes, tau, CONFIDENCE).cells
            if defect in (r["defects"] or "")
        }

    for defect in ("class_overlap", "atypical_provenance"):
        sets = [found(tau, defect) for tau in TAUS]
        assert all(b <= a for a, b in zip(sets, sets[1:]))
    # redundancy is a variance at most tau times the typical one: more at a higher tau
    sets = [found(tau, "redundancy") for tau in TAUS]
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    # the synthetic classes overlap, and the tau range shows it
    assert found(TAUS[0], "class_overlap") != found(TAUS[-1], "class_overlap")


def test_the_defects_of_row_do_not_depend_on_tau(reported):
    cells, rivals, tree, classes = _measures(reported)
    rows = ("invalid_values", "duplicates", "label_conflict")
    low, high = (
        judge(cells, rivals, tree, classes, t, CONFIDENCE).cells for t in (0.02, 0.5)
    )
    for a, b in zip(low, high):
        keep = lambda r: {
            d for d in (r["defects"] or "").split(",") if d in rows
        }  # noqa: E731
        assert keep(a) == keep(b)
    shares = [
        judge(cells, rivals, tree, classes, t, CONFIDENCE).stats for t in (0.02, 0.5)
    ]
    for d in rows:
        assert shares[0][f"share_{d}"] == shares[1][f"share_{d}"]


def test_a_region_is_named_by_the_node_at_its_top(reported):
    cells, rivals, tree, classes = _measures(reported)
    parent = dict(zip(tree["node"].to_list(), tree["parent"].to_list()))
    node_of_cell = dict(zip(tree["cell"].to_list(), tree["node"].to_list()))
    kind = dict(zip(tree["node"].to_list(), tree["class"].to_list()))
    for tau in TAUS:
        got = judge(cells, rivals, tree, classes, tau, CONFIDENCE)
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
    got = judge(cells, rivals, tree, classes, 0.5, CONFIDENCE)
    roots = set(tree.filter(pl.col("parent") < 0)["node"].to_list())
    assert roots & {r["region"] for r in got.regions}


def test_the_shares_of_a_class_sum_to_the_dataset(reported):
    cells, rivals, tree, classes = _measures(reported)
    for tau in (0.05, 0.3):
        got = judge(cells, rivals, tree, classes, tau, CONFIDENCE)
        total = sum(r["n_read"] for r in got.classes)
        for name in (*RIVAL_DEFECTS, *CLASS_DEFECTS, "no_defect"):
            by_class = (
                sum(r[f"share_{name}"] * r["n_read"] for r in got.classes) / total
            )
            assert got.stats[f"share_{name}"] == pytest.approx(by_class)
        assert (
            got.stats["n_regions"]
            == len(got.regions)
            == sum(r["n_regions"] for r in got.classes)
        )


def test_a_dataset_without_a_provenance_does_not_measure_atypical_provenance(
    reported,
):
    # as the dashboard reads a quality_cells shared with datasets that have
    # one: the columns are there, null
    cells, rivals, tree, classes = _measures(reported)
    gone = {
        f"{key}_{m}": None for key in PROVENANCE for m in ("excess", "excess_share")
    }
    cells = [{**c, **gone} for c in cells]
    got = judge(cells, rivals, tree, classes, 0.1, CONFIDENCE)
    assert got.stats["share_atypical_provenance"] is None
    assert got.stats["regions_atypical_provenance"] is None
    assert got.stats["undecided_atypical_provenance"] is None
    assert all(r["share_atypical_provenance"] is None for r in got.classes)
    assert not any("atypical_provenance" in (c["defects"] or "") for c in got.cells)
    assert got.stats["share_class_overlap"] is not None


def test_a_higher_confidence_decides_fewer_cells(reported):
    cells, rivals, tree, classes = _measures(reported)
    tau = _tau(reported)
    loose, strict = (
        judge(cells, rivals, tree, classes, tau, c).stats for c in (0.6, 0.999)
    )
    for d in ("class_overlap", "shortcut"):
        assert strict[f"share_{d}"] <= loose[f"share_{d}"]
        assert strict[f"undecided_{d}"] >= loose[f"undecided_{d}"]
