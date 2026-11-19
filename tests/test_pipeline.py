"""Every stage end to end on the small synthetic dataset."""

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from fractions import Fraction

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import (
    COPY_SRC,
    COPY_TIME,
    N_DUPLICATES,
    STAGE_ORDER,
    Workspace,
    make_workspace,
)

from src.config import load_config
from src.data import DUPLICATES, KINDS, LABEL
from src.prepare import MISSING, OTHER
from src.provenance import KEYS, TIME
from src.quality import (
    CLASS_DEFECTS,
    DEFECTS,
    RIVAL_DEFECTS,
    decisive_rows,
    defects_of,
    provenance_verdict,
    redundancy_verdict,
    rival_defects,
)
from src.report import PROVENANCE
from stages.paths import read_points


def _rows(workspace: Workspace, sql: str) -> list[tuple]:
    with sqlite3.connect(workspace.db_path) as conn:
        return conn.execute(sql).fetchall()


def _read(workspace: Workspace, name: str) -> pl.DataFrame:
    return pl.read_parquet(workspace.output_dir / name)


def test_every_run_is_ok(done):
    runs = _rows(done, "SELECT stage, status FROM runs ORDER BY id")[: len(STAGE_ORDER)]
    assert [r[0] for r in runs] == list(STAGE_ORDER)
    assert {r[1] for r in runs} == {"ok"}


def test_clean_drops_the_rare_class_and_the_constant_column_only(done):
    classes = json.loads((done.output_dir / "classes.json").read_text())
    assert "class_10" not in classes and len(classes) == 9
    dropped = _rows(done, "SELECT label FROM dropped_classes")
    assert dropped == [("class_10",)]
    assert _rows(done, "SELECT column FROM dropped_columns") == [("flat",)]
    data = _read(done, "data.parquet")
    n_raw, n_duplicate, n_rare, n_capped, n_kept = _rows(
        done,
        "SELECT n_raw, n_duplicate, n_rare, n_capped, n_kept FROM clean_summary",
    )[0]
    assert n_duplicate == N_DUPLICATES and n_capped == 0
    assert data.height == n_kept == n_raw - n_duplicate - n_rare - n_capped
    (n_unparsable,) = _rows(done, "SELECT n_unparsable FROM clean_summary")[0]
    assert n_unparsable == 1
    counts = dict(_rows(done, "SELECT label, n_rows FROM class_counts"))
    per_code = dict(data[LABEL].value_counts().iter_rows())
    assert counts == {name: per_code[code] for code, name in enumerate(classes)}
    (n_read,) = _rows(done, "SELECT sum(n_read) FROM class_counts")[0]
    assert data[DUPLICATES].sum() == n_read
    numbers = data.select(pl.selectors.starts_with("num_"))
    assert numbers.select(pl.any_horizontal(pl.all().is_infinite().any())).item()
    assert numbers.null_count().sum_horizontal().item() > 0
    assert "flat" not in data.columns
    assert data["cat_1"].dtype == pl.String
    assert data.columns[-1] == LABEL
    assert set(data[LABEL].unique().to_list()) == set(range(9))


def test_clean_keeps_one_copy_of_each_row_and_counts_them(done):
    data = _read(done, "data.parquet")
    assert data[DUPLICATES].dtype == pl.Int64
    assert set(data[DUPLICATES].unique().to_list()) == {1, 2, 3}
    features = data.drop(DUPLICATES, LABEL)
    # Only the "n/a" row and its empty twin look alike in the file: what
    # tells them apart is unparsable.parquet.
    assert data.drop(DUPLICATES).is_duplicated().sum() == 2
    # Row 51 under another class, and the row whose num_1 read "n/a" with
    # its twin of num_1 empty: the same features in the file, two rows each.
    assert features.is_duplicated().sum() == 4


def test_clean_names_the_cell_that_was_not_a_number(done):
    data = _read(done, "data.parquet")
    cells = _read(done, "unparsable.parquet")
    assert cells.rows() == [(cells["row"][0], "num_1")]
    row = cells["row"][0]
    assert data["num_1"][row] is None
    twin = data.drop(DUPLICATES, LABEL).with_row_index("i")
    same = twin.filter(
        pl.all_horizontal(
            pl.col(c).eq_missing(twin[c][row]) for c in twin.columns if c != "i"
        )
    )
    assert same.height == 2
    logged = _rows(done, f"SELECT kind FROM features WHERE column = '{DUPLICATES}'")
    assert logged == []


def test_clean_writes_where_every_csv_row_comes_from(done, synthetic_csv):
    provenance = _read(done, "provenance.parquet")
    data = _read(done, "data.parquet")
    assert provenance.columns == ["row", *KEYS, TIME]
    # One row per copy of every row of the data file.
    reads = np.bincount(provenance["row"].to_numpy(), minlength=data.height)
    assert np.array_equal(reads, data[DUPLICATES].to_numpy())
    # In reading order: the rows of the CSV of the kept classes.
    classes = json.loads((done.output_dir / "classes.json").read_text())
    csv = pl.read_csv(synthetic_csv, infer_schema=False).with_row_index("line")
    kept = csv.filter(pl.col("label").is_in(classes))
    assert provenance.height == kept.height
    for key in KEYS:
        assert provenance[key].cast(pl.String).to_list() == kept[key].to_list()
    assert provenance[TIME][0] == datetime.fromisoformat(kept["timestamp"][0])
    # A copy is the row of its first copy, with its own source and time.
    at = {line: i for i, line in enumerate(kept["line"].to_list())}
    generated = csv.height - N_DUPLICATES - 2
    copied = [i for i in range(50) if i in at]
    assert copied
    for i in copied:
        original, copy = at[i], at[generated + i]
        assert provenance["row"][original] == provenance["row"][copy]
        assert provenance["src_ip"][copy] == COPY_SRC
        assert provenance[TIME][copy] == datetime.fromisoformat(COPY_TIME)
    logged = {
        r[0]: r[1:]
        for r in _rows(
            done,
            "SELECT key, column, n_read, n_empty, n_unparsable, n_values "
            "FROM provenance_columns",
        )
    }
    assert list(logged) == [*KEYS, TIME]
    n_src = kept["src_ip"].drop_nulls().n_unique()
    assert logged["src_ip"] == ("src_ip", kept.height, 1, 0, n_src)
    for key in ("src_port", "dst_ip", "dst_port"):
        assert logged[key][1:4] == (kept.height, 0, 0)
    # The timestamp that is no time.
    assert logged[TIME] == ("timestamp", kept.height, 0, 1, None)
    assert provenance[TIME].null_count() == 1


def test_clean_without_provenance_writes_none(copy_of_done):
    raw = yaml.safe_load(copy_of_done.dataset.read_text())
    del raw["provenance"]
    copy_of_done.dataset.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert (copy_of_done.output_dir / "provenance.parquet").exists()
    copy_of_done.run("clean")
    assert not (copy_of_done.output_dir / "provenance.parquet").exists()
    (run,) = _rows(copy_of_done, "SELECT max(id) FROM runs WHERE stage = 'clean'")[0]
    logged = _rows(
        copy_of_done, f"SELECT count(*) FROM provenance_columns WHERE run_id = {run}"
    )
    assert logged == [(0,)]


def test_prepare_has_one_finite_row_per_data_row(done):
    data = _read(done, "data.parquet")
    prepared = _read(done, "prepared.parquet")
    assert prepared.height == data.height
    assert prepared[LABEL].to_list() == data[LABEL].to_list()
    features = prepared.drop(LABEL)
    assert set(features.dtypes) == {pl.Float64}
    assert np.isfinite(features.to_numpy()).all()
    numbers = [c for c in data.columns if c.startswith("num_")]
    assert DUPLICATES not in features.columns
    assert features.columns[: len(numbers) + 1] == [*numbers, MISSING]
    numeric = data.select(numbers).to_numpy().astype(float)
    n_missing = (~np.isfinite(numeric)).sum(axis=1)
    assert features[MISSING].to_list() == n_missing.astype(float).tolist()
    assert n_missing.sum() > 0
    slots = features.columns[len(numbers) + 1 :]
    assert {slot.split("=")[0] for slot in slots} == {"cat_1", "cat_2"}
    # Every row sets exactly one slot of each categorical column, all to
    # the column's weight.
    for column in ("cat_1", "cat_2"):
        block = features.select(pl.selectors.starts_with(f"{column}=")).to_numpy()
        assert (block > 0).sum(axis=1).tolist() == [1] * data.height
        assert len(np.unique(block[block > 0])) == 1


def test_prepare_logs_every_column(done):
    prepared = _read(done, "prepared.parquet").drop(LABEL)
    logged = _rows(
        done,
        "SELECT column, source, kind, mean, std, value, n_rows, weight "
        "FROM prepared_columns",
    )
    assert [row[0] for row in logged] == prepared.columns
    height = prepared.height
    for column, source, kind, mean, std, value, n_rows, weight in logged:
        if kind == "num":
            assert column == source and std is not None and mean is not None
        elif kind == "missing":
            assert column == source == MISSING
            assert n_rows == int((prepared[MISSING] > 0).sum())
        else:
            assert column == f"{source}={value}"
            assert n_rows == int((prepared[column] > 0).sum())
            assert prepared[column].max() == pytest.approx(weight)
    for source in ("cat_1", "cat_2"):
        counts = [r[6] for r in logged if r[1] == source]
        assert sum(counts) == height
    others = {row[1] for row in logged if row[5] == OTHER}
    assert others == {"cat_2"}
    n_rows, n_num, n_cat, n_columns, n_missing = _rows(
        done,
        "SELECT n_rows, n_num, n_cat, n_columns, n_missing FROM prepare_summary",
    )[0]
    assert (n_rows, n_cat, n_columns) == (height, 2, len(prepared.columns))
    assert n_missing == prepared[MISSING].sum()
    assert n_num == sum(row[2] == "num" for row in logged)


def test_the_points_leave_missing_out(done):
    cfg = load_config(str(done.dataset), method=str(done.method))
    points, labels = read_points(cfg)
    prepared = _read(done, "prepared.parquet")
    without = prepared.drop(MISSING, LABEL)
    assert points.shape == (prepared.height, without.width)
    np.testing.assert_array_equal(points, without.to_numpy())
    assert labels.tolist() == prepared[LABEL].to_list()


def _cells(workspace: Workspace) -> pl.DataFrame:
    return _read(workspace, "cells.parquet").sort("cell")


def _classes(workspace: Workspace) -> list[str]:
    return json.loads((workspace.output_dir / "classes.json").read_text())


def test_assignments_cover_every_row_and_keep_its_class(done):
    assignments = _read(done, "assignments.parquet").sort("row")
    cells = _cells(done)
    data = _read(done, "data.parquet")
    assert assignments.columns == ["row", "cell"]
    assert assignments["row"].to_list() == list(range(data.height))
    assert cells["cell"].to_list() == list(range(cells.height))
    cell_class = dict(zip(cells["cell"].to_list(), cells["class"].to_list()))
    labels = data[LABEL].to_list()
    assert [cell_class[c] for c in assignments["cell"].to_list()] == labels
    assert cells["n_rows"].sum() == data.height


def test_cells_count_the_rows_of_the_csv_and_keep_their_size(done):
    method = load_config(str(done.dataset), method=str(done.method)).method
    cells = _cells(done)
    data = _read(done, "data.parquet")
    read = dict(_rows(done, "SELECT label, n_read FROM class_counts"))
    classes = _classes(done)
    per_class = cells.group_by("class").agg(pl.col("n_read").sum())
    for code, n_read in per_class.iter_rows():
        assert n_read == read[classes[code]]
    assert cells["n_read"].sum() == data[DUPLICATES].sum()
    assert cells["n_rows"].min() >= method.min_rows
    # under 2 · min_rows, but for the rows of the small branches left beside it
    assert cells["n_rows"].median() < 2 * method.min_rows
    logged = _rows(done, "SELECT cell, class, n_rows, n_read FROM cells ORDER BY cell")
    assert [r[0] for r in logged] == cells["cell"].to_list()
    assert [classes.index(r[1]) for r in logged] == cells["class"].to_list()
    assert [r[2] for r in logged] == cells["n_rows"].to_list()
    by_class = _rows(done, "SELECT class, n_rows, n_read, n_cells FROM class_cells")
    assert sorted(r[0] for r in by_class) == sorted(classes)
    assert sum(r[3] for r in by_class) == cells.height
    assert all(r[3] >= 1 for r in by_class)
    ((n_cells, min_rows),) = _rows(done, "SELECT n_cells, min_rows FROM cell_stats")
    assert n_cells == cells.height and min_rows == method.min_rows


def _tree(workspace: Workspace) -> pl.DataFrame:
    columns = ["node", "class", "parent", "depth", "cell", "n_rows", "n_read"]
    return pl.DataFrame(
        _rows(workspace, f"SELECT {', '.join(columns)} FROM cell_tree ORDER BY node"),
        schema=columns,
        orient="row",
    )


def test_cell_tree_counts_the_rows_below_each_node(done):
    cells = _cells(done)
    classes = _classes(done)
    data = _read(done, "data.parquet")
    nodes = _tree(done)
    roots = nodes.filter(pl.col("parent") == -1)
    assert sorted(roots["class"].to_list()) == sorted(classes)
    assert (roots["depth"] == 0).all()
    class_rows = data.group_by(LABEL).len()
    expected = {classes[code]: n for code, n in class_rows.iter_rows()}
    assert dict(zip(roots["class"], roots["n_rows"])) == expected
    leaves = nodes.filter(pl.col("cell").is_not_null()).sort("cell")
    assert leaves["cell"].to_list() == cells["cell"].to_list()
    assert leaves["n_rows"].to_list() == cells["n_rows"].to_list()
    np.testing.assert_allclose(leaves["n_read"], cells["n_read"])
    # the cells come in pre-order: by node, as by cell
    assert leaves.sort("node")["cell"].to_list() == leaves["cell"].to_list()
    by_node = {row["node"]: row for row in nodes.iter_rows(named=True)}
    children = (
        nodes.filter(pl.col("parent") >= 0)
        .group_by("parent")
        .agg(pl.col("n_rows").sum(), pl.col("n_read").sum(), pl.len())
    )
    for parent, n_rows, n_read, n_children in children.iter_rows():
        node = by_node[parent]
        assert node["cell"] is None and n_children == 2
        assert node["n_rows"] == n_rows and node["n_read"] == pytest.approx(n_read)
    for row in nodes.filter(pl.col("parent") >= 0).iter_rows(named=True):
        parent = by_node[row["parent"]]
        assert parent["class"] == row["class"] and parent["node"] < row["node"]
        assert row["depth"] == parent["depth"] + 1
    # the file the report merges on is the same tree
    tree = _read(done, "cell_tree.parquet").sort("node")
    assert tree.columns == ["node", "class", "parent", "cell"]
    assert tree["node"].to_list() == nodes["node"].to_list()
    assert tree["parent"].to_list() == nodes["parent"].to_list()
    assert tree["cell"].to_list() == nodes["cell"].to_list()
    assert [classes[c] for c in tree["class"]] == nodes["class"].to_list()


def test_a_stage_without_its_input_fails_and_is_logged(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, synthetic_csv)
    with pytest.raises(FileNotFoundError, match="run clean first"):
        workspace.run("prepare")
    workspace.run("clean")
    with pytest.raises(FileNotFoundError, match="run prepare first"):
        workspace.run("cells")
    assert _rows(workspace, "SELECT stage, status FROM runs") == [
        ("prepare", "failed"),
        ("clean", "ok"),
        ("cells", "failed"),
    ]


def test_clean_refuses_rows_without_a_label(tmp_path, synthetic_csv):
    csv = tmp_path / "unlabelled.csv"
    rows = pl.read_csv(synthetic_csv, infer_schema=False)
    unlabelled = pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise("label")
    rows.with_columns(unlabelled.alias("label")).write_csv(csv)
    workspace = make_workspace(tmp_path, csv)
    with pytest.raises(ValueError, match="2 rows .* have no label"):
        workspace.run("clean")


def _table(workspace: Workspace, table: str) -> pl.DataFrame:
    with sqlite3.connect(workspace.db_path) as conn:
        cursor = conn.execute(
            f"SELECT * FROM {table} WHERE run_id = (SELECT max(id) FROM runs "
            "WHERE stage = 'quality_report' AND status = 'ok')"
        )
        names = [d[0] for d in cursor.description]
        return pl.DataFrame(cursor.fetchall(), schema=names, orient="row")


def test_quality_report_runs_ok(reported):
    runs = _rows(reported, "SELECT status FROM runs WHERE stage = 'quality_report'")
    assert runs[-1] == ("ok",)


def test_quality_classes_count_the_rows_of_every_class(reported):
    classes = _table(reported, "quality_classes").sort("class")
    counts = pl.DataFrame(
        _rows(reported, "SELECT label, n_read, n_rows FROM class_counts"),
        schema=["class", "n_read", "n_rows"],
        orient="row",
    ).sort("class")
    kept = counts.filter(pl.col("class").is_in(classes["class"].to_list()))
    assert classes["class"].to_list() == kept["class"].to_list()
    assert classes["n_rows"].to_list() == kept["n_rows"].to_list()
    assert classes["n_read"].to_list() == pytest.approx(kept["n_read"].to_list())
    assert classes["n_cells"].sum() == _table(reported, "quality_cells").height
    assert classes["n_regions"].sum() == _table(reported, "quality_regions").height
    assert "dup_share" not in classes.columns


def test_quality_classes_share_the_csv_rows_with_each_defect(reported):
    classes = _table(reported, "quality_classes").sort("class")
    cells = _table(reported, "quality_cells").with_columns(
        of_cell=pl.col("defects")
        .fill_null("")
        .str.contains("|".join(RIVAL_DEFECTS + CLASS_DEFECTS))
    )
    has = lambda d: pl.col("defects").fill_null("").str.contains(d)  # noqa: E731
    by_hand = (
        cells.group_by("class")
        .agg(
            n_read=pl.col("n_read").sum(),
            invalid_values=pl.col("missing_read").sum(),
            duplicates=pl.col("n_duplicates").sum(),
            label_conflict=pl.col("conflict_read").sum(),
            **{
                d: pl.col("n_read").filter(has(d)).sum()
                for d in RIVAL_DEFECTS + CLASS_DEFECTS
            },
            # a row with no defect: a first copy, with no null value and in no
            # conflict, in a cell without a defect of cell
            lo=pl.when(pl.col("of_cell"))
            .then(0)
            .otherwise(
                pl.col("n_rows") - pl.col("missing_rows") - pl.col("conflict_rows")
            )
            .sum(),
            hi=pl.when(pl.col("of_cell"))
            .then(0)
            .otherwise(
                pl.col("n_rows") - pl.max_horizontal("missing_rows", "conflict_rows")
            )
            .sum(),
        )
        .sort("class")
    )
    for defect in DEFECTS:
        assert classes[f"share_{defect}"].to_list() == pytest.approx(
            (by_hand[defect] / by_hand["n_read"]).to_list()
        )
    clean = (classes["share_no_defect"] * classes["n_read"]).to_numpy()
    assert (by_hand["lo"].to_numpy() - 1e-6 <= clean).all()
    assert (clean <= by_hand["hi"].to_numpy() + 1e-6).all()
    assert (classes["share_duplicates"] > 0).any()
    assert (classes["share_label_conflict"] > 0).any()
    assert (classes["share_invalid_values"] > 0).any()


def test_quality_stats_share_the_csv_rows_of_every_class(reported):
    stats = _table(reported, "quality_stats").row(0, named=True)
    classes = _table(reported, "quality_classes")
    weight = classes["n_read"] / classes["n_read"].sum()
    for name in (*DEFECTS, "no_defect"):
        assert stats[f"share_{name}"] == pytest.approx(
            (classes[f"share_{name}"] * weight).sum()
        )


def _codes(names: str | None, classes: list[str]) -> tuple[int, ...]:
    """The codes of the class names joined by ","."""
    return tuple(classes.index(n) for n in names.split(",")) if names else ()


def test_quality_cells_judge_every_cell(reported):
    q = load_config(str(reported.dataset), method=str(reported.method)).method.quality
    tau, confidence = q.tau, q.confidence
    classes = _classes(reported)
    cells = _table(reported, "quality_cells").sort("cell")
    measured = _table(reported, "quality_rivals")
    parquet = _cells(reported)
    assert cells["cell"].to_list() == parquet["cell"].to_list()
    assert cells["n_rows"].to_list() == parquet["n_rows"].to_list()
    assert cells["n_duplicates"].to_list() == pytest.approx(
        (parquet["n_read"] - parquet["n_rows"]).to_list()
    )
    assert cells["dup_share"].to_list() == pytest.approx(
        (1 - parquet["n_rows"] / parquet["n_read"]).to_list()
    )
    for r in cells.iter_rows(named=True):
        own = measured.filter(pl.col("cell") == r["cell"])
        assert own.height > 0
        assert r["n_rivals"] == sum(len(json.loads(c)) for c in own["rival_cells"])
        rivals = []
        for p in own.iter_rows(named=True):
            assert p["region"] == r["region"]
            if p["confusion"] is None:
                continue
            steps = [p["confusion"], *json.loads(p["confusion_without"])]
            k = classes.index(p["rival_class"])
            rivals.append((k, np.array(steps), p["n_scored"]))
            if p["rival_class"] == r["rival_class"]:
                assert r["confusion"] == pytest.approx(p["confusion"])
                assert r["confusion_low"] <= r["confusion"] <= r["confusion_high"]
        assert r["confusion"] == pytest.approx(max(e[0] for _, e, _ in rivals))
        # the classes of each rival defect: those of a True verdict, and
        # class_overlap only where it is no suspect_label
        found = rival_defects(rivals, tau, confidence)
        yes = lambda d: {k for (e, k), v in found.items() if e == d and v}  # noqa: E731
        assert set(_codes(r["label_with"], classes)) == yes("suspect_label")
        assert set(_codes(r["overlap_with"], classes)) == yes("class_overlap") - yes(
            "suspect_label"
        )
        assert set(_codes(r["shortcut_with"], classes)) == yes("shortcut")
        redundant = redundancy_verdict(
            r["variance"],
            r["variance_sd"],
            r["n_rows"],
            r["variance_typical"],
            tau,
            confidence,
        )
        keys = {
            key
            for key in PROVENANCE
            if r[f"{key}_excess"] is not None
            and provenance_verdict(
                r[f"{key}_excess_share"],
                r[f"{key}_excess"],
                r[f"{key}_n"],
                tau,
                confidence,
            )
        }
        assert set(_listed(r["provenance_with"])) == keys
        established = {
            "suspect_label": r["label_with"],
            "class_overlap": r["overlap_with"],
            "shortcut": r["shortcut_with"],
            "redundancy": redundant,
            "atypical_provenance": keys,
        }
        defects = defects_of(
            r["missing_rows"],
            r["n_rows"],
            r["n_read"],
            r["conflict_rows"],
            {d for d, held in established.items() if held},
        )
        assert r["defects"] == (",".join(defects) or None)
        assert not set(_listed(r["undecided"])) & set(defects)
        assert r["missing_rows"] >= max(r[f"{kind}_rows"] for kind in KINDS)


def _listed(value: str | None) -> list[str]:
    return value.split(",") if value else []


def test_quality_regions_are_branches_of_the_cut_with_the_defects_of_their_cells(
    reported,
):
    classes = _classes(reported)
    cells = _table(reported, "quality_cells").sort("cell")
    regions = _table(reported, "quality_regions").sort("region")
    tree = _read(reported, "cell_tree.parquet").sort("node")
    parent = dict(zip(tree["node"].to_list(), tree["parent"].to_list()))
    node_of = dict(zip(tree["cell"].to_list(), tree["node"].to_list()))
    below = {}  # the cells under every node
    for cell, node in node_of.items():
        if cell is None:
            continue
        while node >= 0:
            below.setdefault(node, set()).add(cell)
            node = parent[node]
    assert regions["region"].to_list() == sorted(set(cells["region"]))
    for r in regions.iter_rows(named=True):
        own = cells.filter(pl.col("region") == r["region"])
        # a whole branch, named by the node at its top
        assert set(own["cell"]) == below[r["region"]]
        assert r["n_cells"] == own.height
        for column in ("n_rows", "conflict_rows", "missing_rows"):
            assert r[column] == own[column].sum()
        assert r["n_read"] == pytest.approx(own["n_read"].sum())
        assert r["confusion"] == pytest.approx(own["confusion"].max())
        # the defects its cells have, against the classes or keys they have them
        for key in ("label_with", "overlap_with", "shortcut_with", "provenance_with"):
            union = {n for v in own[key] for n in _listed(v)}
            assert set(_listed(r[key])) == union
        held = {d for v in own["defects"] for d in _listed(v)}
        defects = defects_of(
            r["missing_rows"],
            r["n_rows"],
            r["n_read"],
            r["conflict_rows"],
            held & set(RIVAL_DEFECTS + CLASS_DEFECTS),
        )
        assert r["defects"] == (",".join(defects) or None)
        assert set(_listed(r["label_with"])) <= set(classes)


def _bad_cells(series: pl.Series) -> int:
    n = series.is_null().sum()
    if series.dtype.is_float():
        n += (series.is_nan() | series.is_infinite()).sum()
    return n


def test_cell_missing_counts_every_null_or_invalid_cell(reported):
    data = _read(reported, "data.parquet").drop(DUPLICATES, LABEL)
    missing = _table(reported, "cell_missing")
    cells = _table(reported, "quality_cells")
    bad = sum(_bad_cells(data[c]) for c in data.columns)
    assert missing["n_rows"].sum() == bad > 0
    assert {"empty", "unparsable", "inf"} <= set(missing["kind"]) <= set(KINDS)
    (unparsable,) = missing.filter(pl.col("kind") == "unparsable").rows(named=True)
    assert unparsable["column"] == "num_1" and unparsable["n_rows"] == 1
    region_of = dict(zip(cells["cell"].to_list(), cells["region"].to_list()))
    assert all(
        region_of[c] == r for c, r in missing.select("cell", "region").iter_rows()
    )
    assert 0 < cells["missing_rows"].sum() <= bad
    for table in (cells, _table(reported, "quality_regions")):
        flagged = table.filter(pl.col("missing_rows") > 0)
        assert flagged["defects"].str.contains("invalid_values").all()


def test_the_relabelled_row_is_a_label_conflict(reported):
    groups = _table(reported, "label_conflict_groups")
    n_rows = _read(reported, "data.parquet").height
    of_last = groups.filter(pl.col("row") == n_rows - 1)["group"].to_list()
    assert len(of_last) == 1
    group = groups.filter(pl.col("group") == of_last[0])
    assert group.height == 2 and group["class"].n_unique() == 2
    for table, key in (
        (_table(reported, "quality_cells"), "cell"),
        (_table(reported, "quality_regions"), "region"),
    ):
        for unit, own in group.select(key, "class").iter_rows():
            (row,) = table.filter(pl.col(key) == unit).rows(named=True)
            assert row["conflict_rows"] >= 1 and "label_conflict" in row["defects"]
            other = set(group["class"]) - {own}
            assert other <= set(row["conflict_with"].split(","))


def test_the_copies_are_duplicates(reported):
    for table in (
        _table(reported, "quality_cells"),
        _table(reported, "quality_regions"),
    ):
        copied = table.filter(pl.col("n_read") > pl.col("n_rows"))
        assert copied.height > 0
        assert copied["defects"].str.contains("duplicates").all()
        alone = table.filter(pl.col("n_read") == pl.col("n_rows"))
        assert not alone["defects"].fill_null("").str.contains("duplicates").any()


def test_quality_rivals_measure_every_cell_against_its_rival_classes(reported):
    cfg = load_config(str(reported.dataset), method=str(reported.method))
    quality = cfg.method.quality
    rivals = _table(reported, "quality_rivals")
    cells = _cells(reported)
    classes = _classes(reported)
    class_of = dict(zip(cells["cell"].to_list(), cells["class"].to_list()))
    size = dict(zip(cells["cell"].to_list(), cells["n_rows"].to_list()))
    assert rivals.height and (rivals["class"] != rivals["rival_class"]).all()
    assert rivals.select("cell", "rival_class").is_unique().all()
    assert set(rivals["cell"]) == set(cells["cell"])
    columns = set(cfg.num_cols + cfg.cat_cols)
    for r in rivals.iter_rows(named=True):
        # the cell whole, its rival cells of one class pooled and capped
        others = json.loads(r["rival_cells"])
        assert others == sorted(others) and others
        assert {classes[class_of[o]] for o in others} == {r["rival_class"]}
        n_a, n_b = size[r["cell"]], sum(size[o] for o in others)
        assert r["n_measured"] == n_a
        assert r["n_rival_measured"] == min(n_b, quality.max_rows_per_cell)
        # so a measure is decidable, as min_rows promises
        assert r["n_scored"] >= decisive_rows(quality.tau, quality.confidence)
        assert 0 <= r["bayes_error_low"] <= r["nn_error"] <= 1
        assert 0 <= r["single_column_acc"] <= 1 and r["single_column"] in columns
        removed = r["key_columns"].split(",")
        assert set(removed) <= columns and len(removed) == quality.max_group
        without = json.loads(r["confusion_without"])
        assert len(without) == len(removed) and all(0 <= e <= 1 for e in without)
        assert 0 <= r["confusion"] <= 1


def test_quality_report_refuses_cells_of_other_data(copy_of_done):
    prepared_file = copy_of_done.output_dir / "prepared.parquet"
    prepared = pl.read_parquet(prepared_file)
    shuffled = np.random.default_rng(0).permutation(prepared[LABEL].to_numpy())
    prepared.with_columns(pl.Series(LABEL, shuffled)).write_parquet(prepared_file)
    with pytest.raises(ValueError, match="run cells again"):
        copy_of_done.run("quality_report")


def _central(times: list, share: Fraction) -> tuple:
    """The first and last time left once k = floor((n - 1) (1 - share) / 2) are cut at each end."""
    times = sorted(times)
    k = int((len(times) - 1) * (1 - share) / 2)
    return times[k], times[len(times) - 1 - k]


def test_quality_cells_measure_where_their_csv_rows_come_from(reported):
    cfg = load_config(str(reported.dataset), method=str(reported.method))
    held = Fraction(str(cfg.method.quality.time_share))
    cells = _table(reported, "quality_cells").sort("cell")
    provenance = _read(reported, "provenance.parquet")
    cell = _read(reported, "assignments.parquet").sort("row")["cell"].to_numpy()
    of = cell[provenance["row"].to_numpy()]
    shares = {key: [] for key in KEYS}  # per key, per cell: {value: its share}
    for c, row in enumerate(cells.iter_rows(named=True)):
        assert row["cell"] == c
        mine = provenance.filter(pl.Series(of == c))
        for key in KEYS:
            values = mine[key].cast(pl.String).drop_nulls().to_list()
            counts = Counter(values)
            top = min(counts, key=lambda v: (-counts[v], v))
            p = np.array(list(counts.values())) / len(values)
            assert row[f"{key}_top"] == top
            assert row[f"{key}_top_share"] == pytest.approx(counts[top] / len(values))
            assert row[f"{key}_n_values"] == len(counts)
            assert row[f"{key}_eff_n"] == pytest.approx(np.exp(-(p * np.log(p)).sum()))
            shares[key].append({v: Fraction(n, len(values)) for v, n in counts.items()})
        first, last = _central(mine[TIME].drop_nulls().to_list(), held)
        assert row["time_from"] == first.strftime("%Y-%m-%d %H:%M:%S")
        assert row["time_to"] == last.strftime("%Y-%m-%d %H:%M:%S")
        assert row["time_window"] == pytest.approx((last - first).total_seconds())
    # The excess: a cell's share of one of its values less the typical
    # share of that value in its class, the ceil(n / 2)-th smallest of its
    # shares in the class's n cells (0 in a cell without it).
    of_class = cells["class"].to_list()
    for key in KEYS:
        for c, row in enumerate(cells.iter_rows(named=True)):
            mates = [
                shares[key][o]
                for o in range(cells.height)
                if of_class[o] == of_class[c]
            ]
            over = {}
            for v, share in shares[key][c].items():
                typical = sorted(d.get(v, Fraction(0)) for d in mates)
                over[v] = share - typical[math.ceil(len(mates) / 2) - 1]
            best = max(over.values())
            assert row[f"{key}_excess"] == pytest.approx(float(best))
            assert over[row[f"{key}_excess_value"]] == best
    # The time window as a share of the window of the class's CSV rows.
    for name in set(of_class):
        own = [c for c in range(cells.height) if of_class[c] == name]
        times = provenance.filter(pl.Series(np.isin(of, own)))[TIME]
        first, last = _central(times.drop_nulls().to_list(), held)
        whole = (last - first).total_seconds()
        for c in own:
            share = cells["time_window_share"][c]
            if whole > 0:
                assert share == pytest.approx(cells["time_window"][c] / whole)
            else:
                assert share is None


def test_the_planted_provenance_shows_in_its_cells(reported):
    cells = _table(reported, "quality_cells")
    # class_5 comes from one host: every cell of it, but for the copies.
    single = cells.filter(pl.col("class") == "class_5")
    assert set(single["src_ip_top"]) == {"18.5.0.99"}
    assert (single["src_ip_top_share"] > 0.9).all()
    # The canonical sub-group of class_4: one host, one port, 5 minutes.
    planted = cells.filter(pl.col("class") == "class_4").sort(
        "src_ip_top_share", descending=True
    )
    first = planted.row(0, named=True)
    assert first["src_ip_top"] == "18.99.9.9" and first["src_port_top"] == "31337"
    assert first["src_ip_top_share"] > 0.9 and first["time_window"] <= 300
    # The background has many hosts in every cell.
    background = cells.filter(pl.col("class") == "class_1")
    assert (background["src_ip_eff_n"] > 10).all()


def test_the_planted_source_is_atypical_for_its_class(reported):
    cells = _table(reported, "quality_cells")
    by_host = cells.filter(
        pl.col("provenance_with").fill_null("").str.contains("src_ip")
    )
    # The canonical sub-group of class_4 comes from a host the rest of
    # its class never has: every cell of it, and only of it, in class_4.
    planted = cells.filter(
        (pl.col("class") == "class_4") & (pl.col("src_ip_top") == "18.99.9.9")
    )
    assert planted.height and (planted["src_ip_top_share"] > 0.9).all()
    assert set(planted["cell"]) == set(
        by_host.filter(pl.col("class") == "class_4")["cell"]
    )
    assert planted["defects"].str.contains("atypical_provenance").all()
    assert set(planted["src_ip_excess_value"]) == {"18.99.9.9"}
    assert (planted["src_ip_excess"] > 0.9).all()
    # class_5 comes all from one host: typical of it, so no cell is
    # atypical; nor is any of the background, of many hosts.
    assert not by_host["class"].is_in(["class_1", "class_5"]).any()


def test_quality_without_a_provenance_does_not_measure_atypical_provenance(
    copy_of_done,
):
    raw = yaml.safe_load(copy_of_done.dataset.read_text())
    del raw["provenance"]
    copy_of_done.dataset.write_text(yaml.safe_dump(raw, sort_keys=False))
    copy_of_done.run("quality_report")
    stats = _table(copy_of_done, "quality_stats").row(0, named=True)
    assert stats["share_atypical_provenance"] is None
    assert stats["regions_atypical_provenance"] is None
    assert stats["share_no_defect"] is not None
    assert stats["share_redundancy"] is not None  # geometry alone
    classes = _table(copy_of_done, "quality_classes")
    assert classes["share_atypical_provenance"].is_null().all()
    for table in ("quality_cells", "quality_regions"):
        defects = _table(copy_of_done, table)["defects"].fill_null("")
        assert not defects.str.contains("atypical_provenance").any()
