"""Every stage end to end on the small synthetic dataset."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from conftest import N_DUPLICATES, STAGE_ORDER, Workspace, make_workspace

from src.cells import CellTree, merge_cells
from src.config import load_config
from src.data import DUPLICATES, KINDS, LABEL
from src.prepare import MISSING, OTHER
from src.quality import DEFECTS, defects_of, rival_defects
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
    n_raw, n_duplicate, n_rare, n_kept = _rows(
        done, "SELECT n_raw, n_duplicate, n_rare, n_kept FROM clean_summary"
    )[0]
    assert n_duplicate == N_DUPLICATES
    assert data.height == n_kept == n_raw - n_duplicate - n_rare
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
    limits = load_config(str(done.dataset), method=str(done.method)).method.cells
    cells = _cells(done)
    data = _read(done, "data.parquet")
    read = dict(_rows(done, "SELECT label, n_read FROM class_counts"))
    classes = _classes(done)
    per_class = cells.group_by("class").agg(pl.col("n_read").sum())
    for code, n_read in per_class.iter_rows():
        assert n_read == read[classes[code]]
    assert cells["n_read"].sum() == data[DUPLICATES].sum()
    assert cells["n_rows"].min() >= limits.min_rows
    # at most max_rows, and the rows of the small branches left beside it
    assert cells["n_rows"].max() <= 1.25 * limits.max_rows
    logged = _rows(done, "SELECT cell, class, n_rows, n_read FROM cells ORDER BY cell")
    assert [r[0] for r in logged] == cells["cell"].to_list()
    assert [classes.index(r[1]) for r in logged] == cells["class"].to_list()
    assert [r[2] for r in logged] == cells["n_rows"].to_list()
    by_class = _rows(done, "SELECT class, n_rows, n_read, n_cells FROM class_cells")
    assert sorted(r[0] for r in by_class) == sorted(classes)
    assert sum(r[3] for r in by_class) == cells.height
    assert all(r[3] >= 1 for r in by_class)
    ((n_cells, max_rows, min_rows),) = _rows(
        done, "SELECT n_cells, max_rows, min_rows FROM cell_stats"
    )
    assert n_cells == cells.height
    assert (max_rows, min_rows) == (limits.max_rows, limits.min_rows)


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
        rival=pl.col("defects").fill_null("").str.contains("class_overlap|shortcut")
    )
    has = lambda d: pl.col("defects").fill_null("").str.contains(d)  # noqa: E731
    by_hand = (
        cells.group_by("class")
        .agg(
            n_read=pl.col("n_read").sum(),
            invalid_values=pl.col("missing_read").sum(),
            duplicates=pl.col("n_duplicates").sum(),
            label_conflict=pl.col("conflict_read").sum(),
            class_overlap=pl.col("n_read").filter(has("class_overlap")).sum(),
            shortcut=pl.col("n_read").filter(has("shortcut")).sum(),
            # a row with no defect: a first copy, with no null value and in no
            # conflict, in a cell without class_overlap or shortcut
            lo=pl.when(pl.col("rival"))
            .then(0)
            .otherwise(
                pl.col("n_rows") - pl.col("missing_rows") - pl.col("conflict_rows")
            )
            .sum(),
            hi=pl.when(pl.col("rival"))
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
    tau = load_config(
        str(reported.dataset), method=str(reported.method)
    ).method.quality.tau
    classes = _classes(reported)
    cells = _table(reported, "quality_cells").sort("cell")
    pairs = _table(reported, "quality_pairs")
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
        own = pairs.filter(
            (pl.col("cell_a") == r["cell"]) | (pl.col("cell_b") == r["cell"])
        )
        assert r["n_rivals"] == own.height > 0
        rivals = []
        for p in own.iter_rows(named=True):
            side, other = ("a", "b") if p["cell_a"] == r["cell"] else ("b", "a")
            assert p[f"region_{side}"] == r["region"]
            if p[f"confusion_{side}"] is None:
                continue
            steps = [
                p[f"confusion_{side}"],
                *json.loads(p[f"confusion_without_{side}"]),
            ]
            rivals.append((classes.index(p[f"class_{other}"]), np.array(steps)))
            if p[f"cell_{other}"] == r["rival"]:
                assert r["confusion"] == pytest.approx(p[f"confusion_{side}"])
                assert r["rival_class"] == p[f"class_{other}"]
        assert r["confusion"] == pytest.approx(max(e[0] for _, e in rivals))
        overlap, shortcut = rival_defects(rivals, tau)
        assert _codes(r["overlap_with"], classes) == overlap
        assert _codes(r["shortcut_with"], classes) == shortcut
        defects = defects_of(
            r["missing_rows"],
            r["n_rows"],
            r["n_read"],
            r["conflict_rows"],
            overlap,
            shortcut,
        )
        assert r["defects"] == (",".join(defects) or None)
        if r["shortcut_rival"] is not None:
            assert r["shortcut_columns"]
            assert (r["confusion_without"] >= tau) == ("shortcut" in defects)
        assert r["missing_rows"] >= max(r[f"{kind}_rows"] for kind in KINDS)


def test_quality_regions_are_the_largest_branches_of_cells_that_agree(reported):
    classes = _classes(reported)
    cells = _table(reported, "quality_cells").sort("cell")
    regions = _table(reported, "quality_regions").sort("region")
    tree = _read(reported, "cell_tree.parquet").sort("node")
    profile = [
        (r["overlap_with"], r["shortcut_with"]) for r in cells.iter_rows(named=True)
    ]
    expected, offset = [], 0
    for code in range(len(classes)):
        part = tree.filter(pl.col("class") == code)
        if not part.height:
            continue
        first = part["node"][0]
        own = part["cell"].drop_nulls().to_numpy()
        base = int(own.min())
        cut = CellTree(
            parent=np.where(part["parent"] >= 0, part["parent"] - first, -1),
            cell=np.where(part["cell"].is_null(), -1, part["cell"].fill_null(0) - base),
        )
        merged = merge_cells(cut, profile[base : base + len(own)])
        expected += (merged + offset).tolist()
        offset += int(merged.max()) + 1
    assert cells["region"].to_list() == expected
    assert regions["region"].to_list() == list(range(offset))
    summed = (
        cells.group_by("region")
        .agg(
            pl.len().alias("n_cells"),
            *(
                pl.col(c).sum()
                for c in ("n_rows", "n_read", "conflict_rows", "missing_rows")
            ),
            pl.col("overlap_with").unique(),
            pl.col("shortcut_with").unique(),
            pl.col("confusion").max(),
        )
        .sort("region")
    )
    assert regions["n_cells"].to_list() == summed["n_cells"].to_list()
    for column in ("n_rows", "conflict_rows", "missing_rows"):
        assert regions[column].to_list() == summed[column].to_list()
    assert regions["n_read"].to_list() == pytest.approx(summed["n_read"].to_list())
    assert regions["confusion"].to_list() == pytest.approx(
        summed["confusion"].to_list()
    )
    for r, s in zip(regions.iter_rows(named=True), summed.iter_rows(named=True)):
        # the cells of a region share their classes of class_overlap and shortcut
        assert s["overlap_with"] == [r["overlap_with"]]
        assert s["shortcut_with"] == [r["shortcut_with"]]
        defects = defects_of(
            r["missing_rows"],
            r["n_rows"],
            r["n_read"],
            r["conflict_rows"],
            _codes(r["overlap_with"], classes),
            _codes(r["shortcut_with"], classes),
        )
        assert r["defects"] == (",".join(defects) or None)


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


def test_quality_pairs_are_the_rivals_of_every_cell(reported):
    cfg = load_config(str(reported.dataset), method=str(reported.method))
    quality = cfg.method.quality
    pairs = _table(reported, "quality_pairs")
    cells = _cells(reported)
    assert pairs.height and (pairs["class_a"] != pairs["class_b"]).all()
    assert (pairs["cell_a"] < pairs["cell_b"]).all()
    assert pairs.select("cell_a", "cell_b").is_unique().all()
    measured = set(pairs["cell_a"]) | set(pairs["cell_b"])
    assert measured == set(cells["cell"])
    columns = set(cfg.num_cols + cfg.cat_cols)
    for p in pairs.iter_rows(named=True):
        for key in ("a", "b", ""):
            assert 0 <= p[f"nn_error_{key}".rstrip("_")] <= 1
        assert 0 <= p["bayes_error_low"] <= p["nn_error"]
        assert p["nn_error"] == pytest.approx((p["nn_error_a"] + p["nn_error_b"]) / 2)
        assert 0.5 <= p["single_column_acc"] <= 1 and p["single_column"] in columns
        removed = p["key_columns"].split(",")
        assert set(removed) <= columns and len(removed) == quality.max_group
        one = (1 - p["single_column_acc_a"], 1 - p["single_column_acc_b"])
        nearest = (p["nn_error_a"], p["nn_error_b"])
        assert (p["confusion_a"], p["confusion_b"]) in (one, nearest)
        for side in ("a", "b"):
            without = json.loads(p[f"confusion_without_{side}"])
            assert len(without) == len(removed)
            assert all(0 <= e <= 1 for e in without)


def test_quality_report_refuses_cells_of_other_data(copy_of_done):
    prepared_file = copy_of_done.output_dir / "prepared.parquet"
    prepared = pl.read_parquet(prepared_file)
    shuffled = np.random.default_rng(0).permutation(prepared[LABEL].to_numpy())
    prepared.with_columns(pl.Series(LABEL, shuffled)).write_parquet(prepared_file)
    with pytest.raises(ValueError, match="run cells again"):
        copy_of_done.run("quality_report")
