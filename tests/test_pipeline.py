"""Every stage end to end on the small synthetic dataset."""

import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from conftest import N_DUPLICATES, STAGE_ORDER, Workspace, make_workspace

from src.config import load_config
from src.data import DUPLICATES, KINDS, LABEL
from src.prepare import MISSING, OTHER
from src.quality import ISSUES, VERDICTS, region_issues
from stages.paths import read_points, read_weights


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


def test_assignments_cover_every_row_and_keep_its_class(done):
    assignments = _read(done, "assignments.parquet").sort("row")
    regions = _read(done, "regions.parquet")
    data = _read(done, "data.parquet")
    assert assignments["row"].to_list() == list(range(data.height))
    region_class = dict(zip(regions["region"].to_list(), regions["class"].to_list()))
    labels = data[LABEL].to_list()
    assert [region_class[r] for r in assignments["region"].to_list()] == labels
    assert regions["n_rows"].sum() == data.height


def test_regions_count_the_rows_of_the_csv(done):
    regions = _read(done, "regions.parquet")
    data = _read(done, "data.parquet")
    read = dict(_rows(done, "SELECT label, n_read FROM class_counts"))
    classes = json.loads((done.output_dir / "classes.json").read_text())
    per_class = regions.group_by("class").agg(pl.col("n_read").sum())
    for code, n_read in per_class.iter_rows():
        assert n_read == read[classes[code]]
    assert regions["n_read"].sum() == data[DUPLICATES].sum()
    stability = [row[0] for row in _rows(done, "SELECT stability FROM regions")]
    assert len(stability) == regions.height
    assert all(0 <= value <= 1 for value in stability)
    logged = _rows(
        done,
        "SELECT class, n_rows, n_read, n_regions, n_pieces, n_voted, "
        "split_separation, stability FROM class_clusterings",
    )
    method = load_config(str(done.dataset), method=str(done.method)).method
    assert {row[0] for row in logged} == set(classes)
    assert sum(row[3] for row in logged) == regions.height
    assert all(row[4] >= row[3] and row[5] >= 0 for row in logged)
    assert all(row[6] >= method.clustering.split_separation for row in logged)
    assert all(0 <= row[7] <= 1 for row in logged)
    curve = _rows(
        done,
        "SELECT class, threshold, n_regions FROM granularity_candidates "
        "ORDER BY class, threshold",
    )
    assert {row[0] for row in curve} == set(classes)
    for name in classes:
        counts = [row[2] for row in curve if row[0] == name]
        assert counts == sorted(counts, reverse=True)


def test_region_tree_counts_the_rows_below_each_node(done):
    regions = _read(done, "regions.parquet")
    classes = json.loads((done.output_dir / "classes.json").read_text())
    data = _read(done, "data.parquet")
    nodes = pl.DataFrame(
        _rows(
            done,
            "SELECT node, class, parent, depth, region, n_rows, n_read "
            "FROM region_tree ORDER BY node",
        ),
        schema=["node", "class", "parent", "depth", "region", "n_rows", "n_read"],
        orient="row",
    )
    roots = nodes.filter(pl.col("parent") == -1)
    assert sorted(roots["class"].to_list()) == sorted(classes)
    assert (roots["depth"] == 0).all()
    class_rows = data.group_by(LABEL).len()
    expected = {classes[code]: n for code, n in class_rows.iter_rows()}
    assert dict(zip(roots["class"], roots["n_rows"])) == expected
    leaves = nodes.filter(pl.col("region").is_not_null()).sort("region")
    assert leaves["region"].to_list() == regions["region"].to_list()
    assert leaves["n_rows"].to_list() == regions["n_rows"].to_list()
    np.testing.assert_allclose(leaves["n_read"], regions["n_read"])
    by_node = {row["node"]: row for row in nodes.iter_rows(named=True)}
    children = (
        nodes.filter(pl.col("parent") >= 0)
        .group_by("parent")
        .agg(pl.col("n_rows").sum(), pl.col("n_read").sum(), pl.len())
    )
    for parent, n_rows, n_read, n_children in children.iter_rows():
        node = by_node[parent]
        assert node["region"] is None and n_children == 2
        assert node["n_rows"] == n_rows and node["n_read"] == pytest.approx(n_read)
    for row in nodes.filter(pl.col("parent") >= 0).iter_rows(named=True):
        parent = by_node[row["parent"]]
        assert parent["class"] == row["class"] and parent["node"] < row["node"]
        assert row["depth"] == parent["depth"] + 1


def _zones(done) -> pl.DataFrame:
    columns = [
        "node",
        "zone",
        "parent",
        "depth",
        "separation",
        "region_a",
        "region_b",
        "region",
        "class",
        "n_classes",
        "n_regions",
        "n_rows",
        "n_read",
    ]
    return pl.DataFrame(
        _rows(done, f"SELECT {', '.join(columns)} FROM overlap_zones ORDER BY node"),
        schema=columns,
        orient="row",
    )


def test_overlap_zones_count_the_regions_below_each_node(done):
    regions = _read(done, "regions.parquet")
    classes = json.loads((done.output_dir / "classes.json").read_text())
    ((threshold, n_zones),) = _rows(
        done, "SELECT overlap_separation, n_zones FROM region_stats"
    )
    nodes = _zones(done)
    assert nodes.height > 0, "the synthetic classes overlap somewhere"
    assert nodes["node"].to_list() == list(range(nodes.height))
    roots = nodes.filter(pl.col("parent") == -1)
    assert roots["zone"].to_list() == list(range(n_zones))
    assert (roots["depth"] == 0).all()
    leaves = nodes.filter(pl.col("region").is_not_null())
    by_region = {row["region"]: row for row in regions.iter_rows(named=True)}
    for leaf in leaves.iter_rows(named=True):
        region = by_region[leaf["region"]]
        assert leaf["class"] == classes[region["class"]]
        assert leaf["n_rows"] == region["n_rows"]
        assert leaf["n_read"] == pytest.approx(region["n_read"])
        assert leaf["n_regions"] == leaf["n_classes"] == 1
        assert leaf["separation"] is None and leaf["region_a"] is None
    assert leaves["region"].n_unique() == leaves.height
    by_node = {row["node"]: row for row in nodes.iter_rows(named=True)}
    below: dict[int, list[dict]] = {}
    for row in nodes.filter(pl.col("parent") >= 0).iter_rows(named=True):
        parent = by_node[row["parent"]]
        assert parent["node"] < row["node"] and parent["zone"] == row["zone"]
        assert row["depth"] == parent["depth"] + 1
        below.setdefault(row["parent"], []).append(row)
    for node, kids in below.items():
        row = by_node[node]
        assert len(kids) == 2 and row["region"] is None and row["class"] is None
        assert 0 <= row["separation"] < threshold
        assert all(
            k["separation"] is None or k["separation"] <= row["separation"]
            for k in kids
        )
        assert row["n_rows"] == sum(k["n_rows"] for k in kids)
        assert row["n_regions"] == sum(k["n_regions"] for k in kids)
        assert row["n_read"] == pytest.approx(sum(k["n_read"] for k in kids))
        assert row["n_classes"] >= 2
        a, b = by_region[row["region_a"]], by_region[row["region_b"]]
        assert a["class"] != b["class"]


def test_region_overlaps_are_the_pairs_under_the_threshold(done):
    cfg = load_config(str(done.dataset), method=str(done.method))
    regions = _read(done, "regions.parquet").sort("region")
    centers = np.array(regions["centroid"].to_list())
    region = _read(done, "assignments.parquet").sort("row")["region"].to_numpy()
    points, _ = read_points(cfg)
    weights = read_weights(cfg)
    classes = json.loads((done.output_dir / "classes.json").read_text())
    ((threshold,),) = _rows(done, "SELECT overlap_separation FROM region_stats")
    columns = [
        "zone",
        "region_a",
        "class_a",
        "region_b",
        "class_b",
        "distance",
        "spread_a",
        "spread_b",
        "separation",
    ]
    pairs = [
        dict(zip(columns, r))
        for r in _rows(done, f"SELECT {', '.join(columns)} FROM region_overlaps")
    ]
    assert pairs
    nodes = _zones(done).filter(pl.col("region").is_not_null())
    zone_of = dict(zip(nodes["region"].to_list(), nodes["zone"].to_list()))
    assert {p["region_a"] for p in pairs} | {p["region_b"] for p in pairs} == set(
        zone_of
    )
    for p in pairs:
        a, b = p["region_a"], p["region_b"]
        assert a < b and p["class_a"] != p["class_b"]
        assert p["class_a"] == classes[regions["class"][a]]
        assert p["class_b"] == classes[regions["class"][b]]
        assert zone_of[a] == zone_of[b] == p["zone"]
        line = centers[b] - centers[a]
        assert p["distance"] == pytest.approx(np.linalg.norm(line))
        line /= np.linalg.norm(line)
        for side, own in (("a", a), ("b", b)):
            rows = region == own
            along = (points[rows] - centers[own]) @ line
            spread = np.sqrt(np.average(along**2, weights=weights[rows]))
            assert p[f"spread_{side}"] == pytest.approx(spread, rel=1e-6)
        assert p["separation"] == pytest.approx(
            p["distance"] / np.hypot(p["spread_a"], p["spread_b"])
        )
        assert p["separation"] < threshold


def test_a_stage_without_its_input_fails_and_is_logged(tmp_path, synthetic_csv):
    workspace = make_workspace(tmp_path, synthetic_csv)
    with pytest.raises(FileNotFoundError, match="run clean first"):
        workspace.run("prepare")
    workspace.run("clean")
    with pytest.raises(FileNotFoundError, match="run prepare first"):
        workspace.run("regions")
    assert _rows(workspace, "SELECT stage, status FROM runs") == [
        ("prepare", "failed"),
        ("clean", "ok"),
        ("regions", "failed"),
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
    assert classes["dup_share"].to_list() == pytest.approx(
        (1 - kept["n_rows"] / kept["n_read"]).to_list()
    )
    for column in ("pdd", "tc", "solid_share", "missing_share", "err"):
        assert classes[column].is_between(0, 1).all()
    issues = {i for v in classes["issues"].drop_nulls() for i in v.split(",")}
    assert issues <= {"low_diversity", "collapse"}
    for issue in ISSUES:
        assert classes[f"share_{issue}"].is_between(0, 1).all()


def test_quality_regions_judge_every_region(reported):
    cfg = load_config(str(reported.dataset), method=str(reported.method))
    regions = _table(reported, "quality_regions").sort("region")
    pairs = _table(reported, "quality_pairs")
    parquet = _read(reported, "regions.parquet").sort("region")
    assert regions["region"].to_list() == parquet["region"].to_list()
    assert regions["n_rows"].to_list() == parquet["n_rows"].to_list()
    assert regions["n_duplicates"].to_list() == pytest.approx(
        (parquet["n_read"] - parquet["n_rows"]).to_list()
    )
    assert regions["dup_share"].to_list() == pytest.approx(
        (1 - parquet["n_rows"] / parquet["n_read"]).to_list()
    )
    assert regions["rival"].is_not_null().all()
    assert regions["err"].is_between(0, 1).all()
    for r in regions.iter_rows(named=True):
        own = pairs.filter(
            (pl.col("region_a") == r["region"]) | (pl.col("region_b") == r["region"])
        )
        verdicts = own["verdict"].drop_nulls().to_list()
        assert [r[f"pairs_{v}"] for v in VERDICTS] == [
            verdicts.count(v) for v in VERDICTS
        ]
        issues = region_issues(
            r["missing_rows"],
            r["dup_share"],
            r["conflict_rows"],
            verdicts,
            cfg.method.quality.thresholds["dup_share"],
        )
        assert r["issues"] == (",".join(issues) or None)
        assert r["status"] == ("flagged" if issues else "solid")
        assert (r["depends_on"] is not None) == (r["verdict"] == "shortcut")
        assert r["missing_rows"] >= max(r[f"{kind}_rows"] for kind in KINDS)


def _bad_cells(series: pl.Series) -> int:
    n = series.is_null().sum()
    if series.dtype.is_float():
        n += (series.is_nan() | series.is_infinite()).sum()
    return n


def test_region_missing_counts_every_null_or_invalid_cell(reported):
    data = _read(reported, "data.parquet").drop(DUPLICATES, LABEL)
    missing = _table(reported, "region_missing")
    regions = _table(reported, "quality_regions")
    bad = sum(_bad_cells(data[c]) for c in data.columns)
    assert missing["n_rows"].sum() == bad > 0
    assert {"empty", "unparsable", "inf"} <= set(missing["kind"]) <= set(KINDS)
    (unparsable,) = missing.filter(pl.col("kind") == "unparsable").rows(named=True)
    assert unparsable["column"] == "num_1" and unparsable["n_rows"] == 1
    assert 0 < regions["missing_rows"].sum() <= bad
    flagged = regions.filter(pl.col("missing_rows") > 0)
    assert flagged["issues"].str.contains("missing").all()


def test_the_relabelled_row_is_a_label_conflict(reported):
    groups = _table(reported, "label_conflict_groups")
    n_rows = _read(reported, "data.parquet").height
    of_last = groups.filter(pl.col("row") == n_rows - 1)["group"].to_list()
    assert len(of_last) == 1
    group = groups.filter(pl.col("group") == of_last[0])
    assert group.height == 2 and group["class"].n_unique() == 2
    pairs = _table(reported, "quality_pairs")
    a, b = sorted(group["region"].to_list())
    pair = pairs.filter((pl.col("region_a") == a) & (pl.col("region_b") == b))
    assert pair.height == 1 and "conflict" in pair["reasons"][0].split(",")
    assert pair["n_conflicts"][0] >= 2


def test_quality_pairs_follow_their_reasons(reported):
    cfg = load_config(str(reported.dataset), method=str(reported.method))
    quality = cfg.method.quality
    pairs = _table(reported, "quality_pairs")
    regions = _read(reported, "regions.parquet")
    assert pairs.height and (pairs["class_a"] != pairs["class_b"]).all()
    assert (pairs["region_a"] < pairs["region_b"]).all()
    measured = set(pairs["region_a"]) | set(pairs["region_b"])
    assert measured == set(regions["region"])
    columns = set(cfg.num_cols + cfg.cat_cols)
    for p in pairs.iter_rows(named=True):
        reasons = p["reasons"].split(",")
        assert ("near" in reasons) == (p["sep"] < quality.candidate_separation)
        assert ("conflict" in reasons) == (p["n_conflicts"] > 0)
        assert p["verdict"] in VERDICTS
        for key in ("err_a", "err_b", "err", "err_low", "err_without"):
            assert 0 <= p[key] <= 1
        assert p["err"] == pytest.approx((p["err_a"] + p["err_b"]) / 2)
        assert p["delta"] == pytest.approx(p["err_without"] - p["err"])
        assert 0.5 <= p["one_feature"] <= 1 and p["one_feature_col"] in columns
        removed = p["removed"].split(",") if p["removed"] else []
        assert set(removed) <= columns and len(removed) <= quality.max_group
        if p["verdict"] == "shortcut":
            assert removed and p["delta"] >= quality.thresholds["error"]


def test_quality_report_refuses_regions_of_other_data(copy_of_done):
    prepared_file = copy_of_done.output_dir / "prepared.parquet"
    prepared = pl.read_parquet(prepared_file)
    shuffled = np.random.default_rng(0).permutation(prepared[LABEL].to_numpy())
    prepared.with_columns(pl.Series(LABEL, shuffled)).write_parquet(prepared_file)
    with pytest.raises(ValueError, match="run regions again"):
        copy_of_done.run("quality_report")
