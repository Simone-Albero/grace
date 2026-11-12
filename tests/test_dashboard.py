"""The dashboard API over the done workspace."""

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from dashboard.app import create_app
from dashboard.queries import CONFLICT_GROUPS, MAP_MAX_POINTS, MAP_PER_LEAF


@pytest.fixture(scope="module")
def client(done, tmp_path_factory) -> TestClient:
    """An app on the done dataset, toy, and on empty, never run."""
    folder = tmp_path_factory.mktemp("datasets")
    (folder / "toy.yaml").write_text(done.dataset.read_text())
    empty = yaml.safe_load(done.dataset.read_text())
    empty.update(
        name="empty",
        output_dir=str(folder / "nothing"),
        db_path=str(folder / "nothing.db"),
    )
    (folder / "empty.yaml").write_text(yaml.safe_dump(empty))
    app = create_app(folder)
    # The dataset yamls name the default method, the runs used a smaller
    # one: the queries read only the db and the files, never the method.
    return TestClient(app)


def _get(client: TestClient, path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


def test_index_and_options(client):
    assert client.get("/").status_code == 200
    options = _get(client, "/api/options")
    assert options == {"datasets": ["empty", "toy"]}


def test_page_files_are_revalidated(client):
    # An embedded browser (VS Code's) keeps a stale style.css otherwise.
    for path in ["/", "/static/style.css", "/static/app.js"]:
        assert client.get(path).headers["cache-control"] == "no-cache"


def test_page_asks_for_its_files_by_version(client):
    # A file that changes changes its address, so no cache can serve
    # the old one.
    page = client.get("/").text
    for name in ["style.css", "app.js"]:
        assert f'"/static/{name}?v=' in page
        assert f'"/static/{name}"' not in page


def test_clean(client):
    data = _get(client, "/api/toy/clean")
    assert set(data) >= {"run", "summary", "classes", "dropped_classes", "features"}
    assert len(data["classes"]) == 9
    assert sum(c["n_rows"] for c in data["classes"]) == data["summary"]["n_kept"]
    assert data["summary"]["n_duplicate"] > 0
    assert "n_duplicates" not in data["features"]["num"]
    assert data["dropped_columns"] == ["flat"]
    assert "flat" not in data["features"]["num"]
    assert set(data["features"]) == {"num", "cat"}


def test_prepare(client):
    data = _get(client, "/api/toy/prepare")
    assert set(data) >= {"run", "summary", "columns"}
    summary, columns = data["summary"], data["columns"]
    assert summary["n_columns"] == len(columns)
    (missing,) = [c for c in columns if c["kind"] == "missing"]
    assert missing["n_rows"] == summary["n_missing_rows"] > 0
    assert "n_duplicates" not in [c["source"] for c in columns]
    cat = [c for c in columns if c["kind"] == "cat"]
    assert len({c["source"] for c in cat}) == summary["n_cat"]


def test_regions(client):
    data = _get(client, "/api/toy/regions")
    assert set(data) >= {"run", "stats", "classes", "candidates", "regions", "tree"}
    assert data["stats"]["n_regions"] == len(data["regions"])
    assert "algorithm" not in data["stats"]
    assert all("params" not in c for c in data["classes"] + data["candidates"])
    leaves = [n for n in data["tree"] if n["region"] is not None]
    assert len(leaves) == len(data["regions"])
    assert all("stability" in r for r in data["regions"])
    assert all({"threshold", "n_regions"} <= set(c) for c in data["candidates"])


def _below(tree: list[dict], node: int) -> set[int]:
    """The regions under a node."""
    kids = {}
    for n in tree:
        kids.setdefault(n["parent"], []).append(n)
    by_node = {n["node"]: n for n in tree}
    stack, regions = [by_node[node]], set()
    while stack:
        n = stack.pop()
        if n["region"] is not None:
            regions.add(n["region"])
        stack += kids.get(n["node"], [])
    return regions


def test_region_map_of_a_root_and_of_a_leaf(client):
    tree = _get(client, "/api/toy/regions")["tree"]
    root = max(
        (n for n in tree if n["parent"] == -1),
        key=lambda n: len(_below(tree, n["node"])),
    )
    data = _get(client, f"/api/toy/regions/map?node={root['node']}")
    assert data["node"] == root["node"]
    points = np.array(data["points"])
    regions = points[:, 2].astype(int)
    assert set(regions) <= _below(tree, root["node"])
    leaves = [n for n in tree if n["region"] in _below(tree, root["node"])]
    per_leaf = min(MAP_PER_LEAF, max(1, MAP_MAX_POINTS // len(leaves)))
    assert len(points) == sum(min(n["n_rows"], per_leaf) for n in leaves)
    for n in leaves:
        assert (regions == n["region"]).sum() == min(n["n_rows"], per_leaf)
    # the weight of a point (rows / taken) needs both counts of its region
    assert sorted(data["sampled"]) == sorted(
        [n["region"], n["n_rows"], min(n["n_rows"], per_leaf)] for n in leaves
    )
    s1, s2 = data["shares"]
    assert 0 <= s2 <= s1 <= 1 and s1 + s2 <= 1 + 1e-9
    assert np.isfinite(points[:, :2]).all()
    leaf = next(n for n in tree if n["region"] is not None)
    data = _get(client, f"/api/toy/regions/map?node={leaf['node']}")
    assert {p[2] for p in data["points"]} == {leaf["region"]}
    assert len(data["points"]) == min(leaf["n_rows"], MAP_PER_LEAF)


def test_region_map_of_an_unknown_node_is_404(client):
    assert client.get("/api/toy/regions/map?node=-5").status_code == 404


def test_regions_carry_the_overlap_zones(client):
    data = _get(client, "/api/toy/regions")
    assert data["stats"]["overlap_separation"] > 0
    assert data["zones"] and data["overlaps"]
    roots = [n for n in data["zones"] if n["parent"] == -1]
    assert len(roots) == data["stats"]["n_zones"]
    assert {"zone", "region_a", "region_b", "spread_a", "separation"} <= set(
        data["overlaps"][0]
    )


def test_overlap_map_of_a_zone_and_of_a_leaf(client):
    nodes = _get(client, "/api/toy/regions")["zones"]
    root = next(n for n in nodes if n["parent"] == -1)
    data = _get(client, f"/api/toy/regions/overlap_map?node={root['node']}")
    assert data["node"] == root["node"]
    regions = {p[2] for p in data["points"]}
    leaves = [n for n in nodes if n["region"] is not None and n["zone"] == root["zone"]]
    assert regions == {n["region"] for n in leaves}
    per_leaf = min(MAP_PER_LEAF, max(1, MAP_MAX_POINTS // len(leaves)))
    assert len(data["points"]) == sum(min(n["n_rows"], per_leaf) for n in leaves)
    s1, s2 = data["shares"]
    assert 0 <= s2 <= s1 <= 1
    leaf = leaves[0]
    data = _get(client, f"/api/toy/regions/overlap_map?node={leaf['node']}")
    assert {p[2] for p in data["points"]} == {leaf["region"]}


def test_overlap_map_of_an_unknown_node_is_404(client):
    assert client.get("/api/toy/regions/overlap_map?node=-5").status_code == 404


def test_quality(reported, client):
    data = _get(client, "/api/toy/quality")
    assert len(data["classes"]) == 9 and data["pairs"] and data["regions"]
    assert set(data["stats"]["thresholds"]) == {
        "nn_error",
        "single_column_acc",
        "dup_share",
        "pdd",
        "tc",
    }
    assert data["missing"] and {"region", "column", "kind"} <= set(data["missing"][0])
    assert all(isinstance(p["reasons"], list) for p in data["pairs"])
    assert {"nn_error", "shortcut_columns", "single_column"} <= set(data["pairs"][0])


def _of(rows: list[dict], region: int) -> list[dict]:
    """The pairs that have region on either side."""
    return [p for p in rows if region in (p["region_a"], p["region_b"])]


def test_quality_region_of_a_region_in_a_label_conflict(reported, client):
    quality = _get(client, "/api/toy/quality")
    region = next(r["region"] for r in quality["regions"] if r["conflict_rows"])
    data = _get(client, f"/api/toy/quality/region?region={region}")
    assert data["region"]["region"] == region
    assert data["region"]["conflict_rows"] > 0
    key = lambda p: (p["region_a"], p["region_b"])  # noqa: E731
    assert sorted(map(key, data["pairs"])) == sorted(
        map(key, _of(quality["pairs"], region))
    )
    assert all(isinstance(p["reasons"], list) for p in data["pairs"])
    assert data["missing"] == [m for m in quality["missing"] if m["region"] == region]
    conflicts = data["conflicts"]
    assert 1 <= len(conflicts) <= min(data["n_groups"], CONFLICT_GROUPS)
    weights = [g["n_read"] for g in conflicts]
    assert weights == sorted(weights, reverse=True)
    features = _get(client, "/api/toy/clean")["features"]
    for group in conflicts:
        assert len({r["class"] for r in group["rows"]}) >= 2
        assert any(r["region"] == region for r in group["rows"])
        assert group["n_read"] == pytest.approx(sum(r["n_read"] for r in group["rows"]))
        assert list(group["values"]) == features["num"] + features["cat"]


def test_quality_region_without_label_conflicts(reported, client):
    quality = _get(client, "/api/toy/quality")
    clean = [r for r in quality["regions"] if not r["conflict_rows"]]
    data = _get(client, f"/api/toy/quality/region?region={clean[0]['region']}")
    assert data["n_groups"] == 0 and data["conflicts"] == []


def test_quality_region_of_an_unknown_region_is_404(reported, client):
    assert client.get("/api/toy/quality/region?region=-1").status_code == 404
    assert client.get("/api/toy/quality/map?region=100000").status_code == 404


def test_quality_map_shows_a_region_and_its_rivals(reported, client):
    quality = _get(client, "/api/toy/quality")
    sizes = {r["region"]: r["n_rows"] for r in quality["regions"]}
    region = quality["pairs"][0]["region_a"]
    rivals = sorted(
        {p["region_a"] for p in _of(quality["pairs"], region)}
        | {p["region_b"] for p in _of(quality["pairs"], region)}
    )
    rivals.remove(region)
    data = _get(client, f"/api/toy/quality/map?region={region}")
    assert data["region"] == region and data["regions"] == [region, *rivals]
    points = np.array(data["points"])
    shown = points[:, 2].astype(int)
    per_leaf = min(MAP_PER_LEAF, max(1, MAP_MAX_POINTS // len(data["regions"])))
    for r in data["regions"]:
        assert (shown == r).sum() == min(sizes[r], per_leaf)
    assert set(shown) == set(data["regions"])
    assert sorted(data["sampled"]) == sorted(
        [r, sizes[r], min(sizes[r], per_leaf)] for r in data["regions"]
    )
    s1, s2 = data["shares"]
    assert 0 <= s2 <= s1 <= 1


def test_missing_runs_are_404(client):
    assert client.get("/api/empty/clean").status_code == 404
    assert client.get("/api/empty/prepare").status_code == 404
    assert client.get("/api/empty/regions").status_code == 404
    assert client.get("/api/empty/quality").status_code == 404
    assert client.get("/api/empty/quality/region?region=0").status_code == 404
    assert client.get("/api/nope/clean").status_code == 404
