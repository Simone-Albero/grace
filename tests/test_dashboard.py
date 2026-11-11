"""The dashboard API over the done workspace."""

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from dashboard.app import create_app
from dashboard.queries import MAP_MAX_POINTS, MAP_PER_LEAF


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
    assert data["stats"]["shortcut_cols"] == ["num_1", "cat_1"]
    assert set(data["stats"]["highlight"]) >= {"kdn", "enn", "f3", "pdd"}
    assert all(isinstance(p["reasons"], list) for p in data["pairs"])


def test_missing_runs_are_404(client):
    assert client.get("/api/empty/clean").status_code == 404
    assert client.get("/api/empty/prepare").status_code == 404
    assert client.get("/api/empty/regions").status_code == 404
    assert client.get("/api/empty/quality").status_code == 404
    assert client.get("/api/nope/clean").status_code == 404
