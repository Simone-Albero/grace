"""The dashboard API over the done workspace."""

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from dashboard.app import create_app
from dashboard.queries import CONFLICT_GROUPS, MAP_MAX_POINTS, MAP_PER_CELL
from src.quality import DEFECTS


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


def test_cells(client):
    data = _get(client, "/api/toy/cells")
    assert set(data) == {"run", "stats", "classes", "cells", "tree"}
    assert data["stats"]["n_cells"] == len(data["cells"])
    assert {"max_rows", "min_rows", "distance"} <= set(data["stats"])
    assert sum(c["n_cells"] for c in data["classes"]) == len(data["cells"])
    leaves = [n for n in data["tree"] if n["cell"] is not None]
    assert sorted(n["cell"] for n in leaves) == [c["cell"] for c in data["cells"]]


def _below(tree: list[dict], node: int) -> set[int]:
    """The cells under a node."""
    kids = {}
    for n in tree:
        kids.setdefault(n["parent"], []).append(n)
    by_node = {n["node"]: n for n in tree}
    stack, cells = [by_node[node]], set()
    while stack:
        n = stack.pop()
        if n["cell"] is not None:
            cells.add(n["cell"])
        stack += kids.get(n["node"], [])
    return cells


def test_cell_map_of_a_root_and_of_a_leaf(client):
    data = _get(client, "/api/toy/cells")
    tree, sizes = data["tree"], {c["cell"]: c["n_rows"] for c in data["cells"]}
    root = max(
        (n for n in tree if n["parent"] == -1),
        key=lambda n: len(_below(tree, n["node"])),
    )
    below = _below(tree, root["node"])
    data = _get(client, f"/api/toy/cells/map?node={root['node']}")
    assert data["node"] == root["node"]
    points = np.array(data["points"])
    shown = points[:, 2].astype(int)
    assert set(shown) == below
    per_cell = min(MAP_PER_CELL, max(1, MAP_MAX_POINTS // len(below)))
    for c in below:
        assert (shown == c).sum() == min(sizes[c], per_cell)
    # the weight of a point (rows / taken) needs both counts of its cell
    assert sorted(data["sampled"]) == sorted(
        [c, sizes[c], min(sizes[c], per_cell)] for c in below
    )
    s1, s2 = data["shares"]
    assert 0 <= s2 <= s1 <= 1 and s1 + s2 <= 1 + 1e-9
    assert np.isfinite(points[:, :2]).all()
    leaf = next(n for n in tree if n["cell"] is not None)
    data = _get(client, f"/api/toy/cells/map?node={leaf['node']}")
    assert {p[2] for p in data["points"]} == {leaf["cell"]}


def test_cell_map_of_an_unknown_node_is_404(client):
    assert client.get("/api/toy/cells/map?node=-5").status_code == 404


def test_the_region_routes_of_the_cells_stage_are_gone(client):
    assert client.get("/api/toy/regions").status_code == 404
    assert client.get("/api/toy/regions/map?node=0").status_code == 404
    assert client.get("/api/toy/regions/overlap_map?node=0").status_code == 404


def test_quality(reported, client):
    data = _get(client, "/api/toy/quality")
    assert set(data) == {"run", "stats", "classes", "regions", "missing"}
    assert len(data["classes"]) == 9 and data["regions"]
    shares = {f"share_{d}" for d in DEFECTS} | {"share_no_defect"}
    for row in [data["stats"], *data["classes"]]:
        assert shares <= set(row) and all(0 <= row[k] <= 1 for k in shares)
    assert data["stats"]["tau"] == 0.1 and "thresholds" not in data["stats"]
    assert data["stats"]["n_regions"] == len(data["regions"])
    assert data["missing"] and {"region", "column", "kind"} <= set(data["missing"][0])
    for region in data["regions"]:
        assert set(region["defects"]) <= set(DEFECTS)
        for key in ("overlap_with", "shortcut_with", "conflict_with"):
            assert isinstance(region[key], list)
    assert any(r["defects"] for r in data["regions"])


def test_quality_region_of_a_region_in_a_label_conflict(reported, client):
    quality = _get(client, "/api/toy/quality")
    region = next(r["region"] for r in quality["regions"] if r["conflict_rows"])
    data = _get(client, f"/api/toy/quality/region?region={region}")
    assert data["region"]["region"] == region
    assert data["region"]["conflict_rows"] > 0
    assert "label_conflict" in data["region"]["defects"]
    cells = {c["cell"] for c in data["cells"]}
    assert len(cells) == data["region"]["n_cells"]
    assert all(c["region"] == region for c in data["cells"])
    assert sum(c["n_rows"] for c in data["cells"]) == data["region"]["n_rows"]
    assert data["pairs"] and all(
        {p["cell_a"], p["cell_b"]} & cells for p in data["pairs"]
    )
    assert all(isinstance(p["confusion_without_a"], list) for p in data["pairs"])
    assert data["missing"] == [
        {k: m[k] for k in m if k != "region"}
        for m in quality["missing"]
        if m["region"] == region
    ]
    conflicts = data["conflicts"]
    assert 1 <= len(conflicts) <= min(data["n_groups"], CONFLICT_GROUPS)
    weights = [g["n_read"] for g in conflicts]
    assert weights == sorted(weights, reverse=True)
    features = _get(client, "/api/toy/clean")["features"]
    for group in conflicts:
        assert len({r["class"] for r in group["rows"]}) >= 2
        assert any(r["region"] == region for r in group["rows"])
        assert any(r["cell"] in cells for r in group["rows"])
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
    region = quality["regions"][0]["region"]
    detail = _get(client, f"/api/toy/quality/region?region={region}")
    own = sorted(c["cell"] for c in detail["cells"])
    paired = {p["cell_a"] for p in detail["pairs"]} | {
        p["cell_b"] for p in detail["pairs"]
    }
    rivals = sorted(paired - set(own))
    sizes = {c["cell"]: c["n_rows"] for c in _get(client, "/api/toy/cells")["cells"]}
    data = _get(client, f"/api/toy/quality/map?region={region}")
    assert data["region"] == region and data["cells"] == [*own, *rivals]
    shown = np.array(data["points"])[:, 2].astype(int)
    assert set(shown) == set(data["cells"])
    per_cell = min(MAP_PER_CELL, max(1, MAP_MAX_POINTS // len(data["cells"])))
    assert sorted(data["sampled"]) == sorted(
        [c, sizes[c], min(sizes[c], per_cell)] for c in data["cells"]
    )
    s1, s2 = data["shares"]
    assert 0 <= s2 <= s1 <= 1


def test_missing_runs_are_404(client):
    assert client.get("/api/empty/clean").status_code == 404
    assert client.get("/api/empty/prepare").status_code == 404
    assert client.get("/api/empty/cells").status_code == 404
    assert client.get("/api/empty/quality").status_code == 404
    assert client.get("/api/empty/quality/region?region=0").status_code == 404
    assert client.get("/api/nope/clean").status_code == 404
