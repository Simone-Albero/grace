"""Dashboard tests through TestClient on a root built by the stages."""

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dashboard import artifacts, charts, queries, tabs
from dashboard.app import create_app
from pipeline.runner import run
from tests.helpers import BASE_URL, fast_cfg, query

_TABS = "data classifier tree predictors".split()


def _execution(root: Path, stage: str) -> int:
    sql = f"SELECT MIN(id) FROM executions WHERE stage = '{stage}'"
    return query(root / "grace.db", sql)[0][0]


@pytest.fixture(scope="module")
def client(root: Path) -> Iterator[TestClient]:
    """Serve the dashboard of the shared root."""
    with TestClient(create_app(root / "grace.db"), base_url=BASE_URL) as client:
        yield client


@pytest.fixture(scope="module")
def experiment(client: TestClient) -> dict:
    """Return the one experiment of the shared root."""
    (row,) = client.get("/api/experiments").json()
    return row


@pytest.fixture(scope="module")
def run_id(experiment: dict) -> int:
    """Return the run that represents the shared root's seed."""
    return experiment["seeds"][0]["run_id"]


def test_experiments_page(client: TestClient, experiment: dict) -> None:
    """Check that the experiments table filters, sorts and swaps."""
    page = client.get("/")
    assert page.status_code == 200 and "<html" in page.text
    assert "naive_bayes" in page.text
    assert f"/experiments/{experiment['key']}" in page.text
    fragment = client.get("/", headers={"HX-Request": "true"})
    assert fragment.status_code == 200 and "<html" not in fragment.text
    assert "naive_bayes" in fragment.text
    assert fragment.headers["vary"] == "HX-Request"
    restore = client.get(
        "/", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"}
    )
    assert "<html" in restore.text
    assert "naive_bayes" in client.get("/?classifier=naive_bayes").text
    assert "matches these filters" in client.get("/?classifier=mlp").text
    assert client.get("/?sort=tree_auroc&desc=false").status_code == 400
    assert client.get("/?sort=evaluate.tree_auroc&desc=false").status_code == 200
    assert client.get("/?sort=nope").status_code == 400


def test_api_experiments(client: TestClient, experiment: dict, root: Path) -> None:
    """Check that the API row's ids and aggregates match the DB."""
    assert experiment["n_seeds"] == 1 and experiment["variant"] == ""
    assert experiment["values"]["classifier"] == "naive_bayes"
    assert all(cell["done"] == 1 for cell in experiment["coverage"])
    for key, stage, name in (
        ("classify.accuracy", "classify", "accuracy"),
        ("evaluate.test_error_rate", "evaluate", "test_error_rate"),
        ("evaluate.tree_mse_ratio", "evaluate", "tree_mse_ratio"),
    ):
        sql = (
            f"SELECT value FROM metrics WHERE name = '{name}' "
            f"AND execution_id = {_execution(root, stage)}"
        )
        expected = query(root / "grace.db", sql)[0][0]
        assert experiment["aggregates"][key]["mean"] == expected
        assert experiment["aggregates"][key]["n"] == 1
    assert client.get("/api/experiments?classifier=mlp").json() == []


def test_experiment_page(client: TestClient, experiment: dict) -> None:
    """Check that the experiment page shows seeds, figures, metrics."""
    page = client.get(f"/experiments/{experiment['key']}")
    assert page.status_code == 200
    for text in ("Seeds", "Across the seeds", "All metrics", "classify · accuracy"):
        assert text in page.text
    assert page.text.count('class="chart"') >= 4
    assert client.get("/experiments/nope").status_code == 404


def test_run_overview_and_tabs(client: TestClient, run_id: int) -> None:
    """Check that every tab of a complete run has its content."""
    overview = client.get(f"/runs/{run_id}")
    assert overview.status_code == 200
    for stage in queries.STAGE_ORDER:
        assert stage in overview.text
    assert "disabled" not in overview.text
    expected = {
        "data": ("Preprocessing", "class_1", "num_1", "train"),
        "classifier": (
            "Confusion matrix",
            "precision",
            "accuracy",
            "train rows after the cap",
        ),
        "tree": (
            "Predicted classes",
            "Leaves",
            "Estimate against the raw rate",
            "Leaf sizes, per split",
            "Tree of a class",
            'name="color"',
            "test rate",
        ),
        "predictors": (
            "Tree",
            "Confidence",
            "GBM",
            "MSE ratio",
            "Binned MSE ratio",
            "AUROC",
            "AP / p",
        ),
    }
    for tab in _TABS:
        page = client.get(f"/runs/{run_id}/{tab}")
        assert page.status_code == 200, tab
        for text in expected[tab]:
            assert text in page.text, (tab, text)
        assert "Settings of the stages shown" in page.text
    assert client.get(f"/runs/{run_id}/nope").status_code == 404
    assert client.get("/runs/999999").status_code == 404
    invalid = client.get("/runs/abc")
    assert invalid.status_code == 422 and "<html" in invalid.text


def _leaf_url(root: Path, run_id: int) -> str:
    sql = (
        "SELECT MIN(node_id) FROM tree_nodes WHERE status = 'leaf' "
        f"AND execution_id = {_execution(root, 'tree')}"
    )
    return f"/runs/{run_id}/leaves/{query(root / 'grace.db', sql)[0][0]}"


def test_tree_tab_sorts_and_a_leaf_has_its_page(
    client: TestClient, root: Path, run_id: int
) -> None:
    """Check the leaves table sorts and a leaf page shows its path."""
    url = f"/runs/{run_id}/tree"
    assert client.get(f"{url}?sort=q&desc=true").status_code == 200
    assert client.get(f"{url}?sort=bogus").status_code == 400
    leaf = client.get(_leaf_url(root, run_id))
    assert leaf.status_code == 200
    for text in ("Counts and rates", "From the root of", "Complexity descriptors"):
        assert text in leaf.text
    assert client.get(f"/runs/{run_id}/leaves/999999").status_code == 404


def test_the_tree_of_a_class_draws_and_rejects_a_missing_one(
    client: TestClient, root: Path, run_id: int
) -> None:
    """Check the tree fragment, with its figure, and its 404 and 422."""
    sql = (
        "SELECT class_id FROM tree_nodes WHERE parent_id IS NULL "
        f"AND execution_id = {_execution(root, 'tree')} LIMIT 1"
    )
    class_id = query(root / "grace.db", sql)[0][0]
    page = client.get(f"/runs/{run_id}/trees?class_id={class_id}")
    # The tree, then the map of its leaves.
    assert page.status_code == 200 and page.text.count('class="chart"') == 2
    assert "<html" not in page.text and 'id="chart' not in page.text
    for color in charts.TREE_COLORS:
        recolored = client.get(
            f"/runs/{run_id}/trees?class_id={class_id}&color={color}"
        )
        assert recolored.status_code == 200, color
    assert (
        client.get(f"/runs/{run_id}/trees?class_id={class_id}&color=nope").status_code
        == 400
    )
    for plane in charts.TREE_PLANES:
        redrawn = client.get(f"/runs/{run_id}/trees?class_id={class_id}&plane={plane}")
        assert redrawn.status_code == 200, plane
        assert f"first two components of {charts.TREE_PLANES[plane]}" in redrawn.text
    assert (
        client.get(f"/runs/{run_id}/trees?class_id={class_id}&plane=nope").status_code
        == 400
    )
    assert client.get(f"/runs/{run_id}/trees?class_id=999999").status_code == 404
    assert client.get(f"/runs/{run_id}/trees?class_id=abc").status_code == 422
    assert client.get(f"/runs/{run_id}/trees").status_code == 422


def test_a_node_cuts_the_tree_and_the_map_down_to_its_subtree(
    client: TestClient, root: Path, run_id: int
) -> None:
    """Check that `node` shows its subtree, with a way back, or 404s."""
    tree = _execution(root, "tree")
    rows = query(
        root / "grace.db",
        "SELECT node_id, class_id FROM tree_nodes WHERE status = 'internal' "
        f"AND parent_id IS NOT NULL AND execution_id = {tree} LIMIT 1",
    )
    if not rows:
        pytest.skip("no node below a root splits in the synthetic tree")
    node_id, class_id = rows[0]
    url = f"/runs/{run_id}/trees?class_id={class_id}"
    whole = client.get(url)
    zoomed = client.get(f"{url}&node={node_id}")
    assert zoomed.status_code == 200 and zoomed.text.count('class="chart"') == 2
    assert f"below node {node_id}" in zoomed.text
    assert 'class="trail"' in zoomed.text and "Reset" in zoomed.text
    assert "below node" not in whole.text and "Reset" not in whole.text
    # A leaf is in the tree and in the map: the zoom drops the ones
    # outside.
    link = f"/runs/{run_id}/leaves/"
    assert 0 < zoomed.text.count(link) < whole.text.count(link)
    root_id = query(
        root / "grace.db",
        f"SELECT node_id FROM tree_nodes WHERE parent_id IS NULL AND "
        f"class_id = {class_id} AND execution_id = {tree}",
    )[0][0]
    assert client.get(f"{url}&node={root_id}").text == whole.text
    assert client.get(f"{url}&node=").text == whole.text
    assert client.get(f"{url}&node=999999").status_code == 404
    assert client.get(f"{url}&node=abc").status_code == 404


def test_the_trail_of_a_deep_node_links_each_ancestor_below_the_root(
    client: TestClient, root: Path, run_id: int
) -> None:
    """Check that a way up is offered for every node between root and node."""
    tree = _execution(root, "tree")
    rows = query(
        root / "grace.db",
        "SELECT node_id, parent_id, class_id FROM tree_nodes WHERE status = 'internal' "
        f"AND depth >= 2 AND execution_id = {tree} LIMIT 1",
    )
    if not rows:
        pytest.skip("no internal node two levels down in the synthetic tree")
    node_id, parent_id, class_id = rows[0]
    page = client.get(f"/runs/{run_id}/trees?class_id={class_id}&node={node_id}")
    assert page.status_code == 200
    assert f'data-node="{parent_id}"' in page.text
    assert f'data-node="{node_id}"' not in page.text
    assert page.text.count('data-node=""') == 2


def test_subtree_cuts_a_node_and_its_descendants_into_a_tree_of_its_own() -> None:
    """Check the rows, the new root and depths, and a node's path."""
    nodes = [
        {"node_id": 0, "parent_id": None, "depth": 0},
        {"node_id": 1, "parent_id": 0, "depth": 1},
        {"node_id": 2, "parent_id": 0, "depth": 1},
        {"node_id": 3, "parent_id": 1, "depth": 2},
        {"node_id": 4, "parent_id": 1, "depth": 2},
        {"node_id": 5, "parent_id": 3, "depth": 3},
    ]
    below = queries.subtree(nodes, 1)
    assert [(node["node_id"], node["parent_id"], node["depth"]) for node in below] == [
        (1, None, 0),
        (3, 1, 1),
        (4, 1, 1),
        (5, 3, 2),
    ]
    # The rows it was given stay as they were.
    assert nodes[1]["parent_id"] == 0 and nodes[1]["depth"] == 1
    assert [node["node_id"] for node in queries.subtree(nodes, 2)] == [2]
    assert len(queries.subtree(nodes, 0)) == 6
    assert queries.node_path(nodes, 5) == [0, 1, 3, 5]
    assert queries.node_path(nodes, 0) == [0]


def test_the_test_rate_colors_a_tree_only_when_evaluate_ran() -> None:
    """Check the colors a run offers, which the route accepts too."""
    assert "test_rate" in tabs.tree_colors({"tree": 1, "evaluate": 2})
    assert "test_rate" not in tabs.tree_colors({"tree": 1})
    assert list(tabs.tree_colors({"tree": 1})) == [
        "purity",
        "grow_rate",
        "estimate_rate",
    ]


def test_the_map_offers_the_disagreement_axis_only_when_the_tree_has_one() -> None:
    """Check the planes a run offers, which the route accepts too."""
    grown = {"mu_d": 0.2, "sigma_d": 0.1, "d_scale": 2.0}
    assert list(tabs.tree_planes(grown)) == ["z", "z_sd"]
    assert tabs.disagreement_axis(grown) == artifacts.DisagreementAxis(0.2, 0.1, 2.0)
    # d_weight=0 grows the tree on z alone.
    assert list(tabs.tree_planes({**grown, "d_scale": 0.0})) == ["z"]
    assert tabs.disagreement_axis({**grown, "sigma_d": None}) is None
    # A `d` that never varies is standardized to zeros: the same plane.
    assert tabs.disagreement_axis({**grown, "sigma_d": 0.0}) is None
    assert tabs.disagreement_axis({}) is None


@pytest.mark.parametrize(
    ("plane", "note"),
    [
        (None, "The map needs"),
        (
            artifacts.ClassPlane(np.empty(0, dtype=int), np.empty((0, 2)), (0, 0)),
            "No grown row has",
        ),
    ],
)
def test_the_tree_of_a_class_says_why_it_has_no_map(
    client: TestClient,
    run_id: int,
    monkeypatch: pytest.MonkeyPatch,
    plane: artifacts.ClassPlane | None,
    note: str,
) -> None:
    """Check that the tree still draws, and a note replaces the map."""
    monkeypatch.setattr(artifacts, "class_plane", lambda *args, **kwargs: plane)
    page = client.get(f"/runs/{run_id}/trees?class_id=0")
    assert page.status_code == 200 and page.text.count('class="chart"') == 1
    assert note in page.text


def test_class_tree_sums_the_test_rows_of_the_leaves_into_each_node(
    root: Path,
) -> None:
    """Check a root holds the test rows of every leaf of its class."""
    path = root / "grace.db"
    ids = {stage: _execution(root, stage) for stage in ("prepare", "tree", "evaluate")}
    with closing(queries.connect(path)) as db:
        nodes = queries.class_tree(db, ids, 0)
    (top,) = [node for node in nodes if node["parent_id"] is None]
    leaves = [node for node in nodes if node["status"] == "leaf"]
    assert top["n_test"] == sum(leaf["n_test"] for leaf in leaves) > 0
    assert top["n_test_errors"] == sum(leaf["n_test_errors"] for leaf in leaves)
    assert top["n_grow"] >= sum(leaf["n_grow"] for leaf in leaves)
    assert all(
        node["purity"] == pytest.approx(1 - node["grow_rate"])
        for node in nodes
        if node["n_grow"]
    )
    without_evaluate = {key: value for key, value in ids.items() if key != "evaluate"}
    with closing(queries.connect(path)) as db:
        untested_nodes = queries.class_tree(db, without_evaluate, 0)
    assert all(node["n_test"] is None for node in untested_nodes)


def test_compare(client: TestClient) -> None:
    """Check that the comparison charts draw both families."""
    assert client.get("/compare").status_code == 200
    charts = client.get(
        "/compare/charts?metric=classify.accuracy&x=classifier&color=data"
        "&score=auroc"
    )
    assert charts.status_code == 200 and charts.text.count('class="chart"') == 2
    for score in queries.SCORES:
        assert client.get(f"/compare/charts?score={score}").status_code == 200
    # No id: HTMX would reset the class of a swapped, id-matched div and
    # with it Plotly's js-plotly-plot, which hover and click need.
    assert 'id="chart' not in charts.text
    for name in ("Tree", "Confidence", "GBM"):
        assert name in charts.text
    assert "No run matches" in client.get("/compare/charts?classifier=mlp").text
    queries_400 = ("metric=bogus", "metric=classify", "x=bogus", "score=mse")
    for query_string in (*queries_400, "color=bogus"):
        assert client.get(f"/compare/charts?{query_string}").status_code == 400


def test_read_routes_never_write(client: TestClient, root: Path, run_id: int) -> None:
    """Check that GET routes never write; connections are read-only."""
    path = root / "grace.db"

    def state() -> tuple[str, float]:
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime

    before = state()
    urls = ["/", "/api/experiments", "/compare", "/compare/charts", f"/runs/{run_id}"]
    urls += [f"/runs/{run_id}/{tab}" for tab in _TABS]
    urls += [f"/runs/{run_id}/tree?sort=q&desc=true", _leaf_url(root, run_id)]
    urls.append(f"/runs/{run_id}/trees?class_id=0")
    urls.append(f"/runs/{run_id}/stages?sig=x")
    for url in urls:
        assert client.get(url).status_code == 200, url
    assert state() == before
    with closing(queries.connect(path)) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM runs")


def test_unavailable_db_huge_ids_and_foreign_host(
    client: TestClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the 503 of a DB error, 422 of a huge id, 400 of a Host."""
    assert client.get("/runs/99999999999999999999").status_code == 422
    # Against DNS rebinding: only the dashboard's own names are served.
    with TestClient(create_app(root / "grace.db"), base_url="http://evil.test") as bad:
        assert bad.get("/").status_code == 400

    def broken(path: Path) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(queries, "connect", broken)
    page = client.get("/")
    assert page.status_code == 503 and "Experiments" in page.text
    assert "grace.db" in page.text and "locked" in page.text


def test_fresh_root(tmp_path: Path) -> None:
    """Check that a new root gets a DB and every page says so."""
    path = tmp_path / "grace.db"
    with TestClient(create_app(path), base_url=BASE_URL) as client:
        assert path.exists()
        page = client.get("/")
        assert page.status_code == 200 and "No experiment yet" in page.text
        assert client.get("/compare").status_code == 200
        assert "No run matches" in client.get("/compare/charts").text
        assert client.get("/api/experiments").json() == []


def test_a_run_with_only_prepare_shows_what_exists(tmp_path: Path) -> None:
    """Check that tabs whose stage is missing are off, and say why."""
    run_id = run(["prepare"], fast_cfg(tmp_path), force=False)
    with TestClient(create_app(tmp_path / "grace.db"), base_url=BASE_URL) as client:
        overview = client.get(f"/runs/{run_id}")
        assert overview.status_code == 200 and "split" in overview.text
        assert overview.text.count('class="disabled"') == 3
        data = client.get(f"/runs/{run_id}/data")
        assert "Rows of each class" in data.text and "Preprocessing" in data.text
        for tab in "classifier tree predictors".split():
            page = client.get(f"/runs/{run_id}/{tab}")
            assert page.status_code == 200 and "Nothing to show yet" in page.text
        assert client.get("/api/experiments").json()[0]["n_seeds"] == 1
        assert client.get("/compare/charts").status_code == 200


def test_a_running_run_polls_and_reloads_when_a_stage_finishes(
    tmp_path: Path,
) -> None:
    """Check that the stage strip polls a run, reloads when stale."""
    run_id = run(["split"], fast_cfg(tmp_path), force=False)
    db = tmp_path / "grace.db"
    with TestClient(create_app(db), base_url=BASE_URL) as client:
        done = client.get(f"/runs/{run_id}")
        assert f"/runs/{run_id}/stages" not in done.text
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute("UPDATE runs SET status = 'running'")
        live = client.get(f"/runs/{run_id}")
        assert f"/runs/{run_id}/stages?sig=" in live.text
        # The page's own signature: nothing changed, so no reload.
        sig = live.text.split("stages?sig=")[1].split('"')[0]

        same = client.get(f"/runs/{run_id}/stages?sig={sig}")
        assert "hx-refresh" not in same.headers
        stale = client.get(f"/runs/{run_id}/stages?sig=old")
        assert stale.headers["hx-refresh"] == "true"
