"""Dashboard tests through TestClient on a root built by the stages."""

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard import queries
from dashboard.app import _format_number, create_app
from pipeline.runner import run
from tests.helpers import BASE_URL, fast_cfg, query

_TABS = "data classifier regions failures predictors partitions".split()


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
    assert "naive_bayes" in page.text and "kmeans" in page.text
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
    assert client.get("/?sort=regressor_spearman&desc=false").status_code == 400
    assert client.get("/?sort=regress.regressor_spearman&desc=false").status_code == 200
    assert client.get("/?sort=nope").status_code == 400


def test_api_experiments(client: TestClient, experiment: dict, root: Path) -> None:
    """Check that the API row's ids and aggregates match the DB."""
    assert experiment["n_seeds"] == 1 and experiment["variant"] == ""
    assert experiment["values"]["classifier"] == "naive_bayes"
    assert all(cell["done"] == 1 for cell in experiment["coverage"])
    for key, stage, name in (
        ("classify.accuracy", "classify", "accuracy"),
        ("regress.regressor_spearman", "regress", "regressor_spearman"),
        ("baselines.val_mse", "baselines", "val_mse"),
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
    assert page.text.count('class="chart"') >= 5
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
        "classifier": ("Confusion matrix", "precision", "accuracy"),
        "regions": ("Clustering of each class", "size balance", "n1_mean"),
        "failures": ("Test against val failure rate", "Distribution"),
        "predictors": ("RF regressor", "Val rate", "MCP", "ATC", "MSE", "Outer folds"),
        "partitions": ("Operating points", "intra", "predicted_class", "Brier"),
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


def test_regions_explorer(client: TestClient, root: Path, run_id: int) -> None:
    """Check that regions filter and sort; a region shows its rate."""
    url = f"/runs/{run_id}/regions"
    page = client.get(url)
    failures = _execution(root, "failures")
    region_id, rate = query(
        root / "grace.db",
        "SELECT region_id, test_rate FROM failures "
        f"WHERE execution_id = {failures} ORDER BY region_id DESC",
    )[0]
    link = f"{url}/{region_id}"
    assert link in page.text
    narrowed = client.get(f"{url}?class_id=0")
    assert narrowed.text.count(f"{url}/") < page.text.count(f"{url}/")
    fragment = client.get(f"{url}?class_id=0", headers={"HX-Request": "true"})
    assert "<html" not in fragment.text and f"{url}/" in fragment.text
    assert client.get(f"{url}?sort=abs_error&desc=true").status_code == 200
    assert client.get(f"{url}?x=n1_mean&y=regressor").status_code == 200
    assert client.get(f"{url}?x=bogus").status_code == 400
    region = client.get(link)
    assert region.status_code == 200 and "n1_mean" in region.text
    assert _format_number(rate) in region.text
    assert client.get(f"{url}/999999").status_code == 404


def test_partition_sweeps(client: TestClient, run_id: int) -> None:
    """Check that the sweep charts draw for every arm."""
    charts = client.get(f"/runs/{run_id}/partitions/charts")
    assert charts.status_code == 200
    assert charts.text.count('class="chart"') == 4
    for arm in ("intra", "inter", "null", "confidence", "predicted_class"):
        assert arm in charts.text
    assert client.get(f"/runs/{run_id}/partitions/charts?metric=x").status_code == 400
    assert client.get(f"/runs/{run_id}/partitions/charts?x=x").status_code == 400
    assert client.get("/runs/999999/partitions/charts").status_code == 404


def test_compare(client: TestClient) -> None:
    """Check that the comparison charts draw all three families."""
    assert client.get("/compare").status_code == 200
    charts = client.get(
        "/compare/charts?metric=classify.accuracy&x=classifier&color=distance"
        "&score=mse&arm_metric=brier&rule=cbrt"
    )
    assert charts.status_code == 200 and charts.text.count('class="chart"') == 3
    # No id: HTMX would reset the class of a swapped, id-matched div and
    # with it Plotly's js-plotly-plot, which hover and click need.
    assert 'id="chart' not in charts.text
    for name in ("RF regressor", "MCP", "ATC", "confidence"):
        assert name in charts.text
    assert "No run matches" in client.get("/compare/charts?classifier=mlp").text
    queries_400 = ("metric=bogus", "metric=classify", "x=bogus", "rule=bogus")
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
    urls += [
        f"/runs/{run_id}/stages?sig=x",
        f"/runs/{run_id}/partitions/charts",
        f"/runs/{run_id}/regions/0",
    ]
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
        assert overview.text.count('class="disabled"') == 5
        data = client.get(f"/runs/{run_id}/data")
        assert "Rows of each class" in data.text and "Preprocessing" in data.text
        for tab in "classifier regions failures predictors partitions".split():
            page = client.get(f"/runs/{run_id}/{tab}")
            assert page.status_code == 200 and "Nothing to show yet" in page.text
        assert client.get(f"/runs/{run_id}/regions/0").status_code == 404
        assert client.get(f"/runs/{run_id}/partitions/charts").status_code == 404
        assert client.get("/api/experiments").json()[0]["n_seeds"] == 1
        assert client.get("/compare/charts").status_code == 200


def test_the_regressor_joins_a_run_that_lacked_it(tmp_path: Path) -> None:
    """Check that predictors show without regress, RF only with it."""
    first = run(["baselines"], fast_cfg(tmp_path), force=False)
    with TestClient(create_app(tmp_path / "grace.db"), base_url=BASE_URL) as client:
        page = client.get(f"/runs/{first}/predictors")
        assert "MCP" in page.text and "RF regressor" not in page.text
        assert "need" in page.text and "regress" in page.text
        assert client.get(f"/runs/{first}/classifier").status_code == 200
        second = run(["regress"], fast_cfg(tmp_path), force=False)
        (row,) = client.get("/api/experiments").json()
        # Both runs are seed 42: the latest represents it.
        assert row["n_seeds"] == 1 and row["seeds"][0]["run_id"] == second
        assert "RF regressor" in client.get(f"/runs/{second}/predictors").text


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
