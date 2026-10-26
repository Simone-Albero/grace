"""Dashboard tests through TestClient on a root built by the stages."""

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard import queries
from dashboard.app import _num, create_app
from pipeline.runner import run
from tests.helpers import BASE_URL, fast_cfg, query


def _execution(root: Path, stage: str) -> int:
    sql = f"SELECT MIN(id) FROM executions WHERE stage = '{stage}'"
    return query(root / "grace.db", sql)[0][0]


@pytest.fixture(scope="module")
def client(root: Path) -> Iterator[TestClient]:
    """Serve the dashboard of the shared root."""
    with TestClient(create_app(root / "grace.db"), base_url=BASE_URL) as client:
        yield client


@pytest.fixture(scope="module")
def run_id(client: TestClient) -> int:
    """Return the run that the results row of the shared root shows."""
    return client.get("/api/results").json()[0]["run_id"]


def test_results(client: TestClient, run_id: int) -> None:
    """Check that results filter, sort and serve HTMX fragments."""
    page = client.get("/")
    assert page.status_code == 200 and "<html" in page.text
    assert "naive_bayes" in page.text and "kmeans" in page.text
    # Another page must not frame this one: its POST buttons would be
    # clicked for the user.
    assert page.headers["x-frame-options"] == "DENY"
    assert client.get("/runs/999999").headers["x-frame-options"] == "DENY"
    assert f"/runs/{run_id}" in page.text
    fragment = client.get("/", headers={"HX-Request": "true"})
    assert fragment.status_code == 200 and "<html" not in fragment.text
    assert "naive_bayes" in fragment.text
    assert fragment.headers["vary"] == "HX-Request"
    restore = client.get(
        "/", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"}
    )
    assert "<html" in restore.text
    assert "naive_bayes" in client.get("/?classifier=naive_bayes").text
    empty = client.get("/?classifier=mlp")
    assert "No finished failures execution yet." in empty.text
    assert client.get("/?sort=regressor_spearman&desc=false").status_code == 200
    assert client.get("/?sort=bogus").status_code == 400


def test_api_results(client: TestClient, root: Path) -> None:
    """Check that the results API row's ids and metrics match the DB."""
    rows = client.get("/api/results").json()
    assert len(rows) == 1
    row = rows[0]
    assert {*queries.CONFIG_KEYS, *queries.METRICS} <= row.keys()
    for key in ("failures", "baselines", "regress"):
        assert row[f"{key}_id"] == _execution(root, key)
    for name, stage in (
        ("regressor_spearman", "regress"),
        ("mcp_spearman", "baselines"),
        ("accuracy", "classify"),
    ):
        sql = (
            f"SELECT value FROM metrics WHERE name = '{name}' "
            f"AND execution_id = {_execution(root, stage)}"
        )
        assert row[name] == query(root / "grace.db", sql)[0][0]
    assert client.get("/api/results?classifier=mlp").json() == []


def test_compare(client: TestClient) -> None:
    """Check that the comparison charts draw all three predictors."""
    assert client.get("/compare").status_code == 200
    charts = client.get(
        "/compare/charts?metric=accuracy&x=classifier&color=distance&score=mae"
    )
    assert charts.status_code == 200 and "Plotly.newPlot" in charts.text
    # No id: HTMX would reset the class of a swapped, id-matched div and
    # with it Plotly's js-plotly-plot, which hover and click need.
    assert 'class="chart"' in charts.text and 'id="chart' not in charts.text
    for name in ("RF regressor", "MCP", "ATC"):
        assert name in charts.text
    assert client.get("/compare/charts?metric=bogus").status_code == 400


def test_runs_and_run_detail(client: TestClient, run_id: int) -> None:
    """Check that the run page shows every stage and figures."""
    assert f"/runs/{run_id}" in client.get("/runs").text
    page = client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    stages = (
        "split prepare regions classify failures baselines complexity regress render"
    ).split()
    figures = ("baselines/predicted_vs_observed", "regress/predicted_vs_observed")
    for text in (*stages, *figures, "failures/rate_by_class", "class_1"):
        assert text in page.text
    assert client.get("/runs/999999").status_code == 404
    invalid = client.get("/runs/abc")
    assert invalid.status_code == 422 and "<html" in invalid.text


def test_figures(client: TestClient, root: Path) -> None:
    """Check that render's figures are served as saved."""
    render = _execution(root, "render")
    sql = f"SELECT name, file FROM figures WHERE execution_id = {render}"
    rows = query(root / "grace.db", sql)
    assert rows
    for name, file in rows:
        response = client.get(f"/figures/{render}/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        saved = root / f"artifacts/render/{render}" / file
        assert response.content == saved.read_bytes()
    name, file = rows[0]
    assert client.get(f"/figures/{render}/nope").status_code == 404
    assert client.get(f"/figures/{render}/{file}").status_code == 404
    assert (
        client.get(f"/figures/{_execution(root, 'regress')}/{name}").status_code == 404
    )


def test_regions(client: TestClient, root: Path, run_id: int) -> None:
    """Check that regions filter and sort; a region shows its rate."""
    url = f"/runs/{run_id}/regions"
    page = client.get(url)
    assert page.status_code == 200 and "n1_mean" in page.text
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
    assert client.get(f"{url}?sort=abs_error&desc=true").status_code == 200
    assert client.get(f"{url}?x=n1_mean&y=regressor").status_code == 200
    assert client.get(f"{url}?x=bogus").status_code == 400

    region = client.get(link)
    assert region.status_code == 200 and "n1_mean" in region.text
    assert _num(rate) in region.text
    assert client.get(f"{url}/999999").status_code == 404


def test_read_routes_never_write(client: TestClient, root: Path, run_id: int) -> None:
    """Check that GET routes never write; connections are read-only."""
    path = root / "grace.db"

    def state() -> tuple[str, float]:
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime

    render = _execution(root, "render")
    name = query(
        root / "grace.db", f"SELECT name FROM figures WHERE execution_id = {render}"
    )[0][0]
    before = state()
    for url in (
        "/",
        "/api/results",
        "/compare",
        "/compare/charts",
        "/runs",
        "/presets",
        "/jobs",
        f"/runs/{run_id}",
        f"/runs/{run_id}/regions",
        f"/runs/{run_id}/regions/0",
        f"/figures/{render}/{name}",
    ):
        assert client.get(url).status_code == 200, url
    assert state() == before
    with closing(queries.connect(path)) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM runs")


def test_unavailable_db_and_huge_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that a DB error is a 503 and a huge id a 422."""
    for url in ("/runs/99999999999999999999", "/jobs/99999999999999999999/log"):
        assert client.get(url).status_code == 422, url

    def broken(path: Path) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(queries, "connect", broken)
    page = client.get("/runs")
    assert page.status_code == 503 and "<nav>" in page.text
    assert "grace.db" in page.text and "locked" in page.text


def test_fresh_root(tmp_path: Path) -> None:
    """Check that a new root gets a DB with the dashboard tables."""
    path = tmp_path / "grace.db"
    with TestClient(create_app(path), base_url=BASE_URL) as client:
        assert path.exists()
        page = client.get("/")
        assert (
            page.status_code == 200
            and "No finished failures execution yet." in page.text
        )
        assert client.get("/presets").status_code == 200
        assert client.get("/jobs").status_code == 200
    tables = {
        name
        for (name,) in query(tmp_path / "grace.db", "SELECT name FROM sqlite_master")
    }
    assert {"presets", "jobs"} <= tables


def test_render_without_then_with_regressor(tmp_path: Path) -> None:
    """Check that a run without regress shows; regress joins it."""
    run_id = run(["render"], fast_cfg(tmp_path, "regressor_figures=false"), force=False)
    linked = query(
        tmp_path / "grace.db", f"SELECT stage FROM run_stages WHERE run_id = {run_id}"
    )
    stages = {stage for (stage,) in linked}
    assert stages == set(
        "split prepare regions classify failures baselines render".split()
    )
    names = {
        name for (name,) in query(tmp_path / "grace.db", "SELECT name FROM figures")
    }
    assert {"failures/rate_by_class", "baselines/predicted_vs_observed"} <= names
    assert not any(name.startswith("regress/") for name in names)

    with TestClient(create_app(tmp_path / "grace.db"), base_url=BASE_URL) as client:
        rows = client.get("/api/results").json()
        assert len(rows) == 1 and rows[0]["run_id"] == run_id
        assert rows[0]["regress_id"] is None and rows[0]["regressor_spearman"] is None
        sql = (
            "SELECT value FROM metrics WHERE name = 'mcp_spearman' "
            f"AND execution_id = {_execution(tmp_path, 'baselines')}"
        )
        assert rows[0]["mcp_spearman"] == query(tmp_path / "grace.db", sql)[0][0]
        render = _execution(tmp_path, "render")
        figure = f"/figures/{render}/failures/rate_by_class"
        page = client.get(f"/runs/{run_id}")
        assert page.status_code == 200 and "failures/rate_by_class" in page.text
        charts = client.get("/compare/charts?metric=regressor_spearman")
        assert charts.status_code == 200
        assert "MCP" in charts.text and "RF regressor" not in charts.text
        for url in (
            "/",
            f"/runs/{run_id}/regions",
            f"/runs/{run_id}/regions/0",
            figure,
        ):
            assert client.get(url).status_code == 200, url

        run(["render"], fast_cfg(tmp_path), force=False)
        run(["failures"], fast_cfg(tmp_path), force=False)
        rows = client.get("/api/results").json()
        assert len(rows) == 1 and rows[0]["n_runs"] == 3
        assert rows[0]["run_id"] == run_id + 1 and rows[0]["regress_id"] is not None
