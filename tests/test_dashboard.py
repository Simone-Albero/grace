"""The dashboard API over the done workspace."""

import pytest
import yaml
from fastapi.testclient import TestClient

from dashboard.app import create_app


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
    assert "n_duplicates" in data["features"]["num"]
    assert data["dropped_columns"] == ["flat"]
    assert "flat" not in data["features"]["num"]
    assert set(data["features"]) == {"num", "cat"}


def test_regions(client):
    data = _get(client, "/api/toy/regions")
    assert set(data) >= {"run", "stats", "classes", "candidates", "regions"}
    assert data["stats"]["n_regions"] == len(data["regions"])


def test_complexity(client):
    data = _get(client, "/api/toy/complexity")
    assert data["measure"] == data["measures"][0]
    assert len(data["classes"]) == 9
    other = _get(client, f"/api/toy/complexity?measure={data['measures'][1]}")
    assert other["measure"] == data["measures"][1]
    assert client.get("/api/toy/complexity?measure=nope").status_code == 404


def test_missing_runs_are_404(client):
    assert client.get("/api/empty/clean").status_code == 404
    assert client.get("/api/empty/regions").status_code == 404
    assert client.get("/api/nope/clean").status_code == 404
