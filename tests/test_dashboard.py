"""The dashboard API over the done workspace."""

import pytest
import yaml
from conftest import MODEL
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
    assert options["datasets"] == ["empty", "toy"]
    assert MODEL in options["models"]


def test_split(client):
    data = _get(client, "/api/toy/split")
    assert set(data) >= {"run", "summary", "classes", "dropped_classes", "features"}
    assert len(data["classes"]) == 9
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


def test_train(client):
    for split in ("val", "test"):
        data = _get(client, f"/api/toy/train?model={MODEL}&split={split}")
        assert set(data["overall"]) == {"accuracy", "macro_f1", "weighted_f1"}
        assert len(data["confusion"]["counts"]) == 9
        assert len(data["confidence"]["edges"]) == 21
        assert data["models"] == [MODEL]


def test_failures(client):
    data = _get(client, f"/api/toy/failures?model={MODEL}")
    assert {s["predictor"] for s in data["scores"]} == {"mcp", "atc"}
    assert set(data["stats"]) >= {"n_regions", "n_used", "atc_threshold"}


def test_regress_and_region(client):
    data = _get(client, f"/api/toy/regress?model={MODEL}")
    assert {s["predictor"] for s in data["scores"]} == {"rf", "mcp", "atc"}
    region = data["estimates"][0]["region"]
    detail = _get(client, f"/api/toy/regress/region?model={MODEL}&region={region}")
    assert detail["region"]["region"] == region
    assert detail["contributions"]
    assert detail["descriptors"]
    missing = f"/api/toy/regress/region?model={MODEL}&region=-5"
    assert client.get(missing).status_code == 404


def test_missing_runs_are_404(client):
    assert client.get("/api/empty/split").status_code == 404
    assert client.get("/api/toy/train?model=knn").status_code == 404
    assert client.get("/api/nope/split").status_code == 404
    assert client.get("/api/toy/train?model=nope").status_code == 422
