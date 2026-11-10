"""Dashboard: python -m uvicorn dashboard.app:app (from the repo)."""

from __future__ import annotations

__all__ = ["app", "create_app"]

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard import queries
from src.config import DATASET_DIR, DatasetConfig, load_config, model_names

STATIC = Path(__file__).parent / "static"
ModelName = Literal[tuple(model_names())]


def _query(query: Callable[..., Any], *args: Any) -> Any:
    """Call a query, turning a missing stage output into a 404."""
    try:
        return query(*args)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


def create_app(dataset_dir: Path = DATASET_DIR) -> FastAPI:
    """Build the app over the dataset configs of dataset_dir.

    Args:
        dataset_dir: The folder of the dataset yamls the dashboard
            offers.

    Returns:
        The app: "/" (the page), "/api/options" and one JSON route per
        stage under "/api/{dataset}"; a stage that never ran answers
        404.
    """
    app = FastAPI(title="grace dashboard")

    def offered_datasets() -> list[str]:
        return sorted(path.stem for path in dataset_dir.glob("*.yaml"))

    def load_dataset(dataset: str) -> DatasetConfig:
        # The name is checked first: load_config also takes arbitrary
        # paths.
        if dataset not in offered_datasets():
            raise HTTPException(404, f"unknown dataset: {dataset}")
        return load_config(str(dataset_dir / f"{dataset}.yaml"))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/options")
    def options() -> dict[str, list[str]]:
        return {"datasets": offered_datasets(), "models": model_names()}

    @app.get("/api/{dataset}/split")
    def split(dataset: str) -> Any:
        return _query(queries.split, load_dataset(dataset))

    @app.get("/api/{dataset}/regions")
    def regions(dataset: str) -> Any:
        return _query(queries.regions, load_dataset(dataset))

    @app.get("/api/{dataset}/complexity")
    def complexity(dataset: str, measure: str | None = None) -> Any:
        return _query(queries.complexity, load_dataset(dataset), measure)

    @app.get("/api/{dataset}/train")
    def train(
        dataset: str, model: ModelName, split: Literal["val", "test"] = "test"
    ) -> Any:
        return _query(queries.train, load_dataset(dataset), model, split)

    @app.get("/api/{dataset}/failures")
    def failures(dataset: str, model: ModelName) -> Any:
        return _query(queries.failures, load_dataset(dataset), model)

    @app.get("/api/{dataset}/regress")
    def regress(dataset: str, model: ModelName) -> Any:
        return _query(queries.regress, load_dataset(dataset), model)

    @app.get("/api/{dataset}/ablation")
    def ablation(dataset: str, model: ModelName) -> Any:
        return _query(queries.ablation, load_dataset(dataset), model)

    @app.get("/api/{dataset}/regress/region")
    def region(dataset: str, model: ModelName, region: int) -> Any:
        return _query(queries.region, load_dataset(dataset), model, region)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
