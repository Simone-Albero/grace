"""Dashboard: python -m uvicorn dashboard.app:app (from the repo)."""

from __future__ import annotations

__all__ = ["app", "create_app"]

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard import queries
from src.config import DATASET_DIR, DatasetConfig, load_config

STATIC = Path(__file__).parent / "static"


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
        return {"datasets": offered_datasets()}

    @app.get("/api/{dataset}/clean")
    def clean(dataset: str) -> Any:
        return _query(queries.clean, load_dataset(dataset))

    @app.get("/api/{dataset}/regions")
    def regions(dataset: str) -> Any:
        return _query(queries.regions, load_dataset(dataset))

    @app.get("/api/{dataset}/complexity")
    def complexity(dataset: str, measure: str | None = None) -> Any:
        return _query(queries.complexity, load_dataset(dataset), measure)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
