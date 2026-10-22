"""The FastAPI app: results, runs, regions, presets and jobs."""

__all__ = ["create_app"]

import json
import sqlite3
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi import Path as PathParam
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dashboard import charts, jobs, queries, store
from dashboard.queries import CONFIG_KEYS, METRICS, REGION_COLUMNS, SCORES
from pipeline.db import explain

# The rendered rows of the region table; the scatter keeps all points.
_MAX_ROWS = 500
# The only Host headers served, against DNS rebinding.
_HOSTS = ["127.0.0.1", "localhost"]
# The job statuses that are not final.
_ACTIVE = ("queued", "running")
# Ids are SQLite integers: a larger one is a 422, not an OverflowError.
_Id = Annotated[int, PathParam(ge=0, le=2**63 - 1)]
_RESULT_COLUMNS = ["run_id", "n_runs", *CONFIG_KEYS, *METRICS]


def _choice(name: str, value: str, valid: Collection[str]) -> None:
    """Raise a 400 naming the valid values unless `value` is one."""
    if value not in valid:
        raise HTTPException(400, f"Unknown {name} {value!r}. Valid: {', '.join(valid)}")


def _sort(rows: Sequence[dict], key: str, desc: bool) -> list[dict]:
    """Return `rows` ordered by `key`, with the None values last."""
    known = [row for row in rows if row[key] is not None]
    missing = [row for row in rows if row[key] is None]
    return sorted(known, key=lambda row: row[key], reverse=desc) + missing


def _keep(rows: Sequence[dict], selected: Mapping[str, str]) -> list[dict]:
    """Return the rows whose value of each selected key matches.

    An empty selection value matches any row.
    """
    return [
        row
        for row in rows
        if all(value in ("", str(row[key])) for key, value in selected.items())
    ]


def _options(rows: Sequence[dict]) -> dict[str, list[str]]:
    """Return the distinct values of every config key, for selects."""
    return {key: sorted({str(row[key]) for row in rows}) for key in CONFIG_KEYS}


def _num(value: float | None) -> str:
    """Format a table cell: an en dash for None, else a number."""
    if value is None:
        return "–"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.4f}"
    return str(int(value))


def _filters(request: Request) -> dict[str, str]:
    """Read the results filters of the query string.

    Args:
        request: The request.

    Returns:
        The value of every config key; "" means no filter.
    """
    return {key: request.query_params.get(key, "") for key in CONFIG_KEYS}


def _same_origin(request: Request) -> None:
    """Refuse a POST from another web page.

    The Host is already pinned, so `base_url` is this dashboard's own
    origin.

    Args:
        request: The request.

    Raises:
        HTTPException: The 403 for a cross-origin POST.
    """
    origin = request.headers.get("origin")
    if origin is not None and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, f"Cross-origin POST from {origin!r} refused")


def _found(row: dict | None, detail: str) -> dict:
    """Return `row`, or raise a 404 with `detail`."""
    if row is None:
        raise HTTPException(404, detail)
    return row


def _region_rows(
    connection: sqlite3.Connection, run_id: int
) -> tuple[dict[str, int], list[str], list[dict]]:
    """Return the stage ids, measures and region rows of a run."""
    _found(queries.run(connection, run_id), f"No run {run_id}")
    ids = queries.stage_ids(connection, run_id)
    if "regions" not in ids:
        raise HTTPException(404, f"Run {run_id} has no regions stage")
    measures, rows = queries.regions(connection, ids)
    return ids, measures, rows


def create_app(db_path: Path) -> FastAPI:
    """Build the dashboard of the DB at `db_path`.

    Args:
        db_path: The experiment root's `grace.db`; created if new.

    Returns:
        The app. Its root stays locked until its lifespan ends.

    Raises:
        RootInUseError: Another dashboard already serves the root.
        sqlite3.Error: The DB cannot be opened or created; `explain`
            says why.
    """
    root = db_path.parent.resolve()
    # Before recover: a refused instance must not close live jobs.
    lock = jobs.lock_root(root)
    try:
        store.recover(db_path)
    except BaseException:
        lock.close()
        raise
    worker = jobs.Worker(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        worker.stop()
        # The closure keeps the file, and so the lock, alive until here.
        lock.close()

    app = FastAPI(title="grace", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_HOSTS)

    @app.middleware("http")
    async def refuse_framing(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Forbid framing: another page could click for the user."""
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # Tests stop the worker through the state.
    app.state.worker = worker
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
    templates.env.filters["num"] = _num
    templates.env.globals["ACTIVE"] = _ACTIVE
    artifacts = (db_path.parent / "artifacts").resolve()

    def db() -> Iterator[sqlite3.Connection]:
        connection = queries.connect(db_path)
        try:
            yield connection
        finally:
            connection.close()

    def render(
        request: Request,
        page: str,
        context: dict,
        *,
        fragment: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        """Render the full `page`, or `fragment` for an HTMX request."""
        # A history restore (the back button after a cache miss) needs
        # the full page.
        htmx = request.headers.get("HX-Request") and not request.headers.get(
            "HX-History-Restore-Request"
        )
        return templates.TemplateResponse(
            request,
            fragment if htmx and fragment else page,
            context,
            status_code=status_code,
            headers={"Vary": "HX-Request"},
        )

    @app.exception_handler(StarletteHTTPException)
    def error(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
        context = {"status": exc.status_code, "detail": exc.detail}
        return render(request, "error.html", context, status_code=exc.status_code)

    @app.exception_handler(sqlite3.Error)
    def unavailable(request: Request, exc: sqlite3.Error) -> HTMLResponse:
        context = {"status": 503, "detail": explain(exc, db_path)}
        return render(request, "error.html", context, status_code=503)

    @app.exception_handler(RequestValidationError)
    def invalid(request: Request, exc: RequestValidationError) -> HTMLResponse:
        details = "; ".join(issue["msg"] for issue in exc.errors())
        context = {"status": 422, "detail": details}
        return render(request, "error.html", context, status_code=422)

    @app.get("/", response_class=HTMLResponse)
    def results_page(
        request: Request,
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_filters),
        sort: str = "run_id",
        desc: bool = True,
    ) -> HTMLResponse:
        _choice("sort", sort, _RESULT_COLUMNS)
        rows = queries.results(connection)
        context = {
            "rows": _sort(_keep(rows, selected), sort, desc),
            "options": _options(rows),
            "selected": selected,
            "sort": sort,
            "desc": desc,
            "config_keys": list(CONFIG_KEYS),
            "metrics": list(METRICS),
        }
        return render(request, "results.html", context, fragment="_results_table.html")

    @app.get("/api/results")
    def results_api(
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_filters),
    ) -> JSONResponse:
        return JSONResponse(_keep(queries.results(connection), selected))

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        rows = queries.results(connection)
        context = {
            "options": _options(rows),
            "metrics": list(METRICS),
            "config_keys": list(CONFIG_KEYS),
            "scores": SCORES,
        }
        return render(request, "compare.html", context)

    @app.get("/compare/charts", response_class=HTMLResponse)
    def compare_charts(
        request: Request,
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_filters),
        metric: str = "regressor_spearman",
        x: str = "classifier",
        color: str = "clustering",
        score: str = "spearman",
    ) -> HTMLResponse:
        _choice("metric", metric, METRICS)
        _choice("x", x, CONFIG_KEYS)
        _choice("color", color, ["", *CONFIG_KEYS])
        _choice("score", score, SCORES)
        rows = _keep(queries.results(connection), selected)
        figures = [
            charts.metric_bars(rows, metric=metric, x=x, color=color or None),
            charts.predictor_bars(rows, score=score, x=x),
        ]
        return render(request, "_charts.html", {"figures": figures})

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(
        request: Request, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        context = {"runs": queries.runs(connection), "config_keys": list(CONFIG_KEYS)}
        return render(request, "runs.html", context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(
        request: Request, run_id: _Id, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        run = _found(queries.run(connection, run_id), f"No run {run_id}")
        stages = queries.run_stages(connection, run_id)
        ids = queries.stage_ids(connection, run_id)
        classes = []
        if "prepare" in ids and "classify" in ids:
            classes = queries.class_metrics(connection, ids["prepare"], ids["classify"])
        folds = []
        if "regress" in ids:
            folds = queries.regressor_folds(connection, ids["regress"])
        context = {
            "run": run,
            "config_keys": list(CONFIG_KEYS),
            "config": json.dumps(run["config"], indent=2),
            "stages": stages,
            "metrics": queries.stage_metrics(connection, run_id),
            "has_regions": "regions" in ids,
            "classes": classes,
            "folds": folds,
            "figures": queries.figures(connection, run_id),
        }
        return render(request, "run.html", context)

    @app.get("/runs/{run_id}/regions", response_class=HTMLResponse)
    def regions_page(
        request: Request,
        run_id: _Id,
        connection: sqlite3.Connection = Depends(db),
        class_id: str = "",
        used: str = "",
        sort: str = "region_id",
        desc: bool = False,
        x: str = "regressor",
        y: str = "failure_rate",
    ) -> HTMLResponse:
        _, measures, rows = _region_rows(connection, run_id)
        sortable = ["region_id", "class", *REGION_COLUMNS, *measures]
        _choice("sort", sort, sortable)
        axes = [*REGION_COLUMNS, *measures]
        _choice("x", x, axes)
        _choice("y", y, axes)
        _choice("used", used, ["", "1", "0"])
        classes = sorted({(row["class_id"], row["class"]) for row in rows})
        shown = _sort(_keep(rows, {"class_id": class_id, "used": used}), sort, desc)
        context = {
            "run_id": run_id,
            "rows": shown[:_MAX_ROWS],
            "n_rows": len(shown),
            "max_rows": _MAX_ROWS,
            "figures": [charts.region_scatter(shown, x=x, y=y, run_id=run_id)],
            "classes": classes,
            "axes": axes,
            "columns": REGION_COLUMNS,
            "measures": measures,
            "class_id": class_id,
            "used": used,
            "sort": sort,
            "desc": desc,
            "x": x,
            "y": y,
        }
        return render(request, "regions.html", context, fragment="_regions.html")

    @app.get("/runs/{run_id}/regions/{region_id}", response_class=HTMLResponse)
    def region_page(
        request: Request,
        run_id: _Id,
        region_id: _Id,
        connection: sqlite3.Connection = Depends(db),
    ) -> HTMLResponse:
        ids, _, rows = _region_rows(connection, run_id)
        region = next((row for row in rows if row["region_id"] == region_id), None)
        if region is None:
            raise HTTPException(404, f"Run {run_id} has no region {region_id}")
        descriptors = []
        if "complexity" in ids:
            descriptors = queries.descriptors(
                connection, ids["complexity"], region_id, region["class_id"]
            )
        context = {
            "run_id": run_id,
            "region": region,
            "columns": REGION_COLUMNS,
            "descriptors": descriptors,
        }
        return render(request, "region.html", context)

    @app.get("/figures/{execution_id}/{name:path}")
    def figure(
        execution_id: _Id, name: str, connection: sqlite3.Connection = Depends(db)
    ) -> FileResponse:
        found = queries.figure_file(connection, execution_id, name)
        if found is None:
            raise HTTPException(404, f"No figure {name!r} in execution {execution_id}")
        stage, file = found
        path = (artifacts / stage / str(execution_id) / file).resolve()
        if not path.is_relative_to(artifacts) or not path.is_file():
            raise HTTPException(404, f"Figure file of {name!r} is missing")
        return FileResponse(path)

    @app.get("/presets", response_class=HTMLResponse)
    def presets_page(
        request: Request, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        presets = [
            {**preset, "command": jobs.command(jobs.argv(preset, root))}
            for preset in queries.presets(connection)
        ]
        return render(request, "presets.html", {"presets": presets, "root": root})

    def form(
        request: Request,
        preset: dict,
        *,
        original: str,
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        context = {"preset": preset, "original": original, "error": error}
        return render(request, "preset_form.html", context, status_code=status_code)

    @app.get("/presets/new", response_class=HTMLResponse)
    def new_preset(request: Request) -> HTMLResponse:
        preset = jobs.parse(name="", targets="", overrides="", sweeps="", force=False)
        return form(request, preset, original="")

    @app.get("/presets/{name}", response_class=HTMLResponse)
    def preset_page(
        request: Request, name: str, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        preset = _found(queries.preset(connection, name), f"No preset {name!r}")
        return form(request, preset, original=name)

    @app.post("/presets", dependencies=[Depends(_same_origin)])
    def save_preset(
        request: Request,
        original: Annotated[str, Form()] = "",
        name: Annotated[str, Form()] = "",
        targets: Annotated[str, Form()] = "",
        overrides: Annotated[str, Form()] = "",
        sweeps: Annotated[str, Form()] = "",
        force: Annotated[bool, Form()] = False,
    ) -> Response:
        preset = jobs.parse(
            name=name, targets=targets, overrides=overrides, sweeps=sweeps, force=force
        )
        try:
            jobs.check(preset, root)
            store.save_preset(db_path, preset, original=original)
        except ValueError as error:
            return form(
                request, preset, original=original, error=str(error), status_code=400
            )
        return RedirectResponse("/presets", status_code=303)

    @app.post("/presets/{name}/delete", dependencies=[Depends(_same_origin)])
    def delete_preset(name: str) -> RedirectResponse:
        if not store.delete_preset(db_path, name):
            raise HTTPException(404, f"No preset {name!r}")
        return RedirectResponse("/presets", status_code=303)

    @app.post("/presets/{name}/launch", dependencies=[Depends(_same_origin)])
    def launch_preset(
        name: str, connection: sqlite3.Connection = Depends(db)
    ) -> RedirectResponse:
        preset = _found(queries.preset(connection, name), f"No preset {name!r}")
        try:
            # The configs may have changed since the save.
            jobs.check(preset, root)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        job_id = store.add_job(db_path, name, jobs.argv(preset, root))
        worker.kick()
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(
        request: Request, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        rows = queries.jobs(connection)
        active = any(row["status"] in _ACTIVE for row in rows)
        context = {"jobs": rows, "active": active}
        return render(request, "jobs.html", context, fragment="_jobs.html")

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(
        request: Request, job_id: _Id, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        row = _found(queries.job(connection, job_id), f"No job {job_id}")
        runs = []
        if row["started_at"]:
            runs = queries.job_runs(connection, row["started_at"], row["finished_at"])
        context = {
            "job": row,
            "command": jobs.command(row["argv"]),
            "active": row["status"] in _ACTIVE,
            "runs": runs,
            "log": jobs.tail(jobs.log_path(root, job_id)),
        }
        return render(request, "job.html", context, fragment="_job.html")

    @app.get("/jobs/{job_id}/log")
    def job_log_file(job_id: _Id) -> Response:
        path = jobs.log_path(root, job_id)
        if not path.is_file():
            raise HTTPException(404, f"No log for job {job_id}")
        # A snapshot: a growing file would outrun the Content-Length of
        # a FileResponse.
        return Response(path.read_bytes(), media_type="text/plain")

    @app.post("/jobs/{job_id}/cancel", dependencies=[Depends(_same_origin)])
    def cancel_job(
        job_id: _Id, connection: sqlite3.Connection = Depends(db)
    ) -> RedirectResponse:
        _found(queries.job(connection, job_id), f"No job {job_id}")
        worker.cancel(job_id)
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    return app
