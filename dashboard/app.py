"""The FastAPI app: experiments, runs and comparisons."""

__all__ = ["create_app"]

import sqlite3
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dashboard import charts, experiments, queries, summary, tabs
from dashboard.experiments import MAIN_KEYS, Experiment
from pipeline.db import explain, reap

# The only Host headers served, against DNS rebinding.
_HOSTS = ["127.0.0.1", "localhost"]
# Ids are SQLite integers: a larger one is a 422, not an OverflowError.
_Id = Annotated[int, PathParam(ge=0, le=2**63 - 1)]
# The columns of the leaves table, in reading order.
_LEAF_COLUMNS = (
    "node_id class depth n_train oof_rate n_val val_rate q gamma regressor n_test "
    "test_rate mean_q"
).split()
_MAX_LEAVES = 500
_SORTS = ["latest", "n_seeds", *MAIN_KEYS, *summary.HEADLINE]
# What the pages poll for while a run is going, in seconds.
_REFRESH_SECONDS = 5
# The trace of a grouped-bar chart: name, color, values by x.
_Trace = tuple[str, str, dict[str, list[float]]]


def _check_choice(name: str, value: str, valid: Collection[str]) -> None:
    """Raise a 400 naming the valid values unless `value` is one."""
    if value not in valid:
        raise HTTPException(400, f"Unknown {name} {value!r}. Valid: {', '.join(valid)}")


def _found(row: dict | None, detail: str) -> dict:
    """Return `row`, or raise a 404 with `detail`."""
    if row is None:
        raise HTTPException(404, detail)
    return row


def _sort_rows(rows: Sequence[dict], key: str, desc: bool) -> list[dict]:
    """Return `rows` ordered by `key`, with the None values last."""
    known = [row for row in rows if row[key] is not None]
    missing = [row for row in rows if row[key] is None]
    return sorted(known, key=lambda row: row[key], reverse=desc) + missing


def _matches(values: Mapping, selected: Mapping[str, str]) -> bool:
    """Say whether `values` matches every selected key; "" is any."""
    return all(value in ("", str(values[key])) for key, value in selected.items())


def _filter_rows(rows: Sequence[Mapping], selected: Mapping[str, str]) -> list[Mapping]:
    """Return the rows whose value of each selected key matches."""
    return [row for row in rows if _matches(row, selected)]


def _format_number(value: float | None) -> str:
    """Format a table cell: an en dash for None, else a number."""
    if value is None:
        return "–"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}" if abs(value) >= 100 else f"{value:.4f}"
    return f"{int(value):,}"


def _format_aggregate(
    aggregate: tuple[float | None, float | None, int], n_seeds: int
) -> str:
    """Format `(mean, std, n)`; the count shows when a seed lacks it."""
    mean, std, n_values = aggregate
    if mean is None:
        return "–"
    text = (
        _format_number(mean)
        if std is None
        else f"{_format_number(mean)} ± {_format_number(std)}"
    )
    return text if n_values == n_seeds else f"{text} (n={n_values})"


def _flatten(config: Mapping, prefix: str = "") -> list[tuple[str, object]]:
    """List a nested config as `(dotted key, value)` pairs."""
    pairs: list[tuple[str, object]] = []
    for name, value in config.items():
        if isinstance(value, Mapping):
            pairs += _flatten(value, f"{prefix}{name}.")
        else:
            pairs.append((f"{prefix}{name}", value))
    return pairs


def _read_filters(request: Request) -> dict[str, str]:
    """Read the config filters of the query string; "" means any."""
    return {key: request.query_params.get(key, "") for key in MAIN_KEYS}


def _list_options(items: Sequence[Mapping]) -> dict[str, list[str]]:
    """Return the distinct values of every main config key."""
    return {key: sorted({str(item[key]) for item in items}) for key in MAIN_KEYS}


def _load_experiments(db: sqlite3.Connection) -> list[Experiment]:
    """Group every run of the DB into experiments."""
    return experiments.build(queries.runs(db), queries.run_metrics(db))


def _signature(run: Mapping) -> str:
    """Summarize a run's progress, to tell when its page is stale."""
    states = summary.stage_states(run)
    progress = ",".join(f"{stage}:{state}" for stage, state in states.items())
    return run["status"] + "|" + progress


def _slot_color(slot: int) -> str:
    """Return the categorical color of a slot; a ninth is neutral."""
    return charts.SERIES[slot] if slot < len(charts.SERIES) else charts.NEUTRAL


def _metric_traces(
    rows: Sequence[Mapping],
    all_rows: Sequence[Mapping],
    metric: str,
    x: str,
    color: str,
) -> list[_Trace]:
    """Build the traces of `metric`, one per value of `color`."""
    # Color follows the entity, over all rows, so the survivors of a
    # filter keep theirs.
    tints = sorted({str(row[color]) for row in all_rows}) if color else [""]
    by_tint = experiments.grouped_values(rows, metric, x, color or None)
    return [
        (
            f"{color}={value}" if color else summary.metric_label(metric),
            _slot_color(tints.index(value)),
            groups,
        )
        for value, groups in sorted(by_tint.items())
    ]


def _predictor_traces(rows: Sequence[Mapping], score: str, x: str) -> list[_Trace]:
    """Build one trace per predictor that has a value of `score`."""
    traces = []
    for key, label in queries.PREDICTORS.items():
        metric = f"evaluate.{key}_{score}"
        grouped = experiments.grouped_values(rows, metric, x, None)
        if grouped:
            traces.append((label, charts.PREDICTOR_COLORS[key], grouped[""]))
    return traces


def _pivot(
    traces: Sequence[_Trace], rows: Sequence[Mapping], x: str
) -> dict[str, list]:
    """Lay the traces out as a table, a row per x value."""
    x_values = sorted({str(row[x]) for row in rows})
    return {
        "columns": [name for name, _, _ in traces],
        "rows": x_values,
        "cells": [
            {value: experiments.summarize(groups.get(value, [])) for value in x_values}
            for _, _, groups in traces
        ],
    }


def _run_context(connection: sqlite3.Connection, run_id: int) -> dict:
    """Read what every tab of a run shows in its header."""
    found = _found(queries.run(connection, run_id), f"No run {run_id}")
    ids = queries.stage_ids(found)
    return {
        "run": found,
        "ids": ids,
        "experiment": experiments.config_key(found)[0],
        "states": summary.stage_states(found),
        "signature": _signature(found),
        "open_tabs": {
            tab: not needs or any(stage in ids for stage in needs)
            for tab, needs in tabs.TABS.items()
        },
    }


def create_app(db_path: Path) -> FastAPI:
    """Build the dashboard of the DB at `db_path`.

    Args:
        db_path: The experiment root's `grace.db`; created if new.

    Returns:
        The app.

    Raises:
        OSError: The root cannot be written to.
        sqlite3.Error: The DB cannot be opened or created; `explain`
            says why.
    """
    root = db_path.parent.resolve()
    # Creates the DB if new, and fails the runs that nothing is running.
    reap(root)

    app = FastAPI(title="grace", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_HOSTS)
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
    templates.env.filters["num"] = _format_number
    templates.env.filters["pm"] = _format_aggregate
    templates.env.filters["flat"] = _flatten
    templates.env.globals.update(
        STAGES=queries.STAGE_ORDER,
        TABS=tabs.TABS,
        REFRESH=_REFRESH_SECONDS,
        MAIN_KEYS=list(MAIN_KEYS),
    )

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
        headers: Mapping[str, str] | None = None,
    ) -> HTMLResponse:
        """Render the full `page`, or `fragment` for an HTMX request."""
        # A history restore (the back button after a cache miss) needs
        # the full page.
        is_htmx = request.headers.get("HX-Request") and not request.headers.get(
            "HX-History-Restore-Request"
        )
        return templates.TemplateResponse(
            request,
            fragment if is_htmx and fragment else page,
            context,
            status_code=status_code,
            headers={"Vary": "HX-Request", **(headers or {})},
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
    def experiments_page(
        request: Request,
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_read_filters),
        sort: str = "latest",
        desc: bool = True,
    ) -> HTMLResponse:
        _check_choice("sort", sort, _SORTS)
        rows = [
            summary.experiment_row(experiment)
            for experiment in _load_experiments(connection)
        ]
        shown = [row for row in rows if _matches(row["values"], selected)]
        for row in shown:
            row.update(row["values"])
            row.update(
                {key: aggregate[0] for key, aggregate in row["aggregates"].items()}
            )
        context = {
            "rows": _sort_rows(shown, sort, desc),
            "options": _list_options([row["values"] for row in rows]),
            "selected": selected,
            "sort": sort,
            "desc": desc,
            "headline": summary.HEADLINE,
            "active": any(row["running"] for row in shown),
        }
        return render(
            request, "experiments.html", context, fragment="_experiments_table.html"
        )

    @app.get("/api/experiments")
    def experiments_api(
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_read_filters),
    ) -> JSONResponse:
        rows = []
        for experiment in _load_experiments(connection):
            row = summary.experiment_row(experiment)
            if _matches(row["values"], selected):
                row["seeds"] = [
                    {
                        "seed": seed.seed,
                        "run_id": seed.run["id"],
                        "metrics": seed.metrics,
                    }
                    for seed in experiment.seeds
                ]
                row["aggregates"] = {
                    key: {"mean": mean, "std": std, "n": n}
                    for key, (mean, std, n) in row["aggregates"].items()
                }
                rows.append(row)
        return JSONResponse(rows)

    @app.get("/experiments/{key}", response_class=HTMLResponse)
    def experiment_page(
        request: Request, key: str, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        experiment = next(
            (found for found in _load_experiments(connection) if found.key == key),
            None,
        )
        if experiment is None:
            raise HTTPException(404, f"No experiment {key!r}")
        seeds = experiment.seeds
        context = {
            "experiment": experiment,
            "headline": summary.HEADLINE,
            "states": {seed.seed: summary.stage_states(seed.run) for seed in seeds},
            "aggregates": {
                metric: experiments.summarize(
                    seed.metrics.get(metric) for seed in seeds
                )
                for metric in summary.HEADLINE
            },
            "figures": summary.experiment_figures(experiment),
            "table": summary.metric_table(experiment),
            "active": any(seed.run["status"] == "running" for seed in seeds),
        }
        return render(request, "experiment.html", context, fragment="_experiment.html")

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(
        request: Request, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        rows = experiments.compare_rows(_load_experiments(connection))
        names = sorted(
            {key for row in rows for key in row["metrics"]},
            key=summary.metric_order,
        )
        context = {
            "options": _list_options(rows),
            "metrics": names,
            "labels": {name: summary.metric_label(name) for name in names},
            "scores": queries.SCORES,
        }
        return render(request, "compare.html", context)

    @app.get("/compare/charts", response_class=HTMLResponse)
    def compare_charts(
        request: Request,
        connection: sqlite3.Connection = Depends(db),
        selected: dict[str, str] = Depends(_read_filters),
        metric: str = "classify.accuracy",
        x: str = "classifier",
        color: str = "",
        score: str = "spearman",
    ) -> HTMLResponse:
        _check_choice("x", x, MAIN_KEYS)
        _check_choice("color", color, ["", *MAIN_KEYS])
        _check_choice("score", score, queries.SCORES)
        all_rows = experiments.compare_rows(_load_experiments(connection))
        # A metric of a stage no run has finished is an empty chart, not
        # an error: it is the form's default on a root with few stages.
        stage, _, name = metric.partition(".")
        _check_choice("metric", stage, queries.STAGE_ORDER)
        if not name:
            raise HTTPException(400, f"Metric {metric!r} is not <stage>.<name>")
        rows = _filter_rows(all_rows, selected)
        main = _metric_traces(rows, all_rows, metric, x, color)
        predictors = _predictor_traces(rows, score, x)
        figures = [
            (
                summary.metric_label(metric),
                charts.group_bars(main, x_title=x, y_title=metric.split(".")[-1]),
            ),
            (
                f"Predictors, {score}",
                charts.group_bars(predictors, x_title=x, y_title=score),
            ),
        ]
        context = {
            "figures": [(title, fig) for title, fig in figures if fig["data"]],
            "pivot": _pivot(main, rows, x),
            "x": x,
            "n_rows": len(rows),
        }
        return render(request, "_compare.html", context)

    @app.get("/runs/{run_id}/stages", response_class=HTMLResponse)
    def run_stages(
        request: Request,
        run_id: _Id,
        sig: str = "",
        connection: sqlite3.Connection = Depends(db),
    ) -> HTMLResponse:
        context = _run_context(connection, run_id)
        # The page is stale once a stage finished: reload it whole.
        headers = {} if sig == context["signature"] else {"HX-Refresh": "true"}
        return render(request, "_stages.html", context, headers=headers)

    @app.get("/runs/{run_id}/leaves/{node_id}", response_class=HTMLResponse)
    def leaf_page(
        request: Request,
        run_id: _Id,
        node_id: _Id,
        connection: sqlite3.Connection = Depends(db),
    ) -> HTMLResponse:
        context = _run_context(connection, run_id)
        ids = context["ids"]
        if "tree" not in ids:
            raise HTTPException(404, f"Run {run_id} has no done tree stage")
        leaf = next(
            (
                row
                for row in queries.leaves(connection, ids)
                if row["node_id"] == node_id
            ),
            None,
        )
        if leaf is None:
            raise HTTPException(404, f"Run {run_id} has no leaf {node_id}")
        descriptors = []
        if "complexity" in ids:
            descriptors = queries.leaf_descriptors(
                connection, ids["complexity"], node_id, leaf["class_id"]
            )
        context.update(
            tab="tree",
            available=True,
            settings={},
            leaf=leaf,
            columns=_LEAF_COLUMNS[2:],
            path=queries.leaf_path(connection, ids["tree"], node_id),
            descriptors=descriptors,
        )
        return render(request, "leaf.html", context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_overview(
        request: Request, run_id: _Id, connection: sqlite3.Connection = Depends(db)
    ) -> HTMLResponse:
        return run_tab(request, run_id, "overview", connection)

    @app.get("/runs/{run_id}/{tab}", response_class=HTMLResponse)
    def run_tab(
        request: Request,
        run_id: _Id,
        tab: str,
        connection: sqlite3.Connection = Depends(db),
        sort: str = "node_id",
        desc: bool = False,
    ) -> HTMLResponse:
        if tab not in tabs.TABS:
            raise HTTPException(404, f"No tab {tab!r}. Valid: {', '.join(tabs.TABS)}")
        context = _run_context(connection, run_id)
        context.update(tabs.build(connection, context["run"], tab, root))
        if tab == "tree" and context["available"]:
            _check_choice("sort", sort, _LEAF_COLUMNS)
            context.update(
                leaves=_sort_rows(context["leaves"], sort, desc),
                columns=_LEAF_COLUMNS,
                max_rows=_MAX_LEAVES,
                sort=sort,
                desc=desc,
            )
        return render(request, f"run_{tab}.html", context)

    return app
