"""Dashboard jobs: presets, worker, lock and the job pages."""

import html
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Collection, Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dashboard import jobs, queries, store
from dashboard.app import create_app
from stages.db import KILLED
from tests.helpers import BASE_URL, query

SPLIT = {"name": "split", "targets": "split", "overrides": "data=synthetic_test"}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return the DB path of a new root."""
    return tmp_path / "grace.db"


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    """Serve a dashboard on a new root; stop its worker after."""
    with TestClient(create_app(db_path), base_url=BASE_URL) as client:
        yield client


def _post(client: TestClient, url: str, **fields: str) -> httpx.Response:
    return client.post(url, data=fields, follow_redirects=False)


def _text(response: httpx.Response) -> str:
    return html.unescape(response.text)


def _job(path: Path, job_id: int) -> dict[str, object] | None:
    with closing(queries.connect(path)) as db:
        return queries.job(db, job_id)


def _wait(
    path: Path, job_id: int, statuses: Collection[str], timeout: float = 300
) -> dict[str, object]:
    """Wait until the job has one of `statuses`, and return it.

    The machine may be loaded, so the timeout is long.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _job(path, job_id)
        if job and job["status"] in statuses:
            return job
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} is {job and job['status']}, not {statuses}")


def test_preset_crud(client: TestClient, db_path: Path) -> None:
    """Check that presets are created, shown, renamed and deleted."""
    root = db_path.parent.resolve()
    fields = {
        "name": "quick",
        "targets": "split",
        "overrides": "data=synthetic_test\nseed=7",
        "sweeps": "distance=euclidean,cosine",
    }
    assert _post(client, "/presets", **fields).status_code == 303
    command = (
        f"python -m stages split data=synthetic_test seed=7 root={root} "
        "--sweep distance=euclidean,cosine"
    )
    assert command in _text(client.get("/presets"))
    page = _text(client.get("/presets/quick"))
    assert "seed=7" in page and "distance=euclidean,cosine" in page
    assert client.get("/presets/nope").status_code == 404

    rename = fields | {"original": "quick", "name": "quick2"}
    renamed = _post(client, "/presets", **rename)
    assert renamed.status_code == 303
    listing = _text(client.get("/presets"))
    assert "quick2" in listing and "/presets/quick<" not in listing
    again = _post(client, "/presets", **fields | {"name": "quick2"})
    assert again.status_code == 400 and "already exists" in _text(again)
    assert _post(client, "/presets/quick2/delete").status_code == 303
    assert "No preset yet." in client.get("/presets").text
    assert _post(client, "/presets/quick2/delete").status_code == 404


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"targets": "bogus"}, "Unknown stage 'bogus'"),
        ({"targets": ""}, "No target given"),
        ({"targets": "data=x"}, "put it in the overrides"),
        ({"overrides": "nope=1"}, "Unknown config key in 'nope=1'"),
        ({"overrides": "seed"}, "is not key=value"),
        ({"overrides": "classifier=bogus"}, "Unknown classifier 'bogus'"),
        ({"overrides": "distance=manhattan"}, "distance 'manhattan' invalid"),
        ({"sweeps": "classifier="}, "is not key=value1,value2"),
        ({"sweeps": "distance=euclidean,manhattan"}, "distance 'manhattan' invalid"),
        ({"overrides": "root=/tmp/x"}, "root is the dashboard's"),
        ({"overrides": "[root]=/tmp/x"}, "root is the dashboard's"),
        ({"sweeps": "[root]=/tmp/x,/tmp/y"}, "root is the dashboard's"),
        ({"sweeps": "root=/tmp/x,/tmp/y"}, "root is the dashboard's"),
        ({"overrides": "root.x=1"}, "root {'x': 1} must be"),
        ({"overrides": "seed=${oc.env:HOME}"}, "Interpolation in"),
        ({"sweeps": "seed=${nope},1"}, "Interpolation in"),
        ({"name": "a b"}, "Preset name"),
        ({"name": "new"}, "Preset name"),
        ({"name": ".."}, "Preset name"),
    ],
)
def test_preset_validation(
    client: TestClient, changes: dict[str, str], message: str
) -> None:
    """Check that an invalid preset is refused with a 400 naming why."""
    response = _post(client, "/presets", **SPLIT | changes)
    assert response.status_code == 400 and message in _text(response)
    assert "No preset yet." in client.get("/presets").text


def test_preset_root_must_be_the_literal_root(
    client: TestClient, db_path: Path, tmp_path: Path
) -> None:
    """Check that a sweep cannot spell the root another way."""
    link = tmp_path / "link"
    link.symlink_to(db_path.parent)
    for spelling in (link, f"{db_path.parent.resolve()}/"):
        response = _post(client, "/presets", **SPLIT | {"sweeps": f"root={spelling}"})
        assert response.status_code == 400, spelling
        assert "root is the dashboard's" in _text(response)


def test_job_runs_preset_sequentially(client: TestClient, db_path: Path) -> None:
    """Check that launched jobs run one at a time and log their runs."""
    root = db_path.parent.resolve()
    assert _post(client, "/presets", **SPLIT).status_code == 303
    for i in (1, 2, 3):
        launched = _post(client, "/presets/split/launch")
        assert launched.status_code == 303
        assert launched.headers["location"] == f"/jobs/{i}"
    # Job 1 needs seconds of imports: right after the launches the
    # others still wait.
    assert _job(db_path, 1)["status"] in ("queued", "running")
    assert _job(db_path, 2)["status"] == "queued"
    assert _post(client, "/jobs/3/cancel").status_code == 303
    cancelled = _job(db_path, 3)
    assert cancelled["status"] == "cancelled" and cancelled["finished_at"]
    assert not (root / "jobs/3.log").exists()

    first = _wait(db_path, 1, {"done", "failed"})
    second = _wait(db_path, 2, {"done", "failed"})
    assert first["status"] == second["status"] == "done"
    assert first["returncode"] == second["returncode"] == 0
    assert second["started_at"] >= first["finished_at"]
    log = (root / "jobs/1.log").read_text()
    assert log.startswith("$ python -m stages split") and "run 1 done" in log
    configs = query(db_path, "SELECT status, json_extract(config, '$.root') FROM runs")
    assert configs == [("done", str(root))] * 2
    cached = query(db_path, "SELECT cached FROM run_stages ORDER BY run_id")
    assert cached == [(0,), (1,)]

    page = _text(client.get("/jobs/1"))
    assert "done" in page and "run 1 done" in page and "/runs/1" in page
    logged = client.get("/jobs/1/log")
    assert logged.status_code == 200
    assert logged.headers["content-type"].startswith("text/plain")
    assert client.get("/jobs/3/log").status_code == 404
    assert client.get("/jobs/99").status_code == 404
    idle = client.get("/jobs", headers={"HX-Request": "true"})
    assert "<html" not in idle.text and "hx-trigger" not in idle.text


def test_cancel_running_job(client: TestClient, db_path: Path) -> None:
    """Check that a running job can be cancelled; the queue goes on."""
    prepare = SPLIT | {"name": "prepare", "targets": "prepare"}
    assert _post(client, "/presets", **prepare).status_code == 303
    assert _post(client, "/presets/prepare/launch").status_code == 303
    _wait(db_path, 1, {"running"})
    assert _post(client, "/jobs/1/cancel").status_code == 303
    job = _wait(db_path, 1, {"cancelled"})
    assert job["returncode"] != 0 and job["finished_at"]

    assert _post(client, "/presets", **SPLIT).status_code == 303
    assert _post(client, "/presets/split/launch").status_code == 303
    assert _wait(db_path, 2, {"done", "failed"})["status"] == "done"


def test_killed_job_fails_its_run(client: TestClient, db_path: Path) -> None:
    """Check that a SIGKILLed job fails its run and executions."""
    prepare = SPLIT | {"name": "prepare", "targets": "prepare"}
    assert _post(client, "/presets", **prepare).status_code == 303
    assert _post(client, "/presets/prepare/launch").status_code == 303
    _wait(db_path, 1, {"running"})
    deadline = time.monotonic() + 300
    while not query(db_path, "SELECT id FROM executions WHERE status = 'running'"):
        assert time.monotonic() < deadline, "the job never started an execution"
        time.sleep(0.05)
    os.kill(client.app.state.worker._process.pid, signal.SIGKILL)
    job = _wait(db_path, 1, {"failed"})
    assert job["returncode"] == -signal.SIGKILL
    assert not query(db_path, "SELECT id FROM runs WHERE status = 'running'")
    assert not query(db_path, "SELECT id FROM executions WHERE status = 'running'")
    errors = query(db_path, "SELECT DISTINCT error FROM runs")
    assert errors == [(KILLED,)]


def test_late_kill_spares_a_reaped_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the grace's SIGKILL skips a job already reaped."""
    killed = []
    monkeypatch.setattr(jobs, "_kill_group", killed.append)
    done = subprocess.Popen(["true"], start_new_session=True)
    done.wait()
    jobs._kill_group_unless_reaped(done)
    assert killed == []

    running = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        jobs._kill_group_unless_reaped(running)
        assert killed == [running.pid]
    finally:
        running.kill()
        running.wait()


def test_restart_closes_open_jobs(db_path: Path) -> None:
    """Check that a restart fails running and cancels queued jobs."""
    store.recover(db_path)
    store.add_job(db_path, "p", ["split"])
    store.add_job(db_path, "p", ["split"])
    assert store.claim_job(db_path) == (1, ["split"])
    with TestClient(create_app(db_path), base_url=BASE_URL):
        for job_id, status, error in (
            (1, "failed", store.RESTARTED_RUNNING),
            (2, "cancelled", store.RESTARTED),
        ):
            job = _job(db_path, job_id)
            assert job["status"] == status and job["error"] == error


def test_second_dashboard_refused(db_path: Path) -> None:
    """Check that a second dashboard is refused, jobs untouched."""
    with TestClient(create_app(db_path), base_url=BASE_URL):
        store.add_job(db_path, "p", ["split"])
        with pytest.raises(jobs.RootInUseError, match="Another dashboard"):
            create_app(db_path)
        assert _job(db_path, 1)["status"] == "queued"
    # The lifespan released the lock.
    with TestClient(create_app(db_path), base_url=BASE_URL):
        pass


def test_second_dashboard_process_exits(db_path: Path) -> None:
    """Check that `python -m dashboard` on a served root exits 1."""
    with TestClient(create_app(db_path), base_url=BASE_URL):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dashboard",
                "--root",
                str(db_path.parent),
                "--port",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=jobs.REPO,
        )
    assert result.returncode == 1
    assert "Another dashboard" in result.stderr


def test_cross_origin_post_refused(client: TestClient) -> None:
    """Check that a cross-origin POST is refused with a 403."""
    evil = {"Origin": "http://evil.example"}
    refused = client.post("/presets", data=SPLIT, headers=evil, follow_redirects=False)
    assert refused.status_code == 403
    assert "No preset yet." in client.get("/presets").text
    assert _post(client, "/presets", **SPLIT).status_code == 303


def test_foreign_host_refused(client: TestClient) -> None:
    """Check that a foreign Host header is refused with a 400."""
    host = "attacker.example:8000"
    rebound = {"Host": host, "Origin": f"http://{host}"}
    refused = client.post(
        "/presets", data=SPLIT, headers=rebound, follow_redirects=False
    )
    assert refused.status_code == 400
    assert "No preset yet." in client.get("/presets").text
    assert client.get("/", headers={"Host": host}).status_code == 400
    assert client.get("/", headers={"Host": "localhost:9000"}).status_code == 200


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_worker_survives_a_database_error(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that a DB error in claim_job does not block the queue."""
    calls: list[int] = []

    def claim(path: Path) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "claim_job", claim)
    worker = jobs.Worker(db_path)
    worker.kick()
    dead = worker._thread
    dead.join(10)
    assert not dead.is_alive()
    # A dead thread must not block the queue for good.
    worker.kick()
    deadline = time.monotonic() + 10
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(calls) == 2


def test_stop_closes_the_running_job(client: TestClient, db_path: Path) -> None:
    """Check that stop() returns only once the running job is closed."""
    prepare = SPLIT | {"name": "prepare", "targets": "prepare"}
    assert _post(client, "/presets", **prepare).status_code == 303
    assert _post(client, "/presets/prepare/launch").status_code == 303
    _wait(db_path, 1, {"running"})
    client.app.state.worker.stop()
    # No waiting: stop() returns once the row is closed.
    job = _job(db_path, 1)
    assert job["status"] == "cancelled" and job["finished_at"]
