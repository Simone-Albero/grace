"""Presets as `python -m stages` calls, and the job worker."""

__all__ = [
    "REPO",
    "RootInUseError",
    "Worker",
    "argv",
    "check",
    "command",
    "lock_root",
    "log_path",
    "parse",
    "tail",
]

import fcntl
import itertools
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from dashboard import store
from pipeline.db import reap

# Config paths are relative to the repo.
REPO = Path(__file__).resolve().parents[1]
# The seconds between SIGTERM and SIGKILL when cancelling a running job.
_GRACE_S = 20
# The bytes of a log that its page shows.
_TAIL_BYTES = 64_000
# The pattern refuses "." and "..".
_PRESET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
logger = logging.getLogger(__name__)


class RootInUseError(RuntimeError):
    """The error of a root whose lock another dashboard holds."""


def lock_root(root: Path) -> TextIO:
    """Take the one-dashboard-per-root lock, held while it is open.

    Args:
        root: The experiment root.

    Returns:
        The open lock file; closing it (or the process ending) releases
        the lock.

    Raises:
        RootInUseError: Another dashboard already serves `root`.
    """
    path = root / "jobs" / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a")
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.close()
        raise RootInUseError(
            f"Another dashboard already serves {root} (it holds {path}). Stop "
            "it, or start this one with another --root."
        ) from None
    return file


def _lines(text: str) -> list[str]:
    """Return the non-blank lines of `text`, stripped."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse(
    name: str, targets: str, overrides: str, sweeps: str, force: bool
) -> dict[str, object]:
    """Build a preset from the form fields.

    Args:
        name: The preset name.
        targets: The stage names, split on spaces.
        overrides: The `key=value` overrides, one per line.
        sweeps: The `key=v1,v2` sweeps, one per line.
        force: Whether the targets run again even when cached.

    Returns:
        The preset: `name`, `targets`, `overrides`, `sweeps`, `force`.
    """
    return {
        "name": name.strip(),
        "targets": targets.split(),
        "overrides": _lines(overrides),
        "sweeps": _lines(sweeps),
        "force": force,
    }


def check(preset: Mapping[str, object], root: Path) -> None:
    """Fail like the CLI would on the preset.

    Every sweep combination is loaded as `python -m stages` would.

    Args:
        preset: The preset, as `parse` builds it.
        root: The experiment root the dashboard serves.

    Raises:
        ValueError: The name is invalid, a target is `key=value`, an
            item holds an interpolation, a combination sets `root`, or a
            target or the config is invalid.
    """
    name = preset["name"]
    if not _PRESET_NAME.fullmatch(name) or name == "new":
        raise ValueError(
            f"Preset name {name!r} invalid; start with a letter or digit, then "
            "use up to 63 letters, digits, '_', '.' or '-' ('new' is reserved)."
        )
    for item in preset["targets"]:
        if "=" in item:
            raise ValueError(f"Target {item!r} is key=value; put it in the overrides.")
    for item in [*preset["overrides"], *preset["sweeps"]]:
        if "${" in item:
            raise ValueError(f"Interpolation in {item!r} is not allowed in presets.")
    # Imported here: they import torch (seconds), which only a save or a
    # launch needs.
    from pipeline.config import load_config, sweep
    from pipeline.runner import plan

    for combo in sweep(preset["sweeps"]):
        # `root` first: any later item that sets it, also as
        # `[root]=...`, which OmegaConf reads as the key `root`,
        # changes it. The string, not the resolved path: the job
        # resolves it later, in another cwd, and a symlink may move.
        cfg = load_config([f"root={root}", *preset["overrides"], *combo])
        if cfg.root != str(root):
            raise ValueError(
                f"root is the dashboard's ({root}); remove the override or "
                "sweep that sets it."
            )
        # What a target reads depends on the config, so plan on each.
        plan(preset["targets"], cfg)


def argv(preset: Mapping[str, object], root: Path) -> list[str]:
    """Return the `python -m stages` arguments of the preset on `root`.

    Args:
        preset: The preset, as `parse` builds it.
        root: The experiment root.

    Returns:
        The targets, the overrides, `root=<root>`, a `--sweep` per
        sweep and `--force` if set.
    """
    return [
        *preset["targets"],
        *preset["overrides"],
        f"root={root}",
        *itertools.chain.from_iterable(
            ("--sweep", sweep) for sweep in preset["sweeps"]
        ),
        *(["--force"] if preset["force"] else []),
    ]


def log_path(root: Path, job_id: int) -> Path:
    """Return where the log of a job is, or will be.

    Args:
        root: The experiment root.
        job_id: The job id.

    Returns:
        The log file.
    """
    return root / "jobs" / f"{job_id}.log"


def command(args: Sequence[str]) -> str:
    """Return the shell-quoted command line of a job, for display.

    Args:
        args: The arguments after `python -m stages`.

    Returns:
        The command line.
    """
    return f"python -m stages {shlex.join(args)}"


def tail(path: Path) -> str:
    """Return the end of a log, from its first full line.

    Args:
        path: The log file.

    Returns:
        The last `_TAIL_BYTES` bytes, from the first full line, or ""
        if the file is absent.
    """
    try:
        file = path.open("rb")
    except FileNotFoundError:
        return ""
    with file:
        start = max(0, file.seek(0, os.SEEK_END) - _TAIL_BYTES)
        file.seek(start)
        text = file.read().decode(errors="replace")
    return text.partition("\n")[2] if start else text


def _kill_group(pgid: int) -> None:
    """SIGKILL the process group `pgid`: the job and its workers."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        # Nothing of the job is left.
        pass


def _kill_group_unless_reaped(process: subprocess.Popen) -> None:
    """SIGKILL the job's group unless the job was already reaped."""
    # Once reaped, its pgid may belong to an unrelated group; _run has
    # already killed the workers it left.
    if process.returncode is None:
        _kill_group(process.pid)


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM the job now, SIGKILL its group after the grace."""
    # Alone, so that the runner marks its run failed.
    process.terminate()
    timer = threading.Timer(_GRACE_S, _kill_group_unless_reaped, (process,))
    timer.daemon = True
    timer.start()


class Worker:
    """The runner of the queued jobs of one DB, one process at a time.

    Args:
        db_path: The grace.db whose jobs it runs.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        # The id of the job being run.
        self._running: int | None = None
        # The id of the running job to stop.
        self._cancelled: int | None = None
        # Set by stop(): no further job starts.
        self._closed = False

    def kick(self) -> None:
        """Start the worker thread unless it is already running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, daemon=True, name="jobs"
                )
                self._thread.start()

    def cancel(self, job_id: int) -> None:
        """Cancel a queued job, or terminate it if it is running.

        Args:
            job_id: The job id.
        """
        if store.cancel_queued(self._db_path, job_id):
            return
        with self._lock:
            if self._running == job_id:
                self._cancelled = job_id
                if self._process is not None:
                    _terminate(self._process)

    def stop(self) -> None:
        """Terminate the running job and wait until it closes."""
        with self._lock:
            self._closed = True
            thread, self._cancelled = self._thread, self._running
            if self._process is not None:
                _terminate(self._process)
        if thread is not None:
            # Until the job row is closed: a daemon thread dies with us.
            thread.join(_GRACE_S + 5)

    def _loop(self) -> None:
        while True:
            # The same lock as kick(): no launch is lost.
            with self._lock:
                job = None if self._closed else store.claim_job(self._db_path)
                if job is None:
                    self._thread = None
                    return
                job_id, args = job
                self._running = job_id
                self._cancelled = None
            try:
                self._run(job_id, args)
            except Exception as error:  # A broken job must not stop the queue.
                logger.exception("job %d failed to run", job_id)
                store.finish_job(self._db_path, job_id, "failed", None, repr(error))
            finally:
                with self._lock:
                    self._running = self._process = None

    def _run(self, job_id: int, args: Sequence[str]) -> None:
        path = log_path(self._db_path.parent, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as log:
            log.write(f"$ {command(args)}\n")
            log.flush()
            process = subprocess.Popen(
                [sys.executable, "-m", "stages", *args],
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                # One group: joblib workers die with it.
                start_new_session=True,
            )
            with self._lock:
                self._process = process
                # The job may have been cancelled before the Popen.
                if self._cancelled == job_id:
                    _terminate(process)
            returncode = process.wait()
        # Kill the workers a killed or cancelled job left.
        _kill_group(process.pid)
        # Killed by a signal: the run could not mark itself failed.
        if returncode < 0:
            reap(self._db_path.parent)
        with self._lock:
            cancelled = self._cancelled == job_id
        if returncode == 0:
            status = "done"
        elif cancelled:
            status = "cancelled"
        else:
            status = "failed"
        store.finish_job(self._db_path, job_id, status, returncode)
