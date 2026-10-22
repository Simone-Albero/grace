"""The dashboard's only writes: presets and jobs, no result tables."""

__all__ = [
    "RESTARTED",
    "RESTARTED_RUNNING",
    "add_job",
    "cancel_queued",
    "claim_job",
    "delete_preset",
    "finish_job",
    "recover",
    "save_preset",
]

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from stages.db import connect, now, reap

RESTARTED = "dashboard restarted"
# Its process may outlive a killed dashboard: the log and the run tell.
RESTARTED_RUNNING = "dashboard restarted; the job's process may still be running"


@contextmanager
def _db(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived writable connection; commit, then close it."""
    db = connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()


def recover(path: Path) -> None:
    """Create the DB if new; close the jobs and runs left open.

    Args:
        path: The grace.db file.
    """
    with _db(path) as db:
        db.execute(
            "UPDATE jobs SET status = CASE status WHEN 'running' THEN 'failed' "
            "ELSE 'cancelled' END, error = CASE status WHEN 'running' THEN ? "
            "ELSE ? END, finished_at = ? WHERE status IN ('queued', 'running')",
            (RESTARTED_RUNNING, RESTARTED, now()),
        )
    reap(path.parent)


def save_preset(path: Path, preset: Mapping[str, object], *, original: str) -> None:
    """Insert a preset, or replace the one named `original`.

    Args:
        path: The grace.db file.
        preset: The preset: `name`, `targets`, `overrides`, `sweeps`
            and `force`.
        original: The name of the preset to replace; "" inserts a new
            one.

    Raises:
        ValueError: The name is taken, or there is no preset
            `original`.
    """
    values = {
        "name": preset["name"],
        "targets": json.dumps(preset["targets"]),
        "overrides": json.dumps(preset["overrides"]),
        "sweeps": json.dumps(preset["sweeps"]),
        "force": int(preset["force"]),
        "updated_at": now(),
    }
    with _db(path) as db:
        try:
            if original:
                cursor = db.execute(
                    "UPDATE presets SET name = :name, targets = :targets, "
                    "overrides = :overrides, sweeps = :sweeps, force = :force, "
                    "updated_at = :updated_at WHERE name = :original",
                    {**values, "original": original},
                )
            else:
                cursor = db.execute(
                    "INSERT INTO presets VALUES (:name, :targets, :overrides, "
                    ":sweeps, :force, :updated_at)",
                    values,
                )
        except sqlite3.IntegrityError:
            raise ValueError(
                f"Preset {preset['name']!r} already exists; pick another name."
            ) from None
        if original and cursor.rowcount == 0:
            raise ValueError(f"No preset {original!r}.")


def delete_preset(path: Path, name: str) -> bool:
    """Delete a preset.

    Args:
        path: The grace.db file.
        name: The preset name.

    Returns:
        False if there was no such preset.
    """
    with _db(path) as db:
        return db.execute("DELETE FROM presets WHERE name = ?", (name,)).rowcount > 0


def add_job(path: Path, preset_name: str, argv: Sequence[str]) -> int:
    """Queue a job that runs `argv`.

    Args:
        path: The grace.db file.
        preset_name: The preset the job was launched from.
        argv: The arguments after `python -m stages`.

    Returns:
        The job id.
    """
    with _db(path) as db:
        cursor = db.execute(
            "INSERT INTO jobs (preset, argv, status, created_at) "
            "VALUES (?, ?, 'queued', ?)",
            (preset_name, json.dumps(argv), now()),
        )
        return cursor.lastrowid


def claim_job(path: Path) -> tuple[int, list[str]] | None:
    """Mark the oldest queued job as running.

    Args:
        path: The grace.db file.

    Returns:
        Its id and `argv`, or None if the queue is empty.
    """
    with _db(path) as db:
        row = db.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = "
            "(SELECT MIN(id) FROM jobs WHERE status = 'queued') RETURNING id, argv",
            (now(),),
        ).fetchone()
    return (row[0], json.loads(row[1])) if row is not None else None


def finish_job(
    path: Path,
    job_id: int,
    status: str,
    returncode: int | None,
    error: str | None = None,
) -> None:
    """Close a job with its final status.

    Args:
        path: The grace.db file.
        job_id: The job id.
        status: The final status: done, failed or cancelled.
        returncode: The exit code of the job's process, if it ran.
        error: Why the dashboard ended the job, if it did.
    """
    with _db(path) as db:
        db.execute(
            "UPDATE jobs SET status = ?, returncode = ?, error = ?, finished_at = ? "
            "WHERE id = ?",
            (status, returncode, error, now(), job_id),
        )


def cancel_queued(path: Path, job_id: int) -> bool:
    """Cancel a job that is still queued.

    Args:
        path: The grace.db file.
        job_id: The job id.

    Returns:
        False if the job is not queued.
    """
    with _db(path) as db:
        cursor = db.execute(
            "UPDATE jobs SET status = 'cancelled', finished_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (now(), job_id),
        )
        return cursor.rowcount > 0
