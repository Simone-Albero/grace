"""The DAG runner: cache keys, stage executions and run bookkeeping."""

__all__ = ["STAGES", "plan", "run"]

import fcntl
import hashlib
import json
import logging
import shutil
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from sqlite3 import Connection

from omegaconf import DictConfig, OmegaConf

from pipeline.db import DB_NAME, RUNS_LOCK, connect, now, reap
from stages import (
    Context,
    Execution,
    baselines,
    classify,
    complexity,
    crossfit,
    evaluate,
    prepare,
    render,
    space,
    split,
    tree,
)

logger = logging.getLogger(__name__)

# The DAG, in execution order.
STAGES = {
    "split": split,
    "prepare": prepare,
    "space": space,
    "classify": classify,
    "crossfit": crossfit,
    "tree": tree,
    "complexity": complexity,
    "baselines": baselines,
    "evaluate": evaluate,
    "render": render,
}
_MISSING = object()
# The size of every output file, to notice a lost or cut one.
_SIZES = "sizes.json"


def _stage_inputs(name: str, cfg: DictConfig) -> tuple[str, ...]:
    """Return the stages `name` reads under this config."""
    module = STAGES[name]
    return module.inputs(cfg) if hasattr(module, "inputs") else module.INPUTS


def plan(targets: Sequence[str], cfg: DictConfig) -> list[str]:
    """Return `targets` and every stage they read, in execution order.

    Args:
        targets: The stages to run.
        cfg: The validated config; what some stages read depends on
            it.

    Returns:
        The stage names in execution order.

    Raises:
        ValueError: There is no target, or one is not a stage.
    """
    if not targets:
        raise ValueError(f"No target given. Valid: {list(STAGES)}")
    for target in targets:
        if target not in STAGES:
            raise ValueError(f"Unknown stage {target!r}. Valid: {list(STAGES)}")
    needed = set(targets)
    # Inputs always come earlier in STAGES.
    for name in reversed(STAGES):
        if name in needed:
            needed.update(_stage_inputs(name, cfg))
    return [name for name in STAGES if name in needed]


def _select(name: str, cfg: DictConfig, key: str) -> object:
    """Return the config value at the dotted `key` a stage declares."""
    value = OmegaConf.select(cfg, key, default=_MISSING)
    if value is _MISSING:
        raise ValueError(f"{name} declares config key {key!r}, absent from the config")
    return value


def _stage_config_keys(name: str, cfg: DictConfig) -> tuple[str, ...]:
    """Return the dotted config keys the stage's outputs depend on."""
    module = STAGES[name]
    return module.config_keys(cfg) if hasattr(module, "config_keys") else module.CONFIG


def _config_slice(name: str, cfg: DictConfig) -> dict[str, object]:
    """Return the values of the stage's config keys, nested by path."""
    values = OmegaConf.create({})
    for key in _stage_config_keys(name, cfg):
        OmegaConf.update(values, key, _select(name, cfg, key), merge=False)
    return OmegaConf.to_container(values)


def _file_stamps(name: str, cfg: DictConfig) -> dict[str, list[int] | None]:
    """Return the size and mtime of each FILES path, None if absent."""
    stamps = {}
    for key in STAGES[name].FILES:
        path = Path(_select(name, cfg, key))
        stat = path.stat() if path.exists() else None
        stamps[str(path)] = [stat.st_size, stat.st_mtime_ns] if stat else None
    return stamps


def _intact(directory: Path) -> bool:
    """Return whether every file in sizes.json exists with its size."""
    try:
        sizes = json.loads((directory / _SIZES).read_text())
    except (OSError, ValueError):
        # Missing, or cut by a crash: run the stage again.
        return False
    return all(
        (directory / name).is_file() and (directory / name).stat().st_size == size
        for name, size in sizes.items()
    )


def _cached(db: Connection, name: str, key: str, root: Path) -> Execution | None:
    """Return the latest done execution of `name` with this key.

    Its outputs must all be there, as sizes.json lists them.
    """
    row = db.execute(
        "SELECT id, dir FROM executions "
        "WHERE stage = ? AND key = ? AND status = 'done' ORDER BY id DESC LIMIT 1",
        (name, key),
    ).fetchone()
    if row is None:
        return None
    execution = Execution(row[0], root / row[1])
    outputs = all((execution.dir / out).exists() for out in STAGES[name].OUTPUTS)
    if outputs and _intact(execution.dir):
        return execution
    logger.info("%s: execution %d lost or cut its outputs, running again", name, row[0])
    return None


def _link(
    db: Connection, run_id: int, name: str, execution_id: int, cached: int
) -> None:
    """Record that the run used this execution of the stage."""
    db.execute(
        "INSERT INTO run_stages (run_id, stage, execution_id, cached) "
        "VALUES (?, ?, ?, ?)",
        (run_id, name, execution_id, cached),
    )


def _store(db: Connection, ctx: Context) -> None:
    """Insert the rows the stage queued with `ctx.write`.

    Not `to_sql`: it would commit.
    """
    for table, frame in ctx.pending:
        columns = ["execution_id", *frame.columns]
        names = ", ".join(f'"{column}"' for column in columns)
        marks = ", ".join(f":{column}" for column in columns)
        rows = frame.assign(execution_id=ctx.execution_id).to_dict("records")
        db.executemany(f"INSERT INTO {table} ({names}) VALUES ({marks})", rows)


def _execute(
    db: Connection,
    run_id: int,
    name: str,
    cfg: DictConfig,
    inputs: dict[str, Execution],
    *,
    force: bool,
) -> Execution:
    """Return the cached or new execution of stage `name`."""
    module = STAGES[name]
    values = _config_slice(name, cfg)
    ids = {stage: execution.id for stage, execution in inputs.items()}
    # Input execution ids, not keys: any upstream re-execution
    # invalidates this one.
    key = hashlib.sha256(
        json.dumps(
            {"config": values, "inputs": ids, "files": _file_stamps(name, cfg)},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    hit = None if force else _cached(db, name, key, Path(cfg.root))
    if hit is not None:
        _link(db, run_id, name, hit.id, 1)
        db.commit()
        logger.info("%s: cached, execution %d", name, hit.id)
        return hit

    execution_id = db.execute(
        "INSERT INTO executions (stage, key, config, inputs, dir, status, started_at) "
        "VALUES (?, ?, ?, ?, '', 'running', ?)",
        (name, key, json.dumps(values), json.dumps(ids), now()),
    ).lastrowid
    # Relative, so that a root can be moved.
    relative_dir = f"artifacts/{name}/{execution_id}"
    out = Path(cfg.root) / relative_dir
    db.execute(
        "UPDATE executions SET dir = ? WHERE id = ?", (relative_dir, execution_id)
    )
    _link(db, run_id, name, execution_id, 0)
    db.commit()

    started = time.perf_counter()
    try:
        # Remove a folder left over from a deleted DB.
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        stage_cfg = OmegaConf.create(values)
        OmegaConf.set_readonly(stage_cfg, True)
        OmegaConf.set_struct(stage_cfg, True)
        runtime = OmegaConf.create(OmegaConf.to_container(cfg.runtime))
        OmegaConf.set_readonly(runtime, True)
        ctx = Context(
            cfg=stage_cfg,
            inputs=inputs,
            out=out,
            db=db,
            execution_id=execution_id,
            runtime=runtime,
        )
        logger.info("%s: running execution %d in %s", name, execution_id, out)
        module.run(ctx)
        missing = [path for path in module.OUTPUTS if not (out / path).exists()]
        if missing:
            raise RuntimeError(f"{name} did not write its declared outputs: {missing}")
        sizes = {
            str(file.relative_to(out)): file.stat().st_size
            for file in sorted(out.rglob("*"))
            if file.is_file()
        }
        (out / _SIZES).write_text(json.dumps(sizes))
        # One short transaction: the DB is write-locked only here.
        _store(db, ctx)
        duration = time.perf_counter() - started
        db.execute(
            "UPDATE executions SET status = 'done', duration_s = ? WHERE id = ?",
            (duration, execution_id),
        )
        db.commit()
    except BaseException as error:  # Ctrl-C leaves a failed row too.
        db.rollback()
        db.execute(
            "UPDATE executions "
            "SET status = 'failed', error = ?, duration_s = ? WHERE id = ?",
            (repr(error), time.perf_counter() - started, execution_id),
        )
        db.commit()
        # Ctrl-C and a job cancel are not failures of the stage.
        if isinstance(error, Exception):
            logger.error(
                "%s: execution %d failed, its outputs are in %s; if an upstream "
                "file is damaged, re-run that stage with --force",
                name,
                execution_id,
                out,
            )
        raise
    logger.info("%s: done in %.1f s", name, duration)
    return Execution(execution_id, out)


def _finish(db: Connection, run_id: int, status: str, error: str | None = None) -> None:
    """Mark the run `done` or `failed` and commit."""
    db.execute(
        "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, error, now(), run_id),
    )
    db.commit()


def run(targets: Sequence[str], cfg: DictConfig, *, force: bool) -> int:
    """Run `targets` and the stages they read, reusing the cached ones.

    Args:
        targets: The stages to run.
        cfg: The validated config.
        force: Run the targets again even when cached; the stages they
            read stay cached.

    Returns:
        The id of the run.

    Raises:
        ValueError: There is no target, or one is not a stage.
        RuntimeError: A stage did not write its declared outputs.
    """
    order = plan(targets, cfg)
    root = Path(cfg.root)
    with closing(connect(root / DB_NAME)) as db:
        # Before our own run exists: only dead runs are `running`.
        reap(root)
        with (root / RUNS_LOCK).open("a") as lock:
            # Held until the run ends, or dies.
            fcntl.flock(lock, fcntl.LOCK_SH)
            run_id = db.execute(
                "INSERT INTO runs (targets, config, status, started_at) "
                "VALUES (?, ?, 'running', ?)",
                (json.dumps(targets), json.dumps(OmegaConf.to_container(cfg)), now()),
            ).lastrowid
            db.commit()
            done: dict[str, Execution] = {}
            try:
                for name in order:
                    inputs = {stage: done[stage] for stage in _stage_inputs(name, cfg)}
                    done[name] = _execute(
                        db, run_id, name, cfg, inputs, force=force and name in targets
                    )
            except BaseException as error:
                _finish(db, run_id, "failed", repr(error))
                raise
            _finish(db, run_id, "done")
            logger.info("run %d done", run_id)
    return run_id
