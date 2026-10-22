"""The command line: `python -m stages <target...> k=v...`."""

__all__ = ["main"]

import argparse
import logging
import signal
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from pipeline.config import load_config, sweep
from pipeline.db import DB_NAME, explain
from pipeline.runner import run

logger = logging.getLogger(__name__)


def _terminate(signum: int, frame: object) -> NoReturn:
    """Turn SIGTERM into SystemExit, so the runner fails the run."""
    raise SystemExit(f"terminated by signal {signum}")


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m stages",
        description="Run pipeline stages (and the stages they read) on one config. "
        "Run it from the repo root: paths in the config are relative to it.",
    )
    parser.add_argument(
        "items",
        nargs="+",
        metavar="TARGET|KEY=VALUE",
        help="stages to run, and config overrides such as data=nb15_v2 or seed=7",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="run once per value (repeatable: the cartesian product of every sweep); "
        "values are split on the commas outside square brackets, so lists sweep "
        "too: 'key=[1,2],[3]' (quoted for the shell)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run the targets even when cached (after a code change)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the stages and config overrides named on the command line.

    Every sweep combination is validated before the first one runs.

    Args:
        argv: The arguments; `None` reads sys.argv.

    Raises:
        ValueError: An override, a sweep or a target is invalid.
        RuntimeError: The DB cannot be used; the message names the
            file and, when known, the remedy.
    """
    signal.signal(signal.SIGTERM, _terminate)
    args = _parser().parse_args(argv)
    overrides = [item for item in args.items if "=" in item]
    targets = [item for item in args.items if "=" not in item]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    configs = [(combo, load_config(overrides + combo)) for combo in sweep(args.sweep)]
    for i, (combo, cfg) in enumerate(configs, start=1):
        if len(configs) > 1:
            logger.info("sweep %d/%d: %s", i, len(configs), " ".join(combo))
        try:
            run(targets, cfg, force=args.force)
        except sqlite3.Error as error:
            message = explain(error, Path(cfg.root) / DB_NAME)
            raise RuntimeError(message) from error


if __name__ == "__main__":
    main()
