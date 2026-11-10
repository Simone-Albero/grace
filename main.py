"""Run a stage: python main.py --stage clean --dataset synthetic_test"""

from __future__ import annotations

__all__ = ["main", "parse_args"]

import argparse
from collections.abc import Sequence
from dataclasses import asdict

from src import db
from src.config import load_config
from stages import STAGES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: The arguments, without the program name; None reads
            sys.argv.

    Returns:
        stage (a key of STAGES), dataset (name or yaml path) and method
        (name or yaml path).
    """
    parser = argparse.ArgumentParser(description="Run a pipeline stage.")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--dataset", required=True, help="name or yaml path")
    parser.add_argument("--method", default="default", help="name or yaml path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage inside a logged run.

    Args:
        argv: The arguments, without the program name; None reads
            sys.argv.

    Returns:
        0; a failed stage marks its run "failed" and re-raises.
    """
    args = parse_args(argv)
    cfg = load_config(args.dataset, args.method)
    conn = db.connect(cfg.db_path)
    run_id = db.start_run(conn, args.stage, cfg.name, asdict(cfg))
    try:
        STAGES[args.stage](cfg, conn, run_id)
    except BaseException:
        db.finish_run(conn, run_id, "failed")
        raise
    db.finish_run(conn, run_id, "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
