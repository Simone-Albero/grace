"""Run a stage: python main.py --stage split --dataset synthetic_test"""

from __future__ import annotations

__all__ = ["main", "parse_args"]

import argparse
from collections.abc import Sequence
from dataclasses import asdict

from src import db
from src.config import load_config, model_names
from stages import MODEL_STAGES, STAGES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: The arguments, without the program name; None reads
            sys.argv.

    Returns:
        stage (a key of STAGES), dataset (name or yaml path), model
        (name or yaml path, required by MODEL_STAGES) and method (name
        or yaml path).
    """
    parser = argparse.ArgumentParser(description="Run a pipeline stage.")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--dataset", required=True, help="name or yaml path")
    parser.add_argument(
        "--model",
        help=f"name ({', '.join(model_names())}) or yaml path, "
        f"for {', '.join(MODEL_STAGES)}",
    )
    parser.add_argument("--method", default="default", help="name or yaml path")
    args = parser.parse_args(argv)
    if args.stage in MODEL_STAGES and args.model is None:
        parser.error(f"--stage {args.stage} requires --model")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage inside a logged run.

    Args:
        argv: The arguments, without the program name; None reads
            sys.argv.

    Returns:
        0; a failed stage marks its run "failed" and re-raises.
    """
    args = parse_args(argv)
    model = args.model if args.stage in MODEL_STAGES else None
    cfg = load_config(args.dataset, model, args.method)
    conn = db.connect(cfg.db_path)
    model_name = None if cfg.model is None else cfg.model.name
    run_id = db.start_run(conn, args.stage, cfg.name, model_name, asdict(cfg))
    try:
        STAGES[args.stage](cfg, conn, run_id)
    except BaseException:
        db.finish_run(conn, run_id, "failed")
        raise
    db.finish_run(conn, run_id, "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
