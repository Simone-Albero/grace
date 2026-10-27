"""The dashboard server: `python -m dashboard --root <root>`."""

__all__ = ["main"]

import argparse
import sqlite3
import sys
from pathlib import Path

import uvicorn

from dashboard.app import create_app
from pipeline.db import DB_NAME, explain


def main() -> None:
    """Serve the dashboard of `<root>/grace.db` on 127.0.0.1."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", default="resources", help="experiment root")
    parser.add_argument(
        "--port", type=int, default=8000, help="port to listen on (default: 8000)"
    )
    args = parser.parse_args()
    path = Path(args.root) / DB_NAME
    try:
        app = create_app(path)
    except OSError as error:
        sys.exit(f"error: {error}")
    except sqlite3.Error as error:
        sys.exit(f"error: {explain(error, path)}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
