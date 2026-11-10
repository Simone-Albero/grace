"""Stage registry, in pipeline order."""

from __future__ import annotations

__all__ = ["STAGES"]

import sqlite3
from collections.abc import Callable

from src.config import DatasetConfig
from stages import clean, complexity, prepare, regions

StageFn = Callable[[DatasetConfig, sqlite3.Connection, int], None]

STAGES: dict[str, StageFn] = {
    "clean": clean.run,
    "prepare": prepare.run,
    "regions": regions.run,
    "complexity": complexity.run,
}
