"""Stage registry, in pipeline order."""

from __future__ import annotations

__all__ = ["MODEL_STAGES", "STAGES"]

import sqlite3
from collections.abc import Callable

from src.config import DatasetConfig
from stages import complexity, failures, regions, regress, split, train

StageFn = Callable[[DatasetConfig, sqlite3.Connection, int], None]

STAGES: dict[str, StageFn] = {
    "split": split.run,
    "regions": regions.run,
    "complexity": complexity.run,
    "train": train.run,
    "failures": failures.run,
    "regress": regress.run,
}
MODEL_STAGES = ("train", "failures", "regress")  # need --model
