"""The stage contract: a stage's execution context and its DB access."""

__all__ = ["LABEL", "SPLITS", "Context", "Execution"]

import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.plot.style import STYLE_PATH

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
# The class column of every split.
LABEL = "label"


class Execution(NamedTuple):
    """A finished stage execution.

    Attributes:
        id: The execution id.
        dir: Its output folder.
    """

    id: int
    dir: Path


@dataclass(frozen=True)
class Context:
    """What a stage's run gets: config, inputs, output folder and DB.

    Attributes:
        cfg: Only the stage's config keys (CONFIG or config_keys),
            read-only.
        inputs: The executions of its INPUTS (or `inputs(cfg)`) this
            run uses.
        out: Its own output folder, created empty.
        db: The DB connection.
        execution_id: The id of this execution.
        runtime: The `runtime` section, read-only, in no cache key.
        pending: The rows queued with `write`, inserted at the end.
    """

    cfg: DictConfig
    inputs: dict[str, Execution]
    out: Path
    db: sqlite3.Connection
    execution_id: int
    runtime: DictConfig
    pending: list[tuple[str, pd.DataFrame]] = field(default_factory=list)

    def read(self, table: str, stage: str) -> pd.DataFrame:
        """Return the rows of `table` that an input stage wrote.

        Args:
            table: The table to read.
            stage: The input stage whose execution wrote the rows.

        Returns:
            The rows in insertion order, without `execution_id`.
        """
        frame = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE execution_id = ? ORDER BY rowid",
            self.db,
            params=(self.inputs[stage].id,),
        )
        return frame.drop(columns="execution_id")

    def features(self) -> tuple[list[str], list[str]]:
        """Return the numerical and categorical feature names.

        Returns:
            The names of the `features` table of split, numerical
            first, then categorical.
        """
        features = self.read("features", "split")
        num = features.loc[features["kind"] == "num", "name"].tolist()
        cat = features.loc[features["kind"] == "cat", "name"].tolist()
        return num, cat

    def split_rows(
        self, stage: str, file: str, split: str, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Return the rows of one split of a stage's parquet file.

        Args:
            stage: The input stage that wrote the file.
            file: The parquet file in the stage's output folder.
            split: The split to read.
            columns: The columns to read besides `row`; `None` reads
                every column.

        Returns:
            The rows in row order.
        """
        rows = pd.read_parquet(
            self.inputs[stage].dir / file,
            columns=None if columns is None else ["row", *columns],
            filters=[("split", "==", split)],
        )
        return rows.sort_values("row")

    def labels(self, split: str) -> np.ndarray:
        """Return the class ids of one split of prepare.

        Args:
            split: The split to read.

        Returns:
            The labels in row order.
        """
        return pd.read_parquet(
            self.inputs["prepare"].dir / f"{split}.parquet", columns=[LABEL]
        )[LABEL].to_numpy()

    def write(self, table: str, frame: pd.DataFrame) -> None:
        """Queue `frame` as rows of `table` for the runner to insert.

        Args:
            table: The table to insert into.
            frame: The rows, one column per table column except
                `execution_id`.
        """
        self.pending.append((table, frame))

    def save_metrics(self, values: Mapping[str, float]) -> None:
        """Store this execution's scalar results in `metrics`.

        Args:
            values: The metric values by name.
        """
        self.write(
            "metrics",
            pd.DataFrame(
                {
                    "name": list(values),
                    "value": [float(value) for value in values.values()],
                }
            ),
        )

    def save_figures(self, draw: Callable[[], dict[str, Figure]]) -> None:
        """Draw the figures, save them and list them in `figures`.

        Files go to figures/<name>.<figure_format> of the output folder;
        the stage must declare `figure_format` in CONFIG.

        Args:
            draw: Returns the figures by name, e.g.
                "predictors/mse_ratio"; called under the plot style.
        """
        extension = self.cfg.figure_format
        rows = []
        # A style context, not plt.style.use: no global style leaks.
        with plt.style.context(STYLE_PATH):
            figures = draw()
            try:
                for name, figure in figures.items():
                    path = self.out / "figures" / f"{name}.{extension}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(path, format=extension, bbox_inches="tight")
                    rows.append((name, str(path.relative_to(self.out))))
            finally:
                # Many stages run in one process: figures would pile up.
                for figure in figures.values():
                    plt.close(figure)
        self.write("figures", pd.DataFrame(rows, columns=["name", "file"]))
        self.save_metrics({"n_figures": len(rows)})
        logger.info("%d figures in %s", len(rows), self.out / "figures")
