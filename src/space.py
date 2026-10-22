"""Space (numerics, weighted one-hot) and its scaling helpers."""

__all__ = ["Space", "l2_normalize", "scale_for_metric"]

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-8


@dataclass(frozen=True)
class Space:
    """The euclidean space of geometry: numerics, one-hot categoricals.

    Numerics stay as prepared.

    Attributes:
        num_cols: The numerical columns, which come first as they are.
        kept: By categorical column, the train codes with a slot of
            their own; the rest share one.
        cat_cost: The squared distance between two rows differing on
            one categorical column. Numerics are IQR-scaled, so 1 weighs
            a mismatch like one IQR step.
    """

    num_cols: list[str]
    kept: dict[str, list[int]]
    cat_cost: float

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        num_cols: list[str],
        cat_cols: list[str],
        top_k: int,
        cat_cost: float,
    ) -> "Space":
        """Fit the `top_k` most frequent codes of each categorical.

        Args:
            train: The train rows, with the prepared codes.
            num_cols: The numerical column names.
            cat_cols: The categorical column names.
            top_k: The codes per categorical column with a slot of
                their own.
            cat_cost: The squared distance between two rows differing
                on one categorical column.

        Returns:
            The fitted space.
        """
        kept = {
            col: [int(code) for code in train[col].value_counts().index[:top_k]]
            for col in cat_cols
        }
        return cls(list(num_cols), kept, cat_cost)

    def columns(self) -> list[str]:
        """Return the names of the coordinates `embed` returns.

        Returns:
            The names, in coordinate order.
        """
        columns = list(self.num_cols)
        for col, codes in self.kept.items():
            columns += [f"{col}={code}" for code in codes] + [f"{col}=other"]
        return columns

    def embed(self, rows: pd.DataFrame) -> np.ndarray:
        """Return the rows as points of the space.

        Args:
            rows: The rows, with the prepared columns.

        Returns:
            The points, shape (n, len(columns())): float64 without
            categoricals, else float32.
        """
        numerics = rows[self.num_cols].to_numpy(dtype=np.float64)
        if not self.kept:
            return numerics
        weight = np.sqrt(self.cat_cost / 2.0)
        # float32: the one-hot blocks are most of the matrix, and a
        # class of millions of rows has to fit in memory next to its
        # copies.
        blocks = [numerics.astype(np.float32)]
        for col, codes in self.kept.items():
            slot = (
                rows[col]
                .map({code: i for i, code in enumerate(codes)})
                .fillna(len(codes))
                .to_numpy(dtype=np.int64)
            )
            block = np.zeros((len(rows), len(codes) + 1), dtype=np.float32)
            block[np.arange(len(rows)), slot] = weight
            blocks.append(block)
        return np.hstack(blocks)


def l2_normalize(points: np.ndarray) -> np.ndarray:
    """Return the rows scaled to unit L2 norm.

    Euclidean distance on unit vectors is monotone in the cosine
    distance.

    Args:
        points: The points, shape (n, d).

    Returns:
        The normalized points; a row of norm under 1e-8 is divided by
        1e-8 instead.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, _EPS)


def scale_for_metric(points: np.ndarray, metric: str) -> np.ndarray:
    """Return the points as the graph measures them.

    Args:
        points: The points, shape (n, d).
        metric: `euclidean` or `cosine`.

    Returns:
        The unit rows under cosine, else the points unchanged.
    """
    return l2_normalize(points) if metric == "cosine" else points
