"""RowsBy: the rows of every label with one sort."""

__all__ = ["RowsBy"]

import numpy as np


class RowsBy:
    """The rows of each distinct label, found with one sort.

    Labels are a region or a class. Each group's rows stay in their
    original order, so a reduction gives exactly what
    `values[labels == g]` would, at a cost of n log n plus one slice per
    group.

    Attributes:
        ids: The distinct labels, ascending.
    """

    def __init__(self, labels: np.ndarray) -> None:
        self._order = np.argsort(labels, kind="stable")
        self.ids, self._starts = np.unique(labels[self._order], return_index=True)
        self._ends = np.append(self._starts[1:], len(labels))

    def members(self, index: int) -> np.ndarray:
        """Return the rows of the `index`-th group, in original order.

        Args:
            index: The position of the group in `ids`.

        Returns:
            The row positions.
        """
        return self._order[self._starts[index] : self._ends[index]]

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Return the mean of `values` over each group.

        Args:
            values: One value per row.

        Returns:
            The means, in ascending label order.
        """
        ordered = np.asarray(values)[self._order]
        return np.array(
            [ordered[start:end].mean() for start, end in zip(self._starts, self._ends)]
        )
