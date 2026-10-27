"""What the dashboard reads from the artifact files, not the DB."""

__all__ = ["confusion"]

from pathlib import Path

import numpy as np
import pandas as pd


def confusion(
    prepare_dir: Path, classify_dir: Path, n_classes: int
) -> np.ndarray | None:
    """Count the test rows by true and predicted class.

    Rows pair by position, as in the stages: prepare's test file holds
    the labels, classify's predictions say `row` within the split.

    Args:
        prepare_dir: The folder of a prepare execution.
        classify_dir: The folder of a classify execution.
        n_classes: The number of classes.

    Returns:
        The counts, true class by rows and predicted by columns; None if
        a file is missing or the two do not have the same test length.
    """
    labels_file = prepare_dir / "test.parquet"
    predictions_file = classify_dir / "predictions.parquet"
    if not (labels_file.is_file() and predictions_file.is_file()):
        return None
    true = pd.read_parquet(labels_file, columns=["label"])["label"].to_numpy()
    predictions = pd.read_parquet(predictions_file, columns=["split", "row", "y_pred"])
    test = predictions[predictions["split"] == "test"].sort_values("row")
    if len(test) != len(true):
        return None
    predicted = test["y_pred"].to_numpy()
    cells = n_classes * true.astype(np.int64) + predicted.astype(np.int64)
    return np.bincount(cells, minlength=n_classes**2).reshape(n_classes, n_classes)
