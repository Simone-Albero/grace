"""The plot types the render stages draw, and their label helpers."""

__all__ = [
    "bar_plot",
    "confusion_matrix_plot",
    "line_plot",
]

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.plot.style import PALETTE


def _apply_labels(ax: Axes, x_label: str, y_label: str) -> None:
    """Set the non-empty labels, as setting "" could move the layout."""
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def bar_plot(
    labels: Sequence[str],
    values: Sequence[float] | np.ndarray,
    *,
    y_label: str,
    colors: Sequence[str],
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Draw vertical bars in the given order.

    Args:
        labels: The bar names.
        values: The bar heights.
        y_label: The y axis label.
        colors: One color per bar.
        ylim: The y axis limits; automatic by default.

    Returns:
        The figure.
    """
    items = list(zip(labels, values))
    plot_labels = [str(name) for name, _ in items]
    plot_values = np.array([value for _, value in items], dtype=float)
    n_bars = len(plot_labels)

    fig, ax = plt.subplots(figsize=(max(8, n_bars * 0.5 + 1.5), 6))

    ax.bar(
        plot_labels,
        plot_values,
        color=list(colors),
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel(y_label)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if n_bars:
        value_range = plot_values.max() - plot_values.min() or 1.0
        offset = max(value_range * 0.01, 1e-6)
        for i, value in enumerate(plot_values):
            ax.text(i, value + offset, f"{value:.3f}", ha="center", va="bottom")

    if ylim is not None:
        ax.set_ylim(ylim)

    return fig


def line_plot(values: np.ndarray, *, y_label: str) -> Figure:
    """Draw a series against its step index, e.g. a training loss.

    Args:
        values: The series, one value per step.
        y_label: The y axis label.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    series = np.asarray(values, dtype=float)
    ax.plot(np.arange(series.size), series, color=PALETTE[0], linewidth=1.2)

    _apply_labels(ax, x_label="step", y_label=y_label)
    ax.grid(True, axis="both", alpha=0.15, linewidth=0.5)
    return fig


def confusion_matrix_plot(cm: np.ndarray, *, class_names: Sequence[str]) -> Figure:
    """Draw a row-normalized confusion matrix.

    The cells are annotated up to 20 classes.

    Args:
        cm: The row-normalized confusion matrix, shape (n, n).
        class_names: The n class names, in row order.

    Returns:
        The figure.

    Raises:
        ValueError: If `cm` is not square or `class_names` does not
            match its size.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            "`cm` must be a square 2D array (n_classes x n_classes), "
            f"got shape {cm.shape}."
        )

    n_classes = cm.shape[0]
    if len(class_names) != n_classes:
        raise ValueError(
            f"`class_names` has {len(class_names)} names for a matrix of "
            f"{n_classes} classes."
        )
    display_names = [
        name if len(name) <= 18 else f"{name[:17]}…" for name in class_names
    ]
    side = max(5.0, 0.55 * n_classes + 1.5)

    matrix = cm.astype(float)
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(
        matrix,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    if n_classes <= 20:
        cell_fontsize = max(6.0, 12.0 - 0.35 * n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                value = matrix[i, j]
                text = f"{value:.2f}" if np.isfinite(value) else "-"
                if text.startswith("0."):
                    text = text[1:]
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if matrix[i, j] > 0.5 else "black",
                )

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(display_names)
    ax.grid(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion")

    _apply_labels(ax, x_label="Predicted label", y_label="True label")
    return fig
