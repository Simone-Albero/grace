"""The plot types the render stages draw, and their label helpers."""

__all__ = [
    "bar_plot",
    "confusion_matrix_plot",
    "line_plot",
    "predicted_vs_observed_plot",
    "strip_box_plot",
    "violin_plot",
]

from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.plot.style import MUTED_COLOR, NEUTRAL_COLOR, PALETTE, extended_palette


def _apply_labels(
    ax: Axes,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Set the non-empty labels, as setting "" could move the layout."""
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)


def _format_value(value: float | None, *, kind: str) -> str:
    """Return a value as text, or '-' if it is missing.

    A score has three decimals and a proportion two.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.3f}" if kind == "score" else f"{value:.2f}"


def _gradient(base: str, n_colors: int) -> list[str]:
    """Return `n_colors` hex colors from light to dark `base`."""
    base_rgb = np.array(mcolors.to_rgb(base))
    white = np.ones(3)
    weights = 0.25 + 0.75 * np.arange(n_colors) / max(n_colors - 1, 1)
    return [
        mcolors.to_hex(weight * base_rgb + (1 - weight) * white) for weight in weights
    ]


def bar_plot(
    labels: Sequence[str],
    values: Sequence[float] | np.ndarray,
    *,
    y_label: str,
    colors: Sequence[str] | None = None,
    top_k: int | None = None,
    annotate_values: bool = True,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw vertical bars in the given order, or the `top_k` largest.

    Args:
        labels: The bar names.
        values: The bar heights.
        y_label: The y axis label.
        colors: One color per bar; by default a light-to-dark gradient
            of the first palette color.
        top_k: Keep only the largest bars, in descending order.
        annotate_values: Whether to print each height above its bar.
        ylim: The y axis limits; automatic by default.
        figsize: The figure size in inches; by default it grows with
            the bar count.

    Returns:
        The figure.
    """
    items = list(zip(labels, values))
    if top_k is not None:
        items.sort(key=lambda item: item[1], reverse=True)
        items = items[:top_k]

    plot_labels = [str(name) for name, _ in items]
    plot_values = np.array([value for _, value in items], dtype=float)
    n_bars = len(plot_labels)

    if colors is not None:
        bar_color = list(colors)
    elif n_bars > 0:
        bar_color = _gradient(PALETTE[0], n_bars)
    else:
        bar_color = PALETTE[0]

    if figsize is None:
        figsize = (max(8, n_bars * 0.5 + 1.5), 6)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        plot_labels,
        plot_values,
        color=bar_color,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel(y_label)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if annotate_values and n_bars:
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


def _draw_box(ax: Axes, position: int, values: np.ndarray) -> None:
    """Draw a quartile box, the median and whiskers at `position`."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    whisker_low = float(np.min(values))
    whisker_high = float(np.max(values))

    box_half = 0.05
    ax.add_patch(
        plt.Rectangle(
            (position - box_half, q1),
            box_half * 2,
            q3 - q1,
            facecolor="white",
            edgecolor=MUTED_COLOR,
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.plot(
        [position - box_half, position + box_half],
        [med, med],
        color=MUTED_COLOR,
        linewidth=1.8,
        zorder=5,
    )
    for y0, y1 in ((whisker_low, q1), (q3, whisker_high)):
        ax.plot(
            [position, position],
            [y0, y1],
            color=MUTED_COLOR,
            linewidth=0.8,
            zorder=4,
        )


def violin_plot(
    categories: np.ndarray,
    values: np.ndarray,
    *,
    category_order: Sequence[object],
    x_label: str,
    y_label: str,
) -> Figure:
    """Draw a violin with an inner box (quartiles, median) per category.

    Args:
        categories: The category of every value.
        values: The values to draw.
        category_order: The categories, left to right; one with fewer
            than two distinct values is left empty.
        x_label: The x axis label.
        y_label: The y axis label.

    Returns:
        The figure.
    """
    categories = np.asarray(categories)
    values = np.asarray(values)

    ordered_categories = list(category_order)
    n_categories = len(ordered_categories)
    category_colors = extended_palette(n_categories)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    for position, (category, color) in enumerate(
        zip(ordered_categories, category_colors)
    ):
        mask = categories == category
        category_values = values[mask]
        if category_values.size < 2 or np.unique(category_values).size < 2:
            continue

        parts = ax.violinplot(
            category_values,
            positions=[position],
            showmedians=False,
            showextrema=False,
            widths=0.85,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        _draw_box(ax, position, category_values)

    ax.set_xticks(range(n_categories))
    ax.set_xticklabels([str(category) for category in ordered_categories])

    _apply_labels(ax, x_label, y_label)
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
                text = _format_value(matrix[i, j], kind="proportion")
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


def _scatter_panel(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: np.ndarray,
    cmap: str,
    vmax: float | None,
    notes: Mapping[str, float],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Draw the y = x line, colored points and a `notes` box."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":")

    ax.scatter(
        x,
        y,
        c=color,
        cmap=cmap,
        s=22,
        alpha=0.7,
        edgecolors="none",
        vmin=0.0,
        vmax=vmax,
    )

    text = "\n".join(
        f"{name} = {_format_value(value, kind='score')}"
        for name, value in notes.items()
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": NEUTRAL_COLOR,
            "boxstyle": "round,pad=0.3",
        },
    )

    ax.grid(True, alpha=0.15, linewidth=0.5)
    _apply_labels(ax, x_label, y_label, title)


def predicted_vs_observed_plot(
    observed: np.ndarray,
    predicted: Mapping[str, tuple[np.ndarray, Mapping[str, float]]],
    *,
    figsize: tuple[float, float],
) -> Figure:
    """Draw predicted against observed failure rates per predictor.

    Args:
        observed: The observed failure rate of each region.
        predicted: For each predictor name, its predicted rates and the
            scores noted in its panel.
        figsize: The figure size in inches.

    Returns:
        The figure, one panel per predictor, points colored by squared
        error on a shared scale.
    """
    cmap = "RdYlGn_r"
    errors = {name: (rates - observed) ** 2 for name, (rates, _) in predicted.items()}
    all_errors = np.concatenate(list(errors.values()))
    finite = np.isfinite(all_errors)
    vmax = float(np.quantile(all_errors[finite], 0.95)) if finite.any() else None

    fig, axes = plt.subplots(
        1, len(predicted), figsize=figsize, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for i, (ax, (name, (rates, notes))) in enumerate(zip(axes, predicted.items())):
        _scatter_panel(
            ax,
            observed,
            rates,
            color=errors[name],
            cmap=cmap,
            vmax=vmax,
            notes=notes,
            x_label="Observed failure rate",
            y_label="Predicted failure rate" if i == 0 else "",
            title=name,
        )

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax if vmax is not None else 1.0)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=list(axes), fraction=0.046, pad=0.04)
    colorbar.set_label("Squared error")
    return fig


def strip_box_plot(
    groups: Sequence[str],
    values: Sequence[np.ndarray],
    sizes: Sequence[np.ndarray],
    *,
    x_label: str,
    y_label: str,
    size_label: str,
) -> Figure:
    """Draw a box and jittered points per group, sized by `sizes`.

    Args:
        groups: The group names, left to right.
        values: The values of each group.
        sizes: The size of each value, shaped like `values`; it sets
            the area of its point.
        x_label: The x axis label.
        y_label: The y axis label.
        size_label: The title of the size legend.

    Returns:
        The figure.
    """
    n_groups = len(groups)
    colors = extended_palette(n_groups)
    point_values = [np.asarray(value, dtype=float) for value in values]
    point_sizes = [np.asarray(size, dtype=float) for size in sizes]
    all_sizes = np.concatenate(point_sizes)
    top = float(all_sizes.max())

    def area(size: np.ndarray | float) -> np.ndarray | float:
        return 8 + 120 * size / top

    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * n_groups + 1.5), 3.8))
    ax.boxplot(
        point_values,
        positions=range(n_groups),
        widths=0.5,
        showfliers=False,
        boxprops={"color": MUTED_COLOR},
        medianprops={"color": MUTED_COLOR, "linewidth": 1.8},
        whiskerprops={"color": MUTED_COLOR},
        capprops={"color": MUTED_COLOR},
    )
    rng = np.random.default_rng(0)
    for i, (value, size, color) in enumerate(
        zip(point_values, point_sizes, colors, strict=True)
    ):
        jitter = rng.uniform(-0.18, 0.18, size=len(value))
        ax.scatter(
            i + jitter,
            value,
            s=area(size),
            color=color,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )

    marks = [float(all_sizes.min()), float(np.median(all_sizes)), top]
    handles = [
        ax.scatter([], [], s=area(mark), color=NEUTRAL_COLOR, label=f"{mark:.0f}")
        for mark in marks
    ]
    ax.legend(
        handles=handles,
        title=size_label,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    ax.set_xticks(range(n_groups), list(groups))
    if n_groups > 6:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax.grid(False, axis="x")
    _apply_labels(ax, x_label, y_label)
    return fig
