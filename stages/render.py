"""Static figures of the classifier, its baselines and the regressor.

The regressor's figures need `complexity` and `regress`: render reads
them, and draws those figures, only when `regressor_figures` is true.
Files: figures/<group>/<name>.<figure_format>. DB rows written:
`figures` (name and file, relative to the execution folder) and metric
n_figures.
"""

__all__ = ["CONFIG", "FILES", "OUTPUTS", "inputs", "run"]

import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix

from src.plot.charts import (
    bar_plot,
    confusion_matrix_plot,
    dual_axis_bar_plot,
    line_plot,
    predicted_vs_observed_plot,
    strip_box_plot,
    violin_plot,
)
from src.plot.style import PALETTE, descriptor_label, extended_palette
from src.regressor import region_descriptors
from stages import Context

CONFIG = ("figure_format", "regressor_figures")
FILES = ()
OUTPUTS = ("figures",)
_BASE_INPUTS = ("prepare", "regions", "classify", "failures", "baselines")
# The number of most important features that get a descriptor violin.
_TOP_DESCRIPTORS = 4
_BASELINES = (("mcp", "MCP"), ("atc", "ATC"))
_PREDICTORS = (("regressor", "RF regressor"), *_BASELINES)


def inputs(cfg: DictConfig) -> tuple[str, ...]:
    """Return the stages render reads under this config.

    Args:
        cfg: The config; `regressor_figures` adds complexity and
            regress.

    Returns:
        The input stage names.
    """
    if cfg.regressor_figures:
        return (*_BASE_INPUTS, "complexity", "regress")
    return _BASE_INPUTS


def _used_rates(ctx: Context) -> tuple[pd.DataFrame, pd.Series]:
    """Return the baselines table and the used regions' failure rates.

    Returns:
        The `baselines` table by region, and the failure rate of each
        region that the predictors are scored on.
    """
    table = ctx.read("baselines", "baselines").set_index("region_id")
    observed = ctx.read("failures", "failures").set_index("region_id")["failure_rate"]
    return table, observed.loc[table.index[table["used"] == 1]]


def _classification_figures(ctx: Context) -> dict[str, Figure]:
    """Draw the confusion matrix, F1 per class and, for DL, the loss."""
    names = ctx.read("classes", "prepare").set_index("class_id")["name"]
    predictions = ctx.split_rows("classify", "predictions.parquet", "test", ["y_pred"])
    matrix = confusion_matrix(
        ctx.labels("test"), predictions["y_pred"], labels=names.index, normalize="true"
    )
    class_metrics = ctx.read("class_metrics", "classify")
    figures = {
        "classification/confusion_matrix": confusion_matrix_plot(
            matrix, class_names=names.tolist()
        ),
        "classification/f1_per_class": bar_plot(
            names.loc[class_metrics["class_id"]].tolist(),
            class_metrics["f1"].tolist(),
            colors=extended_palette(len(class_metrics)),
            y_label="F1",
            ylim=(0, 1),
        ),
    }
    history = ctx.read("history", "classify")
    if not history.empty:
        figures["classification/training_loss"] = line_plot(
            history["loss"].to_numpy(), y_label="loss"
        )
    return figures


def _baseline_figures(ctx: Context) -> dict[str, Figure]:
    """Draw the MCP and ATC rates against the observed failure rates."""
    metrics = ctx.read("metrics", "baselines").set_index("name")["value"]
    table, rate = _used_rates(ctx)

    predicted = {}
    for key, label in _BASELINES:
        notes = {"Spearman": metrics[f"{key}_spearman"], "MAE": metrics[f"{key}_mae"]}
        predicted[label] = (table.loc[rate.index, key].to_numpy(dtype=float), notes)
    return {
        "baselines/predicted_vs_observed": predicted_vs_observed_plot(
            rate.to_numpy(), predicted, figsize=(9.0, 4.2)
        )
    }


def _failure_figure(ctx: Context) -> Figure:
    """Draw the observed failure rate of each class's regions."""
    names = ctx.read("classes", "prepare").set_index("class_id")["name"]
    class_of = ctx.read("regions", "regions").set_index("region_id")["class_id"]
    failures = ctx.read("failures", "failures").set_index("region_id")
    failures = failures[failures["n_eval"] > 0]
    class_ids = sorted(class_of.loc[failures.index].unique())
    per_class = [
        failures[class_of.loc[failures.index] == class_id] for class_id in class_ids
    ]
    return strip_box_plot(
        [names.loc[class_id] for class_id in class_ids],
        [frame["failure_rate"].to_numpy(dtype=float) for frame in per_class],
        [frame["n_eval"].to_numpy(dtype=float) for frame in per_class],
        x_label="Class",
        y_label="Observed failure rate",
        size_label="Test rows",
    )


def _violin_by_rate(
    descriptors: pd.DataFrame, rate: pd.Series, feature: str
) -> Figure | None:
    """Draw the descriptor's distribution per failure-rate quartile.

    Returns:
        The figure, or None when the data is too thin for quartiles.
    """
    bins = pd.qcut(rate, q=4, duplicates="drop")
    values = descriptors[feature]
    valid = values.notna()
    if bins.nunique() < 2 or valid.sum() < 4:
        return None
    mean_rate = rate.groupby(bins, observed=True).mean()
    label_of = {
        bin_: f"G{i + 1}\n({mean:.2f})"
        for i, (bin_, mean) in enumerate(mean_rate.items())
    }
    return violin_plot(
        bins[valid].map(label_of).to_numpy(),
        values[valid].to_numpy(dtype=float),
        category_order=list(label_of.values()),
        x_label="Failure rate bin",
        y_label=descriptor_label(feature),
    )


def _regressor_figures(ctx: Context) -> dict[str, Figure]:
    """Draw the regressor figures: rates, scores and descriptors."""
    scores = pd.concat(
        [ctx.read("metrics", "regress"), ctx.read("metrics", "baselines")]
    ).set_index("name")["value"]
    _, rate = _used_rates(ctx)
    estimates = ctx.read("estimates", "regress").set_index("region_id")
    importances = ctx.read("importances", "regress").set_index("feature")["importance"]

    predicted = estimates.loc[rate.index, "regressor"].to_numpy(dtype=float)
    notes = {"Spearman": scores["regressor_spearman"], "MAE": scores["regressor_mae"]}
    keys = [key for key, _ in _PREDICTORS]
    figures = {
        "regress/predicted_vs_observed": predicted_vs_observed_plot(
            rate.to_numpy(),
            {"RF regressor": (predicted, notes)},
            figsize=(5.2, 4.2),
        ),
        "regress/scores": dual_axis_bar_plot(
            [label for _, label in _PREDICTORS],
            ("Spearman ρ", [scores[f"{key}_spearman"] for key in keys], PALETTE[0]),
            ("MAE", [scores[f"{key}_mae"] for key in keys], PALETTE[1]),
            x_label="Predictor",
        ),
        "regress/feature_importances": bar_plot(
            [descriptor_label(feature) for feature in importances.index],
            importances.to_numpy(),
            top_k=10,
            annotate_values=False,
            y_label="Importance",
            figsize=(5.6, 3.9),
        ),
    }

    descriptors = region_descriptors(
        ctx.read("complexity", "complexity"),
        ctx.read("regions", "regions").set_index("region_id")["class_id"],
    ).loc[rate.index]
    for feature in importances.nlargest(_TOP_DESCRIPTORS).index:
        violin = _violin_by_rate(descriptors, rate, feature)
        if violin is not None:
            figures[f"descriptors/{feature}"] = violin
    return figures


def _figures(ctx: Context) -> dict[str, Figure]:
    """Draw the classifier and baseline figures, and the regressor's."""
    figures = {
        **_classification_figures(ctx),
        **_baseline_figures(ctx),
        "failures/rate_by_class": _failure_figure(ctx),
    }
    if ctx.cfg.regressor_figures:
        figures.update(_regressor_figures(ctx))
    return figures


def run(ctx: Context) -> None:
    """Draw the figures and list the files in `figures`.

    Args:
        ctx: The stage context.
    """
    ctx.save_figures(lambda: _figures(ctx))
