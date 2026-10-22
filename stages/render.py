"""Static figures of the classifier and its MCP/ATC baselines.

Files: figures/<group>/<name>.<figure_format>. DB rows written:
`figures` (name and file, relative to the execution folder) and metric
n_figures.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from src.plot.charts import (
    bar_plot,
    confusion_matrix_plot,
    line_plot,
    predicted_vs_observed_plot,
    strip_box_plot,
)
from src.plot.style import extended_palette
from stages import Context

CONFIG = ("figure_format",)
INPUTS = ("prepare", "regions", "classify", "failures", "baselines")
FILES = ()
OUTPUTS = ("figures",)
_PREDICTORS = (("mcp", "MCP"), ("atc", "ATC"))


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
    table = ctx.read("baselines", "baselines").set_index("region_id")
    observed = ctx.read("failures", "failures").set_index("region_id")["failure_rate"]
    used = table.index[table["used"] == 1]
    rate = observed.loc[used]

    predicted = {}
    for key, label in _PREDICTORS:
        notes = {"Spearman": metrics[f"{key}_spearman"], "MAE": metrics[f"{key}_mae"]}
        predicted[label] = (table.loc[used, key].to_numpy(dtype=float), notes)
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


def _figures(ctx: Context) -> dict[str, Figure]:
    """Draw the classifier and baseline figures."""
    return {
        **_classification_figures(ctx),
        **_baseline_figures(ctx),
        "failures/rate_by_class": _failure_figure(ctx),
    }


def run(ctx: Context) -> None:
    """Draw the classifier and baseline figures, listed in `figures`.

    Args:
        ctx: The stage context.
    """
    ctx.save_figures(lambda: _figures(ctx))
