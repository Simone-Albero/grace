"""Static figures of the classifier and the predictors.

Files: figures/<group>/<name>.<figure_format>. DB rows written:
`figures` (name and file, relative to the execution folder) and metric
n_figures.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from src.plot.charts import bar_plot, confusion_matrix_plot, line_plot
from src.plot.style import extended_palette
from stages import Context

CONFIG = ("figure_format",)
INPUTS = ("prepare", "classify", "evaluate")
FILES = ()
OUTPUTS = ("figures",)
_PREDICTORS = (
    ("tree", "Tree"),
    ("tree_leaf", "Tree leaf"),
    ("val_rate", "Val rate"),
    ("confidence", "Confidence"),
    ("binned", "Binned"),
    ("gbm", "GBM"),
)


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


def _predictor_figures(ctx: Context) -> dict[str, Figure]:
    """Draw each predictor's per-sample MSE ratio and Spearman."""
    metrics = ctx.read("metrics", "evaluate").set_index("name")["value"]
    labels = [label for _, label in _PREDICTORS]
    colors = extended_palette(len(_PREDICTORS))
    return {
        "predictors/mse_ratio": bar_plot(
            labels,
            [metrics[f"{key}_mse_ratio"] for key, _ in _PREDICTORS],
            colors=colors,
            y_label="MSE / MSE of the val rate",
        ),
        "predictors/spearman": bar_plot(
            labels,
            [metrics[f"{key}_spearman"] for key, _ in _PREDICTORS],
            colors=colors,
            y_label="Spearman with the failures",
        ),
    }


def _figures(ctx: Context) -> dict[str, Figure]:
    """Draw the classifier and predictor figures."""
    return {**_classification_figures(ctx), **_predictor_figures(ctx)}


def run(ctx: Context) -> None:
    """Draw the figures and list the files in `figures`.

    Args:
        ctx: The stage context.
    """
    ctx.save_figures(lambda: _figures(ctx))
