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
_SCORES = {
    "mse_ratio": "MSE / MSE of the val rate",
    "binned_mse_ratio": "Binned MSE / binned MSE of the val rate",
    "auroc": "AUROC",
    "ap_lift": "Average precision / failure rate",
}


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
    """Draw each predictor's four scores against the test errors."""
    metrics = ctx.read("metrics", "evaluate").set_index("name")["value"]
    labels = [label for _, label in _PREDICTORS]
    colors = extended_palette(len(_PREDICTORS))
    return {
        f"predictors/{score}": bar_plot(
            labels,
            [metrics[f"{key}_{score}"] for key, _ in _PREDICTORS],
            colors=colors,
            y_label=y_label,
        )
        for score, y_label in _SCORES.items()
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
