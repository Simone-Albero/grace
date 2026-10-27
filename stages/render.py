"""Static figures of the classifier, the predictors and the regressor.

The regressor's figures need `tree`, `complexity` and `regress`: render
reads them, and draws those figures, only when `regressor` is true.
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
    line_plot,
    predicted_vs_observed_plot,
    violin_plot,
)
from src.plot.style import descriptor_label, extended_palette
from src.regressor import leaf_descriptors
from src.scores import rate_scores
from stages import Context

CONFIG = ("figure_format", "regressor")
FILES = ()
OUTPUTS = ("figures",)
_BASE_INPUTS = ("prepare", "classify", "evaluate")
# The number of most important features that get a descriptor violin.
_TOP_DESCRIPTORS = 4
_PREDICTORS = (
    ("tree", "Tree"),
    ("tree_leaf", "Tree leaf"),
    ("regressor", "RF regressor"),
    ("val_rate", "Val rate"),
    ("confidence", "Confidence"),
    ("binned", "Binned"),
    ("gbm", "GBM"),
)


def inputs(cfg: DictConfig) -> tuple[str, ...]:
    """Return the stages render reads under this config.

    Args:
        cfg: The config; `regressor` adds tree, complexity and regress.

    Returns:
        The input stage names.
    """
    if cfg.regressor:
        return (*_BASE_INPUTS, "tree", "complexity", "regress")
    return _BASE_INPUTS


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


def _test_figure(ctx: Context, estimates: pd.DataFrame) -> dict[str, Figure]:
    """Draw the regressor's leaf estimates against test rates."""
    tested = ctx.read("leaf_tests", "evaluate").set_index("node_id")
    tested = tested[tested["n_test"] > 0]
    predicted = estimates.loc[tested.index, "regressor"].to_numpy(dtype=float)
    observed = tested["test_rate"].to_numpy(dtype=float)
    score = rate_scores(predicted, observed)
    return {
        "regress/predicted_vs_test": predicted_vs_observed_plot(
            observed,
            {
                "RF regressor": (
                    predicted,
                    {"Spearman": score["spearman"], "MAE": score["mae"]},
                )
            },
            figsize=(5.2, 4.2),
        )
    }


def _regressor_figures(ctx: Context) -> dict[str, Figure]:
    """Draw the regressor figures: leaf rates and descriptors."""
    scores = ctx.read("metrics", "regress").set_index("name")["value"]
    nodes = ctx.read("tree_nodes", "tree")
    leaves = nodes[nodes["status"] == "leaf"].set_index("node_id")
    val_leaves = leaves[leaves["n_val"] > 0]
    val_rate = val_leaves["n_val_errors"] / val_leaves["n_val"]
    estimates = ctx.read("estimates", "regress").set_index("node_id")
    importances = ctx.read("importances", "regress").set_index("feature")["importance"]

    predicted = estimates.loc[val_rate.index, "regressor"].to_numpy(dtype=float)
    notes = {"Spearman": scores["oof_spearman"], "MAE": scores["oof_mae"]}
    figures = {
        "regress/predicted_vs_val": predicted_vs_observed_plot(
            val_rate.to_numpy(),
            {"RF regressor": (predicted, notes)},
            figsize=(5.2, 4.2),
        ),
        **_test_figure(ctx, estimates),
        "regress/feature_importances": bar_plot(
            [descriptor_label(feature) for feature in importances.index],
            importances.to_numpy(),
            top_k=10,
            annotate_values=False,
            y_label="Importance",
            figsize=(5.6, 3.9),
        ),
    }

    descriptors = leaf_descriptors(
        ctx.read("complexity", "complexity"), leaves["class_id"]
    ).loc[val_rate.index]
    for feature in importances.nlargest(_TOP_DESCRIPTORS).index:
        violin = _violin_by_rate(descriptors, val_rate, feature)
        if violin is not None:
            figures[f"descriptors/{feature}"] = violin
    return figures


def _predictor_figures(ctx: Context) -> dict[str, Figure]:
    """Draw each predictor's per-sample MSE ratio and Spearman."""
    metrics = ctx.read("metrics", "evaluate").set_index("name")["value"]
    shown = [(key, label) for key, label in _PREDICTORS if f"{key}_mse" in metrics]
    labels = [label for _, label in shown]
    colors = extended_palette(len(shown))
    return {
        "predictors/mse_ratio": bar_plot(
            labels,
            [metrics[f"{key}_mse_ratio"] for key, _ in shown],
            colors=colors,
            y_label="MSE / MSE of the val rate",
        ),
        "predictors/spearman": bar_plot(
            labels,
            [metrics[f"{key}_spearman"] for key, _ in shown],
            colors=colors,
            y_label="Spearman with the failures",
        ),
    }


def _figures(ctx: Context) -> dict[str, Figure]:
    """Draw the classifier, predictor and regressor figures."""
    figures = {**_classification_figures(ctx), **_predictor_figures(ctx)}
    if ctx.cfg.regressor:
        figures.update(_regressor_figures(ctx))
    return figures


def run(ctx: Context) -> None:
    """Draw the figures and list the files in `figures`.

    Args:
        ctx: The stage context.
    """
    ctx.save_figures(lambda: _figures(ctx))
