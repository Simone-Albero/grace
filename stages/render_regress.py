"""Static figures of the failure regressor against MCP and ATC.

Files: figures/<group>/<name>.<figure_format>. DB rows written:
`figures` (name and file, relative to the execution folder) and metric
n_figures.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import pandas as pd
from matplotlib.figure import Figure

from src.plot.charts import (
    bar_plot,
    dual_axis_bar_plot,
    predicted_vs_observed_plot,
    violin_plot,
)
from src.plot.style import PALETTE, descriptor_label
from src.regressor import region_descriptors
from stages import Context

CONFIG = ("figure_format",)
INPUTS = ("regions", "complexity", "failures", "baselines", "regress")
FILES = ()
OUTPUTS = ("figures",)
# The number of most important features that get a descriptor violin.
_TOP_DESCRIPTORS = 4
_PREDICTORS = (("regressor", "RF regressor"), ("mcp", "MCP"), ("atc", "ATC"))


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


def _figures(ctx: Context) -> dict[str, Figure]:
    """Draw the regressor figures: rates, scores and descriptors."""
    scores = pd.concat(
        [ctx.read("metrics", "regress"), ctx.read("metrics", "baselines")]
    )
    scores = scores.set_index("name")["value"]
    table = ctx.read("baselines", "baselines").set_index("region_id")
    estimates = ctx.read("estimates", "regress").set_index("region_id")
    observed = ctx.read("failures", "failures").set_index("region_id")["failure_rate"]
    used = table.index[table["used"] == 1]
    rate = observed.loc[used]
    importances = ctx.read("importances", "regress").set_index("feature")["importance"]

    predicted = estimates.loc[used, "regressor"].to_numpy(dtype=float)
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
    ).loc[used]
    for feature in importances.nlargest(_TOP_DESCRIPTORS).index:
        violin = _violin_by_rate(descriptors, rate, feature)
        if violin is not None:
            figures[f"descriptors/{feature}"] = violin
    return figures


def run(ctx: Context) -> None:
    """Draw the regressor figures and list the files in `figures`.

    Args:
        ctx: The stage context.
    """
    ctx.save_figures(lambda: _figures(ctx))
