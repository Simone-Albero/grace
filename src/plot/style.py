"""Plot style: STYLE_PATH, palette and colors, descriptor labels."""

__all__ = [
    "MUTED_COLOR",
    "NEUTRAL_COLOR",
    "PALETTE",
    "STYLE_PATH",
    "descriptor_label",
    "extended_palette",
]

from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH: Path = Path(__file__).parent / "default.mplstyle"

PALETTE: list[str] = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC78BC",
    "#CA9161",
    "#949494",
    "#ECE133",
    "#56B4E9",
]

MUTED_COLOR: str = "#777777"
NEUTRAL_COLOR: str = "#cccccc"


def extended_palette(n_colors: int) -> list[str]:
    """Return `n_colors` distinct hex colors, PALETTE first.

    Args:
        n_colors: The number of colors; tab20 supplies the overflow and
            the pool repeats past it.

    Returns:
        The hex colors.
    """
    if n_colors <= len(PALETTE):
        return PALETTE[:n_colors]
    tab20_rgb = plt.get_cmap("tab20").colors
    tab20_hex = [
        f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"
        for red, green, blue in tab20_rgb
    ]
    pool = PALETTE + [color for color in tab20_hex if color not in PALETTE]
    return (pool * ((n_colors // len(pool)) + 1))[:n_colors]


_MEASURE_LABEL = {
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "n1": "N1",
    "n2": "N2",
    "n3": "N3",
    "n4": "N4",
    "network_density": "Density",
    "cls_coef": "ClsCoef",
    "hub": "Hubs",
    "t2": "T2",
    "t3": "T3",
    "t4": "T4",
    "max_dispersion": r"$\delta_{\max}$",
    "p95_dispersion": r"$\delta_{95}$",
    "dist_to_nearest_rival": r"$\delta_{\mathrm{near}}$",
    "p5_silhouette": r"$s_{5}$",
    "frac_at_risk": r"$s^{-}$",
}


def descriptor_label(feature: str) -> str:
    """Return the display notation of a descriptor name.

    Args:
        feature: A `{region|class}_{measure}[_agg]` descriptor name.

    Returns:
        The measure label, with the class level and the aggregate in
        parentheses.
    """
    level, _, measure = feature.partition("_")
    agg = None
    for suffix in ("_mean", "_max", "_min"):
        if measure.endswith(suffix):
            measure, agg = measure[: -len(suffix)], suffix[1:]
            break
    label = _MEASURE_LABEL.get(measure, measure)
    qualifiers = ["class"] if level == "class" else []
    if agg:
        qualifiers.append(agg)
    return f"{label} ({', '.join(qualifiers)})" if qualifiers else label
