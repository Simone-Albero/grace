"""Plot style: STYLE_PATH and the color palette."""

__all__ = [
    "PALETTE",
    "STYLE_PATH",
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
