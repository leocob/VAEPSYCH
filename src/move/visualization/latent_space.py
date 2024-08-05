__all__ = ["plot_latent_space_with_cat", "plot_latent_space_with_con"]

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from move.core.typing import BoolArray, FloatArray
from move.visualization.style import (
    DEFAULT_DIVERGING_PALETTE,
    DEFAULT_PLOT_STYLE,
    DEFAULT_QUALITATIVE_PALETTE,
    color_cycle,
    style_settings,
)


def plot_latent_space_with_cat(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    feature_mapping: dict[str, Any],
    is_nan: BoolArray,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a 2D latent space together with a legend mapping the latent
    space to the values of a discrete feature.

    Args:
        latent_space:
            Embedding, a ND array with at least two dimensions.
        feature_name:
            Name of categorical feature
        feature_values:
            Values of categorical feature
        feature_mapping:
            Mapping of codes to categories for the categorical feature
        is_nan:
            Array of bool values indicating which feature values are NaNs
        style:
            Name of style to apply to the plot
        colormap:
            Name of qualitative colormap to use for each category

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    with style_settings(style), color_cycle(colormap):
        fig, ax = plt.subplots()
        codes = np.unique(feature_values)
        for code in codes:
            category = feature_mapping[str(code)]
            is_category = (feature_values == code) & ~is_nan
            dims = np.take(latent_space.compress(is_category, axis=0), [0, 1], axis=1).T
            ax.scatter(*dims, label=category, s=10)
        dims = np.take(latent_space.compress(is_nan, axis=0), [0, 1], axis=1).T
        ax.scatter(*dims, label="NaN", s=10)
        ax.set(xlabel="dim 0", ylabel="dim 1")
        legend = ax.legend()
        legend.set_title(feature_name)
    return fig


def plot_latent_space_with_con(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a 2D latent space together with a colorbar mapping the latent
    space to the values of a continuous feature.

    Args:
        latent_space: Embedding, a ND array with at least two dimensions.
        feature_name: Name of continuous feature
        feature_values: Values of continuous feature
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the colorbar

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")

    # print(f"feature_name: {feature_name}")
    # print(f"feature_values: {feature_values}")
    # print(f"min(feature_values): {min(feature_values)}")
    # print(f"max(feature_values): {max(feature_values)}")
    # print(f"mean(feature_values): {np.mean(feature_values)}")
    # print(f"std(feature_values): {np.std(feature_values)}")

    # norm = TwoSlopeNorm(0.0, min(feature_values), max(feature_values))

    vmin = min(feature_values)
    vmax = max(feature_values)
    vcenter = (vmin + vmax) / 2

    # Ensure the norm parameters are in ascending order
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    # I get the ValueError: vmin, vcenter, and vmax must be in ascending order
    with style_settings(style):
        fig, ax = plt.subplots()
        dims = latent_space[:, 0], latent_space[:, 1]
        pts = ax.scatter(*dims, c=feature_values, cmap=colormap, norm=norm, s=10)  # reduce the size of the dots
        cbar = fig.colorbar(pts, ax=ax)
        cbar.ax.set(ylabel=feature_name)
        ax.set(xlabel="dim 0", ylabel="dim 1")
    return fig
