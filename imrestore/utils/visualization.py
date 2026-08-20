"""Plotting helpers: convergence curves, before/after restoration panels."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    histories: Dict[str, Sequence[float]],
    ylabel: str = "objective",
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot one or more solver histories (e.g. `OptimizationResult.objective_history`)
    on shared axes, keyed by label.
    """
    if ax is None:
        _, ax = plt.subplots()
    for label, values in histories.items():
        values = np.asarray(values, dtype=float)
        if log_scale:
            ax.semilogy(np.maximum(values, 1e-16), label=label)
        else:
            ax.plot(values, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_restoration_grid(
    images: Dict[str, np.ndarray],
    cmap: str = "gray",
) -> plt.Figure:
    """Show a row of labeled images (e.g. ground truth / noisy / restored)."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images.items()):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    return fig
