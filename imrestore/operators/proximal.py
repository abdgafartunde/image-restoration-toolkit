"""Generic proximal operator library: prox_g(x0) = argmin_x g(x) + 1/2||x-x0||^2_D.

Standard closed-form convex-analysis operators (not tied to a specific
paper); reimplemented directly from their optimality conditions. `metric`,
where accepted, is the diagonal of D (elementwise weights); omit it (or
pass None) for the ordinary Euclidean prox.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def prox_l1(x0: np.ndarray, weight: float, metric: Optional[np.ndarray] = None) -> np.ndarray:
    """Soft-thresholding: prox of `weight * ||x||_1`.

    Coordinatewise argmin of `weight*|x_i| + 0.5*D_i*(x_i-x0_i)^2` gives a
    soft threshold of `weight/D_i` on each coordinate.
    """
    threshold = weight if metric is None else weight / metric
    return np.sign(x0) * np.maximum(np.abs(x0) - threshold, 0.0)


def prox_l2_squared(x0: np.ndarray, weight: float, metric: Optional[np.ndarray] = None) -> np.ndarray:
    """Prox of `weight/2 * ||x||_2^2` (shrinkage)."""
    if metric is None:
        return x0 / (1.0 + weight)
    return (metric * x0) / (metric + weight)


def prox_box(x0: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Projection onto the box constraint [lo, hi]^N.

    The prox of an indicator function is the metric-weighted projection
    onto its set; for a box this is plain clipping regardless of the
    (positive) metric used, so no `metric` parameter is needed.
    """
    return np.clip(x0, lo, hi)
