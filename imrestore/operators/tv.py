"""Total variation (TV) regularizer.

Isotropic TV, `weight * ||Dx||_{2,1}`, has no closed-form proximal
operator of its own - which is exactly why `pdal` / `var_pdal` solve
TV-regularized problems via the saddle-point (primal-dual) formulation
`min_x max_y <Dx,y> + g(x) [+ h(x)] - f*(y)` rather than forward-backward
splitting. There, `K = D` (the operator below) and `f* = ` the indicator
of the dual ball `{y : ||y||_{2,inf} <= weight}`, whose prox *is*
closed-form (a per-pixel Euclidean-ball projection) - that is `prox_conj`
below. So this class exposes `forward`/`adjoint` (to pass as `K`/`K_adjoint`)
and `prox_conj` (to pass as `prox_f_conj`), plus `__call__` to evaluate the
TV seminorm itself for objective tracking.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-12


class TotalVariation:
    """Isotropic total variation regularizer `weight * TV(x)` via forward
    finite differences (zero/Neumann boundary).
    """

    def __init__(self, weight: float):
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.weight = weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Discrete gradient: returns shape (2, *x.shape), stacking the
        row- and column-direction forward differences.
        """
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:-1, :] = x[1:, :] - x[:-1, :]
        gy[:, :-1] = x[:, 1:] - x[:, :-1]
        return np.stack([gx, gy], axis=0)

    def adjoint(self, p: np.ndarray) -> np.ndarray:
        """Adjoint of `forward` (= -divergence), matched to the same
        boundary convention so that `<forward(x), p> == <x, adjoint(p)>`
        exactly (verified in tests/test_operators.py).
        """
        gx, gy = p[0], p[1]
        dx = np.zeros_like(gx)
        dy = np.zeros_like(gy)
        dx[1:-1, :] = gx[1:-1, :] - gx[:-2, :]
        dx[0, :] = gx[0, :]
        dx[-1, :] = -gx[-2, :]
        dy[:, 1:-1] = gy[:, 1:-1] - gy[:, :-2]
        dy[:, 0] = gy[:, 0]
        dy[:, -1] = -gy[:, -2]
        return -(dx + dy)

    def __call__(self, x: np.ndarray) -> float:
        g = self.forward(x)
        return float(self.weight * np.sum(np.sqrt(g[0] ** 2 + g[1] ** 2)))

    def prox_conj(self, p: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Prox of the indicator of `{p : ||p||_{2,inf} <= weight}` (the
        Fenchel conjugate of `weight * ||.||_{2,1}`): a per-pixel
        Euclidean-ball projection. Independent of `sigma`, like any
        indicator's prox - accepted only so this can be passed directly as
        `prox_f_conj` in `pdal` / `var_pdal`.
        """
        norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
        factor = np.minimum(1.0, self.weight / np.maximum(norm, _EPS))
        return p * factor[np.newaxis, ...]
