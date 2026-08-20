"""Shared machinery for the rank-1 ("0SR1") quasi-Newton metric and its
resolvent calculus, common to both:

  - Wang, Fadili & Ochs, "Inertial Quasi-Newton Methods for Monotone
    Inclusion..." (arXiv:2209.14019) - Section 5 (0SR1 metric) and
    Theorem 4.1 / Corollary 4.3, Propositions 4.8-4.9 (resolvent calculus).
  - Wang, Fadili & Ochs, "A Quasi-Newton Primal-Dual Algorithm with Line
    Search" (arXiv:2405.06824) - Section 5 describes a more general
    multi-memory L-BFGS/L-SR1 metric; `imrestore.algorithms.var_pdal`
    deliberately instantiates the simpler 0SR1 special case defined in the
    companion paper above, for consistency with `quasi_newton_fbs` and to
    keep the resolvent calculus a 1-D root-finding problem.

Not part of the public API; imported by `adaptive_gd`-adjacent algorithm
modules only.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

_EPS = 1e-12


def solve_rank1_root(
    residual: Callable[[float], float],
    bracket_iters: int = 40,
    bisection_iters: int = 60,
) -> float:
    """Find the root of a strictly increasing scalar function `residual`
    by expanding a bracket around 0 and bisecting (paper's Algorithm 4,
    arXiv:2209.14019).
    """
    r = 1.0
    f_minus, f_plus = residual(-r), residual(r)
    n_expand = 0
    while f_minus > 0 or f_plus < 0:
        r *= 2.0
        f_minus, f_plus = residual(-r), residual(r)
        n_expand += 1
        if n_expand > bracket_iters:
            break

    lo, hi = -r, r
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def rank1_curvature_update(
    theta0: float,
    beta: float,
    s: np.ndarray,
    y: np.ndarray,
    gamma: float,
) -> Tuple[Optional[np.ndarray], int]:
    """0SR1 rank-1 correction (eq. 28-31 of arXiv:2209.14019): given the
    secant pair `(s, y)` and base metric scale `theta0` (with
    `theta0 > 1/beta`), return `(u, sign)` such that
    `M_k = theta0*I + sign * u u^T`, or `(None, 1)` if the curvature
    condition is degenerate and `M_k = theta0*I` should be used unchanged.

    `gamma` targets the rank-1 scale (Lemma 5.1(i)); on negative-curvature
    steps it is automatically clipped to the bound of Lemma 5.1(ii) that
    keeps `M_k` positive definite.
    """
    curvature = float(np.vdot(y, s)) - theta0 * float(np.vdot(s, s))
    if abs(curvature) <= _EPS or np.linalg.norm(s) <= _EPS:
        return None, 1

    u_hat = (y - theta0 * s) / math.sqrt(abs(curvature))
    sign = 1 if curvature > 0 else -1
    g = gamma
    if sign < 0:
        c = theta0 - 1.0 / beta
        bound = c * c / (1.0 / beta + theta0) ** 2
        g = min(gamma, 0.5 * bound)
    return math.sqrt(g) * u_hat, sign


def rank1_resolvent_step(
    resolvent_base: Callable[[np.ndarray, float], np.ndarray],
    theta0: float,
    z: np.ndarray,
    w: np.ndarray,
    u: Optional[np.ndarray],
    sign: int,
) -> np.ndarray:
    """Compute `J_A^{M}(w)` where `M = theta0*I + sign*u u^T`, via the
    rank-1 resolvent calculus (Theorem 4.1 / Corollary 4.3, Prop. 4.9):
    reduces to evaluating `resolvent_base` (the resolvent/prox w.r.t. the
    simple base metric `theta0*I`) plus a 1-D root-find, never forming or
    inverting `M`.

    `z` is the un-shifted evaluation point; `w` is the forward step
    already computed by the caller (e.g. `z - Bz/theta0`).
    """
    if u is None:
        return resolvent_base(w, theta0)

    def residual(xi: float) -> float:
        shifted = w - (sign * xi / theta0) * u
        return float(np.vdot(u, z - resolvent_base(shifted, theta0))) + xi

    xi_star = solve_rank1_root(residual)
    return resolvent_base(w - (sign * xi_star / theta0) * u, theta0)
