"""Inertial Quasi-Newton Forward-Backward Splitting.

Wang, Fadili & Ochs, "Inertial Quasi-Newton Methods for Monotone Inclusion:
Efficient Resolvent Calculus and Primal-Dual Methods" (arXiv:2209.14019).
Reimplemented directly from the paper's Algorithm 1 (Section 3), the 0SR1
rank-1 metric (Section 5), and the resolvent calculus for a rank-1
perturbed metric (Theorem 4.1 / Corollary 4.3, Propositions 4.8-4.9). No
code reused from the authors' repository. See THEORY.md, Section 3.

Problem solved: find z such that 0 in Az + Bz, where B is beta-cocoercive
(e.g. the gradient of an L-smooth convex function, beta = 1/L) and A is
maximally monotone with an easy-to-evaluate resolvent under a *scalar*
metric theta*I (e.g. A = the subdifferential of a simple g, so that
resolvent w.r.t. theta*I is prox_{g/theta}).

This module restricts the general framework to a scalar base metric
M_0 = theta_0 * I with a single rank-1 (0-memory SR1) correction per
iteration, which is the simplest instance the paper's Section 5 covers and
keeps the resolvent calculus a 1-D root-finding problem solvable by
bisection, rather than the semismooth-Newton machinery needed for general
rank-r metrics. See `_rank1_metric.py` for the shared machinery (also used
by `var_pdal`, whose companion paper reuses the same construction).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from imrestore.algorithms._rank1_metric import (
    rank1_curvature_update,
    rank1_resolvent_step,
)
from imrestore.algorithms.adaptive_gd import OptimizationResult


def inertial_quasi_newton_fbs(
    B: Callable[[np.ndarray], np.ndarray],
    resolvent_A: Callable[[np.ndarray, float], np.ndarray],
    z0: np.ndarray,
    beta: float,
    theta0: Optional[float] = None,
    inertial_alpha: float = 0.0,
    gamma: float = 1.0,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Solve `0 in Az + Bz` via inertial quasi-Newton FBS with a rank-1
    (0SR1) metric (paper's Algorithm 1 + Section 5).

    Parameters
    ----------
    B:
        beta-cocoercive operator, e.g. the gradient of an L-smooth convex
        function (beta = 1/L).
    resolvent_A:
        `resolvent_A(v, theta)` returns `J_A^{theta*I}(v)`, the resolvent
        of A with respect to the scalar metric `theta*I`. For A = dg of a
        simple convex g, this is `prox_{g/theta}(v)`.
    z0:
        Initial point.
    beta:
        Co-coercivity constant of B.
    theta0:
        Base metric scale, M_0 = theta0 * I. Must satisfy theta0 > 1/beta
        (Section 5 requires M_0 - (1/beta) I positive definite). Defaults
        to 1.1/beta if not given.
    inertial_alpha:
        Constant inertial (extrapolation) coefficient in [0, 1). Use 0.0
        (the default) for the non-inertial step. Note on guarantees: the
        paper's Theorem 3.1 needs a summable-type inertial schedule and its
        Assumption 1 on the metrics, (1 + eta_k) M_k >= M_{k+1} with
        summable eta_k, which Lemma 5.2 obtains with a summable sequence in
        the rank-1 scale. This implementation uses a constant `gamma`
        (clipped as in Lemma 5.1(ii)), which gives only the paper's
        Assumption 2, used for its relaxed Algorithm 2 (not implemented).
        So no theorem in the paper covers this iteration as implemented,
        with or without inertia; it converges in the tests and demos.
    gamma:
        Target scale for the rank-1 curvature update (paper's gamma_k in
        eq. 29-31, Lemma 5.1). Automatically clipped on negative-curvature
        steps to the bound in Lemma 5.1(ii) that keeps M_k positive
        definite.
    objective:
        Optional scalar objective for diagnostics only (e.g. f + g for a
        composite problem A=dg, B=grad f); never evaluated by the
        algorithm itself.

    Returns
    -------
    OptimizationResult with the final iterate and, if `objective` is
    given, its value at each accepted iterate.
    """
    z_prev = np.asarray(z0, dtype=float)
    z = np.asarray(z0, dtype=float)
    if theta0 is None:
        theta0 = 1.1 / beta
    if theta0 <= 1.0 / beta:
        raise ValueError("theta0 must exceed 1/beta so that M_0 - I/beta is positive definite")

    Bz_prev = Bz = B(z)
    objective_history = [objective(z)] if objective is not None else []
    step_size_history = [theta0]

    for k in range(1, max_iter + 1):
        s = z - z_prev
        y = Bz - Bz_prev
        u, sign = rank1_curvature_update(theta0, beta, s, y, gamma)

        z_bar = z + inertial_alpha * (z - z_prev)
        Bz_bar = B(z_bar)
        w = z_bar - Bz_bar / theta0

        z_next = rank1_resolvent_step(resolvent_A, theta0, z_bar, w, u, sign)

        step_size_history.append(theta0)
        if objective is not None:
            objective_history.append(objective(z_next))

        if np.linalg.norm(z_next - z) <= tol:
            return OptimizationResult(z_next, objective_history, step_size_history, k)

        z_prev, Bz_prev = z, Bz
        z, Bz = z_next, B(z_next)

    return OptimizationResult(z, objective_history, step_size_history, max_iter)
