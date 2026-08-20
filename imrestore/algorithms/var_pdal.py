"""PDAL and its variable-metric extension (VarPDAL).

Base method: Malitsky & Pock, "A First-Order Primal-Dual Algorithm with
Linesearch" (SIAM J. Optim. 28(1), 2018, arXiv:1608.08883), Algorithm 1.
Variable-metric extension: Wang, Fadili & Ochs, "A Quasi-Newton Primal-Dual
Algorithm with Line Search" (arXiv:2405.06824), Algorithm 1.
Reimplemented directly from the papers' algorithm boxes; no code reused
from either repository. See THEORY.md, Section 4.

Both solve the saddle-point problem

    min_x max_y  <Kx, y> + g(x) [+ h(x)] - f*(y)

with g, f* simple (prox-friendly), K a bounded linear operator, and (for
`var_pdal`) h an L_h-smooth convex function.

Note the two algorithms linesearch *different* variables: base `pdal`
linesearches the primal step tau_k (computing the next dual iterate);
`var_pdal` follows the source paper's convention of linesearching the
dual step sigma_k (computing the next primal iterate) so that the
linesearch lands on the metric-aware primal update.

Scope note: `var_pdal`'s Section 5 describes a general multi-memory
L-BFGS/L-SR1 metric (compact "identity +- low rank" representation,
rank <= 2m) with a semismooth-Newton resolvent solver for the general
rank-r case. This module instead instantiates the simpler rank-1 ("0SR1")
metric construction defined in the companion FBS paper
(arXiv:2209.14019, Section 5 - see `quasi_newton_fbs.py` and
`_rank1_metric.py`), which the two papers share the same resolvent
calculus for. This keeps the metric update and resolvent both O(d) and
consistent with the rest of the toolkit; the general multi-memory metric
is a documented extension, not yet implemented.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from imrestore.algorithms._rank1_metric import (
    rank1_curvature_update,
    rank1_resolvent_step,
)
from imrestore.algorithms.adaptive_gd import OptimizationResult


def pdal(
    K: Callable[[np.ndarray], np.ndarray],
    K_adjoint: Callable[[np.ndarray], np.ndarray],
    prox_g: Callable[[np.ndarray, float], np.ndarray],
    prox_f_conj: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    y1: np.ndarray,
    tau0: float,
    beta: float,
    mu: float = 0.7,
    delta: float = 0.99,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    max_linesearch: int = 50,
) -> OptimizationResult:
    """Primal-Dual Algorithm with Linesearch (Malitsky & Pock 2018,
    Algorithm 1) for `min_x max_y <Kx,y> + g(x) - f*(y)`.

    Parameters
    ----------
    K, K_adjoint:
        The linear operator and its adjoint.
    prox_g, prox_f_conj:
        `prox_g(v, tau)` = prox_{tau*g}(v); `prox_f_conj(v, sigma)` =
        prox_{sigma*f*}(v).
    x0, y1:
        Initial primal and dual points (paper's `x^0`, `y^1`).
    tau0:
        Initial primal step size (this is the linesearched variable).
    beta:
        Fixed ratio between primal and dual steps: the dual prox is
        evaluated with parameter `beta * tau_k` (paper's beta).
    mu:
        Linesearch shrink factor in (0, 1).
    delta:
        Linesearch tolerance in (0, 1) (paper recommends close to 1).
    """
    x_prev = np.asarray(x0, dtype=float)
    y_cur = np.asarray(y1, dtype=float)
    tau_prev = tau0
    theta_prev = 1.0
    Kx_prev = K(x_prev)
    Ky_cur = K_adjoint(y_cur)

    objective_history = [objective(x_prev)] if objective is not None else []
    step_size_history = [tau0]

    for k in range(1, max_iter + 1):
        x_cur = prox_g(x_prev - tau_prev * Ky_cur, tau_prev)
        Kx_cur = K(x_cur)

        tau_k = tau_prev * math.sqrt(1 + theta_prev)
        for _ in range(max_linesearch):
            theta_k = tau_k / tau_prev
            Kx_bar = Kx_cur + theta_k * (Kx_cur - Kx_prev)
            y_next = prox_f_conj(y_cur + beta * tau_k * Kx_bar, beta * tau_k)
            Ky_next = K_adjoint(y_next)
            lhs = math.sqrt(beta) * tau_k * np.linalg.norm(Ky_next - Ky_cur)
            rhs = delta * np.linalg.norm(y_next - y_cur)
            if lhs <= rhs:
                break
            tau_k *= mu

        step_size_history.append(tau_k)
        if objective is not None:
            objective_history.append(objective(x_cur))

        residual = np.linalg.norm(x_cur - x_prev) + np.linalg.norm(y_next - y_cur)

        x_prev, Kx_prev = x_cur, Kx_cur
        y_cur, Ky_cur = y_next, Ky_next
        tau_prev, theta_prev = tau_k, theta_k

        if residual <= tol:
            return OptimizationResult(x_prev, objective_history, step_size_history, k)

    return OptimizationResult(x_prev, objective_history, step_size_history, max_iter)


def var_pdal(
    K: Callable[[np.ndarray], np.ndarray],
    K_adjoint: Callable[[np.ndarray], np.ndarray],
    prox_g: Callable[[np.ndarray, float], np.ndarray],
    prox_f_conj: Callable[[np.ndarray, float], np.ndarray],
    h: Callable[[np.ndarray], float],
    grad_h: Callable[[np.ndarray], np.ndarray],
    x1: np.ndarray,
    y0: np.ndarray,
    L_h: float,
    sigma0: float,
    beta: float,
    theta0: Optional[float] = None,
    gamma: float = 1.0,
    mu: float = 0.7,
    delta: float = 0.99,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    max_linesearch: int = 50,
) -> OptimizationResult:
    """Variable-metric PDAL (Wang, Fadili & Ochs 2024, Algorithm 1) for
    `min_x max_y <Kx,y> + g(x) + h(x) - f*(y)`, `h` convex with
    `L_h`-Lipschitz gradient. The primal proximal step is taken with
    respect to a rank-1 quasi-Newton metric built from consecutive
    `grad_h` evaluations (see module docstring for scope).

    Unlike `pdal`, this linesearches the *dual* step `sigma_k` (paper's
    convention), which drives the metric-aware primal update.

    Additional parameters beyond `pdal`
    ------------------------------------
    h, grad_h:
        The smooth primal term and its gradient.
    L_h:
        Lipschitz constant of grad_h.
    x1, y0:
        Initial primal and dual points (paper's `x^1`, `y^0`).
    sigma0:
        Initial dual step size (the linesearched variable here).
    beta:
        Fixed ratio tau_k = beta * sigma_k between primal and dual steps.
    theta0:
        Base metric scale, M_0 = theta0 * I. Must exceed L_h (mirrors
        `quasi_newton_fbs`'s theta0 > 1/beta, with beta = 1/L_h here).
        Defaults to 1.1 * L_h.
    gamma:
        Target rank-1 curvature scale, see `quasi_newton_fbs`.
    """
    x_prev = np.asarray(x1, dtype=float)
    x_cur = x_prev.copy()
    y_prev = np.asarray(y0, dtype=float)

    if theta0 is None:
        theta0 = 1.1 * L_h
    if theta0 <= L_h:
        raise ValueError("theta0 must exceed L_h so that M_0 - L_h*I is positive definite")
    beta_h = 1.0 / L_h

    grad_h_prev = grad_h(x_prev)
    sigma_prev = sigma0
    theta_prev = 1.0

    objective_history = [objective(x_prev)] if objective is not None else []
    step_size_history = [sigma0]

    for k in range(1, max_iter + 1):
        y_cur = prox_f_conj(y_prev + sigma_prev * K(x_cur), sigma_prev)

        grad_h_cur = grad_h(x_cur)
        s = x_cur - x_prev
        y_secant = grad_h_cur - grad_h_prev
        u, sign = rank1_curvature_update(theta0, beta_h, s, y_secant, gamma)

        h_cur = h(x_cur)
        Kx_cur = K(x_cur)

        sigma_k = sigma_prev * math.sqrt(1 + theta_prev)
        for _ in range(max_linesearch):
            theta_k = sigma_k / sigma_prev
            tau_k = beta * sigma_k
            y_bar = y_cur + theta_k * (y_cur - y_prev)
            force = K_adjoint(y_bar) + grad_h_cur
            w = x_cur - (tau_k / theta0) * force

            def prox_g_scaled(v, th, _tau_k=tau_k):
                return prox_g(v, _tau_k / th)

            x_next = rank1_resolvent_step(prox_g_scaled, theta0, x_cur, w, u, sign)

            diff = x_next - x_cur
            m_norm_sq = theta0 * float(np.vdot(diff, diff))
            if u is not None:
                m_norm_sq += sign * float(np.vdot(u, diff)) ** 2

            Kx_next = K(x_next)
            h_gap = h(x_next) - h_cur - float(np.vdot(grad_h_cur, diff))
            lhs_val = tau_k * sigma_k * float(np.linalg.norm(Kx_next - Kx_cur) ** 2) + 2 * tau_k * h_gap

            if lhs_val <= delta * m_norm_sq:
                break
            sigma_k *= mu

        step_size_history.append(sigma_k)
        if objective is not None:
            objective_history.append(objective(x_next))

        residual = np.linalg.norm(x_next - x_cur) + np.linalg.norm(y_cur - y_prev)

        x_prev, grad_h_prev = x_cur, grad_h_cur
        x_cur = x_next
        y_prev = y_cur
        sigma_prev, theta_prev = sigma_k, theta_k

        if residual <= tol:
            return OptimizationResult(x_cur, objective_history, step_size_history, k)

    return OptimizationResult(x_cur, objective_history, step_size_history, max_iter)
