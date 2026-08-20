"""Adaptive Gradient Descent (Malitsky & Mishchenko, ICML 2020, arXiv:1910.09529).

Implements:
  - Algorithm 1 (AdGD): proven O(1/k) convergence for convex, locally
    smooth f, with no line search and no Lipschitz constant required.
  - Algorithm 2 (AdGD-accel): the paper's adaptive accelerated variant.
    The paper presents this in its "Heuristics" section and is explicit
    that it comes with no formal convergence guarantee, unlike AdGD.

Reimplemented directly from the paper's algorithm boxes; no code reused
from the paper's reference implementation. See THEORY.md, Section 2.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import numpy as np

_EPS = 1e-15


class OptimizationResult(NamedTuple):
    """Solution and diagnostics returned by every solver in this package."""

    x: np.ndarray
    objective_history: list
    step_size_history: list
    n_iter: int


def adaptive_gradient_descent(
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    lambda_0: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Minimize a convex, locally smooth f via Algorithm 1 (AdGD).

    Parameters
    ----------
    gradient:
        Callable returning grad f(x).
    x0:
        Initial point.
    objective:
        Optional f, used only to populate `objective_history` for
        diagnostics; the algorithm itself never evaluates it.
    lambda_0:
        Initial step size. The paper recommends a tiny value (e.g. 1e-10)
        so that x^1 stays close to x^0 and gives a good first estimate.
    max_iter:
        Maximum number of iterations.
    tol:
        Stop early once ||grad f(x^k)|| <= tol.
    """
    x_prev = np.asarray(x0, dtype=float)
    grad_prev = gradient(x_prev)
    x = x_prev - lambda_0 * grad_prev

    theta = math.inf
    lam = lambda_0
    objective_history = [objective(x_prev)] if objective is not None else []
    step_size_history = [lambda_0]

    for k in range(1, max_iter + 1):
        grad = gradient(x)
        if objective is not None:
            objective_history.append(objective(x))
        if np.linalg.norm(grad) <= tol:
            return OptimizationResult(x, objective_history, step_size_history, k)

        grad_diff_norm = np.linalg.norm(grad - grad_prev)
        x_diff_norm = np.linalg.norm(x - x_prev)

        candidate_a = math.sqrt(1 + theta) * lam if math.isfinite(theta) else math.inf
        if grad_diff_norm > _EPS:
            lam_new = min(candidate_a, x_diff_norm / (2 * grad_diff_norm))
        else:
            # paper's convention 1/0 = +inf: the curvature bound is vacuous.
            lam_new = candidate_a if math.isfinite(candidate_a) else lam
        step_size_history.append(lam_new)

        x_next = x - lam_new * grad

        theta = lam_new / lam
        lam = lam_new
        x_prev, grad_prev = x, grad
        x = x_next

    return OptimizationResult(x, objective_history, step_size_history, max_iter)


def accelerated_adaptive_gradient_descent(
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    lambda_0: float = 1e-10,
    Lambda_0: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Adaptive accelerated gradient descent (Algorithm 2).

    Estimates both an inverse-smoothness step `lambda_k` and an inverse
    strong-convexity step `Lambda_k` online, and uses them to set a
    Nesterov-style momentum coefficient `beta_k`. The paper is explicit
    that, unlike `adaptive_gradient_descent`, this variant has no proven
    convergence guarantee.
    """
    x_prev = np.asarray(x0, dtype=float)
    grad_prev = gradient(x_prev)
    y_prev = x_prev - lambda_0 * grad_prev
    x = y_prev  # x^1 = y^1

    theta = math.inf
    Theta = math.inf
    lam = lambda_0
    Lam = Lambda_0

    objective_history = [objective(x_prev)] if objective is not None else []
    step_size_history = [lambda_0]

    for k in range(1, max_iter + 1):
        grad = gradient(x)
        if objective is not None:
            objective_history.append(objective(x))
        if np.linalg.norm(grad) <= tol:
            return OptimizationResult(x, objective_history, step_size_history, k)

        grad_diff_norm = np.linalg.norm(grad - grad_prev)
        x_diff_norm = np.linalg.norm(x - x_prev)

        cand_a = math.sqrt(1 + theta / 2) * lam if math.isfinite(theta) else math.inf
        if grad_diff_norm > _EPS:
            lam_new = min(cand_a, x_diff_norm / (2 * grad_diff_norm))
        else:
            lam_new = cand_a if math.isfinite(cand_a) else lam

        cand_A = math.sqrt(1 + Theta / 2) * Lam if math.isfinite(Theta) else math.inf
        if x_diff_norm > _EPS:
            Lam_new = min(cand_A, grad_diff_norm / (2 * x_diff_norm))
        else:
            Lam_new = cand_A if math.isfinite(cand_A) else Lam

        beta = (math.sqrt(1 / lam_new) - math.sqrt(Lam_new)) / (
            math.sqrt(1 / lam_new) + math.sqrt(Lam_new)
        )

        y_next = x - lam_new * grad
        x_next = y_next + beta * (y_next - y_prev)
        step_size_history.append(lam_new)

        theta = lam_new / lam
        Theta = Lam_new / Lam
        lam, Lam = lam_new, Lam_new
        x_prev, grad_prev, y_prev = x, grad, y_next
        x = x_next

    return OptimizationResult(x, objective_history, step_size_history, max_iter)
