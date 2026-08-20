"""Convergence tests for imrestore.algorithms.adaptive_gd."""

import numpy as np

from imrestore.algorithms.adaptive_gd import (
    accelerated_adaptive_gradient_descent,
    adaptive_gradient_descent,
)


def _random_quadratic(d=20, condition_number=50.0, seed=0):
    rng = np.random.default_rng(seed)
    eigs = np.geomspace(1.0, condition_number, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Q @ np.diag(eigs) @ Q.T
    x_star = rng.standard_normal(d)
    b = A @ x_star

    def objective(x):
        return 0.5 * x @ (A @ x) - b @ x

    def gradient(x):
        return A @ x - b

    return objective, gradient, x_star


def test_converges_on_quadratic():
    objective, gradient, x_star = _random_quadratic()
    x0 = np.zeros_like(x_star)

    result = adaptive_gradient_descent(gradient, x0, objective=objective, max_iter=2000, tol=1e-10)

    assert np.linalg.norm(result.x - x_star) < 1e-4
    # objective history should be (eventually) monotonically non-increasing
    # in a windowed sense: last value close to the optimum.
    assert objective(result.x) - objective(x_star) < 1e-6


def test_step_size_history_matches_iterations():
    _, gradient, x_star = _random_quadratic()
    x0 = np.zeros_like(x_star)

    result = adaptive_gradient_descent(gradient, x0, max_iter=100, tol=0.0)

    assert len(result.step_size_history) == result.n_iter + 1
    assert all(s > 0 for s in result.step_size_history)


def test_accelerated_variant_converges():
    objective, gradient, x_star = _random_quadratic(condition_number=200.0)
    x0 = np.zeros_like(x_star)

    result = accelerated_adaptive_gradient_descent(
        gradient, x0, objective=objective, max_iter=2000, tol=1e-10
    )

    assert np.linalg.norm(result.x - x_star) < 1e-4


def test_accelerated_variant_uses_fewer_iterations_on_illconditioned_problem():
    objective, gradient, x_star = _random_quadratic(condition_number=500.0)
    x0 = np.zeros_like(x_star)

    plain = adaptive_gradient_descent(gradient, x0, objective=objective, max_iter=5000, tol=1e-8)
    accel = accelerated_adaptive_gradient_descent(
        gradient, x0, objective=objective, max_iter=5000, tol=1e-8
    )

    assert accel.n_iter <= plain.n_iter
