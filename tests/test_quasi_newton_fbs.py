"""Convergence tests for imrestore.algorithms.quasi_newton_fbs."""

import numpy as np

from imrestore.algorithms.quasi_newton_fbs import inertial_quasi_newton_fbs


def _random_quadratic(d=15, condition_number=30.0, seed=1):
    rng = np.random.default_rng(seed)
    eigs = np.geomspace(1.0, condition_number, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Q @ np.diag(eigs) @ Q.T
    x_star = rng.standard_normal(d)
    b = A @ x_star
    L = eigs.max()

    def objective(x):
        return 0.5 * x @ (A @ x) - b @ x

    def gradient(x):
        return A @ x - b

    return objective, gradient, x_star, L


def test_converges_on_unconstrained_quadratic():
    # A = 0 (resolvent_A = identity) reduces the method to a
    # curvature-adaptive quasi-Newton gradient step (paper, Section 5).
    objective, gradient, x_star, L = _random_quadratic()
    identity_resolvent = lambda v, theta: v

    result = inertial_quasi_newton_fbs(
        B=gradient,
        resolvent_A=identity_resolvent,
        z0=np.zeros_like(x_star),
        beta=1.0 / L,
        max_iter=500,
        tol=1e-10,
    )

    assert np.linalg.norm(result.x - x_star) < 1e-4


def _project_box(x, lo, hi):
    return np.clip(x, lo, hi)


def test_converges_on_box_constrained_quadratic():
    objective, gradient, x_star_unconstrained, L = _random_quadratic(d=10, seed=2)
    lo, hi = -0.5, 0.5
    box_resolvent = lambda v, theta: _project_box(v, lo, hi)

    result = inertial_quasi_newton_fbs(
        B=gradient,
        resolvent_A=box_resolvent,
        z0=np.zeros_like(x_star_unconstrained),
        beta=1.0 / L,
        max_iter=1000,
        tol=1e-10,
    )

    # reference: long-run projected gradient descent from the same start
    x_ref = np.zeros_like(x_star_unconstrained)
    step = 1.0 / L
    for _ in range(20000):
        x_ref = _project_box(x_ref - step * gradient(x_ref), lo, hi)

    assert np.all(result.x >= lo - 1e-8) and np.all(result.x <= hi + 1e-8)
    assert np.linalg.norm(result.x - x_ref) < 1e-3


def test_inertial_variant_also_converges():
    objective, gradient, x_star, L = _random_quadratic(condition_number=100.0, seed=3)
    identity_resolvent = lambda v, theta: v

    result = inertial_quasi_newton_fbs(
        B=gradient,
        resolvent_A=identity_resolvent,
        z0=np.zeros_like(x_star),
        beta=1.0 / L,
        inertial_alpha=0.2,
        max_iter=2000,
        tol=1e-10,
    )

    assert np.linalg.norm(result.x - x_star) < 1e-4


def test_rejects_theta0_below_inverse_beta():
    import pytest

    objective, gradient, x_star, L = _random_quadratic()
    with pytest.raises(ValueError):
        inertial_quasi_newton_fbs(
            B=gradient,
            resolvent_A=lambda v, theta: v,
            z0=np.zeros_like(x_star),
            beta=1.0 / L,
            theta0=0.5 / L,
        )
