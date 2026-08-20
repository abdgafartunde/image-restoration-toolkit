"""Convergence tests for imrestore.algorithms.var_pdal (pdal, var_pdal).

Test problem: min_x max_y <Kx,y> + g(x) [+ h(x)] - f*(y), with K = I,
g = indicator of a box [lo, hi]^d, and f*(y) = 0.5||y-c||^2 (smooth,
strongly convex). Since sup_y <x,y> - f*(y) = 0.5||x||^2 + <c,x> (the
conjugate of a shifted quadratic is again a shifted quadratic), the primal
problem collapses to a box-constrained quadratic with a closed-form
solution: x* = clip(-c [- rho*shift term for var_pdal's h], lo, hi). This
gives an exact reference to check convergence against, independent of the
algorithms under test.
"""

import numpy as np

from imrestore.algorithms.var_pdal import pdal, var_pdal


def _project_box(x, lo, hi):
    return np.clip(x, lo, hi)


def test_pdal_converges_on_toy_saddle_point():
    rng = np.random.default_rng(0)
    d = 8
    lo, hi = -1.0, 1.0
    c = rng.uniform(-2.0, 2.0, d)

    identity = lambda v: v
    prox_g = lambda v, tau: _project_box(v, lo, hi)
    prox_f_conj = lambda v, sigma: (v + sigma * c) / (1.0 + sigma)

    result = pdal(
        K=identity,
        K_adjoint=identity,
        prox_g=prox_g,
        prox_f_conj=prox_f_conj,
        x0=np.zeros(d),
        y1=np.zeros(d),
        tau0=0.1,
        beta=1.0,
        max_iter=500,
        tol=1e-10,
    )

    x_ref = _project_box(-c, lo, hi)
    assert np.linalg.norm(result.x - x_ref) < 1e-4


def test_var_pdal_converges_on_toy_saddle_point_with_smooth_term():
    rng = np.random.default_rng(1)
    d = 8
    lo, hi = -1.0, 1.0
    c = rng.uniform(-2.0, 2.0, d)
    rho = 2.0
    shift = rng.uniform(-1.0, 1.0, d)

    identity = lambda v: v
    prox_g = lambda v, tau: _project_box(v, lo, hi)
    prox_f_conj = lambda v, sigma: (v + sigma * c) / (1.0 + sigma)
    h = lambda x: 0.5 * rho * np.sum((x - shift) ** 2)
    grad_h = lambda x: rho * (x - shift)

    result = var_pdal(
        K=identity,
        K_adjoint=identity,
        prox_g=prox_g,
        prox_f_conj=prox_f_conj,
        h=h,
        grad_h=grad_h,
        x1=np.zeros(d),
        y0=np.zeros(d),
        L_h=rho,
        sigma0=0.1,
        beta=1.0,
        max_iter=500,
        tol=1e-10,
    )

    x_unconstrained = -(c - rho * shift) / (1.0 + rho)
    x_ref = _project_box(x_unconstrained, lo, hi)
    assert np.linalg.norm(result.x - x_ref) < 1e-3


def test_var_pdal_rejects_theta0_below_Lh():
    import pytest

    identity = lambda v: v
    with pytest.raises(ValueError):
        var_pdal(
            K=identity,
            K_adjoint=identity,
            prox_g=lambda v, tau: v,
            prox_f_conj=lambda v, sigma: v,
            h=lambda x: 0.0,
            grad_h=lambda x: np.zeros_like(x),
            x1=np.zeros(4),
            y0=np.zeros(4),
            L_h=1.0,
            sigma0=0.1,
            beta=1.0,
            theta0=0.5,
        )
