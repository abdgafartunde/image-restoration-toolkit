"""Tests for imrestore.operators (adjoint checks, prox correctness, TV norm)."""

import numpy as np
import pytest

from imrestore.operators.blur import GaussianBlur
from imrestore.operators.proximal import prox_box, prox_l1, prox_l2_squared
from imrestore.operators.tv import TotalVariation


def test_gaussian_blur_adjoint_is_consistent():
    rng = np.random.default_rng(0)
    blur = GaussianBlur(sigma=1.5)
    x = rng.standard_normal((16, 20))
    y = rng.standard_normal((16, 20))

    lhs = np.sum(blur(x) * y)
    rhs = np.sum(x * blur.adjoint(y))
    assert abs(lhs - rhs) < 1e-8


def test_gaussian_blur_preserves_constant_image():
    blur = GaussianBlur(sigma=2.0)
    x = np.full((24, 24), 3.0)
    assert np.allclose(blur(x), 3.0, atol=1e-10)


def test_gaussian_blur_rejects_kernel_larger_than_image():
    blur = GaussianBlur(sigma=3.0)  # auto kernel_size = 19
    with pytest.raises(ValueError, match="exceeds image shape"):
        blur(np.zeros((17, 23)))


def test_prox_l1_matches_closed_form_on_scalar():
    x0 = np.array([2.0, -0.5, 0.3])
    result = prox_l1(x0, weight=0.4)
    expected = np.sign(x0) * np.maximum(np.abs(x0) - 0.4, 0.0)
    assert np.allclose(result, expected)


def test_prox_l1_satisfies_optimality_condition():
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(10)
    weight = 0.7
    x_star = prox_l1(x0, weight)
    # subgradient optimality: (x0 - x*) / weight in subdiff of |.| at x*
    subgrad = (x0 - x_star) / weight
    nonzero = x_star != 0
    assert np.allclose(subgrad[nonzero], np.sign(x_star[nonzero]), atol=1e-10)
    assert np.all(np.abs(subgrad[~nonzero]) <= 1.0 + 1e-10)


def test_prox_l2_squared_matches_closed_form():
    x0 = np.array([1.0, -2.0, 3.0])
    weight = 0.5
    assert np.allclose(prox_l2_squared(x0, weight), x0 / (1.0 + weight))


def test_prox_box_clips():
    x0 = np.array([-2.0, 0.5, 2.0])
    assert np.allclose(prox_box(x0, lo=-1.0, hi=1.0), [-1.0, 0.5, 1.0])


def test_tv_adjoint_is_consistent():
    rng = np.random.default_rng(2)
    tv = TotalVariation(weight=0.1)
    x = rng.standard_normal((10, 8))
    p = rng.standard_normal((2, 10, 8))

    lhs = np.sum(tv.forward(x) * p)
    rhs = np.sum(x * tv.adjoint(p))
    assert abs(lhs - rhs) < 1e-8


def test_tv_zero_on_constant_image():
    tv = TotalVariation(weight=1.0)
    x = np.full((6, 6), 5.0)
    assert tv(x) == 0.0


def test_tv_prox_conj_projects_onto_dual_ball():
    tv = TotalVariation(weight=0.3)
    p = np.zeros((2, 4, 4))
    p[0, 0, 0] = 10.0  # norm 10, should be clipped to weight=0.3
    p[:, 1, 1] = 0.1  # norm ~0.14, inside the ball, should be unchanged
    result = tv.prox_conj(p)
    assert np.isclose(np.linalg.norm(result[:, 0, 0]), 0.3)
    assert np.allclose(result[:, 1, 1], p[:, 1, 1])
