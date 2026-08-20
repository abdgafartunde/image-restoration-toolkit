"""Tests for imrestore.utils.io."""

import numpy as np
import pytest

from imrestore.utils.io import add_gaussian_noise, load_phantom


def test_load_phantom_shepp_logan_shape_and_range():
    img = load_phantom("shepp-logan", size=64)
    assert img.shape == (64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.std() > 0  # not blank


def test_load_phantom_circles_shape_and_range():
    img = load_phantom("circles", size=32)
    assert img.shape == (32, 32)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_load_phantom_rejects_unknown_name():
    with pytest.raises(ValueError):
        load_phantom("not-a-real-phantom")


def test_add_gaussian_noise_matches_requested_std():
    rng_seed = 0
    x = np.zeros((200, 200))
    noisy = add_gaussian_noise(x, sigma=0.1, seed=rng_seed)
    assert abs(noisy.std() - 0.1) < 0.01
    assert abs(noisy.mean()) < 0.01
