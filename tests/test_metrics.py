"""Tests for imrestore.metrics.image_quality."""

import numpy as np
import pytest

from imrestore.metrics.image_quality import psnr, ssim


def test_psnr_is_infinite_for_identical_images():
    x = np.random.default_rng(0).uniform(0, 1, (16, 16))
    assert psnr(x, x) == float("inf")


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (32, 32))
    low_noise = x + rng.normal(0, 0.01, x.shape)
    high_noise = x + rng.normal(0, 0.2, x.shape)
    assert psnr(low_noise, x) > psnr(high_noise, x)


def test_ssim_is_one_for_identical_images():
    x = np.random.default_rng(2).uniform(0, 1, (32, 32))
    assert ssim(x, x) > 0.999


def test_ssim_decreases_with_more_noise():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, (32, 32))
    low_noise = x + rng.normal(0, 0.01, x.shape)
    high_noise = x + rng.normal(0, 0.3, x.shape)
    assert ssim(low_noise, x) > ssim(high_noise, x)


def test_ssim_rejects_image_too_small_for_window():
    # default sigma=1.5, truncate=3.5 -> pad=5, needs shape > 10
    x = np.random.default_rng(4).uniform(0, 1, (10, 10))
    with pytest.raises(ValueError, match="too small"):
        ssim(x, x)


def test_ssim_accepts_image_just_above_minimum_size():
    x = np.random.default_rng(4).uniform(0, 1, (11, 11))
    assert 0.0 <= ssim(x, x) <= 1.0 + 1e-9
