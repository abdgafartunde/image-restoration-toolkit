"""Image loading, synthetic phantoms, and noise simulation for demos and tests."""

from __future__ import annotations

from typing import Optional

import numpy as np

# Standard Shepp-Logan phantom ellipse parameters (Shepp & Logan, 1974,
# "The Fourier Reconstruction of a Head Section"): A, a, b, x0, y0, phi(deg).
_SHEPP_LOGAN_ELLIPSES = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
    (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0),
    (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0),
    (0.1, 0.2100, 0.2500, 0.0, 0.35, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, 0.1, 0.0),
    (0.1, 0.0460, 0.0460, 0.0, -0.1, 0.0),
    (0.1, 0.0460, 0.0230, -0.08, -0.605, 0.0),
    (0.1, 0.0230, 0.0230, 0.0, -0.606, 0.0),
    (0.1, 0.0230, 0.0460, 0.06, -0.605, 0.0),
]


def _shepp_logan(size: int) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(ax, ax)
    img = np.zeros((size, size))
    for amplitude, a, b, x0, y0, phi in _SHEPP_LOGAN_ELLIPSES:
        phi_r = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_r), np.sin(phi_r)
        xr = (xx - x0) * cos_p + (yy - y0) * sin_p
        yr = -(xx - x0) * sin_p + (yy - y0) * cos_p
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += amplitude
    return np.clip(img, 0.0, 1.0)


def _circles(size: int) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(ax, ax)
    img = np.zeros((size, size))
    img[xx**2 + yy**2 <= 0.8**2] = 0.3
    img[(xx - 0.3) ** 2 + (yy - 0.2) ** 2 <= 0.25**2] = 0.7
    img[(xx + 0.35) ** 2 + (yy + 0.3) ** 2 <= 0.2**2] = 1.0
    return img


_PHANTOMS = {"shepp-logan": _shepp_logan, "circles": _circles}


def load_phantom(name: str, size: int = 128) -> np.ndarray:
    """Load a built-in synthetic test image, values in [0, 1].

    `name` is one of `"shepp-logan"` (the classic 10-ellipse CT phantom) or
    `"circles"` (a simpler three-circle test image).
    """
    if name not in _PHANTOMS:
        raise ValueError(f"unknown phantom {name!r}; choose from {sorted(_PHANTOMS)}")
    return _PHANTOMS[name](size)


def add_gaussian_noise(x: np.ndarray, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """Add i.i.d. Gaussian noise with standard deviation `sigma`."""
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, sigma, size=x.shape)
