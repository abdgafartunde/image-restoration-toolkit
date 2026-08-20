"""Standard image restoration quality metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio between a restoration and a reference image."""
    mse = float(np.mean((x - x_ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((data_range**2) / mse)


def ssim(
    x: np.ndarray,
    x_ref: np.ndarray,
    data_range: float = 1.0,
    sigma: float = 1.5,
    truncate: float = 3.5,
) -> float:
    """Structural similarity index (Wang, Bovik, Sheikh & Simoncelli 2004,
    "Image Quality Assessment: From Error Visibility to Structural
    Similarity"), single-scale, grayscale, Gaussian window. Reimplemented
    directly from the paper's formula rather than pulling in an extra
    dependency for one function; cross-checked to match
    `skimage.metrics.structural_similarity(..., gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False)` to float precision.

    The local SSIM map is only meaningful `truncate*sigma` pixels away from
    the border (closer than that, the Gaussian window is computed over
    reflected/extrapolated boundary content rather than real image data);
    following the paper and standard implementations, that border strip is
    cropped before averaging so it doesn't drag the score down with
    boundary-filter artifacts rather than genuine structural mismatch.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    filter_kwargs = {"sigma": sigma, "truncate": truncate, "mode": "reflect"}

    mu_x = gaussian_filter(x, **filter_kwargs)
    mu_y = gaussian_filter(x_ref, **filter_kwargs)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    var_x = gaussian_filter(x * x, **filter_kwargs) - mu_x2
    var_y = gaussian_filter(x_ref * x_ref, **filter_kwargs) - mu_y2
    cov_xy = gaussian_filter(x * x_ref, **filter_kwargs) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * cov_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    ssim_map = numerator / denominator

    pad = int(truncate * sigma + 0.5)
    if pad > 0:
        if x.shape[0] <= 2 * pad or x.shape[1] <= 2 * pad:
            raise ValueError(
                f"image shape {x.shape} is too small for sigma={sigma}, "
                f"truncate={truncate} (needs both dimensions > {2 * pad}); "
                f"use a smaller sigma/truncate or a larger image"
            )
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(np.mean(ssim_map))
