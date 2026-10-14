"""Forward operators used as the data-fidelity map `A` in restoration problems."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _gaussian_kernel_2d(sigma: float, size: int) -> np.ndarray:
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


class GaussianBlur:
    """2-D Gaussian blur forward operator, implemented as *circular*
    (periodic) convolution via FFT.

    Circular convolution is used deliberately: since the Gaussian kernel is
    symmetric, its adjoint under circular boundary conditions is exactly
    itself (`adjoint == __call__`), with no boundary-truncation mismatch to
    account for - unlike 'same'-mode convolution, whose true adjoint is not
    convolution with the same kernel near the image edges.
    """

    def __init__(self, sigma: float, kernel_size: Optional[int] = None):
        if kernel_size is None:
            kernel_size = 2 * int(round(3 * sigma)) + 1
        if kernel_size % 2 != 1:
            # An even kernel has no centre pixel, so the circular convolution
            # is not symmetric and `adjoint` would not be the adjoint.
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.sigma = sigma
        self.kernel_size = kernel_size
        self._kernel = _gaussian_kernel_2d(sigma, kernel_size)
        self._kernel_fft: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None

    def _fft_kernel(self, shape: Tuple[int, int]) -> np.ndarray:
        if self._shape != shape:
            kh, kw = self._kernel.shape
            if kh > shape[0] or kw > shape[1]:
                raise ValueError(
                    f"kernel_size {self.kernel_size} (sigma={self.sigma}) exceeds "
                    f"image shape {shape}; use a smaller sigma, pass an explicit "
                    f"kernel_size, or use a larger image"
                )
            padded = np.zeros(shape)
            padded[:kh, :kw] = self._kernel
            padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
            self._kernel_fft = np.fft.rfft2(padded)
            self._shape = shape
        return self._kernel_fft

    def __call__(self, x: np.ndarray) -> np.ndarray:
        kernel_fft = self._fft_kernel(x.shape)
        return np.fft.irfft2(np.fft.rfft2(x) * kernel_fft, s=x.shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self(y)
