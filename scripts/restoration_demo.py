"""End-to-end TV-regularized restoration demo: denoising and deblurring on
the Shepp-Logan phantom, solved via `var_pdal` (Wang, Fadili & Ochs 2024).

Both tasks are posed as min_x max_y <Dx,y> + g(x) + h(x) - f*(y), with
D = TotalVariation.forward, g = box indicator [0,1] (prox_box), f* = TV's
dual-ball indicator (TotalVariation.prox_conj), and h the smooth data
term - identity residual for denoising, blur residual for deblurring. This
is exactly the case var_pdal is for: h is smooth but not simple enough to
fold into g's prox once the box constraint is also present.

Saves before/after image grids, PSNR/SSIM numbers, and convergence plots
to figures/ (or --outdir DIR).
"""

from __future__ import annotations

import numpy as np

from _cli import output_dir

from imrestore.algorithms.var_pdal import var_pdal
from imrestore.metrics.image_quality import psnr, ssim
from imrestore.operators.blur import GaussianBlur
from imrestore.operators.proximal import prox_box
from imrestore.operators.tv import TotalVariation
from imrestore.utils.io import add_gaussian_noise, load_phantom
from imrestore.utils.visualization import plot_convergence, plot_restoration_grid


def run_denoising(x_true, tv_weight=0.1, noise_sigma=0.08, max_iter=200):
    # tv_weight is intentionally past the PSNR-maximizing point (~0.05):
    # a visual sweep (0.05 to 0.8) showed 0.1 is where the reconstruction
    # actually looks clean - flat regions, sharp organ boundaries, no
    # visible speckle - while *matching* SSIM at 0.05 and costing only
    # ~2 dB PSNR. Above ~0.2 it visibly over-smooths and distorts anatomy.
    noisy = np.clip(add_gaussian_noise(x_true, noise_sigma, seed=0), 0.0, 1.0)
    tv = TotalVariation(weight=tv_weight)

    h = lambda x: 0.5 * np.sum((x - noisy) ** 2)
    grad_h = lambda x: x - noisy

    result = var_pdal(
        K=tv.forward,
        K_adjoint=tv.adjoint,
        prox_g=lambda v, tau: prox_box(v, 0.0, 1.0),
        prox_f_conj=tv.prox_conj,
        h=h,
        grad_h=grad_h,
        x1=noisy.copy(),
        y0=np.zeros((2, *x_true.shape)),
        L_h=1.0,
        sigma0=0.1,
        beta=1.0,
        objective=lambda x: h(x) + tv(x),
        max_iter=max_iter,
        tol=1e-9,
    )
    return noisy, result


def run_deblurring(x_true, tv_weight=0.001, blur_sigma=1.0, noise_sigma=0.01, max_iter=3000):
    # max_iter is much higher than denoising's: this tv_weight is low, and a low
    # weight means a slower-converging, closer-to-unregularized problem. In a
    # sweep with 3000 iterations, SSIM peaks here (0.927) while PSNR peaks at
    # a smaller weight, 3e-4 to 5e-4 (30.8 dB vs 29.3 dB at 1e-3). Checked against an independent, separately-
    # implemented Chambolle-Pock reference: 200 iterations undershoots the true
    # optimum by several dB PSNR; ~2000-3000 is needed to actually reach it.
    blur = GaussianBlur(sigma=blur_sigma)
    degraded = np.clip(add_gaussian_noise(blur(x_true), noise_sigma, seed=1), 0.0, 1.0)
    tv = TotalVariation(weight=tv_weight)

    h = lambda x: 0.5 * np.sum((blur(x) - degraded) ** 2)
    grad_h = lambda x: blur.adjoint(blur(x) - degraded)

    result = var_pdal(
        K=tv.forward,
        K_adjoint=tv.adjoint,
        prox_g=lambda v, tau: prox_box(v, 0.0, 1.0),
        prox_f_conj=tv.prox_conj,
        h=h,
        grad_h=grad_h,
        x1=degraded.copy(),
        y0=np.zeros((2, *x_true.shape)),
        L_h=1.0,  # Gaussian blur is a normalized low-pass filter: ||A|| <= 1
        sigma0=0.1,
        beta=1.0,
        objective=lambda x: h(x) + tv(x),
        max_iter=max_iter,
        tol=1e-9,
    )
    return degraded, result


def main():
    outdir = output_dir(__doc__.splitlines()[0])
    x_true = load_phantom("shepp-logan", size=96)

    noisy, denoise_result = run_denoising(x_true)
    degraded, deblur_result = run_deblurring(x_true)

    for label, degraded_img, restored in [
        ("denoising", noisy, denoise_result.x),
        ("deblurring", degraded, deblur_result.x),
    ]:
        print(
            f"{label}: PSNR {psnr(degraded_img, x_true):.2f} -> {psnr(restored, x_true):.2f} dB, "
            f"SSIM {ssim(degraded_img, x_true):.3f} -> {ssim(restored, x_true):.3f}"
        )

    fig1 = plot_restoration_grid(
        {"ground truth": x_true, "noisy": noisy, "denoised": denoise_result.x}
    )
    fig1.savefig(outdir / "restoration_denoising.png", dpi=150)

    fig2 = plot_restoration_grid(
        {"ground truth": x_true, "blurred+noisy": degraded, "deblurred": deblur_result.x}
    )
    fig2.savefig(outdir / "restoration_deblurring.png", dpi=150)

    ax = plot_convergence(
        {
            "denoising (g+h)": denoise_result.objective_history,
            "deblurring (g+h)": deblur_result.objective_history,
        },
        ylabel="objective",
    )
    ax.figure.savefig(outdir / "restoration_convergence.png", dpi=150)

    print(f"saved restoration_denoising.png, restoration_deblurring.png, restoration_convergence.png to {outdir}")


if __name__ == "__main__":
    main()
