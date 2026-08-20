# Figures

Generated plots from the `scripts/` demos, checked in so the README/paper
write-ups can reference stable images.

**Algorithm-core sanity checks** (synthetic problems, no image data):

- `adaptive_gd_convergence.png` — AdGD vs. AdGD-accel on an ill-conditioned
  strongly-convex quadratic (κ=500), from `scripts/demo_adaptive_gd.py`.
- `quasi_newton_fbs_convergence.png` — inertial quasi-Newton FBS (0SR1
  metric) vs. projected gradient descent on a box-constrained quadratic
  (κ=300), from `scripts/demo_quasi_newton_fbs.py`.
- `var_pdal_convergence.png` — PDAL vs. VarPDAL (0SR1 metric) on a
  box-constrained saddle-point problem, each plotted against its own
  closed-form reference (PDAL solves a `g`-only problem, VarPDAL solves the
  same problem plus a smooth `h` term), from `scripts/demo_var_pdal.py`.

**End-to-end restoration** (Shepp-Logan phantom, via `var_pdal` +
`TotalVariation`, from `scripts/restoration_demo.py`):

- `restoration_denoising.png` — ground truth / noisy (σ=0.08) / denoised
  (PSNR 23.6→25.8 dB, SSIM 0.40→0.62). `tv_weight=0.1` is deliberately past
  the PSNR-maximizing value (~0.05, PSNR 27.8) — a visual sweep from 0.05 to
  0.8 showed 0.1 is where the result actually *looks* clean (flat regions,
  sharp organ boundaries, no visible speckle) while matching SSIM and
  costing only ~2 dB PSNR; past ~0.2 it visibly over-smooths and distorts
  anatomy. PSNR-argmax and visual quality aren't the same optimum for TV
  denoising.
- `restoration_deblurring.png` — ground truth / Gaussian-blurred+noisy
  (σ=1.0 blur) / deblurred (PSNR 19.7→29.3 dB, SSIM 0.76→0.93), visually
  close to indistinguishable from ground truth. Here (unlike denoising)
  the visual-quality sweep and the PSNR-maximizing weight landed at the
  same place (`tv_weight=0.001`). Needs `max_iter=3000` (not `var_pdal`'s
  usual few hundred) to fully converge at this low weight — checked
  against an independent, separately-coded Chambolle-Pock reference; 200
  iterations undershoots the true optimum by several dB.
- `restoration_convergence.png` — objective history for both runs.
