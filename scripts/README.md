# Scripts

- `demo_adaptive_gd.py` — runs AdGD / AdGD-accel on a synthetic
  ill-conditioned quadratic and saves a convergence figure to `figures/`.
  A sanity check for the optimizer core, independent of image data.
- `demo_quasi_newton_fbs.py` — runs inertial quasi-Newton FBS (0SR1
  metric) on a box-constrained quadratic vs. projected gradient descent,
  saves a convergence figure to `figures/`.
- `demo_var_pdal.py` — runs PDAL and VarPDAL (0SR1 metric) on box-constrained
  saddle-point problems, saves a convergence figure to `figures/`.
- `restoration_demo.py` — end-to-end TV-regularized denoising and
  deblurring on the Shepp-Logan phantom via `var_pdal`, wiring together
  `operators/` (blur, TV), `metrics/` (PSNR, SSIM), and `utils/` (phantom,
  noise, plotting). Prints before/after PSNR/SSIM and saves image grids +
  a convergence plot to `figures/`.

All three algorithm families (`imrestore/algorithms/`), the image operators,
metrics, and utilities are implemented and covered by the demos above.
`restoration_demo.py` is the full pipeline; the others are algorithm-core
sanity checks on synthetic (non-image) problems with closed-form or
long-run references.

Planned: `benchmark_methods.py` — a more thorough comparison across all
three algorithm families on the same restoration problems (convergence
speed, PSNR/SSIM vs. iteration), mirroring
`eit-reconstruction-toolkit/scripts/reconstruct_and_plot.py`.
