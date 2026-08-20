# Examples

- `01_restoration_benchmark.ipynb` — runs all three algorithm families on
  the same denoising problem (Shepp-Logan phantom), each solving the
  problem structure it's actually built for: `adaptive_gradient_descent`
  on unconstrained Tikhonov regularization, `inertial_quasi_newton_fbs` on
  box-constrained Tikhonov, and `var_pdal` on box-constrained TV (the only
  one of the three that can handle TV, since it has no closed-form prox).
  Shows why each algorithm exists rather than picking an artificial winner
  — TV clearly outperforms Tikhonov (24.4 vs. 21.3 dB PSNR here), but only
  `var_pdal` can use it. Pre-executed with real outputs; re-run via
  `jupyter nbconvert --to notebook --execute --inplace
  examples/01_restoration_benchmark.ipynb` (needs the `dev` extra:
  `pip install -e ".[dev]"`).

`scripts/restoration_demo.py` covers the VarPDAL half of this (denoising +
deblurring) as a plain script.
