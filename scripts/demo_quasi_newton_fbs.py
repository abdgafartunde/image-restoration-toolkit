"""Run inertial quasi-Newton FBS on an ill-conditioned box-constrained
quadratic and save a convergence figure to figures/ (or --outdir DIR), comparing against
plain (non-adaptive-metric) projected gradient descent as a baseline.

Not a restoration benchmark yet (operators/ isn't wired in); this is a
sanity-check / illustration for the algorithm core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _cli import output_dir

from imrestore.algorithms.quasi_newton_fbs import inertial_quasi_newton_fbs


def make_box_constrained_quadratic(d=40, condition_number=300.0, seed=0):
    rng = np.random.default_rng(seed)
    eigs = np.geomspace(1.0, condition_number, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Q @ np.diag(eigs) @ Q.T
    x_shift = rng.uniform(-1.0, 1.0, d)
    b = A @ x_shift
    L = eigs.max()

    def objective(x):
        return 0.5 * x @ (A @ x) - b @ x

    def gradient(x):
        return A @ x - b

    return objective, gradient, L


def projected_gd(gradient, x0, lo, hi, step, n_iter, objective):
    x = x0.copy()
    history = [objective(x)]
    for _ in range(n_iter):
        x = np.clip(x - step * gradient(x), lo, hi)
        history.append(objective(x))
    return x, history


def main():
    outdir = output_dir(__doc__.splitlines()[0])
    lo, hi = -0.3, 0.3
    objective, gradient, L = make_box_constrained_quadratic()
    x0 = np.zeros(40)
    box_resolvent = lambda v, theta: np.clip(v, lo, hi)

    n_iter = 300
    result = inertial_quasi_newton_fbs(
        B=gradient,
        resolvent_A=box_resolvent,
        z0=x0,
        beta=1.0 / L,
        objective=objective,
        max_iter=n_iter,
        tol=0.0,
    )
    _, pgd_history = projected_gd(gradient, x0, lo, hi, 1.0 / L, n_iter, objective)

    f_ref = objective(result.x)  # best available reference (converged QN-FBS point)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.maximum(np.array(result.objective_history) - f_ref, 1e-16), label="quasi-Newton FBS (0SR1)")
    ax.semilogy(np.maximum(np.array(pgd_history) - f_ref, 1e-16), label="projected GD (fixed step 1/L)")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$f(z^k) - f_{\mathrm{ref}}$")
    ax.set_title(r"Box-constrained quadratic, $\kappa=300$")
    ax.legend()
    fig.tight_layout()

    out_path = outdir / "quasi_newton_fbs_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
