"""Run AdGD and AdGD-accel on an ill-conditioned strongly-convex quadratic
and save a convergence figure to figures/.

Not a benchmark against blurring/denoising yet (operators aren't
implemented); this is a sanity-check / illustration for the optimizer core
ahead of wiring it into the restoration examples.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imrestore.algorithms.adaptive_gd import (
    accelerated_adaptive_gradient_descent,
    adaptive_gradient_descent,
)


def make_illconditioned_quadratic(d=50, condition_number=500.0, seed=0):
    rng = np.random.default_rng(seed)
    eigs = np.geomspace(1.0, condition_number, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Q @ np.diag(eigs) @ Q.T
    x_star = rng.standard_normal(d)
    b = A @ x_star

    def objective(x):
        return 0.5 * x @ (A @ x) - b @ x

    def gradient(x):
        return A @ x - b

    return objective, gradient, x_star, objective(x_star)


def main():
    objective, gradient, x_star, f_star = make_illconditioned_quadratic()
    x0 = np.zeros_like(x_star)

    results = {
        "AdGD": adaptive_gradient_descent(gradient, x0, objective=objective, max_iter=800),
        "AdGD-accel": accelerated_adaptive_gradient_descent(
            gradient, x0, objective=objective, max_iter=800
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for label, res in results.items():
        err = np.maximum(np.array(res.objective_history) - f_star, 1e-16)
        axes[0].semilogy(err, label=label)
        axes[1].semilogy(res.step_size_history, label=label)

    axes[0].set_xlabel("iteration k")
    axes[0].set_ylabel(r"$f(x^k) - f_*$")
    axes[0].set_title("Convergence (ill-conditioned quadratic, $\\kappa=500$)")
    axes[0].legend()

    axes[1].set_xlabel("iteration k")
    axes[1].set_ylabel(r"step size $\lambda_k$")
    axes[1].set_title("Adaptive step size (no tuning, no line search)")
    axes[1].legend()

    fig.tight_layout()
    out_path = "figures/adaptive_gd_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
