"""Run PDAL and VarPDAL on box-constrained saddle-point problems and save a
convergence figure to figures/.

PDAL solves min_x max_y <x,y> + g(x) - f*(y) (no smooth term - it isn't
part of the base algorithm); VarPDAL solves the same problem plus a smooth
h(x) = 0.5*rho*||x-shift||^2, exercising its quasi-Newton metric on that
term. These are genuinely different problems with different optima, so
each curve is plotted against its *own* closed-form reference (see
tests/test_var_pdal.py for the derivation) rather than a shared one -
comparing both to the same reference would make PDAL look like it fails
to converge, when it is actually correctly solving the h-free problem it
was given.

Not a restoration benchmark yet (operators/ isn't wired in); this is a
sanity-check / illustration for the algorithm core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imrestore.algorithms.var_pdal import pdal, var_pdal


def project_box(x, lo, hi):
    return np.clip(x, lo, hi)


def main():
    rng = np.random.default_rng(2)
    d = 30
    lo, hi = -1.0, 1.0
    c = rng.uniform(-3.0, 3.0, d)
    rho = 5.0
    shift = rng.uniform(-1.0, 1.0, d)

    identity = lambda v: v
    prox_g = lambda v, tau: project_box(v, lo, hi)
    prox_f_conj = lambda v, sigma: (v + sigma * c) / (1.0 + sigma)
    h = lambda x: 0.5 * rho * np.sum((x - shift) ** 2)
    grad_h = lambda x: rho * (x - shift)

    # g(x) + sup_y <x,y> - f*(y) = g(x) + 0.5||x||^2 + <c,x>  (g(x)=0 on the box)
    objective_g_only = lambda x: 0.5 * np.sum(x**2) + c @ x
    objective_g_plus_h = lambda x: h(x) + objective_g_only(x)

    n_iter = 150
    res_pdal = pdal(
        K=identity, K_adjoint=identity, prox_g=prox_g, prox_f_conj=prox_f_conj,
        x0=np.zeros(d), y1=np.zeros(d), tau0=0.1, beta=1.0,
        objective=objective_g_only, max_iter=n_iter, tol=0.0,
    )
    res_var = var_pdal(
        K=identity, K_adjoint=identity, prox_g=prox_g, prox_f_conj=prox_f_conj,
        h=h, grad_h=grad_h, x1=np.zeros(d), y0=np.zeros(d), L_h=rho,
        sigma0=0.1, beta=1.0, objective=objective_g_plus_h, max_iter=n_iter, tol=0.0,
    )

    x_ref_pdal = project_box(-c, lo, hi)
    x_ref_var = project_box(-(c - rho * shift) / (1.0 + rho), lo, hi)
    f_ref_pdal = objective_g_only(x_ref_pdal)
    f_ref_var = objective_g_plus_h(x_ref_var)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.maximum(np.array(res_pdal.objective_history) - f_ref_pdal, 1e-16), label="PDAL (g only)")
    ax.semilogy(np.maximum(np.array(res_var.objective_history) - f_ref_var, 1e-16), label="VarPDAL, 0SR1 (g + h)")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$f(x^k) - f_*$  (each vs. its own reference)")
    ax.set_title("Box-constrained saddle point: PDAL vs. VarPDAL")
    ax.legend()
    fig.tight_layout()

    out_path = "figures/var_pdal_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
