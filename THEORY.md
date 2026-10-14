# Theory

This document lays out the common mathematical framework shared by the three
methods in this toolkit, then summarizes each method individually, with the
exact formulas each `imrestore.algorithms` module implements, transcribed
from the papers' own algorithm boxes.

## 1. Problem setup

Image restoration problems in this toolkit are posed as composite convex
optimization / monotone-inclusion problems of the form

```
min_x  h(x) + g(x)                      (composite / FBS form)
min_x max_y  <Kx, y> + h(x) - f*(y)     (saddle-point / primal-dual form)
```

where, depending on the task:

- **Denoising**: `h(x) = 1/2 ||x - b||^2` (data fidelity), `g` or the
  `<Kx,y>` term encodes a regularizer such as total variation (TV).
- **Deblurring**: `h(x) = 1/2 ||Ax - b||^2` with `A` a blur/convolution
  operator, same regularizer choices as above.
- `f*` is the convex conjugate used in the primal-dual (Fenchel–Rockafellar)
  formulation, e.g. the conjugate of the TV norm.

All three methods below solve instances of this problem without requiring a
known Lipschitz constant of `∇h` or a fixed step size tuned by hand.

## 2. Adaptive Gradient Descent

**Paper:** Malitsky & Mishchenko, *Adaptive Gradient Descent without
Descent*, ICML 2020 ([arXiv:1910.09529](https://arxiv.org/abs/1910.09529)).
Implemented in `imrestore.algorithms.adaptive_gd`.

### Algorithm 1 — AdGD (proven)

```
Input: x^0, λ_0 > 0, θ_0 = +∞
x^1 = x^0 - λ_0 ∇f(x^0)
for k = 1, 2, ...:
    λ_k = min{ sqrt(1 + θ_{k-1}) λ_{k-1},  ||x^k - x^{k-1}|| / (2||∇f(x^k) - ∇f(x^{k-1})||) }
    x^{k+1} = x^k - λ_k ∇f(x^k)
    θ_k = λ_k / λ_{k-1}
```

Convention: `1/0 = +∞`, i.e. if the gradient difference is zero the second
term is dropped from the `min`.

**Theorem 1 (paper).** If `f` is convex with locally Lipschitz gradient,
`(x^k)` converges to a solution and
`f(x̂^k) - f_* ≤ D / (2 S_k) = O(1/k)`, where `x̂^k` is a specific weighted
average of the iterates and `D` depends only on the initial data.

**Theorem 2 (paper).** If `f` is additionally locally strongly convex
(constant `μ` on the relevant compact set, local smoothness `L`), AdGD
converges linearly: `O(κ log(1/ε))` iterations to reach `||x^k - x*||² ≤ ε`,
with `κ = L/μ` — both constants *local*, so the rate can beat global-Lipschitz
GD even though AdGD never estimates `L` or `μ` explicitly.

### Algorithm 2 — AdGD-accel (heuristic, no proof)

```
Input: x^0, λ_0 > 0, Λ_0 > 0, θ_0 = Θ_0 = +∞
y^1 = x^1 = x^0 - λ_0 ∇f(x^0)
for k = 1, 2, ...:
    λ_k = min{ sqrt(1 + θ_{k-1}/2) λ_{k-1},  ||x^k - x^{k-1}|| / (2||∇f(x^k) - ∇f(x^{k-1})||) }
    Λ_k = min{ sqrt(1 + Θ_{k-1}/2) Λ_{k-1},  ||∇f(x^k) - ∇f(x^{k-1})|| / (2||x^k - x^{k-1}||) }
    β_k = (sqrt(1/λ_k) - sqrt(Λ_k)) / (sqrt(1/λ_k) + sqrt(Λ_k))
    y^{k+1} = x^k - λ_k ∇f(x^k)
    x^{k+1} = y^{k+1} + β_k (y^{k+1} - y^k)
    θ_k = λ_k/λ_{k-1},  Θ_k = Λ_k/Λ_{k-1}
```

`λ_k` estimates the inverse local smoothness (as in AdGD); `Λ_k` estimates
the inverse local strong-convexity constant using the same online rule
applied to the *dual* quantities. The paper states explicitly (Section 3,
"Heuristics") that this method has **no formal convergence guarantee** —
it is included because it performs well empirically, not because it is
proven. `imrestore` implements it as-is and documents this caveat in the
docstring.

Role in this toolkit: baseline smooth solver, and a building block inside
proximal-gradient (forward-backward) variants for regularized restoration.

## 3. Inertial Quasi-Newton Forward-Backward Splitting

**Paper:** Wang, Fadili & Ochs, *Inertial Quasi-Newton Methods for Monotone
Inclusion: Efficient Resolvent Calculus and Primal-Dual Methods*, 2022
([arXiv:2209.14019](https://arxiv.org/abs/2209.14019)).
Implemented in `imrestore.algorithms.quasi_newton_fbs`.

Solves `0 ∈ Ax + Bx` where `B` is `β`-cocoercive (e.g. `∇f` of an
`L`-smooth convex `f`, `β = 1/L`) and `A` is maximally monotone with an
easy resolvent under a *scalar* metric (e.g. `A = ∂g`, resolvent =
`prox_{g/θ}`). `imrestore` implements the paper's **Algorithm 1**
(inertial variant) restricted to a scalar base metric `M_0 = θ_0 I` with a
single rank-1 ("0SR1") correction per iteration (paper's Section 5) — the
simplest case the paper's framework covers, and the one for which the
resolvent calculus (Theorem 4.1 / Corollary 4.3) reduces to a 1-D
root-finding problem solvable by bisection (paper's Algorithm 4), rather
than the general semismooth-Newton machinery needed for rank-`r` metrics.

### 0SR1 rank-1 metric (Section 5, eq. 28-31)

Given the secant pair `s_k = z_k - z_{k-1}`, `y_k = Bz_k - Bz_{k-1}`:

```
if <y_k - θ_0 s_k, s_k> == 0:  M_k = M_0                      (skip update)
else:
    û_k = (y_k - θ_0 s_k) / sqrt(|<y_k - θ_0 s_k, s_k>|)
    M_k = M_0 + sign · γ_k û_k û_k^T,   sign = sign(<y_k - θ_0 s_k, s_k>)
```

`γ_k` is a free positive scale (Lemma 5.1): any bounded-away-from-0/∞
constant works for the `+` branch; the `-` branch must additionally stay
below `c² / (1/β + θ_0)²` with `c = θ_0 - 1/β` to keep `M_k` positive
definite — `imrestore` clips `γ_k` to this bound automatically.

### Resolvent calculus for the rank-1 update (Prop. 4.8-4.9)

Rather than forming `M_k` and inverting it, the forward-backward step
`J_A^{M_k}(z̄_k - M_k^{-1} B z̄_k)` is computed as:

```
w = z̄_k - Bz̄_k / θ_0                                    (forward step wrt M_0)
find ξ* : u·(z̄_k - J_A^{M_0}(w - sign·ξ·u/θ_0)) + ξ = 0   (1-D, strictly increasing)
z_{k+1} = J_A^{M_0}(w - sign·ξ*·u/θ_0)
```

i.e. two evaluations of the *simple* resolvent `J_A^{M_0}` (= `prox_{g/θ_0}`
for `A = ∂g`) per bisection step, never an evaluation with respect to the
full rank-1-perturbed metric.

### Inertial step and convergence

`z̄_k = z_k + α_k(z_k - z_{k-1})`. **Theorem 3.1 (paper)** proves weak
convergence to a solution provided `α_k` follows a *summable-type*
schedule (`Σ_k α_k max{...} < ∞`), plus linear convergence if `A` or `B`
is strongly monotone. `imrestore` exposes a constant `inertial_alpha` for
convenience; a constant sequence is a common heuristic relaxation of the
paper's schedule, not covered by Theorem 3.1. Theorem 3.1 also needs the
paper's Assumption 1 on the metrics, `(1 + η_k) M_k ⪰ M_{k+1}` with summable
`η_k`, which Lemma 5.2 obtains with a rank-1 scale proportional to a summable
sequence. `imrestore` uses a constant `γ` (clipped as in Lemma 5.1(ii)), which
gives only Assumption 2, the one for the paper's relaxed Algorithm 2 (not
implemented). So no theorem in the paper covers the iteration as
implemented, even with `inertial_alpha=0`; it converges on all the test and
demo problems.

Role in this toolkit: applied to deblurring/denoising as an inclusion
problem (`A = ∂(regularizer)`, `B = ∇(data fidelity)`), including
box-constrained and other simple-`g` variants.

## 4. PDAL and Variable-Metric PDAL (VarPDAL)

Both solve the saddle-point problem `min_x max_y <Kx,y> + g(x) [+ h(x)] - f*(y)`
with `g, f*` simple (prox-friendly) and `K` a bounded linear operator.
Implemented in `imrestore.algorithms.var_pdal` (`pdal`, `var_pdal`).

### PDAL (Malitsky & Pock, SIAM J. Optim. 28(1), 2018, [arXiv:1608.08883](https://arxiv.org/abs/1608.08883))

Solves the `h ≡ 0` case. Replaces the fixed step-size condition
`στ||K||² < 1` (which requires estimating `||K||`) with an adaptive
**linesearch** on the primal step `τ_k`:

```
Init: x^0, y^1, τ_0 > 0, μ, δ ∈ (0,1), β > 0, θ_0 = 1
for k = 1, 2, ...:
    x^k = prox_{τ_{k-1} g}(x^{k-1} - τ_{k-1} K* y^k)
    pick trial τ_k ∈ [τ_{k-1}, τ_{k-1}√(1+θ_{k-1})]     (imrestore starts at the upper end)
    repeat (linesearch):
        θ_k = τ_k / τ_{k-1}
        x̄^k = x^k + θ_k(x^k - x^{k-1})
        y^{k+1} = prox_{βτ_k f*}(y^k + βτ_k K x̄^k)
        break if √β·τ_k·||K*y^{k+1} - K*y^k|| ≤ δ·||y^{k+1} - y^k||
        else τ_k ← τ_k·μ
```

**Theorem 1 (paper).** `(x^k, y^k)` is bounded and every cluster point
solves the saddle-point problem; if `g|_{dom g}` is continuous and `(τ_k)`
stays bounded above, the whole sequence converges. Ergodic `O(1/N)` rate;
the paper also gives accelerated `O(1/N²)` variants when `g` or `f*` is
strongly convex (not implemented here).

### VarPDAL (Wang, Fadili & Ochs, [arXiv:2405.06824](https://arxiv.org/abs/2405.06824))

Solves the more general `min_x max_y <Kx,y> + g(x) + h(x) - f*(y)` with `h`
convex, `L_h`-Lipschitz gradient. Same linesearch idea, but now
linesearching the *dual* step `σ_k` (the paper's convention, chosen so the
linesearch lands on the metric-aware primal update) and the primal proximal
step uses a quasi-Newton metric `M_k`:

```
Init: x^1, y^0, σ_0 > 0, μ, δ ∈ (0,1), β > 0, θ_0 = 1
for k = 1, 2, ...:
    compute M_k (quasi-Newton framework, Section 5)
    y^k = prox_{σ_{k-1} f*}(y^{k-1} + σ_{k-1} K x^k)
    pick trial σ_k ∈ [σ_{k-1}, σ_{k-1}√(1+θ_{k-1})]   (β_k ≡ β here; imrestore starts at the upper end)
    repeat (linesearch):
        θ_k = σ_k / σ_{k-1};  τ_k = β·σ_k
        ȳ^k = y^k + θ_k(y^k - y^{k-1})
        x^{k+1} = prox_{τ_k g}^{M_k}(x^k - τ_k M_k^{-1} K* ȳ^k - τ_k M_k^{-1} ∇h(x^k))
        break if τ_k σ_k ||Kx^{k+1}-Kx^k||² + 2τ_k(h(x^{k+1})-h(x^k)-<∇h(x^k),x^{k+1}-x^k>) ≤ δ||x^{k+1}-x^k||²_{M_k}
        else σ_k ← σ_k·μ
```

**Theorem (paper).** Bounded and weakly convergent to a saddle point under
the metric-boundedness Assumption 1 (analogous to `quasi_newton_fbs`'s
Assumption 1); `O(1/N)` primal convergence, `O(1/N²)` ergodic gap.

**Metric `M_k` — scope of this implementation.** The paper's own Section 5
default is a general multi-memory L-BFGS/L-SR1 metric (compact
`M_{k,0} + U₁U₁ᵀ - U₂U₂ᵀ` representation, rank ≤ 2m, eq. 10-17) with a
coupled semismooth-Newton resolvent solver (its Algorithm 2, Theorem 6.1).
`imrestore` instead reuses the **same rank-1 ("0SR1") metric and resolvent
calculus** already implemented for `quasi_newton_fbs` (companion paper,
§3 above) — secant pairs built from consecutive `∇h(x^k)` evaluations,
`M_k = θ_0 I + sign·γ_k û_k û_k^T`, resolved via the identical 1-D
root-find/bisection. This keeps both the metric update and its resolvent
`O(d)` and consistent across the toolkit; the general multi-memory metric
is a documented extension, not yet implemented.

Role in this toolkit: the primary saddle-point solver — `pdal` for
TV-type regularization with a *simple* data term (`g`), `var_pdal` when the
data term is smooth-but-not-simple (e.g. quadratic deblurring, KL-divergence
under Poisson noise) and folded into `h` instead.

## 5. Comparison

| | Adaptive GD | Quasi-Newton FBS | (Var)PDAL |
|---|---|---|---|
| Problem form | smooth / composite | monotone inclusion | saddle-point |
| Step-size source | adaptive scalar | fixed / quasi-Newton metric | linesearch (+ optional metric) |
| Curvature-aware | no | yes | optional (Var) |
| Needs `L` or `\|\|K\|\|`? | no | problem-dependent | no (linesearch) |
| Acceleration | Nesterov-style variant | inertial term | momentum via `θ_k` |
| Module | `adaptive_gd` | `quasi_newton_fbs` | `var_pdal` (`pdal`, `var_pdal`) |

## 6. References

1. Y. Malitsky, K. Mishchenko. *Adaptive Gradient Descent without Descent.*
   ICML 2020. arXiv:1910.09529.
2. S. Wang, J. Fadili, P. Ochs. *Inertial Quasi-Newton Methods for Monotone
   Inclusion: Efficient Resolvent Calculus and Primal-Dual Methods.* 2022.
   arXiv:2209.14019.
3. Y. Malitsky, T. Pock. *A First-Order Primal-Dual Algorithm with
   Linesearch.* SIAM J. Optim. 28(1), 2018, pp. 411–432. arXiv:1608.08883.
4. S. Wang, J. Fadili, P. Ochs. *A Quasi-Newton Primal-Dual Algorithm with
   Line Search.* 2024. arXiv:2405.06824.
