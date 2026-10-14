# Theory

This document lays out the common mathematical framework shared by the three
methods in this toolkit, then summarizes each method individually, with the
exact formulas each `imrestore.algorithms` module implements, transcribed
from the papers' own algorithm boxes.

## 1. Problem setup

Image restoration problems in this toolkit are posed as composite convex
optimization or monotone-inclusion problems, in one of two forms:

$$
\min_x \thinspace h(x) + g(x) \qquad \text{(composite, forward-backward form)}
$$

$$
\min_x \max_y \thinspace \langle Kx, y\rangle + g(x) + h(x) - f^{\ast}(y) \qquad \text{(saddle-point, primal-dual form)}
$$

where, depending on the task:

- **Denoising**: $h(x) = \tfrac{1}{2}\lVert x - b\rVert^2$ (data fidelity), and
  $g$ or the $\langle Kx, y\rangle$ term encodes a regularizer such as total
  variation (TV).
- **Deblurring**: $h(x) = \tfrac{1}{2}\lVert Ax - b\rVert^2$ with $A$ a blur
  (convolution) operator, with the same regularizer choices.
- $f^{\ast}$ is the convex conjugate used in the primal-dual
  (Fenchel–Rockafellar) formulation. For TV, $K = \nabla$ and
  $f = \alpha\lVert\cdot\rVert_{2,1}$, so $f^{\ast}$ is the indicator of the
  dual-norm ball $\lbrace p : \lVert p\rVert_{2,\infty} \le \alpha\rbrace$.

AdGD and PDAL need no Lipschitz constant and no hand-tuned step size.
QN-FBS needs the cocoercivity constant $\beta$ of $B$, and VarPDAL the
Lipschitz constant $L_h$ of $\nabla h$, because these set their metrics. No
method needs the operator norm $\lVert K\rVert$.

## 2. Adaptive Gradient Descent

**Paper:** Malitsky & Mishchenko, *Adaptive Gradient Descent without
Descent*, ICML 2020 ([arXiv:1910.09529](https://arxiv.org/abs/1910.09529)).
Implemented in `imrestore.algorithms.adaptive_gd`.

### Algorithm 1: AdGD (proven)

Input $x^0$, $\lambda_0 > 0$ and $\theta_0 = +\infty$, and set
$x^1 = x^0 - \lambda_0 \nabla f(x^0)$. For $k = 1, 2, \dots$:

$$
\lambda_k = \min\left\lbrace \sqrt{1 + \theta_{k-1}}\thinspace\lambda_{k-1}, \quad \frac{\lVert x^k - x^{k-1}\rVert}{2\lVert \nabla f(x^k) - \nabla f(x^{k-1})\rVert} \right\rbrace,
$$

$$
x^{k+1} = x^k - \lambda_k \nabla f(x^k), \qquad \theta_k = \lambda_k / \lambda_{k-1}.
$$

Convention: $1/0 = +\infty$, that is, if the gradient difference is zero the
second term is dropped from the minimum.

**Theorem 1 (paper).** If $f$ is convex with locally Lipschitz gradient,
$(x^k)$ converges to a solution and
$f(\hat x^k) - f_{\ast} \le D/(2S_k) = O(1/k)$, where $\hat x^k$ is a specific
weighted average of the iterates and $D$ depends only on the initial data.

**Theorem 2 (paper).** If $f$ is additionally locally strongly convex
(constant $\mu$ on the relevant compact set, local smoothness $L$), AdGD
converges linearly: $O(\kappa \log(1/\varepsilon))$ iterations reach
$\lVert x^k - x^{\ast}\rVert^2 \le \varepsilon$, with $\kappa = L/\mu$. Both
constants are *local*, so the rate can beat gradient descent with a global
Lipschitz constant, even though AdGD never estimates $L$ or $\mu$ explicitly.

### Algorithm 2: AdGD-accel (heuristic, no proof)

Input $x^0$, $\lambda_0 > 0$, $\Lambda_0 > 0$ and
$\theta_0 = \Theta_0 = +\infty$, and set
$y^1 = x^1 = x^0 - \lambda_0 \nabla f(x^0)$. For $k = 1, 2, \dots$:

$$
\lambda_k = \min\left\lbrace \sqrt{1 + \theta_{k-1}/2}\thinspace\lambda_{k-1}, \quad \frac{\lVert x^k - x^{k-1}\rVert}{2\lVert \nabla f(x^k) - \nabla f(x^{k-1})\rVert} \right\rbrace,
$$

$$
\Lambda_k = \min\left\lbrace \sqrt{1 + \Theta_{k-1}/2}\thinspace\Lambda_{k-1}, \quad \frac{\lVert \nabla f(x^k) - \nabla f(x^{k-1})\rVert}{2\lVert x^k - x^{k-1}\rVert} \right\rbrace,
$$

$$
\beta_k = \frac{\sqrt{1/\lambda_k} - \sqrt{\Lambda_k}}{\sqrt{1/\lambda_k} + \sqrt{\Lambda_k}},
$$

$$
y^{k+1} = x^k - \lambda_k \nabla f(x^k), \qquad x^{k+1} = y^{k+1} + \beta_k (y^{k+1} - y^k),
$$

with $\theta_k = \lambda_k/\lambda_{k-1}$ and $\Theta_k = \Lambda_k/\Lambda_{k-1}$.

$\lambda_k$ estimates the inverse local smoothness $1/L$ (as in AdGD), and
$\Lambda_k$ estimates the local strong-convexity constant $\mu$ by the same
online rule applied to the reciprocal ratio, so that $\beta_k$ mimics
Nesterov's momentum $(\sqrt{L} - \sqrt{\mu})/(\sqrt{L} + \sqrt{\mu})$. The
paper states explicitly (Section 3, "Heuristics") that this method has **no
formal convergence guarantee**. It is included because it performs well
empirically, not because it is proven; `imrestore` implements it as is and
documents this caveat in the docstring.

Role in this toolkit: baseline smooth solver, and a building block inside
proximal-gradient (forward-backward) variants for regularized restoration.

## 3. Inertial Quasi-Newton Forward-Backward Splitting

**Paper:** Wang, Fadili & Ochs, *Inertial Quasi-Newton Methods for Monotone
Inclusion: Efficient Resolvent Calculus and Primal-Dual Methods*, 2022
([arXiv:2209.14019](https://arxiv.org/abs/2209.14019)).
Implemented in `imrestore.algorithms.quasi_newton_fbs`.

The method solves $0 \in Ax + Bx$, where $B$ is $\beta$-cocoercive (for
example $B = \nabla f$ for an $L$-smooth convex $f$, with $\beta = 1/L$) and
$A$ is maximally monotone with an easy resolvent under a *scalar* metric (for
example $A = \partial g$, whose resolvent is $\operatorname{prox}_{g/\theta}$).
`imrestore` implements the paper's **Algorithm 1** (inertial variant)
restricted to a scalar base metric $M_0 = \theta_0 I$ with
$\theta_0 > 1/\beta$ and a single rank-one ("0SR1") correction per iteration
(the paper's Section 5). This is the simplest case the paper's framework
covers, and the one for which the resolvent calculus (Theorem 4.1 and
Corollary 4.3) reduces to a one-dimensional root-finding problem solvable by
bisection (the paper's Algorithm 4), rather than the general semismooth Newton
machinery needed for metrics of rank $r$.

### 0SR1 rank-one metric (Section 5, equations 28 to 31)

Given the secant pair $s_k = z_k - z_{k-1}$ and $y_k = Bz_k - Bz_{k-1}$, let
$q_k = \langle y_k - \theta_0 s_k, s_k\rangle$. If $q_k = 0$, then
$M_k = M_0$ (the update is skipped). Otherwise, with
$\epsilon_k = \operatorname{sign}(q_k)$,

$$
\hat u_k = \frac{y_k - \theta_0 s_k}{\sqrt{|q_k|}}, \qquad M_k = M_0 + \epsilon_k \gamma_k \hat u_k \hat u_k^{T}.
$$

The scale $\gamma_k > 0$ is free within Lemma 5.1. On the minus branch it must
stay below $\bar\gamma = c^2/(1/\beta + \theta_0)^2$, with
$c = \theta_0 - 1/\beta$, to keep $M_k$ positive definite; `imrestore` uses
$\gamma_k = \min(\gamma, \bar\gamma/2)$ there, as in Lemma 5.1(ii), which gives
$M_k \succeq \tfrac{1}{2}(\theta_0 + 1/\beta) I$.

Because $B$ is $\beta$-cocoercive, it is $(1/\beta)$-Lipschitz, so
$\langle y_k, s_k\rangle \le \lVert s_k\rVert^2/\beta < \theta_0\lVert s_k\rVert^2$.
Hence $q_k < 0$ whenever $s_k \ne 0$: in practice only the minus branch
occurs, and the clip is always in force.

### Resolvent calculus for the rank-one update (Propositions 4.8 and 4.9)

Each iteration takes the forward-backward step in the metric $M_k$,

$$
z_{k+1} = (M_k + A)^{-1}(M_k \bar z_k - B\bar z_k) = J_A^{M_k}\bigl(\bar z_k - M_k^{-1} B\bar z_k\bigr),
$$

without forming or inverting $M_k$. With $w = \bar z_k - B\bar z_k/\theta_0$,
$u = \sqrt{\gamma_k}\thinspace\hat u_k$ and

$$
p(\xi) = J_A^{\theta_0 I}\bigl(w - \epsilon_k \xi u/\theta_0\bigr),
$$

the scalar $\xi^{\ast}$ solves
$\langle u, \bar z_k - p(\xi)\rangle + \xi = 0$, whose left side is strictly
increasing in $\xi$. It is found by expanding a bracket around zero and
bisecting, and then $z_{k+1} = p(\xi^{\ast})$. Each bisection step evaluates
the simple resolvent $J_A^{\theta_0 I}$ (which is
$\operatorname{prox}_{g/\theta_0}$ for $A = \partial g$) once; no resolvent
with respect to the full rank-one-perturbed metric is ever evaluated.

### Inertial step and convergence

The inertial point is $\bar z_k = z_k + \alpha_k(z_k - z_{k-1})$.
**Theorem 3.1 (paper)** proves weak convergence to a solution provided
$\alpha_k$ follows a summable-type schedule, plus linear convergence if $A$ or
$B$ is strongly monotone. `imrestore` exposes a constant `inertial_alpha` for
convenience; a constant sequence is a common heuristic relaxation of the
paper's schedule, not covered by Theorem 3.1. Theorem 3.1 also needs the
paper's Assumption 1 on the metrics, $(1 + \eta_k) M_k \succeq M_{k+1}$ with
summable $\eta_k$, which Lemma 5.2 obtains with a rank-one scale proportional
to a summable sequence. `imrestore` uses a constant $\gamma$ (clipped as in
Lemma 5.1(ii)), which gives only Assumption 2, the one for the paper's relaxed
Algorithm 2 (not implemented). So no theorem in the paper covers the iteration
as implemented, even with `inertial_alpha=0`; it converges on all the test and
demo problems.

Role in this toolkit: applied to deblurring and denoising as an inclusion
problem, with $A = \partial(\text{regularizer})$ and
$B = \nabla(\text{data fidelity})$, including box constraints and other
simple choices of $g$.

## 4. PDAL and Variable-Metric PDAL (VarPDAL)

Both solve the saddle-point problem
$\min_x \max_y \langle Kx, y\rangle + g(x) + h(x) - f^{\ast}(y)$ with $g$ and
$f^{\ast}$ simple (prox-friendly) and $K$ a bounded linear operator.
Implemented in `imrestore.algorithms.var_pdal` (`pdal`, `var_pdal`).

### PDAL (Malitsky & Pock, SIAM J. Optim. 28(1), 2018, [arXiv:1608.08883](https://arxiv.org/abs/1608.08883))

PDAL solves the case $h \equiv 0$. It replaces the fixed step-size condition
$\sigma\tau\lVert K\rVert^2 < 1$, which requires an estimate of
$\lVert K\rVert$, with an adaptive **linesearch** on the primal step
$\tau_k$. Initialize $x^0$, $y^1$, $\tau_0 > 0$, $\mu, \delta \in (0, 1)$,
$\beta > 0$ and $\theta_0 = 1$. For $k = 1, 2, \dots$:

1. Primal step: $x^k = \operatorname{prox}\nolimits_{\tau_{k-1} g}\bigl(x^{k-1} - \tau_{k-1} K^{\ast} y^k\bigr)$.
2. Pick a trial $\tau_k \in [\tau_{k-1}, \tau_{k-1}\sqrt{1 + \theta_{k-1}}]$
   (`imrestore` starts at the upper end).
3. Linesearch: set $\theta_k = \tau_k/\tau_{k-1}$,
   $\bar x^k = x^k + \theta_k(x^k - x^{k-1})$ and
   $y^{k+1} = \operatorname{prox}_{\beta\tau_k f^{\ast}}\bigl(y^k + \beta\tau_k K\bar x^k\bigr)$.
   Accept if
   $\sqrt{\beta}\thinspace\tau_k\lVert K^{\ast} y^{k+1} - K^{\ast} y^k\rVert \le \delta\lVert y^{k+1} - y^k\rVert$;
   otherwise set $\tau_k \leftarrow \mu\tau_k$ and repeat step 3.

**Theorem 1 (paper).** $(x^k, y^k)$ is bounded and every cluster point solves
the saddle-point problem; if $g$ is continuous on its domain and $(\tau_k)$
stays bounded above, the whole sequence converges. The ergodic rate is
$O(1/N)$; the paper also gives accelerated $O(1/N^2)$ variants when $g$ or
$f^{\ast}$ is strongly convex (not implemented here).

### VarPDAL (Wang, Fadili & Ochs, [arXiv:2405.06824](https://arxiv.org/abs/2405.06824))

VarPDAL solves the general problem, with $h$ convex and $\nabla h$
$L_h$-Lipschitz. It uses the same linesearch idea, but on the *dual* step
$\sigma_k$ (the paper's convention, chosen so that the linesearch lands on the
metric-aware primal update), and the primal proximal step uses a quasi-Newton
metric $M_k$. Initialize $x^1$, $y^0$, $\sigma_0 > 0$,
$\mu, \delta \in (0, 1)$, $\beta > 0$ and $\theta_0 = 1$. For
$k = 1, 2, \dots$:

1. Compute $M_k$ (quasi-Newton framework, Section 5).
2. Dual step: $y^k = \operatorname{prox}\nolimits_{\sigma_{k-1} f^{\ast}}\bigl(y^{k-1} + \sigma_{k-1} K x^k\bigr)$.
3. Pick a trial $\sigma_k \in [\sigma_{k-1}, \sigma_{k-1}\sqrt{1 + \theta_{k-1}}]$
   (with $\beta_k \equiv \beta$; `imrestore` starts at the upper end).
4. Linesearch: set $\theta_k = \sigma_k/\sigma_{k-1}$, $\tau_k = \beta\sigma_k$,
   $\bar y^k = y^k + \theta_k(y^k - y^{k-1})$ and compute $x^{k+1}$ from the
   update below. Accept if the inequality below holds; otherwise set
   $\sigma_k \leftarrow \mu\sigma_k$ and repeat step 4.

$$
x^{k+1} = \operatorname{prox}^{M_k}_{\tau_k g}\bigl(x^k - \tau_k M_k^{-1} K^{\ast}\bar y^k - \tau_k M_k^{-1}\nabla h(x^k)\bigr)
$$

$$
\tau_k\sigma_k\lVert Kx^{k+1} - Kx^k\rVert^2 + 2\tau_k\bigl(h(x^{k+1}) - h(x^k) - \langle \nabla h(x^k), x^{k+1} - x^k\rangle\bigr) \le \delta\lVert x^{k+1} - x^k\rVert_{M_k}^2
$$

**Theorem (paper).** The iterates are bounded and converge weakly to a saddle
point under the metric-boundedness Assumption 1 (analogous to the assumption
for `quasi_newton_fbs`), with $O(1/N)$ primal convergence and an $O(1/N^2)$
ergodic gap.

**Metric $M_k$: scope of this implementation.** The paper's own Section 5
default is a general multi-memory L-BFGS/L-SR1 metric (compact representation
$M_{k,0} + U_1U_1^{T} - U_2U_2^{T}$ of rank at most $2m$, equations 10 to 17)
with a coupled semismooth Newton resolvent solver (its Algorithm 2 and
Theorem 6.1). `imrestore` instead reuses the **same rank-one ("0SR1") metric
and resolvent calculus** already implemented for `quasi_newton_fbs` (the
companion paper, Section 3 above): secant pairs built from consecutive
$\nabla h(x^k)$ evaluations, $M_k = \theta_0 I + \epsilon_k\gamma_k\hat u_k\hat u_k^{T}$,
resolved by the identical one-dimensional root-finding. This keeps both the
metric update and its resolvent $O(d)$ and consistent across the toolkit; the
general multi-memory metric is a documented extension, not yet implemented.

Role in this toolkit: the primary saddle-point solver. `pdal` handles
TV-type regularization with a *simple* data term ($g$), and `var_pdal` handles
a data term that is smooth but not simple (for example quadratic deblurring,
or the Kullback–Leibler divergence under Poisson noise), folded into $h$.

## 5. Comparison

| | Adaptive GD | Quasi-Newton FBS | (Var)PDAL |
|---|---|---|---|
| Problem form | smooth or composite | monotone inclusion | saddle-point |
| Step-size source | adaptive scalar | fixed quasi-Newton metric | linesearch (plus optional metric) |
| Curvature-aware | no | yes | optional (VarPDAL) |
| Constants needed | none | cocoercivity $\beta$ of $B$ | PDAL none; VarPDAL $L_h$; never $\lVert K\rVert$ |
| Acceleration | Nesterov-style variant | inertial term | momentum via $\theta_k$ |
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
