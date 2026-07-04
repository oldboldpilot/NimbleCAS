# `nimblecas.optimize` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/optimize/optimize.cppm`

**Unconstrained numerical optimization over `R^n`** — minimize a scalar
objective `f : R^n -> R` by iterative descent. The module offers steepest
(gradient) descent, damped Newton, BFGS and limited-memory L-BFGS quasi-Newton,
nonlinear conjugate gradient, the derivative-free Nelder–Mead simplex, and
C.T. Kelley's implicit filtering (derivative-free, noise-aware, optionally
bound-constrained). Everything here is **IEEE-754 double-precision, LOCAL,
iterative** — this is the **floating-point** optimizer, so it does **not** touch
the exact `ratpoly` / `polynomial` layers.

The honesty boundary is stated up front because a numerical minimizer is easy to
over-trust:

- It finds a **LOCAL stationary point** (`grad ~ 0`), **not** a global minimum.
  There is no claim of global optimality and no claim of exactness.
- Results **depend on the initial guess `x0`** — different basins yield different
  minima.
- Convergence is **not guaranteed** for non-convex, ill-conditioned,
  discontinuous, or non-smooth objectives; on a convex, well-conditioned problem
  the methods converge, elsewhere they may stall, cycle, or crawl.
- Finite-difference gradients/Hessians carry discretization error `O(h^2)` plus
  round-off `O(eps/h)`, so a reported `grad_norm` is **approximate**.
- Line searches, damping, and the simplex are heuristics with tuning constants
  that trade robustness for guarantees.
- **A run that exhausts `max_iterations` is NOT an error.** It returns a valid
  `OptimizeResult` with `converged == false` and the best iterate found. Only
  genuinely invalid input (empty `x0`, a non-finite entry, or a non-finite
  objective/gradient at the start) is reported as `MathError::domain_error`.

```cpp
import nimblecas.optimize;   // namespace nimblecas::optimize
```

Depends on [`core`](core.md) only.

## Callable contracts

Objectives and derivatives are supplied as `std::function` callables. The `x`
argument is a **non-owning view valid only for the call**; copy it if you need to
retain it.

```cpp
using Objective = std::function<double(std::span<const double>)>;
using Gradient  = std::function<std::vector<double>(std::span<const double>)>;
using HessianFn = std::function<std::vector<double>(std::span<const double>)>;
```

| Alias | Contract |
| :--- | :--- |
| `Objective` | `x -> f(x)`, a scalar. Required by every optimizer. |
| `Gradient` | `x -> grad f(x)`, an `n`-vector (same length as `x`). **Optional**: a default-constructed (empty) `Gradient` selects a central **finite-difference** gradient over `f`. |
| `HessianFn` | `x -> Hessian`, **ROW-MAJOR** `n*n` (`H[i*n+j]`). **Optional** (Newton only): when absent a finite-difference Hessian is used — a difference of the gradient if one is given, else a second difference of `f`. |

### `CGVariant` — nonlinear-CG update rule

```cpp
enum class CGVariant : std::uint8_t {
    fletcher_reeves,  // beta = (g'.g') / (g.g)
    polak_ribiere,    // beta = max(0, g'.(g'-g) / (g.g))  (PR+, auto-restart)
};
```

## `Options` — tuning knobs

All defaults are conservative, general-purpose values.

```cpp
struct Options {
    // Convergence.
    double grad_tol = 1e-6;             // stop when ||grad f||_2 <= grad_tol.
    double step_tol = 1e-12;            // stop when ||x_{k+1}-x_k||_2 <= step_tol.
    std::size_t max_iterations = 1000;  // hard cap; hitting it => converged=false.

    // Finite differences.
    double fd_step = 1e-6;              // base step h; per-coord h_i = h*max(1,|x_i|).

    // Line search (Armijo sufficient-decrease + strong-Wolfe curvature).
    double armijo_c1 = 1e-4;            // sufficient-decrease constant (0 < c1 < c2 < 1).
    double wolfe_c2 = 0.9;             // curvature constant (0.9 quasi-Newton, 0.1 CG).
    double backtrack_rho = 0.5;        // Armijo step contraction factor.
    std::size_t max_line_search = 50;  // max line-search / zoom iterations.

    // L-BFGS history length (number of (s,y) pairs retained).
    std::size_t lbfgs_memory = 10;

    // Implicit filtering stencil-scale schedule.
    double imf_h0 = 0.5;        // initial finite-difference / stencil scale.
    double imf_h_min = 1e-8;    // finest scale; the schedule halves h down to this.

    // Nelder-Mead simplex coefficients.
    double nm_reflect = 1.0;
    double nm_expand = 2.0;
    double nm_contract = 0.5;
    double nm_shrink = 0.5;
};
```

The convergence tolerances are reused by the derivative-free methods with a
different meaning: Nelder–Mead treats `grad_tol` as the tolerance on the
**relative spread of objective values** across the simplex, and implicit
filtering compares `grad_tol` against a **projected-gradient stationarity
measure**.

## `OptimizeResult` — outcome of a run

```cpp
struct OptimizeResult {
    std::vector<double> x;        // best iterate found.
    double fx = 0.0;              // objective at x.
    std::size_t iterations = 0;   // iterations actually performed.
    bool converged = false;       // true iff a stopping tolerance was met (NOT max-iter).
    double grad_norm = 0.0;       // ||grad f(x)||_2 (approx; 0 reported for derivative-free).
};
```

`converged` is `true` **only** when a stopping tolerance (`grad_tol` or
`step_tol`) was met; hitting `max_iterations` leaves it `false`. `grad_norm` is
approximate whenever it comes from finite differences, and is reported as `0` by
Nelder–Mead (which uses no gradient).

## Public finite-difference utilities

Exported so callers (and the tests) can build or inspect derivatives directly.

```cpp
[[nodiscard]] auto finite_difference_gradient(const Objective& f, std::span<const double> x,
                                              double h) -> std::vector<double>;

[[nodiscard]] auto finite_difference_hessian(const Objective& f, const Gradient& grad,
                                             std::span<const double> x, double h)
    -> std::vector<double>;
```

| Function | Behavior |
| :--- | :--- |
| `finite_difference_gradient` | Central-difference gradient of `f` at `x`. Per-coordinate step `h_i = h * max(1, |x_i|)` balances truncation against round-off. `O(2n)` objective evaluations. |
| `finite_difference_hessian` | Row-major `n*n`, symmetrized. If `grad` is non-empty the Hessian is a central difference of the gradient (`O(2n)` gradient calls); otherwise a second difference of `f` (`O(2n^2)` objective calls). Carries the usual FD error. |

## Optimizers

Every optimizer returns `Result<OptimizeResult>`. An empty `Gradient` /
`HessianFn` argument selects finite differences. All live in namespace
`nimblecas::optimize`.

```cpp
[[nodiscard]] auto gradient_descent(Objective f, std::span<const double> x0,
                                    Gradient grad = {}, Options opts = {})
    -> Result<OptimizeResult>;

[[nodiscard]] auto newton_method(Objective f, std::span<const double> x0, Gradient grad = {},
                                 HessianFn hess = {}, Options opts = {})
    -> Result<OptimizeResult>;

[[nodiscard]] auto bfgs(Objective f, std::span<const double> x0, Gradient grad = {},
                        Options opts = {}) -> Result<OptimizeResult>;

[[nodiscard]] auto l_bfgs(Objective f, std::span<const double> x0, Gradient grad = {},
                          Options opts = {}) -> Result<OptimizeResult>;

[[nodiscard]] auto conjugate_gradient(Objective f, std::span<const double> x0,
                                      Gradient grad = {},
                                      CGVariant variant = CGVariant::polak_ribiere,
                                      Options opts = {}) -> Result<OptimizeResult>;

[[nodiscard]] auto nelder_mead(Objective f, std::span<const double> x0, Options opts = {})
    -> Result<OptimizeResult>;

[[nodiscard]] auto implicit_filtering(Objective f, std::span<const double> x0,
                                      std::span<const double> lo = {},
                                      std::span<const double> hi = {}, Options opts = {})
    -> Result<OptimizeResult>;
```

### `gradient_descent`

Steepest descent with an Armijo backtracking line search. Robust but only
**linearly** convergent, and slow on ill-conditioned (elongated) valleys. Uses
the supplied gradient or a finite-difference one.

### `newton_method`

**Damped Newton.** Solves `H p = -g` by dense LU with partial pivoting on the
Hessian (given or finite-difference). When `H` is not positive definite (or
singular) it adds a Levenberg-style multiple of the identity, `tau*I`, growing
`tau` until `p` is a genuine descent direction; if that fails it falls back to
steepest descent for that step. An Armijo backtracking line search then
globalizes the step. On a convex quadratic it lands in a couple of iterations.

### `bfgs`

**BFGS quasi-Newton** maintaining an **inverse-Hessian** approximation, with a
strong-Wolfe line search (falling back to Armijo backtracking if strong-Wolfe
cannot satisfy its conditions). The rank-2 update is **skipped** when the
curvature condition `s.y > 0` fails, which keeps the approximation positive
definite. Superlinear near the solution.

### `l_bfgs`

**Limited-memory BFGS**: the two-loop recursion over the last `lbfgs_memory`
`(s,y)` pairs with the Nocedal `H0 = (s.y)/(y.y) I` scaling. `O(n * memory)` per
step and no `n*n` storage — the method of choice for large `n`. Same line-search
strategy as `bfgs`.

### `conjugate_gradient`

**Nonlinear conjugate gradient** (Fletcher–Reeves or Polak–Ribière, selected by
`variant`) with a strong-Wolfe line search. Restarts along steepest descent
every `n` steps and whenever the new direction is not a descent direction. The
Polak–Ribière variant is `PR+` (the coefficient is clamped at `0`, giving an
automatic restart).

### `nelder_mead`

**Derivative-free** downhill simplex — no gradient is used or needed. Reflection
/ expansion / contraction / shrink operate on an `(n+1)`-vertex simplex seeded
around `x0`. Robust on non-smooth or noisy objectives but only linearly
convergent with no stationary-point guarantee. Here `converged` means the
simplex's **relative spread of objective values** fell below `grad_tol`;
`grad_norm` is reported as `0`.

### `implicit_filtering`

**C.T. Kelley's implicit filtering** — derivative-free, for **noisy /
nondifferentiable** objectives that are a smooth trend plus low-amplitude noise.
Each outer step forms a central finite-difference **"simplex gradient"** at the
current stencil scale `h`, takes a BFGS-style (or steepest-descent) step, and
accepts it only via an Armijo test that demands decrease beyond what the scale-`h`
model predicts — so the shrinking scale **filters the noise**. On a "stencil
failure" (no productive step at `h`) it **halves `h`** and retries, down to
`opts.imf_h_min`.

Optional **box constraints** `[lo, hi]` are honoured by projecting every iterate
and stencil point onto the box (bound-constrained implicit filtering); pass empty
`lo`/`hi` for the unconstrained problem. Stationarity is measured by the
**projected-gradient norm** `||x - proj(x - g)||_2`, which equals `||g||` at an
interior point and shrinks to `0` at a KKT point on the boundary.

Honesty: numerical, derivative-free, LOCAL. It suits functions with
low-amplitude noise on a smooth trend and gives **no** global-optimality
guarantee; convergence depends on the noise level relative to the scale schedule.
Non-convergence returns `converged == false` (not an error).

## Error model

The **only** failure is `MathError::domain_error`, reserved for genuinely invalid
input. Exhausting `max_iterations`, a stalled line search, or an
indefinite/singular Hessian are all **normal outcomes** that return a value with
`converged == false`.

| Condition | Result |
| :--- | :--- |
| `x0` is empty | `MathError::domain_error` |
| `x0` contains a non-finite entry (NaN / Inf) | `MathError::domain_error` |
| `f(x0)` is non-finite | `MathError::domain_error` |
| A supplied gradient at `x0` has the wrong length or a non-finite entry | `MathError::domain_error` |
| `implicit_filtering`: bounds length ≠ `x0` length, ragged/partial box, non-finite bound, or `lo[i] > hi[i]` | `MathError::domain_error` |
| Reached `max_iterations` without meeting a tolerance | value, `converged == false` |
| Line search stalled / Hessian unusable | value, `converged == false` (best iterate so far) |

`finite_difference_gradient` and `finite_difference_hessian` return a plain
`std::vector<double>` and do not signal errors.

## Worked examples

Mirroring `tests/optimize_tests.cpp`. The tolerances are generous because
these are local, IEEE-754, iteration-capped methods.

```cpp
import nimblecas.optimize;
import nimblecas.core;
namespace opt = nimblecas::optimize;

// Convex quadratic q(x) = 1/2 x^T A x - b^T x with SPD A = [[3,1],[1,2]], b = [1,1].
// Gradient A x - b; unique minimizer x* = A^{-1} b = (0.2, 0.4).
auto quad_f = [](std::span<const double> x) -> double {
    const double ax0 = 3.0 * x[0] + 1.0 * x[1];
    const double ax1 = 1.0 * x[0] + 2.0 * x[1];
    return 0.5 * (x[0] * ax0 + x[1] * ax1) - (x[0] + x[1]);
};
auto quad_grad = [](std::span<const double> x) -> std::vector<double> {
    return {3.0 * x[0] + 1.0 * x[1] - 1.0, 1.0 * x[0] + 2.0 * x[1] - 1.0};
};
const std::array<double, 2> quad_start{5.0, -3.0};

// Every method drives x -> (0.2, 0.4). Newton with a finite-difference Hessian
// (hess left empty) solves a quadratic in a couple of iterations.
opt::gradient_descent(quad_f, quad_start, quad_grad, opt::Options{}).value().x;   // ~ (0.2, 0.4)
opt::newton_method(quad_f, quad_start, quad_grad, {}, opt::Options{}).value().x;  // ~ (0.2, 0.4)
opt::bfgs(quad_f, quad_start, quad_grad, opt::Options{}).value().x;               // ~ (0.2, 0.4)
opt::l_bfgs(quad_f, quad_start, quad_grad, opt::Options{}).value().x;             // ~ (0.2, 0.4)
opt::conjugate_gradient(quad_f, quad_start, quad_grad,
                        opt::CGVariant::polak_ribiere, opt::Options{}).value().x; // ~ (0.2, 0.4)

// Derivative-free Nelder-Mead reaches the same minimizer (no gradient supplied).
opt::Options nm;
nm.grad_tol = 1e-12;
nm.max_iterations = 5000;
opt::nelder_mead(quad_f, quad_start, nm).value().x;                              // ~ (0.2, 0.4)

// Non-convex Rosenbrock f(x,y) = (1-x)^2 + 100 (y - x^2)^2, minimizer (1,1).
auto rosen_f = [](std::span<const double> x) -> double {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
};
auto rosen_grad = [](std::span<const double> x) -> std::vector<double> {
    const double b = x[1] - x[0] * x[0];
    return {-2.0 * (1.0 - x[0]) - 400.0 * x[0] * b, 200.0 * b};
};
const std::array<double, 2> rosen_start{-1.2, 1.0};
opt::Options ro;
ro.grad_tol = 1e-8;
ro.max_iterations = 2000;
opt::bfgs(rosen_f, rosen_start, rosen_grad, ro).value().x;                       // ~ (1, 1)
opt::newton_method(rosen_f, rosen_start, rosen_grad, {}, ro).value().x;          // ~ (1, 1)

// The public finite-difference gradient matches the analytic one.
const std::array<double, 2> p{0.7, -0.4};
opt::finite_difference_gradient(quad_f, p, 1e-6);  // ~ quad_grad(p) = A p - b

// A capped run is NOT an error: it returns converged == false and the best iterate.
opt::Options capped;
capped.max_iterations = 2;    // far too few for Rosenbrock.
capped.grad_tol = 1e-12;
auto r = opt::bfgs(rosen_f, rosen_start, rosen_grad, capped);
r.has_value();                // true  (no error)
r->converged;                 // false (hit the cap)
r->iterations;                // <= 2

// Implicit filtering on a noisy quadratic (smooth trend + low-amplitude oscillation).
auto quad_noisy = [&](std::span<const double> x) -> double {
    return quad_f(x) + 2e-3 * std::sin(40.0 * x[0]) * std::sin(40.0 * x[1]);
};
const std::span<const double> nobound{};
opt::Options imf;
imf.grad_tol = 1e-6;
imf.max_iterations = 3000;
// The scale schedule averages the oscillation away and locates x* = (0.2, 0.4);
// a plain FD-gradient descent, corrupted by the noise slope, stalls farther out.
opt::implicit_filtering(quad_noisy, quad_start, nobound, nobound, imf).value().x; // ~ (0.2, 0.4)

// Bound-constrained implicit filtering: box lo=(0.5,-1), hi=(2,1). The unconstrained
// min (0.2,0.4) is OUTSIDE the box, so the solution sits on the boundary x0 = 0.5,
// x1 = 0.25 (KKT). The returned point is feasible.
const std::array<double, 2> lo{0.5, -1.0};
const std::array<double, 2> hi{2.0, 1.0};
const std::array<double, 2> box_start{1.5, 0.0};
opt::implicit_filtering(quad_f, box_start, lo, hi, imf).value().x;               // ~ (0.5, 0.25)

// Domain errors: invalid input (never a non-convergence).
std::span<const double> empty{};
opt::bfgs(quad_f, empty, quad_grad, opt::Options{}).error();          // MathError::domain_error
const std::array<double, 2> bad{std::numeric_limits<double>::quiet_NaN(), 1.0};
opt::gradient_descent(quad_f, bad, quad_grad, opt::Options{}).error();// MathError::domain_error
const std::array<double, 2> bad_lo{2.0, 0.0};   // bad_lo[0] > bad_hi[0]
const std::array<double, 2> bad_hi{1.0, 1.0};
const std::array<double, 2> s{1.0, 1.0};
opt::implicit_filtering(quad_f, s, bad_lo, bad_hi, opt::Options{}).error(); // MathError::domain_error
```

## See also

- [`nimblecas.numeric`](numeric.md) — the numerical **root-finder** sibling
  (Newton / bisection / secant for a single polynomial), the other
  floating-point solver that depends only on `core`.
- [`nimblecas.lp`](lp.md) — exact-rational **linear programming**: constrained
  optimization of a *linear* objective, kept exact where this module is
  numerical.
- [`nimblecas.roots`](roots.md) — exact rational roots of a polynomial, the
  algebraic counterpart to numerical zero-finding.
- [Documentation hub](../Index.md)
