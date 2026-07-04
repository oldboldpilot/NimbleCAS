# `nimblecas.nlsolve` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/nlsolve/nlsolve.cppm`

**C.T. Kelley iterative solvers** for a nonlinear system `F(x) = 0` in `Rⁿ`
(ROADMAP §7.14) — the numeric counterpart to minimization. This is a
**NUMERIC** module: every quantity is an IEEE-754 `double` and every method is
only **LOCALLY convergent**. Newton's method is locally *quadratically*
convergent near a **simple root** (nonsingular Jacobian) but is **not** globally
convergent; the approximate-Newton variants (chord, Shamanskii, Broyden,
Newton–Krylov) trade Jacobian accuracy for cost and converge at best
superlinearly (linearly for the chord method). Finite-difference Jacobians and
Jacobian–vector products add **truncation error** on top of rounding — an
analytic Jacobian is always more accurate when available. **No exactness and no
global-convergence claims are made anywhere**, and every result depends on the
initial guess. Running out of iterations, a line-search stall, or a mid-iteration
singular Jacobian is **not** an error: it returns a valid `SolveResult` carrying
`converged == false` and the best iterate found.

```cpp
import nimblecas.nlsolve;   // namespace nimblecas::nlsolve
```

Depends on [`core`](core.md) only. The module is self-contained: it implements
its own dense Gaussian-elimination linear solver and a compact restarted
GMRES(m), so no external linear-algebra dependency is pulled in.

## The exact-vs-numerical boundary

Nothing here is exact. Concretely:

- **Newton** is locally quadratically convergent near a simple root and diverges
  from a poor start unless globalised. The **Armijo line search**
  (`Options::line_search`, **on by default**) is what turns most such runs into
  progress — and even it can stall.
- **chord / Shamanskii / Broyden / Newton–Krylov (JFNK)** are approximate-Newton
  variants: only locally convergent, typically superlinearly (linearly for
  chord).
- **Anderson acceleration** accelerates a fixed-point map `g(x) = x`. It is a
  heuristic with **no global guarantee** and can diverge if the map is not
  (locally) contractive.
- **Levenberg–Marquardt** finds a **stationary point** of `||F(x)||²`, which need
  not be a global minimiser and **need not have `F = 0`**.

## API

Everything below lives in `namespace nimblecas::nlsolve`.

### Callable and result types

```cpp
using ResidualFn = std::function<std::vector<double>(std::span<const double>)>;
using JacobianFn = std::function<std::vector<double>(std::span<const double>)>;

struct SolveResult {
    std::vector<double> x{};            // best iterate located
    double residual_norm{0.0};          // ||F(x)||_2 there
    std::size_t iterations{0};          // outer iterations performed
    bool converged{false};              // did the stopping test pass?
    std::size_t function_evals{0};      // number of F/g evaluations (diagnostics)
};

struct Options {
    double tol{1e-10};                  // absolute stopping tolerance on ||F||_2
    std::size_t max_iter{100};          // maximum outer iterations
    double fd_step{1e-7};               // base finite-difference step (relative-scaled)
    bool central_diff{false};           // central (2 evals/col) vs forward FD Jacobian
    bool line_search{true};             // Armijo backtracking on ||F|| (globalisation)
    double armijo_alpha{1e-4};          // sufficient-decrease parameter
    double armijo_reduction{0.5};       // backtracking step-length factor in (0,1)
    std::size_t max_backtrack{30};      // maximum backtracking halvings per step
};

enum class BroydenVariant : std::uint8_t { good, bad };
```

| Type | Notes |
| :--- | :--- |
| `ResidualFn` | The residual map `F : Rⁿ → Rᵐ`, evaluated at the point given as a span. For the square solvers (`newton`, `chord`, `shamanskii`, `broyden`, `newton_krylov`) `m == n`; for `anderson` it is the fixed-point map `g(x)` with `m == n`; for `levenberg_marquardt` `m` may exceed `n`. The callable **must not retain the span**. |
| `JacobianFn` | An analytic Jacobian `J : Rⁿ → R^{m×n}`, returned **DENSE and ROW-MAJOR**: entry `(i, j) = ∂Fᵢ/∂xⱼ` lives at index `i*n + j`, so the returned vector has size `m*n`. |
| `SolveResult` | Outcome of a solve. `x` is the best iterate found (always populated on a non-error return, even when `converged == false`); `residual_norm` is `||F(x)||₂` there. |
| `Options` | Shared tuning knobs. The Armijo line search is **on by default** because plain Newton is not globally convergent. |
| `BroydenVariant` | Which Broyden rank-1 update to apply to the approximate inverse Jacobian. |

### Finite-difference Jacobian

```cpp
[[nodiscard]] auto finite_difference_jacobian(const ResidualFn& F, std::span<const double> x,
                                              double step = 1e-7, bool central = false)
    -> Result<std::vector<double>>;
```

Dense forward (default) or central finite-difference Jacobian of `F` at `x`,
**ROW-MAJOR** with `m = |F(x)|` rows and `n = |x|` columns. Column `j` perturbs
`xⱼ` by a relative step `h = step*(1 + |xⱼ|)` (which avoids a zero increment at
`xⱼ == 0`). Carries `O(step)` (forward) or `O(step²)` (central) truncation error.
Non-finite `x`, or a non-finite / wrong-dimension `F` evaluation, yields
`domain_error`. Exposed for verification and for callers who want the estimate
directly.

### Newton (full Jacobian every step)

```cpp
[[nodiscard]] auto newton(const ResidualFn& F, std::span<const double> x0,
                          const Options& opts = {}) -> Result<SolveResult>;

[[nodiscard]] auto newton(const ResidualFn& F, const JacobianFn& J,
                          std::span<const double> x0, const Options& opts = {})
    -> Result<SolveResult>;
```

Newton's method: the linear step `J s = −F` is solved by dense Gaussian
elimination with partial pivoting. Locally quadratically convergent near a simple
root; globalised by the Armijo line search. The first overload estimates `J` by
finite differences (those inner evaluations are *not* threaded into
`SolveResult::function_evals`); the second uses your analytic `J`, which is more
accurate.

### chord (Jacobian frozen at `x0`)

```cpp
[[nodiscard]] auto chord(const ResidualFn& F, std::span<const double> x0,
                         const Options& opts = {}) -> Result<SolveResult>;

[[nodiscard]] auto chord(const ResidualFn& F, const JacobianFn& J,
                         std::span<const double> x0, const Options& opts = {})
    -> Result<SolveResult>;
```

The chord method forms the Jacobian **once** at `x0` and reuses it for every
step. Cheap per iteration but only **linearly** convergent.

### Shamanskii (Jacobian refreshed every `m` steps)

```cpp
[[nodiscard]] auto shamanskii(const ResidualFn& F, std::span<const double> x0,
                              std::size_t m, const Options& opts = {})
    -> Result<SolveResult>;

[[nodiscard]] auto shamanskii(const ResidualFn& F, const JacobianFn& J,
                              std::span<const double> x0, std::size_t m,
                              const Options& opts = {}) -> Result<SolveResult>;
```

Refreshes the Jacobian every `m` steps: `m == 1` recovers Newton, `m → ∞`
recovers the chord method, and small `m > 1` is superlinear. `m == 0` is treated
as `m == 1` (Newton).

### Broyden (quasi-Newton, rank-1 inverse update)

```cpp
[[nodiscard]] auto broyden(const ResidualFn& F, std::span<const double> x0,
                           const Options& opts = {},
                           BroydenVariant variant = BroydenVariant::good)
    -> Result<SolveResult>;
```

Maintains an approximate **inverse** Jacobian `H` (seeded from the FD Jacobian at
`x0`, or the identity when that seed is singular) and updates it by a rank-1
correction each step. `BroydenVariant::good` uses the Sherman–Morrison inverse
update; `BroydenVariant::bad` uses the (often less robust) alternative. Locally
superlinearly convergent.

### Newton–Krylov (JFNK, matrix-free)

```cpp
[[nodiscard]] auto newton_krylov(const ResidualFn& F, std::span<const double> x0,
                                 const Options& opts = {}, std::size_t krylov_dim = 30)
    -> Result<SolveResult>;
```

Jacobian-**free** Newton–Krylov: each Newton step is solved inexactly by an inner
GMRES(m) that never forms the Jacobian, using finite-difference Jacobian–vector
products `Jv ≈ (F(x + εv) − F(x))/ε`. The inner tolerance `η_k` is chosen
adaptively by the **Eisenstat–Walker "Choice 2"** forcing term (with the standard
safeguards) to avoid oversolving the linear system far from the root. Globalised
by the Armijo line search. `krylov_dim` is the GMRES restart length `m`; `0`
means `min(n, 30)`.

### Anderson acceleration (fixed-point `g(x) = x`)

```cpp
[[nodiscard]] auto anderson(const ResidualFn& g, std::span<const double> x0,
                            std::size_t window = 5, const Options& opts = {})
    -> Result<SolveResult>;
```

Anderson acceleration (Kelley / Walker–Ni) for a fixed-point iteration
`x ← g(x)`, mixing over a `window` of the most recent residuals `f = g(x) − x`
via a small Tikhonov-regularised least-squares problem. `window == 0` degenerates
to plain Picard iteration; an ill-conditioned mixing step falls back to a safe
Picard step. Accelerates linearly-convergent fixed points but is a heuristic with
no global guarantee.

### Levenberg–Marquardt (nonlinear least squares)

```cpp
[[nodiscard]] auto levenberg_marquardt(const ResidualFn& F, std::span<const double> x0,
                                       const Options& opts = {}, double lambda0 = 1e-3)
    -> Result<SolveResult>;

[[nodiscard]] auto levenberg_marquardt(const ResidualFn& F, const JacobianFn& J,
                                       std::span<const double> x0, const Options& opts = {},
                                       double lambda0 = 1e-3) -> Result<SolveResult>;
```

Levenberg–Marquardt for the least-squares problem `min ||F(x)||²`, where `F` may
be **over-determined** (`m = |F(x)| ≥ n`). The damping `λ` interpolates between
the Gauss–Newton step (small `λ`) and scaled gradient descent (large `λ`); it is
grown on rejected steps and shrunk on accepted ones. It converges when either
`||F||₂ ≤ tol` **or** the max component of `JᵀF` (half the gradient of
`||F||²`) is `≤ tol`. Finds
a **stationary point** of the cost, which need not be a global minimiser or a
root. The first overload uses a finite-difference Jacobian; the second uses your
analytic `J`. An under-determined system (`m < n`) is rejected up front with
`domain_error`.

## Error model

The failure policy distinguishes a **bad problem** (an error) from **failure to
converge** (a valid `converged == false` result).

| Condition | Outcome |
| :--- | :--- |
| Empty system (`x0` empty) | `MathError::domain_error` |
| Non-finite `x0` | `MathError::domain_error` |
| `F` / `g` / Jacobian returns a wrong-dimension or non-finite value **at the start point** | `MathError::domain_error` |
| Analytic Jacobian of wrong size (`≠ m*n`) or non-finite at the start | `MathError::domain_error` |
| `levenberg_marquardt` given an under-determined system (`m < n`) | `MathError::domain_error` |
| `finite_difference_jacobian` at a non-finite `x`, or non-finite / mismatched `F` | `MathError::domain_error` |
| Ran out of `max_iter` iterations | valid `SolveResult`, `converged == false` |
| Line-search stagnation (no sufficient decrease within `max_backtrack`) | valid `SolveResult`, `converged == false` |
| Singular Jacobian / breakdown encountered **mid-iteration** | valid `SolveResult`, `converged == false` |

A wrong-dimension or non-finite Jacobian *after* the first iteration is a
mid-iteration breakdown (a `converged == false` result), not an error — only the
**start point** promotes such a failure to `domain_error`.

## Worked examples

Mirroring `tests/nlsolve_tests.cpp` (numeric, local convergence; value
comparisons checked to `< 1e-6`):

```cpp
import nimblecas.nlsolve;
namespace nl = nimblecas::nlsolve;

// F(x) = [x^2 - 2], root sqrt(2); analytic 1x1 Jacobian [2x].
auto sq_minus_two     = [](std::span<const double> x) -> std::vector<double> {
    return {x[0] * x[0] - 2.0};
};
auto sq_minus_two_jac = [](std::span<const double> x) -> std::vector<double> {
    return {2.0 * x[0]};
};
const std::array<double, 1> x0{1.5};

// Newton with an analytic Jacobian, and the finite-difference variant.
nl::newton(sq_minus_two, sq_minus_two_jac, x0).value().x[0];  // ~ 1.41421356 (sqrt 2)
nl::newton(sq_minus_two, x0).value().x[0];                    // ~ 1.41421356 (FD Jacobian)

// 2x2 system F = [x0^2 + x1^2 - 2, x0 - x1]; root (1, 1).
auto system2     = [](std::span<const double> x) -> std::vector<double> {
    return {x[0] * x[0] + x[1] * x[1] - 2.0, x[0] - x[1]};
};
auto system2_jac = [](std::span<const double> x) -> std::vector<double> {
    return {2.0 * x[0], 2.0 * x[1], 1.0, -1.0};  // row-major 2x2
};
const std::array<double, 2> s0{1.5, 1.2};

nl::newton(system2, system2_jac, s0).value().x;                       // (1, 1)
nl::broyden(system2, s0).value().x;                                   // (1, 1) good Broyden
nl::broyden(system2, s0, {}, nl::BroydenVariant::bad).value().x;      // (1, 1) bad Broyden
nl::chord(system2, system2_jac, s0).value().x;                        // (1, 1) linear
nl::shamanskii(system2, system2_jac, s0, 2).value().x;                // (1, 1) refresh every 2
nl::newton_krylov(system2, s0).value().x;                             // (1, 1) matrix-free JFNK

// Armijo globalisation: atan(x) from x0 = 10 diverges undamped, converges damped.
auto atan_res = [](std::span<const double> x) -> std::vector<double> { return {std::atan(x[0])}; };
auto atan_jac = [](std::span<const double> x) -> std::vector<double> {
    return {1.0 / (1.0 + x[0] * x[0])};
};
const std::array<double, 1> far{10.0};

nl::Options plain;  plain.line_search = false;  plain.max_iter = 50;
nl::newton(atan_res, atan_jac, far, plain).value().converged;  // false (undamped diverges)

nl::Options damped; damped.max_iter = 100;                     // line_search on by default
nl::newton(atan_res, atan_jac, far, damped).value().x[0];      // ~ 0 (Armijo saves it)

// Anderson acceleration of the linear fixed point g(x) = cos(x) -> Dottie number.
auto cos_map = [](std::span<const double> x) -> std::vector<double> { return {std::cos(x[0])}; };
const std::array<double, 1> z0{0.0};
auto a = nl::anderson(cos_map, z0, 3).value();
a.x[0];          // ~ 0.7390851 (cos(x) = x); a.iterations well under Picard's ~58

// Levenberg-Marquardt over-determined fit: y = a exp(b t), recover (a, b) = (2, 0.5).
const std::array<double, 5> ts{0.0, 0.5, 1.0, 1.5, 2.0};
std::array<double, 5> ys{};
for (std::size_t i = 0; i < ts.size(); ++i) ys[i] = 2.0 * std::exp(0.5 * ts[i]);
auto resid = [ts, ys](std::span<const double> p) -> std::vector<double> {
    std::vector<double> r(ts.size());
    for (std::size_t i = 0; i < ts.size(); ++i) r[i] = p[0] * std::exp(p[1] * ts[i]) - ys[i];
    return r;
};
const std::array<double, 2> p0{1.0, 1.0};
nl::Options lm; lm.tol = 1e-12; lm.max_iter = 200;
nl::levenberg_marquardt(resid, p0, lm).value().x;  // ~ (2, 0.5)

// Finite-difference Jacobian agrees with the analytic one to ~1e-4.
const std::array<double, 2> xj{1.3, 0.7};
nl::finite_difference_jacobian(system2, xj).value();  // ~ {2.6, 1.4, 1, -1}

// Capped iterations are a valid non-converged result, not an error.
nl::Options cap; cap.tol = 1e-15; cap.max_iter = 1;
auto capped = nl::newton(sq_minus_two, sq_minus_two_jac, x0, cap).value();
capped.converged;   // false
capped.iterations;  // 1  (used exactly the budget)

// The domain-error boundary.
const std::array<double, 1> inf{std::numeric_limits<double>::infinity()};
nl::newton(sq_minus_two, inf).error();               // MathError::domain_error (non-finite x0)
std::span<const double> empty{};
nl::newton(sq_minus_two, empty).error();             // MathError::domain_error (empty system)
auto wrong_dim = [](std::span<const double> x) -> std::vector<double> { return {x[0], x[0]}; };
nl::newton(wrong_dim, x0).error();                   // MathError::domain_error (dim mismatch)
nl::finite_difference_jacobian(sq_minus_two, inf).error();  // MathError::domain_error
```

## See also

- [`nimblecas.numeric`](numeric.md) — the **scalar / single-polynomial**
  floating-point root-finders (Newton–Raphson, bisection, secant), where this
  module solves general nonlinear **systems** `F(x) = 0` in `Rⁿ`.
- [`nimblecas.roots`](roots.md) — the **exact** rational-root counterpart for a
  polynomial over `Q[x]` (rational root theorem + deflation), with no
  floating-point.
- [`nimblecas.matrix`](matrix.md) / [`nimblecas.matdecomp`](matdecomp.md) — dense
  linear algebra, the exact analogue of the internal dense solver used here.
- [Documentation hub](../Index.md)
