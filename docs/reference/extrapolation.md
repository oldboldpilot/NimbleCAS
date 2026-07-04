# `nimblecas.extrapolation` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/extrapolation/extrapolation.cppm`

**Sequence acceleration** for a CAS (ROADMAP §7). Richardson extrapolation,
Romberg integration, Aitken's Δ² process, and Wynn's ε (Shanks) algorithm are all
**rational linear combinations** of their inputs, and this module is deliberately
honest about what that buys you:

- On **exact rational inputs** — a `Rational` sequence, or a rational integrand /
  function sampled at rational abscissae — every routine is **exact over Q**. The
  outputs are pure reduced fractions built by add / subtract / multiply / divide
  of rationals with **no rounding**. These are the `*_exact` paths and the
  `Rational` overloads.
- On **`double`-valued** functions the same algebra runs in IEEE-754 and is
  therefore **numerical**. Extrapolation accelerates convergence **only when the
  assumed asymptotic error expansion actually holds** — i.e. the data is smooth.
  For rough or noisy data it can **amplify** error rather than remove it. **No
  universal-acceleration claim is made.**
- A **stalled sequence** makes the Aitken/epsilon denominator (a `Δ²`, or an ε
  difference) **zero**. That is reported as `MathError::domain_error` on the
  railway — never a wrong result, never a silent NaN.

Every failure travels the railway (`Result<T>` / `MathError`); nothing throws.
All exact paths are overflow-checked through `Rational`'s own guarded arithmetic
(Rule 32).

```cpp
import nimblecas.extrapolation;   // namespace nimblecas
```

Depends on [`core`](core.md) and [`ratpoly`](ratpoly.md).

## Callback and tableau types

All symbols live directly in namespace `nimblecas`.

| Type | Definition | Notes |
| :--- | :--- | :--- |
| `RealFunction` | `using RealFunction = std::function<double(double)>` | Numerical scalar function `R -> R` (IEEE-754). |
| `ExactFunction` | `using ExactFunction = std::function<Result<Rational>(const Rational&)>` | Exact scalar function `Q -> Q`, **on the railway** — an integrand undefined at a point, or one that overflows `int64`, can report it rather than fabricate a value. |
| `RationalTableau` | `struct { std::vector<std::vector<Rational>> table; Rational best; }` | Lower-triangular extrapolation tableau, **exact over Q**. `table[i]` has length `i+1`; column `j` is the `j`-th extrapolation level; `best` is the fully-extrapolated corner `table[n-1][n-1]`. |
| `DoubleTableau` | `struct { std::vector<std::vector<double>> table; double best{}; }` | The **numerical** (IEEE-754) counterpart, same shape. |

## Richardson extrapolation

Given `A(h) = A + c_1 h^p + c_2 h^{2p} + ...`, a single step combines `A(h)` and
the refined `A(h/r)` to cancel the leading `h^p` term:
`A_new = (r^p A(h/r) − A(h)) / (r^p − 1)`.

```cpp
[[nodiscard]] auto richardson_step(const Rational& a_h, const Rational& a_hr,
                                   std::int64_t r, std::int64_t p) -> Result<Rational>;
[[nodiscard]] auto richardson_step(double a_h, double a_hr, double r, double p) -> Result<double>;

[[nodiscard]] auto richardson_tableau(std::span<const Rational> a, std::int64_t r,
                                      std::int64_t p) -> Result<RationalTableau>;
[[nodiscard]] auto richardson_tableau(std::span<const double> a, double r, double p)
    -> Result<DoubleTableau>;
```

| Function | Behavior |
| :--- | :--- |
| `richardson_step` (`Rational`) | One cancellation step, **exact over Q** (`r`, `p` integers ⇒ `r^p` integer ⇒ a rational linear combination). Requires refinement ratio `r >= 2` and leading order `p >= 1`, else `domain_error`. |
| `richardson_step` (`double`) | Numerical form; `r` may be any real `> 1` and `p` any real `> 0` (uses `std::pow`). |
| `richardson_tableau` (`Rational`) | Full iterated Richardson/Neville tableau over `a = [A(h_0), A(h_0/r), A(h_0/r^2), ...]` (successive refinement by `r`, error orders `p, 2p, 3p, ...`). The per-column recurrence is `T[i][j] = T[i][j-1] + (T[i][j-1] − T[i-1][j-1]) / (r^{p·j} − 1)`. **Exact over Q.** `domain_error` on an empty sequence, `r < 2`, or `p < 1`; propagates `Rational` overflow. |
| `richardson_tableau` (`double`) | Numerical form; requires `r > 1` and `p > 0`. |

## Richardson-accelerated derivative

Central differences `D(h) = (f(x+h) − f(x−h)) / (2h)` have an error expansion in
**even** powers `h^2, h^4, ...`, so the tableau is built with refinement `r = 2`
and leading order `p = 2` on `D(h_0), D(h_0/2), ..., D(h_0/2^levels)`. `best` is
the extrapolated estimate.

```cpp
[[nodiscard]] auto richardson_derivative(const RealFunction& f, double x, double h0,
                                         std::size_t levels) -> Result<DoubleTableau>;
[[nodiscard]] auto richardson_derivative_exact(const ExactFunction& f, const Rational& x,
                                               const Rational& h0, std::size_t levels)
    -> Result<RationalTableau>;
```

| Function | Behavior |
| :--- | :--- |
| `richardson_derivative` | **Numerical.** Accelerates for smooth `f`, but a *very* small `h` eventually loses accuracy to cancellation — extrapolating from a moderate `h_0` is the point of the method. `domain_error` if `h0 == 0`. |
| `richardson_derivative_exact` | **Exact over Q** for a rational function sampled at rational points: for a polynomial (or any rational `f` whose central-difference error expansion terminates) enough levels recover the derivative value **exactly**. `domain_error` if `h0 == 0`; `overflow` if `levels` exceeds `int64` range; propagates any error from `f`. |

## Romberg integration

Richardson applied to the composite trapezoidal rule: build
`R[i][0] = T(h_i)`, the composite trapezoid on `2^i` sub-intervals of `[a, b]`
(successive halving, reusing the previous level's points), then Richardson
columns with `r = 2`, `p = 2` (the Euler–Maclaurin error is even in `h`).
`best = R[levels][levels]`.

```cpp
[[nodiscard]] auto romberg(const RealFunction& f, double a, double b, std::size_t levels)
    -> Result<DoubleTableau>;
[[nodiscard]] auto romberg_exact(const ExactFunction& f, const Rational& a, const Rational& b,
                                 std::size_t levels) -> Result<RationalTableau>;
```

| Function | Behavior |
| :--- | :--- |
| `romberg` | **Numerical** double form. `domain_error` on `b < a`. |
| `romberg_exact` | **Exact over Q** when the integrand is rational-valued at the (rational) trapezoid abscissae — e.g. `∫_0^1 x^2 dx` resolves to `1/3` after the first Richardson column (that column is Simpson's rule, exact for cubics). `domain_error` on `b < a`; propagates `Rational` overflow / any error from `f`. |

## Aitken's Δ² process

Accelerate a linearly-convergent sequence:
`x_n' = x_n − (Δx_n)^2 / (Δ² x_n)`, with `Δx_n = x_{n+1} − x_n` and
`Δ² x_n = x_{n+2} − 2x_{n+1} + x_n`. Returns the accelerated sequence
(length `n − 2`).

```cpp
[[nodiscard]] auto aitken(std::span<const Rational> x) -> Result<std::vector<Rational>>;
[[nodiscard]] auto aitken(std::span<const double> x) -> Result<std::vector<double>>;
```

| Function | Behavior |
| :--- | :--- |
| `aitken` (`Rational`) | **Exact over Q.** `domain_error` if the input has fewer than 3 terms, or if any `Δ² x_n == 0` (a stalled window — the denominator would vanish; reported, never divided). |
| `aitken` (`double`) | Numerical form. Guards an **exact-zero** `Δ²` (`domain_error`); a merely *tiny* `Δ²` is the noise-amplification regime the honesty note warns about, **not** an error. |

## Shanks transformation / Wynn's ε algorithm

The epsilon table on a sequence of partial sums `s = [S_0, ..., S_{N-1}]`:
`ε_{-1}^{(n)} = 0`, `ε_0^{(n)} = S_n`,
`ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1 / (ε_k^{(n+1)} − ε_k^{(n)})`. The even-order
columns hold the accelerated estimates (`ε_2` is the Shanks transform,
generalising Aitken). Returns the **deepest even-column entry** — the most
accelerated value.

```cpp
[[nodiscard]] auto wynn_epsilon(std::span<const Rational> s) -> Result<Rational>;
[[nodiscard]] auto wynn_epsilon(std::span<const double> s) -> Result<double>;
```

| Function | Behavior |
| :--- | :--- |
| `wynn_epsilon` (`Rational`) | **Exact over Q.** `domain_error` if `N < 3`, or if any ε difference in the recurrence is `0` (a stalled sequence — division by zero, reported not performed). |
| `wynn_epsilon` (`double`) | Numerical form; guards an **exact-zero** ε difference (`domain_error`). |

## Error model

| Condition | Error |
| :--- | :--- |
| `richardson_step` / `richardson_tableau` with `r < 2` or `p < 1` (`Rational`), or `r <= 1` / `p <= 0` (`double`) | `MathError::domain_error` |
| Empty sequence passed to `richardson_tableau` | `MathError::domain_error` |
| `richardson_derivative(_exact)` with `h0 == 0` | `MathError::domain_error` |
| `romberg(_exact)` with `b < a` | `MathError::domain_error` |
| `aitken` with fewer than 3 terms, or a zero `Δ²` window | `MathError::domain_error` |
| `wynn_epsilon` with `N < 3`, or a zero ε difference | `MathError::domain_error` |
| `richardson_derivative_exact` with `levels` beyond `int64` range | `MathError::overflow` |
| Any `int64` numerator/denominator computation wraps on an exact path | `MathError::overflow` |
| An `ExactFunction` reports an error (undefined point / overflow) | propagated verbatim |

The exact paths never round; the numerical paths never *silently* divide by zero —
an exact-zero denominator is a `domain_error`, while a small-but-nonzero
denominator on smooth data is the intended acceleration (and on noisy data the
documented amplification regime).

## Worked examples

From `tests/extrapolation_tests.cpp` (exact paths compared with `==`; numerical
paths checked to `< 1e-6`):

```cpp
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.extrapolation;
namespace nx = nimblecas;
using nimblecas::Rational;
using nimblecas::Result;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// --- Richardson: one exact step recovers A from A(h) = A + c·h^2 ---
// A = 5, c = 3, p = 2: A(1) = 8, A(1/2) = 5 + 3/4 = 23/4.
nx::richardson_step(ri(8), rat(23, 4), 2, 2).value();     // == 5  (exact over Q)

const std::array<Rational, 2> a{ri(8), rat(23, 4)};       // A(1), A(1/2)
nx::richardson_tableau(std::span<const Rational>{a}, 2, 2).value().best;  // == 5

// Numerical tableau: A(h) = 5 + 3 h^2 at h = 1, 1/2, 1/4 -> best ~ 5.
const std::array<double, 3> ad{8.0, 5.75, 5.1875};
nx::richardson_tableau(std::span<const double>{ad}, 2.0, 2.0).value().best;  // ~ 5

// Domain guards.
nx::richardson_step(ri(8), rat(23, 4), 1, 2).error();     // domain_error (r < 2)
nx::richardson_step(ri(8), rat(23, 4), 2, 0).error();     // domain_error (p < 1)

// --- Romberg: EXACT over Q on a rational integrand ∫_0^1 x^2 dx = 1/3 ---
auto sq = [](const Rational& x) -> Result<Rational> { return x.multiply(x); };
nx::romberg_exact(sq, ri(0), ri(1), 3).value().best;      // == 1/3  (Simpson col is exact)

// Numerical Romberg on a smooth transcendental integrand ∫_0^1 e^x dx = e - 1.
auto ex = [](double x) { return std::exp(x); };
nx::romberg(ex, 0.0, 1.0, 6).value().best;                // ~ e - 1
nx::romberg(ex, 1.0, 0.0, 3).error();                     // domain_error (b < a)

// --- Aitken Δ²: accelerate geometric partial sums to the exact limit ---
// Partial sums of Σ (1/2)^k: 1, 3/2, 7/4 -> limit 2.
const std::array<Rational, 3> xg{ri(1), rat(3, 2), rat(7, 4)};
nx::aitken(std::span<const Rational>{xg}).value()[0];     // == 2  (one accelerated value)

const std::array<Rational, 3> xs{ri(2), ri(2), ri(2)};    // Δ² = 0
nx::aitken(std::span<const Rational>{xs}).error();        // domain_error (stalled)

// --- Wynn ε (Shanks): sum a rational partial-sum sequence exactly ---
const std::array<Rational, 3> sg{ri(1), rat(3, 2), rat(7, 4)};
nx::wynn_epsilon(std::span<const Rational>{sg}).value();  // == 2   (ε_2 exact over Q)

const std::array<Rational, 3> sc{ri(1), ri(1), ri(1)};    // constant -> ε diff 0
nx::wynn_epsilon(std::span<const Rational>{sc}).error();  // domain_error (stalled)

// --- Richardson-accelerated derivative ---
// p(x) = x^3, p'(1) = 3: central diff has a pure h^2 error, one level recovers 3.
auto cube = [](const Rational& x) -> Result<Rational> {
    auto x2 = x.multiply(x);
    if (!x2) { return nimblecas::make_error<Rational>(x2.error()); }
    return x2->multiply(x);
};
nx::richardson_derivative_exact(cube, ri(1), ri(1), 1).value().best;  // == 3 (exact over Q)

// d/dx sin at 0 is cos(0) = 1 (numerical).
auto sinf = [](double x) { return std::sin(x); };
nx::richardson_derivative(sinf, 0.0, 0.5, 5).value().best;            // ~ 1
```

## See also

- [`nimblecas.numeric`](numeric.md) — the other standalone floating-point
  numeric-analysis module (polynomial root-finders): the numerical counterpart to
  these acceleration routines.
- [`nimblecas.series`](series.md) and [`nimblecas.powerseries`](powerseries.md) —
  the truncated-series producers whose partial sums Aitken / Wynn-ε accelerate.
- [`nimblecas.pade`](pade.md) — the Padé rational approximant, a sibling
  series-acceleration technique (exact-rational Toeplitz solve).
- [`nimblecas.integrate`](integrate.md) — exact symbolic rational-function
  integration, where Romberg is the numerical quadrature counterpart.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every `*_exact`
  path computes in.
- [Documentation hub](../Index.md)
