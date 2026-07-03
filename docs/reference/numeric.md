# `nimblecas.numeric` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/numeric/numeric.cppm`

**Numeric root-finders** for polynomials (ROADMAP §7.14). Given a polynomial by
its **double-precision** coefficients in ascending order (low degree first), the
module locates a real root by Newton–Raphson, bisection, or the secant method,
with Horner evaluation of the polynomial and its derivative underneath. It is
deliberately self-contained: this is the **floating-point** solver, so it does
**not** depend on the exact `nimblecas.polynomial` / `ratpoly` layers.

```cpp
import nimblecas.numeric;   // namespace nimblecas::numeric
```

Depends on [`core`](core.md) only.

## Representation

A polynomial `p(x) = c[0] + c[1]·x + … + c[n-1]·x^(n-1)` is a
`std::span<const double>` of coefficients **low degree first**. An **empty** span
denotes the zero polynomial (value `0`); the empty and constant polynomials both
have derivative `0`. Evaluation uses **Horner's scheme** with `std::fma` for a
fused, low-rounding accumulation, and the derivative coefficients `k·c[k]` are
differentiated on the fly so no temporary buffer is allocated.

## API

```cpp
[[nodiscard]] auto eval(std::span<const double> c, double x) noexcept -> double;
[[nodiscard]] auto eval_derivative(std::span<const double> c, double x) noexcept -> double;

[[nodiscard]] auto newton(std::span<const double> coeffs, double x0, double tol,
                          int max_iter) -> Result<double>;
[[nodiscard]] auto bisection(std::span<const double> coeffs, double a, double b,
                             double tol) -> Result<double>;
[[nodiscard]] auto secant(std::span<const double> coeffs, double x0, double x1,
                          double tol, int max_iter) -> Result<double>;
```

### `eval` / `eval_derivative`

`eval` returns `p(x)` by Horner's method; `eval_derivative` returns `p'(x)` for
the same ascending-order span. Both are `noexcept` and total — the zero
polynomial evaluates to `0`, and a zero or constant polynomial has derivative
`0`.

### `newton`

Newton–Raphson: `x_{n+1} = x_n − p(x_n) / p'(x_n)`, starting from `x0`. Converges
(returns the current iterate) when `|p(x)| <= tol` or the step magnitude falls
below `tol`. Fails `not_implemented` on a **(near-)zero derivative**
(`|p'(x)| <= tol`, a flat spot that would divide by ~0) or after exhausting
`max_iter` iterations. With several real roots, Newton follows the basin of the
starting point (e.g. `x^3 − x` from `x0 = 1.5` lands on `+1`, from `−1.5` on
`−1`).

### `bisection`

Bisection on `[a, b]`: **requires a sign change**, `p(a)·p(b) <= 0`, then halves
the bracket until it is narrower than `tol` and returns the midpoint (or an exact
midpoint root if one is hit). A bracket with **no sign change** (`p(a)·p(b) > 0`)
is a `domain_error`.

### `secant`

Secant: a derivative-free Newton variant using the finite-difference slope of the
two most recent iterates, from seeds `x0` and `x1`. Same convergence and failure
contract as `newton`; the collapsed-slope case (`|p(x_curr) − p(x_prev)| <= tol`)
fails `not_implemented` rather than dividing by ~0.

## Error model

| Condition | Error |
| :--- | :--- |
| Empty `coeffs` span (any solver) | `MathError::domain_error` |
| `bisection` bracket with no sign change (`p(a)·p(b) > 0`) | `MathError::domain_error` |
| `newton` / `secant` fail to converge within `max_iter` | `MathError::not_implemented` |
| `newton` (near-)zero derivative, or `secant` collapsed slope | `MathError::not_implemented` |

`eval` and `eval_derivative` are `noexcept` and never fail.

## Worked examples

From `tests/numeric_tests.cpp` (coefficients ascending, low degree first;
convergence checked to `< 1e-6`):

```cpp
import nimblecas.numeric;
namespace num = nimblecas::numeric;

// sqrt(2), the positive root of x^2 - 2      coeffs {-2, 0, 1}
const std::array<double, 3> sq_minus_two{-2.0, 0.0, 1.0};

num::newton(sq_minus_two, 1.5, 1e-12, 100).value();      // ~ 1.41421356  (sqrt 2)
num::bisection(sq_minus_two, 1.0, 2.0, 1e-9).value();    // ~ 1.41421356
num::secant(sq_minus_two, 1.0, 2.0, 1e-12, 100).value(); // ~ 1.41421356
num::eval(sq_minus_two, std::numbers::sqrt2);            // ~ 0  (p(root) ~ 0)

// x^3 - x, roots {-1, 0, 1}: Newton follows the nearest basin   coeffs {0, -1, 0, 1}
const std::array<double, 4> cubic{0.0, -1.0, 0.0, 1.0};
num::newton(cubic, 1.5, 1e-12, 100).value();             // ~ +1
num::newton(cubic, -1.5, 1e-12, 100).value();            // ~ -1

// No sign change: x^2 + 1 over [0, 1]       coeffs {1, 0, 1}
const std::array<double, 3> sq_plus_one{1.0, 0.0, 1.0};
num::bisection(sq_plus_one, 0.0, 1.0, 1e-9).error();     // MathError::domain_error

// Empty coefficients: every solver rejects the zero polynomial.
std::span<const double> empty{};
num::newton(empty, 1.0, 1e-9, 100).error();              // MathError::domain_error
num::bisection(empty, 0.0, 1.0, 1e-9).error();           // MathError::domain_error
num::secant(empty, 0.0, 1.0, 1e-9, 100).error();         // MathError::domain_error
```

## See also

- [`nimblecas.roots`](roots.md) — the **exact** counterpart: every rational root
  of a polynomial over `Q[x]` with multiplicity (rational root theorem +
  deflation), where this module finds real roots numerically.
- [`nimblecas.lp`](lp.md) — the other new numeric-chain module: exact-rational
  linear programming.
- [`nimblecas.polynomial`](polynomial.md) — dense integer `Z[x]` with SIMD-batch
  evaluation.
- [Documentation hub](../Index.md)
