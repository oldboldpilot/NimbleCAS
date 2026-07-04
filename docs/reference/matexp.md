# `nimblecas.matexp` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/matexp/matexp.cppm`

Exact-over-the-rationals routines for the matrix exponential `e^A` (ROADMAP
§7.2 / §7.19) — the object that closes the linear-ODE story, since the solution
of `dx/dt = A x` is `x(t) = e^{At} x0`. Everything here stays inside exact
[`Rational`](ratpoly.md) arithmetic on top of [`Matrix`](matrix.md) and rides the
`Result` railway: **no floating point ever enters**. This module sits in the
numeric tower above `matrix`, alongside the other `ratpoly`-consuming linear
routines, and feeds the dynamics / ODE layer.

**Honesty about exactness.** The true matrix exponential `e^A` is generally
**transcendental** and therefore *not representable over Q*. These routines
never claim the transcendental truth; each returns a specific exact rational
object:

- `matrix_exp_taylor(A, terms)` returns the truncated Taylor series
  `Σ_{k=0}^{terms-1} A^k / k!`. This equals `e^A` **exactly iff `A` is nilpotent
  and `terms` is at least the nilpotency index `m`** (the least `m` with
  `A^m = 0`) — once `terms ≥ m` every omitted term is already zero and the
  series terminates. Otherwise it is only a rational polynomial approximation.
- `matrix_exp_pade(A, q)` returns the diagonal `[q/q]` Padé approximant
  `D^{-1} N`, a matrix-valued rational approximation. The `[q/q]` approximant
  matches the scalar series `e^x` only **through order `x^{2q}`**, so on a
  nilpotent `A` it is **exact only when the nilpotency index `m ≤ 2q+1`**. For a
  nilpotent `A` with `m > 2q+1` — e.g. a 4×4 Jordan block (`m = 4`) with `q = 1`,
  where it returns `I + J + J²/2 + J³/4` versus the true `I + J + J²/2 + J³/6` —
  it is still only an approximation, distinct from Taylor's. Raise `q` so that
  `2q+1 ≥ m` to recover exactness.
- `matrix_exp(A, q, s)` is scaling-and-squaring: `e^A = (e^{A/2^s})^{2^s}`, the
  numerically standard route, with the inner factor formed by the `[q/q]` Padé
  approximant of `A/2^s`. Scaling preserves the nilpotency index, so it inherits
  Padé's condition **exactly**: exact for nilpotent `A` iff `m ≤ 2q+1`, otherwise
  an approximation.

Every fallible step is overflow-checked: any `int64` numerator or denominator
that would wrap surfaces as `MathError::overflow` rather than silently
wrapping (including the `2^s` scale factor). Dimension violations and bad
parameters surface as `MathError::domain_error`.

```cpp
import nimblecas.matexp;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and [`matrix`](matrix.md).

## Free functions

All four routines are free functions in `namespace nimblecas`, `[[nodiscard]]`,
and return `Result<T>`. Every one requires `A` to be square; a non-square input
yields `MathError::domain_error`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `is_nilpotent` | `auto is_nilpotent(const Matrix& a) -> Result<bool>` | `true` iff `A^n` is the zero matrix, where `n = a.rows()`. For an `n×n` matrix this `n`-th power is a complete test — a nilpotent matrix has index at most `n`, so if any power vanishes then `A^n` already does. `A^n` is formed by repeated exact multiplication, so entry overflow is propagated. Requires `A` square, else `domain_error`. |
| `matrix_exp_taylor` | `auto matrix_exp_taylor(const Matrix& a, std::int64_t terms) -> Result<Matrix>` | The truncated Taylor series `Σ_{k=0}^{terms-1} A^k / k!`, exact over `Rational`. A running power `P_k = A^k` and a running reciprocal `1/k!` are accumulated so no large factorial is ever formed. **Exact `e^A` iff `A` is nilpotent and `terms ≥ m`**; otherwise a rational polynomial approximation. Requires `A` square and `terms ≥ 1`, else `domain_error`. Overflow propagated. |
| `matrix_exp_pade` | `auto matrix_exp_pade(const Matrix& a, std::size_t q) -> Result<Matrix>` | The diagonal `[q/q]` Padé approximant: with scalar coefficients `c_k` (`c_0 = 1`), `N = Σ_{k=0}^{q} c_k A^k` and `D = Σ_{k=0}^{q} (-1)^k c_k A^k`, returning `D^{-1} N`. The `c_k` follow the ratio recurrence `c_k = c_{k-1}·(q-k+1)/(k·(2q-k+1))` to avoid forming huge factorials. **Exact for nilpotent `A` only when `m ≤ 2q+1`**; else (or for non-nilpotent `A`) a rational approximation. Requires `A` square and `q ≥ 1`, else `domain_error`; a singular `D` propagates `domain_error` from `inverse()`. Overflow propagated. |
| `matrix_exp` | `auto matrix_exp(const Matrix& a, std::size_t q, std::size_t scaling_power) -> Result<Matrix>` | Scaling-and-squaring (the recommended route): form `B = A / 2^s` exactly, take `X = matrix_exp_pade(B, q)`, then square `X` back `s` times (`X ← X·X`) to recover `e^A`. `s = scaling_power` may be `0` (then it coincides with `matrix_exp_pade(A, q)`). Scaling preserves the nilpotency index, so it inherits Padé's condition: **exact for nilpotent `A` iff `m ≤ 2q+1`**, else an approximation. Requires `A` square and `q ≥ 1`, else `domain_error`. Overflow — including in the `2^s` scale factor — propagated. |

## Error model

| Condition | Error |
| :--- | :--- |
| Any routine given a non-square `A` | `MathError::domain_error` |
| `matrix_exp_taylor` with `terms < 1` | `MathError::domain_error` |
| `matrix_exp_pade` / `matrix_exp` with `q < 1` | `MathError::domain_error` |
| `matrix_exp_pade` denominator `D` singular (guarded; never happens for the standard exp coefficients) | `MathError::domain_error` (from `inverse()`) |
| An `int64` numerator or denominator computation wraps — including a matrix-entry product, a reciprocal factorial, or the `2^s` scale factor | `MathError::overflow` |

The exactness contracts above are **not** error conditions: an inexact result on
a non-nilpotent (or under-`q`) input is a well-defined *rational approximation*,
returned as a success value, not a failure.

## Worked examples

```cpp
import nimblecas.matexp;
import nimblecas.matrix;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri = [](std::int64_t v) { return Rational::from_int(v); };
auto rr = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// e^0 = I exactly, by all three routes.
matrix_exp_taylor(Matrix::zero(2, 2), 5).value().is_equal(Matrix::identity(2));  // true
matrix_exp_pade(Matrix::zero(2, 2), 2).value().is_equal(Matrix::identity(2));    // true
matrix_exp(Matrix::zero(2, 2), 6, 4).value().is_equal(Matrix::identity(2));      // true

// N = [[0,1],[0,0]] has N^2 = 0 (index m = 2), so e^N = I + N = [[1,1],[0,1]]
// EXACTLY. Taylor terminates; Padé (any q >= 1) and scaling-and-squaring all
// reproduce it because m = 2 <= 2q+1.
const Matrix n = Matrix::from_rows({{ri(0), ri(1)}, {ri(0), ri(0)}}).value();
const Matrix eN = Matrix::from_rows({{ri(1), ri(1)}, {ri(0), ri(1)}}).value();
matrix_exp_taylor(n, 2).value().is_equal(eN);  // true
matrix_exp_pade(n, 1).value().is_equal(eN);    // true
matrix_exp(n, 2, 3).value().is_equal(eN);      // true

// On a nilpotent both routes are exact, hence agree.
matrix_exp_taylor(n, 4).value().is_equal(matrix_exp_pade(n, 2).value());  // true

// A non-terminating series pinned entrywise: taylor(diag(1,0), 4).
// (0,0) = 1 + 1 + 1/2 + 1/6 = 8/3;  (1,1) = 1 (only the k=0 term is nonzero).
const Matrix d = Matrix::from_rows({{ri(1), ri(0)}, {ri(0), ri(0)}}).value();
auto e = matrix_exp_taylor(d, 4).value();
e.at(0, 0) == rr(8, 3);  // true
e.at(1, 1) == ri(1);     // true

// The Padé exactness boundary. J4 is the 4x4 Jordan block, J4^4 = 0 (index 4),
// with true e^{J4} carrying J4^3/6 at entry (0,3).
const Matrix j4 = Matrix::from_rows({
    {ri(0), ri(1), ri(0), ri(0)},
    {ri(0), ri(0), ri(1), ri(0)},
    {ri(0), ri(0), ri(0), ri(1)},
    {ri(0), ri(0), ri(0), ri(0)}}).value();
auto exact = matrix_exp_taylor(j4, 4).value();  // terms = index => exact e^{J4}
exact.at(0, 3) == rr(1, 6);                      // true

auto pade_low = matrix_exp_pade(j4, 1).value();  // 2q+1 = 3 < 4: NOT exact
pade_low.at(0, 3) == rr(1, 4);                   // 1/4, an approximation, not 1/6
exact.is_equal(pade_low);                        // false (index 4 > 2q+1 = 3)

auto pade_ok = matrix_exp_pade(j4, 2).value();   // 2q+1 = 5 >= 4: exact
exact.is_equal(pade_ok);                         // true

// Nilpotency test, and its shape guard.
is_nilpotent(n).value();                                    // true
is_nilpotent(Matrix::identity(2)).value();                 // false (identity)
is_nilpotent(Matrix::from_rows({{ri(1), ri(2), ri(3)},
                                {ri(4), ri(5), ri(6)}}).value())
    .error();                                               // MathError::domain_error

// Parameter guards.
matrix_exp_taylor(Matrix::identity(2), 0).error();  // domain_error (terms < 1)
matrix_exp_pade(Matrix::identity(2), 0).error();    // domain_error (q < 1)
matrix_exp(Matrix::identity(2), 0, 2).error();      // domain_error (q < 1)
```

## See also

- [`nimblecas.matrix`](matrix.md) — the exact `Rational` matrix type these
  routines build on (`multiply`, `scale`, `inverse`, `identity`, `zero`).
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field every entry
  lives in.
- [`nimblecas.eigen`](eigen.md), [`nimblecas.dynamics`](dynamics.md), and
  [`nimblecas.laplace`](laplace.md) — sibling linear-algebra and linear-ODE
  consumers.
- [Documentation hub](../Index.md)
```
