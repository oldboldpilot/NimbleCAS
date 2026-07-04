# `nimblecas.pade` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pade/pade.cppm`

Rational resummation of a formal power series (ROADMAP §7.4/7.5). The **[m/n]
Padé approximant** of a series `s = c_0 + c_1 x + c_2 x^2 + …` is the rational
function `P(x)/Q(x)` with `deg P <= m`, `deg Q <= n`, normalised so `Q(0) = 1`,
whose Maclaurin expansion agrees with `s` through the term `x^{m+n}`. Where a
truncated Taylor polynomial diverges past its radius of convergence, the
rational Padé form frequently continues to track the underlying function — and
reproduces it **exactly** when the function is itself rational (recovering `P`
and `Q` up to the `Q(0) = 1` normalisation). This module exports a single free
function, `pade`, sitting one layer above `powerseries`,
[`ratpoly`](ratpoly.md), and [`matrix`](matrix.md).

Everything is computed **exactly over Q** — no floating point ever enters. The
denominator's unknown coefficients `q_1..q_n` (with `q_0 = 1` fixed) come from
the `n` linear equations that force the coefficients of `x^{m+1}..x^{m+n}` in
the product `Q·s` to vanish; this is an `n × n` Toeplitz system `A q = b`
assembled from the series coefficients and solved by the exact rational matrix
solver. `P` is then read off from the low-order coefficients of `Q·s`. The
honest **solvability boundary** is twofold: the series must carry enough terms
(`s.order() >= m + n + 1`), and the Toeplitz system must be nonsingular. A
degenerate (singular) system means the requested approximant does not exist and
surfaces as `MathError::domain_error`, propagated from `Matrix::solve` (Rule 32
railway). Because the whole pipeline runs through `Rational`'s checked
arithmetic, an `int64` numerator or denominator that would wrap surfaces as
`MathError::overflow` rather than silently overflowing.

```cpp
import nimblecas.pade;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), [`matrix`](matrix.md),
and `powerseries`.

## `pade` — the [m/n] approximant

```cpp
[[nodiscard]] auto pade(const PowerSeries& s, std::size_t m, std::size_t n)
    -> Result<std::pair<RationalPoly, RationalPoly>>;
```

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `pade` | `auto pade(const PowerSeries& s, std::size_t m, std::size_t n) -> Result<std::pair<RationalPoly, RationalPoly>>` | Computes the `[m/n]` Padé approximant of the truncated series `s`. On success returns the pair `(P, Q)` of numerator and denominator as [`RationalPoly`](ratpoly.md) in ascending-degree coefficient order, with `Q` normalised so `Q(0) = 1` (`q_0 = 1`). |

Semantics of the returned pair:

- **`P`** has `deg P <= m`: its coefficients are `p_k = sum_{j=0..min(k,n)} q_j c_{k-j}` for `k = 0..m`, i.e. the degree-`<= m` coefficients of the product `Q·s`. Trailing zero coefficients are trimmed by `RationalPoly::from_coeffs`, so a degenerate case may return a `P` of degree strictly less than `m` (see the `[1/1]` of `1/(1-x)` example below, where `p_1 = 0`).
- **`Q`** has `deg Q <= n` with `q_0 = 1`. Its remaining coefficients `q_1..q_n` solve the Toeplitz system; for `n = 0` this step is skipped entirely.
- The construction is **exact**: given exact rational series coefficients, `P` and `Q` are the exact rational solution, verified coefficient-by-coefficient in the tests.

### Construction detail

For rows `r = 0..n-1` (equation `k = m+1+r`, forcing the coefficient of
`x^{m+1+r}` in `Q·s` to vanish) the assembled system is

```
A[r][cc] = c_{m + r - cc}    (cc = 0..n-1, the q_{cc+1} column; c_i = 0 for i < 0)
b[r]     = -c_{m + 1 + r}
```

solved by `Matrix::solve` to give `q_1..q_n`. The **`n = 0` case is handled
directly**: `Q = 1` and `P` is simply the degree-`m` truncation of the series
(`p_k = c_k` for `k = 0..m`), with no linear system to solve — this is
infallible apart from the order check.

## Error model

| Condition | Error |
| :--- | :--- |
| Series too short: `s.order() < m + n + 1` (the coefficients `c_0..c_{m+n}` are not all present) | `MathError::domain_error` |
| The `n × n` Toeplitz system for `q_1..q_n` is singular (the `[m/n]` approximant is degenerate / does not exist) | `MathError::domain_error` (propagated from `Matrix::solve`) |
| An `int64` numerator or denominator computation wraps during the exact rational arithmetic (assembly, solve, or the `P` accumulation) | `MathError::overflow` |

The order bound is written as `m > max_index || n > max_index - m` with
`max_index = s.order() - 1`, so the sum `m + n + 1` cannot itself wrap a
`size_t`. A `PowerSeries` always has `order() >= 1`, so `max_index` never
underflows.

## Worked examples

```cpp
import nimblecas.pade;
import nimblecas.powerseries;
import nimblecas.ratpoly;
using namespace nimblecas;

auto ri  = [](std::int64_t v) { return Rational::from_int(v); };
auto rat = [](std::int64_t n, std::int64_t d) { return Rational::make(n, d).value(); };

// [1/1] Padé of exp: c = {1, 1, 1/2, 1/6}. => P = 1 + x/2, Q = 1 - x/2.
auto exp4 = PowerSeries::from_coeffs({ri(1), ri(1), rat(1, 2), rat(1, 6)}, 4).value();
auto r11  = pade(exp4, 1, 1).value();
r11.first.coefficient(0);   // 1
r11.first.coefficient(1);   // 1/2   (P = 1 + x/2)
r11.second.coefficient(1);  // -1/2  (Q = 1 - x/2)

// [2/2] Padé of exp: c = {1, 1, 1/2, 1/6, 1/24}.
// => P = 1 + x/2 + x^2/12, Q = 1 - x/2 + x^2/12.
auto exp5 = PowerSeries::from_coeffs(
    {ri(1), ri(1), rat(1, 2), rat(1, 6), rat(1, 24)}, 5).value();
auto r22  = pade(exp5, 2, 2).value();
r22.first.coefficient(2);   // 1/12
r22.second.coefficient(2);  // 1/12

// Exact recovery of a rational function. 1/(1-x): c = {1,1,1,1}.
// [0/1] recovers P = 1, Q = 1 - x exactly.
auto geom = PowerSeries::from_coeffs({ri(1), ri(1), ri(1), ri(1)}, 4).value();
auto r01  = pade(geom, 0, 1).value();
r01.first.coefficient(0);   // 1       (P = 1)
r01.second.coefficient(1);  // -1      (Q = 1 - x)

// Degenerate but valid: [1/1] of 1/(1-x) gives Q = 1 - x with P trimmed to 1.
auto r11d = pade(geom, 1, 1).value();
r11d.first.coefficient(0);  // 1
r11d.first.coefficient(1);  // 0       (p_1 = 0, degenerate — P trims to degree 0)

// n = 0: pure series truncation. Q = 1, P = degree-2 truncation of exp.
auto r20  = pade(exp4, 2, 0).value();
r20.first.coefficient(2);   // 1/2     (P = 1 + x + x^2/2)
r20.second.coefficient(0);  // 1       (Q = 1)

// Round-trip from a rational function: (1 + 2x)/(1 - x) = 1 + 3x + 3x^2 + 3x^3 + …
auto series = PowerSeries::from_coeffs({ri(1), ri(3), ri(3), ri(3)}, 4).value();
auto rr   = pade(series, 1, 1).value();
rr.first.coefficient(1);    // 2       (P = 1 + 2x)
rr.second.coefficient(1);   // -1      (Q = 1 - x)

// Error model: an order-2 series cannot pin down a [1/1] approximant.
auto tooShort = PowerSeries::from_coeffs({ri(1), ri(1)}, 2).value();
pade(tooShort, 1, 1).error();   // MathError::domain_error

// Singular Toeplitz system: c = {1, 0, 1, 1}. For [1/1] the 1x1 matrix is [c_1] = [0].
auto singular = PowerSeries::from_coeffs({ri(1), ri(0), ri(1), ri(1)}, 4).value();
pade(singular, 1, 1).error();   // MathError::domain_error (from solve)
```

## See also

- `nimblecas.powerseries` — the truncated formal power series over
  `Q` that `pade` consumes.
- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field and the
  `RationalPoly` numerator/denominator returned.
- [`nimblecas.matrix`](matrix.md) — the exact rational solver that resolves the
  Toeplitz system and raises `domain_error` on a singular one.
- [Documentation hub](../Index.md)
