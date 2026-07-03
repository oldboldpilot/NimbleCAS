# `nimblecas.stats` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/stats/stats.cppm`

Exact descriptive statistics over the **rationals** (ROADMAP §7.7). Every datum
is a [`Rational`](ratpoly.md) — a reduced `int64` fraction — so a mean, variance,
or covariance is the fraction it mathematically **is**, never a `double` that
happens to be close. The headline deliverable is the **covariance matrix** `Σ`:
for `d` variables measured over a common sample of length `n`, `Σ_{jk}` is the
covariance of variables `j` and `k`, a symmetric `d × d` matrix whose diagonal
holds each variable's variance. It is returned as a
[`nimblecas.matrix`](matrix.md) `Matrix`, so it composes directly with the exact
linear algebra — feed it straight to `determinant`, `inverse`, or `rank`.

```cpp
import nimblecas.stats;
```

Depends on [`core`](core.md), [`ratpoly`](ratpoly.md), and [`matrix`](matrix.md).

## The exact-`Rational` contract

Following the rest of the engine, arithmetic is **exact** and
**overflow-checked** (Rule 32): every step flows through `Rational`'s checked
add / subtract / multiply / divide, so an `int64` numerator or denominator that
would overflow surfaces as `MathError::overflow` rather than silently wrapping.
Empty data, mismatched lengths, and too-few-points-for-a-sample-statistic
surface as `MathError::domain_error`.

Computation is a stable **two-pass** scheme: the mean(s) are formed first, then
the (co)moment sum of exact deviations from those means. Because no floating
point is ever used, there is no catastrophic cancellation to guard against — the
naive two-pass sum is already exact.

## Sample vs. population

Every (co)variance takes a `bool sample` flag selecting the divisor (the degrees
of freedom) applied to the (co)moment sum:

| `sample` | Divisor | Minimum `n` | Statistic |
| :--- | :--- | :--- | :--- |
| `true` | `n − 1` | `n ≥ 2` | Bessel-corrected **sample** (co)variance. |
| `false` | `n` | `n ≥ 1` | **population** (co)variance. |

A sample length below the relevant minimum fails with `domain_error` (the sample
form divides by `n − 1`, so it needs at least two points).

## Functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `mean` | `auto mean(std::span<const Rational> data) -> Result<Rational>` | The arithmetic mean `(1/n) · Σ_i data_i`. Empty data has no mean and fails `domain_error`. |
| `variance` | `auto variance(std::span<const Rational> data, bool sample) -> Result<Rational>` | `(1/(n−1)) · Σ_i (data_i − mean)²` when `sample`, else `(1/n) · Σ_i (data_i − mean)²`. Needs `n ≥ 2` (sample) / `n ≥ 1` (population); a shortfall fails `domain_error`. |
| `covariance` | `auto covariance(std::span<const Rational> x, std::span<const Rational> y, bool sample) -> Result<Rational>` | `(1/(n−1)) · Σ_i (x_i − x̄)(y_i − ȳ)` when `sample`, else the `1/n` form. `x` and `y` must have equal length, with the same `n` constraint as `variance`; a length mismatch or shortfall fails `domain_error`. |
| `covariance_matrix` | `auto covariance_matrix(const std::vector<std::span<const Rational>>& variables, bool sample) -> Result<Matrix>` | The symmetric `d × d` matrix `Σ` with `Σ_{jk} = covariance(variables[j], variables[k], sample)`. `variables[j]` is the sample vector of the `j`-th variable; every variable must share the common length `n`. The diagonal `Σ_{jj}` is the variance of variable `j`. An empty variable list, ragged variables (unequal lengths), or a length below the (co)variance constraint fails `domain_error`. |

`covariance(x, x, sample)` equals `variance(x, sample)`, and
`covariance(x, y, sample)` equals `covariance(y, x, sample)` — the covariance
matrix is symmetric by construction, computed once per `(j, k)` pair with
`j ≤ k` and mirrored across the diagonal.

## Error model

| Condition | Error |
| :--- | :--- |
| Empty data in `mean` | `MathError::domain_error` |
| Fewer than `n` points for the chosen statistic (`variance`/`covariance`) | `MathError::domain_error` |
| Unequal-length `x`, `y` in `covariance` | `MathError::domain_error` |
| Empty variable list in `covariance_matrix` | `MathError::domain_error` |
| Ragged variables (unequal lengths) in `covariance_matrix` | `MathError::domain_error` |
| Sample length below the (co)variance constraint in `covariance_matrix` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps | `MathError::overflow` |

## Examples

```cpp
import nimblecas.stats;
import nimblecas.ratpoly;
import nimblecas.matrix;
using namespace nimblecas;

auto ints = [](std::vector<std::int64_t> vs) {
    std::vector<Rational> out;
    for (auto v : vs) out.push_back(Rational::from_int(v));
    return out;
};
auto span = [](const std::vector<Rational>& v) { return std::span<const Rational>{v}; };

// Mean (exact): mean {1,2,3,4} = 5/2.
auto m = mean(span(ints({1, 2, 3, 4}))).value();            // 5/2

// Variance of {1,2,3} (mean 2):
auto d = ints({1, 2, 3});
auto vp = variance(span(d), false).value();                 // population: 2/3
auto vs = variance(span(d), true).value();                  // sample:     1

// Covariance (sample): cov({1,2,3}, {1,2,3}) equals var, = 1;
// perfectly anti-correlated {1,2,3} vs {3,2,1} = -1.
auto x = ints({1, 2, 3});
auto y = ints({3, 2, 1});
auto c1 = covariance(span(x), span(x), true).value();       // 1
auto c2 = covariance(span(x), span(y), true).value();       // -1

// Covariance matrix (sample) of two identical variables X = Y = {1,2,3}:
std::vector<std::span<const Rational>> corr{span(x), span(ints({1, 2, 3}))};
auto sig_corr = covariance_matrix(corr, true).value();      // [[1, 1], [1, 1]]

// Covariance matrix (sample) of anti-correlated X = {1,2,3}, Y = {3,2,1}:
std::vector<std::span<const Rational>> anti{span(x), span(y)};
auto sig_anti = covariance_matrix(anti, true).value();      // [[1, -1], [-1, 1]]
// sig_anti is symmetric; sig_anti.at(0,0) == var(X), sig_anti.at(1,1) == var(Y).
```

Domain edges:

```cpp
mean(span(std::vector<Rational>{})).error();                // domain_error (empty)

auto one = ints({5});
variance(span(one), true).error();                          // domain_error (n < 2)
variance(span(one), false).value();                         // 0 (population var of one point)

// Ragged / mismatched / empty inputs all fail domain_error:
auto a = ints({1, 2, 3});
auto b = ints({1, 2});
std::vector<std::span<const Rational>> ragged{span(a), span(b)};
covariance_matrix(ragged, true).error();                    // domain_error (ragged)
covariance(span(a), span(b), true).error();                 // domain_error (length mismatch)
covariance_matrix({}, true).error();                        // domain_error (empty list)
```

## See also

- [`nimblecas.ratpoly`](ratpoly.md) — the exact `Rational` field the data and
  results live in.
- [`nimblecas.matrix`](matrix.md) — the exact linear algebra the covariance
  matrix `Σ` is returned as, and composes with (`determinant`, `inverse`,
  `rank`).
- [`nimblecas.combinatorics`](combinatorics.md) and
  [`nimblecas.orthopoly`](orthopoly.md) — the sibling `ratpoly`-consuming
  numeric modules.
- [Documentation hub](../Index.md)
