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

## The exact-square / exact-ratio honesty convention

Several textbook descriptive statistics are **irrational** — a standard
deviation, a Pearson correlation `r`, a coefficient of variation, or a
standardised skewness all take a square (or odd) root of a rational. A root of a
rational is almost never rational, so returning one would force a `double` and
break the exact-`ℚ` contract. Rather than lie with a float or throw away the
quantity, this module exposes the **exact rational square or ratio** that lives
underneath the root, and documents precisely what was withheld:

| Irrational textbook quantity | What this module returns (exact `ℚ`) | Recovering the original |
| :--- | :--- | :--- |
| Standard deviation `σ = √var` | `variance` | `σ = √variance` (take the root yourself, in whatever precision you choose). |
| Pearson correlation `r = cov/√(var·var)` | `pearson_correlation_squared` → `r²` | `r = ±√(r²)`; the sign is `sign(covariance(x, y, …))`. |
| Coefficient of variation `cv = σ/μ` | `coefficient_of_variation_squared` → `cv²` | `cv = ±√(cv²)`; the sign is `sign(mean)`. |
| Skewness `g₁ = m₃ / m₂^{3/2}` | `skewness_squared` → `m₃²/m₂³` | `g₁ = ±√(skewness_squared)`; the sign is `sign(central_moment(data, 3))`. |

The **excess kurtosis** `m₄/m₂² − 3` involves **no root** — it is already a ratio
of rationals — so it is returned directly by `excess_kurtosis`, exactly. The rule
of thumb: whenever a name here ends in `_squared`, the value is the exact rational
square of the irrational statistic, never that statistic itself.

## Functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `mean` | `auto mean(std::span<const Rational> data) -> Result<Rational>` | The arithmetic mean `(1/n) · Σ_i data_i`. Empty data has no mean and fails `domain_error`. |
| `variance` | `auto variance(std::span<const Rational> data, bool sample) -> Result<Rational>` | `(1/(n−1)) · Σ_i (data_i − mean)²` when `sample`, else `(1/n) · Σ_i (data_i − mean)²`. Needs `n ≥ 2` (sample) / `n ≥ 1` (population); a shortfall fails `domain_error`. |
| `covariance` | `auto covariance(std::span<const Rational> x, std::span<const Rational> y, bool sample) -> Result<Rational>` | `(1/(n−1)) · Σ_i (x_i − x̄)(y_i − ȳ)` when `sample`, else the `1/n` form. `x` and `y` must have equal length, with the same `n` constraint as `variance`; a length mismatch or shortfall fails `domain_error`. |
| `covariance_matrix` | `auto covariance_matrix(const std::vector<std::span<const Rational>>& variables, bool sample) -> Result<Matrix>` | The symmetric `d × d` matrix `Σ` with `Σ_{jk} = covariance(variables[j], variables[k], sample)`. `variables[j]` is the sample vector of the `j`-th variable; every variable must share the common length `n`. The diagonal `Σ_{jj}` is the variance of variable `j`. An empty variable list, ragged variables (unequal lengths), or a length below the (co)variance constraint fails `domain_error`. |
| `weighted_mean` | `auto weighted_mean(std::span<const Rational> data, std::span<const Rational> weights) -> Result<Rational>` | `(Σ_i w_i x_i) / (Σ_i w_i)`, exact. `data` and `weights` must share the same non-empty length. A length mismatch, empty input, or a **zero total weight** (mean undefined) fails `domain_error`. Weights may be negative if their total is non-zero. |
| `raw_moment` | `auto raw_moment(std::span<const Rational> data, unsigned k) -> Result<Rational>` | The `k`-th raw moment `(1/n) · Σ_i x_i^k`, exact. `k = 0` yields `1`. Empty data fails `domain_error`. |
| `central_moment` | `auto central_moment(std::span<const Rational> data, unsigned k) -> Result<Rational>` | The `k`-th central moment `(1/n) · Σ_i (x_i − x̄)^k` (population form, divisor `n`), exact. `m₀ = 1`, `m₁ = 0`, `m₂ = ` population variance. Empty data fails `domain_error`. |
| `skewness_squared` | `auto skewness_squared(std::span<const Rational> data) -> Result<Rational>` | The **exact** squared skewness `m₃²/m₂³` (see the honesty convention above — the signed skewness is irrational and omitted; its sign is `sign(central_moment(data, 3))`). Requires `m₂ ≠ 0`; zero variance or empty data fails `domain_error`. |
| `excess_kurtosis` | `auto excess_kurtosis(std::span<const Rational> data) -> Result<Rational>` | The **exact** excess kurtosis `m₄/m₂² − 3` (population moments) — no root involved, so returned directly. Requires `m₂ ≠ 0`; zero variance or empty data fails `domain_error`. |
| `median` | `auto median(std::span<const Rational> data) -> Result<Rational>` | The exact median: the middle order statistic for odd `n`, the exact rational average of the two middle order statistics for even `n`. Sorts a copy under an exact cross-multiplied comparison. Empty data fails `domain_error`. |
| `quantile` | `auto quantile(std::span<const Rational> data, const Rational& p) -> Result<Rational>` | The exact **type-7** (linear-interpolation) quantile at `p ∈ [0, 1]`: with the data sorted ascending and `h = (n−1)·p`, `lo = ⌊h⌋`, it is `x_lo + (h − lo)·(x_{lo+1} − x_lo)`. `p ∉ [0, 1]` or empty data fails `domain_error`. |
| `range` | `auto range(std::span<const Rational> data) -> Result<Rational>` | The exact range `max − min`. Empty data fails `domain_error`. |
| `iqr` | `auto iqr(std::span<const Rational> data) -> Result<Rational>` | The exact interquartile range `quantile(data, 3/4) − quantile(data, 1/4)` (type-7). Empty data fails `domain_error`. |
| `mode` | `auto mode(std::span<const Rational> data) -> Result<Rational>` | The most frequent value; ties are broken toward the **smallest** value (deterministic). Empty data fails `domain_error`. |
| `modes` | `auto modes(std::span<const Rational> data) -> Result<std::vector<Rational>>` | Every value attaining the maximum frequency, in ascending order (a single element when the mode is unique). Empty data fails `domain_error`. |
| `pearson_correlation_squared` | `auto pearson_correlation_squared(std::span<const Rational> x, std::span<const Rational> y) -> Result<Rational>` | The **exact** `r² = cov(x,y)² / (var(x)·var(y))` (see honesty convention — the divisor cancels, so no `sample` flag; the signed `r` is irrational and omitted, sign = `sign(covariance)`). `r² ∈ [0, 1]`, `= 1` for perfectly (anti-)correlated data. Length mismatch, empty input, or a zero-variance variable fails `domain_error`. |
| `correlation_squared_matrix` | `auto correlation_squared_matrix(const std::vector<std::span<const Rational>>& variables) -> Result<Matrix>` | The symmetric `d × d` matrix `R²` with `R²_{jk} = pearson_correlation_squared(variables[j], variables[k])`; the diagonal is exactly `1`. Empty/ragged variables or any zero-variance variable fails `domain_error`. |
| `coefficient_of_variation_squared` | `auto coefficient_of_variation_squared(std::span<const Rational> data, bool sample) -> Result<Rational>` | The **exact** `cv² = variance(data, sample) / mean(data)²` (see honesty convention — the signed `cv` is irrational and omitted, sign = `sign(mean)`). Requires a non-zero mean; a zero mean or a data set too small for the chosen variance fails `domain_error`. |

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
| Empty data in `weighted_mean`, `raw_moment`, `central_moment`, `median`, `quantile`, `range`, `iqr`, `mode`, `modes` | `MathError::domain_error` |
| Length mismatch in `weighted_mean` / `pearson_correlation_squared` | `MathError::domain_error` |
| Zero total weight in `weighted_mean` | `MathError::domain_error` |
| Zero variance (`m₂ = 0`) in `skewness_squared` / `excess_kurtosis` | `MathError::domain_error` |
| `p ∉ [0, 1]` in `quantile` | `MathError::domain_error` |
| Zero-variance variable in `pearson_correlation_squared` / `correlation_squared_matrix` | `MathError::domain_error` |
| Zero mean in `coefficient_of_variation_squared` | `MathError::domain_error` |
| An `int64` numerator or denominator computation wraps (including the cross-multiplied order comparison used by the sort) | `MathError::overflow` |

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

### Order statistics, moments, and the exact-square convention

```cpp
// Weighted mean (exact): {1,2,3} weighted by {1,2,3} = 14/6 = 7/3.
auto wm = weighted_mean(span(ints({1, 2, 3})), span(ints({1, 2, 3}))).value();  // 7/3

// Median: even n averages the two middle order statistics, odd n takes the middle.
auto me = median(span(ints({1, 2, 3, 4}))).value();         // 5/2
auto mo = median(span(ints({1, 2, 3}))).value();            // 2

// Type-7 quantiles of {1,2,3,4}: Q1 = 7/4, Q2 = 5/2, Q3 = 13/4; iqr = 3/2.
auto q1 = quantile(span(ints({1, 2, 3, 4})), Rational::make(1, 4).value()).value();  // 7/4
auto md = quantile(span(ints({1, 2, 3, 4})), Rational::make(1, 2).value()).value();  // 5/2
auto iq = iqr(span(ints({1, 2, 3, 4}))).value();            // 3/2
auto rg = range(span(ints({1, 2, 3, 4}))).value();          // 3

// Mode: most frequent value; ties break to the smallest.
auto md1 = mode(span(ints({1, 2, 2, 3, 3, 3}))).value();    // 3
auto mds = modes(span(ints({1, 1, 2, 2}))).value();         // {1, 2} ascending

// Moments of {1,2,3,4,5} (mean 3): m₂ = 2, excess kurtosis = (34/5)/4 − 3 = −13/10.
auto ek = excess_kurtosis(span(ints({1, 2, 3, 4, 5}))).value();   // -13/10  (exact, no root)

// Skewness is irrational; the exact rational SQUARE is returned instead.
// {0,0,0,0,4} has skewness 3/2, so skewness_squared = 9/4; sign is sign(m₃) > 0.
auto sk2 = skewness_squared(span(ints({0, 0, 0, 0, 4}))).value();  // 9/4
auto m3  = central_moment(span(ints({0, 0, 0, 0, 4})), 3).value(); // 768/125 > 0 -> +skew

// Pearson r is irrational; r² is exact. Perfectly (anti-)correlated data -> r² = 1.
auto r2 = pearson_correlation_squared(span(ints({1, 2, 3})), span(ints({2, 4, 6}))).value();  // 1

// Coefficient of variation is irrational; cv² is exact.
// {1,2,3}: sample var 1, mean 2 -> cv² = 1/4.
auto cv2 = coefficient_of_variation_squared(span(ints({1, 2, 3})), true).value();  // 1/4
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
