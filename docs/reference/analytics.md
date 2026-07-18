# `nimblecas.analytics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/analytics/analytics.cppm`

Portfolio and risk analytics: return-series construction, summary statistics,
performance ratios (Sharpe, Sortino, Treynor, information ratio, beta/alpha),
drawdown, value-at-risk and conditional VaR (both historical and
parametric-Gaussian), and **mean-variance portfolio optimization** (global
minimum-variance and tangency/max-Sharpe portfolios in closed form, plus an
efficient frontier by the two-fund theorem), backed by an internal Cholesky solve
of the covariance system. This is the MATLAB `portopt`/`Portfolio` +
financial-ratio surface.

## Honesty boundary

This layer is **NUMERICAL / STATISTICAL**: every output is a **sample estimate**
over the supplied series, not an exact quantity, and VaR/CVaR carry the usual
estimation error of their sample. What the module guarantees instead is
**disciplined failure** — every degenerate input rides the railway rather than
returning a `NaN`:

- An **empty or too-short** series, a **mismatched** pair of series, or a
  **degenerate optimizer** input returns a `MathError` (`domain_error`).
- A **non-positive-definite covariance** (a singular or collinear market) is
  **refused** at the Cholesky factorization → `domain_error`, never silently
  "solved".
- Ratios that would **divide by a zero standard deviation** (or zero downside,
  zero beta, zero market variance) return `MathError::division_by_zero`, never a
  `±inf` dressed up as a number.

Two capabilities are a deliberate **better-than-parity** feature over a plain
spreadsheet: **historical** VaR/CVaR (empirical quantiles, not only the Gaussian
parametric form) and the **equality-constrained efficient frontier** (fully
invested, long/short allowed, via the two-fund closed form).

```cpp
import nimblecas.analytics;
```

Depends only on [`core`](core.md) (the `Result` / `MathError` railway).
Everything lives in namespace `nimblecas::analytics`.

## `Annualisation` — ratio scaling policy

`struct Annualisation { double periods_per_year{1.0}; }` — a ratio's numerator is
multiplied by `periods_per_year` and its denominator by `√periods_per_year` (the
standard √-of-time scaling). `periods_per_year == 1` (the default) leaves the
ratio per-period.

## Return series

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `simple_returns` | `auto simple_returns(std::span<const double> prices) -> Result<std::vector<double>>` | Simple period returns `r_t = P_t/P_{t−1} − 1`. Needs ≥ 2 prices (`domain_error`); a zero prior price → `division_by_zero`. |
| `log_returns` | `auto log_returns(std::span<const double> prices) -> Result<std::vector<double>>` | Log returns `ln(P_t/P_{t−1})`. Needs ≥ 2 prices; a non-positive price → `domain_error`. |

## Summary statistics

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `mean` | `auto mean(std::span<const double> x) -> Result<double>` | Arithmetic mean. Empty → `domain_error`. |
| `variance` | `auto variance(std::span<const double> x, bool sample = true) -> Result<double>` | Sample `(n−1)` variance by default; population when `sample == false`. Needs 2 points (sample) / 1 (population), else `domain_error`. |
| `stddev` | `auto stddev(std::span<const double> x, bool sample = true) -> Result<double>` | `√variance`. |
| `covariance` | `auto covariance(std::span<const double> x, std::span<const double> y, bool sample = true) -> Result<double>` | Sample/population covariance. Length mismatch or too-short → `domain_error`. |
| `correlation` | `auto correlation(std::span<const double> x, std::span<const double> y) -> Result<double>` | Pearson correlation. A zero standard deviation → `division_by_zero`. |
| `covariance_matrix` | `auto covariance_matrix(std::span<const std::vector<double>> series, bool sample = true) -> Result<std::vector<std::vector<double>>>` | Symmetric covariance matrix of a set of series (assets in rows, observations in columns). Ragged/empty → `domain_error`. |

## Performance ratios

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `sharpe_ratio` | `auto sharpe_ratio(std::span<const double> returns, double risk_free_per_period, Annualisation ann = {}) -> Result<double>` | `(mean − rf)/std`, optionally annualised. Zero std → `division_by_zero`. |
| `sortino_ratio` | `auto sortino_ratio(std::span<const double> returns, double mar, Annualisation ann = {}) -> Result<double>` | Like Sharpe but the denominator is the downside deviation below `mar`. Zero downside → `division_by_zero`. |
| `treynor_ratio` | `auto treynor_ratio(std::span<const double> returns, std::span<const double> market, double risk_free_per_period) -> Result<double>` | `(mean − rf)/beta`. Zero beta → `division_by_zero`. |
| `information_ratio` | `auto information_ratio(std::span<const double> returns, std::span<const double> benchmark) -> Result<double>` | `mean(active)/std(active)`, `active = returns − benchmark`. Length mismatch / < 2 points → `domain_error`; zero tracking error → `division_by_zero`. |
| `beta` | `auto beta(std::span<const double> asset, std::span<const double> market) -> Result<double>` | CAPM beta `cov(asset, market)/var(market)`. Zero market variance → `division_by_zero`. |
| `alpha` | `auto alpha(std::span<const double> asset, std::span<const double> market, double risk_free_per_period) -> Result<double>` | Jensen's alpha `mean_asset − [rf + beta·(mean_market − rf)]`. |

## Drawdown & tail risk

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `max_drawdown` | `auto max_drawdown(std::span<const double> equity_curve) -> Result<double>` | Maximum peak-to-trough fractional decline (a non-negative number; `0.2` == a 20% drawdown). Empty → `domain_error`. |
| `value_at_risk_historical` | `auto value_at_risk_historical(std::span<const double> returns, double conf) -> Result<double>` | Empirical `(1−conf)` lower quantile, reported as a **positive** loss fraction. Empty or `conf ∉ (0,1)` → `domain_error`. |
| `conditional_var_historical` | `auto conditional_var_historical(std::span<const double> returns, double conf) -> Result<double>` | Mean loss beyond the historical VaR (expected shortfall), positive. |
| `value_at_risk_gaussian` | `auto value_at_risk_gaussian(double mean_return, double std_return, double conf) -> Result<double>` | Parametric VaR `−(μ + σ·z_{1−conf})`, positive. `std_return < 0` or `conf ∉ (0,1)` → `domain_error`. |
| `conditional_var_gaussian` | `auto conditional_var_gaussian(double mean_return, double std_return, double conf) -> Result<double>` | Gaussian expected shortfall `−μ + σ·φ(z)/(1−conf)`. |

## Portfolio construction & mean-variance optimization

`cov` is a square, symmetric covariance matrix; a non-positive-definite `cov` is
refused at the Cholesky step (`domain_error`).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `portfolio_return` | `auto portfolio_return(std::span<const double> weights, std::span<const double> mean_returns) -> Result<double>` | `Σ wᵢ·μᵢ`. Length mismatch / empty → `domain_error`. |
| `portfolio_variance` | `auto portfolio_variance(std::span<const double> weights, std::span<const std::vector<double>> cov) -> Result<double>` | `wᵀ Σ w`. Non-square or mismatched → `domain_error`. |
| `min_variance_weights` | `auto min_variance_weights(std::span<const std::vector<double>> cov) -> Result<std::vector<double>>` | Global minimum-variance weights `Σ⁻¹1 / (1ᵀΣ⁻¹1)`, fully invested, long/short allowed. Non-PD `cov` → `domain_error`; zero denominator → `division_by_zero`. |
| `tangency_weights` | `auto tangency_weights(std::span<const std::vector<double>> cov, std::span<const double> mean_returns, double risk_free) -> Result<std::vector<double>>` | Max-Sharpe weights `Σ⁻¹(μ − rf)` normalised to sum 1. |
| `efficient_portfolio` | `auto efficient_portfolio(std::span<const std::vector<double>> cov, std::span<const double> mean_returns, double target_return) -> Result<FrontierPoint>` | The minimum-variance portfolio achieving `target_return`, via the two-fund closed form. A degenerate frontier (e.g. equal means → determinant `D ≈ 0`) is **honestly reported** as `domain_error`, not guessed. |
| `efficient_frontier` | `auto efficient_frontier(std::span<const std::vector<double>> cov, std::span<const double> mean_returns, int points) -> Result<std::vector<FrontierPoint>>` | A frontier sampled at `points` targets, evenly spaced from the min-variance return to the max single-asset mean. `points < 2` or empty means → `domain_error`. |

`FrontierPoint` is `struct { std::vector<double> weights; double risk; double ret; }`
— the weights, the portfolio standard deviation, and its return.

## Error model

| Condition | Error |
| :--- | :--- |
| Empty / too-short / ragged / mismatched-length series; `conf ∉ (0,1)`; `std_return < 0`; `points < 2`; a **degenerate frontier** (`D ≈ 0`, e.g. equal means) | `MathError::domain_error` |
| A **non-positive-definite** covariance matrix (fails Cholesky) | `MathError::domain_error` |
| A zero standard deviation (Sharpe / correlation / information ratio), zero downside (Sortino), zero beta (Treynor), zero market variance (beta), a zero prior price (`simple_returns`), or a zero optimization denominator | `MathError::division_by_zero` |

No result is ever claimed exact — all outputs are sample estimates over `double`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.analytics;
using namespace nimblecas;
using namespace nimblecas::analytics;

// Returns and summary statistics on tiny exact series.
const std::array<double, 3> px{100.0, 110.0, 121.0};
simple_returns(px).value();                 // [0.10, 0.10]
const std::array<double, 3> x{1.0, 2.0, 3.0};
mean(x).value();                            // 2
variance(x).value();                        // 1   (sample, n-1)
const std::array<double, 3> y{3.0, 2.0, 1.0};
covariance(x, y).value();                   // -1
correlation(x, y).value();                  // -1

// Performance ratios. ret has mean 0.02, sample std 0.01 -> Sharpe (rf 0) == 2.
const std::array<double, 3> ret{0.01, 0.02, 0.03};
sharpe_ratio(ret, 0.0).value();                        // 2
sharpe_ratio(ret, 0.0, Annualisation{4.0}).value();    // 4   (scaled by sqrt(4))
beta(ret, ret).value();                                // 1

// Drawdown and value-at-risk.
const std::array<double, 4> eq{100.0, 120.0, 90.0, 110.0};
max_drawdown(eq).value();                              // 0.25  (peak 120 -> trough 90)
const std::array<double, 5> r5{-0.05, -0.02, 0.0, 0.01, 0.03};
value_at_risk_historical(r5, 0.95).value();            // 0.05  (worst return, positive loss)
value_at_risk_gaussian(0.0, 1.0, 0.95).value();        // ≈ 1.6449  (standard-normal quantile)

// Mean-variance optimization closed forms. Diagonal cov -> min-var weights ∝ 1/variance.
const std::vector<std::vector<double>> cov{{0.04, 0.0}, {0.0, 0.09}};
auto w = min_variance_weights(cov).value();
w[0] + w[1];                                           // 1  (fully invested)
// 1/0.04 : 1/0.09 -> w0 ≈ 0.6923, w1 ≈ 0.3077.

// A frontier point achieves its target return and sums to 1 (needs distinct means).
const std::vector<std::vector<double>> cov2{{0.04, 0.0}, {0.0, 0.04}};
const std::array<double, 2> mu2{0.08, 0.12};
auto pt = efficient_portfolio(cov2, mu2, 0.10).value();
pt.ret;                                                // 0.10
pt.weights[0] + pt.weights[1];                         // 1

// Honest failure: equal means make the frontier degenerate -> error, not a bogus portfolio.
const std::array<double, 2> mu{0.10, 0.10};
efficient_portfolio(cov2, mu, 0.10).has_value();       // false (domain_error)
// A singular covariance is refused, not silently solved.
const std::vector<std::vector<double>> bad{{0.0, 0.0}, {0.0, 0.0}};
min_variance_weights(bad).has_value();                 // false (not PD)
```

## See also

- [`nimblecas.pricing`](pricing.md) — the derivatives-pricing sibling; a pricing
  `Portfolio`'s value feeds a risk book analysed here.
- [`nimblecas.stats`](stats.md) — the **exact-over-Q** descriptive-statistics
  module this layer mirrors numerically over `double`.
- [`nimblecas.montecarlo`](montecarlo.md) — the numerical sample-statistics
  neighbour on the RNG substrate.
- [`nimblecas.matstruct`](matstruct.md) — exact `LDLᵀ`/Cholesky factorizations,
  the exact counterpart to the numerical Cholesky solve here.
- [Documentation hub](../Index.md)
