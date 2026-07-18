# `nimblecas.portfolio` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/portfolio/portfolio.cppm`

Integrated portfolio analytics layered on [`analytics`](analytics.md): a **one-call risk
report** and a **robust mean-variance optimizer**. Where `analytics` gives the primitives
(ratios, VaR, a Cholesky frontier that refuses a non-positive-definite matrix), this module
aggregates them and adds a Markowitz optimizer that stays well-posed on ill-conditioned or
singular sample covariance via **diagonal ridge regularization** (`Σ + λI`) solved with an
**LU-with-partial-pivoting** solver.

```cpp
import nimblecas.portfolio;
```

Depends on [`core`](core.md) and [`analytics`](analytics.md).

## Honesty boundary

NUMERICAL / STATISTICAL. Every figure is a sample estimate; failure rides the
`Result<T>`/`MathError` railway. The regularizer is **explicit** — `ridge_lambda` is an
argument, never a hidden fudge — and `analyze()` propagates the first primitive failure
rather than filling a partial report. `lu_solve_ridge` returns `nullopt` only when the
regularized matrix is still numerically singular (`λ == 0` on a singular `Σ`).

## Integrated risk report

| Symbol | Signature | Notes |
|---|---|---|
| `RiskReport` | struct: `sharpe, sortino, treynor, jensen_alpha, beta, max_drawdown, var_historical, cvar_historical, var_parametric, cvar_parametric, confidence` | The full one-shot scorecard. |
| `analyze` | `auto analyze(std::span<const double> returns, std::span<const double> market, double risk_free_per_period = 0.0, double confidence = 0.95, analytics::Annualisation ann = {}) -> Result<RiskReport>` | Computes all ten measures. The drawdown equity curve is reconstructed by compounding the returns from 1.0. Mismatched/short series → `domain_error`. |

## Robust mean-variance optimization

| Symbol | Signature | Notes |
|---|---|---|
| `lu_solve_ridge` | `auto lu_solve_ridge(std::span<const std::vector<double>> matrix, std::span<const double> rhs, double ridge_lambda) -> std::optional<std::vector<double>>` | Solves `(Σ + λI) x = rhs` by LU with partial pivoting. The reusable numerical core. |
| `min_variance_weights` | `... (cov, double ridge_lambda = 0.0) -> Result<std::vector<double>>` | GMV weights `(Σ+λI)⁻¹1 / (1ᵀ(Σ+λI)⁻¹1)`, fully invested. |
| `tangency_weights` | `... (cov, mean_returns, risk_free, ridge_lambda = 0.0) -> Result<std::vector<double>>` | Max-Sharpe weights `(Σ+λI)⁻¹(μ−rf)` normalised to sum 1. |
| `efficient_portfolio` | `... (cov, mean_returns, target_return, ridge_lambda = 0.0) -> Result<FrontierPoint>` | Min-variance portfolio at a target return via the A/B/C/D two-fund closed form. |
| `efficient_frontier` | `... (cov, mean_returns, int points, ridge_lambda = 0.0) -> Result<std::vector<FrontierPoint>>` | `points` frontier samples between the GMV return and the largest single-asset mean. |

`FrontierPoint` = `{ std::vector<double> weights; double risk; double ret; }`.

## Error model

| Error | When |
|---|---|
| `domain_error` | non-square/empty covariance, dimension mismatch, `confidence ∉ (0,1)`, a still-singular regularized system, a degenerate (D≈0) frontier, or a propagated primitive failure. |
| `division_by_zero` | zero weight-normaliser, or a zero-volatility/zero-beta ratio denominator inside `analyze`. |

## Worked example

```cpp
import nimblecas.portfolio;
import std;
using namespace nimblecas::portfolio;

// A singular sample covariance (two perfectly collinear assets) is UNSOLVABLE unregularized,
// but ridge regularization rescues it:
const std::vector<std::vector<double>> singular{{0.0, 0.0}, {0.0, 0.0}};
auto bad  = min_variance_weights(singular, 0.0);   // -> error (singular)
auto good = min_variance_weights(singular, 0.1);   // -> [0.5, 0.5] (Σ+0.1I = 0.1·I)

// One-call risk report over a return series and its benchmark:
const std::array<double,3> r{-0.02, 0.01, 0.03}, mkt{-0.02, 0.01, 0.03};
auto rep = analyze(r, mkt, 0.0, 0.95).value();
// rep.beta == 1, rep.jensen_alpha == 0, rep.max_drawdown == 0.02, rep.var_historical == 0.02.
```
