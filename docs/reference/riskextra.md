# `nimblecas.riskextra` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/riskextra/riskextra.cppm`

A **numerical / statistical** companion to [`nimblecas.analytics`](analytics.md) and
[`nimblecas.portfolio`](portfolio.md), covering three families the closed-form mean-variance
layer leaves out: **box-constrained** risk / portfolio optimisation (active-set QP and the
Rockafellar-Uryasev CVaR LP, plus a self-contained two-phase simplex), the **French**
depreciation methods (AMORLINC / AMORDEGRC), and the **continuous-compounding** rate and
irregular-first-period annuity variants. Like `analytics`, every output is a *sample estimate*
over `double` — never claimed exact — and every degenerate input rides the `Result<T>` railway
rather than returning a `NaN`/`inf` dressed as a value. Every dense matrix routine **rejects a
non-finite entry** rather than letting a `NaN` defeat a pivot test, and DoS-sized inputs are
**refused, not allocated**.

```cpp
import nimblecas.riskextra;
```

Depends on [`core`](core.md) (the `Result` / `MathError` railway) and
[`rng`](rng.md) — the counter-based RNG that gives `simulate_correlated_returns` its
seed-reproducible, partition-independent correlated draws. Everything lives in namespace
`nimblecas::riskextra`; all entry points are free functions, `[[nodiscard]]`.

## Design note — why not `nimblecas.lp`?

`riskextra` does **not** reuse `nimblecas.lp`. That module is an **exact-rational, single-phase**
simplex requiring `b ≥ 0`, and it cannot express the Rockafellar-Uryasev program: the RU
formulation needs a **free** VaR variable (`α = ap − an`) and a budget **equality** `Σw = 1`,
neither of which fits the `b ≥ 0` single-phase form. Hence the self-contained **two-phase
double-precision** `linprog` exported here (dense simplex, Bland's rule for guaranteed
termination), which also seeds the active-set QP's feasible point and solves the frontier's
max-return end.

## API

### Section D — risk / portfolio

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `corr2cov` | `auto corr2cov(std::span<const std::vector<double>> corr, std::span<const double> stddevs) -> Result<std::vector<std::vector<double>>>` | Correlation → covariance, `Σ_ij = R_ij · s_i · s_j`. `stddevs` must be non-negative and match the (square) correlation dimension. |
| `cov2corr` | `auto cov2corr(std::span<const std::vector<double>> cov) -> Result<std::vector<std::vector<double>>>` | Covariance → correlation, `R_ij = Σ_ij / (s_i s_j)`, `s_i = √Σ_ii`. A non-positive diagonal (zero/negative variance) → `domain_error`. |
| `lower_partial_moment` | `auto lower_partial_moment(std::span<const double> returns, double order, double threshold) -> Result<double>` | `LPM_order(τ) = (1/N) Σ_k max(τ − r_k, 0)^order`. Only strictly-below-threshold observations contribute, so `LPM_0` is the shortfall probability. Empty series / negative `order` → `domain_error`. |
| `downside_deviation` | `auto downside_deviation(std::span<const double> returns, double mar) -> Result<double>` | `√LPM_2(mar)` (population, divisor `N`). |
| `ew_mean` | `auto ew_mean(std::span<const double> x, double lambda) -> Result<double>` | Exponentially-weighted mean with decay `λ ∈ (0,1]`: the most-recent element `x[N−1]` has weight 1, `x[t]` has weight `λ^(N−1−t)`, normalised by their sum. `λ == 1` recovers the ordinary equal-weight mean. |
| `ew_cov` | `auto ew_cov(std::span<const double> x, std::span<const double> y, double lambda) -> Result<double>` | Exponentially-weighted covariance under the same weighting, with the **unbiased reliability-weights** normalisation (divisor `W1 − W2/W1`, `W1 = Σw`, `W2 = Σw²`), so `λ == 1` recovers the ordinary `(n−1)` sample covariance. Needs `n ≥ 2`, matched lengths. |
| `ew_covariance_matrix` | `auto ew_covariance_matrix(std::span<const std::vector<double>> series, double lambda) -> Result<std::vector<std::vector<double>>>` | EW covariance matrix of a set of series (assets in rows, observations in columns). `λ == 1` → the `(n−1)` sample covariance matrix. Ragged / oversized (> 4096) → `domain_error`. |
| `simulate_correlated_returns` | `auto simulate_correlated_returns(std::span<const double> mean, std::span<const std::vector<double>> cov, std::size_t n, std::uint64_t seed) -> Result<std::vector<std::vector<double>>>` | Draw `n` correlated Gaussian vectors `x = mean + L z` (`z ~ N(0,I)`, `Σ = L Lᵀ` the robust lower Cholesky factor), reproducible for a fixed `seed` via the counter-based RNG. Returns an `n × d` matrix (rows are samples, columns assets). A **non-PD** covariance is rejected → `domain_error`. |
| `linprog` | `auto linprog(std::span<const double> c, std::span<const std::vector<double>> A_le, std::span<const double> b_le, std::span<const std::vector<double>> A_eq, std::span<const double> b_eq) -> Result<LinProgResult>` | Minimise `c·x` s.t. `A_le x ≤ b_le`, `A_eq x = b_eq`, `x ≥ 0`. Self-contained two-phase dense simplex, Bland's rule. Reports `optimal` / `infeasible` / `unbounded` in the result status. |
| `constrained_min_variance` | `auto constrained_min_variance(std::span<const std::vector<double>> cov, std::span<const double> upper, std::span<const double> lower = {}) -> Result<std::vector<double>>` | Global minimum-variance portfolio subject to the box `lower ≤ w ≤ upper` (default `lower` 0) and full investment `Σw = 1`, via the active-set QP. `upper` sizes the problem (`== cov` dimension). |
| `constrained_efficient_portfolio` | `auto constrained_efficient_portfolio(std::span<const std::vector<double>> cov, std::span<const double> mean_returns, double target_return, std::span<const double> upper, std::span<const double> lower = {}) -> Result<FrontierPoint>` | Minimum-variance box-constrained portfolio achieving `target_return` (adds the equality `μ·w = target_return`). Infeasible target under the box → `domain_error`. |
| `constrained_frontier` | `auto constrained_frontier(std::span<const std::vector<double>> cov, std::span<const double> mean_returns, int points, std::span<const double> upper, std::span<const double> lower = {}) -> Result<std::vector<FrontierPoint>>` | A box-constrained frontier of `points` (≥ 2) portfolios, sampled at target returns evenly spaced between the box-constrained global-min-variance return and the maximum return attainable under the box (the latter found by an LP). |
| `cvar_optimal_weights` | `auto cvar_optimal_weights(std::span<const std::vector<double>> scenarios, double beta, std::span<const double> upper = {}) -> Result<CVaRResult>` | CVaR-minimising portfolio via the Rockafellar-Uryasev LP. Given `scenarios` (`S` rows, each an `N`-vector of per-asset returns) and confidence `beta ∈ [0,1)`, minimises the `β`-CVaR of loss `L_k = −(w·r_k)` over fully-invested long-only weights (optionally capped by `upper`). Returns weights, the minimised CVaR (RU objective), and the optimal VaR (RU `α`). |

### Section E — French depreciation (AMORLINC / AMORDEGRC)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `amordegrc_coefficient` | `auto amordegrc_coefficient(double asset_life) -> double` | The life-keyed AMORDEGRC acceleration coefficient (not a `Result` — a pure lookup): `life < 3 → 1.0`; `3 ≤ life < 5 → 1.5` (covers 3- and 4-year lives, matching the `< 5` boundary); `5 ≤ life ≤ 6 → 2.0`; `life > 6 → 2.5`. |
| `amorlinc` | `auto amorlinc(double cost, double salvage, std::int64_t period, double rate, double first_period_fraction) -> Result<double>` | French linear (prorated straight-line) depreciation for 0-based `period`. `first_period_fraction ∈ (0,1]` is the fraction of a full year held in period 0. Each full period depreciates `cost·rate`; period 0 is prorated and the final period takes the remainder so the total is exactly `cost − salvage`. |
| `amordegrc` | `auto amordegrc(double cost, double salvage, std::int64_t period, double rate, double first_period_fraction) -> Result<double>` | French degressive depreciation for 0-based `period`. Charge is `round-half-up(accelerated_rate · book)` with `accelerated_rate = coefficient·rate`, switching to straight-line over the remaining life once that deducts more, never dropping book value below `salvage`. Period 0 is prorated by `first_period_fraction`. |

### Section F — continuous compounding & annuity variants

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `effective_continuous` | `auto effective_continuous(double nominal_rate) -> Result<double>` | Effective annual rate under continuous compounding (force of interest `r`): `e^r − 1` (via `expm1`). Non-finite input → `domain_error`; overflow → `overflow`. |
| `nominal_continuous` | `auto nominal_continuous(double effective_rate) -> Result<double>` | Inverse: the continuously-compounded nominal rate giving an effective annual rate, `ln(1 + effective)`. `effective ≤ −1` → `domain_error` (log of a non-positive). |
| `amortize` | `auto amortize(double rate, std::int64_t nper, double principal) -> Result<AmortSchedule>` | Full level-payment amortisation schedule (ordinary annuity, end-of-period). `interest[t]`/`principal[t]`/`balance[t]` are the split and the balance **after** payment `t` (`balance.back() ≈ 0`; the last payment absorbs residual round-off). |
| `pay_per` | `auto pay_per(double rate, std::int64_t nper, double pv) -> Result<double>` | Level periodic payment fully amortising `pv` over `nper` periods (ordinary annuity, fv 0): `pv·r / (1 − (1+r)^−n)`, or `pv/n` when `r == 0`. |
| `pay_odd` | `auto pay_odd(double rate, double odd_period_fraction, double pv, bool simple = true) -> Result<double>` | Interest accruing over an irregular first period of length `odd_period_fraction` (in periods): simple `pv·r·odd` (default) or compounded `pv·((1+r)^odd − 1)`. |
| `pay_uni` | `auto pay_uni(double rate, std::int64_t nper, double pv, double odd_period_fraction, bool simple = true) -> Result<double>` | Uniform (level) payment with the odd-period interest **capitalised** into principal: `pay_per(rate, nper, pv + pay_odd(...))`. `odd_period_fraction == 0` recovers `pay_per(rate, nper, pv)`. |

### Result structs

```cpp
// One point on a (constrained) efficient frontier.
struct FrontierPoint {
    std::vector<double> weights;
    double risk{0.0};   // portfolio standard deviation
    double ret{0.0};    // portfolio expected return
};

// linprog outcome.
enum class LinProgStatus : std::uint8_t { optimal, infeasible, unbounded };
struct LinProgResult {
    LinProgStatus status{LinProgStatus::optimal};
    double objective{0.0};
    std::vector<double> x{};
};

// cvar_optimal_weights outcome.
struct CVaRResult {
    std::vector<double> weights;
    double cvar{0.0};   // minimised beta-CVaR (the RU objective value)
    double var{0.0};    // optimal VaR (the RU alpha)
};

// amortize schedule (balance[t] is the remaining balance AFTER payment t).
struct AmortSchedule {
    double payment{0.0};
    std::vector<double> interest{};
    std::vector<double> principal{};
    std::vector<double> balance{};
};
```

## Error model

Every fallible routine returns `Result<T>`; nothing throws, nothing returns a `NaN`/`inf`
dressed as a value. A singular KKT system, a non-positive-definite covariance, an infeasible
constrained program, or an exhausted iteration budget rides the railway as `domain_error` /
`not_converged`. DoS-sized inputs are refused before allocation.

| Condition | Error |
| :--- | :--- |
| Any non-finite entry in a matrix, series, or scalar input; ragged / non-square / mismatched-length matrix; empty series where one is required | `MathError::domain_error` |
| `cov2corr`: a non-positive diagonal (zero/negative variance) | `MathError::domain_error` |
| `lower_partial_moment`: empty series or `order < 0`; `ew_mean`/`ew_cov`/`ew_covariance_matrix`: `λ ∉ (0,1]` (and `ew_cov` needs `n ≥ 2`) | `MathError::domain_error` |
| `simulate_correlated_returns`: **non-positive-definite** `cov` (fails the robust Cholesky), dimension mismatch, `n == 0` | `MathError::domain_error` |
| Matrix dimension > 4096 (`corr2cov`, `cov2corr`, `ew_covariance_matrix`, `simulate_correlated_returns`) | `MathError::domain_error` |
| `simulate_correlated_returns`: `n > 100000` | `MathError::domain_error` |
| `linprog`: `c` empty, or variables + constraints > 8192 (`kMaxLpDim`) | `MathError::domain_error` |
| Constrained QP: infeasible box/target (e.g. `Σ upper < 1`), or QP asset dimension > 512 (`kMaxQpDim`) | `MathError::domain_error` |
| Constrained QP: singular/indefinite KKT solve, or the active-set / simplex iteration budget exhausted | `MathError::not_converged` |
| `cvar_optimal_weights`: empty scenarios, `S > 1024` (`kMaxScenarios`), `N == 0`, `N > 128` (`kMaxCvarAssets`), `beta ∉ [0,1)`, ragged scenarios, or `upper` size ≠ `N` | `MathError::domain_error` |
| `cvar_optimal_weights`: the LP reports unbounded (a malformed instance — RU is bounded below) or a non-finite result | `MathError::not_converged` |
| `amorlinc` / `amordegrc`: `rate ≤ 0`, `cost ≤ 0`, `salvage < 0` or `> cost`, `period < 0` or `> 100000` (`kMaxPeriods`), `first_period_fraction ∉ (0,1]` | `MathError::domain_error` |
| `nominal_continuous`: `effective ≤ −1`; `pay_per`/`pay_odd`/`pay_uni`/`amortize`: `rate ≤ −1`; `pay_per`/`amortize`: `nper < 1` or `> 100000`, `principal`/`pv ≤ 0` (`amortize`); `pay_odd`: `odd_period_fraction < 0` | `MathError::domain_error` |
| A finite computation overflowing to a non-finite result (`lower_partial_moment`, `ew_mean`, `ew_cov`, `effective_continuous`, `pay_per`, `pay_odd`) | `MathError::overflow` |
| A zero normalising weight (`ew_mean`) or zero discount denominator (`ew_cov`, `pay_per`) | `MathError::division_by_zero` |

No result is ever claimed exact — all outputs are sample estimates over `double`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.riskextra;
using namespace nimblecas;
using namespace nimblecas::riskextra;

// (D) corr <-> cov round-trip. R = [[1, 0.5],[0.5, 1]], sd = (0.2, 0.3).
const std::vector<std::vector<double>> R{{1.0, 0.5}, {0.5, 1.0}};
const std::array<double, 2> sd{0.2, 0.3};
auto cov = corr2cov(R, sd).value();          // [[0.04, 0.03],[0.03, 0.09]]
cov2corr(cov).value();                        // recovers R

// Downside risk. Only strictly-below-threshold returns contribute.
const std::array<double, 5> r5{-0.05, -0.02, 0.0, 0.01, 0.03};
lower_partial_moment(r5, 0.0, 0.0).value();   // 0.4  (shortfall probability: 2 of 5)
downside_deviation(r5, 0.0).value();          // sqrt(LPM_2(0))

// Exponentially-weighted moments. lambda == 1 degenerates to the plain estimators.
const std::array<double, 4> a{0.01, 0.02, 0.03, 0.04};
const std::array<double, 4> b{0.02, 0.01, 0.04, 0.03};
ew_mean(a, 1.0).value();                       // 0.025  (equal-weight mean)
ew_cov(a, b, 1.0).value();                     // == analytics::covariance(a, b) (n-1)
ew_cov(a, b, 0.94).value();                    // RiskMetrics-style decayed covariance

// Reproducible correlated draws through the robust Cholesky. Same seed -> same matrix.
const std::array<double, 2> mu{0.01, 0.02};
auto sims = simulate_correlated_returns(mu, cov, 1000, /*seed*/42ULL).value();
sims.size();                                   // 1000 rows (samples), each of 2 assets
// A non-PD covariance is refused, not factored into garbage.
const std::vector<std::vector<double>> bad{{0.0, 0.0}, {0.0, 0.0}};
simulate_correlated_returns(mu, bad, 10, 1ULL).has_value();  // false (domain_error)

// (D) Box-constrained global minimum variance: 0 <= w_i <= 0.6, fully invested.
const std::array<double, 2> ub{0.6, 0.6};
auto w = constrained_min_variance(cov, ub).value();
w[0] + w[1];                                   // 1  (fully invested, each <= 0.6)

// (D) CVaR-minimising weights via the Rockafellar-Uryasev LP.
const std::vector<std::vector<double>> scen{
    {-0.02, 0.01}, {0.03, -0.01}, {-0.05, 0.02}, {0.01, 0.00}};
auto cv = cvar_optimal_weights(scen, /*beta*/0.95).value();
cv.cvar;                                       // minimised 95%-CVaR of the loss
cv.var;                                        // the optimal VaR (RU alpha)

// (D) Self-contained LP. minimise -x0 - x1 s.t. x0 + x1 <= 1, x >= 0.
const std::array<double, 2> c{-1.0, -1.0};
const std::vector<std::vector<double>> A_le{{1.0, 1.0}};
const std::array<double, 1> b_le{1.0};
auto lp = linprog(c, A_le, b_le, {}, {}).value();
lp.status;                                     // LinProgStatus::optimal
lp.objective;                                  // -1

// (E) French depreciation. cost 2400, salvage 300, rate 0.15, first-period fraction 0.7.
amordegrc_coefficient(1.0 / 0.15);             // 2.5  (life ~ 6.67 years > 6 -> 2.5)
amorlinc(2400.0, 300.0, /*period*/0, 0.15, 0.7).value();   // prorated first charge
amordegrc(2400.0, 300.0, /*period*/1, 0.15, 0.7).value();  // accelerated degressive charge

// (F) Continuous compounding and annuity variants.
effective_continuous(0.05).value();            // e^0.05 - 1
nominal_continuous(effective_continuous(0.05).value()).value();  // ~ 0.05 (inverse)
pay_per(0.01, 12, 1000.0).value();             // level monthly payment on a 1000 loan
auto sch = amortize(0.01, 12, 1000.0).value();
sch.balance.back();                            // ~ 0  (fully amortised)
pay_uni(0.01, 12, 1000.0, /*odd*/0.0).value(); // == pay_per(0.01, 12, 1000.0)
```

## See also

- [`nimblecas.analytics`](analytics.md) — the closed-form mean-variance and financial-ratio
  sibling whose gaps this module fills (constrained optimisation, CVaR, EW moments).
- [`nimblecas.portfolio`](portfolio.md) — the portfolio-construction neighbour.
- [Documentation hub](../Index.md)
