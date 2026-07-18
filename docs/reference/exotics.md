# `nimblecas.exotics` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/exotics/exotics.cppm`

The exotic-derivatives companion to [`pricing`](pricing.md): lattices, closed forms,
a finite-difference PDE, and Monte Carlo for options that a plain Black-Scholes call
cannot express. It covers the Cox-Ross-Rubinstein binomial tree (European / American,
plus a discrete-cash-dividend variant), the Reiner-Rubinstein analytic single barriers
(all eight types with an optional rebate), the Goldman-Sosin-Gatto floating-strike
lookback, a Crank-Nicolson PDE pricer with Rannacher startup and projected-SOR American
exercise, the Margrabe exchange and Kirk spread approximations, a Drezner bivariate-normal
CDF, the Geske compound option, a simple chooser, and a correlated basket Monte Carlo.

## Honesty boundary

This module is **NUMERICAL / STATISTICAL by nature** — an option price is a limit (of a
lattice as steps → ∞, of a PDE grid as `h` → 0, of a sample mean as paths → ∞), and every
quantity rides on transcendental `exp` / `log` / `erf`. **Nothing here claims exactness.**

- **Closed forms** (barrier, lookback, Margrabe, Kirk, Geske, chooser) are
  correctly-rounded `double` formulas returning `Result<double>`.
- **Lattice / PDE prices** carry discretization error that shrinks with steps / grid size.
- **Monte Carlo returns the estimate AND its standard error** (`pricing::McResult`), and is
  **bit-reproducible and partition-independent**: every draw is a pure function of its global
  index via [`rng`](rng.md)'s `counter_u64`.

A NaN or infinity is **never** returned as a value. Degenerate inputs (`S <= 0`, `σ <= 0`,
`T <= 0`, a barrier on the wrong side of spot, a non-positive-definite correlation) return
`MathError::domain_error`; a non-convergent iteration (PSOR, a critical-price root) returns
`MathError::not_converged`. Every grid / step / path size is bounded against a DoS blow-up.
All failure rides the railway (`Result<T>` / `MathError`); nothing throws.

```cpp
import nimblecas.exotics;
```

Depends on [`core`](core.md) (`Result` / `MathError`), [`pricing`](pricing.md) — it reuses
that module's `OptionSpec` / `OptionType` / `Exercise` / `McResult` and its
correctly-rounded standard-normal helpers (`norm_cdf` / `inverse_norm_cdf`) and
`black_scholes_price` — and [`rng`](rng.md) (`splitmix64`, `counter_u64`, `uniform_unit` —
the counter-based reproducible draw substrate) for the Monte Carlo pieces. Everything lives
in namespace `nimblecas::exotics`; all entry points are free functions, `[[nodiscard]]`.

## Types

| Type | Description |
| :--- | :--- |
| `Barrier` | `enum class : std::uint8_t { down, up }`. Selects the barrier side in `barrier_analytic`. |
| `CashDividend` | A discrete cash dividend: `time` (years from valuation, default `0.0`) and `amount` (default `0.0`). |
| `BasketAsset` | One leg of a basket: `spot` (default `100.0`), `volatility` (`0.2`), `dividend_yield` (`0.0`), `weight` (`1.0`). |

## API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `crr_binomial` | `auto crr_binomial(const OptionSpec& spec, int steps, Exercise exercise) -> Result<double>` | Cox-Ross-Rubinstein binomial lattice with `steps` time steps. `u = e^{σ√dt}`, `d = 1/u`, up-probability `p = (e^{(r−q)dt} − d)/(u − d)`; American exercise takes `max(continuation, intrinsic)` at every node. Converges to Black-Scholes as `steps` grow. |
| `crr_binomial_discrete_div` | `auto crr_binomial_discrete_div(const OptionSpec& spec, int steps, Exercise exercise, std::span<const CashDividend> dividends) -> Result<double>` | CRR binomial with discrete cash dividends via the **escrowed-spot approximation**: the present value of every dividend paid on `(0, T]` is subtracted from spot, and the lattice grown from that reduced spot with the continuous yield forced to zero. Quoted as an approximation. |
| `barrier_analytic` | `auto barrier_analytic(const OptionSpec& spec, double barrier, Barrier side, bool knock_in, double rebate = 0.0) -> Result<double>` | Analytic single-barrier price under continuous monitoring (Reiner-Rubinstein / Merton). `side` selects down/up, `knock_in` selects in/out, and the call/put flavour comes from `spec.type` — together the eight standard types. `rebate` (default `0`) is paid on knock-out at the knock time, or on knock-in at expiry if never triggered. In/out parity `knock_in + knock_out == vanilla` holds exactly for `rebate == 0`. Cost of carry `b = r − q`. Result clamped to `≥ 0`. |
| `lookback_price` | `auto lookback_price(const OptionSpec& spec, double running_extremum = 0.0) -> Result<double>` | Goldman-Sosin-Gatto (1979) floating-strike lookback closed form: a call pays `S_T − min` over the life, a put pays `max − S_T`. `running_extremum` is the extremum observed so far (running minimum for a call, running maximum for a put); pass `0` to start a fresh lookback at the current spot. Cost of carry `b = r − q`. The `b → 0` limit of the `σ²/(2b)` term is **deferred** — `|b| < 1e-8` returns `domain_error`. Result clamped to `≥ 0`. |
| `fd_pde_price` | `auto fd_pde_price(const OptionSpec& spec, int n_space, int n_time, Exercise exercise) -> Result<double>` | Vanilla European / American price by solving the Black-Scholes PDE on a uniform spot grid `[0, S_max]` with `n_space` intervals and `n_time` time steps. **Crank-Nicolson** in time with a **2-step Rannacher startup** (the first two steps fully implicit / backward Euler, damping the payoff-kink oscillation). European steps solved by the **Thomas** tridiagonal algorithm; American exercise enforced by **projected SOR** (PSOR, `ω = 1.5`) against intrinsic. The price at the requested spot is linearly interpolated between grid nodes. `S_max = max(S, K)·e^{5σ√T}`. |
| `margrabe_exchange` | `auto margrabe_exchange(double spot1, double spot2, double div1, double div2, double vol1, double vol2, double rho, double time) -> Result<double>` | Margrabe (1978) exchange option, payoff `max(S1 − S2, 0)`. The risk-free rate cancels; only the yields `q1`, `q2` and the effective volatility `√(σ1² + σ2² − 2ρσ1σ2)` enter. A degenerate case (perfectly correlated equal vols, or zero vols) collapses to discounted intrinsic. |
| `kirk_spread` | `auto kirk_spread(double spot1, double spot2, double strike, double rate, double div1, double div2, double vol1, double vol2, double rho, double time) -> Result<double>` | Kirk (1995) spread-option approximation, payoff `max(S1 − S2 − K, 0)`. Treats `(S2-forward + K)` as a single lognormal and applies a Margrabe-style formula; exact at `K == 0` (reduces to Margrabe). Result clamped to `≥ 0`. |
| `bivariate_normal_cdf` | `auto bivariate_normal_cdf(double a, double b, double rho) -> double` | Standardised bivariate-normal CDF `P(X ≤ a, Y ≤ b)` with correlation `rho`, via the Drezner reduction evaluated by composite Simpson (2000 steps). Exact special cases: `rho == 0` → `Φ(a)Φ(b)`; `rho ≥ 1` → `Φ(min(a,b))`; `rho ≤ −1` → `max(Φ(a) + Φ(b) − 1, 0)`. **Pure — no error channel**; the result is clamped to `[0, 1]`. |
| `geske_compound` | `auto geske_compound(const OptionSpec& spec, double strike1, double strike2, double t1) -> Result<double>` | Geske (1979) call-on-call compound option: the right, at `t1` for premium `strike1`, to buy a Black-Scholes call struck at `strike2` expiring at `spec.time_to_expiry` (`> t1`). Uses `bivariate_normal_cdf` and a bracketed **bisection** (200 halvings) for the critical spot at `t1`. As `strike1 → 0` it collapses to the underlying call. Result clamped to `≥ 0`. |
| `chooser_price` | `auto chooser_price(const OptionSpec& spec, double t1) -> Result<double>` | Simple chooser (Rubinstein 1991): at `t1` the holder chooses call or put, both struck at `spec.strike` and expiring at `spec.time_to_expiry`. Value lies between `max(call, put)` and `call + put`. Result clamped to `≥ 0`. |
| `basket_mc` | `auto basket_mc(std::span<const BasketAsset> assets, std::span<const double> correlation, double strike, double rate, double time, OptionType type, std::uint64_t paths, std::uint64_t seed) -> Result<McResult>` | Correlated basket option by reproducible Monte Carlo. Payoff `max(Σᵢ wᵢ Sᵢ(T) − K, 0)` for a call (or `K − basket` for a put). `correlation` is the row-major `n×n` matrix; its **Cholesky** factor drives the correlated terminal draws. **Antithetic** variates (`±z` through the same factor) halve the variance. Every normal is a pure function of its global index, so the estimate is bit-reproducible under a fixed seed and partition-independent. Returns the estimate and its standard error. |

### Result structs

```cpp
enum class Barrier : std::uint8_t { down, up };

struct CashDividend {
    double time{0.0};    // years from valuation
    double amount{0.0};  // cash paid at `time`
};

struct BasketAsset {
    double spot{100.0};
    double volatility{0.2};
    double dividend_yield{0.0};
    double weight{1.0};
};
```

`crr_binomial` and `fd_pde_price` return `Result<double>` from
`nimblecas::pricing::OptionSpec`; `basket_mc` returns `Result<McResult>` — the
`{ price, std_error, paths }` estimate reused from [`pricing`](pricing.md).

## Bounds

| Quantity | Bound | Constant |
| :--- | :--- | :--- |
| `crr_binomial` / `crr_binomial_discrete_div` `steps` | `[1, 100000]` | `kMaxLatticeSteps = 100000` |
| `fd_pde_price` `n_space` | `[4, 20000]` | `kMaxGrid = 20000` |
| `fd_pde_price` `n_time` | `[1, 20000]` | `kMaxGrid = 20000` |
| `fd_pde_price` `n_space · n_time` | `≤ 2·10⁸` | `kMaxGridCells = 200000000` |
| `basket_mc` asset count `n` | `[1, 256]` | `kMaxBasketAssets = 256` |
| `basket_mc` `paths` | `[1, 10⁹]` with `n · paths ≤ 10⁹` | `kMaxPaths = 1000000000` |

## Error model

| Condition | Error |
| :--- | :--- |
| `crr_binomial`: `steps < 1` or `> 100000`; degenerate spec (`S`/`K`/`T`/`σ ≤ 0`) | `MathError::domain_error` |
| `crr_binomial`: too few steps for these parameters (`p` outside `[0, 1]`) | `MathError::not_converged` |
| `crr_binomial_discrete_div`: a negative dividend `time`; escrowed spot `≤ 0` (dividends exceed spot) | `MathError::domain_error` |
| `barrier_analytic`: degenerate spec (`S`/`K`/`H`/`σ`/`T ≤ 0`); spot on the wrong (dead) side of the barrier (`S ≤ H` for a down-barrier, `S ≥ H` for an up-barrier) | `MathError::domain_error` |
| `lookback_price`: degenerate spec (`S`/`σ`/`T ≤ 0`); `|b| < 1e-8` (the `b → 0` limit is deferred); running minimum above spot (call) or running maximum below spot (put) | `MathError::domain_error` |
| `fd_pde_price`: `n_space ∉ [4, 20000]`, `n_time ∉ [1, 20000]`, or `n_space·n_time > 2·10⁸`; degenerate spec (`S`/`K`/`T`/`σ ≤ 0`) | `MathError::domain_error` |
| `fd_pde_price`: PSOR fails to converge within 10000 sweeps; a zero Thomas pivot | `MathError::not_converged` |
| `margrabe_exchange` / `kirk_spread`: `spot1 ≤ 0`, `spot2 ≤ 0`, `vol1 < 0`, `vol2 < 0`, `time ≤ 0`, or `|rho| > 1` | `MathError::domain_error` |
| `kirk_spread`: `S2-forward + K ≤ 0` | `MathError::domain_error` |
| `geske_compound`: `S`/`σ`/`strike2`/`t1 ≤ 0`, `strike1 < 0`, or `t1 ≥ spec.time_to_expiry`; degenerate inner call | `MathError::domain_error` |
| `geske_compound`: the critical-spot bracket / root fails | `MathError::not_converged` |
| `chooser_price`: `S`/`K`/`σ ≤ 0`, `t1 ≤ 0`, or `t1 > spec.time_to_expiry` | `MathError::domain_error` |
| `basket_mc`: `n == 0` or `> 256`; `correlation.size() ≠ n·n`; `paths == 0`, `> 10⁹`, or `n·paths > 10⁹`; `time ≤ 0`; a degenerate leg (`spot ≤ 0` or `volatility < 0`); a non-positive-definite correlation (no real Cholesky factor) | `MathError::domain_error` |

`bivariate_normal_cdf` is **total** — it returns a `double` clamped to `[0, 1]` with no
error channel. No result is ever claimed exact; Monte Carlo results always carry
`std_error`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.exotics;
using namespace nimblecas;
using namespace nimblecas::exotics;

// The canonical ATM one-year option: S=K=100, r=5%, sigma=20%, T=1.
const auto call = OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
                      .with_volatility(0.2).with_expiry(1.0);
const auto put  = call.with_type(OptionType::put);

// (1) CRR binomial converges to Black-Scholes; American call (no dividend) == European.
crr_binomial(call, 500, Exercise::european).value();   // ≈ BS
crr_binomial(call, 500, Exercise::american).value();   // ≈ BS (no early exercise value)

// Discrete cash dividend via the escrowed-spot approximation.
const std::array<CashDividend, 1> divs{{ {0.5, 2.0} }};   // 2.0 paid at T=0.5
crr_binomial_discrete_div(call, 500, Exercise::american, divs).value();

// (2) Reiner-Rubinstein single barrier. In + out == vanilla (rebate 0).
const double din = barrier_analytic(call, 90.0, Barrier::down, true).value();   // down-and-in
const double dout = barrier_analytic(call, 90.0, Barrier::down, false).value(); // down-and-out
// din + dout ≈ black_scholes_price(call).value()

// A barrier on the wrong side of spot is an honest error (already knocked).
barrier_analytic(call, 110.0, Barrier::down, false).error();  // MathError::domain_error

// (3) Goldman-Sosin-Gatto floating-strike lookback (fresh at current spot).
lookback_price(call, 0.0).value();
lookback_price(call.with_dividend(0.05), 0.0).error();  // b == 0 => domain_error (deferred)

// (4) Crank-Nicolson PDE with Rannacher startup; American via PSOR.
fd_pde_price(call, 400, 400, Exercise::european).value();  // ≈ BS
fd_pde_price(put,  400, 400, Exercise::american).value();  // American put >= European

// (5) Margrabe exchange and Kirk spread.
margrabe_exchange(100, 100, 0.0, 0.0, 0.2, 0.3, 0.5, 1.0).value();
kirk_spread(100, 90, 5.0, 0.05, 0.0, 0.0, 0.2, 0.3, 0.5, 1.0).value();

// (6) Bivariate-normal CDF (total, no error channel).
bivariate_normal_cdf(0.0, 0.0, 0.0);   // 0.25  (independent halves)

// (7) Geske compound (call-on-call) and chooser.
geske_compound(call, 3.0, 100.0, 0.5).value();  // right at t1=0.5 to buy the call
chooser_price(call, 0.5).value();               // choose call/put at t1=0.5

// (8) Correlated basket Monte Carlo — reproducible, antithetic.
const std::array<BasketAsset, 2> assets{{
    {100.0, 0.2, 0.0, 0.5}, {100.0, 0.3, 0.0, 0.5} }};
const std::array<double, 4> corr{{ 1.0, 0.5, 0.5, 1.0 }};  // row-major 2x2
const auto b = basket_mc(assets, corr, 100.0, 0.05, 1.0,
                         OptionType::call, 100000, 12345).value();
// b.price with b.std_error; a second call with the same seed reproduces b.price bit-for-bit.
```

## See also

- [`nimblecas.pricing`](pricing.md) — the vanilla Black-Scholes-Merton oracle, trinomial
  trees, and Monte Carlo whose `OptionSpec` / `Exercise` / `McResult` and standard-normal
  helpers this module reuses.
- [`nimblecas.rng`](rng.md) — the counter-based RNG (`counter_u64`, `splitmix64`,
  `uniform_unit`) that makes the basket Monte Carlo reproducible and partition-independent.
- [`nimblecas.core`](core.md) — the `Result` / `MathError` railway every entry point rides.
- [Documentation hub](../Index.md)
