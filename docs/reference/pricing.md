# `nimblecas.pricing` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/pricing/pricing.cppm`

Derivatives pricing and instrument valuation: the Black-Scholes-Merton closed
form with a full Greek set (the analytic oracle), higher-order "exotic" Greeks,
Kamrad-Ritchken trinomial trees for European / American / Bermudan exercise,
reproducible Monte Carlo for European and Asian payoffs (with the geometric
average as a control variate), Longstaff-Schwartz least-squares Monte Carlo for
American exercise, Black-76, digital and barrier options, a composable
`Portfolio`, and probability / payoff / price-sweep plotting through
[`svgplot`](svgplot.md).

## Honesty boundary

This module is **NUMERICAL / STATISTICAL by nature** — an option price is a limit
(of a lattice as steps → ∞, of a sample mean as paths → ∞), and the very first
quantity a tree computes, `exp(σ·√dt)`, is transcendental. **Nothing here claims
exactness.** Concretely:

- **Closed forms** (Black-Scholes, geometric Asian, Black-76, digitals) are
  correctly-rounded `double` formulas, checked against analytic oracles and
  structural invariants (put-call parity, `call == AON − K·CON`) rather than lone
  magic numbers.
- **Lattice prices** carry `O(dt)` discretization error that shrinks with `steps`;
  the trinomial tree converges to Black-Scholes for European exercise and honours
  `American >= European`. Parameters too coarse for a valid (non-negative)
  probability triple return `MathError::not_converged` rather than misleading.
- **Monte Carlo returns the estimate AND its standard error** (`McResult`), and is
  **bit-reproducible and partition-independent by construction**: every draw is a
  pure function of its global index via [`rng`](rng.md)'s `counter_u64`, inheriting
  the design guarantee documented in [`montecarlo`](montecarlo.md) — no time or
  entropy seeding anywhere, so equal seeds reproduce equal results bit-for-bit and
  any partition of the path range reproduces the serial mean. The `bits → N(0,1)`
  transform (`normal_fill`) and the European GBM `exp` are **dynamically dispatched**
  (AVX-512 → scalar): the AVX-512 inverse-normal evaluates Acklam's branchless central
  rational for the ~95 % bulk and scalar-fixes the ~5 % log/sqrt tail lanes, **bit-identical
  to the per-index scalar path** — a `perf`-gated speedup (all MC pricers ~1.3–1.9× faster on
  a Skylake-SP Xeon) that changes no result. Off AVX-512, the scalar path is the identical
  fallback.

All failure rides the railway (`Result<T>` / `MathError`); nothing throws.

```cpp
import nimblecas.pricing;
```

Depends on [`core`](core.md), [`rng`](rng.md) (`splitmix64`, `counter_u64`,
`uniform_unit` — the reproducible draw substrate), and [`svgplot`](svgplot.md)
(`PlotOptions`, `plot_function` for the diagrams). Everything lives in namespace
`nimblecas::pricing`.

## Types

| Type | Description |
| :--- | :--- |
| `OptionType` | `enum class { call, put }`. |
| `Exercise` | `enum class { european, american, bermudan }`. |
| `OptionSpec` | Immutable option contract + market state, built fluently via a `with_*` chain. Fields: `spot` (S, default 100), `strike` (K, 100), `rate` (r, continuously compounded, 0), `dividend_yield` (q, 0), `volatility` (σ annualised, 0.2), `time_to_expiry` (T years, 1.0), `type` (call). `payoff(s)` returns the intrinsic. Every `with_spot/strike/rate/dividend/volatility/expiry/type` returns a modified copy. |
| `Greeks` | `price`, `delta`, `gamma`, `vega`, `theta` (per-year), `rho` (all per unit change; divide vega/rho by 100 for per-percent). |
| `ExtendedGreeks` | `vanna`, `charm`, `vomma` (= volga), `veta`, `speed`, `zomma`, `color`, `lambda` (elasticity), `dual_delta`, `dual_gamma`, `epsilon` (= psi, ∂price/∂dividend), `vera` (∂rho/∂vol), `ultima` (∂vomma/∂vol). |
| `McResult` | A Monte Carlo estimate: `price`, `std_error`, `paths`; `confidence_half_width()` returns `1.96 · std_error`. |
| `Position` | A signed `quantity` of an `OptionSpec`. |
| `Portfolio` | A composable bag of positions valued and risk-aggregated as a unit. |

## Standard-normal helpers

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `norm_pdf` | `auto norm_pdf(double x) -> double` | Standard-normal density. |
| `norm_cdf` | `auto norm_cdf(double x) -> double` | Standard-normal CDF via the C-library `erfc` (accurate into the tails). |
| `inverse_norm_cdf` | `auto inverse_norm_cdf(double p) -> Result<double>` | Inverse CDF (Acklam + one Halley refinement). `p` outside `(0, 1)` → `domain_error`. |

## Black-Scholes-Merton (European closed form)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `black_scholes_price` | `auto black_scholes_price(const OptionSpec& spec) -> Result<double>` | The BSM price. Degenerate `T == 0` / `σ == 0` collapse to discounted intrinsic. |
| `black_scholes_greeks` | `auto black_scholes_greeks(const OptionSpec& spec) -> Result<Greeks>` | Price + first- and second-order Greeks. `S <= 0`, `K <= 0`, `T < 0`, or `σ < 0` → `domain_error`. |
| `black_scholes_extended_greeks` | `auto black_scholes_extended_greeks(const OptionSpec& spec) -> Result<ExtendedGreeks>` | The full higher-order set; time-derivatives (charm, color, veta) via central differences of the analytic Greeks. Requires strictly positive `T` and `σ` → else `domain_error`. |
| `implied_volatility` | `auto implied_volatility(const OptionSpec& spec, double market_price) -> Result<double>` | Bracketed root of `BS(vol) = market_price` on `(0, 5]`. `market_price <= 0` → `domain_error`; no bracket → `not_converged`. |
| `option_pnl_at_expiry` | `auto option_pnl_at_expiry(const OptionSpec& spec, double premium, double s_T) -> double` | Realised long-option P&L at `s_T`, net of `premium` (total). |
| `option_mark_to_market_pnl` | `auto option_mark_to_market_pnl(const OptionSpec& spec, double premium) -> Result<double>` | Unrealised P&L = current BS value − premium. |

## Black-76, digitals, and barriers

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `black76_price` | `auto black76_price(bool is_call, double forward, double strike, double rate, double volatility, double time) -> Result<double>` | Option on a forward/futures price `F`: `e^{−rT}[F·N(d1) − K·N(d2)]` for a call. `forward <= 0`, `strike <= 0`, `time < 0`, or `volatility < 0` → `domain_error`. |
| `digital_cash_or_nothing` | `auto digital_cash_or_nothing(const OptionSpec& spec, double cash) -> Result<double>` | Pays `cash` if in the money. Strict-positive `S`/`K`/`T`/`σ` required. |
| `digital_asset_or_nothing` | `auto digital_asset_or_nothing(const OptionSpec& spec) -> Result<double>` | Pays the underlying if in the money. |
| `barrier_option_mc` | `auto barrier_option_mc(const OptionSpec& spec, double barrier, bool knock_in, std::uint64_t paths, int steps, std::uint64_t seed) -> Result<McResult>` | Single-barrier option by reproducible MC with discrete monitoring at `steps` dates. **Statistical**: carries MC error and discrete-monitoring bias (documented). In/out parity `in + out == vanilla` holds up to MC error. `paths == 0`, `steps < 1`, `barrier <= 0`, or a bad spec → `domain_error`. |

## Trinomial lattice (Kamrad-Ritchken)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `trinomial_price` | `auto trinomial_price(const OptionSpec& spec, int steps, Exercise exercise, std::span<const double> exercise_times = {}) -> Result<double>` | European / American use `steps` time steps; for Bermudan, `exercise_times` lists the years at which early exercise is permitted (mapped to the nearest step; an empty list behaves European). `steps < 1` or a bad spec → `domain_error`; a negative probability from too-coarse `steps` → `not_converged`. |

## Monte Carlo (reproducible, partition-independent)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `monte_carlo_european` | `auto monte_carlo_european(const OptionSpec& spec, std::uint64_t paths, std::uint64_t seed) -> Result<McResult>` | European MC with antithetic variates (each index `i` produces the pair `±z`). Bit-reproducible under a fixed seed. `paths == 0` or a bad spec → `domain_error`. |
| `monte_carlo_european_parallel` | `auto monte_carlo_european_parallel(const OptionSpec& spec, std::uint64_t paths, std::uint64_t seed) -> Result<McResult>` | The parallel counterpart: identical model and draw stream, but paths are priced in a **fixed block decomposition** across the fork-join runtime ([`parallel`](parallel.md)) and the block partials combined in index order. **Deterministic** — the result depends only on `(spec, paths, seed)`, never on thread count or scheduling (asserted bit-identical across calls). It equals `monte_carlo_european` to **FP tolerance** (`\|Δprice\| ≈ 1.7e-12`, i.e. non-associativity of the sum, far below the MC standard error), *not* bit-for-bit. A `perf` pass (Xeon Gold 6152) showed European MC is **compute/transcendental-bound** (IPC 2.09, ~25 % in scalar libm `log`+`exp`, RNG core already AVX-512), so it scales near-linearly with cores. Both pricers share `price_euro_block`, whose two GBM exponentials per path are batched through the deterministic [`simd::exp_into`](simd.md) (~1.38× on the serial pricer alone); `mc_bench` then measures the parallel pricer at **~1.02 B paths/s (≈26× the serial)** on that 88-thread box. Same guards as the serial pricer. |
| `geometric_asian_price` | `auto geometric_asian_price(const OptionSpec& spec, int steps) -> Result<double>` | **Closed form** for the discrete geometric-average Asian — the control-variate expectation and a validation oracle. At `steps == 1` it reduces **exactly** to Black-Scholes. `steps < 1` or a bad spec → `domain_error`. |
| `monte_carlo_asian` | `auto monte_carlo_asian(const OptionSpec& spec, std::uint64_t paths, int steps, std::uint64_t seed, bool control_variate = true) -> Result<McResult>` | Arithmetic-average Asian by MC, using the geometric closed form as a control variate (a large variance reduction) when `control_variate`. `paths == 0` / `steps < 1` / bad spec → `domain_error`. |
| `longstaff_schwartz_american` | `auto longstaff_schwartz_american(const OptionSpec& spec, std::uint64_t paths, int steps, std::uint64_t seed) -> Result<McResult>` | American exercise by least-squares MC (polynomial basis `{1, S, S²}`); floored at the immediate-exercise value. `paths < 4`, `steps < 1`, or a bad spec → `domain_error`. |

## `Portfolio` — composable position graph

Built fluently: `Portfolio::create().add(spec, quantity)` or `.with(position)`.
Pricing and Greeks are the linear combination of the legs.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add` / `with` | `auto add(const OptionSpec& spec, double quantity) -> Portfolio&` | Append a signed leg. |
| `legs` | `auto legs() const noexcept -> std::span<const Position>` | The positions. |
| `value` | `auto value() const -> Result<double>` | Aggregate Black-Scholes value. Propagates a leg's `domain_error`. |
| `greeks` | `auto greeks() const -> Result<Greeks>` | Aggregate (quantity-weighted) Greeks. |
| `payoff_at` | `auto payoff_at(double s) const -> double` | Aggregate payoff at expiry across legs (total). |

## Plotting (SVG, via `svgplot`)

Each returns an SVG string. `s_min`/`s_max` bound the horizontal sweep; `samples`
sets resolution; `opt` is a `PlotOptions`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `terminal_density_svg` | `auto terminal_density_svg(const OptionSpec& spec, double s_min, double s_max, int samples, const PlotOptions& opt) -> Result<std::string>` | Risk-neutral lognormal terminal density of `S_T`. Bad spec → `domain_error`. |
| `payoff_diagram_svg` | `auto payoff_diagram_svg(const OptionSpec& spec, double s_min, double s_max, int samples, const PlotOptions& opt, double premium = 0.0) -> Result<std::string>` | Payoff at expiry, optionally net of `premium` for a P&L line. |
| `portfolio_pnl_svg` | `auto portfolio_pnl_svg(const Portfolio& book, double s_min, double s_max, int samples, const PlotOptions& opt) -> Result<std::string>` | Portfolio P&L at expiry, net of the book's current BS value. |
| `price_vs_spot_svg` | `auto price_vs_spot_svg(const OptionSpec& spec, double s_min, double s_max, int samples, const PlotOptions& opt) -> Result<std::string>` | Black-Scholes price as spot sweeps `[s_min, s_max]`. |

## Error model

| Condition | Error |
| :--- | :--- |
| A closed form / tree / MC with a non-physical spec (`S <= 0`, `K <= 0`, `T < 0`, `σ < 0`, or strict-positive where required) | `MathError::domain_error` |
| `monte_carlo_*` / `barrier_option_mc` with `paths == 0` (or `paths < 4` for Longstaff-Schwartz); a lattice/MC with `steps < 1` | `MathError::domain_error` |
| `inverse_norm_cdf` / `implied_volatility` with `p`/`market_price` out of range | `MathError::domain_error` |
| `implied_volatility` when no bracket is found on `(0, 5]` | `MathError::not_converged` |
| `trinomial_price` when `steps` is too coarse for a valid probability triple | `MathError::not_converged` |

`black_scholes_price`, `norm_pdf`, `norm_cdf`, `option_pnl_at_expiry`, and
`Portfolio::payoff_at` are total on physical inputs. **No result is ever claimed
exact**; MC results always carry `std_error`.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.svgplot;
using namespace nimblecas;
using namespace nimblecas::pricing;

// The canonical ATM one-year option: S=K=100, r=5%, sigma=20%, T=1.
const auto call = OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
                      .with_volatility(0.2).with_expiry(1.0);
const auto put  = call.with_type(OptionType::put);

// Black-Scholes value and delta match the textbook oracle.
black_scholes_price(call).value();                 // ≈ 10.4506
black_scholes_greeks(call).value().delta;          // ≈ 0.6368  (== N(d1))
norm_cdf(0.0);                                      // 0.5

// Put-call parity: C - P == S - K*e^{-r}.
const double c = black_scholes_price(call).value();
const double p = black_scholes_price(put).value();
// c - p ≈ 100 - 100*exp(-0.05)

// The trinomial tree converges to BS; American call (no dividend) == European.
trinomial_price(call, 400, Exercise::european).value();   // ≈ BS
trinomial_price(call, 400, Exercise::american).value();   // ≈ BS

// Geometric Asian reduces EXACTLY to Black-Scholes at one averaging date.
geometric_asian_price(call, 1).value() == c;              // true (to machine precision)
geometric_asian_price(call, 50).value() < c;              // averaging lowers value

// Monte Carlo: estimate + standard error, bit-reproducible under a fixed seed.
const auto mc  = monte_carlo_european(call, 200000, 12345).value();  // mc.price ≈ BS
const auto mc2 = monte_carlo_european(call, 200000, 12345).value();
mc.price == mc2.price;                                    // true (reproducible)
// The geometric control variate shrinks the Asian MC standard error:
monte_carlo_asian(call, 50000, 20, 999, true).value().std_error
    < monte_carlo_asian(call, 50000, 20, 999, false).value().std_error;   // true

// Composable Portfolio: a long straddle.
auto straddle = Portfolio::create().add(call, 1.0).add(put, 1.0);
straddle.value().value();                                 // ≈ c + p
straddle.payoff_at(120.0);                                // 20  (|S - K|)

// Black-76 on the forward equals Black-Scholes; vanilla call == AON - K*CON.
const double fwd = call.spot * std::exp(call.rate * call.time_to_expiry);
black76_price(true, fwd, call.strike, call.rate, call.volatility,
              call.time_to_expiry).value();               // ≈ c

// Plotting yields an SVG string.
PlotOptions opt{}; opt.title = "payoff";
auto svg = payoff_diagram_svg(call, 50.0, 150.0, 64, opt);   // svg->find("<svg") != npos
```

## See also

- [`nimblecas.rng`](rng.md) — the counter-based RNG (`counter_u64`, `splitmix64`,
  `uniform_unit`) that makes the MC draws reproducible and partition-independent.
- [`nimblecas.montecarlo`](montecarlo.md) — the reproducibility design this module
  inherits.
- [`nimblecas.svgplot`](svgplot.md) — the SVG plotter the diagrams render through.
- [`nimblecas.analytics`](analytics.md) — portfolio-level risk and performance
  analytics downstream of a pricing book.
- [`nimblecas.finance`](finance.md) — the exact/numerical TVM and fixed-income
  sibling.
- [Documentation hub](../Index.md)
