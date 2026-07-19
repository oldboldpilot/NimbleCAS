# `nimblecas.optstrat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/optstrat/optstrat.cppm`

A **composable, fluent option-strategy layer** over [`pricing`](pricing.md). A strategy is
a signed bag of `StrategyLeg`s (long/short calls, puts, and the underlying); the module
supplies the standard **named structures** as one-line builders and a general
**expiry-P&L analytics engine** that returns the net premium (debit/credit), max profit,
max loss, and every breakeven for *any* such structure. Aggregate Black-Scholes Greeks
are available by delegating each option leg to `pricing` against a market template.

## Honesty boundary

The expiry P&L of a vanilla option book is **exactly a continuous piecewise-linear
function** of the terminal price, with kinks only at the strikes. The analytics here are
therefore computed **exactly on that structure, not sampled**: max/min are attained at
strikes or at ±∞, and breakevens are the exact linear roots between kinks. When a payoff
direction is genuinely **unbounded** (a long call's upside, a short straddle's downside)
the analytics say so — `unbounded_profit` / `unbounded_loss` — rather than fabricating a
finite number. The Greeks bridge is **numerical** (Black-Scholes) and inherits pricing's
honesty boundary. All failure rides the railway; nothing throws.

```cpp
import nimblecas.optstrat;
```

Depends on [`core`](core.md) and [`pricing`](pricing.md) (`OptionSpec`, `OptionType`,
`Greeks`, `black_scholes_greeks`). Everything lives in namespace `nimblecas::optstrat`.
Premiums typically come from a [`marketdata`](marketdata.md) `OptionChain` or a
`pricing::black_scholes_price` theoretical.

## Types

| Type | Description |
| :--- | :--- |
| `LegKind` | `enum class { call, put, underlying }`. |
| `StrategyLeg` | One signed leg: `kind`, `strike` (ignored for `underlying`), `quantity` (signed), `premium` (per-unit price). `terminal_value(s)` = per-unit intrinsic; `cost()` = `quantity·premium`; `pnl_at(s)`. |
| `StrategyAnalytics` | `net_premium` (>0 debit, <0 credit), `unbounded_profit`, `unbounded_loss`, `max_profit`, `max_loss`, `breakevens` (ascending). |
| `OptionStrategy` | Composable bag of legs, valued at expiry as a unit. |

## `OptionStrategy`

Built fluently: `OptionStrategy::create().add_call(...).add_put(...).add_underlying(...)`,
or via a named builder.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `add_call` / `add_put` | `auto add_call(double strike, double quantity, double premium) -> OptionStrategy&` | Append a signed option leg. |
| `add_underlying` | `auto add_underlying(double quantity, double entry_price) -> OptionStrategy&` | Append a linear underlying leg. |
| `with_leg` / `legs` / `size` | — | Append a `StrategyLeg`; the legs; the count. |
| `payoff_at` | `auto payoff_at(double s) const noexcept -> double` | Gross terminal value `Σ q·terminal_value(s)` (payoff diagram, no premium). |
| `net_premium` | `auto net_premium() const noexcept -> double` | `Σ cost` (>0 debit, <0 credit). |
| `pnl_at` | `auto pnl_at(double s) const noexcept -> double` | `payoff_at(s) − net_premium`. |
| `analytics` | `auto analytics() const -> StrategyAnalytics` | The exact expiry-P&L profile via piecewise-linear analysis on the strikes. |
| `aggregate_greeks` | `auto aggregate_greeks(const OptionSpec& market) const -> Result<Greeks>` | Aggregate Black-Scholes Greeks: `market`'s rate/vol/dividend/expiry/spot apply to every option leg (each overriding strike + type); underlying legs contribute delta == quantity. A leg's domain error propagates. |

## Named builders

Every option quantity is per `lots` (default 1); premiums are the per-unit prices
transacted.

| Builder | Structure |
| :--- | :--- |
| `long_call` / `long_put` / `short_call` / `short_put` | Single-leg. |
| `covered_call` | long underlying + short call. |
| `protective_put` | long underlying + long put. |
| `collar` | long underlying + long put + short call. |
| `bull_call_spread` / `bear_call_spread` / `bull_put_spread` / `bear_put_spread` | Vertical spreads. |
| `long_straddle` / `short_straddle` | Same-strike call + put. |
| `long_strangle` / `short_strangle` | OTM put + OTM call. |
| `strip` / `strap` | 1 call + 2 puts / 2 calls + 1 put (long). |
| `long_call_butterfly` / `short_call_butterfly` | +1/−2/+1 calls (and inverse). |
| `iron_butterfly` | short ATM straddle + long OTM strangle wings. |
| `long_call_condor` | +1/−1/−1/+1 calls. |
| `iron_condor` | short put spread + short call spread. |
| `box_spread` | bull call spread + bear put spread (synthetic fixed width). |
| `risk_reversal` | long call + short put (synthetic long). |
| `ratio_call_spread` | long 1 lo call, short `ratio` hi calls. |

## Error model

`aggregate_greeks` propagates a leg's `pricing::black_scholes_greeks` error (e.g.
`domain_error` on a non-physical `market`). `payoff_at`, `net_premium`, `pnl_at`, and
`analytics` are total on finite inputs. **Unbounded profit/loss is reported as unbounded,
never as a finite number.**

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.optstrat;
using namespace nimblecas;
using namespace nimblecas::optstrat;

// Bull call spread: long 100 call @5, short 110 call @2 -> net debit 3.
const auto bcs = bull_call_spread(100, 5, 110, 2);
const auto a = bcs.analytics();
a.net_premium;   // 3   (debit)
a.max_profit;    // 7   (width − debit)
a.max_loss;      // −3  (−debit)
a.breakevens;    // { 103 }

// Long straddle: unbounded upside, two breakevens.
const auto st = long_straddle(100, 5, 4).analytics();
st.unbounded_profit;   // true
st.max_loss;           // −9
st.breakevens;         // { 91, 109 }

// Iron condor: net credit, capped both ways.
const auto ic = iron_condor(90, 1, 95, 3, 105, 3, 110, 1).analytics();
ic.net_premium;   // −4  (credit)
ic.max_profit;    // 4
ic.max_loss;      // −1

// Aggregate Greeks against a market template (reuses pricing's Black-Scholes).
const auto mkt = OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
                     .with_volatility(0.2).with_expiry(1.0);
long_straddle(100, 10.45, 5.57).aggregate_greeks(mkt).value().delta;   // ≈ 0.2737
```

## See also

- [`nimblecas.pricing`](pricing.md) — the Black-Scholes / Greeks engine this reuses, and
  its composable `Portfolio` for premium calculation.
- [`nimblecas.marketdata`](marketdata.md) — the `OptionChain` supplying strikes/premiums.
- [`nimblecas.futures`](futures.md) — the futures-strategy sibling.
- [Documentation hub](../Index.md)
