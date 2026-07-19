# `nimblecas.futures` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/futures/futures.cppm`

The **cost-of-carry** model for forward/futures fair value — `F = S·e^{b·T}` with a
carry rate `b = r − q + u − y` (financing `r`, dividend/foreign yield `q`, storage `u`,
convenience yield `y`) — plus its inverses (implied cost of carry, implied convenience
yield), the basis and its contango/backwardation classification, the value of an
off-market forward, futures mark-to-market P&L, and notional/tick arithmetic. On top of
valuation sits a **composable, fluent strategy layer**: a signed `FuturesLeg` and a
reusable `FuturesStrategy` with named builders for the standard structures.

## Honesty boundary

This layer is **numerical**: the first quantity, `e^{b·T}`, is transcendental, so
**nothing here claims exactness** — prices are correctly-rounded `double`. Under
**deterministic rates the futures price equals the forward price** (the standard
no-arbitrage result); when they would differ — stochastic-rate convexity — this module
prices the **forward** and says so, rather than fabricating a convexity adjustment it
cannot justify. Futures P&L is **linear and therefore unbounded** for an outright
position: `StrategyProfile` reports `unbounded_profit` / `unbounded_loss` truthfully
instead of inventing a finite max. All failure rides the railway (`Result<T>` /
`MathError`); nothing throws.

```cpp
import nimblecas.futures;
```

Depends only on [`core`](core.md) (`Result` / `MathError`). Everything lives in namespace
`nimblecas::futures`. Consumes spot/entry prices from [`marketdata`](marketdata.md)
`Quote`s upstream.

## Types

| Type | Description |
| :--- | :--- |
| `MarketState` | `enum class { backwardation, flat, contango }`. |
| `FuturesSpec` | Immutable contract + market state, built fluently. Fields: `spot` (S), `rate` (r), `dividend_yield` (q), `storage_cost` (u), `convenience_yield` (y), `time_to_expiry` (T), `contract_size` (multiplier), `tick_size`. `carry_rate()` = `r − q + u − y`. `with_*` chain returns a modified copy. |
| `FuturesLeg` | A signed position in one contract: `label`, `entry_price`, `quantity` (signed, contracts), `contract_size`, `tick_size`. `pnl_at(p)` = `quantity·contract_size·(p − entry)`; `notional()`; fluent `with_*`. |
| `StrategyProfile` | Realised-risk profile: `unbounded_profit`, `unbounded_loss`, `max_profit` (≥ 0), `max_loss` (≤ 0), `locked_pnl` (the constant P&L of a matched net-zero spread; else 0), `breakeven` (optional), `net_quantity` (signed net exposure). |
| `FuturesStrategy` | A reusable bag of legs valued as a unit; fluent `create().with_leg(...)`. |

## Valuation

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `forward_price` / `futures_price` | `auto forward_price(const FuturesSpec&) -> Result<double>` | `F = S·e^{b·T}`. `futures_price` is identical under deterministic rates. `S <= 0` or `T < 0` → `domain_error`. |
| `forward_price_discrete_income` | `auto forward_price_discrete_income(double spot, double pv_income, double rate, double time) -> Result<double>` | `(S − pv_income)·e^{rT}` for a known-income underlying. `S − pv_income <= 0` or `T < 0` → `domain_error`. |
| `basis` | `auto basis(double spot, double futures) noexcept -> double` | Cash basis `spot − futures` (positive = backwardation). Total. |
| `market_state` | `auto market_state(double spot, double futures) noexcept -> MarketState` | `F>S` contango, `F<S` backwardation, else flat. |
| `implied_cost_of_carry` | `auto implied_cost_of_carry(double spot, double futures, double time) -> Result<double>` | `ln(F/S)/T` (the implied repo rate when `q=u=y=0`). Non-positive `S`/`F`/`T` → `domain_error`. |
| `implied_convenience_yield` | `auto implied_convenience_yield(const FuturesSpec&, double market_futures) -> Result<double>` | `y = r − q + u − ln(F/S)/T` from a market price. |
| `forward_contract_value` | `auto forward_contract_value(const FuturesSpec&, double delivery_price) -> Result<double>` | Value of a **long** forward with delivery `K`: `(F − K)·e^{−rT}` (short is the negation). |
| `mark_to_market_pnl` | `auto mark_to_market_pnl(double entry, double current, double contract_size, double quantity) noexcept -> double` | `quantity·contract_size·(current − entry)`; `quantity` signed. Total. |
| `notional_value` | `auto notional_value(double price, double contract_size, double quantity) noexcept -> double` | `price·contract_size·|quantity|`. |
| `tick_value` | `auto tick_value(double tick_size, double contract_size) noexcept -> double` | `tick_size·contract_size`. |

## `FuturesStrategy` — composable strategy graph

Built fluently: `FuturesStrategy::create().with_leg(a).with_leg(b)`, or via a named
builder. P&L is the linear combination of the legs.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `with_leg` | `auto with_leg(const FuturesLeg&) -> FuturesStrategy&` | Append a signed leg. |
| `pnl_at` | `auto pnl_at(std::span<const double> settlement_prices) const -> Result<double>` | Total P&L at per-leg settlement prices (parallel to `legs()`). Length mismatch → `domain_error` (never a silent partial sum). |
| `pnl_at_uniform` | `auto pnl_at_uniform(double p) const noexcept -> double` | Total P&L when every leg settles at the same price (single-instrument / convergence case). |
| `net_quantity` / `notional` | accessors | Signed net exposure `Σ q·mult`; summed leg notionals. |
| `profile` | `auto profile() const -> StrategyProfile` | Risk profile under the single-price interpretation: non-zero net exposure → unbounded both ways with `breakeven = −intercept/net`; zero net exposure (matched spread) → constant P&L in `locked_pnl` (the captured convergence spread), split into `max_profit` (≥ 0) and `max_loss` (≤ 0). |

### Named builders

| Builder | Legs | Meaning |
| :--- | :--- | :--- |
| `long_futures` / `short_futures` | `[outright]` | Directional; `short` negates the quantity. |
| `calendar_spread` | `[near, far]` | Long near, short far (bull calendar; negative `quantity` for bear). |
| `inter_commodity_spread` | `[A, B]` | Long `A`, short `B` (crack / crush / relative-value). |
| `long_hedge` / `short_hedge` | `[futures]` | The futures leg of a hedge against a physical exposure. |
| `cash_and_carry` | `[spot, futures]` | Long spot, short futures — locks the basis (reverse negates). |

## Error model

| Condition | Error |
| :--- | :--- |
| `forward_price` with `S <= 0` or `T < 0`; `forward_price_discrete_income` with `S − I <= 0` or `T < 0`; `implied_cost_of_carry` / `implied_convenience_yield` with non-positive `S`/`F`/`T` | `MathError::domain_error` |
| `FuturesStrategy::pnl_at` with a price-vector length ≠ leg count | `MathError::domain_error` |

`basis`, `market_state`, `mark_to_market_pnl`, `notional_value`, `tick_value`,
`FuturesLeg::pnl_at`, and the strategy aggregators are total on finite inputs. **No price
is claimed exact**, and an unbounded P&L is never reported as a finite number.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.futures;
using namespace nimblecas;
using namespace nimblecas::futures;

// Cost of carry: S=100, r=5%, T=1 -> F = 100·e^0.05 = 105.1271.
const auto spec = FuturesSpec{}.with_spot(100).with_rate(0.05).with_expiry(1.0);
forward_price(spec).value();                         // 105.12710964
// A 2% dividend yield lowers the carry to 3%.
forward_price(spec.with_dividend(0.02)).value();     // 103.04545340
// Back out a convenience yield from a market price.
implied_convenience_yield(spec, 103.04545340).value();   // 0.02

// A calendar spread captures the convergence spread when net exposure is zero.
const auto cal = calendar_spread("ESM", 4500, "ESU", 4520, 1.0, 50.0);
cal.pnl_at(std::array{4510.0, 4525.0}).value();      // 250 (distinct settles)
cal.profile().locked_pnl;                            // 1000 == (far − near)·mult

// An outright is honestly unbounded, with breakeven at its entry.
const auto lng = long_futures("ES", 4500, 2.0, 50.0);
lng.pnl_at_uniform(4510.0);                          // 1000
lng.profile().unbounded_profit;                      // true
*lng.profile().breakeven;                            // 4500
```

## See also

- [`nimblecas.pricing`](pricing.md) — Black-76 options on these futures.
- [`nimblecas.marketdata`](marketdata.md) — the normalised `Quote` feeding `spot`/entry.
- [`nimblecas.currency`](currency.md) — covered-interest-parity FX forwards.
- [Documentation hub](../Index.md)
