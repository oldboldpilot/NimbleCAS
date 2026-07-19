# `nimblecas.fxstrat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/fxstrat/fxstrat.cppm`

Currency-trading valuation the option/futures strategy engines cannot supply on their
own: **Garman-Kohlhagen** FX option pricing (Black-Scholes with the foreign rate as the
dividend yield — a direct reuse of [`pricing`](pricing.md)), **covered-interest-parity**
forwards and forward points, the **covered-interest-arbitrage** mispricing, and a fluent
**`CarryTrade`**. Option-based FX structures (straddles, risk reversals) are built by
feeding Garman-Kohlhagen premiums into [`optstrat`](optstrat.md), which owns the multi-leg
P&L — this module does not duplicate that.

## Honesty boundary

**Numerical**: BS/GK `double` prices and `e^{(r_d−r_f)T}` parities — nothing claims
exactness. The quoting convention is fixed and asserted in the tests: `spot` is
**domestic per one unit of foreign** (e.g. USD per EUR), so `rate_domestic` funds and
`rate_foreign` accrues; the carry breakeven equalling the CIP forward is the check that
pins that convention. All failure rides the railway; nothing throws.

```cpp
import nimblecas.fxstrat;
```

Depends on [`core`](core.md) and [`pricing`](pricing.md). Namespace `nimblecas::fxstrat`.

## Functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `garman_kohlhagen` | `auto garman_kohlhagen(bool is_call, double spot, double strike, double rate_domestic, double rate_foreign, double vol, double time) -> Result<double>` | FX option price = Black-Scholes with `q := rate_foreign`. Non-physical spec → `domain_error`. |
| `cip_forward` | `auto cip_forward(double spot, double rate_domestic, double rate_foreign, double time) -> Result<double>` | `S·e^{(r_d − r_f)·T}`. `spot <= 0` or `time < 0` → `domain_error`. |
| `forward_points` | `auto forward_points(double spot, double rate_domestic, double rate_foreign, double time) -> Result<double>` | CIP forward − spot. |
| `covered_interest_arbitrage` | `auto covered_interest_arbitrage(double spot, double forward_market, double rate_domestic, double rate_foreign, double time) -> Result<double>` | `forward_market − CIP forward` (per unit foreign); non-zero beyond costs is an arbitrage. |

## `CarryTrade`

Fluent struct (borrow domestic / invest foreign). Fields: `notional`, `rate_borrow`,
`rate_invest`, `spot_entry`, `time`; `with_*` builders.

| Method | Behavior |
| :--- | :--- |
| `carry_rate()` | `rate_invest − rate_borrow` (the pickup if FX is flat). |
| `pnl_at(spot_exit)` | `(notional/spot_entry)·e^{r_invest·T}·spot_exit − notional·e^{r_borrow·T}`. |
| `breakeven_spot()` | `spot_entry·e^{(r_borrow − r_invest)·T}` — exactly the CIP forward (uncovered interest parity). |

## Error model

| Condition | Error |
| :--- | :--- |
| `garman_kohlhagen` with a non-physical spec | `MathError::domain_error` (from pricing) |
| `cip_forward` / `forward_points` / `covered_interest_arbitrage` with `spot <= 0` or `time < 0` | `MathError::domain_error` |

## Worked example

```cpp
import nimblecas.fxstrat;
using namespace nimblecas::fxstrat;

// EURUSD 1.10, r_dom 5%, r_for 2%, T=1.
cip_forward(1.10, 0.05, 0.02, 1.0).value();        // 1.13350
forward_points(1.10, 0.05, 0.02, 1.0).value();     // 0.03350

const auto ct = CarryTrade{}.with_borrow(0.02).with_invest(0.05).with_spot(1.10).with_time(1.0);
ct.carry_rate();          // 0.03
ct.breakeven_spot();      // 1.06749  (== the CIP forward)
ct.pnl_at(1.10);          // e^0.05 − e^0.02  (interest differential when FX is flat)
```

## See also

- [`nimblecas.currency`](currency.md) — exact FX rates, cross rates, arbitrage.
- [`nimblecas.pricing`](pricing.md) — the Black-Scholes engine GK reuses.
- [`nimblecas.optstrat`](optstrat.md) — multi-leg P&L for FX option structures.
- [Documentation hub](../Index.md)
