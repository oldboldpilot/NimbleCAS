# `nimblecas.bondstrat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/bondstrat/bondstrat.cppm`

The bond strategy layer above [`finance`](finance.md) / [`fixedincome`](fixedincome.md)
analytics: weighted portfolio duration / convexity / yield; the classic curve structures
built from per-bond durations — the two-bond **barbell** that hits a target duration, the
duration-and-cash-neutral **butterfly** (long wings, short body), and the duration-neutral
**hedge ratio** between two bonds; plus carry-and-roll-down return and duration-convexity
P&L. It takes per-bond analytics as inputs (price, modified duration, convexity, yield)
rather than re-deriving them — those come from `finance` (`bond_price`, `bond_mduration`,
`bond_convexity`).

## Honesty boundary

The weighting algebra is **exact** on its `double` inputs (linear solves of small systems);
it introduces no approximation beyond the durations it is handed. A degenerate system
(equal durations, zero denominator) returns an honest `MathError` — `division_by_zero`
when a denominator vanishes, `domain_error` on a length mismatch or non-positive price —
never a fabricated ratio. Nothing throws.

```cpp
import nimblecas.bondstrat;
```

Depends only on [`core`](core.md). Namespace `nimblecas::bondstrat`.

## Functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `weighted_average` | `auto weighted_average(std::span<const double> weights, std::span<const double> values) -> Result<double>` | `Σ wᵢ·xᵢ` (portfolio duration/convexity/yield). Empty or length mismatch → `domain_error`. |
| `barbell_weights` | `auto barbell_weights(double target_duration, double dur_short, double dur_long) -> Result<std::array<double,2>>` | `(w_short, w_long)` summing to 1 with blended duration `= target`. `dur_long == dur_short` → `division_by_zero`. |
| `duration_hedge_ratio` | `auto duration_hedge_ratio(double dur_a, double price_a, double dur_b, double price_b) -> Result<double>` | Units of B to short per unit of A for zero net DV01: `(D_a·P_a)/(D_b·P_b)`. Zero denominator → `division_by_zero`. |
| `butterfly_weights` | `auto butterfly_weights(double dur_lo, double dur_body, double dur_hi) -> Result<std::array<double,2>>` | Wing weights `(w_lo, w_hi)` that are cash-neutral (sum 1) and duration-neutral (`= dur_body`). `dur_lo == dur_hi` → `division_by_zero`. |
| `carry_roll_return` | `auto carry_roll_return(double price_now, double price_rolled, double coupon_income) -> Result<double>` | `(price_rolled − price_now + coupon_income)/price_now`. `price_now <= 0` → `domain_error`. |
| `duration_convexity_pnl` | `auto duration_convexity_pnl(double modified_duration, double convexity, double dy) noexcept -> double` | `−D·dy + 0.5·C·dy²` per unit price. Total. |

## Worked example

```cpp
import nimblecas.bondstrat;
using namespace nimblecas::bondstrat;

// A 6y target from a 2y/10y barbell -> 50/50.
barbell_weights(6.0, 2.0, 10.0).value();          // { 0.5, 0.5 }
// Duration- & cash-neutral 2/5/10 butterfly.
butterfly_weights(2.0, 5.0, 10.0).value();        // { 0.625, 0.375 }
// Second-order P&L of a 7y-duration, 50-convexity bond under a +100bp shift.
duration_convexity_pnl(7.0, 50.0, 0.01);          // −0.0675
```

## See also

- [`nimblecas.finance`](finance.md) / [`nimblecas.fixedincome`](fixedincome.md) — the bond
  analytics (price, duration, convexity) feeding these strategies.
- [`nimblecas.yieldcurve`](yieldcurve.md) — the curve behind roll-down.
- [Documentation hub](../Index.md)
