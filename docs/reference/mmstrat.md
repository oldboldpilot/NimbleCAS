# `nimblecas.mmstrat` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/mmstrat/mmstrat.cppm`

The short-end trading toolkit above [`finance`](finance.md)'s T-bill / discount helpers:
**repo** (and reverse-repo) interest and repayment on an actual/360 basis; discount
instrument **price ↔ bank-discount-rate** conversion; the bank-discount → **bond-equivalent
(investment) yield** conversion for like-for-like comparison; simple holding-period return;
and the effective compounded rate of a **deposit strip** (a ladder of forward money-market
deposits — the money-market analogue of a futures/FRA strip).

## Honesty boundary

These are **exact arithmetic identities** on `double` day-count conventions — no
root-finding, no series. The day-count basis (360 vs 365) is an **explicit parameter,
never hidden**, because the bank-discount vs bond-equivalent distinction is exactly where
money-market quoting bugs live. A non-physical input (non-positive face/price/basis, an
empty or mismatched strip) returns `domain_error`; a zeroed denominator returns
`division_by_zero`. Nothing throws.

```cpp
import nimblecas.mmstrat;
```

Depends only on [`core`](core.md). Namespace `nimblecas::mmstrat`.

## Functions

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `repo_interest` | `auto repo_interest(double principal, double repo_rate, double days, double basis = 360.0) -> Result<double>` | `principal·repo_rate·days/basis`. `basis <= 0` or `days < 0` → `domain_error`. |
| `repo_repayment` | `auto repo_repayment(double principal, double repo_rate, double days, double basis = 360.0) -> Result<double>` | `principal + repo_interest`. |
| `discount_price` | `auto discount_price(double face, double discount_rate, double days, double basis = 360.0) -> Result<double>` | `face·(1 − discount_rate·days/basis)`. Non-physical → `domain_error`. |
| `discount_rate_from_price` | `auto discount_rate_from_price(double face, double price, double days, double basis = 360.0) -> Result<double>` | Inverse: `(face − price)/face · basis/days`. `days <= 0` → `domain_error`. |
| `bond_equivalent_yield` | `auto bond_equivalent_yield(double discount_rate, double days) -> Result<double>` | `(365·d)/(360 − d·days)` (short-bill form). `d·days == 360` → `division_by_zero`; `days <= 0` → `domain_error`. |
| `holding_period_return` | `auto holding_period_return(double begin_price, double end_price, double income) -> Result<double>` | `(end − begin + income)/begin`. `begin <= 0` → `domain_error`. |
| `deposit_strip_effective_rate` | `auto deposit_strip_effective_rate(std::span<const double> rates, std::span<const double> accruals) -> Result<double>` | `(∏(1 + rateᵢ·accrualᵢ))^(1/Σaccrual) − 1`. Empty / mismatched / non-positive tenor → `domain_error`. |

## Worked example

```cpp
import nimblecas.mmstrat;
using namespace nimblecas::mmstrat;

repo_interest(1'000'000.0, 0.05, 30.0).value();      // 4166.67  (actual/360)
discount_price(100.0, 0.05, 90.0).value();           // 98.75    (90-day bill @5%)
discount_rate_from_price(100.0, 98.75, 90.0).value();// 0.05     (round trip)
bond_equivalent_yield(0.05, 90.0).value();           // 18.25/355.5

// Effective compounded rate of a two-period 4%/5% deposit strip.
deposit_strip_effective_rate(std::array{0.04, 0.05}, std::array{0.5, 0.5}).value();  // 0.0455
```

## See also

- [`nimblecas.finance`](finance.md) — T-bill (`tbill_price`/`tbill_yield`/`tbill_eq`) and
  discount-security primitives.
- [`nimblecas.futures`](futures.md) — implied-repo / cash-and-carry at the short end.
- [Documentation hub](../Index.md)
