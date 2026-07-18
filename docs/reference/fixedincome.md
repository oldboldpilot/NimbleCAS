# `nimblecas.fixedincome` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/fixedincome/fixedincome.cppm`

A second-tier fixed-income analytics layer built **on top of** [`nimblecas.finance`](finance.md).
It reuses that module's calendar and pricing primitives — `Date` / `DayCount` / `year_fraction`
(dates are integers, so year fractions are exact rationals) and the bond `PRICE` / `MDURATION`
family — and adds: sensitivities (DV01 / PV01, effective and key-rate durations), dated cashflow
schedules, coupon-date functions, odd-period pricing, maturity-paying securities, the z-spread,
yield-convention adapters, floating-rate-note valuation, and level-payment amortization schedules.

```cpp
import nimblecas.fixedincome;
```

Everything lives in namespace `nimblecas::fixedincome`, all entry points are free functions marked
`[[nodiscard]]`, and every failure rides the railway (`Result<T>` / `MathError`) — nothing throws.

## Dependencies

Imports [`nimblecas.core`](core.md) (`Result` / `MathError` / `make_error`) and
[`nimblecas.finance`](finance.md). It reuses `finance::Date`, `finance::DayCount`, and
`finance::year_fraction` for the calendar and day-count arithmetic, and `finance::bond_price` /
`finance::bond_mduration` (via the sensitivity functions in group A). Numeric interfaces use
`std::span<const double>`, `std::vector<double>`, `std::function<double(double)>`, and
`std::optional<finance::Date>` from `import std;`.

## Honesty boundary

This module is **numerical**. Every output is a discounted-cashflow limit whose discount factor
`(1 + y/f)^t` carries a fractional — hence transcendental — power, so nothing here claims exactness:
results are `Result<double>` / `Result<std::vector<double>>` computed to a stated tolerance, with
bracketed root-finding (a local Brent method) where a yield or spread is inverted from a price.

The one exact exception is group **C**, the integer coupon-day counts: those are pure calendar
differences and return `Result<std::int64_t>`. They satisfy the identity
`coup_days_bs + coup_days_nc == coup_days`.

Nothing returns NaN or infinity. A singular divisor, an out-of-domain fractional power, or a
non-bracketing root yields `MathError::division_by_zero`, `MathError::domain_error`, or
`MathError::not_converged` respectively. All iteration is bounded, and DoS-sized inputs are refused
rather than allocated: cashflow / period counts above `100000` and schedule walks beyond `4000`
coupons (an absurd tenor) are rejected with `domain_error`.

### Convention note (documented divergence, never hidden)

The discounting exponent for a cashflow at date `D` is
`year_fraction(settlement, D, basis) * frequency` — the **uniform-period** convention. For the
30/360 family this is identical to the street/SIA grid (each period is exactly 180 / 90 days), so
`odd_first_price` / `odd_last_price` with a **regular** stub reproduce `finance::bond_price`
bit-for-bit on a 30/360 basis. For `actual_actual` and `actual_365` it is instead the consistent
**true-yield** convention rather than a per-period street count, and diverges in the low-order
digits — the same class of divergence `finance` already documents for `DayCount::actual_actual`.

## API

### A. Sensitivities

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `dv01` | `auto dv01(const finance::Date& settlement, const finance::Date& maturity, double coupon_rate, double yield, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Dollar value of a basis point, per 100 face: the price drop for a 1bp yield rise, from a symmetric ±½bp reprice, so `dv01 ≈ modified_duration · price · 1e-4`. Positive for an ordinary bond. Propagates the underlying `bond_price` error. |
| `pv01` | `auto pv01(const finance::Date& settlement, const finance::Date& maturity, double coupon_rate, double yield, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Price value of a basis point, per 100 face: the one-sided drop `P(y) − P(y + 1bp)`. Like `dv01`, approximates `modified_duration · price · 1e-4`. |
| `duration_from_price` | `auto duration_from_price(const finance::Date& settlement, const finance::Date& maturity, double coupon_rate, double yield, double redemption, int frequency, finance::DayCount basis, double dy = 1e-4) -> Result<double>` | Effective duration (years) by ±`dy` reprice: `(P(y−dy) − P(y+dy)) / (2·P(y)·dy)`. For an option-free bond this recovers the modified duration. `dy` defaults to 1bp. |
| `effective_duration` | `auto effective_duration(std::span<const double> times, std::span<const double> amounts, const std::function<double(double)>& df, double bump = 1e-4) -> Result<double>` | Parallel-shift duration of an arbitrary cashflow set priced off a discount-factor curve `df(t)`: `−(P(+bump) − P(−bump)) / (2·P0·bump)`, shifting the whole zero curve by ±`bump` (`df'(t) = df(t)·exp(−shift·t)`). `times` in years; `amounts` absolute. |
| `key_rate_durations` | `auto key_rate_durations(std::span<const double> times, std::span<const double> amounts, const std::function<double(double)>& df, std::span<const double> key_tenors, double bump = 1e-4) -> Result<std::vector<double>>` | Partial durations: for each key tenor, bump only that tenor's zero rate by ±`bump` using triangular (tent) weights that form a partition of unity, and reprice. The tents sum to 1 everywhere, so the key-rate durations sum to the `effective_duration` above. `key_tenors` must be strictly increasing; returns one duration per key tenor. |

### B. Dated cashflow schedule

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `cfdates` | `auto cfdates(const finance::Date& settlement, const finance::Date& maturity, int frequency) -> Result<std::vector<finance::Date>>` | Coupon payment dates strictly after settlement through maturity (ascending), on the regular schedule anchored at maturity stepping back `12 / frequency` months. |
| `cfamounts` | `auto cfamounts(const finance::Date& settlement, const finance::Date& maturity, double coupon_rate, int frequency, finance::DayCount basis, double face = 100.0, double redemption = -1.0, std::optional<finance::Date> issue = std::nullopt) -> Result<DatedCashflows>` | Coupon + principal cashflows and their dates. Each regular coupon is `face·coupon_rate/frequency`; the final date adds `redemption`. If `issue` is supplied and precedes the first payment period, the first coupon is **prorated** over its (possibly odd/stub) accrual `issue → first-payment` via the day-count basis (a short stub pays less, a long stub more). `redemption < 0` defaults to `face`. |

### C. Coupon-date functions (exact integer day counts)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `coup_pcd` | `auto coup_pcd(const finance::Date& settlement, const finance::Date& maturity, int frequency) -> Result<finance::Date>` | The coupon date on or before settlement (previous coupon date). Schedule anchored at maturity, stepping back `12 / frequency` months. |
| `coup_ncd` | `auto coup_ncd(const finance::Date& settlement, const finance::Date& maturity, int frequency) -> Result<finance::Date>` | The coupon date strictly after settlement (next coupon date). |
| `coup_days_bs` | `auto coup_days_bs(const finance::Date& settlement, const finance::Date& maturity, int frequency, finance::DayCount basis) -> Result<std::int64_t>` | Days from the previous coupon date to settlement. Exact integer. |
| `coup_days_nc` | `auto coup_days_nc(const finance::Date& settlement, const finance::Date& maturity, int frequency, finance::DayCount basis) -> Result<std::int64_t>` | Days from settlement to the next coupon date. Exact integer. |
| `coup_days` | `auto coup_days(const finance::Date& settlement, const finance::Date& maturity, int frequency, finance::DayCount basis) -> Result<std::int64_t>` | Days in the coupon period containing settlement. Under a 30/360 basis a period is exactly `360 / frequency` days (180 semiannual); actual bases use calendar days. Satisfies `coup_days_bs + coup_days_nc == coup_days`. |

### D. Odd-period pricing

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `odd_first_price` | `auto odd_first_price(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& issue, const finance::Date& first_coupon, double coupon_rate, double yield, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Clean price per 100 face of a bond with an odd (short or long) **first** coupon period. `issue` is the dated/issue date, `first_coupon` the first (odd) coupon date; thereafter coupons are regular through maturity. With a regular first period (`issue == first_coupon − one period`) this equals `finance::bond_price` on a 30/360 basis (see the convention note). |
| `odd_first_yield` | `auto odd_first_yield(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& issue, const finance::Date& first_coupon, double coupon_rate, double clean_price, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Yield to maturity implied by a clean price for an odd-first bond, by bracketed root-finding. |
| `odd_last_price` | `auto odd_last_price(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& last_interest, double coupon_rate, double yield, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Clean price per 100 face of a bond with an odd **last** coupon period. `last_interest` is the last regular coupon date; the final period `last_interest → maturity` is odd. |
| `odd_last_yield` | `auto odd_last_yield(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& last_interest, double coupon_rate, double clean_price, double redemption, int frequency, finance::DayCount basis) -> Result<double>` | Yield implied by a clean price for an odd-last bond, by bracketed root-finding. |

### E. Maturity-paying securities (interest paid once, at maturity)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `price_mat` | `auto price_mat(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& issue, double coupon_rate, double yield, finance::DayCount basis) -> Result<double>` | Clean price per 100 face of a security that pays interest once at maturity, given the issue date, coupon rate and yield (simple discounting to maturity). |
| `yield_mat` | `auto yield_mat(const finance::Date& settlement, const finance::Date& maturity, const finance::Date& issue, double coupon_rate, double price, finance::DayCount basis) -> Result<double>` | The simple annual yield implied by a clean price; closed form (inverts `price_mat`). |
| `accrint_mat` | `auto accrint_mat(const finance::Date& issue, const finance::Date& settlement, double coupon_rate, finance::DayCount basis, double par = 100.0) -> Result<double>` | Accrued interest per 100 face at maturity: `par · coupon_rate · year_fraction(issue, settlement, basis)`. |

### F. Z-spread

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `z_spread` | `auto z_spread(std::span<const double> times, std::span<const double> amounts, const std::function<double(double)>& df, double target_price) -> Result<double>` | The constant continuously-compounded spread `s` such that discounting each cashflow at `df(t)·exp(−s·t)` reprices the bond to `target_price`: solves `Σ amountsᵢ·df(tᵢ)·exp(−s·tᵢ) == target_price` for `s` by bracketed root-finding. `times` in years, `amounts` absolute. Against a curve that already reprices the bond to target (e.g. the bond's own flat curve), `s ≈ 0`. |

### G. Yield-convention adapters

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `simple_to_compound` | `auto simple_to_compound(double simple_rate, double t) -> Result<double>` | Simple (money-market) rate over `t` periods/years to the equivalent compound rate: `(1 + compound)^t == 1 + simple·t`. Also the street→true relationship for one stub of length `t` (street = simple discounting, true = compound). |
| `compound_to_simple` | `auto compound_to_simple(double compound_rate, double t) -> Result<double>` | Inverse of `simple_to_compound`. |
| `nominal_convert` | `auto nominal_convert(double rate, int from_m, int to_m) -> Result<double>` | Convert a nominal rate compounded `from_m` times/yr to the equivalent nominal compounded `to_m` times/yr: `(1 + r/from_m)^from_m == (1 + r'/to_m)^to_m`. `from_m` and `to_m` must be `>= 1`. |
| `annual_to_semiannual_nominal` | `auto annual_to_semiannual_nominal(double annual_rate) -> Result<double>` | Convenience wrapper: `nominal_convert(annual_rate, 1, 2)`. |
| `semiannual_to_annual_nominal` | `auto semiannual_to_annual_nominal(double semi_rate) -> Result<double>` | Convenience wrapper: `nominal_convert(semi_rate, 2, 1)`. |

### H. FRN valuation and amortization

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `frn_value` | `auto frn_value(std::span<const double> accruals, std::span<const double> forward_rates, std::span<const double> discount_factors, double margin, double notional = 100.0) -> Result<double>` | Floating-rate note value: `Σ (forwardᵢ + margin)·accrualᵢ·dfᵢ` plus the principal `df_last`, all scaled by `notional`. The three spans are parallel (one entry per period). A par floater (`margin == 0`) priced off discount factors consistent with its own forwards values to par (`notional`). |
| `amortizing_schedule` | `auto amortizing_schedule(double principal, double rate_per_period, std::int64_t nper) -> Result<AmortizationSchedule>` | Level-payment (fully amortizing) loan schedule: the constant payment and the per-period interest / principal / balance vectors. The final principal is set to the remaining balance so the schedule closes exactly (sum of principal == loan, closing balance == 0). Handles `rate_per_period == 0` as the straight-line limit. |

### Result structs

```cpp
// Parallel cashflow amounts (per the supplied face convention) and their calendar dates.
struct DatedCashflows {
    std::vector<double> amounts{};
    std::vector<finance::Date> dates{};
};

// Level-payment amortization output: the constant payment, and per-period interest, principal
// and end-of-period balance vectors (each of length nper). The final principal is set to the
// remaining balance so the schedule closes exactly (sum of principal == loan, final balance 0).
struct AmortizationSchedule {
    double payment{0.0};
    std::vector<double> interest{};
    std::vector<double> principal{};
    std::vector<double> balance{};
};
```

## Error model

| Condition | Error |
| :--- | :--- |
| Invalid coupon `frequency` (not 1, 2, or 4); `maturity <= settlement`; a negative `face`; a cashflow dated before settlement; an odd-period bond with `first_coupon <= issue`, `maturity < first_coupon`, `settlement < issue`, or `maturity <= last_interest`; `cfamounts` with `issue` after the first coupon | `MathError::domain_error` |
| An out-of-domain fractional power: `(1 + y/f) <= 0` in discounting; `1 + simple·t <= 0`; `1 + rate/from_m <= 0`; `compound_rate <= −1`; `nominal_convert` with `from_m < 1` or `to_m < 1` | `MathError::domain_error` |
| Empty / length-mismatched spans, or a span longer than `100000` entries, in `effective_duration`, `key_rate_durations`, `z_spread`, `frn_value`; `key_tenors` empty or not strictly increasing; a non-positive `bump` or `dy` | `MathError::domain_error` |
| A non-finite discount factor `df(t)`, a non-finite intermediate price, or an amortization `principal` / `payment` that is not finite | `MathError::domain_error` |
| A schedule walk exceeding `4000` coupons (absurd tenor) | `MathError::domain_error` |
| `amortizing_schedule` with `nper < 1`, `nper > 100000`, or `rate_per_period <= −1` | `MathError::domain_error` |
| A zero base price in `duration_from_price` / `effective_duration` / `key_rate_durations`; a zero denominator `1 + yield·(settle→maturity)` or zero dirty price in `price_mat` / `yield_mat`; `t == 0` in the simple↔compound adapters; a zero amortization denominator | `MathError::division_by_zero` |
| A yield / spread inversion (`odd_first_yield`, `odd_last_yield`, `z_spread`) whose objective cannot be bracketed within the bounded search | `MathError::not_converged` |
| Any error propagated from `finance::year_fraction` or `finance::bond_price` | as returned by `finance` |

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.finance;
import nimblecas.fixedincome;
using namespace nimblecas;
using namespace nimblecas::fixedincome;

const auto settle   = finance::Date::of(2025, 1, 15).value();
const auto maturity = finance::Date::of(2030, 1, 15).value();

// (A) Sensitivities of a 5%-coupon semiannual bond yielding 6%, per 100 face.
dv01(settle, maturity, 0.05, 0.06, 100.0, 2, finance::DayCount::thirty_360).value();   // > 0
pv01(settle, maturity, 0.05, 0.06, 100.0, 2, finance::DayCount::thirty_360).value();   // ≈ dv01
duration_from_price(settle, maturity, 0.05, 0.06, 100.0, 2,
                    finance::DayCount::thirty_360).value();                             // years

// Curve-based effective and key-rate durations. The tents partition unity, so the key-rate
// durations sum to the effective duration.
const std::vector<double> times{1.0, 2.0, 3.0};
const std::vector<double> flows{5.0, 5.0, 105.0};
auto df = [](double t) { return std::exp(-0.04 * t); };   // flat 4% continuous curve
effective_duration(times, flows, df).value();
const std::vector<double> tenors{1.0, 3.0};
auto krd = key_rate_durations(times, flows, df, tenors).value();   // krd[0] + krd[1] ≈ eff. dur.

// (B) Dated cashflows with a prorated odd first coupon (issue precedes the first period).
const auto issue = finance::Date::of(2025, 2, 1).value();
auto cf = cfamounts(settle, maturity, 0.05, 2, finance::DayCount::actual_actual,
                    100.0, -1.0, issue).value();          // cf.amounts.back() includes redemption

// (C) Coupon-date day counts obey coup_days_bs + coup_days_nc == coup_days (exact integers).
const auto b = finance::DayCount::actual_actual;
auto bs = coup_days_bs(settle, maturity, 2, b).value();
auto nc = coup_days_nc(settle, maturity, 2, b).value();
auto dd = coup_days(settle,   maturity, 2, b).value();
// bs + nc == dd.

// (E) Maturity-paying security: yield_mat inverts price_mat.
auto pm = price_mat(settle, maturity, issue, 0.05, 0.06, b).value();
yield_mat(settle, maturity, issue, 0.05, pm, b).value();  // ≈ 0.06

// (F) Z-spread of a bond already repriced by its own curve is ~0.
auto tgt = 0.0;
for (std::size_t i = 0; i < times.size(); ++i) { tgt += flows[i] * df(times[i]); }
z_spread(times, flows, df, tgt).value();                  // ≈ 0

// (G) Nominal-rate compounding conversion round-trips.
auto semi = annual_to_semiannual_nominal(0.10).value();
semiannual_to_annual_nominal(semi).value();               // ≈ 0.10

// (H) A fully-amortizing loan closes exactly: closing balance == 0.
auto sched = amortizing_schedule(1000.0, 0.01, 12).value();
sched.balance.back();                                     // == 0.0
```

## See also

- [`nimblecas.finance`](finance.md) — the underlying calendar (`Date` / `DayCount` / `year_fraction`)
  and bond primitives (`bond_price`, `bond_mduration`, `coupon_num`) this module builds on.
- [`nimblecas.yieldcurve`](yieldcurve.md) — discount-factor curve construction, the natural source
  of the `df(t)` callbacks passed to `effective_duration`, `key_rate_durations`, and `z_spread`.
- [Documentation hub](../Index.md)
