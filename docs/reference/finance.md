# `nimblecas.finance` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/finance/finance.cppm`

The Excel / Mathematica / MATLAB / Maple financial-function suite: time value of
money (PV, FV, PMT, IPMT, PPMT, NPER, RATE, CUMIPMT, CUMPRINC), cashflow analysis
(NPV, XNPV, IRR, XIRR, MIRR, FVSCHEDULE), depreciation (SLN, SYD, DDB, DB, VDB),
rate conversion (EFFECT, NOMINAL, RRI, PDURATION), dollar-fraction conversion,
fixed-income analytics (bond PRICE, YIELD, DURATION, MDURATION, convexity, accrued
interest, discount securities, T-bills), swaps, and a fluent
`CashflowSchedule` / `TvmProblem` builder layer that mirrors Wolfram's `Cashflow`
/ `TimeValue` and exceeds a spreadsheet's fixed arity.

## Honesty boundary — the two-tier contract

This module keeps a **sharp, per-function honesty boundary**. Every function is
in exactly one of two tiers, and the return type tells you which:

- **Tier A — EXACT over Q.** Every closed form whose result is a finite rational
  function of rational inputs is computed on [`BigRational`](bigrational.md) and is
  **exact and complete**. These return `Result<BigRational>`; the only rounding is
  the caller's final quantize to a [`BigDecimal`](bigdecimal.md) money amount.
- **Tier B — NUMERICAL, with a stated tolerance.** Functions whose answer is an
  algebraic irrational (an nth root, a logarithm) or the root of a transcendental
  equation are computed on `double` and return `Result<double>`. Exact iterates
  would be *wasted exactness* on an irrational answer (and would blow up rational
  denominators every Newton step), so root-finding runs on `double` via a
  **bracketed Brent method** that returns `MathError::not_converged` rather than a
  silently-arbitrary root.
- **DB is HYBRID.** Its rate is Excel-mandated to be **rounded to three decimals**
  (a numeric nth root, then an explicit `BigDecimal::quantize(3, half_up)`); the
  depreciation schedule is then exact given that rounded rate.

**The Excel-compat caveat, stated plainly and never hidden:** Tier-A exact mode
will **disagree with Excel in low-order digits, because Excel computes the same
quantities in IEEE double**. The honest contract is *"exact over Q — more accurate
than Excel"*, and never "exact" and "Excel-bit-identical" in the same breath. Sign
and timing conventions do follow Excel (a cash *outflow* is negative;
`PaymentTiming::begin` is Excel type 1), and the `Date` type uses the Excel serial
epoch **without** the Excel 1900-leap-year bug — a documented, deliberate
divergence (see `Date` below).

### Per-function tier table

| Tier | Functions | Return | Why |
| :--- | :--- | :--- | :--- |
| **A (exact over Q)** | `fv`, `pv`, `pmt`, `ipmt`, `ppmt`, `cumipmt`, `cumprinc`, `npv`, `fvschedule`, `effect`, `sln`, `syd`, `ddb`, `vdb`, `ispmt`, `growing_annuity_pv`, `perpetuity_pv`, `growing_perpetuity_pv`, `dollarde`, `dollarfr`, `year_fraction`, `coupon_num`, `CashflowSchedule::net_present_value` | `Result<BigRational>` (or `Result<std::int64_t>` for `coupon_num`) | Finite rational function of rational inputs. |
| **B (numerical)** | `nper`, `rate`, `irr`, `xnpv`, `xirr`, `mirr`, `nominal`, `rri`, `pduration`, `db`, `bond_price`, `bond_yield`, `bond_duration`, `bond_mduration`, `bond_convexity`, `disc`, `intrate`, `received`, `accrued_interest`, `price_disc`, `yield_disc`, `tbill_price`, `tbill_yield`, `tbill_eq`, the swaps, `CashflowSchedule::internal_rate_of_return` / `extended_irr` | `Result<double>` | An nth root, a logarithm, or a transcendental root. |

All failure rides the railway (`Result<T>` / `MathError`); nothing throws.

```cpp
import nimblecas.finance;
```

Depends on [`core`](core.md), [`bigint`](bigint.md),
[`bigrational`](bigrational.md), and [`bigdecimal`](bigdecimal.md) (the exact
substrate and the money quantizer at the boundary). Everything lives in namespace
`nimblecas::finance`.

## Conventions — timing, day counts, and dates

The Excel TVM identity, for per-period rate `r`, integer term `n`, level payment
`pmt`, present value `pv`, future value `fv`, timing `t ∈ {0, 1}`:

```
pv·(1+r)^n + pmt·(1 + r·t)·((1+r)^n − 1)/r + fv = 0        (r ≠ 0)
pv + pmt·n + fv = 0                                        (r == 0, the limit)
```

Each Tier-A TVM function solves this identity for one unknown in exact rational
arithmetic.

| Type | Definition |
| :--- | :--- |
| `PaymentTiming` | `enum class { end = 0, begin = 1 }` — end-of-period (ordinary annuity, Excel type 0) or start-of-period (annuity due, Excel type 1). |
| `DayCount` | `enum class { thirty_360, actual_actual, actual_360, actual_365, thirty_360e }` — Excel bases 0–4. Year fractions are **exact rationals** because calendar dates are integers; only a subsequent `(1+y)^t` with fractional `t` is numerical. `actual_actual` uses the act/act(ISDA) `(days·4)/1461` convention to stay exact and monotone. |

### `Date` — a serial-day calendar date

`struct Date { std::int64_t serial; }` — days since 1899-12-31, so `serial == 1`
is 1900-01-01 (Excel-compatible epoch). Differences are exact integers. The
calendar is **proleptic Gregorian throughout — the astronomical count, WITHOUT
Excel's 1900-leap-year bug** (a documented divergence: Excel treats 1900 as a leap
year, this does not).

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `of` | `static auto of(int year, int month, int day) -> Result<Date>` | Validates the calendar; bad `y/m/d` → `domain_error`. |
| `ymd` | `auto ymd() const -> std::tuple<int, int, int>` | Back to `(year, month, day)`. |
| `operator<=>` / `operator==` | compare on `serial` | Chronological order. |
| `year_fraction` | `auto year_fraction(const Date& start, const Date& end, DayCount basis) -> Result<BigRational>` | Exact year-fraction between two dates under a day-count convention. `end < start` → `domain_error`. |

## Tier A — time value of money (exact over Q)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fv` | `auto fv(const BigRational& rate, std::int64_t nper, const BigRational& pmt, const BigRational& pv, PaymentTiming timing = end) -> Result<BigRational>` | Future value. Never fails (all operations total over Q). |
| `pv` | `auto pv(const BigRational& rate, std::int64_t nper, const BigRational& pmt, const BigRational& fv, PaymentTiming timing = end) -> Result<BigRational>` | Present value. |
| `pmt` | `auto pmt(const BigRational& rate, std::int64_t nper, const BigRational& pv, const BigRational& fv, PaymentTiming timing = end) -> Result<BigRational>` | Level payment. `division_by_zero` only in the degenerate `nper == 0` case. |
| `ipmt` / `ppmt` | `auto ipmt(const BigRational& rate, std::int64_t per, std::int64_t nper, const BigRational& pv, const BigRational& fv, PaymentTiming timing = end) -> Result<BigRational>` (and `ppmt`, same shape) | Interest / principal part of the payment in period `per`. Follows the LibreOffice/Excel reference decomposition. `per < 1` or `per > nper` → `domain_error`. |
| `cumipmt` / `cumprinc` | `auto cumipmt(const BigRational& rate, std::int64_t nper, const BigRational& pv, std::int64_t start, std::int64_t end, PaymentTiming timing = end) -> Result<BigRational>` (and `cumprinc`) | Cumulative interest / principal between periods `start..end` inclusive, with `fv == 0` as in Excel. `start < 1`, `end < start`, or `end > nper` → `domain_error`. |
| `ispmt` | `auto ispmt(const BigRational& rate, std::int64_t per, std::int64_t nper, const BigRational& pv) -> Result<BigRational>` | Lotus-compatible straight-line interest: `−pv·rate·(nper−per)/nper`. `nper == 0` → `division_by_zero`; `per` out of range → `domain_error`. |
| `growing_annuity_pv` | `auto growing_annuity_pv(const BigRational& rate, const BigRational& growth, std::int64_t nper, const BigRational& first_payment) -> Result<BigRational>` | PV of `nper` payments growing at `growth`. Handles the `rate == growth` limit `n·C₁/(1+r)` exactly. `nper < 0` → `domain_error`; `1+rate == 0` → `division_by_zero`. |
| `perpetuity_pv` | `auto perpetuity_pv(const BigRational& rate, const BigRational& payment) -> Result<BigRational>` | `payment / rate`. `rate == 0` → `division_by_zero`. |
| `growing_perpetuity_pv` | `auto growing_perpetuity_pv(const BigRational& rate, const BigRational& growth, const BigRational& payment) -> Result<BigRational>` | `payment / (rate − growth)`; requires `rate > growth` (else the series diverges) → `domain_error`. |

## Tier A — cashflow analysis and rate conversion (exact)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `npv` | `auto npv(const BigRational& rate, std::span<const BigRational> cashflows) -> Result<BigRational>` | Net present value; the first value is discounted **one** period (Excel convention). `rate == −1` → `division_by_zero`. |
| `fvschedule` | `auto fvschedule(const BigRational& principal, std::span<const BigRational> rates) -> BigRational` | Compound a principal through a series of period rates. Total (returns a bare `BigRational`). |
| `effect` | `auto effect(const BigRational& nominal_rate, std::int64_t npery) -> Result<BigRational>` | Effective annual rate `(1 + nominal/npery)^npery − 1`, exact. `npery < 1` → `domain_error`. |

## Tier B — numerical solvers (root-finding, roots, logs)

All bracketed roots use a deterministic Brent method that expands a bounded search
window outward from a guess; failure to bracket a sign change is
`MathError::not_converged`, **never** a fabricated root.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `nper` | `auto nper(double rate, double pmt, double pv, double fv, PaymentTiming timing = end) -> Result<double>` | Number of periods via logarithms (closed form, so `not_converged` is impossible). `domain_error` when no real solution (log of a non-positive ratio); `division_by_zero` in the degenerate `rate == 0, pmt == 0` case. |
| `rate` | `auto rate(std::int64_t nper, double pmt, double pv, double fv, PaymentTiming timing = end, double guess = 0.1) -> Result<double>` | Per-period rate solving the TVM identity, by bracketed root-finding. `not_converged` if no bracket. |
| `irr` | `auto irr(std::span<const double> values, double guess = 0.1) -> Result<double>` | IRR of period-0..N cashflows (`values[0]` is the period-0 flow). Descartes sign-change hazards surface as `not_converged`. Fewer than 2 values → `domain_error`. |
| `xnpv` / `xirr` | `auto xnpv(double rate, std::span<const double> values, std::span<const Date> dates) -> Result<double>`; `auto xirr(std::span<const double> values, std::span<const Date> dates, double guess = 0.1) -> Result<double>` | Date-aware NPV / IRR with actual/365 fractional exponents. Length mismatch / too few points → `domain_error`. |
| `mirr` | `auto mirr(std::span<const double> values, double finance_rate, double reinvest_rate) -> Result<double>` | Modified IRR with distinct finance and reinvestment rates. No outflow or no inflow → `division_by_zero`; a non-positive ratio → `domain_error`. |
| `nominal` | `auto nominal(double effect_rate, std::int64_t npery) -> Result<double>` | Inverse of `effect` (an npery-th root). `npery < 1` or `effect_rate <= −1` → `domain_error`. |
| `rri` | `auto rri(double nper, double pv, double fv) -> Result<double>` | Equivalent per-period growth rate `(fv/pv)^(1/nper) − 1`. |
| `pduration` | `auto pduration(double rate, double pv, double fv) -> Result<double>` | Periods to reach `fv` from `pv` at fixed `rate`. Non-positive `rate`/`pv`/`fv` → `domain_error`. |
| `db` | `auto db(double cost, double salvage, std::int64_t life, std::int64_t period, std::int64_t month = 12) -> Result<double>` | **Hybrid** fixed-declining-balance: rate `= 1 − (salvage/cost)^(1/life)` rounded to 3 decimals (half_up), then an exact schedule. `cost <= 0`, `life <= 0`, `period < 1` → `domain_error`. |

## Depreciation (exact over Q, except DB above)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `sln` | `auto sln(const BigRational& cost, const BigRational& salvage, std::int64_t life) -> Result<BigRational>` | Straight-line per period. `life <= 0` → `division_by_zero`. |
| `syd` | `auto syd(const BigRational& cost, const BigRational& salvage, std::int64_t life, std::int64_t per) -> Result<BigRational>` | Sum-of-years'-digits for period `per`. `life <= 0` → `division_by_zero`; `per` out of `[1, life]` → `domain_error`. |
| `ddb` | `auto ddb(const BigRational& cost, const BigRational& salvage, std::int64_t life, std::int64_t per, const BigRational& factor = 2) -> Result<BigRational>` | Double- (or `factor`-) declining balance, capped so book value never drops below salvage. Same guards as `syd`. |
| `vdb` | `auto vdb(const BigRational& cost, const BigRational& salvage, std::int64_t life, std::int64_t start_period, std::int64_t end_period, const BigRational& factor = 2, bool no_switch = false) -> Result<BigRational>` | Variable declining balance from `start_period` to `end_period`, optionally switching to straight line when that deducts more (Excel default). `life <= 0` → `division_by_zero`; a bad period range → `domain_error`. |

## Dollar-fraction conversion (exact over Q)

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `dollarde` | `auto dollarde(const BigRational& fractional_dollar, std::int64_t fraction) -> Result<BigRational>` | A "fractional dollar" `1.02` read as `1 + 2/fraction` → decimal. `fraction < 1` → `domain_error`. |
| `dollarfr` | `auto dollarfr(const BigRational& decimal_dollar, std::int64_t fraction) -> Result<BigRational>` | The inverse of `dollarde`. `fraction < 1` → `domain_error`. |

## Fixed income (numerical)

Prices are per 100 face. Coupon frequency must be 1, 2, or 4.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `bond_price` | `auto bond_price(const Date& settlement, const Date& maturity, double coupon_rate, double yield, double redemption, int frequency, DayCount basis) -> Result<double>` | Clean price per 100 face, coupons discounted at their fractional-period offsets, minus accrued interest. |
| `bond_yield` | `auto bond_yield(const Date& settlement, const Date& maturity, double coupon_rate, double clean_price, double redemption, int frequency, DayCount basis) -> Result<double>` | Yield to maturity: bracketed root of `price(yield) = clean_price`. |
| `bond_duration` / `bond_mduration` | `auto bond_duration(const Date& settlement, const Date& maturity, double coupon_rate, double yield, int frequency, DayCount basis) -> Result<double>` (and `bond_mduration`) | Macaulay and modified duration, in years. |
| `bond_convexity` | `auto bond_convexity(...) -> Result<double>` (same params as `bond_duration`) | Convexity in years², the second-order price/yield sensitivity. |
| `coupon_num` | `auto coupon_num(const Date& settlement, const Date& maturity, int frequency) -> Result<std::int64_t>` | Number of coupons between settlement and maturity (Excel COUPNUM). Exact integer. |
| `accrued_interest` | `auto accrued_interest(const Date& last_coupon, const Date& settlement, double coupon_rate, int frequency, DayCount basis, double par = 100.0) -> Result<double>` | Accrued interest per 100 face from the last coupon to settlement (single period). |
| `disc` / `intrate` / `received` | see source | Discount-security helpers (simple ratios over the day-count year fraction). |
| `price_disc` / `yield_disc` | see source | Discounted (zero-coupon) securities, per 100 face. |
| `tbill_price` / `tbill_yield` / `tbill_eq` | e.g. `auto tbill_price(const Date& settlement, const Date& maturity, double discount_rate) -> Result<double>` | T-bills on the actual/360 discount basis. `tbill_eq` uses the ≤182-day simple bond-equivalent form, which **diverges from Excel for longer bills** — a documented convention divergence. Maturities beyond 366 days → `domain_error`. |

## Swaps (numerical, discount-factor based)

The caller supplies per-period accrual factors, discount factors to each payment
date, and (for the floating leg) forward rates. A length mismatch or empty span →
`domain_error`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `interest_rate_swap_value` | `auto interest_rate_swap_value(double notional, double fixed_rate, std::span<const double> accruals, std::span<const double> discount_factors, std::span<const double> forward_rates) -> Result<double>` | Payer swap (pay fixed, receive floating): `notional · Σ dfᵢ·accrualᵢ·(forwardᵢ − fixed_rate)`. |
| `swap_par_rate` | `auto swap_par_rate(std::span<const double> accruals, std::span<const double> discount_factors, std::span<const double> forward_rates) -> Result<double>` | Fair fixed rate making the swap value zero. Zero annuity → `division_by_zero`. |
| `currency_swap_value` | `auto currency_swap_value(double domestic_leg_pv, double foreign_leg_pv, double fx_spot) -> Result<double>` | `domestic_leg_pv − fx_spot · foreign_leg_pv`. |

## Fluent builders

| Type | Purpose |
| :--- | :--- |
| `CashflowSchedule` | Built by chaining `.at(date, amount)` (dated) or `.flow(amount)` (period-indexed). `net_present_value(rate)` is Tier A exact (flow `i` discounted `i` periods); `internal_rate_of_return(guess)` and `extended_irr(guess)` are Tier B numerical. `extended_irr` needs dated flows → `domain_error` otherwise. Static factory `create()`. |
| `TvmProblem` | Set the knowns fluently (`.rate` / `.nper` / `.pmt` / `.present_value` / `.future_value` / `.timing`), then solve one unknown: `.solve_fv()` / `.solve_pv()` / `.solve_pmt()` return the raw exact `Result<BigRational>`; `.solve_*_money(scale, mode)` quantize to a `BigDecimal` at the boundary. Static factory `create()`. |

## Error model

| Condition | Error |
| :--- | :--- |
| `pmt`/`ispmt` with `nper == 0`; `sln`/`syd`/`ddb`/`vdb` with `life <= 0`; `npv` with `rate == −1`; `perpetuity_pv` with `rate == 0`; swap/duration/convexity with a zero denominator | `MathError::division_by_zero` |
| `ipmt`/`ppmt`/`cumipmt`/`cumprinc`/`syd`/`ddb`/`ispmt` with a period out of range; `effect`/`nominal` with `npery < 1`; `growing_perpetuity_pv` with `rate <= growth`; `dollarde`/`dollarfr` with `fraction < 1`; `Date::of` on a bad calendar date; `year_fraction` with `end < start`; bad frequency; span length mismatch; a numerical objective with no real solution | `MathError::domain_error` |
| `rate`/`irr`/`xirr`/`bond_yield` when no sign change can be bracketed | `MathError::not_converged` |
| A scale that overflows `int32` when quantizing to money | `MathError::overflow` (propagated from `bigdecimal`) |

`fv`, `pv`, and `fvschedule` are total over Q and never error on well-formed
rational input.

## Worked examples

```cpp
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;
import nimblecas.finance;
using namespace nimblecas;
using namespace nimblecas::finance;

auto q  = [](std::string_view s) { return BigRational::from_string(s).value(); };
auto qi = [](std::int64_t v)     { return BigRational::from_int(v); };

// TIER A — exact identities (hold on the nose, not "to tolerance").
fv(qi(0), 10, qi(-100), qi(0)).value() == qi(1000);   // FV(0,10,-100,0) == 1000
fv(qi(1), 1, qi(0), qi(100)).value()  == qi(-200);    // 100 doubling at 100% -> -200

// FV then PV recovers the present value exactly (5%/period, 12 periods).
const auto f    = fv(q("1/20"), 12, qi(-100), qi(1000)).value();
pv(q("1/20"), 12, qi(-100), f).value() == qi(1000);   // true

// NPV at 100% of [100, 100]: 100/2 + 100/4 == 75.
const std::array<BigRational, 2> cf{qi(100), qi(100)};
npv(qi(1), cf).value() == qi(75);                      // true

// FVSCHEDULE: 100 * 1.1 * 1.2 == 132.
const std::array<BigRational, 2> rates{q("1/10"), q("1/5")};
fvschedule(qi(100), rates) == qi(132);                 // true

// Depreciation and rate conversion, exact.
sln(qi(1000), qi(100), 5).value() == qi(180);          // SLN == 180
effect(q("1/10"), 2).value() == q("41/400");           // (21/20)^2 - 1 == 0.1025

// TIER B — numerical, recovered to tolerance.
const std::array<double, 2> flows{-100.0, 110.0};
irr(flows).value();                                    // ≈ 0.10
rate(2, 0.0, -100.0, 121.0).value();                   // ≈ 0.10 (100 -> 121 over 2 periods)
nper(0.10, 0.0, -100.0, 121.0).value();                // ≈ 2

// Fluent TvmProblem quantizes to money at the boundary.
auto prob = TvmProblem::create().rate(q("1/20")).nper(12)
                .pmt(qi(-100)).present_value(qi(1000));
prob.solve_fv().value();                               // raw exact BigRational
prob.solve_fv_money(2, Rounding::half_even).value();   // BigDecimal, rounded to cents
```

## See also

- [`nimblecas.bigrational`](bigrational.md) — the exact Tier-A substrate.
- [`nimblecas.bigdecimal`](bigdecimal.md) — the money quantizer the fluent layer
  rounds to at the boundary.
- [`nimblecas.currency`](currency.md) — exact FX rates, cross rates, and forwards,
  building `Money` on the same decimal core.
- [`nimblecas.pricing`](pricing.md) — the derivatives-pricing sibling (numerical
  by nature).
- [Documentation hub](../Index.md)
