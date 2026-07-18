# `nimblecas.yieldcurve` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/yieldcurve/yieldcurve.cppm`

The fixed-income **term-structure** layer: a pillar-based zero (spot) curve that discounts
arbitrary cashflows, the four classical term-structure conversions, sequential bootstrapping
of a zero curve from bond prices or yields, par-yield ↔ zero conversion, parametric
Nelson-Siegel and Svensson fits, and a Hull-White one-factor trinomial short-rate lattice
calibrated to an initial discount curve. Where [`nimblecas.finance`](finance.md) prices *one*
instrument against a flat scalar yield, this module builds and manipulates the whole **curve**.

This is the **numerical tier** throughout: times are years (`double`), rates and discount
factors are `double`, and every fallible entry point returns `Result<T>`. A NaN/inf is
**never** returned as a value — an ill-posed or singular problem yields
`MathError::domain_error`, and an iterative fit that exhausts its bounded budget without
meeting tolerance yields `MathError::not_converged`. Every user-supplied size is bounded
(`kMaxPillars`, `kMaxSteps`, `kMaxJmax`) so a hostile input count cannot drive an
allocation/CPU DoS. The curve does **no extrapolation**: evaluating outside `[t_first, t_last]`
is a `domain_error`, never a silent flat guess.

```cpp
import nimblecas.yieldcurve;
```

Depends on [`core`](core.md) (`Result` / `MathError`), [`finance`](finance.md), and
[`bigrational`](bigrational.md). The module deliberately does **not** reinvent a calendar: the
optional date bridge (`times_from_dates`) reuses `finance::Date` / `finance::DayCount` /
`finance::year_fraction` (whose `Result<BigRational>` is taken to `double` via `.to_double()`).
Everything lives in namespace `nimblecas::yieldcurve`; all entry points are `[[nodiscard]]`.

## Conventions

Two enums fix how values are interpolated and compounded.

```cpp
enum class Interp : std::uint8_t { linear_zero, loglinear_df };
//   linear_zero  — the zero (spot) rate is linear in t between pillars.
//   loglinear_df — ln(discount factor) is linear in t between pillars (the standard
//                  bootstrapped-curve choice); equivalently DF is log-linear.

enum class Compounding : std::uint8_t { continuous, annual, semiannual, k_per_year };
//   continuous  : DF = exp(-z t),        z = -ln(DF)/t     (m = 0)
//   annual      : DF = (1+z)^(-t),                          m = 1
//   semiannual  : DF = (1+z/2)^(-2t),                       m = 2
//   k_per_year  : DF = (1+z/k)^(-k t),                      m = k  (the `freq` argument)
```

For `k_per_year` the `freq` argument is the periods per year `k`; it must lie in `[1, 366]`
(any other value → `domain_error`). For the other three conventions `freq` is ignored.

## API

### Point conversions (free functions)

Single zero ↔ discount-factor conversions, so a caller need not build a whole `Curve`. The
two are exact inverses of one another on their valid domain.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `discount_from_zero` | `auto discount_from_zero(double zero, double t, Compounding comp, int freq = 1) -> Result<double>` | Discount factor for zero rate `zero` at time `t` (years). `t == 0` → `1`. `t < 0`, a non-positive periodic base (`1 + z/m ≤ 0`), or a non-finite input → `domain_error` (the honest result would be complex/non-finite). |
| `zero_from_discount` | `auto zero_from_discount(double df, double t, Compounding comp, int freq = 1) -> Result<double>` | Zero rate implied by discount factor `df` (> 0) at time `t` (> 0) under `comp`/`freq`. `df ≤ 0` or `t ≤ 0` → `domain_error`. |

### The `Curve` object

A zero-rate term structure sampled at strictly increasing positive pillar times, with the
interpolation and compounding conventions fixed at construction.

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| `Curve::create` | `static auto create(std::vector<double> times, std::vector<double> zeros, Interp interp, Compounding comp, int freq = 1) -> Result<Curve>` | Build from parallel pillar-time / zero-rate arrays. `domain_error` on: empty input, size mismatch, more than `kMaxPillars` pillars, a non-finite entry, a non-positive first time, non-strictly-increasing times, or an invalid `k_per_year` frequency. |
| `discount_factor` | `auto discount_factor(double t) const -> Result<double>` | `DF(t)` for `t ∈ [t_first, t_last]`. Out of range → `domain_error` (a tiny endpoint tolerance keeps a node time recomputed by arithmetic in range). |
| `zero_rate` | `auto zero_rate(double t) const -> Result<double>` | Zero (spot) rate `z(t)` under the curve's compounding. Consistent with `discount_factor`: `z(t) = zero_from_discount(DF(t), t)`. Out of range → `domain_error`. |
| `forward_rate` | `auto forward_rate(double t1, double t2) const -> Result<double>` | Forward over `[t1, t2]` (both in range). Continuous → `ln(DF(t1)/DF(t2))/(t2−t1)`; periodic-`m` → `m·((DF(t1)/DF(t2))^(1/(m·dt)) − 1)`. `t2 ≤ t1` → `domain_error`. |
| `pillar_times` | `auto pillar_times() const noexcept -> std::span<const double>` | The pillar times. |
| `zero_rates` | `auto zero_rates() const noexcept -> std::span<const double>` | The pillar zero rates. |
| `interpolation` | `auto interpolation() const noexcept -> Interp` | The fixed interpolation convention. |
| `compounding` | `auto compounding() const noexcept -> Compounding` | The fixed compounding convention. |
| `frequency` | `auto frequency() const noexcept -> int` | The `freq` argument as supplied at construction. |

### Term-structure conversions on pillar arrays

Whole-array conversions over a shared strictly-increasing positive time grid. `zero2disc` /
`disc2zero` are exact inverses, as are `zero2fwd` / `fwd2zero`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `zero2disc` | `auto zero2disc(std::span<const double> times, std::span<const double> zeros, Compounding comp, int freq = 1) -> Result<std::vector<double>>` | Discount factors from zero rates at each pillar time. |
| `disc2zero` | `auto disc2zero(std::span<const double> times, std::span<const double> discs, Compounding comp, int freq = 1) -> Result<std::vector<double>>` | Zero rates from discount factors (exact inverse of `zero2disc`). |
| `zero2fwd` | `auto zero2fwd(std::span<const double> times, std::span<const double> zeros, Compounding comp, int freq = 1) -> Result<std::vector<double>>` | Interval forwards: element `i` is the forward over `(t_{i−1}, t_i]` with `t_{−1} = 0`, so element `0` is the spot rate to `t_0`. |
| `fwd2zero` | `auto fwd2zero(std::span<const double> times, std::span<const double> fwds, Compounding comp, int freq = 1) -> Result<std::vector<double>>` | Zero rates from those interval forwards (exact inverse of `zero2fwd`). |

### Bootstrapping from coupon-bond instruments

Sequential bootstrap of a zero `Curve` from bond prices or yields. Instruments must have
strictly increasing maturities. **Documented, enforced assumption:** every *non-final* coupon
date of each bond must fall on or before an already-bootstrapped maturity, so its discount
factor is known by interpolation on the curve built so far; otherwise the system is
underdetermined and the result is `domain_error` rather than an invented value. When coupon
dates coincide with pillar times (the textbook grid), re-pricing the inputs reproduces their
prices exactly.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `zbtprice` | `auto zbtprice(std::span<const CouponBond> bonds, std::span<const double> prices, Compounding comp, int freq = 1, Interp interp = Interp::loglinear_df) -> Result<Curve>` | Bootstrap from dirty bond **prices** (PV of *all* cashflows, per `face`). |
| `zbtyield` | `auto zbtyield(std::span<const CouponBond> bonds, std::span<const double> yields, Compounding comp, int freq = 1, Interp interp = Interp::loglinear_df) -> Result<Curve>` | Bootstrap from bond **yields to maturity**: each yield is converted to a dirty price (cashflows discounted at that bond's own periodic yield; a zero-coupon bill is priced annually) and then `zbtprice` is applied. Same assumptions as `zbtprice`. |

### Par-yield ↔ zero conversion

The pillar times are treated as the coupon dates of par bonds paying `couponfreq` times per
year (one coupon per pillar interval); `comp`/`freq` is the compounding used to *quote* the
resulting zero rates. `par2zero` and `zero2par` are exact inverses. `couponfreq` must lie in
`[1, 366]` (else `domain_error`).

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `par2zero` | `auto par2zero(std::span<const double> times, std::span<const double> par_yields, Compounding comp, int freq = 1, int couponfreq = 1) -> Result<std::vector<double>>` | Zero rates from par (swap) yields by sequential DF bootstrap `DF_i = (1 − (p_i/f)·Σ_{k<i} DF_k) / (1 + p_i/f)`, then quoted as zero rates. |
| `zero2par` | `auto zero2par(std::span<const double> times, std::span<const double> zeros, Compounding comp, int freq = 1, int couponfreq = 1) -> Result<std::vector<double>>` | Par yields from zero rates: `p_i = f·(1 − DF_i) / Σ_{k≤i} DF_k`. |

### Parametric fits — Nelson-Siegel and Svensson

Both fits are **linear in the loadings once the decay(s) are fixed**, so each nests a linear
least-squares solve (via normal equations with a small Gaussian-elimination dense solve)
inside a bounded search over the decay parameter(s): a coarse log-spaced grid plus
golden-section refine. A configuration that yields no solvable, finite least-squares system
anywhere in the search → `not_converged`.

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `fit_nelson_siegel` | `auto fit_nelson_siegel(std::span<const double> times, std::span<const double> zeros) -> Result<NelsonSiegel>` | Fit a Nelson-Siegel curve to observed `(time, zero-rate)` pairs. Requires **≥ 4 points**, all times `> 0` and finite (else `domain_error`); size mismatch or more than `kMaxPillars` points → `domain_error`. No solvable τ → `not_converged`. |
| `fit_svensson` | `auto fit_svensson(std::span<const double> times, std::span<const double> zeros) -> Result<Svensson>` | Fit a Svensson curve. Requires **≥ 6 points**, otherwise as `fit_nelson_siegel`. |

The fitted structs are themselves evaluable:

| Member | Signature | Behavior |
| :--- | :--- | :--- |
| `NelsonSiegel::zero_rate` | `auto zero_rate(double t) const -> Result<double>` | `z(t) = β0 + β1·(1−e^{−u})/u + β2·((1−e^{−u})/u − e^{−u})`, `u = t/τ`. `t < 0`, non-finite `t`, `τ ≤ 0`, or a non-finite result → `domain_error`. |
| `NelsonSiegel::discount_factor` | `auto discount_factor(double t, Compounding comp, int freq = 1) const -> Result<double>` | `discount_from_zero(zero_rate(t), t, comp, freq)`. |
| `Svensson::zero_rate` | `auto zero_rate(double t) const -> Result<double>` | Adds a second hump: `+ β3·((1−e^{−u2})/u2 − e^{−u2})`, `u1 = t/τ1`, `u2 = t/τ2`. `τ1 ≤ 0` or `τ2 ≤ 0` (as well as bad `t`) → `domain_error`. |
| `Svensson::discount_factor` | `auto discount_factor(double t, Compounding comp, int freq = 1) const -> Result<double>` | `discount_from_zero(zero_rate(t), t, comp, freq)`. |

### Hull-White trinomial short-rate lattice

Short rate `dr = (θ(t) − a·r) dt + σ dW`, discretised on Hull's symmetric trinomial tree.
Stage 1 builds the mean-reverting tree for the zero-drift process; stage 2 displaces each time
slice by `alpha_i` via **forward induction on Arrow-Debreu (state) prices** so the tree
*exactly* reprices the input discount curve. Consequently the lattice zero-coupon price
`discount(i·Δt)` equals the curve's `DF(i·Δt)` by construction — the self-consistency oracle
used in the tests.

| Function / member | Signature | Behavior |
| :--- | :--- | :--- |
| `build_hull_white` | `auto build_hull_white(const Curve& curve, double a, double sigma, double dt, int steps) -> Result<HullWhiteLattice>` | Calibrate to `curve` with mean reversion `a > 0`, vol `sigma > 0`, step `dt > 0`, and `steps` steps. `steps·dt` must be within the curve's range. `domain_error` on: `a ≤ 0`, `sigma ≤ 0`, `dt ≤ 0`, `steps` outside `[1, kMaxSteps]`, derived half-width `jmax` outside `[1, kMaxJmax]` (`a·dt` too small explodes `jmax`), a negative branching probability from an out-of-range `(a, dt)`, or a curve `discount_factor` failure. |
| `HullWhiteLattice::discount` | `auto discount(double T) const -> Result<double>` | Zero-coupon bond price to a node time `T = n·dt` (`1 ≤ n ≤ steps`). `T` must land on a node (within tolerance) and within the built horizon; otherwise `domain_error`. |
| `HullWhiteLattice::bond_price` | `auto bond_price(std::span<const double> times, std::span<const double> cashflows) const -> Result<double>` | `Σ_k cashflows[k]·discount(times[k])`. Empty/mismatched/over-`kMaxPillars` spans → `domain_error`; propagates any `discount` error. |
| `HullWhiteLattice::dt` | `auto dt() const noexcept -> double` | The calibration step size. |
| `HullWhiteLattice::steps` | `auto steps() const noexcept -> int` | The number of steps. |
| `HullWhiteLattice::j_max` | `auto j_max() const noexcept -> int` | The tree half-width `jmax`. |

### Calendar bridge

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `times_from_dates` | `auto times_from_dates(const finance::Date& settlement, std::span<const finance::Date> pillars, finance::DayCount basis) -> Result<std::vector<double>>` | Year fractions from `settlement` to each pillar date under `basis`, suitable as the `times` argument to `Curve::create`. Dates must be strictly increasing and after settlement; anything else (empty span, more than `kMaxPillars` pillars, or `finance::year_fraction` failure) → `domain_error`. |

### Result structs

```cpp
class Curve {                       // zero-rate term structure at strictly increasing pillars
  public:
    static auto create(std::vector<double> times, std::vector<double> zeros,
                       Interp interp, Compounding comp, int freq = 1) -> Result<Curve>;
    auto discount_factor(double t) const -> Result<double>;
    auto zero_rate(double t) const -> Result<double>;
    auto forward_rate(double t1, double t2) const -> Result<double>;
    auto pillar_times()   const noexcept -> std::span<const double>;
    auto zero_rates()     const noexcept -> std::span<const double>;
    auto interpolation()  const noexcept -> Interp;
    auto compounding()    const noexcept -> Compounding;
    auto frequency()      const noexcept -> int;
};

struct CouponBond {                 // a bootstrap instrument
    double maturity{};              // years to the final cashflow (> 0)
    double coupon_rate{};           // annual coupon as a fraction of face (0 => zero-coupon)
    int    frequency{2};            // coupons per year in {1,2,4}; ignored when coupon_rate == 0
    double face{100.0};             // redemption paid with the final coupon
};

struct NelsonSiegel {
    double beta0{}, beta1{}, beta2{}, tau{1.0};
    auto zero_rate(double t) const -> Result<double>;
    auto discount_factor(double t, Compounding comp, int freq = 1) const -> Result<double>;
};

struct Svensson {
    double beta0{}, beta1{}, beta2{}, beta3{}, tau1{1.0}, tau2{3.0};
    auto zero_rate(double t) const -> Result<double>;
    auto discount_factor(double t, Compounding comp, int freq = 1) const -> Result<double>;
};

class HullWhiteLattice {            // built only by build_hull_white (its friend)
  public:
    auto discount(double T) const -> Result<double>;
    auto bond_price(std::span<const double> times,
                    std::span<const double> cashflows) const -> Result<double>;
    auto dt()    const noexcept -> double;
    auto steps() const noexcept -> int;
    auto j_max() const noexcept -> int;
};
```

DoS caps (internal `constexpr`, applied to every untrusted size):

```cpp
kMaxPillars = 100'000;   // largest pillar count for any curve / conversion / bootstrap
kMaxSteps   =  20'000;   // Hull-White lattice: node-count guard on `steps`
kMaxJmax    =   2'000;   // Hull-White lattice: guard on the derived tree half-width
```

## Error model

| Condition | Error |
| :--- | :--- |
| Empty input, size mismatch, or more than `kMaxPillars` pillars, in any curve/conversion/bootstrap entry point | `MathError::domain_error` |
| Non-finite entry, non-positive first pillar time, or non-strictly-increasing times | `MathError::domain_error` |
| `k_per_year` with `freq` outside `[1, 366]`; `couponfreq` outside `[1, 366]`; `CouponBond::frequency` not in `{1,2,4}` | `MathError::domain_error` |
| `discount_from_zero` with `t < 0`, `1 + z/m ≤ 0`, or non-finite input; `zero_from_discount` with `df ≤ 0` or `t ≤ 0` | `MathError::domain_error` |
| `Curve` evaluation outside `[t_first, t_last]` (no extrapolation); `forward_rate` with `t2 ≤ t1` | `MathError::domain_error` |
| Bootstrap: maturities not strictly increasing, a non-final coupon before the first bootstrapped pillar (underdetermined), a zero final cashflow, or a non-positive/non-finite implied final DF | `MathError::domain_error` |
| `par2zero` / `zero2par`: a non-positive discount factor or annuity along the recursion | `MathError::domain_error` |
| Nelson-Siegel / Svensson evaluation with `t < 0`, non-finite `t`, or non-positive decay; too few points or bad times for a fit | `MathError::domain_error` |
| Hull-White: `a ≤ 0`, `sigma ≤ 0`, `dt ≤ 0`, `steps ∉ [1, kMaxSteps]`, `jmax ∉ [1, kMaxJmax]`, a negative branch probability, or a node/time off the built grid | `MathError::domain_error` |
| A fit whose bounded search finds no solvable, finite least-squares system (singular normal equations everywhere, or a non-finite fitted loading) | `MathError::not_converged` |

Every numerical `Result<double>` / `Result<Curve>` is a genuine finite value on success; a NaN
or inf is never surfaced as a result.

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.finance;
import nimblecas.yieldcurve;
using namespace nimblecas;
using namespace nimblecas::yieldcurve;

// (1) Point conversions are exact inverses.
const auto df = discount_from_zero(0.03, 2.0, Compounding::continuous).value();  // e^{-0.06}
zero_from_discount(df, 2.0, Compounding::continuous).value();                    // == 0.03

// (2) A log-linear-on-DF curve; no extrapolation past the last pillar.
const auto curve = Curve::create({0.5, 1.0, 2.0, 5.0},          // pillar times (years)
                                 {0.020, 0.024, 0.028, 0.033},  // zero rates
                                 Interp::loglinear_df,
                                 Compounding::continuous).value();
curve.discount_factor(3.0).value();          // interpolated DF, 3y in range
curve.forward_rate(1.0, 2.0).value();         // 1y-forward-1y
curve.discount_factor(6.0).error();           // MathError::domain_error — past t_last = 5

// (3) Term-structure round trips.
const std::array<double, 3> t{1.0, 2.0, 3.0};
const std::array<double, 3> z{0.02, 0.025, 0.03};
const auto d  = zero2disc(t, z, Compounding::annual).value();
disc2zero(t, d, Compounding::annual).value();  // recovers z exactly
const auto fw = zero2fwd(t, z, Compounding::annual).value();
fwd2zero(t, fw, Compounding::annual).value();  // recovers z exactly

// (4) Bootstrap a curve from coupon bonds priced on the pillar grid, then reprice exactly.
const std::array<CouponBond, 2> bonds{
    CouponBond{.maturity = 1.0, .coupon_rate = 0.00, .frequency = 1, .face = 100.0},  // bill
    CouponBond{.maturity = 2.0, .coupon_rate = 0.05, .frequency = 1, .face = 100.0}}; // annual
const std::array<double, 2> prices{97.5, 101.0};
const auto boot = zbtprice(bonds, prices, Compounding::continuous).value();
boot.zero_rate(2.0).value();                   // 2y spot implied by the two instruments

// (5) Par <-> zero, exact inverses.
const auto zz = par2zero(t, {0.021, 0.026, 0.031}, Compounding::annual, 1, 1).value();
zero2par(t, zz, Compounding::annual, 1, 1).value();  // back to the par yields

// (6) Fit Nelson-Siegel (>= 4 points) and evaluate the smooth curve.
const std::array<double, 5> ft{0.5, 1.0, 2.0, 5.0, 10.0};
const std::array<double, 5> fz{0.018, 0.022, 0.027, 0.032, 0.035};
const auto ns = fit_nelson_siegel(ft, fz).value();
ns.zero_rate(3.0).value();                     // model-implied 3y spot
ns.discount_factor(3.0, Compounding::continuous).value();

// (7) Hull-White lattice calibrated to the curve reprices its own DFs by construction.
const auto hw = build_hull_white(curve, /*a=*/0.1, /*sigma=*/0.01,
                                 /*dt=*/0.5, /*steps=*/10).value();
hw.discount(2.0).value();                      // == curve.discount_factor(2.0) up to rounding
const std::array<double, 2> ct{1.0, 2.0};
const std::array<double, 2> ca{5.0, 105.0};
hw.bond_price(ct, ca).value();                 // discounts the dated cashflows on the tree

// (8) Calendar bridge — year fractions straight into Curve::create.
const auto settle = finance::Date::of(2026, 1, 15).value();
const std::array<finance::Date, 2> pillars{
    finance::Date::of(2027, 1, 15).value(), finance::Date::of(2028, 1, 15).value()};
const auto times = times_from_dates(settle, pillars, finance::DayCount::actual_365).value();
```

## See also

- [`nimblecas.finance`](finance.md) — single-instrument analytics (bond price/yield/duration,
  TVM, swaps) and the shared `Date` / `DayCount` / `year_fraction` calendar reused here.
- [Documentation hub](../Index.md)
