// NimbleCAS fixed-income analytics — sensitivities, dated cashflows, coupon-date functions,
// odd-period pricing, maturity-paying securities, z-spread, yield adapters, FRNs and amortization.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A second-tier fixed-income layer built ON TOP OF nimblecas.finance. It reuses that
// module's Date / DayCount / year_fraction (dates are integers, so year fractions are exact
// rationals) and its bond PRICE / YIELD / MDURATION primitives, and adds:
//   A. Sensitivities — DV01 / PV01 (dollar value of a basis point), effective duration by
//      ±Δy reprice, and key-rate durations (tent-weighted bumps of a discount-factor curve).
//   B. Dated cashflow schedules — cfdates / cfamounts, with optional odd (stub) first period.
//   C. Coupon-date functions — COUPPCD / COUPNCD / COUPDAYBS / COUPDAYS / COUPDAYSNC.
//   D. Odd-period pricing — ODDFPRICE / ODDFYIELD / ODDLPRICE / ODDLYIELD.
//   E. Maturity-paying securities — PRICEMAT / YIELDMAT / ACCRINTM (interest at maturity).
//   F. Z-spread — the constant continuously-compounded spread over a supplied discount curve
//      that reprices a bond to a target, by bracketed root-finding.
//   G. Yield-convention adapters — simple↔compound, street↔true (stub discounting), and
//      nominal-rate compounding-frequency conversion (annual↔semiannual and general).
//   H. FRN valuation (projected-forward + margin over a discount curve) and level-payment
//      amortization schedules.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). This module is NUMERICAL: every output
// is a discounted-cashflow limit whose discount factor (1+y/f)^t carries a fractional (hence
// transcendental) power, so nothing here claims exactness — results are Result<double> /
// Result<std::vector<...>> with a stated tolerance. The two exceptions that ARE exact are the
// integer coupon-day counts of part C (calendar differences) — they return Result<std::int64_t>.
// Nothing returns NaN or infinity: a singular divisor, an out-of-domain power, or a
// non-bracketing root yields MathError::division_by_zero / domain_error / not_converged. All
// iteration is bounded and DoS-sized inputs (> 100000 cashflows/periods, absurd tenors) are
// refused, not allocated.
//
// CONVENTION NOTE (documented divergence, never hidden). The discounting exponent for a
// cashflow at date D is year_fraction(settlement, D, basis) * frequency — the "uniform-period"
// convention. For the 30/360 family this is identical to the street/SIA grid (each period is
// exactly 180/90 days), so odd_first_price/odd_last_price with a REGULAR stub reproduce
// finance::bond_price bit-for-bit on a 30/360 basis. For actual/actual and actual/365 it is the
// consistent "true-yield" convention rather than Excel's actual/actual-in-period street count,
// and will diverge from Excel's ODDFPRICE/ODDLPRICE in the low-order digits (same class of
// divergence finance already documents for DayCount::actual_actual).

module;
#include <cassert>

export module nimblecas.fixedincome;

import std;
import nimblecas.core;
import nimblecas.finance;

export namespace nimblecas::fixedincome {

// Parallel cashflow amounts (per the face convention supplied) and their calendar dates.
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

// ===========================================================================
// A. Sensitivities.
// ===========================================================================

// DV01 (dollar value of a basis point), per 100 face: the price drop for a 1bp yield RISE,
// computed by a symmetric ±½bp reprice so DV01 ≈ modified_duration * price * 1e-4. Positive for
// an ordinary bond. Returns the underlying bond_price error on failure.
[[nodiscard]] auto dv01(const finance::Date& settlement, const finance::Date& maturity,
                        double coupon_rate, double yield, double redemption, int frequency,
                        finance::DayCount basis) -> Result<double>;

// PV01 (price value of a basis point), per 100 face: the one-sided price drop for a +1bp yield
// bump, P(y) - P(y + 1bp). Like DV01 it approximates modified_duration * price * 1e-4.
[[nodiscard]] auto pv01(const finance::Date& settlement, const finance::Date& maturity,
                        double coupon_rate, double yield, double redemption, int frequency,
                        finance::DayCount basis) -> Result<double>;

// Effective duration (years) by ±dy reprice: (P(y-dy) - P(y+dy)) / (2 * P(y) * dy). For an
// option-free bond this recovers the modified duration. dy defaults to 1bp.
[[nodiscard]] auto duration_from_price(const finance::Date& settlement,
                                       const finance::Date& maturity, double coupon_rate,
                                       double yield, double redemption, int frequency,
                                       finance::DayCount basis, double dy = 1e-4) -> Result<double>;

// Effective (parallel-shift) duration of an arbitrary cashflow set priced off a discount-factor
// curve df(t): -(P(+bump) - P(-bump)) / (2 * P0 * bump), where the whole zero curve is shifted by
// ±bump (df'(t) = df(t) * exp(-shift * t)). times are in years; amounts are absolute cashflows.
[[nodiscard]] auto effective_duration(std::span<const double> times, std::span<const double> amounts,
                                      const std::function<double(double)>& df, double bump = 1e-4)
    -> Result<double>;

// Key-rate (partial) durations: for each key tenor, bump ONLY that tenor's zero rate by ±bump
// using triangular (tent) weights that form a partition of unity over the key tenors, and
// reprice. The tents sum to 1 everywhere, so the key-rate durations sum to the effective
// duration above. key_tenors must be strictly increasing. Returns one duration per key tenor.
[[nodiscard]] auto key_rate_durations(std::span<const double> times, std::span<const double> amounts,
                                      const std::function<double(double)>& df,
                                      std::span<const double> key_tenors, double bump = 1e-4)
    -> Result<std::vector<double>>;

// ===========================================================================
// B. Dated cashflow schedule.
// ===========================================================================

// Coupon payment dates strictly after settlement through maturity (ascending), on the regular
// schedule anchored at maturity stepping back 12/frequency months.
[[nodiscard]] auto cfdates(const finance::Date& settlement, const finance::Date& maturity,
                           int frequency) -> Result<std::vector<finance::Date>>;

// Coupon + principal cashflows and their dates. Each regular coupon is face*coupon_rate/frequency;
// the final date adds `redemption`. If `issue` is supplied and precedes the first payment period,
// the first coupon is prorated over its (possibly odd/stub) accrual issue→first-payment via the
// day-count basis (a short stub pays less, a long stub more). redemption < 0 defaults to face.
[[nodiscard]] auto cfamounts(const finance::Date& settlement, const finance::Date& maturity,
                             double coupon_rate, int frequency, finance::DayCount basis,
                             double face = 100.0, double redemption = -1.0,
                             std::optional<finance::Date> issue = std::nullopt)
    -> Result<DatedCashflows>;

// ===========================================================================
// C. Coupon-date functions (exact integer day counts).
// ===========================================================================

// COUPPCD: the coupon date on or before settlement. COUPNCD: the coupon date strictly after
// settlement. Schedule anchored at maturity, stepping back 12/frequency months.
[[nodiscard]] auto coup_pcd(const finance::Date& settlement, const finance::Date& maturity,
                            int frequency) -> Result<finance::Date>;
[[nodiscard]] auto coup_ncd(const finance::Date& settlement, const finance::Date& maturity,
                            int frequency) -> Result<finance::Date>;
// COUPDAYBS: days from the previous coupon date to settlement. COUPDAYSNC: days from settlement to
// the next coupon date. COUPDAYS: days in the coupon period containing settlement. Under a 30/360
// basis a period is exactly 360/frequency days (180 semiannual); actual bases use calendar days.
// The three satisfy coup_days_bs + coup_days_nc == coup_days.
[[nodiscard]] auto coup_days_bs(const finance::Date& settlement, const finance::Date& maturity,
                                int frequency, finance::DayCount basis) -> Result<std::int64_t>;
[[nodiscard]] auto coup_days_nc(const finance::Date& settlement, const finance::Date& maturity,
                                int frequency, finance::DayCount basis) -> Result<std::int64_t>;
[[nodiscard]] auto coup_days(const finance::Date& settlement, const finance::Date& maturity,
                             int frequency, finance::DayCount basis) -> Result<std::int64_t>;

// ===========================================================================
// D. Odd-period pricing (ODDFPRICE / ODDFYIELD / ODDLPRICE / ODDLYIELD).
// ===========================================================================

// ODDFPRICE: clean price per 100 face of a bond with an odd (short or long) FIRST coupon period.
// `issue` is the dated/issue date, `first_coupon` the first (odd) coupon date; thereafter coupons
// are regular through maturity. With a REGULAR first period (issue == first_coupon - one period)
// this equals finance::bond_price on a 30/360 basis (see the module convention note).
[[nodiscard]] auto odd_first_price(const finance::Date& settlement, const finance::Date& maturity,
                                   const finance::Date& issue, const finance::Date& first_coupon,
                                   double coupon_rate, double yield, double redemption,
                                   int frequency, finance::DayCount basis) -> Result<double>;
// ODDFYIELD: yield to maturity implied by a clean price for an odd-first bond (bracketed root).
[[nodiscard]] auto odd_first_yield(const finance::Date& settlement, const finance::Date& maturity,
                                   const finance::Date& issue, const finance::Date& first_coupon,
                                   double coupon_rate, double clean_price, double redemption,
                                   int frequency, finance::DayCount basis) -> Result<double>;
// ODDLPRICE: clean price per 100 face of a bond with an odd LAST coupon period. `last_interest`
// is the last regular coupon date; the final period last_interest→maturity is odd.
[[nodiscard]] auto odd_last_price(const finance::Date& settlement, const finance::Date& maturity,
                                  const finance::Date& last_interest, double coupon_rate,
                                  double yield, double redemption, int frequency,
                                  finance::DayCount basis) -> Result<double>;
// ODDLYIELD: yield implied by a clean price for an odd-last bond (bracketed root).
[[nodiscard]] auto odd_last_yield(const finance::Date& settlement, const finance::Date& maturity,
                                  const finance::Date& last_interest, double coupon_rate,
                                  double clean_price, double redemption, int frequency,
                                  finance::DayCount basis) -> Result<double>;

// ===========================================================================
// E. Maturity-paying securities (interest paid once, at maturity).
// ===========================================================================

// PRICEMAT: clean price per 100 face of a security that pays interest at maturity, given the
// issue date, coupon rate and yield (simple discounting to maturity). Excel PRICEMAT.
[[nodiscard]] auto price_mat(const finance::Date& settlement, const finance::Date& maturity,
                             const finance::Date& issue, double coupon_rate, double yield,
                             finance::DayCount basis) -> Result<double>;
// YIELDMAT: the simple annual yield implied by a clean price (closed form; inverts price_mat).
[[nodiscard]] auto yield_mat(const finance::Date& settlement, const finance::Date& maturity,
                             const finance::Date& issue, double coupon_rate, double price,
                             finance::DayCount basis) -> Result<double>;
// ACCRINTM: accrued interest per 100 face at maturity = 100 * rate * year_fraction(issue, settle).
[[nodiscard]] auto accrint_mat(const finance::Date& issue, const finance::Date& settlement,
                               double coupon_rate, finance::DayCount basis, double par = 100.0)
    -> Result<double>;

// ===========================================================================
// F. Z-spread.
// ===========================================================================

// The constant continuously-compounded spread s such that discounting each cashflow at
// df(t)*exp(-s*t) reprices the bond to target_price: solve sum amounts_i*df(t_i)*exp(-s*t_i) ==
// target_price for s by bracketed root-finding. times in years, amounts absolute. Against a curve
// that already reprices the bond to target (e.g. the bond's own flat curve), s is ~0.
[[nodiscard]] auto z_spread(std::span<const double> times, std::span<const double> amounts,
                            const std::function<double(double)>& df, double target_price)
    -> Result<double>;

// ===========================================================================
// G. Yield-convention adapters.
// ===========================================================================

// Simple (money-market) rate over t periods/years to the equivalent compound rate and back:
// (1 + compound)^t == 1 + simple*t. This is also the street↔true relationship for one stub of
// length t (street = simple discounting of the stub, true = compound).
[[nodiscard]] auto simple_to_compound(double simple_rate, double t) -> Result<double>;
[[nodiscard]] auto compound_to_simple(double compound_rate, double t) -> Result<double>;
// Convert a nominal rate compounded from_m times/yr to the equivalent nominal compounded to_m
// times/yr: (1 + r/from_m)^from_m == (1 + r'/to_m)^to_m. from_m/to_m must be >= 1.
[[nodiscard]] auto nominal_convert(double rate, int from_m, int to_m) -> Result<double>;
// Convenience wrappers: annual-compounded nominal ↔ semiannual-compounded nominal.
[[nodiscard]] auto annual_to_semiannual_nominal(double annual_rate) -> Result<double>;
[[nodiscard]] auto semiannual_to_annual_nominal(double semi_rate) -> Result<double>;

// ===========================================================================
// H. FRN valuation and amortization.
// ===========================================================================

// Floating-rate note value: sum over periods of (forward_i + margin) * accrual_i * df_i, plus the
// principal df_last, all scaled by notional. accruals/forward_rates/discount_factors are parallel
// spans (one entry per period). A par floater (margin 0) priced off discount factors consistent
// with its own forwards values to par (notional).
[[nodiscard]] auto frn_value(std::span<const double> accruals, std::span<const double> forward_rates,
                             std::span<const double> discount_factors, double margin,
                             double notional = 100.0) -> Result<double>;

// Level-payment (fully amortizing) loan schedule for a principal at a fixed per-period rate over
// nper periods. Returns the constant payment and the per-period interest / principal / balance.
[[nodiscard]] auto amortizing_schedule(double principal, double rate_per_period, std::int64_t nper)
    -> Result<AmortizationSchedule>;

}  // namespace nimblecas::fixedincome

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::fixedincome {

// Reuse finance's calendar types throughout the implementation (they are exported from
// nimblecas.finance; the interface above already spells them out as finance::Date / DayCount).
using finance::Date;
using finance::DayCount;

namespace {

// DoS guards. kMaxPeriods bounds vector-producing loops (amortization, cashflow counts); a bond
// with > 100000 coupons (or an absurd tenor) is refused rather than allocated. kMaxCoupons bounds
// the schedule-generation walk (250 years of monthly coupons is already absurd for a bond).
inline constexpr std::int64_t kMaxPeriods = 100'000;
inline constexpr int kMaxCoupons = 4000;

[[nodiscard]] auto is_leap(int y) -> bool { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

[[nodiscard]] auto days_in_month(int y, int m) -> int {
    static constexpr std::array<int, 12> base{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m == 2 && is_leap(y)) { return 29; }
    return base[static_cast<std::size_t>(m - 1)];
}

// Add `months` (may be negative) to a date, clamping the day to the target month length. Mirrors
// finance's internal add_months so quasi-coupon anchors never drift through a short month.
[[nodiscard]] auto add_months(const Date& d, int months) -> Date {
    const auto [y, m, day] = d.ymd();
    const int total = (y * 12 + (m - 1)) + months;
    int ny = total / 12;
    int nm = total % 12;
    if (nm < 0) { nm += 12; --ny; }
    const int clamped = std::min(day, days_in_month(ny, nm + 1));
    return Date::of(ny, nm + 1, clamped).value_or(Date{d.serial});
}

[[nodiscard]] auto valid_frequency(int f) -> bool { return f == 1 || f == 2 || f == 4; }
[[nodiscard]] auto step_months(int frequency) -> int { return 12 / frequency; }

// year_fraction(a, b, basis) as a double, propagating the finance error. Requires b >= a.
[[nodiscard]] auto yf(const Date& a, const Date& b, DayCount basis) -> Result<double> {
    auto r = finance::year_fraction(a, b, basis);
    if (!r) { return make_error<double>(r.error()); }
    return r->to_double();
}

// The coupon period [pcd, ncd) containing `settlement` on the grid anchored at `anchor` stepping
// `step` months, with pcd <= settlement < ncd. The scan is bounded by kMaxCoupons each way.
[[nodiscard]] auto period_around(const Date& anchor, int step, const Date& settlement)
    -> Result<std::pair<Date, Date>> {
    if (add_months(anchor, 0).serial <= settlement.serial) {
        int k = 0;
        while (add_months(anchor, step * (k + 1)).serial <= settlement.serial) {
            if (++k > kMaxCoupons) { return make_error<std::pair<Date, Date>>(MathError::domain_error); }
        }
        return std::pair<Date, Date>{add_months(anchor, step * k), add_months(anchor, step * (k + 1))};
    }
    int k = 0;
    while (add_months(anchor, step * k).serial > settlement.serial) {
        if (-(--k) > kMaxCoupons) { return make_error<std::pair<Date, Date>>(MathError::domain_error); }
    }
    return std::pair<Date, Date>{add_months(anchor, step * k), add_months(anchor, step * (k + 1))};
}

// Ascending regular coupon dates strictly after `settlement`, up to and including `maturity`, on
// the grid anchored at `maturity` stepping back `step` months.
[[nodiscard]] auto regular_dates_after(const Date& settlement, const Date& maturity, int step)
    -> Result<std::vector<Date>> {
    std::vector<Date> dates{};
    int k = 0;
    Date c = maturity;
    while (c.serial > settlement.serial) {
        dates.push_back(c);
        if (++k > kMaxCoupons) { return make_error<std::vector<Date>>(MathError::domain_error); }
        c = add_months(maturity, -step * k);
    }
    std::ranges::reverse(dates);
    return dates;
}

// Dirty price per 100 (or absolute) of dated cashflows discounted at yield `yield` compounded
// `frequency`/yr, with the "uniform-period" exponent year_fraction(settlement, date)*frequency.
[[nodiscard]] auto dirty_price(std::span<const Date> dates, std::span<const double> amounts,
                               const Date& settlement, double yield, int frequency, DayCount basis)
    -> Result<double> {
    const double periodic = yield / static_cast<double>(frequency);
    const double base = 1.0 + periodic;
    if (base <= 0.0) { return make_error<double>(MathError::domain_error); }  // (1+y/f)^t undefined
    double price = 0.0;
    for (std::size_t i = 0; i < dates.size(); ++i) {
        if (dates[i].serial < settlement.serial) { return make_error<double>(MathError::domain_error); }
        auto t = yf(settlement, dates[i], basis);
        if (!t) { return t; }
        const double exponent = *t * static_cast<double>(frequency);
        const double pw = std::pow(base, exponent);
        if (!std::isfinite(pw) || pw == 0.0) { return make_error<double>(MathError::domain_error); }
        price += amounts[i] / pw;
    }
    if (!std::isfinite(price)) { return make_error<double>(MathError::domain_error); }
    return price;
}

// 30/360-family day count as an exact integer (mirrors finance::year_fraction's 30/360 numerator).
[[nodiscard]] auto days_thirty_360(const Date& a, const Date& b, bool european) -> std::int64_t {
    const auto [y1, m1, d1raw] = a.ymd();
    const auto [y2, m2, d2raw] = b.ymd();
    int d1 = std::min(d1raw, 30);
    int d2 = d2raw;
    if (european) {
        d2 = std::min(d2, 30);
    } else if (d2 == 31 && d1 == 30) {
        d2 = 30;
    }
    return 360LL * (y2 - y1) + 30LL * (m2 - m1) + (d2 - d1);
}

// Period-boundary days (bs = to settlement, nc = from settlement) under a basis. 30/360 uses the
// exact 30/360 count; actual bases use calendar days. Both give exact integers.
[[nodiscard]] auto period_days_bs(const Date& pcd, const Date& settlement, DayCount basis)
    -> std::int64_t {
    switch (basis) {
        case DayCount::thirty_360:  return days_thirty_360(pcd, settlement, false);
        case DayCount::thirty_360e: return days_thirty_360(pcd, settlement, true);
        default:                    return settlement.serial - pcd.serial;  // actual calendar days
    }
}

[[nodiscard]] auto period_days_nc(const Date& settlement, const Date& ncd, DayCount basis)
    -> std::int64_t {
    switch (basis) {
        case DayCount::thirty_360:  return days_thirty_360(settlement, ncd, false);
        case DayCount::thirty_360e: return days_thirty_360(settlement, ncd, true);
        default:                    return ncd.serial - settlement.serial;
    }
}

// --- Bracketed root-finder (local copies of finance's private Brent + bracket search) ----------

template <typename F>
[[nodiscard]] auto brent(F&& f, double a, double b, double tol, int max_iter) -> Result<double> {
    double fa = f(a);
    double fb = f(b);
    if (!std::isfinite(fa) || !std::isfinite(fb)) { return make_error<double>(MathError::domain_error); }
    if (fa == 0.0) { return a; }
    if (fb == 0.0) { return b; }
    if ((fa > 0.0) == (fb > 0.0)) { return make_error<double>(MathError::not_converged); }
    if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
    double c = a;
    double fc = fa;
    double d = b - a;
    double e = d;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        const double xm = 0.5 * (c - b);
        if (std::abs(xm) <= tol1 || fb == 0.0) { return b; }
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            const double s = fb / fa;
            double p = 0.0;
            double q = 0.0;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const double qq = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) { q = -q; }
            p = std::abs(p);
            if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                e = d; d = p / q;
            } else {
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }
        a = b; fa = fb;
        b += (std::abs(d) > tol1) ? d : (xm > 0.0 ? tol1 : -tol1);
        fb = f(b);
        if (!std::isfinite(fb)) { return make_error<double>(MathError::domain_error); }
        if ((fb > 0.0) == (fc > 0.0)) { c = a; fc = fa; d = b - a; e = d; }
    }
    return make_error<double>(MathError::not_converged);
}

template <typename F>
[[nodiscard]] auto bracket_and_solve(F&& f, double guess) -> Result<double> {
    const double f0 = f(guess);
    if (!std::isfinite(f0)) { return make_error<double>(MathError::domain_error); }
    if (std::abs(f0) < 1e-12) { return guess; }
    for (int i = 0; i < 128; ++i) {
        const double stepv = 1e-4 * std::exp2(static_cast<double>(i) * 0.5);
        const double lo = std::max(guess - stepv, -0.999999999);
        const double hi = guess + stepv;
        const double flo = f(lo);
        const double fhi = f(hi);
        if (std::isfinite(flo) && std::isfinite(fhi) && (flo > 0.0) != (fhi > 0.0)) {
            return brent(f, lo, hi, 1e-12, 200);
        }
    }
    return make_error<double>(MathError::not_converged);
}

// Triangular (tent) weight of key tenor `j` at time t: 1 at key_tenors[j], linear to 0 at the
// neighbours, and FLAT (==1) beyond the first/last tenor. Over strictly-increasing tenors the
// tents form a partition of unity (sum_j == 1 for every t).
[[nodiscard]] auto tent_weight(double t, std::span<const double> key_tenors, std::size_t j)
    -> double {
    const std::size_t n = key_tenors.size();
    const double kj = key_tenors[j];
    if (n == 1) { return 1.0; }
    if (j == 0) {
        if (t <= kj) { return 1.0; }
        const double right = key_tenors[1];
        return t >= right ? 0.0 : (right - t) / (right - kj);
    }
    if (j == n - 1) {
        if (t >= kj) { return 1.0; }
        const double left = key_tenors[j - 1];
        return t <= left ? 0.0 : (t - left) / (kj - left);
    }
    const double left = key_tenors[j - 1];
    const double right = key_tenors[j + 1];
    if (t <= left || t >= right) { return 0.0; }
    return t <= kj ? (t - left) / (kj - left) : (right - t) / (right - kj);
}

}  // namespace

// --- A. Sensitivities -------------------------------------------------------

auto dv01(const Date& settlement, const Date& maturity, double coupon_rate, double yield,
          double redemption, int frequency, DayCount basis) -> Result<double> {
    constexpr double half_bp = 0.5e-4;
    auto p_up = finance::bond_price(settlement, maturity, coupon_rate, yield + half_bp, redemption,
                                    frequency, basis);
    if (!p_up) { return p_up; }
    auto p_dn = finance::bond_price(settlement, maturity, coupon_rate, yield - half_bp, redemption,
                                    frequency, basis);
    if (!p_dn) { return p_dn; }
    // Central ±½bp reprice spans exactly 1bp, so -(dP/dy)*1e-4 == P(y-½bp) - P(y+½bp).
    return *p_dn - *p_up;
}

auto pv01(const Date& settlement, const Date& maturity, double coupon_rate, double yield,
          double redemption, int frequency, DayCount basis) -> Result<double> {
    auto p0 = finance::bond_price(settlement, maturity, coupon_rate, yield, redemption, frequency,
                                  basis);
    if (!p0) { return p0; }
    auto p1 = finance::bond_price(settlement, maturity, coupon_rate, yield + 1e-4, redemption,
                                  frequency, basis);
    if (!p1) { return p1; }
    return *p0 - *p1;  // one-sided price value of +1bp
}

auto duration_from_price(const Date& settlement, const Date& maturity, double coupon_rate,
                         double yield, double redemption, int frequency, DayCount basis, double dy)
    -> Result<double> {
    if (!(dy > 0.0)) { return make_error<double>(MathError::domain_error); }
    auto p0 = finance::bond_price(settlement, maturity, coupon_rate, yield, redemption, frequency,
                                  basis);
    if (!p0) { return p0; }
    if (*p0 == 0.0) { return make_error<double>(MathError::division_by_zero); }
    auto p_up = finance::bond_price(settlement, maturity, coupon_rate, yield + dy, redemption,
                                    frequency, basis);
    if (!p_up) { return p_up; }
    auto p_dn = finance::bond_price(settlement, maturity, coupon_rate, yield - dy, redemption,
                                    frequency, basis);
    if (!p_dn) { return p_dn; }
    return (*p_dn - *p_up) / (2.0 * *p0 * dy);  // effective duration in years
}

auto effective_duration(std::span<const double> times, std::span<const double> amounts,
                        const std::function<double(double)>& df, double bump) -> Result<double> {
    if (times.size() != amounts.size() || times.empty() ||
        static_cast<std::int64_t>(times.size()) > kMaxPeriods || !(bump > 0.0)) {
        return make_error<double>(MathError::domain_error);
    }
    double base = 0.0;
    double up = 0.0;
    double dn = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double d = df(times[i]);
        if (!std::isfinite(d)) { return make_error<double>(MathError::domain_error); }
        base += amounts[i] * d;
        up += amounts[i] * d * std::exp(-bump * times[i]);
        dn += amounts[i] * d * std::exp(+bump * times[i]);
    }
    if (base == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double result = -(up - dn) / (2.0 * base * bump);
    if (!std::isfinite(result)) { return make_error<double>(MathError::domain_error); }
    return result;
}

auto key_rate_durations(std::span<const double> times, std::span<const double> amounts,
                        const std::function<double(double)>& df, std::span<const double> key_tenors,
                        double bump) -> Result<std::vector<double>> {
    if (times.size() != amounts.size() || times.empty() || key_tenors.empty() ||
        static_cast<std::int64_t>(times.size()) > kMaxPeriods ||
        static_cast<std::int64_t>(key_tenors.size()) > kMaxPeriods || !(bump > 0.0)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    for (std::size_t j = 1; j < key_tenors.size(); ++j) {
        if (!(key_tenors[j] > key_tenors[j - 1])) {  // strictly increasing (partition of unity)
            return make_error<std::vector<double>>(MathError::domain_error);
        }
    }
    // Precompute base discount factors once.
    std::vector<double> dfs(times.size());
    double base = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double d = df(times[i]);
        if (!std::isfinite(d)) { return make_error<std::vector<double>>(MathError::domain_error); }
        dfs[i] = d;
        base += amounts[i] * d;
    }
    if (base == 0.0) { return make_error<std::vector<double>>(MathError::division_by_zero); }
    std::vector<double> krd(key_tenors.size());
    for (std::size_t j = 0; j < key_tenors.size(); ++j) {
        double up = 0.0;
        double dn = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            const double w = tent_weight(times[i], key_tenors, j);
            const double shift = w * bump * times[i];
            up += amounts[i] * dfs[i] * std::exp(-shift);
            dn += amounts[i] * dfs[i] * std::exp(+shift);
        }
        const double val = -(up - dn) / (2.0 * base * bump);
        if (!std::isfinite(val)) { return make_error<std::vector<double>>(MathError::domain_error); }
        krd[j] = val;
    }
    return krd;
}

// --- B. Dated cashflow schedule ---------------------------------------------

auto cfdates(const Date& settlement, const Date& maturity, int frequency)
    -> Result<std::vector<Date>> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial) {
        return make_error<std::vector<Date>>(MathError::domain_error);
    }
    return regular_dates_after(settlement, maturity, step_months(frequency));
}

auto cfamounts(const Date& settlement, const Date& maturity, double coupon_rate, int frequency,
               DayCount basis, double face, double redemption, std::optional<Date> issue)
    -> Result<DatedCashflows> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial || face < 0.0) {
        return make_error<DatedCashflows>(MathError::domain_error);
    }
    const int step = step_months(frequency);
    auto dates = regular_dates_after(settlement, maturity, step);
    if (!dates) { return make_error<DatedCashflows>(dates.error()); }
    if (dates->empty()) { return make_error<DatedCashflows>(MathError::domain_error); }
    const double redeem = redemption < 0.0 ? face : redemption;
    const double regular = face * coupon_rate / static_cast<double>(frequency);
    DatedCashflows out{};
    out.dates = *dates;
    out.amounts.assign(dates->size(), regular);
    // Odd first (stub) coupon: prorate over its accrual start → first payment. accrual_start is the
    // issue date when it falls after the prior quasi-coupon (a genuine stub), else the quasi-coupon.
    if (issue.has_value()) {
        const Date first = dates->front();
        const Date prior_quasi = add_months(first, -step);
        const Date accrual_start = issue->serial > prior_quasi.serial ? *issue : prior_quasi;
        if (accrual_start.serial > first.serial) {
            return make_error<DatedCashflows>(MathError::domain_error);  // issue after first coupon
        }
        auto frac = yf(accrual_start, first, basis);
        if (!frac) { return make_error<DatedCashflows>(frac.error()); }
        out.amounts.front() = face * coupon_rate * *frac;
    }
    out.amounts.back() += redeem;
    return out;
}

// --- C. Coupon-date functions -----------------------------------------------

auto coup_pcd(const Date& settlement, const Date& maturity, int frequency) -> Result<Date> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial) {
        return make_error<Date>(MathError::domain_error);
    }
    auto p = period_around(maturity, step_months(frequency), settlement);
    if (!p) { return make_error<Date>(p.error()); }
    return p->first;
}

auto coup_ncd(const Date& settlement, const Date& maturity, int frequency) -> Result<Date> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial) {
        return make_error<Date>(MathError::domain_error);
    }
    auto p = period_around(maturity, step_months(frequency), settlement);
    if (!p) { return make_error<Date>(p.error()); }
    // A settlement exactly on a coupon date has pcd == settlement; the "next" coupon is ncd.
    return p->second;
}

auto coup_days_bs(const Date& settlement, const Date& maturity, int frequency, DayCount basis)
    -> Result<std::int64_t> {
    auto pcd = coup_pcd(settlement, maturity, frequency);
    if (!pcd) { return make_error<std::int64_t>(pcd.error()); }
    return period_days_bs(*pcd, settlement, basis);
}

auto coup_days_nc(const Date& settlement, const Date& maturity, int frequency, DayCount basis)
    -> Result<std::int64_t> {
    auto ncd = coup_ncd(settlement, maturity, frequency);
    if (!ncd) { return make_error<std::int64_t>(ncd.error()); }
    return period_days_nc(settlement, *ncd, basis);
}

auto coup_days(const Date& settlement, const Date& maturity, int frequency, DayCount basis)
    -> Result<std::int64_t> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    // 30/360 makes every coupon period exactly 360/frequency days by construction; actual bases
    // report the calendar length of the period around settlement.
    if (basis == DayCount::thirty_360 || basis == DayCount::thirty_360e) {
        return static_cast<std::int64_t>(360 / frequency);
    }
    auto p = period_around(maturity, step_months(frequency), settlement);
    if (!p) { return make_error<std::int64_t>(p.error()); }
    return p->second.serial - p->first.serial;
}

// --- D. Odd-period pricing ---------------------------------------------------

auto odd_first_price(const Date& settlement, const Date& maturity, const Date& issue,
                     const Date& first_coupon, double coupon_rate, double yield, double redemption,
                     int frequency, DayCount basis) -> Result<double> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial ||
        first_coupon.serial <= issue.serial || maturity.serial < first_coupon.serial ||
        settlement.serial < issue.serial) {
        return make_error<double>(MathError::domain_error);
    }
    const int step = step_months(frequency);
    const double regular = 100.0 * coupon_rate / static_cast<double>(frequency);
    // Cashflow dates: first_coupon, first_coupon+step, ..., maturity — those strictly after
    // settlement. (For a well-formed bond maturity == first_coupon + k*step.)
    std::vector<Date> dates{};
    std::vector<double> amounts{};
    int k = 0;
    Date c = first_coupon;
    while (c.serial <= maturity.serial) {
        if (c.serial > settlement.serial) {
            dates.push_back(c);
            // The first payment (issue→first_coupon) is the odd/stub coupon.
            if (c.serial == first_coupon.serial) {
                auto frac = yf(issue, first_coupon, basis);
                if (!frac) { return frac; }
                amounts.push_back(100.0 * coupon_rate * *frac);
            } else {
                amounts.push_back(regular);
            }
        }
        if (++k > kMaxCoupons) { return make_error<double>(MathError::domain_error); }
        c = add_months(first_coupon, step * k);
    }
    if (dates.empty()) { return make_error<double>(MathError::domain_error); }
    amounts.back() += redemption;
    auto dirty = dirty_price(dates, amounts, settlement, yield, frequency, basis);
    if (!dirty) { return dirty; }
    // Accrued: inside the odd first period accrual runs from issue; otherwise from the regular
    // previous coupon (grid anchored at first_coupon).
    Date accrual_start = issue;
    if (settlement.serial >= first_coupon.serial) {
        auto p = period_around(first_coupon, step, settlement);
        if (!p) { return make_error<double>(p.error()); }
        accrual_start = p->first;
    }
    auto acc = yf(accrual_start, settlement, basis);
    if (!acc) { return acc; }
    return *dirty - 100.0 * coupon_rate * *acc;  // clean price
}

auto odd_first_yield(const Date& settlement, const Date& maturity, const Date& issue,
                     const Date& first_coupon, double coupon_rate, double clean_price,
                     double redemption, int frequency, DayCount basis) -> Result<double> {
    auto f = [&](double y) -> double {
        auto p = odd_first_price(settlement, maturity, issue, first_coupon, coupon_rate, y,
                                 redemption, frequency, basis);
        return p ? (*p - clean_price) : std::numeric_limits<double>::quiet_NaN();
    };
    return bracket_and_solve(f, coupon_rate > 0.0 ? coupon_rate : 0.05);
}

auto odd_last_price(const Date& settlement, const Date& maturity, const Date& last_interest,
                    double coupon_rate, double yield, double redemption, int frequency,
                    DayCount basis) -> Result<double> {
    if (!valid_frequency(frequency) || maturity.serial <= settlement.serial ||
        maturity.serial <= last_interest.serial) {
        return make_error<double>(MathError::domain_error);
    }
    const int step = step_months(frequency);
    const double regular = 100.0 * coupon_rate / static_cast<double>(frequency);
    // Regular coupon dates strictly after settlement, up to and including last_interest (grid
    // anchored at last_interest stepping back), then the odd final payment at maturity.
    std::vector<Date> dates{};
    std::vector<double> amounts{};
    int k = 0;
    Date c = last_interest;
    while (c.serial > settlement.serial) {
        dates.push_back(c);
        amounts.push_back(regular);
        if (++k > kMaxCoupons) { return make_error<double>(MathError::domain_error); }
        c = add_months(last_interest, -step * k);
    }
    std::ranges::reverse(dates);
    std::ranges::reverse(amounts);
    // Odd final coupon covers last_interest → maturity.
    auto final_frac = yf(last_interest, maturity, basis);
    if (!final_frac) { return final_frac; }
    dates.push_back(maturity);
    amounts.push_back(100.0 * coupon_rate * *final_frac + redemption);
    auto dirty = dirty_price(dates, amounts, settlement, yield, frequency, basis);
    if (!dirty) { return dirty; }
    // Accrued: within the odd final period accrual runs from last_interest; otherwise from the
    // regular previous coupon on the last_interest-anchored grid.
    Date accrual_start = last_interest;
    if (settlement.serial < last_interest.serial) {
        auto p = period_around(last_interest, step, settlement);
        if (!p) { return make_error<double>(p.error()); }
        accrual_start = p->first;
    }
    auto acc = yf(accrual_start, settlement, basis);
    if (!acc) { return acc; }
    return *dirty - 100.0 * coupon_rate * *acc;
}

auto odd_last_yield(const Date& settlement, const Date& maturity, const Date& last_interest,
                    double coupon_rate, double clean_price, double redemption, int frequency,
                    DayCount basis) -> Result<double> {
    auto f = [&](double y) -> double {
        auto p = odd_last_price(settlement, maturity, last_interest, coupon_rate, y, redemption,
                                frequency, basis);
        return p ? (*p - clean_price) : std::numeric_limits<double>::quiet_NaN();
    };
    return bracket_and_solve(f, coupon_rate > 0.0 ? coupon_rate : 0.05);
}

// --- E. Maturity-paying securities ------------------------------------------

auto price_mat(const Date& settlement, const Date& maturity, const Date& issue, double coupon_rate,
               double yield, DayCount basis) -> Result<double> {
    if (maturity.serial <= settlement.serial || settlement.serial < issue.serial) {
        return make_error<double>(MathError::domain_error);
    }
    auto life = yf(issue, maturity, basis);      // total life issue→maturity
    if (!life) { return life; }
    auto s2m = yf(settlement, maturity, basis);  // settlement→maturity
    if (!s2m) { return s2m; }
    auto i2s = yf(issue, settlement, basis);     // accrued term issue→settlement
    if (!i2s) { return i2s; }
    const double den = 1.0 + yield * *s2m;
    if (den == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double redemption_plus_interest = 100.0 * (1.0 + coupon_rate * *life);
    return redemption_plus_interest / den - 100.0 * coupon_rate * *i2s;  // clean price
}

auto yield_mat(const Date& settlement, const Date& maturity, const Date& issue, double coupon_rate,
               double price, DayCount basis) -> Result<double> {
    if (maturity.serial <= settlement.serial || settlement.serial < issue.serial) {
        return make_error<double>(MathError::domain_error);
    }
    auto life = yf(issue, maturity, basis);
    if (!life) { return life; }
    auto s2m = yf(settlement, maturity, basis);
    if (!s2m) { return s2m; }
    auto i2s = yf(issue, settlement, basis);
    if (!i2s) { return i2s; }
    if (*s2m == 0.0) { return make_error<double>(MathError::division_by_zero); }
    // Invert price_mat: dirty = price + accrued, then yield = (RPI/dirty - 1) / s2m.
    const double dirty = price + 100.0 * coupon_rate * *i2s;
    if (dirty == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double redemption_plus_interest = 100.0 * (1.0 + coupon_rate * *life);
    return (redemption_plus_interest / dirty - 1.0) / *s2m;
}

auto accrint_mat(const Date& issue, const Date& settlement, double coupon_rate, DayCount basis,
                 double par) -> Result<double> {
    auto frac = yf(issue, settlement, basis);
    if (!frac) { return frac; }
    return par * coupon_rate * *frac;
}

// --- F. Z-spread -------------------------------------------------------------

auto z_spread(std::span<const double> times, std::span<const double> amounts,
              const std::function<double(double)>& df, double target_price) -> Result<double> {
    if (times.size() != amounts.size() || times.empty() ||
        static_cast<std::int64_t>(times.size()) > kMaxPeriods) {
        return make_error<double>(MathError::domain_error);
    }
    // Precompute base discounted cashflows so the root objective is a cheap sum of exponentials.
    std::vector<double> pv0(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double d = df(times[i]);
        if (!std::isfinite(d)) { return make_error<double>(MathError::domain_error); }
        pv0[i] = amounts[i] * d;
    }
    auto f = [&](double s) -> double {
        double acc = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) { acc += pv0[i] * std::exp(-s * times[i]); }
        return acc - target_price;
    };
    return bracket_and_solve(f, 0.0);  // brent inside bounds iterations
}

// --- G. Yield-convention adapters -------------------------------------------

auto simple_to_compound(double simple_rate, double t) -> Result<double> {
    if (t == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double growth = 1.0 + simple_rate * t;
    if (growth <= 0.0) { return make_error<double>(MathError::domain_error); }  // fractional power
    return std::pow(growth, 1.0 / t) - 1.0;
}

auto compound_to_simple(double compound_rate, double t) -> Result<double> {
    if (t == 0.0) { return make_error<double>(MathError::division_by_zero); }
    if (compound_rate <= -1.0) { return make_error<double>(MathError::domain_error); }
    return (std::pow(1.0 + compound_rate, t) - 1.0) / t;
}

auto nominal_convert(double rate, int from_m, int to_m) -> Result<double> {
    if (from_m < 1 || to_m < 1) { return make_error<double>(MathError::domain_error); }
    const double mf = static_cast<double>(from_m);
    const double mt = static_cast<double>(to_m);
    const double per = 1.0 + rate / mf;
    if (per <= 0.0) { return make_error<double>(MathError::domain_error); }  // fractional power base
    // (1 + rate/from_m)^from_m == (1 + r'/to_m)^to_m  ->  r' = to_m*((...)^(from_m/to_m) - 1).
    return mt * (std::pow(per, mf / mt) - 1.0);
}

auto annual_to_semiannual_nominal(double annual_rate) -> Result<double> {
    return nominal_convert(annual_rate, 1, 2);
}

auto semiannual_to_annual_nominal(double semi_rate) -> Result<double> {
    return nominal_convert(semi_rate, 2, 1);
}

// --- H. FRN valuation and amortization --------------------------------------

auto frn_value(std::span<const double> accruals, std::span<const double> forward_rates,
               std::span<const double> discount_factors, double margin, double notional)
    -> Result<double> {
    const std::size_t n = accruals.size();
    if (n == 0 || forward_rates.size() != n || discount_factors.size() != n ||
        static_cast<std::int64_t>(n) > kMaxPeriods) {
        return make_error<double>(MathError::domain_error);
    }
    double coupons = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        coupons += (forward_rates[i] + margin) * accruals[i] * discount_factors[i];
    }
    const double value = notional * (coupons + discount_factors[n - 1]);  // + principal at last DF
    if (!std::isfinite(value)) { return make_error<double>(MathError::domain_error); }
    return value;
}

auto amortizing_schedule(double principal, double rate_per_period, std::int64_t nper)
    -> Result<AmortizationSchedule> {
    if (nper < 1 || nper > kMaxPeriods || rate_per_period <= -1.0 || !std::isfinite(principal)) {
        return make_error<AmortizationSchedule>(MathError::domain_error);
    }
    const auto n = static_cast<std::size_t>(nper);
    double payment = 0.0;
    if (rate_per_period == 0.0) {
        payment = principal / static_cast<double>(nper);
    } else {
        const double disc = std::pow(1.0 + rate_per_period, -static_cast<double>(nper));
        const double denom = 1.0 - disc;
        if (denom == 0.0) { return make_error<AmortizationSchedule>(MathError::division_by_zero); }
        payment = principal * rate_per_period / denom;
    }
    if (!std::isfinite(payment)) { return make_error<AmortizationSchedule>(MathError::domain_error); }
    AmortizationSchedule out{};
    out.payment = payment;
    out.interest.reserve(n);
    out.principal.reserve(n);
    out.balance.reserve(n);
    double balance = principal;
    for (std::size_t i = 0; i < n; ++i) {
        const double interest = balance * rate_per_period;
        // Final period pays off the exact remaining balance so the schedule closes (sum of
        // principal == loan, closing balance == 0) despite floating round-off in the level payment.
        double prin = (i + 1 == n) ? balance : payment - interest;
        balance -= prin;
        if (i + 1 == n) { balance = 0.0; }
        out.interest.push_back(interest);
        out.principal.push_back(prin);
        out.balance.push_back(balance);
    }
    return out;
}

}  // namespace nimblecas::fixedincome
