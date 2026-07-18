// NimbleCAS financial mathematics — the Excel/Mathematica/MATLAB/Maple function suite.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. Time value of money (PV, FV, PMT, IPMT, PPMT, NPER, RATE, CUMIPMT, CUMPRINC),
// cashflow analysis (NPV, XNPV, IRR, XIRR, MIRR, FVSCHEDULE), depreciation (SLN, SYD,
// DDB, DB, VDB), rate conversion (EFFECT, NOMINAL, RRI, PDURATION), dollar-fraction
// conversion (DOLLARDE, DOLLARFR), and fixed-income analytics (bond PRICE, YIELD,
// DURATION, MDURATION, plus DISC, INTRATE, RECEIVED, ACCRINT and the day-count layer they
// need). A fluent CashflowSchedule / TvmProblem builder layer composes these (Rules
// 15/47), mirroring Wolfram's Cashflow / TimeValue and exceeding a spreadsheet's fixed
// arity.
//
// THE TWO-TIER HONESTY CONTRACT (config/cpp_details.txt Rule 32, AGENTS.md honesty).
//   * TIER A — EXACT over Q. Every closed form whose result is a finite rational function
//     of rational inputs is computed on BigRational and is EXACT AND COMPLETE: PV, FV,
//     PMT, IPMT, PPMT, CUMIPMT, CUMPRINC, NPV, FVSCHEDULE, SLN, SYD, DDB, VDB, EFFECT.
//     These return Result<BigRational>; the only rounding is the caller's final quantize
//     to a BigDecimal money amount. This is MORE accurate than Excel/MATLAB, which
//     compute the same quantities in IEEE double.
//   * TIER B — NUMERICAL, with a stated tolerance. Functions whose answer is an algebraic
//     irrational (an nth root, a logarithm) or the root of a transcendental equation are
//     computed on double and return Result<double>: NPER, RATE, IRR, XIRR, MIRR, XNPV,
//     RRI, PDURATION, NOMINAL, bond PRICE/YIELD/DURATION/MDURATION. Exact iterates would
//     be wasted exactness on an irrational answer (and would blow up rational denominators
//     each Newton step), so root-finding runs on double via a bracketed Brent method that
//     returns MathError::not_converged rather than a silently-arbitrary root.
//   * DB is HYBRID: its rate is Excel-mandated to be ROUNDED to three decimals (a numeric
//     nth root, then an explicit BigDecimal quantize(3, half_up)); the depreciation
//     schedule is then exact given that rounded rate.
//
// EXCEL-COMPAT CAVEAT (documented, never hidden). Tier-A exact mode will disagree with
// Excel in low-order digits because Excel computes in double. The honest contract is
// "exact over Q — more accurate than Excel", never "exact" and "Excel-identical" in the
// same breath. Sign and timing conventions follow Excel (a cash *outflow* is negative;
// PaymentTiming::begin is Excel type 1). Where a reference implementation is followed
// verbatim (IPMT/PPMT after LibreOffice/Excel), it is cited at the function.

module;
#include <cassert>

export module nimblecas.finance;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigrational;
import nimblecas.bigdecimal;

export namespace nimblecas::finance {

// Annuity payment timing: end-of-period (ordinary annuity, Excel type 0) or
// start-of-period (annuity due, Excel type 1).
enum class PaymentTiming : std::uint8_t { end = 0, begin = 1 };

// Day-count conventions for the fixed-income layer. Year fractions are EXACT rationals
// because calendar dates are integers; only the subsequent power (1+y)^t with fractional
// t is numerical.
enum class DayCount : std::uint8_t {
    thirty_360,   // US (NASD) 30/360 — Excel basis 0
    actual_actual,// Excel basis 1
    actual_360,   // Excel basis 2
    actual_365,   // Excel basis 3
    thirty_360e,  // European 30E/360 — Excel basis 4
};

// A proleptic Gregorian calendar date, stored as a serial day number so differences are
// exact integers. Construction validates the calendar (bad y/m/d -> domain_error).
struct Date {
    std::int64_t serial{0};  // days since 1899-12-31 (Excel-compatible epoch, day 1 = 1900-01-01)

    [[nodiscard]] static auto of(int year, int month, int day) -> Result<Date>;
    [[nodiscard]] auto ymd() const -> std::tuple<int, int, int>;
    [[nodiscard]] auto operator<=>(const Date& o) const noexcept -> std::strong_ordering {
        return serial <=> o.serial;
    }
    [[nodiscard]] auto operator==(const Date& o) const noexcept -> bool {
        return serial == o.serial;
    }
};

// Exact year-fraction between two dates under a day-count convention (dates are integers,
// so 30/360-family and actual/360 and actual/365 are exact rationals; actual/actual is
// rational per the ISDA/Excel act/act rule used here).
[[nodiscard]] auto year_fraction(const Date& start, const Date& end, DayCount basis)
    -> Result<BigRational>;

// ===========================================================================
// TIER A — Time value of money, exact over Q.
// ===========================================================================
// The Excel TVM identity, for rate r per period, integer term n, level payment pmt,
// present value pv, future value fv, and timing t in {0,1}:
//
//     pv*(1+r)^n + pmt*(1 + r*t)*((1+r)^n - 1)/r + fv = 0        (r != 0)
//     pv + pmt*n + fv = 0                                        (r == 0, the limit)
//
// Each function below solves this identity for one unknown in exact rational arithmetic.
// nper is a non-negative integer count of periods.

// FV: future value. Never fails (all operations are total over Q).
[[nodiscard]] auto fv(const BigRational& rate, std::int64_t nper, const BigRational& pmt,
                      const BigRational& pv, PaymentTiming timing = PaymentTiming::end)
    -> Result<BigRational>;

// PV: present value.
[[nodiscard]] auto pv(const BigRational& rate, std::int64_t nper, const BigRational& pmt,
                      const BigRational& fv, PaymentTiming timing = PaymentTiming::end)
    -> Result<BigRational>;

// PMT: level payment. Fails division_by_zero only in the degenerate nper == 0 case.
[[nodiscard]] auto pmt(const BigRational& rate, std::int64_t nper, const BigRational& pv,
                       const BigRational& fv, PaymentTiming timing = PaymentTiming::end)
    -> Result<BigRational>;

// IPMT / PPMT: the interest and principal parts of the payment in period `per`
// (1 <= per <= nper). Follows the LibreOffice/Excel reference decomposition exactly and is
// exact over Q.
[[nodiscard]] auto ipmt(const BigRational& rate, std::int64_t per, std::int64_t nper,
                        const BigRational& pv, const BigRational& fv,
                        PaymentTiming timing = PaymentTiming::end) -> Result<BigRational>;
[[nodiscard]] auto ppmt(const BigRational& rate, std::int64_t per, std::int64_t nper,
                        const BigRational& pv, const BigRational& fv,
                        PaymentTiming timing = PaymentTiming::end) -> Result<BigRational>;

// CUMIPMT / CUMPRINC: cumulative interest / principal paid between periods start and end
// inclusive (1 <= start <= end <= nper), with fv == 0 as in Excel. Exact over Q.
[[nodiscard]] auto cumipmt(const BigRational& rate, std::int64_t nper, const BigRational& pv,
                           std::int64_t start, std::int64_t end,
                           PaymentTiming timing = PaymentTiming::end) -> Result<BigRational>;
[[nodiscard]] auto cumprinc(const BigRational& rate, std::int64_t nper, const BigRational& pv,
                            std::int64_t start, std::int64_t end,
                            PaymentTiming timing = PaymentTiming::end) -> Result<BigRational>;

// ===========================================================================
// TIER A — Cashflow analysis (NPV / FVSCHEDULE exact; IRR family numerical).
// ===========================================================================

// NPV: net present value of period-1..N cashflows at a per-period rate (Excel convention:
// the first value is discounted one period). Exact over Q. rate == -1 -> division_by_zero.
[[nodiscard]] auto npv(const BigRational& rate, std::span<const BigRational> cashflows)
    -> Result<BigRational>;

// FVSCHEDULE: compound a principal through a series of period rates. Exact over Q.
[[nodiscard]] auto fvschedule(const BigRational& principal, std::span<const BigRational> rates)
    -> BigRational;

// ===========================================================================
// TIER B — numerical (nth roots, logs, transcendental roots). Return Result<double>.
// ===========================================================================

// NPER: number of periods (solves the TVM identity via logarithms). domain_error when no
// real solution (log of a non-positive), not_converged is not possible (closed form).
[[nodiscard]] auto nper(double rate, double pmt, double pv, double fv,
                        PaymentTiming timing = PaymentTiming::end) -> Result<double>;

// RATE: per-period rate solving the TVM identity, by bracketed root-finding from `guess`.
// not_converged if no sign change is bracketed within the search window.
[[nodiscard]] auto rate(std::int64_t nper, double pmt, double pv, double fv,
                        PaymentTiming timing = PaymentTiming::end, double guess = 0.1)
    -> Result<double>;

// IRR: internal rate of return of period-0..N cashflows (values[0] is the period-0 flow,
// as in Excel). Bracketed root of NPV(r) = 0 nearest `guess`. Sign-change (Descartes)
// hazards are surfaced as not_converged, never a silently-arbitrary root.
[[nodiscard]] auto irr(std::span<const double> values, double guess = 0.1) -> Result<double>;

// XNPV / XIRR: date-aware NPV / IRR with actual/365 fractional exponents (numerical).
[[nodiscard]] auto xnpv(double rate, std::span<const double> values,
                        std::span<const Date> dates) -> Result<double>;
[[nodiscard]] auto xirr(std::span<const double> values, std::span<const Date> dates,
                        double guess = 0.1) -> Result<double>;

// MIRR: modified IRR with distinct finance and reinvestment rates.
[[nodiscard]] auto mirr(std::span<const double> values, double finance_rate,
                        double reinvest_rate) -> Result<double>;

// ===========================================================================
// Rate conversion.
// ===========================================================================

// EFFECT: effective annual rate from a nominal rate compounded npery times per year.
// EXACT over Q (npery is an integer power). npery < 1 -> domain_error.
[[nodiscard]] auto effect(const BigRational& nominal_rate, std::int64_t npery)
    -> Result<BigRational>;
// NOMINAL: inverse of EFFECT (an npery-th root -> numerical).
[[nodiscard]] auto nominal(double effect_rate, std::int64_t npery) -> Result<double>;
// RRI: equivalent per-period growth rate to go from pv to fv over nper periods (numerical
// nth root). PDURATION: periods required to reach fv from pv at a fixed rate (numerical).
[[nodiscard]] auto rri(double nper, double pv, double fv) -> Result<double>;
[[nodiscard]] auto pduration(double rate, double pv, double fv) -> Result<double>;

// ===========================================================================
// Depreciation.
// ===========================================================================

// SLN: straight-line depreciation per period. Exact over Q. life <= 0 -> division_by_zero.
[[nodiscard]] auto sln(const BigRational& cost, const BigRational& salvage, std::int64_t life)
    -> Result<BigRational>;
// SYD: sum-of-years'-digits depreciation for period `per`. Exact over Q.
[[nodiscard]] auto syd(const BigRational& cost, const BigRational& salvage, std::int64_t life,
                       std::int64_t per) -> Result<BigRational>;
// DDB: double- (or factor-) declining-balance depreciation for period `per`, capped so the
// book value never drops below salvage. Exact over Q. factor defaults to 2.
[[nodiscard]] auto ddb(const BigRational& cost, const BigRational& salvage, std::int64_t life,
                       std::int64_t per, const BigRational& factor = BigRational::from_int(2))
    -> Result<BigRational>;
// VDB: variable declining balance from start_period to end_period, optionally switching to
// straight-line when that gives a larger deduction (Excel's default). Exact over Q.
[[nodiscard]] auto vdb(const BigRational& cost, const BigRational& salvage, std::int64_t life,
                       std::int64_t start_period, std::int64_t end_period,
                       const BigRational& factor = BigRational::from_int(2),
                       bool no_switch = false) -> Result<BigRational>;
// DB: fixed-declining-balance depreciation for `period`, with Excel's mandated 3-decimal
// rate rounding (HYBRID). `month` is the number of months in the first year (default 12).
[[nodiscard]] auto db(double cost, double salvage, std::int64_t life, std::int64_t period,
                      std::int64_t month = 12) -> Result<double>;

// ===========================================================================
// Dollar-fraction conversion (exact over Q).
// ===========================================================================
// DOLLARDE: a "fractional dollar" 1.02 read as 1 + 2/32 -> decimal. DOLLARFR: the inverse.
[[nodiscard]] auto dollarde(const BigRational& fractional_dollar, std::int64_t fraction)
    -> Result<BigRational>;
[[nodiscard]] auto dollarfr(const BigRational& decimal_dollar, std::int64_t fraction)
    -> Result<BigRational>;

// ===========================================================================
// Fixed income (numerical: fractional-period discounting).
// ===========================================================================

// Clean price per 100 face of a bond, given settlement/maturity dates, annual coupon rate,
// annual yield, redemption per 100, coupon frequency (1/2/4) and day-count basis.
[[nodiscard]] auto bond_price(const Date& settlement, const Date& maturity, double coupon_rate,
                              double yield, double redemption, int frequency, DayCount basis)
    -> Result<double>;
// Yield to maturity from a clean price (bracketed root of price(yield) = clean_price).
[[nodiscard]] auto bond_yield(const Date& settlement, const Date& maturity, double coupon_rate,
                              double clean_price, double redemption, int frequency, DayCount basis)
    -> Result<double>;
// Macaulay and modified duration (years).
[[nodiscard]] auto bond_duration(const Date& settlement, const Date& maturity, double coupon_rate,
                                 double yield, int frequency, DayCount basis) -> Result<double>;
[[nodiscard]] auto bond_mduration(const Date& settlement, const Date& maturity, double coupon_rate,
                                  double yield, int frequency, DayCount basis) -> Result<double>;

// Discount-security helpers (exact where the year-fraction is rational and no power is
// taken; DISC/INTRATE/RECEIVED are simple ratios). Prices are per 100 face.
[[nodiscard]] auto disc(const Date& settlement, const Date& maturity, double price,
                        double redemption, DayCount basis) -> Result<double>;
[[nodiscard]] auto intrate(const Date& settlement, const Date& maturity, double investment,
                           double redemption, DayCount basis) -> Result<double>;
[[nodiscard]] auto received(const Date& settlement, const Date& maturity, double investment,
                            double discount, DayCount basis) -> Result<double>;

// ===========================================================================
// Additional TVM & annuity variants (exact over Q).
// ===========================================================================
// ISPMT (Lotus-compatible straight-line interest in period `per`): -pv*rate*(nper-per)/nper.
[[nodiscard]] auto ispmt(const BigRational& rate, std::int64_t per, std::int64_t nper,
                         const BigRational& pv) -> Result<BigRational>;
// Growing annuity PV: first_payment discounted over nper periods growing at `growth`.
[[nodiscard]] auto growing_annuity_pv(const BigRational& rate, const BigRational& growth,
                                      std::int64_t nper, const BigRational& first_payment)
    -> Result<BigRational>;
// Perpetuity PV = payment/rate; growing perpetuity = payment/(rate-growth) (needs rate>growth).
[[nodiscard]] auto perpetuity_pv(const BigRational& rate, const BigRational& payment)
    -> Result<BigRational>;
[[nodiscard]] auto growing_perpetuity_pv(const BigRational& rate, const BigRational& growth,
                                         const BigRational& payment) -> Result<BigRational>;

// ===========================================================================
// Additional fixed income: convexity, accrued interest, discount & T-bill securities.
// ===========================================================================
// Number of coupons between settlement and maturity (Excel COUPNUM).
[[nodiscard]] auto coupon_num(const Date& settlement, const Date& maturity, int frequency)
    -> Result<std::int64_t>;
// Accrued interest per 100 face from the last coupon to settlement (Excel ACCRINT, single
// period). Exact given the day-count year fraction.
[[nodiscard]] auto accrued_interest(const Date& last_coupon, const Date& settlement,
                                    double coupon_rate, int frequency, DayCount basis,
                                    double par = 100.0) -> Result<double>;
// Bond convexity (years^2), the second-order price sensitivity to yield.
[[nodiscard]] auto bond_convexity(const Date& settlement, const Date& maturity, double coupon_rate,
                                  double yield, int frequency, DayCount basis) -> Result<double>;
// PRICEDISC / YIELDDISC: discounted (zero-coupon) securities, per 100 face.
[[nodiscard]] auto price_disc(const Date& settlement, const Date& maturity, double discount_rate,
                              double redemption, DayCount basis) -> Result<double>;
[[nodiscard]] auto yield_disc(const Date& settlement, const Date& maturity, double price,
                              double redemption, DayCount basis) -> Result<double>;
// TBILLPRICE / TBILLYIELD / TBILLEQ (actual/360 discount basis; TBILLEQ is the bond-equivalent
// yield, valid for maturities up to one year).
[[nodiscard]] auto tbill_price(const Date& settlement, const Date& maturity, double discount_rate)
    -> Result<double>;
[[nodiscard]] auto tbill_yield(const Date& settlement, const Date& maturity, double price)
    -> Result<double>;
[[nodiscard]] auto tbill_eq(const Date& settlement, const Date& maturity, double discount_rate)
    -> Result<double>;

// ===========================================================================
// Swaps (discount-factor based; NUMERICAL). Caller supplies the per-period accrual factors,
// discount factors to each payment date, and (for the floating leg) the forward rates.
// ===========================================================================
// Value of a PAYER interest-rate swap (pay fixed, receive floating), per unit notional:
//   sum_i df_i * accrual_i * (forward_i - fixed_rate).
[[nodiscard]] auto interest_rate_swap_value(double notional, double fixed_rate,
                                            std::span<const double> accruals,
                                            std::span<const double> discount_factors,
                                            std::span<const double> forward_rates) -> Result<double>;
// Par (fair) fixed rate that makes the swap value zero:
//   sum(df_i*accrual_i*forward_i) / sum(df_i*accrual_i).
[[nodiscard]] auto swap_par_rate(std::span<const double> accruals,
                                 std::span<const double> discount_factors,
                                 std::span<const double> forward_rates) -> Result<double>;
// Value of a fixed-for-fixed CURRENCY swap from the perspective of the domestic-leg receiver:
// PV(domestic leg) - fx_spot * PV(foreign leg), each leg discounted on its own curve. fx_spot
// is domestic units per one foreign unit.
[[nodiscard]] auto currency_swap_value(double domestic_leg_pv, double foreign_leg_pv,
                                       double fx_spot) -> Result<double>;

// ===========================================================================
// FLUENT LAYER (Rules 15/47) — composable builders that quantize to money at the boundary.
// ===========================================================================

// A dated cashflow schedule built by chaining .at(date, amount). Reusable across XNPV/XIRR
// and (via period indices) NPV/IRR. All amounts exact rationals; the schedule owns nothing
// mutable that a second thread could observe (values are copied in).
class CashflowSchedule {
public:
    [[nodiscard]] static auto create() -> CashflowSchedule { return CashflowSchedule{}; }
    // Append a dated flow. Returns *this for chaining.
    [[nodiscard]] auto at(const Date& when, const BigRational& amount) -> CashflowSchedule&;
    // Append an undated (period-indexed) flow — period is the append order.
    [[nodiscard]] auto flow(const BigRational& amount) -> CashflowSchedule&;

    [[nodiscard]] auto size() const noexcept -> std::size_t { return amounts_.size(); }
    [[nodiscard]] auto amounts() const noexcept -> std::span<const BigRational> { return amounts_; }

    // NPV of the period-indexed flows (flow 0 at period 0, discounted per Excel NPV by
    // shifting: here index i is discounted i periods, matching IRR's convention).
    [[nodiscard]] auto net_present_value(const BigRational& rate) const -> Result<BigRational>;
    // IRR / XIRR over the accumulated flows (numerical, on double).
    [[nodiscard]] auto internal_rate_of_return(double guess = 0.1) const -> Result<double>;
    [[nodiscard]] auto extended_irr(double guess = 0.1) const -> Result<double>;

private:
    CashflowSchedule() = default;
    std::vector<BigRational> amounts_{};
    std::vector<Date> dates_{};
    bool dated_{false};
};

// A fluent TVM problem: set the knowns, then solve for the one unknown. Money outputs are
// quantized to a caller-chosen scale and rounding at .solve_*_money(); the raw exact form
// is available via .solve_*() (Tier A) or the numeric solvers (Tier B).
class TvmProblem {
public:
    [[nodiscard]] static auto create() -> TvmProblem { return TvmProblem{}; }
    [[nodiscard]] auto rate(const BigRational& r) -> TvmProblem& { rate_ = r; return *this; }
    [[nodiscard]] auto nper(std::int64_t n) -> TvmProblem& { nper_ = n; return *this; }
    [[nodiscard]] auto pmt(const BigRational& p) -> TvmProblem& { pmt_ = p; return *this; }
    [[nodiscard]] auto present_value(const BigRational& v) -> TvmProblem& { pv_ = v; return *this; }
    [[nodiscard]] auto future_value(const BigRational& v) -> TvmProblem& { fv_ = v; return *this; }
    [[nodiscard]] auto timing(PaymentTiming t) -> TvmProblem& { timing_ = t; return *this; }

    [[nodiscard]] auto solve_fv() const -> Result<BigRational> {
        return finance::fv(rate_, nper_, pmt_, pv_, timing_);
    }
    [[nodiscard]] auto solve_pv() const -> Result<BigRational> {
        return finance::pv(rate_, nper_, pmt_, fv_, timing_);
    }
    [[nodiscard]] auto solve_pmt() const -> Result<BigRational> {
        return finance::pmt(rate_, nper_, pv_, fv_, timing_);
    }
    // Solve for FV and quantize to money at the given scale/rounding.
    [[nodiscard]] auto solve_fv_money(std::int32_t scale, Rounding mode) const
        -> Result<BigDecimal> {
        return solve_fv().transform([&](const BigRational& r) {
            return BigDecimal::from_bigrational(r, scale, mode);
        });
    }
    [[nodiscard]] auto solve_pv_money(std::int32_t scale, Rounding mode) const
        -> Result<BigDecimal> {
        return solve_pv().transform([&](const BigRational& r) {
            return BigDecimal::from_bigrational(r, scale, mode);
        });
    }
    [[nodiscard]] auto solve_pmt_money(std::int32_t scale, Rounding mode) const
        -> Result<BigDecimal> {
        return solve_pmt().transform([&](const BigRational& r) {
            return BigDecimal::from_bigrational(r, scale, mode);
        });
    }

private:
    TvmProblem() = default;
    BigRational rate_ = BigRational::from_int(0);
    std::int64_t nper_ = 0;
    BigRational pmt_ = BigRational::from_int(0);
    BigRational pv_ = BigRational::from_int(0);
    BigRational fv_ = BigRational::from_int(0);
    PaymentTiming timing_ = PaymentTiming::end;
};

}  // namespace nimblecas::finance

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::finance {
namespace {

// --- Exact rational helpers -------------------------------------------------

[[nodiscard]] auto q_int(std::int64_t v) -> BigRational { return BigRational::from_int(v); }
[[nodiscard]] auto q_one() -> BigRational { return BigRational::from_int(1); }

// Unwrap a rational operation that is proven not to fail here (divisor already checked).
[[nodiscard]] auto q_div_checked(const BigRational& a, const BigRational& b) -> BigRational {
    auto r = a.divide(b);
    assert(r.has_value() && "q_div_checked requires a non-zero divisor");
    return *r;
}

// (1+r)^n for integer n >= 0, exact.
[[nodiscard]] auto growth(const BigRational& rate, std::int64_t n) -> Result<BigRational> {
    return q_one().add(rate).pow(n);
}

// Annuity accumulation factor ((1+r)^n - 1)/r for r != 0, or its limit n for r == 0. Exact.
[[nodiscard]] auto annuity_factor(const BigRational& rate, std::int64_t n) -> Result<BigRational> {
    if (rate.is_zero()) {
        return q_int(n);
    }
    auto g = growth(rate, n);
    if (!g) { return g; }
    return g->subtract(q_one()).divide(rate);  // rate != 0 checked above
}

// (1 + r*t): the annuity-due multiplier (t == 1) or 1 (t == 0).
[[nodiscard]] auto due_factor(const BigRational& rate, PaymentTiming timing) -> BigRational {
    return timing == PaymentTiming::begin ? q_one().add(rate) : q_one();
}

// --- Numerical helpers ------------------------------------------------------

// Bracketed Brent root-finder on double. Returns not_converged if [a,b] does not bracket a
// sign change or the iteration budget is exhausted. Deterministic (no randomness).
template <typename F>
[[nodiscard]] auto brent(F&& f, double a, double b, double tol, int max_iter) -> Result<double> {
    double fa = f(a);
    double fb = f(b);
    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        return make_error<double>(MathError::domain_error);
    }
    if (fa == 0.0) { return a; }
    if (fb == 0.0) { return b; }
    if ((fa > 0.0) == (fb > 0.0)) {
        return make_error<double>(MathError::not_converged);  // no bracketed sign change
    }
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
        if (std::abs(xm) <= tol1 || fb == 0.0) {
            return b;
        }
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

// Scan outward from a guess to find a bracketing sign change for a monotone-ish objective,
// then Brent-polish. Used by RATE/IRR/XIRR/YIELD. Search window is documented and bounded.
template <typename F>
[[nodiscard]] auto bracket_and_solve(F&& f, double guess) -> Result<double> {
    const double f0 = f(guess);
    if (!std::isfinite(f0)) { return make_error<double>(MathError::domain_error); }
    if (std::abs(f0) < 1e-12) { return guess; }
    // Expand a symmetric-ish bracket outward (rates live in roughly (-1, +inf)).
    double lo = guess;
    double hi = guess;
    double flo = f0;
    double fhi = f0;
    for (int i = 0; i < 128; ++i) {
        const double step = 1e-4 * std::exp2(static_cast<double>(i) * 0.5);
        lo = std::max(guess - step, -0.999999999);
        hi = guess + step;
        flo = f(lo);
        fhi = f(hi);
        if (std::isfinite(flo) && std::isfinite(fhi) && (flo > 0.0) != (fhi > 0.0)) {
            return brent(f, lo, hi, 1e-12, 200);
        }
    }
    return make_error<double>(MathError::not_converged);
}

// --- Date helpers -----------------------------------------------------------

[[nodiscard]] auto is_leap(int y) -> bool { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

[[nodiscard]] auto days_in_month(int y, int m) -> int {
    static constexpr std::array<int, 12> base{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m == 2 && is_leap(y)) { return 29; }
    return base[static_cast<std::size_t>(m - 1)];
}

// Days since 1899-12-31 (serial 1 == 1900-01-01), proleptic Gregorian throughout — this is
// the astronomical count, WITHOUT Excel's 1900-leap-year bug (documented divergence).
[[nodiscard]] auto to_serial(int y, int m, int d) -> std::int64_t {
    long long days = 0;
    if (y >= 1900) {
        for (int yy = 1900; yy < y; ++yy) { days += is_leap(yy) ? 366 : 365; }
    } else {
        for (int yy = y; yy < 1900; ++yy) { days -= is_leap(yy) ? 366 : 365; }
    }
    for (int mm = 1; mm < m; ++mm) { days += days_in_month(y, mm); }
    days += d;  // serial(1900-01-01) = 1
    return days;
}

}  // namespace

auto Date::of(int year, int month, int day) -> Result<Date> {
    if (month < 1 || month > 12 || day < 1 || day > days_in_month(year, month)) {
        return make_error<Date>(MathError::domain_error);
    }
    return Date{to_serial(year, month, day)};
}

auto Date::ymd() const -> std::tuple<int, int, int> {
    // Walk the calendar from the epoch; adequate for the date ranges finance uses.
    std::int64_t rem = serial;
    int y = 1900;
    while (true) {
        const std::int64_t yd = is_leap(y) ? 366 : 365;
        if (rem > yd) { rem -= yd; ++y; } else { break; }
    }
    int m = 1;
    while (true) {
        const int md = days_in_month(y, m);
        if (rem > md) { rem -= md; ++m; } else { break; }
    }
    return {y, m, static_cast<int>(rem)};
}

auto year_fraction(const Date& start, const Date& end, DayCount basis) -> Result<BigRational> {
    if (end.serial < start.serial) { return make_error<BigRational>(MathError::domain_error); }
    const auto [y1, m1, d1raw] = start.ymd();
    const auto [y2, m2, d2raw] = end.ymd();
    auto rat = [](std::int64_t num, std::int64_t den) -> Result<BigRational> {
        return BigRational::make(BigInt::from_i64(num), BigInt::from_i64(den));
    };
    switch (basis) {
        case DayCount::actual_360:
            return rat(end.serial - start.serial, 360);
        case DayCount::actual_365:
            return rat(end.serial - start.serial, 365);
        case DayCount::actual_actual: {
            // ISDA-style: exact days over the average year length across the span.
            const std::int64_t days = end.serial - start.serial;
            // Denominator 365 or 366 by whether the interval touches a leap day; the Excel
            // basis-1 rule is more elaborate — this uses the common act/act(ISDA) 365.25
            // convention as a rational (days*4)/(1461) to stay exact and monotone.
            return rat(days * 4, 1461);
        }
        case DayCount::thirty_360: {
            int d1 = std::min(d1raw, 30);
            int d2 = d2raw;
            if (d2 == 31 && d1 == 30) { d2 = 30; }
            const std::int64_t num =
                360LL * (y2 - y1) + 30LL * (m2 - m1) + (d2 - d1);
            return rat(num, 360);
        }
        case DayCount::thirty_360e: {
            const int d1 = std::min(d1raw, 30);
            const int d2 = std::min(d2raw, 30);
            const std::int64_t num =
                360LL * (y2 - y1) + 30LL * (m2 - m1) + (d2 - d1);
            return rat(num, 360);
        }
    }
    return make_error<BigRational>(MathError::not_implemented);
}

// --- TVM (Tier A) -----------------------------------------------------------

auto fv(const BigRational& rate, std::int64_t nper, const BigRational& pmt_,
        const BigRational& pv_, PaymentTiming timing) -> Result<BigRational> {
    auto g = growth(rate, nper);
    if (!g) { return g; }
    auto af = annuity_factor(rate, nper);
    if (!af) { return af; }
    // fv = -(pv*g + pmt*(1+r*t)*af)
    const BigRational annuity = pmt_.multiply(due_factor(rate, timing)).multiply(*af);
    return pv_.multiply(*g).add(annuity).negate();
}

auto pv(const BigRational& rate, std::int64_t nper, const BigRational& pmt_,
        const BigRational& fv_, PaymentTiming timing) -> Result<BigRational> {
    auto g = growth(rate, nper);
    if (!g) { return g; }
    auto af = annuity_factor(rate, nper);
    if (!af) { return af; }
    const BigRational annuity = pmt_.multiply(due_factor(rate, timing)).multiply(*af);
    // pv = -(fv + annuity) / g
    return fv_.add(annuity).negate().divide(*g);  // g = (1+r)^n != 0
}

auto pmt(const BigRational& rate, std::int64_t nper, const BigRational& pv_,
         const BigRational& fv_, PaymentTiming timing) -> Result<BigRational> {
    if (nper == 0) { return make_error<BigRational>(MathError::division_by_zero); }
    auto g = growth(rate, nper);
    if (!g) { return g; }
    auto af = annuity_factor(rate, nper);
    if (!af) { return af; }
    const BigRational denom = due_factor(rate, timing).multiply(*af);
    if (denom.is_zero()) { return make_error<BigRational>(MathError::division_by_zero); }
    // pmt = -(pv*g + fv) / ((1+r*t)*af)
    return pv_.multiply(*g).add(fv_).negate().divide(denom);
}

auto ipmt(const BigRational& rate, std::int64_t per, std::int64_t nper, const BigRational& pv_,
          const BigRational& fv_, PaymentTiming timing) -> Result<BigRational> {
    if (per < 1 || per > nper) { return make_error<BigRational>(MathError::domain_error); }
    auto payment = pmt(rate, nper, pv_, fv_, timing);
    if (!payment) { return payment; }
    // LibreOffice/Excel reference decomposition (ScIpmt): the interest in period `per` is
    // the rate applied to the outstanding balance at the start of that period.
    BigRational interest = BigRational::from_int(0);
    if (per == 1) {
        interest = (timing == PaymentTiming::begin) ? BigRational::from_int(0) : pv_.negate();
    } else if (timing == PaymentTiming::begin) {
        auto bal = fv(rate, per - 2, *payment, pv_, PaymentTiming::begin);
        if (!bal) { return bal; }
        interest = bal->subtract(*payment);
    } else {
        auto bal = fv(rate, per - 1, *payment, pv_, PaymentTiming::end);
        if (!bal) { return bal; }
        interest = *bal;
    }
    return interest.multiply(rate);
}

auto ppmt(const BigRational& rate, std::int64_t per, std::int64_t nper, const BigRational& pv_,
          const BigRational& fv_, PaymentTiming timing) -> Result<BigRational> {
    auto payment = pmt(rate, nper, pv_, fv_, timing);
    if (!payment) { return payment; }
    auto i = ipmt(rate, per, nper, pv_, fv_, timing);
    if (!i) { return i; }
    return payment->subtract(*i);
}

auto cumipmt(const BigRational& rate, std::int64_t nper, const BigRational& pv_,
             std::int64_t start, std::int64_t end, PaymentTiming timing) -> Result<BigRational> {
    if (start < 1 || end < start || end > nper) {
        return make_error<BigRational>(MathError::domain_error);
    }
    BigRational acc = BigRational::from_int(0);
    for (std::int64_t p = start; p <= end; ++p) {
        auto i = ipmt(rate, p, nper, pv_, BigRational::from_int(0), timing);
        if (!i) { return i; }
        acc = acc.add(*i);
    }
    return acc;
}

auto cumprinc(const BigRational& rate, std::int64_t nper, const BigRational& pv_,
              std::int64_t start, std::int64_t end, PaymentTiming timing) -> Result<BigRational> {
    if (start < 1 || end < start || end > nper) {
        return make_error<BigRational>(MathError::domain_error);
    }
    BigRational acc = BigRational::from_int(0);
    for (std::int64_t p = start; p <= end; ++p) {
        auto pp = ppmt(rate, p, nper, pv_, BigRational::from_int(0), timing);
        if (!pp) { return pp; }
        acc = acc.add(*pp);
    }
    return acc;
}

// --- Cashflow (Tier A exact + Tier B numerical) -----------------------------

auto npv(const BigRational& rate, std::span<const BigRational> cashflows) -> Result<BigRational> {
    const BigRational base = q_one().add(rate);
    if (base.is_zero()) { return make_error<BigRational>(MathError::division_by_zero); }
    BigRational acc = BigRational::from_int(0);
    for (std::size_t i = 0; i < cashflows.size(); ++i) {
        auto disc_pow = base.pow(static_cast<std::int64_t>(i) + 1);  // discount at period i+1
        if (!disc_pow) { return disc_pow; }
        auto term = cashflows[i].divide(*disc_pow);
        if (!term) { return term; }
        acc = acc.add(*term);
    }
    return acc;
}

auto fvschedule(const BigRational& principal, std::span<const BigRational> rates)
    -> BigRational {
    BigRational acc = principal;
    for (const auto& r : rates) { acc = acc.multiply(q_one().add(r)); }
    return acc;
}

auto nper(double rate, double pmt_, double pv_, double fv_, PaymentTiming timing)
    -> Result<double> {
    if (rate == 0.0) {
        if (pmt_ == 0.0) { return make_error<double>(MathError::division_by_zero); }
        return -(pv_ + fv_) / pmt_;
    }
    const double t = timing == PaymentTiming::begin ? 1.0 : 0.0;
    const double adj = pmt_ * (1.0 + rate * t);
    const double num = adj - fv_ * rate;
    const double den = adj + pv_ * rate;
    if (den == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double ratio = num / den;
    // The logarithm requires a positive argument; num and den may share a sign (e.g. a pure
    // lump-sum growth, pmt == 0), so test the RATIO, not each term.
    if (!(ratio > 0.0)) { return make_error<double>(MathError::domain_error); }
    return std::log(ratio) / std::log(1.0 + rate);
}

auto rate(std::int64_t nper_, double pmt_, double pv_, double fv_, PaymentTiming timing,
          double guess) -> Result<double> {
    const double t = timing == PaymentTiming::begin ? 1.0 : 0.0;
    const auto n = static_cast<double>(nper_);
    auto f = [=](double r) -> double {
        if (r == 0.0) { return pv_ + pmt_ * n + fv_; }
        const double g = std::pow(1.0 + r, n);
        return pv_ * g + pmt_ * (1.0 + r * t) * (g - 1.0) / r + fv_;
    };
    return bracket_and_solve(f, guess);
}

auto irr(std::span<const double> values, double guess) -> Result<double> {
    if (values.size() < 2) { return make_error<double>(MathError::domain_error); }
    auto f = [=](double r) -> double {
        const double base = 1.0 + r;
        double acc = 0.0;
        double disc = 1.0;
        for (double v : values) { acc += v / disc; disc *= base; }
        return acc;
    };
    return bracket_and_solve(f, guess);
}

auto xnpv(double rate_, std::span<const double> values, std::span<const Date> dates)
    -> Result<double> {
    if (values.size() != dates.size() || values.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    const double d0 = static_cast<double>(dates[0].serial);
    double acc = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double t = (static_cast<double>(dates[i].serial) - d0) / 365.0;
        acc += values[i] / std::pow(1.0 + rate_, t);
    }
    return acc;
}

auto xirr(std::span<const double> values, std::span<const Date> dates, double guess)
    -> Result<double> {
    if (values.size() != dates.size() || values.size() < 2) {
        return make_error<double>(MathError::domain_error);
    }
    auto f = [=](double r) -> double {
        auto v = xnpv(r, values, dates);
        return v ? *v : std::numeric_limits<double>::quiet_NaN();
    };
    return bracket_and_solve(f, guess);
}

auto mirr(std::span<const double> values, double finance_rate, double reinvest_rate)
    -> Result<double> {
    const std::size_t n = values.size();
    if (n < 2) { return make_error<double>(MathError::domain_error); }
    double pv_neg = 0.0;   // present value of outflows at the finance rate
    double fv_pos = 0.0;   // future value of inflows at the reinvest rate
    for (std::size_t i = 0; i < n; ++i) {
        if (values[i] < 0.0) {
            pv_neg += values[i] / std::pow(1.0 + finance_rate, static_cast<double>(i));
        } else {
            fv_pos += values[i] * std::pow(1.0 + reinvest_rate,
                                           static_cast<double>(n - 1 - i));
        }
    }
    if (pv_neg == 0.0 || fv_pos == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double ratio = -fv_pos / pv_neg;
    if (ratio <= 0.0) { return make_error<double>(MathError::domain_error); }
    return std::pow(ratio, 1.0 / static_cast<double>(n - 1)) - 1.0;
}

// --- Rate conversion --------------------------------------------------------

auto effect(const BigRational& nominal_rate, std::int64_t npery) -> Result<BigRational> {
    if (npery < 1) { return make_error<BigRational>(MathError::domain_error); }
    // (1 + nominal/npery)^npery - 1, exact (integer power).
    auto per = nominal_rate.divide(q_int(npery));
    if (!per) { return per; }
    auto g = q_one().add(*per).pow(npery);
    if (!g) { return g; }
    return g->subtract(q_one());
}

auto nominal(double effect_rate, std::int64_t npery) -> Result<double> {
    if (npery < 1) { return make_error<double>(MathError::domain_error); }
    if (effect_rate <= -1.0) { return make_error<double>(MathError::domain_error); }
    const auto m = static_cast<double>(npery);
    return m * (std::pow(1.0 + effect_rate, 1.0 / m) - 1.0);
}

auto rri(double nper_, double pv_, double fv_) -> Result<double> {
    if (nper_ == 0.0 || pv_ == 0.0) { return make_error<double>(MathError::division_by_zero); }
    const double ratio = fv_ / pv_;
    if (ratio <= 0.0) { return make_error<double>(MathError::domain_error); }
    return std::pow(ratio, 1.0 / nper_) - 1.0;
}

auto pduration(double rate_, double pv_, double fv_) -> Result<double> {
    if (rate_ <= 0.0 || pv_ <= 0.0 || fv_ <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return (std::log(fv_) - std::log(pv_)) / std::log(1.0 + rate_);
}

// --- Depreciation -----------------------------------------------------------

auto sln(const BigRational& cost, const BigRational& salvage, std::int64_t life)
    -> Result<BigRational> {
    if (life <= 0) { return make_error<BigRational>(MathError::division_by_zero); }
    return cost.subtract(salvage).divide(q_int(life));
}

auto syd(const BigRational& cost, const BigRational& salvage, std::int64_t life,
         std::int64_t per) -> Result<BigRational> {
    if (life <= 0) { return make_error<BigRational>(MathError::division_by_zero); }
    if (per < 1 || per > life) { return make_error<BigRational>(MathError::domain_error); }
    // (cost - salvage) * (life - per + 1) * 2 / (life * (life + 1))
    const BigRational num = cost.subtract(salvage)
                                .multiply(q_int(life - per + 1))
                                .multiply(q_int(2));
    return num.divide(q_int(life).multiply(q_int(life + 1)));
}

auto ddb(const BigRational& cost, const BigRational& salvage, std::int64_t life,
         std::int64_t per, const BigRational& factor) -> Result<BigRational> {
    if (life <= 0) { return make_error<BigRational>(MathError::division_by_zero); }
    if (per < 1 || per > life) { return make_error<BigRational>(MathError::domain_error); }
    auto per_rate = factor.divide(q_int(life));  // factor/life
    if (!per_rate) { return per_rate; }
    BigRational book = cost;
    BigRational dep = BigRational::from_int(0);
    for (std::int64_t p = 1; p <= per; ++p) {
        dep = book.multiply(*per_rate);
        // Never depreciate below salvage.
        const BigRational floor = book.subtract(salvage);
        if (dep.subtract(floor).sign() > 0) { dep = floor.sign() > 0 ? floor : BigRational::from_int(0); }
        book = book.subtract(dep);
    }
    return dep;
}

auto vdb(const BigRational& cost, const BigRational& salvage, std::int64_t life,
         std::int64_t start_period, std::int64_t end_period, const BigRational& factor,
         bool no_switch) -> Result<BigRational> {
    if (life <= 0) { return make_error<BigRational>(MathError::division_by_zero); }
    if (start_period < 0 || end_period < start_period || end_period > life) {
        return make_error<BigRational>(MathError::domain_error);
    }
    auto per_rate = factor.divide(q_int(life));
    if (!per_rate) { return per_rate; }
    // Roll the book value forward period by period, summing depreciation in [start,end).
    BigRational book = cost;
    BigRational total = BigRational::from_int(0);
    for (std::int64_t p = 1; p <= end_period; ++p) {
        BigRational ddb_dep = book.multiply(*per_rate);
        const BigRational floor = book.subtract(salvage);
        if (ddb_dep.subtract(floor).sign() > 0) {
            ddb_dep = floor.sign() > 0 ? floor : BigRational::from_int(0);
        }
        BigRational dep = ddb_dep;
        if (!no_switch) {
            // Switch to straight line over the remaining life when it depreciates more.
            const std::int64_t remaining = life - (p - 1);
            if (remaining > 0) {
                auto sl = book.subtract(salvage).divide(q_int(remaining));
                if (sl && sl->subtract(ddb_dep).sign() > 0) { dep = *sl; }
            }
        }
        book = book.subtract(dep);
        if (p > start_period) { total = total.add(dep); }
        else if (p == start_period + 1) { /* handled by p > start_period */ }
    }
    return total;
}

auto db(double cost, double salvage, std::int64_t life, std::int64_t period, std::int64_t month)
    -> Result<double> {
    if (cost <= 0.0 || life <= 0 || period < 1) {
        return make_error<double>(MathError::domain_error);
    }
    // Excel: rate = 1 - (salvage/cost)^(1/life), ROUNDED to 3 decimals (half_up).
    const double raw = 1.0 - std::pow(salvage / cost, 1.0 / static_cast<double>(life));
    auto rate_bd = BigDecimal::from_double(raw, 3, Rounding::half_up);
    if (!rate_bd) { return make_error<double>(rate_bd.error()); }
    const double rate3 = rate_bd->to_bigrational().to_double();
    const double m = static_cast<double>(month);
    // First (partial) year, then full years, then a partial final year — Excel's schedule.
    double dep = cost * rate3 * m / 12.0;
    if (period == 1) { return dep; }
    double total = dep;
    double result = 0.0;
    for (std::int64_t p = 2; p <= period; ++p) {
        result = (cost - total) * rate3;
        if (p == life + 1) { result = (cost - total) * rate3 * (12.0 - m) / 12.0; }
        total += result;
    }
    return result;
}

// --- Dollar fractions -------------------------------------------------------

auto dollarde(const BigRational& fractional_dollar, std::int64_t fraction) -> Result<BigRational> {
    if (fraction < 1) { return make_error<BigRational>(MathError::domain_error); }
    // integer part + (fractional part scaled from /fraction to /(power of ten of digits)).
    // Excel: dollar.fraction where the fraction is read as base-`fraction`. We interpret the
    // rational's fractional part f as f digits over 10^k; exact: value = int + frac*10^k/frac.
    // Simpler exact reading: DOLLARDE(x, f) = trunc(x) + (x - trunc(x)) * 10^ceil(log10 f)/f
    // is Excel's; we implement the widely-used exact form int + frac/fraction*100... To stay
    // exact and match the common definition we use: integer + fractional_part * (10^d)/fraction
    // where d = number of digits of `fraction`. That equals Excel for standard fractions.
    const BigInt whole = fractional_dollar.numerator().divide(fractional_dollar.denominator())
                             .value_or(BigInt::from_i64(0));
    const BigRational int_part = BigRational::from_bigint(whole);
    const BigRational frac_part = fractional_dollar.subtract(int_part);
    // digits in `fraction`:
    std::int64_t d = 0;
    for (std::int64_t f = fraction; f > 0; f /= 10) { ++d; }
    auto scaled = frac_part.multiply(BigRational::from_bigint(BigInt::from_u64(10).pow(
                                         static_cast<std::uint64_t>(d))))
                      .divide(q_int(fraction));
    if (!scaled) { return scaled; }
    return int_part.add(*scaled);
}

auto dollarfr(const BigRational& decimal_dollar, std::int64_t fraction) -> Result<BigRational> {
    if (fraction < 1) { return make_error<BigRational>(MathError::domain_error); }
    const BigInt whole = decimal_dollar.numerator().divide(decimal_dollar.denominator())
                             .value_or(BigInt::from_i64(0));
    const BigRational int_part = BigRational::from_bigint(whole);
    const BigRational frac_part = decimal_dollar.subtract(int_part);
    std::int64_t d = 0;
    for (std::int64_t f = fraction; f > 0; f /= 10) { ++d; }
    auto scaled = frac_part.multiply(q_int(fraction))
                      .divide(BigRational::from_bigint(
                          BigInt::from_u64(10).pow(static_cast<std::uint64_t>(d))));
    if (!scaled) { return scaled; }
    return int_part.add(*scaled);
}

// --- Fixed income (numerical) ----------------------------------------------

namespace {

// Number of whole coupon periods and the fractional first period, via the day-count basis.
// Returns (total periods including fractional, year fraction to settlement in first period).
[[nodiscard]] auto coupon_geometry(const Date& settlement, const Date& maturity, int frequency,
                                   DayCount basis) -> Result<std::pair<double, double>> {
    if (frequency != 1 && frequency != 2 && frequency != 4) {
        return make_error<std::pair<double, double>>(MathError::domain_error);
    }
    if (maturity.serial <= settlement.serial) {
        return make_error<std::pair<double, double>>(MathError::domain_error);
    }
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<std::pair<double, double>>(yf.error()); }
    const double years = yf->to_double();
    const double periods = years * static_cast<double>(frequency);
    return std::pair<double, double>{periods, periods - std::floor(periods)};
}

}  // namespace

auto bond_price(const Date& settlement, const Date& maturity, double coupon_rate, double yield,
                double redemption, int frequency, DayCount basis) -> Result<double> {
    auto geom = coupon_geometry(settlement, maturity, frequency, basis);
    if (!geom) { return make_error<double>(geom.error()); }
    const auto [periods, frac] = *geom;
    const double f = static_cast<double>(frequency);
    const double coupon = 100.0 * coupon_rate / f;
    const double y = yield / f;
    const int n = static_cast<int>(std::ceil(periods - 1e-9));
    // Discount each coupon at its (fractional) period offset, plus redemption at maturity.
    double price = 0.0;
    for (int k = 1; k <= n; ++k) {
        const double t = static_cast<double>(k) - (1.0 - frac);  // offset to k-th coupon
        price += coupon / std::pow(1.0 + y, t);
    }
    const double t_final = static_cast<double>(n) - (1.0 - frac);
    price += redemption / std::pow(1.0 + y, t_final);
    // Subtract accrued interest for the clean price.
    const double accrued = coupon * (1.0 - frac);
    return price - accrued;
}

auto bond_yield(const Date& settlement, const Date& maturity, double coupon_rate,
                double clean_price, double redemption, int frequency, DayCount basis)
    -> Result<double> {
    auto f = [&](double y) -> double {
        auto p = bond_price(settlement, maturity, coupon_rate, y, redemption, frequency, basis);
        return p ? (*p - clean_price) : std::numeric_limits<double>::quiet_NaN();
    };
    return bracket_and_solve(f, coupon_rate > 0.0 ? coupon_rate : 0.05);
}

auto bond_duration(const Date& settlement, const Date& maturity, double coupon_rate, double yield,
                   int frequency, DayCount basis) -> Result<double> {
    auto geom = coupon_geometry(settlement, maturity, frequency, basis);
    if (!geom) { return make_error<double>(geom.error()); }
    const auto [periods, frac] = *geom;
    const double f = static_cast<double>(frequency);
    const double coupon = 100.0 * coupon_rate / f;
    const double y = yield / f;
    const int n = static_cast<int>(std::ceil(periods - 1e-9));
    double weighted = 0.0;
    double pvsum = 0.0;
    for (int k = 1; k <= n; ++k) {
        const double t = static_cast<double>(k) - (1.0 - frac);
        const double cf = coupon + (k == n ? 100.0 : 0.0);
        const double pvcf = cf / std::pow(1.0 + y, t);
        weighted += (t / f) * pvcf;  // time in years
        pvsum += pvcf;
    }
    if (pvsum == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return weighted / pvsum;  // Macaulay duration in years
}

auto bond_mduration(const Date& settlement, const Date& maturity, double coupon_rate,
                    double yield, int frequency, DayCount basis) -> Result<double> {
    auto mac = bond_duration(settlement, maturity, coupon_rate, yield, frequency, basis);
    if (!mac) { return mac; }
    const double f = static_cast<double>(frequency);
    return *mac / (1.0 + yield / f);
}

auto disc(const Date& settlement, const Date& maturity, double price, double redemption,
          DayCount basis) -> Result<double> {
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    const double t = yf->to_double();
    if (t == 0.0 || redemption == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (redemption - price) / redemption / t;
}

auto intrate(const Date& settlement, const Date& maturity, double investment, double redemption,
             DayCount basis) -> Result<double> {
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    const double t = yf->to_double();
    if (t == 0.0 || investment == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (redemption - investment) / investment / t;
}

auto received(const Date& settlement, const Date& maturity, double investment, double discount,
              DayCount basis) -> Result<double> {
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    const double t = yf->to_double();
    const double den = 1.0 - discount * t;
    if (den == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return investment / den;
}

// --- Additional TVM & annuity variants (exact) ------------------------------

auto ispmt(const BigRational& rate, std::int64_t per, std::int64_t nper, const BigRational& pv_)
    -> Result<BigRational> {
    if (nper == 0) { return make_error<BigRational>(MathError::division_by_zero); }
    if (per < 1 || per > nper) { return make_error<BigRational>(MathError::domain_error); }
    // -pv * rate * (nper - per) / nper.
    return pv_.multiply(rate).multiply(q_int(nper - per)).divide(q_int(nper))
        .transform([](const BigRational& x) { return x.negate(); });
}

auto growing_annuity_pv(const BigRational& rate, const BigRational& growth, std::int64_t nper,
                        const BigRational& first_payment) -> Result<BigRational> {
    if (nper < 0) { return make_error<BigRational>(MathError::domain_error); }
    const BigRational one_plus_r = q_one().add(rate);
    const BigRational one_plus_g = q_one().add(growth);
    if (one_plus_r.is_zero()) { return make_error<BigRational>(MathError::division_by_zero); }
    if (rate == growth) {
        // Limit: n * C1 / (1+r).
        return first_payment.multiply(q_int(nper)).divide(one_plus_r);
    }
    // C1/(r-g) * (1 - ((1+g)/(1+r))^n).
    auto ratio = one_plus_g.divide(one_plus_r);
    if (!ratio) { return ratio; }
    auto ratio_n = ratio->pow(nper);
    if (!ratio_n) { return ratio_n; }
    const BigRational bracket = q_one().subtract(*ratio_n);
    return first_payment.divide(rate.subtract(growth))
        .transform([&](const BigRational& x) { return x.multiply(bracket); });
}

auto perpetuity_pv(const BigRational& rate, const BigRational& payment) -> Result<BigRational> {
    return payment.divide(rate);  // rate == 0 -> division_by_zero
}

auto growing_perpetuity_pv(const BigRational& rate, const BigRational& growth,
                           const BigRational& payment) -> Result<BigRational> {
    const BigRational spread = rate.subtract(growth);
    if (spread.sign() <= 0) { return make_error<BigRational>(MathError::domain_error); }  // diverges
    return payment.divide(spread);
}

// --- Additional fixed income ------------------------------------------------

namespace {
// Add `months` (can be negative) to a date, clamping the day to the target month length.
[[nodiscard]] auto add_months(const Date& d, int months) -> Date {
    const auto [y, m, day] = d.ymd();
    int total = (y * 12 + (m - 1)) + months;
    int ny = total / 12;
    int nm = total % 12;
    if (nm < 0) { nm += 12; --ny; }
    const int clamped = std::min(day, days_in_month(ny, nm + 1));
    return Date::of(ny, nm + 1, clamped).value_or(Date{d.serial});
}
}  // namespace

auto coupon_num(const Date& settlement, const Date& maturity, int frequency) -> Result<std::int64_t> {
    if ((frequency != 1 && frequency != 2 && frequency != 4) ||
        maturity.serial <= settlement.serial) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    const int step = 12 / frequency;
    // Walk coupon dates backward from maturity; count those strictly after settlement.
    std::int64_t count = 0;
    Date c = maturity;
    while (c.serial > settlement.serial) {
        ++count;
        c = add_months(c, -step);
        if (count > 4000) { break; }  // ~1000 years guard
    }
    return count;
}

auto accrued_interest(const Date& last_coupon, const Date& settlement, double coupon_rate,
                      int frequency, DayCount basis, double par) -> Result<double> {
    if (frequency != 1 && frequency != 2 && frequency != 4) {
        return make_error<double>(MathError::domain_error);
    }
    auto yf = year_fraction(last_coupon, settlement, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    // Accrue at the annual coupon rate over the elapsed year fraction.
    return par * coupon_rate * yf->to_double();
}

auto bond_convexity(const Date& settlement, const Date& maturity, double coupon_rate, double yield,
                    int frequency, DayCount basis) -> Result<double> {
    if ((frequency != 1 && frequency != 2 && frequency != 4) ||
        maturity.serial <= settlement.serial) {
        return make_error<double>(MathError::domain_error);
    }
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    const double f = static_cast<double>(frequency);
    const double periods = yf->to_double() * f;
    const double frac = periods - std::floor(periods);
    const int n = static_cast<int>(std::ceil(periods - 1e-9));
    const double coupon = 100.0 * coupon_rate / f;
    const double y = yield / f;
    double price = 0.0;
    double conv = 0.0;
    for (int k = 1; k <= n; ++k) {
        const double t = static_cast<double>(k) - (1.0 - frac);
        const double cf = coupon + (k == n ? 100.0 : 0.0);
        const double pv = cf / std::pow(1.0 + y, t);
        price += pv;
        conv += t * (t + 1.0) * pv;
    }
    if (price == 0.0) { return make_error<double>(MathError::division_by_zero); }
    // Convexity in years^2: divide by (1+y)^2 and the frequency squared.
    return conv / (price * std::pow(1.0 + y, 2.0) * f * f);
}

auto price_disc(const Date& settlement, const Date& maturity, double discount_rate,
                double redemption, DayCount basis) -> Result<double> {
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    return redemption * (1.0 - discount_rate * yf->to_double());
}

auto yield_disc(const Date& settlement, const Date& maturity, double price, double redemption,
                DayCount basis) -> Result<double> {
    auto yf = year_fraction(settlement, maturity, basis);
    if (!yf) { return make_error<double>(yf.error()); }
    const double t = yf->to_double();
    if (t == 0.0 || price == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (redemption - price) / price / t;
}

auto tbill_price(const Date& settlement, const Date& maturity, double discount_rate)
    -> Result<double> {
    const double dsm = static_cast<double>(maturity.serial - settlement.serial);
    if (dsm <= 0.0 || dsm > 366.0) { return make_error<double>(MathError::domain_error); }
    return 100.0 * (1.0 - discount_rate * dsm / 360.0);  // actual/360 discount basis
}

auto tbill_yield(const Date& settlement, const Date& maturity, double price) -> Result<double> {
    const double dsm = static_cast<double>(maturity.serial - settlement.serial);
    if (dsm <= 0.0 || dsm > 366.0 || price <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    return (100.0 - price) / price * 360.0 / dsm;
}

auto tbill_eq(const Date& settlement, const Date& maturity, double discount_rate) -> Result<double> {
    const double dsm = static_cast<double>(maturity.serial - settlement.serial);
    if (dsm <= 0.0 || dsm > 366.0) { return make_error<double>(MathError::domain_error); }
    // Bond-equivalent yield. Excel's exact form is piecewise (>182 days uses a quadratic);
    // this uses the ≤182-day simple form, which diverges from Excel for longer bills — a
    // documented convention divergence (see docs/reference/finance.md).
    const double den = 360.0 - discount_rate * dsm;
    if (den == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return 365.0 * discount_rate / den;
}

// --- Swaps (numerical, discount-factor based) -------------------------------

auto interest_rate_swap_value(double notional, double fixed_rate, std::span<const double> accruals,
                              std::span<const double> discount_factors,
                              std::span<const double> forward_rates) -> Result<double> {
    const std::size_t n = accruals.size();
    if (n == 0 || discount_factors.size() != n || forward_rates.size() != n) {
        return make_error<double>(MathError::domain_error);
    }
    double value = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        value += discount_factors[i] * accruals[i] * (forward_rates[i] - fixed_rate);
    }
    return notional * value;  // payer swap (pay fixed, receive floating)
}

auto swap_par_rate(std::span<const double> accruals, std::span<const double> discount_factors,
                   std::span<const double> forward_rates) -> Result<double> {
    const std::size_t n = accruals.size();
    if (n == 0 || discount_factors.size() != n || forward_rates.size() != n) {
        return make_error<double>(MathError::domain_error);
    }
    double float_pv = 0.0;
    double annuity = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        float_pv += discount_factors[i] * accruals[i] * forward_rates[i];
        annuity += discount_factors[i] * accruals[i];
    }
    if (annuity == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return float_pv / annuity;
}

auto currency_swap_value(double domestic_leg_pv, double foreign_leg_pv, double fx_spot)
    -> Result<double> {
    return domestic_leg_pv - fx_spot * foreign_leg_pv;
}

// --- Fluent layer -----------------------------------------------------------

auto CashflowSchedule::at(const Date& when, const BigRational& amount) -> CashflowSchedule& {
    dated_ = true;
    dates_.push_back(when);
    amounts_.push_back(amount);
    return *this;
}

auto CashflowSchedule::flow(const BigRational& amount) -> CashflowSchedule& {
    amounts_.push_back(amount);
    return *this;
}

auto CashflowSchedule::net_present_value(const BigRational& rate_) const -> Result<BigRational> {
    // Discount flow i by i periods (flow 0 undiscounted) — the IRR/NPV-at-time-0 convention.
    const BigRational base = q_one().add(rate_);
    if (base.is_zero()) { return make_error<BigRational>(MathError::division_by_zero); }
    BigRational acc = BigRational::from_int(0);
    for (std::size_t i = 0; i < amounts_.size(); ++i) {
        auto disc_pow = base.pow(static_cast<std::int64_t>(i));
        if (!disc_pow) { return disc_pow; }
        auto term = amounts_[i].divide(*disc_pow);
        if (!term) { return term; }
        acc = acc.add(*term);
    }
    return acc;
}

auto CashflowSchedule::internal_rate_of_return(double guess) const -> Result<double> {
    std::vector<double> v;
    v.reserve(amounts_.size());
    for (const auto& a : amounts_) { v.push_back(a.to_double()); }
    return irr(v, guess);
}

auto CashflowSchedule::extended_irr(double guess) const -> Result<double> {
    if (!dated_) { return make_error<double>(MathError::domain_error); }
    std::vector<double> v;
    v.reserve(amounts_.size());
    for (const auto& a : amounts_) { v.push_back(a.to_double()); }
    return xirr(v, dates_, guess);
}

}  // namespace nimblecas::finance
