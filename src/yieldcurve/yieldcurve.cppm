// NimbleCAS yield-curve / term-structure toolkit — the fixed-income curve layer that
// complements nimblecas.finance's single-instrument analytics.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. A pillar-based zero (spot) curve with selectable interpolation (LINEAR on zero
// rates, LOG-LINEAR on discount factors) and a selectable compounding convention
// (continuous / annual / semiannual / k-per-year); the four classical term-structure
// conversions (zero<->discount, zero<->forward); sequential bootstrapping of a zero curve
// from bond PRICES (zbtprice) or bond YIELDS (zbtyield); par-yield <-> zero conversion
// (par2zero / zero2par); parametric Nelson-Siegel and Svensson fits; and a Hull-White
// trinomial short-rate lattice calibrated to an initial discount curve.
//
// HONESTY CONTRACT (config/cpp_details.txt Rule 32, AGENTS.md honesty). This whole module
// is the NUMERICAL tier: times are years (double), rates/discount factors are double, and
// every fallible entry point returns Result<T>. A NaN/inf is NEVER returned as a value —
// an ill-posed or singular problem yields MathError::domain_error, and an iterative fit
// that exhausts its (bounded) budget without meeting tolerance yields not_converged. Every
// user-supplied size is bounded (kMaxPillars, kMaxSteps, kMaxJmax) so a hostile input
// count cannot drive an allocation/CPU DoS. Time is measured in YEARS as a double; this
// module deliberately does NOT reinvent a calendar — the optional date bridge
// (times_from_dates) reuses nimblecas.finance's Date / DayCount / year_fraction.
//
// RELATION TO nimblecas.finance. finance prices ONE instrument against a flat scalar
// yield; this module builds and manipulates the whole CURVE and discounts arbitrary
// cashflows against it. The two share the honesty discipline and the Date/DayCount types.

export module nimblecas.yieldcurve;

import std;
import nimblecas.core;
import nimblecas.finance;      // Date, DayCount, year_fraction — reused, not reinvented
import nimblecas.bigrational;  // year_fraction returns Result<BigRational>; we take .to_double()

export namespace nimblecas::yieldcurve {

// ---------------------------------------------------------------------------
// Conventions.
// ---------------------------------------------------------------------------

// How a value is interpolated between pillars.
//   linear_zero  — the zero (spot) rate is linear in t between pillars.
//   loglinear_df — ln(discount factor) is linear in t between pillars (equivalently the
//                  discount factor is log-linear), the standard bootstrapped-curve choice.
enum class Interp : std::uint8_t { linear_zero, loglinear_df };

// Compounding convention relating a zero rate z at time t to its discount factor DF:
//   continuous  : DF = exp(-z t),              z = -ln(DF)/t
//   annual      : DF = (1+z)^(-t),             m = 1
//   semiannual  : DF = (1+z/2)^(-2t),          m = 2
//   k_per_year  : DF = (1+z/k)^(-k t),         m = k (the `freq` argument)
enum class Compounding : std::uint8_t { continuous, annual, semiannual, k_per_year };

// ---------------------------------------------------------------------------
// Point conversions between a zero rate and a discount factor (free functions, so callers
// need not build a whole Curve for a single conversion).
// ---------------------------------------------------------------------------

// Discount factor for a zero rate `zero` quoted under `comp`/`freq` at time t (years).
// t == 0 -> 1. t < 0, a non-positive periodic base (1 + z/m <= 0), or a non-finite input
// -> domain_error (the result would be complex / non-finite; honesty forbids faking it).
[[nodiscard]] auto discount_from_zero(double zero, double t, Compounding comp, int freq = 1)
    -> Result<double>;

// Zero rate implied by a discount factor `df` (> 0) at time t (> 0) under `comp`/`freq`.
// df <= 0 or t <= 0 -> domain_error.
[[nodiscard]] auto zero_from_discount(double df, double t, Compounding comp, int freq = 1)
    -> Result<double>;

// ---------------------------------------------------------------------------
// The Curve object.
// ---------------------------------------------------------------------------
// A zero-rate term structure sampled at strictly increasing positive pillar times. The
// interpolation and compounding conventions are fixed at construction. No extrapolation:
// evaluating outside [t_first, t_last] is a domain_error, never a silent flat guess.
class Curve {
public:
    // Build a curve from parallel pillar-time / zero-rate arrays. Rejects (domain_error):
    // empty input, a size mismatch, more than kMaxPillars pillars (DoS guard), a
    // non-finite entry, a non-positive first time, non-strictly-increasing times, or an
    // invalid k_per_year frequency.
    [[nodiscard]] static auto create(std::vector<double> times, std::vector<double> zeros,
                                     Interp interp, Compounding comp, int freq = 1)
        -> Result<Curve>;

    // Discount factor DF(t) for t in [t_first, t_last]. Out of range -> domain_error.
    [[nodiscard]] auto discount_factor(double t) const -> Result<double>;
    // Zero (spot) rate z(t) under the curve's compounding convention. Out of range ->
    // domain_error. Consistent with discount_factor: z(t) = zero_from_discount(DF(t), t).
    [[nodiscard]] auto zero_rate(double t) const -> Result<double>;
    // Forward rate over [t1, t2] (t1 < t2, both in range) under the curve's compounding:
    // continuous -> ln(DF(t1)/DF(t2))/(t2-t1); periodic-m -> m*((DF(t1)/DF(t2))^(1/(m dt))-1).
    [[nodiscard]] auto forward_rate(double t1, double t2) const -> Result<double>;

    [[nodiscard]] auto pillar_times() const noexcept -> std::span<const double> { return times_; }
    [[nodiscard]] auto zero_rates() const noexcept -> std::span<const double> { return zeros_; }
    [[nodiscard]] auto interpolation() const noexcept -> Interp { return interp_; }
    [[nodiscard]] auto compounding() const noexcept -> Compounding { return comp_; }
    [[nodiscard]] auto frequency() const noexcept -> int { return freq_; }

private:
    Curve() = default;
    std::vector<double> times_{};
    std::vector<double> zeros_{};
    std::vector<double> df_{};     // DF at each pillar (precomputed)
    std::vector<double> logdf_{};  // ln(DF) at each pillar (for log-linear interpolation)
    Interp interp_ = Interp::linear_zero;
    Compounding comp_ = Compounding::continuous;
    int freq_ = 1;
    int m_ = 0;  // resolved periods/year; 0 == continuous
};

// ---------------------------------------------------------------------------
// Term-structure conversions on pillar arrays (MATLAB-style zero2disc etc.).
// ---------------------------------------------------------------------------
// Discount factors from zero rates at the given pillar times.
[[nodiscard]] auto zero2disc(std::span<const double> times, std::span<const double> zeros,
                             Compounding comp, int freq = 1) -> Result<std::vector<double>>;
// Zero rates from discount factors (exact inverse of zero2disc).
[[nodiscard]] auto disc2zero(std::span<const double> times, std::span<const double> discs,
                             Compounding comp, int freq = 1) -> Result<std::vector<double>>;
// Forward rates: element i is the forward over the interval (t_{i-1}, t_i] (with t_{-1}=0,
// so element 0 is the spot rate to t_0), all under `comp`/`freq`.
[[nodiscard]] auto zero2fwd(std::span<const double> times, std::span<const double> zeros,
                            Compounding comp, int freq = 1) -> Result<std::vector<double>>;
// Zero rates from those interval forwards (exact inverse of zero2fwd).
[[nodiscard]] auto fwd2zero(std::span<const double> times, std::span<const double> fwds,
                            Compounding comp, int freq = 1) -> Result<std::vector<double>>;

// ---------------------------------------------------------------------------
// Bootstrapping a zero curve from coupon-bond instruments.
// ---------------------------------------------------------------------------
// A (possibly coupon-bearing) bond used as a bootstrap instrument.
//   maturity     : years to the final cashflow (> 0)
//   coupon_rate  : annual coupon as a fraction of face (0 => zero-coupon / bill)
//   frequency    : coupons per year in {1,2,4}; ignored when coupon_rate == 0
//   face         : redemption paid with the final coupon (default 100)
struct CouponBond {
    double maturity{};
    double coupon_rate{};
    int frequency{2};
    double face{100.0};
};

// Sequential bootstrap of a zero curve from bond PRICES (dirty price per `face`, i.e. the
// present value of ALL cashflows). Instruments must have strictly increasing maturities.
// ASSUMPTION (documented, enforced): every NON-final coupon date of each bond must fall on
// or before an already-bootstrapped maturity, so its discount factor is known by
// interpolation on the curve built so far; otherwise the system is underdetermined and we
// return domain_error rather than inventing a value. When coupon dates coincide with
// pillar times (the textbook grid), re-pricing the inputs reproduces their prices exactly.
[[nodiscard]] auto zbtprice(std::span<const CouponBond> bonds, std::span<const double> prices,
                            Compounding comp, int freq = 1, Interp interp = Interp::loglinear_df)
    -> Result<Curve>;

// Sequential bootstrap from bond YIELDS to maturity: each yield is converted to a dirty
// price (cashflows discounted at that bond's own periodic yield; a zero-coupon bill is
// priced annually) and then zbtprice is applied. Same assumptions as zbtprice.
[[nodiscard]] auto zbtyield(std::span<const CouponBond> bonds, std::span<const double> yields,
                            Compounding comp, int freq = 1, Interp interp = Interp::loglinear_df)
    -> Result<Curve>;

// ---------------------------------------------------------------------------
// Par-yield <-> zero conversion.
// ---------------------------------------------------------------------------
// These treat the pillar times as the coupon dates of par bonds paying `couponfreq` times
// per year (one coupon per pillar interval); `comp`/`freq` is the compounding used to quote
// the resulting zero rates. par2zero and zero2par are exact inverses of one another.

// Zero rates from par (swap) yields: sequential bootstrap of the discount factors
// DF_i = (1 - (p_i/f) * sum_{k<i} DF_k) / (1 + p_i/f), then quoted as zero rates.
[[nodiscard]] auto par2zero(std::span<const double> times, std::span<const double> par_yields,
                            Compounding comp, int freq = 1, int couponfreq = 1)
    -> Result<std::vector<double>>;
// Par yields from zero rates: p_i = f * (1 - DF_i) / sum_{k<=i} DF_k.
[[nodiscard]] auto zero2par(std::span<const double> times, std::span<const double> zeros,
                            Compounding comp, int freq = 1, int couponfreq = 1)
    -> Result<std::vector<double>>;

// ---------------------------------------------------------------------------
// Parametric fits: Nelson-Siegel and Svensson.
// ---------------------------------------------------------------------------
// Nelson-Siegel zero-rate model:
//   z(t) = b0 + b1 * (1-e^{-u})/u + b2 * ((1-e^{-u})/u - e^{-u}),  u = t/tau,  tau > 0.
// The model is LINEAR in (b0,b1,b2) once tau is fixed, so the fit nests a linear
// least-squares solve inside a 1-D search over tau (coarse log grid + golden-section
// refine) — more robust than a full nonlinear Levenberg-Marquardt for this shape.
struct NelsonSiegel {
    double beta0{}, beta1{}, beta2{}, tau{1.0};
    [[nodiscard]] auto zero_rate(double t) const -> Result<double>;
    [[nodiscard]] auto discount_factor(double t, Compounding comp, int freq = 1) const
        -> Result<double>;
};

// Svensson extends Nelson-Siegel with a second hump:
//   z(t) = b0 + b1*(1-e^{-u1})/u1 + b2*((1-e^{-u1})/u1 - e^{-u1})
//              + b3*((1-e^{-u2})/u2 - e^{-u2}),  u1=t/tau1, u2=t/tau2, tau1,tau2 > 0.
// Fitted by a 2-D search over (tau1,tau2) with the inner 4-parameter linear least squares.
struct Svensson {
    double beta0{}, beta1{}, beta2{}, beta3{}, tau1{1.0}, tau2{3.0};
    [[nodiscard]] auto zero_rate(double t) const -> Result<double>;
    [[nodiscard]] auto discount_factor(double t, Compounding comp, int freq = 1) const
        -> Result<double>;
};

// Fit a Nelson-Siegel curve to observed (time, zero-rate) pairs (>= 4 points; times > 0).
// not_converged if no tau yields a solvable, finite least-squares system.
[[nodiscard]] auto fit_nelson_siegel(std::span<const double> times, std::span<const double> zeros)
    -> Result<NelsonSiegel>;
// Fit a Svensson curve (>= 6 points). not_converged as above.
[[nodiscard]] auto fit_svensson(std::span<const double> times, std::span<const double> zeros)
    -> Result<Svensson>;

// ---------------------------------------------------------------------------
// Hull-White one-factor trinomial short-rate lattice (P1).
// ---------------------------------------------------------------------------
// Short rate dr = (theta(t) - a r) dt + sigma dW, discretised on Hull's symmetric
// trinomial tree. Stage 1 builds the mean-reverting tree for the zero-drift process;
// stage 2 displaces each time slice by alpha_i via forward induction on Arrow-Debreu
// (state) prices so the tree EXACTLY reprices the input discount curve. Consequently the
// lattice zero-coupon price discount(iΔt) equals the curve's DF(iΔt) by construction — the
// honest self-consistency oracle used in the tests.
class HullWhiteLattice {
public:
    // Zero-coupon bond price to a node time T = n*dt (1 <= n <= steps). T must land on a
    // node (within tolerance) and within the built horizon; otherwise domain_error.
    [[nodiscard]] auto discount(double T) const -> Result<double>;
    // Price of a set of dated cashflows: sum_k cashflows[k] * discount(times[k]).
    [[nodiscard]] auto bond_price(std::span<const double> times,
                                  std::span<const double> cashflows) const -> Result<double>;

    [[nodiscard]] auto dt() const noexcept -> double { return dt_; }
    [[nodiscard]] auto steps() const noexcept -> int { return steps_; }
    [[nodiscard]] auto j_max() const noexcept -> int { return jmax_; }

private:
    friend auto build_hull_white(const Curve&, double, double, double, int)
        -> Result<HullWhiteLattice>;
    HullWhiteLattice() = default;
    double a_{}, sigma_{}, dt_{}, dx_{};
    int jmax_{}, steps_{};
    std::vector<double> alpha_{};  // slice displacements, index 0..steps-1
};

// Calibrate a Hull-White trinomial lattice to `curve` with mean reversion a>0, vol
// sigma>0, step dt>0 and `steps` steps (steps*dt must be within the curve's range). DoS
// guards: steps in [1, kMaxSteps], and the derived tree half-width jmax in [1, kMaxJmax]
// (a*dt too small explodes jmax -> domain_error). A negative branching probability from an
// out-of-range (a, dt) is surfaced as domain_error, never silently used.
[[nodiscard]] auto build_hull_white(const Curve& curve, double a, double sigma, double dt,
                                    int steps) -> Result<HullWhiteLattice>;

// ---------------------------------------------------------------------------
// Optional calendar bridge (reuses nimblecas.finance — no new date type).
// ---------------------------------------------------------------------------
// Year fractions from a settlement date to each pillar date under a day-count basis,
// suitable as the `times` argument to Curve::create. Dates must be strictly increasing and
// after settlement; anything else propagates finance's domain_error.
[[nodiscard]] auto times_from_dates(const finance::Date& settlement,
                                    std::span<const finance::Date> pillars, finance::DayCount basis)
    -> Result<std::vector<double>>;

}  // namespace nimblecas::yieldcurve

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::yieldcurve {
namespace {

// Largest pillar count we will accept for any curve/conversion (DoS guard on untrusted
// sizes — well beyond any real term structure, which has tens of pillars).
inline constexpr std::size_t kMaxPillars = 100'000;
// Hull-White lattice bounds (DoS guards on node count).
inline constexpr int kMaxSteps = 20'000;
inline constexpr int kMaxJmax = 2'000;

// Resolve a (Compounding, freq) pair to periods-per-year m; m == 0 means continuous.
[[nodiscard]] auto resolve_m(Compounding comp, int freq) -> Result<int> {
    switch (comp) {
        case Compounding::continuous: return 0;
        case Compounding::annual:     return 1;
        case Compounding::semiannual: return 2;
        case Compounding::k_per_year:
            if (freq < 1 || freq > 366) { return make_error<int>(MathError::domain_error); }
            return freq;
    }
    return make_error<int>(MathError::domain_error);
}

// DF from a zero rate given resolved m (0 == continuous). t == 0 -> 1.
[[nodiscard]] auto df_from_zero_m(double z, double t, int m) -> Result<double> {
    if (!std::isfinite(z) || !std::isfinite(t) || t < 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    if (t == 0.0) { return 1.0; }
    double df = 0.0;
    if (m == 0) {
        df = std::exp(-z * t);
    } else {
        const double base = 1.0 + z / static_cast<double>(m);
        if (base <= 0.0) { return make_error<double>(MathError::domain_error); }  // complex power
        df = std::pow(base, -static_cast<double>(m) * t);
    }
    if (!std::isfinite(df) || df <= 0.0) { return make_error<double>(MathError::domain_error); }
    return df;
}

// Zero rate from a DF given resolved m. df > 0 and t > 0 required.
[[nodiscard]] auto zero_from_df_m(double df, double t, int m) -> Result<double> {
    if (!std::isfinite(df) || df <= 0.0 || !std::isfinite(t) || t <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    double z = 0.0;
    if (m == 0) {
        z = -std::log(df) / t;
    } else {
        z = static_cast<double>(m) * (std::pow(df, -1.0 / (static_cast<double>(m) * t)) - 1.0);
    }
    if (!std::isfinite(z)) { return make_error<double>(MathError::domain_error); }
    return z;
}

// Growth factor over an interval of length tau at forward rate f: the factor by which one
// unit of currency grows, so DF over the interval is its reciprocal.
[[nodiscard]] auto growth_factor_m(double f, double tau, int m) -> Result<double> {
    if (!std::isfinite(f) || !std::isfinite(tau) || tau <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    double g = 0.0;
    if (m == 0) {
        g = std::exp(f * tau);
    } else {
        const double base = 1.0 + f / static_cast<double>(m);
        if (base <= 0.0) { return make_error<double>(MathError::domain_error); }
        g = std::pow(base, static_cast<double>(m) * tau);
    }
    if (!std::isfinite(g) || g <= 0.0) { return make_error<double>(MathError::domain_error); }
    return g;
}

// Forward rate over an interval of length tau given the growth factor g = DF_start/DF_end.
[[nodiscard]] auto forward_from_growth_m(double g, double tau, int m) -> Result<double> {
    if (!std::isfinite(g) || g <= 0.0 || !std::isfinite(tau) || tau <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    double f = 0.0;
    if (m == 0) {
        f = std::log(g) / tau;
    } else {
        f = static_cast<double>(m) * (std::pow(g, 1.0 / (static_cast<double>(m) * tau)) - 1.0);
    }
    if (!std::isfinite(f)) { return make_error<double>(MathError::domain_error); }
    return f;
}

// Validate a strictly-increasing positive time grid of matching lengths, and reject
// oversized inputs. Returns the resolved m on success.
[[nodiscard]] auto validate_grid(std::span<const double> times, std::size_t vals_size,
                                 Compounding comp, int freq) -> Result<int> {
    if (times.empty() || times.size() != vals_size || times.size() > kMaxPillars) {
        return make_error<int>(MathError::domain_error);
    }
    if (!std::isfinite(times.front()) || times.front() <= 0.0) {
        return make_error<int>(MathError::domain_error);
    }
    for (std::size_t i = 1; i < times.size(); ++i) {
        if (!std::isfinite(times[i]) || times[i] <= times[i - 1]) {
            return make_error<int>(MathError::domain_error);
        }
    }
    return resolve_m(comp, freq);
}

// Linear interpolation weight for x in [x0, x1] (x1 > x0 guaranteed by the grid invariant).
[[nodiscard]] auto lerp(double y0, double y1, double x0, double x1, double x) -> double {
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// Small dense linear solve A x = b (n <= 6) by Gaussian elimination with partial pivoting.
// A near-singular pivot -> domain_error (an honest "no unique solution").
[[nodiscard]] auto solve_dense(std::vector<std::vector<double>> A, std::vector<double> b)
    -> Result<std::vector<double>> {
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        double best = std::abs(A[col][col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            if (std::abs(A[r][col]) > best) { best = std::abs(A[r][col]); piv = r; }
        }
        if (best < 1e-14) { return make_error<std::vector<double>>(MathError::domain_error); }
        if (piv != col) { std::swap(A[piv], A[col]); std::swap(b[piv], b[col]); }
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = A[r][col] / A[col][col];
            for (std::size_t c = col; c < n; ++c) { A[r][c] -= factor * A[col][c]; }
            b[r] -= factor * b[col];
        }
    }
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        double s = b[ii];
        for (std::size_t c = ii + 1; c < n; ++c) { s -= A[ii][c] * x[c]; }
        x[ii] = s / A[ii][ii];
    }
    return x;
}

// Ordinary least squares beta = argmin ||X beta - y|| via the normal equations XᵀX b = Xᵀy.
[[nodiscard]] auto least_squares(const std::vector<std::vector<double>>& X,
                                 std::span<const double> y) -> Result<std::vector<double>> {
    if (X.empty() || X.size() != y.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t p = X.front().size();
    std::vector<std::vector<double>> A(p, std::vector<double>(p, 0.0));
    std::vector<double> g(p, 0.0);
    for (std::size_t r = 0; r < X.size(); ++r) {
        for (std::size_t i = 0; i < p; ++i) {
            g[i] += X[r][i] * y[r];
            for (std::size_t j = 0; j < p; ++j) { A[i][j] += X[r][i] * X[r][j]; }
        }
    }
    return solve_dense(std::move(A), std::move(g));
}

// Nelson-Siegel basis functions [1, slope, curvature] at time t for decay tau.
[[nodiscard]] auto ns_basis(double t, double tau) -> std::array<double, 3> {
    const double u = t / tau;
    if (u < 1e-8) { return {1.0, 1.0, 0.0}; }  // limits as u -> 0
    const double e = std::exp(-u);
    const double slope = (1.0 - e) / u;
    return {1.0, slope, slope - e};
}

// Svensson curvature term for decay tau at time t (the b3 loading uses this with tau2).
[[nodiscard]] auto sv_curv(double t, double tau) -> double {
    const double u = t / tau;
    if (u < 1e-8) { return 0.0; }
    const double e = std::exp(-u);
    return (1.0 - e) / u - e;
}

// Golden-section minimiser of a 1-D objective on [lo, hi] (bounded iterations, no
// derivatives). Returns the argmin estimate.
template <typename F>
[[nodiscard]] auto golden_min(F&& f, double lo, double hi, int iters) -> double {
    constexpr double inv_phi = 0.6180339887498949;  // 1/phi
    double c = hi - inv_phi * (hi - lo);
    double d = lo + inv_phi * (hi - lo);
    double fc = f(c);
    double fd = f(d);
    for (int i = 0; i < iters; ++i) {
        if (fc < fd) {
            hi = d; d = c; fd = fc;
            c = hi - inv_phi * (hi - lo); fc = f(c);
        } else {
            lo = c; c = d; fc = fd;
            d = lo + inv_phi * (hi - lo); fd = f(d);
        }
    }
    return 0.5 * (lo + hi);
}

}  // namespace

// --- Point conversions ------------------------------------------------------

auto discount_from_zero(double zero, double t, Compounding comp, int freq) -> Result<double> {
    auto m = resolve_m(comp, freq);
    if (!m) { return make_error<double>(m.error()); }
    return df_from_zero_m(zero, t, *m);
}

auto zero_from_discount(double df, double t, Compounding comp, int freq) -> Result<double> {
    auto m = resolve_m(comp, freq);
    if (!m) { return make_error<double>(m.error()); }
    return zero_from_df_m(df, t, *m);
}

// --- Curve ------------------------------------------------------------------

auto Curve::create(std::vector<double> times, std::vector<double> zeros, Interp interp,
                   Compounding comp, int freq) -> Result<Curve> {
    auto m = validate_grid(times, zeros.size(), comp, freq);
    if (!m) { return make_error<Curve>(m.error()); }
    Curve c;
    c.times_ = std::move(times);
    c.zeros_ = std::move(zeros);
    c.interp_ = interp;
    c.comp_ = comp;
    c.freq_ = freq;
    c.m_ = *m;
    c.df_.reserve(c.times_.size());
    c.logdf_.reserve(c.times_.size());
    for (std::size_t i = 0; i < c.times_.size(); ++i) {
        if (!std::isfinite(c.zeros_[i])) { return make_error<Curve>(MathError::domain_error); }
        auto df = df_from_zero_m(c.zeros_[i], c.times_[i], c.m_);
        if (!df) { return make_error<Curve>(df.error()); }
        c.df_.push_back(*df);
        c.logdf_.push_back(std::log(*df));
    }
    return c;
}

auto Curve::discount_factor(double t) const -> Result<double> {
    if (times_.empty() || !std::isfinite(t)) { return make_error<double>(MathError::domain_error); }
    const double lo = times_.front();
    const double hi = times_.back();
    // A tiny endpoint tolerance so a node time recomputed via arithmetic still counts as
    // in-range; genuine out-of-range queries remain a domain_error (no extrapolation).
    const double tol = 1e-9 * std::max(1.0, hi);
    if (t < lo - tol || t > hi + tol) { return make_error<double>(MathError::domain_error); }
    const double tc = std::clamp(t, lo, hi);
    if (times_.size() == 1) { return df_[0]; }  // single pillar: only t == t0 is in range
    // Locate the enclosing interval [times_[i], times_[i+1]].
    std::size_t i = 0;
    while (i + 1 < times_.size() && times_[i + 1] < tc) { ++i; }
    if (i + 1 >= times_.size()) { i = times_.size() - 2; }  // tc == hi
    if (interp_ == Interp::linear_zero) {
        const double z = lerp(zeros_[i], zeros_[i + 1], times_[i], times_[i + 1], tc);
        return df_from_zero_m(z, tc, m_);
    }
    const double lz = lerp(logdf_[i], logdf_[i + 1], times_[i], times_[i + 1], tc);
    const double df = std::exp(lz);
    if (!std::isfinite(df) || df <= 0.0) { return make_error<double>(MathError::domain_error); }
    return df;
}

auto Curve::zero_rate(double t) const -> Result<double> {
    auto df = discount_factor(t);
    if (!df) { return df; }
    return zero_from_df_m(*df, std::clamp(t, times_.front(), times_.back()), m_);
}

auto Curve::forward_rate(double t1, double t2) const -> Result<double> {
    if (!(t2 > t1)) { return make_error<double>(MathError::domain_error); }
    auto d1 = discount_factor(t1);
    if (!d1) { return d1; }
    auto d2 = discount_factor(t2);
    if (!d2) { return d2; }
    if (*d2 <= 0.0) { return make_error<double>(MathError::domain_error); }
    return forward_from_growth_m(*d1 / *d2, t2 - t1, m_);
}

// --- Term-structure conversions --------------------------------------------

auto zero2disc(std::span<const double> times, std::span<const double> zeros, Compounding comp,
               int freq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, zeros.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    std::vector<double> out;
    out.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto df = df_from_zero_m(zeros[i], times[i], *m);
        if (!df) { return make_error<std::vector<double>>(df.error()); }
        out.push_back(*df);
    }
    return out;
}

auto disc2zero(std::span<const double> times, std::span<const double> discs, Compounding comp,
               int freq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, discs.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    std::vector<double> out;
    out.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto z = zero_from_df_m(discs[i], times[i], *m);
        if (!z) { return make_error<std::vector<double>>(z.error()); }
        out.push_back(*z);
    }
    return out;
}

auto zero2fwd(std::span<const double> times, std::span<const double> zeros, Compounding comp,
              int freq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, zeros.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    std::vector<double> out;
    out.reserve(times.size());
    double df_prev = 1.0;   // DF(0)
    double t_prev = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto df = df_from_zero_m(zeros[i], times[i], *m);
        if (!df) { return make_error<std::vector<double>>(df.error()); }
        auto f = forward_from_growth_m(df_prev / *df, times[i] - t_prev, *m);
        if (!f) { return make_error<std::vector<double>>(f.error()); }
        out.push_back(*f);
        df_prev = *df;
        t_prev = times[i];
    }
    return out;
}

auto fwd2zero(std::span<const double> times, std::span<const double> fwds, Compounding comp,
              int freq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, fwds.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    std::vector<double> out;
    out.reserve(times.size());
    double df_prev = 1.0;
    double t_prev = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto g = growth_factor_m(fwds[i], times[i] - t_prev, *m);
        if (!g) { return make_error<std::vector<double>>(g.error()); }
        const double df = df_prev / *g;
        auto z = zero_from_df_m(df, times[i], *m);
        if (!z) { return make_error<std::vector<double>>(z.error()); }
        out.push_back(*z);
        df_prev = df;
        t_prev = times[i];
    }
    return out;
}

// --- Bootstrapping ----------------------------------------------------------

namespace {

// Cashflow schedule (times in years, amounts per the bond's `face`) for a bond. A
// zero-coupon bond is a single `face` cashflow at maturity; otherwise coupon dates are
// maturity, maturity-1/f, ... back from maturity so the anchor never drifts.
[[nodiscard]] auto bond_cashflows(const CouponBond& b)
    -> Result<std::pair<std::vector<double>, std::vector<double>>> {
    if (!(b.maturity > 0.0) || !std::isfinite(b.maturity) || !std::isfinite(b.face)) {
        return make_error<std::pair<std::vector<double>, std::vector<double>>>(
            MathError::domain_error);
    }
    std::vector<double> times;
    std::vector<double> amts;
    if (b.coupon_rate == 0.0) {
        times.push_back(b.maturity);
        amts.push_back(b.face);
        return std::pair{std::move(times), std::move(amts)};
    }
    if (b.frequency != 1 && b.frequency != 2 && b.frequency != 4) {
        return make_error<std::pair<std::vector<double>, std::vector<double>>>(
            MathError::domain_error);
    }
    const double f = static_cast<double>(b.frequency);
    const int np = static_cast<int>(std::llround(b.maturity * f));
    if (np < 1 || np > static_cast<int>(kMaxPillars)) {
        return make_error<std::pair<std::vector<double>, std::vector<double>>>(
            MathError::domain_error);
    }
    const double coupon = b.face * b.coupon_rate / f;
    for (int k = 1; k <= np; ++k) {
        times.push_back(b.maturity - static_cast<double>(np - k) / f);
        amts.push_back(coupon + (k == np ? b.face : 0.0));
    }
    return std::pair{std::move(times), std::move(amts)};
}

}  // namespace

auto zbtprice(std::span<const CouponBond> bonds, std::span<const double> prices, Compounding comp,
              int freq, Interp interp) -> Result<Curve> {
    if (bonds.empty() || bonds.size() != prices.size() || bonds.size() > kMaxPillars) {
        return make_error<Curve>(MathError::domain_error);
    }
    auto m = resolve_m(comp, freq);
    if (!m) { return make_error<Curve>(m.error()); }
    std::vector<double> ptimes;  // bootstrapped pillar times
    std::vector<double> pdfs;    // discount factors at those pillars
    ptimes.reserve(bonds.size());
    pdfs.reserve(bonds.size());

    // Discount factor at time t on the curve built so far (log-linear in ln DF; no
    // extrapolation below the first pillar -> underdetermined -> domain_error).
    auto df_at = [&](double t) -> Result<double> {
        if (ptimes.empty() || t < ptimes.front() - 1e-9 || t > ptimes.back() + 1e-9) {
            return make_error<double>(MathError::domain_error);
        }
        std::size_t i = 0;
        while (i + 1 < ptimes.size() && ptimes[i + 1] < t) { ++i; }
        if (i + 1 >= ptimes.size()) { return pdfs.back(); }
        const double lz = lerp(std::log(pdfs[i]), std::log(pdfs[i + 1]), ptimes[i], ptimes[i + 1], t);
        return std::exp(lz);
    };

    for (std::size_t j = 0; j < bonds.size(); ++j) {
        if (j > 0 && !(bonds[j].maturity > bonds[j - 1].maturity)) {
            return make_error<Curve>(MathError::domain_error);  // maturities must increase
        }
        auto cf = bond_cashflows(bonds[j]);
        if (!cf) { return make_error<Curve>(cf.error()); }
        const auto& [ct, ca] = *cf;
        const std::size_t last = ct.size() - 1;
        double known = 0.0;
        for (std::size_t i = 0; i < last; ++i) {  // intermediate coupons discount on prior curve
            auto d = df_at(ct[i]);
            if (!d) { return make_error<Curve>(d.error()); }
            known += ca[i] * *d;
        }
        if (ca[last] == 0.0) { return make_error<Curve>(MathError::domain_error); }
        const double df_final = (prices[j] - known) / ca[last];
        if (!std::isfinite(df_final) || df_final <= 0.0) {
            return make_error<Curve>(MathError::domain_error);
        }
        ptimes.push_back(bonds[j].maturity);
        pdfs.push_back(df_final);
    }

    std::vector<double> zeros;
    zeros.reserve(pdfs.size());
    for (std::size_t i = 0; i < pdfs.size(); ++i) {
        auto z = zero_from_df_m(pdfs[i], ptimes[i], *m);
        if (!z) { return make_error<Curve>(z.error()); }
        zeros.push_back(*z);
    }
    return Curve::create(std::move(ptimes), std::move(zeros), interp, comp, freq);
}

auto zbtyield(std::span<const CouponBond> bonds, std::span<const double> yields, Compounding comp,
              int freq, Interp interp) -> Result<Curve> {
    if (bonds.empty() || bonds.size() != yields.size() || bonds.size() > kMaxPillars) {
        return make_error<Curve>(MathError::domain_error);
    }
    std::vector<double> prices;
    prices.reserve(bonds.size());
    for (std::size_t j = 0; j < bonds.size(); ++j) {
        const CouponBond& b = bonds[j];
        const double y = yields[j];
        if (!std::isfinite(y)) { return make_error<Curve>(MathError::domain_error); }
        if (b.coupon_rate == 0.0) {
            const double base = 1.0 + y;  // zero-coupon bill priced annually
            if (base <= 0.0) { return make_error<Curve>(MathError::domain_error); }
            prices.push_back(b.face * std::pow(base, -b.maturity));
            continue;
        }
        auto cf = bond_cashflows(b);
        if (!cf) { return make_error<Curve>(cf.error()); }
        const auto& [ct, ca] = *cf;
        const double f = static_cast<double>(b.frequency);
        const double per = 1.0 + y / f;
        if (per <= 0.0) { return make_error<Curve>(MathError::domain_error); }
        double price = 0.0;
        for (std::size_t i = 0; i < ct.size(); ++i) {
            const double k = ct[i] * f;  // integer period index of this coupon
            price += ca[i] * std::pow(per, -k);
        }
        prices.push_back(price);
    }
    return zbtprice(bonds, prices, comp, freq, interp);
}

// --- Par <-> zero -----------------------------------------------------------

auto par2zero(std::span<const double> times, std::span<const double> par_yields, Compounding comp,
              int freq, int couponfreq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, par_yields.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    if (couponfreq < 1 || couponfreq > 366) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const double f = static_cast<double>(couponfreq);
    std::vector<double> dfs;
    dfs.reserve(times.size());
    double annuity = 0.0;  // sum of DF over prior pillars
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double c = par_yields[i] / f;
        const double denom = 1.0 + c;
        if (denom <= 0.0) { return make_error<std::vector<double>>(MathError::domain_error); }
        const double df = (1.0 - c * annuity) / denom;
        if (!std::isfinite(df) || df <= 0.0) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        dfs.push_back(df);
        annuity += df;
    }
    return disc2zero(times, dfs, comp, freq);
}

auto zero2par(std::span<const double> times, std::span<const double> zeros, Compounding comp,
              int freq, int couponfreq) -> Result<std::vector<double>> {
    auto m = validate_grid(times, zeros.size(), comp, freq);
    if (!m) { return make_error<std::vector<double>>(m.error()); }
    if (couponfreq < 1 || couponfreq > 366) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const double f = static_cast<double>(couponfreq);
    std::vector<double> out;
    out.reserve(times.size());
    double annuity = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto df = df_from_zero_m(zeros[i], times[i], *m);
        if (!df) { return make_error<std::vector<double>>(df.error()); }
        annuity += *df;
        if (annuity <= 0.0) { return make_error<std::vector<double>>(MathError::domain_error); }
        out.push_back(f * (1.0 - *df) / annuity);
    }
    return out;
}

// --- Nelson-Siegel / Svensson ----------------------------------------------

auto NelsonSiegel::zero_rate(double t) const -> Result<double> {
    if (!(t >= 0.0) || !std::isfinite(t) || tau <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const auto b = ns_basis(t, tau);
    const double z = beta0 * b[0] + beta1 * b[1] + beta2 * b[2];
    if (!std::isfinite(z)) { return make_error<double>(MathError::domain_error); }
    return z;
}

auto NelsonSiegel::discount_factor(double t, Compounding comp, int freq) const -> Result<double> {
    auto z = zero_rate(t);
    if (!z) { return z; }
    return discount_from_zero(*z, t, comp, freq);
}

auto Svensson::zero_rate(double t) const -> Result<double> {
    if (!(t >= 0.0) || !std::isfinite(t) || tau1 <= 0.0 || tau2 <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const auto b1 = ns_basis(t, tau1);
    const double z = beta0 * b1[0] + beta1 * b1[1] + beta2 * b1[2] + beta3 * sv_curv(t, tau2);
    if (!std::isfinite(z)) { return make_error<double>(MathError::domain_error); }
    return z;
}

auto Svensson::discount_factor(double t, Compounding comp, int freq) const -> Result<double> {
    auto z = zero_rate(t);
    if (!z) { return z; }
    return discount_from_zero(*z, t, comp, freq);
}

auto fit_nelson_siegel(std::span<const double> times, std::span<const double> zeros)
    -> Result<NelsonSiegel> {
    if (times.size() != zeros.size() || times.size() < 4 || times.size() > kMaxPillars) {
        return make_error<NelsonSiegel>(MathError::domain_error);
    }
    double t_max = 0.0;
    for (double t : times) {
        if (!std::isfinite(t) || t <= 0.0) { return make_error<NelsonSiegel>(MathError::domain_error); }
        t_max = std::max(t_max, t);
    }
    // SSE(tau): inner linear LSQ for the three betas, then residual sum of squares. A
    // singular/non-finite system reports the sentinel max() so the search skips it.
    auto sse = [&](double tau) -> double {
        if (tau <= 0.0) { return std::numeric_limits<double>::max(); }
        std::vector<std::vector<double>> X;
        X.reserve(times.size());
        for (double t : times) {
            const auto b = ns_basis(t, tau);
            X.push_back({b[0], b[1], b[2]});
        }
        auto beta = least_squares(X, zeros);
        if (!beta) { return std::numeric_limits<double>::max(); }
        double s = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            const auto b = ns_basis(times[i], tau);
            const double pred = (*beta)[0] * b[0] + (*beta)[1] * b[1] + (*beta)[2] * b[2];
            const double r = pred - zeros[i];
            s += r * r;
        }
        return std::isfinite(s) ? s : std::numeric_limits<double>::max();
    };
    // Coarse log-spaced scan over tau, then golden-section refine within the best bracket.
    const double lo = 0.05;
    const double hi = std::max(30.0, 2.0 * t_max);
    constexpr int kGrid = 80;
    double best_tau = lo;
    double best_sse = std::numeric_limits<double>::max();
    std::size_t best_k = 0;
    std::vector<double> grid(kGrid);
    for (int k = 0; k < kGrid; ++k) {
        const double frac = static_cast<double>(k) / static_cast<double>(kGrid - 1);
        grid[static_cast<std::size_t>(k)] = lo * std::pow(hi / lo, frac);
        const double s = sse(grid[static_cast<std::size_t>(k)]);
        if (s < best_sse) { best_sse = s; best_tau = grid[static_cast<std::size_t>(k)]; best_k = static_cast<std::size_t>(k); }
    }
    if (best_sse == std::numeric_limits<double>::max()) {
        return make_error<NelsonSiegel>(MathError::not_converged);
    }
    const double rlo = grid[best_k > 0 ? best_k - 1 : 0];
    const double rhi = grid[best_k + 1 < grid.size() ? best_k + 1 : grid.size() - 1];
    best_tau = golden_min(sse, rlo, rhi, 80);
    std::vector<std::vector<double>> X;
    X.reserve(times.size());
    for (double t : times) {
        const auto b = ns_basis(t, best_tau);
        X.push_back({b[0], b[1], b[2]});
    }
    auto beta = least_squares(X, zeros);
    if (!beta) { return make_error<NelsonSiegel>(MathError::not_converged); }
    NelsonSiegel ns{(*beta)[0], (*beta)[1], (*beta)[2], best_tau};
    if (!std::isfinite(ns.beta0) || !std::isfinite(ns.beta1) || !std::isfinite(ns.beta2)) {
        return make_error<NelsonSiegel>(MathError::not_converged);
    }
    return ns;
}

auto fit_svensson(std::span<const double> times, std::span<const double> zeros)
    -> Result<Svensson> {
    if (times.size() != zeros.size() || times.size() < 6 || times.size() > kMaxPillars) {
        return make_error<Svensson>(MathError::domain_error);
    }
    double t_max = 0.0;
    for (double t : times) {
        if (!std::isfinite(t) || t <= 0.0) { return make_error<Svensson>(MathError::domain_error); }
        t_max = std::max(t_max, t);
    }
    auto solve_beta = [&](double tau1, double tau2) -> Result<std::array<double, 4>> {
        std::vector<std::vector<double>> X;
        X.reserve(times.size());
        for (double t : times) {
            const auto b = ns_basis(t, tau1);
            X.push_back({b[0], b[1], b[2], sv_curv(t, tau2)});
        }
        auto beta = least_squares(X, zeros);
        if (!beta) { return make_error<std::array<double, 4>>(MathError::domain_error); }
        return std::array<double, 4>{(*beta)[0], (*beta)[1], (*beta)[2], (*beta)[3]};
    };
    auto sse = [&](double tau1, double tau2) -> double {
        if (tau1 <= 0.0 || tau2 <= 0.0 || std::abs(tau1 - tau2) < 1e-6) {
            return std::numeric_limits<double>::max();  // coincident decays -> collinear columns
        }
        auto beta = solve_beta(tau1, tau2);
        if (!beta) { return std::numeric_limits<double>::max(); }
        double s = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            const auto b = ns_basis(times[i], tau1);
            const double pred = (*beta)[0] * b[0] + (*beta)[1] * b[1] + (*beta)[2] * b[2] +
                                (*beta)[3] * sv_curv(times[i], tau2);
            const double r = pred - zeros[i];
            s += r * r;
        }
        return std::isfinite(s) ? s : std::numeric_limits<double>::max();
    };
    const double lo = 0.1;
    const double hi = std::max(30.0, 2.0 * t_max);
    constexpr int kGrid = 24;
    double best1 = 1.0;
    double best2 = 3.0;
    double best_sse = std::numeric_limits<double>::max();
    for (int i = 0; i < kGrid; ++i) {
        const double t1 = lo * std::pow(hi / lo, static_cast<double>(i) / (kGrid - 1));
        for (int k = 0; k < kGrid; ++k) {
            const double t2 = lo * std::pow(hi / lo, static_cast<double>(k) / (kGrid - 1));
            if (t2 <= t1 * 1.05) { continue; }  // keep tau2 > tau1 for identifiability
            const double s = sse(t1, t2);
            if (s < best_sse) { best_sse = s; best1 = t1; best2 = t2; }
        }
    }
    if (best_sse == std::numeric_limits<double>::max()) {
        return make_error<Svensson>(MathError::not_converged);
    }
    // Alternating (coordinate) golden-section refine on tau1 then tau2, a few rounds.
    for (int round = 0; round < 6; ++round) {
        best1 = golden_min([&](double x) { return sse(x, best2); }, lo, best2 * 0.98, 40);
        best2 = golden_min([&](double x) { return sse(best1, x); }, best1 * 1.02, hi, 40);
    }
    auto beta = solve_beta(best1, best2);
    if (!beta) { return make_error<Svensson>(MathError::not_converged); }
    Svensson sv{(*beta)[0], (*beta)[1], (*beta)[2], (*beta)[3], best1, best2};
    for (double v : {sv.beta0, sv.beta1, sv.beta2, sv.beta3}) {
        if (!std::isfinite(v)) { return make_error<Svensson>(MathError::not_converged); }
    }
    return sv;
}

// --- Hull-White trinomial lattice ------------------------------------------

namespace {

// The three (child level offset, probability) pairs for a node at level j, given the tree
// half-width jmax and a*dt. Interior nodes branch (+1,0,-1); the top (j==jmax) branches
// down (0,-1,-2) and the bottom (j==-jmax) branches up (+2,+1,0) so the tree stays finite.
// A negative probability (a*dt out of the valid range) is surfaced as domain_error.
[[nodiscard]] auto hw_branch(int j, int jmax, double adt)
    -> Result<std::array<std::pair<int, double>, 3>> {
    const double aj = adt * static_cast<double>(j);  // a*j*dt
    const double a2 = aj * aj;                        // a^2 * j^2 * dt^2
    std::array<std::pair<int, double>, 3> br{};
    if (j == jmax) {
        br = {std::pair{0, 7.0 / 6.0 + (a2 - 3.0 * aj) / 2.0},
              std::pair{-1, -1.0 / 3.0 - a2 + 2.0 * aj},
              std::pair{-2, 1.0 / 6.0 + (a2 - aj) / 2.0}};
    } else if (j == -jmax) {
        br = {std::pair{2, 1.0 / 6.0 + (a2 + aj) / 2.0},
              std::pair{1, -1.0 / 3.0 - a2 - 2.0 * aj},
              std::pair{0, 7.0 / 6.0 + (a2 + 3.0 * aj) / 2.0}};
    } else {
        br = {std::pair{1, 1.0 / 6.0 + (a2 - aj) / 2.0},
              std::pair{0, 2.0 / 3.0 - a2},
              std::pair{-1, 1.0 / 6.0 + (a2 + aj) / 2.0}};
    }
    for (const auto& [off, p] : br) {
        if (!(p > -1e-12) || !std::isfinite(p)) {
            return make_error<std::array<std::pair<int, double>, 3>>(MathError::domain_error);
        }
    }
    return br;
}

}  // namespace

auto build_hull_white(const Curve& curve, double a, double sigma, double dt, int steps)
    -> Result<HullWhiteLattice> {
    if (!(a > 0.0) || !(sigma > 0.0) || !(dt > 0.0) || steps < 1 || steps > kMaxSteps) {
        return make_error<HullWhiteLattice>(MathError::domain_error);
    }
    const double adt = a * dt;
    const int jmax = static_cast<int>(std::ceil(0.184 / adt));
    if (jmax < 1 || jmax > kMaxJmax) {
        return make_error<HullWhiteLattice>(MathError::domain_error);  // a*dt too small -> huge tree
    }
    const double dx = sigma * std::sqrt(3.0 * dt);

    // Precompute the input discount factors P(0, i*dt) for i = 1..steps.
    std::vector<double> P(static_cast<std::size_t>(steps) + 1, 1.0);  // P[0] = 1
    for (int i = 1; i <= steps; ++i) {
        auto df = curve.discount_factor(static_cast<double>(i) * dt);
        if (!df) { return make_error<HullWhiteLattice>(df.error()); }
        P[static_cast<std::size_t>(i)] = *df;
    }

    // Forward induction on Arrow-Debreu (state) prices Q to determine the slice
    // displacements alpha_i that reprice P exactly.
    HullWhiteLattice lat;
    lat.a_ = a; lat.sigma_ = sigma; lat.dt_ = dt; lat.dx_ = dx; lat.jmax_ = jmax; lat.steps_ = steps;
    lat.alpha_.reserve(static_cast<std::size_t>(steps));

    std::vector<double> Q(1, 1.0);  // step 0: single node at j = 0
    int wprev = 0;                  // current half-width
    for (int i = 0; i < steps; ++i) {
        double numer = 0.0;
        for (int k = 0; k <= 2 * wprev; ++k) {
            const int j = k - wprev;
            numer += Q[static_cast<std::size_t>(k)] * std::exp(-static_cast<double>(j) * dx * dt);
        }
        const double pf = P[static_cast<std::size_t>(i) + 1];
        if (!(pf > 0.0) || numer <= 0.0) { return make_error<HullWhiteLattice>(MathError::domain_error); }
        const double alpha_i = std::log(numer / pf) / dt;
        if (!std::isfinite(alpha_i)) { return make_error<HullWhiteLattice>(MathError::domain_error); }
        lat.alpha_.push_back(alpha_i);

        const int wnext = std::min(i + 1, jmax);
        std::vector<double> Qn(static_cast<std::size_t>(2 * wnext + 1), 0.0);
        for (int k = 0; k <= 2 * wprev; ++k) {
            const int j = k - wprev;
            const double disc = std::exp(-(alpha_i + static_cast<double>(j) * dx) * dt);
            auto br = hw_branch(j, jmax, adt);
            if (!br) { return make_error<HullWhiteLattice>(br.error()); }
            for (const auto& [off, p] : *br) {
                const int childk = (j + off) + wnext;
                if (childk < 0 || childk > 2 * wnext) {
                    return make_error<HullWhiteLattice>(MathError::domain_error);
                }
                Qn[static_cast<std::size_t>(childk)] += Q[static_cast<std::size_t>(k)] * p * disc;
            }
        }
        Q = std::move(Qn);
        wprev = wnext;
    }
    return lat;
}

auto HullWhiteLattice::discount(double T) const -> Result<double> {
    if (alpha_.empty() || !std::isfinite(T) || T <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const int n = static_cast<int>(std::llround(T / dt_));
    if (n < 1 || n > steps_ || std::abs(static_cast<double>(n) * dt_ - T) > 1e-7 * std::max(1.0, T)) {
        return make_error<double>(MathError::domain_error);
    }
    const double adt = a_ * dt_;
    // Backward induction: unit payoff at the maturity slice, rolled to the root.
    const int wn = std::min(n, jmax_);
    std::vector<double> V(static_cast<std::size_t>(2 * wn + 1), 1.0);
    for (int i = n - 1; i >= 0; --i) {
        const int wi = std::min(i, jmax_);
        const int wchild = std::min(i + 1, jmax_);
        std::vector<double> Vn(static_cast<std::size_t>(2 * wi + 1), 0.0);
        for (int k = 0; k <= 2 * wi; ++k) {
            const int j = k - wi;
            const double disc = std::exp(-(alpha_[static_cast<std::size_t>(i)] +
                                           static_cast<double>(j) * dx_) * dt_);
            auto br = hw_branch(j, jmax_, adt);
            if (!br) { return make_error<double>(br.error()); }
            double cont = 0.0;
            for (const auto& [off, p] : *br) {
                const int childk = (j + off) + wchild;
                if (childk < 0 || childk > 2 * wchild) {
                    return make_error<double>(MathError::domain_error);
                }
                cont += p * V[static_cast<std::size_t>(childk)];
            }
            Vn[static_cast<std::size_t>(k)] = disc * cont;
        }
        V = std::move(Vn);
    }
    if (!std::isfinite(V[0]) || V[0] <= 0.0) { return make_error<double>(MathError::domain_error); }
    return V[0];
}

auto HullWhiteLattice::bond_price(std::span<const double> times, std::span<const double> cashflows)
    const -> Result<double> {
    if (times.empty() || times.size() != cashflows.size() || times.size() > kMaxPillars) {
        return make_error<double>(MathError::domain_error);
    }
    double price = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        auto d = discount(times[i]);
        if (!d) { return d; }
        price += cashflows[i] * *d;
    }
    return price;
}

// --- Calendar bridge --------------------------------------------------------

auto times_from_dates(const finance::Date& settlement, std::span<const finance::Date> pillars,
                      finance::DayCount basis) -> Result<std::vector<double>> {
    if (pillars.empty() || pillars.size() > kMaxPillars) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> out;
    out.reserve(pillars.size());
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        if (pillars[i].serial <= settlement.serial ||
            (i > 0 && pillars[i].serial <= pillars[i - 1].serial)) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        auto yf = finance::year_fraction(settlement, pillars[i], basis);
        if (!yf) { return make_error<std::vector<double>>(yf.error()); }
        out.push_back(yf->to_double());
    }
    return out;
}

}  // namespace nimblecas::yieldcurve
