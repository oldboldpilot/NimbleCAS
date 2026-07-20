// NimbleCAS derivatives pricing — options & instrument valuation.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. Black-Scholes-Merton closed form with full Greeks (the analytic oracle);
// Kamrad-Ritchken TRINOMIAL trees for European / American / Bermudan exercise (a trinomial
// lattice converges faster and more smoothly than a binomial one — the accuracy the owner
// asked for); reproducible MONTE CARLO for European and Asian (arithmetic-average) payoffs
// with the geometric-average closed form as a control variate AND a validation oracle;
// Longstaff-Schwartz least-squares Monte Carlo for American exercise; a composable
// Portfolio that aggregates price and Greeks across positions; and probability-curve /
// payoff / price-sweep plotting (terminal risk-neutral density, payoff and P&L diagrams,
// price-vs-parameter curves) rendered through nimblecas.svgplot.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). This module is NUMERICAL /
// STATISTICAL by nature — an option price is a limit (of a lattice as steps -> infinity, of
// a sample mean as paths -> infinity), and the very first quantity a tree computes,
// exp(sigma*sqrt(dt)), is transcendental. So NOTHING here claims exactness. Instead:
//   * closed forms (Black-Scholes, geometric Asian) are correctly-rounded double formulas;
//   * lattice prices carry O(dt) discretization error that shrinks with `steps`;
//   * Monte Carlo returns the estimate AND its standard error, and is BIT-REPRODUCIBLE and
//     PARTITION-INDEPENDENT by construction — every draw is a pure function of its global
//     index via nimblecas.rng's counter_u64, inheriting the design guarantee documented in
//     nimblecas.montecarlo (no time/entropy seeding anywhere; equal seeds reproduce equal
//     results bit-for-bit, and any partition of the path range reproduces the serial mean).
// All failure rides the railway (Result<T> / MathError); nothing throws.

module;
#include <cassert>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>  // AVX-512 batched inverse-normal transform (normal_fill hot path)
#endif

export module nimblecas.pricing;

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.svgplot;
import nimblecas.parallel;
import nimblecas.simd;

export namespace nimblecas::pricing {

enum class OptionType : std::uint8_t { call, put };
enum class Exercise : std::uint8_t { european, american, bermudan };

// Immutable option contract + market state. Built fluently (Rules 15/47) via the with_*
// chain; every field has a sensible default so only the relevant ones need setting.
struct OptionSpec {
    double spot{100.0};              // S — current underlying price
    double strike{100.0};            // K
    double rate{0.0};                // r — continuously-compounded risk-free rate
    double dividend_yield{0.0};      // q — continuous dividend yield
    double volatility{0.2};          // sigma — annualised
    double time_to_expiry{1.0};      // T in years
    OptionType type{OptionType::call};

    [[nodiscard]] auto with_spot(double s) const -> OptionSpec { auto c = *this; c.spot = s; return c; }
    [[nodiscard]] auto with_strike(double k) const -> OptionSpec { auto c = *this; c.strike = k; return c; }
    [[nodiscard]] auto with_rate(double r) const -> OptionSpec { auto c = *this; c.rate = r; return c; }
    [[nodiscard]] auto with_dividend(double q) const -> OptionSpec { auto c = *this; c.dividend_yield = q; return c; }
    [[nodiscard]] auto with_volatility(double v) const -> OptionSpec { auto c = *this; c.volatility = v; return c; }
    [[nodiscard]] auto with_expiry(double t) const -> OptionSpec { auto c = *this; c.time_to_expiry = t; return c; }
    [[nodiscard]] auto with_type(OptionType t) const -> OptionSpec { auto c = *this; c.type = t; return c; }

    // Intrinsic payoff at underlying price s.
    [[nodiscard]] auto payoff(double s) const -> double {
        return type == OptionType::call ? std::max(s - strike, 0.0) : std::max(strike - s, 0.0);
    }
};

// Price plus first- and second-order Greeks. theta is per-year; vega/rho per unit (1.00)
// change in vol/rate (divide by 100 for per-percent).
struct Greeks {
    double price{0.0};
    double delta{0.0};
    double gamma{0.0};
    double vega{0.0};
    double theta{0.0};
    double rho{0.0};
};

// A Monte Carlo estimate with its sampling error and the antithetic/CV method used.
struct McResult {
    double price{0.0};
    double std_error{0.0};
    std::uint64_t paths{0};
    // A 95%-ish confidence half-width (1.96 * std_error), for reporting.
    [[nodiscard]] auto confidence_half_width() const -> double { return 1.96 * std_error; }
};

// --- Standard-normal helpers (exposed: useful to callers and to plotting) ---
[[nodiscard]] auto norm_pdf(double x) -> double;
[[nodiscard]] auto norm_cdf(double x) -> double;
[[nodiscard]] auto inverse_norm_cdf(double p) -> Result<double>;

// --- Black-Scholes-Merton closed form (European) ---------------------------
[[nodiscard]] auto black_scholes_price(const OptionSpec& spec) -> Result<double>;
[[nodiscard]] auto black_scholes_greeks(const OptionSpec& spec) -> Result<Greeks>;
// Implied volatility from a market price (bracketed root of BS(vol) = market_price).
[[nodiscard]] auto implied_volatility(const OptionSpec& spec, double market_price) -> Result<double>;

// Higher-order ("exotic") Greeks — the full analytic sensitivity set beyond the first five.
// vanna = d delta / d vol; charm = -d delta / d time; vomma/volga = d vega / d vol;
// veta = d vega / d time; speed = d gamma / d spot; zomma = d gamma / d vol;
// color = d gamma / d time; lambda = delta * spot / price (elasticity/gearing);
// dual_delta = d price / d strike; dual_gamma = d^2 price / d strike^2;
// epsilon = d price / d dividend_yield; vera = d rho / d vol;
// ultima = d vomma / d vol (third-order vol).
struct ExtendedGreeks {
    double vanna{0.0};
    double charm{0.0};
    double vomma{0.0};   // == volga
    double veta{0.0};
    double speed{0.0};
    double zomma{0.0};
    double color{0.0};
    double lambda{0.0};
    double dual_delta{0.0};
    double dual_gamma{0.0};
    double epsilon{0.0};  // d price / d dividend_yield (a.k.a. psi)
    double vera{0.0};     // d rho / d vol (a.k.a. rhova)
    double ultima{0.0};   // d vomma / d vol == d^3 price / d vol^3
};
[[nodiscard]] auto black_scholes_extended_greeks(const OptionSpec& spec) -> Result<ExtendedGreeks>;

// Option profit/loss. pnl_at_expiry: realised P&L of a long option at underlying price s_T,
// net of the premium paid. mark_to_market_pnl: unrealised P&L = current BS value - premium.
[[nodiscard]] auto option_pnl_at_expiry(const OptionSpec& spec, double premium, double s_T)
    -> double;
[[nodiscard]] auto option_mark_to_market_pnl(const OptionSpec& spec, double premium)
    -> Result<double>;

// Black-76: option on a FORWARD/FUTURES price F (futures options, caps/floors substrate).
// price = e^{-rT}[F N(d1) - K N(d2)] for a call. Returns price and delta w.r.t. F.
[[nodiscard]] auto black76_price(bool is_call, double forward, double strike, double rate,
                                 double volatility, double time) -> Result<double>;

// European digital (binary) options, closed form. cash_or_nothing pays `cash` if in the
// money; asset_or_nothing pays the underlying if in the money.
[[nodiscard]] auto digital_cash_or_nothing(const OptionSpec& spec, double cash) -> Result<double>;
[[nodiscard]] auto digital_asset_or_nothing(const OptionSpec& spec) -> Result<double>;

// Single-barrier option by reproducible Monte Carlo with discrete monitoring at `steps`
// dates. `knock_in` selects in vs out; `barrier` is the level. Returns the estimate and its
// standard error. STATISTICAL: price carries MC error and monitoring bias (documented). The
// in/out parity in-price + out-price == vanilla holds up to MC error — a checked invariant.
[[nodiscard]] auto barrier_option_mc(const OptionSpec& spec, double barrier, bool knock_in,
                                     std::uint64_t paths, int steps, std::uint64_t seed)
    -> Result<McResult>;

// --- Trinomial lattice (Kamrad-Ritchken) -----------------------------------
// European/American use `steps` time steps. For Bermudan, `exercise_times` lists the years
// at which early exercise is permitted (mapped to the nearest step); an empty list with
// Exercise::bermudan behaves European.
[[nodiscard]] auto trinomial_price(const OptionSpec& spec, int steps, Exercise exercise,
                                   std::span<const double> exercise_times = {}) -> Result<double>;

// --- Monte Carlo (reproducible, partition-independent) ---------------------
// European MC with antithetic variates. `paths` sample terminal prices; `seed` fully
// determines the draw stream.
[[nodiscard]] auto monte_carlo_european(const OptionSpec& spec, std::uint64_t paths,
                                        std::uint64_t seed) -> Result<McResult>;
// Parallel European MC: identical model/draws as monte_carlo_european, but the paths are priced
// in a FIXED block decomposition across the fork-join runtime and the block partials combined in
// index order. The result depends only on (spec, paths, seed) — NOT on thread count or scheduling
// (deterministic/reproducible). It equals monte_carlo_european to floating-point tolerance (the
// difference is FP non-associativity of the sum, far below the MC standard error), NOT bit-for-bit.
// perf shows European MC is compute/transcendental-bound, so this scales ~linearly with cores.
[[nodiscard]] auto monte_carlo_european_parallel(const OptionSpec& spec, std::uint64_t paths,
                                                 std::uint64_t seed) -> Result<McResult>;
// Arithmetic-average Asian option via MC, using the geometric-average price as a CONTROL
// VARIATE (exact closed form below) — a large variance reduction. `steps` averaging dates.
[[nodiscard]] auto monte_carlo_asian(const OptionSpec& spec, std::uint64_t paths, int steps,
                                     std::uint64_t seed, bool control_variate = true)
    -> Result<McResult>;
// Geometric-average Asian option CLOSED FORM (the CV control and a validation oracle).
[[nodiscard]] auto geometric_asian_price(const OptionSpec& spec, int steps) -> Result<double>;
// Longstaff-Schwartz least-squares MC for American exercise (polynomial basis {1,S,S^2}).
[[nodiscard]] auto longstaff_schwartz_american(const OptionSpec& spec, std::uint64_t paths,
                                               int steps, std::uint64_t seed) -> Result<McResult>;

// ===========================================================================
// Composable position / portfolio graph (fluent, reusable).
// ===========================================================================
// A signed quantity of an option (or, via a zero-vol/zero-strike trick, the underlying).
struct Position {
    OptionSpec spec{};
    double quantity{1.0};  // signed: long (+) / short (-)
};

// A portfolio is a composable bag of positions valued and risk-aggregated as a unit — the
// "computational graph" for a book of trades. Pricing/Greeks are the linear combination of
// the legs (bump-and-reprice would give cross-Greeks; the analytic legs suffice here).
class Portfolio {
public:
    [[nodiscard]] static auto create() -> Portfolio { return Portfolio{}; }
    [[nodiscard]] auto add(const OptionSpec& spec, double quantity) -> Portfolio& {
        legs_.push_back(Position{spec, quantity});
        return *this;
    }
    [[nodiscard]] auto with(const Position& p) -> Portfolio& { legs_.push_back(p); return *this; }
    [[nodiscard]] auto legs() const noexcept -> std::span<const Position> { return legs_; }

    // Aggregate Black-Scholes value and Greeks (sum of quantity-weighted legs).
    [[nodiscard]] auto value() const -> Result<double>;
    [[nodiscard]] auto greeks() const -> Result<Greeks>;
    // Aggregate payoff at expiry across all legs at underlying price s (for P&L diagrams).
    [[nodiscard]] auto payoff_at(double s) const -> double;

private:
    Portfolio() = default;
    std::vector<Position> legs_{};
};

// ===========================================================================
// Probability curves, payoff & price plotting (SVG, via nimblecas.svgplot).
// ===========================================================================
// Risk-neutral terminal density of S_T (lognormal) over [s_min, s_max].
[[nodiscard]] auto terminal_density_svg(const OptionSpec& spec, double s_min, double s_max,
                                        int samples, const PlotOptions& opt) -> Result<std::string>;
// Option payoff at expiry over [s_min, s_max] (optionally net of a premium for a P&L line).
[[nodiscard]] auto payoff_diagram_svg(const OptionSpec& spec, double s_min, double s_max,
                                      int samples, const PlotOptions& opt,
                                      double premium = 0.0) -> Result<std::string>;
// Portfolio P&L at expiry over [s_min, s_max], net of the portfolio's current BS value.
[[nodiscard]] auto portfolio_pnl_svg(const Portfolio& book, double s_min, double s_max,
                                     int samples, const PlotOptions& opt) -> Result<std::string>;
// Black-Scholes price as spot sweeps [s_min, s_max] (the price curve).
[[nodiscard]] auto price_vs_spot_svg(const OptionSpec& spec, double s_min, double s_max,
                                     int samples, const PlotOptions& opt) -> Result<std::string>;

}  // namespace nimblecas::pricing

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::pricing {
namespace {

constexpr double kSqrt2   = 1.4142135623730951;
constexpr double kInvSqrt2Pi = 0.3989422804014327;

// A single standard-normal draw as a pure function of a global index (partition-independent
// reproducibility). uniform in (0,1) from the counter core, mapped through the inverse CDF.
[[nodiscard]] auto normal_at(std::uint64_t key, std::uint64_t index) -> double;

// Small dense solve of the 3x3 normal equations (Longstaff-Schwartz regression). Returns
// nullopt on a singular system, so the caller falls back to "never exercise early".
[[nodiscard]] auto solve3(std::array<std::array<double, 3>, 3> a, std::array<double, 3> b)
    -> std::optional<std::array<double, 3>> {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) { piv = r; }
        }
        if (std::abs(a[piv][col]) < 1e-14) { return std::nullopt; }
        std::swap(a[col], a[piv]);
        std::swap(b[col], b[piv]);
        for (int r = 0; r < 3; ++r) {
            if (r == col) { continue; }
            const double factor = a[r][col] / a[col][col];
            for (int c = col; c < 3; ++c) { a[r][c] -= factor * a[col][c]; }
            b[r] -= factor * b[col];
        }
    }
    return std::array<double, 3>{b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]};
}

}  // namespace

auto norm_pdf(double x) -> double { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

auto norm_cdf(double x) -> double {
    // Correctly-rounded via the C library erfc (available through import std): more accurate
    // than a rational approximation, especially in the tails.
    return 0.5 * std::erfc(-x / kSqrt2);
}

auto inverse_norm_cdf(double p) -> Result<double> {
    if (!(p > 0.0 && p < 1.0)) { return make_error<double>(MathError::domain_error); }
    // Acklam's rational approximation, then one Halley refinement against erfc for ~machine
    // accuracy across the whole (0,1) range.
    static constexpr std::array<double, 6> a{
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr std::array<double, 5> b{
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01};
    static constexpr std::array<double, 6> c{
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static constexpr std::array<double, 4> d{
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};
    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;
    double x = 0.0;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // Halley step: e = Phi(x) - p, u = e / phi(x); x -= u/(1 + x*u/2).
    const double e = norm_cdf(x) - p;
    const double u = e / norm_pdf(x);
    x -= u / (1.0 + 0.5 * x * u);
    return x;
}

namespace {
// Acklam's rational inverse-normal WITHOUT the Halley refinement — ~1.15e-9 relative error,
// which is ~six orders of magnitude below Monte Carlo sampling error, so the erfc+exp of the
// Halley step (measured at ~12% of MC runtime by perf on oluwasanmi-tradingbot-server) is
// pure waste in the path engine. The public inverse_norm_cdf keeps the refinement for
// callers that need machine accuracy; only the hot MC path uses this.
[[nodiscard]] auto fast_inv_norm(double p) -> double {
    static constexpr std::array<double, 6> a{-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static constexpr std::array<double, 5> b{-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static constexpr std::array<double, 6> c{-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static constexpr std::array<double, 4> d{7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double plow = 0.02425;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p <= 1.0 - plow) {
        const double q = p - 0.5;
        const double r = q * q;
        // Explicit std::fma so the central branch is bit-identical to the AVX-512 batched
        // normal_fill regardless of -ffp-contract (Rule 55). Under the default contract=on build
        // this is exactly the value the contracted Horner would produce (verified: 0 mismatches).
        double num = a[0];
        num = std::fma(num, r, a[1]);
        num = std::fma(num, r, a[2]);
        num = std::fma(num, r, a[3]);
        num = std::fma(num, r, a[4]);
        num = std::fma(num, r, a[5]);
        num = num * q;
        double den = b[0];
        den = std::fma(den, r, b[1]);
        den = std::fma(den, r, b[2]);
        den = std::fma(den, r, b[3]);
        den = std::fma(den, r, b[4]);
        den = std::fma(den, r, 1.0);
        return num / den;
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

// Map a single raw counter draw to a standard normal. Shared by the scalar and batched paths
// so both produce identical values for the same underlying bits.
[[nodiscard]] auto bits_to_normal(std::uint64_t bits) -> double {
    const double u = uniform_unit(bits);
    // Clamp away from the exact endpoints so the inverse CDF stays finite (uniform_unit
    // already returns (0,1); this is belt-and-braces against a boundary bit pattern).
    const double uc = std::min(std::max(u, 1e-15), 1.0 - 1e-15);
    return fast_inv_norm(uc);
}

auto normal_at(std::uint64_t key, std::uint64_t index) -> double {
    return bits_to_normal(counter_u64(key, index));
}

// --- Batched bits -> standard-normal transform, dynamically dispatched (Rule 29) ---
// out[i] = bits_to_normal(bits[i]) for every i. The scalar path is the reference and portable
// fallback; the AVX-512 path vectorises Acklam's CENTRAL region (~95 % of draws: a branchless
// rational, evaluated with _mm512_fmadd to match the explicit-fma scalar central bit-for-bit)
// and fixes up the ~5 % tail lanes (which need log/sqrt) with the scalar path. Every ISA path is
// BIT-IDENTICAL to the per-index scalar bits_to_normal (verified in pricing_tests), so MC stays
// reproducible across hosts.
void normal_transform_scalar(const std::uint64_t* bits, double* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bits_to_normal(bits[i]);
    }
}
#if defined(__x86_64__) || defined(__i386__)
[[gnu::target("avx512f,avx512dq")]] void normal_transform_avx512(const std::uint64_t* bits,
                                                                 double* out,
                                                                 std::size_t n) noexcept {
    constexpr double kA[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double kB[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
    constexpr double kPlow = 0.02425;
    const __m512d scale = _mm512_set1_pd(1.0 / 9007199254740992.0);  // 1 / 2^53
    const __m512d lo = _mm512_set1_pd(1e-15);
    const __m512d hi = _mm512_set1_pd(1.0 - 1e-15);
    const __m512d half = _mm512_set1_pd(0.5);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i b = _mm512_loadu_si512(bits + i);
        __m512d u = _mm512_mul_pd(_mm512_cvtepu64_pd(_mm512_srli_epi64(b, 11)), scale);
        u = _mm512_min_pd(_mm512_max_pd(u, lo), hi);  // clamp to (0,1) endpoints
        const __m512d q = _mm512_sub_pd(u, half);
        const __m512d r = _mm512_mul_pd(q, q);
        __m512d num = _mm512_set1_pd(kA[0]);
        num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(kA[1]));
        num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(kA[2]));
        num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(kA[3]));
        num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(kA[4]));
        num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(kA[5]));
        num = _mm512_mul_pd(num, q);
        __m512d den = _mm512_set1_pd(kB[0]);
        den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(kB[1]));
        den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(kB[2]));
        den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(kB[3]));
        den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(kB[4]));
        den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(1.0));
        _mm512_storeu_pd(out + i, _mm512_div_pd(num, den));  // central value for all 8 lanes
        // Overwrite the rare tail lanes (u outside [kPlow, 1-kPlow]) with the scalar tail path.
        alignas(64) double ub[8];
        _mm512_store_pd(ub, u);
        for (int j = 0; j < 8; ++j) {
            if (ub[j] < kPlow || ub[j] > 1.0 - kPlow) {
                out[i + j] = fast_inv_norm(ub[j]);
            }
        }
    }
    for (; i < n; ++i) {  // ragged tail
        out[i] = bits_to_normal(bits[i]);
    }
}
#endif

using NormalTransformFn = void (*)(const std::uint64_t*, double*, std::size_t);
[[nodiscard]] auto resolve_normal_transform() noexcept -> NormalTransformFn {
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512dq") != 0) {
        return &normal_transform_avx512;
    }
#endif
    return &normal_transform_scalar;
}
const NormalTransformFn g_normal_transform = resolve_normal_transform();  // resolved once

// Batched standard-normal draws: out[j] = normal_at(key, base_index + j) for every j, sourcing
// the raw bits through counter_u64_batch (AVX-512 when available) and transforming them through
// the dispatched g_normal_transform. BIT-IDENTICAL to calling normal_at per index — this is the
// hot path that lets the Monte Carlo engines feed the vectorised RNG core instead of one draw at
// a time. `scratch` must be at least out.size(); it holds the raw u64 draws.
void normal_fill(std::uint64_t key, std::uint64_t base_index, std::span<double> out,
                 std::span<std::uint64_t> scratch) {
    const std::span<std::uint64_t> bits = scratch.first(out.size());
    counter_u64_batch(key, base_index, bits);
    g_normal_transform(bits.data(), out.data(), out.size());
}
}  // namespace

// --- Black-Scholes ----------------------------------------------------------

auto black_scholes_greeks(const OptionSpec& spec) -> Result<Greeks> {
    const double S = spec.spot;
    const double K = spec.strike;
    const double r = spec.rate;
    const double q = spec.dividend_yield;
    const double sig = spec.volatility;
    const double T = spec.time_to_expiry;
    if (S <= 0.0 || K <= 0.0 || T < 0.0 || sig < 0.0) {
        return make_error<Greeks>(MathError::domain_error);
    }
    const bool call = spec.type == OptionType::call;
    Greeks g{};
    if (T == 0.0 || sig == 0.0) {
        // Degenerate: value is the discounted intrinsic on the forward; Greeks are limits.
        const double fwd = S * std::exp((r - q) * T);
        const double intrinsic = call ? std::max(fwd - K, 0.0) : std::max(K - fwd, 0.0);
        g.price = std::exp(-r * T) * intrinsic;
        g.delta = call ? (fwd > K ? std::exp(-q * T) : 0.0) : (fwd < K ? -std::exp(-q * T) : 0.0);
        return g;
    }
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT);
    const double d2 = d1 - sig * sqrtT;
    const double disc_r = std::exp(-r * T);
    const double disc_q = std::exp(-q * T);
    const double Nd1 = norm_cdf(call ? d1 : -d1);
    const double Nd2 = norm_cdf(call ? d2 : -d2);
    const double pdf_d1 = norm_pdf(d1);
    if (call) {
        g.price = S * disc_q * Nd1 - K * disc_r * Nd2;
        g.delta = disc_q * norm_cdf(d1);
        g.rho = K * T * disc_r * norm_cdf(d2);
        g.theta = -S * disc_q * pdf_d1 * sig / (2.0 * sqrtT)
                  - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1);
    } else {
        g.price = K * disc_r * Nd2 - S * disc_q * Nd1;
        g.delta = -disc_q * norm_cdf(-d1);
        g.rho = -K * T * disc_r * norm_cdf(-d2);
        g.theta = -S * disc_q * pdf_d1 * sig / (2.0 * sqrtT)
                  + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1);
    }
    g.gamma = disc_q * pdf_d1 / (S * sig * sqrtT);
    g.vega = S * disc_q * pdf_d1 * sqrtT;
    return g;
}

auto black_scholes_price(const OptionSpec& spec) -> Result<double> {
    return black_scholes_greeks(spec).transform([](const Greeks& g) { return g.price; });
}

auto implied_volatility(const OptionSpec& spec, double market_price) -> Result<double> {
    if (market_price <= 0.0) { return make_error<double>(MathError::domain_error); }
    auto f = [&](double vol) -> double {
        auto p = black_scholes_price(spec.with_volatility(vol));
        return p ? (*p - market_price) : std::numeric_limits<double>::quiet_NaN();
    };
    // Bracket vol in (0, 5] (500% covers essentially all quoted markets).
    double lo = 1e-6;
    double hi = 5.0;
    double flo = f(lo);
    double fhi = f(hi);
    if (!std::isfinite(flo) || !std::isfinite(fhi) || (flo > 0.0) == (fhi > 0.0)) {
        return make_error<double>(MathError::not_converged);
    }
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double fm = f(mid);
        if (std::abs(fm) < 1e-10 || (hi - lo) < 1e-12) { return mid; }
        if ((fm > 0.0) == (flo > 0.0)) { lo = mid; flo = fm; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}

auto black_scholes_extended_greeks(const OptionSpec& spec) -> Result<ExtendedGreeks> {
    const double S = spec.spot;
    const double K = spec.strike;
    const double r = spec.rate;
    const double q = spec.dividend_yield;
    const double sig = spec.volatility;
    const double T = spec.time_to_expiry;
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sig <= 0.0) {
        return make_error<ExtendedGreeks>(MathError::domain_error);
    }
    const bool call = spec.type == OptionType::call;
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT);
    const double d2 = d1 - sig * sqrtT;
    const double phi = norm_pdf(d1);
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double gamma = disc_q * phi / (S * sig * sqrtT);
    const double vega = S * disc_q * phi * sqrtT;
    auto base = black_scholes_greeks(spec);
    if (!base) { return make_error<ExtendedGreeks>(base.error()); }

    ExtendedGreeks g{};
    g.vanna = -disc_q * phi * d2 / sig;
    g.vomma = vega * d1 * d2 / sig;
    g.speed = -gamma / S * (d1 / (sig * sqrtT) + 1.0);
    g.zomma = gamma * (d1 * d2 - 1.0) / sig;
    g.lambda = base->price != 0.0 ? base->delta * S / base->price : 0.0;
    g.dual_delta = call ? -disc_r * norm_cdf(d2) : disc_r * norm_cdf(-d2);
    g.dual_gamma = disc_r * norm_pdf(d2) / (K * sig * sqrtT);
    // epsilon (psi): d price / d dividend_yield, closed form.
    g.epsilon = call ? -S * T * disc_q * norm_cdf(d1) : S * T * disc_q * norm_cdf(-d1);
    // ultima = d vomma / d vol == d^3 price / d vol^3, closed form.
    g.ultima = -vega / (sig * sig) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2);
    // Time-derivatives (charm, color, veta) via central differences of the analytic delta /
    // gamma / vega w.r.t. time_to_expiry — robust and sign-consistent with our BS functions.
    const double h = 1e-4 * T;
    auto up = black_scholes_greeks(spec.with_expiry(T + h));
    auto dn = black_scholes_greeks(spec.with_expiry(T - h));
    if (up && dn) {
        // The struct documents these per unit of CALENDAR time (decay), which is -d/dT since
        // calendar time t = expiry - T. charm's doc carries the minus already (charm =
        // -d delta/d time == +d delta/dT); veta and color are d/d time, so negate the dT slope.
        g.charm = (up->delta - dn->delta) / (2.0 * h);      // +d delta/dT == -d delta/d time
        g.color = -(up->gamma - dn->gamma) / (2.0 * h);     // d gamma / d time
        g.veta  = -(up->vega - dn->vega) / (2.0 * h);       // d vega  / d time
    }
    // vera = d rho / d vol, via central difference of the analytic rho (avoids a fragile
    // hand-derived cross-partial; sign-consistent with our BS rho).
    const double hv = 1e-4 * sig;
    auto upv = black_scholes_greeks(spec.with_volatility(sig + hv));
    auto dnv = black_scholes_greeks(spec.with_volatility(sig - hv));
    if (upv && dnv) {
        g.vera = (upv->rho - dnv->rho) / (2.0 * hv);
    }
    return g;
}

auto option_pnl_at_expiry(const OptionSpec& spec, double premium, double s_T) -> double {
    return spec.payoff(s_T) - premium;
}

auto option_mark_to_market_pnl(const OptionSpec& spec, double premium) -> Result<double> {
    return black_scholes_price(spec).transform([&](double p) { return p - premium; });
}

auto black76_price(bool is_call, double forward, double strike, double rate, double volatility,
                   double time) -> Result<double> {
    if (forward <= 0.0 || strike <= 0.0 || time < 0.0 || volatility < 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double disc = std::exp(-rate * time);
    if (time == 0.0 || volatility == 0.0) {
        const double intrinsic = is_call ? std::max(forward - strike, 0.0)
                                         : std::max(strike - forward, 0.0);
        return disc * intrinsic;
    }
    const double sqrtT = std::sqrt(time);
    const double d1 = (std::log(forward / strike) + 0.5 * volatility * volatility * time) /
                      (volatility * sqrtT);
    const double d2 = d1 - volatility * sqrtT;
    if (is_call) {
        return disc * (forward * norm_cdf(d1) - strike * norm_cdf(d2));
    }
    return disc * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1));
}

auto digital_cash_or_nothing(const OptionSpec& spec, double cash) -> Result<double> {
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double sqrtT = std::sqrt(spec.time_to_expiry);
    const double d2 = (std::log(spec.spot / spec.strike) +
                       (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) *
                           spec.time_to_expiry) /
                      (spec.volatility * sqrtT);
    const double disc = std::exp(-spec.rate * spec.time_to_expiry);
    return cash * disc *
           (spec.type == OptionType::call ? norm_cdf(d2) : norm_cdf(-d2));
}

auto digital_asset_or_nothing(const OptionSpec& spec) -> Result<double> {
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double sqrtT = std::sqrt(spec.time_to_expiry);
    const double d1 = (std::log(spec.spot / spec.strike) +
                       (spec.rate - spec.dividend_yield + 0.5 * spec.volatility * spec.volatility) *
                           spec.time_to_expiry) /
                      (spec.volatility * sqrtT);
    const double disc_q = std::exp(-spec.dividend_yield * spec.time_to_expiry);
    return spec.spot * disc_q *
           (spec.type == OptionType::call ? norm_cdf(d1) : norm_cdf(-d1));
}

auto barrier_option_mc(const OptionSpec& spec, double barrier, bool knock_in, std::uint64_t paths,
                       int steps, std::uint64_t seed) -> Result<McResult> {
    // The inner work is paths*steps iterations (O(1) memory), so bound the PRODUCT, not just each
    // factor: two individually-in-range values (1e9 paths * 1e5 steps = 1e14) would still hang.
    constexpr std::uint64_t kMaxPathSteps = 1'000'000'000;
    constexpr int kMaxSteps = 100'000;  // caps the per-path draw buffer (O(steps)) independently
    if (paths == 0 || steps < 1 || steps > kMaxSteps || barrier <= 0.0 ||
        paths > kMaxPathSteps / static_cast<std::uint64_t>(steps)) {
        return make_error<McResult>(MathError::domain_error);
    }
    if (spec.spot <= 0.0 || spec.volatility < 0.0 || spec.time_to_expiry <= 0.0) {
        return make_error<McResult>(MathError::domain_error);
    }
    const bool down = barrier < spec.spot;  // barrier below spot -> down-barrier, else up
    const double dt = spec.time_to_expiry / static_cast<double>(steps);
    const double drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * dt;
    const double vol_sqrtdt = spec.volatility * std::sqrt(dt);
    const double disc = std::exp(-spec.rate * spec.time_to_expiry);
    const std::uint64_t key = splitmix64(seed);
    // Per-path scratch: the `steps` draws of a path occupy the contiguous counter range
    // [i*steps, i*steps+steps), so they batch through the vectorised RNG core in one call.
    const std::size_t nsteps = static_cast<std::size_t>(steps);
    std::vector<double> z(nsteps);
    std::vector<std::uint64_t> zbits(nsteps);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t i = 0; i < paths; ++i) {
        double s = spec.spot;
        bool hit = false;
        normal_fill(key, i * static_cast<std::uint64_t>(steps), z, zbits);
        for (int step = 0; step < steps; ++step) {
            s *= std::exp(drift + vol_sqrtdt * z[static_cast<std::size_t>(step)]);
            if ((down && s <= barrier) || (!down && s >= barrier)) { hit = true; }
        }
        const bool alive = knock_in ? hit : !hit;
        const double payoff = alive ? spec.payoff(s) * disc : 0.0;
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double var = std::max((sum_sq - n * mean * mean) / std::max(n - 1.0, 1.0), 0.0);
    return McResult{mean, std::sqrt(var / n), paths};
}

// --- Trinomial lattice ------------------------------------------------------

auto trinomial_price(const OptionSpec& spec, int steps, Exercise exercise,
                     std::span<const double> exercise_times) -> Result<double> {
    // steps drives a (2*steps+1)-wide lattice. Cap it so `2*steps + 1` cannot overflow int
    // (UB / negative width) and the two O(steps) buffers stay bounded (~a few hundred MB at the
    // ceiling). 100k steps is far past any convergence need.
    constexpr int kMaxSteps = 100'000;
    if (steps < 1 || steps > kMaxSteps) { return make_error<double>(MathError::domain_error); }
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double dt = spec.time_to_expiry / static_cast<double>(steps);
    const double sig = spec.volatility;
    const double nu = spec.rate - spec.dividend_yield - 0.5 * sig * sig;
    const double dx = sig * std::sqrt(3.0 * dt);          // Kamrad-Ritchken space step
    const double var = sig * sig * dt + nu * nu * dt * dt;
    const double pu = 0.5 * (var / (dx * dx) + nu * dt / dx);
    const double pm = 1.0 - var / (dx * dx);
    const double pd = 0.5 * (var / (dx * dx) - nu * dt / dx);
    if (pu < 0.0 || pm < 0.0 || pd < 0.0) {
        // Instability (too few steps for these parameters): report rather than mislead.
        return make_error<double>(MathError::not_converged);
    }
    const double disc = std::exp(-spec.rate * dt);

    // Which steps permit early exercise.
    std::vector<char> can_exercise(static_cast<std::size_t>(steps) + 1, 0);
    if (exercise == Exercise::american) {
        std::ranges::fill(can_exercise, static_cast<char>(1));
    } else if (exercise == Exercise::bermudan) {
        for (double t : exercise_times) {
            const int idx = static_cast<int>(std::lround(t / dt));
            if (idx >= 0 && idx <= steps) { can_exercise[static_cast<std::size_t>(idx)] = 1; }
        }
    }

    // Terminal layer: 2*steps + 1 nodes, index k in [0, 2*steps], log-offset (k - steps).
    const std::size_t width = static_cast<std::size_t>(2 * steps + 1);
    std::vector<double> cur(width);   // holds step i+1 during the sweep of step i
    std::vector<double> next(width);  // receives step i
    for (std::size_t k = 0; k < width; ++k) {
        const double s = spec.spot * std::exp((static_cast<double>(k) - steps) * dx);
        cur[k] = spec.payoff(s);
    }
    // Backward induction. At step i the live nodes span index [steps-i, steps+i]. We read the
    // three step-(i+1) children from `cur` and write step i into `next`, then swap — reading
    // and writing distinct buffers so a node's down-child is never overwritten before use.
    for (int i = steps - 1; i >= 0; --i) {
        const std::size_t lo = static_cast<std::size_t>(steps - i);
        const std::size_t hi = static_cast<std::size_t>(steps + i);
        const bool exercisable = can_exercise[static_cast<std::size_t>(i)] != 0;
        for (std::size_t k = lo; k <= hi; ++k) {
            const double cont = disc * (pu * cur[k + 1] + pm * cur[k] + pd * cur[k - 1]);
            if (exercisable) {
                const double s = spec.spot * std::exp((static_cast<double>(k) - steps) * dx);
                next[k] = std::max(cont, spec.payoff(s));
            } else {
                next[k] = cont;
            }
        }
        std::swap(cur, next);
    }
    return cur[static_cast<std::size_t>(steps)];
}

// --- Monte Carlo ------------------------------------------------------------

namespace {
// (sum, sum_sq) of antithetic European payoffs over the contiguous counter range
// [base, base+count). Shared by the serial and parallel pricers so both use the identical
// vectorised exp path. The two GBM exponentials per path are batched through simd::exp_into
// (deterministic AVX-512 exp, ~1 ulp) instead of scalar libm exp — a perf pass showed the MC
// hot loop was ~10 % scalar libm exp. `z`, `ep`, `em`, `zbits` are caller scratch (>= tile).
struct EuroPartial {
    double sum;
    double sum_sq;
};
[[nodiscard]] auto price_euro_block(const OptionSpec& spec, std::uint64_t key, double drift,
                                    double vol_sqrtT, double disc, std::uint64_t base,
                                    std::uint64_t count, std::span<double> z, std::span<double> ep,
                                    std::span<double> em, std::span<std::uint64_t> zbits)
    -> EuroPartial {
    const std::uint64_t tile = z.size();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t off = 0; off < count; off += tile) {
        const std::size_t m = static_cast<std::size_t>(std::min<std::uint64_t>(count - off, tile));
        normal_fill(key, base + off, z.first(m), zbits);
        for (std::size_t j = 0; j < m; ++j) {
            ep[j] = drift + vol_sqrtT * z[j];
            em[j] = drift - vol_sqrtT * z[j];
        }
        simd::exp_into(ep.first(m), ep.first(m));  // in-place (exp_into reads before it writes)
        simd::exp_into(em.first(m), em.first(m));
        for (std::size_t j = 0; j < m; ++j) {
            const double sp = spec.spot * ep[j];
            const double sm = spec.spot * em[j];
            const double payoff = 0.5 * (spec.payoff(sp) + spec.payoff(sm)) * disc;
            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }
    return {sum, sum_sq};
}
}  // namespace

auto monte_carlo_european(const OptionSpec& spec, std::uint64_t paths, std::uint64_t seed)
    -> Result<McResult> {
    // Cap paths to bound runtime (O(1) memory, but a ~1e18-iteration loop is an effective hang).
    constexpr std::uint64_t kMaxPaths = 1'000'000'000;
    if (paths == 0 || paths > kMaxPaths) { return make_error<McResult>(MathError::domain_error); }
    if (spec.time_to_expiry < 0.0 || spec.volatility < 0.0 || spec.spot <= 0.0) {
        return make_error<McResult>(MathError::domain_error);
    }
    const double T = spec.time_to_expiry;
    const double drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * T;
    const double vol_sqrtT = spec.volatility * std::sqrt(T);
    const double disc = std::exp(-spec.rate * T);
    const std::uint64_t key = splitmix64(seed);
    // Antithetic variates: each index i produces the pair (+z, -z), halving variance cheaply.
    // Draws are the contiguous counter range [0, paths); price_euro_block tiles them so the
    // vectorised RNG core + vectorised exp fill each tile at once while memory stays O(tile).
    constexpr std::size_t kTile = 8192;
    const std::size_t tile = static_cast<std::size_t>(std::min<std::uint64_t>(paths, kTile));
    std::vector<double> z(tile);
    std::vector<double> ep(tile);
    std::vector<double> em(tile);
    std::vector<std::uint64_t> zbits(tile);
    const auto part = price_euro_block(spec, key, drift, vol_sqrtT, disc, 0, paths, z, ep, em, zbits);
    const double sum = part.sum;
    const double sum_sq = part.sum_sq;
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double var = std::max((sum_sq - n * mean * mean) / std::max(n - 1.0, 1.0), 0.0);
    return McResult{mean, std::sqrt(var / n), paths};
}

auto monte_carlo_european_parallel(const OptionSpec& spec, std::uint64_t paths, std::uint64_t seed)
    -> Result<McResult> {
    // Same validation and model constants as the serial pricer.
    constexpr std::uint64_t kMaxPaths = 1'000'000'000;
    if (paths == 0 || paths > kMaxPaths) { return make_error<McResult>(MathError::domain_error); }
    if (spec.time_to_expiry < 0.0 || spec.volatility < 0.0 || spec.spot <= 0.0) {
        return make_error<McResult>(MathError::domain_error);
    }
    const double T = spec.time_to_expiry;
    const double drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * T;
    const double vol_sqrtT = spec.volatility * std::sqrt(T);
    const double disc = std::exp(-spec.rate * T);
    const std::uint64_t key = splitmix64(seed);

    // FIXED block decomposition -> the result depends only on (paths, seed), never on thread
    // count or scheduling. Each block prices its own contiguous counter sub-range with the exact
    // antithetic loop of the serial pricer (its own O(tile) scratch, so blocks never share
    // mutable state); transform_index returns the block partials IN ORDER, and they are combined
    // sequentially. This is numerically equal to the serial running sum (the gap is FP
    // non-associativity, orders of magnitude below the MC standard error) but not bit-identical.
    constexpr std::uint64_t kBlock = std::uint64_t{1} << 18;  // 262144 paths per block
    constexpr std::size_t kTile_ = 8192;                      // per-block RNG/exp fill tile
    const std::size_t n_blocks = static_cast<std::size_t>((paths + kBlock - 1) / kBlock);

    // grain 1: each fixed block is an independent task the backend load-balances across cores
    // (a block is 262144 paths of transcendental work — coarse enough that per-task overhead is
    // negligible). The grain affects only scheduling, never the result: transform_index is
    // order-preserving, so the combine below stays deterministic regardless of chunking. Each
    // block runs the SAME price_euro_block (vectorised exp) as the serial pricer.
    const auto partials = parallel::transform_index(n_blocks, [&](std::size_t b) -> EuroPartial {
        const std::uint64_t base = static_cast<std::uint64_t>(b) * kBlock;
        const std::uint64_t count = std::min<std::uint64_t>(paths - base, kBlock);
        const std::size_t tile = static_cast<std::size_t>(std::min<std::uint64_t>(count, kTile_));
        std::vector<double> z(tile);
        std::vector<double> ep(tile);
        std::vector<double> em(tile);
        std::vector<std::uint64_t> zbits(tile);
        return price_euro_block(spec, key, drift, vol_sqrtT, disc, base, count, z, ep, em, zbits);
    }, /*grain=*/std::size_t{1});

    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& p : partials) {  // in-order combine -> deterministic
        sum += p.sum;
        sum_sq += p.sum_sq;
    }
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double var = std::max((sum_sq - n * mean * mean) / std::max(n - 1.0, 1.0), 0.0);
    return McResult{mean, std::sqrt(var / n), paths};
}

auto geometric_asian_price(const OptionSpec& spec, int steps) -> Result<double> {
    if (steps < 1) { return make_error<double>(MathError::domain_error); }
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    // EXACT closed form for the DISCRETE geometric average G = (prod_{i=1..n} S_{t_i})^(1/n)
    // over equally-spaced dates t_i = i*dt. ln G is normal with the mean/variance below (the
    // variance uses sum_{i,j} min(i,j) = n(n+1)(2n+1)/6). Because this matches the MC's
    // discrete geometric average exactly, it is BOTH the control-variate expectation and a
    // validation oracle. With steps == 1 it reduces to Black-Scholes (G == S_T) — a checked
    // invariant in the tests.
    const double n = static_cast<double>(steps);
    const double T = spec.time_to_expiry;
    const double dt = T / n;
    const double sig = spec.volatility;
    const double drift = spec.rate - spec.dividend_yield - 0.5 * sig * sig;
    const double mu_g = std::log(spec.spot) + drift * dt * (n + 1.0) / 2.0;
    const double var_g = sig * sig * dt * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n);
    const double sg = std::sqrt(var_g);
    const double eg = std::exp(mu_g + 0.5 * var_g);          // E[G] under the risk-neutral law
    const double lnK = std::log(spec.strike);
    const double d1 = (mu_g + var_g - lnK) / sg;
    const double d2 = d1 - sg;
    const double disc = std::exp(-spec.rate * T);
    if (spec.type == OptionType::call) {
        return disc * (eg * norm_cdf(d1) - spec.strike * norm_cdf(d2));
    }
    return disc * (spec.strike * norm_cdf(-d2) - eg * norm_cdf(-d1));
}

auto monte_carlo_asian(const OptionSpec& spec, std::uint64_t paths, int steps, std::uint64_t seed,
                       bool control_variate) -> Result<McResult> {
    // Bound the PRODUCT paths*steps (the iteration count), not just each factor: 1e9 paths *
    // 1e5 steps = 1e14 iterations would hang even with both factors individually in range.
    constexpr std::uint64_t kMaxPathSteps = 1'000'000'000;
    constexpr int kMaxSteps = 100'000;  // caps the per-path draw buffer (O(steps)) independently
    if (paths == 0 || steps < 1 || steps > kMaxSteps ||
        paths > kMaxPathSteps / static_cast<std::uint64_t>(steps)) {
        return make_error<McResult>(MathError::domain_error);
    }
    if (spec.spot <= 0.0 || spec.volatility < 0.0 || spec.time_to_expiry <= 0.0) {
        return make_error<McResult>(MathError::domain_error);
    }
    const double dt = spec.time_to_expiry / static_cast<double>(steps);
    const double drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * dt;
    const double vol_sqrtdt = spec.volatility * std::sqrt(dt);
    const double disc = std::exp(-spec.rate * spec.time_to_expiry);
    const std::uint64_t key = splitmix64(seed);
    double geo_closed = 0.0;
    if (control_variate) {
        auto g = geometric_asian_price(spec, steps);
        if (!g) { return make_error<McResult>(g.error()); }
        geo_closed = *g;
    }
    const std::size_t nsteps = static_cast<std::size_t>(steps);
    std::vector<double> z(nsteps);
    std::vector<std::uint64_t> zbits(nsteps);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t i = 0; i < paths; ++i) {
        double s = spec.spot;
        double arith_sum = 0.0;
        double log_sum = 0.0;
        normal_fill(key, i * static_cast<std::uint64_t>(steps), z, zbits);
        for (int step = 0; step < steps; ++step) {
            s *= std::exp(drift + vol_sqrtdt * z[static_cast<std::size_t>(step)]);
            arith_sum += s;
            log_sum += std::log(s);
        }
        const double arith_avg = arith_sum / static_cast<double>(steps);
        const double geo_avg = std::exp(log_sum / static_cast<double>(steps));
        double payoff = spec.type == OptionType::call ? std::max(arith_avg - spec.strike, 0.0)
                                                      : std::max(spec.strike - arith_avg, 0.0);
        payoff *= disc;
        if (control_variate) {
            double geo_payoff = spec.type == OptionType::call ? std::max(geo_avg - spec.strike, 0.0)
                                                              : std::max(spec.strike - geo_avg, 0.0);
            geo_payoff *= disc;
            payoff += geo_closed - geo_payoff;  // CV with beta = 1 (geometric is highly correlated)
        }
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double var = std::max((sum_sq - n * mean * mean) / std::max(n - 1.0, 1.0), 0.0);
    return McResult{mean, std::sqrt(var / n), paths};
}

auto longstaff_schwartz_american(const OptionSpec& spec, std::uint64_t paths, int steps,
                                 std::uint64_t seed) -> Result<McResult> {
    if (paths < 4 || steps < 1) { return make_error<McResult>(MathError::domain_error); }
    // O(paths*(steps+1)) doubles are materialized as one grid; bound the product so it cannot
    // overflow size_t or ask for tens of GB (5e8 doubles ~ 4 GB at the ceiling).
    constexpr std::uint64_t kMaxCells = 500'000'000;
    if (steps > 100'000 ||
        paths > kMaxCells / (static_cast<std::uint64_t>(steps) + 1)) {
        return make_error<McResult>(MathError::domain_error);
    }
    if (spec.spot <= 0.0 || spec.volatility <= 0.0 || spec.time_to_expiry <= 0.0) {
        return make_error<McResult>(MathError::domain_error);
    }
    const std::size_t P = static_cast<std::size_t>(paths);
    const int N = steps;
    const double dt = spec.time_to_expiry / static_cast<double>(N);
    const double drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * dt;
    const double vol_sqrtdt = spec.volatility * std::sqrt(dt);
    const double disc = std::exp(-spec.rate * dt);
    const std::uint64_t key = splitmix64(seed);

    // Simulate and store the full price grid (paths x (N+1)). Memory is O(P*N). Each path's N
    // draws are the contiguous counter range [p*N, p*N+N), filled through the vectorised RNG core.
    std::vector<double> S(P * static_cast<std::size_t>(N + 1));
    const std::size_t Nn = static_cast<std::size_t>(N);
    std::vector<double> z(Nn);
    std::vector<std::uint64_t> zbits(Nn);
    for (std::size_t p = 0; p < P; ++p) {
        S[p * (N + 1)] = spec.spot;
        normal_fill(key, static_cast<std::uint64_t>(p) * static_cast<std::uint64_t>(N), z, zbits);
        for (int t = 0; t < N; ++t) {
            S[p * (N + 1) + static_cast<std::size_t>(t + 1)] =
                S[p * (N + 1) + static_cast<std::size_t>(t)] *
                std::exp(drift + vol_sqrtdt * z[static_cast<std::size_t>(t)]);
        }
    }
    // Cashflow at expiry, then roll backward regressing continuation on {1,S,S^2}.
    std::vector<double> cash(P);
    for (std::size_t p = 0; p < P; ++p) {
        cash[p] = spec.payoff(S[p * (N + 1) + static_cast<std::size_t>(N)]);
    }
    for (int t = N - 1; t >= 1; --t) {
        std::array<std::array<double, 3>, 3> ata{};
        std::array<double, 3> atb{};
        std::vector<std::size_t> itm;
        itm.reserve(P);
        for (std::size_t p = 0; p < P; ++p) {
            const double s = S[p * (N + 1) + static_cast<std::size_t>(t)];
            const double ex = spec.payoff(s);
            if (ex > 0.0) { itm.push_back(p); }
            cash[p] *= disc;  // discount every path's future cashflow one step
        }
        if (itm.size() >= 3) {
            for (std::size_t p : itm) {
                const double s = S[p * (N + 1) + static_cast<std::size_t>(t)];
                const std::array<double, 3> x{1.0, s, s * s};
                const double y = cash[p];  // already discounted to time t
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) { ata[a][b] += x[a] * x[b]; }
                    atb[a] += x[a] * y;
                }
            }
            if (auto beta = solve3(ata, atb)) {
                for (std::size_t p : itm) {
                    const double s = S[p * (N + 1) + static_cast<std::size_t>(t)];
                    const double cont = (*beta)[0] + (*beta)[1] * s + (*beta)[2] * s * s;
                    const double ex = spec.payoff(s);
                    if (ex > cont) { cash[p] = ex; }  // exercise now
                }
            }
        }
    }
    // Discount the time-1 cashflows back to time 0.
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::size_t p = 0; p < P; ++p) {
        const double v = cash[p] * disc;
        sum += v;
        sum_sq += v * v;
    }
    const double n = static_cast<double>(P);
    const double mean = sum / n;
    const double var = std::max((sum_sq - n * mean * mean) / std::max(n - 1.0, 1.0), 0.0);
    // American value is at least the immediate exercise value at t0.
    const double lower = spec.payoff(spec.spot);
    return McResult{std::max(mean, lower), std::sqrt(var / n), paths};
}

// --- Portfolio --------------------------------------------------------------

auto Portfolio::value() const -> Result<double> {
    double total = 0.0;
    for (const auto& leg : legs_) {
        auto p = black_scholes_price(leg.spec);
        if (!p) { return p; }
        total += leg.quantity * *p;
    }
    return total;
}

auto Portfolio::greeks() const -> Result<Greeks> {
    Greeks agg{};
    for (const auto& leg : legs_) {
        auto g = black_scholes_greeks(leg.spec);
        if (!g) { return g; }
        agg.price += leg.quantity * g->price;
        agg.delta += leg.quantity * g->delta;
        agg.gamma += leg.quantity * g->gamma;
        agg.vega  += leg.quantity * g->vega;
        agg.theta += leg.quantity * g->theta;
        agg.rho   += leg.quantity * g->rho;
    }
    return agg;
}

auto Portfolio::payoff_at(double s) const -> double {
    double total = 0.0;
    for (const auto& leg : legs_) { total += leg.quantity * leg.spec.payoff(s); }
    return total;
}

// --- Plotting ---------------------------------------------------------------

auto terminal_density_svg(const OptionSpec& spec, double s_min, double s_max, int samples,
                          const PlotOptions& opt) -> Result<std::string> {
    if (spec.spot <= 0.0 || spec.volatility <= 0.0 || spec.time_to_expiry <= 0.0) {
        return make_error<std::string>(MathError::domain_error);
    }
    const double T = spec.time_to_expiry;
    const double mu = std::log(spec.spot) +
                      (spec.rate - spec.dividend_yield - 0.5 * spec.volatility * spec.volatility) * T;
    const double sd = spec.volatility * std::sqrt(T);
    auto density = [=](double s) -> double {
        if (s <= 0.0) { return 0.0; }
        const double z = (std::log(s) - mu) / sd;
        return norm_pdf(z) / (s * sd);
    };
    return plot_function(density, s_min, s_max, samples, opt);
}

auto payoff_diagram_svg(const OptionSpec& spec, double s_min, double s_max, int samples,
                        const PlotOptions& opt, double premium) -> Result<std::string> {
    auto f = [=](double s) -> double { return spec.payoff(s) - premium; };
    return plot_function(f, s_min, s_max, samples, opt);
}

auto portfolio_pnl_svg(const Portfolio& book, double s_min, double s_max, int samples,
                       const PlotOptions& opt) -> Result<std::string> {
    auto cost = book.value();
    if (!cost) { return make_error<std::string>(cost.error()); }
    const double c = *cost;
    auto f = [&book, c](double s) -> double { return book.payoff_at(s) - c; };
    return plot_function(f, s_min, s_max, samples, opt);
}

auto price_vs_spot_svg(const OptionSpec& spec, double s_min, double s_max, int samples,
                       const PlotOptions& opt) -> Result<std::string> {
    auto f = [spec](double s) -> double {
        auto p = black_scholes_price(spec.with_spot(s));
        return p ? *p : std::numeric_limits<double>::quiet_NaN();
    };
    return plot_function(f, s_min, s_max, samples, opt);
}

}  // namespace nimblecas::pricing
