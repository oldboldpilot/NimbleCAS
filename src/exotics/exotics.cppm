// NimbleCAS exotic-derivatives pricing — closed forms, lattices, PDE & Monte Carlo.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. The exotic-option companion to nimblecas.pricing. Everything here reuses that
// module's OptionSpec / OptionType / Exercise / McResult and its correctly-rounded
// standard-normal helpers (norm_cdf / norm_pdf / inverse_norm_cdf), and the parallelisable
// counter-based RNG (nimblecas.rng) for the Monte Carlo pieces, rather than reinventing
// them. The tier is NUMERICAL / STATISTICAL:
//   * Cox-Ross-Rubinstein BINOMIAL lattice (European / American), plus an escrowed-spot
//     variant that handles discrete cash dividends via a dividend schedule;
//   * Reiner-Rubinstein CLOSED FORMS for all eight continuously-monitored single-barrier
//     options (up/down x in/out x call/put) — the analytic counterpart to the discrete
//     barrier Monte Carlo in nimblecas.pricing;
//   * the Goldman-Sosin-Gatto floating-strike LOOKBACK closed form (call & put);
//   * a Crank-Nicolson FINITE-DIFFERENCE PDE pricer with Rannacher startup (a few
//     fully-implicit steps that damp the payoff-kink oscillation) and American exercise
//     via projected SOR (PSOR);
//   * MARGRABE exchange, KIRK spread approximation, GESKE compound (with a Drezner-style
//     bivariate-normal CDF helper), a simple CHOOSER, and a correlated BASKET Monte Carlo
//     driven by a Cholesky factor of the correlation matrix.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). Nothing here claims exactness — an
// option price is a limit (of a lattice as steps -> infinity, a PDE grid as h -> 0, a
// sample mean as paths -> infinity), and every quantity rides on transcendental
// exp/log/erf. So:
//   * closed forms (barrier, lookback, Margrabe, Kirk, Geske, chooser) are
//     correctly-rounded double formulas returning Result<double>;
//   * lattice / PDE prices carry discretization error that shrinks with steps / grid size;
//   * Monte Carlo returns the estimate AND its standard error (pricing::McResult), and is
//     bit-reproducible and partition-independent (every draw is a pure function of its
//     global index via nimblecas.rng's counter_u64).
// A NaN or infinity is NEVER returned as a value. Degenerate inputs (S<=0, vol<=0, T<=0, a
// barrier on the wrong side of spot, a non-positive-definite correlation) return
// domain_error; a non-convergent iteration (PSOR, a critical-price root) returns
// not_converged. Every grid / step / path size is bounded against a DoS blow-up.

export module nimblecas.exotics;

import std;
import nimblecas.core;
import nimblecas.pricing;
import nimblecas.rng;

export namespace nimblecas::exotics {

using pricing::OptionSpec;
using pricing::OptionType;
using pricing::Exercise;
using pricing::McResult;

// ---------------------------------------------------------------------------
// 1. Cox-Ross-Rubinstein binomial lattice.
// ---------------------------------------------------------------------------

// CRR binomial price with `steps` time steps. u = e^{sigma sqrt(dt)}, d = 1/u, and the
// risk-neutral up-probability p = (e^{(r-q)dt} - d)/(u - d). American exercise takes
// max(continuation, intrinsic) at every node. Converges to Black-Scholes as steps grow
// (O(1/steps), with the classic even/odd oscillation). `steps` is capped: > 100000 (or
// < 1) -> domain_error; parameters with p outside [0,1] (far too few steps) ->
// not_converged rather than a silently-negative "probability".
[[nodiscard]] auto crr_binomial(const OptionSpec& spec, int steps, Exercise exercise)
    -> Result<double>;

// A discrete cash dividend: `amount` paid at time `time` (years from valuation).
struct CashDividend {
    double time{0.0};
    double amount{0.0};
};

// CRR binomial with DISCRETE cash dividends via the escrowed-spot approximation: the
// present value of every dividend paid on (0, T] is subtracted from the spot, and the
// lattice is grown from that reduced spot with dividend_yield forced to zero. This is the
// standard practitioner approximation (exact discrete-dividend trees need a
// non-recombining lattice); the honesty boundary is that it is an APPROXIMATION, quoted as
// such. If the escrowed spot goes non-positive (dividends exceed the spot) -> domain_error;
// a negative dividend time -> domain_error. `steps` is capped as in crr_binomial.
[[nodiscard]] auto crr_binomial_discrete_div(const OptionSpec& spec, int steps,
                                             Exercise exercise,
                                             std::span<const CashDividend> dividends)
    -> Result<double>;

// ---------------------------------------------------------------------------
// 2. Reiner-Rubinstein single-barrier closed forms (continuous monitoring).
// ---------------------------------------------------------------------------

enum class Barrier : std::uint8_t { down, up };

// Analytic price of a single-barrier option under continuous monitoring (Reiner-Rubinstein
// / Merton). `side` selects down- vs up-barrier, `knock_in` selects in vs out, and the
// call/put flavour is taken from spec.type — together the eight standard types. `rebate` is
// an optional cash amount paid (out: at the knock time; in: at expiry if never knocked in);
// default 0. Cost of carry b = r - q. The in/out parity knock_in + knock_out == vanilla
// holds exactly for rebate 0. Preconditions: barrier > 0 and spot strictly on the LIVE side
// (spot > barrier for a down-barrier, spot < barrier for an up-barrier) — otherwise the
// contract is already knocked in/out and the continuous formula does not apply -> the
// wrong-side case returns domain_error. Degenerate spec (S/K/vol/T <= 0) -> domain_error.
[[nodiscard]] auto barrier_analytic(const OptionSpec& spec, double barrier, Barrier side,
                                    bool knock_in, double rebate = 0.0) -> Result<double>;

// ---------------------------------------------------------------------------
// 3. Goldman-Sosin-Gatto floating-strike lookback closed form.
// ---------------------------------------------------------------------------

// Floating-strike lookback: a call pays S_T - min over the life, a put pays max - S_T.
// `running_extremum` is the extremum observed SO FAR (running minimum for a call, running
// maximum for a put); pass 0 to start a fresh lookback at the current spot. Closed form of
// Goldman, Sosin & Gatto (1979); cost of carry b = r - q. The formula has a removable
// singularity at b == 0 (the sig^2/(2b) term); this implementation requires |b| >= 1e-8 and
// returns domain_error at b == 0 — the b -> 0 limit is // deferred:. Degenerate spec, a
// running minimum above spot (call), or a running maximum below spot (put) -> domain_error.
[[nodiscard]] auto lookback_price(const OptionSpec& spec, double running_extremum = 0.0)
    -> Result<double>;

// ---------------------------------------------------------------------------
// 4. Crank-Nicolson finite-difference PDE pricer (Rannacher startup; PSOR for American).
// ---------------------------------------------------------------------------

// Prices a vanilla European or American option by solving the Black-Scholes PDE on a
// uniform spot grid [0, S_max] with `n_space` intervals and `n_time` time steps.
// Crank-Nicolson in time with RANNACHER startup: the first two steps are fully implicit
// (backward Euler) to damp the oscillation the non-smooth payoff otherwise induces in pure
// CN. American exercise is enforced with projected SOR (PSOR) against the intrinsic value.
// The price at the requested spot is linearly interpolated between grid nodes. Grid sizes
// are bounded: n_space in [4, 20000], n_time in [1, 20000], and n_space*n_time <= 2e8.
// Degenerate spec -> domain_error; PSOR that fails to converge -> not_converged.
[[nodiscard]] auto fd_pde_price(const OptionSpec& spec, int n_space, int n_time,
                                Exercise exercise) -> Result<double>;

// ---------------------------------------------------------------------------
// 5. Multi-asset closed forms + basket Monte Carlo.
// ---------------------------------------------------------------------------

// Margrabe (1978): option to exchange asset 2 for asset 1, payoff max(S1 - S2, 0). The
// risk-free rate cancels; only the dividend yields q1, q2 and the effective volatility
// sqrt(sig1^2 + sig2^2 - 2 rho sig1 sig2) enter. With one asset's volatility set to zero
// (and its yield equal to r) this reduces to a Black-Scholes vanilla — a checked invariant.
// Preconditions: S1,S2 > 0, sig1,sig2 >= 0, |rho| <= 1, T > 0.
[[nodiscard]] auto margrabe_exchange(double spot1, double spot2, double div1, double div2,
                                     double vol1, double vol2, double rho, double time)
    -> Result<double>;

// Kirk (1995) approximation for a spread option, payoff max(S1 - S2 - K, 0). Treats
// (S2-forward + K) as a single lognormal and applies a Margrabe-style formula; exact when
// K == 0 (it reduces to Margrabe) and accurate for modest strikes. Preconditions:
// S1,S2 > 0, S2-forward + K > 0, sig1,sig2 >= 0, |rho| <= 1, T > 0.
[[nodiscard]] auto kirk_spread(double spot1, double spot2, double strike, double rate,
                               double div1, double div2, double vol1, double vol2,
                               double rho, double time) -> Result<double>;

// Standardised bivariate-normal CDF: P(X <= a, Y <= b) for a standard bivariate normal
// with correlation rho. Drezner reduction M(a,b;rho) = Phi(a)Phi(b) + integral_0^rho of the
// bivariate density in the correlation parameter, evaluated by composite Simpson. Exact
// special cases: rho == 0 factorises to Phi(a)Phi(b); rho == 1 gives Phi(min(a,b));
// rho == -1 gives max(Phi(a) + Phi(b) - 1, 0). Pure (no error channel); the near-|rho|=1
// integrand is mildly ill-conditioned but the endpoints are handled exactly.
[[nodiscard]] auto bivariate_normal_cdf(double a, double b, double rho) -> double;

// Geske (1979) compound option: a call-on-call — the right, at t1 for premium `strike1`, to
// buy a Black-Scholes call struck at `strike2` expiring at spec.time_to_expiry (> t1) on the
// underlying described by spec. Uses the bivariate-normal CDF and a bracketed root for the
// critical spot at t1. As strike1 -> 0 the compound value collapses to the underlying call
// (a checked invariant). Preconditions: 0 < t1 < spec.time_to_expiry, strike1 >= 0,
// strike2 > 0, spec spot/vol > 0. A non-convergent critical-price root -> not_converged.
[[nodiscard]] auto geske_compound(const OptionSpec& spec, double strike1, double strike2,
                                  double t1) -> Result<double>;

// Simple chooser (Rubinstein 1991): at t1 the holder chooses call or put, both struck at
// spec.strike and expiring at spec.time_to_expiry. Value lies between max(call, put) and
// call + put (a straddle). Preconditions: 0 < t1 <= spec.time_to_expiry, spec spot/K/vol > 0.
[[nodiscard]] auto chooser_price(const OptionSpec& spec, double t1) -> Result<double>;

// One leg of a basket: its spot, volatility, continuous dividend yield and basket weight.
struct BasketAsset {
    double spot{100.0};
    double volatility{0.2};
    double dividend_yield{0.0};
    double weight{1.0};
};

// Correlated basket option by reproducible Monte Carlo. Payoff max(sum_i w_i S_i(T) - K, 0)
// for a call (or K - basket for a put). `correlation` is the row-major n*n correlation
// matrix; its Cholesky factor drives the correlated terminal draws. Antithetic variates
// halve the variance. Reuses nimblecas.rng: every normal is a pure function of its global
// index, so the estimate is bit-reproducible under a fixed seed and partition-independent.
// Returns the estimate and its standard error. Bounds: 1 <= n <= 256, paths in
// [1, 1e9] with n*paths <= 1e9. A non-positive-definite correlation, a size mismatch, or a
// degenerate leg -> domain_error.
[[nodiscard]] auto basket_mc(std::span<const BasketAsset> assets,
                             std::span<const double> correlation, double strike, double rate,
                             double time, OptionType type, std::uint64_t paths,
                             std::uint64_t seed) -> Result<McResult>;

}  // namespace nimblecas::exotics

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::exotics {
namespace {

using pricing::norm_cdf;

constexpr int kMaxLatticeSteps = 100'000;
constexpr int kMaxGrid = 20'000;
constexpr std::uint64_t kMaxGridCells = 200'000'000;
constexpr std::size_t kMaxBasketAssets = 256;
constexpr std::uint64_t kMaxPaths = 1'000'000'000;

// A single standard-normal draw as a pure function of a global index (partition-independent
// reproducibility), mirroring the counter-based construction in nimblecas.pricing. The
// public inverse_norm_cdf is used with the uniform clamped strictly inside (0,1) so it is
// always engaged; the fallback 0.0 is unreachable and only satisfies the type.
[[nodiscard]] auto normal_at(std::uint64_t key, std::uint64_t index) -> double {
    const double u = uniform_unit(counter_u64(key, index));
    const double uc = std::min(std::max(u, 1e-15), 1.0 - 1e-15);
    auto z = pricing::inverse_norm_cdf(uc);
    return z ? *z : 0.0;
}

// Black-Scholes call value as a bare number (used by the Geske critical-price root and the
// Margrabe/chooser reductions). Returns nullopt on a degenerate sub-problem.
[[nodiscard]] auto bs_call(double spot, double strike, double rate, double div, double vol,
                           double time) -> std::optional<double> {
    auto p = pricing::black_scholes_price(
        OptionSpec{}.with_spot(spot).with_strike(strike).with_rate(rate).with_dividend(div)
            .with_volatility(vol).with_expiry(time).with_type(OptionType::call));
    if (!p) { return std::nullopt; }
    return *p;
}

// Thomas algorithm: solve a tridiagonal system for interior indices [1, n] (0 and n+1 are
// Dirichlet boundaries already folded into rhs). lower/diag/upper/rhs are indexed by node.
// Writes the solution into x[1..n]. Returns false on a zero pivot (should not occur for the
// diagonally-dominant BS operator, but guarded for honesty).
[[nodiscard]] auto thomas(std::span<const double> lower, std::span<const double> diag,
                          std::span<const double> upper, std::vector<double>& rhs,
                          std::vector<double>& x, std::size_t n) -> bool {
    std::vector<double> c_star(n + 2, 0.0);
    std::vector<double> d_star(n + 2, 0.0);
    if (std::abs(diag[1]) < 1e-300) { return false; }
    c_star[1] = upper[1] / diag[1];
    d_star[1] = rhs[1] / diag[1];
    for (std::size_t j = 2; j <= n; ++j) {
        const double m = diag[j] - lower[j] * c_star[j - 1];
        if (std::abs(m) < 1e-300) { return false; }
        c_star[j] = upper[j] / m;
        d_star[j] = (rhs[j] - lower[j] * d_star[j - 1]) / m;
    }
    x[n] = d_star[n];
    for (std::size_t j = n - 1; j >= 1; --j) {
        x[j] = d_star[j] - c_star[j] * x[j + 1];
        if (j == 1) { break; }
    }
    return true;
}

}  // namespace

auto crr_binomial(const OptionSpec& spec, int steps, Exercise exercise) -> Result<double> {
    if (steps < 1 || steps > kMaxLatticeSteps) {
        return make_error<double>(MathError::domain_error);
    }
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double dt = spec.time_to_expiry / static_cast<double>(steps);
    const double u = std::exp(spec.volatility * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((spec.rate - spec.dividend_yield) * dt);
    const double p = (growth - d) / (u - d);
    if (!(p >= 0.0 && p <= 1.0)) {
        // Too few steps for these parameters: a risk-neutral probability outside [0,1] would
        // otherwise produce a meaningless price. Report rather than mislead (Rule 32).
        return make_error<double>(MathError::not_converged);
    }
    const double disc = std::exp(-spec.rate * dt);
    const bool american = exercise == Exercise::american;

    const std::size_t width = static_cast<std::size_t>(steps) + 1;
    std::vector<double> v(width);
    // Terminal layer: node i has i up-moves and (steps - i) down-moves.
    for (std::size_t i = 0; i < width; ++i) {
        const double s = spec.spot * std::pow(u, static_cast<double>(i)) *
                         std::pow(d, static_cast<double>(steps) - static_cast<double>(i));
        v[i] = spec.payoff(s);
    }
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            const double cont =
                disc * (p * v[static_cast<std::size_t>(i) + 1] + (1.0 - p) * v[static_cast<std::size_t>(i)]);
            if (american) {
                const double s = spec.spot * std::pow(u, static_cast<double>(i)) *
                                 std::pow(d, static_cast<double>(step) - static_cast<double>(i));
                v[static_cast<std::size_t>(i)] = std::max(cont, spec.payoff(s));
            } else {
                v[static_cast<std::size_t>(i)] = cont;
            }
        }
    }
    return v[0];
}

auto crr_binomial_discrete_div(const OptionSpec& spec, int steps, Exercise exercise,
                               std::span<const CashDividend> dividends) -> Result<double> {
    double pv_div = 0.0;
    for (const auto& div : dividends) {
        if (div.time < 0.0) { return make_error<double>(MathError::domain_error); }
        if (div.time > 0.0 && div.time <= spec.time_to_expiry) {
            pv_div += div.amount * std::exp(-spec.rate * div.time);
        }
    }
    const double escrowed_spot = spec.spot - pv_div;
    if (escrowed_spot <= 0.0) {
        // Dividends escrow more than the whole spot: the approximation breaks down.
        return make_error<double>(MathError::domain_error);
    }
    // Grow the lattice from the reduced spot with the continuous yield removed (the discrete
    // cash dividends are already captured in the escrowed spot).
    return crr_binomial(spec.with_spot(escrowed_spot).with_dividend(0.0), steps, exercise);
}

auto barrier_analytic(const OptionSpec& spec, double barrier, Barrier side, bool knock_in,
                      double rebate) -> Result<double> {
    const double S = spec.spot;
    const double K = spec.strike;
    const double H = barrier;
    const double r = spec.rate;
    const double sig = spec.volatility;
    const double T = spec.time_to_expiry;
    if (S <= 0.0 || K <= 0.0 || H <= 0.0 || sig <= 0.0 || T <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const bool down = side == Barrier::down;
    // Spot must sit on the live side of the barrier for the continuous formula to apply.
    if ((down && S <= H) || (!down && S >= H)) {
        return make_error<double>(MathError::domain_error);
    }
    const double b = r - spec.dividend_yield;                 // cost of carry
    const bool call = spec.type == OptionType::call;
    const double phi = call ? 1.0 : -1.0;
    const double eta = down ? 1.0 : -1.0;

    const double sqrtT = std::sqrt(T);
    const double sst = sig * sqrtT;
    const double mu = (b - 0.5 * sig * sig) / (sig * sig);
    const double lam = std::sqrt(std::max(mu * mu + 2.0 * r / (sig * sig), 0.0));
    const double hs = H / S;

    const double x1 = std::log(S / K) / sst + (1.0 + mu) * sst;
    const double x2 = std::log(S / H) / sst + (1.0 + mu) * sst;
    const double y1 = std::log(H * H / (S * K)) / sst + (1.0 + mu) * sst;
    const double y2 = std::log(H / S) / sst + (1.0 + mu) * sst;
    const double z = std::log(H / S) / sst + lam * sst;

    const double dq = std::exp((b - r) * T);
    const double dr = std::exp(-r * T);
    const double p2mu = std::pow(hs, 2.0 * mu);
    const double p2mu2 = std::pow(hs, 2.0 * (mu + 1.0));

    const double A = phi * S * dq * norm_cdf(phi * x1) -
                     phi * K * dr * norm_cdf(phi * x1 - phi * sst);
    const double B = phi * S * dq * norm_cdf(phi * x2) -
                     phi * K * dr * norm_cdf(phi * x2 - phi * sst);
    const double C = phi * S * dq * p2mu2 * norm_cdf(eta * y1) -
                     phi * K * dr * p2mu * norm_cdf(eta * y1 - eta * sst);
    const double D = phi * S * dq * p2mu2 * norm_cdf(eta * y2) -
                     phi * K * dr * p2mu * norm_cdf(eta * y2 - eta * sst);
    const double E = rebate * dr *
                     (norm_cdf(eta * x2 - eta * sst) - p2mu * norm_cdf(eta * y2 - eta * sst));
    const double F = rebate * (std::pow(hs, mu + lam) * norm_cdf(eta * z) +
                               std::pow(hs, mu - lam) * norm_cdf(eta * z - 2.0 * eta * lam * sst));

    const bool kge = K >= H;  // strike at or above the barrier
    double value = 0.0;
    if (call && down && knock_in) {          // down-and-in call
        value = kge ? (C + E) : (A - B + D + E);
    } else if (call && !down && knock_in) {  // up-and-in call
        value = kge ? (A + E) : (B - C + D + E);
    } else if (call && down && !knock_in) {  // down-and-out call
        value = kge ? (A - C + F) : (B - D + F);
    } else if (call && !down && !knock_in) { // up-and-out call
        value = kge ? F : (A - B + C - D + F);
    } else if (!call && down && knock_in) {  // down-and-in put
        value = kge ? (B - C + D + E) : (A + E);
    } else if (!call && !down && knock_in) { // up-and-in put
        value = kge ? (A - B + D + E) : (C + E);
    } else if (!call && down && !knock_in) { // down-and-out put
        value = kge ? (A - B + C - D + F) : F;
    } else {                                 // up-and-out put
        value = kge ? (B - D + F) : (A - C + F);
    }
    // A barrier value is a non-negative discounted expectation; tiny negative round-off near
    // the knock boundary is clamped to zero (never a negative price).
    return std::max(value, 0.0);
}

auto lookback_price(const OptionSpec& spec, double running_extremum) -> Result<double> {
    const double S = spec.spot;
    const double r = spec.rate;
    const double sig = spec.volatility;
    const double T = spec.time_to_expiry;
    if (S <= 0.0 || sig <= 0.0 || T <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double b = r - spec.dividend_yield;
    if (std::abs(b) < 1e-8) {
        // Removable singularity of the sig^2/(2b) term; the b -> 0 limit is // deferred:.
        return make_error<double>(MathError::domain_error);
    }
    const bool call = spec.type == OptionType::call;
    double m = running_extremum;
    if (m <= 0.0) { m = S; }  // fresh lookback: extremum starts at the current spot
    if (call && m > S) { return make_error<double>(MathError::domain_error); }
    if (!call && m < S) { return make_error<double>(MathError::domain_error); }

    const double sqrtT = std::sqrt(T);
    const double sst = sig * sqrtT;
    const double dq = std::exp((b - r) * T);
    const double dr = std::exp(-r * T);
    const double coeff = sig * sig / (2.0 * b);
    const double power = -2.0 * b / (sig * sig);

    if (call) {
        const double a1 = (std::log(S / m) + (b + 0.5 * sig * sig) * T) / sst;
        const double a2 = a1 - sst;
        const double value = S * dq * norm_cdf(a1) - m * dr * norm_cdf(a2) +
                             S * dr * coeff *
                                 (std::pow(S / m, power) * norm_cdf(-a1 + 2.0 * b * sqrtT / sig) -
                                  std::exp(b * T) * norm_cdf(-a1));
        return std::max(value, 0.0);
    }
    const double b1 = (std::log(S / m) + (b + 0.5 * sig * sig) * T) / sst;
    const double b2 = b1 - sst;
    const double value = m * dr * norm_cdf(-b2) - S * dq * norm_cdf(-b1) +
                         S * dr * coeff *
                             (-std::pow(S / m, power) * norm_cdf(b1 - 2.0 * b * sqrtT / sig) +
                              std::exp(b * T) * norm_cdf(b1));
    return std::max(value, 0.0);
}

auto fd_pde_price(const OptionSpec& spec, int n_space, int n_time, Exercise exercise)
    -> Result<double> {
    if (n_space < 4 || n_space > kMaxGrid || n_time < 1 || n_time > kMaxGrid) {
        return make_error<double>(MathError::domain_error);
    }
    if (static_cast<std::uint64_t>(n_space) * static_cast<std::uint64_t>(n_time) > kMaxGridCells) {
        return make_error<double>(MathError::domain_error);
    }
    if (spec.spot <= 0.0 || spec.strike <= 0.0 || spec.time_to_expiry <= 0.0 ||
        spec.volatility <= 0.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double r = spec.rate;
    const double q = spec.dividend_yield;
    const double sig = spec.volatility;
    const double T = spec.time_to_expiry;
    const bool call = spec.type == OptionType::call;
    const bool american = exercise == Exercise::american;

    const std::size_t M = static_cast<std::size_t>(n_space);
    const int N = n_time;
    // Domain wide enough that the far Dirichlet boundary contributes negligibly at spot.
    const double base = std::max(spec.spot, spec.strike);
    const double s_max = base * std::exp(5.0 * sig * std::sqrt(T));
    const double dS = s_max / static_cast<double>(M);
    const double dt = T / static_cast<double>(N);

    std::vector<double> S(M + 1);
    std::vector<double> V(M + 1);
    std::vector<double> obstacle(M + 1);
    for (std::size_t j = 0; j <= M; ++j) {
        S[j] = static_cast<double>(j) * dS;
        V[j] = spec.payoff(S[j]);   // terminal condition
        obstacle[j] = V[j];         // intrinsic value (American early-exercise floor)
    }

    const std::size_t n = M - 1;  // number of interior unknowns (indices 1..M-1)
    std::vector<double> lower(M + 1, 0.0);
    std::vector<double> diag(M + 1, 0.0);
    std::vector<double> upper(M + 1, 0.0);
    std::vector<double> rhs(M + 1, 0.0);
    std::vector<double> x(M + 1, 0.0);

    const int rannacher = std::min(2, N);  // fully-implicit startup steps

    for (int step = 1; step <= N; ++step) {
        const double tau = static_cast<double>(step) * dt;  // time to maturity at new level
        const bool implicit = step <= rannacher;
        const double theta = implicit ? 1.0 : 0.5;  // implicitness weight

        // New Dirichlet boundaries at S=0 and S=S_max for this time level.
        double b0 = 0.0;
        double bM = 0.0;
        if (call) {
            b0 = 0.0;
            bM = american ? std::max(s_max - spec.strike, 0.0)
                          : std::max(s_max * std::exp(-q * tau) - spec.strike * std::exp(-r * tau), 0.0);
        } else {
            b0 = american ? spec.strike : spec.strike * std::exp(-r * tau);
            bM = 0.0;
        }

        for (std::size_t j = 1; j <= n; ++j) {
            const double jj = static_cast<double>(j);
            const double sig2j2 = sig * sig * jj * jj;
            const double muj = (r - q) * jj;
            const double aj = 0.5 * theta * dt * (sig2j2 - muj);
            const double bj = -theta * dt * (sig2j2 + r);
            const double cj = 0.5 * theta * dt * (sig2j2 + muj);
            lower[j] = -aj;
            diag[j] = 1.0 - bj;
            upper[j] = -cj;
            // Explicit part (weight 1 - theta) of Crank-Nicolson on the previous level.
            const double ae = 0.5 * (1.0 - theta) * dt * (sig2j2 - muj);
            const double be = -(1.0 - theta) * dt * (sig2j2 + r);
            const double ce = 0.5 * (1.0 - theta) * dt * (sig2j2 + muj);
            rhs[j] = ae * V[j - 1] + (1.0 + be) * V[j] + ce * V[j + 1];
        }

        if (american) {
            // Projected SOR against the intrinsic obstacle. The boundaries stay explicit in
            // x (not folded into rhs), so each sweep reads x[0]/x[M] directly.
            x[0] = b0;
            x[M] = bM;
            for (std::size_t j = 1; j <= n; ++j) { x[j] = std::max(V[j], obstacle[j]); }
            constexpr double omega = 1.5;
            constexpr int max_iter = 10'000;
            bool converged = false;
            for (int it = 0; it < max_iter; ++it) {
                double err = 0.0;
                for (std::size_t j = 1; j <= n; ++j) {
                    const double gs =
                        (rhs[j] - lower[j] * x[j - 1] - upper[j] * x[j + 1]) / diag[j];
                    double xn = x[j] + omega * (gs - x[j]);
                    xn = std::max(xn, obstacle[j]);
                    err = std::max(err, std::abs(xn - x[j]));
                    x[j] = xn;
                }
                if (err < 1e-8) { converged = true; break; }
            }
            if (!converged) { return make_error<double>(MathError::not_converged); }
        } else {
            // Fold the Dirichlet boundaries into the first/last interior equations, then
            // solve the tridiagonal system directly.
            rhs[1] -= lower[1] * b0;
            rhs[n] -= upper[n] * bM;
            if (!thomas(lower, diag, upper, rhs, x, n)) {
                return make_error<double>(MathError::not_converged);
            }
            x[0] = b0;
            x[M] = bM;
        }
        for (std::size_t j = 0; j <= M; ++j) { V[j] = x[j]; }
    }

    // Linear interpolation of the solution at the requested spot.
    const double pos = spec.spot / dS;
    auto jlo = static_cast<std::size_t>(std::floor(pos));
    if (jlo >= M) { jlo = M - 1; }
    const double frac = pos - static_cast<double>(jlo);
    return V[jlo] * (1.0 - frac) + V[jlo + 1] * frac;
}

auto margrabe_exchange(double spot1, double spot2, double div1, double div2, double vol1,
                       double vol2, double rho, double time) -> Result<double> {
    if (spot1 <= 0.0 || spot2 <= 0.0 || vol1 < 0.0 || vol2 < 0.0 || time <= 0.0 ||
        std::abs(rho) > 1.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double var = vol1 * vol1 + vol2 * vol2 - 2.0 * rho * vol1 * vol2;
    const double sig = std::sqrt(std::max(var, 0.0));
    const double d1disc = spot1 * std::exp(-div1 * time);
    const double d2disc = spot2 * std::exp(-div2 * time);
    if (sig <= 0.0 || time <= 0.0) {
        // Degenerate (perfectly correlated equal vols, or zero vols): discounted intrinsic.
        return std::max(d1disc - d2disc, 0.0);
    }
    const double sst = sig * std::sqrt(time);
    const double d1 = (std::log(spot1 / spot2) + (div2 - div1) * time) / sst + 0.5 * sst;
    const double d2 = d1 - sst;
    return d1disc * norm_cdf(d1) - d2disc * norm_cdf(d2);
}

auto kirk_spread(double spot1, double spot2, double strike, double rate, double div1,
                 double div2, double vol1, double vol2, double rho, double time)
    -> Result<double> {
    if (spot1 <= 0.0 || spot2 <= 0.0 || vol1 < 0.0 || vol2 < 0.0 || time <= 0.0 ||
        std::abs(rho) > 1.0) {
        return make_error<double>(MathError::domain_error);
    }
    const double f1 = spot1 * std::exp((rate - div1) * time);
    const double f2 = spot2 * std::exp((rate - div2) * time);
    const double denom = f2 + strike;
    if (denom <= 0.0) { return make_error<double>(MathError::domain_error); }
    const double w = f2 / denom;
    const double var = vol1 * vol1 - 2.0 * rho * vol1 * vol2 * w + vol2 * vol2 * w * w;
    const double sig = std::sqrt(std::max(var, 0.0));
    const double disc = std::exp(-rate * time);
    if (sig <= 0.0) {
        return disc * std::max(f1 - denom, 0.0);
    }
    const double sst = sig * std::sqrt(time);
    const double d1 = (std::log(f1 / denom) + 0.5 * sig * sig * time) / sst;
    const double d2 = d1 - sst;
    const double value = disc * (f1 * norm_cdf(d1) - denom * norm_cdf(d2));
    return std::max(value, 0.0);
}

auto bivariate_normal_cdf(double a, double b, double rho) -> double {
    if (rho >= 1.0) { return norm_cdf(std::min(a, b)); }
    if (rho <= -1.0) { return std::max(norm_cdf(a) + norm_cdf(b) - 1.0, 0.0); }
    const double base = norm_cdf(a) * norm_cdf(b);
    if (rho == 0.0) { return base; }
    // Drezner reduction: integrate the bivariate density in the correlation parameter from 0
    // to rho by composite Simpson. Exact at rho == 0 (the integral vanishes).
    constexpr int steps = 2000;  // even
    const double h = rho / static_cast<double>(steps);
    const double two_pi = 2.0 * std::numbers::pi;
    auto integrand = [&](double t) -> double {
        const double omt = 1.0 - t * t;
        return std::exp(-(a * a - 2.0 * t * a * b + b * b) / (2.0 * omt)) /
               (two_pi * std::sqrt(omt));
    };
    double sum = integrand(0.0) + integrand(rho);
    for (int i = 1; i < steps; ++i) {
        const double t = static_cast<double>(i) * h;
        sum += (i % 2 == 1 ? 4.0 : 2.0) * integrand(t);
    }
    const double integral = (h / 3.0) * sum;
    const double value = base + integral;
    // A probability, clamped to [0,1] against Simpson round-off (never outside the range).
    return std::min(std::max(value, 0.0), 1.0);
}

auto geske_compound(const OptionSpec& spec, double strike1, double strike2, double t1)
    -> Result<double> {
    const double S = spec.spot;
    const double r = spec.rate;
    const double qd = spec.dividend_yield;
    const double sig = spec.volatility;
    const double T2 = spec.time_to_expiry;
    if (S <= 0.0 || sig <= 0.0 || strike2 <= 0.0 || strike1 < 0.0 || t1 <= 0.0 || t1 >= T2) {
        return make_error<double>(MathError::domain_error);
    }
    // The inner (underlying) call struck at strike2, expiring at T2.
    auto inner = bs_call(S, strike2, r, qd, sig, T2);
    if (!inner) { return make_error<double>(MathError::domain_error); }
    if (strike1 == 0.0) {
        // A zero-premium option to acquire the call is just the call itself.
        return *inner;
    }
    // Critical spot S* at t1 where the (T2 - t1)-maturity call is worth exactly strike1.
    const double tau = T2 - t1;
    auto call_at = [&](double s) -> double {
        auto c = bs_call(s, strike2, r, qd, sig, tau);
        return c ? *c : std::numeric_limits<double>::quiet_NaN();
    };
    double lo = 1e-8;
    double hi = std::max(S, strike2) * 100.0;
    double flo = call_at(lo) - strike1;
    double fhi = call_at(hi) - strike1;
    if (!std::isfinite(flo) || !std::isfinite(fhi) || (flo > 0.0) == (fhi > 0.0)) {
        // The call value is monotone in spot from 0 to unbounded, so a sign change must
        // exist; failing to bracket it means the sub-problem is degenerate.
        return make_error<double>(MathError::not_converged);
    }
    // Bisection: the call value is monotone in spot, so 200 halvings of a bracketed sign
    // change drive the interval below any tolerance — the midpoint is the critical spot.
    double s_star = 0.5 * (lo + hi);
    for (int i = 0; i < 200; ++i) {
        s_star = 0.5 * (lo + hi);
        const double fm = call_at(s_star) - strike1;
        if (!std::isfinite(fm)) { return make_error<double>(MathError::not_converged); }
        if (std::abs(fm) < 1e-10 || (hi - lo) < 1e-12) { break; }
        if ((fm > 0.0) == (flo > 0.0)) { lo = s_star; flo = fm; } else { hi = s_star; }
    }

    const double sqrt_t1 = std::sqrt(t1);
    const double sqrt_T2 = std::sqrt(T2);
    const double a1 = (std::log(S / s_star) + (r - qd + 0.5 * sig * sig) * t1) / (sig * sqrt_t1);
    const double a2 = a1 - sig * sqrt_t1;
    const double b1 = (std::log(S / strike2) + (r - qd + 0.5 * sig * sig) * T2) / (sig * sqrt_T2);
    const double b2 = b1 - sig * sqrt_T2;
    const double corr = std::sqrt(t1 / T2);

    const double value = S * std::exp(-qd * T2) * bivariate_normal_cdf(a1, b1, corr) -
                         strike2 * std::exp(-r * T2) * bivariate_normal_cdf(a2, b2, corr) -
                         strike1 * std::exp(-r * t1) * norm_cdf(a2);
    return std::max(value, 0.0);
}

auto chooser_price(const OptionSpec& spec, double t1) -> Result<double> {
    const double S = spec.spot;
    const double K = spec.strike;
    const double r = spec.rate;
    const double qd = spec.dividend_yield;
    const double sig = spec.volatility;
    const double T2 = spec.time_to_expiry;
    if (S <= 0.0 || K <= 0.0 || sig <= 0.0 || t1 <= 0.0 || t1 > T2) {
        return make_error<double>(MathError::domain_error);
    }
    const double sqrtT2 = std::sqrt(T2);
    const double sqrt_t1 = std::sqrt(t1);
    const double d1 = (std::log(S / K) + (r - qd + 0.5 * sig * sig) * T2) / (sig * sqrtT2);
    const double d2 = d1 - sig * sqrtT2;
    const double y1 = (std::log(S / K) + (r - qd) * T2 + 0.5 * sig * sig * t1) / (sig * sqrt_t1);
    const double y2 = y1 - sig * sqrt_t1;
    const double value = S * std::exp(-qd * T2) * norm_cdf(d1) -
                         K * std::exp(-r * T2) * norm_cdf(d2) -
                         S * std::exp(-qd * T2) * norm_cdf(-y1) +
                         K * std::exp(-r * T2) * norm_cdf(-y2);
    return std::max(value, 0.0);
}

auto basket_mc(std::span<const BasketAsset> assets, std::span<const double> correlation,
               double strike, double rate, double time, OptionType type,
               std::uint64_t paths, std::uint64_t seed) -> Result<McResult> {
    const std::size_t nn = assets.size();
    if (nn == 0 || nn > kMaxBasketAssets) { return make_error<McResult>(MathError::domain_error); }
    if (correlation.size() != nn * nn) { return make_error<McResult>(MathError::domain_error); }
    if (paths == 0 || paths > kMaxPaths ||
        paths > kMaxPaths / static_cast<std::uint64_t>(nn)) {
        return make_error<McResult>(MathError::domain_error);
    }
    if (time <= 0.0) { return make_error<McResult>(MathError::domain_error); }
    for (const auto& a : assets) {
        if (a.spot <= 0.0 || a.volatility < 0.0) {
            return make_error<McResult>(MathError::domain_error);
        }
    }
    // Cholesky factor L (lower triangular) of the correlation matrix; L L^T = R.
    std::vector<double> L(nn * nn, 0.0);
    for (std::size_t i = 0; i < nn; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = correlation[i * nn + j];
            for (std::size_t k = 0; k < j; ++k) { s -= L[i * nn + k] * L[j * nn + k]; }
            if (i == j) {
                if (s <= 0.0) {
                    // Not positive-definite: no real Cholesky factor exists.
                    return make_error<McResult>(MathError::domain_error);
                }
                L[i * nn + j] = std::sqrt(s);
            } else {
                L[i * nn + j] = s / L[j * nn + j];
            }
        }
    }

    const double sqrtT = std::sqrt(time);
    const double disc = std::exp(-rate * time);
    const std::uint64_t key = splitmix64(seed);

    std::vector<double> zc(nn);   // correlated normals for the +z leg
    std::vector<double> zraw(nn); // independent draws
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t p = 0; p < paths; ++p) {
        for (std::size_t i = 0; i < nn; ++i) {
            zraw[i] = normal_at(key, p * static_cast<std::uint64_t>(nn) + static_cast<std::uint64_t>(i));
        }
        // Antithetic pair: use +zraw and -zraw through the same Cholesky factor.
        double payoff_pair = 0.0;
        for (int sign_idx = 0; sign_idx < 2; ++sign_idx) {
            const double s = sign_idx == 0 ? 1.0 : -1.0;
            for (std::size_t i = 0; i < nn; ++i) {
                double acc = 0.0;
                for (std::size_t k = 0; k <= i; ++k) { acc += L[i * nn + k] * (s * zraw[k]); }
                zc[i] = acc;
            }
            double basket = 0.0;
            for (std::size_t i = 0; i < nn; ++i) {
                const auto& a = assets[i];
                const double st = a.spot * std::exp((rate - a.dividend_yield -
                                                     0.5 * a.volatility * a.volatility) * time +
                                                    a.volatility * sqrtT * zc[i]);
                basket += a.weight * st;
            }
            const double intrinsic = type == OptionType::call ? std::max(basket - strike, 0.0)
                                                              : std::max(strike - basket, 0.0);
            payoff_pair += 0.5 * intrinsic * disc;
        }
        sum += payoff_pair;
        sum_sq += payoff_pair * payoff_pair;
    }
    const double np = static_cast<double>(paths);
    const double mean = sum / np;
    const double variance = std::max((sum_sq - np * mean * mean) / std::max(np - 1.0, 1.0), 0.0);
    return McResult{mean, std::sqrt(variance / np), paths};
}

}  // namespace nimblecas::exotics
