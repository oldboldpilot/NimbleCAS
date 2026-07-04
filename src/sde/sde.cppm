// NimbleCAS stochastic differential equations — seeded, parallelisable strong/weak schemes.
// @author Olumuyiwa Oluwasanmi
//
// Deterministic (seeded) numerical integrators for scalar Itô stochastic differential
// equations  dX = a(X) dt + b(X) dW_t,  X(0) = x0,  on [0, T] with `steps` uniform steps
// dt = T / steps and standard-normal Wiener increments dW = sqrt(dt) · Z, Z ~ N(0, 1). Five
// schemes are provided (all consuming the SAME one-normal-per-step Brownian stream, so a given
// (seed, path) is reproducible across schemes and b ≡ 0 makes the Itô schemes coincide):
//
//   • Euler-Maruyama  — strong order 1/2, weak order 1                    (Itô).
//   • Milstein        — strong order 1, adds ½ b b' (dW² − dt)           (Itô; needs b').
//   • Stochastic Heun — strong order 1, predictor-corrector trapezoid    (STRATONOVICH — see below).
//   • SRK (Platen)    — strong order 1, derivative-FREE Milstein variant  (Itô; needs no b').
//   • Tamed Euler     — strong order 1/2, drift-tamed for stiff/superlinear drift (Itô).
//
// CONVENTION WARNING (Itô vs Stratonovich). Euler-Maruyama, Milstein, the derivative-free SRK
// and Tamed Euler all approximate the Itô solution of dX = a dt + b dW. Stochastic Heun, being
// a predictor-corrector that re-uses the SAME increment dW in both stages, instead converges to
// the STRATONOVICH solution of dX = a dt + b ∘ dW — equivalently the Itô SDE with drift
// a + ½ b b'. The two agree only when b b' ≡ 0 (e.g. additive noise). For geometric Brownian
// motion a = μx, b = σx this means E[X_T] = x0·e^{μT} for the Itô schemes but x0·e^{(μ+½σ²)T}
// for Heun; the tests reflect exactly this. Do not mix the conventions unknowingly.
//
// HONESTY. This is a NUMERICAL solver in IEEE-754 double precision, NOT an exact symbolic
// one. The paths of an SDE are almost surely non-differentiable and are not representable
// over ℚ, so — unlike the exact power-series ODE tools (nimblecas.ode / nimblecas.perturbation),
// which return exact rational/series coefficients — these routines return floating-point
// approximations carrying both discretisation error (O(√dt) or O(dt) strong, per the orders
// listed above) and Monte Carlo sampling error in any ensemble average. What they DO guarantee
// is determinism: every draw is a pure function of a seed, so equal seeds reproduce bit-identical
// paths. None of these schemes is exact; none claims exactness over ℚ.
//
// PARALLELISM. Each path is generated from the STATELESS counter core counter_u64(key, i)
// of nimblecas.rng: for a fixed key, increment i of the driving Brownian motion is a pure
// function of i alone, and each path index p in an ensemble is seeded independently via
// splitmix64(seed ^ p) — mirroring mcmc's run_parallel_chains contract. Consequently a path
// is a pure function of (seed, p): any split of the path range 0..paths-1 across workers,
// reassembled in index order, reproduces the same ensemble bit-for-bit, regardless of worker
// count or scheduling. There is no time/entropy seeding and no global mutable state, and all
// failure travels the railway (Result<T> / MathError), never an exception.

export module nimblecas.sde;

import std;
import nimblecas.core;
import nimblecas.rng;

export namespace nimblecas {

// A single simulated sample path: `times[n]` is the grid time n·dt (with the final entry set
// exactly to T) and `values[n]` the scheme's approximation of X at that time. Both vectors
// have length steps + 1 (the initial point X(0) = x0 included), so times.size() == values.size().
struct SdePath {
    std::vector<double> times;
    std::vector<double> values;
};

// Numerical integration scheme selector for the generic ensemble drivers
// (simulate_terminal_scheme / terminal_moments_scheme). Each value names one of the
// single-path integrators below; see the file header for convergence orders and the
// Itô/Stratonovich convention of each. `milstein` is the only value that consumes `b_prime`.
enum class Scheme : std::uint8_t {
    euler_maruyama,   // strong 1/2, weak 1   (Itô)
    milstein,         // strong 1             (Itô; requires b')
    stochastic_heun,  // strong 1             (STRATONOVICH)
    srk,              // strong 1             (Itô; derivative-free)
    tamed_euler,      // strong 1/2           (Itô; stiff-stable)
};

// Euler-Maruyama integration of dX = a(X) dt + b(X) dW on [0, T]:
//
//     X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n,   dW_n = sqrt(dt) · Z_n,  Z_n ~ N(0, 1).
//
// Strong order 1/2, weak order 1. The Brownian increments are drawn from the stateless
// counter core keyed by splitmix64(seed): increment n consumes counter draws 2n and 2n+1 and
// is a pure function of n, so the path is fully reproducible from `seed`. Returns
// domain_error if steps == 0, T <= 0, or either `a` or `b` is an empty std::function.
[[nodiscard]] auto euler_maruyama(std::function<double(double)> a, std::function<double(double)> b,
                                  double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;

// Milstein integration of dX = a(X) dt + b(X) dW on [0, T], adding the first-order Itô
// correction that raises the strong order to 1:
//
//     X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n + ½ b(X_n) b'(X_n) (dW_n² − dt).
//
// `b_prime` is the derivative b'(x) of the diffusion coefficient, supplied by the caller (the
// scheme needs it explicitly). Uses the same seeded Brownian increments as euler_maruyama, so
// with b ≡ 0 the two schemes coincide. Returns domain_error if steps == 0, T <= 0, or any of
// `a`, `b`, `b_prime` is an empty std::function.
[[nodiscard]] auto milstein(std::function<double(double)> a, std::function<double(double)> b,
                            std::function<double(double)> b_prime, double x0, double T,
                            std::uint64_t steps, std::uint64_t seed) -> Result<SdePath>;

// Simulate `paths` independent sample paths and return only their terminal values X_T (the
// natural input for estimating a terminal expectation such as an option price). With
// use_milstein == false the Euler-Maruyama scheme is used and `b_prime` is ignored (pass {});
// with use_milstein == true the Milstein scheme is used and `b_prime` must be non-empty.
//
// PARTITION-INDEPENDENCE: path p is seeded with splitmix64(seed ^ p), so X_T for path p is a
// pure function of (seed, p) and independent of `paths` and of the order the paths are run —
// any decomposition of 0..paths-1 across workers reproduces the same vector element-wise.
// Returns domain_error if steps == 0, T <= 0, paths == 0, `a`/`b` is empty, or use_milstein
// is set with an empty `b_prime`.
[[nodiscard]] auto simulate_terminal(std::function<double(double)> a,
                                     std::function<double(double)> b,
                                     std::function<double(double)> b_prime, double x0, double T,
                                     std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                                     bool use_milstein) -> Result<std::vector<double>>;

// Estimate the first two moments of the terminal distribution of X_T over `paths` independent
// seeded paths: returns { sample mean, sample variance } of the terminal values. The variance
// uses the unbiased (Bessel, n − 1) estimator when paths >= 2 and is 0 for a single path.
// Scheme selection and the seeding/partition contract are exactly as for simulate_terminal.
// Returns domain_error under the same conditions as simulate_terminal.
[[nodiscard]] auto terminal_moments(std::function<double(double)> a, std::function<double(double)> b,
                                    std::function<double(double)> b_prime, double x0, double T,
                                    std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                                    bool use_milstein) -> Result<std::pair<double, double>>;

// ---------------------------------------------------------------------------
// Additional single-path integrators. Each mirrors euler_maruyama's signature/return style and
// draws from the SAME seeded Brownian stream (one N(0,1) per step via standard_normal), so with
// b ≡ 0 the Itô schemes reproduce euler_maruyama bit-for-bit, and (seed, path) is reproducible
// across schemes. All reject steps == 0, non-finite/non-positive T, non-finite x0, or an empty
// `a`/`b` with MathError::domain_error, exactly as the existing solvers do. None needs b'(x).
// ---------------------------------------------------------------------------

// Stochastic Heun (predictor-corrector trapezoidal) integration of dX = a(X) dt + b(X) dW:
//
//     X̃      = X_n + a(X_n) dt + b(X_n) dW_n                              (Euler predictor)
//     X_{n+1} = X_n + ½(a(X_n)+a(X̃)) dt + ½(b(X_n)+b(X̃)) dW_n            (trapezoidal corrector)
//
// CONVENTION: because the SAME increment dW_n is used in predictor and corrector, this scheme
// converges to the STRATONOVICH solution dX = a dt + b ∘ dW (equivalently the Itô SDE with drift
// a + ½ b b'), NOT the Itô solution the other schemes target. Strong order 1.0, weak order 1.0.
// Derivative-free (no b' needed). With b ≡ 0 it reduces to the deterministic Heun/RK2 ODE step,
// so it does NOT coincide with Euler-Maruyama there (Euler-Maruyama is forward Euler on the drift).
[[nodiscard]] auto stochastic_heun(std::function<double(double)> a, std::function<double(double)> b,
                                   double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;

// Derivative-free stochastic Runge-Kutta (Platen's order-1.0 SRK, a.k.a. the derivative-free
// Milstein scheme, Kloeden-Platen §11.1) integration of dX = a(X) dt + b(X) dW:
//
//     Ŷ      = X_n + a(X_n) dt + b(X_n) √dt                               (supporting value)
//     X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n
//               + (b(Ŷ) − b(X_n)) (dW_n² − dt) / (2 √dt)
//
// This reproduces Milstein's strong order 1.0 in the ITÔ sense while replacing the analytic
// derivative b'(x) that Milstein requires with a finite difference of b — so it composes with
// the plain a/b callback signature and needs no b'. Weak order 1.0. With b ≡ 0 the correction
// term vanishes and it coincides with euler_maruyama bit-for-bit.
[[nodiscard]] auto srk(std::function<double(double)> a, std::function<double(double)> b, double x0,
                       double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath>;

// Tamed Euler-Maruyama (Hutzenthaler-Jentzen-Kloeden 2012) integration of dX = a(X) dt + b(X) dW:
//
//     X_{n+1} = X_n + a(X_n) dt / (1 + |a(X_n)| dt) + b(X_n) dW_n
//
// STABILITY MOTIVATION: for superlinearly growing drift (e.g. a(x) = −x³, one-sided Lipschitz
// but NOT globally Lipschitz) explicit Euler-Maruyama diverges — its absolute moments blow up to
// +∞ as the step count grows, because a single large excursion is amplified by the unbounded
// drift. Taming caps the per-step drift increment at 1/dt in magnitude (|a dt/(1+|a|dt)| < 1),
// so the step stays finite where plain Euler overflows, while leaving the scheme's strong order
// at 1/2 and its weak/strong limit unchanged (the taming perturbation is O(dt) per step). Itô
// convention. Derivative-free.
[[nodiscard]] auto tamed_euler(std::function<double(double)> a, std::function<double(double)> b,
                               double x0, double T, std::uint64_t steps, std::uint64_t seed)
    -> Result<SdePath>;

// ---------------------------------------------------------------------------
// Generic ensemble drivers (scheme-parameterised). These mirror simulate_terminal /
// terminal_moments exactly — same per-path seeding path_seed = splitmix64(seed ^ p), same
// deterministic in-index-order reduction, same partition/thread-count independence — but select
// the integrator through the Scheme enum instead of the use_milstein bool, so the Heun, SRK and
// tamed schemes get the same reproducible multi-path driver the original schemes have.
// ---------------------------------------------------------------------------

// Simulate `paths` independent seeded paths with the chosen `scheme` and return their terminal
// values X_T (index p is a pure function of (seed, p), independent of `paths` and of how the
// range 0..paths-1 is partitioned across workers). `b_prime` is consulted ONLY when
// scheme == Scheme::milstein; for every other scheme it is ignored (pass {}). Returns
// domain_error if steps == 0, T is non-finite or ≤ 0, x0 is non-finite, paths == 0, `a`/`b` is
// empty, or scheme == Scheme::milstein with an empty `b_prime`.
[[nodiscard]] auto simulate_terminal_scheme(std::function<double(double)> a,
                                            std::function<double(double)> b,
                                            std::function<double(double)> b_prime, double x0,
                                            double T, std::uint64_t steps, std::uint64_t paths,
                                            std::uint64_t seed, Scheme scheme)
    -> Result<std::vector<double>>;

// Estimate { sample mean, unbiased (n−1) sample variance } of X_T over `paths` seeded paths using
// the chosen `scheme`. Seeding/partition contract and domain-error conditions are exactly as for
// simulate_terminal_scheme.
[[nodiscard]] auto terminal_moments_scheme(std::function<double(double)> a,
                                           std::function<double(double)> b,
                                           std::function<double(double)> b_prime, double x0,
                                           double T, std::uint64_t steps, std::uint64_t paths,
                                           std::uint64_t seed, Scheme scheme)
    -> Result<std::pair<double, double>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// One standard-normal draw Z ~ N(0, 1) for Brownian increment `index` of a path with stream
// key `key`, via the Box-Muller transform on two decorrelated counter draws (indices 2·index
// and 2·index+1). Keeping each Z a pure function of (key, index) is what makes a path
// reproducible and its increments partition-independent along the index axis. uniform_unit
// yields [0, 1); u1 == 0 would send log to −inf, so it is nudged to the smallest positive
// double before the logarithm (a negligible, deterministic perturbation).
[[nodiscard]] auto standard_normal(std::uint64_t key, std::uint64_t index) noexcept -> double {
    const double u1 = uniform_unit(counter_u64(key, 2 * index));
    const double u2 = uniform_unit(counter_u64(key, 2 * index + 1));
    const double safe = u1 > 0.0 ? u1 : std::numeric_limits<double>::min();
    const double radius = std::sqrt(-2.0 * std::log(safe));
    return radius * std::cos(2.0 * std::numbers::pi * u2);
}

// Advance one step of the chosen scheme from state `x` given the (already scaled) Brownian
// increment `dW` and step size `dt` (`sqrt_dt = √dt`). Split out so the path builder and any
// future adaptive driver share one definition of each update. `b_prime` is invoked ONLY by the
// Milstein branch, so an empty b_prime is fine for every other scheme.
[[nodiscard]] auto step_update(const std::function<double(double)>& a,
                               const std::function<double(double)>& b,
                               const std::function<double(double)>& b_prime, double x, double dt,
                               double sqrt_dt, double dW, Scheme scheme) -> double {
    switch (scheme) {
        case Scheme::euler_maruyama: {
            // X + a dt + b dW. Kept byte-identical to the original Euler-Maruyama expression.
            const double bx = b(x);
            return x + a(x) * dt + bx * dW;
        }
        case Scheme::milstein: {
            // Itô correction ½ b b' (dW² − dt): vanishes in mean but lifts the strong order to 1.
            const double bx = b(x);
            return x + a(x) * dt + bx * dW + 0.5 * bx * b_prime(x) * (dW * dW - dt);
        }
        case Scheme::stochastic_heun: {
            // Predictor-corrector trapezoid re-using dW in both stages ⇒ Stratonovich limit.
            const double a0 = a(x);
            const double b0 = b(x);
            const double x_pred = x + a0 * dt + b0 * dW;
            return x + 0.5 * (a0 + a(x_pred)) * dt + 0.5 * (b0 + b(x_pred)) * dW;
        }
        case Scheme::srk: {
            // Derivative-free Milstein (Platen): finite-difference of b along a √dt support point
            // replaces the analytic b'. Reduces to Euler-Maruyama when b ≡ 0 (correction is 0).
            const double a0 = a(x);
            const double b0 = b(x);
            const double y_hat = x + a0 * dt + b0 * sqrt_dt;
            return x + a0 * dt + b0 * dW + (b(y_hat) - b0) * (dW * dW - dt) / (2.0 * sqrt_dt);
        }
        case Scheme::tamed_euler: {
            // Cap the per-step drift increment at magnitude < 1 so superlinear drift cannot
            // overflow the step (|a dt|/(1+|a| dt) < 1); diffusion term is the plain Euler one.
            const double a0 = a(x);
            const double tamed_drift = a0 * dt / (1.0 + std::abs(a0) * dt);
            return x + tamed_drift + b(x) * dW;
        }
    }
    return x;  // unreachable; every Scheme value is handled above.
}

// Generate one full sample path with the chosen scheme. Preconditions (steps > 0, T > 0, the
// required std::functions non-empty) are checked by the public callers; this builder assumes
// them. The Brownian stream is one N(0,1) per step, identical across schemes for a fixed seed.
[[nodiscard]] auto simulate_path(const std::function<double(double)>& a,
                                 const std::function<double(double)>& b,
                                 const std::function<double(double)>& b_prime, double x0, double T,
                                 std::uint64_t steps, std::uint64_t seed, Scheme scheme) -> SdePath {
    const std::uint64_t key = splitmix64(seed);
    const double dt = T / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);

    SdePath path;
    path.times.reserve(static_cast<std::size_t>(steps) + 1);
    path.values.reserve(static_cast<std::size_t>(steps) + 1);

    double x = x0;
    path.times.push_back(0.0);
    path.values.push_back(x);

    for (std::uint64_t n = 0; n < steps; ++n) {
        const double dW = sqrt_dt * standard_normal(key, n);
        x = step_update(a, b, b_prime, x, dt, sqrt_dt, dW, scheme);

        // Set the final grid time to T exactly; interior nodes are n·dt. This avoids the tiny
        // rounding drift of steps·(T/steps) accumulating into the reported end time.
        const std::uint64_t step_index = n + 1;
        path.times.push_back(step_index == steps ? T : static_cast<double>(step_index) * dt);
        path.values.push_back(x);
    }

    return path;
}

// { sample mean, unbiased (n−1) sample variance } of a non-empty terminal-value vector. Two-pass
// (mean, then summed squared deviations) for numerical stability over E[x²] − E[x]². Shared by
// terminal_moments and terminal_moments_scheme.
[[nodiscard]] auto compute_moments(const std::vector<double>& xs) noexcept
    -> std::pair<double, double> {
    double sum = 0.0;
    for (const double v : xs) {
        sum += v;
    }
    const double n = static_cast<double>(xs.size());
    const double mean = sum / n;

    double sq = 0.0;
    for (const double v : xs) {
        const double d = v - mean;
        sq += d * d;
    }
    const double variance = xs.size() >= 2 ? sq / (n - 1.0) : 0.0;
    return std::pair<double, double>{mean, variance};
}

// Shared input guard for the single-path integrators (railway domain check). `b_prime_ok`
// must already fold in any scheme-specific b' requirement.
[[nodiscard]] auto path_inputs_ok(bool has_a, bool has_b, double x0, double T,
                                  std::uint64_t steps) noexcept -> bool {
    return steps != 0 && std::isfinite(T) && T > 0.0 && std::isfinite(x0) && has_a && has_b;
}

}  // namespace

auto euler_maruyama(std::function<double(double)> a, std::function<double(double)> b, double x0,
                    double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || !a || !b) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, /*b_prime=*/{}, x0, T, steps, seed, Scheme::euler_maruyama);
}

auto milstein(std::function<double(double)> a, std::function<double(double)> b,
              std::function<double(double)> b_prime, double x0, double T, std::uint64_t steps,
              std::uint64_t seed) -> Result<SdePath> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || !a || !b ||
        !b_prime) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, b_prime, x0, T, steps, seed, Scheme::milstein);
}

auto simulate_terminal(std::function<double(double)> a, std::function<double(double)> b,
                       std::function<double(double)> b_prime, double x0, double T,
                       std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                       bool use_milstein) -> Result<std::vector<double>> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || paths == 0 || !a ||
        !b || (use_milstein && !b_prime)) {
        return make_error<std::vector<double>>(MathError::domain_error);  // NaN/inf also rejected
    }

    const Scheme scheme = use_milstein ? Scheme::milstein : Scheme::euler_maruyama;

    std::vector<double> terminals;
    terminals.reserve(static_cast<std::size_t>(paths));

    for (std::uint64_t p = 0; p < paths; ++p) {
        // Per-path-index seed: bijective mixing of (seed, p) gives independent, reproducible
        // paths that do not depend on the path count or scheduling order.
        const std::uint64_t path_seed = splitmix64(seed ^ p);
        const SdePath path = simulate_path(a, b, b_prime, x0, T, steps, path_seed, scheme);
        terminals.push_back(path.values.back());
    }

    return terminals;
}

auto terminal_moments(std::function<double(double)> a, std::function<double(double)> b,
                      std::function<double(double)> b_prime, double x0, double T,
                      std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                      bool use_milstein) -> Result<std::pair<double, double>> {
    auto terminals = simulate_terminal(std::move(a), std::move(b), std::move(b_prime), x0, T, steps,
                                       paths, seed, use_milstein);
    if (!terminals) {
        return make_error<std::pair<double, double>>(terminals.error());
    }
    return compute_moments(terminals.value());
}

// --- Additional single-path integrators (all derivative-free) ---

auto stochastic_heun(std::function<double(double)> a, std::function<double(double)> b, double x0,
                     double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps)) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, /*b_prime=*/{}, x0, T, steps, seed, Scheme::stochastic_heun);
}

auto srk(std::function<double(double)> a, std::function<double(double)> b, double x0, double T,
         std::uint64_t steps, std::uint64_t seed) -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps)) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, /*b_prime=*/{}, x0, T, steps, seed, Scheme::srk);
}

auto tamed_euler(std::function<double(double)> a, std::function<double(double)> b, double x0,
                 double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps)) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, /*b_prime=*/{}, x0, T, steps, seed, Scheme::tamed_euler);
}

// --- Generic scheme-parameterised ensemble drivers ---

auto simulate_terminal_scheme(std::function<double(double)> a, std::function<double(double)> b,
                              std::function<double(double)> b_prime, double x0, double T,
                              std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                              Scheme scheme) -> Result<std::vector<double>> {
    // Only Milstein consumes b'; requesting it without one is a domain error, not a thrown
    // std::bad_function_call off the railway.
    const bool needs_b_prime = scheme == Scheme::milstein;
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps) || paths == 0 ||
        (needs_b_prime && !b_prime)) {
        return make_error<std::vector<double>>(MathError::domain_error);  // NaN/inf also rejected
    }

    std::vector<double> terminals;
    terminals.reserve(static_cast<std::size_t>(paths));

    for (std::uint64_t p = 0; p < paths; ++p) {
        // Same per-path-index seed as simulate_terminal: X_T for path p is a pure function of
        // (seed, p), independent of `paths` and of how 0..paths-1 is split across workers.
        const std::uint64_t path_seed = splitmix64(seed ^ p);
        const SdePath path = simulate_path(a, b, b_prime, x0, T, steps, path_seed, scheme);
        terminals.push_back(path.values.back());
    }

    return terminals;
}

auto terminal_moments_scheme(std::function<double(double)> a, std::function<double(double)> b,
                             std::function<double(double)> b_prime, double x0, double T,
                             std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                             Scheme scheme) -> Result<std::pair<double, double>> {
    auto terminals = simulate_terminal_scheme(std::move(a), std::move(b), std::move(b_prime), x0, T,
                                              steps, paths, seed, scheme);
    if (!terminals) {
        return make_error<std::pair<double, double>>(terminals.error());
    }
    return compute_moments(terminals.value());
}

}  // namespace nimblecas
