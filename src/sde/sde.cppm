// NimbleCAS stochastic differential equations — Euler-Maruyama & Milstein (seeded, parallelisable).
// @author Olumuyiwa Oluwasanmi
//
// Deterministic (seeded) numerical integrators for scalar Itô stochastic differential
// equations  dX = a(X) dt + b(X) dW_t,  X(0) = x0,  on [0, T] with `steps` uniform steps
// dt = T / steps and standard-normal Wiener increments dW = sqrt(dt) · Z, Z ~ N(0, 1). Two
// schemes are provided: Euler-Maruyama (strong order 1/2, weak order 1) and Milstein (strong
// order 1), the latter adding the Itô-correction term ½ b(X) b'(X) (dW² − dt).
//
// HONESTY. This is a NUMERICAL solver in IEEE-754 double precision, NOT an exact symbolic
// one. The paths of an SDE are almost surely non-differentiable and are not representable
// over ℚ, so — unlike the exact power-series ODE tools (nimblecas.ode / nimblecas.perturbation),
// which return exact rational/series coefficients — these routines return floating-point
// approximations carrying both discretisation error (O(√dt) strong for Euler-Maruyama,
// O(dt) strong for Milstein) and Monte Carlo sampling error in any ensemble average. What
// they DO guarantee is determinism: every draw is a pure function of a seed, so equal seeds
// reproduce bit-identical paths.
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

// Generate one full sample path with either scheme. Preconditions (steps > 0, T > 0, the
// required std::functions non-empty) are checked by the public callers; this builder assumes
// them. When use_milstein is false the correction term is skipped and `b_prime` is never
// invoked (so an empty b_prime is fine for Euler-Maruyama).
[[nodiscard]] auto simulate_path(const std::function<double(double)>& a,
                                 const std::function<double(double)>& b,
                                 const std::function<double(double)>& b_prime, double x0, double T,
                                 std::uint64_t steps, std::uint64_t seed, bool use_milstein)
    -> SdePath {
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
        const double bx = b(x);
        double next = x + a(x) * dt + bx * dW;
        if (use_milstein) {
            // Itô correction ½ b b' (dW² − dt): vanishes in mean but lifts the strong order.
            next += 0.5 * bx * b_prime(x) * (dW * dW - dt);
        }
        x = next;

        // Set the final grid time to T exactly; interior nodes are n·dt. This avoids the tiny
        // rounding drift of steps·(T/steps) accumulating into the reported end time.
        const std::uint64_t step_index = n + 1;
        path.times.push_back(step_index == steps ? T : static_cast<double>(step_index) * dt);
        path.values.push_back(x);
    }

    return path;
}

}  // namespace

auto euler_maruyama(std::function<double(double)> a, std::function<double(double)> b, double x0,
                    double T, std::uint64_t steps, std::uint64_t seed) -> Result<SdePath> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || !a || !b) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, /*b_prime=*/{}, x0, T, steps, seed, /*use_milstein=*/false);
}

auto milstein(std::function<double(double)> a, std::function<double(double)> b,
              std::function<double(double)> b_prime, double x0, double T, std::uint64_t steps,
              std::uint64_t seed) -> Result<SdePath> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || !a || !b ||
        !b_prime) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    return simulate_path(a, b, b_prime, x0, T, steps, seed, /*use_milstein=*/true);
}

auto simulate_terminal(std::function<double(double)> a, std::function<double(double)> b,
                       std::function<double(double)> b_prime, double x0, double T,
                       std::uint64_t steps, std::uint64_t paths, std::uint64_t seed,
                       bool use_milstein) -> Result<std::vector<double>> {
    if (steps == 0 || !std::isfinite(T) || T <= 0.0 || !std::isfinite(x0) || paths == 0 || !a ||
        !b || (use_milstein && !b_prime)) {
        return make_error<std::vector<double>>(MathError::domain_error);  // NaN/inf also rejected
    }

    std::vector<double> terminals;
    terminals.reserve(static_cast<std::size_t>(paths));

    for (std::uint64_t p = 0; p < paths; ++p) {
        // Per-path-index seed: bijective mixing of (seed, p) gives independent, reproducible
        // paths that do not depend on the path count or scheduling order.
        const std::uint64_t path_seed = splitmix64(seed ^ p);
        const SdePath path = simulate_path(a, b, b_prime, x0, T, steps, path_seed, use_milstein);
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
    const std::vector<double>& xs = terminals.value();

    // Two-pass mean then sum of squared deviations (more stable than E[x²] − E[x]²).
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
    // Unbiased (n − 1) estimator of the terminal variance; a single path has no dispersion.
    const double variance = xs.size() >= 2 ? sq / (n - 1.0) : 0.0;

    return std::pair<double, double>{mean, variance};
}

}  // namespace nimblecas
