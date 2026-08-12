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
import nimblecas.nlsolve;
import nimblecas.parallel;

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

// ---------------------------------------------------------------------------
// Jump-diffusion and drift-implicit (theta) additions. ADDITIVE ONLY: none of the
// schemes above change, and every scheme here consumes the Brownian stream EXACTLY as
// euler_maruyama does — standard_normal(key, n) at counter indices 2n/2n+1 of
// key = splitmix64(seed) — so a variable jump count or an implicit drift solve can
// never shift a Brownian draw. All jump randomness (Poisson counts and jump marks)
// lives on a SEPARATE, domain-separated sub-key derived from the same path key, so it
// is likewise a pure function of (seed, path, step, mark index) and never touches the
// Brownian indices. CONSEQUENCES, both bit-for-bit and tested: lambda == 0 makes
// jump_euler_maruyama IDENTICAL to euler_maruyama, and theta == 0 makes theta_euler
// IDENTICAL to euler_maruyama (both special-cased to the exact same arithmetic
// expression, not merely "close").
// ---------------------------------------------------------------------------

// A compound-Poisson jump specification for a jump-diffusion SDE
// dX = a(X) dt + b(X) dW + dJ, where dJ is a compound Poisson process with intensity
// `lambda` (jumps per unit time): `size_quantile(u)` maps a uniform u in (0,1) to a
// jump mark J via the inverse CDF of the mark distribution (so marks are drawn by
// inversion, one uniform per mark), and `impulse(x, J)` maps the PRE-JUMP state x and a
// mark J to the state increment applied at a jump (e.g. a multiplicative return jump
// for a price process, or an additive shock for an additive process). lambda == 0.0
// (the default) disables jumps entirely; `size_quantile`/`impulse` may then be empty.
struct JumpSpec {
    double lambda{0.0};                                // jump intensity (jumps / unit time)
    std::function<double(double)> size_quantile{};      // u in (0,1) -> jump mark J
    std::function<double(double, double)> impulse{};    // (x, J) -> state increment c(x, J)
};

// Builds a Merton (1976) log-normal jump specification: jump marks J ~ N(mu_j, sigma_j^2),
// drawn by the inverse-normal quantile transform of a single uniform, applied to a
// price-like process via the multiplicative impulse c(x, J) = x * (e^J - 1) = x * expm1(J)
// (a jump multiplies the pre-jump state by e^J). Returns domain_error if lambda is
// negative or non-finite, mu_j is non-finite, or sigma_j is negative or non-finite.
[[nodiscard]] auto merton_jumps(double lambda, double mu_j, double sigma_j) -> Result<JumpSpec>;

// Jump-diffusion Euler-Maruyama: explicit Euler-Maruyama on the diffusion PLUS an
// explicit compound-Poisson jump sum, per step:
//
//     X_{n+1} = X_n + a(X_n) dt + b(X_n) dW_n + sum_{i=1}^{N_n} impulse(X_n, J_{n,i}),
//     N_n ~ Poisson(lambda dt).
//
// The Brownian draw dW_n is EXACTLY euler_maruyama's (same key, same counter indices);
// the Poisson count N_n and every mark J_{n,i} are drawn from a domain-separated
// sub-key (see the file-level charter above), so they never perturb the Brownian
// stream. With jumps.lambda == 0.0 this is BIT-FOR-BIT IDENTICAL to euler_maruyama (the
// jump sum is an exact 0.0 added to the Euler step, and the domain-separated draws are
// simply not made). Returns domain_error if steps == 0, T is non-finite or <= 0, x0 is
// non-finite, `a`/`b` is empty, jumps.lambda is negative/non-finite, jumps.lambda * dt
// exceeds 700 (the point at which e^{-lambda dt} underflows to 0 in double precision),
// or jumps.lambda > 0 with an empty `size_quantile`/`impulse`.
[[nodiscard]] auto jump_euler_maruyama(std::function<double(double)> a,
                                       std::function<double(double)> b, const JumpSpec& jumps,
                                       double x0, double T, std::uint64_t steps,
                                       std::uint64_t seed) -> Result<SdePath>;

// Drift-implicit (theta) Euler-Maruyama for stiff SDEs, diffusion explicit (Itô):
//
//     xi = X_n + (theta a(xi) + (1-theta) a(X_n)) dt + b(X_n) dW_n,   X_{n+1} = xi,
//
// solved for xi at each step via nlsolve::newton on the 1-dimensional residual
// r(xi) = xi - X_n - (theta a(xi) + (1-theta) a(X_n)) dt - b(X_n) dW_n, using the
// analytic Jacobian 1 - theta a'(xi) dt when `a_prime` is supplied (empty falls back to
// nlsolve's finite-difference Jacobian). theta == 1 is fully implicit backward Euler on
// the drift (unconditionally A-stable in the deterministic/linear sense); theta == 0.5
// is the (drift) trapezoidal / Crank-Nicolson rule; theta == 0.0 is EXACTLY
// euler_maruyama, special-cased to the identical arithmetic expression so the two are
// bit-for-bit identical (not merely close) rather than routed through Newton on a
// trivial linear residual. STABILITY MOTIVATION: for stiff linear mean reversion
// a(x) = -kappa(x - m) with kappa dt >> 1 (e.g. a fast-reverting short-rate or variance
// process on a coarse grid), explicit Euler-Maruyama's amplification factor
// |1 - kappa dt| exceeds 1 and the scheme oscillates/blows up; theta >= 1/2 keeps the
// implicit drift step stable regardless of kappa dt. Newton's converged == false
// (stagnation, singular Jacobian, iteration budget exhausted) is a HARD not_converged
// for the WHOLE path — an unconverged iterate is never accepted as a step. Returns
// domain_error if steps == 0, T is non-finite or <= 0, x0 is non-finite, `a`/`b` is
// empty, or theta is outside [0, 1] or non-finite.
[[nodiscard]] auto theta_euler(std::function<double(double)> a,
                               std::function<double(double)> a_prime,
                               std::function<double(double)> b, double x0, double T,
                               std::uint64_t steps, std::uint64_t seed, double theta)
    -> Result<SdePath>;

// theta_euler's drift-implicit, diffusion-explicit step PLUS jump_euler_maruyama's
// explicit compound-Poisson jump sum (jump marks and impulse applied to the PRE-STEP
// state X_n, exactly as in jump_euler_maruyama). Combines the stiff-drift stability of
// theta_euler with jump risk. theta == 0 and jumps.lambda == 0 together reduce this to
// euler_maruyama bit-for-bit (the same theta==0 special-casing as theta_euler, plus the
// same domain-separated, zero-jump-sum reasoning as jump_euler_maruyama). Domain-error
// conditions are the union of theta_euler's and jump_euler_maruyama's.
[[nodiscard]] auto jump_theta_euler(std::function<double(double)> a,
                                    std::function<double(double)> a_prime,
                                    std::function<double(double)> b, const JumpSpec& jumps,
                                    double x0, double T, std::uint64_t steps,
                                    std::uint64_t seed, double theta) -> Result<SdePath>;

// A terminal-distribution estimate carrying its own Monte Carlo statistical error, in
// the same spirit as nimblecas::pricing::McResult (nimblecas.pricing): `mean`/`variance` are the
// sample mean and unbiased (n-1) sample variance of X_T over `paths` seeded paths, and
// `std_error = sqrt(variance / paths)` is the standard error of the MEAN (not of a
// single draw). `paths` is echoed back for convenience.
struct EnsembleEstimate {
    double mean{0.0};
    double variance{0.0};
    double std_error{0.0};
    std::uint64_t paths{0};
};

// Simulate `paths` independent seeded jump_theta_euler paths (jump_theta_euler with the
// given `theta`/`jumps`; theta == 0 and jumps.lambda == 0 together recover plain
// euler_maruyama) and return their terminal values X_T. PARTITION-INDEPENDENCE: path p
// is seeded with splitmix64(seed ^ p) — EXACTLY simulate_terminal_scheme's contract —
// so X_T for path p is a pure function of (seed, p), independent of `paths` and of how
// 0..paths-1 is partitioned across workers. If ANY path's implicit solve fails to
// converge the WHOLE call returns not_converged (a partially-valid ensemble is never
// returned). Returns domain_error under the union of jump_theta_euler's conditions,
// plus paths == 0.
[[nodiscard]] auto simulate_terminal_jump(std::function<double(double)> a,
                                          std::function<double(double)> a_prime,
                                          std::function<double(double)> b, const JumpSpec& jumps,
                                          double x0, double T, std::uint64_t steps,
                                          std::uint64_t paths, std::uint64_t seed, double theta)
    -> Result<std::vector<double>>;

// { mean, variance, std_error, paths } of the terminal distribution X_T over `paths`
// seeded jump_theta_euler paths — the EnsembleEstimate counterpart of
// simulate_terminal_jump. Seeding/partition contract and domain-error conditions are
// exactly as for simulate_terminal_jump.
[[nodiscard]] auto terminal_estimate_jump(std::function<double(double)> a,
                                          std::function<double(double)> a_prime,
                                          std::function<double(double)> b, const JumpSpec& jumps,
                                          double x0, double T, std::uint64_t steps,
                                          std::uint64_t paths, std::uint64_t seed, double theta)
    -> Result<EnsembleEstimate>;

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

// ---------------------------------------------------------------------------
// Jump-diffusion / theta-implicit internals. Domain-separated from the Brownian stream
// (see the export-block charter above): every jump-related draw is keyed off
// jump_key = counter_u64(key, kJumpDomain), a sub-key that never coincides with the
// Brownian counter indices 2n/2n+1 the Box-Muller draws above consume, so a variable
// jump count can never shift where the Brownian stream is read from.
// ---------------------------------------------------------------------------

// Domain-separation constants (xxHash's PRIME64_2 / PRIME64_4 — arbitrary well-mixed odd
// 64-bit constants, chosen only to be far from any realistic step index so `jump_key`
// and `mark_key_n` never collide with the small counters n/i they are combined with).
inline constexpr std::uint64_t kJumpDomain = 0xC2B2AE3D27D4EB4FULL;
inline constexpr std::uint64_t kMarkDomain = 0x165667B19E3779F9ULL;

// Inverse standard-normal CDF (quantile function) of a single uniform p in (0, 1):
// Acklam's rational approximation plus one Halley refinement against erfc, giving
// ~machine accuracy across the whole range. Self-contained (module-local) rather than
// reusing nimblecas.pricing::inverse_norm_cdf so nimblecas.sde keeps its existing
// core+rng(+nlsolve) dependency footprint. p is nudged away from the exact endpoint 0.0
// (uniform_unit CAN return exactly 0.0) to keep the tail branch finite.
[[nodiscard]] auto inv_norm_quantile(double p) noexcept -> double {
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
    const double pc = p > 0.0 ? p : std::numeric_limits<double>::min();

    double x = 0.0;
    if (pc < p_low) {
        const double q = std::sqrt(-2.0 * std::log(pc));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (pc <= p_high) {
        const double q = pc - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - pc));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // Halley step: e = Phi(x) - p, u = e / phi(x); x -= u/(1 + x*u/2).
    const double phi = std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
    const double Phi = 0.5 * std::erfc(-x / std::numbers::sqrt2);
    const double e = Phi - pc;
    const double u = e / phi;
    x -= u / (1.0 + 0.5 * x * u);
    return x;
}

// Poisson(lambda*dt) count for grid step `step`, drawn by CDF inversion on a SINGLE
// uniform from the jump sub-stream: standard sequential-search inversion (Devroye),
// walking the Poisson pmf p_k = e^{-m} m^k / k! (m = lambda*dt) until the cumulative
// probability F_k passes the drawn uniform. Hard-capped at k_max = 4096 terms: this is
// a deterministic, DOCUMENTED cap (not a silent truncation) that only a wildly
// unrealistic lambda*dt (already rejected above 700 by the caller's domain check)
// could approach — for lambda*dt <= 700 the true count is astronomically unlikely to
// reach 4096 (a Poisson(700) count has essentially zero mass past ~900), so the cap is
// a safety valve, not an operating regime.
[[nodiscard]] auto poisson_count(std::uint64_t jump_key, std::uint64_t step,
                                 double lambda_dt) noexcept -> std::uint64_t {
    constexpr std::uint64_t k_max = 4096;
    const double u = uniform_unit(counter_u64(jump_key, step));
    double pmf = std::exp(-lambda_dt);
    double cdf = pmf;
    std::uint64_t k = 0;
    while (u > cdf && k < k_max) {
        ++k;
        pmf *= lambda_dt / static_cast<double>(k);
        cdf += pmf;
    }
    return k;
}

// Shared domain guard for the jump-consuming schemes: JumpSpec is a public aggregate a
// caller may build by hand (not only via merton_jumps), so every consumer re-validates
// it rather than trusting a factory was used. `dt` is the step size (T / steps),
// already known finite/positive by the caller's path_inputs_ok check.
[[nodiscard]] auto jump_inputs_ok(const JumpSpec& jumps, double dt) noexcept -> bool {
    if (!std::isfinite(jumps.lambda) || jumps.lambda < 0.0) {
        return false;
    }
    if (jumps.lambda * dt > 700.0) {  // e^{-lambda dt} underflows to 0.0 past this point
        return false;
    }
    if (jumps.lambda > 0.0 && (!jumps.size_quantile || !jumps.impulse)) {
        return false;
    }
    return true;
}

[[nodiscard]] auto theta_ok(double theta) noexcept -> bool {
    return std::isfinite(theta) && theta >= 0.0 && theta <= 1.0;
}

// One theta-implicit drift step (diffusion explicit), shared by theta_euler and
// jump_theta_euler: solves xi = X_n + (theta a(xi) + (1-theta) a(X_n)) dt + b(X_n) dW_n
// for xi via nlsolve::newton on a 1-vector, with the explicit Euler value as the
// initial guess. theta == 0.0 is special-cased to step_update's PLAIN Euler-Maruyama
// expression (not routed through Newton on a trivial linear residual), so theta_euler
// and jump_theta_euler are BIT-FOR-BIT identical to euler_maruyama when theta == 0.0 —
// required for the reproducibility charter, not just numerically close. A Newton
// converged == false (stagnation / singular Jacobian / iteration budget exhausted) maps
// to not_converged — the caller must never accept the unconverged iterate as a step.
[[nodiscard]] auto solve_theta_step(const std::function<double(double)>& a,
                                    const std::function<double(double)>& a_prime,
                                    const std::function<double(double)>& b, double x, double dt,
                                    double sqrt_dt, double dW, double theta) -> Result<double> {
    if (theta == 0.0) {
        return step_update(a, b, /*b_prime=*/{}, x, dt, sqrt_dt, dW, Scheme::euler_maruyama);
    }
    const double a_x = a(x);
    const double b_dW = b(x) * dW;
    const double xi0 = x + a_x * dt + b_dW;  // explicit Euler predictor (initial guess)

    nlsolve::ResidualFn F = [&](std::span<const double> xi) -> std::vector<double> {
        const double x1 = xi[0];
        return std::vector<double>{x1 - x - (theta * a(x1) + (1.0 - theta) * a_x) * dt - b_dW};
    };

    const std::array<double, 1> x0{xi0};
    const nlsolve::Options opts{};
    Result<nlsolve::SolveResult> solved;
    if (a_prime) {
        nlsolve::JacobianFn J = [&](std::span<const double> xi) -> std::vector<double> {
            return std::vector<double>{1.0 - theta * a_prime(xi[0]) * dt};
        };
        solved = nlsolve::newton(F, J, std::span<const double>{x0}, opts);
    } else {
        solved = nlsolve::newton(F, std::span<const double>{x0}, opts);
    }
    if (!solved) {
        return make_error<double>(solved.error());
    }
    if (!solved->converged) {
        return make_error<double>(MathError::not_converged);
    }
    return solved->x[0];
}

// Shared path builder for theta_euler / jump_euler_maruyama / jump_theta_euler:
// theta-implicit drift + explicit diffusion (solve_theta_step) plus an OPTIONAL
// explicit compound-Poisson jump sum (skipped entirely when jumps.lambda == 0.0 — which
// also means theta_euler's internal call, JumpSpec{}, never touches the jump-domain
// stream and never requires non-empty size_quantile/impulse). Preconditions
// (path_inputs_ok, jump_inputs_ok, theta_ok) are checked by the public callers.
[[nodiscard]] auto build_jump_theta_path(const std::function<double(double)>& a,
                                         const std::function<double(double)>& a_prime,
                                         const std::function<double(double)>& b,
                                         const JumpSpec& jumps, double x0, double T,
                                         std::uint64_t steps, std::uint64_t seed, double theta)
    -> Result<SdePath> {
    const std::uint64_t key = splitmix64(seed);
    const double dt = T / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);
    const double lambda_dt = jumps.lambda * dt;
    const bool has_jumps = jumps.lambda > 0.0;
    const std::uint64_t jump_key = has_jumps ? counter_u64(key, kJumpDomain) : std::uint64_t{0};

    SdePath path;
    path.times.reserve(static_cast<std::size_t>(steps) + 1);
    path.values.reserve(static_cast<std::size_t>(steps) + 1);

    double x = x0;
    path.times.push_back(0.0);
    path.values.push_back(x);

    for (std::uint64_t n = 0; n < steps; ++n) {
        const double dW = sqrt_dt * standard_normal(key, n);
        auto stepped = solve_theta_step(a, a_prime, b, x, dt, sqrt_dt, dW, theta);
        if (!stepped) {
            return make_error<SdePath>(stepped.error());
        }
        double x_new = *stepped;

        if (has_jumps) {
            const std::uint64_t k = poisson_count(jump_key, n, lambda_dt);
            if (k > 0) {
                const std::uint64_t mark_key_n = counter_u64(jump_key ^ kMarkDomain, n);
                double jump_sum = 0.0;
                for (std::uint64_t i = 0; i < k; ++i) {
                    const double u_mark = uniform_unit(counter_u64(mark_key_n, i));
                    const double J = jumps.size_quantile(u_mark);
                    jump_sum += jumps.impulse(x, J);  // impulse uses the PRE-STEP state X_n
                }
                x_new += jump_sum;
            }
        }

        x = x_new;
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

// --- Jump-diffusion and drift-implicit (theta) additions ---

auto merton_jumps(double lambda, double mu_j, double sigma_j) -> Result<JumpSpec> {
    if (!std::isfinite(lambda) || lambda < 0.0 || !std::isfinite(mu_j) || !std::isfinite(sigma_j) ||
        sigma_j < 0.0) {
        return make_error<JumpSpec>(MathError::domain_error);
    }
    return JumpSpec{
        lambda,
        [mu_j, sigma_j](double u) -> double { return mu_j + sigma_j * inv_norm_quantile(u); },
        [](double x, double J) -> double { return x * std::expm1(J); }};
}

auto jump_euler_maruyama(std::function<double(double)> a, std::function<double(double)> b,
                         const JumpSpec& jumps, double x0, double T, std::uint64_t steps,
                         std::uint64_t seed) -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps)) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T or x0 also rejected
    }
    const double dt = T / static_cast<double>(steps);
    if (!jump_inputs_ok(jumps, dt)) {
        return make_error<SdePath>(MathError::domain_error);
    }
    return build_jump_theta_path(a, /*a_prime=*/{}, b, jumps, x0, T, steps, seed, /*theta=*/0.0);
}

auto theta_euler(std::function<double(double)> a, std::function<double(double)> a_prime,
                 std::function<double(double)> b, double x0, double T, std::uint64_t steps,
                 std::uint64_t seed, double theta) -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps) ||
        !theta_ok(theta)) {
        return make_error<SdePath>(MathError::domain_error);  // NaN/inf T, x0, theta also rejected
    }
    return build_jump_theta_path(a, a_prime, b, JumpSpec{}, x0, T, steps, seed, theta);
}

auto jump_theta_euler(std::function<double(double)> a, std::function<double(double)> a_prime,
                      std::function<double(double)> b, const JumpSpec& jumps, double x0,
                      double T, std::uint64_t steps, std::uint64_t seed, double theta)
    -> Result<SdePath> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps) ||
        !theta_ok(theta)) {
        return make_error<SdePath>(MathError::domain_error);
    }
    const double dt = T / static_cast<double>(steps);
    if (!jump_inputs_ok(jumps, dt)) {
        return make_error<SdePath>(MathError::domain_error);
    }
    return build_jump_theta_path(a, a_prime, b, jumps, x0, T, steps, seed, theta);
}

auto simulate_terminal_jump(std::function<double(double)> a, std::function<double(double)> a_prime,
                            std::function<double(double)> b, const JumpSpec& jumps, double x0,
                            double T, std::uint64_t steps, std::uint64_t paths,
                            std::uint64_t seed, double theta) -> Result<std::vector<double>> {
    if (!path_inputs_ok(static_cast<bool>(a), static_cast<bool>(b), x0, T, steps) || paths == 0 ||
        !theta_ok(theta)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const double dt = T / static_cast<double>(steps);
    if (!jump_inputs_ok(jumps, dt)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }

    // Ensemble over paths in parallel via the TBB/PPL fork-join layer (nimblecas.parallel).
    // Partition-independence — path p is a pure function of (seed, p) — makes the parallel
    // result BIT-IDENTICAL to the serial one: transform_index is order-preserving and each
    // path writes its own slot, so the returned vector (and every moment reduced from it) is
    // deterministic regardless of worker count. The callbacks a/b/a_prime and the jump spec
    // must be pure (the documented partition-independence contract) since they are invoked
    // concurrently for distinct path indices.
    auto per_path = parallel::transform_index(
        static_cast<std::size_t>(paths), [&](std::size_t p) -> Result<double> {
            const std::uint64_t path_seed = splitmix64(seed ^ static_cast<std::uint64_t>(p));
            auto path =
                build_jump_theta_path(a, a_prime, b, jumps, x0, T, steps, path_seed, theta);
            if (!path) {
                return make_error<double>(path.error());
            }
            return path->values.back();
        });

    std::vector<double> terminals;
    terminals.reserve(static_cast<std::size_t>(paths));
    for (auto& r : per_path) {
        if (!r) {
            // A single path's implicit solve failing to converge invalidates the whole
            // ensemble call — never silently drop a path or accept an unconverged step.
            // Deterministic: the lowest-index failing path is reported first.
            return make_error<std::vector<double>>(r.error());
        }
        terminals.push_back(*r);
    }
    return terminals;
}

auto terminal_estimate_jump(std::function<double(double)> a, std::function<double(double)> a_prime,
                            std::function<double(double)> b, const JumpSpec& jumps, double x0,
                            double T, std::uint64_t steps, std::uint64_t paths,
                            std::uint64_t seed, double theta) -> Result<EnsembleEstimate> {
    auto terminals = simulate_terminal_jump(std::move(a), std::move(a_prime), std::move(b), jumps,
                                            x0, T, steps, paths, seed, theta);
    if (!terminals) {
        return make_error<EnsembleEstimate>(terminals.error());
    }
    const auto [mean, variance] = compute_moments(*terminals);
    return EnsembleEstimate{mean, variance, std::sqrt(variance / static_cast<double>(paths)),
                            paths};
}

}  // namespace nimblecas
