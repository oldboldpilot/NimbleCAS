// NimbleCAS Sequential Monte Carlo / stochastic-process simulation (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// Deep stochastic-process simulation built on the parallelisable, counter-based RNG
// substrate (nimblecas.rng), extending the sampling/MCMC primitives with particle
// filtering (Sequential Importance Resampling) and variance-reduced Monte Carlo
// integration.
//
// NUMERICAL / STATISTICAL NATURE — READ FIRST. Unlike the exact-over-Q engines in this
// repo, everything here operates on IEEE-754 `double` and every public result is a
// STATISTICAL ESTIMATE carrying Monte Carlo variance. These are NOT exact values: an
// estimate is a random quantity whose realised number depends on the seed, and its
// std_error field quantifies the sampling noise around the true integral / filtered mean.
// The honesty contract is therefore two-fold:
//   (a) DETERMINISM & REPRODUCIBILITY. Given a seed, the returned numbers are fixed and
//       bit-reproducible. Every random draw is a pure function of (seed, step, particle)
//       obtained from the STATELESS counter core counter_u64/Rng::split — there is no
//       shared mutable RNG state across particles, so the results are independent of the
//       thread count or how any parallel executor partitions the particle set. A parallel
//       run reproduces the serial numbers exactly.
//   (b) DOCUMENTED STATISTICAL PROPERTIES, NOT EXACTNESS. Each estimator's bias and
//       consistency are stated plainly below. The plain/antithetic/stratified integral
//       estimators are unbiased; the control-variate and particle-filter estimators are
//       consistent (asymptotically unbiased) but carry an O(1/N) finite-sample bias from
//       estimating the control coefficient / from resampling. We never claim an exact
//       result.
//
// All failure travels the railway (Result<T> / MathError), never an exception:
//   * domain_error  — structural misuse: empty inputs, zero particle/sample counts,
//                     non-normalizable (all-zero or negative) weight vectors, b < a.
//   * undefined_value — a non-finite number entered the computation: NaN/Inf weights or
//                     NaN/+Inf log-weights that cannot be normalised on the railway.
// (MathError has no dedicated `numeric_error`; non-finite numeric failure maps to
// undefined_value, structural failure to domain_error.)

export module nimblecas.smc;

import std;
import nimblecas.core;
import nimblecas.rng;

// A NESTED namespace deliberately isolates the generic names used here (resample,
// Particle-style weight vectors, Estimate) from the flat nimblecas:: surface.
export namespace nimblecas::smc {

// ---------------------------------------------------------------------------
// Resampling schemes.
//
// Each takes a NORMALIZED weight vector (non-negative, summing to ~1; the routines
// re-normalise internally for robustness) and a seed, and returns exactly N parent
// indices in [0, N) — the particle count is preserved. All draws come from the
// counter-based RNG, so the returned index vector is deterministic given the seed and is
// independent of any partitioning of the work.
//
// Guards (all schemes): empty weights => domain_error; any weight NaN/Inf =>
// undefined_value; any weight < 0 or total <= 0 (non-normalizable) => domain_error.
// ---------------------------------------------------------------------------

// Multinomial: N independent inverse-CDF draws u_i = uniform_unit(counter_u64(key, i)).
// Unbiased (E[#copies of i] = N w_i) but highest variance of the four schemes.
[[nodiscard]] auto multinomial_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>>;

// Systematic (low-variance): a SINGLE uniform u0 in [0, 1/N) placed at u0 + k/N for
// k in [0, N). Deterministic given that one uniform; minimal Monte Carlo variance but the
// draws are correlated. Unbiased in expectation.
[[nodiscard]] auto systematic_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>>;

// Stratified (low-variance): one uniform per stratum, position (k + U_k)/N with
// U_k = uniform_unit(counter_u64(key, k)). Unbiased; variance between multinomial and
// systematic, but each stratum is sampled exactly once so no region is starved.
[[nodiscard]] auto stratified_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>>;

// Residual: deterministically copy floor(N w_i) times, then fill the R remaining slots by
// multinomial resampling on the normalised fractional residuals. Lowers variance versus
// pure multinomial by removing the integer part of the allocation from the random draw.
[[nodiscard]] auto residual_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>>;

// Kish effective sample size: (sum w)^2 / sum w^2. For a normalized weight vector this is
// 1 / sum w_i^2 and lies in [1, N] — N for uniform weights (no degeneracy), 1 for a
// one-hot vector (total collapse). A diagnostic of weight degeneracy, not an exact count.
// Empty => domain_error; NaN/Inf or negative weight => undefined_value/domain_error;
// sum of squares 0 (all-zero) => domain_error.
[[nodiscard]] auto effective_sample_size(std::span<const double> weights) -> Result<double>;

// ---------------------------------------------------------------------------
// Bootstrap particle filter (Sequential Importance Resampling).
// ---------------------------------------------------------------------------

// Which resampling scheme the filter uses when it decides to resample.
enum class ResampleScheme : std::uint8_t { multinomial, systematic, stratified, residual };

// General model callbacks. State and observation are scalar `double` (the filter tracks a
// 1-D marginal so that "filtered mean" is well defined); the DYNAMICS are fully general
// through these callbacks, each fed its own independent, deterministically-derived Rng.
using StateSampler     = std::function<double(Rng&)>;          // draw x_0
using TransitionSampler = std::function<double(double, Rng&)>; // propagate(x_{t-1}, rng)->x_t
using LogWeight        = std::function<double(double, double)>; // log-likelihood(x_t, y_t)

// Output of a filter run. filtered_means[t] estimates E[x_t | y_{1:t}], one per
// observation; effective_sample_sizes[t] is the post-weighting ESS diagnostic;
// log_marginal_likelihood is the running estimate of log p(y_{1:T}).
struct BootstrapFilterResult {
    std::vector<double> filtered_means;
    std::vector<double> effective_sample_sizes;
    double log_marginal_likelihood;
};

// Run a bootstrap particle filter over `observations` with `particles` particles.
//
// At each step the particles are propagated through `propagate`, weighted by
// exp(log_weight(x, y)) (carrying the previous normalized weights so ADAPTIVE resampling
// is exact), the filtered mean and ESS are recorded, the log-marginal increment
// log sum_j W_j^prev exp(loglik_j) is accumulated, and the particles are resampled with
// `scheme` ONLY when ESS < ess_threshold * particles (adaptive resampling; pass a
// threshold >= 1 to resample every step, <= 0 to never resample). Weights are reset to
// uniform after a resample.
//
// STATISTICAL PROPERTY: the estimator of the marginal likelihood is unbiased and the
// filtered means are consistent (converge to the true posterior means as particles->inf);
// for finite N they carry Monte Carlo variance and a small resampling bias — NOT exact.
//
// DETERMINISM: particle j at step t draws from Rng streams derived purely from
// (seed, t, j) via Rng::split, so the whole run is reproducible and thread-count/partition
// independent.
//
// Guards: particles == 0 or observations empty => domain_error; a NaN/+Inf log-weight, or
// a step where every weight underflows to zero (non-normalizable) => undefined_value /
// domain_error; a resample failure is propagated.
[[nodiscard]] auto bootstrap_particle_filter(
    const StateSampler& initial_sampler, const TransitionSampler& propagate,
    const LogWeight& log_weight, std::span<const double> observations,
    std::uint64_t particles, std::uint64_t seed, double ess_threshold = 0.5,
    ResampleScheme scheme = ResampleScheme::systematic) -> Result<BootstrapFilterResult>;

// ---------------------------------------------------------------------------
// Variance-reduced Monte Carlo integration of a scalar f over [a, b].
//
// Each estimator returns the point estimate of ∫_a^b f(x) dx AND an empirical standard
// error (the estimated standard deviation of that point estimate). The std_error is the
// honest advertisement of remaining Monte Carlo noise: it is NOT a claim of exactness.
// ---------------------------------------------------------------------------

// An integral estimate together with its empirical standard error.
struct Estimate {
    double value;      // point estimate of ∫_a^b f
    double std_error;  // estimated standard deviation of `value` (>= 0; may be 0/NaN if
                       // fewer than 2 independent replicates are available)
};

// Plain Monte Carlo: (b - a) * mean f(U_i), U_i ~ Uniform[a, b]. Unbiased. Provided as the
// baseline against which the variance-reduced estimators below are compared at equal
// sample count. Guards: b < a or samples == 0 => domain_error.
[[nodiscard]] auto plain_estimate(std::function<double(double)> f, double a, double b,
                                  std::uint64_t samples, std::uint64_t seed) -> Result<Estimate>;

// Antithetic variates: for each of `pairs` draws U, average f at U and at its reflection
// 1 - U. Uses 2 * pairs function evaluations. Unbiased; for a MONOTONE integrand the pair
// is negatively correlated, so the variance (and std_error) is strictly below plain Monte
// Carlo at the same 2*pairs sample count. Guards: b < a or pairs == 0 => domain_error.
[[nodiscard]] auto antithetic_estimate(std::function<double(double)> f, double a, double b,
                                       std::uint64_t pairs, std::uint64_t seed)
    -> Result<Estimate>;

// Control variates: reduce variance using a `control` whose mean over Uniform[a, b] is the
// KNOWN value `control_mean` (i.e. (1/(b-a)) ∫_a^b control). The optimal coefficient
// c* = Cov(f, control)/Var(control) is ESTIMATED from the same samples, which introduces
// an O(1/samples) finite-sample bias — the estimator is CONSISTENT (asymptotically
// unbiased), not exact. When f and control are correlated the std_error falls below plain
// Monte Carlo. Guards: b < a or samples == 0 => domain_error (samples < 2 yields a valid
// value but a NaN std_error, since variance needs >= 2 points).
[[nodiscard]] auto control_variate_estimate(std::function<double(double)> f,
                                            std::function<double(double)> control,
                                            double control_mean, double a, double b,
                                            std::uint64_t samples, std::uint64_t seed)
    -> Result<Estimate>;

// Stratified sampling: partition [a, b] into `strata` equal strata and draw `per_stratum`
// uniforms within each. Unbiased; the std_error is built from the WITHIN-stratum variances
// only, so it is <= plain Monte Carlo at the same strata*per_stratum sample count (the
// between-stratum variance is removed by construction). Guards: b < a, strata == 0, or
// per_stratum == 0 => domain_error.
[[nodiscard]] auto stratified_estimate(std::function<double(double)> f, double a, double b,
                                       std::uint64_t strata, std::uint64_t per_stratum,
                                       std::uint64_t seed) -> Result<Estimate>;

}  // namespace nimblecas::smc

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::smc {

namespace {

// Re-normalise a weight vector, validating it on the railway. Returns w_i / sum(w). Empty
// => domain_error; any NaN/Inf entry => undefined_value; any negative entry or a
// non-positive total (all-zero, non-normalizable) => domain_error.
[[nodiscard]] auto normalized_weights(std::span<const double> w)
    -> Result<std::vector<double>> {
    if (w.empty()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    double total = 0.0;
    for (const double v : w) {
        if (!std::isfinite(v)) {
            return make_error<std::vector<double>>(MathError::undefined_value);
        }
        if (v < 0.0) {
            return make_error<std::vector<double>>(MathError::domain_error);
        }
        total += v;
    }
    if (!(total > 0.0)) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> out(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
        out[i] = w[i] / total;
    }
    return out;
}

// Inclusive prefix sums (empirical CDF) of a normalized weight vector.
[[nodiscard]] auto build_cdf(std::span<const double> normalized) -> std::vector<double> {
    std::vector<double> cdf(normalized.size());
    double acc = 0.0;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        acc += normalized[i];
        cdf[i] = acc;
    }
    return cdf;
}

// First index i with cdf[i] > u, clamped into [0, n) so floating drift at the tail can
// never index past the end. cdf is non-decreasing, so upper_bound is exact.
[[nodiscard]] auto cdf_pick(std::span<const double> cdf, double u) -> std::size_t {
    const auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    std::size_t idx = static_cast<std::size_t>(it - cdf.begin());
    if (idx >= cdf.size()) {
        idx = cdf.size() - 1;
    }
    return idx;
}

// Numerically stable log(sum_j exp(a_j)). Returns -inf if every entry is -inf (the
// degenerate, non-normalizable case the caller must reject).
[[nodiscard]] auto log_sum_exp(std::span<const double> a) -> double {
    double m = -std::numeric_limits<double>::infinity();
    for (const double v : a) {
        if (v > m) {
            m = v;
        }
    }
    if (!std::isfinite(m)) {
        return m;  // -inf (all entries -inf) or, defensively, +inf
    }
    double s = 0.0;
    for (const double v : a) {
        s += std::exp(v - m);
    }
    return m + std::log(s);
}

// Dispatch to the requested public resampling scheme.
[[nodiscard]] auto resample_with(ResampleScheme scheme, std::span<const double> weights,
                                 std::uint64_t seed) -> Result<std::vector<std::size_t>> {
    switch (scheme) {
        case ResampleScheme::multinomial: return multinomial_resample(weights, seed);
        case ResampleScheme::systematic:  return systematic_resample(weights, seed);
        case ResampleScheme::stratified:  return stratified_resample(weights, seed);
        case ResampleScheme::residual:    return residual_resample(weights, seed);
    }
    return make_error<std::vector<std::size_t>>(MathError::domain_error);
}

}  // namespace

auto multinomial_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>> {
    auto norm = normalized_weights(weights);
    if (!norm) {
        return make_error<std::vector<std::size_t>>(norm.error());
    }
    const auto& w = norm.value();
    const std::size_t n = w.size();
    const std::vector<double> cdf = build_cdf(w);
    const std::uint64_t key = splitmix64(seed);

    std::vector<std::size_t> parents(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Independent inverse-CDF draw per particle; index i keeps every draw a pure
        // function of its position, so the result is partition-independent.
        const double u = uniform_unit(counter_u64(key, static_cast<std::uint64_t>(i)));
        parents[i] = cdf_pick(cdf, u);
    }
    return parents;
}

auto systematic_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>> {
    auto norm = normalized_weights(weights);
    if (!norm) {
        return make_error<std::vector<std::size_t>>(norm.error());
    }
    const auto& w = norm.value();
    const std::size_t n = w.size();
    const std::vector<double> cdf = build_cdf(w);
    const std::uint64_t key = splitmix64(seed);

    // A SINGLE uniform seeds the whole comb: positions (u0 + k) / n, k in [0, n).
    const double u0 = uniform_unit(counter_u64(key, 0));
    const double inv_n = 1.0 / static_cast<double>(n);

    std::vector<std::size_t> parents(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double pos = (u0 + static_cast<double>(k)) * inv_n;
        parents[k] = cdf_pick(cdf, pos);
    }
    return parents;
}

auto stratified_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>> {
    auto norm = normalized_weights(weights);
    if (!norm) {
        return make_error<std::vector<std::size_t>>(norm.error());
    }
    const auto& w = norm.value();
    const std::size_t n = w.size();
    const std::vector<double> cdf = build_cdf(w);
    const std::uint64_t key = splitmix64(seed);
    const double inv_n = 1.0 / static_cast<double>(n);

    std::vector<std::size_t> parents(n);
    for (std::size_t k = 0; k < n; ++k) {
        // One uniform per stratum: position (k + U_k) / n keeps exactly one draw in each
        // 1/n band.
        const double uk = uniform_unit(counter_u64(key, static_cast<std::uint64_t>(k)));
        const double pos = (static_cast<double>(k) + uk) * inv_n;
        parents[k] = cdf_pick(cdf, pos);
    }
    return parents;
}

auto residual_resample(std::span<const double> weights, std::uint64_t seed)
    -> Result<std::vector<std::size_t>> {
    auto norm = normalized_weights(weights);
    if (!norm) {
        return make_error<std::vector<std::size_t>>(norm.error());
    }
    const auto& w = norm.value();
    const std::size_t n = w.size();
    const double dn = static_cast<double>(n);

    std::vector<std::size_t> parents;
    parents.reserve(n);

    // Deterministic integer allocation: floor(n w_i) guaranteed copies of particle i.
    std::vector<double> residual(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double scaled = dn * w[i];
        const double floored = std::floor(scaled);
        const auto copies = static_cast<std::size_t>(floored);
        for (std::size_t c = 0; c < copies; ++c) {
            parents.push_back(i);
        }
        residual[i] = scaled - floored;  // fractional part in [0, 1)
    }

    // Fill the remaining R = n - deterministic slots by multinomial draws on the
    // normalised residuals. R equals the (integer) sum of the fractional parts.
    const std::size_t remaining = n - parents.size();
    if (remaining > 0) {
        double res_total = 0.0;
        for (const double r : residual) {
            res_total += r;
        }
        // remaining > 0 implies some fractional part is positive, so res_total > 0.
        if (!(res_total > 0.0)) {
            return make_error<std::vector<std::size_t>>(MathError::domain_error);
        }
        std::vector<double> res_norm(n);
        for (std::size_t i = 0; i < n; ++i) {
            res_norm[i] = residual[i] / res_total;
        }
        const std::vector<double> cdf = build_cdf(res_norm);
        const std::uint64_t key = splitmix64(seed);
        for (std::size_t m = 0; m < remaining; ++m) {
            const double u = uniform_unit(counter_u64(key, static_cast<std::uint64_t>(m)));
            parents.push_back(cdf_pick(cdf, u));
        }
    }
    return parents;
}

auto effective_sample_size(std::span<const double> weights) -> Result<double> {
    if (weights.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double s1 = 0.0;
    double s2 = 0.0;
    for (const double v : weights) {
        if (!std::isfinite(v)) {
            return make_error<double>(MathError::undefined_value);
        }
        if (v < 0.0) {
            return make_error<double>(MathError::domain_error);
        }
        s1 += v;
        s2 += v * v;
    }
    if (!(s2 > 0.0)) {
        return make_error<double>(MathError::domain_error);
    }
    // General (sum w)^2 / sum w^2 == 1 / sum w^2 for a normalized vector; lies in [1, N].
    return (s1 * s1) / s2;
}

auto bootstrap_particle_filter(const StateSampler& initial_sampler,
                               const TransitionSampler& propagate, const LogWeight& log_weight,
                               std::span<const double> observations, std::uint64_t particles,
                               std::uint64_t seed, double ess_threshold, ResampleScheme scheme)
    -> Result<BootstrapFilterResult> {
    if (particles == 0 || observations.empty()) {
        return make_error<BootstrapFilterResult>(MathError::domain_error);
    }

    const std::size_t n = static_cast<std::size_t>(particles);
    const double log_uniform = -std::log(static_cast<double>(n));

    // Deterministic per-(step, particle) streams from the counter core: no shared mutable
    // RNG state, so the run is reproducible and thread-count/partition independent.
    auto root = Rng::seeded(seed);

    std::vector<double> state(n);
    std::vector<double> log_w(n, log_uniform);  // normalized log-weights, uniform to start
    {
        auto init_root = root.split(0);  // split index 0 reserved for initialisation
        for (std::size_t j = 0; j < n; ++j) {
            auto rj = init_root.split(static_cast<std::uint64_t>(j));
            state[j] = initial_sampler(rj);
        }
    }

    BootstrapFilterResult result{};
    result.filtered_means.reserve(observations.size());
    result.effective_sample_sizes.reserve(observations.size());
    result.log_marginal_likelihood = 0.0;

    std::vector<double> a(n);  // scratch: log W_prev + loglik, reused each step

    for (std::size_t t = 0; t < observations.size(); ++t) {
        // Observation steps use split index t+1 so they never collide with init's index 0.
        auto step_root = root.split(static_cast<std::uint64_t>(t) + 1);
        auto prop_root = step_root.split(0);      // propagation streams
        auto resample_root = step_root.split(1);  // resampling stream, disjoint from above

        // Propagate each particle through its own independent stream.
        for (std::size_t j = 0; j < n; ++j) {
            auto rj = prop_root.split(static_cast<std::uint64_t>(j));
            state[j] = propagate(state[j], rj);
        }

        // Unnormalised log-weights a_j = log W_prev_j + loglik_j.
        const double y = observations[t];
        for (std::size_t j = 0; j < n; ++j) {
            const double ll = log_weight(state[j], y);
            // Reject non-finite-upward log-likelihoods (NaN or +inf): they cannot be
            // normalised honestly. -inf (zero likelihood) is allowed and handled below.
            if (std::isnan(ll) || ll == std::numeric_limits<double>::infinity()) {
                return make_error<BootstrapFilterResult>(MathError::undefined_value);
            }
            a[j] = log_w[j] + ll;
        }

        const double lse = log_sum_exp(a);
        if (!std::isfinite(lse)) {
            // Every weight collapsed to zero: the observation is impossible under all
            // particles — a non-normalizable step, not a garbage number.
            return make_error<BootstrapFilterResult>(MathError::domain_error);
        }
        result.log_marginal_likelihood += lse;

        // Normalize: log W_j = a_j - lse; accumulate filtered mean and ESS.
        double mean = 0.0;
        double sum_sq = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            log_w[j] = a[j] - lse;
            const double wj = std::exp(log_w[j]);
            mean += wj * state[j];
            sum_sq += wj * wj;
        }
        const double ess = 1.0 / sum_sq;  // weights sum to 1 by construction
        result.filtered_means.push_back(mean);
        result.effective_sample_sizes.push_back(ess);

        // Adaptive resampling: only when the ESS drops below the threshold fraction of N.
        if (ess < ess_threshold * static_cast<double>(n)) {
            std::vector<double> w(n);
            for (std::size_t j = 0; j < n; ++j) {
                w[j] = std::exp(log_w[j]);
            }
            auto idx = resample_with(scheme, w, resample_root.key());
            if (!idx) {
                return make_error<BootstrapFilterResult>(idx.error());
            }
            const auto& parents = idx.value();
            std::vector<double> resampled(n);
            for (std::size_t j = 0; j < n; ++j) {
                resampled[j] = state[parents[j]];
            }
            state = std::move(resampled);
            std::fill(log_w.begin(), log_w.end(), log_uniform);  // reset to uniform weights
        }
    }

    return result;
}

// --- Variance-reduced integration ---

auto plain_estimate(std::function<double(double)> f, double a, double b, std::uint64_t samples,
                    std::uint64_t seed) -> Result<Estimate> {
    if (b < a || samples == 0) {
        return make_error<Estimate>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;
    const double dn = static_cast<double>(samples);

    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t i = 0; i < samples; ++i) {
        const double x = a + uniform_unit(counter_u64(key, i)) * width;
        const double fx = f(x);
        sum += fx;
        sum_sq += fx * fx;
    }

    const double mean = sum / dn;
    const double value = width * mean;
    // Sample variance of f (Bessel-corrected), then std error of the width-scaled mean.
    double std_error = std::numeric_limits<double>::quiet_NaN();
    if (samples >= 2) {
        const double var_f = (sum_sq - sum * sum / dn) / (dn - 1.0);
        std_error = width * std::sqrt(std::max(var_f, 0.0) / dn);
    }
    return Estimate{value, std_error};
}

auto antithetic_estimate(std::function<double(double)> f, double a, double b,
                         std::uint64_t pairs, std::uint64_t seed) -> Result<Estimate> {
    if (b < a || pairs == 0) {
        return make_error<Estimate>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;
    const double dp = static_cast<double>(pairs);

    // Each pair yields an unbiased replicate Y_k = width * (f(x(U)) + f(x(1-U)))/2.
    double sum = 0.0;
    double sum_sq = 0.0;
    for (std::uint64_t k = 0; k < pairs; ++k) {
        const double u = uniform_unit(counter_u64(key, k));
        const double x1 = a + u * width;
        const double x2 = a + (1.0 - u) * width;  // antithetic reflection
        const double yk = width * 0.5 * (f(x1) + f(x2));
        sum += yk;
        sum_sq += yk * yk;
    }

    const double value = sum / dp;
    double std_error = std::numeric_limits<double>::quiet_NaN();
    if (pairs >= 2) {
        const double var_y = (sum_sq - sum * sum / dp) / (dp - 1.0);
        std_error = std::sqrt(std::max(var_y, 0.0) / dp);
    }
    return Estimate{value, std_error};
}

auto control_variate_estimate(std::function<double(double)> f, std::function<double(double)> control,
                              double control_mean, double a, double b, std::uint64_t samples,
                              std::uint64_t seed) -> Result<Estimate> {
    if (b < a || samples == 0) {
        return make_error<Estimate>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;
    const double dn = static_cast<double>(samples);

    std::vector<double> fs(static_cast<std::size_t>(samples));
    std::vector<double> hs(static_cast<std::size_t>(samples));
    double sum_f = 0.0;
    double sum_h = 0.0;
    double sum_hh = 0.0;
    double sum_fh = 0.0;
    for (std::uint64_t i = 0; i < samples; ++i) {
        const double x = a + uniform_unit(counter_u64(key, i)) * width;
        const double fx = f(x);
        const double hx = control(x);
        fs[static_cast<std::size_t>(i)] = fx;
        hs[static_cast<std::size_t>(i)] = hx;
        sum_f += fx;
        sum_h += hx;
        sum_hh += hx * hx;
        sum_fh += fx * hx;
    }

    const double mf = sum_f / dn;
    const double mh = sum_h / dn;
    // Optimal coefficient c* = Cov(f, h) / Var(h), estimated from the sample (introduces a
    // small O(1/n) bias, so the estimator is consistent rather than exactly unbiased).
    double c_hat = 0.0;
    if (samples >= 2) {
        const double cov = (sum_fh - dn * mf * mh) / (dn - 1.0);
        const double var_h = (sum_hh - dn * mh * mh) / (dn - 1.0);
        if (var_h > 0.0) {
            c_hat = cov / var_h;
        }
    }

    // Adjusted samples g_i = f_i - c*(h_i - control_mean); their mean estimates E[f].
    double sum_g = 0.0;
    double sum_gg = 0.0;
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const double g = fs[i] - c_hat * (hs[i] - control_mean);
        sum_g += g;
        sum_gg += g * g;
    }
    const double mean_g = sum_g / dn;
    const double value = width * mean_g;
    double std_error = std::numeric_limits<double>::quiet_NaN();
    if (samples >= 2) {
        const double var_g = (sum_gg - sum_g * sum_g / dn) / (dn - 1.0);
        std_error = width * std::sqrt(std::max(var_g, 0.0) / dn);
    }
    return Estimate{value, std_error};
}

auto stratified_estimate(std::function<double(double)> f, double a, double b,
                         std::uint64_t strata, std::uint64_t per_stratum, std::uint64_t seed)
    -> Result<Estimate> {
    if (b < a || strata == 0 || per_stratum == 0) {
        return make_error<Estimate>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double stratum_width = (b - a) / static_cast<double>(strata);
    const double dm = static_cast<double>(per_stratum);

    double value = 0.0;
    double var_est = 0.0;  // variance of the point estimate (sum of within-stratum terms)
    for (std::uint64_t k = 0; k < strata; ++k) {
        const double lo = a + static_cast<double>(k) * stratum_width;
        double sum = 0.0;
        double sum_sq = 0.0;
        for (std::uint64_t m = 0; m < per_stratum; ++m) {
            const std::uint64_t idx = k * per_stratum + m;
            const double u = uniform_unit(counter_u64(key, idx));
            const double x = lo + u * stratum_width;
            const double fx = f(x);
            sum += fx;
            sum_sq += fx * fx;
        }
        const double mean_k = sum / dm;
        value += stratum_width * mean_k;  // stratum contributes width * mean_k to the integral
        if (per_stratum >= 2) {
            const double var_k = (sum_sq - sum * sum / dm) / (dm - 1.0);
            // Var of this stratum's contribution: (stratum_width)^2 * var_k / per_stratum.
            var_est += stratum_width * stratum_width * (std::max(var_k, 0.0) / dm);
        }
    }
    const double std_error =
        per_stratum >= 2 ? std::sqrt(var_est) : std::numeric_limits<double>::quiet_NaN();
    return Estimate{value, std_error};
}

}  // namespace nimblecas::smc
