// NimbleCAS Monte Carlo integration & rejection sampling (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// Deterministic (seeded) Monte Carlo primitives built on the parallelisable RNG
// substrate (nimblecas.rng). Two integration/sampling estimators are expressed over the
// STATELESS counter core `counter_u64(key, i)`: because sample i is a pure function of
// its index, any partition of the index range 0..N-1 across workers, concatenated in
// index order, reproduces the single-threaded estimate bit-for-bit — the estimators are
// partition-independent by construction (they are summed serially here, but the design
// admits parallel summation). `rejection_sample` instead consumes a stateful
// `Rng::seeded` trial stream and is hard-capped at `max_trials` proposals so it can never
// loop forever. There is no time/entropy seeding anywhere: equal seeds reproduce equal
// results. All failure is reported on the railway (Result<T> / MathError), never by
// throwing.

export module nimblecas.montecarlo;

import std;
import nimblecas.core;
import nimblecas.rng;

export namespace nimblecas {

// Monte Carlo estimate of the definite integral ∫_a^b f(x) dx = (b − a) · mean(f(U)),
// with U drawn uniformly on [a, b]. Returns domain_error if b < a or samples == 0
// (b == a is allowed and integrates to 0).
//
// PARTITION-INDEPENDENCE: a key is derived from `seed` via splitmix64, then sample i uses
// x_i = a + uniform_unit(counter_u64(key, i)) · (b − a). Every sample is a pure function
// of its index i alone (no running state), so splitting 0..samples-1 into any set of
// disjoint index ranges and summing them in any grouping yields the same total — the
// estimate does not depend on how (or across how many workers) the range is partitioned.
// The sum is evaluated serially here; the property is a design guarantee for parallel use.
[[nodiscard]] auto integrate(std::function<double(double)> f, double a, double b,
                             std::uint64_t samples, std::uint64_t seed) -> Result<double>;

// Classic dart estimate of π: draw (x, y) in [0, 1)² from two decorrelated counter draws
// per sample — counter_u64(key, 2i) and counter_u64(key, 2i+1) — count the darts with
// x² + y² < 1 (inside the unit quarter circle), and return 4 · hits / samples. Like
// `integrate`, each sample is a pure function of its index, so the estimate is
// partition-independent. Returns domain_error if samples == 0.
[[nodiscard]] auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double>;

// Draw up to `want` samples on [a, b] with density proportional to `pdf` using rejection
// sampling against a uniform proposal with ceiling `m_bound`: a candidate x ~ Uniform[a,b]
// is accepted when U · m_bound ≤ pdf(x) for a fresh U ~ Uniform[0,1). `pdf` need NOT be
// normalised — only its shape and the ceiling m_bound (an upper bound on pdf over [a, b])
// matter. A stateful Rng::seeded(seed) supplies the trial stream.
//
// TERMINATION GUARANTEE: the routine issues at most `max_trials` proposals and stops as
// soon as either `want` samples are accepted or the trial budget is exhausted, returning
// however many were accepted (possibly fewer than `want`, possibly empty). It therefore
// always terminates and can never spin forever, even for a pathological pdf/m_bound. The
// returned vector holds only accepted x values, each in [a, b]. Returns domain_error if
// b < a, m_bound ≤ 0, want == 0, or max_trials == 0.
[[nodiscard]] auto rejection_sample(std::function<double(double)> pdf, double a, double b,
                                    double m_bound, std::uint64_t want, std::uint64_t seed,
                                    std::uint64_t max_trials) -> Result<std::vector<double>>;

// Arithmetic mean of a sample. Returns domain_error on an empty span.
[[nodiscard]] auto sample_mean(std::span<const double> data) -> Result<double>;

// Variance of a sample. With sample == true the unbiased estimator with Bessel's (n − 1)
// correction is used and at least 2 points are required (fewer → domain_error). With
// sample == false the population variance (divide by n) is returned and a single point is
// allowed. An empty span is always a domain_error.
[[nodiscard]] auto sample_variance(std::span<const double> data, bool sample = true)
    -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto integrate(std::function<double(double)> f, double a, double b, std::uint64_t samples,
               std::uint64_t seed) -> Result<double> {
    if (b < a || samples == 0) {
        return make_error<double>(MathError::domain_error);
    }

    // Expand the seed into a stream key; each sample then reads the stateless counter core
    // at its own index, making the sum independent of any partitioning of [0, samples).
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;

    double sum = 0.0;
    for (std::uint64_t i = 0; i < samples; ++i) {
        const double x = a + uniform_unit(counter_u64(key, i)) * width;
        sum += f(x);
    }

    const double mean = sum / static_cast<double>(samples);
    return width * mean;
}

auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    if (samples == 0) {
        return make_error<double>(MathError::domain_error);
    }

    const std::uint64_t key = splitmix64(seed);
    std::uint64_t hits = 0;
    for (std::uint64_t i = 0; i < samples; ++i) {
        // Two decorrelated counter draws per sample keep x and y independent; using
        // indices 2i and 2i+1 keeps every draw a pure function of the sample index.
        const double x = uniform_unit(counter_u64(key, 2 * i));
        const double y = uniform_unit(counter_u64(key, 2 * i + 1));
        if (x * x + y * y < 1.0) {
            ++hits;
        }
    }

    return 4.0 * static_cast<double>(hits) / static_cast<double>(samples);
}

auto rejection_sample(std::function<double(double)> pdf, double a, double b, double m_bound,
                      std::uint64_t want, std::uint64_t seed, std::uint64_t max_trials)
    -> Result<std::vector<double>> {
    if (b < a || m_bound <= 0.0 || want == 0 || max_trials == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }

    auto rng = Rng::seeded(seed);
    const double width = b - a;

    std::vector<double> accepted;
    // At most min(want, max_trials) samples can ever be accepted, so reserve that — never
    // the raw `want`, which an un-clamped caller could set huge enough to make reserve()
    // throw std::bad_alloc/length_error and escape the no-throw railway contract.
    accepted.reserve(static_cast<std::size_t>(std::min(want, max_trials)));

    // Hard cap on proposals: the loop makes at most `max_trials` iterations and also exits
    // early once `want` samples are collected — so it always terminates.
    for (std::uint64_t trial = 0; trial < max_trials && accepted.size() < want; ++trial) {
        const double x = a + rng.next_unit() * width;   // proposal ~ Uniform[a, b)
        const double u = rng.next_unit();               // acceptance draw ~ Uniform[0, 1)
        if (u * m_bound <= pdf(x)) {
            accepted.push_back(x);
        }
    }

    return accepted;
}

auto sample_mean(std::span<const double> data) -> Result<double> {
    if (data.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double sum = 0.0;
    for (const double v : data) {
        sum += v;
    }
    return sum / static_cast<double>(data.size());
}

auto sample_variance(std::span<const double> data, bool sample) -> Result<double> {
    const std::size_t n = data.size();
    if (n == 0 || (sample && n < 2)) {
        return make_error<double>(MathError::domain_error);
    }

    // Two-pass computation: mean first, then the sum of squared deviations, which is more
    // numerically stable than the naïve E[x²] − E[x]² form.
    double sum = 0.0;
    for (const double v : data) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(n);

    double sq = 0.0;
    for (const double v : data) {
        const double d = v - mean;
        sq += d * d;
    }

    const double denom = sample ? static_cast<double>(n - 1) : static_cast<double>(n);
    return sq / denom;
}

}  // namespace nimblecas
