// NimbleCAS Monte Carlo integration & rejection sampling (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// Deterministic (seeded) Monte Carlo primitives built on the parallelisable RNG
// substrate (nimblecas.rng). Two integration/sampling estimators are expressed over the
// STATELESS counter core `counter_u64(key, i)`, so sample i is a pure function of its
// index. Both are evaluated over a FIXED block decomposition of 0..N-1 (block partials
// combined in index order), which is what lets `integrate_parallel` / `estimate_pi_parallel`
// run the blocks on the `parallel` runtime and still return the serial result bit-for-bit,
// at any thread count. `rejection_sample` instead consumes a stateful
// `Rng::seeded` trial stream and is hard-capped at `max_trials` proposals so it can never
// loop forever. There is no time/entropy seeding anywhere: equal seeds reproduce equal
// results. All failure is reported on the railway (Result<T> / MathError), never by
// throwing.

export module nimblecas.montecarlo;

import std;
import nimblecas.core;
import nimblecas.rng;
import nimblecas.parallel;

export namespace nimblecas {

// Monte Carlo estimate of the definite integral ∫_a^b f(x) dx = (b − a) · mean(f(U)),
// with U drawn uniformly on [a, b]. Returns domain_error if b < a or samples == 0
// (b == a is allowed and integrates to 0).
//
// REPRODUCIBILITY: a key is derived from `seed` via splitmix64, then sample i uses
// x_i = a + uniform_unit(counter_u64(key, i)) · (b − a). Every sample is a pure function
// of its index i alone (no running state). Floating-point addition is NOT associative, so
// the grouping of the sum is fixed rather than left to the scheduler: 0..samples-1 is cut
// into blocks of `montecarlo_block` samples, each block is summed in index order, and the
// block partials are added in block order. The serial and parallel entry points share that
// grouping, so they agree bit-for-bit.
[[nodiscard]] auto integrate(const std::function<double(double)>& f, double a, double b,
                             std::uint64_t samples, std::uint64_t seed) -> Result<double>;

// Samples per block of the fixed decomposition shared by the serial and parallel
// estimators. Part of the RESULT, not a tuning knob: changing it changes `integrate`'s
// low-order bits (never `estimate_pi`'s, whose integer hit count is grouping-free).
inline constexpr std::uint64_t montecarlo_block = std::uint64_t{1} << 16U;

// `integrate`, with the blocks evaluated concurrently on the `parallel` runtime (TBB on
// Linux/macOS, PPL on Windows). Returns exactly what `integrate` returns for the same
// arguments — bit-identical, independent of thread count and scheduling — and the same
// domain errors. PRECONDITION: `f` is invoked concurrently from several threads, so it must
// be safe to call concurrently (a pure function is; one mutating captured state is not).
[[nodiscard]] auto integrate_parallel(const std::function<double(double)>& f, double a,
                                      double b, std::uint64_t samples, std::uint64_t seed)
    -> Result<double>;

// Classic dart estimate of π: draw (x, y) in [0, 1)² from two decorrelated counter draws
// per sample — counter_u64(key, 2i) and counter_u64(key, 2i+1) — count the darts with
// x² + y² < 1 (inside the unit quarter circle), and return 4 · hits / samples. Like
// `integrate`, each sample is a pure function of its index, so the estimate is
// partition-independent. Returns domain_error if samples == 0.
[[nodiscard]] auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double>;

// `estimate_pi`, with the blocks counted concurrently. The hit count is an integer, so any
// grouping gives the same total: bit-identical to `estimate_pi` at any thread count.
[[nodiscard]] auto estimate_pi_parallel(std::uint64_t samples, std::uint64_t seed)
    -> Result<double>;

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
[[nodiscard]] auto rejection_sample(const std::function<double(double)>& pdf, double a, double b,
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

namespace {

auto block_count(std::uint64_t samples) -> std::size_t {
    return static_cast<std::size_t>((samples + montecarlo_block - 1) / montecarlo_block);
}

// Sum of f over the samples of block `blk`, in index order. Shared by both entry points, so
// the per-block partials (and hence the final sum) cannot differ between them.
auto integrate_block(const std::function<double(double)>& f, std::uint64_t key, double a,
                     double width, std::uint64_t samples, std::size_t blk) -> double {
    const std::uint64_t begin = static_cast<std::uint64_t>(blk) * montecarlo_block;
    const std::uint64_t end = std::min(samples, begin + montecarlo_block);
    double sum = 0.0;
    for (std::uint64_t i = begin; i < end; ++i) {
        const double x = a + uniform_unit(counter_u64(key, i)) * width;
        sum += f(x);
    }
    return sum;
}

// Darts of block `blk` landing inside the unit quarter circle. Two decorrelated counter
// draws per sample (indices 2i and 2i+1) keep x and y independent and every draw a pure
// function of the sample index.
auto pi_block(std::uint64_t key, std::uint64_t samples, std::size_t blk) -> std::uint64_t {
    const std::uint64_t begin = static_cast<std::uint64_t>(blk) * montecarlo_block;
    const std::uint64_t end = std::min(samples, begin + montecarlo_block);
    std::uint64_t hits = 0;
    for (std::uint64_t i = begin; i < end; ++i) {
        const double x = uniform_unit(counter_u64(key, 2 * i));
        const double y = uniform_unit(counter_u64(key, 2 * i + 1));
        if (x * x + y * y < 1.0) {
            ++hits;
        }
    }
    return hits;
}

// The one combine both integrate entry points end in: partials added in block order.
auto finish_integrate(std::span<const double> partials, double width, std::uint64_t samples)
    -> double {
    double sum = 0.0;
    for (const double p : partials) {
        sum += p;
    }
    return width * (sum / static_cast<double>(samples));
}

auto finish_pi(std::span<const std::uint64_t> partials, std::uint64_t samples) -> double {
    std::uint64_t hits = 0;
    for (const std::uint64_t p : partials) {
        hits += p;
    }
    return 4.0 * static_cast<double>(hits) / static_cast<double>(samples);
}

}  // namespace

auto integrate(const std::function<double(double)>& f, double a, double b, std::uint64_t samples,
               std::uint64_t seed) -> Result<double> {
    if (b < a || samples == 0) {
        return make_error<double>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;
    const std::size_t blocks = block_count(samples);
    std::vector<double> partials;
    partials.reserve(blocks);
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        partials.push_back(integrate_block(f, key, a, width, samples, blk));
    }
    return finish_integrate(partials, width, samples);
}

auto integrate_parallel(const std::function<double(double)>& f, double a, double b,
                        std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    if (b < a || samples == 0) {
        return make_error<double>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const double width = b - a;
    // grain 1: a block is 65536 evaluations of f, coarse enough that each is its own task.
    // transform_index is order-preserving, so the grain moves work between threads but never
    // changes which partial lands in which slot.
    const auto partials = parallel::transform_index(
        block_count(samples),
        [&](std::size_t blk) { return integrate_block(f, key, a, width, samples, blk); },
        std::size_t{1});
    return finish_integrate(partials, width, samples);
}

auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    if (samples == 0) {
        return make_error<double>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const std::size_t blocks = block_count(samples);
    std::vector<std::uint64_t> partials;
    partials.reserve(blocks);
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        partials.push_back(pi_block(key, samples, blk));
    }
    return finish_pi(partials, samples);
}

auto estimate_pi_parallel(std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    if (samples == 0) {
        return make_error<double>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const auto partials = parallel::transform_index(
        block_count(samples), [&](std::size_t blk) { return pi_block(key, samples, blk); },
        std::size_t{1});
    return finish_pi(partials, samples);
}

auto rejection_sample(const std::function<double(double)>& pdf, double a, double b, double m_bound,
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
