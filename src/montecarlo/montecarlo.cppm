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
// with U drawn uniformly on [a, b]. Returns domain_error if `f` is empty, b < a, either bound
// is not finite, or samples == 0 (b == a is allowed and integrates to 0); overflow if the
// finite bounds are so far apart that b − a is not finite.
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
// domain errors. PRECONDITIONS: `f` is invoked concurrently from several threads, so it must
// be safe to call concurrently (a pure function is; one mutating captured state is not). And
// the caller's floating-point environment must be the default one: the rounding mode and the
// flush-to-zero/denormals-are-zero flags are per thread, pool threads do not inherit the
// caller's, so a caller that changed them would see blocks rounded two different ways.
[[nodiscard]] auto integrate_parallel(const std::function<double(double)>& f, double a,
                                      double b, std::uint64_t samples, std::uint64_t seed)
    -> Result<double>;

// Classic dart estimate of π: draw (x, y) in [0, 1)² from two decorrelated counter draws
// per sample — counter_u64(key, 2i) and counter_u64(key, 2i+1) — count the darts with
// x² + y² < 1 (inside the unit quarter circle), and return 4 · hits / samples. Like
// `integrate`, each sample is a pure function of its index, so the estimate is
// partition-independent. Returns domain_error if samples == 0 or samples > 2^63 (sample i
// reads counters 2i and 2i+1, which would wrap and replay earlier darts beyond that).
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
// `pdf` is empty, b < a, a bound is not finite, m_bound is not a positive number (≤ 0 or
// NaN), want == 0, or max_trials == 0; overflow if b − a is not finite.
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

// Written as quotient + remainder test rather than (samples + block - 1) / block, which wraps
// for samples within a block of UINT64_MAX.
auto block_count(std::uint64_t samples) -> std::uint64_t {
    return samples / montecarlo_block + (samples % montecarlo_block != 0 ? 1U : 0U);
}

// Blocks handed to transform_index per round. The parallel entry points walk the blocks in
// rounds of this many, so their memory is bounded by the round, not by `samples` -- the serial
// entry points allocate nothing, and a huge sample count must not make the parallel ones throw
// out of the no-throw railway from a vector allocation. Adding each round's partials in order
// to one running total is the same sequence of additions as adding all partials in order.
inline constexpr std::uint64_t kBlocksPerRound = 4096;

// Sum of f over the samples of block `blk`, in index order. Shared by both entry points, so
// the per-block partials (and hence the final sum) cannot differ between them.
auto integrate_block(const std::function<double(double)>& f, std::uint64_t key, double a,
                     double width, std::uint64_t samples, std::uint64_t blk) -> double {
    const std::uint64_t begin = blk * montecarlo_block;
    // samples - begin cannot wrap (begin < samples); begin + montecarlo_block could.
    const std::uint64_t end = begin + std::min(samples - begin, montecarlo_block);
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
auto pi_block(std::uint64_t key, std::uint64_t samples, std::uint64_t blk) -> std::uint64_t {
    const std::uint64_t begin = blk * montecarlo_block;
    const std::uint64_t end = begin + std::min(samples - begin, montecarlo_block);
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

// Runs `block_fn` over every block and adds the partials, in block order, to `total`. Serial
// when `parallel` is false; otherwise rounds of kBlocksPerRound blocks go through
// transform_index (grain 1: a block is 65536 samples, coarse enough to be its own task). Either
// way the additions into `total` happen in the same order, which is the bit-identity guarantee.
template <typename T, typename BlockFn>
auto sum_blocks(std::uint64_t samples, bool parallel, T total, BlockFn block_fn) -> T {
    const std::uint64_t blocks = block_count(samples);
    if (!parallel) {
        for (std::uint64_t blk = 0; blk < blocks; ++blk) {
            total += block_fn(blk);
        }
        return total;
    }
    for (std::uint64_t first = 0; first < blocks; first += kBlocksPerRound) {
        const std::uint64_t round = std::min(blocks - first, kBlocksPerRound);
        const auto partials = parallel::transform_index(
            static_cast<std::size_t>(round),
            [&](std::size_t k) { return block_fn(first + k); }, std::size_t{1});
        for (const T p : partials) {
            total += p;
        }
    }
    return total;
}

auto integrate_impl(const std::function<double(double)>& f, double a, double b,
                    std::uint64_t samples, std::uint64_t seed, bool parallel) -> Result<double> {
    // isfinite rejects NaN as well as ±inf: a NaN bound compares false both ways and would
    // otherwise slip past `b < a` to a NaN "estimate". An empty f would throw
    // bad_function_call out of the railway on the first sample.
    if (!f || !std::isfinite(a) || !std::isfinite(b) || b < a || samples == 0) {
        return make_error<double>(MathError::domain_error);
    }
    const double width = b - a;
    if (!std::isfinite(width)) {  // e.g. [-1e308, 1e308]; same verdict as rng::uniform_double
        return make_error<double>(MathError::overflow);
    }
    const std::uint64_t key = splitmix64(seed);
    const double sum = sum_blocks(samples, parallel, 0.0, [&](std::uint64_t blk) {
        return integrate_block(f, key, a, width, samples, blk);
    });
    return width * (sum / static_cast<double>(samples));
}

auto estimate_pi_impl(std::uint64_t samples, std::uint64_t seed, bool parallel)
    -> Result<double> {
    if (samples == 0 || samples > (std::uint64_t{1} << 63U)) {
        return make_error<double>(MathError::domain_error);
    }
    const std::uint64_t key = splitmix64(seed);
    const std::uint64_t hits = sum_blocks(samples, parallel, std::uint64_t{0},
                                          [&](std::uint64_t blk) { return pi_block(key, samples, blk); });
    return 4.0 * static_cast<double>(hits) / static_cast<double>(samples);
}

}  // namespace

auto integrate(const std::function<double(double)>& f, double a, double b, std::uint64_t samples,
               std::uint64_t seed) -> Result<double> {
    return integrate_impl(f, a, b, samples, seed, /*parallel=*/false);
}

auto integrate_parallel(const std::function<double(double)>& f, double a, double b,
                        std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    return integrate_impl(f, a, b, samples, seed, /*parallel=*/true);
}

auto estimate_pi(std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    return estimate_pi_impl(samples, seed, /*parallel=*/false);
}

auto estimate_pi_parallel(std::uint64_t samples, std::uint64_t seed) -> Result<double> {
    return estimate_pi_impl(samples, seed, /*parallel=*/true);
}

auto rejection_sample(const std::function<double(double)>& pdf, double a, double b, double m_bound,
                      std::uint64_t want, std::uint64_t seed, std::uint64_t max_trials)
    -> Result<std::vector<double>> {
    // Negated comparison on m_bound so a NaN ceiling fails it, as NaN bounds fail isfinite.
    if (!pdf || !std::isfinite(a) || !std::isfinite(b) || b < a || !(m_bound > 0.0) ||
        want == 0 || max_trials == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const double width = b - a;
    if (!std::isfinite(width)) {
        return make_error<std::vector<double>>(MathError::overflow);
    }

    auto rng = Rng::seeded(seed);

    std::vector<double> accepted;
    // At most min(want, max_trials) samples can ever be accepted, but both may be huge, and a
    // reserve() of that size would throw bad_alloc/length_error out of the railway before a
    // single trial ran. Reserve at most 2^20 up front and let push_back grow past it; only a
    // run that really accepts that many samples needs the memory.
    accepted.reserve(static_cast<std::size_t>(
        std::min({want, max_trials, std::uint64_t{1} << 20U})));

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
