// CPU micro-benchmark for the SIMD polynomial batch-evaluation fast path.
// @author Olumuyiwa Oluwasanmi
//
// A profiling harness (perf / Intel VTune) for nimblecas.polynomial::evaluate_batch* — the
// AVX-512 -> AVX2 -> scalar dynamic-dispatch waterfall. Not a correctness test; built as a
// plain executable (no ctest registration) purely to give the profilers a hot loop. It
// contrasts the allocating evaluate_batch() against the allocation-free evaluate_batch_into()
// so the per-call-allocation cost that perf flagged (page faults / sys time) is visible.

import std;
import nimblecas.core;
import nimblecas.polynomial;

using nimblecas::Polynomial;

namespace {

template <typename F>
[[nodiscard]] auto timed(F&& fn) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

auto main() -> int {
    constexpr std::size_t n = 50'000'000;
    constexpr int iters = 10;
    std::vector<float> xs(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = static_cast<float>(i % 1000) * 0.001F - 0.5F;
    }
    const Polynomial p{{1, 2, 3, 4, 5}};  // 1 + 2x + 3x^2 + 4x^3 + 5x^4

    // Warm up: fault in xs and populate caches before timing.
    volatile float sink = p.evaluate_batch(xs).front();
    (void)sink;

    // Allocating variant: a fresh output vector every call.
    double checksum_alloc = 0.0;
    const double t_alloc = timed([&] {
        for (int it = 0; it < iters; ++it) {
            const std::vector<float> out = p.evaluate_batch(xs);
            checksum_alloc += out[static_cast<std::size_t>(it)];
        }
    });

    // Allocation-free variant: one reused output buffer.
    std::vector<float> out(n);
    double checksum_inplace = 0.0;
    const double t_inplace = timed([&] {
        for (int it = 0; it < iters; ++it) {
            static_cast<void>(p.evaluate_batch_into(xs, out));
            checksum_inplace += out[static_cast<std::size_t>(it)];
        }
    });

    // Parallel in-place variant: same allocation-free buffer, sharded across the fork-join
    // runtime. Serial evaluate_batch_into is memory-latency bound (perf: IPC 0.31, ~56% of
    // cycles stalled on L3 misses), so distributing shards across cores hides that latency.
    std::vector<float> out_par(n);
    double checksum_par = 0.0;
    const double t_par = timed([&] {
        for (int it = 0; it < iters; ++it) {
            static_cast<void>(p.evaluate_batch_parallel_into(xs, out_par));
            checksum_par += out_par[static_cast<std::size_t>(it)];
        }
    });

    const double melem = static_cast<double>(n) * iters / 1.0e6;
    std::cout << std::format(
        "evaluate_batch          (alloc):     {:.3f}s  {:>7.1f} Melem/s\n"
        "evaluate_batch_into     (in-place):  {:.3f}s  {:>7.1f} Melem/s   ({:.2f}x vs alloc)\n"
        "evaluate_batch_parallel (in-place):  {:.3f}s  {:>7.1f} Melem/s   ({:.2f}x vs serial in-place)\n"
        "checksums: {} / {} / {}\n",
        t_alloc, melem / t_alloc, t_inplace, melem / t_inplace, t_alloc / t_inplace,
        t_par, melem / t_par, t_inplace / t_par,
        checksum_alloc, checksum_inplace, checksum_par);
    return 0;
}
