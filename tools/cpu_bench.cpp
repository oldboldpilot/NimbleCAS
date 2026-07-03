// CPU micro-benchmark for the SIMD polynomial batch-evaluation fast path.
// @author Olumuyiwa Oluwasanmi
//
// A profiling harness (perf / Intel VTune) for nimblecas.polynomial::evaluate_batch — the
// AVX-512 -> AVX2 -> scalar dynamic-dispatch waterfall. Not a correctness test; built as a
// plain executable (no ctest registration) purely to give the profilers a hot loop.

import std;
import nimblecas.core;
import nimblecas.polynomial;

using nimblecas::Polynomial;

auto main() -> int {
    constexpr std::size_t n = 50'000'000;
    std::vector<float> xs(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = static_cast<float>(i % 1000) * 0.001F - 0.5F;
    }
    const Polynomial p{{1, 2, 3, 4, 5}};  // 1 + 2x + 3x^2 + 4x^3 + 5x^4

    // Warm up: fault in pages and populate caches before timing.
    volatile float sink = p.evaluate_batch(xs).front();
    (void)sink;

    constexpr int iters = 10;
    const auto t0 = std::chrono::steady_clock::now();
    double checksum = 0.0;
    for (int it = 0; it < iters; ++it) {
        const std::vector<float> out = p.evaluate_batch(xs);
        checksum += out[static_cast<std::size_t>(it)];
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double melem = static_cast<double>(n) * iters / sec / 1.0e6;
    std::cout << std::format(
        "evaluate_batch: n={} iters={} time={:.3f}s throughput={:.1f} Melem/s (checksum={})\n",
        n, iters, sec, melem, checksum);
    return 0;
}
