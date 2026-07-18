// Throughput micro-benchmark: scalar counter_u64 loop vs the batched (AVX-512) core.
// @author Olumuyiwa Oluwasanmi
import std;
import nimblecas.rng;

using nimblecas::counter_u64;
using nimblecas::counter_u64_batch;

auto main() -> int {
    constexpr std::size_t N = 200'000'000;
    constexpr std::uint64_t key = 0xA5A5A5A5DEADBEEFULL;

    // --- Scalar ---
    {
        auto t0 = std::chrono::steady_clock::now();
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < N; ++i) {
            acc ^= counter_u64(key, i);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        std::println("scalar : {:.3f} s  {:.1f} M draws/s  (acc={:#x})",
                     s, N / s / 1e6, acc);
    }

    // --- Batched (AVX-512 when available) ---
    {
        std::vector<std::uint64_t> buf(1 << 16);
        auto t0 = std::chrono::steady_clock::now();
        std::uint64_t acc = 0;
        std::size_t done = 0;
        while (done < N) {
            const std::size_t m = std::min(buf.size(), N - done);
            counter_u64_batch(key, done, std::span<std::uint64_t>(buf).first(m));
            for (std::size_t j = 0; j < m; ++j) {
                acc ^= buf[j];
            }
            done += m;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        std::println("batched: {:.3f} s  {:.1f} M draws/s  (acc={:#x})",
                     s, N / s / 1e6, acc);
    }
    return 0;
}
