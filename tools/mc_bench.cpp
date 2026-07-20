// CPU micro-benchmark / perf harness for the Monte Carlo pricers.
// @author Olumuyiwa Oluwasanmi
//
// Gives perf/VTune a hot loop over nimblecas.pricing's reproducible Monte Carlo. A perf pass
// (Xeon Gold 6152, AVX-512) showed European MC is COMPUTE/transcendental-bound — IPC 2.09,
// L1 miss 1.4 %, ~25 % of time in scalar libm log+exp, the RNG core already AVX-512 — so unlike
// the memory-bound SIMD poly eval it scales ~linearly with cores. This harness contrasts the
// serial monte_carlo_european against the deterministic monte_carlo_european_parallel, and also
// times the path-dependent Asian (control-variate) and Longstaff-Schwartz American pricers.
// Not a correctness test — prices are printed with their sampling error to confirm they land
// near the closed-form / expected value.

import std;
import nimblecas.core;
import nimblecas.pricing;

using namespace nimblecas::pricing;

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
    const auto spec = OptionSpec{}.with_spot(100).with_strike(100).with_rate(0.05)
                          .with_volatility(0.2).with_expiry(1.0);
    const std::uint64_t seed = 0xC0FFEEULL;

    // European (antithetic): serial vs deterministic parallel. Each path is 2 std::exp + payoff
    // over a vectorised normal fill — the primary hot path.
    const std::uint64_t euro_paths = 100'000'000;
    McResult es{};
    const double ts = timed([&] { if (auto r = monte_carlo_european(spec, euro_paths, seed)) es = *r; });
    McResult ep{};
    const double tp = timed([&] { if (auto r = monte_carlo_european_parallel(spec, euro_paths, seed)) ep = *r; });
    std::cout << std::format(
        "European MC serial   : {:>12} paths          {:.3f}s  {:>8.1f} M paths/s  price {:.6f} +/- {:.6f}  (BS 10.45058)\n"
        "European MC parallel : {:>12} paths          {:.3f}s  {:>8.1f} M paths/s  price {:.6f}   ({:.1f}x vs serial; |dprice| {:.2e})\n",
        euro_paths, ts, static_cast<double>(euro_paths) / ts / 1e6, es.price, es.std_error,
        euro_paths, tp, static_cast<double>(euro_paths) / tp / 1e6, ep.price, ts / tp,
        std::abs(ep.price - es.price));

    // Asian (arithmetic average, control-variate) — steps add per-path cost.
    const std::uint64_t asian_paths = 5'000'000;
    const int asian_steps = 50;
    McResult ar{};
    const double ta = timed([&] {
        if (auto r = monte_carlo_asian(spec, asian_paths, asian_steps, seed)) ar = *r;
    });
    std::cout << std::format(
        "Asian MC             : {:>12} paths x{:>3} steps  {:.3f}s  {:>8.1f} M steps/s  price {:.6f} +/- {:.6f}\n",
        asian_paths, asian_steps, ta,
        static_cast<double>(asian_paths) * asian_steps / ta / 1e6, ar.price, ar.std_error);

    // Longstaff-Schwartz American (regression on in-the-money paths).
    const std::uint64_t ls_paths = 2'000'000;
    const int ls_steps = 50;
    McResult lr{};
    const double tl = timed([&] {
        if (auto r = longstaff_schwartz_american(spec, ls_paths, ls_steps, seed)) lr = *r;
    });
    std::cout << std::format(
        "LS American          : {:>12} paths x{:>3} steps  {:.3f}s  {:>8.1f} M steps/s  price {:.6f} +/- {:.6f}\n",
        ls_paths, ls_steps, tl,
        static_cast<double>(ls_paths) * ls_steps / tl / 1e6, lr.price, lr.std_error);
    return 0;
}
