// CPU micro-benchmark for nimblecas.montecarlo's serial vs parallel estimators.
// @author Olumuyiwa Oluwasanmi
//
// Times integrate / integrate_parallel and estimate_pi / estimate_pi_parallel on the same
// arguments and checks the results are bit-identical (the module's contract — the two share a
// fixed block decomposition). Not a correctness test; montecarlo_tests is. The integrand is a
// cheap pure function, so this measures the estimator machinery (counter RNG + per-block sums),
// the case least favourable to parallelism.

import std;
import nimblecas.core;
import nimblecas.montecarlo;
import nimblecas.parallel;

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
    const std::uint64_t samples = 200'000'000;
    const std::uint64_t seed = 0xC0FFEEULL;
    const std::function<double(double)> f = [](double x) { return std::exp(-x * x); };

    // NaN until a call succeeds, so a failed call can never print as "bit-identical true".
    const double unset = std::numeric_limits<double>::quiet_NaN();
    double is = unset;
    double ip = unset;
    const double t_is = timed([&] { if (auto r = nimblecas::integrate(f, 0.0, 2.0, samples, seed)) is = *r; });
    const double t_ip = timed([&] { if (auto r = nimblecas::integrate_parallel(f, 0.0, 2.0, samples, seed)) ip = *r; });
    double ps = unset;
    double pp = unset;
    const double t_ps = timed([&] { if (auto r = nimblecas::estimate_pi(samples, seed)) ps = *r; });
    const double t_pp = timed([&] { if (auto r = nimblecas::estimate_pi_parallel(samples, seed)) pp = *r; });
    const auto same = [](double x, double y) {
        return !std::isnan(x) && std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
    };

    const auto rate = [&](double t) { return static_cast<double>(samples) / t / 1e6; };
    std::cout << std::format("backend {} with {} workers, {} samples\n",
                             nimblecas::parallel::backend(),
                             nimblecas::parallel::max_concurrency(), samples);
    std::cout << std::format(
        "integrate            serial {:.3f}s ({:>8.1f} M/s)  parallel {:.3f}s ({:>8.1f} M/s)  "
        "{:.1f}x  value {:.9f}  bit-identical {}\n",
        t_is, rate(t_is), t_ip, rate(t_ip), t_is / t_ip, is, same(is, ip));
    std::cout << std::format(
        "estimate_pi          serial {:.3f}s ({:>8.1f} M/s)  parallel {:.3f}s ({:>8.1f} M/s)  "
        "{:.1f}x  value {:.9f}  bit-identical {}\n",
        t_ps, rate(t_ps), t_pp, rate(t_pp), t_ps / t_pp, ps, same(ps, pp));
    return 0;
}
