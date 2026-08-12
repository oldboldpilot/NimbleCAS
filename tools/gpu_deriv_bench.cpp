// NimbleCAS GPU derivatives benchmark tool.
// @author Olumuyiwa Oluwasanmi
//
// Standalone benchmark harness for timing the 8 shipped GPU derivative pricing & grid sweep
// entry points in nimblecas.gpu (European/Asian/barrier Monte Carlo, first-order & extended
// Greeks, strategy payoff/P&L and futures P&L grid sweeps) against CPU references across
// realistic batch sizes.
// Emits raw wall-clock timings (median of 5 reps after 1 warmup) and throughput figures.
// Kernel-level attribution requires profiling with nsys / ncu.

import std;
import nimblecas.core;
import nimblecas.gpu;
import nimblecas.pricing;
import nimblecas.optstrat;
import nimblecas.futures;

namespace gpu = nimblecas::gpu;
namespace pricing = nimblecas::pricing;
namespace optstrat = nimblecas::optstrat;
namespace futures = nimblecas::futures;

namespace {

[[nodiscard]] auto to_spec(const gpu::BsOption& o) -> pricing::OptionSpec {
    return pricing::OptionSpec{}
        .with_spot(o.spot)
        .with_strike(o.strike)
        .with_rate(o.rate)
        .with_dividend(o.dividend)
        .with_volatility(o.volatility)
        .with_expiry(o.time)
        .with_type(o.is_call ? pricing::OptionType::call : pricing::OptionType::put);
}

template <typename F>
[[nodiscard]] auto time_median_ms(F&& fn, int reps = 5) -> double {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(ms);
    }
    std::ranges::sort(samples);
    return samples[static_cast<std::size_t>(reps / 2)];
}

[[nodiscard]] auto bench_monte_carlo_european() -> bool {
    std::cout << "--- European Monte Carlo Batch (monte_carlo_european_batch) ---\n";
    std::cout << std::format("{:>8} {:>12} {:>14} {:>12} {:>12} {:>20}\n",
                             "Options", "Paths", "Total Paths", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<std::size_t> opt_sizes = {1, 64, 1024};
    const std::vector<std::uint64_t> path_counts = {100'000, 1'000'000};
    const std::uint64_t seed = 42;

    for (const std::size_t n_opts : opt_sizes) {
        std::vector<gpu::BsOption> opts;
        opts.reserve(n_opts);
        for (std::size_t i = 0; i < n_opts; ++i) {
            const double strike = 80.0 + static_cast<double>(i % 40) * 1.0;
            opts.push_back(gpu::BsOption{100.0, strike, 0.05, 0.01, 0.2, 1.0, (i % 2 == 0)});
        }

        for (const std::uint64_t paths : path_counts) {
            // Warmup & Correctness spot-check
            auto gpu_res = gpu::monte_carlo_european_batch(opts, paths, seed);
            if (!gpu_res || gpu_res->size() != n_opts) {
                std::cerr << "ERROR: monte_carlo_european_batch returned error or invalid size\n";
                return false;
            }

            std::vector<pricing::McResult> cpu_check;
            cpu_check.reserve(n_opts);
            for (const auto& o : opts) {
                auto r = pricing::monte_carlo_european_parallel(to_spec(o), paths, seed);
                if (!r) {
                    std::cerr << "ERROR: pricing::monte_carlo_european_parallel failed\n";
                    return false;
                }
                cpu_check.push_back(*r);
            }

            const double p_gpu = (*gpu_res)[0].price;
            const double p_cpu = cpu_check[0].price;
            if (!std::isfinite(p_gpu) || std::abs(p_gpu - p_cpu) > 1e-4) {
                std::cerr << std::format("ERROR: Spot check failed for MC batch (GPU {:.6f} vs CPU {:.6f})\n",
                                         p_gpu, p_cpu);
                return false;
            }

            // Timings
            const double gpu_ms = time_median_ms([&] {
                std::ignore = gpu::monte_carlo_european_batch(opts, paths, seed);
            });

            const double cpu_ms = time_median_ms([&] {
                std::vector<pricing::McResult> out;
                out.reserve(n_opts);
                for (const auto& o : opts) {
                    out.push_back(*pricing::monte_carlo_european_parallel(to_spec(o), paths, seed));
                }
            });

            const std::uint64_t total_paths = static_cast<std::uint64_t>(n_opts) * paths;
            const double gpu_tput = (static_cast<double>(total_paths) / 1e6) / (gpu_ms / 1000.0);

            std::cout << std::format("{:>8} {:>12} {:>14} {:>12.3f} {:>12.3f} {:>14.2f} M paths/s\n",
                                     n_opts, paths, total_paths, gpu_ms, cpu_ms, gpu_tput);
        }
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_black_scholes_greeks() -> bool {
    std::cout << "--- Black-Scholes Greeks Batch (black_scholes_greeks_batch) ---\n";
    std::cout << std::format("{:>10} {:>12} {:>12} {:>20}\n",
                             "Options", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (const std::size_t n : sizes) {
        std::vector<gpu::BsOption> opts;
        opts.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double strike = 80.0 + static_cast<double>(i % 40) * 1.0;
            opts.push_back(gpu::BsOption{100.0, strike, 0.05, 0.01, 0.2, 1.0, (i % 2 == 0)});
        }

        // Warmup & Spot-check
        auto gpu_res = gpu::black_scholes_greeks_batch(opts);
        if (!gpu_res || gpu_res->size() != n) {
            std::cerr << "ERROR: black_scholes_greeks_batch failed\n";
            return false;
        }

        auto cpu_first = pricing::black_scholes_greeks(to_spec(opts[0]));
        if (!cpu_first || !std::isfinite((*gpu_res)[0].delta) ||
            std::abs((*gpu_res)[0].delta - cpu_first->delta) > 1e-7) {
            std::cerr << "ERROR: Spot check failed for BS Greeks\n";
            return false;
        }

        const double gpu_ms = time_median_ms([&] {
            std::ignore = gpu::black_scholes_greeks_batch(opts);
        });

        const double cpu_ms = time_median_ms([&] {
            std::vector<pricing::Greeks> out;
            out.reserve(n);
            for (const auto& o : opts) {
                out.push_back(*pricing::black_scholes_greeks(to_spec(o)));
            }
        });

        const double gpu_tput = (static_cast<double>(n) / 1e6) / (gpu_ms / 1000.0);
        std::cout << std::format("{:>10} {:>12.3f} {:>12.3f} {:>14.2f} M opts/s\n",
                                 n, gpu_ms, cpu_ms, gpu_tput);
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_black_scholes_extended_greeks() -> bool {
    std::cout << "--- Black-Scholes Extended Greeks Batch (black_scholes_extended_greeks_batch) ---\n";
    std::cout << std::format("{:>10} {:>12} {:>12} {:>20}\n",
                             "Options", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (const std::size_t n : sizes) {
        std::vector<gpu::BsOption> opts;
        opts.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double strike = 80.0 + static_cast<double>(i % 40) * 1.0;
            opts.push_back(gpu::BsOption{100.0, strike, 0.05, 0.01, 0.2, 1.0, (i % 2 == 0)});
        }

        // Warmup & Spot-check
        auto gpu_res = gpu::black_scholes_extended_greeks_batch(opts);
        if (!gpu_res || gpu_res->size() != n) {
            std::cerr << "ERROR: black_scholes_extended_greeks_batch failed\n";
            return false;
        }

        auto cpu_first = pricing::black_scholes_extended_greeks(to_spec(opts[0]));
        if (!cpu_first || !std::isfinite((*gpu_res)[0].vanna) ||
            std::abs((*gpu_res)[0].vanna - cpu_first->vanna) > 1e-7) {
            std::cerr << "ERROR: Spot check failed for Extended BS Greeks\n";
            return false;
        }

        const double gpu_ms = time_median_ms([&] {
            std::ignore = gpu::black_scholes_extended_greeks_batch(opts);
        });

        const double cpu_ms = time_median_ms([&] {
            std::vector<pricing::ExtendedGreeks> out;
            out.reserve(n);
            for (const auto& o : opts) {
                out.push_back(*pricing::black_scholes_extended_greeks(to_spec(o)));
            }
        });

        const double gpu_tput = (static_cast<double>(n) / 1e6) / (gpu_ms / 1000.0);
        std::cout << std::format("{:>10} {:>12.3f} {:>12.3f} {:>14.2f} M opts/s\n",
                                 n, gpu_ms, cpu_ms, gpu_tput);
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_strategy_payoff_grid() -> bool {
    std::cout << "--- Option Strategy Payoff Grid (strategy_payoff_grid) ---\n";
    std::cout << std::format("{:>12} {:>12} {:>12} {:>20}\n",
                             "Grid Points", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<optstrat::StrategyLeg> legs = {
        optstrat::StrategyLeg{optstrat::LegKind::call, 90.0, 1.0, 12.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 95.0, -1.0, 8.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 105.0, -1.0, 5.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 110.0, 1.0, 2.0}
    };

    auto strat = optstrat::OptionStrategy::create();
    for (const auto& l : legs) { std::ignore = strat.with_leg(l); }

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (const std::size_t n_grid : sizes) {
        std::vector<double> grid(n_grid);
        for (std::size_t i = 0; i < n_grid; ++i) {
            grid[i] = 50.0 + static_cast<double>(i) * (100.0 / static_cast<double>(n_grid));
        }

        // Warmup & Spot-check
        auto gpu_res = gpu::strategy_payoff_grid(legs, grid);
        if (!gpu_res || gpu_res->size() != n_grid) {
            std::cerr << "ERROR: strategy_payoff_grid failed\n";
            return false;
        }

        const double cpu_first = strat.payoff_at(grid[0]);
        if (!std::isfinite((*gpu_res)[0]) || std::abs((*gpu_res)[0] - cpu_first) > 1e-10) {
            std::cerr << "ERROR: Spot check failed for Strategy Payoff Grid\n";
            return false;
        }

        const double gpu_ms = time_median_ms([&] {
            std::ignore = gpu::strategy_payoff_grid(legs, grid);
        });

        const double cpu_ms = time_median_ms([&] {
            std::vector<double> out(n_grid);
            for (std::size_t i = 0; i < n_grid; ++i) {
                out[i] = strat.payoff_at(grid[i]);
            }
        });

        const double gpu_tput = (static_cast<double>(n_grid) / 1e6) / (gpu_ms / 1000.0);
        std::cout << std::format("{:>12} {:>12.3f} {:>12.3f} {:>14.2f} M pts/s\n",
                                 n_grid, gpu_ms, cpu_ms, gpu_tput);
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_strategy_pnl_grid() -> bool {
    std::cout << "--- Option Strategy P&L Grid (strategy_pnl_grid) ---\n";
    std::cout << std::format("{:>12} {:>12} {:>12} {:>20}\n",
                             "Grid Points", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<optstrat::StrategyLeg> legs = {
        optstrat::StrategyLeg{optstrat::LegKind::call, 90.0, 1.0, 12.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 95.0, -1.0, 8.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 105.0, -1.0, 5.0},
        optstrat::StrategyLeg{optstrat::LegKind::call, 110.0, 1.0, 2.0}
    };

    auto strat = optstrat::OptionStrategy::create();
    for (const auto& l : legs) { std::ignore = strat.with_leg(l); }

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (const std::size_t n_grid : sizes) {
        std::vector<double> grid(n_grid);
        for (std::size_t i = 0; i < n_grid; ++i) {
            grid[i] = 50.0 + static_cast<double>(i) * (100.0 / static_cast<double>(n_grid));
        }

        // Warmup & Spot-check
        auto gpu_res = gpu::strategy_pnl_grid(legs, grid);
        if (!gpu_res || gpu_res->size() != n_grid) {
            std::cerr << "ERROR: strategy_pnl_grid failed\n";
            return false;
        }

        const double cpu_first = strat.pnl_at(grid[0]);
        if (!std::isfinite((*gpu_res)[0]) || std::abs((*gpu_res)[0] - cpu_first) > 1e-10) {
            std::cerr << "ERROR: Spot check failed for Strategy P&L Grid\n";
            return false;
        }

        const double gpu_ms = time_median_ms([&] {
            std::ignore = gpu::strategy_pnl_grid(legs, grid);
        });

        const double cpu_ms = time_median_ms([&] {
            std::vector<double> out(n_grid);
            for (std::size_t i = 0; i < n_grid; ++i) {
                out[i] = strat.pnl_at(grid[i]);
            }
        });

        const double gpu_tput = (static_cast<double>(n_grid) / 1e6) / (gpu_ms / 1000.0);
        std::cout << std::format("{:>12} {:>12.3f} {:>12.3f} {:>14.2f} M pts/s\n",
                                 n_grid, gpu_ms, cpu_ms, gpu_tput);
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_futures_pnl_grid() -> bool {
    std::cout << "--- Futures Strategy P&L Grid (futures_pnl_grid) ---\n";
    std::cout << std::format("{:>12} {:>12} {:>12} {:>20}\n",
                             "Grid Points", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<futures::FuturesLeg> legs = {
        futures::FuturesLeg{"CL_L", 100.0, 2.0, 50.0, 0.01},
        futures::FuturesLeg{"CL_S", 105.0, -1.0, 50.0, 0.01}
    };

    auto strat = futures::FuturesStrategy::create();
    for (const auto& l : legs) { std::ignore = strat.with_leg(l); }

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};

    for (const std::size_t n_grid : sizes) {
        std::vector<double> grid(n_grid);
        for (std::size_t i = 0; i < n_grid; ++i) {
            grid[i] = 50.0 + static_cast<double>(i) * (100.0 / static_cast<double>(n_grid));
        }

        // Warmup & Spot-check
        auto gpu_res = gpu::futures_pnl_grid(legs, grid);
        if (!gpu_res || gpu_res->size() != n_grid) {
            std::cerr << "ERROR: futures_pnl_grid failed\n";
            return false;
        }

        const double cpu_first = strat.pnl_at_uniform(grid[0]);
        if (!std::isfinite((*gpu_res)[0]) || std::abs((*gpu_res)[0] - cpu_first) > 1e-10) {
            std::cerr << "ERROR: Spot check failed for Futures P&L Grid\n";
            return false;
        }

        const double gpu_ms = time_median_ms([&] {
            std::ignore = gpu::futures_pnl_grid(legs, grid);
        });

        const double cpu_ms = time_median_ms([&] {
            std::vector<double> out(n_grid);
            for (std::size_t i = 0; i < n_grid; ++i) {
                out[i] = strat.pnl_at_uniform(grid[i]);
            }
        });

        const double gpu_tput = (static_cast<double>(n_grid) / 1e6) / (gpu_ms / 1000.0);
        std::cout << std::format("{:>12} {:>12.3f} {:>12.3f} {:>14.2f} M pts/s\n",
                                 n_grid, gpu_ms, cpu_ms, gpu_tput);
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_monte_carlo_asian() -> bool {
    std::cout << "--- Asian Monte Carlo Batch (monte_carlo_asian_batch) ---\n";
    std::cout << std::format("{:>8} {:>10} {:>6} {:>14} {:>12} {:>12} {:>22}\n", "Options", "Paths",
                             "Steps", "Path-Steps", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<std::size_t> opt_sizes = {1, 64, 256};
    const std::vector<std::uint64_t> path_counts = {50'000, 200'000};
    const int steps = 64;
    const std::uint64_t seed = 42;

    for (const std::size_t n_opts : opt_sizes) {
        std::vector<gpu::BsOption> opts;
        opts.reserve(n_opts);
        for (std::size_t i = 0; i < n_opts; ++i) {
            const double strike = 80.0 + static_cast<double>(i % 40) * 1.0;
            opts.push_back(gpu::BsOption{100.0, strike, 0.05, 0.01, 0.2, 1.0, (i % 2 == 0)});
        }

        for (const std::uint64_t paths : path_counts) {
            auto gpu_res = gpu::monte_carlo_asian_batch(opts, paths, steps, seed);
            if (!gpu_res || gpu_res->size() != n_opts) {
                std::cerr << "ERROR: monte_carlo_asian_batch returned error or invalid size\n";
                return false;
            }
            const auto cpu0 = pricing::monte_carlo_asian(to_spec(opts[0]), paths, steps, seed, false);
            if (!cpu0) {
                std::cerr << "ERROR: pricing::monte_carlo_asian failed\n";
                return false;
            }
            const double p_gpu = (*gpu_res)[0].price;
            if (!std::isfinite(p_gpu) || std::abs(p_gpu - cpu0->price) > 1e-4) {
                std::cerr << std::format("ERROR: Asian spot check failed (GPU {:.6f} vs CPU {:.6f})\n",
                                         p_gpu, cpu0->price);
                return false;
            }

            const double gpu_ms = time_median_ms([&] {
                std::ignore = gpu::monte_carlo_asian_batch(opts, paths, steps, seed);
            });
            const double cpu_ms = time_median_ms([&] {
                for (const auto& o : opts) {
                    std::ignore = pricing::monte_carlo_asian(to_spec(o), paths, steps, seed, false);
                }
            });

            const std::uint64_t path_steps =
                static_cast<std::uint64_t>(n_opts) * paths * static_cast<std::uint64_t>(steps);
            const double gpu_tput = (static_cast<double>(path_steps) / 1e6) / (gpu_ms / 1000.0);
            std::cout << std::format("{:>8} {:>10} {:>6} {:>14} {:>12.3f} {:>12.3f} {:>14.2f} M ps/s\n",
                                     n_opts, paths, steps, path_steps, gpu_ms, cpu_ms, gpu_tput);
        }
    }
    std::cout << "\n";
    return true;
}

[[nodiscard]] auto bench_barrier_option_mc() -> bool {
    std::cout << "--- Barrier Monte Carlo Batch (barrier_option_mc_batch, down-and-out @0.8x) ---\n";
    std::cout << std::format("{:>8} {:>10} {:>6} {:>14} {:>12} {:>12} {:>22}\n", "Options", "Paths",
                             "Steps", "Path-Steps", "GPU (ms)", "CPU (ms)", "GPU Throughput");

    const std::vector<std::size_t> opt_sizes = {1, 64, 256};
    const std::vector<std::uint64_t> path_counts = {50'000, 200'000};
    const int steps = 64;
    const std::uint64_t seed = 42;
    const double barrier = 80.0;  // non-grazing (spot 100)
    const bool knock_in = false;

    for (const std::size_t n_opts : opt_sizes) {
        std::vector<gpu::BsOption> opts;
        opts.reserve(n_opts);
        for (std::size_t i = 0; i < n_opts; ++i) {
            const double strike = 95.0 + static_cast<double>(i % 20) * 1.0;
            opts.push_back(gpu::BsOption{100.0, strike, 0.05, 0.01, 0.2, 1.0, (i % 2 == 0)});
        }

        for (const std::uint64_t paths : path_counts) {
            auto gpu_res = gpu::barrier_option_mc_batch(opts, barrier, knock_in, paths, steps, seed);
            if (!gpu_res || gpu_res->size() != n_opts) {
                std::cerr << "ERROR: barrier_option_mc_batch returned error or invalid size\n";
                return false;
            }
            const auto cpu0 =
                pricing::barrier_option_mc(to_spec(opts[0]), barrier, knock_in, paths, steps, seed);
            if (!cpu0) {
                std::cerr << "ERROR: pricing::barrier_option_mc failed\n";
                return false;
            }
            const double p_gpu = (*gpu_res)[0].price;
            if (!std::isfinite(p_gpu) || std::abs(p_gpu - cpu0->price) > 1e-3) {
                std::cerr << std::format("ERROR: Barrier spot check failed (GPU {:.6f} vs CPU {:.6f})\n",
                                         p_gpu, cpu0->price);
                return false;
            }

            const double gpu_ms = time_median_ms([&] {
                std::ignore = gpu::barrier_option_mc_batch(opts, barrier, knock_in, paths, steps, seed);
            });
            const double cpu_ms = time_median_ms([&] {
                for (const auto& o : opts) {
                    std::ignore =
                        pricing::barrier_option_mc(to_spec(o), barrier, knock_in, paths, steps, seed);
                }
            });

            const std::uint64_t path_steps =
                static_cast<std::uint64_t>(n_opts) * paths * static_cast<std::uint64_t>(steps);
            const double gpu_tput = (static_cast<double>(path_steps) / 1e6) / (gpu_ms / 1000.0);
            std::cout << std::format("{:>8} {:>10} {:>6} {:>14} {:>12.3f} {:>12.3f} {:>14.2f} M ps/s\n",
                                     n_opts, paths, steps, path_steps, gpu_ms, cpu_ms, gpu_tput);
        }
    }
    std::cout << "\n";
    return true;
}

}  // namespace

auto main() -> int {
    std::cout << std::format(
        "========================================================================\n"
        " NimbleCAS GPU Derivatives Benchmark\n"
        "========================================================================\n"
        " Raw wall-clock timings (std::chrono::steady_clock, median of 5 reps after 1 warmup).\n"
        " Note: Kernel-level attribution & occupancy require profiling with nsys / ncu.\n"
        "========================================================================\n\n");

    if (!gpu::available()) {
        std::cout << std::format(
            "========================================================================\n"
            " NOTICE: No CUDA device detected — GPU columns reflect CPU-fallback timings.\n"
            " Kernel-level attribution requires nsys/ncu on a machine with a CUDA GPU.\n"
            "========================================================================\n\n");
    }

    if (!bench_monte_carlo_european()) { return 1; }
    if (!bench_monte_carlo_asian()) { return 1; }
    if (!bench_barrier_option_mc()) { return 1; }
    if (!bench_black_scholes_greeks()) { return 1; }
    if (!bench_black_scholes_extended_greeks()) { return 1; }
    if (!bench_strategy_payoff_grid()) { return 1; }
    if (!bench_strategy_pnl_grid()) { return 1; }
    if (!bench_futures_pnl_grid()) { return 1; }

    return 0;
}
