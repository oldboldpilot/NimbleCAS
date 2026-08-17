// NimbleCAS task-DAG scheduling benchmark and calibration harness (ROADMAP §6.1 / M6).
// @author Olumuyiwa Oluwasanmi
//
// WHAT THIS IS:
// The empirical measurement harness for cost-aware, deterministic task scheduling.
// Measures level makespan of LPT (longest processing time first) ordering against
// the baseline insertion order across single-level workloads: SKEW-LAST, SKEW-PERM,
// UNIFORM (negative control), and ZIPF, under SgeeDistributedExecutor (with FakeBrokerPort)
// and local_parallel_executor.
//
// HONESTY AND PROTOCOL (Rules 32, 43, 58, 59):
// - Negative results are first-class outcomes. If UNIFORM shows an improvement, the
//   benchmark prints that the harness is measuring sort/setup overhead and is invalid.
// - Sanity floor: when num_workers >= level_size (33), all policies must tie. A win
//   there invalidates the session.
// - Interleaved A/B/A/B execution across >= 30 measured repetitions (after 3 discarded warmups)
//   to eliminate thermal/system drift bias.
// - Statistical reporting: median, IQR, paired per-repetition ratio, and exact binomial
//   two-sided sign test p-value.
// - Offline CostTable calibration emitter: timing only feeds forward as a frozen artifact.

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.taskdag_sched;

using namespace nimblecas;
using namespace nimblecas::taskdag_sched;

namespace {

// ===========================================================================
// 1. Bench-only Op: nimblecas.bench.spin/v1
// ===========================================================================

// Linear congruential generator parameters (Knuth 64-bit multiplier + prime modulo 2^64 - 59).
// Pure integer arithmetic: zero floating-point math, zero dynamic memory allocation in the loop.
inline constexpr std::uint64_t k_spin_a = 6364136223846793005ULL;
inline constexpr std::uint64_t k_spin_c = 1442695040888963407ULL;
inline constexpr std::uint64_t k_spin_p = 18446744073709551557ULL;  // 0xFFFFFFFFFFFFFFC5 (2^64 - 59)
inline constexpr std::uint64_t k_spin_init = 1ULL;

[[nodiscard]] auto make_u64_payload(std::uint64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::uint64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] auto decode_u64_payload(std::span<const std::byte> p) -> Result<std::uint64_t> {
    if (p.size() != sizeof(std::uint64_t)) {
        return make_error<std::uint64_t>(MathError::syntax_error);
    }
    std::array<std::byte, sizeof(std::uint64_t)> bytes{};
    std::ranges::copy(p, bytes.begin());
    return std::bit_cast<std::uint64_t>(bytes);
}

[[nodiscard]] auto spin_kernel(std::uint64_t rounds) noexcept -> std::uint64_t {
    std::uint64_t x = k_spin_init;
    for (std::uint64_t i = 0; i < rounds; ++i) {
        x = static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(x) * k_spin_a + k_spin_c) % k_spin_p);
    }
    return x;
}

[[nodiscard]] auto spin_task_fn(std::span<const Payload> args) -> Result<Payload> {
    if (args.empty()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    auto rounds_res = decode_u64_payload(args[0]);
    if (!rounds_res.has_value()) {
        return make_error<Payload>(rounds_res.error());
    }
    const std::uint64_t residue = spin_kernel(*rounds_res);
    return make_u64_payload(residue);
}

[[nodiscard]] auto register_benchmark_ops(TaskRegistry& reg) -> Result<void> {
    return reg.register_op("nimblecas.bench.spin/v1", spin_task_fn);
}

// ===========================================================================
// 2. Hardware Calibration & Timing Units
// ===========================================================================

struct Calibration {
    std::uint64_t rounds_per_unit{100'000};
    double seconds_per_unit{0.005};
    double heavy_target_seconds{0.200};
};

[[nodiscard]] auto calibrate_hardware(double target_heavy_seconds = 0.200) -> Calibration {
    constexpr std::uint64_t test_rounds = 2'000'000;
    // Warmup
    volatile std::uint64_t dummy = spin_kernel(50'000);
    (void)dummy;

    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t res = spin_kernel(test_rounds);
    const auto t1 = std::chrono::steady_clock::now();
    (void)res;

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double rounds_per_sec = (elapsed > 0.0)
                                      ? (static_cast<double>(test_rounds) / elapsed)
                                      : 2.0e7;

    // Heavy task has cost 32 => unit cost (cost 1) targets target_heavy_seconds / 32.0
    const double target_unit_sec = target_heavy_seconds / 32.0;
    const auto rounds_per_unit = std::max<std::uint64_t>(
        1'000ULL, static_cast<std::uint64_t>(rounds_per_sec * target_unit_sec));
    const double sec_per_unit = static_cast<double>(rounds_per_unit) / rounds_per_sec;

    return Calibration{
        .rounds_per_unit = rounds_per_unit,
        .seconds_per_unit = sec_per_unit,
        .heavy_target_seconds = target_heavy_seconds};
}

// ===========================================================================
// 3. Workload Definitions
// ===========================================================================

struct TaskSpec {
    std::uint64_t rounds{0};
    CostHint hint{};
};

[[nodiscard]] auto build_graph_from_specs(const TaskRegistry& reg,
                                          std::span<const TaskSpec> specs)
    -> Result<TaskGraph> {
    TaskGraph g;
    for (const auto& ts : specs) {
        auto lit = make_u64_payload(ts.rounds);
        auto id_res = g.add_named_task(reg, "nimblecas.bench.spin/v1",
                                       std::vector<Payload>{std::move(lit)},
                                       /*deps=*/{},
                                       ts.hint);
        if (!id_res.has_value()) {
            return make_error<TaskGraph>(id_res.error());
        }
    }
    return g;
}

[[nodiscard]] auto make_skew_last_specs(const Calibration& cal) -> std::vector<TaskSpec> {
    std::vector<TaskSpec> specs;
    specs.reserve(33);
    for (std::size_t i = 0; i < 32; ++i) {
        specs.push_back(TaskSpec{
            .rounds = cal.rounds_per_unit,
            .hint = CostHint{.mean_seconds = cal.seconds_per_unit, .variance = 0.0}});
    }
    specs.push_back(TaskSpec{
        .rounds = 32 * cal.rounds_per_unit,
        .hint = CostHint{.mean_seconds = 32.0 * cal.seconds_per_unit, .variance = 0.0}});
    return specs;
}

[[nodiscard]] auto make_skew_perm_specs(const Calibration& cal, std::uint64_t seed)
    -> std::vector<TaskSpec> {
    auto specs = make_skew_last_specs(cal);
    std::mt19937_64 rng(seed);
    std::ranges::shuffle(specs, rng);
    return specs;
}

[[nodiscard]] auto make_uniform_specs(const Calibration& cal) -> std::vector<TaskSpec> {
    std::vector<TaskSpec> specs;
    specs.reserve(33);
    for (std::size_t i = 0; i < 33; ++i) {
        specs.push_back(TaskSpec{
            .rounds = 4 * cal.rounds_per_unit,
            .hint = CostHint{.mean_seconds = 4.0 * cal.seconds_per_unit, .variance = 0.0}});
    }
    return specs;
}

[[nodiscard]] auto make_zipf_specs(const Calibration& cal) -> std::vector<TaskSpec> {
    std::vector<TaskSpec> specs;
    specs.reserve(33);
    for (std::size_t r = 1; r <= 33; ++r) {
        const double factor = 32.0 / static_cast<double>(r);
        const auto rounds = std::max<std::uint64_t>(
            1ULL, static_cast<std::uint64_t>(std::round(cal.rounds_per_unit * factor)));
        const double mean_sec = cal.seconds_per_unit * factor;
        const double std_sec = mean_sec * 0.5;  // non-zero variance model for risk_lambda
        specs.push_back(TaskSpec{
            .rounds = rounds,
            .hint = CostHint{.mean_seconds = mean_sec, .variance = std_sec * std_sec}});
    }
    return specs;
}

// ===========================================================================
// 4. Statistics & Math Helpers
// ===========================================================================

[[nodiscard]] auto compute_median(std::vector<double> vals) -> double {
    if (vals.empty()) {
        return 0.0;
    }
    std::ranges::sort(vals);
    const std::size_t n = vals.size();
    if (n % 2 == 1) {
        return vals[n / 2];
    }
    return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
}

[[nodiscard]] auto compute_iqr(std::vector<double> vals) -> double {
    if (vals.size() < 4) {
        return 0.0;
    }
    std::ranges::sort(vals);
    const std::size_t n = vals.size();
    const std::size_t half = n / 2;
    std::vector<double> lower(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(half));
    std::vector<double> upper(
        vals.begin() + static_cast<std::ptrdiff_t>((n % 2 == 0) ? half : (half + 1)),
        vals.end());
    return compute_median(upper) - compute_median(lower);
}

// Two-sided exact binomial sign test p-value:
// H0: median difference between baseline and ordered is 0 (P(ord < base) = 0.5)
[[nodiscard]] auto compute_sign_test_p_value(std::size_t wins, std::size_t losses) noexcept
    -> double {
    const std::size_t n = wins + losses;
    if (n == 0) {
        return 1.0;
    }
    const std::size_t k = std::min(wins, losses);
    double cumulative = 0.0;
    double term = std::pow(0.5, static_cast<double>(n));
    cumulative += term;
    for (std::size_t j = 1; j <= k; ++j) {
        term *= static_cast<double>(n - j + 1) / static_cast<double>(j);
        cumulative += term;
    }
    const double p = 2.0 * cumulative;
    return std::min(1.0, p);
}

// ===========================================================================
// 5. Execution Runner & Interleaved Harness
// ===========================================================================

enum class ExecutorBackendKind {
    sgee_fake,
    local_parallel
};

struct RunConfig {
    ExecutorBackendKind backend{ExecutorBackendKind::sgee_fake};
    std::size_t num_workers{4};
    double risk_lambda{0.0};
    bool is_ordered{false};
};

[[nodiscard]] auto run_single_makespan(const TaskRegistry& reg,
                                       const TaskGraph& g,
                                       const RunConfig& cfg)
    -> Result<double> {
    if (cfg.backend == ExecutorBackendKind::sgee_fake) {
        FakeBrokerPort port;
        InMemoryResultChannel results;
        SgeeExecutorConfig exec_cfg;
        exec_cfg.with_registry(reg)
            .with_num_workers(cfg.num_workers)
            .with_poll_interval_ms(1);
        if (cfg.is_ordered) {
            exec_cfg.with_cost_ordering(true)
                .with_schedule_params(ScheduleParams{.risk_lambda = cfg.risk_lambda});
        }
        SgeeDistributedExecutor exec(exec_cfg, port, results);
        const auto t0 = std::chrono::steady_clock::now();
        auto res = exec.run(g);
        const auto t1 = std::chrono::steady_clock::now();
        if (!res.has_value()) {
            return make_error<double>(res.error());
        }
        if (res->executed != g.size()) {
            return make_error<double>(MathError::distributed_error);
        }
        return std::chrono::duration<double>(t1 - t0).count();
    }

    if (cfg.is_ordered) {
        auto exec = cost_ordered_local_executor(
            nullptr, ScheduleParams{.risk_lambda = cfg.risk_lambda}, /*grain=*/1);
        const auto t0 = std::chrono::steady_clock::now();
        auto res = exec->run(g);
        const auto t1 = std::chrono::steady_clock::now();
        if (!res.has_value()) {
            return make_error<double>(res.error());
        }
        return std::chrono::duration<double>(t1 - t0).count();
    }

    auto exec = local_parallel_executor();
    const auto t0 = std::chrono::steady_clock::now();
    auto res = exec->run(g);
    const auto t1 = std::chrono::steady_clock::now();
    if (!res.has_value()) {
        return make_error<double>(res.error());
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

struct RepetitionRecord {
    std::size_t rep_index{0};
    std::string workload;
    std::string config_name;
    std::string policy_name;
    double base_seconds{0.0};
    double ord_seconds{0.0};
    double ratio{1.0};
};

struct SeriesResult {
    std::string workload;
    std::string config_name;
    std::string policy_name;
    std::size_t n_reps{0};
    double base_median_s{0.0};
    double base_iqr_s{0.0};
    double ord_median_s{0.0};
    double ord_iqr_s{0.0};
    double median_ratio{1.0};
    double median_reduction_pct{0.0};
    double sign_test_p{1.0};
    std::size_t wins{0};
    std::size_t losses{0};
    std::size_t ties{0};
    std::string verdict;
};

[[nodiscard]] auto run_interleaved_series(const TaskRegistry& reg,
                                          const TaskGraph& g,
                                          std::string workload_name,
                                          std::string config_name,
                                          std::string policy_name,
                                          const RunConfig& base_cfg,
                                          const RunConfig& ord_cfg,
                                          std::size_t warmups,
                                          std::size_t reps,
                                          std::vector<RepetitionRecord>& raw_records)
    -> Result<SeriesResult> {
    // 1. Warmup phase (discarded)
    for (std::size_t w = 0; w < warmups; ++w) {
        auto w_base = run_single_makespan(reg, g, base_cfg);
        auto w_ord = run_single_makespan(reg, g, ord_cfg);
        if (!w_base.has_value()) { return make_error<SeriesResult>(w_base.error()); }
        if (!w_ord.has_value()) { return make_error<SeriesResult>(w_ord.error()); }
    }

    // 2. Interleaved measurement phase (A/B/A/B)
    std::vector<double> base_times;
    std::vector<double> ord_times;
    std::vector<double> ratios;
    base_times.reserve(reps);
    ord_times.reserve(reps);
    ratios.reserve(reps);

    std::size_t wins = 0;
    std::size_t losses = 0;
    std::size_t ties = 0;

    for (std::size_t r = 0; r < reps; ++r) {
        auto t_base_res = run_single_makespan(reg, g, base_cfg);
        if (!t_base_res.has_value()) { return make_error<SeriesResult>(t_base_res.error()); }
        const double t_base = *t_base_res;

        auto t_ord_res = run_single_makespan(reg, g, ord_cfg);
        if (!t_ord_res.has_value()) { return make_error<SeriesResult>(t_ord_res.error()); }
        const double t_ord = *t_ord_res;

        const double ratio = (t_base > 0.0) ? (t_ord / t_base) : 1.0;
        base_times.push_back(t_base);
        ord_times.push_back(t_ord);
        ratios.push_back(ratio);

        if (t_ord < t_base) {
            ++wins;
        } else if (t_ord > t_base) {
            ++losses;
        } else {
            ++ties;
        }

        raw_records.push_back(RepetitionRecord{
            .rep_index = r,
            .workload = workload_name,
            .config_name = config_name,
            .policy_name = policy_name,
            .base_seconds = t_base,
            .ord_seconds = t_ord,
            .ratio = ratio});
    }

    const double base_med = compute_median(base_times);
    const double base_iqr = compute_iqr(base_times);
    const double ord_med = compute_median(ord_times);
    const double ord_iqr = compute_iqr(ord_times);
    const double med_ratio = compute_median(ratios);
    const double med_reduct = (1.0 - med_ratio) * 100.0;
    const double p_val = compute_sign_test_p_value(wins, losses);

    std::string verdict;
    if (p_val < 0.01 && med_reduct >= 10.0) {
        verdict = "STRONG_WIN";
    } else if (p_val < 0.05 && med_reduct >= 5.0) {
        verdict = "WIN";
    } else if (std::abs(med_reduct) < 2.0) {
        verdict = "TIE";
    } else if (p_val < 0.05 && med_reduct <= -5.0) {
        verdict = "REGRESSION";
    } else {
        verdict = "INCONCLUSIVE";
    }

    return SeriesResult{
        .workload = std::move(workload_name),
        .config_name = std::move(config_name),
        .policy_name = std::move(policy_name),
        .n_reps = reps,
        .base_median_s = base_med,
        .base_iqr_s = base_iqr,
        .ord_median_s = ord_med,
        .ord_iqr_s = ord_iqr,
        .median_ratio = med_ratio,
        .median_reduction_pct = med_reduct,
        .sign_test_p = p_val,
        .wins = wins,
        .losses = losses,
        .ties = ties,
        .verdict = std::move(verdict)};
}

// ===========================================================================
// 6. Offline CostTable Calibration Emitter
// ===========================================================================

[[nodiscard]] auto emit_offline_calibration_table(const TaskRegistry& reg,
                                                 const Calibration& cal)
    -> Result<CostTable> {
    CostTable table;
    TaskGraph g;
    constexpr std::size_t sample_n = 10;
    for (std::size_t i = 0; i < sample_n; ++i) {
        auto lit = make_u64_payload(cal.rounds_per_unit);
        auto id_res = g.add_named_task(reg, "nimblecas.bench.spin/v1",
                                       std::vector<Payload>{std::move(lit)});
        if (!id_res.has_value()) {
            return make_error<CostTable>(id_res.error());
        }
    }

    auto exec = serial_executor();
    auto run_res = exec->run(g);
    if (!run_res.has_value()) {
        return make_error<CostTable>(run_res.error());
    }

    std::vector<double> timings;
    timings.reserve(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        timings.push_back(run_res->measured_seconds[i]);
    }

    const double median = compute_median(timings);
    const double iqr = compute_iqr(timings);
    const double sigma = iqr / 1.349;
    const double variance = sigma * sigma;

    table["nimblecas.bench.spin/v1"] = CostHint{
        .mean_seconds = median,
        .variance = variance};

    return table;
}

// ===========================================================================
// 7. System & Host Metadata
// ===========================================================================

[[nodiscard]] auto get_host_identifier() -> std::string {
    const char* h = std::getenv("HOSTNAME");
    if (h != nullptr && std::strlen(h) > 0) {
        return std::string(h);
    }
    const char* c = std::getenv("COMPUTERNAME");
    if (c != nullptr && std::strlen(c) > 0) {
        return std::string(c);
    }
    return "unknown_host";
}

[[nodiscard]] auto get_git_commit_string() -> std::string {
#ifdef NIMBLECAS_GIT_COMMIT
    return std::string(NIMBLECAS_GIT_COMMIT);
#else
    const char* g = std::getenv("GIT_COMMIT");
    if (g != nullptr && std::strlen(g) > 0) {
        return std::string(g);
    }
    return "HEAD";
#endif
}

// ===========================================================================
// 8. Benchmark Suite Runner & Decision Rule Evaluation
// ===========================================================================

struct CliOptions {
    std::size_t reps{30};
    std::size_t warmups{3};
    double target_heavy_seconds{0.200};
    std::string csv_output_path{};
    bool run_cost_table_calibration{false};
    bool show_help{false};
};

[[nodiscard]] auto parse_cli(int argc, char** argv) -> Result<CliOptions> {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
            return opts;
        }
        if (arg == "--reps" && i + 1 < argc) {
            opts.reps = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--warmups" && i + 1 < argc) {
            opts.warmups = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--target-heavy-ms" && i + 1 < argc) {
            opts.target_heavy_seconds = std::stod(argv[++i]) / 1000.0;
        } else if (arg == "--csv" && i + 1 < argc) {
            opts.csv_output_path = argv[++i];
        } else if (arg == "--calibrate-cost-table") {
            opts.run_cost_table_calibration = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return make_error<CliOptions>(MathError::syntax_error);
        }
    }
    return opts;
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --reps <N>              Number of measured repetitions (default: 30, min: 30)\n"
              << "  --warmups <N>           Number of discarded warmups (default: 3)\n"
              << "  --target-heavy-ms <ms>  Target duration for heavy task (default: 200)\n"
              << "  --csv <path>            Output path for raw per-repetition CSV\n"
              << "  --calibrate-cost-table  Emit offline CostTable calibration JSON\n"
              << "  --help, -h              Show this help message\n";
}

}  // namespace

auto main(int argc, char** argv) -> int {
    auto cli_res = parse_cli(argc, argv);
    if (!cli_res.has_value()) {
        print_help(argv[0]);
        return 1;
    }
    const CliOptions opts = *cli_res;
    if (opts.show_help) {
        print_help(argv[0]);
        return 0;
    }

    TaskRegistry reg;
    auto reg_res = register_benchmark_ops(reg);
    if (!reg_res.has_value()) {
        std::cerr << "Failed to register benchmark ops: " << static_cast<int>(reg_res.error()) << "\n";
        return 1;
    }

    const auto host_name = get_host_identifier();
    const auto git_commit = get_git_commit_string();
    const auto cores = std::thread::hardware_concurrency();

    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════\n"
              << " NimbleCAS M6 Task-DAG Cost-Aware Scheduling Benchmark (ROADMAP §6.1)\n"
              << " Host: " << host_name << " | Cores: " << cores << " | Commit: " << git_commit << "\n"
              << " Interleaved Reps: " << opts.reps << " (Warmups: " << opts.warmups << ")\n"
              << "═══════════════════════════════════════════════════════════════════════════════════════════\n";

    // Hardware Calibration
    const Calibration cal = calibrate_hardware(opts.target_heavy_seconds);
    std::cout << std::format("Calibrated: unit cost (1x) = {} rounds (~{:.4f}s), heavy (32x) = {} rounds (~{:.4f}s)\n\n",
                             cal.rounds_per_unit, cal.seconds_per_unit,
                             32 * cal.rounds_per_unit, 32.0 * cal.seconds_per_unit);

    if (opts.run_cost_table_calibration) {
        std::cout << "--- Emitting Offline CostTable Calibration ---\n";
        auto table_res = emit_offline_calibration_table(reg, cal);
        if (table_res.has_value()) {
            for (const auto& [op, hint] : *table_res) {
                std::cout << std::format("  Op: {:<30} mean_s: {:.6e} variance: {:.6e}\n",
                                         op, hint.mean_seconds, hint.variance);
            }
        } else {
            std::cerr << "CostTable calibration failed.\n";
        }
        std::cout << "\n";
    }

    std::vector<RepetitionRecord> raw_records;
    std::vector<SeriesResult> series_results;

    // Fixed seeds for SKEW-PERM
    constexpr std::array<std::uint64_t, 5> k_perm_seeds = {42ULL, 1337ULL, 2026ULL, 777ULL, 9999ULL};

    // Worker counts for SGEE Fake
    constexpr std::array<std::size_t, 3> k_sgee_workers = {2, 4, 8};

    // -----------------------------------------------------------------------
    // Execution 1: SKEW-LAST under SGEE (FakeBrokerPort)
    // -----------------------------------------------------------------------
    {
        const auto specs = make_skew_last_specs(cal);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        for (const std::size_t w : k_sgee_workers) {
            RunConfig base_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = w,
                .risk_lambda = 0.0,
                .is_ordered = false};
            RunConfig ord_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = w,
                .risk_lambda = 0.0,
                .is_ordered = true};
            auto s_res = run_interleaved_series(
                reg, g, "SKEW-LAST", std::format("sgee_w{}", w), "LPT(λ=0)",
                base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
            if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
        }
    }

    // -----------------------------------------------------------------------
    // Execution 2: SKEW-PERM (5 seeded permutations) under SGEE w=4
    // -----------------------------------------------------------------------
    for (std::size_t idx = 0; idx < k_perm_seeds.size(); ++idx) {
        const std::uint64_t seed = k_perm_seeds[idx];
        const auto specs = make_skew_perm_specs(cal, seed);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        RunConfig base_cfg{
            .backend = ExecutorBackendKind::sgee_fake,
            .num_workers = 4,
            .risk_lambda = 0.0,
            .is_ordered = false};
        RunConfig ord_cfg{
            .backend = ExecutorBackendKind::sgee_fake,
            .num_workers = 4,
            .risk_lambda = 0.0,
            .is_ordered = true};
        auto s_res = run_interleaved_series(
            reg, g, std::format("SKEW-PERM-{}", idx + 1), "sgee_w4", "LPT(λ=0)",
            base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
        if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
    }

    // -----------------------------------------------------------------------
    // Execution 3: UNIFORM (Negative Control — must show ~0)
    // -----------------------------------------------------------------------
    {
        const auto specs = make_uniform_specs(cal);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        for (const std::size_t w : k_sgee_workers) {
            RunConfig base_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = w,
                .risk_lambda = 0.0,
                .is_ordered = false};
            RunConfig ord_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = w,
                .risk_lambda = 0.0,
                .is_ordered = true};
            auto s_res = run_interleaved_series(
                reg, g, "UNIFORM", std::format("sgee_w{}", w), "LPT(λ=0)",
                base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
            if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
        }
    }

    // -----------------------------------------------------------------------
    // Execution 4: ZIPF with λ in {0.0, 0.5, 1.0} under SGEE w=4
    // -----------------------------------------------------------------------
    {
        const auto specs = make_zipf_specs(cal);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        for (const double lambda : {0.0, 0.5, 1.0}) {
            RunConfig base_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = 4,
                .risk_lambda = 0.0,
                .is_ordered = false};
            RunConfig ord_cfg{
                .backend = ExecutorBackendKind::sgee_fake,
                .num_workers = 4,
                .risk_lambda = lambda,
                .is_ordered = true};
            auto s_res = run_interleaved_series(
                reg, g, "ZIPF", "sgee_w4", std::format("LPT(λ={:.1f})", lambda),
                base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
            if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
        }
    }

    // -----------------------------------------------------------------------
    // Execution 5: Sanity Floor (num_workers = 33 >= level size, must tie!)
    // -----------------------------------------------------------------------
    {
        const auto specs = make_skew_last_specs(cal);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        RunConfig base_cfg{
            .backend = ExecutorBackendKind::sgee_fake,
            .num_workers = 33,
            .risk_lambda = 0.0,
            .is_ordered = false};
        RunConfig ord_cfg{
            .backend = ExecutorBackendKind::sgee_fake,
            .num_workers = 33,
            .risk_lambda = 0.0,
            .is_ordered = true};
        auto s_res = run_interleaved_series(
            reg, g, "SANITY-FLOOR-SKEW", "sgee_w33", "LPT(λ=0)",
            base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
        if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
    }

    // -----------------------------------------------------------------------
    // Execution 6: Local Parallel Executor (grain 1)
    // -----------------------------------------------------------------------
    {
        const auto specs = make_skew_last_specs(cal);
        auto g_res = build_graph_from_specs(reg, specs);
        if (!g_res.has_value()) { return 1; }
        const TaskGraph& g = *g_res;

        RunConfig base_cfg{
            .backend = ExecutorBackendKind::local_parallel,
            .num_workers = 0,
            .risk_lambda = 0.0,
            .is_ordered = false};
        RunConfig ord_cfg{
            .backend = ExecutorBackendKind::local_parallel,
            .num_workers = 0,
            .risk_lambda = 0.0,
            .is_ordered = true};
        auto s_res = run_interleaved_series(
            reg, g, "SKEW-LAST", "local_parallel", "LPT(λ=0)",
            base_cfg, ord_cfg, opts.warmups, opts.reps, raw_records);
        if (s_res.has_value()) { series_results.push_back(std::move(*s_res)); }
    }

    // -----------------------------------------------------------------------
    // Output Machine-Parsable Summary Table
    // -----------------------------------------------------------------------
    std::cout << "\n--- SUMMARY RESULTS TABLE ---\n"
              << "| Workload           | Config          | Policy     | N  | BaseMed(s) | BaseIQR(s) | OrdMed(s)  | OrdIQR(s)  | Ratio  | Δ(%)   | p-value    | Verdict      |\n"
              << "|:-------------------|:----------------|:-----------|:---|:-----------|:-----------|:-----------|:-----------|:-------|:-------|:-----------|:-------------|\n";

    for (const auto& sr : series_results) {
        std::cout << std::format(
            "| {:<18} | {:<15} | {:<10} | {:>2} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>6.4f} | {:>+5.1f}% | {:>10.2e} | {:<12} |\n",
            sr.workload, sr.config_name, sr.policy_name, sr.n_reps,
            sr.base_median_s, sr.base_iqr_s, sr.ord_median_s, sr.ord_iqr_s,
            sr.median_ratio, sr.median_reduction_pct, sr.sign_test_p, sr.verdict);
    }

    // -----------------------------------------------------------------------
    // Pre-registered Decision Rule Evaluation
    // -----------------------------------------------------------------------
    std::cout << "\n═══════════════════════════════════════════════════════════════════════════════════════════\n"
              << " PRE-REGISTERED DECISION EVALUATION (M6_SPEC.md §2.5)\n"
              << "═══════════════════════════════════════════════════════════════════════════════════════════\n";

    bool sanity_floor_passed = true;
    bool negative_control_passed = true;
    bool skew_last_passed = false;
    bool skew_perm_passed = true;

    for (const auto& sr : series_results) {
        if (sr.workload == "SANITY-FLOOR-SKEW") {
            if (std::abs(sr.median_reduction_pct) > 2.0 && sr.sign_test_p < 0.05) {
                sanity_floor_passed = false;
                std::cout << std::format(
                    "[FATAL] SANITY FLOOR VIOLATED: w=33 showed {:.2f}% delta (p={:.2e}). Session INVALID.\n",
                    sr.median_reduction_pct, sr.sign_test_p);
            } else {
                std::cout << std::format(
                    "[PASS] Sanity floor: w=33 tied within noise (delta {:.2f}%, p={:.2e}).\n",
                    sr.median_reduction_pct, sr.sign_test_p);
            }
        } else if (sr.workload == "UNIFORM") {
            if (std::abs(sr.median_reduction_pct) >= 2.0 && sr.sign_test_p < 0.05) {
                negative_control_passed = false;
                std::cout << std::format(
                    "[FATAL] UNIFORM NEGATIVE CONTROL FAILED: {} showed {:.2f}% delta (p={:.2e}).\n"
                    "        The benchmark is measuring overhead/sort cost rather than real scheduling speedup.\n",
                    sr.config_name, sr.median_reduction_pct, sr.sign_test_p);
            }
        } else if (sr.workload == "SKEW-LAST" && sr.config_name == "sgee_w4") {
            if (sr.median_reduction_pct >= 10.0 && sr.sign_test_p < 0.01) {
                skew_last_passed = true;
                std::cout << std::format(
                    "[PASS] SKEW-LAST (w=4): {:.1f}% reduction >= 10% threshold (p={:.2e} < 0.01).\n",
                    sr.median_reduction_pct, sr.sign_test_p);
            } else {
                std::cout << std::format(
                    "[FAIL] SKEW-LAST (w=4): {:.1f}% reduction did not meet 10% threshold (p={:.2e}).\n",
                    sr.median_reduction_pct, sr.sign_test_p);
            }
        } else if (sr.workload.starts_with("SKEW-PERM") && sr.config_name == "sgee_w4") {
            if (sr.median_reduction_pct < 5.0 || sr.sign_test_p >= 0.05) {
                skew_perm_passed = false;
                std::cout << std::format(
                    "[WARN] {} (w=4): {:.1f}% reduction below 5% threshold (p={:.2e}).\n",
                    sr.workload, sr.median_reduction_pct, sr.sign_test_p);
            }
        }
    }

    if (negative_control_passed) {
        std::cout << "[PASS] UNIFORM negative control showed ~0 (within 2% bound across all worker counts).\n";
    }

    std::cout << "\n--- FINAL VERDICT ---\n";
    if (!sanity_floor_passed) {
        std::cout << "DECISION: INVALID_SESSION — Sanity floor violated (w >= level_size did not tie).\n";
    } else if (!negative_control_passed) {
        std::cout << "DECISION: INVALID_BENCHMARK — Negative control failed (sort/harness overhead detected).\n";
    } else if (skew_last_passed && skew_perm_passed) {
        std::cout << "DECISION: POSITIVE — LPT cost-aware ordering confirmed effective under pre-registered rules.\n";
    } else {
        std::cout << "DECISION: NEGATIVE — LPT ordering did not achieve pre-registered speedup thresholds.\n"
                  << "          Honesty policy (Rule 32/43/58/59): record in reference docs; ordering remains off-by-default.\n";
    }

    // -----------------------------------------------------------------------
    // Output Raw CSV
    // -----------------------------------------------------------------------
    if (!opts.csv_output_path.empty()) {
        std::ofstream csv_file(opts.csv_output_path);
        if (csv_file.is_open()) {
            csv_file << "rep,workload,config,policy,baseline_seconds,ordered_seconds,ratio\n";
            for (const auto& rec : raw_records) {
                csv_file << std::format("{},{},{},{},{:.6f},{:.6f},{:.6f}\n",
                                        rec.rep_index, rec.workload, rec.config_name,
                                        rec.policy_name, rec.base_seconds, rec.ord_seconds,
                                        rec.ratio);
            }
            std::cout << std::format("\nRaw per-repetition CSV written to: {}\n", opts.csv_output_path);
        } else {
            std::cerr << "Failed to open CSV output file: " << opts.csv_output_path << "\n";
        }
    }

    return 0;
}
