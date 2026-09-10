// NimbleCAS content-addressed distributed memoization benchmark (ROADMAP §6.2 / M7).
// @author Olumuyiwa Oluwasanmi
//
// WHAT THIS HARNESS MEASURES:
// This benchmark empirically measures the wall-clock execution time and coordination overhead
// of task deduplication (dedup_identical_tasks) and content-addressed distributed memoization
// (DistributedMemo / InProcessMemo) under SgeeDistributedExecutor across varying task granularities
// (task-ms) and duplication ratios in single-level DAG workloads.
//
// WHAT THIS HARNESS CANNOT CONCLUDE:
// Deduplication and memoization deterministically reduce the number of dispatched tasks,
// but whether that translates into a wall-clock speedup depends strictly on whether the compute
// cost of the omitted tasks exceeds the overhead of hashing, table lookup, and coordination.
// This harness measures those trade-offs directly and reports raw timings without presupposing
// or claiming a speedup. Negative results where coordination overhead exceeds task savings are
// first-class empirical findings.

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.memo_dist;

using namespace nimblecas;

namespace {

// ===========================================================================
// 1. Bench-only Op: nimblecas.bench.spin/v1 (Verbatim from M6 harness)
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
// 2. Hardware Calibration (Verbatim from M6 harness with plausibility guard)
// ===========================================================================

[[nodiscard]] auto calibrate_hardware() -> double {
    constexpr std::uint64_t test_rounds = 2'000'000;
    // Warmup
    volatile std::uint64_t dummy = spin_kernel(50'000);
    (void)dummy;

    // The sink MUST be volatile. spin_kernel is [[nodiscard]] noexcept and touches no memory,
    // so discarding its result with (void) lets -O3 delete the entire timing loop as dead code.
    const auto t0 = std::chrono::steady_clock::now();
    volatile std::uint64_t sink = spin_kernel(test_rounds);
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double rounds_per_sec = (elapsed > 0.0)
                                      ? (static_cast<double>(test_rounds) / elapsed)
                                      : 2.0e7;

    // Refuse to proceed on a physically impossible rate rather than silently mis-size the
    // workload. A benchmark that quietly calibrates wrong is worse than one that stops: it
    // still emits a table, and the table looks fine.
    constexpr double k_max_plausible_rounds_per_sec = 1.0e10;  // ~10 GHz of 128-bit modmuls
    constexpr double k_min_plausible_rounds_per_sec = 1.0e5;
    if (!std::isfinite(rounds_per_sec) || rounds_per_sec > k_max_plausible_rounds_per_sec ||
        rounds_per_sec < k_min_plausible_rounds_per_sec) {
        std::cerr << std::format(
            "CALIBRATION IMPLAUSIBLE: {:.3e} rounds/s from {} rounds in {:.9f}s.\n"
            "The timing loop was probably optimized away. Refusing to emit measurements.\n",
            rounds_per_sec, test_rounds, elapsed);
        std::exit(2);
    }

    return rounds_per_sec;
}

// ===========================================================================
// 3. System & Host Metadata
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
// 4. Statistics & Correctness Verification Helpers
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

[[nodiscard]] auto compute_min(std::span<const double> vals) -> double {
    if (vals.empty()) {
        return 0.0;
    }
    return *std::ranges::min_element(vals);
}

[[nodiscard]] auto compute_max(std::span<const double> vals) -> double {
    if (vals.empty()) {
        return 0.0;
    }
    return *std::ranges::max_element(vals);
}

[[nodiscard]] auto compute_iqr(std::vector<double> vals) -> double {
    if (vals.size() < 4) {
        return 0.0;
    }
    std::ranges::sort(vals);
    const std::size_t n = vals.size();
    const std::size_t half = n / 2;
    const std::vector<double> lower(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(half));
    const std::vector<double> upper(
        vals.begin() + static_cast<std::ptrdiff_t>((n % 2 == 0) ? half : (half + 1)),
        vals.end());
    return compute_median(upper) - compute_median(lower);
}

[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

[[nodiscard]] auto find_differing_index(std::span<const Result<Payload>> a,
                                        std::span<const Result<Payload>> b)
    -> std::optional<std::size_t> {
    if (a.size() != b.size()) {
        return 0;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!results_equal(a[i], b[i])) {
            return i;
        }
    }
    return std::nullopt;
}

// ===========================================================================
// 5. Workload Generation
// ===========================================================================

[[nodiscard]] auto build_benchmark_graph(const TaskRegistry& reg,
                                        std::size_t num_tasks,
                                        double duplicate_ratio,
                                        std::uint64_t rounds,
                                        std::uint64_t seed)
    -> Result<TaskGraph> {
    if (num_tasks == 0) {
        return TaskGraph{};
    }

    const double clamped_ratio = std::clamp(duplicate_ratio, 0.0, 1.0);
    const std::size_t num_dups = static_cast<std::size_t>(
        std::round(static_cast<double>(num_tasks) * clamped_ratio));
    const std::size_t num_unique = (num_tasks > num_dups) ? (num_tasks - num_dups) : 1;

    std::vector<std::uint64_t> group_ids(num_tasks);
    for (std::size_t i = 0; i < num_unique; ++i) {
        group_ids[i] = static_cast<std::uint64_t>(i);
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, num_unique - 1);
    for (std::size_t i = num_unique; i < num_tasks; ++i) {
        group_ids[i] = static_cast<std::uint64_t>(dist(rng));
    }

    std::ranges::shuffle(group_ids, rng);

    TaskGraph g;
    for (std::size_t i = 0; i < num_tasks; ++i) {
        auto rounds_payload = make_u64_payload(rounds);
        auto group_payload = make_u64_payload(group_ids[i]);
        auto id_res = g.add_named_task(
            reg,
            "nimblecas.bench.spin/v1",
            std::vector<Payload>{std::move(rounds_payload), std::move(group_payload)});
        if (!id_res.has_value()) {
            return make_error<TaskGraph>(id_res.error());
        }
    }

    return g;
}

// ===========================================================================
// 6. CLI Options & Parser
// ===========================================================================

struct CliOptions {
    std::size_t tasks{64};
    std::vector<double> duplicate_ratios{0.5};
    std::vector<double> task_ms_list{0.1, 1.0, 10.0, 100.0};
    std::size_t reps{7};
    std::uint64_t seed{1337ULL};
    std::size_t num_workers{4};
    std::string csv_output_path{};
    // Arm C against a DURABLE memo instead of the in-memory one, so the cost of durability is
    // measured rather than assumed. Empty = InProcessMemo.
    std::string file_memo_path{};
    bool show_help{false};
};

[[nodiscard]] auto parse_double_list(std::string_view str) -> std::vector<double> {
    std::vector<double> result;
    std::size_t start = 0;
    while (start < str.size()) {
        const auto end = str.find(',', start);
        const auto token = (end == std::string_view::npos)
                               ? str.substr(start)
                               : str.substr(start, end - start);
        if (!token.empty()) {
            result.push_back(std::stod(std::string(token)));
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return result;
}

[[nodiscard]] auto parse_cli(int argc, char* const* argv) -> Result<CliOptions> {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
            return opts;
        }
        if (arg == "--tasks" && i + 1 < argc) {
            opts.tasks = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if ((arg == "--duplicate-ratio" || arg == "--duplicate-ratios") && i + 1 < argc) {
            opts.duplicate_ratios = parse_double_list(argv[++i]);
        } else if ((arg == "--task-ms" || arg == "--tasks-ms") && i + 1 < argc) {
            opts.task_ms_list = parse_double_list(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            opts.reps = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        } else if ((arg == "--workers" || arg == "--num-workers") && i + 1 < argc) {
            opts.num_workers = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--csv" && i + 1 < argc) {
            opts.csv_output_path = argv[++i];
        } else if (arg == "--file-memo" && i + 1 < argc) {
            opts.file_memo_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return make_error<CliOptions>(MathError::syntax_error);
        }
    }
    return opts;
}

auto print_help(const char* prog) -> void {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --tasks <N>               Number of tasks per graph (default: 64)\n"
              << "  --duplicate-ratio <R>     Duplicate ratio(s), comma-separated (default: 0.5)\n"
              << "  --task-ms <list>          Task spin duration(s) in ms, comma-separated (default: 0.1,1,10,100)\n"
              << "  --reps <N>                Measured repetitions per arm per cell (default: 7)\n"
              << "  --seed <S>                PRNG seed for graph duplicate generation (default: 1337)\n"
              << "  --workers <N>             Number of workers for SgeeDistributedExecutor (default: 4)\n"
              << "  --csv <path>              Output path for raw per-repetition CSV\n"
              << "  --file-memo <path>        Arm C uses a durable FileMemo (default: in-memory)\n"
              << "  --help, -h                Show this help message\n";
}

// ===========================================================================
// 7. Benchmark Records & Summary
// ===========================================================================

struct RepetitionRecord {
    std::uint64_t seed{0};
    double task_ms{0.0};
    double duplicate_ratio{0.0};
    std::size_t tasks{0};
    std::size_t rep{0};
    std::string arm;  // "A", "B", "C"
    double wall_seconds{0.0};
    std::size_t executed{0};
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t key_mismatches{0};
};

struct CellSummary {
    double task_ms{0.0};
    double duplicate_ratio{0.0};
    std::size_t tasks{0};
    std::size_t reps{0};
    std::size_t executed_A{0};
    std::size_t executed_B{0};
    std::size_t executed_C{0};
    double median_A_s{0.0};
    double min_A_s{0.0};
    double max_A_s{0.0};
    double median_B_s{0.0};
    double min_B_s{0.0};
    double max_B_s{0.0};
    double median_C_s{0.0};
    double min_C_s{0.0};
    double max_C_s{0.0};
    double delta_B_pct{0.0};
    double delta_C_pct{0.0};
    MemoStats memo_stats{};
};

}  // namespace

auto main(int argc, char* const* argv) -> int {
    auto cli_res = parse_cli(argc, argv);
    if (!cli_res.has_value()) {
        print_help(argv[0]);
        return 1;
    }
    const CliOptions& opts = *cli_res;
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
              << " NimbleCAS M7 Distributed Memoization & Deduplication Benchmark (ROADMAP §6.2)\n"
              << " Host: " << host_name << " | Cores: " << cores << " | Commit: " << git_commit << "\n"
              << " Tasks: " << opts.tasks << " | Reps: " << opts.reps << " | Seed: " << opts.seed << "\n"
              << " Workers: " << opts.num_workers << "\n"
              << "═══════════════════════════════════════════════════════════════════════════════════════════\n";

    // Hardware Calibration
    const double cal_rounds_per_sec = calibrate_hardware();
    std::cout << std::format("Calibrated rate: {:.3e} spin rounds/second\n\n", cal_rounds_per_sec);

    std::vector<RepetitionRecord> raw_records;
    std::vector<CellSummary> cell_summaries;

    for (const double task_ms : opts.task_ms_list) {
        const double target_task_sec = task_ms / 1000.0;
        const auto task_rounds = std::max<std::uint64_t>(
            1ULL, static_cast<std::uint64_t>(std::round(cal_rounds_per_sec * target_task_sec)));

        for (const double dup_ratio : opts.duplicate_ratios) {
            auto g_res = build_benchmark_graph(reg, opts.tasks, dup_ratio, task_rounds, opts.seed);
            if (!g_res.has_value()) {
                std::cerr << "Failed to build benchmark graph: " << static_cast<int>(g_res.error()) << "\n";
                return 1;
            }
            const TaskGraph& g = *g_res;

            // Arm C's memo, pre-warmed once per cell. FileMemo when a path was given, so the
            // cost of DURABILITY is measured against the in-memory table rather than assumed.
            // A FRESH file per cell: a store carried across cells would make later cells look
            // faster for a reason that has nothing to do with the cell.
            InProcessMemo mem_memo(32, 100'000, 16u * 1024u * 1024u);
            std::unique_ptr<FileMemo> file_memo;
            if (!opts.file_memo_path.empty()) {
                std::error_code fec;
                const std::string cell_path =
                    opts.file_memo_path + std::format(".{}_{}", task_ms, dup_ratio);
                std::filesystem::remove(cell_path, fec);
                auto created = FileMemo::create(cell_path);
                if (!created.has_value()) {
                    std::cerr << "FileMemo::create failed for " << cell_path << "\n";
                    std::exit(4);
                }
                file_memo = std::move(*created);
            }
            DistributedMemo& memo = file_memo
                                        ? static_cast<DistributedMemo&>(*file_memo)
                                        : static_cast<DistributedMemo&>(mem_memo);

            // Pre-warm Arm C's memo with one run of the exact graph
            {
                FakeBrokerPort warm_port;
                InMemoryResultChannel warm_results;
                SgeeExecutorConfig warm_cfg;
                warm_cfg.with_registry(reg)
                    .with_num_workers(opts.num_workers)
                    .with_poll_interval_ms(1)
                    .with_dedup_identical_tasks(true)
                    .with_memo(memo);
                SgeeDistributedExecutor warm_exec(warm_cfg, warm_port, warm_results);
                auto warm_res = warm_exec.run(g);
                if (!warm_res.has_value()) {
                    std::cerr << "Pre-warming run failed: " << static_cast<int>(warm_res.error()) << "\n";
                    std::exit(1);
                }
            }

            std::vector<double> times_A;
            std::vector<double> times_B;
            std::vector<double> times_C;
            times_A.reserve(opts.reps);
            times_B.reserve(opts.reps);
            times_C.reserve(opts.reps);

            std::size_t exec_A = 0;
            std::size_t exec_B = 0;
            std::size_t exec_C = 0;

            // Interleaved repetitions: A, B, C, A, B, C...
            for (std::size_t r = 0; r < opts.reps; ++r) {
                // Arm A: Plain distributed executor
                FakeBrokerPort port_A;
                InMemoryResultChannel results_A;
                SgeeExecutorConfig cfg_A;
                cfg_A.with_registry(reg)
                    .with_num_workers(opts.num_workers)
                    .with_poll_interval_ms(1);
                SgeeDistributedExecutor exec_A_inst(cfg_A, port_A, results_A);
                const auto t0_A = std::chrono::steady_clock::now();
                auto res_A = exec_A_inst.run(g);
                const auto t1_A = std::chrono::steady_clock::now();
                if (!res_A.has_value()) {
                    std::cerr << "Arm A failed on rep " << r << ": " << static_cast<int>(res_A.error()) << "\n";
                    std::exit(1);
                }
                const double wall_s_A = std::chrono::duration<double>(t1_A - t0_A).count();
                times_A.push_back(wall_s_A);
                exec_A = res_A->executed;

                // Arm B: .with_dedup_identical_tasks(true)
                FakeBrokerPort port_B;
                InMemoryResultChannel results_B;
                SgeeExecutorConfig cfg_B;
                cfg_B.with_registry(reg)
                    .with_num_workers(opts.num_workers)
                    .with_poll_interval_ms(1)
                    .with_dedup_identical_tasks(true);
                SgeeDistributedExecutor exec_B_inst(cfg_B, port_B, results_B);
                const auto t0_B = std::chrono::steady_clock::now();
                auto res_B = exec_B_inst.run(g);
                const auto t1_B = std::chrono::steady_clock::now();
                if (!res_B.has_value()) {
                    std::cerr << "Arm B failed on rep " << r << ": " << static_cast<int>(res_B.error()) << "\n";
                    std::exit(1);
                }
                const double wall_s_B = std::chrono::duration<double>(t1_B - t0_B).count();
                times_B.push_back(wall_s_B);
                exec_B = res_B->executed;

                // Arm C: .with_dedup_identical_tasks(true).with_memo(memo) on pre-warmed memo
                const auto memo_stats_before = memo.stats();
                FakeBrokerPort port_C;
                InMemoryResultChannel results_C;
                SgeeExecutorConfig cfg_C;
                cfg_C.with_registry(reg)
                    .with_num_workers(opts.num_workers)
                    .with_poll_interval_ms(1)
                    .with_dedup_identical_tasks(true)
                    .with_memo(memo);
                SgeeDistributedExecutor exec_C_inst(cfg_C, port_C, results_C);
                const auto t0_C = std::chrono::steady_clock::now();
                auto res_C = exec_C_inst.run(g);
                const auto t1_C = std::chrono::steady_clock::now();
                if (!res_C.has_value()) {
                    std::cerr << "Arm C failed on rep " << r << ": " << static_cast<int>(res_C.error()) << "\n";
                    std::exit(1);
                }
                const double wall_s_C = std::chrono::duration<double>(t1_C - t0_C).count();
                times_C.push_back(wall_s_C);
                exec_C = res_C->executed;
                const auto memo_stats_after = memo.stats();

                // ── MANDATORY CORRECTNESS GATE (checked EVERY repetition) ──────────
                if (const auto diff_b = find_differing_index(res_B->outputs, res_A->outputs)) {
                    std::cerr << std::format(
                        "CORRECTNESS_VIOLATION: Arm B output differs from Arm A at index {}\n", *diff_b);
                    std::exit(3);
                }
                if (const auto diff_c = find_differing_index(res_C->outputs, res_A->outputs)) {
                    std::cerr << std::format(
                        "CORRECTNESS_VIOLATION: Arm C output differs from Arm A at index {}\n", *diff_c);
                    std::exit(3);
                }

                const std::uint64_t rep_hits = memo_stats_after.hits - memo_stats_before.hits;
                const std::uint64_t rep_misses = memo_stats_after.misses - memo_stats_before.misses;
                const std::uint64_t rep_mismatches =
                    memo_stats_after.key_mismatches - memo_stats_before.key_mismatches;

                raw_records.push_back(RepetitionRecord{
                    .seed = opts.seed,
                    .task_ms = task_ms,
                    .duplicate_ratio = dup_ratio,
                    .tasks = opts.tasks,
                    .rep = r,
                    .arm = "A",
                    .wall_seconds = wall_s_A,
                    .executed = exec_A,
                    .hits = 0,
                    .misses = 0,
                    .key_mismatches = 0});

                raw_records.push_back(RepetitionRecord{
                    .seed = opts.seed,
                    .task_ms = task_ms,
                    .duplicate_ratio = dup_ratio,
                    .tasks = opts.tasks,
                    .rep = r,
                    .arm = "B",
                    .wall_seconds = wall_s_B,
                    .executed = exec_B,
                    .hits = 0,
                    .misses = 0,
                    .key_mismatches = 0});

                raw_records.push_back(RepetitionRecord{
                    .seed = opts.seed,
                    .task_ms = task_ms,
                    .duplicate_ratio = dup_ratio,
                    .tasks = opts.tasks,
                    .rep = r,
                    .arm = "C",
                    .wall_seconds = wall_s_C,
                    .executed = exec_C,
                    .hits = rep_hits,
                    .misses = rep_misses,
                    .key_mismatches = rep_mismatches});
            }

            const double med_A = compute_median(times_A);
            const double min_A = compute_min(times_A);
            const double max_A = compute_max(times_A);

            const double med_B = compute_median(times_B);
            const double min_B = compute_min(times_B);
            const double max_B = compute_max(times_B);

            const double med_C = compute_median(times_C);
            const double min_C = compute_min(times_C);
            const double max_C = compute_max(times_C);

            const double delta_B = (med_A > 0.0) ? ((med_A - med_B) / med_A * 100.0) : 0.0;
            const double delta_C = (med_A > 0.0) ? ((med_A - med_C) / med_A * 100.0) : 0.0;

            cell_summaries.push_back(CellSummary{
                .task_ms = task_ms,
                .duplicate_ratio = dup_ratio,
                .tasks = opts.tasks,
                .reps = opts.reps,
                .executed_A = exec_A,
                .executed_B = exec_B,
                .executed_C = exec_C,
                .median_A_s = med_A,
                .min_A_s = min_A,
                .max_A_s = max_A,
                .median_B_s = med_B,
                .min_B_s = min_B,
                .max_B_s = max_B,
                .median_C_s = med_C,
                .min_C_s = min_C,
                .max_C_s = max_C,
                .delta_B_pct = delta_B,
                .delta_C_pct = delta_C,
                .memo_stats = memo.stats()});
        }
    }

    // -----------------------------------------------------------------------
    // Output Human-Readable Summary Tables
    // -----------------------------------------------------------------------
    std::cout << "\n--- SUMMARY RESULTS TABLE ---\n"
              << "| Task(ms) | DupRatio | Tasks | Reps | Arm A Median [Min, Max] (s) | Arm B Median [Min, Max] (s) | Arm C Median [Min, Max] (s) | Δ B vs A (%) | Δ C vs A (%) | Exec (A/B/C) |\n"
              << "|:---------|:---------|:------|:-----|:----------------------------|:----------------------------|:----------------------------|:-------------|:-------------|:-------------|\n";

    for (const auto& cs : cell_summaries) {
        const std::string a_str = std::format("{:.6f} [{:.6f}, {:.6f}]", cs.median_A_s, cs.min_A_s, cs.max_A_s);
        const std::string b_str = std::format("{:.6f} [{:.6f}, {:.6f}]", cs.median_B_s, cs.min_B_s, cs.max_B_s);
        const std::string c_str = std::format("{:.6f} [{:.6f}, {:.6f}]", cs.median_C_s, cs.min_C_s, cs.max_C_s);
        const std::string exec_str = std::format("{}/{}/{}", cs.executed_A, cs.executed_B, cs.executed_C);
        std::cout << std::format(
            "| {:>8.3f} | {:>8.3f} | {:>5} | {:>4} | {:<27} | {:<27} | {:<27} | {:>+11.1f}% | {:>+11.1f}% | {:<12} |\n",
            cs.task_ms, cs.duplicate_ratio, cs.tasks, cs.reps,
            a_str, b_str, c_str,
            cs.delta_B_pct, cs.delta_C_pct, exec_str);
    }

    std::cout << "\n--- MEMO TELEMETRY (ARM C) ---\n"
              << "| Task(ms) | DupRatio | Hits       | Misses     | Publishes  | KeyMismatches | Rejected   |\n"
              << "|:---------|:---------|:-----------|:-----------|:-----------|:--------------|:-----------|\n";

    for (const auto& cs : cell_summaries) {
        std::cout << std::format(
            "| {:>8.3f} | {:>8.3f} | {:>10} | {:>10} | {:>10} | {:>13} | {:>10} |\n",
            cs.task_ms, cs.duplicate_ratio,
            cs.memo_stats.hits, cs.memo_stats.misses, cs.memo_stats.publishes,
            cs.memo_stats.key_mismatches, cs.memo_stats.rejected);
    }

    // -----------------------------------------------------------------------
    // Output Raw CSV
    // -----------------------------------------------------------------------
    if (!opts.csv_output_path.empty()) {
        std::ofstream csv_file(opts.csv_output_path);
        if (csv_file.is_open()) {
            csv_file << "seed,task_ms,duplicate_ratio,tasks,rep,arm,wall_seconds,executed,hits,misses,key_mismatches\n";
            for (const auto& rec : raw_records) {
                csv_file << std::format("{},{:.4f},{:.4f},{},{},{},{:.9f},{},{},{},{}\n",
                                        rec.seed, rec.task_ms, rec.duplicate_ratio,
                                        rec.tasks, rec.rep, rec.arm,
                                        rec.wall_seconds, rec.executed,
                                        rec.hits, rec.misses, rec.key_mismatches);
            }
            std::cout << std::format("\nRaw per-repetition CSV written to: {}\n", opts.csv_output_path);
        } else {
            std::cerr << "Failed to open CSV output file: " << opts.csv_output_path << "\n";
        }
    }

    return 0;
}
