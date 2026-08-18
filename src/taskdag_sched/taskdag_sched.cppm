// NimbleCAS cost-aware task scheduling module: deterministic wavefront ordering
// and cost resolution for TaskGraph execution.
// @author Olumuyiwa Oluwasanmi
//
// WHAT THIS IS. A deterministic cost-resolution and task-ordering layer built on top of
// nimblecas.taskdag. Within each wavefront level of a TaskGraph, tasks are independent by
// construction and may be scheduled in an order determined by execution-time estimates.
//
// THREE-TIER COST RESOLUTION:
// 1. Caller-supplied per-task CostHint (if known: finite, mean > 0.0).
// 2. Frozen CostTable per-OpId entry (for named tasks when per-task hint is unknown).
// 3. Unknown (effective cost = 0.0, sorting last in ascending TaskId issuance order).
// Closure tasks (empty OpId) never match the CostTable.
//
// ORDERING POLICY:
// Longest-expected-processing-time-first (LPT) with an optional variance risk margin:
//   effective_cost = mean + lambda * sqrt(variance)  (lambda >= 0, default 0.0)
// Tasks are sorted by (effective_cost descending, TaskId ascending).
//
// ── HONESTY BOUNDARY (Rule 32) ───────────────────────────────────────────────────────────
// A cost model is a HINT about TIME (wall-clock scheduling order), never about RESULTS.
// A wrong, negative, NaN, infinite, or absent estimate changes only the order in which
// independent tasks within a wavefront level are executed, never what is computed.
// TaskRunResult::outputs is guaranteed bit-identical to serial_executor() under every
// policy and every cost input.
//
// DETERMINISM:
// Given the same TaskGraph, CostTable, and ScheduleParams, level_order() produces the
// exact same order across runs and platforms. Ties are broken strictly by ascending TaskId
// (issuance order). Costs are frozen inputs (from offline calibration), never dynamic
// feedback from live clocks inside executors.
//
// UNMEASURED PERFORMANCE:
// Whether cost-aware ordering yields any wall-clock reduction on a given hardware/workload
// configuration is unmeasured pending execution of the pre-registered benchmark harness
// (ROADMAP §6.1 / M6_SPEC §2.5). On uniform-cost levels or hint-free graphs, ordering is a
// strict no-op.

module;
#include <cassert>

export module nimblecas.taskdag_sched;

import std;
import nimblecas.core;
import nimblecas.parallel;
import nimblecas.taskdag;

export namespace nimblecas {

// Frozen table mapping registered OpId strings to offline-calibrated CostHints.
// Deterministic ordered map (std::less<>) matching TaskRegistry.
using CostTable = std::map<OpId, CostHint, std::less<>>;

// Scheduling configuration parameters.
struct ScheduleParams {
    // Risk hedge parameter for variance: effective_cost = mean + risk_lambda * sqrt(var).
    // Defaults to 0.0 (pure LPT). Negative/non-finite values are sanitized to 0.0.
    double risk_lambda{0.0};
};

// Target device affinity category for a registered operation.
enum class Affinity : std::uint8_t {
    cpu_only = 0,
    gpu_only = 1,
    hybrid = 2
};

// Frozen table mapping OpId strings to target device affinity.
using AffinityTable = std::map<OpId, Affinity, std::less<>>;

// Sanitizes a CostHint to prevent non-finite/hostile values from violating strict weak ordering.
// - Non-finite or non-positive mean_seconds (<= 0.0) -> unknown ({0.0, 0.0}).
// - Finite positive mean clamped to [1e-9, 1e9] seconds.
// - Non-finite or negative variance (< 0.0) -> 0.0.
// - Finite non-negative variance clamped to [0.0, 1e18] seconds^2.
[[nodiscard]] auto sanitize(const CostHint& hint) noexcept -> CostHint;

// Resolves the effective scheduling cost for a task according to the three-tier precedence:
// 1. Caller per-task hint (if known after sanitization).
// 2. Frozen CostTable lookup by `op` (if task hint is unknown and `op` is non-empty).
// 3. Unknown (0.0).
// Returns a finite non-negative cost: 0.0 if unknown, else mean + lambda * sqrt(variance).
[[nodiscard]] auto effective_cost(const CostHint& task_hint, std::string_view op,
                                  const CostTable* table, const ScheduleParams& params) noexcept
    -> double;

// Computes the deterministic execution order for a wavefront level of tasks.
// Sorts by (effective_cost descending, TaskId ascending).
// Strict no-op (preserves original level order) if all tasks are hint-free / unknown.
[[nodiscard]] auto level_order(const TaskGraph& g, std::span<const TaskId> level,
                               const CostTable* table, const ScheduleParams& params)
    -> std::vector<TaskId>;

// Deterministic mapping from Affinity enum to SGEE wire placement tag:
// - cpu_only -> 1 (SgeePlacement::cpu)
// - gpu_only -> 2 (SgeePlacement::gpu)
// - hybrid   -> 1 (SgeePlacement::cpu; hybrid resolves to cpu until a GPU task worker exists)
// Note: returns std::uint8_t to keep module imports acyclic (nimblecas.taskdag_sgee imports
// nimblecas.taskdag_sched).
[[nodiscard]] auto to_placement(Affinity a) noexcept -> std::uint8_t;

// Wavefront local executor that orders independent tasks within each level by effective cost
// before fanning out across nimblecas.parallel::transform_index.
// Produces outputs bit-identical to serial_executor().
[[nodiscard]] auto cost_ordered_local_executor(const CostTable* table = nullptr,
                                               ScheduleParams params = {},
                                               std::size_t grain = 1) -> std::unique_ptr<Executor>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Outcome of running or poisoning a single task.
struct TaskOutcome {
    Result<Payload> output;
    std::optional<std::size_t> poison_origin;
    double seconds{0.0};
    bool executed{false};
};

// Pure function running task `id` or propagating error poisoning from failing ancestors.
[[nodiscard]] auto run_or_poison(const TaskGraph& g, TaskId id,
                                 std::span<const Result<Payload>> outputs,
                                 std::span<const std::optional<std::size_t>> origins)
    -> TaskOutcome {
    const std::span<const TaskId> task_deps = g.deps(id);

    // Resolve lowest-topological-index failing ancestor.
    std::optional<std::size_t> best_origin;
    for (const TaskId d : task_deps) {
        if (!outputs[d.value].has_value()) {
            const std::size_t origin = origins[d.value].value();
            if (!best_origin.has_value() || origin < *best_origin) {
                best_origin = origin;
            }
        }
    }
    if (best_origin.has_value()) {
        return TaskOutcome{.output = make_error<Payload>(outputs[*best_origin].error()),
                           .poison_origin = best_origin,
                           .seconds = 0.0,
                           .executed = false};
    }

    // Pass bound literals first, then parent outputs in declared dependency order.
    const std::span<const Payload> lits = g.literals(id);
    std::vector<Payload> args;
    args.reserve(lits.size() + task_deps.size());
    for (const Payload& lit : lits) {
        args.push_back(lit);
    }
    for (const TaskId d : task_deps) {
        args.push_back(outputs[d.value].value());
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto result = g.invoke(id, args);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    if (!result.has_value()) {
        return TaskOutcome{.output = std::move(result),
                           .poison_origin = id.value,
                           .seconds = seconds,
                           .executed = true};
    }
    return TaskOutcome{.output = std::move(result),
                       .poison_origin = std::nullopt,
                       .seconds = seconds,
                       .executed = true};
}

class CostOrderedLocalExecutor final : public Executor {
public:
    explicit CostOrderedLocalExecutor(const CostTable* table, ScheduleParams params,
                                      std::size_t grain)
        : table_(table), sched_(params), grain_(grain == 0 ? 1 : grain) {}

    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "cost_ordered_local";
    }

    [[nodiscard]] auto run(const TaskGraph& g) -> Result<TaskRunResult> override {
        const std::size_t n = g.size();
        std::vector<Result<Payload>> outputs(n);
        std::vector<std::optional<std::size_t>> origins(n);
        std::vector<double> seconds(n, 0.0);
        std::size_t executed = 0;

        for (std::size_t lvl = 0; lvl < g.num_levels(); ++lvl) {
            const std::span<const TaskId> raw_level = g.level(lvl);
            const std::vector<TaskId> ordered_ids = level_order(g, raw_level, table_, sched_);

            std::vector<TaskOutcome> outcomes =
                parallel::transform_index(ordered_ids.size(), [&](std::size_t i) {
                    return run_or_poison(g, ordered_ids[i], outputs, origins);
                }, grain_);

            for (std::size_t i = 0; i < ordered_ids.size(); ++i) {
                const TaskId id = ordered_ids[i];
                if (outcomes[i].executed) {
                    ++executed;
                }
                outputs[id.value] = std::move(outcomes[i].output);
                origins[id.value] = outcomes[i].poison_origin;
                seconds[id.value] = outcomes[i].seconds;
            }
        }
        return TaskRunResult{.outputs = std::move(outputs),
                             .measured_seconds = std::move(seconds),
                             .executed = executed};
    }

private:
    const CostTable* table_{nullptr};
    ScheduleParams sched_{};
    std::size_t grain_{1};
};

}  // namespace

auto sanitize(const CostHint& hint) noexcept -> CostHint {
    if (!std::isfinite(hint.mean_seconds) || hint.mean_seconds <= 0.0) {
        return CostHint{.mean_seconds = 0.0, .variance = 0.0};
    }
    const double mean = std::clamp(hint.mean_seconds, 1e-9, 1e9);
    double var = 0.0;
    if (std::isfinite(hint.variance) && hint.variance >= 0.0) {
        var = std::clamp(hint.variance, 0.0, 1e18);
    }
    return CostHint{.mean_seconds = mean, .variance = var};
}

auto effective_cost(const CostHint& task_hint, std::string_view op, const CostTable* table,
                    const ScheduleParams& params) noexcept -> double {
    const CostHint s_task = sanitize(task_hint);
    CostHint selected_hint = s_task;

    // Tier 2: If per-task hint is unknown, check CostTable by non-empty OpId.
    if (selected_hint.mean_seconds <= 0.0 && table != nullptr && !op.empty()) {
        const auto it = table->find(op);
        if (it != table->end()) {
            selected_hint = sanitize(it->second);
        }
    }

    // Tier 3: Unknown cost defaults to exactly 0.0.
    if (selected_hint.mean_seconds <= 0.0) {
        return 0.0;
    }

    double lambda = params.risk_lambda;
    if (!std::isfinite(lambda) || lambda < 0.0) {
        lambda = 0.0;
    }

    const double c = selected_hint.mean_seconds + lambda * std::sqrt(selected_hint.variance);
    if (!std::isfinite(c) || c < 0.0) {
        return 0.0;
    }
    return c;
}

auto level_order(const TaskGraph& g, std::span<const TaskId> level, const CostTable* table,
                 const ScheduleParams& params) -> std::vector<TaskId> {
    if (level.empty()) {
        return {};
    }
    if (level.size() == 1) {
        return {level[0]};
    }

    struct TaskCostEntry {
        TaskId id;
        double cost;
    };

    std::vector<TaskCostEntry> entries;
    entries.reserve(level.size());
    for (const TaskId id : level) {
        const CostHint& task_hint = g.hint(id);
        const std::string_view op = g.op_id(id);
        const double cost = effective_cost(task_hint, op, table, params);
        entries.push_back(TaskCostEntry{.id = id, .cost = cost});
    }

    std::ranges::sort(entries, [](const TaskCostEntry& a, const TaskCostEntry& b) noexcept -> bool {
        if (a.cost != b.cost) {
            return a.cost > b.cost;  // Descending cost
        }
        return a.id.value < b.id.value;  // Ascending TaskId tie-break (issuance order)
    });

    std::vector<TaskId> result;
    result.reserve(entries.size());
    for (const auto& entry : entries) {
        result.push_back(entry.id);
    }
    return result;
}

auto to_placement(Affinity a) noexcept -> std::uint8_t {
    switch (a) {
        case Affinity::cpu_only:
            return 1;  // SgeePlacement::cpu
        case Affinity::gpu_only:
            return 2;  // SgeePlacement::gpu
        case Affinity::hybrid:
            return 1;  // SgeePlacement::cpu (hybrid resolves to cpu until a GPU worker exists)
    }
    return 1;
}

auto cost_ordered_local_executor(const CostTable* table, ScheduleParams params,
                                 std::size_t grain) -> std::unique_ptr<Executor> {
    return std::make_unique<CostOrderedLocalExecutor>(table, params, grain);
}

}  // namespace nimblecas
