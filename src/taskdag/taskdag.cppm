// NimbleCAS task-DAG scheduler: deterministic, single-node wavefront execution of a
// dependency graph of pure computation steps, with a pluggable Executor seam.
// @author Olumuyiwa Oluwasanmi
//
// WHAT THIS IS. A TaskGraph is built incrementally: add_task(fn, deps, hint) appends one
// task whose dependencies must already have been issued TaskIds, so the graph is ACYCLIC
// BY CONSTRUCTION — no cycle check is ever needed or possible to fail. Each task's
// depth = 1 + max(parent depth) (0 for a task with no deps) places it in a wavefront
// LEVEL; an Executor runs level 0, then level 1, and so on, and within a level every task
// is independent of every other task in that level (their dependencies all sit at
// strictly smaller depth), so an executor is free to run a level's tasks concurrently.
//
// DETERMINISM CONTRACT. serial_executor() and local_parallel_executor() produce
// BIT-IDENTICAL TaskRunResult::outputs for the same TaskGraph, for any thread count. Each
// TaskFn is a pure function of its parents' payloads (never of scheduling), and every
// level's parallel evaluation goes through nimblecas.parallel::transform_index, which is
// order-preserving, so the accumulation back into `outputs` always happens in a single
// fixed index order — never in thread-completion order. No atomics-driven nondeterminism.
//
// ERROR POISONING. If task T's own TaskFn returns an error, T IS executed (that is how the
// error was discovered) and the error becomes T's output; T counts toward `executed`. Any
// task that depends, directly or transitively, on T is POISONED: it is NOT executed (its
// TaskFn is never called), its output slot carries the error of its lowest-topological-
// index failing ancestor (TaskId order == issuance order == a topological order here, so
// "lowest index" is well-defined and independent of scheduling), and this propagates
// deterministically to further descendants. An independent subgraph with no failing
// ancestor still runs to completion.
//
// ── HONESTY BOUNDARY (Rule 32) ───────────────────────────────────────────────────────────
// This is a SINGLE-NODE, LOCAL fork-join scheduler, now with an additive TaskRegistry
// and add_named_task path for named operations. Payload (std::vector<std::byte>) is
// serializable by construction. Unnamed TaskFn closures cannot be serialized to remote
// processes; named tasks identified by a registered OpId ("<domain>.<op>/v<N>") allow a
// remote/distributed backend (such as nimblecas.taskdag_sgee) to ship op identifiers and
// arguments while executing identical code across nodes. Named tasks run on local
// executors bit-identically to plain closure tasks.

module;
#include <cassert>  // assert (unavailable via `import std`); guards internal-accessor preconditions

export module nimblecas.taskdag;

import std;
import nimblecas.core;
import nimblecas.parallel;

export namespace nimblecas {

// A task's output (or another task's input): an opaque byte blob. Serializable by
// construction (it already IS bytes); how a caller encodes/decodes domain values into it
// is outside this module's concern.
using Payload = std::vector<std::byte>;

// A task's compute step: given its parents' outputs (in the exact order dependencies were
// declared to add_task), produces this task's own output or fails. MUST be a PURE function
// of `args` — no shared mutable state, no visible side effects — because
// local_parallel_executor invokes every task in a wavefront level concurrently; levels are
// independent by construction (a task's depth is strictly greater than every one of its
// dependencies' depths), so this is the only requirement placed on TaskFn.
using TaskFn = std::function<Result<Payload>(std::span<const Payload>)>;

// Canonical identifier for a registered task operation string (e.g. "nimblecas.poly.eval_batch/v1").
using OpId = std::string;

// Registry mapping versioned OpId strings to executable TaskFn bodies.
// The coordinator and worker processes populate an identical registry at startup.
class TaskRegistry {
public:
    TaskRegistry() = default;

    // Registers `fn` under `id`. Fails with domain_error if `id` violates the OpId grammar
    // (<domain>.<op>/v<N>, ASCII, <= 256 bytes) or is already registered.
    [[nodiscard]] auto register_op(OpId id, TaskFn fn) -> Result<void> {
        if (!validate_op_id(id)) {
            return make_error<void>(MathError::domain_error);
        }
        auto [it, inserted] = ops_.emplace(std::move(id), std::move(fn));
        if (!inserted) {
            return make_error<void>(MathError::domain_error);
        }
        return {};
    }

    // Lookup. Returns nullptr if absent.
    [[nodiscard]] auto find(std::string_view id) const noexcept -> const TaskFn* {
        const auto it = ops_.find(id);
        if (it == ops_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    // Number of registered operations.
    [[nodiscard]] auto size() const noexcept -> std::size_t { return ops_.size(); }

    // FNV-1a fingerprint over sorted OpIds, each followed by '\0'.
    [[nodiscard]] auto fingerprint() const noexcept -> std::uint64_t {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        constexpr std::uint64_t prime = 0x00000100000001b3ULL;
        for (const auto& [id, fn] : ops_) {
            for (const char c : id) {
                hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
                hash *= prime;
            }
            hash ^= 0ULL;
            hash *= prime;
        }
        return hash;
    }

private:
    [[nodiscard]] static auto validate_op_id(std::string_view id) noexcept -> bool {
        if (id.empty() || id.size() > 256) {
            return false;
        }
        for (const char c : id) {
            const auto uc = static_cast<unsigned char>(c);
            if (uc >= 128 || c == '\0') {
                return false;
            }
        }
        const auto v_pos = id.rfind("/v");
        if (v_pos == std::string_view::npos || v_pos == 0) {
            return false;
        }
        const std::string_view ver = id.substr(v_pos + 2);
        if (ver.empty()) {
            return false;
        }
        for (const char c : ver) {
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    std::map<OpId, TaskFn, std::less<>> ops_{};
};

// Opaque handle to a task within the TaskGraph that issued it. TaskId order == issuance
// order == a valid topological order (a task can only depend on already-issued ids).
struct TaskId {
    std::size_t value{0};

    [[nodiscard]] auto operator==(const TaskId&) const noexcept -> bool = default;
};

// Recorded (not yet exploited) cost estimate for a task, in seconds, e.g. from a prior
// profiling run. A future cost-aware scheduler (critical-path ordering, load balancing
// across a distributed backend) can read it via TaskGraph::hint(id); neither shipped
// executor here uses it — both are plain level-synchronous wavefront schedulers.
struct CostHint {
    double mean_seconds{0.0};
    double variance{0.0};
};

// Builds a dependency graph of pure computation tasks. Immutable-by-append: add_task only
// ever appends (existing tasks are never removed or mutated), and every dependency must
// already have a TaskId issued by THIS graph — which makes the graph acyclic by
// construction, with no separate cycle check needed or possible to fail.
class TaskGraph {
public:
    // Appends a new task depending on `deps` (parents' outputs are passed to `fn` in this
    // exact order) with an optional cost hint, returning the new task's TaskId. Fails with
    // domain_error, WITHOUT adding the task, if any entry of `deps` references a TaskId
    // unknown to this graph — i.e. dep.value >= size() at the time of this call (it names
    // a task not yet issued, or one from a different graph).
    [[nodiscard]] auto add_task(TaskFn fn, std::span<const TaskId> deps = {}, CostHint hint = {})
        -> Result<TaskId> {
        for (const TaskId d : deps) {
            if (d.value >= tasks_.size()) {
                return make_error<TaskId>(MathError::domain_error);
            }
        }
        std::size_t new_depth = 0;
        for (const TaskId d : deps) {
            new_depth = std::max(new_depth, tasks_[d.value].depth + 1);
        }
        const TaskId id{tasks_.size()};
        tasks_.push_back(Task{.fn = std::move(fn),
                               .deps = std::vector<TaskId>(deps.begin(), deps.end()),
                               .hint = hint,
                               .depth = new_depth,
                               .op_id = {}});
        if (new_depth >= levels_.size()) {
            levels_.resize(new_depth + 1);
        }
        levels_[new_depth].push_back(id);
        return id;
    }

    // Appends a new task whose TaskFn body is copied from `reg` for `op`. Fails with
    // domain_error (adding nothing) if `op` is not in `reg` or any entry of `deps` is unknown.
    [[nodiscard]] auto add_named_task(const TaskRegistry& reg, OpId op,
                                      std::span<const TaskId> deps = {},
                                      CostHint hint = {}) -> Result<TaskId> {
        const TaskFn* fn_ptr = reg.find(op);
        if (fn_ptr == nullptr) {
            return make_error<TaskId>(MathError::domain_error);
        }
        for (const TaskId d : deps) {
            if (d.value >= tasks_.size()) {
                return make_error<TaskId>(MathError::domain_error);
            }
        }
        std::size_t new_depth = 0;
        for (const TaskId d : deps) {
            new_depth = std::max(new_depth, tasks_[d.value].depth + 1);
        }
        const TaskId id{tasks_.size()};
        tasks_.push_back(Task{.fn = *fn_ptr,
                               .deps = std::vector<TaskId>(deps.begin(), deps.end()),
                               .hint = hint,
                               .depth = new_depth,
                               .op_id = std::move(op)});
        if (new_depth >= levels_.size()) {
            levels_.resize(new_depth + 1);
        }
        levels_[new_depth].push_back(id);
        return id;
    }

    // Number of tasks issued so far.
    [[nodiscard]] auto size() const noexcept -> std::size_t { return tasks_.size(); }

    // ── Accessors below are for Executor implementations (this module's two shipped
    // executors, and any future one built against this same interface) ─────────────────
    // The curated, stable surface for callers assembling a graph is add_task/size plus
    // TaskRunResult; these exist so an Executor never needs a friend declaration or a
    // second, parallel representation of the graph.

    // This task's dependencies, in the exact order given to add_task.
    [[nodiscard]] auto deps(TaskId id) const noexcept -> std::span<const TaskId> {
        assert(id.value < tasks_.size() && "TaskGraph::deps: id not issued by this graph");
        return tasks_[id.value].deps;
    }

    // This task's wavefront level: 0 for a task with no dependencies, else
    // 1 + max(parent depth).
    [[nodiscard]] auto depth(TaskId id) const noexcept -> std::size_t {
        assert(id.value < tasks_.size() && "TaskGraph::depth: id not issued by this graph");
        return tasks_[id.value].depth;
    }

    // The cost hint recorded at add_task.
    [[nodiscard]] auto hint(TaskId id) const noexcept -> const CostHint& {
        assert(id.value < tasks_.size() && "TaskGraph::hint: id not issued by this graph");
        return tasks_[id.value].hint;
    }

    // The op_id recorded at add_named_task, or empty for a plain add_task closure.
    [[nodiscard]] auto op_id(TaskId id) const noexcept -> std::string_view {
        assert(id.value < tasks_.size() && "TaskGraph::op_id: id not issued by this graph");
        return tasks_[id.value].op_id;
    }

    // Number of distinct depths present (0 for an empty graph).
    [[nodiscard]] auto num_levels() const noexcept -> std::size_t { return levels_.size(); }

    // Every TaskId at the given depth, in ascending TaskId (== issuance) order.
    [[nodiscard]] auto level(std::size_t d) const noexcept -> std::span<const TaskId> {
        assert(d < levels_.size() && "TaskGraph::level: depth out of range");
        return levels_[d];
    }

    // Invokes this task's TaskFn with the given parent outputs. Safe to call concurrently
    // for distinct `id`s, since TaskFn is required to be pure (see TaskFn's doc comment).
    [[nodiscard]] auto invoke(TaskId id, std::span<const Payload> args) const -> Result<Payload> {
        assert(id.value < tasks_.size() && "TaskGraph::invoke: id not issued by this graph");
        return tasks_[id.value].fn(args);
    }

private:
    struct Task {
        TaskFn fn;
        std::vector<TaskId> deps;
        CostHint hint;
        std::size_t depth{0};
        std::string op_id{};
    };

    std::vector<Task> tasks_{};
    std::vector<std::vector<TaskId>> levels_{};  // levels_[d] = tasks at depth d, issuance order
};

// Result of running a TaskGraph to completion.
struct TaskRunResult {
    // outputs[id.value] is task id's result: its own value/error, or (if poisoned) the
    // error of its lowest-topological-index failing ancestor.
    std::vector<Result<Payload>> outputs;
    // measured_seconds[id.value] is the wall-clock time TaskFn took to run, or 0.0 for a
    // task that was poisoned (never invoked).
    std::vector<double> measured_seconds;
    // Number of tasks actually invoked (poisoned tasks are not counted; a task whose own
    // TaskFn returned an error IS counted — it had to run to discover that).
    std::size_t executed{0};
};

// Pluggable execution backend for a TaskGraph. Every Executor must honour the determinism
// contract documented at the top of this module: for a given TaskGraph, `run` must return
// outputs bit-identical to every other Executor's, because a TaskFn's result depends only
// on its own parents' payloads.
class Executor {
public:
    virtual ~Executor() = default;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;

    [[nodiscard]] virtual auto run(const TaskGraph& g) -> Result<TaskRunResult> = 0;
};

// The deterministic reference implementation: runs every task strictly one at a time, in
// wavefront (level-by-level, ascending TaskId within a level) order. Always available; use
// this to validate a faster executor's output.
[[nodiscard]] auto serial_executor() -> std::unique_ptr<Executor>;

// Wavefront execution with each level's independent tasks fanned out across
// nimblecas.parallel::transform_index. Bit-identical to serial_executor() for the same
// graph (see the determinism contract above) — this is purely a wall-clock optimization.
[[nodiscard]] auto local_parallel_executor() -> std::unique_ptr<Executor>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Outcome of attempting to run (or poison) a single task within a wavefront level.
struct TaskOutcome {
    Result<Payload> output;
    // Set iff `output` is an INHERITED error (this task was poisoned): the id of the
    // ancestor whose own TaskFn actually returned that error. Unset when this task
    // succeeded, and — deliberately — also SET (to this task's own id) when this task's
    // own TaskFn returned the error, so a descendant reading this via TaskGraph::deps can
    // always find the true origin in one hop without re-walking history.
    std::optional<std::size_t> poison_origin;
    double seconds{0.0};
    bool executed{false};
};

// Runs task `id`, or poisons it, given the ALREADY-COMPLETED results of every task at a
// strictly smaller depth (that suffices: `outputs`/`origins` need only be valid at the
// indices `id`'s dependencies occupy, which — by the depth invariant — were all finished
// in an earlier wavefront level). A pure function of its arguments: no shared mutable
// state, so distinct `id`s in the same level may call this concurrently.
[[nodiscard]] auto run_or_poison(const TaskGraph& g, TaskId id,
                                 std::span<const Result<Payload>> outputs,
                                 std::span<const std::optional<std::size_t>> origins)
    -> TaskOutcome {
    const std::span<const TaskId> task_deps = g.deps(id);

    // Find the lowest-topological-index ancestor (among direct deps, but each dep's own
    // `origins` entry already resolves to the TRUE originating failure, so this one hop
    // is enough) whose error should poison this task, if any.
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

    // Every dependency succeeded (or there are none): actually run this task.
    std::vector<Payload> args;
    args.reserve(task_deps.size());
    for (const TaskId d : task_deps) {
        args.push_back(outputs[d.value].value());
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto result = g.invoke(id, args);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    if (!result.has_value()) {
        // This task ran and failed: it is its own poison origin for any descendant.
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

class SerialExecutor final : public Executor {
public:
    [[nodiscard]] auto name() const noexcept -> std::string_view override { return "serial"; }

    [[nodiscard]] auto run(const TaskGraph& g) -> Result<TaskRunResult> override {
        const std::size_t n = g.size();
        std::vector<Result<Payload>> outputs(n);
        std::vector<std::optional<std::size_t>> origins(n);
        std::vector<double> seconds(n, 0.0);
        std::size_t executed = 0;

        for (std::size_t lvl = 0; lvl < g.num_levels(); ++lvl) {
            for (const TaskId id : g.level(lvl)) {
                TaskOutcome outcome = run_or_poison(g, id, outputs, origins);
                if (outcome.executed) {
                    ++executed;
                }
                outputs[id.value] = std::move(outcome.output);
                origins[id.value] = outcome.poison_origin;
                seconds[id.value] = outcome.seconds;
            }
        }
        return TaskRunResult{
            .outputs = std::move(outputs), .measured_seconds = std::move(seconds), .executed = executed};
    }
};

class LocalParallelExecutor final : public Executor {
public:
    [[nodiscard]] auto name() const noexcept -> std::string_view override {
        return "local_parallel";
    }

    [[nodiscard]] auto run(const TaskGraph& g) -> Result<TaskRunResult> override {
        const std::size_t n = g.size();
        std::vector<Result<Payload>> outputs(n);
        std::vector<std::optional<std::size_t>> origins(n);
        std::vector<double> seconds(n, 0.0);
        std::size_t executed = 0;

        for (std::size_t lvl = 0; lvl < g.num_levels(); ++lvl) {
            const std::span<const TaskId> ids = g.level(lvl);
            // Tasks within a level are independent by construction (their dependencies
            // all sit at a strictly smaller depth, i.e. an earlier, already-completed
            // level), so they may be evaluated concurrently. transform_index is
            // order-preserving: outcomes[i] always corresponds to ids[i] regardless of
            // scheduling, so the write-back loop below is a fixed, deterministic index
            // order — bit-identical to SerialExecutor's sequential one.
            std::vector<TaskOutcome> outcomes =
                parallel::transform_index(ids.size(), [&](std::size_t i) {
                    return run_or_poison(g, ids[i], outputs, origins);
                });
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const TaskId id = ids[i];
                if (outcomes[i].executed) {
                    ++executed;
                }
                outputs[id.value] = std::move(outcomes[i].output);
                origins[id.value] = outcomes[i].poison_origin;
                seconds[id.value] = outcomes[i].seconds;
            }
        }
        return TaskRunResult{
            .outputs = std::move(outputs), .measured_seconds = std::move(seconds), .executed = executed};
    }
};

}  // namespace

auto serial_executor() -> std::unique_ptr<Executor> {
    return std::make_unique<SerialExecutor>();
}

auto local_parallel_executor() -> std::unique_ptr<Executor> {
    return std::make_unique<LocalParallelExecutor>();
}

}  // namespace nimblecas
