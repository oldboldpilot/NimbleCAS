// NimbleCAS distributed classical planning over task graphs.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.planning` searches in one process. This module expresses the same searches as
// explicit `nimblecas.taskdag` task graphs, so they run under any conforming `Executor`:
// `serial_executor`, `local_parallel_executor`, or `SgeeDistributedExecutor` across a cluster.
//
// WHAT IS ACTUALLY PARALLEL HERE, AND WHAT IS NOT.
//
// Heuristic search is not embarrassingly parallel and this module does not pretend otherwise.
// The open list is a sequential dependency: which state is expanded next depends on every
// expansion before it. What IS data-parallel is the expensive part -- EVALUATING the successors.
// A delete-relaxation heuristic solves a relaxed planning task per state, and on any task worth
// distributing that dominates everything else the search does. So the decomposition is:
//
//   the coordinator owns the open and closed lists and decides WHAT to expand;
//   the workers expand and evaluate, and own nothing.
//
// A worker is handed the task and a batch of states, and returns each state's successors with
// their heuristic values. It holds no search state between rounds, which is what lets a round be
// an ordinary task graph with no messages between workers.
//
// PARALLEL EXPANSION OF AN f-LAYER, AND WHY IT KEEPS OPTIMALITY.
//
// `distributed_astar_plan` expands, in each round, EVERY open state sharing the minimum f. A*
// requires only that states be expanded in non-decreasing order of f; it does not care in which
// order the states of one f-layer are taken. Expanding a whole layer at once is therefore a legal
// A* expansion order, and the plan that comes out is optimal for the same reason the serial one
// is. What parallelism buys is that the layer's evaluations happen at once; what it costs is that
// a layer may contain one state or ten thousand, so the speedup is a property of the task and is
// not claimed here in advance.
//
// THE EQUIVALENCE CONTRACT, STATED EXACTLY.
//
// `distributed_astar_plan` returns a plan of the SAME COST as `astar_plan`, always. It does NOT
// promise the same plan. `astar_plan` breaks ties toward the state discovered earlier, and
// "earlier" is a property of the serial expansion order that expanding a layer at once does not
// have. Where a task has several optimal plans the two may pick different ones -- both optimal,
// both valid. Within this module the choice is deterministic: successors merge in shard-index
// then batch order, and ties break toward the lower operator index, so the SAME executor and the
// same task give the same plan on every run, at any shard or worker count. Claiming the stronger
// property would mean serialising the layer, which is the entire thing being avoided.
//
// `distributed_gbfs_plan` makes no claim about plan length, exactly as `gbfs_plan` does not.
//
// HONESTY. Planning is PSPACE-complete and distributing it does not repeal that. Every plan is
// replayed from the initial state by `nimblecas::planning::validate_plan` before it is returned,
// so an invalid plan is an error and never a result. `undefined_value` means the reachable state
// space was exhausted -- a proof that no plan exists -- and `not_converged` means the expansion
// budget ran out, which is not a proof of anything. The two are different answers and the error
// codes keep them apart.
//
// `max_expansions` is enforced at LAYER granularity, not exactly, and that is a real limitation
// rather than an oversight. An f-layer is atomic to the optimality argument -- stopping part-way
// through one and returning what had been found would forfeit precisely the guarantee this entry
// point exists to make -- so a run may overshoot the budget by up to the size of the layer it is
// in. It bounds when the search gives up; it is not a promise about the exact count.

export module nimblecas.planning_dist;

import std;
import nimblecas.core;
import nimblecas.planning;
import nimblecas.taskdag;

export namespace nimblecas::planning_dist {

using nimblecas::Executor;
using nimblecas::MathError;
using nimblecas::OpId;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::TaskGraph;
using nimblecas::TaskRegistry;
using nimblecas::planning::Plan;
using nimblecas::planning::PlanResult;
using nimblecas::planning::SearchStats;
using nimblecas::planning::State;
using nimblecas::planning::Task;

// Which heuristic the workers evaluate with. The admissibility of the choice is what decides
// whether the search above it may claim optimality, so it is named rather than implied.
enum class Heuristic : std::uint8_t {
    h_max,  // ADMISSIBLE -- the only one `distributed_astar_plan` accepts
    h_add,  // inadmissible
    h_ff,   // inadmissible, and carries preferred operators
};

// Operation names registered in the task registry. The executor ships an op id and arguments,
// never code, so a worker that has not registered these cannot run what it is handed.
inline constexpr std::string_view expand_op_id = "nimblecas.planning.expand_shard/v1";
inline constexpr std::string_view evaluate_op_id = "nimblecas.planning.evaluate_shard/v1";

// The largest number of states one round may carry. A round materialises every state of the
// layer into task payloads, so an unbounded layer would make the payloads, rather than the
// search, the thing that fails.
inline constexpr std::size_t max_layer_states = 1U << 20U;

// One generated successor, as it comes back from a worker.
struct Successor {
    State state;
    std::size_t parent{0};       // index into the batch the worker was handed
    std::size_t op_index{0};     // the operator applied
    std::int64_t step_cost{0};   // that operator's cost
    std::int64_t heuristic{-1};  // the successor's h, or -1 for "unreachable under the relaxation"
    bool preferred{false};       // reached by a preferred operator (h_ff only)

    [[nodiscard]] auto operator==(const Successor&) const -> bool = default;
};

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------
//
// Little-endian fixed-width integers, matching the convention `nimblecas.taskdag_sgee` uses for
// envelope framing. Signed values travel as their two's-complement bit pattern, which C++20
// onward defines exactly, so INT64_MIN survives a round trip unchanged.

[[nodiscard]] auto encode_task(const Task& task) -> Result<Payload>;
[[nodiscard]] auto decode_task(std::span<const std::byte> bytes) -> Result<Task>;
[[nodiscard]] auto encode_states(std::span<const State> states, Heuristic h) -> Result<Payload>;
[[nodiscard]] auto decode_successors(std::span<const std::byte> bytes)
    -> Result<std::vector<Successor>>;

// Registers both operations. BOTH the coordinator and every worker must call this on their own
// registry at startup.
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

// ---------------------------------------------------------------------------
// Graph construction.
// ---------------------------------------------------------------------------

// One round: the states split across `shards` tasks, with no dependencies between them.
// Exposed because a round is the unit a scheduler or a profiler wants to see.
//
// `domain_error` for a malformed task, a zero shard count, or an empty batch; `overflow` if the
// batch exceeds `max_layer_states`.
[[nodiscard]] auto build_round_graph(const TaskRegistry& reg, const Task& task,
                                     std::span<const State> states, Heuristic h,
                                     std::size_t shards) -> Result<TaskGraph>;

// ---------------------------------------------------------------------------
// Distributed entry points.
// ---------------------------------------------------------------------------

// The heuristic value of each state, evaluated across `exec`. Exposed because it is the actual
// parallel primitive the searches are built on, and because batch evaluation is useful on its
// own -- for a portfolio, or for measuring a heuristic over a sample of states.
//
// A `std::nullopt` entry means the goal is unreachable from that state even under the delete
// relaxation, which for `h_max` is a proof that it is unreachable in the real task.
[[nodiscard]] auto distributed_evaluate(const Task& task, std::span<const State> states,
                                        Heuristic h, std::size_t shards, Executor& exec)
    -> Result<std::vector<std::optional<std::int64_t>>>;

// OPTIMAL distributed planning: A* with h_max, expanding a whole f-layer per round.
//
// Returns a plan of the same COST as `astar_plan` -- see the equivalence contract in the header
// for why the plan itself may differ when several optimal plans exist. `undefined_value` when
// the task is provably unsolvable, `not_converged` when `max_expansions` ran out.
[[nodiscard]] auto distributed_astar_plan(const Task& task, std::uint64_t max_expansions,
                                          std::size_t shards, Executor& exec)
    -> Result<PlanResult>;

// SATISFICING distributed planning: greedy best-first search, expanding a batch of the best
// states per round. NO CLAIM OF OPTIMALITY, exactly as `gbfs_plan` makes none.
//
// `beam` is how many of the best-h states are expanded per round; 1 reproduces a strictly serial
// greedy search, and larger values trade more work per round for fewer rounds.
[[nodiscard]] auto distributed_gbfs_plan(const Task& task, std::uint64_t max_expansions,
                                         std::size_t beam, std::size_t shards, Executor& exec)
    -> Result<PlanResult>;

}  // namespace nimblecas::planning_dist

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::planning_dist {

namespace {

constexpr std::uint64_t task_magic = 0x4e43535f5054534bULL;  // "NCS_PTSK"
constexpr std::uint64_t batch_magic = 0x4e43535f50424348ULL;  // "NCS_PBCH"
constexpr std::uint64_t succ_magic = 0x4e43535f50535543ULL;   // "NCS_PSUC"

// Addition that saturates instead of overflowing, matching `nimblecas.planning`'s own
// `add_capped` exactly -- including its convention that INT64_MAX means "unreachable" and is
// absorbing.
//
// This is not defensiveness for its own sake. `g + step_cost` and `g + h` are plain signed
// additions on caller-supplied operator costs, so a task with costs near the top of the range
// overflows: undefined behaviour, and in practice a negative g that sorts to the FRONT of the
// f-order and makes the search return a nonsense "optimum". The serial planner saturates at
// exactly these three sums, so anything else here would also be a silent divergence from the
// answer this module promises to reproduce.
[[nodiscard]] auto add_capped(std::int64_t a, std::int64_t b) noexcept -> std::int64_t {
    constexpr std::int64_t unreachable = std::numeric_limits<std::int64_t>::max();
    if (a == unreachable || b == unreachable) {
        return unreachable;
    }
    std::int64_t out = 0;
    if (__builtin_add_overflow(a, b, &out)) {
        return unreachable;
    }
    return out;
}

auto put_u64(std::vector<std::byte>& out, std::uint64_t v) -> void {
    for (const unsigned int i : std::views::iota(0U, 8U)) {
        out.push_back(static_cast<std::byte>((v >> (8U * i)) & 0xffULL));
    }
}

auto put_i64(std::vector<std::byte>& out, std::int64_t v) -> void {
    put_u64(out, static_cast<std::uint64_t>(v));
}

[[nodiscard]] auto take_u64(std::span<const std::byte> bytes, std::size_t& offset)
    -> std::optional<std::uint64_t> {
    if (offset + 8 > bytes.size()) {
        return std::nullopt;
    }
    std::uint64_t v = 0;
    for (const unsigned int i : std::views::iota(0U, 8U)) {
        const auto byte =
            std::to_integer<unsigned char>(bytes[offset + static_cast<std::size_t>(i)]);
        v |= static_cast<std::uint64_t>(byte) << (8U * i);
    }
    offset += 8;
    return v;
}

[[nodiscard]] auto take_i64(std::span<const std::byte> bytes, std::size_t& offset)
    -> std::optional<std::int64_t> {
    const auto v = take_u64(bytes, offset);
    if (!v.has_value()) {
        return std::nullopt;
    }
    return static_cast<std::int64_t>(*v);
}

// A length that must survive coming off the wire. A corrupt count is a syntax error, never an
// allocation attempt: `per` is the minimum bytes each element occupies, so a claimed count
// larger than the remaining bytes could hold is rejected before anything is reserved.
[[nodiscard]] auto count_is_sane(std::uint64_t n, std::size_t remaining, std::size_t per) -> bool {
    if (per == 0) {
        return false;
    }
    return n <= remaining / per;
}

auto put_condition(std::vector<std::byte>& out, const nimblecas::planning::Condition& c) -> void {
    put_u64(out, static_cast<std::uint64_t>(c.size()));
    for (const auto& fp : c) {
        put_u64(out, static_cast<std::uint64_t>(fp.var));
        put_i64(out, fp.value);
    }
}

[[nodiscard]] auto take_condition(std::span<const std::byte> bytes, std::size_t& off)
    -> std::optional<nimblecas::planning::Condition> {
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 16)) {
        return std::nullopt;
    }
    nimblecas::planning::Condition c;
    c.reserve(static_cast<std::size_t>(*n));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *n)) {
        const auto var = take_u64(bytes, off);
        const auto val = take_i64(bytes, off);
        if (!var.has_value() || !val.has_value()) {
            return std::nullopt;
        }
        c.push_back(nimblecas::planning::FactPair{
            .var = static_cast<std::size_t>(*var), .value = static_cast<std::int32_t>(*val)});
    }
    return c;
}

// The heuristic, applied to one state. Kept in one place so the searches and the batch
// evaluator cannot drift apart on which heuristic means what.
struct Evaluated {
    std::optional<std::int64_t> value;
    std::vector<std::size_t> preferred;
};

[[nodiscard]] auto evaluate_one(const Task& task, const State& s, Heuristic h) -> Evaluated {
    switch (h) {
        case Heuristic::h_max:
            return Evaluated{.value = nimblecas::planning::h_max(task, s), .preferred = {}};
        case Heuristic::h_add:
            return Evaluated{.value = nimblecas::planning::h_add(task, s), .preferred = {}};
        case Heuristic::h_ff: {
            auto ff = nimblecas::planning::h_ff(task, s);
            return Evaluated{.value = ff.value, .preferred = std::move(ff.preferred)};
        }
    }
    return Evaluated{};
}

}  // namespace

auto encode_task(const Task& task) -> Result<Payload> {
    if (const auto ok = nimblecas::planning::validate(task); !ok.has_value()) {
        return make_error<Payload>(ok.error());
    }
    Payload out;
    put_u64(out, task_magic);
    put_u64(out, static_cast<std::uint64_t>(task.domain_size.size()));
    for (const std::int32_t d : task.domain_size) {
        put_i64(out, d);
    }
    put_u64(out, static_cast<std::uint64_t>(task.initial.size()));
    for (const std::int32_t v : task.initial) {
        put_i64(out, v);
    }
    put_condition(out, task.goal);
    put_u64(out, static_cast<std::uint64_t>(task.operators.size()));
    for (const auto& op : task.operators) {
        put_u64(out, static_cast<std::uint64_t>(op.name.size()));
        for (const char ch : op.name) {
            out.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
        }
        put_condition(out, op.preconditions);
        put_condition(out, op.effects);
        put_i64(out, op.cost);
    }
    return out;
}

auto decode_task(std::span<const std::byte> bytes) -> Result<Task> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != task_magic) {
        return make_error<Task>(MathError::syntax_error);
    }
    Task task;
    const auto nvars = take_u64(bytes, off);
    if (!nvars.has_value() || !count_is_sane(*nvars, bytes.size() - off, 8)) {
        return make_error<Task>(MathError::syntax_error);
    }
    task.domain_size.reserve(static_cast<std::size_t>(*nvars));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *nvars)) {
        const auto d = take_i64(bytes, off);
        if (!d.has_value()) {
            return make_error<Task>(MathError::syntax_error);
        }
        task.domain_size.push_back(static_cast<std::int32_t>(*d));
    }
    const auto ninit = take_u64(bytes, off);
    if (!ninit.has_value() || !count_is_sane(*ninit, bytes.size() - off, 8)) {
        return make_error<Task>(MathError::syntax_error);
    }
    task.initial.reserve(static_cast<std::size_t>(*ninit));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *ninit)) {
        const auto v = take_i64(bytes, off);
        if (!v.has_value()) {
            return make_error<Task>(MathError::syntax_error);
        }
        task.initial.push_back(static_cast<std::int32_t>(*v));
    }
    auto goal = take_condition(bytes, off);
    if (!goal.has_value()) {
        return make_error<Task>(MathError::syntax_error);
    }
    task.goal = std::move(*goal);
    const auto nops = take_u64(bytes, off);
    if (!nops.has_value() || !count_is_sane(*nops, bytes.size() - off, 32)) {
        return make_error<Task>(MathError::syntax_error);
    }
    task.operators.reserve(static_cast<std::size_t>(*nops));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *nops)) {
        nimblecas::planning::Operator op;
        const auto nlen = take_u64(bytes, off);
        if (!nlen.has_value() || !count_is_sane(*nlen, bytes.size() - off, 1)) {
            return make_error<Task>(MathError::syntax_error);
        }
        op.name.reserve(static_cast<std::size_t>(*nlen));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *nlen)) {
            if (off >= bytes.size()) {
                return make_error<Task>(MathError::syntax_error);
            }
            op.name.push_back(static_cast<char>(std::to_integer<unsigned char>(bytes[off])));
            ++off;
        }
        auto pre = take_condition(bytes, off);
        auto eff = take_condition(bytes, off);
        const auto cost = take_i64(bytes, off);
        if (!pre.has_value() || !eff.has_value() || !cost.has_value()) {
            return make_error<Task>(MathError::syntax_error);
        }
        op.preconditions = std::move(*pre);
        op.effects = std::move(*eff);
        op.cost = *cost;
        task.operators.push_back(std::move(op));
    }
    // A task that arrives malformed is a syntax fault on the wire, not a silently-accepted one:
    // every worker would otherwise rediscover the same fault independently.
    if (const auto ok = nimblecas::planning::validate(task); !ok.has_value()) {
        return make_error<Task>(MathError::syntax_error);
    }
    return task;
}

auto encode_states(std::span<const State> states, Heuristic h) -> Result<Payload> {
    if (states.empty()) {
        return make_error<Payload>(MathError::domain_error);
    }
    Payload out;
    put_u64(out, batch_magic);
    put_u64(out, static_cast<std::uint64_t>(h));
    put_u64(out, static_cast<std::uint64_t>(states.size()));
    for (const State& s : states) {
        put_u64(out, static_cast<std::uint64_t>(s.size()));
        for (const std::int32_t v : s) {
            put_i64(out, v);
        }
    }
    return out;
}

namespace {

[[nodiscard]] auto decode_states(std::span<const std::byte> bytes, std::size_t& off)
    -> std::optional<std::pair<std::vector<State>, Heuristic>> {
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != batch_magic) {
        return std::nullopt;
    }
    const auto h_raw = take_u64(bytes, off);
    if (!h_raw.has_value() || *h_raw > static_cast<std::uint64_t>(Heuristic::h_ff)) {
        return std::nullopt;
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 8)) {
        return std::nullopt;
    }
    std::vector<State> states;
    states.reserve(static_cast<std::size_t>(*n));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *n)) {
        const auto len = take_u64(bytes, off);
        if (!len.has_value() || !count_is_sane(*len, bytes.size() - off, 8)) {
            return std::nullopt;
        }
        State s;
        s.reserve(static_cast<std::size_t>(*len));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *len)) {
            const auto v = take_i64(bytes, off);
            if (!v.has_value()) {
                return std::nullopt;
            }
            s.push_back(static_cast<std::int32_t>(*v));
        }
        states.push_back(std::move(s));
    }
    return std::pair{std::move(states), static_cast<Heuristic>(*h_raw)};
}

auto put_successor(std::vector<std::byte>& out, const Successor& s) -> void {
    put_u64(out, static_cast<std::uint64_t>(s.state.size()));
    for (const std::int32_t v : s.state) {
        put_i64(out, v);
    }
    put_u64(out, static_cast<std::uint64_t>(s.parent));
    put_u64(out, static_cast<std::uint64_t>(s.op_index));
    put_i64(out, s.step_cost);
    put_i64(out, s.heuristic);
    put_u64(out, s.preferred ? 1ULL : 0ULL);
}

}  // namespace

auto decode_successors(std::span<const std::byte> bytes) -> Result<std::vector<Successor>> {
    using Ret = std::vector<Successor>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != succ_magic) {
        return make_error<Ret>(MathError::syntax_error);
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 48)) {
        return make_error<Ret>(MathError::syntax_error);
    }
    Ret out;
    out.reserve(static_cast<std::size_t>(*n));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *n)) {
        Successor s;
        const auto len = take_u64(bytes, off);
        if (!len.has_value() || !count_is_sane(*len, bytes.size() - off, 8)) {
            return make_error<Ret>(MathError::syntax_error);
        }
        s.state.reserve(static_cast<std::size_t>(*len));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *len)) {
            const auto v = take_i64(bytes, off);
            if (!v.has_value()) {
                return make_error<Ret>(MathError::syntax_error);
            }
            s.state.push_back(static_cast<std::int32_t>(*v));
        }
        const auto parent = take_u64(bytes, off);
        const auto op = take_u64(bytes, off);
        const auto step = take_i64(bytes, off);
        const auto h = take_i64(bytes, off);
        const auto pref = take_u64(bytes, off);
        if (!parent.has_value() || !op.has_value() || !step.has_value() || !h.has_value() ||
            !pref.has_value() || *pref > 1) {
            return make_error<Ret>(MathError::syntax_error);
        }
        s.parent = static_cast<std::size_t>(*parent);
        s.op_index = static_cast<std::size_t>(*op);
        s.step_cost = *step;
        s.heuristic = *h;
        s.preferred = *pref == 1;
        out.push_back(std::move(s));
    }
    return out;
}

namespace {

// The shard body for `expand_op_id`: generate every applicable successor of every state in the
// batch, and evaluate each one.
[[nodiscard]] auto expand_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto task = decode_task(args[0]);
    if (!task) {
        return make_error<Payload>(task.error());
    }
    std::size_t off = 0;
    auto batch = decode_states(args[1], off);
    if (!batch.has_value()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    const auto& [states, h] = *batch;

    std::vector<Successor> out;
    for (const auto si : std::views::iota(std::size_t{0}, states.size())) {
        const State& s = states[si];
        if (s.size() != task->num_vars()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        // Preferred operators come from the PARENT's relaxed plan, so they are computed once
        // per state rather than once per successor.
        const auto parent_eval = evaluate_one(*task, s, h);
        for (const auto oi : std::views::iota(std::size_t{0}, task->operators.size())) {
            if (!nimblecas::planning::applicable(*task, s, oi)) {
                continue;
            }
            auto next = nimblecas::planning::apply(*task, s, oi);
            if (!next) {
                return make_error<Payload>(next.error());
            }
            const auto child = evaluate_one(*task, *next, h);
            out.push_back(Successor{
                .state = std::move(*next),
                .parent = si,
                .op_index = oi,
                .step_cost = task->operators[oi].cost,
                .heuristic = child.value.value_or(-1),
                .preferred = std::ranges::find(parent_eval.preferred, oi) !=
                             parent_eval.preferred.end()});
        }
    }
    Payload payload;
    put_u64(payload, succ_magic);
    put_u64(payload, static_cast<std::uint64_t>(out.size()));
    for (const auto& s : out) {
        put_successor(payload, s);
    }
    return payload;
}

// The shard body for `evaluate_op_id`: evaluate the batch and nothing else. Reported in the same
// `Successor` shape with `op_index` unused, so one decoder serves both operations.
[[nodiscard]] auto evaluate_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto task = decode_task(args[0]);
    if (!task) {
        return make_error<Payload>(task.error());
    }
    std::size_t off = 0;
    auto batch = decode_states(args[1], off);
    if (!batch.has_value()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    const auto& [states, h] = *batch;
    Payload payload;
    put_u64(payload, succ_magic);
    put_u64(payload, static_cast<std::uint64_t>(states.size()));
    for (const auto si : std::views::iota(std::size_t{0}, states.size())) {
        if (states[si].size() != task->num_vars()) {
            return make_error<Payload>(MathError::syntax_error);
        }
        const auto e = evaluate_one(*task, states[si], h);
        put_successor(payload, Successor{.state = states[si],
                                         .parent = si,
                                         .op_index = 0,
                                         .step_cost = 0,
                                         .heuristic = e.value.value_or(-1),
                                         .preferred = false});
    }
    return payload;
}

// The shard boundaries for `total` items over `shards` shards, balanced and contiguous.
[[nodiscard]] auto shard_range(std::size_t total, std::size_t shards, std::size_t s)
    -> std::pair<std::size_t, std::size_t> {
    const std::size_t base = total / shards;
    const std::size_t rest = total % shards;
    const std::size_t lo = s * base + std::min(s, rest);
    const std::size_t hi = lo + base + (s < rest ? 1 : 0);
    return {lo, hi};
}

// Builds a round graph and runs it, returning the successors in SHARD-INDEX then batch order --
// which is what makes every merge below deterministic without any further sorting.
[[nodiscard]] auto run_round(const Task& task, std::span<const State> states, Heuristic h,
                             std::size_t shards, Executor& exec, std::string_view op)
    -> Result<std::vector<std::vector<Successor>>> {
    using Ret = std::vector<std::vector<Successor>>;
    if (states.empty() || shards == 0) {
        return make_error<Ret>(MathError::domain_error);
    }
    if (states.size() > max_layer_states) {
        return make_error<Ret>(MathError::overflow);
    }
    TaskRegistry reg;
    if (const auto ok = register_ops(reg); !ok) {
        return make_error<Ret>(ok.error());
    }
    auto encoded = encode_task(task);
    if (!encoded) {
        return make_error<Ret>(encoded.error());
    }
    const std::size_t used = std::min(shards, states.size());
    TaskGraph graph;
    std::vector<std::size_t> offsets;
    offsets.reserve(used);
    for (const auto s : std::views::iota(std::size_t{0}, used)) {
        const auto [lo, hi] = shard_range(states.size(), used, s);
        offsets.push_back(lo);
        auto batch = encode_states(states.subspan(lo, hi - lo), h);
        if (!batch) {
            return make_error<Ret>(batch.error());
        }
        auto id = graph.add_named_task(reg, OpId{op},
                                       std::vector<Payload>{*encoded, std::move(*batch)});
        if (!id) {
            return make_error<Ret>(id.error());
        }
    }
    auto run = exec.run(graph);
    if (!run) {
        return make_error<Ret>(run.error());
    }
    Ret out;
    out.reserve(run->outputs.size());
    for (const auto s : std::views::iota(std::size_t{0}, run->outputs.size())) {
        const Result<Payload>& outcome = run->outputs[s];
        if (!outcome) {
            return make_error<Ret>(outcome.error());
        }
        auto succ = decode_successors(*outcome);
        if (!succ) {
            return make_error<Ret>(succ.error());
        }
        // A worker numbers parents within its own batch; the coordinator needs them numbered
        // within the round.
        for (auto& item : *succ) {
            item.parent += offsets[s];
        }
        out.push_back(std::move(*succ));
    }
    return out;
}

// One node of the coordinator's search tree, kept so a plan can be walked back from the goal.
struct Node {
    State state;
    std::int64_t g{0};
    std::size_t parent{0};      // index into `nodes`, self for the root
    std::size_t op_index{0};    // the operator that reached it
    bool is_root{false};
};

[[nodiscard]] auto reconstruct(const std::vector<Node>& nodes, std::size_t goal_index)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> steps;
    std::size_t cur = goal_index;
    while (!nodes[cur].is_root) {
        steps.push_back(nodes[cur].op_index);
        cur = nodes[cur].parent;
    }
    std::ranges::reverse(steps);
    return steps;
}

}  // namespace

auto register_ops(TaskRegistry& reg) -> Result<void> {
    auto r = reg.register_op(OpId{expand_op_id}, expand_shard);
    if (!r) {
        return r;
    }
    return reg.register_op(OpId{evaluate_op_id}, evaluate_shard);
}

auto build_round_graph(const TaskRegistry& reg, const Task& task, std::span<const State> states,
                       Heuristic h, std::size_t shards) -> Result<TaskGraph> {
    if (states.empty() || shards == 0) {
        return make_error<TaskGraph>(MathError::domain_error);
    }
    if (states.size() > max_layer_states) {
        return make_error<TaskGraph>(MathError::overflow);
    }
    auto encoded = encode_task(task);
    if (!encoded) {
        return make_error<TaskGraph>(encoded.error());
    }
    const std::size_t used = std::min(shards, states.size());
    TaskGraph graph;
    for (const auto s : std::views::iota(std::size_t{0}, used)) {
        const auto [lo, hi] = shard_range(states.size(), used, s);
        auto batch = encode_states(states.subspan(lo, hi - lo), h);
        if (!batch) {
            return make_error<TaskGraph>(batch.error());
        }
        auto id = graph.add_named_task(reg, OpId{expand_op_id},
                                       std::vector<Payload>{*encoded, std::move(*batch)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

auto distributed_evaluate(const Task& task, std::span<const State> states, Heuristic h,
                          std::size_t shards, Executor& exec)
    -> Result<std::vector<std::optional<std::int64_t>>> {
    using Ret = std::vector<std::optional<std::int64_t>>;
    if (const auto ok = nimblecas::planning::validate(task); !ok.has_value()) {
        return make_error<Ret>(ok.error());
    }
    auto rounds = run_round(task, states, h, shards, exec, evaluate_op_id);
    if (!rounds) {
        return make_error<Ret>(rounds.error());
    }
    Ret out(states.size(), std::nullopt);
    for (const auto& shard : *rounds) {
        for (const auto& s : shard) {
            if (s.parent >= out.size()) {
                return make_error<Ret>(MathError::syntax_error);
            }
            out[s.parent] = s.heuristic < 0 ? std::nullopt : std::optional{s.heuristic};
        }
    }
    return out;
}

auto distributed_astar_plan(const Task& task, std::uint64_t max_expansions, std::size_t shards,
                            Executor& exec) -> Result<PlanResult> {
    if (const auto ok = nimblecas::planning::validate(task); !ok.has_value()) {
        return make_error<PlanResult>(ok.error());
    }
    if (shards == 0) {
        return make_error<PlanResult>(MathError::domain_error);
    }
    SearchStats stats;

    // The root. h_max is the only admissible choice, and optimality is the point of this entry
    // point, so it is not a parameter.
    const auto root_h = nimblecas::planning::h_max(task, task.initial);
    ++stats.evaluated;
    if (!root_h.has_value()) {
        // Unreachable even under the relaxation, which is a proof it is unreachable outright.
        return make_error<PlanResult>(MathError::undefined_value);
    }

    std::vector<Node> nodes;
    nodes.push_back(Node{.state = task.initial, .g = 0, .parent = 0, .op_index = 0, .is_root = true});
    std::map<State, std::size_t> best;  // state -> index in `nodes`, holding its best known g
    best.emplace(task.initial, std::size_t{0});

    // open: node index -> f. A std::map keyed by state keeps the coordinator's decisions
    // independent of hash order, so the search is reproducible across standard libraries.
    std::vector<std::size_t> open{0};
    std::vector<std::int64_t> open_f{*root_h};

    while (!open.empty()) {
        // The whole f-layer: A* may expand the states of one layer in any order.
        const std::int64_t f_min = *std::ranges::min_element(open_f);
        std::vector<std::size_t> layer;
        std::vector<std::size_t> rest;
        std::vector<std::int64_t> rest_f;
        for (const auto i : std::views::iota(std::size_t{0}, open.size())) {
            if (open_f[i] != f_min) {
                rest.push_back(open[i]);
                rest_f.push_back(open_f[i]);
                continue;
            }
            // Skip STALE entries. When a state is re-reached more cheaply a new node is pushed,
            // and the old one stays in `open` carrying the dearer g. Expanding it is harmless --
            // every successor it generates loses to the cheaper path in `best` -- but harmless
            // work still costs a round trip to a worker for every one of its successors, which
            // is exactly the compute this module exists to spend well. A node is current iff it
            // is the one `best` holds for its state.
            const auto it = best.find(nodes[open[i]].state);
            if (it == best.end() || it->second != open[i]) {
                continue;
            }
            layer.push_back(open[i]);
        }
        if (layer.empty()) {
            // The whole layer was stale. Nothing to expand, and the remaining entries carry
            // strictly larger f, so the search simply moves on rather than spinning.
            open = std::move(rest);
            open_f = std::move(rest_f);
            continue;
        }
        open = std::move(rest);
        open_f = std::move(rest_f);

        // A goal in this layer settles it: nothing cheaper can appear later, because every
        // remaining f is at least f_min and f is a lower bound on the total cost.
        for (const std::size_t ni : layer) {
            if (nimblecas::planning::holds(nodes[ni].state, task.goal)) {
                Plan plan{.steps = reconstruct(nodes, ni), .cost = nodes[ni].g};
                auto valid = nimblecas::planning::validate_plan(task, plan);
                if (!valid) {
                    return make_error<PlanResult>(valid.error());
                }
                return PlanResult{.plan = std::move(plan), .stats = stats};
            }
        }

        // The budget is checked HERE, after the goal test and before any expansion, because
        // that is the order the serial planner uses: it goal-tests a popped node before
        // checking its budget, so a task whose goal is already in hand is solved even at a
        // budget of zero. Checking first would make the two disagree at every budget boundary.
        if (stats.expanded >= max_expansions) {
            return make_error<PlanResult>(MathError::not_converged);
        }
        std::vector<State> batch;
        batch.reserve(layer.size());
        for (const std::size_t ni : layer) {
            batch.push_back(nodes[ni].state);
        }
        // The budget is enforced at LAYER granularity, and that is a real limitation rather than
        // an oversight: an f-layer is atomic to the optimality argument, so stopping part-way
        // through one and reporting what was found would forfeit exactly the guarantee this entry
        // point exists to make. A run may therefore overshoot `max_expansions` by up to the size
        // of the layer in progress. The check at the top of the loop is what stops it.
        stats.expanded += layer.size();

        auto rounds = run_round(task, batch, Heuristic::h_max, shards, exec, expand_op_id);
        if (!rounds) {
            return make_error<PlanResult>(rounds.error());
        }
        // Shard-index then batch order: deterministic without any further sorting.
        for (const auto& shard : *rounds) {
            for (const auto& s : shard) {
                ++stats.generated;
                ++stats.evaluated;
                if (s.parent >= layer.size()) {
                    return make_error<PlanResult>(MathError::syntax_error);
                }
                if (s.heuristic < 0) {
                    continue;  // unreachable under the relaxation: no plan runs through here
                }
                const std::size_t parent_node = layer[s.parent];
                const std::int64_t g = add_capped(nodes[parent_node].g, s.step_cost);
                const auto it = best.find(s.state);
                if (it != best.end() && nodes[it->second].g <= g) {
                    continue;  // already reached at least as cheaply
                }
                nodes.push_back(Node{.state = s.state,
                                     .g = g,
                                     .parent = parent_node,
                                     .op_index = s.op_index,
                                     .is_root = false});
                const std::size_t idx = nodes.size() - 1;
                if (it != best.end()) {
                    it->second = idx;
                } else {
                    best.emplace(s.state, idx);
                }
                open.push_back(idx);
                open_f.push_back(add_capped(g, s.heuristic));
            }
        }
    }
    // Every reachable state was expanded and no goal was among them.
    return make_error<PlanResult>(MathError::undefined_value);
}

auto distributed_gbfs_plan(const Task& task, std::uint64_t max_expansions, std::size_t beam,
                           std::size_t shards, Executor& exec) -> Result<PlanResult> {
    if (const auto ok = nimblecas::planning::validate(task); !ok.has_value()) {
        return make_error<PlanResult>(ok.error());
    }
    if (shards == 0 || beam == 0) {
        return make_error<PlanResult>(MathError::domain_error);
    }
    SearchStats stats;

    const auto root = nimblecas::planning::h_ff(task, task.initial);
    ++stats.evaluated;
    if (!root.value.has_value()) {
        return make_error<PlanResult>(MathError::undefined_value);
    }

    std::vector<Node> nodes;
    nodes.push_back(Node{.state = task.initial, .g = 0, .parent = 0, .op_index = 0, .is_root = true});
    std::set<State> seen;
    seen.insert(task.initial);

    // (h, preferred-first, node index). Ordered so the best h wins, then a preferred successor,
    // then the earlier node -- all three deterministic.
    struct OpenEntry {
        std::int64_t h{0};
        int not_preferred{0};
        std::size_t node{0};

        [[nodiscard]] auto operator<=>(const OpenEntry&) const noexcept = default;
    };
    std::vector<OpenEntry> open{OpenEntry{.h = *root.value, .not_preferred = 0, .node = 0}};

    while (!open.empty()) {
        std::ranges::sort(open);
        const std::size_t take = std::min(beam, open.size());
        std::vector<std::size_t> layer;
        layer.reserve(take);
        for (const auto i : std::views::iota(std::size_t{0}, take)) {
            layer.push_back(open[i].node);
        }
        open.erase(open.begin(), open.begin() + static_cast<std::ptrdiff_t>(take));

        for (const std::size_t ni : layer) {
            if (nimblecas::planning::holds(nodes[ni].state, task.goal)) {
                Plan plan{.steps = reconstruct(nodes, ni), .cost = nodes[ni].g};
                auto valid = nimblecas::planning::validate_plan(task, plan);
                if (!valid) {
                    return make_error<PlanResult>(valid.error());
                }
                return PlanResult{.plan = std::move(plan), .stats = stats};
            }
        }

        // After the goal test, as above: a goal already in hand is returned even at a budget
        // of zero, which is what the serial search does.
        if (stats.expanded >= max_expansions) {
            return make_error<PlanResult>(MathError::not_converged);
        }
        std::vector<State> batch;
        batch.reserve(layer.size());
        for (const std::size_t ni : layer) {
            batch.push_back(nodes[ni].state);
        }
        stats.expanded += layer.size();

        auto rounds = run_round(task, batch, Heuristic::h_ff, shards, exec, expand_op_id);
        if (!rounds) {
            return make_error<PlanResult>(rounds.error());
        }
        for (const auto& shard : *rounds) {
            for (const auto& s : shard) {
                ++stats.generated;
                ++stats.evaluated;
                if (s.parent >= layer.size()) {
                    return make_error<PlanResult>(MathError::syntax_error);
                }
                if (s.heuristic < 0 || seen.contains(s.state)) {
                    continue;
                }
                seen.insert(s.state);
                const std::size_t parent_node = layer[s.parent];
                nodes.push_back(Node{.state = s.state,
                                     .g = add_capped(nodes[parent_node].g, s.step_cost),
                                     .parent = parent_node,
                                     .op_index = s.op_index,
                                     .is_root = false});
                open.push_back(OpenEntry{.h = s.heuristic,
                                         .not_preferred = s.preferred ? 0 : 1,
                                         .node = nodes.size() - 1});
            }
        }
    }
    return make_error<PlanResult>(MathError::undefined_value);
}

}  // namespace nimblecas::planning_dist
