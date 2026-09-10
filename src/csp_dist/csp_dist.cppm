// NimbleCAS distributed finite-domain constraint satisfaction over task graphs.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.csp` solves a CSP in one process. This module expresses the same search as an
// explicit `nimblecas.taskdag` task graph, so it can run under any conforming `Executor`:
// `serial_executor`, `local_parallel_executor`, or `SgeeDistributedExecutor` across a cluster.
//
// THE DECOMPOSITION. Fix the first k variables to one combination of values from their
// domains. That yields one independent sub-problem, and over the ascending product of those k
// domains the sub-problems PARTITION the assignment space exactly: every complete assignment
// lies in exactly one of them. Nothing is examined twice and nothing is missed. Two
// consequences follow, and they are the whole reason this decomposition is the one used:
//
//   - a solution to any sub-problem is a solution to the original, and
//   - the original is unsatisfiable only if EVERY sub-problem is unsatisfiable.
//
// The second is what a portfolio of differently-seeded solvers can never give. A portfolio
// that finds nothing has only exhausted its own budgets and must honestly answer "unknown";
// an exhausted partition is a complete, cluster-wide proof that no solution exists. Because
// the sub-problems also partition the space, their solution COUNTS sum to the total, which is
// distributed model counting rather than merely distributed search.
//
// DETERMINISM. `nimblecas.csp` promises the lexicographically-first assignment. That promise
// survives distribution: `prefix_assignment` enumerates the prefixes in ascending
// lexicographic order, each shard returns the lexicographically-first extension of its own
// prefix, and the merge takes the lowest-indexed shard that found anything. The answer is
// therefore identical to `backtracking_search` on the same problem, for any shard count, any
// worker count, and any arrival order -- the merge is associative and index-ordered, never
// dependent on which shard happens to finish first.
//
// HONESTY. CSP is NP-complete and this module does not repeal that: partitioning divides the
// work, it does not shrink it. The task count is the PRODUCT of the fixed domains, and every
// task is materialised eagerly in the graph, so `max_prefix_tasks` caps it -- past that point
// graph construction and payload encoding cost more than the search they distribute. An
// unsatisfiable CSP is not an error; it is an honest empty result. What this module does NOT
// do is claim a verdict it has not earned: `count_distributed` takes a per-shard limit, and a
// count that hits that limit is reported as having done so rather than passed off as exact.

export module nimblecas.csp_dist;

import std;
import nimblecas.core;
import nimblecas.csp;
import nimblecas.taskdag;

export namespace nimblecas::csp_dist {

using nimblecas::Csp;
using nimblecas::Executor;
using nimblecas::MathError;
using nimblecas::OpId;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::TaskGraph;
using nimblecas::TaskRegistry;
using nimblecas::WireCsp;

// Operation names registered in the task registry. The executor ships an op id and arguments,
// never code, so a worker that has not registered these cannot run what it is handed.
inline constexpr std::string_view solve_op_id = "nimblecas.csp.prefix_solve/v1";
inline constexpr std::string_view count_op_id = "nimblecas.csp.prefix_count/v1";

// The largest number of prefix sub-problems a split may produce. The tasks are materialised
// eagerly, each carrying its own encoded sub-problem, so beyond this the graph itself is the
// bottleneck rather than the search.
inline constexpr std::uint64_t max_prefix_tasks = 1U << 16U;

// The outcome of a distributed count. `exact` is false exactly when some shard stopped at its
// limit, in which case `count` is a LOWER BOUND on the true number of solutions and is
// reported as such rather than as a total.
struct CountResult {
    std::uint64_t count{0};
    bool exact{true};

    [[nodiscard]] auto operator==(const CountResult&) const noexcept -> bool = default;
};

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------
//
// Little-endian fixed-width integers, matching the convention `nimblecas.taskdag_sgee` uses
// for envelope framing. Signed values travel as their two's-complement bit pattern, which
// C++20 onward defines exactly, so INT64_MIN survives a round trip unchanged.

[[nodiscard]] auto encode_csp(const WireCsp& w) -> Result<Payload>;
[[nodiscard]] auto decode_csp(std::span<const std::byte> bytes) -> Result<WireCsp>;
[[nodiscard]] auto encode_solution(const std::optional<std::vector<std::int64_t>>& s)
    -> Result<Payload>;
[[nodiscard]] auto decode_solution(std::span<const std::byte> bytes)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// Registers both operations. BOTH the coordinator and every worker must call this on their
// own registry at startup.
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

// ---------------------------------------------------------------------------
// Graph construction.
// ---------------------------------------------------------------------------

// One task per prefix sub-problem, with no dependencies between them. Exposed because a graph
// is the unit a scheduler or a profiler wants to see.
//
// `domain_error` if `fixed_vars` is zero or exceeds the variable count, or if the problem is
// malformed; `overflow` if the prefix product exceeds `max_prefix_tasks`.
[[nodiscard]] auto build_prefix_graph(const TaskRegistry& reg, const WireCsp& w,
                                      std::size_t fixed_vars) -> Result<TaskGraph>;

[[nodiscard]] auto build_count_graph(const TaskRegistry& reg, const WireCsp& w,
                                     std::size_t fixed_vars, std::uint64_t per_shard_limit)
    -> Result<TaskGraph>;

// ---------------------------------------------------------------------------
// Distributed entry points.
// ---------------------------------------------------------------------------

// The lexicographically-first solution, or std::nullopt when the CSP is unsatisfiable --
// which, because the prefixes partition the space, is a COMPLETE answer and not a budget
// having run out. Identical to `backtracking_search(as_csp(w))` for any executor.
[[nodiscard]] auto solve_distributed(const WireCsp& w, std::size_t fixed_vars, Executor& exec)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// The total number of solutions, summed over the partition. `per_shard_limit` of 0 means no
// cap, and the result is then exact; a non-zero limit caps each shard's enumeration and the
// result says whether any shard reached it.
[[nodiscard]] auto count_distributed(const WireCsp& w, std::size_t fixed_vars,
                                     std::uint64_t per_shard_limit, Executor& exec)
    -> Result<CountResult>;

}  // namespace nimblecas::csp_dist

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::csp_dist {

namespace {

constexpr std::uint64_t csp_magic = 0x4e43535f43535031ULL;  // "NCS_CSP1"
constexpr std::uint64_t sol_magic = 0x4e43535f434f4c31ULL;  // "NCS_COL1"
constexpr std::uint64_t cnt_magic = 0x4e43535f43434e31ULL;  // "NCS_CCN1"

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

}  // namespace

auto encode_csp(const WireCsp& w) -> Result<Payload> {
    if (auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<Payload>(ok.error());
    }
    Payload out;
    put_u64(out, csp_magic);
    put_u64(out, static_cast<std::uint64_t>(w.domains.size()));
    for (const auto& dom : w.domains) {
        put_u64(out, static_cast<std::uint64_t>(dom.size()));
        for (const std::int64_t v : dom) {
            put_i64(out, v);
        }
    }
    put_u64(out, static_cast<std::uint64_t>(w.constraints.size()));
    for (const auto& c : w.constraints) {
        put_u64(out, static_cast<std::uint64_t>(c.kind));
        put_u64(out, static_cast<std::uint64_t>(c.scope.size()));
        for (const std::size_t idx : c.scope) {
            put_u64(out, static_cast<std::uint64_t>(idx));
        }
        put_u64(out, static_cast<std::uint64_t>(c.params.size()));
        for (const std::int64_t p : c.params) {
            put_i64(out, p);
        }
    }
    return out;
}

auto decode_csp(std::span<const std::byte> bytes) -> Result<WireCsp> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != csp_magic) {
        return make_error<WireCsp>(MathError::syntax_error);
    }
    const auto nvars = take_u64(bytes, off);
    if (!nvars.has_value() || !count_is_sane(*nvars, bytes.size() - off, 8)) {
        return make_error<WireCsp>(MathError::syntax_error);
    }
    WireCsp w;
    w.domains.reserve(static_cast<std::size_t>(*nvars));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *nvars)) {
        const auto dsize = take_u64(bytes, off);
        if (!dsize.has_value() || !count_is_sane(*dsize, bytes.size() - off, 8)) {
            return make_error<WireCsp>(MathError::syntax_error);
        }
        std::vector<std::int64_t> dom;
        dom.reserve(static_cast<std::size_t>(*dsize));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *dsize)) {
            const auto v = take_i64(bytes, off);
            if (!v.has_value()) {
                return make_error<WireCsp>(MathError::syntax_error);
            }
            dom.push_back(*v);
        }
        w.domains.push_back(std::move(dom));
    }
    const auto ncons = take_u64(bytes, off);
    if (!ncons.has_value() || !count_is_sane(*ncons, bytes.size() - off, 24)) {
        return make_error<WireCsp>(MathError::syntax_error);
    }
    w.constraints.reserve(static_cast<std::size_t>(*ncons));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *ncons)) {
        const auto kind_raw = take_u64(bytes, off);
        if (!kind_raw.has_value() ||
            *kind_raw > static_cast<std::uint64_t>(nimblecas::ConstraintKind::table_allowed)) {
            return make_error<WireCsp>(MathError::syntax_error);
        }
        nimblecas::WireConstraint c;
        c.kind = static_cast<nimblecas::ConstraintKind>(*kind_raw);
        const auto ssize = take_u64(bytes, off);
        if (!ssize.has_value() || !count_is_sane(*ssize, bytes.size() - off, 8)) {
            return make_error<WireCsp>(MathError::syntax_error);
        }
        c.scope.reserve(static_cast<std::size_t>(*ssize));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *ssize)) {
            const auto idx = take_u64(bytes, off);
            if (!idx.has_value()) {
                return make_error<WireCsp>(MathError::syntax_error);
            }
            c.scope.push_back(static_cast<std::size_t>(*idx));
        }
        const auto psize = take_u64(bytes, off);
        if (!psize.has_value() || !count_is_sane(*psize, bytes.size() - off, 8)) {
            return make_error<WireCsp>(MathError::syntax_error);
        }
        c.params.reserve(static_cast<std::size_t>(*psize));
        for ([[maybe_unused]] const auto k : std::views::iota(std::uint64_t{0}, *psize)) {
            const auto p = take_i64(bytes, off);
            if (!p.has_value()) {
                return make_error<WireCsp>(MathError::syntax_error);
            }
            c.params.push_back(*p);
        }
        w.constraints.push_back(std::move(c));
    }
    // A problem that arrives malformed is a syntax fault on the wire, not a silently-accepted
    // one: every shard would otherwise rediscover the same fault independently.
    if (const auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<WireCsp>(MathError::syntax_error);
    }
    return w;
}

auto encode_solution(const std::optional<std::vector<std::int64_t>>& s) -> Result<Payload> {
    Payload out;
    put_u64(out, sol_magic);
    put_u64(out, s.has_value() ? 1ULL : 0ULL);
    if (s.has_value()) {
        put_u64(out, static_cast<std::uint64_t>(s->size()));
        for (const std::int64_t v : *s) {
            put_i64(out, v);
        }
    }
    return out;
}

auto decode_solution(std::span<const std::byte> bytes)
    -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != sol_magic) {
        return make_error<Ret>(MathError::syntax_error);
    }
    const auto engaged = take_u64(bytes, off);
    if (!engaged.has_value() || *engaged > 1) {
        return make_error<Ret>(MathError::syntax_error);
    }
    if (*engaged == 0) {
        return Ret{std::nullopt};
    }
    const auto n = take_u64(bytes, off);
    if (!n.has_value() || !count_is_sane(*n, bytes.size() - off, 8)) {
        return make_error<Ret>(MathError::syntax_error);
    }
    std::vector<std::int64_t> vals;
    vals.reserve(static_cast<std::size_t>(*n));
    for ([[maybe_unused]] const auto i : std::views::iota(std::uint64_t{0}, *n)) {
        const auto v = take_i64(bytes, off);
        if (!v.has_value()) {
            return make_error<Ret>(MathError::syntax_error);
        }
        vals.push_back(*v);
    }
    return Ret{std::move(vals)};
}

namespace {

// The shard body for `solve_op_id`: decode the restricted sub-problem, search it, return its
// lexicographically-first solution or none.
[[nodiscard]] auto prefix_solve(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 1) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto w = decode_csp(args[0]);
    if (!w) {
        return make_error<Payload>(w.error());
    }
    auto c = nimblecas::as_csp(*w);
    if (!c) {
        return make_error<Payload>(c.error());
    }
    auto found = nimblecas::backtracking_search_fc(*c);
    if (!found) {
        return make_error<Payload>(found.error());
    }
    return encode_solution(*found);
}

// The shard body for `count_op_id`: decode, count up to the limit, return the count and
// whether the limit was reached.
[[nodiscard]] auto prefix_count_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto w = decode_csp(args[0]);
    if (!w) {
        return make_error<Payload>(w.error());
    }
    std::size_t off = 0;
    const auto limit = take_u64(args[1], off);
    if (!limit.has_value()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    auto c = nimblecas::as_csp(*w);
    if (!c) {
        return make_error<Payload>(c.error());
    }
    auto n = nimblecas::solution_count(*c, *limit);
    if (!n) {
        return make_error<Payload>(n.error());
    }
    Payload out;
    put_u64(out, cnt_magic);
    put_u64(out, *n);
    // A shard that stopped exactly at a non-zero limit may have had more to find. Saying so is
    // the difference between a total and a lower bound.
    put_u64(out, (*limit != 0 && *n == *limit) ? 1ULL : 0ULL);
    return out;
}

[[nodiscard]] auto decode_count(std::span<const std::byte> bytes) -> Result<CountResult> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != cnt_magic) {
        return make_error<CountResult>(MathError::syntax_error);
    }
    const auto n = take_u64(bytes, off);
    const auto capped = take_u64(bytes, off);
    if (!n.has_value() || !capped.has_value() || *capped > 1) {
        return make_error<CountResult>(MathError::syntax_error);
    }
    return CountResult{.count = *n, .exact = *capped == 0};
}

// Shared shape checking and prefix enumeration for both graph builders.
[[nodiscard]] auto prefix_tasks(const WireCsp& w, std::size_t fixed_vars)
    -> Result<std::vector<WireCsp>> {
    using Ret = std::vector<WireCsp>;
    if (fixed_vars == 0 || fixed_vars > w.domains.size()) {
        return make_error<Ret>(MathError::domain_error);
    }
    const auto total = nimblecas::prefix_count(w, fixed_vars);
    if (!total) {
        return make_error<Ret>(total.error());
    }
    if (*total > max_prefix_tasks) {
        return make_error<Ret>(MathError::overflow);
    }
    Ret out;
    out.reserve(static_cast<std::size_t>(*total));
    for (const std::uint64_t i : std::views::iota(std::uint64_t{0}, *total)) {
        auto prefix = nimblecas::prefix_assignment(w, fixed_vars, i);
        if (!prefix) {
            return make_error<Ret>(prefix.error());
        }
        auto sub = nimblecas::restrict_prefix(w, std::span<const std::int64_t>(*prefix));
        if (!sub) {
            return make_error<Ret>(sub.error());
        }
        out.push_back(std::move(*sub));
    }
    return out;
}

}  // namespace

auto register_ops(TaskRegistry& reg) -> Result<void> {
    auto r = reg.register_op(OpId{solve_op_id}, prefix_solve);
    if (!r) {
        return r;
    }
    return reg.register_op(OpId{count_op_id}, prefix_count_shard);
}

auto build_prefix_graph(const TaskRegistry& reg, const WireCsp& w, std::size_t fixed_vars)
    -> Result<TaskGraph> {
    auto subs = prefix_tasks(w, fixed_vars);
    if (!subs) {
        return make_error<TaskGraph>(subs.error());
    }
    TaskGraph graph;
    for (const auto& sub : *subs) {
        auto payload = encode_csp(sub);
        if (!payload) {
            return make_error<TaskGraph>(payload.error());
        }
        auto id =
            graph.add_named_task(reg, OpId{solve_op_id}, std::vector<Payload>{std::move(*payload)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

auto build_count_graph(const TaskRegistry& reg, const WireCsp& w, std::size_t fixed_vars,
                       std::uint64_t per_shard_limit) -> Result<TaskGraph> {
    auto subs = prefix_tasks(w, fixed_vars);
    if (!subs) {
        return make_error<TaskGraph>(subs.error());
    }
    TaskGraph graph;
    for (const auto& sub : *subs) {
        auto payload = encode_csp(sub);
        if (!payload) {
            return make_error<TaskGraph>(payload.error());
        }
        Payload args;
        put_u64(args, per_shard_limit);
        auto id = graph.add_named_task(
            reg, OpId{count_op_id}, std::vector<Payload>{std::move(*payload), std::move(args)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

auto solve_distributed(const WireCsp& w, std::size_t fixed_vars, Executor& exec)
    -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    TaskRegistry reg;
    if (auto ok = register_ops(reg); !ok) {
        return make_error<Ret>(ok.error());
    }
    auto graph = build_prefix_graph(reg, w, fixed_vars);
    if (!graph) {
        return make_error<Ret>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<Ret>(run.error());
    }
    // Outputs arrive in TASK order, which is prefix order, which is ascending lexicographic.
    // The first engaged one is therefore the lexicographically-first solution overall, with no
    // dependence on completion order.
    for (const Result<Payload>& outcome : run->outputs) {
        if (!outcome) {
            return make_error<Ret>(outcome.error());
        }
        auto s = decode_solution(*outcome);
        if (!s) {
            return make_error<Ret>(s.error());
        }
        if (s->has_value()) {
            // Verify against the ORIGINAL problem, not the restricted sub-problem a shard saw.
            // A shard solves a CSP with collapsed domains; checking the answer against the
            // unrestricted original is what makes a corrupted or buggy shard a failure rather
            // than a wrong answer that looks right.
            auto original = nimblecas::as_csp(w);
            if (!original) {
                return make_error<Ret>(original.error());
            }
            const auto& assign = **s;
            if (assign.size() != w.domains.size()) {
                return make_error<Ret>(MathError::undefined_value);
            }
            for (const auto i : std::views::iota(std::size_t{0}, assign.size())) {
                const auto& dom = w.domains[i];
                if (std::ranges::find(dom, assign[i]) == dom.end()) {
                    return make_error<Ret>(MathError::undefined_value);
                }
            }
            for (const auto& c : w.constraints) {
                std::vector<std::int64_t> vals;
                vals.reserve(c.scope.size());
                for (const std::size_t idx : c.scope) {
                    vals.push_back(assign[idx]);
                }
                if (!nimblecas::holds(c, std::span<const std::int64_t>(vals))) {
                    return make_error<Ret>(MathError::undefined_value);
                }
            }
            return *s;
        }
    }
    // Every sub-problem reported no solution, and they partition the space: this is a complete
    // answer, not an exhausted budget.
    return Ret{std::nullopt};
}

auto count_distributed(const WireCsp& w, std::size_t fixed_vars, std::uint64_t per_shard_limit,
                       Executor& exec) -> Result<CountResult> {
    TaskRegistry reg;
    if (auto ok = register_ops(reg); !ok) {
        return make_error<CountResult>(ok.error());
    }
    auto graph = build_count_graph(reg, w, fixed_vars, per_shard_limit);
    if (!graph) {
        return make_error<CountResult>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<CountResult>(run.error());
    }
    CountResult merged;
    for (const Result<Payload>& outcome : run->outputs) {
        if (!outcome) {
            return make_error<CountResult>(outcome.error());
        }
        auto c = decode_count(*outcome);
        if (!c) {
            return make_error<CountResult>(c.error());
        }
        if (merged.count > std::numeric_limits<std::uint64_t>::max() - c->count) {
            return make_error<CountResult>(MathError::overflow);
        }
        merged.count += c->count;
        merged.exact = merged.exact && c->exact;
    }
    return merged;
}

}  // namespace nimblecas::csp_dist
