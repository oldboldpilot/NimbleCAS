// NimbleCAS distributed SAT — portfolio and cube-and-conquer as task graphs over SGEE.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.sat` already has the unit of distributed work and says so: `solve_shard` is one
// portfolio worker, keyed only by (shard index, base seed), with no shared state. This module
// takes that unit and the cube decomposition below and expresses both as `nimblecas.taskdag`
// graphs, which any `Executor` runs -- the serial reference, the local parallel one, or the SGEE
// distributed executor over a broker.
//
// TWO DECOMPOSITIONS, AND THEY ANSWER DIFFERENT QUESTIONS.
//
// PORTFOLIO runs differently-configured solvers on the WHOLE formula and takes the first useful
// answer. It is the right shape when a formula is probably satisfiable, since one lucky
// configuration finishes the job and the rest are wasted work that cost nothing. Its weakness is
// that it inherits every worker's budget: if no worker is complete, the merged verdict is
// `unknown`, and `unknown` is all it can honestly say.
//
// CUBE-AND-CONQUER splits the search SPACE instead. Fixing k variables to every one of their 2^k
// combinations partitions the assignments exactly, so each cube is a smaller independent formula
// and the verdicts compose: SATISFIABLE if any cube is, UNSATISFIABLE only if EVERY cube is. That
// second half is what portfolio cannot give -- a complete UNSAT proof across a cluster -- and it
// is the reason both are here rather than just the simpler one.
//
// The partition is exact, which matters more than it might look. Because the cubes are disjoint
// and cover everything, no assignment is tested twice and none is missed, so the merged verdict
// is the verdict of the original formula rather than an approximation of it.
//
// WHAT THE MERGE GUARANTEES. Both reductions are associative and order-independent, so the answer
// does not depend on which worker finished first or how many there were. Where several cubes are
// satisfiable, the model from the LOWEST cube index wins -- a rule, not a race. A caller can run
// the same formula on one thread or a hundred machines and compare the results directly, and the
// tests do exactly that.
//
// EVERY MODEL IS VERIFIED BEFORE IT IS RETURNED, by `verify_assignment` against the ORIGINAL
// formula rather than the restricted cube. A cube's solver only ever saw a formula with extra unit
// clauses; checking its model against the real thing is what makes a returned `satisfiable`
// trustworthy, and a model that fails that check is reported as `undefined_value` rather than
// passed on.

export module nimblecas.sat_dist;

import std;
import nimblecas.core;
import nimblecas.sat;
import nimblecas.taskdag;

export namespace nimblecas::sat_dist {

// The registered operations, versioned because a wire format is part of the contract between a
// coordinator and a worker that may be running a different build.
inline constexpr std::string_view portfolio_op_id = "nimblecas.sat.portfolio_shard/v1";
inline constexpr std::string_view cube_op_id = "nimblecas.sat.cube_shard/v1";

// The most variables a cube split may fix. 2^k cubes are built eagerly, so this is the point at
// which the GRAPH itself becomes the problem rather than the search.
inline constexpr std::size_t max_cube_vars = 20;

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------
//
// Little-endian fixed-width integers, the same convention `nimblecas.taskdag_sgee` uses for
// envelope framing. A CNF travels as its literals; a verdict travels as a byte plus the model.

[[nodiscard]] auto encode_cnf(const Cnf& cnf) -> Result<Payload>;
[[nodiscard]] auto decode_cnf(std::span<const std::byte> bytes) -> Result<Cnf>;
[[nodiscard]] auto encode_result(const SatResult& r) -> Result<Payload>;
[[nodiscard]] auto decode_result(std::span<const std::byte> bytes) -> Result<SatResult>;

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

// Registers both operations. BOTH the coordinator and every worker must call this on their own
// registry at startup: the distributed executor ships an op id and arguments, never code.
[[nodiscard]] auto register_ops(TaskRegistry& reg) -> Result<void>;

// The cubes of a k-variable split: `cube_index` fixes variable v (1-based, v <= cube_vars) to bit
// (v-1) of the index. Returns the formula restricted to that cube, which is the original clauses
// with one unit clause appended per fixed variable.
//
// Exposed because it is the whole content of the decomposition, and a caller checking that the
// cubes really do partition the space should be able to look at them.
//
// `domain_error` if `cube_vars` exceeds the variable count or `max_cube_vars`, or if `cube_index`
// is out of range for that many variables.
[[nodiscard]] auto cube_of(const Cnf& cnf, std::size_t cube_vars, std::uint64_t cube_index)
    -> Result<Cnf>;

// Builds the portfolio graph: one task per shard, each carrying the whole formula and its own
// (index, seed), with no dependencies between them.
[[nodiscard]] auto build_portfolio_graph(const TaskRegistry& reg, const Cnf& cnf,
                                         std::uint64_t base_seed, std::size_t shards)
    -> Result<TaskGraph>;

// Builds the cube graph: one task per cube, 2^cube_vars of them.
[[nodiscard]] auto build_cube_graph(const TaskRegistry& reg, const Cnf& cnf, std::size_t cube_vars,
                                    std::uint64_t max_conflicts) -> Result<TaskGraph>;

// Runs the portfolio across `exec` and merges with the reduction `solve_portfolio` uses: a
// definitive UNSAT wins; otherwise the model from the lowest shard index; otherwise `unknown`.
//
// Equal to `nimblecas.sat`'s in-process `solve_portfolio` for the same formula, seed and worker
// count -- which is the contract, and what the tests assert on both executors.
//
// `domain_error` for a malformed formula or `shards == 0`.
[[nodiscard]] auto solve_portfolio_distributed(const Cnf& cnf, std::uint64_t base_seed,
                                               std::size_t shards, Executor& exec)
    -> Result<SatResult>;

// Runs the cube decomposition across `exec` and merges.
//
// The verdict is COMPLETE when every cube's own solver was complete: `satisfiable` as soon as one
// cube is, `unsatisfiable` only when all of them are, and `unknown` when no cube found a model and
// at least one hit its conflict budget -- because then the formula's status genuinely is not
// known, and saying `unsatisfiable` there would be a wrong answer rather than a cautious one.
//
// `max_conflicts` of 0 means no budget: each cube runs its CDCL to a definitive verdict, so the
// result is always complete. That is the right setting when completeness is the point and the
// wrong one when a stuck cube would hold up the run.
//
// `domain_error` for a malformed formula, or a `cube_vars` outside 1..min(num_vars, max_cube_vars).
[[nodiscard]] auto solve_cubes_distributed(const Cnf& cnf, std::size_t cube_vars,
                                           std::uint64_t max_conflicts, Executor& exec)
    -> Result<SatResult>;

}  // namespace nimblecas::sat_dist

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::sat_dist {

namespace {

constexpr std::uint64_t cnf_magic = 0x4e43535f434e4631ULL;  // "NCS_CNF1"
constexpr std::uint64_t res_magic = 0x4e43535f52455331ULL;  // "NCS_RES1"

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
    for (const std::size_t i : std::views::iota(std::size_t{0}, std::size_t{8})) {
        const auto byte =
            std::to_integer<unsigned char>(bytes[offset + i]);
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
// allocation attempt.
[[nodiscard]] auto count_is_sane(std::uint64_t n, std::size_t remaining, std::size_t per) -> bool {
    if (per == 0) {
        return false;
    }
    return n <= remaining / per;
}

// The reduction both entry points share: a definitive UNSAT wins outright; otherwise the model
// from the lowest index; otherwise unknown. Associative and order-independent, which is what lets
// results be collected in whatever order they arrive.
struct Merge {
    bool saw_unsat{false};
    bool saw_unknown{false};
    std::optional<SatResult> best;
};

auto accept(Merge& m, const SatResult& r) -> void {
    switch (r.verdict) {
        case SatVerdict::unsatisfiable:
            m.saw_unsat = true;
            break;
        case SatVerdict::satisfiable:
            if (!m.best.has_value()) {
                m.best = r;  // shards are visited in index order, so the first is the lowest
            }
            break;
        case SatVerdict::unknown:
            m.saw_unknown = true;
            break;
    }
}

}  // namespace

auto encode_cnf(const Cnf& cnf) -> Result<Payload> {
    if (cnf.num_vars == 0) {
        return make_error<Payload>(MathError::domain_error);
    }
    Payload out;
    put_u64(out, cnf_magic);
    put_u64(out, static_cast<std::uint64_t>(cnf.num_vars));
    put_u64(out, static_cast<std::uint64_t>(cnf.clauses.size()));
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        put_u64(out, static_cast<std::uint64_t>(clause.size()));
        for (const std::int64_t lit : clause) {
            put_i64(out, lit);
        }
    }
    return out;
}

auto decode_cnf(std::span<const std::byte> bytes) -> Result<Cnf> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != cnf_magic) {
        return make_error<Cnf>(MathError::syntax_error);
    }
    const auto nvars = take_u64(bytes, off);
    const auto nclauses = take_u64(bytes, off);
    if (!nvars.has_value() || !nclauses.has_value() || *nvars == 0 ||
        !count_is_sane(*nclauses, bytes.size() - off, 8)) {
        return make_error<Cnf>(MathError::syntax_error);
    }
    Cnf cnf{.num_vars = static_cast<std::size_t>(*nvars), .clauses = {}};
    cnf.clauses.resize(static_cast<std::size_t>(*nclauses));
    for (std::vector<std::int64_t>& clause : cnf.clauses) {
        const auto len = take_u64(bytes, off);
        if (!len.has_value() || !count_is_sane(*len, bytes.size() - off, 8)) {
            return make_error<Cnf>(MathError::syntax_error);
        }
        clause.resize(static_cast<std::size_t>(*len));
        for (std::int64_t& lit : clause) {
            const auto v = take_i64(bytes, off);
            if (!v.has_value()) {
                return make_error<Cnf>(MathError::syntax_error);
            }
            lit = *v;
        }
    }
    if (off != bytes.size()) {
        return make_error<Cnf>(MathError::syntax_error);
    }
    return cnf;
}

auto encode_result(const SatResult& r) -> Result<Payload> {
    Payload out;
    put_u64(out, res_magic);
    put_u64(out, static_cast<std::uint64_t>(r.verdict));
    put_u64(out, static_cast<std::uint64_t>(r.model.size()));
    for (const bool b : r.model) {
        out.push_back(static_cast<std::byte>(b ? 1 : 0));
    }
    return out;
}

auto decode_result(std::span<const std::byte> bytes) -> Result<SatResult> {
    std::size_t off = 0;
    const auto magic = take_u64(bytes, off);
    if (!magic.has_value() || *magic != res_magic) {
        return make_error<SatResult>(MathError::syntax_error);
    }
    const auto verdict = take_u64(bytes, off);
    const auto len = take_u64(bytes, off);
    if (!verdict.has_value() || !len.has_value() || *verdict > 2 ||
        !count_is_sane(*len, bytes.size() - off, 1)) {
        return make_error<SatResult>(MathError::syntax_error);
    }
    SatResult r{.verdict = static_cast<SatVerdict>(*verdict), .model = {}};
    r.model.resize(static_cast<std::size_t>(*len));
    for (std::size_t i = 0; i < r.model.size(); ++i) {
        if (off >= bytes.size()) {
            return make_error<SatResult>(MathError::syntax_error);
        }
        r.model[i] = std::to_integer<unsigned char>(bytes[off]) != 0;
        ++off;
    }
    if (off != bytes.size()) {
        return make_error<SatResult>(MathError::syntax_error);
    }
    return r;
}

auto cube_of(const Cnf& cnf, std::size_t cube_vars, std::uint64_t cube_index) -> Result<Cnf> {
    if (cube_vars == 0 || cube_vars > cnf.num_vars || cube_vars > max_cube_vars) {
        return make_error<Cnf>(MathError::domain_error);
    }
    if (cube_index >= (1ULL << cube_vars)) {
        return make_error<Cnf>(MathError::domain_error);
    }
    Cnf out = cnf;
    out.clauses.reserve(cnf.clauses.size() + cube_vars);
    for (const std::size_t i : std::views::iota(std::size_t{0}, cube_vars)) {
        // Variable i+1 fixed to bit i of the cube index, as a unit clause. Unit clauses are how a
        // cube is expressed to a solver that knows nothing about cubes: the restriction becomes
        // part of the formula, so nothing downstream needs a special case.
        const std::int64_t v = static_cast<std::int64_t>(i) + 1;
        const bool positive = ((cube_index >> i) & 1ULL) != 0ULL;
        out.clauses.push_back(std::vector<std::int64_t>{positive ? v : -v});
    }
    return out;
}

namespace {

// One portfolio worker. Literals are [formula, (index, num_shards, seed)].
[[nodiscard]] auto portfolio_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto cnf = decode_cnf(args[0]);
    if (!cnf) {
        return make_error<Payload>(cnf.error());
    }
    std::size_t off = 0;
    const auto index = take_u64(args[1], off);
    const auto shards = take_u64(args[1], off);
    const auto seed = take_u64(args[1], off);
    if (!index.has_value() || !shards.has_value() || !seed.has_value() || off != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    auto r = solve_shard(*cnf, static_cast<std::size_t>(*index), static_cast<std::size_t>(*shards),
                         *seed);
    if (!r) {
        return make_error<Payload>(r.error());
    }
    return encode_result(*r);
}

// One cube. Literals are [restricted formula, (max_conflicts)].
//
// The RESTRICTED formula travels, not the original plus a cube index, so a worker needs to know
// nothing about the decomposition -- it solves an ordinary formula. That keeps the worker side
// free of any assumption the coordinator might later change.
[[nodiscard]] auto cube_shard(std::span<const Payload> args) -> Result<Payload> {
    if (args.size() != 2) {
        return make_error<Payload>(MathError::domain_error);
    }
    auto cnf = decode_cnf(args[0]);
    if (!cnf) {
        return make_error<Payload>(cnf.error());
    }
    std::size_t off = 0;
    const auto budget = take_u64(args[1], off);
    if (!budget.has_value() || off != args[1].size()) {
        return make_error<Payload>(MathError::syntax_error);
    }
    // A budget of 0 means "no budget": run to a definitive verdict. CDCL reads it as an unlimited
    // conflict allowance, which is what makes the cube's answer complete.
    auto r = cdcl(*cnf, *budget == 0 ? std::numeric_limits<std::uint64_t>::max() : *budget);
    if (!r) {
        return make_error<Payload>(r.error());
    }
    return encode_result(*r);
}

}  // namespace

auto register_ops(TaskRegistry& reg) -> Result<void> {
    auto r = reg.register_op(OpId{portfolio_op_id}, portfolio_shard);
    if (!r) {
        return r;
    }
    return reg.register_op(OpId{cube_op_id}, cube_shard);
}

auto build_portfolio_graph(const TaskRegistry& reg, const Cnf& cnf, std::uint64_t base_seed,
                           std::size_t shards) -> Result<TaskGraph> {
    if (shards == 0) {
        return make_error<TaskGraph>(MathError::domain_error);
    }
    auto payload = encode_cnf(cnf);
    if (!payload) {
        return make_error<TaskGraph>(payload.error());
    }
    TaskGraph graph;
    for (const std::size_t s : std::views::iota(std::size_t{0}, shards)) {
        Payload args;
        put_u64(args, static_cast<std::uint64_t>(s));
        put_u64(args, static_cast<std::uint64_t>(shards));
        put_u64(args, base_seed);
        auto id = graph.add_named_task(reg, OpId{portfolio_op_id},
                                       std::vector<Payload>{*payload, std::move(args)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

auto build_cube_graph(const TaskRegistry& reg, const Cnf& cnf, std::size_t cube_vars,
                      std::uint64_t max_conflicts) -> Result<TaskGraph> {
    if (cube_vars == 0 || cube_vars > cnf.num_vars || cube_vars > max_cube_vars) {
        return make_error<TaskGraph>(MathError::domain_error);
    }
    TaskGraph graph;
    const std::uint64_t cubes = 1ULL << cube_vars;
    for (const std::uint64_t c : std::views::iota(std::uint64_t{0}, cubes)) {
        auto restricted = cube_of(cnf, cube_vars, c);
        if (!restricted) {
            return make_error<TaskGraph>(restricted.error());
        }
        auto payload = encode_cnf(*restricted);
        if (!payload) {
            return make_error<TaskGraph>(payload.error());
        }
        Payload args;
        put_u64(args, max_conflicts);
        auto id = graph.add_named_task(reg, OpId{cube_op_id},
                                       std::vector<Payload>{std::move(*payload), std::move(args)});
        if (!id) {
            return make_error<TaskGraph>(id.error());
        }
    }
    return graph;
}

namespace {

// Collects the per-task verdicts in TASK ORDER, which is index order, so "the lowest index wins"
// needs no separate bookkeeping.
[[nodiscard]] auto gather(const TaskRunResult& run) -> Result<Merge> {
    Merge m;
    for (const Result<Payload>& outcome : run.outputs) {
        if (!outcome) {
            return make_error<Merge>(outcome.error());
        }
        auto r = decode_result(*outcome);
        if (!r) {
            return make_error<Merge>(r.error());
        }
        accept(m, *r);
    }
    return m;
}

}  // namespace

auto solve_portfolio_distributed(const Cnf& cnf, std::uint64_t base_seed, std::size_t shards,
                                 Executor& exec) -> Result<SatResult> {
    if (shards == 0) {
        return make_error<SatResult>(MathError::domain_error);
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<SatResult>(reg_ok.error());
    }
    auto graph = build_portfolio_graph(reg, cnf, base_seed, shards);
    if (!graph) {
        return make_error<SatResult>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<SatResult>(run.error());
    }
    auto merged = gather(*run);
    if (!merged) {
        return make_error<SatResult>(merged.error());
    }

    // UNSAT from a complete worker is definitive and outranks a model, which cannot exist if the
    // formula is unsatisfiable -- if both appeared, one of them is a bug, and the check below is
    // what turns that into an error instead of a coin toss.
    if (merged->saw_unsat) {
        if (merged->best.has_value()) {
            return make_error<SatResult>(MathError::undefined_value);
        }
        return SatResult{.verdict = SatVerdict::unsatisfiable, .model = {}};
    }
    if (merged->best.has_value()) {
        if (!verify_assignment(cnf, merged->best->model)) {
            return make_error<SatResult>(MathError::undefined_value);
        }
        return *merged->best;
    }
    return SatResult{.verdict = SatVerdict::unknown, .model = {}};
}

auto solve_cubes_distributed(const Cnf& cnf, std::size_t cube_vars, std::uint64_t max_conflicts,
                             Executor& exec) -> Result<SatResult> {
    if (cube_vars == 0 || cube_vars > cnf.num_vars || cube_vars > max_cube_vars) {
        return make_error<SatResult>(MathError::domain_error);
    }
    TaskRegistry reg;
    auto reg_ok = register_ops(reg);
    if (!reg_ok) {
        return make_error<SatResult>(reg_ok.error());
    }
    auto graph = build_cube_graph(reg, cnf, cube_vars, max_conflicts);
    if (!graph) {
        return make_error<SatResult>(graph.error());
    }
    auto run = exec.run(*graph);
    if (!run) {
        return make_error<SatResult>(run.error());
    }
    auto merged = gather(*run);
    if (!merged) {
        return make_error<SatResult>(merged.error());
    }

    // A model from ANY cube settles it, and it is checked against the ORIGINAL formula rather than
    // the restricted one the cube's solver saw.
    if (merged->best.has_value()) {
        if (!verify_assignment(cnf, merged->best->model)) {
            return make_error<SatResult>(MathError::undefined_value);
        }
        return *merged->best;
    }
    // No model anywhere. UNSAT only if every cube reached a definitive verdict: the cubes
    // partition the assignment space exactly, so "none of them has a solution" is "the formula has
    // no solution" -- but only when none of them merely ran out of budget.
    if (merged->saw_unknown) {
        return SatResult{.verdict = SatVerdict::unknown, .model = {}};
    }
    return SatResult{.verdict = SatVerdict::unsatisfiable, .model = {}};
}

}  // namespace nimblecas::sat_dist
