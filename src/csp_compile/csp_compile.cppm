// NimbleCAS constraint-satisfaction compiler: a finite-domain CSP emitted as parallel source.
// @author Olumuyiwa Oluwasanmi
//
// This module EMITS SOURCE. It is not a JIT: it produces C++23, CUDA or Triton text that a
// downstream toolchain compiles, and it never generates PTX, loads a module, or launches a
// kernel itself. What it returns is a string, and the honesty boundary starts there -- nothing
// here has been compiled or run, so nothing here may claim a runtime property it cannot show.
//
// It compiles the DECLARATIVE `WireCsp` of `nimblecas.csp`, not the functional `Csp`. That is
// forced rather than chosen: `Csp` carries its constraints as `std::function`, and a closure
// over this process's address space cannot be read by a code generator. The declarative form
// is less expressive -- it covers the constraint kinds `ConstraintKind` names and nothing else
// -- and that limit is inherited here in full.
//
// TWO STRATEGIES, WITH DIFFERENT GUARANTEES. The difference is the whole design, so it is
// stated before anything else:
//
//   * `exhaustive` is COMPLETE. It walks the entire assignment space as a mixed-radix index
//     range, so it settles satisfiability BOTH ways: it finds a solution if one exists, and
//     its silence is a proof that none does. It is also the product of the domain sizes, which
//     grows about as fast as anything in this repository. `max_search_space` caps it, and the
//     cap is low enough to be honest rather than aspirational.
//
//   * `min_conflicts` is INCOMPLETE, and no amount of running it changes that. It is local
//     search: from a starting assignment it repeatedly re-values a variable that is currently
//     in conflict. It has no variable-count ceiling and it is what makes this module useful at
//     a size `exhaustive` cannot reach. But it CANNOT PROVE UNSATISFIABILITY. Running out of
//     flips means UNKNOWN and is emitted as unknown -- never as "no solution".
//
// CONSTRAINT GRANULARITY MATTERS TO LOCAL SEARCH, AND THE MODEL DECIDES IT. `min_conflicts`
// steers by the NUMBER OF VIOLATED CONSTRAINTS, so how a problem is carved into constraints is
// not a presentational choice -- it is the shape of the landscape the search descends. One
// `all_different` over k variables is a single constraint: violated or not, it says nothing
// about whether two variables collide or five, so every assignment that is wrong at all looks
// equally wrong and the search has no gradient to follow. The same requirement written as the
// k(k-1)/2 pairwise `not_equal` constraints it is logically equivalent to gives a count that
// falls as the assignment improves, which is exactly what the walk needs. The two models have
// identical SOLUTION SETS and very different local-search behaviour, and `exhaustive` is
// indifferent between them. This is a property of the algorithm rather than a defect in it, so
// it is documented here rather than papered over by silently re-weighting the count -- which
// would make `conflict_count` mean something other than what it says.
//
// WHY NO BACKTRACKING SEARCH IS EMITTED. The obvious third strategy would be the backtracking
// solver `nimblecas.csp` already has, and it is deliberately absent. Backtracking's efficiency
// comes from pruning: a partial assignment is abandoned, and the shape of the remaining tree
// depends on every decision taken above it. That is a sequential dependency chain, not a
// data-parallel one, and an emitted "parallel backtracking" kernel would either serialise on
// the shared bound or duplicate the whole tree per thread. The in-process `backtracking_search`
// on one core beats anything that could honestly be emitted here, and `nimblecas.csp_dist`
// already distributes it by partitioning the space instead. Claiming otherwise would be the
// dishonest option, so the strategy is simply not offered.
//
// DETERMINISM. Every emitted program is reproducible from its inputs. `exhaustive` returns the
// SMALLEST satisfying assignment index, which is the lexicographically-first assignment and
// exactly what `backtracking_search` returns for the same problem -- independent of how the
// index range was split across threads or blocks. `min_conflicts` returns the result of the
// LOWEST-INDEXED walker that succeeded, and each walker's random choices come from a
// counter-based generator seeded by its own index, so the answer does not depend on worker
// count, scheduling, or arrival order.
//
// WHAT TRITON GETS, AND WHY IT IS NOT THE SEARCH. Triton is for dense tensor kernels; a local
// search is a data-dependent walk with a branch at every step, which is precisely what it is
// not for. So the Triton target emits a SCORER rather than a solver: given a batch of candidate
// assignments it returns the number of violated constraints for each. That is a genuine dense
// kernel, it is the expensive inner quantity of any local search, and it composes with a driver
// written in Python. Emitting a walk in Triton and calling it a solver would be a claim the
// kernel could not support.

export module nimblecas.csp_compile;

import std;
import nimblecas.core;
import nimblecas.csp;

export namespace nimblecas::csp_compile {

using nimblecas::ConstraintKind;
using nimblecas::MathError;
using nimblecas::Result;
using nimblecas::WireConstraint;
using nimblecas::WireCsp;

// The language the emitted source is written in.
enum class Target : std::uint8_t {
    cpp,     // C++23, parallel over std::jthread
    cuda,    // a __global__ kernel plus a host launcher
    triton,  // a @triton.jit conflict-count scorer, not a solver -- see the header
};

// How the emitted program searches.
enum class Strategy : std::uint8_t {
    exhaustive,     // complete: settles SAT and UNSAT, but is the whole domain product
    min_conflicts,  // incomplete: scales, and can never report unsatisfiable
};

// The largest assignment space `exhaustive` may be asked to enumerate: the product of the
// domain sizes. Beyond this the emitted loop would not finish in any useful time, and emitting
// it anyway would be promising something the generated code cannot deliver.
inline constexpr std::uint64_t max_search_space = 1ULL << 32U;

// The largest shard or walker count that may be requested. Two reasons, both real. The emitted
// exhaustive scan computes a shard's range as `space * s / shards`, and with `space` bounded by
// `max_search_space` this cap keeps that product inside std::uint64_t -- an unbounded shard
// count would wrap it and hand a thread the wrong range, silently. And each shard or walker
// becomes a thread in the emitted program, so an unbounded count would emit code that tries to
// create arbitrarily many of them.
inline constexpr std::size_t max_parallelism = 1U << 16U;

// Knobs for the emitted program. The defaults are the ones the tests use.
struct EmitOptions {
    Target target{Target::cpp};
    Strategy strategy{Strategy::exhaustive};
    // exhaustive: how many index shards the work is split into. Purely a parallelism decision
    // -- it cannot change the answer, only how quickly it arrives.
    std::size_t shards{8};
    // min_conflicts: independent walkers, and each walker's step budget.
    std::size_t walkers{8};
    std::uint64_t max_steps{100000};
    // min_conflicts: the probability, in parts per 1024, of taking a random value rather than
    // the conflict-minimising one. Expressed as an integer so the emitted code needs no
    // floating point and stays bit-reproducible across toolchains.
    std::uint32_t noise_per_1024{200};
    std::uint64_t seed{0x9E3779B97F4A7C15ULL};
    // The prefix every emitted symbol carries, so two compiled problems can coexist in one
    // translation unit.
    std::string name_prefix{"nc_csp"};
};

// Whether this problem can be compiled with this strategy at all, and if not, why.
//
// `domain_error` for a malformed problem (whatever `nimblecas::validate` rejects) and for
// options that make no sense, such as zero shards or zero walkers. `overflow` when
// `exhaustive` is asked for a space larger than `max_search_space`, or when the domain product
// does not fit in a std::uint64_t at all. `not_implemented` for a combination the target
// genuinely cannot express -- Triton with a strategy, since Triton emits only the scorer.
[[nodiscard]] auto is_compilable_for(const WireCsp& w, const EmitOptions& opts) -> Result<void>;

// The size of the assignment space: the product of the domain sizes. `overflow` if that
// product exceeds std::uint64_t.
[[nodiscard]] auto search_space(const WireCsp& w) -> Result<std::uint64_t>;

// Emits the source. The returned text is self-contained apart from the standard headers it
// includes for its target, and defines the entry points documented in `entry_point_name`.
[[nodiscard]] auto emit(const WireCsp& w, const EmitOptions& opts) -> Result<std::string>;

// The name of the primary entry point the emitted source defines, given the options. Provided
// so a caller wiring the output into a build does not have to reconstruct the mangling.
[[nodiscard]] auto entry_point_name(const EmitOptions& opts) -> std::string;

// ---------------------------------------------------------------------------
// Reference implementations.
// ---------------------------------------------------------------------------
//
// These run IN THIS PROCESS the same algorithms the emitted source implements. They exist so
// the emitted program's semantics can be checked against something executable: the compiler
// cannot run its own output, so without these the only claim available would be "the text
// looks right", which is not a claim at all.

// The complete scan, returning the lexicographically-first satisfying assignment or
// std::nullopt when there is none. This is what `Strategy::exhaustive` emits.
[[nodiscard]] auto reference_exhaustive(const WireCsp& w)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// Local search, returning an assignment or std::nullopt for UNKNOWN -- never for "no solution",
// which this algorithm is not able to establish. This is what `Strategy::min_conflicts` emits.
[[nodiscard]] auto reference_min_conflicts(const WireCsp& w, std::size_t walkers,
                                           std::uint64_t max_steps, std::uint32_t noise_per_1024,
                                           std::uint64_t seed)
    -> Result<std::optional<std::vector<std::int64_t>>>;

// The number of constraints an assignment violates: zero exactly when it is a solution. This is
// the quantity the Triton kernel computes for a whole batch.
[[nodiscard]] auto conflict_count(const WireCsp& w, std::span<const std::int64_t> assignment)
    -> Result<std::uint64_t>;

}  // namespace nimblecas::csp_compile

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::csp_compile {

namespace {

// A counter-based step generator. Splitmix64 is used rather than a stateful engine so that a
// walker's k-th decision is a pure function of (seed, walker, k) -- which is what makes the
// emitted code's choices reproducible without any shared state between walkers.
[[nodiscard]] auto mix64(std::uint64_t z) noexcept -> std::uint64_t {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

// Decodes an assignment index into values, LAST variable varying fastest, so ascending index
// walks the assignments in ascending lexicographic order.
auto decode_index(const WireCsp& w, std::uint64_t index, std::vector<std::int64_t>& out) -> void {
    const std::size_t n = w.domains.size();
    out.resize(n);
    std::uint64_t rest = index;
    for (const auto i : std::views::iota(std::size_t{0}, n) | std::views::reverse) {
        const auto size = static_cast<std::uint64_t>(w.domains[i].size());
        out[i] = w.domains[i][static_cast<std::size_t>(rest % size)];
        rest /= size;
    }
}

[[nodiscard]] auto values_of(const WireConstraint& c, std::span<const std::int64_t> assignment)
    -> std::vector<std::int64_t> {
    std::vector<std::int64_t> vals;
    vals.reserve(c.scope.size());
    for (const std::size_t idx : c.scope) {
        vals.push_back(assignment[idx]);
    }
    return vals;
}

// The emitted name for a constraint's predicate helper.
[[nodiscard]] auto kind_name(ConstraintKind k) -> std::string_view {
    switch (k) {
        case ConstraintKind::not_equal:
            return "not_equal";
        case ConstraintKind::equal:
            return "equal";
        case ConstraintKind::less_equal:
            return "less_equal";
        case ConstraintKind::abs_diff_ne:
            return "abs_diff_ne";
        case ConstraintKind::all_different:
            return "all_different";
        case ConstraintKind::linear_eq:
            return "linear_eq";
        case ConstraintKind::linear_le:
            return "linear_le";
        case ConstraintKind::table_allowed:
            return "table_allowed";
    }
    return "unknown";
}

// One integer, spelled so the target actually accepts it.
//
// INT64_MIN is the trap. Written as plain decimal it is `-9223372036854775808`, which in C and
// C++ is unary minus applied to the literal `9223372036854775808` -- a value too large for
// `long long`, so the emitted table or expression would be ill-formed. It is spelled as
// `(-9223372036854775807LL - 1)` instead. Python has arbitrary-precision integers and no
// suffix, so it takes the plain decimal and must NOT be given the `LL`.
[[nodiscard]] auto emit_int(std::int64_t v, bool python) -> std::string {
    if (python) {
        return std::to_string(v);
    }
    if (v == std::numeric_limits<std::int64_t>::min()) {
        return "(-9223372036854775807LL - 1)";
    }
    return std::format("{}LL", v);
}

// Emits the expression deciding one constraint, reading values out of `arr`.
[[nodiscard]] auto emit_constraint_expr(const WireConstraint& c, std::string_view arr, bool python)
    -> std::string {
    const auto v = [&](std::size_t k) { return std::format("{}[{}]", arr, c.scope[k]); };
    switch (c.kind) {
        case ConstraintKind::not_equal:
            return std::format("({} != {})", v(0), v(1));
        case ConstraintKind::equal:
            return std::format("({} == {})", v(0), v(1));
        case ConstraintKind::less_equal:
            // x + k <= y, rearranged as x - y <= -k. `validate` has already refused a k of
            // INT64_MIN, whose negation is not representable.
            return std::format("(({} - {}) <= {})", v(0), v(1), emit_int(-c.params[0], python));
        case ConstraintKind::abs_diff_ne: {
            if (python) {
                // Triton is Python: it has no `?:`, so the C spelling below would not even
                // parse. `tl.abs` is the elementwise magnitude the walk wants anyway.
                return std::format("(tl.abs({} - {}) != {})", v(0), v(1),
                                   emit_int(c.params[0], python));
            }
            return std::format("((({} - {}) < 0 ? -({} - {}) : ({} - {})) != {})", v(0), v(1),
                               v(0), v(1), v(0), v(1), emit_int(c.params[0], python));
        }
        case ConstraintKind::all_different: {
            std::string out = "(";
            bool first = true;
            for (const auto i : std::views::iota(std::size_t{0}, c.scope.size())) {
                for (const auto j : std::views::iota(i + 1, c.scope.size())) {
                    if (!first) {
                        out += " && ";
                    }
                    first = false;
                    out += std::format("({} != {})", v(i), v(j));
                }
            }
            out += ')';
            return out;
        }
        case ConstraintKind::linear_eq:
        case ConstraintKind::linear_le: {
            std::string sum;
            for (const auto i : std::views::iota(std::size_t{0}, c.scope.size())) {
                if (i != 0) {
                    sum += " + ";
                }
                sum += std::format("({} * {})", emit_int(c.params[i], python), v(i));
            }
            const char* op = c.kind == ConstraintKind::linear_eq ? "==" : "<=";
            return std::format("(({}) {} {})", sum, op, emit_int(c.params.back(), python));
        }
        case ConstraintKind::table_allowed: {
            const std::size_t width = c.scope.size();
            std::string out = "(";
            bool first = true;
            for (std::size_t row = 0; row + width <= c.params.size(); row += width) {
                if (!first) {
                    out += " || ";
                }
                first = false;
                out += '(';
                for (const auto k : std::views::iota(std::size_t{0}, width)) {
                    if (k != 0) {
                        out += " && ";
                    }
                    out += std::format("({} == {})", v(k), emit_int(c.params[row + k], python));
                }
                out += ')';
            }
            out += ')';
            return out;
        }
    }
    return "(0)";
}

// The domain table, emitted once and shared by every entry point.
[[nodiscard]] auto emit_domains(const WireCsp& w, std::string_view p, std::string_view qual)
    -> std::string {
    std::string out;
    std::string flat;
    std::string offsets;
    std::string sizes;
    std::size_t running = 0;
    for (const auto i : std::views::iota(std::size_t{0}, w.domains.size())) {
        if (i != 0) {
            offsets += ", ";
            sizes += ", ";
        }
        offsets += std::to_string(running);
        sizes += std::to_string(w.domains[i].size());
        for (const std::int64_t v : w.domains[i]) {
            if (!flat.empty()) {
                flat += ", ";
            }
            flat += emit_int(v, /*python=*/false);
            ++running;
        }
    }
    out += std::format("{}long long {}_domain_values[] = {{{}}};\n", qual, p, flat);
    out += std::format("{}unsigned long long {}_domain_offset[] = {{{}}};\n", qual, p, offsets);
    out += std::format("{}unsigned long long {}_domain_size[] = {{{}}};\n", qual, p, sizes);
    return out;
}

// The same tables under `_host` names, for the CUDA launcher. __constant__ memory cannot be
// read from the host, so the two sides genuinely need separate copies.
[[nodiscard]] auto emit_host_domains(const WireCsp& w, std::string_view p) -> std::string {
    std::string flat;
    std::string offsets;
    std::string sizes;
    std::size_t running = 0;
    for (const auto i : std::views::iota(std::size_t{0}, w.domains.size())) {
        if (i != 0) {
            offsets += ", ";
            sizes += ", ";
        }
        offsets += std::to_string(running);
        sizes += std::to_string(w.domains[i].size());
        for (const std::int64_t v : w.domains[i]) {
            if (!flat.empty()) {
                flat += ", ";
            }
            flat += emit_int(v, /*python=*/false);
            ++running;
        }
    }
    std::string out;
    out += std::format("static constexpr long long {}_domain_values_host[] = {{{}}};\n", p, flat);
    out += std::format("static constexpr unsigned long long {}_domain_offset_host[] = {{{}}};\n", p,
                       offsets);
    out += std::format("static constexpr unsigned long long {}_domain_size_host[] = {{{}}};\n", p,
                       sizes);
    return out;
}

[[nodiscard]] auto emit_check_fn(const WireCsp& w, std::string_view p, std::string_view qual)
    -> std::string {
    std::string out;
    out += std::format("// True when `a` satisfies every constraint.\n");
    out += std::format("{}bool {}_satisfies(const long long* a) {{\n", qual, p);
    for (const auto& c : w.constraints) {
        out += std::format("    if (!{}) {{ return false; }}  // {}\n",
                           emit_constraint_expr(c, "a", /*python=*/false), kind_name(c.kind));
    }
    out += "    return true;\n}\n\n";

    out += std::format("// How many constraints `a` violates; zero exactly when it is a solution.\n");
    out += std::format("{}unsigned long long {}_conflicts(const long long* a) {{\n", qual, p);
    out += "    unsigned long long n = 0;\n";
    for (const auto& c : w.constraints) {
        out += std::format("    if (!{}) {{ ++n; }}  // {}\n", emit_constraint_expr(c, "a", /*python=*/false),
                           kind_name(c.kind));
    }
    out += "    return n;\n}\n\n";

    // The variables that appear in at least one VIOLATED constraint. Local search re-values one
    // of these rather than any variable at all: a variable in no violated constraint cannot be
    // the reason the assignment fails, so changing it is a wasted step -- and on a problem of
    // any size, wasting most steps is the difference between solving it and not.
    out += "// The variables taking part in at least one violated constraint, written into\n";
    out += "// `out`; returns how many. Re-valuing anything else cannot repair the assignment.\n";
    out += std::format("{}unsigned long long {}_conflicted(const long long* a, unsigned long long* out) {{\n",
                       qual, p);
    out += std::format("    bool m[{}];\n", w.domains.size());
    out += std::format("    for (unsigned long long i = 0; i < {}ULL; ++i) {{ m[i] = false; }}\n",
                       w.domains.size());
    for (const auto& c : w.constraints) {
        std::string marks;
        for (const std::size_t idx : c.scope) {
            marks += std::format(" m[{}] = true;", idx);
        }
        out += std::format("    if (!{}) {{{} }}  // {}\n", emit_constraint_expr(c, "a", /*python=*/false), marks,
                           kind_name(c.kind));
    }
    out += "    unsigned long long k = 0;\n";
    out += std::format("    for (unsigned long long i = 0; i < {}ULL; ++i) {{\n",
                       w.domains.size());
    out += "        if (m[i]) { out[k] = i; ++k; }\n";
    out += "    }\n    return k;\n}\n\n";
    return out;
}

[[nodiscard]] auto emit_decode_fn(const WireCsp& w, std::string_view p, std::string_view qual)
    -> std::string {
    std::string out;
    out += "// Decodes an assignment index into values, the LAST variable varying fastest, so\n";
    out += "// ascending index walks the assignments in ascending lexicographic order.\n";
    out += std::format("{}void {}_decode(unsigned long long index, long long* out) {{\n", qual, p);
    out += "    unsigned long long rest = index;\n";
    out += std::format("    for (long long i = {}; i >= 0; --i) {{\n",
                       static_cast<std::int64_t>(w.domains.size()) - 1);
    out += std::format("        const unsigned long long sz = {}_domain_size[i];\n", p);
    out += std::format(
        "        out[i] = {}_domain_values[{}_domain_offset[i] + (rest % sz)];\n", p, p);
    out += "        rest /= sz;\n";
    out += "    }\n}\n\n";
    return out;
}

}  // namespace

auto search_space(const WireCsp& w) -> Result<std::uint64_t> {
    if (auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<std::uint64_t>(ok.error());
    }
    std::uint64_t total = 1;
    for (const auto& dom : w.domains) {
        const auto size = static_cast<std::uint64_t>(dom.size());
        if (total > std::numeric_limits<std::uint64_t>::max() / size) {
            return make_error<std::uint64_t>(MathError::overflow);
        }
        total *= size;
    }
    return total;
}

auto is_compilable_for(const WireCsp& w, const EmitOptions& opts) -> Result<void> {
    if (auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<void>(ok.error());
    }
    if (opts.name_prefix.empty()) {
        return make_error<void>(MathError::domain_error);
    }
    if (opts.target == Target::triton) {
        // Triton emits the SCORER, which has no strategy. Asking for one is asking for
        // something this target does not produce, and saying so is better than emitting a
        // kernel that quietly is not what was requested.
        return make_error<void>(MathError::not_implemented);
    }
    switch (opts.strategy) {
        case Strategy::exhaustive: {
            if (opts.shards == 0 || opts.shards > max_parallelism) {
                return make_error<void>(MathError::domain_error);
            }
            const auto space = search_space(w);
            if (!space) {
                return make_error<void>(space.error());
            }
            if (*space > max_search_space) {
                return make_error<void>(MathError::overflow);
            }
            break;
        }
        case Strategy::min_conflicts: {
            if (opts.walkers == 0 || opts.walkers > max_parallelism || opts.max_steps == 0) {
                return make_error<void>(MathError::domain_error);
            }
            if (opts.noise_per_1024 > 1024) {
                return make_error<void>(MathError::domain_error);
            }
            break;
        }
    }
    return {};
}

auto entry_point_name(const EmitOptions& opts) -> std::string {
    if (opts.target == Target::triton) {
        return std::format("{}_conflict_kernel", opts.name_prefix);
    }
    switch (opts.strategy) {
        case Strategy::exhaustive:
            return std::format("{}_solve_exhaustive", opts.name_prefix);
        case Strategy::min_conflicts:
            return std::format("{}_solve_min_conflicts", opts.name_prefix);
    }
    return opts.name_prefix;
}

namespace {

[[nodiscard]] auto emit_cpp(const WireCsp& w, const EmitOptions& opts) -> std::string {
    const std::string& p = opts.name_prefix;
    const std::size_t n = w.domains.size();
    std::string out;
    out += "// Generated by nimblecas.csp_compile. Do not edit.\n";
    out += "//\n";
    if (opts.strategy == Strategy::exhaustive) {
        out += "// COMPLETE search: the whole assignment space is scanned, so a false return is a\n";
        out += "// PROOF that no solution exists, not a budget having run out.\n";
    } else {
        out += "// INCOMPLETE search: a false return means UNKNOWN. This algorithm cannot prove\n";
        out += "// unsatisfiability, and nothing it does should be read as such a proof.\n";
    }
    out += "#include <algorithm>\n#include <cstdint>\n#include <thread>\n#include <vector>\n\n";
    out += "namespace {\n\n";
    out += emit_domains(w, p, "constexpr ");
    out += '\n';
    out += emit_check_fn(w, p, "inline ");
    out += emit_decode_fn(w, p, "inline ");

    if (opts.strategy == Strategy::exhaustive) {
        out += std::format(
            "// Scans [begin, end) and reports the SMALLEST satisfying index it saw.\n"
            "inline bool {}_scan(unsigned long long begin, unsigned long long end,\n"
            "                    unsigned long long* found) {{\n"
            "    long long a[{}];\n"
            "    for (unsigned long long i = begin; i < end; ++i) {{\n"
            "        {}_decode(i, a);\n"
            "        if ({}_satisfies(a)) {{ *found = i; return true; }}\n"
            "    }}\n"
            "    return false;\n"
            "}}\n\n",
            p, n, p, p);
    } else {
        out += std::format(
            "inline unsigned long long {}_mix(unsigned long long z) {{\n"
            "    z += 0x9E3779B97F4A7C15ULL;\n"
            "    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;\n"
            "    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;\n"
            "    return z ^ (z >> 31);\n"
            "}}\n\n",
            p);
        out += std::format(
            "// One walker. Its k-th decision is a pure function of (seed, walker, k), so the\n"
            "// walk is reproducible and shares no state with any other walker.\n"
            "inline bool {}_walk(unsigned long long walker, long long* a) {{\n"
            "    for (unsigned long long v = 0; v < {}ULL; ++v) {{\n"
            "        const unsigned long long r = {}_mix({}ULL + walker * 0x1000193ULL + v);\n"
            "        a[v] = {}_domain_values[{}_domain_offset[v] + (r % {}_domain_size[v])];\n"
            "    }}\n"
            "    for (unsigned long long step = 0; step < {}ULL; ++step) {{\n"
            "        if ({}_conflicts(a) == 0) {{ return true; }}\n"
            "        const unsigned long long r = {}_mix(walker * 0x9E3779B9ULL + step);\n"
            "        // Re-value a variable that is actually IN CONFLICT. A variable in no\n"
            "        // violated constraint cannot be the reason the assignment fails, so\n"
            "        // changing it is a wasted step.\n"
            "        unsigned long long conf[{}];\n"
            "        const unsigned long long nconf = {}_conflicted(a, conf);\n"
            "        if (nconf == 0) {{ return true; }}\n"
            "        const unsigned long long var = conf[r % nconf];\n"
            "        const unsigned long long sz = {}_domain_size[var];\n"
            "        const unsigned long long off = {}_domain_offset[var];\n"
            "        if ((r >> 32) % 1024ULL < {}ULL) {{\n"
            "            // Noise: take some value regardless of whether it helps, which is what\n"
            "            // lets the walk leave a local minimum at all.\n"
            "            a[var] = {}_domain_values[off + ((r >> 40) % sz)];\n"
            "            continue;\n"
            "        }}\n"
            "        // Otherwise take the value minimising conflicts, ties to the LOWEST value,\n"
            "        // so the choice does not depend on iteration order.\n"
            "        const long long keep = a[var];\n"
            "        long long best = keep;\n"
            "        unsigned long long best_c = ~0ULL;\n"
            "        for (unsigned long long k = 0; k < sz; ++k) {{\n"
            "            a[var] = {}_domain_values[off + k];\n"
            "            const unsigned long long c = {}_conflicts(a);\n"
            "            if (c < best_c) {{ best_c = c; best = a[var]; }}\n"
            "        }}\n"
            "        a[var] = best;\n"
            "        (void)keep;\n"
            "    }}\n"
            "    return {}_conflicts(a) == 0;\n"
            "}}\n\n",
            p, n, p, opts.seed, p, p, p, opts.max_steps, p, p, n, p, p, p,
            opts.noise_per_1024, p, p, p, p);
    }
    out += "}  // namespace\n\n";

    if (opts.strategy == Strategy::exhaustive) {
        out += std::format(
            "// Returns true and fills `out` with the LEXICOGRAPHICALLY-FIRST solution, or false\n"
            "// when there is none -- and because the scan is exhaustive, false is a proof.\n"
            "// The index range is split across {} shards; the split cannot change the answer,\n"
            "// because the smallest satisfying index wins regardless of which shard found it.\n"
            "inline bool {}(long long* out) {{\n"
            "    constexpr unsigned long long space = {}ULL;\n"
            "    constexpr unsigned long long shards = {}ULL;\n"
            "    std::vector<unsigned long long> hit(shards, ~0ULL);\n"
            "    {{\n"
            "        std::vector<std::jthread> ts;\n"
            "        ts.reserve(shards);\n"
            "        for (unsigned long long s = 0; s < shards; ++s) {{\n"
            "            ts.emplace_back([s, &hit]() {{\n"
            "                const unsigned long long lo = (space * s) / shards;\n"
            "                const unsigned long long hi = (space * (s + 1)) / shards;\n"
            "                unsigned long long f = 0;\n"
            "                if ({}_scan(lo, hi, &f)) {{ hit[s] = f; }}\n"
            "            }});\n"
            "        }}\n"
            "    }}\n"
            "    const auto it = std::min_element(hit.begin(), hit.end());\n"
            "    if (it == hit.end() || *it == ~0ULL) {{ return false; }}\n"
            "    {}_decode(*it, out);\n"
            "    return true;\n"
            "}}\n",
            opts.shards, entry_point_name(opts), search_space(w).value_or(0), opts.shards, p, p);
    } else {
        out += std::format(
            "// Returns true and fills `out` on success. FALSE MEANS UNKNOWN: this search cannot\n"
            "// prove that no solution exists, and false must not be read as such a proof.\n"
            "// The answer is the LOWEST-INDEXED walker that succeeded, so it does not depend on\n"
            "// which finished first.\n"
            "inline bool {}(long long* out) {{\n"
            "    constexpr unsigned long long walkers = {}ULL;\n"
            "    std::vector<int> ok(walkers, 0);\n"
            "    std::vector<std::vector<long long>> res(walkers, std::vector<long long>({}));\n"
            "    {{\n"
            "        std::vector<std::jthread> ts;\n"
            "        ts.reserve(walkers);\n"
            "        for (unsigned long long wi = 0; wi < walkers; ++wi) {{\n"
            "            ts.emplace_back([wi, &ok, &res]() {{\n"
            "                if ({}_walk(wi, res[wi].data())) {{ ok[wi] = 1; }}\n"
            "            }});\n"
            "        }}\n"
            "    }}\n"
            "    for (unsigned long long wi = 0; wi < walkers; ++wi) {{\n"
            "        if (ok[wi] != 0) {{\n"
            "            for (unsigned long long v = 0; v < {}ULL; ++v) {{ out[v] = res[wi][v]; }}\n"
            "            return true;\n"
            "        }}\n"
            "    }}\n"
            "    return false;\n"
            "}}\n",
            entry_point_name(opts), opts.walkers, n, p, n);
    }
    return out;
}

[[nodiscard]] auto emit_cuda(const WireCsp& w, const EmitOptions& opts) -> std::string {
    const std::string& p = opts.name_prefix;
    const std::size_t n = w.domains.size();
    std::string out;
    out += "// Generated by nimblecas.csp_compile. Do not edit.\n";
    out += "//\n";
    out += "// A CUDA kernel plus a host launcher. Nothing here has been compiled or run by the\n";
    out += "// emitter, which produces text only.\n";
    out += "#include <cstdint>\n#include <vector>\n\n";
    out += emit_domains(w, p, "__device__ __constant__ ");
    out += '\n';
    out += emit_check_fn(w, p, "__device__ inline ");
    out += emit_decode_fn(w, p, "__device__ inline ");

    if (opts.strategy == Strategy::exhaustive) {
        out += std::format(
            "// One thread per index chunk. The winner is the SMALLEST satisfying index, reduced\n"
            "// with atomicMin, so the result does not depend on which thread finished first.\n"
            "__global__ void {}_kernel(unsigned long long space, unsigned long long* best) {{\n"
            "    const unsigned long long tid =\n"
            "        blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
            "    const unsigned long long stride =\n"
            "        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;\n"
            "    long long a[{}];\n"
            "    for (unsigned long long i = tid; i < space; i += stride) {{\n"
            "        {}_decode(i, a);\n"
            "        if ({}_satisfies(a)) {{\n"
            "            atomicMin((unsigned long long*)best, i);\n"
            "            return;\n"
            "        }}\n"
            "    }}\n"
            "}}\n\n",
            p, n, p, p);
        // The launcher decodes the winning index on the HOST, and __constant__ memory is not
        // host-addressable, so the two sides genuinely need separate copies of the tables.
        // Emitted here, before the launcher that reads them.
        out += "\n// Host-side copies of the tables; the device ones are not host-addressable.\n";
        out += emit_host_domains(w, p);
        out += '\n';
        out += std::format(
            "// Host launcher. Returns true and fills `out` with the lexicographically-first\n"
            "// solution; false is a PROOF of unsatisfiability, the scan being exhaustive.\n"
            "inline bool {}(long long* out) {{\n"
            "    const unsigned long long space = {}ULL;\n"
            "    unsigned long long* d_best = nullptr;\n"
            "    unsigned long long host_best = ~0ULL;\n"
            "    if (cudaMalloc(&d_best, sizeof(unsigned long long)) != cudaSuccess) "
            "{{ return false; }}\n"
            "    cudaMemcpy(d_best, &host_best, sizeof(host_best), cudaMemcpyHostToDevice);\n"
            "    {}_kernel<<<256, 256>>>(space, d_best);\n"
            "    cudaMemcpy(&host_best, d_best, sizeof(host_best), cudaMemcpyDeviceToHost);\n"
            "    cudaFree(d_best);\n"
            "    if (host_best == ~0ULL) {{ return false; }}\n"
            "    long long a[{}];\n"
            "    unsigned long long rest = host_best;\n"
            "    for (long long i = {}; i >= 0; --i) {{\n"
            "        const unsigned long long sz = {}_domain_size_host[i];\n"
            "        a[i] = {}_domain_values_host[{}_domain_offset_host[i] + (rest % sz)];\n"
            "        rest /= sz;\n"
            "    }}\n"
            "    for (unsigned long long v = 0; v < {}ULL; ++v) {{ out[v] = a[v]; }}\n"
            "    return true;\n"
            "}}\n",
            entry_point_name(opts), search_space(w).value_or(0), p, n,
            static_cast<std::int64_t>(n) - 1, p, p, p, n);
    } else {
        out += std::format(
            "__device__ inline unsigned long long {}_mix(unsigned long long z) {{\n"
            "    z += 0x9E3779B97F4A7C15ULL;\n"
            "    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;\n"
            "    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;\n"
            "    return z ^ (z >> 31);\n"
            "}}\n\n",
            p);
        out += std::format(
            "// One walker per thread. INCOMPLETE: a thread that exhausts its steps has shown\n"
            "// nothing about satisfiability.\n"
            "__global__ void {}_kernel(int* ok, long long* results) {{\n"
            "    const unsigned long long wi =\n"
            "        blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
            "    if (wi >= {}ULL) {{ return; }}\n"
            "    long long* a = results + wi * {}ULL;\n"
            "    for (unsigned long long v = 0; v < {}ULL; ++v) {{\n"
            "        const unsigned long long r = {}_mix({}ULL + wi * 0x1000193ULL + v);\n"
            "        a[v] = {}_domain_values[{}_domain_offset[v] + (r % {}_domain_size[v])];\n"
            "    }}\n"
            "    for (unsigned long long step = 0; step < {}ULL; ++step) {{\n"
            "        if ({}_conflicts(a) == 0) {{ ok[wi] = 1; return; }}\n"
            "        const unsigned long long r = {}_mix(wi * 0x9E3779B9ULL + step);\n"
            "        // Re-value a variable that is actually IN CONFLICT; see the host walk.\n"
            "        unsigned long long conf[{}];\n"
            "        const unsigned long long nconf = {}_conflicted(a, conf);\n"
            "        if (nconf == 0) {{ ok[wi] = 1; return; }}\n"
            "        const unsigned long long var = conf[r % nconf];\n"
            "        const unsigned long long sz = {}_domain_size[var];\n"
            "        const unsigned long long off = {}_domain_offset[var];\n"
            "        if ((r >> 32) % 1024ULL < {}ULL) {{\n"
            "            a[var] = {}_domain_values[off + ((r >> 40) % sz)];\n"
            "            continue;\n"
            "        }}\n"
            "        long long best = a[var];\n"
            "        unsigned long long best_c = ~0ULL;\n"
            "        for (unsigned long long k = 0; k < sz; ++k) {{\n"
            "            a[var] = {}_domain_values[off + k];\n"
            "            const unsigned long long c = {}_conflicts(a);\n"
            "            if (c < best_c) {{ best_c = c; best = a[var]; }}\n"
            "        }}\n"
            "        a[var] = best;\n"
            "    }}\n"
            "    ok[wi] = ({}_conflicts(a) == 0) ? 1 : 0;\n"
            "}}\n\n",
            std::format("{}_kernel", entry_point_name(opts)), opts.walkers, n, n, p, opts.seed, p,
            p, p, opts.max_steps, p, p, n, p, p, p, opts.noise_per_1024, p, p, p, p);
        // The host launcher. Without it `entry_point_name` would name a symbol that does not
        // exist, and the LOWEST-INDEXED-walker determinism the header promises would have
        // nowhere to be enforced -- the kernel alone leaves every successful walker equally
        // eligible.
        out += std::format(
            "// Host launcher. Returns true and fills `out` on success. FALSE MEANS UNKNOWN:\n"
            "// local search cannot prove that no solution exists.\n"
            "// The answer is the LOWEST-INDEXED walker that succeeded, chosen here on the\n"
            "// host, so it does not depend on which thread finished first.\n"
            "inline bool {}(long long* out) {{\n"
            "    const unsigned long long walkers = {}ULL;\n"
            "    const unsigned long long nvars = {}ULL;\n"
            "    int* d_ok = nullptr;\n"
            "    long long* d_res = nullptr;\n"
            "    if (cudaMalloc(&d_ok, walkers * sizeof(int)) != cudaSuccess) {{ return false; }}\n"
            "    if (cudaMalloc(&d_res, walkers * nvars * sizeof(long long)) != cudaSuccess) {{\n"
            "        cudaFree(d_ok);\n"
            "        return false;\n"
            "    }}\n"
            "    cudaMemset(d_ok, 0, walkers * sizeof(int));\n"
            "    const unsigned long long block = 128ULL;\n"
            "    const unsigned long long grid = (walkers + block - 1ULL) / block;\n"
            "    {}_kernel<<<(unsigned)grid, (unsigned)block>>>(d_ok, d_res);\n"
            "    std::vector<int> ok(walkers, 0);\n"
            "    std::vector<long long> res(walkers * nvars, 0);\n"
            "    cudaMemcpy(ok.data(), d_ok, walkers * sizeof(int), cudaMemcpyDeviceToHost);\n"
            "    cudaMemcpy(res.data(), d_res, walkers * nvars * sizeof(long long),\n"
            "               cudaMemcpyDeviceToHost);\n"
            "    cudaFree(d_ok);\n"
            "    cudaFree(d_res);\n"
            "    for (unsigned long long wi = 0; wi < walkers; ++wi) {{\n"
            "        if (ok[wi] != 0) {{\n"
            "            for (unsigned long long v = 0; v < nvars; ++v) {{\n"
            "                out[v] = res[wi * nvars + v];\n"
            "            }}\n"
            "            return true;\n"
            "        }}\n"
            "    }}\n"
            "    return false;\n"
            "}}\n",
            entry_point_name(opts), opts.walkers, n, entry_point_name(opts));
    }
    return out;
}

[[nodiscard]] auto emit_triton(const WireCsp& w, const EmitOptions& opts) -> std::string {
    const std::string& p = opts.name_prefix;
    const std::size_t n = w.domains.size();
    std::string out;
    out += "# Generated by nimblecas.csp_compile. Do not edit.\n";
    out += "#\n";
    out += "# A SCORER, not a solver. Given a batch of candidate assignments it returns the\n";
    out += "# number of constraints each one violates -- zero meaning it is a solution.\n";
    out += "#\n";
    out += "# The search itself is deliberately NOT emitted here. A local search is a\n";
    out += "# data-dependent walk with a branch at every step, which is the opposite of what\n";
    out += "# Triton is for; the conflict count is the dense, expensive inner quantity, and it\n";
    out += "# is what a Python driver actually needs a kernel for.\n";
    out += "import triton\n";
    out += "import triton.language as tl\n\n";
    out += std::format("{}_NUM_VARS = {}\n", p, n);
    out += std::format("{}_NUM_CONSTRAINTS = {}\n\n", p, w.constraints.size());
    out += "@triton.jit\n";
    out += std::format("def {}(assignments_ptr, conflicts_ptr, n_rows, BLOCK: tl.constexpr):\n",
                       entry_point_name(opts));
    out += "    pid = tl.program_id(0)\n";
    out += "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
    out += "    mask = offs < n_rows\n";
    out += "    # One row per candidate assignment, laid out contiguously.\n";
    for (const auto i : std::views::iota(std::size_t{0}, n)) {
        out += std::format("    v{} = tl.load(assignments_ptr + offs * {} + {}, mask=mask, other=0)\n",
                           i, n, i);
    }
    out += "    conflicts = tl.zeros((BLOCK,), dtype=tl.int64)\n";
    for (const auto& c : w.constraints) {
        std::string expr = emit_constraint_expr(c, "v", /*python=*/true);
        // The emitted expression indexes as `v[k]`; in Triton each variable is its own tensor.
        expr = std::regex_replace(expr, std::regex(R"(v\[(\d+)\])"), "v$1");
        expr = std::regex_replace(expr, std::regex(R"(&&)"), "&");
        expr = std::regex_replace(expr, std::regex(R"(\|\|)"), "|");
        out += std::format("    ok = {}\n", expr);
        out += "    conflicts += tl.where(ok, 0, 1)\n";
    }
    out += "    tl.store(conflicts_ptr + offs, conflicts, mask=mask)\n";
    return out;
}

}  // namespace

auto emit(const WireCsp& w, const EmitOptions& opts) -> Result<std::string> {
    if (opts.target == Target::triton) {
        // The scorer has no strategy, so the strategy-based precondition does not apply; the
        // problem still has to be well formed and the prefix usable.
        if (auto ok = nimblecas::validate(w); !ok.has_value()) {
            return make_error<std::string>(ok.error());
        }
        if (opts.name_prefix.empty()) {
            return make_error<std::string>(MathError::domain_error);
        }
        return emit_triton(w, opts);
    }
    if (auto ok = is_compilable_for(w, opts); !ok.has_value()) {
        return make_error<std::string>(ok.error());
    }
    switch (opts.target) {
        case Target::cpp:
            return emit_cpp(w, opts);
        case Target::cuda:
            return emit_cuda(w, opts);
        case Target::triton:
            break;  // handled above
    }
    return make_error<std::string>(MathError::not_implemented);
}

auto conflict_count(const WireCsp& w, std::span<const std::int64_t> assignment)
    -> Result<std::uint64_t> {
    if (auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<std::uint64_t>(ok.error());
    }
    if (assignment.size() != w.domains.size()) {
        return make_error<std::uint64_t>(MathError::domain_error);
    }
    std::uint64_t n = 0;
    for (const auto& c : w.constraints) {
        const auto vals = values_of(c, assignment);
        if (!nimblecas::holds(c, std::span<const std::int64_t>(vals))) {
            ++n;
        }
    }
    return n;
}

auto reference_exhaustive(const WireCsp& w) -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    const auto space = search_space(w);
    if (!space) {
        return make_error<Ret>(space.error());
    }
    if (*space > max_search_space) {
        return make_error<Ret>(MathError::overflow);
    }
    std::vector<std::int64_t> a;
    for (const std::uint64_t i : std::views::iota(std::uint64_t{0}, *space)) {
        decode_index(w, i, a);
        bool ok = true;
        for (const auto& c : w.constraints) {
            const auto vals = values_of(c, a);
            if (!nimblecas::holds(c, std::span<const std::int64_t>(vals))) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return Ret{a};
        }
    }
    return Ret{std::nullopt};
}

auto reference_min_conflicts(const WireCsp& w, std::size_t walkers, std::uint64_t max_steps,
                             std::uint32_t noise_per_1024, std::uint64_t seed)
    -> Result<std::optional<std::vector<std::int64_t>>> {
    using Ret = std::optional<std::vector<std::int64_t>>;
    if (auto ok = nimblecas::validate(w); !ok.has_value()) {
        return make_error<Ret>(ok.error());
    }
    if (walkers == 0 || max_steps == 0 || noise_per_1024 > 1024) {
        return make_error<Ret>(MathError::domain_error);
    }
    const std::size_t n = w.domains.size();
    // Walkers are visited in ascending order and the first success wins, which is the same rule
    // the emitted code applies -- so the answer does not depend on how they were scheduled.
    for (const auto wi : std::views::iota(std::uint64_t{0}, static_cast<std::uint64_t>(walkers))) {
        std::vector<std::int64_t> a(n, 0);
        for (const auto v : std::views::iota(std::uint64_t{0}, static_cast<std::uint64_t>(n))) {
            const std::uint64_t r = mix64(seed + wi * 0x1000193ULL + v);
            const auto& dom = w.domains[static_cast<std::size_t>(v)];
            a[static_cast<std::size_t>(v)] =
                dom[static_cast<std::size_t>(r % static_cast<std::uint64_t>(dom.size()))];
        }
        for (const auto step : std::views::iota(std::uint64_t{0}, max_steps)) {
            const auto c0 = conflict_count(w, std::span<const std::int64_t>(a));
            if (!c0) {
                return make_error<Ret>(c0.error());
            }
            if (*c0 == 0) {
                return Ret{a};
            }
            const std::uint64_t r = mix64(wi * 0x9E3779B9ULL + step);
            // Re-value a variable that is actually IN CONFLICT, which is what makes this
            // min-conflicts rather than a random walk. A variable in no violated constraint
            // cannot be the reason the assignment fails, so re-valuing it is a wasted step --
            // and on any problem of size, wasting most steps is the difference between solving
            // it and not. The emitted code applies the identical rule.
            std::vector<std::size_t> conflicted;
            for (const auto& c : w.constraints) {
                const auto vals = values_of(c, std::span<const std::int64_t>(a));
                if (!nimblecas::holds(c, std::span<const std::int64_t>(vals))) {
                    for (const std::size_t idx : c.scope) {
                        conflicted.push_back(idx);
                    }
                }
            }
            std::ranges::sort(conflicted);
            const auto dup = std::ranges::unique(conflicted);
            conflicted.erase(dup.begin(), dup.end());
            if (conflicted.empty()) {
                return Ret{a};  // nothing is violated, so this is a solution
            }
            const auto var = conflicted[static_cast<std::size_t>(
                r % static_cast<std::uint64_t>(conflicted.size()))];
            const auto& dom = w.domains[var];
            const auto sz = static_cast<std::uint64_t>(dom.size());
            if ((r >> 32U) % 1024ULL < noise_per_1024) {
                a[var] = dom[static_cast<std::size_t>((r >> 40U) % sz)];
                continue;
            }
            std::int64_t best = a[var];
            std::uint64_t best_c = std::numeric_limits<std::uint64_t>::max();
            for (const std::int64_t candidate : dom) {
                a[var] = candidate;
                const auto c = conflict_count(w, std::span<const std::int64_t>(a));
                if (!c) {
                    return make_error<Ret>(c.error());
                }
                if (*c < best_c) {
                    best_c = *c;
                    best = candidate;
                }
            }
            a[var] = best;
        }
        const auto final_c = conflict_count(w, std::span<const std::int64_t>(a));
        if (final_c.has_value() && *final_c == 0) {
            return Ret{a};
        }
    }
    // Out of flips. This is UNKNOWN and nothing more: local search cannot show that no solution
    // exists, so this must never be reported as unsatisfiable.
    return Ret{std::nullopt};
}

}  // namespace nimblecas::csp_compile
