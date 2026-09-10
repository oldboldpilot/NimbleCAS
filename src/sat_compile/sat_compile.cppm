// NimbleCAS SAT compiler — a CNF formula emitted as parallel C++, CUDA and Triton source.
// @author Olumuyiwa Oluwasanmi
//
// `nimblecas.logic_compile` turns a Prolog program into source; this turns a CNF FORMULA into
// source, in the same three targets and with the same discipline: what comes out is code a
// compiler reads, never PTX and never a JIT.
//
// WHAT IS DATA-PARALLEL ABOUT SAT, AND WHAT IS NOT.
//
// This is the honest part, and it should be read before the API.
//
// CDCL — the algorithm every serious solver uses, and the one `nimblecas.sat` implements — CANNOT
// be emitted as a data-parallel kernel. Its power comes from clause learning, and each learned
// clause is derived by analysing the conflict that the PREVIOUS learned clauses led to. That is a
// sequential dependency chain by construction: there is no wavefront in it, no independent pieces
// to hand to lanes, and a "GPU CDCL solver" that claimed otherwise would be claiming something
// false. For a hard formula, `nimblecas.sat`'s `cdcl` on one core beats anything emitted here.
//
// What IS data-parallel is EVALUATING MANY ASSIGNMENTS AT ONCE. A clause under a fixed assignment
// is an OR of literals; a formula is an AND of clauses; and both are bitwise operations that do
// not care how many assignments are packed into a word. Pack 64 assignments into a `uint64_t` and
// one AND evaluates all 64. Pack eight of those into a vector register and one instruction
// evaluates 512. Give a GPU a million such blocks and it evaluates 64 million assignments per
// launch. That is a real algorithm with a real speedup, and it is what this module emits.
//
// The cost of that honesty is stated just as plainly: the emitted solver is COMPLETE BUT
// EXPONENTIAL. It enumerates every one of the 2^n assignments. There is no learning, no
// propagation and no backtracking to prune with -- the whole point is that there are no branches
// to diverge on. That makes it excellent up to roughly 30 variables on a GPU and roughly 26 on a
// CPU, and useless well before 60. Above that range, use `nimblecas.sat`. This module refuses
// nothing in that range because the emitted code stays CORRECT however long it would run; what it
// will not do is pretend the run is short.
//
// HOW A VARIABLE BECOMES A BIT PATTERN.
//
// Number the assignments 0, 1, 2, ... and let block b cover assignments [64b, 64b + 64). Variable
// v (1-based, so index i = v - 1) takes the value of bit i of the assignment number. Within a
// block the low six bits vary and the rest do not, so:
//
//   * i < 6: the variable's 64 values form a FIXED pattern, the same in every block. Index 0 is
//     0xAAAA...A (alternating), index 1 is 0xCCCC...C, and so on up to index 5, which is
//     0xFFFFFFFF00000000. These are compile-time constants in the emitted code.
//   * i >= 6: the variable is CONSTANT across the whole block, equal to bit (i - 6) of b. It
//     emits as `0 - ((b >> (i - 6)) & 1)`, which is all-ones or all-zeros with no branch.
//
// So every literal is a word, every clause an OR of words, and the formula an AND of clauses --
// straight-line code with no data-dependent control flow anywhere, which is exactly why it
// vectorises and exactly why it suits a GPU. The formula is BAKED IN as constants: that is the
// compilation, and it is what lets the compiler fold, reassociate and schedule the whole thing.
//
// THE ANSWER IS THE SMALLEST SATISFYING ASSIGNMENT, on every target. Not "a" satisfying
// assignment: the smallest. Lanes are searched by lowest set bit and blocks are reduced by
// minimum, so a CPU run, a SIMD run, a CUDA run and a Triton run over the same formula return the
// same number. That is a testable claim rather than a hope, and the tests check it -- a solver
// whose answer depended on the thread count would be far harder to trust.

export module nimblecas.sat_compile;

import std;
import nimblecas.core;
import nimblecas.sat;

export namespace nimblecas::sat_compile {

// The three source languages, matching `nimblecas.logic_compile`'s vocabulary so a caller that
// knows one compiler knows the other.
enum class Target : std::uint8_t { cpp, cuda, triton };

// WHICH ALGORITHM IS EMITTED, and the choice is not a tuning knob -- the two answer different
// questions and fail in different ways.
//
//   exhaustive: COMPLETE. Enumerates every assignment, so it settles satisfiable AND
//     unsatisfiable. Bit-parallel, branch-free, and capped by 2^n: excellent to about thirty
//     variables and useless well before sixty. It is the right choice only when the formula is
//     small enough that a definite UNSAT is worth having.
//
//   walksat: INCOMPLETE, and that is the whole trade. Stochastic local search cannot prove
//     unsatisfiability -- it can only ever exhibit a model or run out of flips -- so a formula it
//     does not solve is `unknown` and never `unsatisfiable`. In exchange it does not care about
//     2^n at all: it scales to thousands of variables and hundreds of thousands of clauses, which
//     is the range real formulas live in. The parallelism is ACROSS WALKERS, each an independent
//     random walk, which is as close to embarrassing as parallelism gets.
enum class Strategy : std::uint8_t { exhaustive, walksat };

// How much a variable count costs. `max_variables` is the representability limit, not a taste
// judgement: assignment numbers are unsigned 64-bit and `~0` is reserved as the "no solution"
// sentinel, so 63 variables is where the numbering itself runs out.
inline constexpr std::size_t max_variables = 63;

// Assignments evaluated by one word, and by one emitted SIMD group. The group is eight words
// wide, which is one AVX-512 register; on a narrower machine the compiler splits it, and the
// emitted code is identical either way.
inline constexpr std::uint64_t assignments_per_block = 64;
inline constexpr std::uint64_t blocks_per_group = 8;

struct CompileOptions {
    Target target{Target::cpp};
    Strategy strategy{Strategy::exhaustive};
    // Base name for the emitted entry points, e.g. "puzzle" yields `puzzle_solve`.
    std::string entry{"formula"};
    // C++ only. The vector-extension path, eight blocks at a time.
    bool emit_simd{true};
    // C++ only. A std::jthread fan-out over disjoint block ranges, reduced by minimum so the
    // answer does not depend on the thread count.
    bool emit_parallel{true};

    // ---- walksat only ----
    // Independent random walks. Each has its own seed and its own state, so they never
    // communicate; this is the axis the emitted code parallelises over.
    std::uint32_t walkers{64};
    // Flips per walker before it gives up. The single knob that trades time for solve rate.
    std::uint64_t max_flips{100000};
    // The random-walk probability, as an integer PERCENT rather than a double: the decision is
    // then exact and identical on every machine, where a float comparison could differ. 50 is a
    // reasonable default for 3-SAT; the literature's 0.5 to 0.57 band lives here.
    std::uint32_t noise_percent{50};
    // Walkers are seeded from this, deterministically, so the same base seed gives the same
    // answer whatever order they finish in.
    std::uint64_t base_seed{0x9E3779B97F4A7C15ULL};
};

// Whether this formula can be compiled at all, and why not when it cannot.
//
// `domain_error` for a malformed CNF -- zero variables, a literal of 0, or a literal naming a
// variable outside 1..num_vars. An EMPTY clause is accepted and compiles to a formula no
// assignment satisfies, because that is what an empty clause means; an empty clause LIST is
// accepted and every assignment satisfies it.
//
// The `max_variables` cap applies to `exhaustive` ALONE. It is the enumerator's limit, not SAT's:
// exhaustive search numbers every assignment in a 64-bit word and so runs out at 63 variables,
// while a random walk never numbers an assignment and has no such ceiling. Applying one cap to
// both would have made the walker refuse exactly the formulas it exists for.
[[nodiscard]] auto is_compilable_for(const Cnf& cnf, Strategy strategy) -> Result<void>;

// Shorthand for the exhaustive strategy, which is what this asked before there was a choice.
[[nodiscard]] auto is_compilable(const Cnf& cnf) -> Result<void>;

// Emits the solver source for `cnf`.
//
// The returned text is complete and standalone: it includes what it needs, defines what it uses,
// and depends on no NimbleCAS header. That is deliberate -- emitted code is meant to be dropped
// into another build, and a hidden dependency discovered at link time is the worst place to find
// one.
//
// `domain_error` when `is_compilable` refuses, or when `entry` is not a valid identifier.
[[nodiscard]] auto compile(const Cnf& cnf, const CompileOptions& opts) -> Result<std::string>;

// The smallest satisfying assignment of `cnf`, or nullopt when there is none, computed by exactly
// the algorithm the emitted code implements.
//
// This exists to be the ORACLE: emitted C++, emitted CUDA and emitted Triton must all agree with
// it, and a test can check that without a GPU or a compiler in the loop. It is not the fast path
// and is not meant to be -- for a real solve, use `nimblecas.sat`.
//
// `not_converged` if the enumeration would exceed `max_blocks` blocks, so a caller can bound the
// work rather than discover its size by waiting.
[[nodiscard]] auto reference_solve(const Cnf& cnf, std::uint64_t max_blocks)
    -> Result<std::optional<std::uint64_t>>;

// The model as `nimblecas.sat` represents it -- one bool per variable, `model[v - 1]` for
// variable v -- given an assignment number from any of the solvers above.
[[nodiscard]] auto model_of(std::uint64_t assignment, std::size_t num_vars) -> std::vector<bool>;

// What a WalkSAT run found.
struct WalkResult {
    bool found{false};
    // The lowest-indexed walker that succeeded; meaningless when `found` is false.
    std::uint32_t walker{0};
    std::vector<bool> model;
    // Walkers actually run. Fewer than requested when one succeeded early, which is the honest
    // way to report that the rest were never needed.
    std::uint32_t walkers_run{0};
};

// Runs the emitted WalkSAT algorithm in process, EXACTLY as the generated code runs it: the same
// seeding, the same move rule, the same tie-breaks, the same walker ordering.
//
// This is the ORACLE. Emitted C++ and emitted CUDA are checked against it, and because the walk
// is a pure function of (base_seed, walker index) the check is for an identical model rather than
// merely a satisfying one. It is not the fast path; for a real solve, run the emitted code.
//
// INCOMPLETE, and the return type says so: `found` false means no walk succeeded within its
// budget, which is UNKNOWN and never a proof of unsatisfiability.
//
// `domain_error` for a malformed formula, zero walkers, or a noise percentage above 100.
[[nodiscard]] auto reference_walksat(const Cnf& cnf, std::uint32_t walkers,
                                     std::uint64_t max_flips, std::uint32_t noise_percent,
                                     std::uint64_t base_seed) -> Result<WalkResult>;

}  // namespace nimblecas::sat_compile

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::sat_compile {

namespace {

// The six fixed patterns. Pattern[i] has bit L set exactly when bit i of L is set, which is the
// value of variable index i for the assignment in lane L.
constexpr std::array<std::uint64_t, 6> lane_patterns{
    0xAAAAAAAAAAAAAAAAULL,  // i = 0: every other lane
    0xCCCCCCCCCCCCCCCCULL,  // i = 1
    0xF0F0F0F0F0F0F0F0ULL,  // i = 2
    0xFF00FF00FF00FF00ULL,  // i = 3
    0xFFFF0000FFFF0000ULL,  // i = 4
    0xFFFFFFFF00000000ULL,  // i = 5: the top half of the block
};

[[nodiscard]] auto is_identifier(std::string_view s) -> bool {
    if (s.empty() || s.size() > 128) {
        return false;
    }
    if ((std::isalpha(static_cast<unsigned char>(s.front())) == 0) && s.front() != '_') {
        return false;
    }
    return std::ranges::all_of(s, [](char c) {
        return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_';
    });
}

[[nodiscard]] auto var_of(std::int64_t lit) -> std::size_t {
    const std::uint64_t mag = lit < 0 ? (0ULL - static_cast<std::uint64_t>(lit))
                                      : static_cast<std::uint64_t>(lit);
    return static_cast<std::size_t>(mag);
}

[[nodiscard]] auto hex64(std::uint64_t v) -> std::string {
    return std::format("0x{:016X}ULL", v);
}

// The word holding variable index `i`'s value across one 64-assignment block.
[[nodiscard]] auto block_word_cpp(std::size_t i, std::string_view block) -> std::string {
    if (i < lane_patterns.size()) {
        return hex64(lane_patterns[i]);
    }
    return std::format("(0ULL - (({} >> {}) & 1ULL))", block, i - lane_patterns.size());
}

// The same for a SIMD group of eight consecutive blocks, covering 512 assignments.
//
// Three regimes rather than two, because the eight LANES of the group are eight consecutive block
// numbers: indices 6, 7 and 8 vary between lanes and are therefore lane-constant vectors, and only
// index 9 upward is constant across the whole group.
[[nodiscard]] auto group_word_cpp(std::size_t i, std::string_view group, std::string_view entry)
    -> std::string {
    if (i < lane_patterns.size()) {
        return std::format("nc_{}_splat({})", entry, hex64(lane_patterns[i]));
    }
    const std::size_t b = i - lane_patterns.size();
    if (b < 3) {
        std::string lanes;
        for (const std::uint64_t lane : std::views::iota(std::uint64_t{0}, blocks_per_group)) {
            if (!lanes.empty()) {
                lanes += ", ";
            }
            lanes += ((lane >> b) & 1ULL) != 0 ? "~0ULL" : "0ULL";
        }
        return std::format("nc_{}_u64x8{{{}}}", entry, lanes);
    }
    return std::format("nc_{}_splat(0ULL - (({} >> {}) & 1ULL))", entry, group, b - 3);
}

// One clause as an OR of literal words. An EMPTY clause is falsity, and emits as the zero word --
// which is exactly right, since ANDing it leaves nothing satisfied.
[[nodiscard]] auto clause_expr(const std::vector<std::int64_t>& clause,
                               const std::function<std::string(std::size_t)>& word,
                               std::string_view zero) -> std::string {
    if (clause.empty()) {
        return std::string(zero);
    }
    std::string out;
    for (const std::int64_t lit : clause) {
        if (!out.empty()) {
            out += " | ";
        }
        const std::string w = word(var_of(lit) - 1);
        out += lit < 0 ? std::format("~({})", w) : w;
    }
    return std::format("({})", out);
}

// ---------------------------------------------------------------------------
// C++ emission.
// ---------------------------------------------------------------------------

[[nodiscard]] auto emit_cpp(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    std::string out;

    out += std::format(
        "// Generated by nimblecas.sat_compile. Do not edit.\n"
        "//\n"
        "// A complete SAT solver for one fixed formula: {} variables, {} clauses, baked in as\n"
        "// straight-line bitwise code with no data-dependent branching. Each 64-bit word carries\n"
        "// 64 candidate assignments at once, so one AND evaluates 64 of them.\n"
        "//\n"
        "// COMPLETE BUT EXPONENTIAL: this enumerates all 2^{} assignments. It has no propagation\n"
        "// and no learning to prune with -- that absence is what makes it branch-free and fast\n"
        "// per assignment, and it is also why a CDCL solver wins on any formula large enough to\n"
        "// need one.\n"
        "//\n"
        "// Every entry point returns the SMALLEST satisfying assignment, or nc_{}_none. The answer\n"
        "// does not depend on the thread count.\n\n",
        cnf.num_vars, cnf.clauses.size(), cnf.num_vars, e);

    out += "#include <cstdint>\n#include <cstddef>\n";
    if (opts.emit_parallel) {
        out += "#include <thread>\n#include <vector>\n#include <algorithm>\n";
    }
    out += "\n";

    out += std::format("inline constexpr unsigned nc_{}_vars = {};\n", e, cnf.num_vars);
    out += std::format("inline constexpr unsigned long long nc_{}_none = ~0ULL;\n", e);
    out += std::format("inline constexpr unsigned long long nc_{}_blocks = {}ULL;\n\n", e,
                       cnf.num_vars <= 6 ? 1ULL : (1ULL << (cnf.num_vars - 6)));

    // ---- scalar block kernel ----
    out += std::format(
        "// The satisfying lanes of block `b`, one bit per assignment in [64b, 64b + 64).\n"
        "[[nodiscard]] inline auto nc_{}_block(unsigned long long b) noexcept\n"
        "    -> unsigned long long {{\n"
        "    unsigned long long sat = ~0ULL;\n",
        e);
    const auto scalar_word = [](std::size_t i) { return block_word_cpp(i, "b"); };
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        out += std::format("    sat &= {};\n", clause_expr(clause, scalar_word, "0ULL"));
    }
    if (cnf.num_vars < 6) {
        // Fewer than six variables do not fill a block, so the lanes above 2^n repeat assignments
        // that do not exist. Masking them off is what keeps "smallest satisfying assignment"
        // meaningful for a tiny formula.
        out += std::format("    sat &= {};  // only 2^{} lanes are real assignments\n",
                           hex64((1ULL << (1ULL << cnf.num_vars)) - 1ULL), cnf.num_vars);
    }
    out += "    return sat;\n}\n\n";

    out += std::format(
        "// The smallest satisfying assignment in blocks [first, first + count), or nc_{0}_none.\n"
        "[[nodiscard]] inline auto {0}_solve_range(unsigned long long first,\n"
        "                                         unsigned long long count) noexcept\n"
        "    -> unsigned long long {{\n"
        "    for (unsigned long long b = first; b < first + count; ++b) {{\n"
        "        const unsigned long long sat = nc_{0}_block(b);\n"
        "        if (sat != 0ULL) {{\n"
        "            return (b << 6) | static_cast<unsigned long long>(__builtin_ctzll(sat));\n"
        "        }}\n"
        "    }}\n"
        "    return nc_{0}_none;\n"
        "}}\n\n",
        e);

    // ---- SIMD ----
    if (opts.emit_simd) {
        out += std::format(
            "// Eight blocks -- 512 assignments -- per iteration, through a compiler vector type.\n"
            "//\n"
            "// A vector extension rather than intrinsics, deliberately: the same source compiles\n"
            "// to one AVX-512 instruction, to a pair of AVX2 instructions, or to scalar code,\n"
            "// according to the -march the caller builds with, and the RESULT is identical in\n"
            "// every case. There is no ISA dispatch here to get wrong.\n"
            "//\n"
            "// The vector type and its splat carry the entry name, like everything else here.\n"
            "// Two emitted solvers must be includable in ONE translation unit, and a shared\n"
            "// unprefixed helper collides the moment anyone tries.\n"
            "using nc_{0}_u64x8 [[gnu::vector_size(64)]] = unsigned long long;\n\n"
            "[[nodiscard]] inline auto nc_{0}_splat(unsigned long long v) noexcept\n"
            "    -> nc_{0}_u64x8 {{\n"
            "    return nc_{0}_u64x8{{v, v, v, v, v, v, v, v}};\n"
            "}}\n\n"
            "// The satisfying lanes of the eight blocks 8g .. 8g+7, one word per block.\n"
            "[[nodiscard]] inline auto nc_{0}_group(unsigned long long g) noexcept\n"
            "    -> nc_{0}_u64x8 {{\n"
            "    nc_{0}_u64x8 sat = nc_{0}_splat(~0ULL);\n",
            e);
        const auto simd_word = [&e](std::size_t i) { return group_word_cpp(i, "g", e); };
        for (const std::vector<std::int64_t>& clause : cnf.clauses) {
            out += std::format(
                "    sat &= {};\n",
                clause_expr(clause, simd_word, std::format("nc_{}_splat(0ULL)", e)));
        }
        if (cnf.num_vars < 6) {
            out += std::format("    sat &= nc_{}_splat({});\n", e,
                               hex64((1ULL << (1ULL << cnf.num_vars)) - 1ULL));
        }
        out += "    return sat;\n}\n\n";

        out += std::format(
            "// As {0}_solve_range, eight blocks at a time. Same answer, by construction: the\n"
            "// lanes are scanned in block order and the lowest set bit wins.\n"
            "[[nodiscard]] inline auto {0}_solve_range_simd(unsigned long long first,\n"
            "                                              unsigned long long count) noexcept\n"
            "    -> unsigned long long {{\n"
            "    unsigned long long b = first;\n"
            "    const unsigned long long last = first + count;\n"
            "    // Lead-in until the group boundary, so groups stay aligned to multiples of 8.\n"
            "    for (; b < last && (b % 8ULL) != 0ULL; ++b) {{\n"
            "        const unsigned long long sat = nc_{0}_block(b);\n"
            "        if (sat != 0ULL) {{\n"
            "            return (b << 6) | static_cast<unsigned long long>(__builtin_ctzll(sat));\n"
            "        }}\n"
            "    }}\n"
            "    for (; b + 8ULL <= last; b += 8ULL) {{\n"
            "        const nc_{0}_u64x8 sat = nc_{0}_group(b / 8ULL);\n"
            "        for (unsigned lane = 0; lane < 8U; ++lane) {{\n"
            "            if (sat[lane] != 0ULL) {{\n"
            "                return ((b + lane) << 6) |\n"
            "                       static_cast<unsigned long long>(__builtin_ctzll(sat[lane]));\n"
            "            }}\n"
            "        }}\n"
            "    }}\n"
            "    for (; b < last; ++b) {{\n"
            "        const unsigned long long sat = nc_{0}_block(b);\n"
            "        if (sat != 0ULL) {{\n"
            "            return (b << 6) | static_cast<unsigned long long>(__builtin_ctzll(sat));\n"
            "        }}\n"
            "    }}\n"
            "    return nc_{0}_none;\n"
            "}}\n\n",
            e);
    }

    // ---- whole-space entry points ----
    const std::string range_fn =
        opts.emit_simd ? std::format("{}_solve_range_simd", e) : std::format("{}_solve_range", e);

    out += std::format(
        "// The smallest satisfying assignment of the whole formula, or nc_{}_none.\n"
        "[[nodiscard]] inline auto {}_solve() noexcept -> unsigned long long {{\n"
        "    return {}(0ULL, nc_{}_blocks);\n"
        "}}\n\n",
        e, e, range_fn, e);

    if (opts.emit_parallel) {
        out += std::format(
            "// The same answer, across `threads` workers.\n"
            "//\n"
            "// Each worker owns a disjoint block range and reports its own smallest hit; the\n"
            "// reduction is a MINIMUM, so the result is the smallest satisfying assignment\n"
            "// whatever order the workers finish in and however many there are. A worker that\n"
            "// finds nothing reports the sentinel, which is the largest value and so loses the\n"
            "// minimum without needing a special case.\n"
            "[[nodiscard]] inline auto {}_solve_parallel(unsigned threads) noexcept\n"
            "    -> unsigned long long {{\n"
            "    if (threads == 0U) {{\n"
            "        threads = 1U;\n"
            "    }}\n"
            "    const unsigned long long total = nc_{}_blocks;\n"
            "    const unsigned long long per = (total + threads - 1U) / threads;\n"
            "    std::vector<unsigned long long> found(threads, nc_{}_none);\n"
            "    {{\n"
            "        std::vector<std::jthread> workers;\n"
            "        workers.reserve(threads);\n"
            "        for (unsigned t = 0; t < threads; ++t) {{\n"
            "            const unsigned long long first = static_cast<unsigned long long>(t) * per;\n"
            "            if (first >= total) {{\n"
            "                break;\n"
            "            }}\n"
            "            const unsigned long long count =\n"
            "                (first + per > total) ? (total - first) : per;\n"
            "            workers.emplace_back([&found, t, first, count]() {{\n"
            "                found[t] = {}(first, count);\n"
            "            }});\n"
            "        }}\n"
            "    }}\n"
            "    return *std::min_element(found.begin(), found.end());\n"
            "}}\n",
            e, e, e, range_fn);
    }
    return out;
}

// ---------------------------------------------------------------------------
// CUDA emission.
// ---------------------------------------------------------------------------

[[nodiscard]] auto emit_cuda(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    std::string out;

    out += std::format(
        "// Generated by nimblecas.sat_compile. Do not edit.\n"
        "//\n"
        "// One fixed formula -- {} variables, {} clauses -- as a CUDA kernel. Each thread owns\n"
        "// one block of 64 candidate assignments and evaluates all 64 with straight-line bitwise\n"
        "// code. There is no data-dependent branching in the evaluation, so no warp divergence.\n"
        "//\n"
        "// The result is reduced with atomicMin, which makes it the SMALLEST satisfying\n"
        "// assignment and therefore independent of launch geometry and scheduling -- the same\n"
        "// number the emitted C++ returns.\n\n"
        "#include <cstdint>\n\n"
        "__device__ __constant__ unsigned long long nc_{}_none_d = ~0ULL;\n"
        "static const unsigned long long nc_{}_none = ~0ULL;\n"
        "static const unsigned long long nc_{}_blocks = {}ULL;\n\n",
        cnf.num_vars, cnf.clauses.size(), e, e, e,
        cnf.num_vars <= 6 ? 1ULL : (1ULL << (cnf.num_vars - 6)));

    out += std::format(
        "__device__ __forceinline__ unsigned long long nc_{}_block(unsigned long long b) {{\n"
        "    unsigned long long sat = ~0ULL;\n",
        e);
    const auto word = [](std::size_t i) { return block_word_cpp(i, "b"); };
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        out += std::format("    sat &= {};\n", clause_expr(clause, word, "0ULL"));
    }
    if (cnf.num_vars < 6) {
        out += std::format("    sat &= {};\n", hex64((1ULL << (1ULL << cnf.num_vars)) - 1ULL));
    }
    out += "    return sat;\n}\n\n";

    out += std::format(
        "// `out` must be initialised to ~0ULL by the host before launch.\n"
        "__global__ void {0}_kernel(unsigned long long first, unsigned long long count,\n"
        "                          unsigned long long* out) {{\n"
        "    const unsigned long long stride =\n"
        "        static_cast<unsigned long long>(blockDim.x) * gridDim.x;\n"
        "    unsigned long long best = ~0ULL;\n"
        "    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count;\n"
        "         i += stride) {{\n"
        "        const unsigned long long b = first + i;\n"
        "        const unsigned long long sat = nc_{0}_block(b);\n"
        "        if (sat != 0ULL) {{\n"
        "            const unsigned long long cand =\n"
        "                (b << 6) | static_cast<unsigned long long>(__ffsll(\n"
        "                    static_cast<long long>(sat)) - 1);\n"
        "            if (cand < best) {{\n"
        "                best = cand;\n"
        "            }}\n"
        "        }}\n"
        "    }}\n"
        "    // One atomic per thread rather than per hit: the per-thread minimum is computed in\n"
        "    // a register first, so a formula with many solutions does not serialise on the\n"
        "    // atomic.\n"
        "    if (best != ~0ULL) {{\n"
        "        atomicMin(out, best);\n"
        "    }}\n"
        "}}\n\n",
        e);

    out += std::format(
        "// Host wrapper: launches over the whole assignment space and returns the smallest\n"
        "// satisfying assignment, or nc_{}_none. Returns the sentinel on any CUDA error rather\n"
        "// than a value that would read as an answer.\n"
        "inline unsigned long long {}_solve_cuda(int grid = 1024, int block = 256) {{\n"
        "    unsigned long long* d_out = nullptr;\n"
        "    if (cudaMalloc(&d_out, sizeof(unsigned long long)) != cudaSuccess) {{\n"
        "        return nc_{}_none;\n"
        "    }}\n"
        "    const unsigned long long init = ~0ULL;\n"
        "    unsigned long long host = ~0ULL;\n"
        "    bool ok = cudaMemcpy(d_out, &init, sizeof(init), cudaMemcpyHostToDevice) ==\n"
        "              cudaSuccess;\n"
        "    if (ok) {{\n"
        "        {}_kernel<<<grid, block>>>(0ULL, nc_{}_blocks, d_out);\n"
        "        ok = cudaDeviceSynchronize() == cudaSuccess;\n"
        "    }}\n"
        "    if (ok) {{\n"
        "        ok = cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost) ==\n"
        "             cudaSuccess;\n"
        "    }}\n"
        "    cudaFree(d_out);\n"
        "    return ok ? host : nc_{}_none;\n"
        "}}\n",
        e, e, e, e, e, e);
    return out;
}

// ---------------------------------------------------------------------------
// Triton emission.
// ---------------------------------------------------------------------------

// Triton has no count-trailing-zeros, so the kernel reduces to the smallest satisfying BLOCK and
// the host resolves the lane within it. That costs one scalar step on a single block and keeps the
// kernel to operations Triton actually has -- which is better than emitting a clever bit trick
// that a future Triton release might evaluate differently.
[[nodiscard]] auto emit_triton(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    const std::uint64_t blocks = cnf.num_vars <= 6 ? 1ULL : (1ULL << (cnf.num_vars - 6));

    // Triton integers are signed, so the masks are emitted as their signed two's-complement value.
    const auto signed_of = [](std::uint64_t v) {
        return std::format("{}", static_cast<std::int64_t>(v));
    };
    const auto tri_word = [&signed_of](std::size_t i) -> std::string {
        if (i < lane_patterns.size()) {
            return std::format("tl.full(b.shape, {}, tl.int64)", signed_of(lane_patterns[i]));
        }
        return std::format("(-((b >> {}) & 1))", i - lane_patterns.size());
    };

    std::string out;
    out += std::format(
        "# Generated by nimblecas.sat_compile. Do not edit.\n"
        "#\n"
        "# One fixed formula -- {} variables, {} clauses -- as a Triton kernel. Each lane owns one\n"
        "# block of 64 candidate assignments and evaluates all 64 with bitwise operations, so a\n"
        "# tile of BLOCK lanes tests BLOCK * 64 assignments with no branching.\n"
        "#\n"
        "# The kernel reduces to the smallest satisfying BLOCK via tl.atomic_min; the host resolves\n"
        "# which lane inside that block, because Triton has no count-trailing-zeros. The answer is\n"
        "# therefore the smallest satisfying assignment, the same number the C++ and CUDA emissions\n"
        "# return.\n"
        "#\n"
        "# COMPLETE BUT EXPONENTIAL: {} blocks of 64 assignments each.\n\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "NC_VARS = {}\n"
        "NC_BLOCKS = {}\n"
        "NC_NONE = -1\n\n\n"
        "@triton.jit\n"
        "def {}_kernel(out_ptr, n_blocks, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    in_range = offs < n_blocks\n"
        "    b = offs.to(tl.int64)\n"
        "    sat = tl.full(b.shape, -1, tl.int64)\n",
        cnf.num_vars, cnf.clauses.size(), blocks, cnf.num_vars, blocks, e);

    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        out += std::format("    sat = sat & {}\n",
                           clause_expr(clause, tri_word, "tl.full(b.shape, 0, tl.int64)"));
    }
    if (cnf.num_vars < 6) {
        out += std::format("    sat = sat & tl.full(b.shape, {}, tl.int64)\n",
                           signed_of((1ULL << (1ULL << cnf.num_vars)) - 1ULL));
    }

    out += std::format(
        "    hit = in_range & (sat != 0)\n"
        "    # Lanes that missed are pushed to the maximum so they lose the minimum.\n"
        "    cand = tl.where(hit, b, tl.full(b.shape, 0x7FFFFFFFFFFFFFFF, tl.int64))\n"
        "    best = tl.min(cand, axis=0)\n"
        "    if best != 0x7FFFFFFFFFFFFFFF:\n"
        "        tl.atomic_min(out_ptr, best)\n\n\n"
        "def {}_block_mask(b):\n"
        "    \"\"\"The satisfying lanes of one block, on the host -- the same formula in Python.\n\n"
        "    Used to resolve which assignment inside the winning block satisfies the formula.\n"
        "    \"\"\"\n"
        "    mask = (1 << 64) - 1\n"
        "    sat = mask\n",
        e, e);

    const auto py_word = [](std::size_t i) -> std::string {
        if (i < lane_patterns.size()) {
            return std::format("0x{:016X}", lane_patterns[i]);
        }
        return std::format("(0 - ((b >> {}) & 1))", i - lane_patterns.size());
    };
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        out += std::format("    sat &= {} & mask\n", clause_expr(clause, py_word, "0"));
    }
    if (cnf.num_vars < 6) {
        out += std::format("    sat &= 0x{:016X}\n", (1ULL << (1ULL << cnf.num_vars)) - 1ULL);
    }

    out += std::format(
        "    return sat & mask\n\n\n"
        "def {}_solve(device='cuda', BLOCK=1024):\n"
        "    \"\"\"The smallest satisfying assignment, or None.\"\"\"\n"
        "    import torch\n"
        "    out = torch.full((1,), 0x7FFFFFFFFFFFFFFF, dtype=torch.int64, device=device)\n"
        "    grid = (triton.cdiv(NC_BLOCKS, BLOCK),)\n"
        "    {}_kernel[grid](out, NC_BLOCKS, BLOCK=BLOCK)\n"
        "    best_block = int(out[0].item())\n"
        "    if best_block == 0x7FFFFFFFFFFFFFFF:\n"
        "        return None\n"
        "    sat = {}_block_mask(best_block)\n"
        "    if sat == 0:\n"
        "        return None\n"
        "    lane = (sat & -sat).bit_length() - 1\n"
        "    return (best_block << 6) | lane\n",
        e, e, e);
    return out;
}

// ---------------------------------------------------------------------------
// WalkSAT — the tables, the walk, and the three targets.
// ---------------------------------------------------------------------------

// The flattened formula every WalkSAT emission needs.
//
// Occurrences are split BY SIGN, which is the one representation choice that matters here.
// Flipping variable v breaks exactly those clauses in which v's literal is currently the only
// true one -- and if v is currently true, those are among its POSITIVE occurrences and nowhere
// else. Splitting by sign turns the break count into a scan of one contiguous list with no sign
// test per element, which is both faster and vectorisable.
struct FlatFormula {
    std::vector<std::int32_t> lits;      // clause literals, flattened
    std::vector<std::uint32_t> cstart;   // clause c spans lits[cstart[c] .. cstart[c+1])
    std::vector<std::uint32_t> occ_pos;  // clauses in which +v appears
    std::vector<std::uint32_t> op_start; // variable v (0-based) spans occ_pos[op_start[v] ..]
    std::vector<std::uint32_t> occ_neg;
    std::vector<std::uint32_t> on_start;
};

[[nodiscard]] auto flatten(const Cnf& cnf) -> FlatFormula {
    FlatFormula f;
    f.cstart.reserve(cnf.clauses.size() + 1);
    f.cstart.push_back(0);
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        for (const std::int64_t lit : clause) {
            f.lits.push_back(static_cast<std::int32_t>(lit));
        }
        f.cstart.push_back(static_cast<std::uint32_t>(f.lits.size()));
    }

    std::vector<std::vector<std::uint32_t>> pos(cnf.num_vars);
    std::vector<std::vector<std::uint32_t>> neg(cnf.num_vars);
    for (const std::size_t c : std::views::iota(std::size_t{0}, cnf.clauses.size())) {
        for (const std::int64_t lit : cnf.clauses[c]) {
            const std::size_t v = var_of(lit) - 1;
            (lit > 0 ? pos : neg)[v].push_back(static_cast<std::uint32_t>(c));
        }
    }
    f.op_start.reserve(cnf.num_vars + 1);
    f.on_start.reserve(cnf.num_vars + 1);
    f.op_start.push_back(0);
    f.on_start.push_back(0);
    for (const std::size_t v : std::views::iota(std::size_t{0}, cnf.num_vars)) {
        f.occ_pos.insert(f.occ_pos.end(), pos[v].begin(), pos[v].end());
        f.occ_neg.insert(f.occ_neg.end(), neg[v].begin(), neg[v].end());
        f.op_start.push_back(static_cast<std::uint32_t>(f.occ_pos.size()));
        f.on_start.push_back(static_cast<std::uint32_t>(f.occ_neg.size()));
    }
    return f;
}

// A table as a C array initialiser, wrapped so a large formula does not become one enormous line.
template <typename T>
[[nodiscard]] auto table_of(std::string_view qualifier, std::string_view type,
                            std::string_view name, const std::vector<T>& values) -> std::string {
    std::string out = std::format("{}{} {}[] = {{", qualifier, type, name);
    if (values.empty()) {
        // A zero-length array is not valid C++, and an empty occurrence list is entirely normal
        // for a variable that appears with only one sign. One unused element costs nothing and
        // keeps every index expression uniform.
        return out + "0};\n";
    }
    for (const std::size_t i : std::views::iota(std::size_t{0}, values.size())) {
        if (i % 16 == 0) {
            out += "\n    ";
        }
        out += std::format("{},", values[i]);
    }
    out += "\n};\n";
    return out;
}

// The formula tables, shared by every WalkSAT target.
[[nodiscard]] auto emit_tables(const Cnf& cnf, const FlatFormula& f, std::string_view e,
                               std::string_view qualifier) -> std::string {
    std::string out;
    out += std::format("{}unsigned nc_{}_nvars = {}u;\n", qualifier, e, cnf.num_vars);
    out += std::format("{}unsigned nc_{}_nclauses = {}u;\n\n", qualifier, e,
                       cnf.clauses.size());
    out += table_of(qualifier, "int", std::format("nc_{}_lits", e), f.lits);
    out += table_of(qualifier, "unsigned", std::format("nc_{}_cstart", e), f.cstart);
    out += table_of(qualifier, "unsigned", std::format("nc_{}_occp", e), f.occ_pos);
    out += table_of(qualifier, "unsigned", std::format("nc_{}_opstart", e), f.op_start);
    out += table_of(qualifier, "unsigned", std::format("nc_{}_occn", e), f.occ_neg);
    out += table_of(qualifier, "unsigned", std::format("nc_{}_onstart", e), f.on_start);
    out += "\n";
    return out;
}

// ---------------------------------------------------------------------------
// WalkSAT — C++.
// ---------------------------------------------------------------------------

[[nodiscard]] auto emit_cpp_walksat(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    const FlatFormula f = flatten(cnf);
    std::string out;

    out += std::format(
        "// Generated by nimblecas.sat_compile. Do not edit.\n"
        "//\n"
        "// WalkSAT/SKC for one fixed formula: {} variables, {} clauses, flattened into static\n"
        "// tables so the whole search is index arithmetic over constant arrays.\n"
        "//\n"
        "// INCOMPLETE. This cannot prove unsatisfiability -- it exhibits a model or it runs out\n"
        "// of flips, and running out means UNKNOWN, never `no solution`. In exchange it does not\n"
        "// care about 2^n and scales to formulas an enumerator could not begin.\n"
        "//\n"
        "// The parallelism is ACROSS WALKERS. Each walk is an independent Markov chain with its\n"
        "// own seed and its own state; they never communicate, so W walkers are W times the\n"
        "// search with no synchronisation at all. The reduction takes the LOWEST-INDEXED walker\n"
        "// that succeeded rather than the first to finish, so the answer is reproducible however\n"
        "// many threads run it.\n\n"
        "#include <cstdint>\n"
        "#include <cstddef>\n"
        "#include <vector>\n",
        cnf.num_vars, cnf.clauses.size());
    if (opts.emit_parallel) {
        out += "#include <thread>\n";
    }
    out += "\n";
    out += emit_tables(cnf, f, e, "static const ");

    out += std::format(
        "// xorshift64*, seeded through splitmix64. Self-contained on purpose: emitted code must\n"
        "// not depend on the host's <random>, whose engines differ between standard libraries\n"
        "// and would make the same seed give different answers on different machines.\n"
        "static inline unsigned long long nc_{0}_mix(unsigned long long z) {{\n"
        "    z += 0x9E3779B97F4A7C15ULL;\n"
        "    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;\n"
        "    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;\n"
        "    return z ^ (z >> 31);\n"
        "}}\n\n"
        "static inline unsigned long long nc_{0}_next(unsigned long long* s) {{\n"
        "    unsigned long long x = *s;\n"
        "    x ^= x >> 12;\n"
        "    x ^= x << 25;\n"
        "    x ^= x >> 27;\n"
        "    *s = x;\n"
        "    return x * 0x2545F4914F6CDD1DULL;\n"
        "}}\n\n"
        "// Scratch for one walker. Allocated once per walker rather than per restart: the walk is\n"
        "// long and the allocation is not part of what is being measured.\n"
        "struct nc_{0}_scratch {{\n"
        "    std::vector<unsigned char> assign;    // one byte per variable\n"
        "    std::vector<unsigned> true_count;     // true literals in each clause\n"
        "    std::vector<unsigned> unsat;          // the currently unsatisfied clauses\n"
        "    std::vector<unsigned> unsat_at;       // where clause c sits in `unsat`\n"
        "}};\n\n"
        "static inline void nc_{0}_alloc(nc_{0}_scratch* s) {{\n"
        "    s->assign.assign(nc_{0}_nvars, 0);\n"
        "    s->true_count.assign(nc_{0}_nclauses, 0);\n"
        "    s->unsat.clear();\n"
        "    s->unsat.reserve(nc_{0}_nclauses);\n"
        "    s->unsat_at.assign(nc_{0}_nclauses, 0);\n"
        "}}\n\n",
        e);

    out += std::format(
        "// The number of clauses that flipping variable v would BREAK: those where v's literal is\n"
        "// currently the only true one. Because occurrences are split by sign, the clauses at risk\n"
        "// are exactly one contiguous list and no sign test is needed inside the loop.\n"
        "static inline unsigned nc_{0}_break_count(const nc_{0}_scratch* s, unsigned v) {{\n"
        "    const unsigned char val = s->assign[v];\n"
        "    const unsigned* list = val ? nc_{0}_occp : nc_{0}_occn;\n"
        "    const unsigned* start = val ? nc_{0}_opstart : nc_{0}_onstart;\n"
        "    const unsigned from = start[v];\n"
        "    const unsigned to = start[v + 1u];\n"
        "    const unsigned* counts = s->true_count.data();\n"
        "    unsigned broken = 0;\n",
        e);
    if (opts.emit_simd) {
        out += std::format(
            "    // Eight clause counts per iteration through a compiler vector type. This is the\n"
            "    // hot loop of WalkSAT -- a scan of a contiguous index list, gathering counts and\n"
            "    // testing each against one -- and it is the one part of the walk that is genuinely\n"
            "    // data-parallel. Everything else in the flip is a dependent gather.\n"
            "    using nc_{0}_u32x8 [[gnu::vector_size(32)]] = unsigned;\n"
            "    unsigned i = from;\n"
            "    for (; i + 8u <= to; i += 8u) {{\n"
            "        nc_{0}_u32x8 g;\n"
            "        for (unsigned k = 0; k < 8u; ++k) {{\n"
            "            g[k] = counts[list[i + k]];\n"
            "        }}\n"
            "        const nc_{0}_u32x8 hit = (g == 1u);\n"
            "        for (unsigned k = 0; k < 8u; ++k) {{\n"
            "            broken += (hit[k] != 0u) ? 1u : 0u;\n"
            "        }}\n"
            "    }}\n"
            "    for (; i < to; ++i) {{\n"
            "        broken += (counts[list[i]] == 1u) ? 1u : 0u;\n"
            "    }}\n",
            e);
    } else {
        out += std::format(
            "    for (unsigned i = from; i < to; ++i) {{\n"
            "        broken += (counts[list[i]] == 1u) ? 1u : 0u;\n"
            "    }}\n");
    }
    out += "    return broken;\n}\n\n";

    out += std::format(
        "static inline void nc_{0}_set_unsat(nc_{0}_scratch* s, unsigned c) {{\n"
        "    s->unsat_at[c] = static_cast<unsigned>(s->unsat.size());\n"
        "    s->unsat.push_back(c);\n"
        "}}\n\n"
        "static inline void nc_{0}_clear_unsat(nc_{0}_scratch* s, unsigned c) {{\n"
        "    // Swap-with-last removal, so dropping a clause from the unsatisfied set is O(1) and\n"
        "    // the set stays a dense array a random index can address directly.\n"
        "    const unsigned at = s->unsat_at[c];\n"
        "    const unsigned last = s->unsat.back();\n"
        "    s->unsat[at] = last;\n"
        "    s->unsat_at[last] = at;\n"
        "    s->unsat.pop_back();\n"
        "}}\n\n"
        "// Flips v and repairs the incremental state. The counts and the unsatisfied set are\n"
        "// maintained rather than recomputed -- that is what makes a flip cost the size of one\n"
        "// variable's occurrence list instead of the whole formula.\n"
        "static inline void nc_{0}_flip(nc_{0}_scratch* s, unsigned v) {{\n"
        "    const unsigned char was = s->assign[v];\n"
        "    s->assign[v] = static_cast<unsigned char>(was ^ 1u);\n"
        "    const unsigned* losing = was ? nc_{0}_occp : nc_{0}_occn;\n"
        "    const unsigned* lstart = was ? nc_{0}_opstart : nc_{0}_onstart;\n"
        "    const unsigned* gaining = was ? nc_{0}_occn : nc_{0}_occp;\n"
        "    const unsigned* gstart = was ? nc_{0}_onstart : nc_{0}_opstart;\n"
        "    for (unsigned i = gstart[v]; i < gstart[v + 1u]; ++i) {{\n"
        "        const unsigned c = gaining[i];\n"
        "        if (s->true_count[c]++ == 0u) {{\n"
        "            nc_{0}_clear_unsat(s, c);\n"
        "        }}\n"
        "    }}\n"
        "    for (unsigned i = lstart[v]; i < lstart[v + 1u]; ++i) {{\n"
        "        const unsigned c = losing[i];\n"
        "        if (--s->true_count[c] == 0u) {{\n"
        "            nc_{0}_set_unsat(s, c);\n"
        "        }}\n"
        "    }}\n"
        "}}\n\n",
        e);

    out += std::format(
        "// One independent walk. Returns 1 and leaves the model in `s->assign` on success.\n"
        "//\n"
        "// The move rule is WalkSAT/SKC: pick a random unsatisfied clause; if some variable in it\n"
        "// breaks nothing, take that one (a free move is never worth gambling against); otherwise\n"
        "// with probability `noise` take a random variable from the clause and with the remaining\n"
        "// probability the one that breaks fewest. The random move is what escapes local minima,\n"
        "// and without it this degenerates into greedy descent that sticks.\n"
        "static int nc_{0}_walk(nc_{0}_scratch* s, unsigned long long seed,\n"
        "                       unsigned long long max_flips, unsigned noise_percent) {{\n"
        "    unsigned long long rng = nc_{0}_mix(seed) | 1ULL;\n"
        "    for (unsigned v = 0; v < nc_{0}_nvars; ++v) {{\n"
        "        s->assign[v] = static_cast<unsigned char>(nc_{0}_next(&rng) & 1ULL);\n"
        "    }}\n"
        "    s->unsat.clear();\n"
        "    for (unsigned c = 0; c < nc_{0}_nclauses; ++c) {{\n"
        "        unsigned t = 0;\n"
        "        for (unsigned i = nc_{0}_cstart[c]; i < nc_{0}_cstart[c + 1u]; ++i) {{\n"
        "            const int lit = nc_{0}_lits[i];\n"
        "            const unsigned v = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "            t += ((lit > 0) == (s->assign[v] != 0u)) ? 1u : 0u;\n"
        "        }}\n"
        "        s->true_count[c] = t;\n"
        "        if (t == 0u) {{\n"
        "            nc_{0}_set_unsat(s, c);\n"
        "        }}\n"
        "    }}\n\n"
        "    for (unsigned long long flip = 0; flip < max_flips; ++flip) {{\n"
        "        if (s->unsat.empty()) {{\n"
        "            return 1;\n"
        "        }}\n"
        "        const unsigned pick =\n"
        "            static_cast<unsigned>(nc_{0}_next(&rng) % s->unsat.size());\n"
        "        const unsigned c = s->unsat[pick];\n"
        "        const unsigned from = nc_{0}_cstart[c];\n"
        "        const unsigned to = nc_{0}_cstart[c + 1u];\n"
        "        if (from == to) {{\n"
        "            return 0;  // an empty clause can never be satisfied by any walk\n"
        "        }}\n"
        "        unsigned best_v = 0;\n"
        "        unsigned best_break = 0xFFFFFFFFu;\n"
        "        int freebie = 0;\n"
        "        for (unsigned i = from; i < to; ++i) {{\n"
        "            const int lit = nc_{0}_lits[i];\n"
        "            const unsigned v = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "            const unsigned b = nc_{0}_break_count(s, v);\n"
        "            // Ties go to the lower variable index, so a walk is reproducible from its\n"
        "            // seed rather than depending on how the loop happened to be ordered.\n"
        "            if (b < best_break || (b == best_break && v < best_v)) {{\n"
        "                best_break = b;\n"
        "                best_v = v;\n"
        "            }}\n"
        "            if (b == 0u) {{\n"
        "                freebie = 1;\n"
        "            }}\n"
        "        }}\n"
        "        unsigned chosen = best_v;\n"
        "        if (freebie == 0 && (nc_{0}_next(&rng) % 100ULL) < noise_percent) {{\n"
        "            const unsigned idx = from + static_cast<unsigned>(\n"
        "                nc_{0}_next(&rng) % static_cast<unsigned long long>(to - from));\n"
        "            const int lit = nc_{0}_lits[idx];\n"
        "            chosen = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "        }}\n"
        "        nc_{0}_flip(s, chosen);\n"
        "    }}\n"
        "    return s->unsat.empty() ? 1 : 0;\n"
        "}}\n\n",
        e);

    out += std::format(
        "// The result of a solve: `found` is 0 or 1, and `model` is meaningful only when found.\n"
        "struct {0}_result {{\n"
        "    int found;\n"
        "    unsigned walker;                    // which walk succeeded; 0xFFFFFFFF if none\n"
        "    std::vector<unsigned char> model;   // one byte per variable\n"
        "}};\n\n"
        "// Runs walkers [first, first + count) in this thread and returns the LOWEST-INDEXED one\n"
        "// that succeeded.\n"
        "inline {0}_result {0}_solve_range(unsigned first, unsigned count,\n"
        "                                  unsigned long long max_flips, unsigned noise_percent,\n"
        "                                  unsigned long long base_seed) {{\n"
        "    {0}_result out;\n"
        "    out.found = 0;\n"
        "    out.walker = 0xFFFFFFFFu;\n"
        "    nc_{0}_scratch s;\n"
        "    nc_{0}_alloc(&s);\n"
        "    for (unsigned w = first; w < first + count; ++w) {{\n"
        "        // Each walker's seed is derived from the base and its own index, so a walker's\n"
        "        // trajectory depends on nothing but those two -- not on the thread it ran on,\n"
        "        // nor on how many walkers there were.\n"
        "        if (nc_{0}_walk(&s, base_seed + static_cast<unsigned long long>(w) * 0x9E3779B97F4A7C15ULL,\n"
        "                        max_flips, noise_percent) != 0) {{\n"
        "            out.found = 1;\n"
        "            out.walker = w;\n"
        "            out.model = s.assign;\n"
        "            return out;\n"
        "        }}\n"
        "    }}\n"
        "    return out;\n"
        "}}\n\n",
        e);

    if (opts.emit_parallel) {
        out += std::format(
            "// Every walker, across `threads` workers.\n"
            "//\n"
            "// Walkers are dealt to threads in contiguous blocks and each thread reports its own\n"
            "// lowest success; the reduction then takes the lowest of those. The result is the\n"
            "// lowest-indexed successful walker overall, which does not depend on the thread count\n"
            "// or on which thread finished first -- so a run on one core and a run on sixty-four\n"
            "// return the same model.\n"
            "inline {0}_result {0}_solve_parallel(unsigned walkers, unsigned threads,\n"
            "                                     unsigned long long max_flips,\n"
            "                                     unsigned noise_percent,\n"
            "                                     unsigned long long base_seed) {{\n"
            "    if (threads == 0u) {{\n"
            "        threads = 1u;\n"
            "    }}\n"
            "    if (walkers == 0u) {{\n"
            "        walkers = 1u;\n"
            "    }}\n"
            "    if (threads > walkers) {{\n"
            "        threads = walkers;\n"
            "    }}\n"
            "    const unsigned per = (walkers + threads - 1u) / threads;\n"
            "    std::vector<{0}_result> found(threads);\n"
            "    {{\n"
            "        std::vector<std::jthread> ws;\n"
            "        ws.reserve(threads);\n"
            "        for (unsigned t = 0; t < threads; ++t) {{\n"
            "            const unsigned first = t * per;\n"
            "            if (first >= walkers) {{\n"
            "                found[t].found = 0;\n"
            "                found[t].walker = 0xFFFFFFFFu;\n"
            "                continue;\n"
            "            }}\n"
            "            const unsigned count =\n"
            "                (first + per > walkers) ? (walkers - first) : per;\n"
            "            ws.emplace_back([&found, t, first, count, max_flips, noise_percent,\n"
            "                             base_seed]() {{\n"
            "                found[t] = {0}_solve_range(first, count, max_flips, noise_percent,\n"
            "                                           base_seed);\n"
            "            }});\n"
            "        }}\n"
            "    }}\n"
            "    {0}_result best;\n"
            "    best.found = 0;\n"
            "    best.walker = 0xFFFFFFFFu;\n"
            "    for (unsigned t = 0; t < threads; ++t) {{\n"
            "        if (found[t].found != 0 && found[t].walker < best.walker) {{\n"
            "            best = found[t];\n"
            "        }}\n"
            "    }}\n"
            "    return best;\n"
            "}}\n\n",
            e);
    }

    out += std::format(
        "// Every walker on this thread.\n"
        "inline {0}_result {0}_solve(unsigned walkers, unsigned long long max_flips,\n"
        "                            unsigned noise_percent, unsigned long long base_seed) {{\n"
        "    return {0}_solve_range(0u, walkers, max_flips, noise_percent, base_seed);\n"
        "}}\n",
        e);
    return out;
}

// ---------------------------------------------------------------------------
// WalkSAT — CUDA.
// ---------------------------------------------------------------------------

[[nodiscard]] auto emit_cuda_walksat(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    const FlatFormula f = flatten(cnf);
    std::string out;

    out += std::format(
        "// Generated by nimblecas.sat_compile. Do not edit.\n"
        "//\n"
        "// WalkSAT/SKC for one fixed formula -- {} variables, {} clauses -- with ONE THREAD PER\n"
        "// WALKER. Thousands of independent random walks run at once, which is the shape this\n"
        "// algorithm has always wanted and the reason it suits a GPU at all.\n"
        "//\n"
        "// The threads DIVERGE, and that is expected rather than a defect: each walk visits its\n"
        "// own clauses and flips its own variables. What makes it work anyway is that there is no\n"
        "// communication and no synchronisation -- a diverged warp still makes progress on every\n"
        "// lane, and the only shared write is one atomic at the end.\n"
        "//\n"
        "// INCOMPLETE: exhibits a model or reports nothing. It never proves unsatisfiability.\n"
        "//\n"
        "// SCRATCH IS CALLER-ALLOCATED because each walker needs its own counts and its own\n"
        "// unsatisfied set -- (nvars + 3 * nclauses) words per walker. Sizing that on the host,\n"
        "// where the failure is a clear allocation error, beats discovering it inside a kernel.\n\n"
        "#include <cstdint>\n\n",
        cnf.num_vars, cnf.clauses.size());

    // __managed__ rather than __constant__: the tables run to tens of kilobytes for a real
    // formula and would overflow the 64 KB constant bank, and unified memory lets the HOST read
    // the same arrays -- so a caller can verify a returned model without a second copy of the
    // formula.
    out += emit_tables(cnf, f, e, "__managed__ ");

    out += std::format(
        "__device__ __forceinline__ unsigned long long nc_{0}_mix(unsigned long long z) {{\n"
        "    z += 0x9E3779B97F4A7C15ULL;\n"
        "    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;\n"
        "    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;\n"
        "    return z ^ (z >> 31);\n"
        "}}\n\n"
        "__device__ __forceinline__ unsigned long long nc_{0}_next(unsigned long long* s) {{\n"
        "    unsigned long long x = *s;\n"
        "    x ^= x >> 12;\n"
        "    x ^= x << 25;\n"
        "    x ^= x >> 27;\n"
        "    *s = x;\n"
        "    return x * 0x2545F4914F6CDD1DULL;\n"
        "}}\n\n"
        "// Words of scratch one walker needs. The host multiplies by the walker count.\n"
        "inline unsigned long long {0}_scratch_words() {{\n"
        "    return static_cast<unsigned long long>({1}) + 3ULL * static_cast<unsigned long long>({2});\n"
        "}}\n\n",
        e, cnf.num_vars, cnf.clauses.size());

    out += std::format(
        "// `out_walker` must be initialised to 0xFFFFFFFF; `out_model` holds nvars bytes.\n"
        "__global__ void {0}_kernel(unsigned walkers, unsigned long long max_flips,\n"
        "                           unsigned noise_percent, unsigned long long base_seed,\n"
        "                           unsigned* scratch, unsigned* out_walker,\n"
        "                           unsigned char* out_model) {{\n"
        "    const unsigned w = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (w >= walkers) {{\n"
        "        return;\n"
        "    }}\n"
        "    const unsigned long long per = {1}ULL + 3ULL * {2}ULL;\n"
        "    unsigned* base = scratch + static_cast<unsigned long long>(w) * per;\n"
        "    unsigned* assign = base;\n"
        "    unsigned* tcount = base + {1}u;\n"
        "    unsigned* unsat = tcount + {2}u;\n"
        "    unsigned* unsat_at = unsat + {2}u;\n"
        "    unsigned nunsat = 0;\n\n"
        "    unsigned long long rng =\n"
        "        nc_{0}_mix(base_seed + static_cast<unsigned long long>(w) * 0x9E3779B97F4A7C15ULL) | 1ULL;\n"
        "    for (unsigned v = 0; v < nc_{0}_nvars; ++v) {{\n"
        "        assign[v] = static_cast<unsigned>(nc_{0}_next(&rng) & 1ULL);\n"
        "    }}\n"
        "    for (unsigned c = 0; c < nc_{0}_nclauses; ++c) {{\n"
        "        unsigned t = 0;\n"
        "        for (unsigned i = nc_{0}_cstart[c]; i < nc_{0}_cstart[c + 1u]; ++i) {{\n"
        "            const int lit = nc_{0}_lits[i];\n"
        "            const unsigned v = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "            t += ((lit > 0) == (assign[v] != 0u)) ? 1u : 0u;\n"
        "        }}\n"
        "        tcount[c] = t;\n"
        "        if (t == 0u) {{\n"
        "            unsat_at[c] = nunsat;\n"
        "            unsat[nunsat++] = c;\n"
        "        }}\n"
        "    }}\n\n"
        "    for (unsigned long long flip = 0; flip < max_flips && nunsat > 0u; ++flip) {{\n"
        "        const unsigned c = unsat[nc_{0}_next(&rng) % nunsat];\n"
        "        const unsigned from = nc_{0}_cstart[c];\n"
        "        const unsigned to = nc_{0}_cstart[c + 1u];\n"
        "        if (from == to) {{\n"
        "            return;\n"
        "        }}\n"
        "        unsigned best_v = 0;\n"
        "        unsigned best_break = 0xFFFFFFFFu;\n"
        "        int freebie = 0;\n"
        "        for (unsigned i = from; i < to; ++i) {{\n"
        "            const int lit = nc_{0}_lits[i];\n"
        "            const unsigned v = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "            const unsigned* list = assign[v] ? nc_{0}_occp : nc_{0}_occn;\n"
        "            const unsigned* st = assign[v] ? nc_{0}_opstart : nc_{0}_onstart;\n"
        "            unsigned b = 0;\n"
        "            for (unsigned k = st[v]; k < st[v + 1u]; ++k) {{\n"
        "                b += (tcount[list[k]] == 1u) ? 1u : 0u;\n"
        "            }}\n"
        "            if (b < best_break || (b == best_break && v < best_v)) {{\n"
        "                best_break = b;\n"
        "                best_v = v;\n"
        "            }}\n"
        "            if (b == 0u) {{\n"
        "                freebie = 1;\n"
        "            }}\n"
        "        }}\n"
        "        unsigned chosen = best_v;\n"
        "        if (freebie == 0 && (nc_{0}_next(&rng) % 100ULL) < noise_percent) {{\n"
        "            const unsigned idx =\n"
        "                from + static_cast<unsigned>(nc_{0}_next(&rng) % (to - from));\n"
        "            const int lit = nc_{0}_lits[idx];\n"
        "            chosen = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1u;\n"
        "        }}\n\n"
        "        const unsigned was = assign[chosen];\n"
        "        assign[chosen] = was ^ 1u;\n"
        "        const unsigned* losing = was ? nc_{0}_occp : nc_{0}_occn;\n"
        "        const unsigned* lst = was ? nc_{0}_opstart : nc_{0}_onstart;\n"
        "        const unsigned* gaining = was ? nc_{0}_occn : nc_{0}_occp;\n"
        "        const unsigned* gst = was ? nc_{0}_onstart : nc_{0}_opstart;\n"
        "        for (unsigned i = gst[chosen]; i < gst[chosen + 1u]; ++i) {{\n"
        "            const unsigned cc = gaining[i];\n"
        "            if (tcount[cc]++ == 0u) {{\n"
        "                const unsigned at = unsat_at[cc];\n"
        "                const unsigned last = unsat[--nunsat];\n"
        "                unsat[at] = last;\n"
        "                unsat_at[last] = at;\n"
        "            }}\n"
        "        }}\n"
        "        for (unsigned i = lst[chosen]; i < lst[chosen + 1u]; ++i) {{\n"
        "            const unsigned cc = losing[i];\n"
        "            if (--tcount[cc] == 0u) {{\n"
        "                unsat_at[cc] = nunsat;\n"
        "                unsat[nunsat++] = cc;\n"
        "            }}\n"
        "        }}\n"
        "    }}\n\n"
        "    if (nunsat == 0u) {{\n"
        "        // atomicMin, so the reported walker is the LOWEST that succeeded rather than\n"
        "        // whichever warp got there first -- the same rule the CPU emission uses, which is\n"
        "        // what lets the two be compared directly.\n"
        "        const unsigned prev = atomicMin(out_walker, w);\n"
        "        if (w < prev) {{\n"
        "            for (unsigned v = 0; v < nc_{0}_nvars; ++v) {{\n"
        "                out_model[v] = static_cast<unsigned char>(assign[v]);\n"
        "            }}\n"
        "        }}\n"
        "    }}\n"
        "}}\n\n",
        e, cnf.num_vars, cnf.clauses.size());

    out += std::format(
        "// Host wrapper. Returns the winning walker index, or 0xFFFFFFFF when no walk succeeded\n"
        "// or CUDA failed -- never a model that was not actually found.\n"
        "//\n"
        "// A NOTE ON THE MODEL WRITE-BACK. The winner is decided by atomicMin, but a lower walker\n"
        "// can finish after a higher one has already written its model, so the model in\n"
        "// `out_model` is the last writer's rather than strictly the winner's. Both satisfy the\n"
        "// formula, and the caller is expected to verify whichever it gets; if the exact\n"
        "// lowest-walker model is required, re-run that single walker on the host, whose seed is\n"
        "// a pure function of (base_seed, walker).\n"
        "inline unsigned {0}_solve_cuda(unsigned walkers, unsigned long long max_flips,\n"
        "                               unsigned noise_percent, unsigned long long base_seed,\n"
        "                               unsigned char* host_model, int block = 128) {{\n"
        "    unsigned* d_scratch = nullptr;\n"
        "    unsigned* d_walker = nullptr;\n"
        "    unsigned char* d_model = nullptr;\n"
        "    const unsigned long long words =\n"
        "        static_cast<unsigned long long>(walkers) * {0}_scratch_words();\n"
        "    if (cudaMalloc(&d_scratch, words * sizeof(unsigned)) != cudaSuccess) {{\n"
        "        return 0xFFFFFFFFu;\n"
        "    }}\n"
        "    unsigned result = 0xFFFFFFFFu;\n"
        "    bool ok = cudaMalloc(&d_walker, sizeof(unsigned)) == cudaSuccess &&\n"
        "              cudaMalloc(&d_model, nc_{0}_nvars) == cudaSuccess;\n"
        "    if (ok) {{\n"
        "        const unsigned none = 0xFFFFFFFFu;\n"
        "        ok = cudaMemcpy(d_walker, &none, sizeof(none), cudaMemcpyHostToDevice) ==\n"
        "             cudaSuccess;\n"
        "    }}\n"
        "    if (ok) {{\n"
        "        const int grid = static_cast<int>((walkers + block - 1) / block);\n"
        "        {0}_kernel<<<grid, block>>>(walkers, max_flips, noise_percent, base_seed,\n"
        "                                    d_scratch, d_walker, d_model);\n"
        "        ok = cudaDeviceSynchronize() == cudaSuccess;\n"
        "    }}\n"
        "    if (ok) {{\n"
        "        ok = cudaMemcpy(&result, d_walker, sizeof(result), cudaMemcpyDeviceToHost) ==\n"
        "             cudaSuccess;\n"
        "    }}\n"
        "    if (ok && result != 0xFFFFFFFFu && host_model != nullptr) {{\n"
        "        ok = cudaMemcpy(host_model, d_model, nc_{0}_nvars, cudaMemcpyDeviceToHost) ==\n"
        "             cudaSuccess;\n"
        "    }}\n"
        "    cudaFree(d_scratch);\n"
        "    cudaFree(d_walker);\n"
        "    cudaFree(d_model);\n"
        "    return ok ? result : 0xFFFFFFFFu;\n"
        "}}\n",
        e);
    return out;
}

// ---------------------------------------------------------------------------
// WalkSAT — Triton.
// ---------------------------------------------------------------------------

// WHAT TRITON GETS, AND WHY IT IS NOT THE WALK.
//
// A random walk is control flow: a data-dependent gather, a divergent branch, and a dynamic list
// that grows and shrinks. Triton is built for dense tensor kernels, and expressing a walk in it
// would produce something slower than the CUDA emission while being far harder to read. Emitting
// it anyway, so the target list looks complete, would be the sort of box-ticking this repository
// exists not to do.
//
// What IS tensor-shaped is the Monte Carlo part: draw many random assignments at once and score
// every clause under every one of them. That is a dense (assignments x clauses) reduction, it is
// exactly what Triton is good at, and it is genuinely useful -- restart scoring is how a portfolio
// picks promising starting points, and it is the step that dominates when restarts are frequent.
// So Triton gets the scorer, and the header says plainly that the flip loop belongs elsewhere.
[[nodiscard]] auto emit_triton_walksat(const Cnf& cnf, const CompileOptions& opts) -> std::string {
    const std::string& e = opts.entry;
    const FlatFormula f = flatten(cnf);

    const auto py_list = [](std::string_view name, const auto& values) {
        std::string out = std::format("{} = [", name);
        for (const std::size_t i : std::views::iota(std::size_t{0}, values.size())) {
            if (i % 20 == 0) {
                out += "\n    ";
            }
            out += std::format("{}, ", values[i]);
        }
        out += "\n]\n";
        return out;
    };

    std::string out;
    out += std::format(
        "# Generated by nimblecas.sat_compile. Do not edit.\n"
        "#\n"
        "# MONTE CARLO RESTART SCORING for one fixed formula: {} variables, {} clauses.\n"
        "#\n"
        "# This is deliberately NOT the walk. A random walk is control flow -- a data-dependent\n"
        "# gather, a divergent branch, and a list that grows and shrinks -- and Triton is built for\n"
        "# dense tensor kernels. Expressing the walk here would be slower than the CUDA emission\n"
        "# and much harder to read, so it is not attempted; use the CUDA target for the walk.\n"
        "#\n"
        "# What Triton is genuinely good at is the Monte Carlo step: draw many random assignments\n"
        "# and score every clause under every one of them at once. That is a dense\n"
        "# (samples x clauses) reduction, and it is the step that dominates when restarts are\n"
        "# frequent -- a portfolio uses it to pick promising starting points before walking.\n"
        "#\n"
        "# score_kernel returns, per sample, the NUMBER OF UNSATISFIED CLAUSES. Zero means the\n"
        "# sample is already a model.\n\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "NC_VARS = {}\n"
        "NC_CLAUSES = {}\n\n",
        cnf.num_vars, cnf.clauses.size(), cnf.num_vars, cnf.clauses.size());

    out += py_list(std::format("NC_{}_LITS", e), f.lits);
    out += py_list(std::format("NC_{}_CSTART", e), f.cstart);

    out += std::format(
        "\n\n@triton.jit\n"
        "def {0}_score_kernel(assign_ptr, lits_ptr, cstart_ptr, out_ptr,\n"
        "                     n_samples, n_vars, n_clauses,\n"
        "                     BLOCK: tl.constexpr):\n"
        "    \"\"\"One program per sample; counts that sample's unsatisfied clauses.\n\n"
        "    assign_ptr is (n_samples, n_vars) of int32 in {{0, 1}}.\n"
        "    out_ptr is (n_samples,) of int32.\n"
        "    \"\"\"\n"
        "    sample = tl.program_id(0)\n"
        "    if sample >= n_samples:\n"
        "        return\n"
        "    row = assign_ptr + sample * n_vars\n"
        "    unsat = 0\n"
        "    c = 0\n"
        "    while c < n_clauses:\n"
        "        start = tl.load(cstart_ptr + c)\n"
        "        end = tl.load(cstart_ptr + c + 1)\n"
        "        # A clause is short, so its literals are read as one small tile.\n"
        "        offs = start + tl.arange(0, BLOCK)\n"
        "        active = offs < end\n"
        "        lit = tl.load(lits_ptr + offs, mask=active, other=0)\n"
        "        var = tl.abs(lit) - 1\n"
        "        val = tl.load(row + var, mask=active, other=0)\n"
        "        # The literal is true when its sign agrees with the variable's value.\n"
        "        pos = lit > 0\n"
        "        is_true = active & ((pos & (val == 1)) | ((~pos) & (val == 0)))\n"
        "        any_true = tl.sum(is_true.to(tl.int32), axis=0)\n"
        "        unsat += tl.where(any_true > 0, 0, 1)\n"
        "        c += 1\n"
        "    tl.store(out_ptr + sample, unsat)\n\n\n"
        "def {0}_sample_and_score(n_samples=4096, seed=0, device='cuda', BLOCK=16):\n"
        "    \"\"\"Draws n_samples random assignments, scores them, and returns\n"
        "    (assignments, unsatisfied_counts) sorted best-first.\n\n"
        "    A count of zero is a model. Anything else is a starting point for a walk.\n"
        "    \"\"\"\n"
        "    import torch\n"
        "    g = torch.Generator(device=device)\n"
        "    g.manual_seed(seed)\n"
        "    assign = torch.randint(0, 2, (n_samples, NC_VARS), dtype=torch.int32,\n"
        "                           device=device, generator=g)\n"
        "    lits = torch.tensor(NC_{0}_LITS, dtype=torch.int32, device=device)\n"
        "    cstart = torch.tensor(NC_{0}_CSTART, dtype=torch.int32, device=device)\n"
        "    out = torch.zeros((n_samples,), dtype=torch.int32, device=device)\n"
        "    {0}_score_kernel[(n_samples,)](assign, lits, cstart, out,\n"
        "                                   n_samples, NC_VARS, NC_CLAUSES, BLOCK=BLOCK)\n"
        "    order = torch.argsort(out)\n"
        "    return assign[order], out[order]\n",
        e);
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public surface.
// ---------------------------------------------------------------------------

auto is_compilable(const Cnf& cnf) -> Result<void> {
    return is_compilable_for(cnf, Strategy::exhaustive);
}

auto is_compilable_for(const Cnf& cnf, Strategy strategy) -> Result<void> {
    if (cnf.num_vars == 0) {
        return make_error<void>(MathError::domain_error);
    }
    // The variable cap belongs to the ENUMERATOR, not to SAT. Exhaustive search numbers every
    // assignment in a 64-bit word, so it runs out at 63 variables; a random walk never numbers an
    // assignment at all and has no such limit. Applying one cap to both would have made the
    // walker refuse exactly the formulas it exists to solve.
    if (strategy == Strategy::exhaustive && cnf.num_vars > max_variables) {
        return make_error<void>(MathError::domain_error);
    }
    for (const std::vector<std::int64_t>& clause : cnf.clauses) {
        for (const std::int64_t lit : clause) {
            if (lit == 0) {
                return make_error<void>(MathError::domain_error);
            }
            const std::size_t v = var_of(lit);
            if (v == 0 || v > cnf.num_vars) {
                return make_error<void>(MathError::domain_error);
            }
        }
    }
    return {};
}

auto compile(const Cnf& cnf, const CompileOptions& opts) -> Result<std::string> {
    auto ok = is_compilable_for(cnf, opts.strategy);
    if (!ok) {
        return make_error<std::string>(ok.error());
    }
    if (!is_identifier(opts.entry)) {
        return make_error<std::string>(MathError::domain_error);
    }
    if (opts.strategy == Strategy::walksat &&
        (opts.walkers == 0 || opts.noise_percent > 100)) {
        return make_error<std::string>(MathError::domain_error);
    }
    if (opts.strategy == Strategy::walksat) {
        switch (opts.target) {
            case Target::cpp:
                return emit_cpp_walksat(cnf, opts);
            case Target::cuda:
                return emit_cuda_walksat(cnf, opts);
            case Target::triton:
                return emit_triton_walksat(cnf, opts);
        }
        return make_error<std::string>(MathError::domain_error);
    }
    switch (opts.target) {
        case Target::cpp:
            return emit_cpp(cnf, opts);
        case Target::cuda:
            return emit_cuda(cnf, opts);
        case Target::triton:
            return emit_triton(cnf, opts);
    }
    return make_error<std::string>(MathError::domain_error);
}

auto reference_solve(const Cnf& cnf, std::uint64_t max_blocks)
    -> Result<std::optional<std::uint64_t>> {
    using Answer = std::optional<std::uint64_t>;
    auto ok = is_compilable(cnf);
    if (!ok) {
        return make_error<Answer>(ok.error());
    }
    const std::uint64_t blocks =
        cnf.num_vars <= 6 ? 1ULL : (1ULL << (cnf.num_vars - lane_patterns.size()));
    if (blocks > max_blocks) {
        return make_error<Answer>(MathError::not_converged);
    }
    // Fewer than six variables do not fill a block, so lanes above 2^n stand for assignments that
    // do not exist and must not be reported as solutions.
    const std::uint64_t valid_lanes =
        cnf.num_vars >= 6 ? ~0ULL : (1ULL << (1ULL << cnf.num_vars)) - 1ULL;

    for (const std::uint64_t b : std::views::iota(std::uint64_t{0}, blocks)) {
        std::uint64_t sat = valid_lanes;
        for (const std::vector<std::int64_t>& clause : cnf.clauses) {
            std::uint64_t c = 0;
            for (const std::int64_t lit : clause) {
                const std::size_t i = var_of(lit) - 1;
                const std::uint64_t w = i < lane_patterns.size()
                                            ? lane_patterns[i]
                                            : (0ULL - ((b >> (i - lane_patterns.size())) & 1ULL));
                c |= lit < 0 ? ~w : w;
            }
            sat &= c;
            if (sat == 0) {
                break;
            }
        }
        if (sat != 0) {
            return Answer{(b << 6) | static_cast<std::uint64_t>(std::countr_zero(sat))};
        }
    }
    return Answer{std::nullopt};
}


auto reference_walksat(const Cnf& cnf, std::uint32_t walkers, std::uint64_t max_flips,
                       std::uint32_t noise_percent, std::uint64_t base_seed) -> Result<WalkResult> {
    auto ok = is_compilable_for(cnf, Strategy::walksat);
    if (!ok) {
        return make_error<WalkResult>(ok.error());
    }
    if (walkers == 0 || noise_percent > 100) {
        return make_error<WalkResult>(MathError::domain_error);
    }

    const FlatFormula f = flatten(cnf);
    const std::size_t nv = cnf.num_vars;
    const std::size_t nc = cnf.clauses.size();

    // The same generators the emitted code carries, so a walk here and a walk there are the same
    // sequence of moves rather than two algorithms that merely resemble each other.
    const auto mix = [](std::uint64_t z) -> std::uint64_t {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    const auto next = [](std::uint64_t& s) -> std::uint64_t {
        std::uint64_t x = s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s = x;
        return x * 0x2545F4914F6CDD1DULL;
    };

    std::vector<std::uint8_t> assign(nv, 0);
    std::vector<std::uint32_t> true_count(nc, 0);
    std::vector<std::uint32_t> unsat;
    std::vector<std::uint32_t> unsat_at(nc, 0);
    unsat.reserve(nc);

    const auto set_unsat = [&](std::uint32_t c) {
        unsat_at[c] = static_cast<std::uint32_t>(unsat.size());
        unsat.push_back(c);
    };
    const auto clear_unsat = [&](std::uint32_t c) {
        const std::uint32_t at = unsat_at[c];
        const std::uint32_t last = unsat.back();
        unsat[at] = last;
        unsat_at[last] = at;
        unsat.pop_back();
    };
    const auto break_count = [&](std::uint32_t v) -> std::uint32_t {
        const bool val = assign[v] != 0;
        const std::vector<std::uint32_t>& list = val ? f.occ_pos : f.occ_neg;
        const std::vector<std::uint32_t>& start = val ? f.op_start : f.on_start;
        std::uint32_t broken = 0;
        for (std::uint32_t i = start[v]; i < start[v + 1]; ++i) {
            broken += true_count[list[i]] == 1 ? 1U : 0U;
        }
        return broken;
    };
    const auto flip = [&](std::uint32_t v) {
        const bool was = assign[v] != 0;
        assign[v] = static_cast<std::uint8_t>(was ? 0 : 1);
        const std::vector<std::uint32_t>& losing = was ? f.occ_pos : f.occ_neg;
        const std::vector<std::uint32_t>& lstart = was ? f.op_start : f.on_start;
        const std::vector<std::uint32_t>& gaining = was ? f.occ_neg : f.occ_pos;
        const std::vector<std::uint32_t>& gstart = was ? f.on_start : f.op_start;
        for (std::uint32_t i = gstart[v]; i < gstart[v + 1]; ++i) {
            const std::uint32_t c = gaining[i];
            if (true_count[c]++ == 0) {
                clear_unsat(c);
            }
        }
        for (std::uint32_t i = lstart[v]; i < lstart[v + 1]; ++i) {
            const std::uint32_t c = losing[i];
            if (--true_count[c] == 0) {
                set_unsat(c);
            }
        }
    };

    WalkResult result;
    for (const std::uint32_t w : std::views::iota(std::uint32_t{0}, walkers)) {
        ++result.walkers_run;
        std::uint64_t rng =
            mix(base_seed + static_cast<std::uint64_t>(w) * 0x9E3779B97F4A7C15ULL) | 1ULL;
        for (std::size_t v = 0; v < nv; ++v) {
            assign[v] = static_cast<std::uint8_t>(next(rng) & 1ULL);
        }
        unsat.clear();
        for (std::size_t c = 0; c < nc; ++c) {
            std::uint32_t t = 0;
            for (std::uint32_t i = f.cstart[c]; i < f.cstart[c + 1]; ++i) {
                const std::int32_t lit = f.lits[i];
                const auto v = static_cast<std::uint32_t>(var_of(lit) - 1);
                t += ((lit > 0) == (assign[v] != 0)) ? 1U : 0U;
            }
            true_count[c] = t;
            if (t == 0) {
                set_unsat(static_cast<std::uint32_t>(c));
            }
        }

        bool solved = unsat.empty();
        for (std::uint64_t step = 0; step < max_flips && !solved; ++step) {
            const auto pick = static_cast<std::size_t>(next(rng) % unsat.size());
            const std::uint32_t c = unsat[pick];
            const std::uint32_t from = f.cstart[c];
            const std::uint32_t to = f.cstart[c + 1];
            if (from == to) {
                break;  // an empty clause is unsatisfiable by any walk; this one is finished
            }
            std::uint32_t best_v = 0;
            std::uint32_t best_break = std::numeric_limits<std::uint32_t>::max();
            bool freebie = false;
            for (std::uint32_t i = from; i < to; ++i) {
                const auto v = static_cast<std::uint32_t>(var_of(f.lits[i]) - 1);
                const std::uint32_t b = break_count(v);
                if (b < best_break || (b == best_break && v < best_v)) {
                    best_break = b;
                    best_v = v;
                }
                if (b == 0) {
                    freebie = true;
                }
            }
            std::uint32_t chosen = best_v;
            if (!freebie && (next(rng) % 100ULL) < noise_percent) {
                const std::uint32_t idx =
                    from + static_cast<std::uint32_t>(next(rng) % static_cast<std::uint64_t>(to - from));
                chosen = static_cast<std::uint32_t>(var_of(f.lits[idx]) - 1);
            }
            flip(chosen);
            solved = unsat.empty();
        }

        if (solved) {
            result.found = true;
            result.walker = w;
            result.model.assign(nv, false);
            for (std::size_t v = 0; v < nv; ++v) {
                result.model[v] = assign[v] != 0;
            }
            // The FIRST walker to succeed is also the lowest-indexed, because walkers run in
            // order here. The emitted parallel code reaches the same answer by reduction rather
            // than by ordering, which is what lets the two be compared.
            return result;
        }
    }
    return result;
}
auto model_of(std::uint64_t assignment, std::size_t num_vars) -> std::vector<bool> {
    std::vector<bool> model(num_vars, false);
    for (const std::size_t i : std::views::iota(std::size_t{0}, num_vars)) {
        model[i] = ((assignment >> i) & 1ULL) != 0ULL;
    }
    return model;
}

}  // namespace nimblecas::sat_compile
