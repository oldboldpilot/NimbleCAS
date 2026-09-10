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
    // Base name for the emitted entry points, e.g. "puzzle" yields `puzzle_solve`.
    std::string entry{"formula"};
    // C++ only. The vector-extension path, eight blocks at a time.
    bool emit_simd{true};
    // C++ only. A std::jthread fan-out over disjoint block ranges, reduced by minimum so the
    // answer does not depend on the thread count.
    bool emit_parallel{true};
};

// Whether this formula can be compiled at all, and why not when it cannot.
//
// `domain_error` for a malformed CNF -- zero variables, a literal of 0, or a literal naming a
// variable outside 1..num_vars -- and for more than `max_variables` variables. An EMPTY clause is
// accepted and compiles to a formula no assignment satisfies, because that is what an empty
// clause means; an empty clause LIST is accepted and every assignment satisfies it.
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

}  // namespace

// ---------------------------------------------------------------------------
// Public surface.
// ---------------------------------------------------------------------------

auto is_compilable(const Cnf& cnf) -> Result<void> {
    if (cnf.num_vars == 0 || cnf.num_vars > max_variables) {
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
    auto ok = is_compilable(cnf);
    if (!ok) {
        return make_error<std::string>(ok.error());
    }
    if (!is_identifier(opts.entry)) {
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

auto model_of(std::uint64_t assignment, std::size_t num_vars) -> std::vector<bool> {
    std::vector<bool> model(num_vars, false);
    for (const std::size_t i : std::views::iota(std::size_t{0}, num_vars)) {
        model[i] = ((assignment >> i) & 1ULL) != 0ULL;
    }
    return model;
}

}  // namespace nimblecas::sat_compile
