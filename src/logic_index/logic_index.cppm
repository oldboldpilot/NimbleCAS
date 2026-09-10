// NimbleCAS logic clause indexing — the batched first-argument probe, CPU kernel.
// @author Olumuyiwa Oluwasanmi
//
// WHAT IS AND IS NOT PARALLELISABLE IN A PROLOG ENGINE, stated plainly because the honest
// answer is more useful than an optimistic one.
//
// SLD resolution itself is irregular: the AND-side is a data-dependent backtracking walk whose
// control flow diverges at every step, which is a poor fit for SIMT and not much better for
// SIMD. `nimblecas.logic` therefore keeps the search scalar and takes its parallelism from
// OR-parallelism over clause branches.
//
// One part of the inner loop IS regular, and it is the part that runs most often: FIRST-ARGUMENT
// INDEXING. Deciding which clauses of a predicate could possibly match a goal is a comparison of
// one integer key against a contiguous array of them, with no branching and no data dependence.
// That is exactly a SIMD/SIMT workload, and batching it across many goals at once — which is
// what a distributed or speculative engine does anyway — makes it a genuinely wide one.
//
// This module is that kernel. `index_probe_batch` takes the clause keys of a predicate and a
// batch of goal keys, and produces one candidate BITMASK per goal.
//
// THE CPU PATH IS AUTHORITATIVE. The scalar loop below defines the answer; the AVX2 and AVX-512
// paths must agree with it BIT FOR BIT, and the tests check exactly that rather than checking
// each path against its own expectations. The GPU mirror in `nimblecas.gpu` is held to the same
// standard. A faster path that computes something subtly different is not an optimisation.
//
// KEY SEMANTICS — the one thing to get right:
//   key 0 means UNKNOWN, and unknown matches everything.
// A goal whose first argument is an unbound variable has key 0 and must consider every clause;
// a clause whose first head argument is a variable has key 0 and must be considered by every
// goal. Two non-zero keys are candidates only when EQUAL. The filter may only ever rule out a
// clause it can prove cannot match: a false negative would silently lose an answer, which is
// the one outcome Rule 32 forbids. A false POSITIVE merely costs a unification attempt that
// fails, so when in doubt the answer is "candidate".

module;
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

export module nimblecas.logic_index;

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.parallel;

export namespace nimblecas::logic_index {

// Which instruction set the batched probe actually used. Reported so a benchmark can say which
// path it measured instead of guessing from the host's advertised capabilities.
enum class ProbeIsa : std::uint8_t { scalar, avx2, avx512 };

[[nodiscard]] constexpr auto to_string_view(ProbeIsa isa) noexcept -> std::string_view {
    switch (isa) {
        case ProbeIsa::scalar: return "scalar";
        case ProbeIsa::avx2:   return "avx2";
        case ProbeIsa::avx512: return "avx512";
    }
    return "scalar";
}

// The widest probe path this host can actually run, decided at run time.
[[nodiscard]] auto detect_probe_isa() noexcept -> ProbeIsa;

// The index key of a term used as a first argument: its principal functor, hashed.
//
// Returns 0 for a VARIABLE, which is the "matches anything" key. Every other term kind gets a
// non-zero key, and two terms with different keys provably cannot unify.
[[nodiscard]] auto first_arg_key(const Term& t) -> std::uint64_t;

// The index keys of a program's clause heads, in clause order: the first argument's key, or 0
// for a clause whose head is an atom (no arguments at all, so nothing to discriminate on).
[[nodiscard]] auto clause_index_keys(const Program& program) -> std::vector<std::uint64_t>;

// Words per goal row in the output bitmask.
[[nodiscard]] constexpr auto probe_row_words(std::size_t clause_count) noexcept -> std::size_t {
    return (clause_count + 63U) / 64U;
}

// The batched first-argument probe.
//
// For goal `g` and clause `c`, bit `c` of row `g` is set iff clause `c` is a CANDIDATE for goal
// `g` — that is, iff `goal_keys[g] == 0 || clause_keys[c] == 0 || goal_keys[g] == clause_keys[c]`.
//
// `out_words` must be exactly `goal_keys.size() * probe_row_words(clause_keys.size())` long;
// a wrong size is a `domain_error` rather than a truncated result. Bits beyond `clause_keys
// .size()` in the final word of each row are written as 0.
//
// Rows are independent, so the batch is parallelised across goals; within a row the comparison
// is vectorised. Both are transparent: the result does not depend on either.
[[nodiscard]] auto index_probe_batch(std::span<const std::uint64_t> clause_keys,
                                     std::span<const std::uint64_t> goal_keys,
                                     std::span<std::uint64_t> out_words) -> Result<void>;

// The same probe forced onto a specific path. Exists so the tests can prove the vector paths
// agree with the scalar one bit for bit on the same input, which is the only way that claim is
// worth making. Asking for a path this host cannot run is a `domain_error`, never a silent
// downgrade to a slower one.
[[nodiscard]] auto index_probe_batch_with(ProbeIsa isa, std::span<const std::uint64_t> clause_keys,
                                          std::span<const std::uint64_t> goal_keys,
                                          std::span<std::uint64_t> out_words) -> Result<void>;

// The candidate clause indices for one goal row, decoded from the bitmask in ascending order.
[[nodiscard]] auto decode_candidates(std::span<const std::uint64_t> row, std::size_t clause_count)
    -> std::vector<std::size_t>;

}  // namespace nimblecas::logic_index

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::logic_index {

namespace {

// A 64-bit FNV-1a over the discriminating text of a principal functor. Any deterministic hash
// would do; FNV-1a is chosen because it is exact, dependency-free and identical on every host,
// which matters because the GPU mirror has to compute the same keys.
constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

[[nodiscard]] auto fnv1a(std::string_view s, std::uint64_t seed) noexcept -> std::uint64_t {
    std::uint64_t h = seed;
    for (const char c : s) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= fnv_prime;
    }
    return h;
}

// 0 is reserved for "unknown", so a hash that lands on it is nudged. Losing one value out of
// 2^64 costs nothing; letting a real key collide with the wildcard would make the filter admit
// every clause for that key, which is slow but still correct — the nudge keeps it fast.
[[nodiscard]] auto non_zero(std::uint64_t h) noexcept -> std::uint64_t {
    return h == 0U ? 1U : h;
}

// The scalar probe. THIS DEFINES THE ANSWER; the vector paths are checked against it.
auto probe_row_scalar(std::span<const std::uint64_t> clause_keys, std::uint64_t goal_key,
                      std::span<std::uint64_t> row) noexcept -> void {
    const std::size_t nc = clause_keys.size();
    for (std::size_t w = 0; w < row.size(); ++w) {
        std::uint64_t bits = 0;
        const std::size_t base = w * 64U;
        const std::size_t lanes = std::min<std::size_t>(64U, nc - base);
        for (std::size_t i = 0; i < lanes; ++i) {
            const std::uint64_t ck = clause_keys[base + i];
            const bool candidate = goal_key == 0U || ck == 0U || ck == goal_key;
            bits |= static_cast<std::uint64_t>(candidate) << i;
        }
        row[w] = bits;
    }
}

#if defined(__x86_64__) || defined(__i386__)

// AVX2 compares four 64-bit keys at a time. `_mm256_cmpeq_epi64` yields an all-ones lane per
// equal pair, and moving the sign bits out of the corresponding double lanes turns four lanes
// into four bits — which is exactly the bit layout the scalar path produces.
[[gnu::target("avx2")]] auto probe_row_avx2(std::span<const std::uint64_t> clause_keys,
                                            std::uint64_t goal_key,
                                            std::span<std::uint64_t> row) noexcept -> void {
    const std::size_t nc = clause_keys.size();
    const __m256i vgoal = _mm256_set1_epi64x(static_cast<long long>(goal_key));
    const __m256i vzero = _mm256_setzero_si256();
    for (std::size_t w = 0; w < row.size(); ++w) {
        std::uint64_t bits = 0;
        const std::size_t base = w * 64U;
        const std::size_t lanes = std::min<std::size_t>(64U, nc - base);
        std::size_t i = 0;
        for (; i + 4U <= lanes; i += 4U) {
            const __m256i vc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(clause_keys.data() + base + i));
            const __m256i eq = _mm256_cmpeq_epi64(vc, vgoal);
            const __m256i wild = _mm256_cmpeq_epi64(vc, vzero);
            const __m256i any = _mm256_or_si256(eq, wild);
            const auto m = static_cast<std::uint64_t>(
                _mm256_movemask_pd(_mm256_castsi256_pd(any)));
            bits |= m << i;
        }
        for (; i < lanes; ++i) {
            const std::uint64_t ck = clause_keys[base + i];
            bits |= static_cast<std::uint64_t>(ck == 0U || ck == goal_key) << i;
        }
        // An unknown goal key admits every clause; applying it here rather than inside the lane
        // loop keeps the hot path branch-free.
        if (goal_key == 0U) {
            bits = lanes == 64U ? ~std::uint64_t{0}
                                : ((std::uint64_t{1} << lanes) - std::uint64_t{1});
        }
        row[w] = bits;
    }
}

// AVX-512 compares eight at a time and yields the bitmask directly in a mask register, which is
// the natural shape for this problem — no sign-bit extraction step at all.
[[gnu::target("avx512f,avx512dq")]] auto probe_row_avx512(
    std::span<const std::uint64_t> clause_keys, std::uint64_t goal_key,
    std::span<std::uint64_t> row) noexcept -> void {
    const std::size_t nc = clause_keys.size();
    const __m512i vgoal = _mm512_set1_epi64(static_cast<long long>(goal_key));
    const __m512i vzero = _mm512_setzero_si512();
    for (std::size_t w = 0; w < row.size(); ++w) {
        std::uint64_t bits = 0;
        const std::size_t base = w * 64U;
        const std::size_t lanes = std::min<std::size_t>(64U, nc - base);
        std::size_t i = 0;
        for (; i + 8U <= lanes; i += 8U) {
            const __m512i vc = _mm512_loadu_si512(
                reinterpret_cast<const void*>(clause_keys.data() + base + i));
            const __mmask8 eq = _mm512_cmpeq_epi64_mask(vc, vgoal);
            const __mmask8 wild = _mm512_cmpeq_epi64_mask(vc, vzero);
            bits |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(eq | wild)) << i;
        }
        for (; i < lanes; ++i) {
            const std::uint64_t ck = clause_keys[base + i];
            bits |= static_cast<std::uint64_t>(ck == 0U || ck == goal_key) << i;
        }
        if (goal_key == 0U) {
            bits = lanes == 64U ? ~std::uint64_t{0}
                                : ((std::uint64_t{1} << lanes) - std::uint64_t{1});
        }
        row[w] = bits;
    }
}

#endif  // x86

[[nodiscard]] auto host_has_avx2() noexcept -> bool {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

[[nodiscard]] auto host_has_avx512() noexcept -> bool {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512dq") != 0;
#else
    return false;
#endif
}

auto probe_row_dispatch(ProbeIsa isa, std::span<const std::uint64_t> clause_keys,
                        std::uint64_t goal_key, std::span<std::uint64_t> row) noexcept -> void {
#if defined(__x86_64__) || defined(__i386__)
    if (isa == ProbeIsa::avx512) {
        probe_row_avx512(clause_keys, goal_key, row);
        return;
    }
    if (isa == ProbeIsa::avx2) {
        probe_row_avx2(clause_keys, goal_key, row);
        return;
    }
#else
    (void)isa;
#endif
    probe_row_scalar(clause_keys, goal_key, row);
}

[[nodiscard]] auto run_probe(ProbeIsa isa, std::span<const std::uint64_t> clause_keys,
                             std::span<const std::uint64_t> goal_keys,
                             std::span<std::uint64_t> out_words) -> Result<void> {
    const std::size_t words = probe_row_words(clause_keys.size());
    if (out_words.size() != goal_keys.size() * words) {
        return make_error<void>(MathError::domain_error);
    }
    if (goal_keys.empty() || clause_keys.empty()) {
        return {};
    }
    // One task per goal row. Rows never touch each other's words, so there is no sharing to
    // synchronise and the result cannot depend on how the work was split.
    const auto rows = parallel::transform_index(
        goal_keys.size(),
        [&](std::size_t g) -> int {
            probe_row_dispatch(isa, clause_keys, goal_keys[g],
                               out_words.subspan(g * words, words));
            return 0;
        },
        1);
    (void)rows;
    return {};
}

}  // namespace

auto detect_probe_isa() noexcept -> ProbeIsa {
    if (host_has_avx512()) {
        return ProbeIsa::avx512;
    }
    if (host_has_avx2()) {
        return ProbeIsa::avx2;
    }
    return ProbeIsa::scalar;
}

auto first_arg_key(const Term& t) -> std::uint64_t {
    if (is_var(t)) {
        return 0U;  // an unbound argument discriminates nothing
    }
    if (is_atom(t)) {
        return non_zero(fnv1a(atom_of(t).name, fnv_offset ^ 0x61ULL));
    }
    if (is_int(t)) {
        // Integers are hashed from their VALUE, not their text, so the integer 1 and the atom
        // '1' get different keys — confusing them would let the filter drop a real candidate.
        const auto v = static_cast<std::uint64_t>(int_of(t).value);
        std::uint64_t h = fnv_offset ^ 0x69ULL;
        for (std::size_t i = 0; i < 8U; ++i) {
            h ^= (v >> (i * 8U)) & 0xFFULL;
            h *= fnv_prime;
        }
        return non_zero(h);
    }
    const CompoundNode& c = compound_of(t);
    // Functor AND arity: `f(a)` and `f(a,b)` cannot unify, so they must not share a key.
    std::uint64_t h = fnv1a(c.functor, fnv_offset ^ 0x63ULL);
    h ^= c.args.size();
    h *= fnv_prime;
    return non_zero(h);
}

auto clause_index_keys(const Program& program) -> std::vector<std::uint64_t> {
    std::vector<std::uint64_t> keys;
    keys.reserve(program.size());
    for (const Clause& c : program) {
        // A 0-arity head has no first argument to discriminate on, so it is a wildcard.
        keys.push_back(is_compound(c.head) ? first_arg_key(compound_of(c.head).args.front()) : 0U);
    }
    return keys;
}

auto index_probe_batch(std::span<const std::uint64_t> clause_keys,
                       std::span<const std::uint64_t> goal_keys,
                       std::span<std::uint64_t> out_words) -> Result<void> {
    return run_probe(detect_probe_isa(), clause_keys, goal_keys, out_words);
}

auto index_probe_batch_with(ProbeIsa isa, std::span<const std::uint64_t> clause_keys,
                            std::span<const std::uint64_t> goal_keys,
                            std::span<std::uint64_t> out_words) -> Result<void> {
    // Refusing a path this host cannot execute is the honest answer: silently running the
    // scalar path while the caller believes it measured AVX-512 would make every number it
    // reports a lie.
    if (isa == ProbeIsa::avx512 && !host_has_avx512()) {
        return make_error<void>(MathError::domain_error);
    }
    if (isa == ProbeIsa::avx2 && !host_has_avx2()) {
        return make_error<void>(MathError::domain_error);
    }
    return run_probe(isa, clause_keys, goal_keys, out_words);
}

auto decode_candidates(std::span<const std::uint64_t> row, std::size_t clause_count)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> out;
    for (std::size_t w = 0; w < row.size(); ++w) {
        std::uint64_t bits = row[w];
        while (bits != 0U) {
            const auto lane = static_cast<std::size_t>(std::countr_zero(bits));
            const std::size_t index = w * 64U + lane;
            if (index < clause_count) {
                out.push_back(index);
            }
            bits &= bits - 1U;  // clear the lowest set bit
        }
    }
    return out;
}

}  // namespace nimblecas::logic_index
