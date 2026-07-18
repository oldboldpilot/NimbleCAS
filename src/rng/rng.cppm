// NimbleCAS parallelisable RNG (ROADMAP 7.8 dependency).
// @author Olumuyiwa Oluwasanmi
//
// A deterministic, reproducible, counter-based random-number substrate for Monte
// Carlo, MCMC and rejection sampling (ROADMAP 7.8). The design is *stateless at its
// core*: the primitive `counter_u64(key, counter)` maps an index to a well-distributed
// 64-bit value with no hidden state, so N workers can each own a disjoint slice of the
// counter space and the concatenation of their outputs is bit-identical to a single
// worker walking the whole range. That is the property that makes the generator
// parallelisable: the produced stream depends only on (key, counter), never on the
// number of threads or how the work was partitioned. The stateful `Rng` class is a thin
// sequential convenience built on top of the counter core, and `Rng::split` performs
// JAX-style tree splitting so a parallel decomposition can deterministically hand each
// subtask its own independent stream. There is no global mutable state and no time /
// entropy seeding — identical seeds always reproduce identical results.

module;
#include <immintrin.h>  // AVX-512 intrinsics for the vectorised counter core (GMF include).

export module nimblecas.rng;

import std;
import nimblecas.core;

export namespace nimblecas {

// SplitMix64 finalizer: a pure, invertible bit-mixing function that turns a sequentially
// incremented state into a well-distributed 64-bit value. Used both directly (to expand
// a seed into a key) and as a mixing primitive inside the counter core. Pure — no error
// channel.
[[nodiscard]] auto splitmix64(std::uint64_t x) noexcept -> std::uint64_t;

// The parallelisable primitive. Deterministically maps (key, counter) -> a
// well-distributed 64-bit value via a Threefry-2x64 keyed bijection (20 rounds), so
// adjacent counters decorrelate fully.
//
// PARALLELISM CONTRACT: this function is stateless and referentially transparent. For a
// fixed key, the sequence counter_u64(key, 0), counter_u64(key, 1), ... is fully defined
// by the index alone. Therefore any partition of the index range 0..N-1 into disjoint
// sub-ranges, evaluated on independent threads and concatenated in index order, yields
// exactly the same sequence as a single-threaded evaluation. Parallel decompositions are
// thus reproducible regardless of thread count or scheduling.
[[nodiscard]] auto counter_u64(std::uint64_t key, std::uint64_t counter) noexcept
    -> std::uint64_t;

// Batched counter core: fills out[i] = counter_u64(key, base_counter + i) for every i in
// [0, out.size()). BIT-IDENTICAL to calling counter_u64 in a loop — it preserves the same
// parallelism and reproducibility contract — but processes eight Threefry-2x64 blocks per
// step through an AVX-512F path when the running CPU supports it, transparently falling back
// to the scalar core otherwise. The dispatch is purely a performance detail: the produced
// values never depend on the ISA, the thread count, or how a range was partitioned. This is
// the hot path for the Monte Carlo engines, whose profile is dominated by RNG self-time.
void counter_u64_batch(std::uint64_t key, std::uint64_t base_counter,
                       std::span<std::uint64_t> out) noexcept;

// ---------------------------------------------------------------------------
// Distributions as free functions over a raw 64-bit draw (`bits`). These are pure
// transforms of an already-generated value, so they compose with either the counter
// core or the stateful Rng.
// ---------------------------------------------------------------------------

// Maps a 64-bit draw to a double in [0, 1) using the top 53 bits over 2^53, matching the
// 53-bit mantissa of an IEEE-754 double so every representable value in the range is
// reachable and the result is never exactly 1.0.
[[nodiscard]] auto uniform_unit(std::uint64_t bits) noexcept -> double;

// Maps a 64-bit draw to a double in [lo, hi). Returns domain_error if hi < lo, and
// overflow if the width hi - lo is not finite (e.g. lo = -1e308, hi = 1e308) — that guard
// keeps a non-finite width from producing a silent NaN outside the railway.
[[nodiscard]] auto uniform_double(std::uint64_t bits, double lo, double hi) -> Result<double>;

// Maps a 64-bit draw to an integer in the inclusive range [lo, hi] via modulo reduction.
// Returns domain_error if hi < lo. The modulo introduces the classic small bias of
// bits % width: the worst-case relative bias is range_size / 2^64, which is negligible for
// the small ranges typical of Monte Carlo work but grows for very wide ranges (a width near
// 2^63 can make some outcomes ~2x as likely). For unbiased draws over a huge range, reject
// sampling on top of counter_u64 is the intended pattern.
[[nodiscard]] auto uniform_int(std::uint64_t bits, std::int64_t lo, std::int64_t hi)
    -> Result<std::int64_t>;

// ---------------------------------------------------------------------------
// Sequential convenience generator built ON TOP of the counter core. Holds {key,
// counter}; every draw is counter_u64(key, counter++). Copyable and trivially cheap.
// ---------------------------------------------------------------------------
class Rng {
public:
    // Constructs a generator whose stream is counter_u64(seeded-key, 0), (…,1), ….
    // The key is derived from `seed` via splitmix64 so nearby seeds still produce
    // well-separated streams. Fully deterministic: equal seeds => equal streams.
    [[nodiscard]] static auto seeded(std::uint64_t seed) noexcept -> Rng;

    // Next raw 64-bit draw; advances the internal counter.
    [[nodiscard]] auto next_u64() noexcept -> std::uint64_t;

    // Fills `out` with the next out.size() raw draws and advances the counter by that many —
    // equivalent to calling next_u64() out.size() times, but routed through counter_u64_batch
    // (AVX-512 when available). The stream is unchanged bit-for-bit, so a batched fill and a
    // scalar loop are freely interchangeable.
    void fill_u64(std::span<std::uint64_t> out) noexcept;

    // Returns an independent child stream whose key is derived from (this key, index)
    // through the counter core. Distinct indices yield independent streams, and children
    // are independent of the parent — JAX-style tree splitting that lets a parallel
    // decomposition deterministically assign each subtask its own generator. Does not
    // advance this generator.
    [[nodiscard]] auto split(std::uint64_t index) const noexcept -> Rng;

    // Distribution methods: each consumes one raw draw from this stream.
    [[nodiscard]] auto next_unit() noexcept -> double;
    [[nodiscard]] auto next_double(double lo, double hi) -> Result<double>;
    [[nodiscard]] auto next_int(std::int64_t lo, std::int64_t hi) -> Result<std::int64_t>;

    // Observers (useful for reproducibility checks and checkpointing).
    [[nodiscard]] auto key() const noexcept -> std::uint64_t { return key_; }
    [[nodiscard]] auto counter() const noexcept -> std::uint64_t { return counter_; }

private:
    constexpr Rng(std::uint64_t key, std::uint64_t counter) noexcept
        : key_(key), counter_(counter) {}

    std::uint64_t key_{};
    std::uint64_t counter_{};
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

namespace {

// Left rotation. Every rotation amount used by the Threefry schedule is in (0, 64), so
// the shift-count precondition of the (64 - r) shift is always satisfied.
[[nodiscard]] constexpr auto rotl64(std::uint64_t x, unsigned r) noexcept -> std::uint64_t {
    return (x << r) | (x >> (64U - r));
}

// Domain-separation constant (fractional golden ratio) that keeps split-derived keys off
// the counter positions a stream draws from during normal generation.
inline constexpr std::uint64_t split_domain = 0x9E3779B97F4A7C15ULL;

// The Threefry-2x64-20 rotation schedule (shared by the scalar and vector cores).
inline constexpr std::array<unsigned, 8> threefry_rot{16U, 42U, 12U, 31U, 16U, 32U, 24U, 21U};

// The Threefry-2x64 key schedule: {k0, k1, k2} with k2 = parity ^ k0 ^ k1. Computed once per
// key and shared by the scalar and vector cores so both derive keys identically.
struct ThreefryKeys {
    std::uint64_t k[3];
};

[[nodiscard]] auto threefry_keys(std::uint64_t key) noexcept -> ThreefryKeys {
    constexpr std::uint64_t parity = 0x1BD11BDAA9FC1A22ULL;
    const std::uint64_t k0 = key;
    const std::uint64_t k1 = splitmix64(key);
    return ThreefryKeys{{k0, k1, parity ^ k0 ^ k1}};
}

#if defined(__x86_64__)
// AVX-512 vector core: eight Threefry-2x64-20 blocks per outer step, one counter per lane.
// Marked with an AVX-512F target attribute so the enclosing translation unit need NOT be
// compiled with -mavx512f (the project default is -march=x86-64-v3, i.e. AVX2); this function
// is reached only after __builtin_cpu_supports("avx512f") confirms the ISA at run time. Every
// operation is a per-lane mod-2^64 add / xor / rotate — identical semantics to the scalar
// core — so each lane's output equals counter_u64(key, block_base + lane) bit-for-bit.
__attribute__((target("avx512f"))) void counter_u64_batch_avx512(
    const ThreefryKeys& kk, std::uint64_t base, std::uint64_t* out, std::size_t blocks) noexcept {
    const __m512i lane_id = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i k0 = _mm512_set1_epi64(static_cast<long long>(kk.k[0]));
    const __m512i k1v = _mm512_set1_epi64(static_cast<long long>(kk.k[1]));
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint64_t block_base = base + b * 8ULL;
        const __m512i ctr =
            _mm512_add_epi64(_mm512_set1_epi64(static_cast<long long>(block_base)), lane_id);
        __m512i x0 = _mm512_add_epi64(ctr, k0);
        __m512i x1 = k1v;
        for (unsigned r = 0; r < 20U; ++r) {
            x0 = _mm512_add_epi64(x0, x1);
            x1 = _mm512_rolv_epi64(
                x1, _mm512_set1_epi64(static_cast<long long>(threefry_rot[r % 8U])));
            x1 = _mm512_xor_si512(x1, x0);
            if ((r + 1U) % 4U == 0U) {
                const unsigned s = (r + 1U) / 4U;  // key-injection index 1..5
                x0 = _mm512_add_epi64(x0, _mm512_set1_epi64(static_cast<long long>(kk.k[s % 3U])));
                x1 = _mm512_add_epi64(
                    x1, _mm512_set1_epi64(static_cast<long long>(kk.k[(s + 1U) % 3U] + s)));
            }
        }
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + b * 8ULL),
                            _mm512_xor_si512(x0, x1));
    }
}
#endif

}  // namespace

auto splitmix64(std::uint64_t x) noexcept -> std::uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

auto counter_u64(std::uint64_t key, std::uint64_t counter) noexcept -> std::uint64_t {
    // Threefry-2x64-20: a keyed bijection over a 128-bit block. We only expose a single
    // 64-bit key, so the second key word is derived by splitmix64 (decorrelating the two
    // words), and the second input word is fixed at 0 — the counter alone indexes the
    // stream. Threefry is a bijection on the full 128-bit block; we return the 64-bit fold
    // x0 ^ x1, which is NOT injective (outputs collide at the birthday rate, ~2^32 draws) —
    // fine for a Monte Carlo substrate, but do not rely on it for collision-free unique IDs.
    // The 20-round schedule gives strong avalanche so adjacent counters look independent.
    constexpr std::uint64_t parity = 0x1BD11BDAA9FC1A22ULL;
    const std::uint64_t k0 = key;
    const std::uint64_t k1 = splitmix64(key);
    const std::uint64_t k2 = parity ^ k0 ^ k1;
    const std::array<std::uint64_t, 3> ks{k0, k1, k2};

    std::uint64_t x0 = counter + k0;
    std::uint64_t x1 = k1;  // second input word is 0, so x1 starts at 0 + k1

    constexpr std::array<unsigned, 8> rot{16U, 42U, 12U, 31U, 16U, 32U, 24U, 21U};
    for (unsigned r = 0; r < 20U; ++r) {
        x0 += x1;
        x1 = rotl64(x1, rot[r % 8U]);
        x1 ^= x0;
        if ((r + 1U) % 4U == 0U) {
            const unsigned s = (r + 1U) / 4U;  // key-injection index 1..5
            x0 += ks[s % 3U];
            x1 += ks[(s + 1U) % 3U] + s;
        }
    }
    return x0 ^ x1;
}

void counter_u64_batch(std::uint64_t key, std::uint64_t base_counter,
                       std::span<std::uint64_t> out) noexcept {
    const ThreefryKeys kk = threefry_keys(key);
    std::size_t done = 0;
#if defined(__x86_64__)
    // Vector core handles whole 8-block groups; __builtin_cpu_supports gates the ISA at run
    // time so a non-AVX-512 host silently uses the scalar loop below with identical output.
    if (out.size() >= 8U && __builtin_cpu_supports("avx512f")) {
        const std::size_t blocks = out.size() / 8U;
        counter_u64_batch_avx512(kk, base_counter, out.data(), blocks);
        done = blocks * 8U;
    }
#endif
    // Scalar tail (and the whole range when AVX-512 is unavailable). Bit-identical by design.
    for (std::size_t i = done; i < out.size(); ++i) {
        out[i] = counter_u64(key, base_counter + static_cast<std::uint64_t>(i));
    }
}

auto uniform_unit(std::uint64_t bits) noexcept -> double {
    // Top 53 bits scaled by 2^-53 gives a uniform double in [0, 1).
    constexpr double scale = 1.0 / 9007199254740992.0;  // 1 / 2^53
    return static_cast<double>(bits >> 11) * scale;
}

auto uniform_double(std::uint64_t bits, double lo, double hi) -> Result<double> {
    if (hi < lo) {
        return make_error<double>(MathError::domain_error);
    }
    const double width = hi - lo;
    if (!std::isfinite(width)) {
        // An infinite/NaN width (e.g. lo = -1e308, hi = 1e308) would make 0 * inf = NaN and
        // leak a bad value through has_value(). Reject it on the railway instead.
        return make_error<double>(MathError::overflow);
    }
    return lo + uniform_unit(bits) * width;
}

auto uniform_int(std::uint64_t bits, std::int64_t lo, std::int64_t hi) -> Result<std::int64_t> {
    if (hi < lo) {
        return make_error<std::int64_t>(MathError::domain_error);
    }
    // Number of steps between lo and hi (unsigned difference is exact even across the
    // signed range, e.g. lo<0<hi). The inclusive range size is span + 1.
    const std::uint64_t span =
        static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
    if (span == std::numeric_limits<std::uint64_t>::max()) {
        // Full 64-bit range [INT64_MIN, INT64_MAX]: every draw is already in range.
        return static_cast<std::int64_t>(bits);
    }
    const std::uint64_t width = span + 1U;
    const std::uint64_t offset = bits % width;
    // Reconstruct via unsigned arithmetic to avoid signed-overflow UB; the value is
    // guaranteed to land in [lo, hi] because offset <= span.
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) + offset);
}

// --- Rng ---

auto Rng::seeded(std::uint64_t seed) noexcept -> Rng {
    return Rng{splitmix64(seed), 0};
}

auto Rng::next_u64() noexcept -> std::uint64_t {
    return counter_u64(key_, counter_++);
}

void Rng::fill_u64(std::span<std::uint64_t> out) noexcept {
    // Draws out.size() values from the current counter position and advances by that many, so
    // the post-fill stream is identical to out.size() successive next_u64() calls.
    counter_u64_batch(key_, counter_, out);
    counter_ += out.size();
}

auto Rng::split(std::uint64_t index) const noexcept -> Rng {
    const std::uint64_t child_key = counter_u64(key_, split_domain ^ index);
    return Rng{child_key, 0};
}

auto Rng::next_unit() noexcept -> double {
    return uniform_unit(next_u64());
}

auto Rng::next_double(double lo, double hi) -> Result<double> {
    return uniform_double(next_u64(), lo, hi);
}

auto Rng::next_int(std::int64_t lo, std::int64_t hi) -> Result<std::int64_t> {
    return uniform_int(next_u64(), lo, hi);
}

}  // namespace nimblecas
