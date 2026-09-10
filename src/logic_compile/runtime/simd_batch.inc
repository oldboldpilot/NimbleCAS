// SIMD checked-arithmetic kernels for the Prolog compiler's generated C++.
// Dynamic-dispatch waterfall: AVX-512 -> AVX2 -> scalar.
// Self-contained standalone compilation unit emitted into generated code.
//
// Every operation computes both results and overflow status masks without per-lane branching.
// The scalar path defines canonical results and respects C++ unsigned modular arithmetic
// to prevent undefined behavior under UndefinedBehaviorSanitizer. Vector paths guarantee
// bit-for-bit exact agreement with scalar execution on all representable values and masks.

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Which path a call actually took. Reported so a benchmark can say what it measured instead of
// assuming; a claim of "AVX-512" from a host that silently ran scalar is a false number.
enum class nc_isa : unsigned char { scalar = 0, avx2 = 1, avx512 = 2 };

// Dynamic runtime ISA detection querying CPU capability flags.
[[nodiscard]] auto nc_detect_isa() noexcept -> nc_isa;

// Element-wise checked arithmetic over arrays of int64.
//
// For each lane i: out[i] = a[i] OP b[i], and ok[i] is 1 when that value is exactly
// representable in int64 and 0 when it is not. A lane whose ok is 0 leaves out[i] UNSPECIFIED
// and the caller must not read it — the honesty invariant forbids reading an unrepresentable
// wrapped value as an exact result.
//
// `n` is the lane count; the arrays are non-overlapping.
auto nc_add_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void;
auto nc_sub_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void;
auto nc_mul_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void;

// Truncating division and its remainder. ok[i] is 0 for a zero divisor AND for the single
// overflowing case INT64_MIN / -1.
auto nc_quot_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                   unsigned char* ok, std::size_t n) noexcept -> void;

// Comparisons: out[i] is 1 when the relation holds, 0 otherwise. These cannot fail, so there is
// no mask — a comparison of two representable integers is always representable.
auto nc_cmp_lt_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void;
auto nc_cmp_le_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void;
auto nc_cmp_eq_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void;

// Forced-path variants, so a test can prove the vector paths agree with the scalar one BIT FOR
// BIT on the same input. Requesting a path the host cannot execute must FAIL (return false)
// rather than quietly running a narrower one — silently substituting scalar makes every number
// a benchmark reports a lie.
[[nodiscard]] auto nc_add_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool;
[[nodiscard]] auto nc_mul_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool;

[[nodiscard]] auto nc_sub_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool;
[[nodiscard]] auto nc_quot_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                      std::int64_t* out, unsigned char* ok,
                                      std::size_t n) noexcept -> bool;
[[nodiscard]] auto nc_cmp_lt_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool;
[[nodiscard]] auto nc_cmp_le_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool;
[[nodiscard]] auto nc_cmp_eq_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool;

// ===========================================================================
// Implementation details (anonymous namespace)
// ===========================================================================
namespace {

// Scalar checked addition for a single lane.
// Signed integer overflow is undefined behavior in C++, so computation is performed
// in uint64_t modulo 2^64. Two's complement addition overflows if and only if operands
// share a sign bit and the result's sign bit differs: (~(a ^ b) & (a ^ sum)) has bit 63 set.
inline auto nc_add_scalar_lane(std::int64_t a, std::int64_t b, std::int64_t& out,
                               unsigned char& ok) noexcept -> void {
    const auto ua = static_cast<std::uint64_t>(a);
    const auto ub = static_cast<std::uint64_t>(b);
    const auto usum = ua + ub;
    const auto ovf = (~(ua ^ ub)) & (ua ^ usum);
    ok = ((ovf & (std::uint64_t{1} << 63)) != 0) ? 0 : 1;
    out = static_cast<std::int64_t>(usum);
}

// Scalar checked subtraction for a single lane.
// Two's complement subtraction overflows if and only if operands' signs differ and the
// difference's sign differs from a's: ((a ^ b) & (a ^ diff)) has bit 63 set.
inline auto nc_sub_scalar_lane(std::int64_t a, std::int64_t b, std::int64_t& out,
                               unsigned char& ok) noexcept -> void {
    const auto ua = static_cast<std::uint64_t>(a);
    const auto ub = static_cast<std::uint64_t>(b);
    const auto udiff = ua - ub;
    const auto ovf = (ua ^ ub) & (ua ^ udiff);
    ok = ((ovf & (std::uint64_t{1} << 63)) != 0) ? 0 : 1;
    out = static_cast<std::int64_t>(udiff);
}

// Scalar checked multiplication for a single lane.
// Uses the compiler built-in __builtin_mul_overflow for exact overflow detection without UB.
inline auto nc_mul_scalar_lane(std::int64_t a, std::int64_t b, std::int64_t& out,
                               unsigned char& ok) noexcept -> void {
    std::int64_t prod = 0;
    if (__builtin_mul_overflow(a, b, &prod)) {
        ok = 0;
        out = prod;
    } else {
        ok = 1;
        out = prod;
    }
}

// Scalar truncating division for a single lane.
// Truncating integer division fails on zero divisor or on INT64_MIN / -1 (which cannot be represented).
inline auto nc_quot_scalar_lane(std::int64_t a, std::int64_t b, std::int64_t& out,
                                unsigned char& ok) noexcept -> void {
    if (b == 0 || (a == std::numeric_limits<std::int64_t>::min() && b == -1)) {
        ok = 0;
        out = 0;
    } else {
        ok = 1;
        out = a / b;
    }
}

auto nc_add_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                   unsigned char* ok, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        nc_add_scalar_lane(a[i], b[i], out[i], ok[i]);
    }
}

auto nc_sub_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                   unsigned char* ok, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        nc_sub_scalar_lane(a[i], b[i], out[i], ok[i]);
    }
}

auto nc_mul_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                   unsigned char* ok, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        nc_mul_scalar_lane(a[i], b[i], out[i], ok[i]);
    }
}

auto nc_quot_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                    unsigned char* ok, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        nc_quot_scalar_lane(a[i], b[i], out[i], ok[i]);
    }
}

auto nc_cmp_lt_scalar(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                      std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (a[i] < b[i]) ? 1 : 0;
    }
}

auto nc_cmp_le_scalar(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                      std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (a[i] <= b[i]) ? 1 : 0;
    }
}

auto nc_cmp_eq_scalar(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                      std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (a[i] == b[i]) ? 1 : 0;
    }
}

#if defined(__x86_64__) || defined(__i386__)

// --- AVX2 (256-bit, 4 lanes of int64) ---

// AVX2 checked addition: performs modular 64-bit addition with _mm256_add_epi64, computes
// the overflow sign mask in parallel using bitwise logic (~(a ^ b) & (a ^ sum)), and extracts
// the lane sign bits into a 4-bit mask with _mm256_movemask_pd without branching.
[[gnu::target("avx2")]] auto nc_add_avx2(const std::int64_t* a, const std::int64_t* b,
                                         std::int64_t* out, unsigned char* ok,
                                         std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i vsum = _mm256_add_epi64(va, vb);
        const __m256i vxor_ab = _mm256_xor_si256(va, vb);
        const __m256i vxor_asum = _mm256_xor_si256(va, vsum);
        const __m256i vovf = _mm256_andnot_si256(vxor_ab, vxor_asum);
        const int mask = _mm256_movemask_pd(_mm256_castsi256_pd(vovf));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vsum);
        ok[i + 0] = static_cast<unsigned char>((mask & 1) ^ 1);
        ok[i + 1] = static_cast<unsigned char>(((mask >> 1) & 1) ^ 1);
        ok[i + 2] = static_cast<unsigned char>(((mask >> 2) & 1) ^ 1);
        ok[i + 3] = static_cast<unsigned char>(((mask >> 3) & 1) ^ 1);
    }
    if (i < n) {
        nc_add_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

// AVX2 checked subtraction: modular subtraction via _mm256_sub_epi64, overflow mask computed
// via ((a ^ b) & (a ^ diff)), sign bit extraction via _mm256_movemask_pd.
[[gnu::target("avx2")]] auto nc_sub_avx2(const std::int64_t* a, const std::int64_t* b,
                                         std::int64_t* out, unsigned char* ok,
                                         std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i vdiff = _mm256_sub_epi64(va, vb);
        const __m256i vxor_ab = _mm256_xor_si256(va, vb);
        const __m256i vxor_adiff = _mm256_xor_si256(va, vdiff);
        const __m256i vovf = _mm256_and_si256(vxor_ab, vxor_adiff);
        const int mask = _mm256_movemask_pd(_mm256_castsi256_pd(vovf));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vdiff);
        ok[i + 0] = static_cast<unsigned char>((mask & 1) ^ 1);
        ok[i + 1] = static_cast<unsigned char>(((mask >> 1) & 1) ^ 1);
        ok[i + 2] = static_cast<unsigned char>(((mask >> 2) & 1) ^ 1);
        ok[i + 3] = static_cast<unsigned char>(((mask >> 3) & 1) ^ 1);
    }
    if (i < n) {
        nc_sub_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

// AVX2 checked multiply: AVX2 does not have a 64x64 full multiply instruction.
// We vectorize the sufficient condition: both operands fitting within [-2^31, 2^31 - 1]
// can never overflow a 64-bit signed integer (max product 2^62 < 2^63 - 1). For lanes
// meeting this condition, _mm256_mul_epi32 computes the exact 64-bit product vectorially.
// Lanes outside this safe envelope fall back to the scalar checked multiply.
[[gnu::target("avx2")]] auto nc_mul_avx2(const std::int64_t* a, const std::int64_t* b,
                                         std::int64_t* out, unsigned char* ok,
                                         std::size_t n) noexcept -> void {
    const __m256i c_min = _mm256_set1_epi64x(-2147483648LL);
    const __m256i c_max = _mm256_set1_epi64x(2147483647LL);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i too_low_a = _mm256_cmpgt_epi64(c_min, va);
        const __m256i too_high_a = _mm256_cmpgt_epi64(va, c_max);
        const __m256i bad_a = _mm256_or_si256(too_low_a, too_high_a);
        const __m256i too_low_b = _mm256_cmpgt_epi64(c_min, vb);
        const __m256i too_high_b = _mm256_cmpgt_epi64(vb, c_max);
        const __m256i bad_b = _mm256_or_si256(too_low_b, too_high_b);
        const __m256i bad = _mm256_or_si256(bad_a, bad_b);
        const int bad_mask = _mm256_movemask_pd(_mm256_castsi256_pd(bad));
        if (bad_mask == 0) {
            const __m256i vprod = _mm256_mul_epi32(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vprod);
            ok[i + 0] = 1;
            ok[i + 1] = 1;
            ok[i + 2] = 1;
            ok[i + 3] = 1;
        } else {
            for (std::size_t k = 0; k < 4; ++k) {
                nc_mul_scalar_lane(a[i + k], b[i + k], out[i + k], ok[i + k]);
            }
        }
    }
    if (i < n) {
        nc_mul_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

// AVX2 truncating quotient: x86 lacks 64-bit integer vector division.
// We vectorize the failure tests (divisor == 0, and dividend == INT64_MIN with divisor == -1)
// across 4 lanes simultaneously, then perform the division scalar for validated safe lanes.
[[gnu::target("avx2")]] auto nc_quot_avx2(const std::int64_t* a, const std::int64_t* b,
                                          std::int64_t* out, unsigned char* ok,
                                          std::size_t n) noexcept -> void {
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vmin = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
    const __m256i vneg1 = _mm256_set1_epi64x(-1LL);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i eq_zero = _mm256_cmpeq_epi64(vb, vzero);
        const __m256i eq_min = _mm256_cmpeq_epi64(va, vmin);
        const __m256i eq_neg1 = _mm256_cmpeq_epi64(vb, vneg1);
        const __m256i ovf_case = _mm256_and_si256(eq_min, eq_neg1);
        const __m256i bad_vec = _mm256_or_si256(eq_zero, ovf_case);
        const int bad_mask = _mm256_movemask_pd(_mm256_castsi256_pd(bad_vec));
        for (std::size_t k = 0; k < 4; ++k) {
            if (((bad_mask >> k) & 1) != 0) {
                ok[i + k] = 0;
                out[i + k] = 0;
            } else {
                ok[i + k] = 1;
                out[i + k] = a[i + k] / b[i + k];
            }
        }
    }
    if (i < n) {
        nc_quot_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

[[gnu::target("avx2")]] auto nc_cmp_lt_avx2(const std::int64_t* a, const std::int64_t* b,
                                            unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i gt = _mm256_cmpgt_epi64(vb, va);
        const int mask = _mm256_movemask_pd(_mm256_castsi256_pd(gt));
        out[i + 0] = static_cast<unsigned char>(mask & 1);
        out[i + 1] = static_cast<unsigned char>((mask >> 1) & 1);
        out[i + 2] = static_cast<unsigned char>((mask >> 2) & 1);
        out[i + 3] = static_cast<unsigned char>((mask >> 3) & 1);
    }
    if (i < n) {
        nc_cmp_lt_scalar(a + i, b + i, out + i, n - i);
    }
}

[[gnu::target("avx2")]] auto nc_cmp_le_avx2(const std::int64_t* a, const std::int64_t* b,
                                            unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i gt = _mm256_cmpgt_epi64(va, vb);
        const int mask = _mm256_movemask_pd(_mm256_castsi256_pd(gt));
        out[i + 0] = static_cast<unsigned char>((mask & 1) ^ 1);
        out[i + 1] = static_cast<unsigned char>(((mask >> 1) & 1) ^ 1);
        out[i + 2] = static_cast<unsigned char>(((mask >> 2) & 1) ^ 1);
        out[i + 3] = static_cast<unsigned char>(((mask >> 3) & 1) ^ 1);
    }
    if (i < n) {
        nc_cmp_le_scalar(a + i, b + i, out + i, n - i);
    }
}

[[gnu::target("avx2")]] auto nc_cmp_eq_avx2(const std::int64_t* a, const std::int64_t* b,
                                            unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i eq = _mm256_cmpeq_epi64(va, vb);
        const int mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq));
        out[i + 0] = static_cast<unsigned char>(mask & 1);
        out[i + 1] = static_cast<unsigned char>((mask >> 1) & 1);
        out[i + 2] = static_cast<unsigned char>((mask >> 2) & 1);
        out[i + 3] = static_cast<unsigned char>((mask >> 3) & 1);
    }
    if (i < n) {
        nc_cmp_eq_scalar(a + i, b + i, out + i, n - i);
    }
}

// --- AVX-512 (512-bit, 8 lanes of int64, AVX512F + AVX512DQ) ---

[[gnu::target("avx512f,avx512dq")]] auto nc_add_avx512(const std::int64_t* a, const std::int64_t* b,
                                                       std::int64_t* out, unsigned char* ok,
                                                       std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __m512i vsum = _mm512_add_epi64(va, vb);
        const __m512i vxor_ab = _mm512_xor_si512(va, vb);
        const __m512i vxor_asum = _mm512_xor_si512(va, vsum);
        const __m512i vovf = _mm512_andnot_si512(vxor_ab, vxor_asum);
        const __mmask8 ovf_mask = _mm512_cmplt_epi64_mask(vovf, _mm512_setzero_si512());
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), vsum);
        for (std::size_t k = 0; k < 8; ++k) {
            ok[i + k] = static_cast<unsigned char>((~ovf_mask >> k) & 1);
        }
    }
    if (i < n) {
        nc_add_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

[[gnu::target("avx512f,avx512dq")]] auto nc_sub_avx512(const std::int64_t* a, const std::int64_t* b,
                                                       std::int64_t* out, unsigned char* ok,
                                                       std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __m512i vdiff = _mm512_sub_epi64(va, vb);
        const __m512i vxor_ab = _mm512_xor_si512(va, vb);
        const __m512i vxor_adiff = _mm512_xor_si512(va, vdiff);
        const __m512i vovf = _mm512_and_si512(vxor_ab, vxor_adiff);
        const __mmask8 ovf_mask = _mm512_cmplt_epi64_mask(vovf, _mm512_setzero_si512());
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), vdiff);
        for (std::size_t k = 0; k < 8; ++k) {
            ok[i + k] = static_cast<unsigned char>((~ovf_mask >> k) & 1);
        }
    }
    if (i < n) {
        nc_sub_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

// AVX-512 checked multiply: uses _mm512_mullo_epi64 (AVX512DQ) to compute the low 64 bits of
// products in parallel. The cheap sufficient condition (operands in [-2^31, 2^31 - 1]) is
// evaluated using 64-bit compare-to-mask instructions; lanes proven safe have ok set to 1.
// Lanes that exceed this range fall back to the scalar checked multiply.
[[gnu::target("avx512f,avx512dq")]] auto nc_mul_avx512(const std::int64_t* a, const std::int64_t* b,
                                                       std::int64_t* out, unsigned char* ok,
                                                       std::size_t n) noexcept -> void {
    const __m512i c_min = _mm512_set1_epi64(-2147483648LL);
    const __m512i c_max = _mm512_set1_epi64(2147483647LL);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __mmask8 bad_a = _mm512_cmplt_epi64_mask(va, c_min) | _mm512_cmpgt_epi64_mask(va, c_max);
        const __mmask8 bad_b = _mm512_cmplt_epi64_mask(vb, c_min) | _mm512_cmpgt_epi64_mask(vb, c_max);
        const __mmask8 bad = bad_a | bad_b;
        const __m512i vprod = _mm512_mullo_epi64(va, vb);
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), vprod);
        if (bad == 0) {
            for (std::size_t k = 0; k < 8; ++k) {
                ok[i + k] = 1;
            }
        } else {
            for (std::size_t k = 0; k < 8; ++k) {
                if (((bad >> k) & 1) == 0) {
                    ok[i + k] = 1;
                } else {
                    nc_mul_scalar_lane(a[i + k], b[i + k], out[i + k], ok[i + k]);
                }
            }
        }
    }
    if (i < n) {
        nc_mul_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

// AVX-512 truncating quotient: failure tests (b == 0 or INT64_MIN / -1) are evaluated
// into mask registers, and division is executed scalar for valid lanes.
[[gnu::target("avx512f,avx512dq")]] auto nc_quot_avx512(const std::int64_t* a, const std::int64_t* b,
                                                        std::int64_t* out, unsigned char* ok,
                                                        std::size_t n) noexcept -> void {
    const __m512i vzero = _mm512_setzero_si512();
    const __m512i vmin = _mm512_set1_epi64(std::numeric_limits<std::int64_t>::min());
    const __m512i vneg1 = _mm512_set1_epi64(-1LL);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __mmask8 eq_zero = _mm512_cmpeq_epi64_mask(vb, vzero);
        const __mmask8 eq_min = _mm512_cmpeq_epi64_mask(va, vmin);
        const __mmask8 eq_neg1 = _mm512_cmpeq_epi64_mask(vb, vneg1);
        const __mmask8 bad = eq_zero | (eq_min & eq_neg1);
        for (std::size_t k = 0; k < 8; ++k) {
            if (((bad >> k) & 1) != 0) {
                ok[i + k] = 0;
                out[i + k] = 0;
            } else {
                ok[i + k] = 1;
                out[i + k] = a[i + k] / b[i + k];
            }
        }
    }
    if (i < n) {
        nc_quot_scalar(a + i, b + i, out + i, ok + i, n - i);
    }
}

[[gnu::target("avx512f,avx512dq")]] auto nc_cmp_lt_avx512(const std::int64_t* a, const std::int64_t* b,
                                                          unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __mmask8 mask = _mm512_cmplt_epi64_mask(va, vb);
        for (std::size_t k = 0; k < 8; ++k) {
            out[i + k] = static_cast<unsigned char>((mask >> k) & 1);
        }
    }
    if (i < n) {
        nc_cmp_lt_scalar(a + i, b + i, out + i, n - i);
    }
}

[[gnu::target("avx512f,avx512dq")]] auto nc_cmp_le_avx512(const std::int64_t* a, const std::int64_t* b,
                                                          unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __mmask8 mask = _mm512_cmple_epi64_mask(va, vb);
        for (std::size_t k = 0; k < 8; ++k) {
            out[i + k] = static_cast<unsigned char>((mask >> k) & 1);
        }
    }
    if (i < n) {
        nc_cmp_le_scalar(a + i, b + i, out + i, n - i);
    }
}

[[gnu::target("avx512f,avx512dq")]] auto nc_cmp_eq_avx512(const std::int64_t* a, const std::int64_t* b,
                                                          unsigned char* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const __mmask8 mask = _mm512_cmpeq_epi64_mask(va, vb);
        for (std::size_t k = 0; k < 8; ++k) {
            out[i + k] = static_cast<unsigned char>((mask >> k) & 1);
        }
    }
    if (i < n) {
        nc_cmp_eq_scalar(a + i, b + i, out + i, n - i);
    }
}

#endif  // __x86_64__ || __i386__

// CPU feature detection WITHOUT `__builtin_cpu_supports`.
//
// That builtin reads `__cpu_model`, a symbol supplied by clang's compiler-rt builtins archive.
// This file is emitted into a STANDALONE translation unit that the user compiles themselves, so
// depending on it would impose a link requirement they were never told about — the failure
// arrives as an undefined-symbol error in generated code, which is the least debuggable place
// for it. Reading CPUID directly keeps the emitted file self-contained.
//
// Checking the OS as well as the CPU is not optional: a CPU can report AVX-512 while the
// operating system has not enabled the wide register state, and using it then faults. XGETBV
// reports what the OS actually preserves across a context switch.
[[nodiscard]] inline auto nc_cpuid(unsigned leaf, unsigned sub, std::array<unsigned, 4>& out)
    noexcept -> bool {
#if defined(_WIN32)
    std::array<int, 4> regs{};
    __cpuidex(regs.data(), static_cast<int>(leaf), static_cast<int>(sub));
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<unsigned>(regs[i]);
    }
    return true;
#else
    return __get_cpuid_count(leaf, sub, &out[0], &out[1], &out[2], &out[3]) != 0;
#endif
}

[[nodiscard]] inline auto nc_xcr0() noexcept -> unsigned long long {
    return static_cast<unsigned long long>(_xgetbv(0));
}

[[nodiscard]] inline auto nc_cpu_has_avx2() noexcept -> bool {
    std::array<unsigned, 4> r{};
    if (!nc_cpuid(1, 0, r)) {
        return false;
    }
    const bool osxsave = (r[2] & (1U << 27)) != 0U;
    const bool avx = (r[2] & (1U << 28)) != 0U;
    if (!osxsave || !avx) {
        return false;
    }
    if ((nc_xcr0() & 0x6ULL) != 0x6ULL) {
        return false;  // the OS is not preserving YMM state
    }
    std::array<unsigned, 4> r7{};
    if (!nc_cpuid(7, 0, r7)) {
        return false;
    }
    return (r7[1] & (1U << 5)) != 0U;
}

[[nodiscard]] inline auto nc_cpu_has_avx512() noexcept -> bool {
    if (!nc_cpu_has_avx2()) {
        return false;
    }
    if ((nc_xcr0() & 0xE6ULL) != 0xE6ULL) {
        return false;  // the OS is not preserving ZMM/opmask state
    }
    std::array<unsigned, 4> r7{};
    if (!nc_cpuid(7, 0, r7)) {
        return false;
    }
    const bool f = (r7[1] & (1U << 16)) != 0U;
    const bool dq = (r7[1] & (1U << 17)) != 0U;
    return f && dq;
}
}  // anonymous namespace

// ===========================================================================
// Public entry points
// ===========================================================================

[[nodiscard]] auto nc_detect_isa() noexcept -> nc_isa {
#if defined(__x86_64__) || defined(__i386__)
    if (nc_cpu_has_avx512()) {
        return nc_isa::avx512;
    }
    if (nc_cpu_has_avx2()) {
        return nc_isa::avx2;
    }
#endif
    return nc_isa::scalar;
}

auto nc_add_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_add_avx512(a, b, out, ok, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_add_avx2(a, b, out, ok, n);
        return;
    }
#endif
    nc_add_scalar(a, b, out, ok, n);
}

auto nc_sub_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_sub_avx512(a, b, out, ok, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_sub_avx2(a, b, out, ok, n);
        return;
    }
#endif
    nc_sub_scalar(a, b, out, ok, n);
}

auto nc_mul_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                  unsigned char* ok, std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_mul_avx512(a, b, out, ok, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_mul_avx2(a, b, out, ok, n);
        return;
    }
#endif
    nc_mul_scalar(a, b, out, ok, n);
}

auto nc_quot_batch(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                   unsigned char* ok, std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_quot_avx512(a, b, out, ok, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_quot_avx2(a, b, out, ok, n);
        return;
    }
#endif
    nc_quot_scalar(a, b, out, ok, n);
}

auto nc_cmp_lt_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_cmp_lt_avx512(a, b, out, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_cmp_lt_avx2(a, b, out, n);
        return;
    }
#endif
    nc_cmp_lt_scalar(a, b, out, n);
}

auto nc_cmp_le_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_cmp_le_avx512(a, b, out, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_cmp_le_avx2(a, b, out, n);
        return;
    }
#endif
    nc_cmp_le_scalar(a, b, out, n);
}

auto nc_cmp_eq_batch(const std::int64_t* a, const std::int64_t* b, unsigned char* out,
                     std::size_t n) noexcept -> void {
    if (n == 0) {
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    const auto isa = nc_detect_isa();
    if (isa == nc_isa::avx512) {
        nc_cmp_eq_avx512(a, b, out, n);
        return;
    }
    if (isa == nc_isa::avx2) {
        nc_cmp_eq_avx2(a, b, out, n);
        return;
    }
#endif
    nc_cmp_eq_scalar(a, b, out, n);
}

// Forced-path variants: fail honestly (return false) if requested ISA is unsupported by hardware.

[[nodiscard]] auto nc_add_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_add_scalar(a, b, out, ok, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_add_avx2(a, b, out, ok, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_add_avx512(a, b, out, ok, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_mul_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_mul_scalar(a, b, out, ok, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_mul_avx2(a, b, out, ok, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_mul_avx512(a, b, out, ok, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_sub_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                     std::int64_t* out, unsigned char* ok,
                                     std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_sub_scalar(a, b, out, ok, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_sub_avx2(a, b, out, ok, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_sub_avx512(a, b, out, ok, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_quot_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                      std::int64_t* out, unsigned char* ok,
                                      std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_quot_scalar(a, b, out, ok, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_quot_avx2(a, b, out, ok, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_quot_avx512(a, b, out, ok, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_cmp_lt_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_cmp_lt_scalar(a, b, out, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_lt_avx2(a, b, out, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_lt_avx512(a, b, out, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_cmp_le_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_cmp_le_scalar(a, b, out, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_le_avx2(a, b, out, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_le_avx512(a, b, out, n);
        }
        return true;
    }
#endif
    return false;
}

[[nodiscard]] auto nc_cmp_eq_batch_with(nc_isa isa, const std::int64_t* a, const std::int64_t* b,
                                        unsigned char* out, std::size_t n) noexcept -> bool {
    if (isa == nc_isa::scalar) {
        if (n > 0) {
            nc_cmp_eq_scalar(a, b, out, n);
        }
        return true;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (isa == nc_isa::avx2) {
        if (!nc_cpu_has_avx2()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_eq_avx2(a, b, out, n);
        }
        return true;
    }
    if (isa == nc_isa::avx512) {
        if (!nc_cpu_has_avx512()) {
            return false;
        }
        if (n > 0) {
            nc_cmp_eq_avx512(a, b, out, n);
        }
        return true;
    }
#endif
    return false;
}
