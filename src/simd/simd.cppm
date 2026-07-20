// NimbleCAS multi-register SIMD engine with runtime dynamic dispatch (ROADMAP 4,
// Code Policy Rule 29).
// @author Olumuyiwa Oluwasanmi
//
// Elementwise float32 array kernels are dispatched at process startup to the best
// instruction set the CPU supports, waterfalling AVX-512 -> AVX2 -> scalar. The
// canonical build targets x86-64-v3 (AVX2+FMA) as its baseline, so AVX2 is the
// practical floor and the scalar path is the portable correctness fallback; AVX-512
// is engaged per-function via [[gnu::target(...)]] so the binary stays runnable on
// machines without it (Rule 50).
//
// All kernels are strictly elementwise (out[i] = f(a[i], b[i], ...)), so every ISA
// path produces bit-identical results regardless of vector width (Rule 55). Reductions
// whose result depends on summation order are deliberately excluded here.
//
// PORTABILITY (e.g. wasm32 under Emscripten): <immintrin.h> and __builtin_cpu_supports
// are x86-specific. Emscripten's <immintrin.h> is a compiling-but-empty stub (it declares
// none of the __m256/__m512 types or _mm*_ intrinsics), so unconditionally including it and
// calling into it, as this file used to, silently made every non-x86 build of this module
// fail. The AVX/AVX2/AVX-512 code paths and the cpu-supports probe are therefore compiled
// only on __x86_64__/__i386__; every other target (wasm32, ARM, ...) uses ONLY the scalar
// path, which is both exact and already the documented "portable correctness fallback" --
// there is no hardware SIMD ISA to waterfall to there, so scalar-only is the honest answer,
// not a degraded one.

module;
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

export module nimblecas.simd;

import std;

export namespace nimblecas::simd {

enum class Isa : std::uint8_t { scalar, avx, avx2, avx512 };

[[nodiscard]] constexpr auto to_string_view(Isa isa) noexcept -> std::string_view {
    switch (isa) {
        case Isa::scalar: return "scalar";
        case Isa::avx:    return "avx";
        case Isa::avx2:   return "avx2";
        case Isa::avx512: return "avx512";
    }
    return "unknown";
}

// The ISA selected for this process (highest supported).
[[nodiscard]] auto active_isa() noexcept -> Isa;

// out = a + b   (out may alias a or b)
auto add(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
    -> void;

// out = a * b
auto mul(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
    -> void;

// out = a * scale + b   (fused multiply-add)
auto axpy(float scale, std::span<const float> a, std::span<const float> b,
          std::span<float> out) noexcept -> void;

// acc = acc * x + c   (elementwise fused multiply-add, c broadcast). One Horner step
// for evaluating a polynomial at many points at once.
auto horner_step(std::span<float> acc, std::span<const float> x, float c) noexcept -> void;

// out[i] = exp(x[i]) in DOUBLE precision. A deterministic (≈1 ulp) exponential: Cody-Waite
// range reduction to |r| <= ln2/2 then a degree-13 Horner polynomial, scaled by 2^k via
// exponent construction. Every ISA path (AVX-512, scalar) is bit-identical (the scalar fallback
// uses std::fma to match the vector FMA — Rule 55), so results are reproducible across hosts.
// Accurate to 1 ulp vs libm (validated in simd_tests). The hot path for batch Black-Scholes /
// Monte-Carlo GBM exponentials.
//
// DOMAIN (honesty boundary): defined for finite x with |x| <~ 700 — the whole representable
// exp range, which comfortably contains every finance exponential (bounded drift + vol*sqrt(T)*z).
// The 2^k step is a raw exponent construction, so inputs that would overflow (x >~ 709) or
// underflow (x <~ -745) double are OUT OF CONTRACT: they return an unspecified value, not a
// saturated inf/0. Callers outside that range must guard or clamp first.
auto exp_into(std::span<const double> x, std::span<double> out) noexcept -> void;

// log(x) for a single DOUBLE — the scalar reference AND public scalar entry, bit-identical to the
// per-lane result of log_into (Rule 55). A deterministic (≈1 ulp) natural log: bit-manipulation
// mantissa/exponent split, an atanh series on s = (m-1)/(m+1), then e*ln2 recombination via std::fma.
//
// DOMAIN (honesty boundary): defined for finite, positive, NORMAL x. The reduction reads the raw
// exponent/mantissa fields, so x <= 0, subnormal, inf or NaN are OUT OF CONTRACT (unspecified value,
// not a diagnosed error). Callers must guard first — the Monte-Carlo tail uses it on p in (1e-15, 1).
[[nodiscard]] auto log_one(double x) noexcept -> double;

// out[i] = log(x[i]) in DOUBLE precision, dynamically dispatched (AVX-512 -> scalar). Every ISA path
// is bit-identical to log_one applied per element, so results are reproducible across hosts. Same
// positive-normal domain as log_one. The hot path for the Monte-Carlo inverse-normal tail (the ~5 %
// of draws that need a log) and any batch logarithm.
auto log_into(std::span<const double> x, std::span<double> out) noexcept -> void;

}  // namespace nimblecas::simd

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::simd {
namespace {

// --- exp(double) constants: Cody-Waite ln2 split + degree-13 Taylor (1/n!) on |r| <= ln2/2 ---
inline constexpr double kLog2e = 1.4426950408889634;
inline constexpr double kLn2Hi = 6.93145751953125e-1;          // ln2 = kLn2Hi + kLn2Lo, split so
inline constexpr double kLn2Lo = 1.42860682030941723212e-6;    // k*kLn2Hi is exact in double.
inline constexpr double kExpC13 = 1.0 / 6227020800.0;
inline constexpr double kExpC12 = 1.0 / 479001600.0;
inline constexpr double kExpC11 = 1.0 / 39916800.0;
inline constexpr double kExpC10 = 1.0 / 3628800.0;
inline constexpr double kExpC9 = 1.0 / 362880.0;
inline constexpr double kExpC8 = 1.0 / 40320.0;
inline constexpr double kExpC7 = 1.0 / 5040.0;
inline constexpr double kExpC6 = 1.0 / 720.0;
inline constexpr double kExpC5 = 1.0 / 120.0;
inline constexpr double kExpC4 = 1.0 / 24.0;
inline constexpr double kExpC3 = 1.0 / 6.0;
inline constexpr double kExpC2 = 0.5;

// Scalar exp(double): the correctness reference AND the portable fallback. Bit-identical to the
// AVX-512 path — same constants, same std::fma order, same 2^k exponent construction (Rule 55).
[[nodiscard]] auto exp_one(double x) noexcept -> double {
    const double k = std::nearbyint(x * kLog2e);
    double r = std::fma(-k, kLn2Hi, x);
    r = std::fma(-k, kLn2Lo, r);
    double p = kExpC13;
    p = std::fma(p, r, kExpC12);
    p = std::fma(p, r, kExpC11);
    p = std::fma(p, r, kExpC10);
    p = std::fma(p, r, kExpC9);
    p = std::fma(p, r, kExpC8);
    p = std::fma(p, r, kExpC7);
    p = std::fma(p, r, kExpC6);
    p = std::fma(p, r, kExpC5);
    p = std::fma(p, r, kExpC4);
    p = std::fma(p, r, kExpC3);
    p = std::fma(p, r, kExpC2);
    p = std::fma(p, r, 1.0);
    p = std::fma(p, r, 1.0);
    const auto ki = static_cast<std::int64_t>(k);
    const double pow2 = std::bit_cast<double>(static_cast<std::uint64_t>((ki + 1023) << 52));
    return p * pow2;
}
auto exp_d_scalar(const double* in, double* out, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = exp_one(in[i]);
    }
}

// --- log(double) constants: reuse the exp ln2 split (kLn2Hi/kLn2Lo) + atanh series 1 + z/3 +
// z^2/5 + ... on z = s^2, s = (m-1)/(m+1), with the mantissa centred in [sqrt2/2, sqrt2). ---
inline constexpr double kSqrt2 = 1.4142135623730951;
inline constexpr double kLogC1 = 1.0 / 3.0;
inline constexpr double kLogC2 = 1.0 / 5.0;
inline constexpr double kLogC3 = 1.0 / 7.0;
inline constexpr double kLogC4 = 1.0 / 9.0;
inline constexpr double kLogC5 = 1.0 / 11.0;
inline constexpr double kLogC6 = 1.0 / 13.0;
inline constexpr double kLogC7 = 1.0 / 15.0;
inline constexpr double kLogC8 = 1.0 / 17.0;

// Scalar log(double): the correctness reference AND the portable fallback. Bit-identical to the
// AVX-512 path — same field extraction, same sqrt2 centring, same std::fma order (Rule 55).
[[nodiscard]] auto log_scalar_one(double x) noexcept -> double {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    double e = static_cast<double>(static_cast<std::int64_t>((bits >> 52) & 0x7ff) - 1023);
    double m = std::bit_cast<double>((bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL);
    if (m > kSqrt2) {  // centre the mantissa in [sqrt2/2, sqrt2) so the atanh argument stays small
        m *= 0.5;
        e += 1.0;
    }
    const double s = (m - 1.0) / (m + 1.0);
    const double z = s * s;
    double poly = kLogC8;
    poly = std::fma(poly, z, kLogC7);
    poly = std::fma(poly, z, kLogC6);
    poly = std::fma(poly, z, kLogC5);
    poly = std::fma(poly, z, kLogC4);
    poly = std::fma(poly, z, kLogC3);
    poly = std::fma(poly, z, kLogC2);
    poly = std::fma(poly, z, kLogC1);
    poly = std::fma(poly, z, 1.0);
    const double log_m = (s + s) * poly;  // 2*s exactly (add, no rounding)
    double res = std::fma(e, kLn2Lo, log_m);
    res = std::fma(e, kLn2Hi, res);
    return res;
}
auto log_d_scalar(const double* in, double* out, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = log_scalar_one(in[i]);
    }
}

// --- scalar reference paths (also the portable fallback) ---
auto add_scalar(const float* a, const float* b, float* out, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
auto mul_scalar(const float* a, const float* b, float* out, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}
auto axpy_scalar(float s, const float* a, const float* b, float* out, std::size_t n) noexcept
    -> void {
    for (std::size_t i = 0; i < n; ++i) {
        // std::fma (single rounding) so this is bit-identical to the vector _mm*_fmadd
        // paths — the whole point of keeping every ISA path elementwise (Rule 55).
        out[i] = std::fma(a[i], s, b[i]);
    }
}
auto horner_scalar(float* acc, const float* x, float c, std::size_t n) noexcept -> void {
    for (std::size_t i = 0; i < n; ++i) {
        acc[i] = std::fma(acc[i], x[i], c);
    }
}

#if defined(__x86_64__) || defined(__i386__)
// --- 256-bit AVX paths (add/mul need only AVX, shared by the AVX and AVX2 tiers) ---
auto add_avx256(const float* a, const float* b, float* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
                         _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    add_scalar(a + i, b + i, out + i, n - i);
}
auto mul_avx256(const float* a, const float* b, float* out, std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
                         _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    mul_scalar(a + i, b + i, out + i, n - i);
}
// AVX tier (no FMA): keep axpy on std::fma so it stays bit-identical to the fused
// vector tiers (Rule 55). add/mul above are already 256-bit-vectorised here.
auto axpy_avx(float s, const float* a, const float* b, float* out, std::size_t n) noexcept
    -> void {
    axpy_scalar(s, a, b, out, n);
}
auto horner_avx(float* acc, const float* x, float c, std::size_t n) noexcept -> void {
    horner_scalar(acc, x, c, n);  // no FMA on this tier -> std::fma keeps it bit-identical
}
// AVX2 tier: FMA is available, so axpy vectorises with a single-rounding fmadd.
auto axpy_avx2(float s, const float* a, const float* b, float* out, std::size_t n) noexcept
    -> void {
    const __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(a + i), vs, _mm256_loadu_ps(b + i)));
    }
    axpy_scalar(s, a + i, b + i, out + i, n - i);
}
auto horner_avx2(float* acc, const float* x, float c, std::size_t n) noexcept -> void {
    const __m256 vc = _mm256_set1_ps(c);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(acc + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(acc + i), _mm256_loadu_ps(x + i), vc));
    }
    horner_scalar(acc + i, x + i, c, n - i);
}

// --- AVX-512 paths (opt-in per-function so the binary stays portable, Rule 50) ---
[[gnu::target("avx512f")]] auto add_avx512(const float* a, const float* b, float* out,
                                           std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
                         _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    add_avx256(a + i, b + i, out + i, n - i);
}
[[gnu::target("avx512f")]] auto mul_avx512(const float* a, const float* b, float* out,
                                           std::size_t n) noexcept -> void {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
                         _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    mul_avx256(a + i, b + i, out + i, n - i);
}
[[gnu::target("avx512f")]] auto axpy_avx512(float s, const float* a, const float* b, float* out,
                                            std::size_t n) noexcept -> void {
    const __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(a + i), vs, _mm512_loadu_ps(b + i)));
    }
    axpy_avx2(s, a + i, b + i, out + i, n - i);
}
[[gnu::target("avx512f")]] auto horner_avx512(float* acc, const float* x, float c,
                                              std::size_t n) noexcept -> void {
    const __m512 vc = _mm512_set1_ps(c);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(acc + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(acc + i), _mm512_loadu_ps(x + i), vc));
    }
    horner_avx2(acc + i, x + i, c, n - i);
}
// AVX-512 exp(double), 8 lanes. Bit-identical to exp_one: the reduction (roundscale = round to
// nearest even), the FMA order, and the 2^k exponent construction all match the scalar path.
// Needs AVX512DQ for _mm512_cvtpd_epi64; the tail falls through to the scalar reference.
[[gnu::target("avx512f,avx512dq")]] auto exp_d_avx512(const double* in, double* out,
                                                      std::size_t n) noexcept -> void {
    const __m512d log2e = _mm512_set1_pd(kLog2e);
    const __m512d ln2hi = _mm512_set1_pd(kLn2Hi);
    const __m512d ln2lo = _mm512_set1_pd(kLn2Lo);
    const __m512d one = _mm512_set1_pd(1.0);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_loadu_pd(in + i);
        const __m512d k = _mm512_roundscale_pd(_mm512_mul_pd(x, log2e),
                                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512d r = _mm512_fnmadd_pd(k, ln2hi, x);
        r = _mm512_fnmadd_pd(k, ln2lo, r);
        __m512d p = _mm512_set1_pd(kExpC13);
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC12));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC11));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC10));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC9));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC8));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC7));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC6));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC5));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC4));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC3));
        p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(kExpC2));
        p = _mm512_fmadd_pd(p, r, one);
        p = _mm512_fmadd_pd(p, r, one);
        __m512i ki = _mm512_cvtpd_epi64(k);
        ki = _mm512_add_epi64(ki, _mm512_set1_epi64(1023));
        ki = _mm512_slli_epi64(ki, 52);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(p, _mm512_castsi512_pd(ki)));
    }
    exp_d_scalar(in + i, out + i, n - i);
}
// AVX-512 log(double), 8 lanes. Bit-identical to log_scalar_one: the same bit-field split, the same
// sqrt2 mantissa centring (via a compare-mask), the same FMA order. Needs AVX512DQ for
// _mm512_cvtepi64_pd; the tail falls through to the scalar reference.
[[gnu::target("avx512f,avx512dq")]] auto log_d_avx512(const double* in, double* out,
                                                      std::size_t n) noexcept -> void {
    const __m512i expmask = _mm512_set1_epi64(0x7ff);
    const __m512i mantmask = _mm512_set1_epi64(0x000fffffffffffffLL);
    const __m512i mantbias = _mm512_set1_epi64(0x3ff0000000000000LL);
    const __m512i bias = _mm512_set1_epi64(1023);
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d sqrt2 = _mm512_set1_pd(kSqrt2);
    const __m512d ln2hi = _mm512_set1_pd(kLn2Hi);
    const __m512d ln2lo = _mm512_set1_pd(kLn2Lo);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i b = _mm512_loadu_si512(in + i);
        const __m512i ei = _mm512_sub_epi64(_mm512_and_si512(_mm512_srli_epi64(b, 52), expmask), bias);
        __m512d e = _mm512_cvtepi64_pd(ei);
        __m512d m = _mm512_castsi512_pd(_mm512_or_si512(_mm512_and_si512(b, mantmask), mantbias));
        const __mmask8 k = _mm512_cmp_pd_mask(m, sqrt2, _CMP_GT_OQ);
        m = _mm512_mask_mul_pd(m, k, m, half);     // m *= 0.5 where m > sqrt2
        e = _mm512_mask_add_pd(e, k, e, one);      // e += 1 there
        const __m512d s = _mm512_div_pd(_mm512_sub_pd(m, one), _mm512_add_pd(m, one));
        const __m512d z = _mm512_mul_pd(s, s);
        __m512d poly = _mm512_set1_pd(kLogC8);
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC7));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC6));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC5));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC4));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC3));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC2));
        poly = _mm512_fmadd_pd(poly, z, _mm512_set1_pd(kLogC1));
        poly = _mm512_fmadd_pd(poly, z, one);
        const __m512d log_m = _mm512_mul_pd(_mm512_add_pd(s, s), poly);
        __m512d res = _mm512_fmadd_pd(e, ln2lo, log_m);
        res = _mm512_fmadd_pd(e, ln2hi, res);
        _mm512_storeu_pd(out + i, res);
    }
    log_d_scalar(in + i, out + i, n - i);
}
#endif  // __x86_64__ || __i386__

// Runtime CPU capability query, evaluated once. Waterfalls
// AVX-512 -> AVX2(+FMA) -> AVX -> scalar. Off x86 there is no such ISA to waterfall
// through, so this always reports scalar (see the portability note at the top of file).
[[nodiscard]] auto detect_isa() noexcept -> Isa {
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx512f") != 0) {
        return Isa::avx512;
    }
    if (__builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0) {
        return Isa::avx2;
    }
    if (__builtin_cpu_supports("avx") != 0) {
        return Isa::avx;
    }
#endif
    return Isa::scalar;
}

// Function-pointer dispatch tables, resolved once at static-init time.
struct Dispatch {
    Isa isa;
    void (*add)(const float*, const float*, float*, std::size_t);
    void (*mul)(const float*, const float*, float*, std::size_t);
    void (*axpy)(float, const float*, const float*, float*, std::size_t);
    void (*horner)(float*, const float*, float, std::size_t);
    void (*exp_d)(const double*, double*, std::size_t);
    void (*log_d)(const double*, double*, std::size_t);
};

[[nodiscard]] auto make_dispatch() noexcept -> Dispatch {
    // detect_isa() is called UNCONDITIONALLY (even though it always returns Isa::scalar
    // off x86) so it is never an unused function there; only the AVX/AVX2/AVX-512 case
    // labels -- the ones referencing intrinsic-backed functions -- are guarded. The
    // `default:` covers the enumerators the non-x86 build leaves unhandled, so neither
    // platform trips -Wswitch.
    switch (detect_isa()) {
#if defined(__x86_64__) || defined(__i386__)
        case Isa::avx512: {
            // exp_d/log_d need AVX512DQ (_mm512_cvtpd_epi64 / _mm512_cvtepi64_pd); fall back to the
            // bit-identical scalar transcendentals on an AVX512F-only part (e.g. KNL) while the
            // float kernels stay vectorised.
            const bool dq = __builtin_cpu_supports("avx512dq") != 0;
            return {Isa::avx512, &add_avx512, &mul_avx512, &axpy_avx512, &horner_avx512,
                    dq ? &exp_d_avx512 : &exp_d_scalar, dq ? &log_d_avx512 : &log_d_scalar};
        }
        case Isa::avx2:
            return {Isa::avx2, &add_avx256, &mul_avx256, &axpy_avx2, &horner_avx2, &exp_d_scalar,
                    &log_d_scalar};
        case Isa::avx:
            return {Isa::avx, &add_avx256, &mul_avx256, &axpy_avx, &horner_avx, &exp_d_scalar,
                    &log_d_scalar};
#endif
        case Isa::scalar:
        default:
            break;
    }
    return {Isa::scalar, &add_scalar, &mul_scalar, &axpy_scalar, &horner_scalar, &exp_d_scalar,
            &log_d_scalar};
}

const Dispatch g_dispatch = make_dispatch();

// Smallest common length across the three spans, so a caller mismatch can never
// read/write out of bounds.
[[nodiscard]] auto common_size(std::size_t a, std::size_t b, std::size_t c) noexcept
    -> std::size_t {
    return std::min({a, b, c});
}

}  // namespace

auto active_isa() noexcept -> Isa { return g_dispatch.isa; }

auto add(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
    -> void {
    const std::size_t n = common_size(a.size(), b.size(), out.size());
    g_dispatch.add(a.data(), b.data(), out.data(), n);
}

auto mul(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
    -> void {
    const std::size_t n = common_size(a.size(), b.size(), out.size());
    g_dispatch.mul(a.data(), b.data(), out.data(), n);
}

auto axpy(float scale, std::span<const float> a, std::span<const float> b,
          std::span<float> out) noexcept -> void {
    const std::size_t n = common_size(a.size(), b.size(), out.size());
    g_dispatch.axpy(scale, a.data(), b.data(), out.data(), n);
}

auto horner_step(std::span<float> acc, std::span<const float> x, float c) noexcept -> void {
    const std::size_t n = std::min(acc.size(), x.size());
    g_dispatch.horner(acc.data(), x.data(), c, n);
}

auto exp_into(std::span<const double> x, std::span<double> out) noexcept -> void {
    const std::size_t n = std::min(x.size(), out.size());
    g_dispatch.exp_d(x.data(), out.data(), n);
}

auto log_one(double x) noexcept -> double { return log_scalar_one(x); }

auto log_into(std::span<const double> x, std::span<double> out) noexcept -> void {
    const std::size_t n = std::min(x.size(), out.size());
    g_dispatch.log_d(x.data(), out.data(), n);
}

}  // namespace nimblecas::simd
