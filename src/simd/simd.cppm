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

module;
#include <immintrin.h>

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

}  // namespace nimblecas::simd

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::simd {
namespace {

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

// Runtime CPU capability query, evaluated once. Waterfalls
// AVX-512 -> AVX2(+FMA) -> AVX -> scalar.
[[nodiscard]] auto detect_isa() noexcept -> Isa {
    if (__builtin_cpu_supports("avx512f") != 0) {
        return Isa::avx512;
    }
    if (__builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0) {
        return Isa::avx2;
    }
    if (__builtin_cpu_supports("avx") != 0) {
        return Isa::avx;
    }
    return Isa::scalar;
}

// Function-pointer dispatch tables, resolved once at static-init time.
struct Dispatch {
    Isa isa;
    void (*add)(const float*, const float*, float*, std::size_t);
    void (*mul)(const float*, const float*, float*, std::size_t);
    void (*axpy)(float, const float*, const float*, float*, std::size_t);
};

[[nodiscard]] auto make_dispatch() noexcept -> Dispatch {
    switch (detect_isa()) {
        case Isa::avx512:
            return {Isa::avx512, &add_avx512, &mul_avx512, &axpy_avx512};
        case Isa::avx2:
            return {Isa::avx2, &add_avx256, &mul_avx256, &axpy_avx2};
        case Isa::avx:
            return {Isa::avx, &add_avx256, &mul_avx256, &axpy_avx};
        case Isa::scalar:
            break;
    }
    return {Isa::scalar, &add_scalar, &mul_scalar, &axpy_scalar};
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

}  // namespace nimblecas::simd
