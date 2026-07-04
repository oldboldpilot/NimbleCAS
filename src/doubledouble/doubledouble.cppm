// NimbleCAS double-double extended-precision floating point (~106-bit; SIMD-batched).
// @author Olumuyiwa Oluwasanmi
//
// A DoubleDouble represents a real number as an unevaluated sum hi + lo of two IEEE-754
// binary64 doubles held under the invariant |lo| <= 0.5 * ulp(hi). Because hi already
// carries 53 significand bits and lo carries the next ~53, the pair models roughly a
// 106-bit significand — about 31-32 decimal digits — at a small multiple of double cost.
//
// HONESTY (read this before reaching for it):
//   * This is ~106-bit, it is NOT IEEE binary128 / "quad". There is no native 128-bit
//     hardware float on x86, so double-double + error-free transforms (EFTs) is the fast,
//     honest route. The exponent range is still binary64's (~1e+/-308), only the
//     significand is extended.
//   * It is still FLOATING POINT, hence still inexact: rounding happens, results are
//     faithful approximations, not exact. When you need EXACT arithmetic (no rounding at
//     all) use the rational / big-integer path instead — nimblecas::Rational (module
//     nimblecas.ratpoly) is closed under +,-,*,/ with zero rounding, and the exact
//     complex/rational layers build on it. Double-double is for "more precision, still
//     fast", not for "provably exact".
//   * two_prod needs a fused multiply-add to recover the product error in one rounding.
//     std::fma is correctly rounded everywhere (in software if the CPU lacks FMA), so
//     the code is correct on any target; when hardware FMA is absent we instead take
//     Dekker's splitting TwoProduct so we never pay for a slow software std::fma. Both
//     recover the SAME exact error, so results are bit-identical either way.
//   * SIMD (AVX/AVX2/AVX-512) accelerates BATCHES: it runs the EFTs of many array
//     elements in parallel. It does NOT make a single scalar op "128-bit" — one add is
//     one add. The batched kernels below (dd_sum / dd_dot / dd_poly_eval) are where the
//     vectorisation pays off.
//
// The batched kernels use a fixed 4-lane (256-bit) reduction layout so their result is
// reproducible across machines and bit-identical whether the scalar or the SIMD path
// runs (see doubledouble_tests.cpp). Runtime CPU-feature detection is reused from
// nimblecas.simd (Code Policy Rules 29/50/55); AVX-512 hosts simply run the 256-bit
// kernel to keep that 4-lane layout, trading a little width for exact reproducibility.
//
// References for the EFTs used here:
//   Knuth,  TAOCP vol. 2, Thm. B (§4.2.2)            — TwoSum.
//   Dekker, Numer. Math. 18 (1971), 224-242          — quick-TwoSum and split TwoProduct.
//   Shewchuk, Discrete & Comp. Geom. 18 (1997)       — adaptive EFT expansions.
//   Hida/Li/Bailey, "Library for Double-Double and Quad-Double Arithmetic" (2000)
//                                                    — the dd add/mul/div/sqrt algorithms.

module;
#include <immintrin.h>

export module nimblecas.doubledouble;

import std;
import nimblecas.core;
import nimblecas.simd;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// DoubleDouble — a real number as the unevaluated sum hi + lo (~106-bit).
// ---------------------------------------------------------------------------
// Kept an aggregate (no user-declared constructors, public members) so it can be
// brace-initialised as {hi, lo} and lives happily in std::array / SIMD spill buffers.
// The invariant |lo| <= 0.5*ulp(hi) is maintained by every operation below, which makes
// (hi, lo) a canonical, order-preserving pair: numeric ordering equals lexicographic
// ordering on (hi, lo), so the defaulted <=> / == are the correct numeric comparisons.
struct DoubleDouble {
    double hi{};
    double lo{};

    // The double d as the exact double-double d + 0.
    [[nodiscard]] static constexpr auto from_double(double d) noexcept -> DoubleDouble {
        return {d, 0.0};
    }

    // Nearest double (the leading component). Loses the extra ~53 bits carried in lo.
    [[nodiscard]] constexpr auto to_double() const noexcept -> double { return hi; }

    [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
        return hi == 0.0 && lo == 0.0;
    }
    // The additive inverse -(hi + lo); never rounds, so it cannot fail.
    [[nodiscard]] constexpr auto negate() const noexcept -> DoubleDouble { return {-hi, -lo}; }

    // Field operations. Add/subtract/multiply cannot fail (no domain, no zero divisor),
    // so they are plain values; divide/sqrt are railway-typed (Rule 32).
    [[nodiscard]] auto add(const DoubleDouble& o) const noexcept -> DoubleDouble;
    [[nodiscard]] auto subtract(const DoubleDouble& o) const noexcept -> DoubleDouble;
    [[nodiscard]] auto multiply(const DoubleDouble& o) const noexcept -> DoubleDouble;
    // Fails with division_by_zero when o is exactly zero.
    [[nodiscard]] auto divide(const DoubleDouble& o) const -> Result<DoubleDouble>;
    // Fails with domain_error for a negative operand (zero maps to zero).
    [[nodiscard]] auto sqrt() const -> Result<DoubleDouble>;

    // Decimal string with enough significant digits to expose the extra precision, e.g.
    // to_string() of the double-double 1/3 shows ~31 threes, not double's ~16.
    [[nodiscard]] auto to_string(int precision = 31) const -> std::string;

    // Structural == and lexicographic <=> coincide with numeric equality/ordering because
    // the (hi, lo) representation is canonical under the invariant. NaN parts make <=>
    // yield std::partial_ordering::unordered, matching IEEE semantics.
    [[nodiscard]] auto operator==(const DoubleDouble& o) const noexcept -> bool {
        return hi == o.hi && lo == o.lo;
    }
    [[nodiscard]] auto operator<=>(const DoubleDouble& o) const noexcept
        -> std::partial_ordering {
        if (const auto c = hi <=> o.hi; c != 0) {
            return c;
        }
        return lo <=> o.lo;
    }
};

// ---------------------------------------------------------------------------
// Error-free transforms (EFTs) — the primitives everything else is built from.
// ---------------------------------------------------------------------------
// Each returns {s, e} where s is the rounded result and e is the EXACT rounding error,
// so the pair reconstructs the true value with no loss: a op b == s + e exactly (for
// representable, non-overflowing inputs). The returned pair already satisfies the
// DoubleDouble invariant |e| <= 0.5*ulp(s).

// Knuth's TwoSum (TAOCP vol. 2, Thm. B): a + b == s + e exactly, for any finite a, b.
[[nodiscard]] auto two_sum(double a, double b) noexcept -> DoubleDouble;

// Dekker's quick-TwoSum (1971): a + b == s + e exactly, but ONLY valid when
// |a| >= |b| (or a == 0). Cheaper than two_sum (3 flops vs 6); used internally after a
// renormalising add where the magnitude order is already known.
[[nodiscard]] auto quick_two_sum(double a, double b) noexcept -> DoubleDouble;

// TwoProduct: a * b == p + e exactly. Uses a single fused multiply-add to capture the
// product error (std::fma when hardware FMA is present) and falls back to Dekker's
// splitting TwoProduct when it is not — both yield the identical exact e.
[[nodiscard]] auto two_prod(double a, double b) noexcept -> DoubleDouble;

// ---------------------------------------------------------------------------
// SIMD-batched reductions — the vectorisation payoff.
// ---------------------------------------------------------------------------
// Each has a scalar reference (the *_scalar overload) AND a runtime-dispatched path that
// vectorises the EFTs across a fixed 4-lane layout. The two paths are bit-for-bit
// identical (asserted in the tests), and the dispatched path is chosen by reusing
// nimblecas.simd's CPU-feature detection, falling back to the scalar reference when the
// baseline (AVX2+FMA) is unavailable — so every result is correct and reproducible
// everywhere.

// Compensated summation of x to a double-double (recovers the bits a naive double sum
// drops; e.g. {1e16, 1, -1e16} sums to 1, not 0).
[[nodiscard]] auto dd_sum(std::span<const double> x) noexcept -> DoubleDouble;
[[nodiscard]] auto dd_sum_scalar(std::span<const double> x) noexcept -> DoubleDouble;

// Compensated dot product sum_i a[i]*b[i] to a double-double (each product is exact via
// two_prod). Uses min(a.size(), b.size()) elements.
[[nodiscard]] auto dd_dot(std::span<const double> a, std::span<const double> b) noexcept
    -> DoubleDouble;
[[nodiscard]] auto dd_dot_scalar(std::span<const double> a, std::span<const double> b) noexcept
    -> DoubleDouble;

// Compensated polynomial evaluation P(x) = sum_i coeffs[i] * x^i (coeffs[i] is the
// coefficient of x^i). Uses a 4-way interleaved Horner scheme: four strided sub-Horners
// run in parallel over y = x^4, then are recombined — accurate and vectorisable, unlike a
// single sequential Horner. (To evaluate one polynomial at MANY points at once, see
// nimblecas.simd::horner_step.)
[[nodiscard]] auto dd_poly_eval(std::span<const double> coeffs, double x) noexcept
    -> DoubleDouble;
[[nodiscard]] auto dd_poly_eval_scalar(std::span<const double> coeffs, double x) noexcept
    -> DoubleDouble;

// Which path the dispatched batched kernels are using on this host ("scalar" or a
// description of the SIMD backend), for diagnostics/tests.
[[nodiscard]] auto batched_backend() noexcept -> std::string_view;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// True when the CPU has hardware FMA; selects the two_prod strategy (see two_prod).
const bool g_has_fma = (__builtin_cpu_supports("fma") != 0);

// The batched kernels vectorise only on the AVX2+FMA baseline (or better); AVX-512 hosts
// still run the 256-bit kernel to preserve the reproducible 4-lane layout. Plain-AVX and
// scalar hosts take the scalar reference. Detection is reused from nimblecas.simd.
[[nodiscard]] auto use_simd() noexcept -> bool {
    const auto isa = simd::active_isa();
    return isa == simd::Isa::avx2 || isa == simd::Isa::avx512;
}

// --- scalar error-free transforms ------------------------------------------

// Dekker's splitting TwoProduct (1971): the FMA-free product error. Veltkamp-splits each
// operand into two 26/27-bit halves whose sub-products are all exactly representable, so
// the assembled e equals a*b - fl(a*b) exactly. (Caveat: SPLIT*a overflows for |a| very
// near DBL_MAX; the FMA path is used on essentially all modern x86 and has no such split.)
[[nodiscard]] auto two_prod_dekker(double a, double b) noexcept -> DoubleDouble {
    constexpr double kSplit = 134217729.0;  // 2^27 + 1
    const double ca = kSplit * a;
    const double a_hi = ca - (ca - a);
    const double a_lo = a - a_hi;
    const double cb = kSplit * b;
    const double b_hi = cb - (cb - b);
    const double b_lo = b - b_hi;
    const double p = a * b;
    const double e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    return {p, e};
}

// --- double-double core arithmetic (Hida/Li/Bailey algorithms) -------------
// These helpers are written so each scalar op has a lane-for-lane SIMD twin below with
// the IDENTICAL sequence of IEEE roundings — that is what makes the scalar and SIMD
// batched paths bit-for-bit equal. In particular every product is materialised into a
// named temporary before any following add, so an aggressive -ffp-contract cannot fuse a
// mul+add into an FMA in one path but not the other.

// dd + double.
[[nodiscard]] auto add_dd_d(DoubleDouble a, double b) noexcept -> DoubleDouble {
    const DoubleDouble s = two_sum(a.hi, b);
    const double s2 = s.lo + a.lo;
    return quick_two_sum(s.hi, s2);
}

// dd + dd (the accurate "ieee_add").
[[nodiscard]] auto add_dd_dd(DoubleDouble a, DoubleDouble b) noexcept -> DoubleDouble {
    const DoubleDouble s = two_sum(a.hi, b.hi);
    const DoubleDouble t = two_sum(a.lo, b.lo);
    const double s2 = s.lo + t.hi;
    const DoubleDouble s1 = quick_two_sum(s.hi, s2);
    const double s2b = s1.lo + t.lo;
    return quick_two_sum(s1.hi, s2b);
}

// dd - dd.
[[nodiscard]] auto sub_dd_dd(DoubleDouble a, DoubleDouble b) noexcept -> DoubleDouble {
    return add_dd_dd(a, b.negate());
}

// dd - double (used by the decimal-digit extraction in to_string).
[[nodiscard]] auto sub_dd_d(DoubleDouble a, double b) noexcept -> DoubleDouble {
    return add_dd_d(a, -b);
}

// dd * dd. The a.lo*b.lo cross term is below the ~106-bit resolution and dropped.
[[nodiscard]] auto mul_dd_dd(DoubleDouble a, DoubleDouble b) noexcept -> DoubleDouble {
    const DoubleDouble p = two_prod(a.hi, b.hi);
    const double t1 = a.hi * b.lo;
    const double t2 = a.lo * b.hi;
    const double cross = t1 + t2;
    const double p2 = p.lo + cross;
    return quick_two_sum(p.hi, p2);
}

// dd * double.
[[nodiscard]] auto mul_dd_d(DoubleDouble a, double b) noexcept -> DoubleDouble {
    const DoubleDouble p = two_prod(a.hi, b);
    const double t = a.lo * b;
    const double p2 = p.lo + t;
    return quick_two_sum(p.hi, p2);
}

// dd / dd, assuming b != 0 (the public divide() guards the zero case). Three long-division
// correction steps give a faithful ~106-bit quotient (Hida/Li/Bailey).
[[nodiscard]] auto div_dd_dd(DoubleDouble a, DoubleDouble b) noexcept -> DoubleDouble {
    const double q1 = a.hi / b.hi;
    DoubleDouble r = sub_dd_dd(a, mul_dd_d(b, q1));
    const double q2 = r.hi / b.hi;
    r = sub_dd_dd(r, mul_dd_d(b, q2));
    const double q3 = r.hi / b.hi;
    const DoubleDouble q = quick_two_sum(q1, q2);
    return add_dd_d(q, q3);
}

// Combine the four reduction lanes in a fixed order (shared by the scalar and SIMD paths
// so the horizontal step is also bit-identical).
[[nodiscard]] auto combine4(const DoubleDouble lane[4]) noexcept -> DoubleDouble {
    const DoubleDouble t01 = add_dd_dd(lane[0], lane[1]);
    const DoubleDouble t23 = add_dd_dd(lane[2], lane[3]);
    return add_dd_dd(t01, t23);
}

// --- vector (256-bit, 4-lane) error-free transforms ------------------------
// One __m256d holds four lanes of hi (or lo). Each helper mirrors its scalar twin above
// op-for-op, so lane j of the vector result equals the scalar result on lane j's data
// bit-for-bit. Compiled at the AVX2+FMA baseline (like nimblecas.simd's AVX2 kernels), so
// no per-function target attribute is needed; two_prod_v relies on _mm256_fmsub_pd.
struct V2 {
    __m256d hi;
    __m256d lo;
};

[[nodiscard]] auto two_sum_v(__m256d a, __m256d b) noexcept -> V2 {
    const __m256d s = _mm256_add_pd(a, b);
    const __m256d bb = _mm256_sub_pd(s, a);
    const __m256d e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb)),
                                    _mm256_sub_pd(b, bb));
    return {s, e};
}

[[nodiscard]] auto quick_two_sum_v(__m256d a, __m256d b) noexcept -> V2 {
    const __m256d s = _mm256_add_pd(a, b);
    const __m256d e = _mm256_sub_pd(b, _mm256_sub_pd(s, a));
    return {s, e};
}

[[nodiscard]] auto two_prod_v(__m256d a, __m256d b) noexcept -> V2 {
    const __m256d p = _mm256_mul_pd(a, b);
    const __m256d e = _mm256_fmsub_pd(a, b, p);  // a*b - p, fused == std::fma(a,b,-p)
    return {p, e};
}

[[nodiscard]] auto add_dd_d_v(V2 a, __m256d b) noexcept -> V2 {
    const V2 s = two_sum_v(a.hi, b);
    const __m256d s2 = _mm256_add_pd(s.lo, a.lo);
    return quick_two_sum_v(s.hi, s2);
}

[[nodiscard]] auto add_dd_dd_v(V2 a, V2 b) noexcept -> V2 {
    const V2 s = two_sum_v(a.hi, b.hi);
    const V2 t = two_sum_v(a.lo, b.lo);
    const __m256d s2 = _mm256_add_pd(s.lo, t.hi);
    const V2 s1 = quick_two_sum_v(s.hi, s2);
    const __m256d s2b = _mm256_add_pd(s1.lo, t.lo);
    return quick_two_sum_v(s1.hi, s2b);
}

[[nodiscard]] auto mul_dd_dd_v(V2 a, V2 b) noexcept -> V2 {
    const V2 p = two_prod_v(a.hi, b.hi);
    const __m256d cross =
        _mm256_add_pd(_mm256_mul_pd(a.hi, b.lo), _mm256_mul_pd(a.lo, b.hi));
    const __m256d p2 = _mm256_add_pd(p.lo, cross);
    return quick_two_sum_v(p.hi, p2);
}

// --- batched kernels: scalar reference + SIMD twin -------------------------
// Canonical 4-lane layout (LANES == 4): element i accrues into lane (i % 4) across blocks
// of 4, lanes are combined in the fixed combine4 order, then the ragged tail is folded in
// sequentially. Both paths implement exactly this, so they agree bit-for-bit and the
// result is independent of which host runs it.
constexpr std::size_t kLanes = 4;

[[nodiscard]] auto dd_sum_scalar_impl(const double* x, std::size_t n) noexcept -> DoubleDouble {
    DoubleDouble lane[kLanes] = {};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (std::size_t r = 0; r < kLanes; ++r) {
            lane[r] = add_dd_d(lane[r], x[i + r]);
        }
    }
    DoubleDouble acc = combine4(lane);
    for (; i < n; ++i) {
        acc = add_dd_d(acc, x[i]);
    }
    return acc;
}

[[nodiscard]] auto dd_sum_simd_impl(const double* x, std::size_t n) noexcept -> DoubleDouble {
    V2 acc{_mm256_setzero_pd(), _mm256_setzero_pd()};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        acc = add_dd_d_v(acc, _mm256_loadu_pd(x + i));
    }
    double hi[kLanes];
    double lo[kLanes];
    _mm256_storeu_pd(hi, acc.hi);
    _mm256_storeu_pd(lo, acc.lo);
    const DoubleDouble lane[kLanes] = {{hi[0], lo[0]}, {hi[1], lo[1]},
                                       {hi[2], lo[2]}, {hi[3], lo[3]}};
    DoubleDouble acc_s = combine4(lane);
    for (; i < n; ++i) {
        acc_s = add_dd_d(acc_s, x[i]);
    }
    return acc_s;
}

[[nodiscard]] auto dd_dot_scalar_impl(const double* a, const double* b, std::size_t n) noexcept
    -> DoubleDouble {
    DoubleDouble lane[kLanes] = {};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (std::size_t r = 0; r < kLanes; ++r) {
            const DoubleDouble p = two_prod(a[i + r], b[i + r]);
            lane[r] = add_dd_dd(lane[r], p);
        }
    }
    DoubleDouble acc = combine4(lane);
    for (; i < n; ++i) {
        acc = add_dd_dd(acc, two_prod(a[i], b[i]));
    }
    return acc;
}

[[nodiscard]] auto dd_dot_simd_impl(const double* a, const double* b, std::size_t n) noexcept
    -> DoubleDouble {
    V2 acc{_mm256_setzero_pd(), _mm256_setzero_pd()};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const V2 p = two_prod_v(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
        acc = add_dd_dd_v(acc, p);
    }
    double hi[kLanes];
    double lo[kLanes];
    _mm256_storeu_pd(hi, acc.hi);
    _mm256_storeu_pd(lo, acc.lo);
    const DoubleDouble lane[kLanes] = {{hi[0], lo[0]}, {hi[1], lo[1]},
                                       {hi[2], lo[2]}, {hi[3], lo[3]}};
    DoubleDouble acc_s = combine4(lane);
    for (; i < n; ++i) {
        acc_s = add_dd_dd(acc_s, two_prod(a[i], b[i]));
    }
    return acc_s;
}

// Horizontal recombine of the four strided sub-Horners: P = ((L3*x + L2)*x + L1)*x + L0.
[[nodiscard]] auto poly_combine(const DoubleDouble lane[4], double x) noexcept -> DoubleDouble {
    DoubleDouble r = lane[3];
    r = add_dd_dd(mul_dd_d(r, x), lane[2]);
    r = add_dd_dd(mul_dd_d(r, x), lane[1]);
    r = add_dd_dd(mul_dd_d(r, x), lane[0]);
    return r;
}

[[nodiscard]] auto dd_poly_scalar_impl(const double* c, std::size_t n, double x) noexcept
    -> DoubleDouble {
    if (n == 0) {
        return {0.0, 0.0};
    }
    const DoubleDouble x2 = two_prod(x, x);      // x^2 (exact)
    const DoubleDouble y = mul_dd_dd(x2, x2);    // y = x^4 = x^kLanes
    DoubleDouble lane[kLanes] = {};
    const std::ptrdiff_t blocks = static_cast<std::ptrdiff_t>((n + kLanes - 1) / kLanes);
    for (std::ptrdiff_t k = blocks - 1; k >= 0; --k) {
        for (std::size_t r = 0; r < kLanes; ++r) {
            const std::size_t idx = static_cast<std::size_t>(k) * kLanes + r;
            const double cc = (idx < n) ? c[idx] : 0.0;  // zero-pad the top block
            lane[r] = add_dd_d(mul_dd_dd(lane[r], y), cc);
        }
    }
    return poly_combine(lane, x);
}

[[nodiscard]] auto dd_poly_simd_impl(const double* c, std::size_t n, double x) noexcept
    -> DoubleDouble {
    if (n == 0) {
        return {0.0, 0.0};
    }
    const DoubleDouble x2 = two_prod(x, x);
    const DoubleDouble y = mul_dd_dd(x2, x2);
    const V2 yv{_mm256_set1_pd(y.hi), _mm256_set1_pd(y.lo)};
    V2 acc{_mm256_setzero_pd(), _mm256_setzero_pd()};

    const std::ptrdiff_t blocks = static_cast<std::ptrdiff_t>((n + kLanes - 1) / kLanes);
    std::ptrdiff_t k = blocks - 1;
    const std::size_t rem = n % kLanes;
    if (rem != 0) {
        // Top block is partial: build it with explicit zero-padding rather than reading
        // past the array, then run the same fused step as a full block.
        double blk[kLanes];
        for (std::size_t r = 0; r < kLanes; ++r) {
            const std::size_t idx = static_cast<std::size_t>(k) * kLanes + r;
            blk[r] = (idx < n) ? c[idx] : 0.0;
        }
        acc = add_dd_d_v(mul_dd_dd_v(acc, yv), _mm256_loadu_pd(blk));
        --k;
    }
    for (; k >= 0; --k) {
        const __m256d bvec = _mm256_loadu_pd(c + static_cast<std::size_t>(k) * kLanes);
        acc = add_dd_d_v(mul_dd_dd_v(acc, yv), bvec);
    }

    double hi[kLanes];
    double lo[kLanes];
    _mm256_storeu_pd(hi, acc.hi);
    _mm256_storeu_pd(lo, acc.lo);
    const DoubleDouble lane[kLanes] = {{hi[0], lo[0]}, {hi[1], lo[1]},
                                       {hi[2], lo[2]}, {hi[3], lo[3]}};
    return poly_combine(lane, x);
}

// --- decimal string helpers ------------------------------------------------

// 10^n as a double-double (binary exponentiation; negative powers via one exact-ish
// reciprocal). Used to scale a value into [1, 10) for digit extraction in to_string.
[[nodiscard]] auto pow10_dd(int n) noexcept -> DoubleDouble {
    const bool neg = n < 0;
    unsigned m = neg ? static_cast<unsigned>(-static_cast<long>(n)) : static_cast<unsigned>(n);
    DoubleDouble result = DoubleDouble::from_double(1.0);
    DoubleDouble base = DoubleDouble::from_double(10.0);
    while (m > 0) {
        if ((m & 1u) != 0u) {
            result = mul_dd_dd(result, base);
        }
        m >>= 1;
        if (m > 0) {
            base = mul_dd_dd(base, base);
        }
    }
    if (neg) {
        result = div_dd_dd(DoubleDouble::from_double(1.0), result);  // result != 0
    }
    return result;
}

}  // namespace

// --- exported error-free transforms ----------------------------------------

auto two_sum(double a, double b) noexcept -> DoubleDouble {
    const double s = a + b;
    const double bb = s - a;
    const double e = (a - (s - bb)) + (b - bb);
    return {s, e};
}

auto quick_two_sum(double a, double b) noexcept -> DoubleDouble {
    const double s = a + b;
    const double e = b - (s - a);
    return {s, e};
}

auto two_prod(double a, double b) noexcept -> DoubleDouble {
    const double p = a * b;
    if (g_has_fma) {
        return {p, std::fma(a, b, -p)};  // correctly-rounded fused product error
    }
    return two_prod_dekker(a, b);  // identical exact error, no software std::fma
}

// --- DoubleDouble field operations -----------------------------------------

auto DoubleDouble::add(const DoubleDouble& o) const noexcept -> DoubleDouble {
    return add_dd_dd(*this, o);
}

auto DoubleDouble::subtract(const DoubleDouble& o) const noexcept -> DoubleDouble {
    return sub_dd_dd(*this, o);
}

auto DoubleDouble::multiply(const DoubleDouble& o) const noexcept -> DoubleDouble {
    return mul_dd_dd(*this, o);
}

auto DoubleDouble::divide(const DoubleDouble& o) const -> Result<DoubleDouble> {
    if (o.is_zero()) {
        return make_error<DoubleDouble>(MathError::division_by_zero);
    }
    return div_dd_dd(*this, o);
}

auto DoubleDouble::sqrt() const -> Result<DoubleDouble> {
    // Negative (sign fixed by hi, or by lo when hi is +0): the real square root is
    // undefined — that belongs to the complex layer, not here.
    if (hi < 0.0 || (hi == 0.0 && lo < 0.0)) {
        return make_error<DoubleDouble>(MathError::domain_error);
    }
    if (hi == 0.0) {
        return DoubleDouble{0.0, 0.0};  // lo == 0 here by the invariant
    }
    // One Newton step on 1/sqrt in double-double (Hida/Li/Bailey): with x ~ 1/sqrt(a) and
    // ax ~ sqrt(a), the correction (a - ax^2) * (x/2) lifts the result to ~106 bits.
    const double x = 1.0 / std::sqrt(hi);
    const double ax = hi * x;
    const DoubleDouble axsq = two_prod(ax, ax);
    const DoubleDouble diff = sub_dd_dd(*this, axsq);
    const double corr = diff.hi * (x * 0.5);
    return two_sum(ax, corr);
}

auto DoubleDouble::to_string(int precision) const -> std::string {
    if (!std::isfinite(hi)) {
        if (std::isnan(hi)) {
            return "nan";
        }
        return hi < 0.0 ? "-inf" : "inf";
    }
    if (is_zero()) {
        return "0";
    }
    if (precision < 1) {
        precision = 1;
    }
    const bool neg = (hi < 0.0) || (hi == 0.0 && lo < 0.0);
    DoubleDouble a = neg ? negate() : *this;  // work with |value|

    // Bring |value| into [1, 10): mantissa = a * 10^(-e), e = floor(log10 a).
    int e = static_cast<int>(std::floor(std::log10(a.hi)));
    DoubleDouble m = mul_dd_dd(a, pow10_dd(-e));
    if (m.hi >= 10.0) {
        e += 1;
        m = mul_dd_dd(a, pow10_dd(-e));
    } else if (m.hi < 1.0) {
        e -= 1;
        m = mul_dd_dd(a, pow10_dd(-e));
    }

    // Extract precision+1 digits (the extra one drives round-to-nearest of the last kept).
    std::vector<int> d(static_cast<std::size_t>(precision) + 1, 0);
    DoubleDouble r = m;
    for (std::size_t i = 0; i <= static_cast<std::size_t>(precision); ++i) {
        int dig = static_cast<int>(r.hi);
        dig = std::clamp(dig, 0, 9);
        d[i] = dig;
        r = mul_dd_d(sub_dd_d(r, static_cast<double>(dig)), 10.0);
    }
    if (d[static_cast<std::size_t>(precision)] >= 5) {  // round up, propagating carry
        std::ptrdiff_t i = precision - 1;
        for (; i >= 0; --i) {
            if (++d[static_cast<std::size_t>(i)] < 10) {
                break;
            }
            d[static_cast<std::size_t>(i)] = 0;
        }
        if (i < 0) {  // carry rippled past the leading digit: 9.99.. -> 10.0.., shift
            for (std::ptrdiff_t j = precision - 1; j > 0; --j) {
                d[static_cast<std::size_t>(j)] = d[static_cast<std::size_t>(j) - 1];
            }
            d[0] = 1;
            e += 1;
        }
    }

    std::string out;
    if (neg) {
        out.push_back('-');
    }
    out.push_back(static_cast<char>('0' + d[0]));
    out.push_back('.');
    for (int i = 1; i < precision; ++i) {
        out.push_back(static_cast<char>('0' + d[static_cast<std::size_t>(i)]));
    }
    out.push_back('e');
    out.push_back(e >= 0 ? '+' : '-');
    out += std::format("{:02}", e < 0 ? -e : e);
    return out;
}

// --- exported batched kernels ----------------------------------------------

auto dd_sum_scalar(std::span<const double> x) noexcept -> DoubleDouble {
    return dd_sum_scalar_impl(x.data(), x.size());
}

auto dd_sum(std::span<const double> x) noexcept -> DoubleDouble {
    return use_simd() ? dd_sum_simd_impl(x.data(), x.size())
                      : dd_sum_scalar_impl(x.data(), x.size());
}

auto dd_dot_scalar(std::span<const double> a, std::span<const double> b) noexcept
    -> DoubleDouble {
    return dd_dot_scalar_impl(a.data(), b.data(), std::min(a.size(), b.size()));
}

auto dd_dot(std::span<const double> a, std::span<const double> b) noexcept -> DoubleDouble {
    const std::size_t n = std::min(a.size(), b.size());
    return use_simd() ? dd_dot_simd_impl(a.data(), b.data(), n)
                      : dd_dot_scalar_impl(a.data(), b.data(), n);
}

auto dd_poly_eval_scalar(std::span<const double> coeffs, double x) noexcept -> DoubleDouble {
    return dd_poly_scalar_impl(coeffs.data(), coeffs.size(), x);
}

auto dd_poly_eval(std::span<const double> coeffs, double x) noexcept -> DoubleDouble {
    return use_simd() ? dd_poly_simd_impl(coeffs.data(), coeffs.size(), x)
                      : dd_poly_scalar_impl(coeffs.data(), coeffs.size(), x);
}

auto batched_backend() noexcept -> std::string_view {
    return use_simd() ? "simd(256-bit AVX2+FMA, 4-lane)" : "scalar";
}

}  // namespace nimblecas
