// NimbleCAS GPU acceleration module (ROADMAP 5): a Result-based C++23 wrapper over the
// CUDA kernels in gpu_kernels.cu.
// @author Olumuyiwa Oluwasanmi
//
// Optional — built only when configured with -DNIMBLECAS_CUDA=ON and an available nvcc
// (see CMakeLists.txt). The kernels are reached through the plain C ABI of gpu_bridge.h,
// so this module holds no CUDA types; it only marshals std::span/std::vector across the
// boundary and maps CUDA failures onto MathError (Rule 32 — no exceptions).

module;
#include "gpu_bridge.h"

export module nimblecas.gpu;

import std;
import nimblecas.core;
import nimblecas.pricing;   // Family A CPU fallback + result types (McResult, Greeks, ExtendedGreeks)
import nimblecas.optstrat;  // Family B CPU fallback + StrategyLeg (the exact piecewise-linear oracle)
import nimblecas.futures;   // Family B CPU fallback + FuturesLeg
import nimblecas.control;   // Family F CPU fallback + TransferFunction / BodePoint / NyquistPoint
import nimblecas.wavelets;  // Wavelet FilterBank + CPU dwt/swt fallback
import nimblecas.qmc;       // Family H CPU fallback & point generators
import nimblecas.krylov;    // CPU fallbacks (csr_matvec + cg / bicgstab / gmres)
import nimblecas.nlsolve;   // FAMILY I CPU fallback + the authoritative LM oracle

export namespace nimblecas::gpu {

// Number of CUDA-capable devices detected (0 when no GPU / CUDA runtime is present).
[[nodiscard]] auto device_count() -> int { return nimblecas_gpu_device_count(); }

// Whether at least one GPU is available for computation.
[[nodiscard]] auto available() -> bool { return device_count() > 0; }

// Maximum FFT length the GPU kernel supports (must match kMaxFftLen in gpu_kernels.cu): each
// signal's 2*n interleaved doubles must fit one block's dynamic shared-memory tile.
inline constexpr int kGpuFftMaxLen = 2048;

// Whether v is a (positive) power of two — the radix-2 FFT precondition. v <= 0 is not.
[[nodiscard]] auto is_power_of_two(int v) -> bool { return v > 0 && (v & (v - 1)) == 0; }

// Evaluate the polynomial `coeffs` (low degree first) at every point in `x` on the GPU,
// returning the vector of p(x_i). Fails with MathError::gpu_error when no device is present
// or a CUDA call fails, and MathError::overflow when a size exceeds the int kernel bound.
[[nodiscard]] auto poly_eval(std::span<const double> coeffs, std::span<const double> x)
    -> Result<std::vector<double>> {
    if (!available()) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (coeffs.size() > int_max || x.size() > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    std::vector<double> out(x.size());
    if (x.empty()) {
        return out;  // nothing to evaluate
    }
    const int rc = nimblecas_gpu_poly_eval(coeffs.data(), static_cast<int>(coeffs.size()),
                                           x.data(), out.data(), static_cast<int>(x.size()));
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// Batched Levenshtein edit distance. The sequences are supplied as flattened code-point arrays
// with prefix-offset arrays of length pairs+1: pair i is a_flat[a_off[i]..a_off[i+1]) against
// b_flat[b_off[i]..b_off[i+1]). Returns one distance per pair. The kernel rolls its DP over
// the SHORTER sequence of each pair (Levenshtein is symmetric), so the SHORTER side of every
// pair must not exceed 256 code points; a pair violating that is rejected with
// MathError::overflow (never silently truncated — the longer side is unbounded). Fails with
// MathError::gpu_error when no device is present, the offset arrays are malformed (empty,
// unequal length, non-monotone, or out of the flat-buffer bounds), or a CUDA call fails, and
// MathError::overflow when a span exceeds the int kernel bound or the 256 short-side limit.
[[nodiscard]] auto edit_distance_batch(std::span<const int> a_flat, std::span<const int> a_off,
                                       std::span<const int> b_flat, std::span<const int> b_off)
    -> Result<std::vector<int>> {
    if (!available()) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    // Offsets must be non-empty and of equal length: they define the same pair count for a and b.
    if (a_off.empty() || a_off.size() != b_off.size()) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (a_flat.size() > int_max || b_flat.size() > int_max || a_off.size() > int_max ||
        b_off.size() > int_max) {
        return make_error<std::vector<int>>(MathError::overflow);
    }
    const std::size_t pairs = a_off.size() - 1;
    std::vector<int> out(pairs);
    if (pairs == 0) {
        return out;  // no pairs to compare
    }
    // The kernel holds its rolling DP rows in bounded per-thread local memory, sized for the
    // SHORTER sequence of each pair (Levenshtein is symmetric). Reject — never silently
    // truncate — any pair whose shorter side exceeds this width, and validate that the offset
    // arrays are non-decreasing and stay within the flattened buffers.
    constexpr int kEditMaxShortLen = 256;  // must match kMaxEditLen in gpu_kernels.cu
    if (a_off.front() < 0 || b_off.front() < 0 ||
        a_off.back() > static_cast<int>(a_flat.size()) ||
        b_off.back() > static_cast<int>(b_flat.size())) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    for (std::size_t i = 0; i < pairs; ++i) {
        const int a_len = a_off[i + 1] - a_off[i];
        const int b_len = b_off[i + 1] - b_off[i];
        if (a_len < 0 || b_len < 0) {  // non-monotone offsets
            return make_error<std::vector<int>>(MathError::gpu_error);
        }
        if (std::min(a_len, b_len) > kEditMaxShortLen) {
            return make_error<std::vector<int>>(MathError::overflow);
        }
    }
    const int rc = nimblecas_gpu_edit_distance_batch(a_flat.data(), a_off.data(), b_flat.data(),
                                                     b_off.data(), static_cast<int>(pairs),
                                                     out.data());
    if (rc != 0) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    return out;
}

// Level-synchronous single-source BFS over a CSR graph. `row_offsets` has length
// num_vertices+1 and `col_indices` holds the flattened adjacency. Returns the hop distance from
// `source` to every vertex (-1 for unreachable). Fails with MathError::gpu_error when no device
// is present, the CSR is malformed, or a CUDA call fails, and MathError::overflow when a span
// exceeds the int kernel bound.
[[nodiscard]] auto bfs(std::span<const int> row_offsets, std::span<const int> col_indices,
                       int source) -> Result<std::vector<int>> {
    if (!available()) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    if (row_offsets.empty()) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (row_offsets.size() > int_max || col_indices.size() > int_max) {
        return make_error<std::vector<int>>(MathError::overflow);
    }
    const std::size_t num_vertices = row_offsets.size() - 1;
    std::vector<int> dist(num_vertices);
    if (num_vertices == 0) {
        return dist;  // empty graph
    }
    const int rc = nimblecas_gpu_bfs(row_offsets.data(), col_indices.data(),
                                     static_cast<int>(num_vertices),
                                     static_cast<int>(col_indices.size()), source, dist.data());
    if (rc != 0) {
        return make_error<std::vector<int>>(MathError::gpu_error);
    }
    return dist;
}

// Count the solutions to the n-queens problem on the GPU (n = 8 -> 92, n = 10 -> 724). Fails
// with MathError::gpu_error when no device is present, n exceeds the 31-wide bitmask bound, or a
// CUDA call fails.
[[nodiscard]] auto nqueens_count(int n) -> Result<std::uint64_t> {
    if (!available()) {
        return make_error<std::uint64_t>(MathError::gpu_error);
    }
    unsigned long long count = 0ull;
    const int rc = nimblecas_gpu_nqueens_count(n, &count);
    if (rc != 0) {
        return make_error<std::uint64_t>(MathError::gpu_error);
    }
    return static_cast<std::uint64_t>(count);
}

// QMC integration reduction on the device: the equal-weight average of the polynomial integrand
// `coeffs` (low degree first) over the supplied sample `points`, i.e. (1/N) * sum_i p(points_i),
// evaluated and summed on the GPU. This mirrors the numerical nimblecas::qmc_integrate but runs
// the per-point evaluation and the reduction on the device; the caller supplies the
// low-discrepancy sample points (e.g. from nimblecas::halton_point / sobol_point) and scales the
// returned mean by the domain measure to obtain an integral.
//
// HONESTY: this is a NUMERICAL (double) estimator — GPU acceleration here applies only to
// regular, data-parallel floating-point work. The exact-rational / symbolic paths (qmc_integrate_exact,
// the CAS) cannot run on the GPU and are unaffected. DETERMINISM: the device reduces in
// block/tree order rather than the CPU's strict left-to-right order, so the estimate may differ
// from a CPU average in the last bits — each is a valid equal-weight estimate.
//
// Fails with MathError::gpu_error when no device is present, the point set is empty (the mean of
// an empty set is undefined), or a CUDA call fails, and MathError::overflow when a span exceeds
// the int kernel bound.
[[nodiscard]] auto qmc_poly_integrate(std::span<const double> coeffs, std::span<const double> points)
    -> Result<double> {
    if (!available()) {
        return make_error<double>(MathError::gpu_error);
    }
    if (points.empty()) {
        return make_error<double>(MathError::gpu_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (coeffs.size() > int_max || points.size() > int_max) {
        return make_error<double>(MathError::overflow);
    }
    double mean = 0.0;
    const int rc = nimblecas_gpu_qmc_poly_integrate(coeffs.data(), static_cast<int>(coeffs.size()),
                                                    points.data(), static_cast<int>(points.size()),
                                                    &mean);
    if (rc != 0) {
        return make_error<double>(MathError::gpu_error);
    }
    return mean;
}

// One-level batch Haar discrete wavelet transform (orthonormal 1/sqrt(2) normalization) over
// `batch` contiguous signal blocks of `len` samples each — `data` is row-major, so
// data.size() == batch*len and `len` must be even. For each block the result packs its len/2
// approximation coefficients followed by its len/2 detail coefficients, so the returned vector
// has the same size and layout as the input.
//
// HONESTY: a REGULAR, DATA-PARALLEL numerical transform (double) — one independent lift per
// output pair — which is exactly the workload shape that maps well to the GPU. It is NOT a
// symbolic/exact operation. DETERMINISM: each output element is a single (e +/- o)/sqrt(2)
// expression with no cross-element reduction, so the result is elementwise deterministic and
// matches a CPU reference to within FMA-contraction last bits.
//
// Fails with MathError::domain_error when batch/len are non-positive, len is odd, or
// data.size() != batch*len; MathError::gpu_error when no device is present or a CUDA call fails;
// and MathError::overflow when the flat size exceeds the int kernel bound.
[[nodiscard]] auto haar_dwt_batch(std::span<const double> data, int batch, int len)
    -> Result<std::vector<double>> {
    if (!available()) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    if (batch <= 0 || len <= 0 || (len % 2) != 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const auto expected = static_cast<std::size_t>(batch) * static_cast<std::size_t>(len);
    if (data.size() != expected) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (expected > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    std::vector<double> out(expected);
    const int rc = nimblecas_gpu_haar_dwt_batch(data.data(), batch, len, out.data());
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// Batched dense double matrix multiply over `batch` independent problems, each C_b = A_b * B_b
// with A_b an m x k and B_b a k x n row-major double matrix; returns all `batch` products C_b
// (each m x n) packed contiguously in the same block layout. The inputs are the concatenated
// A and B blocks, so a.size() must equal batch*m*k and b.size() must equal batch*k*n.
//
// HONESTY: a REGULAR, DATA-PARALLEL numerical routine (double) — one independent dot product per
// output scalar — which is the workload shape the GPU accelerates. It is NOT a symbolic/exact
// operation; the exact-rational / CAS matrix paths cannot run on the device. DETERMINISM: each
// output element is an independent length-k accumulation with no cross-element reduction, so the
// result matches a CPU reference to within FMA-contraction last bits.
//
// Fails with MathError::gpu_error when no device is present or a CUDA call fails;
// MathError::domain_error when any dimension is non-positive or a span size disagrees with the
// dimensions; and MathError::overflow when a flattened element count exceeds the int kernel bound.
[[nodiscard]] auto batched_matmul(std::span<const double> a, std::span<const double> b, int batch,
                                  int m, int k, int n) -> Result<std::vector<double>> {
    if (!available()) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    if (batch <= 0 || m <= 0 || k <= 0 || n <= 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    // Every flattened count must fit the int kernel bound; check the three-factor products without
    // overflowing size_t (each dimension is already a positive int, so <= INT_MAX).
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    const auto sb = static_cast<std::size_t>(batch);
    const auto sm = static_cast<std::size_t>(m);
    const auto sk = static_cast<std::size_t>(k);
    const auto sn = static_cast<std::size_t>(n);
    const auto fits = [](std::size_t x, std::size_t y, std::size_t z, std::size_t limit) -> bool {
        // x, y, z are all > 0 here, so the divisions below are well defined.
        if (x > limit / y) {
            return false;
        }
        return (x * y) <= limit / z;
    };
    if (!fits(sb, sm, sk, int_max) || !fits(sb, sk, sn, int_max) || !fits(sb, sm, sn, int_max)) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    const std::size_t a_count = sb * sm * sk;  // proven <= int_max above, so no overflow
    const std::size_t b_count = sb * sk * sn;
    const std::size_t c_count = sb * sm * sn;
    if (a.size() != a_count || b.size() != b_count) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> out(c_count);
    const int rc = nimblecas_gpu_batched_matmul(a.data(), b.data(), out.data(), batch, m, k, n);
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// Batched radix-2 forward FFT over `batch` independent complex signals, each of length `n` (n a
// power of two). Each signal is supplied as 2*n interleaved doubles (re, im, re, im, ...) and the
// `batch` signals are packed contiguously, so in.size() must equal batch*2*n. Returns the DFT of
// each signal in the same interleaved layout, computed FORWARD as X_k = sum_j x_j e^{-2*pi*i*k*j/n}
// (negative exponent). One CUDA block transforms one signal.
//
// HONESTY: a NUMERICAL (double) transform. The GPU kernel is radix-2 only and length-capped at
// kGpuFftMaxLen; arbitrary (non-power-of-two) lengths are the domain of the CPU fft module
// (Bluestein), not this device path. DETERMINISM: the device evaluates the butterfly network in a
// fixed order, matching a same-order CPU FFT to within floating-point last bits.
//
// Fails with MathError::domain_error when batch <= 0, n is not a power of two, n exceeds
// kGpuFftMaxLen, or in.size() != batch*2*n; MathError::gpu_error when no device is present or a
// CUDA call fails; and MathError::overflow when the flat element count exceeds the int kernel bound.
[[nodiscard]] auto fft_batch(std::span<const double> in, int batch, int n)
    -> Result<std::vector<double>> {
    if (!available()) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    if (batch <= 0 || !is_power_of_two(n) || n > kGpuFftMaxLen) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    // Flattened count = batch * 2 * n must fit the int kernel bound. n <= kGpuFftMaxLen makes
    // 2*n small, so only the batch factor can overflow; check it without overflowing size_t.
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    const auto two_n = static_cast<std::size_t>(2) * static_cast<std::size_t>(n);
    if (static_cast<std::size_t>(batch) > int_max / two_n) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    const std::size_t total = static_cast<std::size_t>(batch) * two_n;  // proven <= int_max
    if (in.size() != total) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    std::vector<double> out(total);
    const int rc = nimblecas_gpu_fft_batch(in.data(), out.data(), batch, n);
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Batched Black-Scholes-Merton pricing (finance) — the batch-valuation MIRROR of the
// authoritative CPU nimblecas.pricing closed form. Fields mirror pricing::OptionSpec.
// ---------------------------------------------------------------------------
struct BsOption {
    double spot{100.0};
    double strike{100.0};
    double rate{0.0};
    double dividend{0.0};
    double volatility{0.2};
    double time{1.0};
    bool is_call{true};
};

namespace detail {
// Validate the physical domain (matching pricing's black_scholes_greeks guards) and repack
// the ergonomic BsOption span into the POD bridge array. A non-physical option ->
// domain_error; too many options -> overflow.
[[nodiscard]] inline auto to_bridge(std::span<const BsOption> opts)
    -> Result<std::vector<NimblecasBsOption>> {
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (opts.size() > int_max) { return make_error<std::vector<NimblecasBsOption>>(MathError::overflow); }
    std::vector<NimblecasBsOption> pod;
    pod.reserve(opts.size());
    for (const auto& o : opts) {
        if (o.spot <= 0.0 || o.strike <= 0.0 || o.time < 0.0 || o.volatility < 0.0) {
            return make_error<std::vector<NimblecasBsOption>>(MathError::domain_error);
        }
        pod.push_back(NimblecasBsOption{o.spot, o.strike, o.rate, o.dividend, o.volatility,
                                        o.time, o.is_call ? 1 : 0});
    }
    return pod;
}
}  // namespace detail

// Price a batch of Black-Scholes options on the device. Returns one price per option, in
// order. No device / a CUDA failure -> gpu_error; a non-physical option -> domain_error.
// The result agrees with pricing::black_scholes_price to floating-point tolerance.
[[nodiscard]] auto black_scholes_batch(std::span<const BsOption> opts)
    -> Result<std::vector<double>> {
    if (!available()) { return make_error<std::vector<double>>(MathError::gpu_error); }
    if (opts.empty()) { return std::vector<double>{}; }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<double>>(pod.error()); }
    std::vector<double> out(opts.size());
    const int rc = nimblecas_gpu_black_scholes_batch(pod->data(), out.data(),
                                                     static_cast<int>(pod->size()));
    if (rc != 0) { return make_error<std::vector<double>>(MathError::gpu_error); }
    return out;
}

// Identical result to black_scholes_batch, but the kernel is captured into a CUDA graph and
// replayed `iterations` times (a fixed-shape re-pricing / risk sweep). `iterations < 1` is
// treated as 1. Same error model.
[[nodiscard]] auto black_scholes_batch_graphed(std::span<const BsOption> opts, int iterations = 1)
    -> Result<std::vector<double>> {
    if (!available()) { return make_error<std::vector<double>>(MathError::gpu_error); }
    if (opts.empty()) { return std::vector<double>{}; }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<double>>(pod.error()); }
    std::vector<double> out(opts.size());
    const int rc = nimblecas_gpu_black_scholes_batch_graphed(
        pod->data(), out.data(), static_cast<int>(pod->size()), iterations);
    if (rc != 0) { return make_error<std::vector<double>>(MathError::gpu_error); }
    return out;
}

// ---------------------------------------------------------------------------
// Conjugate-gradient solve of a symmetric positive-definite CSR sparse system A x = b — the
// batch-solve MIRROR of the authoritative CPU nimblecas::cg.
// ---------------------------------------------------------------------------
struct CgCsrResult {
    std::vector<double> x;    // the solution vector, length n
    int iterations;           // CG iterations actually performed
    bool converged;           // whether ||r|| <= tol*||b|| was reached within max_iters
    double residual;          // final residual 2-norm ||b - A x||
};

// Solve A x = b for a SYMMETRIC POSITIVE-DEFINITE sparse A in CSR form on the device: `row_offsets`
// has length n+1 and `col_indices`/`values` hold the flattened nonzeros (equal length nnz), with
// n = b.size(). The solver starts from a zero initial guess and iterates until the residual 2-norm
// falls to tol*||b|| or `max_iters` is reached, returning the solution together with the iteration
// count, a converged flag, and the final residual norm.
//
// HONESTY: a NUMERICAL (double) iterative solver — GPU acceleration applies only to the regular,
// data-parallel SpMV and vector work. The exact-rational / symbolic linear-algebra paths cannot run
// on the device and are unaffected; the CPU nimblecas::cg remains authoritative.
// DETERMINISM: the device dot-product reductions sum in block/tree order rather than the CPU's
// strict left-to-right order, so the last bits of x can differ from a sequential CPU CG — each is a
// valid numerical solution.
//
// Fails with MathError::gpu_error when no device is present or a CUDA call fails;
// MathError::domain_error when b is empty (n must be > 0), row_offsets.size() != n+1, or
// col_indices.size() != values.size(); and MathError::overflow when a size exceeds the int kernel
// bound.
[[nodiscard]] auto cg_csr(std::span<const int> row_offsets, std::span<const int> col_indices,
                          std::span<const double> values, std::span<const double> b,
                          int max_iters = 1000, double tol = 1e-10) -> Result<CgCsrResult> {
    if (!available()) {
        return make_error<CgCsrResult>(MathError::gpu_error);
    }
    if (!(tol >= 0.0) || b.empty() || row_offsets.size() != b.size() + 1 ||
        col_indices.size() != values.size()) {
        return make_error<CgCsrResult>(MathError::domain_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (row_offsets.size() > int_max || values.size() > int_max) {
        return make_error<CgCsrResult>(MathError::overflow);
    }
    const auto n = static_cast<int>(b.size());
    const auto nnz = static_cast<int>(values.size());
    std::vector<double> x(b.size(), 0.0);  // zero initial guess, as documented
    int iterations = 0;
    int converged = 0;
    double residual = 0.0;
    const int rc = nimblecas_gpu_cg_csr(row_offsets.data(), col_indices.data(), values.data(), n,
                                        nnz, b.data(), x.data(), max_iters, tol, &iterations,
                                        &converged, &residual);
    if (rc != 0) {
        return make_error<CgCsrResult>(MathError::gpu_error);
    }
    return CgCsrResult{std::move(x), iterations, converged != 0, residual};
}

// ---------------------------------------------------------------------------
// BiCGStab solve of a general (possibly non-symmetric) CSR sparse system A x = b — the
// batch-solve MIRROR of the authoritative CPU nimblecas::bicgstab.
// ---------------------------------------------------------------------------
using BicgstabCsrResult = CgCsrResult;

// Solve A x = b for a GENERAL (possibly non-symmetric) sparse A in CSR form on the device or CPU fallback:
// `row_offsets` has length n+1 and `col_indices`/`values` hold the flattened nonzeros (equal length nnz),
// with n = b.size(). The solver starts from a zero initial guess and iterates until the true residual
// 2-norm falls to tol*||b|| or `max_iters` is reached.
//
// HONESTY: a NUMERICAL (double) iterative solver — the CPU nimblecas::bicgstab remains authoritative.
// `converged == false` is a legitimate outcome (budget exhausted or breakdown), never an error. The reported
// residual is the TRUE ||b - A x|| recomputed at exit.
// DETERMINISM: device dot-product reductions sum in block/tree order rather than the CPU's strict left-to-right
// order, so the last bits of x can differ from a sequential CPU BiCGStab — each is a valid numerical solution.
// Bitwise REPEATABLE run-to-run on the same device at fixed launch shape.
// DOMAIN: no SPD requirement; singular/ill-conditioned system breakdown is reported honestly.
//
// Fails with MathError::domain_error when b is empty (n must be > 0), row_offsets.size() != n+1,
// col_indices.size() != values.size(), or CSR interior invariants (monotone row_offsets, col index in [0, n))
// are violated; MathError::overflow when a size exceeds the int kernel bound; and MathError::gpu_error
// when a CUDA call fails on an available device. When no GPU is present, falls back to CPU krylov::bicgstab.
[[nodiscard]] auto bicgstab_csr(std::span<const int> row_offsets, std::span<const int> col_indices,
                                std::span<const double> values, std::span<const double> b,
                                int max_iters = 1000, double tol = 1e-10) -> Result<CgCsrResult> {
    if (max_iters < 0 || !(tol >= 0.0) || b.empty() || row_offsets.size() != b.size() + 1 ||
        col_indices.size() != values.size()) {
        return make_error<CgCsrResult>(MathError::domain_error);
    }
    // Overflow guard runs BEFORE the int truncation of n and the column scan below, so a >2^31
    // element system is honestly reported as overflow rather than a wrapped-n domain_error. Applied
    // before the available() branch so accept/reject never depends on device presence (negative
    // max_iters likewise: it hangs the CPU fallback but no-ops the device — rejected here for both).
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (b.size() > int_max || row_offsets.size() > int_max || values.size() > int_max) {
        return make_error<CgCsrResult>(MathError::overflow);
    }
    if (row_offsets.front() != 0 ||
        static_cast<std::size_t>(row_offsets.back()) != values.size()) {
        return make_error<CgCsrResult>(MathError::domain_error);
    }
    for (std::size_t i = 0; i + 1 < row_offsets.size(); ++i) {
        if (row_offsets[i] < 0 || row_offsets[i + 1] < row_offsets[i]) {
            return make_error<CgCsrResult>(MathError::domain_error);
        }
    }
    const auto n = static_cast<int>(b.size());
    for (const int c : col_indices) {
        if (c < 0 || c >= n) {
            return make_error<CgCsrResult>(MathError::domain_error);
        }
    }

    if (!available()) {
        auto A = nimblecas::csr_matvec(row_offsets, col_indices, values, b.size());
        auto res = nimblecas::bicgstab(A, b, tol, static_cast<std::size_t>(max_iters));
        if (!res) {
            return make_error<CgCsrResult>(res.error());
        }
        return CgCsrResult{std::move(res->x), static_cast<int>(res->iterations), res->converged,
                           res->residual};
    }

    const auto nnz = static_cast<int>(values.size());
    std::vector<double> x(b.size(), 0.0);
    int iterations = 0;
    int converged = 0;
    double residual = 0.0;
    const int rc = nimblecas_gpu_bicgstab_csr(row_offsets.data(), col_indices.data(), values.data(), n,
                                              nnz, b.data(), x.data(), max_iters, tol, &iterations,
                                              &converged, &residual);
    if (rc != 0) {
        return make_error<CgCsrResult>(MathError::gpu_error);
    }
    return CgCsrResult{std::move(x), iterations, converged != 0, residual};
}

// ---------------------------------------------------------------------------
// Restarted GMRES(m) solve of a general (possibly non-symmetric) CSR sparse system A x = b — the
// batch-solve MIRROR of the authoritative CPU nimblecas::gmres.
// ---------------------------------------------------------------------------
using GmresCsrResult = CgCsrResult;

// Solve A x = b for a GENERAL (possibly non-symmetric) sparse A in CSR form on the device or CPU fallback:
// `row_offsets` has length n+1 and `col_indices`/`values` hold the flattened nonzeros (equal length nnz),
// with n = b.size(). The solver starts from a zero initial guess and restarts until the true residual
// 2-norm falls to tol*||b|| or `max_iters` total inner iterations is reached.
//
// HONESTY: a NUMERICAL (double) iterative solver — the CPU nimblecas::gmres remains authoritative.
// `converged == false` is a legitimate outcome (budget exhausted or breakdown), never an error. The reported
// residual is the TRUE ||b - A x|| recomputed at exit.
// DETERMINISM: device dot-product reductions sum in block/tree order rather than the CPU's strict left-to-right
// order, so the last bits of x can differ from a sequential CPU GMRES — each is a valid numerical solution.
// Bitwise REPEATABLE run-to-run on the same device at fixed launch shape.
// DOMAIN: no SPD requirement; `restart` clamped to min(restart, n); memory O(n*(restart+1)) on device.
//
// Fails with MathError::domain_error when max_iters < 0, restart < 1, b is empty (n must be > 0),
// row_offsets.size() != n+1, col_indices.size() != values.size(), or CSR interior invariants (monotone
// row_offsets, col index in [0, n)) are violated; MathError::overflow when a size exceeds the int kernel
// bound or basis allocation sizing wraps; and MathError::gpu_error when a CUDA call fails on an available device.
// When no GPU is present, falls back to CPU krylov::gmres.
[[nodiscard]] auto gmres_csr(std::span<const int> row_offsets, std::span<const int> col_indices,
                             std::span<const double> values, std::span<const double> b,
                             int max_iters = 1000, double tol = 1e-10, int restart = 30)
    -> Result<CgCsrResult> {
    if (max_iters < 0 || restart < 1 || !(tol >= 0.0) || b.empty() ||
        row_offsets.size() != b.size() + 1 || col_indices.size() != values.size()) {
        return make_error<CgCsrResult>(MathError::domain_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (b.size() > int_max || row_offsets.size() > int_max || values.size() > int_max) {
        return make_error<CgCsrResult>(MathError::overflow);
    }
    const std::size_t m_size = std::min<std::size_t>(static_cast<std::size_t>(restart), b.size());
    if (b.size() != 0 && (m_size + 1) > std::numeric_limits<std::size_t>::max() / b.size() / sizeof(double)) {
        return make_error<CgCsrResult>(MathError::overflow);
    }
    if (row_offsets.front() != 0 ||
        static_cast<std::size_t>(row_offsets.back()) != values.size()) {
        return make_error<CgCsrResult>(MathError::domain_error);
    }
    for (std::size_t i = 0; i + 1 < row_offsets.size(); ++i) {
        if (row_offsets[i] < 0 || row_offsets[i + 1] < row_offsets[i]) {
            return make_error<CgCsrResult>(MathError::domain_error);
        }
    }
    const auto n = static_cast<int>(b.size());
    for (const int c : col_indices) {
        if (c < 0 || c >= n) {
            return make_error<CgCsrResult>(MathError::domain_error);
        }
    }

    if (!available()) {
        auto A = nimblecas::csr_matvec(row_offsets, col_indices, values, b.size());
        auto res = nimblecas::gmres(A, b, tol, static_cast<std::size_t>(max_iters),
                                    static_cast<std::size_t>(restart));
        if (!res) {
            return make_error<CgCsrResult>(res.error());
        }
        return CgCsrResult{std::move(res->x), static_cast<int>(res->iterations), res->converged,
                           res->residual};
    }

    const auto nnz = static_cast<int>(values.size());
    std::vector<double> x(b.size(), 0.0);
    int iterations = 0;
    int converged = 0;
    double residual = 0.0;
    const int rc = nimblecas_gpu_gmres_csr(row_offsets.data(), col_indices.data(), values.data(), n,
                                           nnz, b.data(), x.data(), max_iters, tol, restart,
                                           &iterations, &converged, &residual);
    if (rc != 0) {
        return make_error<CgCsrResult>(MathError::gpu_error);
    }
    return CgCsrResult{std::move(x), iterations, converged != 0, residual};
}

// One CSR system view for the batched solver; all spans must outlive the call.
struct CsrSystem {
    std::span<const int> row_offsets;    // length n_i + 1
    std::span<const int> col_indices;    // length nnz_i, entries in [0, n_i)
    std::span<const double> values;      // length nnz_i
    std::span<const double> b;           // length n_i
};

// Batched CG solve over K independent SYMMETRIC POSITIVE-DEFINITE sparse CSR systems — one CUDA
// block per system, the whole iteration in-kernel (no per-iteration host sync).
// Layout & validation: per-system shape/consistency/nnz is validated BEFORE the available() branch.
// SPD-ness is assumed (caller's responsibility); a non-SPD system is stopped honestly by the pap > 0
// guard with converged == false for that system only.
//
// HONESTY: NUMERICAL (double) iterative solver. CPU krylov::cg remains authoritative.
// `converged == false` is a legitimate outcome (budget exhausted or non-SPD breakdown), never an error.
// The reported residual is the TRUE ||b_i - A_i x_i|| recomputed in-kernel at exit.
// DETERMINISM: per-block tree reduction has a fixed 256-thread shape and index-ordered strided
// accumulation, so each system's result is a pure function of its inputs, independent of grid
// geometry and bitwise repeatable run-to-run; NOT bit-for-bit vs CPU krylov::cg, and a batch-of-one
// is NOT guaranteed bitwise equal to cg_csr (single-block 256-slice dot vs multi-block two-stage dot).
//
// Fails with MathError::domain_error when max_iters < 0, a system has empty b, row_offsets.size() != n_i+1,
// col_indices.size() != values.size(), non-monotone row_offsets, or out-of-range col index;
// MathError::overflow when total_n, total_nnz, or total_n + K exceeds INT_MAX;
// and MathError::gpu_error when a CUDA call fails on an available device.
// When no GPU is present (!available()), loops CPU krylov::cg per system.
[[nodiscard]] auto batched_cg_csr(std::span<const CsrSystem> systems, int max_iters = 1000,
                                  double tol = 1e-10) -> Result<std::vector<CgCsrResult>> {
    // Reject invalid parameters unconditionally (matching the scalar solvers), before the
    // empty-span shortcut. `!(tol >= 0.0)` also rejects NaN — a negative/NaN tol would make the
    // per-system stop threshold negative, defeating every convergence test.
    if (max_iters < 0 || !(tol >= 0.0)) {
        return make_error<std::vector<CgCsrResult>>(MathError::domain_error);
    }
    if (systems.empty()) {
        return std::vector<CgCsrResult>{};
    }

    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (systems.size() > int_max) {
        return make_error<std::vector<CgCsrResult>>(MathError::overflow);
    }

    std::size_t total_n = 0;
    std::size_t total_nnz = 0;

    for (const auto& s : systems) {
        if (s.b.empty() || s.row_offsets.size() != s.b.size() + 1 ||
            s.col_indices.size() != s.values.size()) {
            return make_error<std::vector<CgCsrResult>>(MathError::domain_error);
        }
        if (s.row_offsets.front() != 0 ||
            static_cast<std::size_t>(s.row_offsets.back()) != s.values.size()) {
            return make_error<std::vector<CgCsrResult>>(MathError::domain_error);
        }
        for (std::size_t i = 0; i + 1 < s.row_offsets.size(); ++i) {
            if (s.row_offsets[i] < 0 || s.row_offsets[i + 1] < s.row_offsets[i]) {
                return make_error<std::vector<CgCsrResult>>(MathError::domain_error);
            }
        }
        if (s.b.size() > int_max || s.values.size() > int_max) {
            return make_error<std::vector<CgCsrResult>>(MathError::overflow);
        }
        const auto n_i = static_cast<int>(s.b.size());
        for (const int c : s.col_indices) {
            if (c < 0 || c >= n_i) {
                return make_error<std::vector<CgCsrResult>>(MathError::domain_error);
            }
        }
        total_n += s.b.size();
        total_nnz += s.values.size();
    }

    if (total_n > int_max || total_nnz > int_max ||
        (total_n + systems.size()) > int_max) {
        return make_error<std::vector<CgCsrResult>>(MathError::overflow);
    }

    if (!available()) {
        std::vector<CgCsrResult> out;
        out.reserve(systems.size());
        for (const auto& s : systems) {
            auto A = nimblecas::csr_matvec(s.row_offsets, s.col_indices, s.values, s.b.size());
            auto res = nimblecas::cg(A, s.b, tol, static_cast<std::size_t>(max_iters));
            if (!res) {
                return make_error<std::vector<CgCsrResult>>(res.error());
            }
            out.push_back(CgCsrResult{std::move(res->x), static_cast<int>(res->iterations),
                                      res->converged, res->residual});
        }
        return out;
    }

    const auto K = systems.size();
    std::vector<int> x_off(K + 1, 0);
    std::vector<int> nz_off(K + 1, 0);
    std::vector<int> row_offsets_cat;
    row_offsets_cat.reserve(total_n + K);
    std::vector<int> col_indices_cat;
    col_indices_cat.reserve(total_nnz);
    std::vector<double> values_cat;
    values_cat.reserve(total_nnz);
    std::vector<double> b_cat;
    b_cat.reserve(total_n);
    std::vector<double> x_cat(total_n, 0.0);

    for (std::size_t i = 0; i < K; ++i) {
        const auto& s = systems[i];
        x_off[i + 1] = x_off[i] + static_cast<int>(s.b.size());
        nz_off[i + 1] = nz_off[i] + static_cast<int>(s.values.size());
        row_offsets_cat.insert(row_offsets_cat.end(), s.row_offsets.begin(), s.row_offsets.end());
        col_indices_cat.insert(col_indices_cat.end(), s.col_indices.begin(), s.col_indices.end());
        values_cat.insert(values_cat.end(), s.values.begin(), s.values.end());
        b_cat.insert(b_cat.end(), s.b.begin(), s.b.end());
    }

    std::vector<int> out_iters(K, 0);
    std::vector<int> out_converged(K, 0);
    std::vector<double> out_resid(K, 0.0);

    const int rc = nimblecas_gpu_batched_cg_csr(
        row_offsets_cat.data(), col_indices_cat.data(), values_cat.data(), x_off.data(),
        nz_off.data(), static_cast<int>(K), b_cat.data(), x_cat.data(), max_iters, tol,
        out_iters.data(), out_converged.data(), out_resid.data());

    if (rc != 0) {
        return make_error<std::vector<CgCsrResult>>(MathError::gpu_error);
    }

    std::vector<CgCsrResult> results;
    results.reserve(K);
    for (std::size_t i = 0; i < K; ++i) {
        const std::size_t start = static_cast<std::size_t>(x_off[i]);
        const std::size_t end = static_cast<std::size_t>(x_off[i + 1]);
        std::vector<double> xi(x_cat.begin() + static_cast<std::ptrdiff_t>(start),
                               x_cat.begin() + static_cast<std::ptrdiff_t>(end));
        results.push_back(CgCsrResult{std::move(xi), out_iters[i], out_converged[i] != 0,
                                      out_resid[i]});
    }
    return results;
}


// ---------------------------------------------------------------------------
// FAMILY A — Batched derivative pricing (gpu_pricing_kernels.cu).
//
// Every entry point here is a batch MIRROR of the authoritative CPU nimblecas.pricing
// implementation, never a second source of truth. FALLBACK CONTRACT (shared with cg_csr,
// bicgstab_csr, and the module wrappers above; unlike the raw numeric kernels): when NO device
// is present these functions compute the result on the CPU via the CPU module and return real
// values — an honest fallback, not gpu_error. A CUDA
// failure on a machine that HAS a device is still reported as gpu_error (the device was
// asked and the device failed; we never silently switch answers).
// ---------------------------------------------------------------------------

// Paths per Monte-Carlo segment: one device thread serially prices one segment's contiguous
// counter sub-range in index order, making the partial a pure function of the segment index.
// Must match kMcSegPaths in gpu_pricing_kernels.cu.
inline constexpr std::uint64_t kGpuMcSegPaths = 4096;
// Path-count cap, mirroring pricing::monte_carlo_european's kMaxPaths bound.
inline constexpr std::uint64_t kGpuMcMaxPaths = 1'000'000'000;

// Price a batch of European options by GPU Monte Carlo path simulation, returning one
// pricing::McResult { price, std_error, paths } per option, in order. Every option is priced
// over the SAME counter stream [0, paths) with key = splitmix64(seed) — exactly the draw
// indexing of pricing::monte_carlo_european — with antithetic variates, so item i estimates
// the same quantity as monte_carlo_european(spec_i, paths, seed).
//
// HONESTY: STATISTICAL (double) — the estimate carries its standard error. REPRODUCIBILITY:
// the device result is a pure function of (opts, paths, seed): the counter-based Threefry
// draws are bit-identical to nimblecas.rng's counter_u64, segments are a FIXED decomposition
// of the path range summed in index order, and the segment partials are folded by a
// fixed-shape (256-thread, one block per option) reduction — so the result is independent of
// grid/block geometry and identical across repeated calls on the same device/toolkit. It
// EQUALS the CPU monte_carlo_european to floating-point tolerance (documented at 1e-6
// absolute: the divergence sources are the reduction association order plus last-bit
// differences of the device exp/log versus the CPU simd::exp/simd::log_one in the ~5%
// Acklam tail region — orders of magnitude below the MC standard error), NOT bit-for-bit.
// CPU fallback (no device): pricing::monte_carlo_european_parallel per option, which carries
// the same (spec, paths, seed)-only reproducibility contract.
//
// Fails with MathError::domain_error when paths == 0, paths > kGpuMcMaxPaths, or an option is
// non-physical (spot<=0, strike<=0, time<0, volatility<0 — note strike>0 is required here,
// slightly stricter than the CPU MC, and honestly rejected rather than silently accepted);
// MathError::overflow when opts.size() or opts.size()*ceil(paths/kGpuMcSegPaths) exceeds the
// int kernel bound; MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto monte_carlo_european_batch(std::span<const BsOption> opts,
                                              std::uint64_t paths, std::uint64_t seed)
    -> Result<std::vector<pricing::McResult>> {
    if (paths == 0 || paths > kGpuMcMaxPaths) {
        return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
    }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::McResult>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::McResult>{}; }
    const std::uint64_t nseg = (paths + kGpuMcSegPaths - 1) / kGpuMcSegPaths;
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (opts.size() > int_max || nseg > int_max ||
        opts.size() > static_cast<std::size_t>(int_max / nseg)) {
        return make_error<std::vector<pricing::McResult>>(MathError::overflow);
    }
    if (!available()) {
        // Honest CPU fallback: the authoritative reproducible CPU pricer, option by option.
        std::vector<pricing::McResult> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto r = pricing::monte_carlo_european_parallel(spec, paths, seed);
            if (!r) { return make_error<std::vector<pricing::McResult>>(r.error()); }
            out.push_back(*r);
        }
        return out;
    }
    std::vector<NimblecasMcEstimate> est(opts.size());
    const int rc = nimblecas_gpu_mc_european_batch(pod->data(), static_cast<int>(pod->size()),
                                                   paths, seed, est.data());
    if (rc != 0) { return make_error<std::vector<pricing::McResult>>(MathError::gpu_error); }
    std::vector<pricing::McResult> out;
    out.reserve(est.size());
    for (const auto& e : est) { out.push_back(pricing::McResult{e.price, e.std_error, paths}); }
    return out;
}

// Batch analytic Black-Scholes-Merton Greeks on the device — the batch mirror of
// pricing::black_scholes_greeks (same d1/d2, same degenerate T==0/vol==0 collapse with
// limit delta), one pricing::Greeks per option, in order.
//
// HONESTY: a NUMERICAL closed form; the device result agrees with the CPU closed form to
// 1e-9 relative (the only divergence is last-bit differences of the device erfc/exp/log/sqrt
// versus glibc's). DETERMINISM: elementwise, no reduction — fully deterministic.
// CPU fallback (no device): pricing::black_scholes_greeks per option.
//
// A non-physical option -> domain_error; too many options -> overflow; a CUDA failure with a
// device present -> gpu_error.
[[nodiscard]] auto black_scholes_greeks_batch(std::span<const BsOption> opts)
    -> Result<std::vector<pricing::Greeks>> {
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::Greeks>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::Greeks>{}; }
    if (!available()) {
        std::vector<pricing::Greeks> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto g = pricing::black_scholes_greeks(spec);
            if (!g) { return make_error<std::vector<pricing::Greeks>>(g.error()); }
            out.push_back(*g);
        }
        return out;
    }
    std::vector<NimblecasBsGreeks> pg(opts.size());
    const int rc = nimblecas_gpu_bs_greeks_batch(pod->data(), pg.data(),
                                                 static_cast<int>(pod->size()));
    if (rc != 0) { return make_error<std::vector<pricing::Greeks>>(MathError::gpu_error); }
    std::vector<pricing::Greeks> out;
    out.reserve(pg.size());
    for (const auto& g : pg) {
        out.push_back(pricing::Greeks{g.price, g.delta, g.gamma, g.vega, g.theta, g.rho});
    }
    return out;
}

// Batch extended (higher-order) Black-Scholes Greeks on the device — the batch mirror of
// pricing::black_scholes_extended_greeks, INCLUDING its algorithm choices: closed forms for
// vanna/vomma/speed/zomma/lambda/dual_delta/dual_gamma/epsilon/ultima and CENTRAL FINITE
// DIFFERENCES of the analytic Greeks for charm/color/veta (h = 1e-4*T) and vera
// (hv = 1e-4*sig) — mirrored, not re-derived, so the two implementations cannot drift.
//
// HONESTY: NUMERICAL. The finite-difference fields divide last-bit noise by 2e-4-scale
// steps, so the validated agreement bound is 1e-7 * max(1, |cpu|) per field (four orders of
// magnitude of margin over the measured ulp-level divergence, and four orders tighter than
// any real formula error). DETERMINISM: elementwise, no reduction.
// CPU fallback (no device): pricing::black_scholes_extended_greeks per option.
//
// Requires T > 0 and volatility > 0 for every option (the CPU guard) -> domain_error
// otherwise; too many options -> overflow; a CUDA failure with a device present -> gpu_error.
[[nodiscard]] auto black_scholes_extended_greeks_batch(std::span<const BsOption> opts)
    -> Result<std::vector<pricing::ExtendedGreeks>> {
    // Validate the full domain (basic physicality via to_bridge's checks, plus the extended
    // set's strict time>0/vol>0 guard) BEFORE any repacking or the available() split, so the
    // error model is identical with or without a device (Rule 32, device-independent errors).
    for (const auto& o : opts) {  // the extended set needs the strict CPU guard
        if (o.time <= 0.0 || o.volatility <= 0.0) {
            return make_error<std::vector<pricing::ExtendedGreeks>>(MathError::domain_error);
        }
    }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::ExtendedGreeks>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::ExtendedGreeks>{}; }
    if (!available()) {
        std::vector<pricing::ExtendedGreeks> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto g = pricing::black_scholes_extended_greeks(spec);
            if (!g) { return make_error<std::vector<pricing::ExtendedGreeks>>(g.error()); }
            out.push_back(*g);
        }
        return out;
    }
    std::vector<NimblecasBsExtGreeks> pg(opts.size());
    const int rc = nimblecas_gpu_bs_extended_greeks_batch(pod->data(), pg.data(),
                                                          static_cast<int>(pod->size()));
    if (rc != 0) {
        return make_error<std::vector<pricing::ExtendedGreeks>>(MathError::gpu_error);
    }
    std::vector<pricing::ExtendedGreeks> out;
    out.reserve(pg.size());
    for (const auto& g : pg) {
        out.push_back(pricing::ExtendedGreeks{g.vanna, g.charm, g.vomma, g.veta, g.speed,
                                              g.zomma, g.color, g.lambda, g.dual_delta,
                                              g.dual_gamma, g.epsilon, g.vera, g.ultima});
    }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY B — Strategy payoff / P&L grid sweeps (gpu_sweep_kernels.cu).
//
// The expiry P&L of a vanilla option book is EXACTLY a continuous piecewise-linear function
// of the terminal price (optstrat's honesty boundary), and a futures book is exactly linear
// — so unlike the statistical Family A these sweeps are EXACTLY-REPRODUCIBLE computations.
// The device kernel executes the same IEEE-754 double operation sequence as the CPU
// reference (explicit __dadd_rn/__dmul_rn/__dsub_rn/fmax, no FMA contraction), AND the CPU
// reference is itself pinned non-contracted (optstrat::payoff_at / net_premium carry
// `#pragma clang fp contract(off)`) — pinning BOTH sides to the same rounding sequence makes
// the GPU value at every grid point EQUAL to the CPU value BIT-FOR-BIT, independent of the
// compiler's default contraction (tests validate to 1e-12 and pin hand values with exact ==).
// FALLBACK CONTRACT: as in Family A, no device -> the CPU (optstrat / futures) computes the
// result and real values are returned; a CUDA failure with a device present -> gpu_error.
// ---------------------------------------------------------------------------

namespace detail {
// Flatten optstrat legs into the POD bridge encoding (right: 0 call, 1 put, 2 underlying).
[[nodiscard]] inline auto legs_to_bridge(std::span<const optstrat::StrategyLeg> legs)
    -> std::vector<NimblecasSweepLeg> {
    std::vector<NimblecasSweepLeg> pod;
    pod.reserve(legs.size());
    for (const auto& l : legs) {
        pod.push_back(NimblecasSweepLeg{l.strike, l.quantity, l.premium,
                                        static_cast<int>(l.kind)});
    }
    return pod;
}
// The shared sweep driver: validates sizes, falls back to the exact CPU optstrat evaluation
// when no device is present, otherwise crosses the bridge. net_of_premium selects P&L
// (payoff - net premium) versus gross payoff.
[[nodiscard]] inline auto strategy_grid(std::span<const optstrat::StrategyLeg> legs,
                                        std::span<const double> grid, bool net_of_premium)
    -> Result<std::vector<double>> {
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (legs.size() > int_max || grid.size() > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    std::vector<double> out(grid.size());
    if (grid.empty()) { return out; }
    if (!available()) {
        // Exact CPU fallback: rebuild the strategy and evaluate the authoritative
        // piecewise-linear payoff_at / pnl_at point by point (reuse, not re-derivation).
        auto strat = optstrat::OptionStrategy::create();
        for (const auto& l : legs) { std::ignore = strat.with_leg(l); }
        for (std::size_t i = 0; i < grid.size(); ++i) {
            out[i] = net_of_premium ? strat.pnl_at(grid[i]) : strat.payoff_at(grid[i]);
        }
        return out;
    }
    const auto pod = legs_to_bridge(legs);
    const int rc = nimblecas_gpu_strategy_grid(pod.data(), static_cast<int>(pod.size()),
                                               grid.data(), static_cast<int>(grid.size()),
                                               net_of_premium ? 1 : 0, out.data());
    if (rc != 0) { return make_error<std::vector<double>>(MathError::gpu_error); }
    return out;
}
}  // namespace detail

// Gross expiry payoff Σ quantity·terminal_value(s) of a signed option/underlying leg bag at
// every grid point, in grid order — the batch mirror of optstrat::OptionStrategy::payoff_at.
// Legs are summed in span order on both CPU and GPU (fixed ordering -> deterministic).
// Empty legs -> a vector of zeros; empty grid -> an empty vector; a span exceeding the int
// kernel bound -> overflow; a CUDA failure with a device present -> gpu_error.
[[nodiscard]] auto strategy_payoff_grid(std::span<const optstrat::StrategyLeg> legs,
                                        std::span<const double> grid)
    -> Result<std::vector<double>> {
    return detail::strategy_grid(legs, grid, /*net_of_premium=*/false);
}

// Expiry P&L (payoff − net premium) at every grid point — the batch mirror of
// optstrat::OptionStrategy::pnl_at, computed as the SAME two in-order sums (payoff, then
// Σ leg cost) followed by one subtraction, so the device value equals the CPU double
// evaluation exactly. Same error model as strategy_payoff_grid.
[[nodiscard]] auto strategy_pnl_grid(std::span<const optstrat::StrategyLeg> legs,
                                     std::span<const double> grid)
    -> Result<std::vector<double>> {
    return detail::strategy_grid(legs, grid, /*net_of_premium=*/true);
}

// Uniform-settlement P&L of a futures leg bag at every grid point — the batch mirror of
// futures::FuturesStrategy::pnl_at_uniform: Σ (quantityᵢ·contract_sizeᵢ)·(s − entryᵢ) with
// the CPU's association (q·cs first, on the host; then ·(s − entry) on the device), summed
// in span order. Exactly linear, exactly reproducible (1e-12-validated, expected equal).
// Empty legs -> zeros; empty grid -> empty; span too large -> overflow; a CUDA failure with
// a device present -> gpu_error.
[[nodiscard]] auto futures_pnl_grid(std::span<const futures::FuturesLeg> legs,
                                    std::span<const double> grid)
    -> Result<std::vector<double>> {
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (legs.size() > int_max || grid.size() > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    std::vector<double> out(grid.size());
    if (grid.empty()) { return out; }
    if (!available()) {
        auto strat = futures::FuturesStrategy::create();
        for (const auto& l : legs) { std::ignore = strat.with_leg(l); }
        for (std::size_t i = 0; i < grid.size(); ++i) {
            out[i] = strat.pnl_at_uniform(grid[i]);
        }
        return out;
    }
    std::vector<NimblecasFuturesSweepLeg> pod;
    pod.reserve(legs.size());
    for (const auto& l : legs) {
        // q*cs on the host mirrors FuturesLeg::pnl_at's left-to-right association exactly.
        pod.push_back(NimblecasFuturesSweepLeg{l.quantity * l.contract_size, l.entry_price});
    }
    const int rc = nimblecas_gpu_futures_grid(pod.data(), static_cast<int>(pod.size()),
                                              grid.data(), static_cast<int>(grid.size()),
                                              out.data());
    if (rc != 0) { return make_error<std::vector<double>>(MathError::gpu_error); }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY C — Path-dependent derivative pricing (gpu_asian_kernels.cu).
// ---------------------------------------------------------------------------

// Maximum averaging steps, mirroring pricing::monte_carlo_asian's kMaxSteps bound.
inline constexpr int kGpuMcMaxSteps = 100'000;
// Path*steps cap, mirroring pricing::monte_carlo_asian's kMaxPathSteps bound.
inline constexpr std::uint64_t kGpuMcMaxPathSteps = 1'000'000'000;

// Price a batch of arithmetic-average Asian options by GPU Monte Carlo path simulation,
// returning one pricing::McResult { price, std_error, paths } per option, in order.
// Every option is priced over the SAME counter stream [0, paths*steps) with key = splitmix64(seed)
// — step t of path p draws counter index (p*steps + t), matching pricing::monte_carlo_asian —
// so item i estimates the same quantity as monte_carlo_asian(spec_i, paths, steps, seed, false).
//
// HONESTY: STATISTICAL (double). The Threefry draw BITS are bit-identical to the CPU counter RNG;
// the normal z is bit-identical in Acklam's central ~95% region, with <=1 ULP differences in the
// ~5% tail (device libm log vs simd::log_one). Price agrees with CPU monte_carlo_asian(..., false)
// to ~1e-6 relative (hardware exp vs simd::exp_into, plus the tail-z last bit). Geometric control
// variate is CPU-only.
// CPU fallback (no device): pricing::monte_carlo_asian(spec, paths, steps, seed, false) per option.
//
// Fails with MathError::domain_error when paths == 0, steps < 1, steps > kGpuMcMaxSteps,
// paths * steps > kGpuMcMaxPathSteps, or an option is non-physical; MathError::overflow when
// opts.size() or opts.size()*ceil(paths/kGpuMcSegPaths) exceeds the int kernel bound;
// MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto monte_carlo_asian_batch(std::span<const BsOption> opts,
                                           std::uint64_t paths, int steps, std::uint64_t seed)
    -> Result<std::vector<pricing::McResult>> {
    if (paths == 0 || steps < 1 || steps > kGpuMcMaxSteps ||
        paths > kGpuMcMaxPathSteps / static_cast<std::uint64_t>(steps)) {
        return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
    }
    // Physical domain matched EXACTLY to pricing::monte_carlo_asian (spot>0, vol>=0, time>0) so the
    // accept/reject decision — and therefore the result — never depends on whether a device is
    // present. (to_bridge additionally rejects strike<=0, a module-wide batch-POD precondition that
    // is applied identically on the device and CPU-fallback paths, so it is not device-dependent.)
    for (const auto& o : opts) {
        if (o.spot <= 0.0 || o.volatility < 0.0 || o.time <= 0.0) {
            return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
        }
    }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::McResult>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::McResult>{}; }
    const std::uint64_t nseg = (paths + kGpuMcSegPaths - 1) / kGpuMcSegPaths;
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (opts.size() > int_max || nseg > int_max ||
        opts.size() > static_cast<std::size_t>(int_max / nseg)) {
        return make_error<std::vector<pricing::McResult>>(MathError::overflow);
    }
    if (!available()) {
        std::vector<pricing::McResult> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto r = pricing::monte_carlo_asian(spec, paths, steps, seed, /*control_variate=*/false);
            if (!r) { return make_error<std::vector<pricing::McResult>>(r.error()); }
            out.push_back(*r);
        }
        return out;
    }
    std::vector<NimblecasMcEstimate> est(opts.size());
    const int rc = nimblecas_gpu_mc_asian_batch(pod->data(), static_cast<int>(pod->size()),
                                                steps, paths, seed, est.data());
    if (rc != 0) { return make_error<std::vector<pricing::McResult>>(MathError::gpu_error); }
    std::vector<pricing::McResult> out;
    out.reserve(est.size());
    for (const auto& e : est) { out.push_back(pricing::McResult{e.price, e.std_error, paths}); }
    return out;
}

// Price a batch of single-barrier options by GPU Monte Carlo path simulation, returning one
// pricing::McResult { price, std_error, paths } per option, in order. Every option is priced over
// the SAME counter stream [0, paths*steps) with key = splitmix64(seed) — step t of path p draws
// counter index (p*steps + t), matching pricing::barrier_option_mc — so item i estimates the same
// barrier payoff as barrier_option_mc(spec_i, barrier, knock_in, paths, steps, seed). The single
// barrier level and knock_in flag apply to every option; up/down is decided per option from its
// own spot (down = barrier < spot).
//
// HONESTY: STATISTICAL (double) — the estimate carries its standard error. The Threefry draw BITS
// are bit-identical to the CPU counter RNG; the normal z is bit-identical in Acklam's central ~95%
// region, with <=1 ULP differences in the ~5% tail (device libm log vs simd::log_one). Price
// matches pricing::barrier_option_mc to ~1e-5 relative on non-grazing cases (hardware exp vs
// std::exp, plus the tail-z last bit).
// BARRIER GRAZING CAVEAT: knock detection is an exact threshold check (s <= barrier / s >= barrier),
// so paths that graze the barrier within a few ULPs can knock differently on GPU vs CPU, causing
// occasional whole-path payoff divergence — price barriers away from a dense grazing band, or rely
// on the grazing-immune identity knock_in + knock_out == vanilla (exact per path).
// CPU fallback (no device): pricing::barrier_option_mc per option.
//
// Fails with MathError::domain_error when paths == 0, steps < 1, steps > kGpuMcMaxSteps,
// barrier <= 0, paths * steps > kGpuMcMaxPathSteps, or an option is non-physical (spot <= 0,
// vol < 0, time <= 0; strike <= 0 is rejected by the shared batch-POD builder on both the device
// and CPU-fallback paths, as for every Family A/C pricer); MathError::overflow when opts.size() or
// opts.size()*ceil(paths/kGpuMcSegPaths) exceeds the int kernel bound; MathError::gpu_error when a
// device is present but a CUDA call fails.
[[nodiscard]] auto barrier_option_mc_batch(std::span<const BsOption> opts, double barrier,
                                           bool knock_in, std::uint64_t paths, int steps,
                                           std::uint64_t seed)
    -> Result<std::vector<pricing::McResult>> {
    if (paths == 0 || steps < 1 || steps > kGpuMcMaxSteps || barrier <= 0.0 ||
        paths > kGpuMcMaxPathSteps / static_cast<std::uint64_t>(steps)) {
        return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
    }
    // Physical domain matched EXACTLY to pricing::barrier_option_mc (spot>0, vol>=0, time>0) so the
    // accept/reject decision — and therefore the result — never depends on device presence.
    for (const auto& o : opts) {
        if (o.spot <= 0.0 || o.volatility < 0.0 || o.time <= 0.0) {
            return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
        }
    }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::McResult>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::McResult>{}; }
    const std::uint64_t nseg = (paths + kGpuMcSegPaths - 1) / kGpuMcSegPaths;
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (opts.size() > int_max || nseg > int_max ||
        opts.size() > static_cast<std::size_t>(int_max / nseg)) {
        return make_error<std::vector<pricing::McResult>>(MathError::overflow);
    }
    if (!available()) {
        std::vector<pricing::McResult> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto r = pricing::barrier_option_mc(spec, barrier, knock_in, paths, steps, seed);
            if (!r) { return make_error<std::vector<pricing::McResult>>(r.error()); }
            out.push_back(*r);
        }
        return out;
    }
    std::vector<NimblecasMcEstimate> est(opts.size());
    const int rc = nimblecas_gpu_mc_barrier_batch(pod->data(), static_cast<int>(pod->size()),
                                                  barrier, knock_in ? 1 : 0, steps, paths, seed,
                                                  est.data());
    if (rc != 0) { return make_error<std::vector<pricing::McResult>>(MathError::gpu_error); }
    std::vector<pricing::McResult> out;
    out.reserve(est.size());
    for (const auto& e : est) { out.push_back(pricing::McResult{e.price, e.std_error, paths}); }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY E — Longstaff-Schwartz American Monte Carlo (gpu_lsm_kernels.cu).
// ---------------------------------------------------------------------------

// Price a batch of American options by GPU Longstaff-Schwartz Monte Carlo path simulation,
// returning one pricing::McResult { price, std_error, paths } per option, in order. Every option
// is priced over the SAME counter stream [0, paths*steps) with key = splitmix64(seed) — step t of
// path p draws counter index (p*steps + t), matching pricing::longstaff_schwartz_american — so
// item i estimates the same quantity as longstaff_schwartz_american(spec_i, paths, steps, seed).
//
// HONESTY: STATISTICAL (double). Threefry draw BITS are bit-identical to the CPU counter RNG; the
// normal z is bit-identical in Acklam's central ~95% region, with <=1 ULP differences in the ~5%
// tail (device libm log vs simd::log_one); forward grid S agrees to ~1e-6 relative. However, the
// American PRICE matches the CPU oracle only to ~1e-3 relative because (a) the 3x3 normal-equations
// regression on basis {1, s, s^2} is ill-conditioned and summation-order-sensitive, so regression
// coefficients differ beyond ULP level, and (b) exercise decisions (ex > cont) are exact threshold
// checks, so paths near the boundary can flip decisions between GPU and CPU, altering whole-path
// cashflows. The GPU calculation is itself 100% deterministic (pure function of inputs).
// CPU fallback (no device): pricing::longstaff_schwartz_american(spec, paths, steps, seed) per option.
//
// Fails with MathError::domain_error when paths < 4, steps < 1, steps > kGpuMcMaxSteps,
// paths > kMaxCells / (steps + 1) [kMaxCells = 500,000,000], or an option is non-physical
// (spot <= 0, volatility <= 0 [strict <=], time <= 0; strike <= 0 is rejected by shared batch POD);
// MathError::overflow when opts.size() or ceil(paths/kGpuMcSegPaths) exceeds int bounds;
// MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto longstaff_schwartz_american_batch(std::span<const BsOption> opts,
                                                     std::uint64_t paths, int steps,
                                                     std::uint64_t seed)
    -> Result<std::vector<pricing::McResult>> {
    constexpr std::uint64_t kMaxCells = 500'000'000;
    if (paths < 4 || steps < 1 || steps > kGpuMcMaxSteps ||
        paths > kMaxCells / (static_cast<std::uint64_t>(steps) + 1)) {
        return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
    }
    // Physical domain matched EXACTLY to pricing::longstaff_schwartz_american (spot>0, vol>0 strict, time>0)
    // so accept/reject decision never depends on whether a device is present.
    for (const auto& o : opts) {
        if (o.spot <= 0.0 || o.volatility <= 0.0 || o.time <= 0.0) {
            return make_error<std::vector<pricing::McResult>>(MathError::domain_error);
        }
    }
    auto pod = detail::to_bridge(opts);
    if (!pod) { return make_error<std::vector<pricing::McResult>>(pod.error()); }
    if (opts.empty()) { return std::vector<pricing::McResult>{}; }
    const std::uint64_t nseg = (paths + kGpuMcSegPaths - 1) / kGpuMcSegPaths;
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (opts.size() > int_max || nseg > int_max) {
        return make_error<std::vector<pricing::McResult>>(MathError::overflow);
    }
    if (!available()) {
        std::vector<pricing::McResult> out;
        out.reserve(opts.size());
        for (const auto& o : opts) {
            const auto spec = pricing::OptionSpec{}
                                  .with_spot(o.spot).with_strike(o.strike).with_rate(o.rate)
                                  .with_dividend(o.dividend).with_volatility(o.volatility)
                                  .with_expiry(o.time)
                                  .with_type(o.is_call ? pricing::OptionType::call
                                                       : pricing::OptionType::put);
            auto r = pricing::longstaff_schwartz_american(spec, paths, steps, seed);
            if (!r) { return make_error<std::vector<pricing::McResult>>(r.error()); }
            out.push_back(*r);
        }
        return out;
    }
    std::vector<NimblecasMcEstimate> est(opts.size());
    const int rc = nimblecas_gpu_lsm_american_batch(pod->data(), static_cast<int>(pod->size()),
                                                     steps, paths, seed, est.data());
    if (rc != 0) { return make_error<std::vector<pricing::McResult>>(MathError::gpu_error); }
    std::vector<pricing::McResult> out;
    out.reserve(est.size());
    for (const auto& e : est) { out.push_back(pricing::McResult{e.price, e.std_error, paths}); }
    return out;
}

// Frequency-response Bode sweep H(iω) over an angular frequency grid ω (rad/s).
// Returns BodePoint { omega, magnitude_db (20·log10|H|), phase_deg (arg H in deg) }.
//
// HONESTY: NUMERICAL (double complex). Elementwise calculation (one thread per ω),
// so execution order is independent of thread count or block geometry. Results
// match CPU nimblecas::bode to ~1e-9 relative (device log10/atan2/hypot vs libm).
// CPU fallback (no device): nimblecas::bode(tf, omegas). Fails with domain_error when
// the transfer function denominator is zero, overflow when size exceeds int bounds,
// and gpu_error on CUDA launch failure.
[[nodiscard]] auto bode_sweep(const nimblecas::TransferFunction& tf, std::span<const double> omegas)
    -> Result<std::vector<nimblecas::BodePoint>> {
    if (tf.denominator().is_zero()) {
        return make_error<std::vector<nimblecas::BodePoint>>(MathError::domain_error);
    }
    if (omegas.empty()) {
        return std::vector<nimblecas::BodePoint>{};
    }
    const auto num_coeffs_span = tf.numerator().coefficients();
    const auto den_coeffs_span = tf.denominator().coefficients();
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (num_coeffs_span.size() > int_max || den_coeffs_span.size() > int_max ||
        omegas.size() > int_max) {
        return make_error<std::vector<nimblecas::BodePoint>>(MathError::overflow);
    }
    // Domain/overflow validated above (before this branch) so accept/reject never depends on device presence.
    if (!available()) {
        return nimblecas::bode(tf, omegas);
    }
    std::vector<double> num_coeffs;
    num_coeffs.reserve(num_coeffs_span.size());
    for (const auto& r : num_coeffs_span) {
        num_coeffs.push_back(static_cast<double>(r.numerator()) /
                            static_cast<double>(r.denominator()));
    }
    std::vector<double> den_coeffs;
    den_coeffs.reserve(den_coeffs_span.size());
    for (const auto& r : den_coeffs_span) {
        den_coeffs.push_back(static_cast<double>(r.numerator()) /
                            static_cast<double>(r.denominator()));
    }
    std::vector<double> mag_db(omegas.size());
    std::vector<double> phase_deg(omegas.size());
    const int rc = nimblecas_gpu_bode_sweep(
        num_coeffs.data(), static_cast<int>(num_coeffs.size()),
        den_coeffs.data(), static_cast<int>(den_coeffs.size()),
        omegas.data(), static_cast<int>(omegas.size()),
        mag_db.data(), phase_deg.data());
    if (rc != 0) {
        return make_error<std::vector<nimblecas::BodePoint>>(MathError::gpu_error);
    }
    std::vector<nimblecas::BodePoint> out(omegas.size());
    for (std::size_t i = 0; i < omegas.size(); ++i) {
        out[i] = nimblecas::BodePoint{
            .omega = omegas[i],
            .magnitude_db = mag_db[i],
            .phase_deg = phase_deg[i]
        };
    }
    return out;
}

// Frequency-response Nyquist trace H(iω) = re + im·i sampled over an angular frequency grid ω.
//
// HONESTY: NUMERICAL (double complex). Elementwise calculation (one thread per ω),
// so execution order is independent of thread count or block geometry. Results
// match CPU nimblecas::nyquist to ~1e-9 relative (device log10/atan2/hypot vs libm).
// CPU fallback (no device): nimblecas::nyquist(tf, omegas). Fails with domain_error when
// the transfer function denominator is zero, overflow when size exceeds int bounds,
// and gpu_error on CUDA launch failure.
[[nodiscard]] auto nyquist_sweep(const nimblecas::TransferFunction& tf, std::span<const double> omegas)
    -> Result<std::vector<nimblecas::NyquistPoint>> {
    if (tf.denominator().is_zero()) {
        return make_error<std::vector<nimblecas::NyquistPoint>>(MathError::domain_error);
    }
    if (omegas.empty()) {
        return std::vector<nimblecas::NyquistPoint>{};
    }
    const auto num_coeffs_span = tf.numerator().coefficients();
    const auto den_coeffs_span = tf.denominator().coefficients();
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (num_coeffs_span.size() > int_max || den_coeffs_span.size() > int_max ||
        omegas.size() > int_max) {
        return make_error<std::vector<nimblecas::NyquistPoint>>(MathError::overflow);
    }
    // Domain/overflow validated above (before this branch) so accept/reject never depends on device presence.
    if (!available()) {
        return nimblecas::nyquist(tf, omegas);
    }
    std::vector<double> num_coeffs;
    num_coeffs.reserve(num_coeffs_span.size());
    for (const auto& r : num_coeffs_span) {
        num_coeffs.push_back(static_cast<double>(r.numerator()) /
                            static_cast<double>(r.denominator()));
    }
    std::vector<double> den_coeffs;
    den_coeffs.reserve(den_coeffs_span.size());
    for (const auto& r : den_coeffs_span) {
        den_coeffs.push_back(static_cast<double>(r.numerator()) /
                            static_cast<double>(r.denominator()));
    }
    std::vector<double> re(omegas.size());
    std::vector<double> im(omegas.size());
    const int rc = nimblecas_gpu_nyquist_sweep(
        num_coeffs.data(), static_cast<int>(num_coeffs.size()),
        den_coeffs.data(), static_cast<int>(den_coeffs.size()),
        omegas.data(), static_cast<int>(omegas.size()),
        re.data(), im.data());
    if (rc != 0) {
        return make_error<std::vector<nimblecas::NyquistPoint>>(MathError::gpu_error);
    }
    std::vector<nimblecas::NyquistPoint> out(omegas.size());
    for (std::size_t i = 0; i < omegas.size(); ++i) {
        out[i] = nimblecas::NyquistPoint{
            .omega = omegas[i],
            .re = re[i],
            .im = im[i]
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY G — Batched discrete & stationary wavelet transforms (gpu_wavelet_kernels.cu).
// ---------------------------------------------------------------------------

// One-level batch 1D discrete wavelet transform for an arbitrary FilterBank over `batch`
// contiguous signal blocks of `len` samples each (`data` is row-major, so data.size() == batch*len
// and `len` must be even). For each block the result packs its len/2 approximation coefficients
// followed by its len/2 detail coefficients, matching wavelets::dwt layout.
//
// HONESTY: NUMERICAL (double). Bitwise-deterministic run-to-run; matches CPU wavelets::dwt to
// ~1e-12 (same filter FP arithmetic).
// CPU fallback (no device): wavelets::dwt per signal, reassembling the batch layout.
//
// Fails with MathError::domain_error when batch <= 0, len <= 0, len is odd, data.size() != batch*len,
// or filter bank is empty/mismatched; MathError::overflow when a flat size or filter length exceeds
// the int kernel bound; MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto dwt_batch(std::span<const double> data, int batch, int len,
                             const wavelets::FilterBank& fb) -> Result<std::vector<double>> {
    if (batch <= 0 || len <= 0 || (len % 2) != 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (fb.analysis_lo.empty() || fb.analysis_hi.empty() ||
        fb.analysis_lo.size() != fb.analysis_hi.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const auto expected = static_cast<std::size_t>(batch) * static_cast<std::size_t>(len);
    if (data.size() != expected) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (expected > int_max || fb.analysis_lo.size() > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    const int flen = static_cast<int>(fb.analysis_lo.size());
    if (!available()) {
        std::vector<double> out(expected);
        const std::size_t half = static_cast<std::size_t>(len / 2);
        for (int b = 0; b < batch; ++b) {
            const auto sig = data.subspan(static_cast<std::size_t>(b) * static_cast<std::size_t>(len),
                                          static_cast<std::size_t>(len));
            auto res = wavelets::dwt(sig, fb);
            if (!res) {
                return make_error<std::vector<double>>(res.error());
            }
            std::copy(res->approx.begin(), res->approx.end(),
                      out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * len));
            std::copy(res->detail.begin(), res->detail.end(),
                      out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * len + half));
        }
        return out;
    }
    std::vector<double> out(expected);
    const int rc = nimblecas_gpu_dwt_batch(data.data(), batch, len, fb.analysis_lo.data(),
                                           fb.analysis_hi.data(), flen, out.data());
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// Level-1 batch 1D stationary (undecimated / a-trous) wavelet transform for an arbitrary FilterBank
// over `batch` contiguous signal blocks of `len` samples each (`data` is row-major, so data.size() == batch*len).
// For each block the result packs its `len` approximation coefficients followed by its `len` detail
// coefficients (total 2*len per block, so out.size() == batch*2*len), matching wavelets::swt level-1 convention.
//
// HONESTY: NUMERICAL (double). Bitwise-deterministic run-to-run; matches CPU wavelets::swt to
// ~1e-12 (same filter FP arithmetic).
// CPU fallback (no device): wavelets::swt level 1 per signal, reassembling the batch layout.
//
// Fails with MathError::domain_error when batch <= 0, len <= 0, data.size() != batch*len, or
// filter bank is empty/mismatched; MathError::overflow when flat sizes or filter length exceed
// the int kernel bound; MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto swt_batch(std::span<const double> data, int batch, int len,
                             const wavelets::FilterBank& fb) -> Result<std::vector<double>> {
    if (batch <= 0 || len <= 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (fb.analysis_lo.empty() || fb.analysis_hi.empty() ||
        fb.analysis_lo.size() != fb.analysis_hi.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const auto expected_in = static_cast<std::size_t>(batch) * static_cast<std::size_t>(len);
    if (data.size() != expected_in) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const auto expected_out = static_cast<std::size_t>(batch) * static_cast<std::size_t>(2 * len);
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (expected_in > int_max || expected_out > int_max || fb.analysis_lo.size() > int_max) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    const int flen = static_cast<int>(fb.analysis_lo.size());
    if (!available()) {
        std::vector<double> out(expected_out);
        const std::size_t slen = static_cast<std::size_t>(len);
        for (int b = 0; b < batch; ++b) {
            const auto sig = data.subspan(static_cast<std::size_t>(b) * slen, slen);
            auto res = wavelets::swt(sig, fb, 1);
            if (!res) {
                return make_error<std::vector<double>>(res.error());
            }
            std::copy(res->approx.begin(), res->approx.end(),
                      out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * 2 * slen));
            std::copy(res->detail.begin(), res->detail.end(),
                      out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(b) * 2 * slen + slen));
        }
        return out;
    }
    std::vector<double> out(expected_out);
    const int rc = nimblecas_gpu_swt_batch(data.data(), batch, len, fb.analysis_lo.data(),
                                           fb.analysis_hi.data(), flen, out.data());
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY H — Quasi-Monte Carlo primitives (gpu_qmc_kernels.cu).
// ---------------------------------------------------------------------------

namespace detail {
struct GpuSobolSpec {
    std::uint32_t s;
    std::uint32_t a;
    std::array<std::uint32_t, 6> m;
};

inline constexpr std::array<GpuSobolSpec, 7> gpu_sobol_table{{
    {1, 0, {1, 0, 0, 0, 0, 0}},   // dim 2
    {2, 1, {1, 3, 0, 0, 0, 0}},   // dim 3
    {3, 1, {1, 3, 1, 0, 0, 0}},   // dim 4
    {3, 2, {1, 1, 1, 0, 0, 0}},   // dim 5
    {4, 1, {1, 1, 3, 3, 0, 0}},   // dim 6
    {4, 4, {1, 3, 5, 13, 0, 0}},  // dim 7
    {5, 2, {1, 1, 5, 5, 17, 0}},  // dim 8
}};

[[nodiscard]] inline auto build_sobol_directions_table(std::size_t max_dim)
    -> std::vector<unsigned int> {
    std::vector<unsigned int> table(max_dim * 32, 0U);
    for (std::size_t dim = 1; dim <= max_dim; ++dim) {
        std::array<std::uint32_t, 33> v{};
        if (dim == 1) {
            for (std::uint32_t k = 1; k <= 32; ++k) {
                v[k] = 1U << (32 - k);
            }
        } else {
            const GpuSobolSpec& spec = gpu_sobol_table[dim - 2];
            const std::uint32_t s = spec.s;
            for (std::uint32_t k = 1; k <= s; ++k) {
                v[k] = spec.m[k - 1] << (32 - k);
            }
            for (std::uint32_t k = s + 1; k <= 32; ++k) {
                std::uint32_t val = v[k - s] ^ (v[k - s] >> s);
                for (std::uint32_t i = 1; i + 1 <= s; ++i) {
                    const std::uint32_t bit = (spec.a >> (s - 1 - i)) & 1U;
                    val ^= bit * v[k - i];
                }
                v[k] = val;
            }
        }
        for (std::uint32_t k = 1; k <= 32; ++k) {
            table[(dim - 1) * 32 + (k - 1)] = v[k];
        }
    }
    return table;
}

[[nodiscard]] inline auto first_primes_gpu(std::size_t k) -> std::vector<int> {
    std::vector<int> primes;
    primes.reserve(k);
    for (int cand = 2; primes.size() < k; ++cand) {
        bool prime = true;
        for (const int p : primes) {
            if (static_cast<long long>(p) * p > cand) {
                break;
            }
            if (cand % p == 0) {
                prime = false;
                break;
            }
        }
        if (prime) {
            primes.push_back(cand);
        }
    }
    return primes;
}
}  // namespace detail

// L2 star discrepancy of a point set via Warnock's closed form on the GPU.
// Each point must have exactly `dimension` coordinates in [0,1].
//
// HONESTY: NUMERICAL (double) with tree-order last-bit differences vs the CPU serial sum
// (each a valid value). Deterministic run-to-run.
// CPU fallback (no device): nimblecas::l2_star_discrepancy.
//
// Fails with MathError::domain_error on an empty set, dimension == 0, or a point of the wrong size;
// MathError::overflow when N * dimension exceeds the int kernel bound; and MathError::gpu_error
// when a device is present but a CUDA call fails.
[[nodiscard]] auto l2_star_discrepancy(std::span<const std::vector<double>> points,
                                       std::size_t dimension) -> Result<double> {
    const std::size_t N = points.size();
    if (N == 0 || dimension == 0) {
        return make_error<double>(MathError::domain_error);
    }
    for (const auto& p : points) {
        if (p.size() != dimension) {
            return make_error<double>(MathError::domain_error);
        }
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (N > int_max || dimension > int_max || N > int_max / dimension) {
        return make_error<double>(MathError::overflow);
    }
    if (!available()) {
        return nimblecas::l2_star_discrepancy(points, dimension);
    }
    std::vector<double> flat;
    flat.reserve(N * dimension);
    for (const auto& p : points) {
        flat.insert(flat.end(), p.begin(), p.end());
    }
    double out_disc = 0.0;
    const int rc = nimblecas_gpu_l2_star_discrepancy(flat.data(), static_cast<int>(N),
                                                     static_cast<int>(dimension), &out_disc);
    if (rc != 0) {
        return make_error<double>(MathError::gpu_error);
    }
    return out_disc;
}

// Generate `count` Sobol' points in `dimension` dimensions starting at point index `n0` on the GPU.
// Points are row-major packed (count * dimension doubles).
//
// HONESTY: DYADIC-EXACT bit-for-bit double view vs CPU sobol_point for the same index range.
// CPU fallback (no device): nimblecas::sobol_point per index.
//
// Fails with MathError::domain_error when dimension == 0, dimension > 8, or n0 + count > 2^32;
// MathError::overflow when count * dimension exceeds the int kernel bound; and MathError::gpu_error
// when a device is present but a CUDA call fails.
[[nodiscard]] auto sobol_batch(std::uint64_t n0, std::size_t count, std::size_t dimension)
    -> Result<std::vector<double>> {
    if (dimension == 0 || dimension > 8) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (count == 0) {
        return std::vector<double>{};
    }
    if (n0 > 0xFFFFFFFFULL || count > 0xFFFFFFFFULL || n0 + count > 0x100000000ULL) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (count > int_max || dimension > int_max || count > int_max / dimension) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    if (!available()) {
        std::vector<double> out;
        out.reserve(count * dimension);
        for (std::uint64_t i = 0; i < count; ++i) {
            auto pt = nimblecas::sobol_point(n0 + i, dimension);
            if (!pt) {
                return make_error<std::vector<double>>(pt.error());
            }
            out.insert(out.end(), pt->begin(), pt->end());
        }
        return out;
    }
    const auto dir_numbers = detail::build_sobol_directions_table(dimension);
    std::vector<double> out(count * dimension);
    const int rc = nimblecas_gpu_sobol_batch(dir_numbers.data(), 32, n0,
                                             static_cast<int>(count),
                                             static_cast<int>(dimension), out.data());
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// Generate `count` Halton points in `dimension` dimensions starting at point index `n0` on the GPU.
// Points are row-major packed (count * dimension doubles).
//
// HONESTY: NUMERICAL double view matching CPU halton_point to floating-point rounding.
// CPU fallback (no device): nimblecas::halton_point per index.
//
// Fails with MathError::domain_error when dimension == 0; MathError::overflow when count * dimension
// exceeds the int kernel bound; and MathError::gpu_error when a device is present but a CUDA call fails.
[[nodiscard]] auto halton_batch(std::uint64_t n0, std::size_t count, std::size_t dimension)
    -> Result<std::vector<double>> {
    if (dimension == 0) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    if (count == 0) {
        return std::vector<double>{};
    }
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (count > int_max || dimension > int_max || count > int_max / dimension) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    if (!available()) {
        std::vector<double> out;
        out.reserve(count * dimension);
        for (std::uint64_t i = 0; i < count; ++i) {
            auto pt = nimblecas::halton_point(n0 + i, dimension);
            if (!pt) {
                return make_error<std::vector<double>>(pt.error());
            }
            out.insert(out.end(), pt->begin(), pt->end());
        }
        return out;
    }
    const auto primes = detail::first_primes_gpu(dimension);
    std::vector<double> out(count * dimension);
    const int rc = nimblecas_gpu_halton_batch(primes.data(), n0, static_cast<int>(count),
                                              static_cast<int>(dimension), out.data());
    if (rc != 0) {
        return make_error<std::vector<double>>(MathError::gpu_error);
    }
    return out;
}

// ---------------------------------------------------------------------------
// FAMILY I — Batched nonlinear least-squares curve fitting by Levenberg-
// Marquardt (gpu_lmfit_kernels.cu). One CUDA block per problem, the WHOLE LM
// iteration in-kernel — the nonlinear counterpart of batched_cg_csr.
// ---------------------------------------------------------------------------

// The device-expressible parametric model family. The general host-callback LM
// (nlsolve::levenberg_marquardt over an arbitrary ResidualFn) CANNOT run on the
// device — std::function does not cross the bridge — so the GPU path is limited
// to this closed family; anything else stays on the CPU oracle.
enum class FitModel : std::uint8_t {
    polynomial = 0,   // f = sum theta_j t^j        (m = theta0.size() <= 8)
    exponential = 1,  // f = th0*exp(th1*t) + th2   (m = 3)
    gaussian = 2,     // f = th0*exp(-(t-th1)^2/(2 th2^2))   (m = 3)
    logistic = 3,     // f = th0/(1 + exp(-th1*(t - th2)))    (m = 3)
    sinusoid = 4,     // f = th0*sin(th1*t + th2) + th3       (m = 4)
    power_law = 5,    // f = th0 * t^th1 (requires t > 0)     (m = 2)
};

// Hard parameter-count cap (must match kMaxParams in gpu_lmfit_kernels.cu): the
// in-block Cholesky state lives in fixed shared arrays sized by this bound.
inline constexpr int kGpuLmMaxParams = 8;
// Inner damping-loop bound, mirroring the CPU oracle's 40 (nlsolve detail_lm::run).
inline constexpr int kGpuLmMaxInner = 40;

// One curve-fit problem view; all spans must outlive the call. t/y have equal
// length n_k (the points); theta0 has length m_k (the model's parameter count;
// for polynomial, the degree+1). Requires n_k >= m_k (else under-determined).
struct CurveFitProblem {
    FitModel model{FitModel::polynomial};
    std::span<const double> t;       // abscissae, length n_k
    std::span<const double> y;       // observations, length n_k
    std::span<const double> theta0;  // initial parameters, length m_k
};

// Fluent tuning knobs (defaults mirror nlsolve::Options / levenberg_marquardt).
struct LmFitOptions {
    double tol{1e-10};              // stopping tolerance on ||r||_2 AND on ||J^T r||_inf
    int max_iter{100};              // outer LM iterations
    double lambda0{1e-3};           // initial damping
    bool analytic_jacobian{true};   // false -> in-kernel forward FD
    double fd_step{1e-7};           // FD base step (relative-scaled), FD mode only

    [[nodiscard]] auto with_tol(double v) const -> LmFitOptions {
        auto copy = *this;
        copy.tol = v;
        return copy;
    }
    [[nodiscard]] auto with_max_iter(int v) const -> LmFitOptions {
        auto copy = *this;
        copy.max_iter = v;
        return copy;
    }
    [[nodiscard]] auto with_lambda0(double v) const -> LmFitOptions {
        auto copy = *this;
        copy.lambda0 = v;
        return copy;
    }
    [[nodiscard]] auto with_analytic_jacobian(bool v) const -> LmFitOptions {
        auto copy = *this;
        copy.analytic_jacobian = v;
        return copy;
    }
    [[nodiscard]] auto with_fd_step(double v) const -> LmFitOptions {
        auto copy = *this;
        copy.fd_step = v;
        return copy;
    }
};

// Outcome of one fit — the batch mirror of nlsolve::SolveResult.
struct LmFitResult {
    std::vector<double> theta;   // best parameters found (always populated)
    double residual_norm{0.0};   // honest final ||r(theta)||_2, recomputed at exit
    int iterations{0};           // outer LM iterations, counted exactly as the oracle
    bool converged{false};       // did the stopping test pass?
};

namespace detail {

inline auto lm_model_eval(FitModel model, double t, std::span<const double> theta,
                          double& out_f, std::span<double> out_jrow) -> void {
    const std::size_t m = theta.size();
    switch (model) {
        case FitModel::polynomial: {
            double f = 0.0;
            double p = 1.0;
            for (std::size_t j = 0; j < m; ++j) {
                f += theta[j] * p;
                if (!out_jrow.empty()) { out_jrow[j] = p; }
                p *= t;
            }
            out_f = f;
            break;
        }
        case FitModel::exponential: {
            double e = std::exp(theta[1] * t);
            out_f = theta[0] * e + theta[2];
            if (!out_jrow.empty()) {
                out_jrow[0] = e;
                out_jrow[1] = theta[0] * t * e;
                out_jrow[2] = 1.0;
            }
            break;
        }
        case FitModel::gaussian: {
            double u = (t - theta[1]) / theta[2];
            double e = std::exp(-0.5 * u * u);
            out_f = theta[0] * e;
            if (!out_jrow.empty()) {
                out_jrow[0] = e;
                out_jrow[1] = theta[0] * e * u / theta[2];
                out_jrow[2] = theta[0] * e * u * u / theta[2];
            }
            break;
        }
        case FitModel::logistic: {
            double s = 1.0 / (1.0 + std::exp(-theta[1] * (t - theta[2])));
            out_f = theta[0] * s;
            if (!out_jrow.empty()) {
                out_jrow[0] = s;
                out_jrow[1] = theta[0] * s * (1.0 - s) * (t - theta[2]);
                out_jrow[2] = -theta[0] * s * (1.0 - s) * theta[1];
            }
            break;
        }
        case FitModel::sinusoid: {
            double arg = theta[1] * t + theta[2];
            double s = std::sin(arg);
            double c = std::cos(arg);
            out_f = theta[0] * s + theta[3];
            if (!out_jrow.empty()) {
                out_jrow[0] = s;
                out_jrow[1] = theta[0] * t * c;
                out_jrow[2] = theta[0] * c;
                out_jrow[3] = 1.0;
            }
            break;
        }
        case FitModel::power_law: {
            double p = std::pow(t, theta[1]);
            out_f = theta[0] * p;
            if (!out_jrow.empty()) {
                out_jrow[0] = p;
                out_jrow[1] = theta[0] * p * std::log(t);
            }
            break;
        }
    }
}

}  // namespace detail

// Batched Levenberg-Marquardt curve fitting over the device-expressible FitModel
// family — the batch mirror of nlsolve::levenberg_marquardt, which remains
// authoritative. One CUDA block per problem runs the ENTIRE LM trust-region loop
// in-kernel (residual + Jacobian evaluation, J^T J + lambda*diag normal equations
// via in-block Cholesky, strict-decrease accept/reject with lambda*4 / lambda/3
// damping), mirroring the CPU oracle's control flow decision-for-decision.
//
// HONESTY (Rule 32): a NUMERICAL (double) LOCAL method. LM finds a STATIONARY
// point of ||r||^2, which need not be a global minimiser and need not fit the
// data well; the result depends on theta0. converged == false is a legitimate
// OUTCOME (budget exhausted, damping blow-up lambda > 1e18, rank-deficient
// J^T J that damping cannot rescue), never an error; the best iterate found is
// returned with its HONEST final residual_norm, recomputed in-kernel from the
// returned theta. One problem's breakdown never affects the others. The general
// host-callback LM (arbitrary ResidualFn) CANNOT run on the device and stays on
// the CPU oracle — this entry point covers only the closed FitModel family.
// DETERMINISM: fixed-shape in-block reductions make each problem's result a pure
// function of its own inputs — bitwise repeatable run-to-run and independent of
// batch composition — but NOT bit-for-bit vs the CPU oracle: threshold decisions
// (accept/reject, convergence) can flip on last-bit differences, so GPU and CPU
// may take different iterate paths to the same stationary point. Validated
// agreement is on the DESTINATION (final cost/parameters, §tests), not the path.
// Fails with domain_error on structural/model/finiteness/start-point violations
// (decided before the available() branch, identically with or without a device);
// overflow when a concatenated size exceeds the int kernel bound; gpu_error only
// when a device is present and a CUDA call fails. CPU fallback (no device):
// nlsolve::levenberg_marquardt per problem over the same model family.
[[nodiscard]] auto batched_curve_fit_lm(std::span<const CurveFitProblem> problems,
                                        const LmFitOptions& opts = {})
    -> Result<std::vector<LmFitResult>> {
    if (!(opts.tol >= 0.0) || !std::isfinite(opts.tol)) {
        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
    }
    if (opts.max_iter < 0) {
        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
    }
    if (!std::isfinite(opts.lambda0)) {
        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
    }
    if (!opts.analytic_jacobian && !(opts.fd_step > 0.0)) {
        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
    }

    if (problems.empty()) {
        return std::vector<LmFitResult>{};
    }

    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (problems.size() > int_max) {
        return make_error<std::vector<LmFitResult>>(MathError::overflow);
    }

    std::size_t total_n = 0;
    std::size_t total_th = 0;
    std::size_t total_jac = 0;

    for (const auto& p : problems) {
        const std::size_t n_k = p.t.size();
        const std::size_t m_k = p.theta0.size();

        // Size/overflow guard FIRST: a span whose element count exceeds the int
        // kernel bound must be reported as overflow before ANY element is read
        // (the finiteness/model loops below iterate over p.t / p.y / p.theta0).
        if (n_k > int_max || m_k > int_max || p.y.size() > int_max) {
            return make_error<std::vector<LmFitResult>>(MathError::overflow);
        }

        if (p.t.size() != p.y.size() || n_k == 0 || m_k == 0 ||
            m_k > static_cast<std::size_t>(kGpuLmMaxParams)) {
            return make_error<std::vector<LmFitResult>>(MathError::domain_error);
        }

        switch (p.model) {
            case FitModel::exponential:
            case FitModel::gaussian:
            case FitModel::logistic:
                if (m_k != 3) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
                break;
            case FitModel::sinusoid:
                if (m_k != 4) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
                break;
            case FitModel::power_law:
                if (m_k != 2) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
                break;
            case FitModel::polynomial:
                break;
        }

        if (n_k < m_k) {
            return make_error<std::vector<LmFitResult>>(MathError::domain_error);
        }

        for (double tv : p.t) {
            if (!std::isfinite(tv)) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
        }
        for (double yv : p.y) {
            if (!std::isfinite(yv)) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
        }
        for (double thv : p.theta0) {
            if (!std::isfinite(thv)) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
        }

        if (p.model == FitModel::power_law) {
            for (double tv : p.t) {
                if (tv <= 0.0) return make_error<std::vector<LmFitResult>>(MathError::domain_error);
            }
        }

        std::array<double, kGpuLmMaxParams> j_buf{};
        std::span<double> j_span = opts.analytic_jacobian
                                       ? std::span<double>{j_buf.data(), m_k}
                                       : std::span<double>{};
        std::array<double, kGpuLmMaxParams> th_pert{};
        for (std::size_t i = 0; i < n_k; ++i) {
            double f_val = 0.0;
            detail::lm_model_eval(p.model, p.t[i], p.theta0, f_val, j_span);
            if (!std::isfinite(f_val)) {
                return make_error<std::vector<LmFitResult>>(MathError::domain_error);
            }
            if (opts.analytic_jacobian) {
                for (std::size_t j = 0; j < m_k; ++j) {
                    if (!std::isfinite(j_buf[j])) {
                        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
                    }
                }
            } else {
                // FD mode: host-validate the forward-difference Jacobian at theta0 with
                // the SAME h_j = fd_step*(1+|theta_j|) recipe the kernel uses, so the GPU
                // path and the CPU fallback agree on start-point admissibility (this is
                // decided before the available() branch). A finite r(theta0) with a
                // non-finite forward-FD Jacobian would otherwise diverge: the GPU returns
                // {iters=0, converged=false} while the CPU fallback's fd_jacobian fails at
                // it==0 and returns domain_error for the whole batch. m extra model evals
                // per point.
                for (std::size_t j = 0; j < m_k; ++j) {
                    const double h_j = opts.fd_step * (1.0 + std::abs(p.theta0[j]));
                    for (std::size_t q = 0; q < m_k; ++q) {
                        th_pert[q] = p.theta0[q];
                    }
                    th_pert[j] += h_j;
                    double f_pert = 0.0;
                    detail::lm_model_eval(p.model, p.t[i],
                                          std::span<const double>{th_pert.data(), m_k}, f_pert,
                                          std::span<double>{});
                    const double fd = (f_pert - f_val) / h_j;
                    if (!std::isfinite(fd)) {
                        return make_error<std::vector<LmFitResult>>(MathError::domain_error);
                    }
                }
            }
        }

        // n_k <= int_max already proven above; guard only the running total.
        if (total_n > int_max - n_k) return make_error<std::vector<LmFitResult>>(MathError::overflow);
        total_n += n_k;

        if (total_th > int_max - m_k) return make_error<std::vector<LmFitResult>>(MathError::overflow);
        total_th += m_k;

        const std::size_t jac_k = n_k * m_k;
        if (total_jac > int_max - jac_k) return make_error<std::vector<LmFitResult>>(MathError::overflow);
        total_jac += jac_k;
    }

    if (!available()) {
        std::vector<LmFitResult> out;
        out.reserve(problems.size());
        for (const auto& p : problems) {
            nlsolve::ResidualFn F = [&p](std::span<const double> th) -> std::vector<double> {
                std::vector<double> r(p.t.size());
                for (std::size_t i = 0; i < p.t.size(); ++i) {
                    double f_val = 0.0;
                    detail::lm_model_eval(p.model, p.t[i], th, f_val, std::span<double>{});
                    r[i] = f_val - p.y[i];
                }
                return r;
            };
            nlsolve::Options o{};
            o.tol = opts.tol;
            o.max_iter = static_cast<std::size_t>(opts.max_iter);
            o.fd_step = opts.fd_step;
            Result<nlsolve::SolveResult> r;
            if (opts.analytic_jacobian) {
                nlsolve::JacobianFn J = [&p](std::span<const double> th) -> std::vector<double> {
                    const std::size_t n = p.t.size();
                    const std::size_t m = p.theta0.size();
                    std::vector<double> j_flat(n * m);
                    std::array<double, kGpuLmMaxParams> j_row{};
                    for (std::size_t i = 0; i < n; ++i) {
                        double f_val = 0.0;
                        detail::lm_model_eval(p.model, p.t[i], th, f_val,
                                              std::span<double>{j_row.data(), m});
                        for (std::size_t j = 0; j < m; ++j) {
                            j_flat[i * m + j] = j_row[j];
                        }
                    }
                    return j_flat;
                };
                r = nlsolve::levenberg_marquardt(F, J, p.theta0, o, opts.lambda0);
            } else {
                r = nlsolve::levenberg_marquardt(F, p.theta0, o, opts.lambda0);
            }
            if (!r) {
                return make_error<std::vector<LmFitResult>>(r.error());
            }
            out.push_back(LmFitResult{
                .theta = std::move(r->x),
                .residual_norm = r->residual_norm,
                .iterations = static_cast<int>(r->iterations),
                .converged = r->converged
            });
        }
        return out;
    }

    const int num_problems = static_cast<int>(problems.size());
    std::vector<int> model_cat(num_problems);
    std::vector<double> t_cat;
    t_cat.reserve(total_n);
    std::vector<double> y_cat;
    y_cat.reserve(total_n);
    std::vector<int> pt_off(num_problems + 1, 0);

    std::vector<double> theta_cat;
    theta_cat.reserve(total_th);
    std::vector<int> th_off(num_problems + 1, 0);

    std::vector<int> jac_off(num_problems + 1, 0);

    for (int k = 0; k < num_problems; ++k) {
        const auto& p = problems[k];
        model_cat[k] = static_cast<int>(p.model);

        pt_off[k] = static_cast<int>(t_cat.size());
        t_cat.insert(t_cat.end(), p.t.begin(), p.t.end());
        y_cat.insert(y_cat.end(), p.y.begin(), p.y.end());

        th_off[k] = static_cast<int>(theta_cat.size());
        theta_cat.insert(theta_cat.end(), p.theta0.begin(), p.theta0.end());

        jac_off[k] = (k == 0) ? 0 : jac_off[k - 1] + (pt_off[k] - pt_off[k - 1]) * (th_off[k] - th_off[k - 1]);
    }
    pt_off[num_problems] = static_cast<int>(t_cat.size());
    th_off[num_problems] = static_cast<int>(theta_cat.size());
    jac_off[num_problems] = jac_off[num_problems - 1] +
        (pt_off[num_problems] - pt_off[num_problems - 1]) * (th_off[num_problems] - th_off[num_problems - 1]);

    std::vector<int> out_iters(num_problems);
    std::vector<int> out_converged(num_problems);
    std::vector<double> out_resid(num_problems);

    const int use_fd = opts.analytic_jacobian ? 0 : 1;
    const int rc = nimblecas_gpu_batched_lm_curvefit(
        model_cat.data(), t_cat.data(), y_cat.data(), pt_off.data(), theta_cat.data(),
        th_off.data(), jac_off.data(), num_problems, opts.max_iter, opts.tol, opts.lambda0,
        use_fd, opts.fd_step, out_iters.data(), out_converged.data(), out_resid.data());

    if (rc != 0) {
        return make_error<std::vector<LmFitResult>>(MathError::gpu_error);
    }

    std::vector<LmFitResult> results;
    results.reserve(num_problems);
    for (int k = 0; k < num_problems; ++k) {
        int start_th = th_off[k];
        int end_th = th_off[k + 1];
        std::vector<double> th_k(theta_cat.begin() + start_th, theta_cat.begin() + end_th);
        results.push_back(LmFitResult{
            .theta = std::move(th_k),
            .residual_norm = out_resid[k],
            .iterations = out_iters[k],
            .converged = out_converged[k] != 0
        });
    }
    return results;
}

// Alias for public module requirement: batched_lm_curvefit
[[nodiscard]] inline auto batched_lm_curvefit(std::span<const CurveFitProblem> problems,
                                             const LmFitOptions& opts = {})
    -> Result<std::vector<LmFitResult>> {
    return batched_curve_fit_lm(problems, opts);
}

}  // namespace nimblecas::gpu



