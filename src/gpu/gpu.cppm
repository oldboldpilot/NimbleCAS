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
// batch-solve MIRROR of the authoritative CPU nimblecas::krylov::cg.
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
// on the device and are unaffected; the CPU nimblecas::krylov::cg remains authoritative.
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
    if (b.empty() || row_offsets.size() != b.size() + 1 || col_indices.size() != values.size()) {
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
// FAMILY A — Batched derivative pricing (gpu_pricing_kernels.cu).
//
// Every entry point here is a batch MIRROR of the authoritative CPU nimblecas.pricing
// implementation, never a second source of truth. NEW FALLBACK CONTRACT (differs from the
// entries above): when NO device is present these functions compute the result on the CPU
// via nimblecas.pricing and return real values — an honest fallback, not gpu_error. A CUDA
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

}  // namespace nimblecas::gpu

