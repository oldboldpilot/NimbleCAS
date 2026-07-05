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

export namespace nimblecas::gpu {

// Number of CUDA-capable devices detected (0 when no GPU / CUDA runtime is present).
[[nodiscard]] auto device_count() -> int { return nimblecas_gpu_device_count(); }

// Whether at least one GPU is available for computation.
[[nodiscard]] auto available() -> bool { return device_count() > 0; }

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

}  // namespace nimblecas::gpu
