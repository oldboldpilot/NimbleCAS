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

}  // namespace nimblecas::gpu
