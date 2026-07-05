// Plain C ABI bridge to the NimbleCAS CUDA GPU kernels (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// This header is included both by gpu_kernels.cu (compiled by nvcc) and in the global
// module fragment of nimblecas.gpu (compiled by clang + libc++). It deliberately exposes
// only a plain C interface over POD types, so no C++ standard-library object crosses the
// nvcc <-> clang/libc++ boundary and the two objects link cleanly.

#ifndef NIMBLECAS_GPU_BRIDGE_H
#define NIMBLECAS_GPU_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Number of CUDA-capable devices (0 if none or CUDA is unavailable).
int nimblecas_gpu_device_count(void);

// Evaluate the polynomial coeffs[0..n_coeffs-1] (low degree first) at each of the n
// points x[0..n-1], writing p(x_i) to out[i]. Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_poly_eval(const double* coeffs, int n_coeffs,
                            const double* x, double* out, int n);

// Batched Levenshtein edit distance, one thread per (a_i, b_i) pair. The sequences are stored
// flattened as integer code points in a_flat / b_flat; a_off / b_off are prefix-offset arrays
// of length pairs+1 so that pair i spans [a_off[i], a_off[i+1]) and [b_off[i], b_off[i+1]).
// Writes the distance of pair i to out[i]. Returns 0 on success, or a non-zero CUDA error
// code on failure.
int nimblecas_gpu_edit_distance_batch(const int* a_flat, const int* a_off, const int* b_flat,
                                      const int* b_off, int pairs, int* out);

// Level-synchronous single-source BFS over a CSR graph (row_offsets length num_vertices+1,
// col_indices length num_edges). Fills dist[0..num_vertices-1] with the hop distance from
// source (unreachable vertices remain -1). Returns 0 on success, or a non-zero CUDA error
// code on failure.
int nimblecas_gpu_bfs(const int* row_offsets, const int* col_indices, int num_vertices,
                      int num_edges, int source, int* dist);

// Count the solutions to the n-queens problem on the GPU, one thread per partial placement of
// the first row(s), each thread completing its subtree with the classic three-bitmask method.
// Writes the total solution count to *out_count. Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_nqueens_count(int n, unsigned long long* out_count);

// QMC integration reduction. Horner-evaluate the polynomial integrand coeffs[0..n_coeffs-1]
// (low degree first) at each of the n sample points x[0..n-1] on the device and reduce (sum),
// writing the equal-weight average (1/n) * sum p(x_i) to *out_mean. This is the device analogue
// of the numerical qmc_integrate: the caller supplies the (low-discrepancy) sample points. The
// device sums in block/tree order, which differs from a strict left-to-right CPU sum, so the
// last bits of *out_mean may differ from a CPU average (each is a valid estimate). Returns 0 on
// success, or a non-zero CUDA error code on failure.
int nimblecas_gpu_qmc_poly_integrate(const double* coeffs, int n_coeffs, const double* x, int n,
                                     double* out_mean);

// One-level batch Haar discrete wavelet transform (orthonormal 1/sqrt(2) normalization), one
// thread per output coefficient pair, grid-stride over the batch. `data` holds `batch` contiguous
// signal blocks of `len` samples each (len must be even), row-major. For each block, writes its
// len/2 approximation coefficients followed by its len/2 detail coefficients to the matching
// block of `out` (same size and layout as the input). Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_haar_dwt_batch(const double* data, int batch, int len, double* out);

// Batched dense double matrix multiply, one thread per output element. `a` holds `batch`
// contiguous row-major A blocks (each m x k) and `b` holds `batch` contiguous row-major B blocks
// (each k x n); writes the `batch` row-major products C_b = A_b * B_b (each m x n) to `c`, in the
// same contiguous block layout. Returns 0 on success, or a non-zero CUDA error code on failure.
int nimblecas_gpu_batched_matmul(const double* a, const double* b, double* c, int batch, int m,
                                 int k, int n);

// Batched radix-2 forward FFT (Cooley-Tukey, decimation-in-time), one CUDA block per signal.
// `in` holds `batch` complex signals of length `n` (n a power of two), each stored as 2*n
// interleaved doubles (re, im, re, im, ...) contiguously; writes the DFT of each signal
// X_k = sum_j x_j e^{-2*pi*i*k*j/n} (FORWARD, negative exponent) to `out` in the same interleaved
// layout. n must be a power of two with 1 <= n <= 2048 (the shared-memory length cap). Returns 0
// on success, or a non-zero CUDA error code on failure (cudaErrorInvalidValue for a bad shape).
int nimblecas_gpu_fft_batch(const double* in, double* out, int batch, int n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NIMBLECAS_GPU_BRIDGE_H
