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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NIMBLECAS_GPU_BRIDGE_H
