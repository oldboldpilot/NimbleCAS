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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NIMBLECAS_GPU_BRIDGE_H
