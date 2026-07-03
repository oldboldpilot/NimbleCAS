// NimbleCAS CUDA GPU kernels — batch polynomial evaluation (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and
// linked into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code
// here uses ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object
// carries no libstdc++ dependency and links cleanly with the clang/libc++ engine.

#include <cuda_runtime.h>

#include "gpu_bridge.h"

namespace {

// Horner evaluation of a polynomial coeffs[0..n_coeffs-1] (low degree first) at x[i].
__global__ void poly_eval_kernel(const double* coeffs, int n_coeffs, const double* x,
                                 double* out, int n) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i >= n) {
        return;
    }
    const double xi = x[i];
    double acc = 0.0;
    for (int k = n_coeffs - 1; k >= 0; --k) {
        acc = acc * xi + coeffs[k];
    }
    out[i] = acc;
}

}  // namespace

extern "C" int nimblecas_gpu_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

extern "C" int nimblecas_gpu_poly_eval(const double* coeffs, int n_coeffs, const double* x,
                                       double* out, int n) {
    if (n <= 0) {
        return 0;
    }
    if (n_coeffs <= 0) {
        for (int i = 0; i < n; ++i) {
            out[i] = 0.0;  // the zero polynomial evaluates to 0 everywhere
        }
        return 0;
    }
    const size_t coeff_bytes = static_cast<size_t>(n_coeffs) * sizeof(double);
    const size_t point_bytes = static_cast<size_t>(n) * sizeof(double);
    double* dev_coeffs = nullptr;
    double* dev_x = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    if ((err = cudaMalloc(&dev_coeffs, coeff_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_x, point_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, point_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_coeffs, coeffs, coeff_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_x, x, point_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        poly_eval_kernel<<<blocks, threads>>>(dev_coeffs, n_coeffs, dev_x, dev_out, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, point_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dev_coeffs != nullptr) {
        cudaFree(dev_coeffs);
    }
    if (dev_x != nullptr) {
        cudaFree(dev_x);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    return rc;
}
