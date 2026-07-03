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
//
// Grid-stride loop: each thread walks the point array in steps of the total grid size, so a
// fixed, device-sized grid saturates every SM regardless of n and no thread sits idle on
// small n. The coefficient array is tiny and hot, so it stays in L2/registers across the
// stride; the per-point cost is dominated by the streaming loads of x and stores of out.
__global__ void poly_eval_kernel(const double* __restrict__ coeffs, int n_coeffs,
                                 const double* __restrict__ x, double* __restrict__ out,
                                 int n) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         i < n; i += stride) {
        const double xi = x[i];
        double acc = 0.0;
        for (int k = n_coeffs - 1; k >= 0; --k) {
            acc = acc * xi + coeffs[k];
        }
        out[i] = acc;
    }
}

// RAII-free helper: register a host buffer for pinned DMA when it is large enough to amortize
// the fixed registration cost, falling back silently to pageable transfers on failure. A
// non-null returned handle must be paired with host_unregister once the copies complete.
struct PinnedScope {
    const void* ptr;
    bool registered;
};

// Registration has a fixed per-call cost, so only pin buffers past this threshold.
constexpr size_t kPinThresholdBytes = 256u * 1024u;

PinnedScope host_register(const void* ptr, size_t bytes) {
    PinnedScope scope{ptr, false};
    if (ptr != nullptr && bytes >= kPinThresholdBytes) {
        // A failed registration is not fatal: clear the sticky error and use pageable DMA.
        if (cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault) ==
            cudaSuccess) {
            scope.registered = true;
        } else {
            cudaGetLastError();
        }
    }
    return scope;
}

void host_unregister(const PinnedScope& scope) {
    if (scope.registered) {
        cudaHostUnregister(const_cast<void*>(scope.ptr));
        cudaGetLastError();  // swallow any unregister error; transfers already completed
    }
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
    // Pin the caller's pageable buffers so the H2D/D2H copies use DMA instead of a staging
    // bounce. Registration is size-gated and best-effort; a failure just leaves the buffer
    // pageable. The out buffer is pinned before it is written back on D2H.
    PinnedScope pin_coeffs = host_register(coeffs, coeff_bytes);
    PinnedScope pin_x = host_register(x, point_bytes);
    PinnedScope pin_out = host_register(out, point_bytes);
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
        // Launch a device-sized grid (a small multiple of the SM count) rather than one block
        // per point, so occupancy stays high without spawning excessive blocks on large n.
        int sm_count = 0;
        int blocks_by_n = (n + threads - 1) / threads;
        int blocks;
        if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) ==
                cudaSuccess &&
            sm_count > 0) {
            const int grid_cap = sm_count * 32;
            blocks = grid_cap < blocks_by_n ? grid_cap : blocks_by_n;
        } else {
            cudaGetLastError();  // attribute query failed; fall back to one block per point
            blocks = blocks_by_n;
        }
        if (blocks < 1) {
            blocks = 1;
        }
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
    host_unregister(pin_out);
    host_unregister(pin_x);
    host_unregister(pin_coeffs);
    return rc;
}
