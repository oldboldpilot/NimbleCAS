// NimbleCAS CUDA GPU kernels — batched wavelet transforms (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and
// linked into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code
// here uses ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object
// carries no libstdc++ dependency and links cleanly with the clang/libc++ engine.

#include <cuda_runtime.h>

#include "gpu_bridge.h"

namespace {

struct PinnedScope {
    const void* ptr;
    bool registered;
};

constexpr size_t kPinThresholdBytes = 256u * 1024u;

PinnedScope host_register(const void* ptr, size_t bytes) {
    PinnedScope scope{ptr, false};
    if (ptr != nullptr && bytes >= kPinThresholdBytes) {
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
        cudaGetLastError();
    }
}

int choose_blocks(long long work, int threads) {
    const int blocks_by_work = static_cast<int>((work + threads - 1) / threads);
    int sm_count = 0;
    int blocks;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        sm_count > 0) {
        const int grid_cap = sm_count * 32;
        blocks = grid_cap < blocks_by_work ? grid_cap : blocks_by_work;
    } else {
        cudaGetLastError();
        blocks = blocks_by_work;
    }
    if (blocks < 1) {
        blocks = 1;
    }
    return blocks;
}

// Batched 1D DWT kernel (periodic convolution + decimation by 2).
// Each thread handles one output pair (approximation and detail coefficient at decimation index i)
// for signal `blk` in the batch.
__global__ void dwt_batch_kernel(const double* __restrict__ in, int batch, int len, int half,
                                 const double* __restrict__ lo, const double* __restrict__ hi,
                                 int flen, double* __restrict__ out) {
    const long long total_pairs = static_cast<long long>(batch) * static_cast<long long>(half);
    const long long stride = static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    for (long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
                         static_cast<long long>(threadIdx.x);
         idx < total_pairs; idx += stride) {
        const long long blk = idx / half;
        const int i = static_cast<int>(idx - blk * half);
        const long long base_in = blk * len;
        const long long base_out = blk * len;
        double sa = 0.0;
        double sd = 0.0;
        for (int k = 0; k < flen; ++k) {
            const int sample_idx = static_cast<int>((2LL * i + k) % len);
            const double x = in[base_in + sample_idx];
            sa = fma(lo[k], x, sa);
            sd = fma(hi[k], x, sd);
        }
        out[base_out + i] = sa;
        out[base_out + half + i] = sd;
    }
}

// Batched 1D SWT (stationary/undecimated wavelet transform, level 1).
// Each thread handles one sample position `i` for signal `blk` in the batch,
// computing both the approximation and detail coefficient at position i.
__global__ void swt_batch_kernel(const double* __restrict__ in, int batch, int len,
                                 const double* __restrict__ lo, const double* __restrict__ hi,
                                 int flen, double* __restrict__ out) {
    const long long total = static_cast<long long>(batch) * static_cast<long long>(len);
    const long long stride = static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    for (long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
                         static_cast<long long>(threadIdx.x);
         idx < total; idx += stride) {
        const long long blk = idx / len;
        const int i = static_cast<int>(idx - blk * len);
        const long long base_in = blk * len;
        const long long base_out = blk * 2 * len;
        double sa = 0.0;
        double sd = 0.0;
        for (int k = 0; k < flen; ++k) {
            const int sample_idx = static_cast<int>((static_cast<long long>(i) + k) % len);
            const double x = in[base_in + sample_idx];
            sa = fma(lo[k], x, sa);
            sd = fma(hi[k], x, sd);
        }
        out[base_out + i] = sa;
        out[base_out + len + i] = sd;
    }
}

}  // namespace

extern "C" int nimblecas_gpu_dwt_batch(const double* data, int batch, int len, const double* lo,
                                       const double* hi, int flen, double* out) {
    if (batch <= 0 || len <= 0 || flen <= 0) {
        return 0;
    }
    if ((len & 1) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (data == nullptr || lo == nullptr || hi == nullptr || out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int half = len / 2;
    const size_t total_in_bytes =
        static_cast<size_t>(batch) * static_cast<size_t>(len) * sizeof(double);
    const size_t total_out_bytes = total_in_bytes;
    const size_t filter_bytes = static_cast<size_t>(flen) * sizeof(double);

    double* dev_in = nullptr;
    double* dev_out = nullptr;
    double* dev_lo = nullptr;
    double* dev_hi = nullptr;

    cudaError_t err = cudaSuccess;
    int rc = 0;

    PinnedScope pin_in = host_register(data, total_in_bytes);
    PinnedScope pin_out = host_register(out, total_out_bytes);
    PinnedScope pin_lo = host_register(lo, filter_bytes);
    PinnedScope pin_hi = host_register(hi, filter_bytes);

    if ((err = cudaMalloc(&dev_in, total_in_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, total_out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_lo, filter_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_hi, filter_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_in, data, total_in_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_lo, lo, filter_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_hi, hi, filter_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(static_cast<long long>(batch) * half, threads);
        dwt_batch_kernel<<<blocks, threads>>>(dev_in, batch, len, half, dev_lo, dev_hi, flen, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, total_out_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_in != nullptr) {
        cudaFree(dev_in);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    if (dev_lo != nullptr) {
        cudaFree(dev_lo);
    }
    if (dev_hi != nullptr) {
        cudaFree(dev_hi);
    }

    host_unregister(pin_hi);
    host_unregister(pin_lo);
    host_unregister(pin_out);
    host_unregister(pin_in);

    return rc;
}

extern "C" int nimblecas_gpu_swt_batch(const double* data, int batch, int len, const double* lo,
                                       const double* hi, int flen, double* out) {
    if (batch <= 0 || len <= 0 || flen <= 0) {
        return 0;
    }
    if (data == nullptr || lo == nullptr || hi == nullptr || out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t total_in_bytes =
        static_cast<size_t>(batch) * static_cast<size_t>(len) * sizeof(double);
    const size_t total_out_bytes =
        static_cast<size_t>(batch) * static_cast<size_t>(len) * 2u * sizeof(double);
    const size_t filter_bytes = static_cast<size_t>(flen) * sizeof(double);

    double* dev_in = nullptr;
    double* dev_out = nullptr;
    double* dev_lo = nullptr;
    double* dev_hi = nullptr;

    cudaError_t err = cudaSuccess;
    int rc = 0;

    PinnedScope pin_in = host_register(data, total_in_bytes);
    PinnedScope pin_out = host_register(out, total_out_bytes);
    PinnedScope pin_lo = host_register(lo, filter_bytes);
    PinnedScope pin_hi = host_register(hi, filter_bytes);

    if ((err = cudaMalloc(&dev_in, total_in_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, total_out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_lo, filter_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_hi, filter_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_in, data, total_in_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_lo, lo, filter_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_hi, hi, filter_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(static_cast<long long>(batch) * len, threads);
        swt_batch_kernel<<<blocks, threads>>>(dev_in, batch, len, dev_lo, dev_hi, flen, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, total_out_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_in != nullptr) {
        cudaFree(dev_in);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    if (dev_lo != nullptr) {
        cudaFree(dev_lo);
    }
    if (dev_hi != nullptr) {
        cudaFree(dev_hi);
    }

    host_unregister(pin_hi);
    host_unregister(pin_lo);
    host_unregister(pin_out);
    host_unregister(pin_in);

    return rc;
}
