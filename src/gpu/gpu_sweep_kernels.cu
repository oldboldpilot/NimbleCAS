// NimbleCAS CUDA GPU kernels — strategy payoff & P&L grid sweeps (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and
// linked into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code
// here uses ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object
// carries no libstdc++ dependency and links cleanly with the clang/libc++ engine.

#include <cuda_runtime.h>

#include "gpu_bridge.h"

namespace {

int choose_blocks(int work, int threads) {
    // 64-bit intermediate: work can be up to INT_MAX, so (work + threads - 1) would overflow
    // signed int (UB) before the divide. Query the CURRENT device (not a hardcoded 0) for its
    // SM count — geometry only, never affects results.
    const int blocks_by_work =
        static_cast<int>((static_cast<long long>(work) + threads - 1) / threads);
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

__global__ void strategy_grid_kernel(const NimblecasSweepLeg* __restrict__ legs, int n_legs,
                                     const double* __restrict__ grid, int n_grid,
                                     int net_of_premium, double* __restrict__ out) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < n_grid; j += stride) {
        const double s = grid[j];
        double payoff = 0.0;
        double cost = 0.0;
        for (int i = 0; i < n_legs; ++i) {
            const NimblecasSweepLeg l = legs[i];
            double tv;
            if (l.right == 0) {
                tv = fmax(__dsub_rn(s, l.strike), 0.0);
            } else if (l.right == 1) {
                tv = fmax(__dsub_rn(l.strike, s), 0.0);
            } else {
                tv = s;
            }
            payoff = __dadd_rn(payoff, __dmul_rn(l.quantity, tv));
            cost = __dadd_rn(cost, __dmul_rn(l.quantity, l.premium));
        }
        out[j] = (net_of_premium != 0) ? __dsub_rn(payoff, cost) : payoff;
    }
}

__global__ void futures_grid_kernel(const NimblecasFuturesSweepLeg* __restrict__ legs,
                                    int n_legs, const double* __restrict__ grid, int n_grid,
                                    double* __restrict__ out) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < n_grid; j += stride) {
        const double s = grid[j];
        double total = 0.0;
        for (int i = 0; i < n_legs; ++i) {
            const NimblecasFuturesSweepLeg l = legs[i];
            total = __dadd_rn(total, __dmul_rn(l.scaled_quantity, __dsub_rn(s, l.entry_price)));
        }
        out[j] = total;
    }
}

}  // namespace

extern "C" int nimblecas_gpu_strategy_grid(const NimblecasSweepLeg* legs, int n_legs,
                                        const double* grid, int n_grid, int net_of_premium,
                                        double* out) {
    if (n_grid <= 0) {
        return 0;
    }
    const size_t legs_bytes = static_cast<size_t>(n_legs) * sizeof(NimblecasSweepLeg);
    const size_t grid_bytes = static_cast<size_t>(n_grid) * sizeof(double);
    const size_t legs_alloc = legs_bytes != 0 ? legs_bytes : sizeof(NimblecasSweepLeg);

    NimblecasSweepLeg* dev_legs = nullptr;
    double* dev_grid = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_legs, legs_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_grid, grid_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, grid_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_legs > 0 &&
               (err = cudaMemcpy(dev_legs, legs, legs_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_grid, grid, grid_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n_grid, threads);
        strategy_grid_kernel<<<blocks, threads>>>(dev_legs, n_legs, dev_grid, n_grid,
                                                 net_of_premium, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, grid_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dev_legs != nullptr) {
        cudaFree(dev_legs);
    }
    if (dev_grid != nullptr) {
        cudaFree(dev_grid);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    return rc;
}

extern "C" int nimblecas_gpu_futures_grid(const NimblecasFuturesSweepLeg* legs, int n_legs,
                                       const double* grid, int n_grid, double* out) {
    if (n_grid <= 0) {
        return 0;
    }
    const size_t legs_bytes = static_cast<size_t>(n_legs) * sizeof(NimblecasFuturesSweepLeg);
    const size_t grid_bytes = static_cast<size_t>(n_grid) * sizeof(double);
    const size_t legs_alloc = legs_bytes != 0 ? legs_bytes : sizeof(NimblecasFuturesSweepLeg);

    NimblecasFuturesSweepLeg* dev_legs = nullptr;
    double* dev_grid = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_legs, legs_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_grid, grid_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, grid_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_legs > 0 &&
               (err = cudaMemcpy(dev_legs, legs, legs_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_grid, grid, grid_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n_grid, threads);
        futures_grid_kernel<<<blocks, threads>>>(dev_legs, n_legs, dev_grid, n_grid, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, grid_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dev_legs != nullptr) {
        cudaFree(dev_legs);
    }
    if (dev_grid != nullptr) {
        cudaFree(dev_grid);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    return rc;
}
