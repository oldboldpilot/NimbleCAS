// NimbleCAS CUDA GPU kernels — quasi-Monte Carlo primitives (ROADMAP 5 / 7.8).
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and
// linked into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code
// here uses ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object
// carries no libstdc++ dependency and links cleanly with the clang/libc++ engine.

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_bridge.h"

namespace {

// RAII-free helper: register a host buffer for pinned DMA when it is large enough to amortize
// the fixed registration cost, falling back silently to pageable transfers on failure.
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

// ---------------------------------------------------------------------------
// Warnock L2 star discrepancy reduction kernels.
//
// Warnock closed form: D2*^2 = 3^-d - (2^{1-d}/N) sum_i prod_k (1 - x_{i,k}^2)
//                              + (1/N^2) sum_i sum_j prod_k (1 - max(x_{i,k}, x_{j,k})).
// Each thread i computes its row's contribution to sum1 and sum2, and a block/tree reduction
// accumulates partials.
// ---------------------------------------------------------------------------
__global__ void l2_star_discrepancy_kernel(const double* __restrict__ points, int n, int dimension,
                                           double* __restrict__ partials) {
    extern __shared__ double sdata[];
    const int tid = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);

    const double inv_n = 1.0 / static_cast<double>(n);
    const double inv_n2 = inv_n * inv_n;
    const double coeff_s1 = -pow(2.0, 1.0 - static_cast<double>(dimension)) * inv_n;

    double local_sum = 0.0;
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid; i < n;
         i += stride) {
        const long long i_off = static_cast<long long>(i) * dimension;

        // p1 = prod_k (1 - x_ik^2)
        double p1 = 1.0;
        for (int k = 0; k < dimension; ++k) {
            const double xk = points[i_off + k];
            p1 *= (1.0 - xk * xk);
        }

        // s2_i = sum_j prod_k (1 - max(x_ik, x_jk))
        double s2_i = 0.0;
        for (int j = 0; j < n; ++j) {
            const long long j_off = static_cast<long long>(j) * dimension;
            double p2 = 1.0;
            for (int k = 0; k < dimension; ++k) {
                p2 *= (1.0 - fmax(points[i_off + k], points[j_off + k]));
            }
            s2_i += p2;
        }

        local_sum += coeff_s1 * p1 + inv_n2 * s2_i;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_partials_kernel(const double* __restrict__ partials, int n_partials,
                                       double* __restrict__ dev_out) {
    extern __shared__ double sdata[];
    const int tid = static_cast<int>(threadIdx.x);
    double local = 0.0;
    for (int i = tid; i < n_partials; i += static_cast<int>(blockDim.x)) {
        local += partials[i];
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dev_out[0] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Sobol' batch point generation kernel.
// ---------------------------------------------------------------------------
__global__ void sobol_batch_kernel(const unsigned int* __restrict__ dir_numbers, int dir_stride,
                                   unsigned long long n0, int count, int dimension,
                                   double* __restrict__ out) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         i < count; i += stride) {
        const unsigned long long n = n0 + static_cast<unsigned long long>(i);
        const unsigned int gray = static_cast<unsigned int>(n ^ (n >> 1));
        const long long out_off = static_cast<long long>(i) * dimension;

        for (int d = 0; d < dimension; ++d) {
            unsigned int x = 0;
            unsigned int g = gray;
            int bit = 0;
            const int d_off = d * dir_stride;
            while (g != 0U) {
                if (g & 1U) {
                    x ^= dir_numbers[d_off + bit];
                }
                g >>= 1;
                ++bit;
            }
            out[out_off + d] = static_cast<double>(x) * 2.3283064365386962890625e-10;  // 2^-32
        }
    }
}

// ---------------------------------------------------------------------------
// Halton batch point generation kernel.
// ---------------------------------------------------------------------------
__global__ void halton_batch_kernel(const int* __restrict__ primes, unsigned long long n0,
                                    int count, int dimension, double* __restrict__ out) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         i < count; i += stride) {
        const unsigned long long n = n0 + static_cast<unsigned long long>(i);
        const long long out_off = static_cast<long long>(i) * dimension;

        for (int d = 0; d < dimension; ++d) {
            const int base = primes[d];
            const double inv_base = 1.0 / static_cast<double>(base);
            double f = 0.0;
            double q = inv_base;
            unsigned long long m = n;
            while (m > 0ULL) {
                const unsigned long long digit = m % static_cast<unsigned long long>(base);
                m /= static_cast<unsigned long long>(base);
                f += static_cast<double>(digit) * q;
                q *= inv_base;
            }
            out[out_off + d] = f;
        }
    }
}

}  // namespace

extern "C" int nimblecas_gpu_l2_star_discrepancy(const double* points, int n, int dimension,
                                                double* out) {
    if (out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *out = 0.0;
    if (n <= 0 || dimension <= 0 || points == nullptr) {
        return 0;
    }

    const int threads = 256;
    const int blocks = choose_blocks(n, threads);
    const size_t point_bytes = static_cast<size_t>(n) * static_cast<size_t>(dimension) * sizeof(double);
    const size_t part_bytes = static_cast<size_t>(blocks) * sizeof(double);
    const size_t shmem = static_cast<size_t>(threads) * sizeof(double);

    double* dev_points = nullptr;
    double* dev_partials = nullptr;
    double* dev_sum = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    PinnedScope pin_points = host_register(points, point_bytes);

    if ((err = cudaMalloc(&dev_points, point_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_partials, part_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_sum, sizeof(double))) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_points, points, point_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        l2_star_discrepancy_kernel<<<blocks, threads, shmem>>>(dev_points, n, dimension,
                                                               dev_partials);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            reduce_partials_kernel<<<1, threads, shmem>>>(dev_partials, blocks, dev_sum);
            if ((err = cudaGetLastError()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else {
                double total_sum = 0.0;
                if ((err = cudaMemcpy(&total_sum, dev_sum, sizeof(double),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                } else {
                    const double d2 = pow(3.0, -static_cast<double>(dimension)) + total_sum;
                    *out = sqrt(fmax(d2, 0.0));
                }
            }
        }
    }

    if (dev_points != nullptr) {
        cudaFree(dev_points);
    }
    if (dev_partials != nullptr) {
        cudaFree(dev_partials);
    }
    if (dev_sum != nullptr) {
        cudaFree(dev_sum);
    }
    host_unregister(pin_points);
    return rc;
}

extern "C" int nimblecas_gpu_sobol_batch(const unsigned int* dir_numbers, int dir_stride,
                                         unsigned long long n0, int count, int dimension,
                                         double* out) {
    if (count <= 0 || dimension <= 0) {
        return 0;
    }
    if (dir_numbers == nullptr || out == nullptr || dir_stride < 32) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t dir_bytes =
        static_cast<size_t>(dimension) * static_cast<size_t>(dir_stride) * sizeof(unsigned int);
    const size_t out_bytes =
        static_cast<size_t>(count) * static_cast<size_t>(dimension) * sizeof(double);

    unsigned int* dev_dir = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    PinnedScope pin_out = host_register(out, out_bytes);

    if ((err = cudaMalloc(&dev_dir, dir_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_dir, dir_numbers, dir_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(count, threads);
        sobol_batch_kernel<<<blocks, threads>>>(dev_dir, dir_stride, n0, count, dimension,
                                                dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, out_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_dir != nullptr) {
        cudaFree(dev_dir);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    host_unregister(pin_out);
    return rc;
}

extern "C" int nimblecas_gpu_halton_batch(const int* primes, unsigned long long n0, int count,
                                          int dimension, double* out) {
    if (count <= 0 || dimension <= 0) {
        return 0;
    }
    if (primes == nullptr || out == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t prime_bytes = static_cast<size_t>(dimension) * sizeof(int);
    const size_t out_bytes =
        static_cast<size_t>(count) * static_cast<size_t>(dimension) * sizeof(double);

    int* dev_primes = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    PinnedScope pin_out = host_register(out, out_bytes);

    if ((err = cudaMalloc(&dev_primes, prime_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_primes, primes, prime_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(count, threads);
        halton_batch_kernel<<<blocks, threads>>>(dev_primes, n0, count, dimension, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, out_bytes, cudaMemcpyDeviceToHost)) !=
                   cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_primes != nullptr) {
        cudaFree(dev_primes);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    host_unregister(pin_out);
    return rc;
}
