// NimbleCAS CUDA GPU kernels — batched barrier-option Monte Carlo pricing.
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and
// linked into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code
// here uses ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object
// carries no libstdc++ dependency and links cleanly with the clang/libc++ engine.

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "gpu_bridge.h"

namespace {

// Internal POD for precalculated barrier option parameters in MC simulation.
struct McBarrierOptPre {
    double spot;
    double strike;
    double drift;
    double vol_sqrtdt;
    double disc;
    int is_call;
    int down;
};

// Device Threefry/Acklam RNG ports verbatim
__host__ __device__ inline unsigned long long nc_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ inline unsigned long long nc_rotl64(unsigned long long v, unsigned r) {
    return (v << r) | (v >> (64U - r));
}

// Threefry-2x64-20 fold — bit-identical to nimblecas::counter_u64(key, counter).
__device__ inline unsigned long long nc_counter_u64(unsigned long long key,
                                                    unsigned long long counter) {
    const unsigned long long parity = 0x1BD11BDAA9FC1A22ULL;
    const unsigned long long k0 = key;
    const unsigned long long k1 = nc_splitmix64(key);
    const unsigned long long k2 = parity ^ k0 ^ k1;
    const unsigned long long ks[3] = {k0, k1, k2};
    const unsigned rot[8] = {16U, 42U, 12U, 31U, 16U, 32U, 24U, 21U};
    unsigned long long x0 = counter + k0;
    unsigned long long x1 = k1;
    for (unsigned r = 0; r < 20U; ++r) {
        x0 += x1;
        x1 = nc_rotl64(x1, rot[r % 8U]);
        x1 ^= x0;
        if ((r + 1U) % 4U == 0U) {
            const unsigned s = (r + 1U) / 4U;
            x0 += ks[s % 3U];
            x1 += ks[(s + 1U) % 3U] + s;
        }
    }
    return x0 ^ x1;
}

__device__ inline double nc_acklam_central(double p) {
    const double q = p - 0.5;
    const double r = q * q;
    double num = -3.969683028665376e+01;
    num = fma(num, r, 2.209460984245205e+02);
    num = fma(num, r, -2.759285104469687e+02);
    num = fma(num, r, 1.383577518672690e+02);
    num = fma(num, r, -3.066479806614716e+01);
    num = fma(num, r, 2.506628277459239e+00);
    num = num * q;
    double den = -5.447609879822406e+01;
    den = fma(den, r, 1.615858368580409e+02);
    den = fma(den, r, -1.556989798598866e+02);
    den = fma(den, r, 6.680131188771972e+01);
    den = fma(den, r, -1.328068155288572e+01);
    den = fma(den, r, 1.0);
    return num / den;
}

__device__ inline double nc_acklam_tail(double q) {
    double num = -7.784894002430293e-03;
    num = fma(num, q, -3.223964580411365e-01);
    num = fma(num, q, -2.400758277161838e+00);
    num = fma(num, q, -2.549732539343734e+00);
    num = fma(num, q, 4.374664141464968e+00);
    num = fma(num, q, 2.938163982698783e+00);
    double den = 7.784695709041462e-03;
    den = fma(den, q, 3.224671290700398e-01);
    den = fma(den, q, 2.445134137142996e+00);
    den = fma(den, q, 3.754408661907416e+00);
    den = fma(den, q, 1.0);
    return num / den;
}

__device__ inline double nc_z_from_bits(unsigned long long bits) {
    const double u0 = static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
    const double u = fmin(fmax(u0, 1e-15), 1.0 - 1e-15);
    if (u < 0.02425) { return nc_acklam_tail(sqrt(-2.0 * log(u))); }
    if (u <= 1.0 - 0.02425) { return nc_acklam_central(u); }
    return -nc_acklam_tail(sqrt(-2.0 * log(1.0 - u)));
}

constexpr unsigned long long kMcSegPaths = 4096;

__global__ void mc_barrier_segment_kernel(const McBarrierOptPre* __restrict__ pre, int n_opts,
                                          double barrier, int knock_in, int steps,
                                          unsigned long long paths, int nseg,
                                          unsigned long long key, double* __restrict__ psum,
                                          double* __restrict__ psumsq) {
    const long long total = static_cast<long long>(n_opts) * nseg;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int opt = static_cast<int>(idx / nseg);
        const int seg = static_cast<int>(idx - static_cast<long long>(opt) * nseg);
        const McBarrierOptPre o = pre[opt];
        const unsigned long long start_p = static_cast<unsigned long long>(seg) * kMcSegPaths;
        const unsigned long long end_p = (start_p + kMcSegPaths < paths) ? (start_p + kMcSegPaths) : paths;
        double sum = 0.0;
        double sum_sq = 0.0;
        for (unsigned long long p = start_p; p < end_p; ++p) {
            double s = o.spot;
            bool hit = false;
            const unsigned long long base_counter = p * static_cast<unsigned long long>(steps);
            for (int t = 0; t < steps; ++t) {
                const unsigned long long counter = base_counter + static_cast<unsigned long long>(t);
                const unsigned long long bits = nc_counter_u64(key, counter);
                const double z = nc_z_from_bits(bits);
                s *= exp(o.drift + o.vol_sqrtdt * z);
                if ((o.down != 0 && s <= barrier) || (o.down == 0 && s >= barrier)) {
                    hit = true;
                }
            }
            const bool alive = (knock_in != 0) ? hit : !hit;
            const double payoff = alive ? (o.is_call != 0 ? fmax(s - o.strike, 0.0) : fmax(o.strike - s, 0.0)) * o.disc : 0.0;
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        psum[idx] = sum;
        psumsq[idx] = sum_sq;
    }
}

__global__ void mc_barrier_reduce_kernel(const double* __restrict__ psum,
                                         const double* __restrict__ psumsq, int nseg,
                                         double* __restrict__ sums, double* __restrict__ sumsqs) {
    extern __shared__ double sdata[];
    double* s_sum = sdata;
    double* s_sumsq = sdata + 256;
    const int opt = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    const long long base = static_cast<long long>(opt) * nseg;
    for (int i = tid; i < nseg; i += 256) {
        local_sum += psum[base + i];
        local_sumsq += psumsq[base + i];
    }
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sums[opt] = s_sum[0];
        sumsqs[opt] = s_sumsq[0];
    }
}

int choose_blocks(int work, int threads) {
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

}  // namespace

extern "C" int nimblecas_gpu_mc_barrier_batch(const NimblecasBsOption* opts, int n, double barrier,
                                              int knock_in, int steps, unsigned long long paths,
                                              unsigned long long seed, NimblecasMcEstimate* out) {
    if (n <= 0) {
        return 0;
    }
    // Defense-in-depth at the raw C ABI (the nimblecas.gpu wrapper already validates): reject the
    // degenerate arguments that would otherwise divide by zero (dt = time/steps), launch an empty
    // grid, or finalize 0/0 = NaN while falsely returning success. Mirrors the Asian bridge guard.
    if (steps < 1 || paths == 0 || barrier <= 0.0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int nseg = static_cast<int>((paths + kMcSegPaths - 1ULL) / kMcSegPaths);
    if (nseg > 0 && n > INT_MAX / nseg) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int total = n * nseg;
    const size_t pre_bytes = static_cast<size_t>(n) * sizeof(McBarrierOptPre);
    const size_t partial_bytes = static_cast<size_t>(total) * sizeof(double);
    const size_t out_sums_bytes = static_cast<size_t>(n) * sizeof(double);

    McBarrierOptPre* pre_host = (McBarrierOptPre*)malloc(pre_bytes);
    double* sums_host = (double*)malloc(out_sums_bytes);
    double* sumsqs_host = (double*)malloc(out_sums_bytes);

    if (!pre_host || !sums_host || !sumsqs_host) {
        if (pre_host) free(pre_host);
        if (sums_host) free(sums_host);
        if (sumsqs_host) free(sumsqs_host);
        return static_cast<int>(cudaErrorMemoryAllocation);
    }

    for (int i = 0; i < n; ++i) {
        const NimblecasBsOption o = opts[i];
        const double dt = o.time / static_cast<double>(steps);
        pre_host[i].spot = o.spot;
        pre_host[i].strike = o.strike;
        pre_host[i].drift = (o.rate - o.dividend - 0.5 * o.volatility * o.volatility) * dt;
        pre_host[i].vol_sqrtdt = o.volatility * sqrt(dt);
        pre_host[i].disc = exp(-o.rate * o.time);
        pre_host[i].is_call = o.is_call;
        pre_host[i].down = (barrier < o.spot) ? 1 : 0;
    }

    const unsigned long long key = nc_splitmix64(seed);

    McBarrierOptPre* dev_pre = nullptr;
    double* dev_psum = nullptr;
    double* dev_psumsq = nullptr;
    double* dev_sums = nullptr;
    double* dev_sumsqs = nullptr;

    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_pre, pre_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_psum, partial_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_psumsq, partial_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_sums, out_sums_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_sumsqs, out_sums_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_pre, pre_host, pre_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(total, threads);
        mc_barrier_segment_kernel<<<blocks, threads>>>(dev_pre, n, barrier, knock_in, steps, paths, nseg, key, dev_psum, dev_psumsq);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            mc_barrier_reduce_kernel<<<n, 256, 2 * 256 * sizeof(double)>>>(dev_psum, dev_psumsq, nseg, dev_sums, dev_sumsqs);
            if ((err = cudaGetLastError()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else if ((err = cudaMemcpy(sums_host, dev_sums, out_sums_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else if ((err = cudaMemcpy(sumsqs_host, dev_sumsqs, out_sums_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else {
                const double np = static_cast<double>(paths);
                for (int i = 0; i < n; ++i) {
                    const double sum = sums_host[i];
                    const double sum_sq = sumsqs_host[i];
                    const double mean = sum / np;
                    const double var = (sum_sq - np * mean * mean) / (np > 1.0 ? np - 1.0 : 1.0);
                    const double se = sqrt(var > 0.0 ? var / np : 0.0);
                    out[i].price = mean;
                    out[i].std_error = se;
                }
            }
        }
    }

    if (dev_pre) cudaFree(dev_pre);
    if (dev_psum) cudaFree(dev_psum);
    if (dev_psumsq) cudaFree(dev_psumsq);
    if (dev_sums) cudaFree(dev_sums);
    if (dev_sumsqs) cudaFree(dev_sumsqs);

    free(pre_host);
    free(sums_host);
    free(sumsqs_host);

    return rc;
}
