// NimbleCAS CUDA GPU kernels — batched Longstaff-Schwartz American Monte Carlo pricing.
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

// Fixed path-segment size. MUST equal nimblecas::gpu::kGpuMcSegPaths (gpu.cppm) so the wrapper's
// nseg overflow guard describes this kernel's actual launch, and MUST stay a compile-time constant
// (not a function of paths/SM count) so the segmented reduction is a pure function of the inputs.
constexpr unsigned long long kLsmSegPaths = 4096;

__global__ void lsm_forward_kernel(double spot, double strike, int is_call,
                                   double drift, double vol_sqrtdt,
                                   int steps, unsigned long long paths,
                                   unsigned long long key,
                                   double* __restrict__ S,
                                   double* __restrict__ cash) {
    const unsigned long long P = paths;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long p = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < P; p += stride) {
        double s = spot;
        S[0 * P + p] = spot;
        const unsigned long long base = p * static_cast<unsigned long long>(steps);
        for (int t = 0; t < steps; ++t) {
            const unsigned long long ctr_idx = base + static_cast<unsigned long long>(t);
            const unsigned long long bits = nc_counter_u64(key, ctr_idx);
            const double z = nc_z_from_bits(bits);
            s *= exp(fma(vol_sqrtdt, z, drift));
            S[static_cast<size_t>(t + 1) * P + p] = s;
        }
        cash[p] = is_call ? fmax(s - strike, 0.0) : fmax(strike - s, 0.0);
    }
}

__global__ void lsm_accumulate_segment_kernel(double strike, int is_call, double disc,
                                              int t, unsigned long long paths, int nseg,
                                              const double* __restrict__ S,
                                              double* __restrict__ cash,
                                              double* __restrict__ partials) {
    const long long total_segs = nseg;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const unsigned long long P = paths;
    const size_t t_offset = static_cast<size_t>(t) * P;

    for (long long seg = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         seg < total_segs; seg += stride) {
        const unsigned long long start_p = static_cast<unsigned long long>(seg) * kLsmSegPaths;
        const unsigned long long end_p = (start_p + kLsmSegPaths < P) ? (start_p + kLsmSegPaths) : P;

        double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
        double b0 = 0.0, b1 = 0.0, b2 = 0.0;

        for (unsigned long long p = start_p; p < end_p; ++p) {
            cash[p] *= disc;
            const double s = S[t_offset + p];
            const double ex = is_call ? fmax(s - strike, 0.0) : fmax(strike - s, 0.0);
            if (ex > 0.0) {
                const double y = cash[p];
                const double s2 = s * s;
                m0 += 1.0;
                m1 += s;
                m2 += s2;
                m3 += s2 * s;
                m4 += s2 * s2;
                b0 += y;
                b1 += s * y;
                b2 += s2 * y;
            }
        }
        partials[0 * static_cast<size_t>(nseg) + seg] = m0;
        partials[1 * static_cast<size_t>(nseg) + seg] = m1;
        partials[2 * static_cast<size_t>(nseg) + seg] = m2;
        partials[3 * static_cast<size_t>(nseg) + seg] = m3;
        partials[4 * static_cast<size_t>(nseg) + seg] = m4;
        partials[5 * static_cast<size_t>(nseg) + seg] = b0;
        partials[6 * static_cast<size_t>(nseg) + seg] = b1;
        partials[7 * static_cast<size_t>(nseg) + seg] = b2;
    }
}

__global__ void lsm_reduce_accumulate_kernel(const double* __restrict__ partials, int nseg,
                                             double* __restrict__ out8) {
    __shared__ double sdata[8][256];
    const int tid = static_cast<int>(threadIdx.x);

    double local_m[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = tid; i < nseg; i += 256) {
        for (int k = 0; k < 8; ++k) {
            local_m[k] += partials[static_cast<size_t>(k) * static_cast<size_t>(nseg) + static_cast<size_t>(i)];
        }
    }
    for (int k = 0; k < 8; ++k) {
        sdata[k][tid] = local_m[k];
    }
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 8; ++k) {
                sdata[k][tid] += sdata[k][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int k = 0; k < 8; ++k) {
            out8[k] = sdata[k][0];
        }
    }
}

__global__ void lsm_update_kernel(double strike, int is_call, int t,
                                  double beta0, double beta1, double beta2,
                                  unsigned long long paths,
                                  const double* __restrict__ S,
                                  double* __restrict__ cash) {
    const unsigned long long P = paths;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const size_t t_offset = static_cast<size_t>(t) * P;

    for (unsigned long long p = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < P; p += stride) {
        const double s = S[t_offset + p];
        const double ex = is_call ? fmax(s - strike, 0.0) : fmax(strike - s, 0.0);
        if (ex > 0.0) {
            const double cont = beta0 + beta1 * s + beta2 * (s * s);
            if (ex > cont) {
                cash[p] = ex;
            }
        }
    }
}

__global__ void lsm_finalize_segment_kernel(double disc, unsigned long long paths, int nseg,
                                            const double* __restrict__ cash,
                                            double* __restrict__ psum,
                                            double* __restrict__ psumsq) {
    const long long total_segs = nseg;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const unsigned long long P = paths;

    for (long long seg = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         seg < total_segs; seg += stride) {
        const unsigned long long start_p = static_cast<unsigned long long>(seg) * kLsmSegPaths;
        const unsigned long long end_p = (start_p + kLsmSegPaths < P) ? (start_p + kLsmSegPaths) : P;

        double sum = 0.0;
        double sum_sq = 0.0;
        for (unsigned long long p = start_p; p < end_p; ++p) {
            const double v = cash[p] * disc;
            sum += v;
            sum_sq += v * v;
        }
        psum[seg] = sum;
        psumsq[seg] = sum_sq;
    }
}

__global__ void lsm_finalize_reduce_kernel(const double* __restrict__ psum,
                                           const double* __restrict__ psumsq, int nseg,
                                           double* __restrict__ out2) {
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    const int tid = static_cast<int>(threadIdx.x);

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (int i = tid; i < nseg; i += 256) {
        local_sum += psum[i];
        local_sumsq += psumsq[i];
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
        out2[0] = s_sum[0];
        out2[1] = s_sumsq[0];
    }
}

int solve3_host(double a[3][3], double b[3], double out_x[3]) {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r) {
            if (fabs(a[r][col]) > fabs(a[piv][col])) { piv = r; }
        }
        if (fabs(a[piv][col]) < 1e-14) { return 0; }
        for (int c = 0; c < 3; ++c) {
            double tmp = a[col][c];
            a[col][c] = a[piv][c];
            a[piv][c] = tmp;
        }
        double tmpb = b[col];
        b[col] = b[piv];
        b[piv] = tmpb;

        for (int r = 0; r < 3; ++r) {
            if (r == col) { continue; }
            const double factor = a[r][col] / a[col][col];
            for (int c = col; c < 3; ++c) { a[r][c] -= factor * a[col][c]; }
            b[r] -= factor * b[col];
        }
    }
    out_x[0] = b[0] / a[0][0];
    out_x[1] = b[1] / a[1][1];
    out_x[2] = b[2] / a[2][2];
    return 1;
}

int choose_blocks(long long work, int threads) {
    const long long blocks_by_work = (work + threads - 1) / threads;
    int sm_count = 0;
    int blocks;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        sm_count > 0) {
        const long long grid_cap = static_cast<long long>(sm_count) * 32;
        blocks = static_cast<int>(grid_cap < blocks_by_work ? grid_cap : blocks_by_work);
    } else {
        cudaGetLastError();
        blocks = static_cast<int>(blocks_by_work > INT_MAX ? INT_MAX : blocks_by_work);
    }
    if (blocks < 1) { blocks = 1; }
    return blocks;
}

}  // namespace

extern "C" int nimblecas_gpu_lsm_american_batch(const NimblecasBsOption* opts, int n, int steps,
                                                unsigned long long paths, unsigned long long seed,
                                                NimblecasMcEstimate* out) {
    if (n <= 0) { return 0; }
    if (paths < 4 || steps < 1 || steps > 100000) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    constexpr unsigned long long kMaxCells = 500000000ULL;
    if (paths > kMaxCells / (static_cast<unsigned long long>(steps) + 1ULL)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int nseg = static_cast<int>((paths + kLsmSegPaths - 1ULL) / kLsmSegPaths);
    if (nseg <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    for (int i = 0; i < n; ++i) {
        if (opts[i].spot <= 0.0 || opts[i].volatility <= 0.0 || opts[i].time <= 0.0 || opts[i].strike <= 0.0) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    const size_t grid_cells = static_cast<size_t>(paths) * static_cast<size_t>(steps + 1);
    const size_t grid_bytes = grid_cells * sizeof(double);
    const size_t cash_bytes = static_cast<size_t>(paths) * sizeof(double);
    const size_t partials_bytes = static_cast<size_t>(8) * static_cast<size_t>(nseg) * sizeof(double);
    const size_t out8_bytes = 8 * sizeof(double);
    const size_t out2_bytes = 2 * sizeof(double);

    double* dev_S = nullptr;
    double* dev_cash = nullptr;
    double* dev_partials = nullptr;
    double* dev_out8 = nullptr;
    double* dev_out2 = nullptr;

    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_S, grid_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_cash, cash_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_partials, partials_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out8, out8_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out2, out2_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const unsigned long long key = nc_splitmix64(seed);
        const int threads = 256;

        for (int opt_idx = 0; opt_idx < n && rc == 0; ++opt_idx) {
            const NimblecasBsOption o = opts[opt_idx];
            const double dt = o.time / static_cast<double>(steps);
            const double drift = (o.rate - o.dividend - 0.5 * o.volatility * o.volatility) * dt;
            const double vol_sqrtdt = o.volatility * sqrt(dt);
            const double disc = exp(-o.rate * dt);

            const int f_blocks = choose_blocks(static_cast<long long>(paths), threads);
            lsm_forward_kernel<<<f_blocks, threads>>>(o.spot, o.strike, o.is_call,
                                                      drift, vol_sqrtdt, steps, paths,
                                                      key, dev_S, dev_cash);
            if ((err = cudaGetLastError()) != cudaSuccess ||
                (err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
                break;
            }

            const int seg_blocks = choose_blocks(static_cast<long long>(nseg), threads);

            for (int t = steps - 1; t >= 1; --t) {
                lsm_accumulate_segment_kernel<<<seg_blocks, threads>>>(o.strike, o.is_call, disc,
                                                                       t, paths, nseg,
                                                                       dev_S, dev_cash, dev_partials);
                if ((err = cudaGetLastError()) != cudaSuccess ||
                    (err = cudaDeviceSynchronize()) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }

                lsm_reduce_accumulate_kernel<<<1, 256>>>(dev_partials, nseg, dev_out8);
                if ((err = cudaGetLastError()) != cudaSuccess ||
                    (err = cudaDeviceSynchronize()) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }

                double host_out8[8];
                if ((err = cudaMemcpy(host_out8, dev_out8, out8_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }

                const double m0 = host_out8[0];
                if (m0 >= 3.0) {
                    double ata[3][3] = {
                        {host_out8[0], host_out8[1], host_out8[2]},
                        {host_out8[1], host_out8[2], host_out8[3]},
                        {host_out8[2], host_out8[3], host_out8[4]}
                    };
                    double atb[3] = {host_out8[5], host_out8[6], host_out8[7]};
                    double beta[3];
                    if (solve3_host(ata, atb, beta)) {
                        const int u_blocks = choose_blocks(static_cast<long long>(paths), threads);
                        lsm_update_kernel<<<u_blocks, threads>>>(o.strike, o.is_call, t,
                                                                 beta[0], beta[1], beta[2],
                                                                 paths, dev_S, dev_cash);
                        if ((err = cudaGetLastError()) != cudaSuccess ||
                            (err = cudaDeviceSynchronize()) != cudaSuccess) {
                            rc = static_cast<int>(err);
                            break;
                        }
                    }
                }
            }
            if (rc != 0) { break; }

            lsm_finalize_segment_kernel<<<seg_blocks, threads>>>(disc, paths, nseg, dev_cash,
                                                                 dev_partials, dev_partials + nseg);
            if ((err = cudaGetLastError()) != cudaSuccess ||
                (err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
                break;
            }

            lsm_finalize_reduce_kernel<<<1, 256>>>(dev_partials, dev_partials + nseg, nseg, dev_out2);
            if ((err = cudaGetLastError()) != cudaSuccess ||
                (err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
                break;
            }

            double host_out2[2];
            if ((err = cudaMemcpy(host_out2, dev_out2, out2_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                rc = static_cast<int>(err);
                break;
            }

            const double sum = host_out2[0];
            const double sum_sq = host_out2[1];
            const double np = static_cast<double>(paths);
            const double mean = sum / np;
            const double var = (sum_sq - np * mean * mean) / (np > 1.0 ? np - 1.0 : 1.0);
            const double clamped_var = var > 0.0 ? var : 0.0;
            const double se = sqrt(clamped_var / np);
            const double lower = o.is_call ? fmax(o.spot - o.strike, 0.0) : fmax(o.strike - o.spot, 0.0);

            out[opt_idx].price = fmax(mean, lower);
            out[opt_idx].std_error = se;
        }
    }

    if (dev_S) cudaFree(dev_S);
    if (dev_cash) cudaFree(dev_cash);
    if (dev_partials) cudaFree(dev_partials);
    if (dev_out8) cudaFree(dev_out8);
    if (dev_out2) cudaFree(dev_out2);

    return rc;
}
