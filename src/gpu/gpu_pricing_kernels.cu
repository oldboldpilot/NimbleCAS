// NimbleCAS CUDA GPU kernels — batched derivative pricing (ROADMAP 5).
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

// Internal POD for precalculated option parameters in MC simulation.
struct McOptPre {
    double spot;
    double strike;
    double drift;
    double vol_sqrtT;
    double disc;
    int is_call;
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

__global__ void mc_segment_kernel(const McOptPre* __restrict__ pre, int n_opts,
                                  unsigned long long paths, int nseg,
                                  unsigned long long key, double* __restrict__ psum,
                                  double* __restrict__ psumsq) {
    // 64-bit work index/stride: total == n_opts*nseg is capped at INT_MAX by the wrapper,
    // but `idx += stride` can step past INT_MAX on the final increment — signed-int overflow
    // (UB). Widening to long long keeps the grid-stride loop well-defined at the cap.
    const long long total = static_cast<long long>(n_opts) * nseg;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int opt = static_cast<int>(idx / nseg);
        const int seg = static_cast<int>(idx - static_cast<long long>(opt) * nseg);
        const McOptPre o = pre[opt];
        const unsigned long long start_i = static_cast<unsigned long long>(seg) * kMcSegPaths;
        const unsigned long long end_i = (start_i + kMcSegPaths < paths) ? (start_i + kMcSegPaths) : paths;
        double sum = 0.0;
        double sum_sq = 0.0;
        for (unsigned long long i = start_i; i < end_i; ++i) {
            const unsigned long long bits = nc_counter_u64(key, i);
            const double z = nc_z_from_bits(bits);
            const double ep = exp(o.drift + o.vol_sqrtT * z);
            const double em = exp(o.drift - o.vol_sqrtT * z);
            const double sp = o.spot * ep;
            const double sm = o.spot * em;
            const double pay_p = o.is_call ? fmax(sp - o.strike, 0.0) : fmax(o.strike - sp, 0.0);
            const double pay_m = o.is_call ? fmax(sm - o.strike, 0.0) : fmax(o.strike - sm, 0.0);
            const double payoff = 0.5 * (pay_p + pay_m) * o.disc;
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        psum[idx] = sum;
        psumsq[idx] = sum_sq;
    }
}

__global__ void mc_reduce_kernel(const double* __restrict__ psum,
                                 const double* __restrict__ psumsq, int nseg,
                                 double* __restrict__ sums, double* __restrict__ sumsqs) {
    extern __shared__ double sdata[];
    double* s_sum = sdata;
    double* s_sumsq = sdata + 256;
    const int opt = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    const long long base = static_cast<long long>(opt) * nseg;  // 64-bit: base+i indexes the partials
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

__device__ void nc_bs_greeks(double S, double K, double r, double q, double v, double T,
                             int is_call, NimblecasBsGreeks& g) {
    const bool call = (is_call != 0);
    g.price = 0.0;
    g.delta = 0.0;
    g.gamma = 0.0;
    g.vega = 0.0;
    g.theta = 0.0;
    g.rho = 0.0;

    if (T == 0.0 || v == 0.0) {
        const double fwd = S * exp((r - q) * T);
        const double intrinsic = call ? fmax(fwd - K, 0.0) : fmax(K - fwd, 0.0);
        g.price = exp(-r * T) * intrinsic;
        g.delta = call ? (fwd > K ? exp(-q * T) : 0.0) : (fwd < K ? -exp(-q * T) : 0.0);
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * v * v) * T) / (v * sqrtT);
    const double d2 = d1 - v * sqrtT;
    const double disc_r = exp(-r * T);
    const double disc_q = exp(-q * T);
    const double norm_cdf_d1 = 0.5 * erfc(-d1 * 0.7071067811865475244);
    const double norm_cdf_neg_d1 = 0.5 * erfc(d1 * 0.7071067811865475244);
    const double norm_cdf_d2 = 0.5 * erfc(-d2 * 0.7071067811865475244);
    const double norm_cdf_neg_d2 = 0.5 * erfc(d2 * 0.7071067811865475244);
    const double pdf_d1 = 0.3989422804014327 * exp(-0.5 * d1 * d1);

    if (call) {
        g.price = S * disc_q * norm_cdf_d1 - K * disc_r * norm_cdf_d2;
        g.delta = disc_q * norm_cdf_d1;
        g.rho = K * T * disc_r * norm_cdf_d2;
        g.theta = -S * disc_q * pdf_d1 * v / (2.0 * sqrtT)
                  - r * K * disc_r * norm_cdf_d2 + q * S * disc_q * norm_cdf_d1;
    } else {
        g.price = K * disc_r * norm_cdf_neg_d2 - S * disc_q * norm_cdf_neg_d1;
        g.delta = -disc_q * norm_cdf_neg_d1;
        g.rho = -K * T * disc_r * norm_cdf_neg_d2;
        g.theta = -S * disc_q * pdf_d1 * v / (2.0 * sqrtT)
                  + r * K * disc_r * norm_cdf_neg_d2 - q * S * disc_q * norm_cdf_neg_d1;
    }
    g.gamma = disc_q * pdf_d1 / (S * v * sqrtT);
    g.vega = S * disc_q * pdf_d1 * sqrtT;
}

__global__ void bs_greeks_kernel(const NimblecasBsOption* __restrict__ opts,
                                 NimblecasBsGreeks* __restrict__ out, int n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const NimblecasBsOption o = opts[i];
        nc_bs_greeks(o.spot, o.strike, o.rate, o.dividend, o.volatility, o.time, o.is_call, out[i]);
    }
}

__global__ void bs_extended_greeks_kernel(const NimblecasBsOption* __restrict__ opts,
                                          NimblecasBsExtGreeks* __restrict__ out, int n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const NimblecasBsOption o = opts[i];
        const double S = o.spot;
        const double K = o.strike;
        const double r = o.rate;
        const double q = o.dividend;
        const double sig = o.volatility;
        const double T = o.time;
        const bool call = (o.is_call != 0);

        const double sqrtT = sqrt(T);
        const double d1 = (log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT);
        const double d2 = d1 - sig * sqrtT;
        const double phi = 0.3989422804014327 * exp(-0.5 * d1 * d1);
        const double disc_q = exp(-q * T);
        const double disc_r = exp(-r * T);
        const double gamma = disc_q * phi / (S * sig * sqrtT);
        const double vega = S * disc_q * phi * sqrtT;

        NimblecasBsGreeks base{};
        nc_bs_greeks(S, K, r, q, sig, T, o.is_call, base);

        NimblecasBsExtGreeks g{};
        g.vanna = -disc_q * phi * d2 / sig;
        g.vomma = vega * d1 * d2 / sig;
        g.speed = -gamma / S * (d1 / (sig * sqrtT) + 1.0);
        g.zomma = gamma * (d1 * d2 - 1.0) / sig;
        g.lambda = base.price != 0.0 ? base.delta * S / base.price : 0.0;
        const double norm_cdf_d2 = 0.5 * erfc(-d2 * 0.7071067811865475244);
        const double norm_cdf_neg_d2 = 0.5 * erfc(d2 * 0.7071067811865475244);
        g.dual_delta = call ? -disc_r * norm_cdf_d2 : disc_r * norm_cdf_neg_d2;
        g.dual_gamma = disc_r * (0.3989422804014327 * exp(-0.5 * d2 * d2)) / (K * sig * sqrtT);
        const double norm_cdf_d1 = 0.5 * erfc(-d1 * 0.7071067811865475244);
        const double norm_cdf_neg_d1 = 0.5 * erfc(d1 * 0.7071067811865475244);
        g.epsilon = call ? -S * T * disc_q * norm_cdf_d1 : S * T * disc_q * norm_cdf_neg_d1;
        g.ultima = -vega / (sig * sig) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2);

        const double h = 1e-4 * T;
        NimblecasBsGreeks up{}, dn{};
        nc_bs_greeks(S, K, r, q, sig, T + h, o.is_call, up);
        nc_bs_greeks(S, K, r, q, sig, T - h, o.is_call, dn);
        g.charm = (up.delta - dn.delta) / (2.0 * h);
        g.color = -(up.gamma - dn.gamma) / (2.0 * h);
        g.veta = -(up.vega - dn.vega) / (2.0 * h);

        const double hv = 1e-4 * sig;
        NimblecasBsGreeks upv{}, dnv{};
        nc_bs_greeks(S, K, r, q, sig + hv, T, o.is_call, upv);
        nc_bs_greeks(S, K, r, q, sig - hv, T, o.is_call, dnv);
        g.vera = (upv.rho - dnv.rho) / (2.0 * hv);

        out[i] = g;
    }
}

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

}  // namespace

extern "C" int nimblecas_gpu_mc_european_batch(const NimblecasBsOption* opts, int n,
                                    unsigned long long paths, unsigned long long seed,
                                    NimblecasMcEstimate* out) {
    if (n <= 0) {
        return 0;
    }
    const int nseg = static_cast<int>((paths + kMcSegPaths - 1ULL) / kMcSegPaths);
    // Defense in depth: the C++ wrapper already bounds n*nseg <= INT_MAX, but this exported
    // bridge is C-ABI callable directly, so re-check here rather than overflow `total`.
    if (nseg > 0 && n > INT_MAX / nseg) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int total = n * nseg;
    const size_t pre_bytes = static_cast<size_t>(n) * sizeof(McOptPre);
    const size_t partial_bytes = static_cast<size_t>(total) * sizeof(double);
    const size_t out_sums_bytes = static_cast<size_t>(n) * sizeof(double);

    McOptPre* pre_host = (McOptPre*)malloc(pre_bytes);
    double* sums_host = (double*)malloc(out_sums_bytes);
    double* sumsqs_host = (double*)malloc(out_sums_bytes);

    if (!pre_host || !sums_host || !sumsqs_host) {
        if (pre_host) free(pre_host);
        if (sums_host) free(sums_host);
        if (sumsqs_host) free(sumsqs_host);
        return static_cast<int>(cudaErrorMemoryAllocation);
    }

    for (int i = 0; i < n; ++i) {
        // These host-side precomputes are compiled by nvcc's host compiler (whose default
        // FP-contraction may fuse differently from the clang CPU pricer), so drift/vol_sqrtT/
        // disc can differ from the CPU by ~1 ulp. That is absorbed by the documented 1e-6 MC
        // agreement bound (orders of magnitude below the sampling error); the per-path draw
        // bits themselves are exact-integer and bit-identical to the CPU.
        const NimblecasBsOption o = opts[i];
        pre_host[i].spot = o.spot;
        pre_host[i].strike = o.strike;
        pre_host[i].drift = (o.rate - o.dividend - 0.5 * o.volatility * o.volatility) * o.time;
        pre_host[i].vol_sqrtT = o.volatility * sqrt(o.time);
        pre_host[i].disc = exp(-o.rate * o.time);
        pre_host[i].is_call = o.is_call;
    }

    const unsigned long long key = nc_splitmix64(seed);

    McOptPre* dev_pre = nullptr;
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
        mc_segment_kernel<<<blocks, threads>>>(dev_pre, n, paths, nseg, key, dev_psum, dev_psumsq);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            mc_reduce_kernel<<<n, 256, 2 * 256 * sizeof(double)>>>(dev_psum, dev_psumsq, nseg, dev_sums, dev_sumsqs);
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

extern "C" int nimblecas_gpu_bs_greeks_batch(const NimblecasBsOption* opts, NimblecasBsGreeks* out,
                                  int n) {
    if (n <= 0) {
        return 0;
    }
    const size_t in_bytes = static_cast<size_t>(n) * sizeof(NimblecasBsOption);
    const size_t out_bytes = static_cast<size_t>(n) * sizeof(NimblecasBsGreeks);

    NimblecasBsOption* dev_opts = nullptr;
    NimblecasBsGreeks* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_opts, in_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_opts, opts, in_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n, threads);
        bs_greeks_kernel<<<blocks, threads>>>(dev_opts, dev_out, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, out_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_opts) cudaFree(dev_opts);
    if (dev_out) cudaFree(dev_out);

    return rc;
}

extern "C" int nimblecas_gpu_bs_extended_greeks_batch(const NimblecasBsOption* opts,
                                           NimblecasBsExtGreeks* out, int n) {
    if (n <= 0) {
        return 0;
    }
    const size_t in_bytes = static_cast<size_t>(n) * sizeof(NimblecasBsOption);
    const size_t out_bytes = static_cast<size_t>(n) * sizeof(NimblecasBsExtGreeks);

    NimblecasBsOption* dev_opts = nullptr;
    NimblecasBsExtGreeks* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_opts, in_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_opts, opts, in_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n, threads);
        bs_extended_greeks_kernel<<<blocks, threads>>>(dev_opts, dev_out, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, out_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_opts) cudaFree(dev_opts);
    if (dev_out) cudaFree(dev_out);

    return rc;
}
