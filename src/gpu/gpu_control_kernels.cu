// NimbleCAS CUDA GPU kernels — Bode and Nyquist frequency-response sweeps.
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

__device__ inline void eval_complex_poly(const double* __restrict__ coeffs, int n_coeffs,
                                         double w, double* out_re, double* out_im) {
    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int k = n_coeffs - 1; k >= 0; --k) {
        const double new_re = -acc_im * w + coeffs[k];
        const double new_im = acc_re * w;
        acc_re = new_re;
        acc_im = new_im;
    }
    *out_re = acc_re;
    *out_im = acc_im;
}

__device__ inline void complex_div(double n_re, double n_im, double d_re, double d_im,
                                   double* h_re, double* h_im) {
    // Smith's algorithm (scaled division), matching libc++ std::complex::operator/ used by the CPU
    // oracle. The naive den_sq = d_re*d_re + d_im*d_im form overflows to +inf once |den| > ~1.34e154
    // (giving H=(0,0), magnitude_db=-inf) and underflows below ~1.5e-162 — both reachable by a
    // high-order denominator swept at large/small omega. Smith stays correct up to |den| ~ 1e308.
    if (fabs(d_re) >= fabs(d_im)) {
        const double r = d_im / d_re;
        const double t = 1.0 / (d_re + d_im * r);
        *h_re = (n_re + n_im * r) * t;
        *h_im = (n_im - n_re * r) * t;
    } else {
        const double r = d_re / d_im;
        const double t = 1.0 / (d_re * r + d_im);
        *h_re = (n_re * r + n_im) * t;
        *h_im = (n_im * r - n_re) * t;
    }
}

__global__ void bode_sweep_kernel(const double* __restrict__ num, int n_num,
                                   const double* __restrict__ den, int n_den,
                                   const double* __restrict__ omegas, int n_omega,
                                   double* __restrict__ out_mag_db,
                                   double* __restrict__ out_phase_deg) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_omega; i += stride) {
        const double w = omegas[i];
        double n_re = 0.0, n_im = 0.0, d_re = 0.0, d_im = 0.0, h_re = 0.0, h_im = 0.0;
        eval_complex_poly(num, n_num, w, &n_re, &n_im);
        eval_complex_poly(den, n_den, w, &d_re, &d_im);
        complex_div(n_re, n_im, d_re, d_im, &h_re, &h_im);
        const double mag = hypot(h_re, h_im);
        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double rad_to_deg = 180.0 / pi;
        out_mag_db[i] = 20.0 * log10(mag);
        out_phase_deg[i] = atan2(h_im, h_re) * rad_to_deg;
    }
}

__global__ void nyquist_sweep_kernel(const double* __restrict__ num, int n_num,
                                      const double* __restrict__ den, int n_den,
                                      const double* __restrict__ omegas, int n_omega,
                                      double* __restrict__ out_re,
                                      double* __restrict__ out_im) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_omega; i += stride) {
        const double w = omegas[i];
        double n_re = 0.0, n_im = 0.0, d_re = 0.0, d_im = 0.0, h_re = 0.0, h_im = 0.0;
        eval_complex_poly(num, n_num, w, &n_re, &n_im);
        eval_complex_poly(den, n_den, w, &d_re, &d_im);
        complex_div(n_re, n_im, d_re, d_im, &h_re, &h_im);
        out_re[i] = h_re;
        out_im[i] = h_im;
    }
}

}  // namespace

extern "C" int nimblecas_gpu_bode_sweep(const double* num, int n_num, const double* den,
                                       int n_den, const double* omegas, int n_omega,
                                       double* out_mag_db, double* out_phase_deg) {
    if (n_omega <= 0) {
        return 0;
    }
    if (omegas == nullptr || out_mag_db == nullptr || out_phase_deg == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_num > 0 && num == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_den > 0 && den == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t num_bytes = static_cast<size_t>(n_num > 0 ? n_num : 0) * sizeof(double);
    const size_t num_alloc = num_bytes != 0 ? num_bytes : sizeof(double);
    const size_t den_bytes = static_cast<size_t>(n_den > 0 ? n_den : 0) * sizeof(double);
    const size_t den_alloc = den_bytes != 0 ? den_bytes : sizeof(double);
    const size_t omega_bytes = static_cast<size_t>(n_omega) * sizeof(double);

    double* dev_num = nullptr;
    double* dev_den = nullptr;
    double* dev_omegas = nullptr;
    double* dev_out_mag = nullptr;
    double* dev_out_phase = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_num, num_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_den, den_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_omegas, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out_mag, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out_phase, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_num > 0 &&
               (err = cudaMemcpy(dev_num, num, num_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_den > 0 &&
               (err = cudaMemcpy(dev_den, den, den_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_omegas, omegas, omega_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n_omega, threads);
        bode_sweep_kernel<<<blocks, threads>>>(dev_num, n_num, dev_den, n_den, dev_omegas,
                                               n_omega, dev_out_mag, dev_out_phase);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out_mag_db, dev_out_mag, omega_bytes,
                                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out_phase_deg, dev_out_phase, omega_bytes,
                                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_num != nullptr) {
        cudaFree(dev_num);
    }
    if (dev_den != nullptr) {
        cudaFree(dev_den);
    }
    if (dev_omegas != nullptr) {
        cudaFree(dev_omegas);
    }
    if (dev_out_mag != nullptr) {
        cudaFree(dev_out_mag);
    }
    if (dev_out_phase != nullptr) {
        cudaFree(dev_out_phase);
    }
    return rc;
}

extern "C" int nimblecas_gpu_nyquist_sweep(const double* num, int n_num, const double* den,
                                          int n_den, const double* omegas, int n_omega,
                                          double* out_re, double* out_im) {
    if (n_omega <= 0) {
        return 0;
    }
    if (omegas == nullptr || out_re == nullptr || out_im == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_num > 0 && num == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_den > 0 && den == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t num_bytes = static_cast<size_t>(n_num > 0 ? n_num : 0) * sizeof(double);
    const size_t num_alloc = num_bytes != 0 ? num_bytes : sizeof(double);
    const size_t den_bytes = static_cast<size_t>(n_den > 0 ? n_den : 0) * sizeof(double);
    const size_t den_alloc = den_bytes != 0 ? den_bytes : sizeof(double);
    const size_t omega_bytes = static_cast<size_t>(n_omega) * sizeof(double);

    double* dev_num = nullptr;
    double* dev_den = nullptr;
    double* dev_omegas = nullptr;
    double* dev_out_re = nullptr;
    double* dev_out_im = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;

    if ((err = cudaMalloc(&dev_num, num_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_den, den_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_omegas, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out_re, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out_im, omega_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_num > 0 &&
               (err = cudaMemcpy(dev_num, num, num_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (n_den > 0 &&
               (err = cudaMemcpy(dev_den, den, den_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_omegas, omegas, omega_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(n_omega, threads);
        nyquist_sweep_kernel<<<blocks, threads>>>(dev_num, n_num, dev_den, n_den, dev_omegas,
                                                  n_omega, dev_out_re, dev_out_im);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out_re, dev_out_re, omega_bytes,
                                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out_im, dev_out_im, omega_bytes,
                                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }

    if (dev_num != nullptr) {
        cudaFree(dev_num);
    }
    if (dev_den != nullptr) {
        cudaFree(dev_den);
    }
    if (dev_omegas != nullptr) {
        cudaFree(dev_omegas);
    }
    if (dev_out_re != nullptr) {
        cudaFree(dev_out_re);
    }
    if (dev_out_im != nullptr) {
        cudaFree(dev_out_im);
    }
    return rc;
}
