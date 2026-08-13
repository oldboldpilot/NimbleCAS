// CUDA GPU kernels for batched Levenberg-Marquardt curve fitting (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// One CUDA block per problem, the whole LM trust-region loop in-kernel.

#include "gpu_bridge.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace {

constexpr int kMaxParams = 8;
constexpr int kMaxInner = 40;

// In-block tree reduction helper for a single block of 256 threads.
// Computes sum_{i=0..n-1} a[i] * b[i].
// EVERY thread in the 256-thread block must execute block_dot and pass through all
// internal __syncthreads() barriers, including threads where tid >= n.
__device__ double block_dot(const double* __restrict__ a, const double* __restrict__ b, int n,
                            double* __restrict__ sdata) {
    const int tid = static_cast<int>(threadIdx.x);
    double local_sum = 0.0;
    for (int i = tid; i < n; i += 256) {
        local_sum += a[i] * b[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();
    if (tid < 16) { sdata[tid] += sdata[tid + 16]; }
    __syncthreads();
    if (tid < 8) { sdata[tid] += sdata[tid + 8]; }
    __syncthreads();
    if (tid < 4) { sdata[tid] += sdata[tid + 4]; }
    __syncthreads();
    if (tid < 2) { sdata[tid] += sdata[tid + 2]; }
    __syncthreads();
    if (tid < 1) { sdata[tid] += sdata[tid + 1]; }
    __syncthreads();

    const double result = sdata[0];
    __syncthreads();
    return result;
}

// Device model evaluator: computes f(t; theta) and optionally the analytic Jacobian row [df/dtheta_0 ...].
__device__ inline void device_eval_model(int model_type, double t, const double* __restrict__ theta,
                                         int m, double* __restrict__ out_f,
                                         double* __restrict__ out_jrow) {
    switch (model_type) {
        case 0: {  // polynomial: sum_{j<m} theta_j * t^j
            double f = 0.0;
            double p = 1.0;
            for (int j = 0; j < m; ++j) {
                f += theta[j] * p;
                if (out_jrow) { out_jrow[j] = p; }
                p *= t;
            }
            if (out_f) { *out_f = f; }
            break;
        }
        case 1: {  // exponential: th0 * exp(th1 * t) + th2
            double e = exp(theta[1] * t);
            if (out_f) { *out_f = theta[0] * e + theta[2]; }
            if (out_jrow) {
                out_jrow[0] = e;
                out_jrow[1] = theta[0] * t * e;
                out_jrow[2] = 1.0;
            }
            break;
        }
        case 2: {  // gaussian: th0 * exp(-(t - th1)^2 / (2 th2^2))
            double u = (t - theta[1]) / theta[2];
            double e = exp(-0.5 * u * u);
            if (out_f) { *out_f = theta[0] * e; }
            if (out_jrow) {
                out_jrow[0] = e;
                out_jrow[1] = theta[0] * e * u / theta[2];
                out_jrow[2] = theta[0] * e * u * u / theta[2];
            }
            break;
        }
        case 3: {  // logistic: th0 / (1 + exp(-th1 * (t - th2)))
            double s = 1.0 / (1.0 + exp(-theta[1] * (t - theta[2])));
            if (out_f) { *out_f = theta[0] * s; }
            if (out_jrow) {
                out_jrow[0] = s;
                out_jrow[1] = theta[0] * s * (1.0 - s) * (t - theta[2]);
                out_jrow[2] = -theta[0] * s * (1.0 - s) * theta[1];
            }
            break;
        }
        case 4: {  // sinusoid: th0 * sin(th1 * t + th2) + th3
            double arg = theta[1] * t + theta[2];
            double s = sin(arg);
            double c = cos(arg);
            if (out_f) { *out_f = theta[0] * s + theta[3]; }
            if (out_jrow) {
                out_jrow[0] = s;
                out_jrow[1] = theta[0] * t * c;
                out_jrow[2] = theta[0] * c;
                out_jrow[3] = 1.0;
            }
            break;
        }
        case 5: {  // power_law: th0 * t^th1
            double p = pow(t, theta[1]);
            if (out_f) { *out_f = theta[0] * p; }
            if (out_jrow) {
                out_jrow[0] = p;
                out_jrow[1] = theta[0] * p * log(t);
            }
            break;
        }
        default:
            break;
    }
}

// Batched Levenberg-Marquardt megakernel: one CUDA block per problem, whole LM iteration in-kernel.
__global__ void batched_lm_kernel(
    const int* __restrict__ model_cat, const double* __restrict__ t_cat,
    const double* __restrict__ y_cat, const int* __restrict__ pt_off,
    double* __restrict__ theta_cat, const int* __restrict__ th_off,
    const int* __restrict__ jac_off, int num_problems, double* __restrict__ scratch_r,
    double* __restrict__ scratch_rt, double* __restrict__ scratch_J, int max_iter, double tol,
    double lambda0, int use_fd, double fd_step, int* __restrict__ out_iters,
    int* __restrict__ out_converged, double* __restrict__ out_resid) {
    extern __shared__ double sdata[];
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ double sh_A[kMaxParams * kMaxParams];
    __shared__ double sh_M[kMaxParams * kMaxParams];
    __shared__ double sh_g[kMaxParams];
    __shared__ double sh_delta[kMaxParams];
    __shared__ double sh_theta[kMaxParams];
    __shared__ double sh_theta_try[kMaxParams];
    __shared__ double sh_ctrl[4];  // [0]=lambda, [1]=fnorm2, [2]=status, [3]=fn_try2

    for (int sys = static_cast<int>(blockIdx.x); sys < num_problems;
         sys += static_cast<int>(gridDim.x)) {
        const int n = pt_off[sys + 1] - pt_off[sys];
        const int m = th_off[sys + 1] - th_off[sys];
        const int model_type = model_cat[sys];
        const double* t = t_cat + pt_off[sys];
        const double* y = y_cat + pt_off[sys];
        double* theta_out = theta_cat + th_off[sys];
        double* r = scratch_r + pt_off[sys];
        double* rt = scratch_rt + pt_off[sys];
        double* J = scratch_J + jac_off[sys];

        if (tid < m) {
            sh_theta[tid] = theta_out[tid];
        }
        if (tid == 0) {
            sh_ctrl[0] = (lambda0 > 0.0) ? lambda0 : 1e-3;
        }
        __syncthreads();

        for (int i = tid; i < n; i += 256) {
            double f_val = 0.0;
            device_eval_model(model_type, t[i], sh_theta, m, &f_val, nullptr);
            r[i] = f_val - y[i];
        }
        __syncthreads();

        double fnorm2 = block_dot(r, r, n, sdata);
        if (tid == 0) {
            sh_ctrl[1] = fnorm2;
        }
        __syncthreads();

        for (int it = 0; it < max_iter; ++it) {
            // Build Jacobian J (column-major order per problem)
            if (use_fd == 0) {
                for (int j = 0; j < m; ++j) {
                    for (int i = tid; i < n; i += 256) {
                        double f_val = 0.0;
                        double j_row[kMaxParams];
                        device_eval_model(model_type, t[i], sh_theta, m, &f_val, j_row);
                        J[j * n + i] = j_row[j];
                    }
                }
            } else {
                for (int j = 0; j < m; ++j) {
                    double hj = fd_step * (1.0 + fabs(sh_theta[j]));
                    double th_pert[kMaxParams];
                    for (int p = 0; p < m; ++p) {
                        th_pert[p] = sh_theta[p];
                    }
                    th_pert[j] += hj;
                    for (int i = tid; i < n; i += 256) {
                        double f_pert = 0.0;
                        device_eval_model(model_type, t[i], th_pert, m, &f_pert, nullptr);
                        double f_base = r[i] + y[i];
                        J[j * n + i] = (f_pert - f_base) / hj;
                    }
                }
            }
            __syncthreads();

            // Build A = J^T J and g = J^T r using block_dot
            for (int j = 0; j < m; ++j) {
                double gj = block_dot(J + j * n, r, n, sdata);
                if (tid == 0) {
                    sh_g[j] = gj;
                }
                for (int l = j; l < m; ++l) {
                    double Ajl = block_dot(J + j * n, J + l * n, n, sdata);
                    if (tid == 0) {
                        sh_A[j * m + l] = Ajl;
                        sh_A[l * m + j] = Ajl;
                    }
                }
            }
            __syncthreads();

            // Finiteness guard on A and g
            if (tid == 0) {
                bool ok = true;
                for (int j = 0; j < m; ++j) {
                    if (!isfinite(sh_g[j])) {
                        ok = false;
                        break;
                    }
                    for (int l = 0; l < m; ++l) {
                        if (!isfinite(sh_A[j * m + l])) {
                            ok = false;
                            break;
                        }
                    }
                }
                sh_ctrl[2] = ok ? 1.0 : 0.0;
            }
            __syncthreads();

            // Consume-then-barrier: every thread latches the finiteness flag into a
            // private register and synchronises BEFORE thread 0 reuses sh_ctrl[2] for
            // the convergence flag below (:sh_ctrl[2] = 2.0/0.0). Without this barrier a
            // warp stalled after the publish above could read sh_ctrl[2] after thread 0
            // overwrote it, flip its branch decision, and desync at a __syncthreads().
            const double finite_ok = sh_ctrl[2];
            __syncthreads();

            if (finite_ok == 0.0) {
                if (tid == 0) {
                    out_iters[sys] = it;
                    out_converged[sys] = 0;
                }
                goto write_exit;
            }

            // Convergence test at loop top
            if (tid == 0) {
                double gmax = 0.0;
                for (int j = 0; j < m; ++j) {
                    gmax = fmax(gmax, fabs(sh_g[j]));
                }
                double fnorm = sqrt(sh_ctrl[1]);
                if (fnorm <= tol || gmax <= tol) {
                    out_iters[sys] = it;
                    out_converged[sys] = 1;
                    sh_ctrl[2] = 2.0;  // flag for top-of-loop convergence
                } else {
                    sh_ctrl[2] = 0.0;
                }
            }
            __syncthreads();

            // Consume-then-barrier: latch the convergence flag before thread 0 reuses
            // sh_ctrl[2] as the solve-ok flag inside the inner damping loop (:sh_ctrl[2]
            // = solve_ok ? 1.0 : 0.0). Values written there (0.0/1.0) never equal the
            // 2.0 decision value, so this is benign today, but the WAR read/overwrite of
            // a shared slot is still a racecheck hazard; the barrier retires all reads
            // first.
            const double conv_flag = sh_ctrl[2];
            __syncthreads();

            if (conv_flag == 2.0) {
                goto write_exit;
            }

            // Inner damping loop (up to 40 tries)
            bool accepted = false;
            for (int inner = 0; inner < kMaxInner; ++inner) {
                if (tid == 0) {
                    double lambda = sh_ctrl[0];
                    for (int i = 0; i < m * m; ++i) {
                        sh_M[i] = sh_A[i];
                    }
                    for (int j = 0; j < m; ++j) {
                        double djj = sh_A[j * m + j];
                        sh_M[j * m + j] += lambda * (djj > 0.0 ? djj : 1.0);
                    }

                    // In-block Cholesky factorisation L L^T = M
                    bool solve_ok = true;
                    for (int j = 0; j < m; ++j) {
                        for (int k = 0; k <= j; ++k) {
                            double sum = sh_M[j * m + k];
                            for (int p = 0; p < k; ++p) {
                                sum -= sh_M[j * m + p] * sh_M[k * m + p];
                            }
                            if (j == k) {
                                if (sum <= 0.0 || !isfinite(sum)) {
                                    solve_ok = false;
                                    break;
                                }
                                sh_M[j * m + j] = sqrt(sum);
                            } else {
                                sh_M[j * m + k] = sum / sh_M[k * m + k];
                            }
                        }
                        if (!solve_ok) break;
                    }

                    if (solve_ok) {
                        // Forward sub: L y = -g
                        for (int i = 0; i < m; ++i) {
                            double sum = -sh_g[i];
                            for (int k = 0; k < i; ++k) {
                                sum -= sh_M[i * m + k] * sh_delta[k];
                            }
                            sh_delta[i] = sum / sh_M[i * m + i];
                        }
                        // Back sub: L^T delta = y
                        for (int i = m - 1; i >= 0; --i) {
                            double sum = sh_delta[i];
                            for (int k = i + 1; k < m; ++k) {
                                sum -= sh_M[k * m + i] * sh_delta[k];
                            }
                            sh_delta[i] = sum / sh_M[i * m + i];
                        }
                        for (int i = 0; i < m; ++i) {
                            if (!isfinite(sh_delta[i])) {
                                solve_ok = false;
                                break;
                            }
                        }
                    }
                    sh_ctrl[2] = solve_ok ? 1.0 : 0.0;
                }
                __syncthreads();

                if (sh_ctrl[2] == 0.0) {
                    if (tid == 0) {
                        sh_ctrl[0] *= 4.0;
                    }
                    __syncthreads();
                    if (sh_ctrl[0] > 1e18) {
                        if (tid == 0) {
                            out_iters[sys] = it;
                            out_converged[sys] = 0;
                        }
                        goto write_exit;
                    }
                    continue;
                }

                if (tid < m) {
                    sh_theta_try[tid] = sh_theta[tid] + sh_delta[tid];
                }
                __syncthreads();

                if (tid == 0) {
                    bool ok = true;
                    for (int j = 0; j < m; ++j) {
                        if (!isfinite(sh_theta_try[j])) {
                            ok = false;
                            break;
                        }
                    }
                    sh_ctrl[2] = ok ? 1.0 : 0.0;
                }
                __syncthreads();

                if (sh_ctrl[2] == 0.0) {
                    if (tid == 0) {
                        sh_ctrl[0] *= 4.0;
                    }
                    __syncthreads();
                    continue;
                }

                for (int i = tid; i < n; i += 256) {
                    double f_val = 0.0;
                    device_eval_model(model_type, t[i], sh_theta_try, m, &f_val, nullptr);
                    rt[i] = f_val - y[i];
                }
                __syncthreads();

                double fn_try2 = block_dot(rt, rt, n, sdata);
                if (tid == 0) {
                    sh_ctrl[3] = fn_try2;
                }
                __syncthreads();

                if (!isfinite(sh_ctrl[3])) {
                    if (tid == 0) {
                        sh_ctrl[0] *= 4.0;
                    }
                    __syncthreads();
                    if (sh_ctrl[0] > 1e18) {
                        if (tid == 0) {
                            out_iters[sys] = it;
                            out_converged[sys] = 0;
                        }
                        goto write_exit;
                    }
                    continue;
                }

                double fn_try = sqrt(sh_ctrl[3]);
                double fnorm = sqrt(sh_ctrl[1]);
                // Consume-then-barrier: every thread must finish reading sh_ctrl[1]
                // (current fnorm^2) into its private fnorm BEFORE thread 0 overwrites it
                // with the accepted fn_try^2 (sh_ctrl[1] = sh_ctrl[3]) below. Without this
                // barrier a stalled warp could read the just-written value, evaluate
                // fn_try < fnorm as false, take the reject branch (parking at the wrong
                // barrier) AND skip its strided share of the r[i] = rt[i] copy -> corrupt
                // residual on the hot accept path.
                __syncthreads();
                if (fn_try < fnorm) {
                    for (int i = tid; i < n; i += 256) {
                        r[i] = rt[i];
                    }
                    if (tid < m) {
                        sh_theta[tid] = sh_theta_try[tid];
                    }
                    if (tid == 0) {
                        sh_ctrl[1] = sh_ctrl[3];
                        sh_ctrl[0] = fmax(sh_ctrl[0] / 3.0, 1e-12);
                    }
                    __syncthreads();
                    accepted = true;
                    break;
                } else {
                    if (tid == 0) {
                        sh_ctrl[0] *= 4.0;
                    }
                    __syncthreads();
                    if (sh_ctrl[0] > 1e18) {
                        if (tid == 0) {
                            out_iters[sys] = it + 1;
                            out_converged[sys] = 0;
                        }
                        goto write_exit;
                    }
                }
            }

            if (!accepted) {
                if (tid == 0) {
                    out_iters[sys] = it + 1;
                    out_converged[sys] = 0;
                }
                goto write_exit;
            }
        }

        // Budget exhausted
        if (tid == 0) {
            out_iters[sys] = max_iter;
            out_converged[sys] = (sqrt(sh_ctrl[1]) <= tol) ? 1 : 0;
        }

    write_exit:
        for (int i = tid; i < n; i += 256) {
            double f_val = 0.0;
            device_eval_model(model_type, t[i], sh_theta, m, &f_val, nullptr);
            r[i] = f_val - y[i];
        }
        __syncthreads();

        double true_fn2 = block_dot(r, r, n, sdata);
        double true_resid = sqrt(true_fn2);

        if (tid < m) {
            theta_out[tid] = sh_theta[tid];
        }
        if (tid == 0) {
            out_resid[sys] = true_resid;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int nimblecas_gpu_batched_lm_curvefit(
    const int* model, const double* t_cat, const double* y_cat, const int* pt_off,
    double* theta_cat, const int* th_off, const int* jac_off, int num_problems, int max_iter,
    double tol, double lambda0, int use_fd, double fd_step, int* out_iters, int* out_converged,
    double* out_resid) {
    if (num_problems <= 0) {
        return (num_problems == 0) ? 0 : cudaErrorInvalidValue;
    }
    if (!model || !t_cat || !y_cat || !pt_off || !theta_cat || !th_off || !jac_off || !out_iters ||
        !out_converged || !out_resid) {
        return cudaErrorInvalidValue;
    }

    const int total_n = pt_off[num_problems];
    const int total_th = th_off[num_problems];
    const int total_jac = jac_off[num_problems];
    if (total_n < 0 || total_th < 0 || total_jac < 0) {
        return cudaErrorInvalidValue;
    }

    // Base offsets must be zero. Per-problem diffs and totals can all stay positive
    // while pt_off[0]/th_off[0]/jac_off[0] are negative, which would let the kernel
    // index below the allocation (the pointers are base_ptr + off[sys]).
    if (pt_off[0] != 0 || th_off[0] != 0 || jac_off[0] != 0) {
        return cudaErrorInvalidValue;
    }

    for (int k = 0; k < num_problems; ++k) {
        int nk = pt_off[k + 1] - pt_off[k];
        int mk = th_off[k + 1] - th_off[k];
        int jk = jac_off[k + 1] - jac_off[k];
        // nk * mk in 64-bit: the int product can overflow (UB) for a hostile caller.
        if (nk <= 0 || mk <= 0 || mk > kMaxParams || nk < mk ||
            static_cast<long long>(nk) * static_cast<long long>(mk) !=
                static_cast<long long>(jk)) {
            return cudaErrorInvalidValue;
        }
    }

    for (int k = 0; k < num_problems; ++k) {
        out_iters[k] = 0;
        out_converged[k] = 0;
        out_resid[k] = 0.0;
    }

    const size_t model_bytes = static_cast<size_t>(num_problems) * sizeof(int);
    const size_t n_bytes = static_cast<size_t>(total_n) * sizeof(double);
    const size_t th_bytes = static_cast<size_t>(total_th) * sizeof(double);
    const size_t off_bytes = static_cast<size_t>(num_problems + 1) * sizeof(int);
    const size_t jac_bytes = static_cast<size_t>(total_jac) * sizeof(double);
    const size_t out_int_bytes = static_cast<size_t>(num_problems) * sizeof(int);
    const size_t out_dbl_bytes = static_cast<size_t>(num_problems) * sizeof(double);

    int* d_model = nullptr;
    double* d_t = nullptr;
    double* d_y = nullptr;
    int* d_pt_off = nullptr;
    double* d_theta = nullptr;
    int* d_th_off = nullptr;
    int* d_jac_off = nullptr;
    double* d_scratch_r = nullptr;
    double* d_scratch_rt = nullptr;
    double* d_scratch_J = nullptr;
    int* d_out_iters = nullptr;
    int* d_out_converged = nullptr;
    double* d_out_resid = nullptr;

    cudaError_t rc = cudaMalloc(&d_model, model_bytes != 0 ? model_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_t, n_bytes != 0 ? n_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_y, n_bytes != 0 ? n_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_pt_off, off_bytes != 0 ? off_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_theta, th_bytes != 0 ? th_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_th_off, off_bytes != 0 ? off_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_jac_off, off_bytes != 0 ? off_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_scratch_r, n_bytes != 0 ? n_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_scratch_rt, n_bytes != 0 ? n_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_scratch_J, jac_bytes != 0 ? jac_bytes : sizeof(double));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_out_iters, out_int_bytes != 0 ? out_int_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_out_converged, out_int_bytes != 0 ? out_int_bytes : sizeof(int));
    if (rc == cudaSuccess) rc = cudaMalloc(&d_out_resid, out_dbl_bytes != 0 ? out_dbl_bytes : sizeof(double));

    if (rc == cudaSuccess) {
        // Fold the scratch zeroing into the rc ladder: the kernel reads scratch_r
        // (f_base = r[i] + y[i]) before writing it in FD mode, so a failed memset must
        // abort rather than run on garbage.
        if (n_bytes > 0) rc = cudaMemset(d_scratch_r, 0, n_bytes);
        if (rc == cudaSuccess && n_bytes > 0) rc = cudaMemset(d_scratch_rt, 0, n_bytes);
        if (rc == cudaSuccess && jac_bytes > 0) rc = cudaMemset(d_scratch_J, 0, jac_bytes);

        if (rc == cudaSuccess) rc = cudaMemcpy(d_model, model, model_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_t, t_cat, n_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_y, y_cat, n_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_pt_off, pt_off, off_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_theta, theta_cat, th_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_th_off, th_off, off_bytes, cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(d_jac_off, jac_off, off_bytes, cudaMemcpyHostToDevice);

        if (rc == cudaSuccess) {
            int dev = 0;
            cudaGetDevice(&dev);
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
            int blocks = num_problems;
            if (sm_count > 0 && blocks > sm_count * 32) {
                blocks = sm_count * 32;
            }
            const size_t shmem = 256 * sizeof(double);
            batched_lm_kernel<<<blocks, 256, shmem>>>(
                d_model, d_t, d_y, d_pt_off, d_theta, d_th_off, d_jac_off, num_problems,
                d_scratch_r, d_scratch_rt, d_scratch_J, max_iter, tol, lambda0, use_fd, fd_step,
                d_out_iters, d_out_converged, d_out_resid);

            rc = cudaGetLastError();
            if (rc == cudaSuccess) {
                rc = cudaDeviceSynchronize();
            }
            if (rc == cudaSuccess) {
                rc = cudaMemcpy(theta_cat, d_theta, th_bytes, cudaMemcpyDeviceToHost);
            }
            if (rc == cudaSuccess) {
                rc = cudaMemcpy(out_iters, d_out_iters, out_int_bytes, cudaMemcpyDeviceToHost);
            }
            if (rc == cudaSuccess) {
                rc = cudaMemcpy(out_converged, d_out_converged, out_int_bytes, cudaMemcpyDeviceToHost);
            }
            if (rc == cudaSuccess) {
                rc = cudaMemcpy(out_resid, d_out_resid, out_dbl_bytes, cudaMemcpyDeviceToHost);
            }
        }
    }

    if (d_model) cudaFree(d_model);
    if (d_t) cudaFree(d_t);
    if (d_y) cudaFree(d_y);
    if (d_pt_off) cudaFree(d_pt_off);
    if (d_theta) cudaFree(d_theta);
    if (d_th_off) cudaFree(d_th_off);
    if (d_jac_off) cudaFree(d_jac_off);
    if (d_scratch_r) cudaFree(d_scratch_r);
    if (d_scratch_rt) cudaFree(d_scratch_rt);
    if (d_scratch_J) cudaFree(d_scratch_J);
    if (d_out_iters) cudaFree(d_out_iters);
    if (d_out_converged) cudaFree(d_out_converged);
    if (d_out_resid) cudaFree(d_out_resid);

    return static_cast<int>(rc);
}
