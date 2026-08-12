// Plain C ABI bridge to the NimbleCAS CUDA GPU kernels (ROADMAP 5).
// @author Olumuyiwa Oluwasanmi
//
// This header is included both by gpu_kernels.cu (compiled by nvcc) and in the global
// module fragment of nimblecas.gpu (compiled by clang + libc++). It deliberately exposes
// only a plain C interface over POD types, so no C++ standard-library object crosses the
// nvcc <-> clang/libc++ boundary and the two objects link cleanly.

#ifndef NIMBLECAS_GPU_BRIDGE_H
#define NIMBLECAS_GPU_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Number of CUDA-capable devices (0 if none or CUDA is unavailable).
int nimblecas_gpu_device_count(void);

// Evaluate the polynomial coeffs[0..n_coeffs-1] (low degree first) at each of the n
// points x[0..n-1], writing p(x_i) to out[i]. Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_poly_eval(const double* coeffs, int n_coeffs,
                            const double* x, double* out, int n);

// Batched Levenshtein edit distance, one thread per (a_i, b_i) pair. The sequences are stored
// flattened as integer code points in a_flat / b_flat; a_off / b_off are prefix-offset arrays
// of length pairs+1 so that pair i spans [a_off[i], a_off[i+1]) and [b_off[i], b_off[i+1]).
// Writes the distance of pair i to out[i]. Returns 0 on success, or a non-zero CUDA error
// code on failure.
int nimblecas_gpu_edit_distance_batch(const int* a_flat, const int* a_off, const int* b_flat,
                                      const int* b_off, int pairs, int* out);

// Level-synchronous single-source BFS over a CSR graph (row_offsets length num_vertices+1,
// col_indices length num_edges). Fills dist[0..num_vertices-1] with the hop distance from
// source (unreachable vertices remain -1). Returns 0 on success, or a non-zero CUDA error
// code on failure.
int nimblecas_gpu_bfs(const int* row_offsets, const int* col_indices, int num_vertices,
                      int num_edges, int source, int* dist);

// Count the solutions to the n-queens problem on the GPU, one thread per partial placement of
// the first row(s), each thread completing its subtree with the classic three-bitmask method.
// Writes the total solution count to *out_count. Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_nqueens_count(int n, unsigned long long* out_count);

// QMC integration reduction. Horner-evaluate the polynomial integrand coeffs[0..n_coeffs-1]
// (low degree first) at each of the n sample points x[0..n-1] on the device and reduce (sum),
// writing the equal-weight average (1/n) * sum p(x_i) to *out_mean. This is the device analogue
// of the numerical qmc_integrate: the caller supplies the (low-discrepancy) sample points. The
// device sums in block/tree order, which differs from a strict left-to-right CPU sum, so the
// last bits of *out_mean may differ from a CPU average (each is a valid estimate). Returns 0 on
// success, or a non-zero CUDA error code on failure.
int nimblecas_gpu_qmc_poly_integrate(const double* coeffs, int n_coeffs, const double* x, int n,
                                     double* out_mean);

// One-level batch Haar discrete wavelet transform (orthonormal 1/sqrt(2) normalization), one
// thread per output coefficient pair, grid-stride over the batch. `data` holds `batch` contiguous
// signal blocks of `len` samples each (len must be even), row-major. For each block, writes its
// len/2 approximation coefficients followed by its len/2 detail coefficients to the matching
// block of `out` (same size and layout as the input). Returns 0 on success, or a non-zero CUDA
// error code on failure.
int nimblecas_gpu_haar_dwt_batch(const double* data, int batch, int len, double* out);

// Batched dense double matrix multiply, one thread per output element. `a` holds `batch`
// contiguous row-major A blocks (each m x k) and `b` holds `batch` contiguous row-major B blocks
// (each k x n); writes the `batch` row-major products C_b = A_b * B_b (each m x n) to `c`, in the
// same contiguous block layout. Returns 0 on success, or a non-zero CUDA error code on failure.
int nimblecas_gpu_batched_matmul(const double* a, const double* b, double* c, int batch, int m,
                                 int k, int n);

// Batched radix-2 forward FFT (Cooley-Tukey, decimation-in-time), one CUDA block per signal.
// `in` holds `batch` complex signals of length `n` (n a power of two), each stored as 2*n
// interleaved doubles (re, im, re, im, ...) contiguously; writes the DFT of each signal
// X_k = sum_j x_j e^{-2*pi*i*k*j/n} (FORWARD, negative exponent) to `out` in the same interleaved
// layout. n must be a power of two with 1 <= n <= 2048 (the shared-memory length cap). Returns 0
// on success, or a non-zero CUDA error code on failure (cudaErrorInvalidValue for a bad shape).
int nimblecas_gpu_fft_batch(const double* in, double* out, int batch, int n);

// A single Black-Scholes-Merton option in POD form for the batched device pricer. Fields
// mirror nimblecas::pricing::OptionSpec; is_call is 1 for a call, 0 for a put.
typedef struct {
    double spot;
    double strike;
    double rate;
    double dividend;
    double volatility;
    double time;
    int is_call;
} NimblecasBsOption;

// Batched Black-Scholes-Merton pricing, one thread per option (grid-stride). Writes the
// price of opts[i] to out[i]. Inputs must be physically valid (spot>0, strike>0, time>=0,
// volatility>=0) — nimblecas.gpu validates before calling. Returns 0 on success or a
// non-zero CUDA error code.
int nimblecas_gpu_black_scholes_batch(const NimblecasBsOption* opts, double* out, int n);

// Same computation, but the kernel launch is captured once into a CUDA graph and replayed
// `iterations` times on persistent device buffers before the result is read back.
// Numerically identical to the direct version; the graph amortizes per-launch overhead
// across repeated re-pricing of a fixed-shape grid (a live risk sweep). iterations<1 is
// treated as 1. Returns 0 on success or a non-zero CUDA error code.
int nimblecas_gpu_black_scholes_batch_graphed(const NimblecasBsOption* opts, double* out,
                                              int n, int iterations);

// Conjugate-gradient solve of a SYMMETRIC POSITIVE-DEFINITE sparse system A x = b, with A in
// CSR form: row_offsets length n+1, col_indices/values length nnz. x is the solution (the
// caller supplies an initial guess, typically zeros, in x on entry). Iterates until the
// residual 2-norm <= tol*||b|| or max_iters is reached. Writes the iteration count to
// *out_iters, 1/0 (converged?) to *out_converged, and the final residual 2-norm to *out_resid.
// Returns 0 on success, or a non-zero CUDA error code. NOTE: device dot-product reductions sum
// in block/tree order, so the last bits may differ from a sequential CPU CG (each is a valid
// numerical solution) -- the CPU krylov::cg remains authoritative; this is a numerical mirror.
int nimblecas_gpu_cg_csr(const int* row_offsets, const int* col_indices, const double* values,
                         int n, int nnz, const double* b, double* x, int max_iters, double tol,
                         int* out_iters, int* out_converged, double* out_resid);

/* --- FAMILY A: batched derivative pricing (gpu_pricing_kernels.cu) --- */

/* A Monte Carlo estimate in POD form: the price and its standard error. */
typedef struct {
    double price;
    double std_error;
} NimblecasMcEstimate;

/* Batched reproducible European Monte Carlo. Prices each of the n options over the SAME
 * counter stream [0, paths) with key = splitmix64(seed) — the identical Threefry-2x64-20
 * draw indexing as nimblecas.rng's counter_u64 — using antithetic variates, and writes
 * { price, std_error } to out[i]. The path range is decomposed into FIXED segments of
 * kMcSegPaths (4096) paths; one thread sums one segment serially in index order, and a
 * fixed-shape 256-thread block per option folds the segment partials — so the result is a
 * pure function of (opts, paths, seed), independent of launch geometry. Inputs must be
 * physically valid and paths in [1, 1e9] (the C++ wrapper validates). Returns 0 on success
 * or a non-zero CUDA error code. */
int nimblecas_gpu_mc_european_batch(const NimblecasBsOption* opts, int n,
                                    unsigned long long paths, unsigned long long seed,
                                    NimblecasMcEstimate* out);

/* Black-Scholes first-order Greeks in POD form (fields mirror pricing::Greeks). */
typedef struct {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
} NimblecasBsGreeks;

/* Batched analytic Black-Scholes Greeks, one thread per option (grid-stride). Mirrors
 * pricing::black_scholes_greeks exactly (same d1/d2, same degenerate T==0/vol==0 collapse
 * with limit delta). Inputs must be physically valid — the C++ wrapper validates. Returns
 * 0 on success or a non-zero CUDA error code. */
int nimblecas_gpu_bs_greeks_batch(const NimblecasBsOption* opts, NimblecasBsGreeks* out,
                                  int n);

/* Extended Black-Scholes Greeks in POD form (fields mirror pricing::ExtendedGreeks). */
typedef struct {
    double vanna;
    double charm;
    double vomma;
    double veta;
    double speed;
    double zomma;
    double color;
    double lambda;
    double dual_delta;
    double dual_gamma;
    double epsilon;
    double vera;
    double ultima;
} NimblecasBsExtGreeks;

/* Batched extended Greeks, one thread per option (grid-stride). Mirrors
 * pricing::black_scholes_extended_greeks INCLUDING its central-finite-difference fields
 * (charm/color/veta with h = 1e-4*T; vera with hv = 1e-4*sig). Requires T > 0 and vol > 0
 * for every option — the C++ wrapper validates. Returns 0 on success or a non-zero CUDA
 * error code. */
int nimblecas_gpu_bs_extended_greeks_batch(const NimblecasBsOption* opts,
                                           NimblecasBsExtGreeks* out, int n);
/* --- FAMILY B: strategy payoff / P&L grid sweeps (gpu_sweep_kernels.cu) --- */

/* One signed strategy leg in POD form. Fields mirror optstrat::StrategyLeg; right is
 * 0 = call, 1 = put, 2 = underlying (strike ignored for underlying). */
typedef struct {
    double strike;
    double quantity;   /* signed: long (+) / short (-) */
    double premium;    /* per-unit price paid (option premium / underlying entry) */
    int right;         /* 0 call, 1 put, 2 underlying */
} NimblecasSweepLeg;

/* Aggregate expiry payoff / P&L sweep, one thread per grid point (grid-stride). For each
 * grid price s the thread sums quantity*terminal_value(s) over the legs IN ARRAY ORDER and,
 * when net_of_premium is non-zero, separately sums quantity*premium in array order and
 * subtracts — the exact operation sequence of optstrat's payoff_at/pnl_at. All arithmetic
 * uses the non-contracted __dadd_rn/__dsub_rn/__dmul_rn/fmax intrinsics so the result is
 * bit-equal to the CPU double evaluation (whose optstrat oracle is likewise pinned
 * FP_CONTRACT off) -- an exactly-reproducible piecewise-linear
 * computation, not a statistical one). Returns 0 on success or a non-zero CUDA error code. */
int nimblecas_gpu_strategy_grid(const NimblecasSweepLeg* legs, int n_legs,
                                const double* grid, int n_grid, int net_of_premium,
                                double* out);

/* One futures leg in POD form: scaled_quantity = quantity * contract_size (precomputed on
 * the host to preserve the CPU association), entry_price the per-unit entry. */
typedef struct {
    double scaled_quantity;
    double entry_price;
} NimblecasFuturesSweepLeg;

/* Uniform-settlement futures P&L sweep, one thread per grid point (grid-stride):
 * out[j] = sum_i scaled_quantity_i * (grid[j] - entry_price_i), legs in array order, with
 * the non-contracted __dsub_rn/__dmul_rn/__dadd_rn intrinsics — bit-equal to the CPU
 * FuturesStrategy::pnl_at_uniform evaluation. Returns 0 on success or a non-zero CUDA
 * error code. */
int nimblecas_gpu_futures_grid(const NimblecasFuturesSweepLeg* legs, int n_legs,
                               const double* grid, int n_grid, double* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NIMBLECAS_GPU_BRIDGE_H
