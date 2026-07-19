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

// Pick a device-sized grid (a small multiple of the SM count) capped at one block per unit of
// work, mirroring the launch policy of poly_eval: occupancy stays high on large problems while
// small ones never spawn excessive blocks. Grid-stride kernels tolerate any returned value.
int choose_blocks(int work, int threads) {
    const int blocks_by_work = (work + threads - 1) / threads;
    int sm_count = 0;
    int blocks;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess &&
        sm_count > 0) {
        const int grid_cap = sm_count * 32;
        blocks = grid_cap < blocks_by_work ? grid_cap : blocks_by_work;
    } else {
        cudaGetLastError();  // attribute query failed; fall back to one block per work unit
        blocks = blocks_by_work;
    }
    if (blocks < 1) {
        blocks = 1;
    }
    return blocks;
}

// Upper bound on the column count of the DP table held in per-thread local memory. Sequences
// longer than this are clamped so the local arrays never overflow; the batched-distance use
// cases here (short token/character strings) stay well within it.
constexpr int kMaxEditLen = 256;

// Batched Levenshtein edit distance: one thread per (a_i, b_i) pair, grid-stride over pairs.
// Each thread runs a rolling two-row DP whose cell recurrence is the branchless three-way
// min(min(del, ins), sub); the two rows live in local memory and are swapped by pointer so no
// per-row copy is needed. Equal-length pairs execute in lockstep, so warps stay divergence
// free apart from the trip-count of the inner DP.
__global__ void edit_distance_batch_kernel(const int* __restrict__ a_flat,
                                           const int* __restrict__ a_off,
                                           const int* __restrict__ b_flat,
                                           const int* __restrict__ b_off, int pairs,
                                           int* __restrict__ out) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    int row0[kMaxEditLen + 1];
    int row1[kMaxEditLen + 1];
    for (int p = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         p < pairs; p += stride) {
        const int a_begin = a_off[p];
        const int a_raw = a_off[p + 1] - a_begin;
        const int b_begin = b_off[p];
        const int b_raw = b_off[p + 1] - b_begin;
        // Levenshtein is symmetric, so roll the DP over the SHORTER sequence: only that
        // (column) dimension sizes the bounded local arrays, so the array width need only
        // hold min(a,b)+1 cells. The host wrapper guarantees min(a,b) <= kMaxEditLen, so no
        // silent truncation of the longer side can occur. `row` walks the longer sequence.
        const int* row_seq = (a_raw >= b_raw) ? (a_flat + a_begin) : (b_flat + b_begin);
        const int* col_seq = (a_raw >= b_raw) ? (b_flat + b_begin) : (a_flat + a_begin);
        const int row_len = (a_raw >= b_raw) ? a_raw : b_raw;
        const int col_len = (a_raw >= b_raw) ? b_raw : a_raw;
        int* prev = row0;
        int* curr = row1;
        for (int j = 0; j <= col_len; ++j) {
            prev[j] = j;  // distance from the empty row-prefix to each col-prefix
        }
        for (int i = 1; i <= row_len; ++i) {
            curr[0] = i;  // distance from the i-prefix of the long seq to the empty col-prefix
            const int ri = row_seq[i - 1];
            for (int j = 1; j <= col_len; ++j) {
                const int cost = (ri != col_seq[j - 1]);  // branchless 0/1 substitution
                const int del = prev[j] + 1;
                const int ins = curr[j - 1] + 1;
                const int sub = prev[j - 1] + cost;
                curr[j] = min(min(del, ins), sub);
            }
            int* tmp = prev;  // roll the rows without copying
            prev = curr;
            curr = tmp;
        }
        out[p] = prev[col_len];  // final cell after row_len iterations of the roll
    }
}

// Level-synchronous BFS relaxation over a CSR graph. Every thread owns a vertex (grid-stride);
// a vertex in the current frontier (dist == level) relaxes each outgoing edge by branchlessly
// claiming any unvisited neighbour with atomicCAS(-1 -> level+1). A single device flag records
// whether any claim succeeded so the host loop knows when the frontier is empty.
__global__ void bfs_kernel(const int* __restrict__ row_offsets,
                           const int* __restrict__ col_indices, int num_vertices, int level,
                           int* __restrict__ dist, int* __restrict__ changed) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int v = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         v < num_vertices; v += stride) {
        if (dist[v] == level) {
            const int begin = row_offsets[v];
            const int end = row_offsets[v + 1];
            for (int e = begin; e < end; ++e) {
                const int nbr = col_indices[e];
                const int old = atomicCAS(&dist[nbr], -1, level + 1);
                if (old == -1) {
                    *changed = 1;  // benign race: every writer stores the same 1
                }
            }
        }
    }
}

// Depth of the explicit backtracking stack held in per-thread local memory. n-queens places one
// queen per row, so a board of width up to kNQueensStack-1 fits without overflow.
constexpr int kNQueensStack = 33;

// Count the completions of one partial placement with the classic three-bitmask method. The
// masks encode occupied columns and the two diagonals already shifted into the current row, so
// free = ~(cols | d1 | d2) & full and the lowest set bit `free & -free` is the next candidate —
// no data-dependent branch beyond the backtracking loop itself. An explicit stack replaces the
// usual recursion so the routine runs on the device.
__device__ unsigned long long nqueens_count_subtree(unsigned int full, unsigned int cols0,
                                                    unsigned int d1_0, unsigned int d2_0) {
    unsigned int stk_cols[kNQueensStack];
    unsigned int stk_d1[kNQueensStack];
    unsigned int stk_d2[kNQueensStack];
    unsigned int stk_free[kNQueensStack];
    int depth = 0;
    stk_cols[0] = cols0;
    stk_d1[0] = d1_0;
    stk_d2[0] = d2_0;
    stk_free[0] = ~(cols0 | d1_0 | d2_0) & full;
    unsigned long long solutions = 0ull;
    while (depth >= 0) {
        if (stk_cols[depth] == full) {
            ++solutions;  // every column filled -> one complete arrangement
            --depth;
            continue;
        }
        const unsigned int f = stk_free[depth];
        if (f == 0u) {
            --depth;  // no candidate left on this row; backtrack
            continue;
        }
        const unsigned int bit = f & (0u - f);  // isolate the lowest free column
        stk_free[depth] = f ^ bit;              // consume it before descending
        const unsigned int nc = stk_cols[depth] | bit;
        const unsigned int nd1 = (stk_d1[depth] | bit) << 1;
        const unsigned int nd2 = (stk_d2[depth] | bit) >> 1;
        ++depth;
        stk_cols[depth] = nc;
        stk_d1[depth] = nd1;
        stk_d2[depth] = nd2;
        stk_free[depth] = ~(nc | nd1 | nd2) & full;
    }
    return solutions;
}

// One thread per partial placement of the first rows; each accumulates its subtree count into
// the shared total with atomicAdd. Grid-stride keeps the launch device-sized for large n.
__global__ void nqueens_kernel(const unsigned int* __restrict__ part_cols,
                               const unsigned int* __restrict__ part_d1,
                               const unsigned int* __restrict__ part_d2, int num_partials,
                               unsigned int full, unsigned long long* __restrict__ out_count) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int p = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         p < num_partials; p += stride) {
        const unsigned long long sub =
            nqueens_count_subtree(full, part_cols[p], part_d1[p], part_d2[p]);
        atomicAdd(out_count, sub);
    }
}

// QMC integration reduction. Each thread Horner-evaluates the polynomial integrand at its
// grid-stride slice of the sample points, accumulating a private partial sum; a shared-memory
// tree reduction then folds the block's partials and thread 0 writes the block total to
// partials[blockIdx.x]. blockDim.x is a power of two by construction (the host launches 256
// threads), so the halving tree is exact. The final sum over the (few) block partials is done by
// reduce_partials_kernel below — a two-stage reduction that keeps everything on the device and
// avoids a double atomicAdd (portable to pre-sm_60 parts).
//
// DETERMINISM: the device sums in this block/tree order rather than the CPU's strict
// left-to-right order, so the last bits of the estimate can differ from a CPU average. Each is a
// valid equal-weight estimate; neither is "more correct".
__global__ void qmc_poly_integrate_kernel(const double* __restrict__ coeffs, int n_coeffs,
                                          const double* __restrict__ x, int n,
                                          double* __restrict__ partials) {
    extern __shared__ double sdata[];
    const int tid = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    double local = 0.0;
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid; i < n;
         i += stride) {
        const double xi = x[i];
        double acc = 0.0;
        for (int k = n_coeffs - 1; k >= 0; --k) {
            acc = acc * xi + coeffs[k];
        }
        local += acc;
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
        partials[blockIdx.x] = sdata[0];
    }
}

// Final single-block reduction: one block strides over the `count` block-partials, accumulates
// into shared memory, tree-reduces, and thread 0 stores the total to *out. blockDim.x is a power
// of two (256) so the halving tree is exact.
__global__ void reduce_partials_kernel(const double* __restrict__ partials, int count,
                                       double* __restrict__ out) {
    extern __shared__ double sdata[];
    const int tid = static_cast<int>(threadIdx.x);
    double local = 0.0;
    for (int i = tid; i < count; i += static_cast<int>(blockDim.x)) {
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
        *out = sdata[0];
    }
}

// One-level batch Haar DWT (orthonormal 1/sqrt(2) lifting). `in` holds `batch` contiguous signal
// blocks of `len` samples (len even); `half` = len/2. Each thread owns one output pair
// (block, k): it reads the even/odd sample pair in[2k], in[2k+1] of its block and writes the
// approximation coefficient (e + o)/sqrt(2) to out[k] and the detail coefficient (e - o)/sqrt(2)
// to out[half + k], packing all approximations before all details within each block. Grid-stride
// over the batch*half independent pairs; a regular, divergence-free data-parallel transform.
__global__ void haar_dwt_batch_kernel(const double* __restrict__ in, int batch, int len, int half,
                                      double* __restrict__ out) {
    const int total = batch * half;  // <= batch*len <= INT_MAX (guarded by the host wrapper)
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    const double inv_sqrt2 = 0.70710678118654752440;  // 1/sqrt(2)
    for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                   static_cast<int>(threadIdx.x);
         idx < total; idx += stride) {
        const int blk = idx / half;
        const int k = idx - blk * half;
        const int base = blk * len;
        const double e = in[base + 2 * k];
        const double o = in[base + 2 * k + 1];
        out[base + k] = (e + o) * inv_sqrt2;         // approximation (low-pass)
        out[base + half + k] = (e - o) * inv_sqrt2;  // detail (high-pass)
    }
}

// Batched dense double matrix multiply, one thread per output element (b, i, j). The layout is
// row-major throughout: A_b is m x k, B_b is k x n, C_b is m x n, and the `batch` problems are
// stored contiguously back to back. Each thread computes a single output scalar
// C_b[i,j] = sum_{l=0..k-1} A_b[i,l] * B_b[l,j] with a plain global-memory accumulation loop over
// the shared dimension k — the simple, provably-correct formulation (no shared-memory tiling).
// Grid-stride over the batch*m*n independent output elements keeps the launch device-sized. The
// host wrapper guarantees each flattened count (batch*m*k, batch*k*n, batch*m*n) fits in int, so
// every index below stays within int range.
__global__ void batched_matmul_kernel(const double* __restrict__ a, const double* __restrict__ b,
                                      double* __restrict__ c, int batch, int m, int k, int n) {
    const int total = batch * m * n;  // number of output scalars across all problems
    const int mn = m * n;             // elements per C_b block
    const int mk = m * k;             // elements per A_b block
    const int kn = k * n;             // elements per B_b block
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                   static_cast<int>(threadIdx.x);
         idx < total; idx += stride) {
        const int bi = idx / mn;          // which problem in the batch
        const int rem = idx - bi * mn;    // linear offset within this C_b block
        const int row = rem / n;          // output row i
        const int col = rem - row * n;    // output column j
        const int a_off = bi * mk + row * k;  // start of A_b row i
        const int b_off = bi * kn + col;      // start of B_b column j (stride n down the column)
        double acc = 0.0;
        for (int l = 0; l < k; ++l) {
            acc += a[a_off + l] * b[b_off + l * n];
        }
        c[idx] = acc;
    }
}

// Upper bound on the FFT length one block can hold in dynamic shared memory. Each of the n
// complex samples is two interleaved doubles, so one signal needs 2*n*sizeof(double) = 16*n
// bytes of shared memory; 2048 samples occupy 32 KiB, comfortably inside the 48 KiB per-block
// default without an opt-in carveout. Signals longer than this are rejected (honest error) by
// both the host wrapper here and the C++ module wrapper — the CPU fft module (Bluestein) covers
// arbitrary lengths; this GPU kernel stays power-of-two and length-capped.
constexpr int kMaxFftLen = 2048;

// Reverse the low `bits` bits of x — the index permutation of a decimation-in-time FFT, which
// consumes its input in bit-reversed order and produces the transform in natural order.
__device__ inline unsigned int fft_bit_reverse(unsigned int x, int bits) {
    unsigned int r = 0u;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

// Batched radix-2 forward DFT (Cooley-Tukey, decimation-in-time), ONE CUDA BLOCK PER SIGNAL
// (blockIdx.x = signal index). Each signal is n complex samples stored as 2*n interleaved doubles
// (re, im, re, im, ...); `in` and `out` hold `batch` such signals contiguously. The block's
// threads first scatter the samples into shared memory in bit-reversed order, then run log2(n)
// butterfly stages in place, __syncthreads() between stages so every thread sees the previous
// stage's writes. The FORWARD transform uses the twiddle w = e^{-2*pi*i/len} (NEGATIVE exponent):
// X_k = sum_j x_j e^{-2*pi*i*k*j/n}. Threads grid-stride over the n/2 butterflies of each stage
// and over the n samples of the load/store passes, so any blockDim.x (even < n/2) is correct.
// The host wrapper guarantees n is a power of two with 1 <= n <= kMaxFftLen, so the 2*n-double
// shared tile fits and log2(n) is exact.
__global__ void fft_batch_kernel(const double* __restrict__ in, double* __restrict__ out,
                                 int batch, int n) {
    extern __shared__ double s[];  // 2*n doubles: s[2k] = Re(sample k), s[2k+1] = Im(sample k)
    const int sig = static_cast<int>(blockIdx.x);
    if (sig >= batch) {
        return;  // defensive: grid is launched with exactly `batch` blocks
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    // Base offset (in doubles) of this signal; long long so the product cannot overflow int even
    // though the host wrapper guarantees the total element count fits in int.
    const long long base = static_cast<long long>(sig) * (2LL * static_cast<long long>(n));
    int logn = 0;
    while ((1 << logn) < n) {
        ++logn;  // n is a power of two (guaranteed by the wrapper), so 1<<logn == n on exit
    }
    // Bit-reversal permutation: sample k of the input lands at index rev(k) in shared memory.
    for (int k = tid; k < n; k += nthreads) {
        const unsigned int j = fft_bit_reverse(static_cast<unsigned int>(k), logn);
        s[2 * j] = in[base + 2 * k];
        s[2 * j + 1] = in[base + 2 * k + 1];
    }
    __syncthreads();
    const double two_pi = 6.28318530717958647692;
    // Butterfly stages: transform length doubles each stage (2, 4, ..., n). Each stage has n/2
    // independent butterflies; thread `tid` owns butterflies tid, tid+nthreads, ... .
    const int nhalf = n >> 1;
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int b = tid; b < nhalf; b += nthreads) {
            const int group = b / half;             // which length-`len` block
            const int k = b - group * half;         // position within the block's first half
            const int i = group * len + k;          // top index of the butterfly pair
            const double ang = -two_pi * static_cast<double>(k) / static_cast<double>(len);
            const double wr = cos(ang);             // forward twiddle e^{-2*pi*i*k/len}
            const double wi = sin(ang);             // sin is negative here (negative exponent)
            const double ur = s[2 * i];
            const double ui = s[2 * i + 1];
            const double vr = s[2 * (i + half)];
            const double vi = s[2 * (i + half) + 1];
            const double tr = wr * vr - wi * vi;    // t = w * lower sample
            const double ti = wr * vi + wi * vr;
            s[2 * i] = ur + tr;                      // butterfly: top = u + t
            s[2 * i + 1] = ui + ti;
            s[2 * (i + half)] = ur - tr;             // bottom = u - t
            s[2 * (i + half) + 1] = ui - ti;
        }
        __syncthreads();  // publish this stage's writes before the next stage reads them
    }
    for (int k = tid; k < n; k += nthreads) {
        out[base + 2 * k] = s[2 * k];  // transform is now in natural order
        out[base + 2 * k + 1] = s[2 * k + 1];
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

extern "C" int nimblecas_gpu_edit_distance_batch(const int* a_flat, const int* a_off,
                                                 const int* b_flat, const int* b_off, int pairs,
                                                 int* out) {
    if (pairs <= 0) {
        return 0;
    }
    // The offset arrays are host memory, so their final entries give the flattened lengths.
    const int a_flat_len = a_off[pairs];
    const int b_flat_len = b_off[pairs];
    const size_t off_bytes = static_cast<size_t>(pairs + 1) * sizeof(int);
    const size_t out_bytes = static_cast<size_t>(pairs) * sizeof(int);
    const size_t a_flat_bytes = static_cast<size_t>(a_flat_len) * sizeof(int);
    const size_t b_flat_bytes = static_cast<size_t>(b_flat_len) * sizeof(int);
    // A zero-length flattened array still needs a valid device allocation to hand the kernel.
    const size_t a_alloc = a_flat_bytes != 0 ? a_flat_bytes : sizeof(int);
    const size_t b_alloc = b_flat_bytes != 0 ? b_flat_bytes : sizeof(int);
    int* dev_a = nullptr;
    int* dev_aoff = nullptr;
    int* dev_b = nullptr;
    int* dev_boff = nullptr;
    int* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    PinnedScope pin_a = host_register(a_flat, a_flat_bytes);
    PinnedScope pin_b = host_register(b_flat, b_flat_bytes);
    PinnedScope pin_out = host_register(out, out_bytes);
    if ((err = cudaMalloc(&dev_a, a_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_aoff, off_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_b, b_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_boff, off_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, out_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_a, a_flat, a_flat_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_aoff, a_off, off_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_b, b_flat, b_flat_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_boff, b_off, off_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(pairs, threads);
        edit_distance_batch_kernel<<<blocks, threads>>>(dev_a, dev_aoff, dev_b, dev_boff, pairs,
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
    if (dev_a != nullptr) {
        cudaFree(dev_a);
    }
    if (dev_aoff != nullptr) {
        cudaFree(dev_aoff);
    }
    if (dev_b != nullptr) {
        cudaFree(dev_b);
    }
    if (dev_boff != nullptr) {
        cudaFree(dev_boff);
    }
    if (dev_out != nullptr) {
        cudaFree(dev_out);
    }
    host_unregister(pin_out);
    host_unregister(pin_b);
    host_unregister(pin_a);
    return rc;
}

extern "C" int nimblecas_gpu_bfs(const int* row_offsets, const int* col_indices,
                                 int num_vertices, int num_edges, int source, int* dist) {
    if (num_vertices <= 0) {
        return 0;
    }
    const size_t row_bytes = static_cast<size_t>(num_vertices + 1) * sizeof(int);
    const size_t dist_bytes = static_cast<size_t>(num_vertices) * sizeof(int);
    const size_t col_bytes = static_cast<size_t>(num_edges) * sizeof(int);
    // An edgeless graph still needs a valid (unused) device allocation for col_indices.
    const size_t col_alloc = col_bytes != 0 ? col_bytes : sizeof(int);
    int* dev_row = nullptr;
    int* dev_col = nullptr;
    int* dev_dist = nullptr;
    int* dev_changed = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    PinnedScope pin_row = host_register(row_offsets, row_bytes);
    PinnedScope pin_col = host_register(col_indices, col_bytes);
    PinnedScope pin_dist = host_register(dist, dist_bytes);
    if ((err = cudaMalloc(&dev_row, row_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_col, col_alloc)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_dist, dist_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_changed, sizeof(int))) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_row, row_offsets, row_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if (num_edges > 0 &&
               (err = cudaMemcpy(dev_col, col_indices, col_bytes, cudaMemcpyHostToDevice)) !=
                   cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemset(dev_dist, 0xFF, dist_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);  // 0xFF bytes set every distance to -1 (unvisited)
    } else {
        // Seed the source at distance 0; an out-of-range source leaves the graph unreachable.
        const int zero = 0;
        if (source >= 0 && source < num_vertices &&
            (err = cudaMemcpy(dev_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice)) !=
                cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            const int threads = 256;
            const int blocks = choose_blocks(num_vertices, threads);
            // At most num_vertices levels can add a new vertex, bounding the host loop.
            for (int level = 0; level < num_vertices; ++level) {
                const int reset = 0;
                if ((err = cudaMemcpy(dev_changed, &reset, sizeof(int),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }
                bfs_kernel<<<blocks, threads>>>(dev_row, dev_col, num_vertices, level, dev_dist,
                                                dev_changed);
                if ((err = cudaGetLastError()) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }
                if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }
                int changed_host = 0;
                if ((err = cudaMemcpy(&changed_host, dev_changed, sizeof(int),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                    break;
                }
                if (changed_host == 0) {
                    break;  // frontier empty: every reachable vertex now has its distance
                }
            }
            if (rc == 0 &&
                (err = cudaMemcpy(dist, dev_dist, dist_bytes, cudaMemcpyDeviceToHost)) !=
                    cudaSuccess) {
                rc = static_cast<int>(err);
            }
        }
    }
    if (dev_row != nullptr) {
        cudaFree(dev_row);
    }
    if (dev_col != nullptr) {
        cudaFree(dev_col);
    }
    if (dev_dist != nullptr) {
        cudaFree(dev_dist);
    }
    if (dev_changed != nullptr) {
        cudaFree(dev_changed);
    }
    host_unregister(pin_dist);
    host_unregister(pin_col);
    host_unregister(pin_row);
    return rc;
}

extern "C" int nimblecas_gpu_nqueens_count(int n, unsigned long long* out_count) {
    *out_count = 0ull;
    if (n <= 0) {
        return 0;  // no board, no solutions
    }
    if (n == 1) {
        *out_count = 1ull;  // the single-cell board has one trivial solution
        return 0;
    }
    // The bitmasks are 32-bit, so boards wider than 31 are out of range for this kernel.
    if (n > 31) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const unsigned int full = (1u << n) - 1u;
    // Enumerate every non-attacking placement of the first two rows on the host; the count is
    // bounded by n*n <= 31*31 < the capacity below.
    constexpr int kMaxPartials = 1024;
    unsigned int part_cols[kMaxPartials];
    unsigned int part_d1[kMaxPartials];
    unsigned int part_d2[kMaxPartials];
    int np = 0;
    for (int c0 = 0; c0 < n; ++c0) {
        const unsigned int b0 = 1u << c0;
        const unsigned int cols1 = b0;
        const unsigned int d1_1 = b0 << 1;  // shift the diagonals into row 1
        const unsigned int d2_1 = b0 >> 1;
        unsigned int free1 = ~(cols1 | d1_1 | d2_1) & full;
        while (free1 != 0u) {
            const unsigned int b1 = free1 & (0u - free1);
            free1 ^= b1;
            if (np < kMaxPartials) {
                part_cols[np] = cols1 | b1;
                part_d1[np] = (d1_1 | b1) << 1;  // shift again into row 2 (kernel start row)
                part_d2[np] = (d2_1 | b1) >> 1;
                ++np;
            }
        }
    }
    if (np == 0) {
        return 0;  // no valid two-row prefix (e.g. n = 2, 3): zero solutions
    }
    const size_t part_bytes = static_cast<size_t>(np) * sizeof(unsigned int);
    unsigned int* dev_pc = nullptr;
    unsigned int* dev_pd1 = nullptr;
    unsigned int* dev_pd2 = nullptr;
    unsigned long long* dev_count = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    if ((err = cudaMalloc(&dev_pc, part_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_pd1, part_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_pd2, part_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_count, sizeof(unsigned long long))) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_pc, part_cols, part_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_pd1, part_d1, part_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_pd2, part_d2, part_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemset(dev_count, 0, sizeof(unsigned long long))) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(np, threads);
        nqueens_kernel<<<blocks, threads>>>(dev_pc, dev_pd1, dev_pd2, np, full, dev_count);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out_count, dev_count, sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dev_pc != nullptr) {
        cudaFree(dev_pc);
    }
    if (dev_pd1 != nullptr) {
        cudaFree(dev_pd1);
    }
    if (dev_pd2 != nullptr) {
        cudaFree(dev_pd2);
    }
    if (dev_count != nullptr) {
        cudaFree(dev_count);
    }
    return rc;
}

extern "C" int nimblecas_gpu_qmc_poly_integrate(const double* coeffs, int n_coeffs,
                                                const double* x, int n, double* out_mean) {
    *out_mean = 0.0;
    if (n <= 0) {
        return 0;  // no sample points: nothing to average (the wrapper rejects the empty case)
    }
    if (n_coeffs <= 0) {
        return 0;  // the zero integrand averages to 0 everywhere
    }
    const int threads = 256;  // power of two: required by the shared-memory tree reductions
    const int blocks = choose_blocks(n, threads);
    const size_t coeff_bytes = static_cast<size_t>(n_coeffs) * sizeof(double);
    const size_t point_bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t part_bytes = static_cast<size_t>(blocks) * sizeof(double);
    const size_t shmem = static_cast<size_t>(threads) * sizeof(double);
    double* dev_coeffs = nullptr;
    double* dev_x = nullptr;
    double* dev_partials = nullptr;
    double* dev_sum = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    // Pin the caller's pageable inputs so the H2D copies use DMA; best-effort and size-gated.
    PinnedScope pin_coeffs = host_register(coeffs, coeff_bytes);
    PinnedScope pin_x = host_register(x, point_bytes);
    if ((err = cudaMalloc(&dev_coeffs, coeff_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_x, point_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_partials, part_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_sum, sizeof(double))) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_coeffs, coeffs, coeff_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_x, x, point_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        qmc_poly_integrate_kernel<<<blocks, threads, shmem>>>(dev_coeffs, n_coeffs, dev_x, n,
                                                              dev_partials);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            // Fold the block partials into a single device value, then bring back one double.
            reduce_partials_kernel<<<1, threads, shmem>>>(dev_partials, blocks, dev_sum);
            if ((err = cudaGetLastError()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
                rc = static_cast<int>(err);
            } else {
                double total = 0.0;
                if ((err = cudaMemcpy(&total, dev_sum, sizeof(double),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                } else {
                    *out_mean = total / static_cast<double>(n);  // equal-weight average
                }
            }
        }
    }
    if (dev_coeffs != nullptr) {
        cudaFree(dev_coeffs);
    }
    if (dev_x != nullptr) {
        cudaFree(dev_x);
    }
    if (dev_partials != nullptr) {
        cudaFree(dev_partials);
    }
    if (dev_sum != nullptr) {
        cudaFree(dev_sum);
    }
    host_unregister(pin_x);
    host_unregister(pin_coeffs);
    return rc;
}

extern "C" int nimblecas_gpu_haar_dwt_batch(const double* data, int batch, int len, double* out) {
    if (batch <= 0 || len <= 0) {
        return 0;  // nothing to transform
    }
    // A single Haar level pairs adjacent samples, so len must be even. The C++ wrapper already
    // rejects an odd len; guard here too so half = len/2 pairs the block exactly.
    if ((len & 1) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int half = len / 2;
    const size_t total_bytes =
        static_cast<size_t>(batch) * static_cast<size_t>(len) * sizeof(double);
    double* dev_in = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    PinnedScope pin_in = host_register(data, total_bytes);
    PinnedScope pin_out = host_register(out, total_bytes);
    if ((err = cudaMalloc(&dev_in, total_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, total_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_in, data, total_bytes, cudaMemcpyHostToDevice)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(batch * half, threads);
        haar_dwt_batch_kernel<<<blocks, threads>>>(dev_in, batch, len, half, dev_out);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, total_bytes, cudaMemcpyDeviceToHost)) !=
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
    host_unregister(pin_out);
    host_unregister(pin_in);
    return rc;
}

extern "C" int nimblecas_gpu_batched_matmul(const double* a, const double* b, double* c, int batch,
                                            int m, int k, int n) {
    // The C++ wrapper rejects non-positive dimensions up front; guard here too so the sizes below
    // and the launch bound are well defined. Nothing to compute means success with no writes.
    if (batch <= 0 || m <= 0 || k <= 0 || n <= 0) {
        return 0;
    }
    const size_t a_count =
        static_cast<size_t>(batch) * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t b_count =
        static_cast<size_t>(batch) * static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t c_count =
        static_cast<size_t>(batch) * static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t a_bytes = a_count * sizeof(double);
    const size_t b_bytes = b_count * sizeof(double);
    const size_t c_bytes = c_count * sizeof(double);
    double* dev_a = nullptr;
    double* dev_b = nullptr;
    double* dev_c = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    PinnedScope pin_a = host_register(a, a_bytes);
    PinnedScope pin_b = host_register(b, b_bytes);
    PinnedScope pin_c = host_register(c, c_bytes);
    if ((err = cudaMalloc(&dev_a, a_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_b, b_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_c, c_bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_a, a, a_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_b, b, b_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = choose_blocks(batch * m * n, threads);
        batched_matmul_kernel<<<blocks, threads>>>(dev_a, dev_b, dev_c, batch, m, k, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(c, dev_c, c_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dev_a != nullptr) {
        cudaFree(dev_a);
    }
    if (dev_b != nullptr) {
        cudaFree(dev_b);
    }
    if (dev_c != nullptr) {
        cudaFree(dev_c);
    }
    host_unregister(pin_c);
    host_unregister(pin_b);
    host_unregister(pin_a);
    return rc;
}

extern "C" int nimblecas_gpu_fft_batch(const double* in, double* out, int batch, int n) {
    // The C++ wrapper rejects these up front; guard here too so the sizes and launch are well
    // defined. Nothing to transform means success with no writes.
    if (batch <= 0 || n <= 0) {
        return 0;
    }
    // n must be a power of two (radix-2) and within the shared-memory length cap. Both are honest
    // preconditions; a violation is a bad shape, reported to the wrapper as an invalid value.
    if ((n & (n - 1)) != 0 || n > kMaxFftLen) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // Total interleaved-double count = batch * 2 * n; the C++ wrapper proves this fits in int.
    const size_t total =
        static_cast<size_t>(batch) * 2u * static_cast<size_t>(n);
    const size_t bytes = total * sizeof(double);
    double* dev_in = nullptr;
    double* dev_out = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    PinnedScope pin_in = host_register(in, bytes);
    PinnedScope pin_out = host_register(out, bytes);
    if ((err = cudaMalloc(&dev_in, bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dev_out, bytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dev_in, in, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;  // more threads than butterflies is fine (grid-stride in-block)
        // One block per signal; each block needs a 2*n-double shared tile for its samples.
        const size_t shmem = static_cast<size_t>(2 * n) * sizeof(double);
        fft_batch_kernel<<<batch, threads, shmem>>>(dev_in, dev_out, batch, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dev_out, bytes, cudaMemcpyDeviceToHost)) !=
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
    host_unregister(pin_out);
    host_unregister(pin_in);
    return rc;
}

// ---------------------------------------------------------------------------
// Batched Black-Scholes-Merton option pricing (finance, ROADMAP 5).
//
// One thread per option, grid-stride. Matches nimblecas::pricing::black_scholes_greeks'
// price formula exactly (same d1/d2, same degenerate T==0/vol==0 collapse to discounted
// intrinsic), so the device result agrees with the CPU closed form to floating-point
// tolerance — the GPU is a batch-valuation MIRROR of the authoritative CPU pricer, never a
// second source of truth. The `_graphed` launcher captures the launch into a CUDA graph
// and replays it, amortizing per-launch overhead across a repeated fixed-shape risk sweep.
// ---------------------------------------------------------------------------
namespace {

__device__ inline double bs_norm_cdf(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475244);  // 0.5 * erfc(-x / sqrt(2))
}

__global__ void black_scholes_kernel(const NimblecasBsOption* __restrict__ opts,
                                     double* __restrict__ out, int n) {
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
         i < n; i += stride) {
        const NimblecasBsOption o = opts[i];
        const double S = o.spot, K = o.strike, r = o.rate, q = o.dividend, v = o.volatility,
                     T = o.time;
        const bool call = o.is_call != 0;
        double price;
        if (T == 0.0 || v == 0.0) {
            const double fwd = S * exp((r - q) * T);
            const double intr = call ? fmax(fwd - K, 0.0) : fmax(K - fwd, 0.0);
            price = exp(-r * T) * intr;
        } else {
            const double sq = sqrt(T);
            const double d1 = (log(S / K) + (r - q + 0.5 * v * v) * T) / (v * sq);
            const double d2 = d1 - v * sq;
            const double dr = exp(-r * T), dq = exp(-q * T);
            price = call ? (S * dq * bs_norm_cdf(d1) - K * dr * bs_norm_cdf(d2))
                         : (K * dr * bs_norm_cdf(-d2) - S * dq * bs_norm_cdf(-d1));
        }
        out[i] = price;
    }
}

// Device-sized grid (a small multiple of the SM count), capped at one block per option.
inline int bs_blocks(int n, int threads) {
    int sm = 0;
    // 64-bit intermediate: n can be up to INT_MAX (the wrapper's cap), so `n + threads - 1`
    // would overflow a signed int. The ceil-divide result fits back in int (<= ~8.4M).
    const long long by_n64 = (static_cast<long long>(n) + threads - 1) / threads;
    const int by_n = by_n64 > 0 ? static_cast<int>(by_n64) : 1;
    if (cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess && sm > 0) {
        const int cap = sm * 32;
        const int b = cap < by_n ? cap : by_n;
        return b < 1 ? 1 : b;
    }
    cudaGetLastError();  // attribute query failed; fall back to one block per option
    return by_n < 1 ? 1 : by_n;
}

}  // namespace

extern "C" int nimblecas_gpu_black_scholes_batch(const NimblecasBsOption* opts, double* out,
                                                 int n) {
    if (n <= 0) {
        return 0;
    }
    const size_t obytes = static_cast<size_t>(n) * sizeof(NimblecasBsOption);
    const size_t pbytes = static_cast<size_t>(n) * sizeof(double);
    NimblecasBsOption* dopts = nullptr;
    double* dout = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    if ((err = cudaMalloc(&dopts, obytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dout, pbytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dopts, opts, obytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        const int threads = 256;
        const int blocks = bs_blocks(n, threads);
        black_scholes_kernel<<<blocks, threads>>>(dopts, dout, n);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaMemcpy(out, dout, pbytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            rc = static_cast<int>(err);
        }
    }
    if (dopts != nullptr) {
        cudaFree(dopts);
    }
    if (dout != nullptr) {
        cudaFree(dout);
    }
    return rc;
}

extern "C" int nimblecas_gpu_black_scholes_batch_graphed(const NimblecasBsOption* opts,
                                                         double* out, int n, int iterations) {
    if (n <= 0) {
        return 0;
    }
    if (iterations < 1) {
        iterations = 1;
    }
    const size_t obytes = static_cast<size_t>(n) * sizeof(NimblecasBsOption);
    const size_t pbytes = static_cast<size_t>(n) * sizeof(double);
    NimblecasBsOption* dopts = nullptr;
    double* dout = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaSuccess;
    int rc = 0;
    const int threads = 256;
    const int blocks = bs_blocks(n, threads);
    if ((err = cudaMalloc(&dopts, obytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMalloc(&dout, pbytes)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaMemcpy(dopts, opts, obytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
        rc = static_cast<int>(err);
    } else if ((err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)) !=
               cudaSuccess) {
        rc = static_cast<int>(err);
    } else {
        black_scholes_kernel<<<blocks, threads, 0, stream>>>(dopts, dout, n);
        if ((err = cudaStreamEndCapture(stream, &graph)) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else if ((err = cudaGraphInstantiate(&exec, graph, 0)) != cudaSuccess) {
            rc = static_cast<int>(err);
        } else {
            for (int it = 0; it < iterations && rc == 0; ++it) {
                if ((err = cudaGraphLaunch(exec, stream)) != cudaSuccess) {
                    rc = static_cast<int>(err);
                }
            }
            if (rc == 0 && (err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                rc = static_cast<int>(err);
            }
            if (rc == 0 &&
                (err = cudaMemcpy(out, dout, pbytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                rc = static_cast<int>(err);
            }
        }
    }
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (dopts != nullptr) {
        cudaFree(dopts);
    }
    if (dout != nullptr) {
        cudaFree(dout);
    }
    return rc;
}
