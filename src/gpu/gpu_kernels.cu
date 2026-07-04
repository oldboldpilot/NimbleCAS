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
        const int a_len = a_off[p + 1] - a_begin;
        const int b_begin = b_off[p];
        // Clamp the column span to the local-memory width; the row span needs no clamp because
        // only the b dimension sizes the arrays.
        const int b_len = min(b_off[p + 1] - b_begin, kMaxEditLen);
        int* prev = row0;
        int* curr = row1;
        for (int j = 0; j <= b_len; ++j) {
            prev[j] = j;  // distance from the empty a-prefix to each b-prefix
        }
        for (int i = 1; i <= a_len; ++i) {
            curr[0] = i;  // distance from the i-prefix of a to the empty b-prefix
            const int ai = a_flat[a_begin + i - 1];
            for (int j = 1; j <= b_len; ++j) {
                const int cost = (ai != b_flat[b_begin + j - 1]);  // branchless 0/1 substitution
                const int del = prev[j] + 1;
                const int ins = curr[j - 1] + 1;
                const int sub = prev[j - 1] + cost;
                curr[j] = min(min(del, ins), sub);
            }
            int* tmp = prev;  // roll the rows without copying
            prev = curr;
            curr = tmp;
        }
        out[p] = prev[b_len];  // final cell after a_len iterations of the roll
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
