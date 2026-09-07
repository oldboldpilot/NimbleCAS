// NimbleCAS CUDA GPU kernels — the batched first-argument clause probe.
// @author Olumuyiwa Oluwasanmi
//
// Compiled independently by nvcc (see the NIMBLECAS_CUDA block in CMakeLists.txt) and linked
// into the nimblecas.gpu module through the plain C ABI in gpu_bridge.h. Host code here uses
// ONLY the CUDA runtime — no C++ standard library — so the nvcc-produced object carries no
// libstdc++ dependency and links cleanly with the clang/libc++ engine. In particular nothing
// here allocates, throws, or touches std::; that rule has bitten this repo before.
//
// WHY THIS KERNEL AND NOT "PROLOG ON THE GPU".
//
// SLD resolution is a data-dependent backtracking walk: every goal takes a different path,
// which is the worst case for SIMT. Putting the search itself on a GPU would be slower than
// the CPU and far harder to trust, so this module does not pretend to.
//
// What IS regular is deciding WHICH CLAUSES a goal could match. That is one integer compared
// against a contiguous array of them, branch-free and data-independent — and a speculative or
// distributed engine probes many goals at once, so the batch is naturally wide. One block per
// goal, threads striding the clause table, `__ballot_sync` folding 32 lanes into a bitmask
// word: the shape the hardware is built for.
//
// THE CPU IS AUTHORITATIVE. `nimblecas.logic_index` defines the answer; this kernel must
// produce the SAME BITS. It is a performance mirror, not a second opinion, and the tests
// compare it against the CPU result rather than against its own expectations.
//
// KEY SEMANTICS, identical to the CPU path: key 0 means UNKNOWN and matches everything; two
// non-zero keys are candidates only when equal. The filter may only rule out a clause it can
// PROVE cannot match — a false negative silently loses an answer, while a false positive costs
// one failed unification.

#include <cuda_runtime.h>

#include "gpu_bridge.h"

namespace {

// One block per goal; the block's threads stride the clause array. Each warp folds its 32
// lanes into 32 bits with __ballot_sync, and lane 0 of each warp writes the half-word it owns,
// so a 64-clause word is written by exactly two warps with no atomics and no contention.
__global__ void index_probe_kernel(const unsigned long long* __restrict__ clause_keys,
                                   int clause_count,
                                   const unsigned long long* __restrict__ goal_keys,
                                   int goal_count, int row_words,
                                   unsigned long long* __restrict__ out_words) {
    const int goal = blockIdx.x;
    if (goal >= goal_count) {
        return;
    }
    const unsigned long long gk = goal_keys[goal];
    unsigned long long* row = out_words + static_cast<long long>(goal) * row_words;

    // Each iteration handles 64 clauses — one output word — using two warps' worth of lanes.
    for (int word = 0; word < row_words; ++word) {
        const int base = word * 64;
        unsigned long long bits = 0ULL;
        for (int half = 0; half < 2; ++half) {
            const int start = base + half * 32;
            const int lane = threadIdx.x & 31;
            const int warp = static_cast<int>(threadIdx.x) >> 5;
            // Only the first warp of the block participates, so the ballot is over a known set
            // of lanes regardless of how the launch was configured.
            unsigned int mask = 0U;
            if (warp == 0) {
                const int index = start + lane;
                bool candidate = false;
                if (index < clause_count) {
                    const unsigned long long ck = clause_keys[index];
                    candidate = (gk == 0ULL) || (ck == 0ULL) || (ck == gk);
                }
                mask = __ballot_sync(0xFFFFFFFFU, candidate);
            }
            if (threadIdx.x == 0) {
                bits |= static_cast<unsigned long long>(mask) << (half * 32);
            }
        }
        if (threadIdx.x == 0) {
            row[word] = bits;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int nimblecas_gpu_index_probe_batch(const unsigned long long* clause_keys,
                                               int clause_count,
                                               const unsigned long long* goal_keys,
                                               int goal_count, unsigned long long* out_words) {
    if (clause_count < 0 || goal_count < 0) {
        return -1;
    }
    if (clause_count == 0 || goal_count == 0) {
        return 0;  // nothing to do is not a failure
    }
    if (clause_keys == nullptr || goal_keys == nullptr || out_words == nullptr) {
        return -1;
    }

    const int row_words = (clause_count + 63) / 64;
    const size_t ck_bytes = static_cast<size_t>(clause_count) * sizeof(unsigned long long);
    const size_t gk_bytes = static_cast<size_t>(goal_count) * sizeof(unsigned long long);
    const size_t out_bytes =
        static_cast<size_t>(goal_count) * static_cast<size_t>(row_words) * sizeof(unsigned long long);

    unsigned long long* d_ck = nullptr;
    unsigned long long* d_gk = nullptr;
    unsigned long long* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_ck, ck_bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMalloc(&d_gk, gk_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_ck);
        return static_cast<int>(err);
    }
    err = cudaMalloc(&d_out, out_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_ck);
        cudaFree(d_gk);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(d_ck, clause_keys, ck_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_gk, goal_keys, gk_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        // One block per goal. 32 threads is exactly the warp the ballot folds, and more would
        // idle: the kernel's parallelism is across goals, not within a word.
        index_probe_kernel<<<goal_count, 32>>>(d_ck, clause_count, d_gk, goal_count, row_words,
                                               d_out);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(out_words, d_out, out_bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_ck);
    cudaFree(d_gk);
    cudaFree(d_out);
    return static_cast<int>(err);
}
