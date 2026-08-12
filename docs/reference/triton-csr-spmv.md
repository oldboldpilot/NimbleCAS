# Triton CSR SpMV + the Feature-7 SIMD finding — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `python/triton/csr_spmv.py` (+ `python/triton/test_csr_spmv.py`)

This is the **GPU/Triton performance mirror** of the CPU sparse matrix-vector
product `nimblecas.krylov::csr_matvec` (`src/krylov/krylov.cppm`), authored under
the profiling-gated optimization policy (Code Policy Rules 43/58/59: **no speedup
claim without evidence**). It records both what was shipped and — honestly — what
was *not*, and why.

## `csr_spmv` — the Triton kernel

`csr_spmv(row_ptr, col_indices, values, x, *, dtype=None, block_size=128)` computes
`y = A·x` for `A` in CSR form, on a CUDA device, in `float32` **or** `float64`. The
`@triton.jit` kernel uses one program per row: it loads the row's `[start, end)`
extent, gathers `values[k]` and `x[col_indices[k]]` in `BLOCK_SIZE`-wide masked
chunks, and reduces with `tl.sum`. Because Triton JIT-compiles per device, the same
source runs on any CUDA GPU (e.g. sm_120 Blackwell, sm_90 Hopper) without a rebuild,
complementing the raw-CUDA kernels in `src/gpu/gpu_kernels.cu`.

### Honesty boundary — a *numerical mirror*, not a bit-identical one

The authoritative path is the CPU `csr_matvec`, which accumulates each row with a
**sequential fused multiply-add** (`std::fma`, one rounding per term). The GPU
kernel reduces with a `tl.sum` **tree** in a different order, so results may differ
in the last bits — each a valid double. **The CPU path stays authoritative**; the
GPU kernel is a numerical mirror, exactly as `gpu::cg_csr` and the other
`nimblecas.gpu` kernels are.

### Measured evidence (RTX 5090, sm_120, CUDA 12.9, fp64)

| Check | Result |
| :--- | :--- |
| Correctness vs `torch.sparse` CSR SpMV (200k rows, 3.2M nnz) | max abs diff **7.1e-15** (machine precision) |
| Correctness vs dense oracle `A=[[2,0,1],[0,3,0],[1,0,4]] · [1,2,3]` | `[5,6,13]` exact (fp32 & fp64) |
| Throughput (200k rows, 16 nnz/row, fp64) | **~6.2 ms/call, ~6.7 GB/s** |

**Interpretation (honest).** 6.7 GB/s is a small fraction of the 5090's ~1.7 TB/s
peak: the one-program-per-row tiling is **memory-latency-bound on the random
`x[col]` gather** (poor coalescing, low occupancy for small `nnz/row`). So the
kernel is shipped as a **correctness-verified, portable GPU mirror** — *not* a
demonstrated speedup over the CPU. A performance-competitive version would need a
warp-per-row (or vectorized/coalesced) tiling and is deliberately left as future
work rather than claimed here.

## The SIMD leg — an evidence-backed *non*-deliverable

A CPU SIMD `csr_matvec` was scoped but **not shipped**, for a principled reason:

1. `csr_matvec`'s inner loop is a **fused-fma reduction over a gather**
   (`acc = fma(values[k], x[col[k]], acc)`). Any lane-parallel accumulation
   **reorders** that sum, and splitting the fma into a separate multiply and add
   **doubles the rounding** — so a SIMD version **cannot be bit-identical** to the
   sequential reference.
2. `nimblecas.simd` is **elementwise-only by design** (`add`/`mul`/`axpy`/
   `exp_into`/`log_into`) — it exposes *no* dot/reduce primitive, precisely so its
   AVX-512 → AVX2 → scalar waterfall stays bit-identical across ISAs. Reductions are
   intentionally out of scope.
3. A numerical-mirror SIMD variant (scalar gather → `simd::mul` → sequential sum)
   adds a gather pass and a temp buffer whose overhead is not repaid for the small
   `nnz/row` typical of these systems.

Per Rules 43/58/59, the correct outcome is therefore a **measured decision not to
ship**, keeping the fused-fma scalar `csr_matvec` authoritative, rather than adding
a slower or non-bit-identical SIMD path for its own sake. The GPU/Triton mirror is
the shippable perf artifact for SpMV; the already-shipped SIMD work in the codebase
targets **elementwise** transcendentals (`exp_into`/`log_into`), where vectorization
*is* bit-identical.

## Running

```bash
# On a CUDA host with the project .venv (torch + triton):
.venv/bin/python python/triton/test_csr_spmv.py      # correctness (skips cleanly with no GPU)
```

## See also

- [`nimblecas.gpu`](gpu.md) — the raw-CUDA kernels, including `cg_csr` (the CUDA
  conjugate-gradient solver that consumes CSR SpMV).
- [Documentation hub](../Index.md)
