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

### Tiling — an ncu-guided, rows-per-program 2-D kernel

The kernel is wrapped in `@triton.autotune` over
`ROWS_PER_PROG ∈ {1,2,4,8,16}` × `BLOCK_SIZE ∈ {16,32,64}` × `num_warps ∈ {1,2,4}`,
keyed on the mean nonzeros-per-row. Each program owns a **tile of
`ROWS_PER_PROG` rows** and processes them with a 2-D `[ROWS_PER_PROG, BLOCK_SIZE]`
index (per-row masking `k < len[r]`), reducing along the nonzero axis.

This shape was chosen from **`ncu` evidence**, not guesswork. A first
one-row-per-program version (`BLOCK_SIZE=32, num_warps=1`) profiled at:

| | one-row-per-program | rows-per-program (2-D) |
| :--- | ---: | ---: |
| Theoretical occupancy | 50 % (block-count-limited) | **100 %** |
| Achieved occupancy | 29.8 % | **91.6 %** |
| DRAM throughput (% of peak) | 35.5 % | **84.9 %** |

`ncu` showed the one-row kernel was **block-count-limited**: a 1-warp block admits
only 24 warps/SM (of 48), capping occupancy at 50 %. Giving each program several
rows raises the warps-per-block, so occupancy reaches 100 % / 91.6 % achieved and
the kernel becomes **DRAM-bandwidth-bound at ~85 % of peak** — which is the
optimum for a memory-bound operation like SpMV (compute-SM % is correspondingly
low, as it should be). Counters were collected with `sudo ncu` (the driver
restricts GPU performance counters to admin users); `nsys` needs no such elevation.

### Measured evidence (RTX 5090, sm_120, CUDA 12.9, fp64)

Kernel-only timing on **GPU-resident** tensors (CUDA events; inputs placed on the
device once, so the figure is the kernel, not host↔device transfer):

| Check | Result |
| :--- | :--- |
| Correctness vs `torch.sparse` CSR SpMV | max abs diff **5–9e-15** (machine precision) |
| Correctness vs dense oracle `A=[[2,0,1],[0,3,0],[1,0,4]] · [1,2,3]` | `[5,6,13]` exact (fp32 & fp64) |
| Throughput, 200k rows × 16 nnz (3.2M nnz), fp64 | **35.8 µs/call, ~1162 GB/s** (tuner: `ROWS_PER_PROG=4, BLOCK=16, warps=2`) |
| Throughput, 500k rows × 32 nnz (16M nnz), fp64 | **211 µs/call, ~948 GB/s** (tuner: `ROWS_PER_PROG=16, BLOCK=32, warps=2`) |

**Interpretation.** ~1162 GB/s is **~68 % of the 5090's ~1.7 TB/s peak** — a **2.7×**
gain over the one-row kernel (436 GB/s), corroborated by the occupancy/throughput
counters above. (An early end-to-end measurement of ~6.7 GB/s was an artifact of
timing the *convenience wrapper*, which re-uploads all inputs on every call — that
rate is the ~64 MB host↔device marshaling, not the kernel.)

**Using it in a loop (e.g. CG).** Because `csr_spmv(...)` moves its arguments to
the device on each call, iterative use should keep `row_ptr`/`col_indices`/
`values`/`x` **GPU-resident** and launch `_csr_spmv_kernel` directly, so the
per-iteration cost is the kernel, not a repeated upload.

It remains a **numerical mirror** (the `tl.sum` tree reorders the CPU's sequential
`std::fma`), so the CPU `csr_matvec` stays authoritative.

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
