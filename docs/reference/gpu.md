# `nimblecas.gpu` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/gpu/gpu.cppm` (module), `src/gpu/gpu_bridge.h` (C ABI),
`src/gpu/gpu_kernels.cu` (CUDA kernels)

Optional **CUDA GPU acceleration** — the first entry of ROADMAP §5 (GPU). It is a
Result-based C++23 wrapper over hand-written CUDA kernels; today it offers batch
polynomial evaluation (Horner's method) on the device. A portable
[Triton](#the-portable-triton-path) counterpart under `python/triton`
JIT-compiles the same computation across GPU architectures without a rebuild.

```cpp
import nimblecas.gpu;
```

Namespace: `nimblecas::gpu`. Depends only on `core` (and, at build time, on
`nvcc` + the CUDA runtime). Because it is opt-in, importers that do not enable
CUDA never see this module.

## Build isolation

The GPU layer is **opt-in** via `-DNIMBLECAS_CUDA=ON`; `scripts/build.sh`
auto-enables it when `nvcc` is on the path (non-sanitizer builds only — the `.cu`
is compiled independently of the sanitized clang/libc++ objects). It is
Linux/macOS-only (`NOT WIN32`).

The key to mixing two toolchains cleanly is that the `.cu` never goes through
CMake's CUDA language. It is compiled by **`nvcc` independently** — a custom
command runs `nvcc -O3 -std=c++17 -arch=native -Xcompiler -fPIC -lib` to produce
a static archive (`libnimblecas_gpu_kernels.a`) — so the global clang/libc++
canonical flags (which `nvcc` cannot parse) never reach it. The kernels are then
reached over the **plain C ABI** in `gpu_bridge.h`, so:

- Only **POD** (raw `double*`/`int`) crosses the `nvcc` ↔ clang boundary — no C++
  standard-library object does, so the two objects link cleanly.
- The `.cu` host code uses **only the CUDA runtime**, no C++ stdlib, so the
  `nvcc`-produced object carries no libstdc++ dependency.

The `nimblecas.gpu` module holds no CUDA types at all: it only marshals
`std::span`/`std::vector` across the bridge and maps CUDA failures onto
`MathError` (Rule 32 — no exceptions).

## Public API

```cpp
[[nodiscard]] auto device_count() -> int;    // CUDA-capable devices (0 if none)
[[nodiscard]] auto available() -> bool;      // device_count() > 0
[[nodiscard]] auto poly_eval(std::span<const double> coeffs,
                             std::span<const double> x)
    -> Result<std::vector<double>>;
```

- **`device_count()`** — the number of CUDA-capable devices detected, `0` when no
  GPU or CUDA runtime is present.
- **`available()`** — whether at least one device is present (`device_count() > 0`).
- **`poly_eval(coeffs, x)`** — evaluate the polynomial `coeffs` (**low degree
  first**, so `coeffs[0]` is the constant term) at every point in `x`, returning
  the vector of `p(x_i)`. This is the flagship, documented in full below.

### Additional batch kernels

The module has grown into a batch-numeric offload surface. Every entry is a
Result-based wrapper over a hand-written CUDA kernel reached across the `gpu_bridge.h`
C ABI, and every one is cross-checked against a CPU reference in `tests/gpu_tests.cpp`:

| Function | Purpose |
| :--- | :--- |
| `edit_distance_batch(a_flat, a_off, b_flat, b_off, ...)` | Levenshtein distance over a batch of sequence pairs. |
| `bfs(row_offsets, col_indices, n, source)` | Single-source BFS distances on a CSR graph. |
| `nqueens_count(n)` | Exact solution count for the N-queens problem. |
| `qmc_poly_integrate(coeffs, points)` | Quasi-Monte-Carlo polynomial integration (device reduction). |
| `haar_dwt_batch(data, batch, len)` | Batched Haar discrete wavelet transform. |
| `batched_matmul(a, b, batch, ...)` | Batched dense matrix multiply. |
| `fft_batch(in, batch, n)` | Batched FFT (power-of-two length). |
| `black_scholes_batch(opts)` | **Batched Black-Scholes-Merton pricing** — one thread per `BsOption`, grid-stride. Mirrors `pricing::black_scholes_price` to FP tolerance. |
| `black_scholes_batch_graphed(opts, iterations)` | Same result, captured into a **CUDA graph** and replayed `iterations` times (a fixed-shape risk sweep); the replay is bit-identical to the direct launch. |

All eight numeric kernels plus the batched Black-Scholes pricer execute and pass on the
**RTX PRO 6000 Blackwell** and **RTX 5090** (sm_120, CUDA 13.2, nvcc `-arch=native`); see
[Testing](#testing).

### Finance: batched Black-Scholes and the CUDA-graph path

`black_scholes_batch` prices a whole option grid (a chain, or a price-vs-spot sweep) in one
launch, taking a span of the POD `BsOption { spot, strike, rate, dividend, volatility, time,
is_call }`. The device formula is identical to
[`pricing`](pricing.md)'s `black_scholes_greeks` (same `d1`/`d2`, same degenerate
`T==0`/`σ==0` collapse to discounted intrinsic), so **the GPU is a batch-valuation mirror of
the authoritative CPU pricer, never a second source of truth**. `black_scholes_batch_graphed`
captures the kernel launch into a **CUDA graph** and replays it on persistent device buffers,
amortizing per-launch overhead across repeated re-pricing of a fixed-shape grid — its output
is bit-identical to the direct launch (a checked invariant). A non-physical option
(`spot<=0`, `strike<=0`, `time<0`, `σ<0`) → `MathError::domain_error` (validated on the host
before any launch); no device / a CUDA failure → `MathError::gpu_error`.

Profiling (`nsys`/`ncu`, RTX PRO 6000 Blackwell, CUDA 13.2) confirms the honest picture: at a
small grid the kernel is **launch-latency-bound** (SM ~0.1 %, DRAM ~0.2 % of peak, ~11 µs) —
a few transcendentals per option cannot saturate a Blackwell part, so the GPU path is worth
it only at large batch sizes, and the CUDA-graph replay (visible as repeated
`cudaGraphLaunch` in the `nsys` timeline) is what removes per-launch overhead when the same
grid is re-priced many times.

### Error model

| Condition | Result |
| :--- | :--- |
| No device present, or a CUDA call fails | `MathError::gpu_error` |
| A size exceeds the `int` kernel bound (`coeffs.size()` or `x.size()` > `INT_MAX`) | `MathError::overflow` |
| Empty `x` | empty vector (nothing to evaluate) |

## Kernel design

The device kernel (`poly_eval_kernel`) is a **grid-stride Horner loop** over
`__restrict__` pointers:

```cpp
acc = 0;
for (int k = n_coeffs - 1; k >= 0; --k) acc = acc * xi + coeffs[k];
```

Each thread walks the point array in steps of the total grid size. Rather than
launching one block per point, the host launches a **device-sized grid** — a
small multiple of the SM count (`sm_count * 32` blocks, `threads = 256`, capped
by the number of points) — so any `n` saturates the SMs and no thread sits idle
on small `n`. If the SM-count attribute query fails, it falls back to one block
per point. The coefficient array is tiny and hot, so it stays in L2/registers
across the stride; the per-point cost is dominated by streaming loads of `x` and
stores of `out`.

Transfers use **pinned-memory DMA**: before each copy the host buffers are
registered with `cudaHostRegister` (paired with `cudaHostUnregister` after the
copies) so the H2D/D2H copies use DMA instead of a staging bounce. Registration
has a fixed per-call cost, so it is **size-gated at 256 KB** (`kPinThresholdBytes`)
with a graceful **pageable fallback** — a failed registration clears the sticky
CUDA error and simply leaves the buffer pageable.

Two degenerate inputs are handled in the bridge without a launch: `n <= 0`
returns immediately, and `n_coeffs <= 0` (the zero polynomial) writes `0.0` to
every output.

### Profiling result

Nsight (`nsys`/`ncu`) profiling showed the path is **transfer-bound** — the
H2D+D2H copies cost about **2×** the kernel. The two optimizations above (the
device-sized grid-stride loop and pinned-memory DMA) raised end-to-end
throughput from **~12.6 to ~17.2 GB/s** (**~1.36×**, at 50M points) with
correctness preserved (max relative error ≤ `1.7e-16`). Measured on the
**RTX PRO 6000 Blackwell** (sm_120, CUDA 13.2).

## The portable Triton path

`python/triton/poly_eval.py` provides a `@triton.jit` Horner kernel that
complements the in-engine CUDA path. Because Triton **JIT-compiles per device**,
the same source runs across GPU architectures without a rebuild — it
accommodates different kinds of GPU (for example an sm_120 Blackwell part and an
sm_90 Hopper part). The kernel tiles the input into `BLOCK_SIZE`-wide programs
over a 1-D program-id grid and masks the ragged tail; `n_coeffs` is a
`tl.constexpr`, so the Horner loop is unrolled and specialized per polynomial
degree. Both `float32` and `float64` are supported.

Verified on Blackwell sm_120 (torch 2.12.1+cu130, triton 3.7.1): all cases pass
at **~209,800 Melem/s** (float32, kernel-only, 20M points). As with the CUDA
path, `nsys` shows the work is transfer-bound; the benchmark isolates the kernel
by keeping data **device-resident** across calls.

### Triton financial Monte Carlo (`python/triton/mc_option.py`)

`mc_option.py` is a `@triton.jit` European-option Monte-Carlo kernel: each program
simulates a block of terminal prices under geometric Brownian motion and reduces to
partial sum / sum-of-squares, which the host combines into a price ± standard error.
It **accelerates the same computation as the in-engine `pricing::monte_carlo_european`**
(the CAS CPU MC) rather than reimplementing the model — and it validates against the
Black-Scholes closed form. Verified on the **RTX 5090** (torch 2.13.0+cu130, triton
3.7.1): 8,000,000 paths → **10.45238 ± 0.00260**, agreeing with Black-Scholes
(10.45058) to **0.69 standard errors**.

### Triton batched Black-Scholes (`python/triton/black_scholes.py`)

`black_scholes.py` is the closed-form Triton sibling of `mc_option.py` and the portable
counterpart of the in-engine CUDA `black_scholes_batch`: a `@triton.jit` kernel that prices
one `BLOCK_SIZE`-wide tile of options per program (`d1`/`d2`, `Φ` via `tl.erf`, the degenerate
`T==0`/`σ==0` branch). It **mirrors the CPU closed form** rather than reimplementing the
model. Verified on the **RTX 5090** (sm_120, torch 2.13.0+cu130, triton 3.7.1): a 10-option
grid agrees with the CPU closed form to a max `|error|` of **7.1e-15**, with the ATM 1-year
call at **10.45058** (the textbook value).

The Triton kernels live under `python/triton` and run via the managed venv,
complementing — not replacing — the in-engine CUDA and CPU paths.

### CuTile variant — status

The GPU acceleration variants requested for the finance path are: **CUDA** (`black_scholes_batch`)
and **CUDA Graphs** (`black_scholes_batch_graphed`) — both implemented in-engine and validated
on Blackwell above — and **Triton** (`black_scholes.py` + `mc_option.py`) — verified on the
RTX 5090. The **CuTile** variant (NVIDIA's tile-based CUDA DSL) is **not yet implemented**: it
requires the `cuda-python` / cuTile toolchain, which is not currently provisioned in the build
venvs (only `torch`/`triton` are). It is a bounded follow-up — port the same tile structure as
the Triton kernel onto cuTile once that toolchain is installed — and is tracked as the one open
item of ROADMAP §5 GPU for finance. This module does not ship an unverified CuTile kernel.

## Testing

`tests/gpu_tests.cpp` cross-checks **every** GPU kernel against a CPU reference:
poly-eval (small/large/edge cases), batched edit-distance, CSR BFS, N-queens count,
QMC polynomial integration, batched Haar DWT, batched matmul, and batched FFT (vs a
CPU DFT) — **12 groups, 41 checks**. It uses relative tolerances because the GPU may
contract to FMA where the CPU does not. The suite is built and run **only with
`-DNIMBLECAS_CUDA=ON` on a machine with a CUDA device**.

Verified green on the **RTX 5090** (sm_120, CUDA 13.2): all 12 groups pass, and an
`nsys` trace confirms all nine kernels launch and the pipeline is transfer-bound.
`ncu` deep-counter profiling on that host requires elevated GPU-counter permissions
(`ERR_NVGPUCTRPERM` — the `NVreg_RestrictProfilingToAdminUsers=0` driver setting or
root), which was unavailable; `nsys` needs no such permission and profiled cleanly.

## Example

```cpp
import nimblecas.gpu;
import std;
using namespace nimblecas;

// p(x) = 1 + 2x + 3x^2, evaluated at many points on the GPU.
const std::vector<double> coeffs = {1.0, 2.0, 3.0};
const std::vector<double> x = {0.0, 1.0, 2.0, -1.0, 3.5};

if (gpu::available()) {
    auto r = gpu::poly_eval(coeffs, x);
    if (r) {
        for (double y : *r) std::println("{}", y);   // 1, 6, 17, 2, 43.75
    }
}
```

Each `poly_eval` call copies its inputs to the device and the result back, and
the profiling above shows that transfer dominates. Keeping data **device-resident
across operations** — so a chain of evaluations amortizes a single H2D/D2H — is
therefore the path past the transfer bound, and a documented future direction for
this layer.

## See also

- [`nimblecas.simd`](simd.md) — the CPU numeric fast path (runtime-dispatched SIMD).
- [`nimblecas.polynomial`](polynomial.md) — dense polynomials and SIMD batch evaluation.
- [Parallel tree computation §6](../architecture/parallel-tree-computation.md) — linearization, the bridge to SIMD/GPU.
- [Documentation hub](../Index.md)
