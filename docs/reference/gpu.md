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
  the vector of `p(x_i)`.

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

The Triton kernels live under `python/triton` and run via the uv-managed venv,
complementing — not replacing — the in-engine CUDA path.

## Testing

`tests/gpu_tests.cpp` cross-checks the GPU batch evaluation against a **CPU
Horner reference** (matching the kernel's coefficient order) across a small
polynomial, a large batch (exercising multiple blocks), and the empty /
constant-polynomial edge cases. It uses a relative tolerance because the GPU may
contract to FMA where the CPU does not. The suite is built and run **only with
`-DNIMBLECAS_CUDA=ON` on a machine with a CUDA device**.

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
