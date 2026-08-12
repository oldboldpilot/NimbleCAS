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
| `monte_carlo_european_batch(opts, paths, seed)` | **Batched reproducible European Monte Carlo** — price ± standard error per option; bit-identical Threefry draws vs the CPU counter RNG, fixed-segment index-order reduction, result a pure function of (opts, paths, seed). Equals `pricing::monte_carlo_european` to 1e-6 (documented FP-reassociation/device-libm bound), never bit-for-bit. CPU fallback: `monte_carlo_european_parallel`. |
| `monte_carlo_asian_batch(opts, paths, steps, seed)` | **Batched reproducible Asian Monte Carlo** — arithmetic-average price ± standard error per option over `steps` dates; Threefry draw *bits* bit-identical to the CPU counter RNG (inverse-normal bit-identical in the central ~95%, ≤1 ULP in the ~5% tail), matches CPU `monte_carlo_asian(..., false)` to ~1e-6. Control variate (CV) is CPU-only. CPU fallback: `monte_carlo_asian(..., false)`. |
| `barrier_option_mc_batch(opts, barrier, knock_in, paths, steps, seed)` | **Batched reproducible barrier option Monte Carlo** — price ± standard error per option; Threefry draw *bits* bit-identical vs CPU `barrier_option_mc` (inverse-normal ≤1 ULP in the ~5% tail), fixed-segment index-order reduction. Equals CPU to ~1e-5 on non-grazing barriers; grazing-divergence caveat noted (the `knock_in + knock_out == vanilla` sum stays grazing-immune to ~1e-6). CPU fallback: `pricing::barrier_option_mc`. |
| `longstaff_schwartz_american_batch(opts, paths, steps, seed)` | **Batched reproducible Longstaff-Schwartz American Monte Carlo** — price ± standard error per option over `steps` exercise dates; Threefry draw *bits* bit-identical vs CPU `pricing::longstaff_schwartz_american` (inverse-normal ≤1 ULP in ~5% tail, grid S ~1e-6). Matches CPU price to ~1e-3 relative due to ill-conditioned 3x3 normal-equations regression and exact exercise-threshold sensitivity. Deterministic (pure function of inputs). CPU fallback: `pricing::longstaff_schwartz_american`. |
| `black_scholes_greeks_batch(opts)` | Batched analytic BS Greeks (price/delta/gamma/vega/theta/rho), mirror of `pricing::black_scholes_greeks` incl. the degenerate `T==0`/`σ==0` branch; matches the CPU to 1e-9 relative. CPU fallback: the CPU closed form. |
| `black_scholes_extended_greeks_batch(opts)` | Batched 13-field extended Greeks, mirror of `pricing::black_scholes_extended_greeks` **including** its central-finite-difference charm/color/veta/vera; matches the CPU to 1e-7. CPU fallback: the CPU implementation. |
| `strategy_payoff_grid(legs, grid)` / `strategy_pnl_grid(legs, grid)` | **Exact piecewise-linear strategy sweep** — aggregate expiry payoff / P&L of an `optstrat` leg bag at every grid price; non-contracted device arithmetic reproduces the CPU double evaluation bit-for-bit (validated to 1e-12, expected equal). CPU fallback: `OptionStrategy::payoff_at`/`pnl_at`. |
| `futures_pnl_grid(legs, grid)` | Exact linear futures-strategy sweep, mirror of `FuturesStrategy::pnl_at_uniform`, same bit-equality contract. CPU fallback: the CPU evaluation. |

Every GPU entry point — the numeric kernels, the batched Black-Scholes pricer, and the Asian and
barrier path-dependent Monte-Carlo pricers — builds and passes `gpu_tests` under
`-DNIMBLECAS_CUDA=ON` on the **mgpu** host (**RTX PRO 6000 Blackwell** / **RTX 5090**, sm_120,
nvcc `-arch=native`); see [Testing](#testing).

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

### Derivatives batch pricing: reproducible Monte Carlo and batch Greeks

`monte_carlo_european_batch` prices a batch of European options by device path simulation.
Its honesty boundary is STATISTICAL: each estimate is returned with its standard error, and
the reproducibility contract of the CPU engines is preserved on the device — the kernel
ports `nimblecas.rng`'s Threefry-2x64-20 counter core bit-for-bit (pure integer ops), draws
path `i` from counter index `i` under `key = splitmix64(seed)` exactly like
`pricing::monte_carlo_european`, decomposes the path range into FIXED 4096-path segments
summed serially in index order, and folds segment partials with a fixed-shape 256-thread
reduction. The result is therefore a pure function of `(opts, paths, seed)` — independent
of grid/block geometry, identical across repeated calls — and equals the CPU pricer to a
DOCUMENTED 1e-6 absolute bound (sources of divergence: summation association order, plus
last-bit differences of the device `exp`/`log` against the CPU's `simd::exp`/`simd::log_one`
in the ~5 % Acklam tail region; the central ~95 % of inverse-normal values are bit-identical
`fma` chains). It is validated against BOTH the CPU MC (1e-6) and the Black-Scholes closed
form (4 standard errors). `black_scholes_greeks_batch` and
`black_scholes_extended_greeks_batch` are elementwise closed-form mirrors of the CPU
`pricing` functions (the extended set reproduces the CPU's central-finite-difference
charm/color/veta/vera rather than re-deriving them), validated to 1e-9 and 1e-7.

`monte_carlo_asian_batch` prices a batch of arithmetic-average Asian options by GPU path simulation
over `steps` averaging dates. Path `p` step `t` draws counter index `p * steps + t` under
`key = splitmix64(seed)`, so the Threefry draw *bits* are bit-identical to CPU
`pricing::monte_carlo_asian`; the inverse-normal is bit-identical in Acklam's central ~95% region
and differs by ≤1 ULP in the ~5% tail (device libm `log` vs CPU `simd::log_one`). Price and standard
error agree with CPU `monte_carlo_asian(spec, paths, steps, seed, false)` to ~1e-6 relative (the
divergence is transcendental hardware `exp` vs CPU `simd::exp_into`, plus the tail-`z` last bit). The
geometric control variate (CV) is CPU-only (GPU v1 is arithmetic-only).

`barrier_option_mc_batch` prices a batch of single-barrier options (knock-in or knock-out, up or down) by device path simulation. Its draw indexing for path `p` step `t` uses counter `p * steps + t` under `key = splitmix64(seed)`, so the Threefry draw *bits* are bit-identical to CPU `pricing::barrier_option_mc` — the inverse-normal `z[t]` is bit-identical in the central ~95% and differs by ≤1 ULP in the ~5% tail (device libm `log` vs CPU `simd::log_one`), so the sampled `z[t]` are **not** claimed bit-identical. The honesty boundary is STATISTICAL: prices match `pricing::barrier_option_mc` to ~1e-5 relative on non-grazing cases (CUDA hardware `exp` vs CPU `std::exp`, plus the tail-`z` last bit). **Barrier-grazing divergence caveat:** because knock detection is an exact threshold check (`s <= barrier` or `s >= barrier`), paths that graze extremely close to the barrier level can knock differently on GPU vs CPU due to tiny ULP-level path differences, causing occasional whole-path payoff divergence. Prefer barriers away from a dense grazing band; the grazing-immune identity `knock_in + knock_out == vanilla` holds per path (its GPU sum tracks the CPU sum to ~1e-6 regardless of flips). CPU fallback (when no device is present): `pricing::barrier_option_mc` per option.

`longstaff_schwartz_american_batch` prices a batch of American options by GPU path simulation and backward induction (Longstaff-Schwartz regression). Path `p` step `t` draws counter index `p * steps + t` under `key = splitmix64(seed)`, so Threefry draw *bits* are bit-identical to CPU `pricing::longstaff_schwartz_american`; inverse-normal `z[t]` is bit-identical in the central ~95% and differs by ≤1 ULP in the ~5% tail, with stored grid `S` agreeing to ~1e-6 relative. However, the American **price** matches the CPU oracle to ~1e-3 relative (rather than ~1e-6) due to two structural amplifiers: (a) the 3x3 normal-equations regression on basis `{1, s, s^2}` is ill-conditioned (cond ~1e8-1e12) and summation-order-sensitive, so regression coefficients differ beyond ULP level, and (b) `ex > cont` exercise decisions are exact threshold checks, so paths near the boundary can flip exercise decisions between GPU and CPU, altering whole-path cashflows and cascading into earlier regressions. The GPU result is itself 100% deterministic (a pure function of inputs). Robust flip-immune invariant gates (intrinsic lower bound, European put lower bound, q=0 American call equality, strike monotonicity, bitwise repeatability) verify correctness. CPU fallback: `pricing::longstaff_schwartz_american` per option.

**Fallback contract (new with these entry points):** when NO CUDA device is present, these
functions compute the result on the CPU via `nimblecas.pricing` and return real values —
the GPU remains a mirror, never a gatekeeper. A CUDA failure on a machine that has a device
is still an honest `MathError::gpu_error`. With `-DNIMBLECAS_CUDA=OFF` the module is not
built at all (unchanged).
### Strategy payoff / P&L grid sweeps (exactly reproducible)

`strategy_payoff_grid` / `strategy_pnl_grid` sweep an `optstrat` leg bag (calls, puts,
underlying — pass `strategy.legs()`) across a grid of terminal prices, one thread per grid
point, legs summed in span order; `futures_pnl_grid` does the same for a `futures` leg bag
under uniform settlement. Unlike the statistical Monte Carlo path, this is an
EXACTLY-REPRODUCIBLE computation: the expiry P&L is exactly piecewise-linear (optstrat's
honesty boundary), the kernels use only non-contracted IEEE-754 double intrinsics
(`__dadd_rn`/`__dsub_rn`/`__dmul_rn`/`fmax`) in the same operation order as
`OptionStrategy::payoff_at`/`pnl_at` and `FuturesStrategy::pnl_at_uniform`. Those CPU
accumulators are themselves pinned non-contracted (`#pragma clang fp contract(off)`), so
pinning *both* sides to the identical rounding sequence makes the device values EQUAL to the
CPU reference **bit-for-bit**, independent of the compiler's default FMA contraction (tests
validate to 1e-12, pin hand-computed breakeven/floor/cap points with exact `==`, and include
a non-representable-product book asserted bit-exact against the CPU oracle). The same fallback
contract applies: no device → the CPU optstrat/futures evaluation, real values returned.

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

### cuTile tiled variant (`python/cutile/black_scholes.py`)

The fourth GPU flavour of the batched pricer, completing the requested set — **CUDA**
(`black_scholes_batch`), **CUDA Graphs** (`black_scholes_batch_graphed`), **Triton**
(`black_scholes.py`), and now **cuTile**. Each thread block cooperatively stages a **tile** of
`tile_size` contracts into **shared memory** (six `double` lanes), prices the tile, and stores
it back — the canonical cuTile *load → compute → store* pattern — so a whole option array is
valued in one launch. Like every other GPU path it **mirrors the CPU closed form** (same
`d1`/`d2`, same degenerate `T==0`/`σ==0` branch), never a second source of truth.

**Honesty on the "cuTile" name.** NVIDIA's public cuTile *DSL* (a `@cutile.jit` Python
front-end) is not yet released for this CUDA 13.x / sm_120 stack — the `cutile` PyPI name is a
474-byte empty placeholder, no `tile` surface ships under `cuda.*`, and the CUDA 13.2 toolkit
headers carry no cuTile intrinsics. Rather than fabricate a kernel against an API that does not
exist, this delivers the cuTile *programming model* — cooperative shared-memory tiles — on
NVIDIA's officially-shipping CUDA-Python runtime-compilation stack (`cuda-core` 1.1.0 /
`cuda-bindings` 13.3.1): it JIT-compiles a tiled CUDA C++ kernel with `cuda.core.Program` and
launches it with `cuda.core.launch` on the current device. When the cuTile DSL lands, only the
kernel body's surface syntax changes; the tiled decomposition, the host API, and the
verification harness stay identical.

Because Black-Scholes is elementwise, each lane reads only the tile slot it wrote — the
shared-memory tile carries **no cross-lane data reuse**, so it is *not* a bandwidth
optimization (it adds one global→shared→register hop). It is present to express the cuTile
decomposition faithfully; the measured time below already includes that staging, so the
throughput figure is honest, and a naive global-only kernel would produce identical prices.

Verified on the **RTX 5090** (sm_120, CUDA 13.2, `cuda-core` 1.1.0, torch 2.13.0+cu130): a
10-option grid agrees with the CPU closed form to a max `|error|` of **7.1e-15**, ATM 1-year
call **10.45058** (textbook). `nsys` profiling at **1,000,000 options** shows the tiled kernel
at a stable **~262 µs/launch** (median 261,728 ns, StdDev 281 ns; 99.8 % of GPU time) — moving
~64 MB (7 loads + 1 store per option) in that window, an honest **~245 GB/s effective**, ~14 %
of the 5090's ~1.8 TB/s peak: memory/occupancy-bound at this size, not compute-bound, exactly
as expected for a handful of transcendentals per option. `ncu` deep counters need elevated
GPU-counter permissions (`ERR_NVGPUCTRPERM`), unavailable on this host, so no occupancy figure
is claimed beyond the `nsys`-measured throughput above.

**CUDA-graph replay sibling.** `black_scholes_batch_graphed(...)` captures the tiled launch into
a CUDA graph on persistent buffers (via `cuda.core`'s `GraphBuilder` — `begin_building` →
`launch(gb.stream, …)` → `end_building` → `complete()`) and replays it `iterations` times,
mirroring the in-engine `black_scholes_batch_graphed`. The replay is **bit-identical to the
direct launch** (verified `max abs diff = 0.0` over 8 iterations). It is **not** presented as a
speedup: a CUDA graph amortizes the CPU cost of a *sequence* of kernels, and for a single-kernel
pricer there is nothing to amortize — measured on the 5090 it is **on par with a direct launch
(~0.98× at 50k options, ~0.85× at 256; `cudaGraphLaunch` carries its own fixed cost)**. Its value
is API/structural parity with the in-engine graphed path, the bit-identical guarantee, and being
the correct substrate once the captured region grows into a multi-kernel pipeline where graph
amortization pays.

## Testing

`tests/gpu_tests.cpp` cross-checks **every** GPU kernel against a CPU reference:
poly-eval (small/large/edge cases), batched edit-distance, CSR BFS, N-queens count,
QMC polynomial integration, batched Haar DWT, batched matmul, batched FFT (vs a CPU
DFT), batched Black-Scholes, CG, European/Asian/barrier Monte Carlo, first-order and
extended Greeks, and the strategy/futures grid sweeps — **25 test groups**. It uses
relative tolerances because the GPU may contract to FMA where the CPU does not (the
sweep kernels are pinned non-contracted for bit-exactness). The suite is built and run
**only with `-DNIMBLECAS_CUDA=ON` on a machine with a CUDA device**.

Verified green on the **mgpu** host (**RTX PRO 6000 Blackwell** / **RTX 5090**, sm_120,
CUDA 13.2): all 25 groups pass. An `nsys` trace confirms every kernel launches — the
path-dependent `asian_segment_kernel` / `mc_barrier_segment_kernel` dominate wall-clock
(compute-bound path simulation; the reductions are ~2 µs). `ncu` deep-counter profiling
**is available on this host** and shows the segment kernels are latency/occupancy-bound
at small batch sizes — achieved occupancy scales with `nseg = ceil(paths / 4096)`, so
small path counts underutilize the SM while large counts saturate it (the deliberate
determinism-vs-occupancy trade-off shared with the European kernel). No speedup is
claimed; the GPU path is a correctness mirror, and `gpu_deriv_bench` reports raw
wall-clock only.

## Benchmarking

`tools/gpu_deriv_bench.cpp` is a standalone profiling harness for the eight shipped GPU
derivative pricing and grid sweep entry points (`monte_carlo_european_batch`,
`monte_carlo_asian_batch`, `barrier_option_mc_batch`, `black_scholes_greeks_batch`,
`black_scholes_extended_greeks_batch`, `strategy_payoff_grid`, `strategy_pnl_grid`, and
`futures_pnl_grid`).

### Building and Running

When configured with `-DNIMBLECAS_CUDA=ON`, CMake registers the benchmark executable target:

```bash
# Build the benchmark executable
ninja gpu_deriv_bench

# Run the benchmark
./build/gpu_deriv_bench
```

### Output Interpretation

The tool measures raw host wall-clock execution time (`std::chrono::steady_clock`, median
of 5 repetitions following 1 warmup iteration) and calculates throughput for each batch
size alongside raw CPU execution time.

- **Raw Wall-Clock Only**: Emits end-to-end host wall-clock duration including DMA
  transfers and launch overhead. No speedup or acceleration ratios are claimed or asserted.
- **Profiling for Kernel Attribution**: Detailed kernel execution duration, SM occupancy,
  and memory bandwidth attribution require profiling with Nsight tools (`nsys` / `ncu`).
- **Device Availability**: On systems without an available CUDA GPU (`device_count() == 0`),
  a banner is displayed noting that GPU columns reflect CPU-fallback timings.

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
