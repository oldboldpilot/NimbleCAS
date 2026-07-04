# WebAssembly build (Emscripten) — status and recipe

**Author:** Olumuyiwa Oluwasanmi

This page records the state of compiling NimbleCAS to WebAssembly. There are two
distinct tracks, and only the first is a "quick win":

1. **Freestanding compute kernel** (DONE) — `src/webkernel/kernel.cpp` compiled with
   `clang++-22 --target=wasm32` (no Emscripten), exporting a dependency-free numeric
   ABI the browser front-end loads. See [webkernel.md](../reference/webkernel.md).
2. **The full modular CAS to WASM** (SLICE DONE) — the exact symbolic core
   (`core → parallel → symbolic → cache → simplify → diff → latex → reader`) now
   compiles to wasm and runs the real engine in the browser. Build it with
   [`scripts/build-wasm-slice.sh`](../../scripts/build-wasm-slice.sh); the entry ABI
   is [`src/wasm/wasm_entry.cpp`](../../src/wasm/wasm_entry.cpp) and a live REPL is
   [`web/cas-repl.html`](../../web/cas-repl.html). See "Status of the slice" below.

## Toolchain

Emscripten is installed on the build server (`mgpu`) via emsdk:

```
/scratch/emsdk                         # emsdk 6.0.2 (LLVM/clang + node 22 + binaryen)
source /scratch/emsdk/emsdk_env.sh     # puts em++/emcc/node on PATH
em++ --version                         # emcc 6.0.2
```

The wasm libc++ `std` module **source** ships in the emscripten sysroot:

```
/scratch/emsdk/upstream/emscripten/cache/sysroot/share/libc++/v1/std.cppm
```

## What is proven to work (PoCs on mgpu)

- **C++23 STL → wasm → run.** `#include <vector>/<numeric>/<expected>` compiles with
  `em++ -std=c++23 -O2` and runs under node. Crucially `std::expected` (NimbleCAS's
  railway `Result<T>`) is supported.
- **The `std` module precompiles for wasm.** A 32 MB `std.pcm` BMI builds from the
  sysroot `std.cppm`:
  ```
  em++ -std=c++23 -fexperimental-library -Wno-reserved-module-identifier \
       --precompile -x c++-module "$SYSROOT/share/libc++/v1/std.cppm" -o std.pcm
  ```
- **C++23 named modules + `import std` → wasm → run.** A user module
  (`export module demo; import std; ...`) compiles to a BMI and links into a running
  wasm module. The proven two-phase recipe (this is what CMake's module support
  automates):
  ```
  F="-std=c++23 -fexperimental-library"
  # 1. std module BMI (above), then BMI -> object
  em++ $F -fmodule-file=std=std.pcm -c std.pcm -o std.o
  # 2. each user module: source -> BMI, then BMI -> object (std flag needed BOTH times)
  em++ $F -fmodule-file=std=std.pcm --precompile -x c++-module demo.cpp -o demo.pcm
  em++ $F -fmodule-file=std=std.pcm -c demo.pcm -o demo.o
  # 3. consumers, passing every module-file it imports
  em++ $F -fmodule-file=std=std.pcm -fmodule-file=demo=demo.pcm -c main.cpp -o main.o
  # 4. link the objects
  em++ $F std.o demo.o main.o -o out.js        # -> out.js + out.wasm, runs under node
  ```
  Verified end to end: the demo module's `answer()` returned `55` from the wasm.

  Gotcha found: the module declaration (`export module X;`) MUST be the first
  declaration — `import std;` goes *after* it, not before. And turning a `.pcm` into
  a `.o` still needs `-fmodule-file=std=std.pcm` (the std BMI must be visible at every
  phase, not just at `--precompile`).

## Remaining work for a full-CAS wasm build (bounded, not blocked)

1. **Emscripten CMake toolchain.** Configure with `emcmake cmake` (or a toolchain file)
   so `CMAKE_CXX_COMPILER=em++` and CMake's C++23 module dependency scanning drives the
   two-phase build above for all ~69 modules. CMake must treat `em++` as clang-like for
   `CMAKE_CXX_COMPILER_ID` / module scanning.
2. **oneTBB has no wasm build.** `nimblecas.parallel` must select its **serial backend**
   for wasm (no TBB link). Gate the TBB path behind the existing serial fallback when
   targeting wasm; wasm threads (`-pthread` + SharedArrayBuffer) are a later option.
3. **SIMD.** `nimblecas.simd` (and `doubledouble`) use x86 intrinsics; for wasm use the
   scalar fallback or wasm128 SIMD (`-msimd128`). Runtime dispatch must resolve to a
   wasm-valid path. **RESOLVED** — see "Status of the slice" below: `simd.cppm`'s
   AVX/AVX2/AVX-512 code and `__builtin_cpu_supports` are now guarded behind
   `#if defined(__x86_64__) || defined(__i386__)`, so non-x86 targets compile to the
   scalar-only dispatch (already the documented portable-correctness path) with zero
   native-behavior change.
4. **Exclude `nimblecas.gpu`** (CUDA) and the nanobind Python bindings from the wasm
   configuration.
5. **Entry surface.** Decide the exported C ABI / Embind surface the browser calls
   (e.g. parse → simplify → to_latex, or evaluate an expression), then wire it into the
   `web/` front-end alongside the existing freestanding kernel.

Once (1)–(4) are in place, the front-end can call the real engine in-browser rather
than only the freestanding `poly_eval` kernel. The hard unknown (named modules +
`import std` on wasm) is already resolved above.

## Status of the slice (DONE — symbolic core + numeric/linear-algebra chain)

The slice now covers the symbolic core **and** the numeric / linear-algebra chain, built
and verified end to end:

- **Modules** — `core, parallel, simd, polynomial, ratpoly, matrix, roots, numeric,
  matdecomp, bandsolve, eigen, symbolic, cache, simplify, diff, latex, reader` (17
  modules), in dependency order.
- **Toolchain** — rather than CMake's still-unreliable module scanning under emscripten,
  [`scripts/build-wasm-slice.sh`](../../scripts/build-wasm-slice.sh) drives the proven
  two-phase recipe over all 17 modules, links the entry TU, and emits `nimblecas.js` +
  `nimblecas.wasm` (~397 KB).
- **(2) TBB** — handled with no code change: `nimblecas.parallel` selects its serial
  backend on wasm because `__has_include(<tbb/parallel_for.h>)` is false there.
- **(3) SIMD** — handled with a source-level portability guard, not a code-change-free
  no-op like TBB: `<immintrin.h>` under Emscripten **compiles** (its headers are stubs)
  but declares none of the `__m256`/`__m512` types or `_mm*_` intrinsics actually used, so
  the module previously failed to compile off x86. Fixed in `src/simd/simd.cppm` by
  guarding the AVX/AVX2/AVX-512 code paths and `__builtin_cpu_supports` behind
  `#if defined(__x86_64__) || defined(__i386__)`; non-x86 targets fall through to the
  scalar dispatch unconditionally — exact and already the documented portable-correctness
  path, not a degraded one. Verified zero behavior change on native x86 (full suite still
  105/105). Adversarial review of this guard caught one real defect before it landed:
  guarding `detect_isa()`'s only *caller* (inside `make_dispatch`'s switch) left the
  function itself unused on non-x86 (`-Wunused-function` in an anonymous namespace —
  reproduced with `-Wall -Wextra`, would break any future `-Werror` non-x86 build). Fixed
  by calling `detect_isa()` unconditionally and guarding only the AVX-specific `switch`
  cases, with a `default:` covering the enumerators a non-x86 build leaves unhandled.
- **(4) Excluded** — `nimblecas.gpu` (CUDA) and the nanobind Python bindings are simply
  not part of the slice configuration.
- **(5) Entry surface** — [`src/wasm/wasm_entry.cpp`](../../src/wasm/wasm_entry.cpp)
  exposes two endpoints, both the same exact-over-ℚ engine as native, never a crash or a
  wrong value:
  - `nimblecas_eval_latex(const char*)` — text → `parse` → `simplify` → `to_latex`.
  - `nimblecas_matrix_det_latex(const char*)` — `"a,b;c,d"` (semicolon rows, comma cells,
    each cell any expression reducing to an exact rational) → the linear-algebra chain
    (`matrix` → `ratpoly` → `polynomial` → `simd`) → the exact determinant → `to_latex`.
    Malformed/non-square/non-numeric input returns a `\text{…}` marker, never a guess.

**Verified under node**:

| endpoint | input | output |
| :--- | :--- | :--- |
| `eval_latex` | `1 + 2*3` | `7` |
| `eval_latex` | `2/4 + 1/4` | `\frac{3}{4}` (exact rational — not a float) |
| `eval_latex` | `x^2 + x^2` | `2 x^{2}` |
| `eval_latex` | `sin(x) + sin(x)` | `2 \sin\left(x\right)` |
| `eval_latex` | `(x + 1)^2` | `\left(1 + x\right)^{2}` |
| `eval_latex` | `1 +` | `\text{parse error}` |
| `matrix_det_latex` | `1,2;3,4` | `-2` |
| `matrix_det_latex` | `2,0,0;0,3,0;0,0,4` | `24` |
| `matrix_det_latex` | `1/2,1;1,1` | `-\frac{1}{2}` (exact rational) |
| `matrix_det_latex` | `1,2;3,4;5,6` (non-square) | `\text{determinant error}` |
| `matrix_det_latex` | `1,x;2,3` (symbolic cell) | `\text{matrix error}` |
| `matrix_det_latex` | `5,5;5,5` (singular) | `0` |

Two front-ends use this ABI:

- [`web/cas-repl.html`](../../web/cas-repl.html) — a minimal standalone REPL (MathJax
  renders the LaTeX).
- [`web/app.html`](../../web/app.html) — the **full WebGPU document viewer** with the
  engine wired in: executable `nimblecas` document cells, `nimblecas-det` matrix cells,
  and a live CAS box are evaluated in-browser and rendered as native MathML alongside the
  WebGPU/Canvas2D plots (`web/app.js` → `initCAS` / `renderCasCell` / `renderDetCell` /
  `buildCasRepl`). Browser-verified end to end (Chrome, WebGPU active, no console errors,
  screenshot-confirmed `det(1,2;3,4) = -2`, `det(diag(2,3,4)) = 24`,
  `det(1/2,1;1,1) = -1/2`).

Two build notes learned wiring it in: emit a **true ES module** (`EXPORT_ES6`, so
`nimblecas.js` carries `export default` and imports in the browser — `python
http.server` serves `.mjs` as `text/plain`, which browsers reject), and use a **fixed
heap** (`INITIAL_MEMORY=256MB`, no `ALLOW_MEMORY_GROWTH`) so a mid-call heap growth can't
detach the buffer `ccall`'s string marshaling reads.

**Still open:** widening the slice further as the front-end needs (e.g. `bigint`/`bigrational`
for arbitrary precision, `cmatrix`/`quantum` for complex/operator algebra); none of these are
known-blocked, each is the same mechanical add-module-to-`MODS`-and-check exercise as this
round.

## See also
- [webkernel.md](../reference/webkernel.md) — the freestanding kernel (track 1).
- [`web/README.md`](../../web/README.md) — the front-end that would host it.
- [Architecture overview](overview.md) · [Documentation hub](../Index.md)
