# WebAssembly build (Emscripten) — status and recipe

**Author:** Olumuyiwa Oluwasanmi

This page records the state of compiling NimbleCAS to WebAssembly. There are two
distinct tracks, and only the first is a "quick win":

1. **Freestanding compute kernel** (DONE) — `src/webkernel/kernel.cpp` compiled with
   `clang++-22 --target=wasm32` (no Emscripten), exporting a dependency-free numeric
   ABI the browser front-end loads. See [webkernel.md](../reference/webkernel.md).
2. **The full modular CAS to WASM** (VIABILITY PROVEN, integration pending) — the
   subject of this page.

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
3. **SIMD.** `nimblecas.simd` / `doubledouble` use x86 intrinsics; for wasm use the
   scalar fallback or wasm128 SIMD (`-msimd128`). Runtime dispatch must resolve to a
   wasm-valid path.
4. **Exclude `nimblecas.gpu`** (CUDA) and the nanobind Python bindings from the wasm
   configuration.
5. **Entry surface.** Decide the exported C ABI / Embind surface the browser calls
   (e.g. parse → simplify → to_latex, or evaluate an expression), then wire it into the
   `web/` front-end alongside the existing freestanding kernel.

Once (1)–(4) are in place, the front-end can call the real engine in-browser rather
than only the freestanding `poly_eval` kernel. The hard unknown (named modules +
`import std` on wasm) is already resolved above.

## See also
- [webkernel.md](../reference/webkernel.md) — the freestanding kernel (track 1).
- [`web/README.md`](../../web/README.md) — the front-end that would host it.
- [Architecture overview](overview.md) · [Documentation hub](../Index.md)
