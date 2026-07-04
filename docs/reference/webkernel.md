# `nimblecas.webkernel` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/webkernel/kernel.cpp` · Build: `scripts/build_wasm.sh` -> `web/kernel.wasm`

A tiny **freestanding wasm32 compute kernel** that the [browser front-end](../../web/README.md)
loads to sample functions live in-page. It is compiled with clang's native
WebAssembly target — `clang++-22 --target=wasm32` — with **no Emscripten and no
libc / libc++**, and it exports a small, dependency-free numeric ABI (currently
Horner polynomial evaluation). The prebuilt `web/kernel.wasm` (**~842 bytes**) is
committed as a front-end asset so the viewer's "Sample a polynomial live" control
runs native code with a JavaScript fallback when the module is absent.

## Boundary: what this is, and what it is not

This is the **feasible slice** of the "WASM" story. It is deliberately *not* the
NimbleCAS engine in a browser:

- It is **not a C++23 named module** and is **not part of the CAS module graph**.
  The translation unit uses only `extern "C"` C-linkage functions over POD
  (`double*`, `int`, `double`) — no `import std`, no `nimblecas.*` imports, no
  standard-library object of any kind crosses the ABI.
- It is **freestanding**: compiled `-nostdlib -fno-exceptions -fno-rtti`, it has no
  libc/libc++ underneath it. The only state is one BSS scratch array, zeroed by
  the linker.
- Compiling the **full modular CAS** (`import std` + libc++ + oneTBB) to WASM is a
  **separate infrastructure task**, not this file. That would require Emscripten
  plus a wasm-targeted libc++/`std` module and a TBB-free build. This kernel is
  the small, real, dependency-free thing that ships today.

## Exported ABI

Four functions with C linkage, plus the linear memory. All signatures are copied
faithfully from `src/webkernel/kernel.cpp`; `f64 == double`, and an `i32` pointer
is a byte offset into the exported linear memory.

| Export | C signature | Behavior |
| :--- | :--- | :--- |
| `coeff_capacity` | `int coeff_capacity()` | The scratch-buffer capacity in slots, `kCapacity == 256`. |
| `coeff_buffer` | `double* coeff_buffer()` | Byte offset of the 256-slot scratch array in linear memory. JS writes coefficients here. |
| `poly_eval` | `double poly_eval(const double* c, int n, double x)` | Horner evaluation `Σ_{i<n} c[i]·x^i` of `n` coefficients at offset `c`. `n < 0` is clamped to `0`. |
| `poly_eval_buffer` | `double poly_eval_buffer(int n, double x)` | `poly_eval(coeff_buffer(), n, x)` over the loaded scratch buffer; `n > kCapacity` is clamped to `kCapacity`. |
| `memory` | `WebAssembly.Memory` | The wasm linear memory, exported by `wasm-ld` by default; JS views it as a typed array. |

### Calling convention

Coefficients are stored **low index = low degree** — `c[0]` is the constant term,
so `p(x) = c[0] + c[1]·x + … + c[n-1]·x^{n-1}`. To evaluate:

1. Read the buffer offset by calling `coeff_buffer()`.
2. Construct a `Float64Array` view over the exported `memory` at that offset and
   write `c0..c_{n-1}` into it (little-endian `f64`, as the platform is).
3. Call `poly_eval(offset, n, x)` per sample point (or `poly_eval_buffer(n, x)`,
   which reuses the buffer offset internally).

`n` is **clamped defensively** on both entry points — negative `n` in `poly_eval`
becomes `0`, and `n` beyond `kCapacity` in `poly_eval_buffer` is capped — so a bad
length can never read past the scratch array.

> **The buffer offset is queried, not hardcoded.** In the committed build,
> `coeff_buffer()` returns **65536** (verified via Node), the address `wasm-ld`
> chose for the BSS array — *not* a fixed constant. The front-end therefore calls
> `coeff_buffer()` and uses whatever offset it returns, falling back to a
> documented `1024` only when an older module lacks the export. Never assume the
> linker-chosen data address; always ask the module.

## Building

`scripts/build_wasm.sh` compiles the single source file to `web/kernel.wasm`. It
resolves the repo root via `git`, honors `NIMBLECAS_CLANGXX` (default
`clang++-22`), and invokes:

```sh
clang++-22 --target=wasm32 -std=c++23 -O3 -nostdlib -fno-exceptions -fno-rtti \
  -Wl,--no-entry \
  -Wl,--export=poly_eval \
  -Wl,--export=poly_eval_buffer \
  -Wl,--export=coeff_buffer \
  -Wl,--export=coeff_capacity \
  -o web/kernel.wasm src/webkernel/kernel.cpp
```

- `--target=wasm32` selects clang's native WebAssembly backend (no Emscripten).
- `-nostdlib -fno-exceptions -fno-rtti` keeps the build freestanding — no libc,
  no libc++, no unwinding or type-info tables.
- `-Wl,--no-entry` marks a freestanding library with no `_start`; the explicit
  `--export=` flags keep the module surface to exactly the four ABI functions.
- The **linear `memory` is exported automatically** by `wasm-ld`; it does not need
  an explicit `--export`.

Run it from anywhere in the repo:

```sh
scripts/build_wasm.sh
# built .../web/kernel.wasm (842 bytes)
```

The script echoes the output path and byte count on success.

## Usage from the front-end

`web/index.html` loads the kernel best-effort at startup (`initWasm()`), never
blocking rendering. It fetches `kernel.wasm`, checks the required exports, and
**prefers the module's own `coeff_buffer()` offset** over the fixed fallback:

```js
async function initWasm() {
  const resp = await fetch('kernel.wasm');
  if (!resp || !resp.ok) return false;
  const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
  const ex = instance.exports;
  if (typeof ex.poly_eval !== 'function' || !(ex.memory instanceof WebAssembly.Memory))
    return false;
  // Prefer the kernel's own scratch offset; fall back to the documented 1024.
  WASM.coeffPtr = (typeof ex.coeff_buffer === 'function') ? ex.coeff_buffer() : 1024;
  WASM.mod = instance; WASM.ready = true;
  return true;
}

function polyWasm(coeffs, x) {
  const ex = WASM.mod.exports;
  const view = new Float64Array(ex.memory.buffer, WASM.coeffPtr, coeffs.length);
  view.set(coeffs);                              // c0..c_{n-1}, low index = low degree
  return ex.poly_eval(WASM.coeffPtr, coeffs.length, x);
}
```

If `kernel.wasm` is absent, lacks the exports, or fails to instantiate (for
example under `file://`, where `fetch` is blocked), the viewer evaluates the
**identical** polynomial with a plain-JS Horner fallback — and degrades to it
mid-run on any kernel error (such as a bad offset yielding a `Float64Array`
`RangeError`):

```js
function polyJS(coeffs, x) {   // Horner, matches the WASM contract exactly
  let acc = 0;
  for (let k = coeffs.length - 1; k >= 0; k--) acc = acc * x + coeffs[k];
  return acc;
}
```

A tag in the UI shows whether `kernel.wasm` or the JS kernel produced the samples,
so the two paths are visibly distinguished but numerically interchangeable.

## See also

- [`nimblecas.gpu`](gpu.md) — the CUDA/Triton batch `poly_eval` path; same Horner
  contract (low degree first), at engine scale rather than in-page.
- [`nimblecas.svgplot`](svgplot.md) — dependency-free plot rendering, the sibling
  "runs anywhere without the CAS toolchain" surface.
- [`web/` front-end README](../../web/README.md) — the WebGPU/Canvas2D viewer that
  consumes this kernel via the optional live-compute hook.
- [Documentation hub](../Index.md)
