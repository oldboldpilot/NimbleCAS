#!/usr/bin/env bash
# Build the freestanding WebAssembly compute kernel -> web/kernel.wasm
# @author Olumuyiwa Oluwasanmi
#
# Uses clang's native wasm32 target (NO Emscripten, NO libc/libc++). See
# src/webkernel/kernel.cpp for the exported ABI the browser front-end consumes.
#
# This is the FEASIBLE slice of the "WASM" story — a small freestanding compute
# kernel. Compiling the FULL modular CAS (import std + libc++ + oneTBB) to WASM is a
# separate infrastructure task that needs Emscripten plus a wasm-targeted libc++/std
# module and a TBB-free build.

set -euo pipefail
REPO_ROOT="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" rev-parse --show-toplevel)"
CXX="${NIMBLECAS_CLANGXX:-clang++-22}"
SRC="${REPO_ROOT}/src/webkernel/kernel.cpp"
OUT="${REPO_ROOT}/web/kernel.wasm"

mkdir -p "${REPO_ROOT}/web"

# --no-entry: freestanding library, no _start. Explicit function exports keep the
# module surface minimal; linear memory is exported as "memory" by wasm-ld default.
"${CXX}" --target=wasm32 -std=c++23 -O3 -nostdlib -fno-exceptions -fno-rtti \
  -Wl,--no-entry \
  -Wl,--export=poly_eval \
  -Wl,--export=poly_eval_buffer \
  -Wl,--export=coeff_buffer \
  -Wl,--export=coeff_capacity \
  -o "${OUT}" "${SRC}"

echo "built ${OUT} ($(wc -c < "${OUT}") bytes)"
