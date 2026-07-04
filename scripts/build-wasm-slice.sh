#!/usr/bin/env bash
# Build the NimbleCAS "full-CAS to WASM" slice: the exact symbolic engine (core, symbolic,
# cache, simplify, diff, latex, reader) compiled with Emscripten and exposed to the browser
# via the tiny C ABI in src/wasm/wasm_entry.cpp (text -> parse -> simplify -> LaTeX).
#
# Uses the proven two-phase C++23-module recipe from docs/architecture/wasm-build.md rather
# than CMake's module scanning under emscripten (which is not yet reliable). Excludes gpu
# (CUDA), the Python bindings, and simd; nimblecas.parallel auto-selects its serial backend
# on wasm (no TBB headers). Run on the build server (mgpu) with emsdk sourced.
#
# @author Olumuyiwa Oluwasanmi
set -euo pipefail

REPO="${1:-/scratch/NimbleCAS}"
OUT="${2:-/scratch/wasm-cas}"
source /scratch/emsdk/emsdk_env.sh >/dev/null 2>&1
SYSROOT="$(em++ --sysroot-path 2>/dev/null || echo /scratch/emsdk/upstream/emscripten/cache/sysroot)"

mkdir -p "$OUT"
cd "$OUT"
F=(-std=c++23 -fexperimental-library -O2 -Wno-reserved-module-identifier)

# 1. Precompile the std module (cached across runs).
if [[ ! -f std.pcm ]]; then
  echo "[wasm] precompiling std.pcm (slow, ~once)"
  em++ "${F[@]}" --precompile -x c++-module "$SYSROOT/share/libc++/v1/std.cppm" -o std.pcm
fi
em++ "${F[@]}" -fmodule-file=std=std.pcm -c std.pcm -o std.o

# 2. Each CAS module in dependency order: source -> BMI (.pcm) -> object (.o), passing every
#    already-built module-file (over-providing is harmless; the importer uses what it needs).
MODS=(core parallel symbolic cache simplify diff latex reader)
MF=(-fmodule-file=std=std.pcm)
OBJS=(std.o)
for m in "${MODS[@]}"; do
  echo "[wasm] module $m"
  em++ "${F[@]}" "${MF[@]}" --precompile -x c++-module "$REPO/src/$m/$m.cppm" -o "$m.pcm"
  em++ "${F[@]}" "${MF[@]}" -fmodule-file="nimblecas.$m=$m.pcm" -c "$m.pcm" -o "$m.o"
  MF+=(-fmodule-file="nimblecas.$m=$m.pcm")
  OBJS+=("$m.o")
done

# 3. The entry TU (imports the modules, exposes the C ABI).
echo "[wasm] entry wasm_entry.cpp"
em++ "${F[@]}" "${MF[@]}" -c "$REPO/src/wasm/wasm_entry.cpp" -o wasm_entry.o
OBJS+=(wasm_entry.o)

# 4. Link to an ES6 module nimblecas.js + nimblecas.wasm with the eval ABI exported.
#    EXPORT_ES6 makes it a real browser-importable module (`import NimbleCAS from
#    './nimblecas.js'`), which the ES-module web front-end (web/app.js) needs; it also
#    locates the .wasm via import.meta.url when served.
echo "[wasm] link nimblecas.js / nimblecas.wasm"
# A FIXED heap (no ALLOW_MEMORY_GROWTH) is deliberate: with growth the WebAssembly.Memory
# buffer is reallocated mid-call, detaching the JS-side view that ccall's string marshaling
# reads ("TextDecoder ... ArrayBuffer detached"). A generous fixed 256 MB never reallocates,
# so the string ABI is robust; it is plenty for REPL-scale symbolic expressions.
em++ "${F[@]}" "${OBJS[@]}" -o nimblecas.js \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=NimbleCAS \
  -sINITIAL_MEMORY=268435456 -sALLOW_MEMORY_GROWTH=0 \
  -sEXPORTED_FUNCTIONS=_nimblecas_eval_latex,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,stringToUTF8,lengthBytesUTF8

echo "[wasm] built: $OUT/nimblecas.js  $OUT/nimblecas.wasm"
