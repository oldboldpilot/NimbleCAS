#!/usr/bin/env bash
# NimbleCAS Windows build driver (Git Bash / MSYS).
# @author Olumuyiwa Oluwasanmi
#
# Builds with clang + the MSVC STL: clang++ (targeting x86_64-pc-windows-msvc), with
# CMake and Ninja from Visual Studio, using `import std` from the MSVC toolset's
# std.ixx (see cmake/StdModule.cmake). Verified to build and run the full engine +
# tests, identical behaviour to the Linux/libc++ build.
#
# The project always builds with the NEWEST clang, and the copy bundled in Visual
# Studio lags upstream LLVM, so it is the last resort rather than the default. The
# compiler is, in order: $NIMBLECAS_LLVM_BIN; the highest-versioned clang among
# C:/tools/LLVM*/bin and C:/Program Files/LLVM/bin; Visual Studio's bundled clang.
#
# Usage: scripts/build_win.sh

set -euo pipefail
REPO_ROOT="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" rev-parse --show-toplevel)"

# Locate the newest Visual Studio installation.
VS_DIR="$(ls -d "/c/Program Files"*"/Microsoft Visual Studio"/*/* 2>/dev/null \
          | grep -Ei '/(Professional|Community|Enterprise|BuildTools)$' | sort | tail -1)"
if [[ -z "${VS_DIR}" ]]; then
  echo "error: no Visual Studio installation found." >&2
  exit 1
fi

CMAKE="${VS_DIR}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
CTEST="${VS_DIR}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe"
NINJA="${VS_DIR}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"

# Pick the newest clang by the version it reports, not by directory name.
clang_version() {
  "$1/clang++.exe" --version 2>/dev/null | sed -n 's/.*clang version \([0-9][0-9.]*\).*/\1/p' | head -1
}
LLVM_BIN="${NIMBLECAS_LLVM_BIN:-}"
if [[ -z "${LLVM_BIN}" ]]; then
  best_ver=""
  for cand in /c/tools/LLVM*/bin "/c/Program Files/LLVM/bin" "${VS_DIR}/VC/Tools/Llvm/x64/bin"; do
    [[ -x "${cand}/clang++.exe" ]] || continue
    ver="$(clang_version "${cand}")"
    [[ -n "${ver}" ]] || continue
    if [[ -z "${best_ver}" || "$(printf '%s\n%s\n' "${best_ver}" "${ver}" | sort -V | tail -1)" != "${best_ver}" ]]; then
      best_ver="${ver}"
      LLVM_BIN="${cand}"
    fi
  done
fi
if [[ -z "${LLVM_BIN}" || ! -x "${LLVM_BIN}/clang++.exe" ]]; then
  echo "error: no clang++.exe found (set NIMBLECAS_LLVM_BIN)." >&2
  exit 1
fi
echo "Using clang $(clang_version "${LLVM_BIN}") from ${LLVM_BIN}"

BUILD_DIR="${REPO_ROOT}/build-win"

"${CMAKE}" -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_CXX_COMPILER="${LLVM_BIN}/clang++.exe" \
  -DCMAKE_RC_COMPILER="${LLVM_BIN}/llvm-rc.exe" \
  -DCMAKE_MAKE_PROGRAM="${NINJA}"

"${CMAKE}" --build "${BUILD_DIR}"

"${CTEST}" --test-dir "${BUILD_DIR}" --output-on-failure
