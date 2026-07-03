#!/usr/bin/env bash
# NimbleCAS Windows build driver (Git Bash / MSYS).
# @author Olumuyiwa Oluwasanmi
#
# Builds with the clang + MSVC-STL toolchain bundled in Visual Studio: clang++
# (targeting x86_64-pc-windows-msvc), CMake, and Ninja, using `import std` from the
# MSVC toolset's std.ixx (see cmake/StdModule.cmake). Verified to build and run the
# full engine + tests, identical behaviour to the Linux/libc++ build.
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
LLVM_BIN="${VS_DIR}/VC/Tools/Llvm/x64/bin"

BUILD_DIR="${REPO_ROOT}/build-win"

"${CMAKE}" -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_CXX_COMPILER="${LLVM_BIN}/clang++.exe" \
  -DCMAKE_RC_COMPILER="${LLVM_BIN}/llvm-rc.exe" \
  -DCMAKE_MAKE_PROGRAM="${NINJA}"

"${CMAKE}" --build "${BUILD_DIR}"

"${CTEST}" --test-dir "${BUILD_DIR}" --output-on-failure
