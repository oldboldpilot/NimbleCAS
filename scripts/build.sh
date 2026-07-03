#!/usr/bin/env bash
# NimbleCAS build driver — configure + build + test via CMake/Ninja.
# @author Olumuyiwa Oluwasanmi
#
# Usage:
#   scripts/build.sh            # configure, build, run tests
#   NIMBLECAS_SANITIZE=ON scripts/build.sh   # ASan+UBSan+LSan build (Code Policy Rules 36/56)

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

BUILD_DIR="${REPO_ROOT}/build"
SANITIZE="${NIMBLECAS_SANITIZE:-OFF}"

CMAKE_ARGS=(
  -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}"
  -DCMAKE_CXX_COMPILER="${NIMBLECAS_CLANGXX}"
  -DNIMBLECAS_SANITIZE="${SANITIZE}"
)

# Enable the nanobind Python bindings when the project venv (with nanobind) exists.
# Sanitizer builds skip Python: the extension would need an ASan-instrumented
# interpreter (LD_PRELOAD) to load cleanly, which is out of scope here.
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ "${SANITIZE}" == "OFF" && -x "${PYTHON_BIN}" ]] \
   && "${PYTHON_BIN}" -c "import nanobind" >/dev/null 2>&1; then
  NANOBIND_CMAKE_DIR="$("${PYTHON_BIN}" -c 'import nanobind; print(nanobind.cmake_dir())')"
  CMAKE_ARGS+=(
    -DNIMBLECAS_PYTHON=ON
    -DPython_EXECUTABLE="${PYTHON_BIN}"
    -Dnanobind_ROOT="${NANOBIND_CMAKE_DIR}"
  )
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}"

ctest --test-dir "${BUILD_DIR}" --output-on-failure
