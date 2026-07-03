#!/usr/bin/env bash
# NimbleCAS build driver — configure + build + test via CMake/Ninja.
# @author Olumuyiwa Oluwasanmi
#
# Usage:
#   scripts/build.sh                              # configure, build, run tests
#   NIMBLECAS_SANITIZE=ON scripts/build.sh        # ASan+UBSan+LSan (Rules 36/56)
#   NIMBLECAS_SANITIZER=thread scripts/build.sh   # ThreadSanitizer
#   NIMBLECAS_SANITIZER=memory scripts/build.sh   # MemorySanitizer

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

# Separate build dir per sanitizer so BMIs/objects never mix configs.
SANITIZE="${NIMBLECAS_SANITIZE:-OFF}"
SANITIZER="${NIMBLECAS_SANITIZER:-}"
if [[ "${SANITIZE}" == "ON" && -z "${SANITIZER}" ]]; then SANITIZER="address"; fi
if [[ -n "${SANITIZER}" ]]; then
  BUILD_DIR="${REPO_ROOT}/build-san-${SANITIZER}"
else
  BUILD_DIR="${REPO_ROOT}/build"
fi

CMAKE_ARGS=(
  -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}"
  -DCMAKE_CXX_COMPILER="${NIMBLECAS_CLANGXX}"
  -DNIMBLECAS_SANITIZER="${SANITIZER}"
)

# Enable the nanobind Python bindings when the uv-managed venv (with nanobind)
# exists. Sanitizer builds skip Python: the extension would need a sanitizer-
# instrumented interpreter (LD_PRELOAD) to load cleanly, which is out of scope.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

# For a non-sanitizer build, provision the uv env on first use (best-effort).
if [[ -z "${SANITIZER}" && ! -x "${PYTHON_BIN}" ]]; then
  bash "${SCRIPT_DIR}/setup_python.sh" 2>/dev/null || true
fi

if [[ -z "${SANITIZER}" && -x "${PYTHON_BIN}" ]] \
   && "${PYTHON_BIN}" -c "import nanobind" >/dev/null 2>&1; then
  NANOBIND_CMAKE_DIR="$("${PYTHON_BIN}" -c 'import nanobind; print(nanobind.cmake_dir())')"
  CMAKE_ARGS+=(
    -DNIMBLECAS_PYTHON=ON
    -DPython_EXECUTABLE="${PYTHON_BIN}"
    -Dnanobind_ROOT="${NANOBIND_CMAKE_DIR}"
  )
fi

# Enable the CUDA GPU kernels when nvcc is available (non-sanitizer builds only; the
# .cu is compiled by nvcc independently of the sanitized clang/libc++ objects).
if [[ -z "${SANITIZER}" ]]; then
  NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
  for d in /usr/local/cuda/bin /usr/local/cuda-13.2/bin; do
    if [[ -z "${NVCC_BIN}" && -x "${d}/nvcc" ]]; then NVCC_BIN="${d}/nvcc"; fi
  done
  if [[ -n "${NVCC_BIN}" ]]; then
    CMAKE_ARGS+=(-DNIMBLECAS_CUDA=ON -DNIMBLECAS_NVCC="${NVCC_BIN}")
  fi
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}"

# ThreadSanitizer: silence the uninstrumented-TBB false positives (config/tsan.supp)
# so a clean run validates NimbleCAS's own concurrency (Code Policy Rule 36).
if [[ "${SANITIZER}" == "thread" ]]; then
  export TSAN_OPTIONS="suppressions=${REPO_ROOT}/config/tsan.supp ${TSAN_OPTIONS:-}"
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure
