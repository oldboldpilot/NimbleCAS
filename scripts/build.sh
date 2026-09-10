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
  -DCMAKE_C_COMPILER="${NIMBLECAS_CLANGC}"
  -DCMAKE_CXX_COMPILER="${NIMBLECAS_CLANGXX}"
  -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS="${NIMBLECAS_SCAN_DEPS}"
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

# Enable the CUDA GPU kernels when an nvcc that can actually target this machine's GPU is
# available (non-sanitizer builds only; the .cu is compiled by nvcc independently of the
# sanitized clang/libc++ objects).
#
# "Available" is not "first on PATH". A distribution's own nvcc package installs
# /usr/bin/nvcc, which is often years older than the toolkit under /usr/local -- on a
# Blackwell host /usr/bin/nvcc was CUDA 12.0 and died with
#   nvcc fatal : Unsupported gpu architecture 'compute_120'
# while /usr/local/cuda-13.1 built the kernels fine. So: prefer the newest versioned
# toolkit, take PATH only as a last resort, and PROVE the choice by compiling for the
# local device before switching CUDA on. A toolkit that cannot target the GPU in this
# box is not a reason to configure a build that is going to fail; it is a reason to
# leave the GPU kernels out and say so.
if [[ -z "${SANITIZER}" ]]; then
  NVCC_BIN=""
  NVCC_CANDIDATES=()
  # Newest versioned toolkit first: `sort -V -r` puts cuda-13.1 ahead of cuda-12.8.
  while IFS= read -r c; do NVCC_CANDIDATES+=("${c}"); done < <(
    ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r)
  NVCC_CANDIDATES+=(/usr/local/cuda/bin/nvcc)
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_CANDIDATES+=("$(command -v nvcc)")
  fi

  NVCC_PROBE="$(mktemp -d)/probe.cu"
  printf '__global__ void k() {}
' > "${NVCC_PROBE}"
  for c in "${NVCC_CANDIDATES[@]}"; do
    [[ -x "${c}" ]] || continue
    if "${c}" -arch="${NIMBLECAS_CUDA_ARCH:-native}" -c "${NVCC_PROBE}"          -o /dev/null >/dev/null 2>&1; then
      NVCC_BIN="${c}"
      break
    fi
  done
  rm -rf "$(dirname "${NVCC_PROBE}")"

  if [[ -n "${NVCC_BIN}" ]]; then
    CMAKE_ARGS+=(-DNIMBLECAS_CUDA=ON -DNIMBLECAS_NVCC="${NVCC_BIN}")
    if [[ -n "${NIMBLECAS_CUDA_ARCH:-}" ]]; then
      CMAKE_ARGS+=(-DNIMBLECAS_CUDA_ARCH="${NIMBLECAS_CUDA_ARCH}")
    fi
    echo "CUDA: ${NVCC_BIN} (release $("${NVCC_BIN}" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1))"
  elif ((${#NVCC_CANDIDATES[@]})); then
    echo "CUDA: found nvcc but none could target this GPU -- building without the GPU kernels" >&2
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
