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

cmake -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_CXX_COMPILER="${NIMBLECAS_CLANGXX}" \
  -DNIMBLECAS_SANITIZE="${SANITIZE}"

cmake --build "${BUILD_DIR}"

ctest --test-dir "${BUILD_DIR}" --output-on-failure
