#!/usr/bin/env bash
# NimbleCAS clang-tidy runner (Code Policy Rules 34 / 50).
# @author Olumuyiwa Oluwasanmi
#
# Usage:
#   scripts/tidy.sh                     # every TU in the compile database
#   scripts/tidy.sh src/memo_dist       # only TUs whose path contains this substring
#   BUILD_DIR=build-c23 scripts/tidy.sh # against a different configured build
#
# Requires a BUILT build directory, not merely a configured one. CMAKE_EXPORT_COMPILE_COMMANDS
# is ON so cmake writes compile_commands.json at configure time, but clang-tidy on a C++23 module
# translation unit needs the .pcm files its imports resolve to, and those only exist after a
# build. This script itself builds nothing.
#
# Checks come from the repo-root .clang-tidy. Exit status is clang-tidy's, so this is
# usable as a gate; it is not wired into the build, because running tidy over every
# module TU on every compile would dominate build time for no gain per edit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
FILTER="${1:-}"

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "No compile database at ${BUILD_DIR}/compile_commands.json." >&2
  echo "Configure a build first (scripts/build.sh), or set BUILD_DIR." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH; it is needed to read the compile database." >&2
  exit 1
fi

if ! command -v "${NIMBLECAS_CLANG_TIDY}" >/dev/null 2>&1; then
  echo "${NIMBLECAS_CLANG_TIDY} not found on PATH." >&2
  echo "Install it, or override: NIMBLECAS_CLANG_TIDY=clang-tidy-NN scripts/tidy.sh" >&2
  exit 1
fi

mapfile -t FILES < <(
  python3 - "${BUILD_DIR}/compile_commands.json" "${REPO_ROOT}" "${FILTER}" <<'PY'
import json, os, sys
db = json.load(open(sys.argv[1]))
root = os.path.realpath(sys.argv[2]) + os.sep
needle = sys.argv[3]
seen = set()
for e in db:
    f = e["file"]
    # Translation units OUTSIDE the repository are dropped. The compile database names one:
    # libc++'s own std.cppm, pulled in because `import std` builds the standard library module
    # as a project target. Nothing above /usr/lib/llvm-NN provides a .clang-tidy, so clang-tidy
    # resolves an empty check list for that file and aborts the WHOLE invocation with
    # "no checks enabled" -- taking every repo TU with it. That is why this filter exists.
    if not os.path.realpath(f).startswith(root):
        continue
    # Inside the repository is not the same as ours. Enabling the Python bindings puts
    # nanobind's own .cpp files in the compile database, under the uv-managed .venv --
    # gitignored, never edited here, and 610 of the 1248 findings in a full sweep. Dropping
    # them is what makes the reported number mean "findings in NimbleCAS".
    rel = os.path.realpath(f)[len(root):].replace(os.sep, "/")
    if rel.split("/", 1)[0] == ".venv":
        continue
    if needle and needle not in f:
        continue
    if f in seen:
        continue
    seen.add(f)
    print(f)
PY
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No translation units matched${FILTER:+ filter '${FILTER}'}." >&2
  exit 1
fi

echo "Running ${NIMBLECAS_CLANG_TIDY} over ${#FILES[@]} translation unit(s) in ${BUILD_DIR}..."
"${NIMBLECAS_CLANG_TIDY}" -p "${BUILD_DIR}" "${FILES[@]}"
