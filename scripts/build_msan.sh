#!/usr/bin/env bash
# MemorySanitizer verification build against an MSan-instrumented libc++.
# @author Olumuyiwa Oluwasanmi
#
# MSan requires the WHOLE stack (incl. libc++) to be instrumented. Point MSAN_LIBCXX
# at an MSan-instrumented libc++ (built from llvm-project/runtimes with
# -DLLVM_USE_SANITIZER=MemoryWithOrigins). This does an explicit-module MSan build of
# the tests that do not exercise oneTBB (TBB is not MSan-instrumented), giving a clean
# MSan run over the core symbolic/SIMD code.
#
# Usage: MSAN_LIBCXX=/scratch/msan-libcxx scripts/build_msan.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

MSAN_LIBCXX="${MSAN_LIBCXX:-/scratch/msan-libcxx}"
STDCPPM="${MSAN_LIBCXX}/modules/c++/v1/std.cppm"
if [[ ! -f "${STDCPPM}" ]]; then
  echo "error: MSan libc++ std module not found at ${STDCPPM}" >&2
  echo "Build one: cmake -S llvm-project/runtimes -B <dir> -DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind' -DLLVM_USE_SANITIZER=MemoryWithOrigins ..." >&2
  exit 1
fi

CXX="${NIMBLECAS_CLANGXX:-clang++-22}"
OUT="${REPO_ROOT}/build-san-memory"; rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
SRC="${REPO_ROOT}/src"; TST="${REPO_ROOT}/tests"

# MSan + MSan-libc++ compile/link flags.
M=(-std=c++23 -stdlib=libc++ -nostdinc++ -isystem "${MSAN_LIBCXX}/include/c++/v1"
   -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O1
   -Wno-reserved-module-identifier -Wno-include-angled-in-module-purview -pthread)
LINK=(-fsanitize=memory -nostdlib++ -L"${MSAN_LIBCXX}/lib" -Wl,-rpath,"${MSAN_LIBCXX}/lib"
      -lc++ -lc++abi -pthread)

pcm() {  # pcm <module-name> <src> [module-file flags...]
  local name="$1" src="$2"; shift 2
  "${CXX}" "${M[@]}" "$@" --precompile "${src}" -o "${name}.pcm"
  "${CXX}" "${M[@]}" "$@" -c "${name}.pcm" -o "${name}.o"
}
mf() { printf -- "-fmodule-file=%s=%s.pcm" "$1" "$2"; }

echo "== building std + module chain under MSan =="
pcm std "${STDCPPM}"
STD=$(mf std std)
pcm nimblecas.core     "${SRC}/core/core.cppm"        "${STD}"
pcm nimblecas.testing  "${SRC}/testing/testing.cppm"  "${STD}"
pcm nimblecas.simd     "${SRC}/simd/simd.cppm"        "${STD}"
CORE=$(mf nimblecas.core nimblecas.core)
pcm nimblecas.parallel "${SRC}/parallel/parallel.cppm" "${STD}"
PAR=$(mf nimblecas.parallel nimblecas.parallel)
pcm nimblecas.symbolic "${SRC}/symbolic/symbolic.cppm" "${STD}" "${CORE}" "${PAR}"

TESTING=$(mf nimblecas.testing nimblecas.testing)
SIMD=$(mf nimblecas.simd nimblecas.simd)
SYM=$(mf nimblecas.symbolic nimblecas.symbolic)

link_test() {  # link_test <exe> <test-src> <objs...> [extra libs]
  local exe="$1" src="$2"; shift 2
  "${CXX}" "${M[@]}" "$@" -c "${src}" -o "${exe}.o" 2>/dev/null || true
}

echo "== core_tests (no TBB) =="
"${CXX}" "${M[@]}" "${STD}" "${CORE}" "${TESTING}" -c "${TST}/core_tests.cpp" -o core_tests.o
"${CXX}" core_tests.o std.o nimblecas.core.o nimblecas.testing.o "${LINK[@]}" -o core_tests

echo "== simd_tests (no TBB) =="
"${CXX}" "${M[@]}" "${STD}" "${SIMD}" "${TESTING}" -c "${TST}/simd_tests.cpp" -o simd_tests.o
"${CXX}" simd_tests.o std.o nimblecas.simd.o nimblecas.testing.o "${LINK[@]}" -o simd_tests

echo "== symbolic_tests (small trees stay serial; links TBB but does not invoke it) =="
"${CXX}" "${M[@]}" "${STD}" "${CORE}" "${PAR}" "${SYM}" "${TESTING}" \
  -c "${TST}/symbolic_tests.cpp" -o symbolic_tests.o
"${CXX}" symbolic_tests.o std.o nimblecas.core.o nimblecas.parallel.o nimblecas.symbolic.o \
  nimblecas.testing.o "${LINK[@]}" -ltbb -o symbolic_tests

echo "== run under MSan =="
export MSAN_OPTIONS="halt_on_error=0 exitcode=77"
rc=0
for t in core_tests simd_tests symbolic_tests; do
  if ./"${t}" >"${t}.msan.log" 2>&1; then
    echo "  [${t}] OK ($(grep -c '\[PASS\]' "${t}.msan.log") pass, no MSan reports)"
  else
    code=$?
    reps=$(grep -c "WARNING: MemorySanitizer" "${t}.msan.log" || true)
    echo "  [${t}] exit=${code}, MSan reports=${reps}"
    grep -A3 "WARNING: MemorySanitizer" "${t}.msan.log" | head -8
    rc=1
  fi
done
exit "${rc}"
