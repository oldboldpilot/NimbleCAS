#!/usr/bin/env bash
# verify-generated.sh — compile, and where possible RUN, the code this repository generates.
# @author Olumuyiwa Oluwasanmi
#
# Usage: scripts/verify-generated.sh [build-dir]
#
# WHY THIS EXISTS. Three modules emit source: nimblecas.logic_compile (Prolog),
# nimblecas.sat_compile (CNF) and nimblecas.csp_compile (finite-domain CSPs). Their test suites
# check the emitted TEXT -- that it contains the promised entry points, that it says which
# guarantee it carries -- and a string match cannot tell whether the text is a program. Nothing
# had ever run a compiler over it, and that gap hid real defects: emitted 128-bit CUDA that had
# never been compilable, because nvcc treats `__builtin_add_overflow` as a __host__ function.
#
# So this script is the missing half of those suites. It is not run by ctest: it needs clang++-23
# with libc++, and nvcc and a GPU for the CUDA half, none of which a test binary should assume.
# Run it after touching an emitter.
#
# Exit status is 0 only if every variant compiled, and every one that could be run agreed with
# the reference answer the emitting tool printed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-${REPO}/build}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
checks=0

note() { printf '\n=== %s ===\n' "$*"; }
pass() { checks=$((checks + 1)); printf '  [PASS] %s\n' "$*"; }
fail() { checks=$((checks + 1)); failures=$((failures + 1)); printf '  [FAIL] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Toolchain.
# ---------------------------------------------------------------------------
CXX=""
for c in clang++-23 /usr/lib/llvm-23/bin/clang++ clang++; do
    if command -v "${c}" >/dev/null 2>&1; then CXX="$(command -v "${c}")"; break; fi
done
if [[ -z "${CXX}" ]]; then
    echo "verify-generated: no clang++ found; this repository builds only with clang++-23" >&2
    exit 2
fi

# std.cppm must come from the SAME libc++ release as the clang++ compiling it: a mismatched pair
# either fails to build or, worse, builds against a different standard library than the rest.
#
# The build directory has already resolved this, through cmake/LibcxxLocate.cmake, so ask IT
# first -- that guarantees the generated code is checked against the same standard library the
# repository itself is built with. Asking the compiler directly is the fallback, and on its own
# it is not enough: `-print-file-name=libc++.so.1` answers with the runtime the LINKER would
# pick, which on a host that also has a distro libc++ is /lib/x86_64-linux-gnu and has no
# share/libc++ beside it.
STD_CPPM=""
if [[ -f "${BUILD}/CMakeCache.txt" ]]; then
    STD_CPPM="$(sed -n 's/^NIMBLECAS_STD_MODULE_SRC:FILEPATH=//p' "${BUILD}/CMakeCache.txt")"
    [[ -f "${STD_CPPM}" ]] || STD_CPPM=""
fi
if [[ -z "${STD_CPPM}" ]]; then
    LIBCXX_SO="$("${CXX}" -print-file-name=libc++.so.1 2>/dev/null)"
    if [[ -f "${LIBCXX_SO}" ]]; then
        prefix="$(cd "$(dirname "${LIBCXX_SO}")/.." && pwd)"
        [[ -f "${prefix}/share/libc++/v1/std.cppm" ]] &&
            STD_CPPM="${prefix}/share/libc++/v1/std.cppm"
    fi
fi
if [[ -z "${STD_CPPM}" ]]; then
    # The compiler's resource directory pins the LLVM tree it belongs to, which is where a
    # versioned toolchain keeps its own std.cppm.
    resource="$("${CXX}" -print-resource-dir 2>/dev/null)"
    while [[ -n "${resource}" && "${resource}" != "/" ]]; do
        if [[ -f "${resource}/share/libc++/v1/std.cppm" ]]; then
            STD_CPPM="${resource}/share/libc++/v1/std.cppm"
            break
        fi
        resource="$(dirname "${resource}")"
    done
fi
if [[ -z "${STD_CPPM}" ]]; then
    echo "verify-generated: cannot find this clang++'s std.cppm" >&2
    exit 2
fi

# -Werror, because a warning in generated code is a defect in the emitter and there is nobody to
# fix it in the generated file. -Wno-psabi is the one exception: the vector-extension path
# returns a 512-bit vector, and clang notes that doing so without -mavx512f changes the calling
# convention. That is a warning about mixing translation units built with different flags; every
# caller here is in the same file and the note is not actionable from inside an emitter.
CXXFLAGS=(-std=c++23 -stdlib=libc++ -Wall -Wextra -Werror -Wno-psabi -O1)

# Link and RUN against the libc++ that owns that std.cppm, not whichever one the loader finds
# first. A host with a distro libc++ alongside a versioned LLVM will otherwise compile against
# one standard library and run against the other, which surfaces as an undefined symbol at
# startup rather than as anything a reader would connect to the toolchain.
LIBCXX_DIR="$(cd "$(dirname "${STD_CPPM}")/../../.." && pwd)/lib"
#
# `-lc++abi` goes with the `-L`: the distro package links its ABI library in for you, a
# versioned LLVM tree does not, and taking the second one's libc++ without saying so leaves
# std::length_error's destructor undefined at link time.
LDFLAGS=()
if [[ -f "${LIBCXX_DIR}/libc++.so.1" ]]; then
    LDFLAGS=(-L"${LIBCXX_DIR}" -Wl,-rpath,"${LIBCXX_DIR}" -lc++abi)
fi

echo "verify-generated: ${CXX}"
echo "verify-generated: std module ${STD_CPPM}"
"${CXX}" "${CXXFLAGS[@]}" -Wno-reserved-module-identifier --precompile "${STD_CPPM}" \
    -o "${WORK}/std.pcm" || { echo "verify-generated: could not precompile std" >&2; exit 2; }

# Compiles one generated C++ file together with a driver, and runs it. The generated file is
# #included by the driver, which is how these emitters are meant to be consumed.
# $1 label, $2 generated file, $3 driver source
compile_and_run_cpp() {
    local label="$1" gen="$2" driver="$3" exe="${WORK}/run.$$"
    if ! "${CXX}" "${CXXFLAGS[@]}" -fmodule-file=std="${WORK}/std.pcm" \
            -I"$(dirname "${gen}")" "${driver}" "${WORK}/std.pcm" "${LDFLAGS[@]}" -o "${exe}" \
            > "${WORK}/cc.log" 2>&1; then
        fail "${label}: did not compile"
        sed -n '1,15p' "${WORK}/cc.log"
        return 1
    fi
    if ! "${exe}" > "${WORK}/run.log" 2>&1; then
        fail "${label}: compiled but the run disagreed"
        sed -n '1,15p' "${WORK}/run.log"
        rm -f "${exe}"
        return 1
    fi
    pass "${label}: compiles and agrees with the reference"
    rm -f "${exe}"
}

# Compiles one generated CUDA file. Only compiled, not run: running needs a GPU, and a compile
# is what catches the class of defect this script exists for.
# $1 label, $2 generated file
compile_cu() {
    local label="$1" gen="$2"
    if [[ -z "${NVCC}" ]]; then
        printf '  [SKIP] %s: no nvcc that can target this GPU\n' "${label}"
        return 0
    fi
    if ! "${NVCC}" -std=c++20 -arch="${NIMBLECAS_CUDA_ARCH:-native}" -c "${gen}" \
            -o /dev/null > "${WORK}/nvcc.log" 2>&1; then
        fail "${label}: did not compile"
        sed -n '1,15p' "${WORK}/nvcc.log"
        return 1
    fi
    pass "${label}: compiles"
}

# Compiling a .cu proves less than it looks. nvcc checks that the launcher names a kernel that
# exists -- which is how `<entry>_kernel_kernel` was caught -- but it cannot tell whether the
# kernel COMPUTES anything, and an emitted solver that returns a confident wrong number compiles
# perfectly. So where there is a device to run on, the emitted host wrapper is called and its
# answer checked, exactly as the C++ emissions are.
compile_and_run_cu() {
    local label="$1" gen="$2" driver="$3" exe="${WORK}/cu.$$"
    if [[ -z "${NVCC}" ]]; then
        printf '  [SKIP] %s: no nvcc that can target this GPU\n' "${label}"
        return 0
    fi
    if ! "${NVCC}" -std=c++20 -arch="${NIMBLECAS_CUDA_ARCH:-native}" \
            -I"$(dirname "${gen}")" "${driver}" -o "${exe}" > "${WORK}/nvcc.log" 2>&1; then
        fail "${label}: did not compile"
        sed -n '1,15p' "${WORK}/nvcc.log"
        return 1
    fi
    if [[ -z "${CUDA_RUNS}" ]]; then
        pass "${label}: compiles (no device to run it on)"
        rm -f "${exe}"
        return 0
    fi
    if ! "${exe}" > "${WORK}/curun.log" 2>&1; then
        fail "${label}: compiled but the run disagreed"
        sed -n '1,15p' "${WORK}/curun.log"
        rm -f "${exe}"
        return 1
    fi
    pass "${label}: compiles, runs and agrees with the reference"
    rm -f "${exe}"
}

# $1 label, $2 generated file
# A Triton emission gets the same treatment as the others where the machine can give it: the
# kernel goes through the Triton JIT and a driver checks the ANSWER against a reference.
#
# A parse is nowhere near enough, and this is not a hypothetical. `shape = offs.shape` parses,
# and so does `INT64_MAX = 9223372036854775807` at module scope -- and each on its own makes
# Triton refuse the kernel outright, the first because a tensor shape has to be a tuple of
# compile-time integers and the second because a @triton.jit function cannot read a global that
# is not a tl.constexpr. Both were in the logic_compile Triton target, which therefore reached
# this script never once having compiled.
#
# Where there is no Triton or no CUDA device the parse still runs, announced as the weaker
# check it is rather than as a pass for something that was not tested.
check_triton() {
    local label="$1" gen="$2" driver="$3"
    if [[ -z "${PY}" ]]; then
        printf '  [SKIP] %s: no python3\n' "${label}"
        return 0
    fi
    if ! "${PY}" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "${gen}" \
            > "${WORK}/py.log" 2>&1; then
        fail "${label}: is not valid Python"
        sed -n '1,10p' "${WORK}/py.log"
        return 1
    fi
    if [[ -z "${TRITON_PY}" ]]; then
        # A SKIP, not a pass. The parse succeeded, but the parse is not the check, and
        # counting it as one is how a kernel that cannot compile gets a clean report.
        printf '  [SKIP] %s: parses as Python; no Triton with a CUDA device to compile it\n' \
            "${label}"
        return 0
    fi
    if ! "${TRITON_PY}" "${driver}" > "${WORK}/tri.log" 2>&1; then
        fail "${label}: did not compile through Triton and run"
        tail -n 25 "${WORK}/tri.log"
        return 1
    fi
    pass "${label}: compiles through Triton and agrees with the reference"
}

# The newest nvcc that can actually target this GPU, chosen the same way scripts/build.sh does:
# a distro CUDA too old for the installed card answers `Unsupported gpu architecture`.
NVCC=""
probe="${WORK}/probe.cu"
printf '__global__ void k() {}\n' > "${probe}"
while IFS= read -r c; do
    [[ -x "${c}" ]] || continue
    if "${c}" -arch="${NIMBLECAS_CUDA_ARCH:-native}" -c "${probe}" -o /dev/null >/dev/null 2>&1; then
        NVCC="${c}"
        break
    fi
done < <({ ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r; \
           echo /usr/local/cuda/bin/nvcc; command -v nvcc 2>/dev/null; })
[[ -n "${NVCC}" ]] && echo "verify-generated: ${NVCC}"

# nvcc can compile for a card this machine does not have. Running needs one that is actually
# visible to the driver, so ask -- rather than reporting a wrong answer from a failed launch as
# an emitter defect.
CUDA_RUNS=""
if [[ -n "${NVCC}" ]]; then
    cat > "${WORK}/devprobe.cu" <<'EOF'
#include <cstdio>
int main() {
    int n = 0;
    return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 0 : 1;
}
EOF
    if "${NVCC}" -std=c++20 -arch="${NIMBLECAS_CUDA_ARCH:-native}" "${WORK}/devprobe.cu" \
            -o "${WORK}/devprobe" >/dev/null 2>&1 && "${WORK}/devprobe" >/dev/null 2>&1; then
        CUDA_RUNS="yes"
        echo "verify-generated: a CUDA device is visible; the emitted CUDA will be run"
    fi
fi

PY="$(command -v python3 2>/dev/null || true)"

# Compiling a Triton kernel needs Triton, and running one needs a device. Both, or the emitted
# kernels fall back to a parse.
TRITON_PY=""
if [[ -n "${PY}" ]] && "${PY}" - <<'PROBE' >/dev/null 2>&1
import sys
import torch
import triton  # noqa: F401
sys.exit(0 if torch.cuda.is_available() else 1)
PROBE
then
    TRITON_PY="${PY}"
    echo "verify-generated: triton $("${PY}" -c 'import triton; print(triton.__version__)')"
fi

# ---------------------------------------------------------------------------
# The emitting tools.
# ---------------------------------------------------------------------------
for tool in prolog_compile sat_compile csp_compile; do
    if [[ ! -x "${BUILD}/${tool}" ]]; then
        echo "verify-generated: building ${tool}" >&2
        if ! ninja -C "${BUILD}" "${tool}" >/dev/null 2>&1; then
            echo "verify-generated: cannot build ${tool} in ${BUILD}" >&2
            exit 2
        fi
    fi
done

# ---------------------------------------------------------------------------
# nimblecas.logic_compile — a Prolog predicate.
# ---------------------------------------------------------------------------
note "logic_compile: examples/fib.pl"

for width in 64 128; do
    gen="${WORK}/fib_${width}.hpp"
    if ! "${BUILD}/prolog_compile" "${REPO}/examples/fib.pl" fib io cpp "${width}" \
            > "${gen}" 2>"${WORK}/emit.log"; then
        fail "logic_compile cpp/${width}: emission refused"
        sed -n '1,10p' "${WORK}/emit.log"
        continue
    fi
    cat > "${WORK}/fib_driver_${width}.cpp" <<EOF
#include "fib_${width}.hpp"

// fib(0..20) against the closed values, so the check is the ANSWERS and not merely that the
// generated file was accepted by a compiler.
//
// The compiled predicate keeps Prolog's shape: it returns whether it SUCCEEDED and writes its
// output argument through a reference, because a predicate can fail without producing a value.
// That is the mode declaration, not a departure from the arithmetic helpers' std::optional.
auto main() -> int {
    constexpr std::array<std::int64_t, 21> expected{0,   1,   1,    2,    3,    5,    8,
                                                    13,  21,  34,   55,   89,   144,  233,
                                                    377, 610, 987,  1597, 2584, 4181, 6765};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        nc_int got = nc_lit(0);
        if (!p_fib_2(nc_lit(static_cast<std::int64_t>(i)), got) || got != nc_lit(expected[i])) {
            std::println(std::cerr, "fib({}) disagreed", i);
            return 1;
        }
    }
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/${width}" "${gen}" "${WORK}/fib_driver_${width}.cpp"
done

for width in 64 128; do
    gen="${WORK}/fib_${width}.cu"
    if ! "${BUILD}/prolog_compile" "${REPO}/examples/fib.pl" fib io cuda "${width}" \
            > "${gen}" 2>/dev/null; then
        fail "logic_compile cuda/${width}: emission refused"
        continue
    fi
    cat > "${WORK}/fib_cu_driver_${width}.cu" <<EOF
#include "fib_${width}.cu"

#include <cstdio>

// The same closed values the C++ driver checks, through the batch kernel: one thread per row,
// and ok[i] must say the predicate succeeded there.
//
// The device stack limit is raised because the emitted file says to. fib/2 is recursive, and a
// recursive predicate overruns the 1 KB per thread CUDA gives by default -- which surfaces from
// cudaDeviceSynchronize as an illegal memory access and reads like a codegen bug.
int main() {
    constexpr int N = 21;
    const long long expected[N] = {0,   1,   1,    2,    3,    5,    8,
                                   13,  21,  34,   55,   89,   144,  233,
                                   377, 610, 987,  1597, 2584, 4181, 6765};
    if (cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024) != cudaSuccess) {
        std::fprintf(stderr, "could not raise the device stack limit\n");
        return 1;
    }
    nc_int h_in[N];
    nc_int h_out[N];
    unsigned char h_ok[N];
    for (int i = 0; i < N; ++i) { h_in[i] = static_cast<nc_int>(i); }

    nc_int* d_in = nullptr;
    nc_int* d_out = nullptr;
    unsigned char* d_ok = nullptr;
    if (cudaMalloc(&d_in, sizeof(h_in)) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof(h_out)) != cudaSuccess ||
        cudaMalloc(&d_ok, sizeof(h_ok)) != cudaSuccess ||
        cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "device allocation failed\n");
        return 1;
    }
    p_fib_2_batch<<<1, N>>>(N, d_ok, d_in, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "the batch kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    if (cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_ok, d_ok, sizeof(h_ok), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "reading the results back failed\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        if (h_ok[i] != 1) {
            std::fprintf(stderr, "fib(%d) reported failure\n", i);
            return 1;
        }
        if (h_out[i] != static_cast<nc_int>(expected[i])) {
            std::fprintf(stderr, "fib(%d) disagreed\n", i);
            return 1;
        }
    }
    return 0;
}
EOF
    compile_and_run_cu "logic_compile cuda/${width}" "${gen}" \
        "${WORK}/fib_cu_driver_${width}.cu"
done

# Triton refuses recursion -- a kernel is a flat block of lanes with no call stack -- so fib/2
# cannot reach this target at all, and examples/poly.pl is the straight-line arithmetic that can.
# Without it the module's third target would have no coverage here, which is exactly how it came
# to be shipped uncompilable.
gen="${WORK}/poly_triton.py"
if "${BUILD}/prolog_compile" "${REPO}/examples/poly.pl" eval_poly io triton 64 \
        > "${gen}" 2>/dev/null; then
    cat > "${WORK}/poly_tri_driver.py" <<EOF
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import poly_triton as k

# The whole int32-ish neighbourhood of zero, including the negatives: the emitted overflow
# predicates are the interesting part, and they are sign-dependent.
N = 256
x = torch.arange(N, dtype=torch.int64, device="cuda") - (N // 2)
y = torch.zeros(N, dtype=torch.int64, device="cuda")
ok = torch.zeros(N, dtype=torch.int8, device="cuda")
k.p_eval_poly_2_kernel[(1,)](x, y, ok, N, BLOCK=N)

for xi, yi, oki in zip(x.tolist(), y.tolist(), ok.tolist()):
    want = xi * xi + 3 * xi + 5
    if oki != 1:
        print("eval_poly({}) reported failure on a total predicate".format(xi), file=sys.stderr)
        raise SystemExit(1)
    if yi != want:
        print("eval_poly({}) = {}, expected {}".format(xi, yi, want), file=sys.stderr)
        raise SystemExit(1)
EOF
    check_triton "logic_compile triton/64" "${gen}" "${WORK}/poly_tri_driver.py"
else
    fail "logic_compile triton/64: emission refused"
fi

# ---------------------------------------------------------------------------
# nimblecas.logic_compile — the things fib/2 cannot reach: the width boundaries, the tail-call
# rewrite, and continuation-passing emission.
#
# fib(0..20) tops out at 6765. It exercises 64-bit arithmetic and calls the 128-bit path with
# numbers that would fit in a byte, so it is not a width test at all. An accumulator factorial
# is, and it is tail recursive, so it tests the loop rewrite at the same time.
# ---------------------------------------------------------------------------
note "logic_compile: widths, tail recursion and continuation passing"

# The boundaries are exact and they are the point: 20! is the largest factorial that fits in 64
# bits and 33! the largest that fits in 128, so the pairs (20, 21) and (33, 34) sit either side
# of each limit. A refusal is the honest answer at the far side; a wrapped value would not be.
gen="${WORK}/fact_64.hpp"
if ! "${BUILD}/prolog_compile" "${REPO}/examples/fact.pl" fact_acc iio cpp 64 \
        > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/64 widths: emission refused"
else
    cat > "${WORK}/fact64_driver.cpp" <<'EOF'
#include "fact_64.hpp"

// 20! fits in 64 bits and 21! does not, so one must be computed exactly and the other REFUSED.
// Refusing is not a limitation being tolerated here -- it is the honesty invariant: the compiled
// program reports that it cannot represent the answer rather than returning a wrapped one.
auto main() -> int {
    nc_int got = nc_lit(0);
    if (!p_fact_acc_3(nc_lit(20), nc_lit(1), got) || got != nc_lit(2432902008176640000LL)) {
        std::println(std::cerr, "20! is wrong or was refused at 64 bits");
        return 1;
    }
    nc_int overflowed = nc_lit(0);
    if (p_fact_acc_3(nc_lit(21), nc_lit(1), overflowed)) {
        std::println(std::cerr, "21! was answered at 64 bits, where it does not fit");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/64 widths" "${gen}" "${WORK}/fact64_driver.cpp"
fi

gen="${WORK}/fact_128.hpp"
if ! "${BUILD}/prolog_compile" "${REPO}/examples/fact.pl" fact_acc iio cpp 128 \
        > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/128 widths: emission refused"
else
    cat > "${WORK}/fact128_driver.cpp" <<'EOF'
#include "fact_128.hpp"

// The expected values are built from 64-bit factors because a 128-bit literal cannot be written
// directly: 21! = 20! * 21, and 33! = 20! * (21*22*...*33), whose second factor is 3569119343741952000
// and fits in 64 bits. Both are therefore exact constants, not a re-run of the algorithm under test.
auto main() -> int {
    const nc_int f20 = nc_lit(2432902008176640000LL);
    const nc_int f21 = f20 * nc_lit(21);
    const nc_int f33 = f20 * nc_lit(3569119343741952000LL);

    nc_int got = nc_lit(0);
    if (!p_fact_acc_3(nc_lit(21), nc_lit(1), got) || got != f21) {
        std::println(std::cerr, "21! is wrong or was refused at 128 bits");
        return 1;
    }
    got = nc_lit(0);
    if (!p_fact_acc_3(nc_lit(33), nc_lit(1), got) || got != f33) {
        std::println(std::cerr, "33! is wrong or was refused at 128 bits");
        return 1;
    }
    // And 128 bits must be a real boundary, not a wider-looking 64: 34! is past it.
    nc_int overflowed = nc_lit(0);
    if (p_fact_acc_3(nc_lit(34), nc_lit(1), overflowed)) {
        std::println(std::cerr, "34! was answered at 128 bits, where it does not fit");
        return 1;
    }
    // The width must actually BE 128 bits wide, not an alias that compiled.
    static_assert(sizeof(nc_int) == 16, "the 128-bit width did not emit a 128-bit type");
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/128 widths" "${gen}" "${WORK}/fact128_driver.cpp"
fi

# The tail-call rewrite, tested by the only thing that can tell the difference: a recursion deep
# enough that the un-rewritten form would exhaust the stack. A million frames would; a loop does
# not notice. count/3 sums 1..N, so the answer is N(N+1)/2 and stays inside 64 bits.
gen="${WORK}/count_64.hpp"
if ! "${BUILD}/prolog_compile" "${REPO}/examples/countdown.pl" count iio cpp 64 \
        > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/64 tail recursion: emission refused"
elif ! grep -q "for (;;)" "${gen}"; then
    fail "logic_compile cpp/64 tail recursion: no loop in the emitted source, so the self tail call was not rewritten"
else
    cat > "${WORK}/count_driver.cpp" <<'EOF'
#include "count_64.hpp"

// One million frames. Compiled as recursion this overflows the stack and the process dies;
// compiled as the loop the emitter promises, it returns 500000500000.
auto main() -> int {
    nc_int got = nc_lit(0);
    if (!p_count_3(nc_lit(1000000), nc_lit(0), got)) {
        std::println(std::cerr, "count/3 failed on an input it should accept");
        return 1;
    }
    if (got != nc_lit(500000500000LL)) {
        std::println(std::cerr, "count(1000000, 0, R) disagreed");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/64 tail recursion" "${gen}" "${WORK}/count_driver.cpp"
fi

# Continuation-passing emission. A nondeterministic predicate compiled deterministically would be
# truncated to its first solution, so what has to be checked is that EVERY solution arrives, in
# order -- and that a continuation returning false stops the enumeration where it said to.
gen="${WORK}/between_cps.hpp"
if ! "${BUILD}/prolog_compile" "${REPO}/examples/between.pl" between iio cpp 64 cps \
        > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/64 cps: emission refused"
else
    cat > "${WORK}/between_driver.cpp" <<'EOF'
#include "between_cps.hpp"

auto main() -> int {
    std::vector<std::int64_t> seen;
    const bool completed = p_between_3(nc_lit(3), nc_lit(9), [&](nc_int x) -> bool {
        seen.push_back(static_cast<std::int64_t>(x));
        return true;
    });
    if (!completed) {
        std::println(std::cerr, "the enumeration reported that it did not complete");
        return 1;
    }
    const std::vector<std::int64_t> expected{3, 4, 5, 6, 7, 8, 9};
    if (seen != expected) {
        std::println(std::cerr, "between(3, 9, X) enumerated {} solutions, expected {}",
                     seen.size(), expected.size());
        return 1;
    }

    // A continuation that refuses stops the enumeration, and that answer travels back out.
    std::vector<std::int64_t> stopped;
    const bool ran_to_end = p_between_3(nc_lit(3), nc_lit(9), [&](nc_int x) -> bool {
        stopped.push_back(static_cast<std::int64_t>(x));
        return stopped.size() < 3;
    });
    if (ran_to_end) {
        std::println(std::cerr, "the enumeration claimed to complete after being stopped");
        return 1;
    }
    if (stopped != std::vector<std::int64_t>{3, 4, 5}) {
        std::println(std::cerr, "stopping after three solutions did not stop after three");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/64 cps" "${gen}" "${WORK}/between_driver.cpp"
fi

# A clause that CHAINS CALLS, which mode inference could not read until now: a variable bound
# by one call is bound for the next, and a call whose arguments are all inputs is a test rather
# than a producer. Compiling it is half the check -- it also has to fail where the test fails.
cat > "${WORK}/chain.pl" <<'EOF'
positive(X) :- X > 0.
double(X, Y) :- Y is X * 2.
chain(X, Z) :- double(X, T), positive(T), Z is T + 1.
EOF
gen="${WORK}/chain_64.hpp"
if ! "${BUILD}/prolog_compile" "${WORK}/chain.pl" chain io cpp 64 > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/64 chained calls: emission refused"
else
    cat > "${WORK}/chain_driver.cpp" <<'EOF'
#include "chain_64.hpp"

auto main() -> int {
    // double(5) is 10, which is positive, so chain(5) is 11.
    nc_int got = nc_lit(0);
    if (!p_chain_2(nc_lit(5), got) || got != nc_lit(11)) {
        std::println(std::cerr, "chain(5) should be 5*2+1 = 11");
        return 1;
    }
    // double(-1) is -2, which is not, so the clause must FAIL rather than answer
    // -- the all-input call is a guard and has to behave like one.
    nc_int none = nc_lit(0);
    if (p_chain_2(nc_lit(-1), none)) {
        std::println(std::cerr, "chain(-1) succeeded, but the guard should have failed it");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "logic_compile cpp/64 chained calls" "${gen}" "${WORK}/chain_driver.cpp"
fi

# ---------------------------------------------------------------------------
# nimblecas.logic_compile — the 128-bit FIXED-POINT DECIMAL width, on money.
#
# What has to be checked here is not that it compiles but that it is EXACT: that the cents
# survive a rescaled multiplication, and that an answer the scale cannot hold is refused rather
# than rounded. A money type that quietly rounds is worse than one that refuses.
# ---------------------------------------------------------------------------
note "logic_compile: 128-bit decimal money"

gen="${WORK}/money_dec.hpp"
if ! "${BUILD}/prolog_compile" "${REPO}/examples/money.pl" settle iiio cpp dec \
        > "${gen}" 2>/dev/null; then
    fail "logic_compile cpp/decimal: emission refused"
else
    cat > "${WORK}/money_driver.cpp" <<'EOF'
#include "money_dec.hpp"

// Values are minor units: at scale 2, nc_lit(10000) is 100.00 and nc_lit(10) is 0.10.
auto main() -> int {
    int bad = 0;
    const auto check = [&bad](bool ok, std::string_view what) {
        if (!ok) {
            std::println(std::cerr, "{}", what);
            ++bad;
        }
    };

    // The whole transaction: 100.00 plus 10% is 110.00, split four ways is 27.50. It also
    // exercises a CALL between two compiled predicates at this width.
    nc_int each = nc_lit(0);
    check(p_settle_4(nc_lit(10000), nc_lit(10), nc_lit(400), each), "settle/4 failed to run");
    check(nc_to_string(each) == "27.50", "100.00 +10% split four ways was not 27.50");

    // 19.99 scaled by 2.00 is 39.98: the cents must survive the rescale, which is exactly what
    // a naive a*b/10^s gets wrong when it divides at the wrong moment.
    nc_int scaled = nc_lit(0);
    check(p_with_tax_3(nc_lit(1999), nc_lit(200), scaled), "with_tax/3 failed to run");
    check(nc_to_string(scaled) == "59.97", "19.99 plus 200% was not 59.97");

    // 0.01 * 0.01 is 0.0001. Two digits cannot hold it, so it is REFUSED -- not rounded to
    // zero, which is the answer that would look right and cost a business money.
    nc_int lost = nc_lit(0);
    check(!p_with_tax_3(nc_lit(1), nc_lit(1), lost), "0.01 * 0.01 was answered, not refused");

    // Division the same way: exact or nothing.
    nc_int quarter = nc_lit(0);
    check(p_split_3(nc_lit(1000), nc_lit(400), quarter), "10.00 four ways failed to run");
    check(nc_to_string(quarter) == "2.50", "10.00 divided four ways was not 2.50");
    nc_int third = nc_lit(0);
    check(!p_split_3(nc_lit(1000), nc_lit(300), third),
          "10.00 divided three ways was answered, so a cent went missing");

    // Rendering, on the values most likely to be got wrong.
    check(nc_to_string(nc_lit(0)) == "0.00", "zero did not render as 0.00");
    check(nc_to_string(nc_lit(5)) == "0.05", "five minor units did not render as 0.05");
    check(nc_to_string(nc_lit(-1999)) == "-19.99", "a negative amount lost its sign");

    // And it is genuinely 128 bits wide: ten multiplications by 1000.00 reach 10^28 currency
    // units, which 64 bits could not hold.
    static_assert(sizeof(nc_int) == 16, "the decimal width did not emit a 128-bit type");
    nc_int big = nc_lit(1);
    for (int i = 0; i < 10; ++i) {
        const auto step = nc_mul(big, nc_lit(100000));
        check(step.has_value(), "a large but representable multiplication was refused");
        if (!step) {
            break;
        }
        big = *step;
    }
    check(big > (nc_int{1} << 96), "the running total never left 64 bits behind");

    return bad == 0 ? 0 : 1;
}
EOF
    compile_and_run_cpp "logic_compile cpp/decimal" "${gen}" "${WORK}/money_driver.cpp"
fi

# The refusals are as much of the contract as the answers, so they are checked as answers.
mkdir -p "${WORK}/dec"
cat > "${WORK}/dec/bad.pl" <<'EOF'
leftover(A, B, R) :- R is A mod B.
EOF
refusals=0
"${BUILD}/prolog_compile" "${WORK}/dec/bad.pl" leftover iio cpp dec >/dev/null 2>&1 &&
    { echo "    mod was accepted at a decimal width"; refusals=1; }
"${BUILD}/prolog_compile" "${WORK}/dec/bad.pl" leftover iio cpp 64 >/dev/null 2>&1 ||
    { echo "    mod was refused at 64 bits, where it is meaningful"; refusals=1; }
"${BUILD}/prolog_compile" "${REPO}/examples/money.pl" line_total iio cpp dec:18 >/dev/null 2>&1 ||
    { echo "    a scale of 18 was refused"; refusals=1; }
"${BUILD}/prolog_compile" "${REPO}/examples/money.pl" line_total iio cpp dec:19 >/dev/null 2>&1 &&
    { echo "    a scale of 19 was accepted"; refusals=1; }
"${BUILD}/prolog_compile" "${REPO}/examples/money.pl" line_total iio triton dec >/dev/null 2>&1 &&
    { echo "    Triton accepted a decimal, which tl.int64 cannot represent"; refusals=1; }
if [[ "${refusals}" -eq 0 ]]; then
    pass "logic_compile decimal: mod, an out-of-range scale and Triton are all refused"
else
    fail "logic_compile decimal: a refusal did not hold"
fi

# CUDA takes the decimal width too -- __int128 is synthesised there from 64-bit pieces.
gen="${WORK}/money_dec.cu"
if "${BUILD}/prolog_compile" "${REPO}/examples/money.pl" settle iiio cuda dec \
        > "${gen}" 2>/dev/null; then
    cat > "${WORK}/money_cu_driver.cu" <<'EOF'
#include "money_dec.cu"

#include <cstdio>

// The same transaction on the device, and it must give the same minor units. A width that
// disagrees between host and device is worse than one that is missing.
__global__ void settle_kernel(unsigned char* ok, nc_int* out) {
    nc_int each = nc_lit(0);
    *ok = p_settle_4(nc_lit(10000), nc_lit(10), nc_lit(400), each) ? 1 : 0;
    *out = each;
}

int main() {
    unsigned char* d_ok = nullptr;
    nc_int* d_out = nullptr;
    if (cudaMalloc(&d_ok, 1) != cudaSuccess || cudaMalloc(&d_out, sizeof(nc_int)) != cudaSuccess) {
        std::fprintf(stderr, "device allocation failed\n");
        return 1;
    }
    settle_kernel<<<1, 1>>>(d_ok, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "the kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    unsigned char ok = 0;
    nc_int got = 0;
    if (cudaMemcpy(&ok, d_ok, 1, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "reading the result back failed\n");
        return 1;
    }
    if (ok != 1) {
        std::fprintf(stderr, "settle/4 failed on the device\n");
        return 1;
    }
    if (got != static_cast<nc_int>(2750)) {
        std::fprintf(stderr, "the device disagreed with the host on the minor units\n");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cu "logic_compile cuda/decimal" "${gen}" "${WORK}/money_cu_driver.cu"
else
    fail "logic_compile cuda/decimal: emission refused"
fi

# ---------------------------------------------------------------------------
# nimblecas.sat_compile — a CNF formula.
# ---------------------------------------------------------------------------
note "sat_compile: a small satisfiable formula"

cat > "${WORK}/f.cnf" <<'EOF'
c a formula whose smallest satisfying assignment the reference solver also reports
p cnf 6 5
1 -2 3 0
-1 2 0
2 3 -4 0
4 5 -6 0
-3 6 0
EOF

# The reference answer, from the same module, on stderr.
"${BUILD}/sat_compile" "${WORK}/f.cnf" cpp formula > "${WORK}/sat_cpp.hpp" 2>"${WORK}/sat.log"
ANSWER="$(sed -n 's/.*reference smallest assignment = \([0-9]*\).*/\1/p' "${WORK}/sat.log")"
if [[ -z "${ANSWER}" ]]; then
    fail "sat_compile: the reference solver reported no assignment to check against"
else
    cat > "${WORK}/sat_driver.cpp" <<EOF
#include "sat_cpp.hpp"

// Three entry points that must all agree: the scalar scan, the vector one, and the threaded
// fan-out. Disagreement between them is the defect this formula is here to catch.
auto main() -> int {
    const std::uint64_t expected = ${ANSWER}ULL;
    if (formula_solve() != expected) {
        std::println(std::cerr, "formula_solve disagreed");
        return 1;
    }
    if (formula_solve_range(0ULL, nc_formula_blocks) != expected) {
        std::println(std::cerr, "formula_solve_range disagreed");
        return 1;
    }
    if (formula_solve_parallel(4U) != expected) {
        std::println(std::cerr, "formula_solve_parallel disagreed");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "sat_compile cpp/exhaustive" "${WORK}/sat_cpp.hpp" "${WORK}/sat_driver.cpp"
fi

for variant in skc probsat novelty adaptnovelty; do
    gen="${WORK}/sat_walk_${variant}.hpp"
    if ! "${BUILD}/sat_compile" "${WORK}/f.cnf" cpp wsat walksat "${variant}" \
            > "${gen}" 2>/dev/null; then
        fail "sat_compile cpp/walksat/${variant}: emission refused"
        continue
    fi
    cat > "${WORK}/walk_driver_${variant}.cpp" <<EOF
#include "sat_walk_${variant}.hpp"

// Two things are checked, and the first matters more than it looks.
//
// SUCCESS IS REQUIRED, even though the walk is incomplete and UNKNOWN is in general a legitimate
// answer from it. Not here: this formula is satisfiable, and the configuration is fully
// determined -- fixed seed, fixed walker count, fixed flip budget, integer noise -- so the run
// has exactly one outcome, and all four variants are observed to find a model on walker 0.
// Accepting UNKNOWN would mean an emitter defect that breaks the walk outright still passes,
// which is precisely the failure this script exists to catch.
//
// And the model it reports must satisfy every clause. A walker claiming success with an
// assignment that does not is the other half.
auto main() -> int {
    const wsat_result r = wsat_solve(64U, 100000ULL, 50U, 0x9E3779B97F4A7C15ULL);
    if (!r.found) {
        std::println(std::cerr,
                     "the walk found nothing on a satisfiable formula it solves deterministically");
        return 1;
    }
    for (std::uint32_t c = 0; c < nc_wsat_nclauses; ++c) {
        bool sat = false;
        for (std::uint32_t i = nc_wsat_cstart[c]; i < nc_wsat_cstart[c + 1U]; ++i) {
            const std::int32_t lit = nc_wsat_lits[i];
            const auto v = static_cast<std::uint32_t>(lit < 0 ? -lit : lit) - 1U;
            if ((lit > 0) == (r.model[v] != 0U)) {
                sat = true;
                break;
            }
        }
        if (!sat) {
            std::println(std::cerr, "clause {} is not satisfied by the reported model", c);
            return 1;
        }
    }
    return 0;
}
EOF
    compile_and_run_cpp "sat_compile cpp/walksat/${variant}" "${gen}" \
        "${WORK}/walk_driver_${variant}.cpp"
done

gen="${WORK}/sat_exhaustive.cu"
if ! "${BUILD}/sat_compile" "${WORK}/f.cnf" cuda formula exhaustive > "${gen}" 2>/dev/null; then
    fail "sat_compile cuda/exhaustive: emission refused"
elif [[ -z "${ANSWER}" ]]; then
    fail "sat_compile cuda/exhaustive: no reference assignment to check against"
else
    cat > "${WORK}/sat_ex_cu_driver.cu" <<EOF
#include "sat_exhaustive.cu"

#include <cstdio>

// The smallest satisfying assignment, which every target must agree on.
int main() {
    const unsigned long long got = formula_solve_cuda();
    const unsigned long long want = ${ANSWER}ULL;
    if (got != want) {
        std::fprintf(stderr, "the CUDA scan returned %llu, the reference says %llu\n", got, want);
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cu "sat_compile cuda/exhaustive" "${gen}" "${WORK}/sat_ex_cu_driver.cu"
fi

gen="${WORK}/sat_walksat.cu"
if "${BUILD}/sat_compile" "${WORK}/f.cnf" cuda formula walksat > "${gen}" 2>/dev/null; then
    cat > "${WORK}/sat_wk_cu_driver.cu" <<EOF
#include "sat_walksat.cu"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// The oracle is the DIMACS file this script wrote, not the tables the emitter produced: those
// are the kernel's input and so are part of what is being tested.
//
// The emitted wrapper says plainly that the model handed back is the LAST writer's rather than
// strictly the winning walker's, because a lower walker can finish after a higher one. Both
// satisfy the formula, and verifying whichever arrives is exactly what it asks the caller to do.
int main() {
    std::vector<std::vector<int>> clauses;
    std::ifstream in("${WORK}/f.cnf");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == 'c' || line[0] == 'p' || line[0] == '%') { continue; }
        std::istringstream ls(line);
        std::vector<int> lits;
        int lit = 0;
        while (ls >> lit) {
            if (lit != 0) { lits.push_back(lit); }
        }
        if (!lits.empty()) { clauses.push_back(lits); }
    }
    if (clauses.empty()) {
        std::fprintf(stderr, "no clauses were read, so nothing would have been checked\n");
        return 1;
    }

    std::vector<unsigned char> model(nc_formula_nvars, 0);
    const unsigned winner = formula_solve_cuda(64U, 100000ULL, 50U, 0x9E3779B97F4A7C15ULL,
                                               model.data());
    if (winner == 0xFFFFFFFFu) {
        std::fprintf(stderr, "no walker succeeded on a satisfiable formula\n");
        return 1;
    }
    for (const auto& c : clauses) {
        bool sat = false;
        for (const int lit : c) {
            const auto v = static_cast<unsigned>(lit < 0 ? -lit : lit) - 1U;
            if ((lit > 0) == (model[v] != 0)) { sat = true; break; }
        }
        if (!sat) {
            std::fprintf(stderr, "a clause is not satisfied by the model the walk reported\n");
            return 1;
        }
    }
    return 0;
}
EOF
    compile_and_run_cu "sat_compile cuda/walksat" "${gen}" "${WORK}/sat_wk_cu_driver.cu"
else
    fail "sat_compile cuda/walksat: emission refused"
fi

gen="${WORK}/sat_exhaustive.py"
if ! "${BUILD}/sat_compile" "${WORK}/f.cnf" triton formula exhaustive > "${gen}" 2>/dev/null; then
    fail "sat_compile triton/exhaustive: emission refused"
elif [[ -z "${ANSWER}" ]]; then
    fail "sat_compile triton/exhaustive: no reference assignment to check against"
else
    cat > "${WORK}/sat_ex_tri_driver.py" <<EOF
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sat_exhaustive as k

# The same number the C++ and CUDA emissions return, and the same number the reference solver
# in the module printed: the smallest satisfying assignment, not merely a satisfying one.
got = k.formula_solve()
want = ${ANSWER}
if got != want:
    print("the Triton scan returned {}, the reference says {}".format(got, want), file=sys.stderr)
    raise SystemExit(1)
EOF
    check_triton "sat_compile triton/exhaustive" "${gen}" "${WORK}/sat_ex_tri_driver.py"
fi

gen="${WORK}/sat_walksat.py"
if "${BUILD}/sat_compile" "${WORK}/f.cnf" triton formula walksat > "${gen}" 2>/dev/null; then
    cat > "${WORK}/sat_walk_tri_driver.py" <<EOF
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import sat_walksat as k

# The oracle is the DIMACS file this script wrote, parsed here -- not the tables the emitter
# produced, which are half of what is under test.
clauses = []
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "f.cnf")):
    line = line.strip()
    if not line or line[0] in "cp%":
        continue
    lits = [int(tok) for tok in line.split() if int(tok) != 0]
    if lits:
        clauses.append(lits)


def unsat_count(row):
    return sum(
        0 if any((lit > 0) == (row[abs(lit) - 1] == 1) for lit in c) else 1 for c in clauses
    )


# Every assignment of six variables, so the check is exhaustive rather than a sample: the
# emitted kernel must agree with the formula on all 64 of them, and the emitted literal and
# clause-start tables are the kernel's input, so they are under test too.
rows = [[(a >> v) & 1 for v in range(k.NC_VARS)] for a in range(1 << k.NC_VARS)]
assign = torch.tensor(rows, dtype=torch.int32, device="cuda")
lits = torch.tensor(k.NC_formula_LITS, dtype=torch.int32, device="cuda")
cstart = torch.tensor(k.NC_formula_CSTART, dtype=torch.int32, device="cuda")
out = torch.zeros((len(rows),), dtype=torch.int32, device="cuda")
k.formula_score_kernel[(len(rows),)](
    assign, lits, cstart, out, len(rows), k.NC_VARS, k.NC_CLAUSES, BLOCK=16
)

scored = out.tolist()
for row, got in zip(rows, scored):
    want = unsat_count(row)
    if got != want:
        print(
            "the scorer counted {} unsatisfied clauses under {}, the formula says {}".format(
                got, row, want
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
if min(scored) != 0:
    print("no assignment scored zero on a satisfiable formula", file=sys.stderr)
    raise SystemExit(1)

# And the emitted host helper, which is the entry a caller actually uses.
best_assign, best_counts = k.formula_sample_and_score(n_samples=1024, seed=1)
if int(best_counts[0].item()) != 0:
    print("the sampler's best draw was not a model", file=sys.stderr)
    raise SystemExit(1)
if unsat_count(best_assign[0].tolist()) != 0:
    print("the sampler called an assignment a model that is not one", file=sys.stderr)
    raise SystemExit(1)
EOF
    check_triton "sat_compile triton/walksat" "${gen}" "${WORK}/sat_walk_tri_driver.py"
else
    fail "sat_compile triton/walksat: emission refused"
fi

# ---------------------------------------------------------------------------
# nimblecas.csp_compile — n-queens.
# ---------------------------------------------------------------------------
note "csp_compile: 6-queens"

"${BUILD}/csp_compile" 6 cpp ncq exhaustive > "${WORK}/csp_ex.hpp" 2>"${WORK}/csp.log"
FIRST="$(sed -n 's/.*reference first solution = \(.*\) *$/\1/p' "${WORK}/csp.log" | tr -s ' ' | sed 's/ $//')"
if [[ -z "${FIRST}" ]]; then
    fail "csp_compile: the reference search reported no solution to check against"
else
    cat > "${WORK}/csp_driver.cpp" <<EOF
#include "csp_ex.hpp"

// The exhaustive scan reports the LEXICOGRAPHICALLY FIRST solution, which is exactly what the
// module's own backtracking search returns -- so the two are comparable value for value, not
// merely both satisfying.
auto main() -> int {
    const std::vector<std::int64_t> expected{$(echo "${FIRST}" | tr ' ' ',')};
    std::vector<std::int64_t> got(expected.size(), 0);
    if (!ncq_solve_exhaustive(got)) {
        std::println(std::cerr, "the emitted scan found nothing");
        return 1;
    }
    if (got != expected) {
        std::println(std::cerr, "the emitted scan disagreed with the reference");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "csp_compile cpp/exhaustive" "${WORK}/csp_ex.hpp" "${WORK}/csp_driver.cpp"
fi

if "${BUILD}/csp_compile" 6 cpp ncq minconflicts > "${WORK}/csp_mc.hpp" 2>/dev/null; then
    cat > "${WORK}/csp_mc_driver.cpp" <<'EOF'
#include "csp_mc.hpp"

// As with the WalkSAT drivers: local search reports UNKNOWN in general, but not on this input.
// 6-queens is solvable, the seed and walker count are fixed, and the walk is observed to find a
// solution -- so a false return here means the emitted walk is broken, not that it was unlucky.
// And a true return must come with an assignment that actually satisfies every constraint.
auto main() -> int {
    std::vector<std::int64_t> got(6, 0);
    if (!ncq_solve_min_conflicts(got)) {
        std::println(std::cerr,
                     "the walk found nothing on a 6-queens instance it solves deterministically");
        return 1;
    }
    if (!ncq_satisfies(got)) {
        std::println(std::cerr, "the walk reported a solution that violates a constraint");
        return 1;
    }
    return 0;
}
EOF
    compile_and_run_cpp "csp_compile cpp/min_conflicts" "${WORK}/csp_mc.hpp" \
        "${WORK}/csp_mc_driver.cpp"
else
    fail "csp_compile cpp/min_conflicts: emission refused"
fi

gen="${WORK}/csp_exhaustive.cu"
if ! "${BUILD}/csp_compile" 6 cuda ncq exhaustive > "${gen}" 2>/dev/null; then
    fail "csp_compile cuda/exhaustive: emission refused"
elif [[ -z "${FIRST}" ]]; then
    fail "csp_compile cuda/exhaustive: no reference solution to check against"
else
    cat > "${WORK}/csp_ex_cu_driver.cu" <<EOF
#include "csp_exhaustive.cu"

#include <cstdio>

// The lexicographically first solution, the same one the module's own backtracking search
// returns and the C++ emission is checked against.
int main() {
    const long long expected[] = {$(echo "${FIRST}" | tr ' ' ',')};
    constexpr int n = static_cast<int>(sizeof(expected) / sizeof(expected[0]));
    long long got[n] = {};
    if (!ncq_solve_exhaustive(got)) {
        std::fprintf(stderr, "the emitted scan found nothing\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            std::fprintf(stderr, "the emitted scan disagreed with the reference\n");
            return 1;
        }
    }
    return 0;
}
EOF
    compile_and_run_cu "csp_compile cuda/exhaustive" "${gen}" "${WORK}/csp_ex_cu_driver.cu"
fi

gen="${WORK}/csp_minconflicts.cu"
if "${BUILD}/csp_compile" 6 cuda ncq minconflicts > "${gen}" 2>/dev/null; then
    cat > "${WORK}/csp_mc_cu_driver.cu" <<'EOF'
#include "csp_minconflicts.cu"

#include <cstdio>
#include <cstdlib>

// As with the C++ walk: 6-queens is solvable and the configuration is fixed, so a false return
// is a broken walk rather than bad luck. And the assignment must actually be a solution --
// checked here from the PROBLEM, because the emitted ncq_satisfies is __device__ and the point
// is not to ask the code under test whether it is right.
int main() {
    constexpr int n = 6;
    long long got[n] = {};
    if (!ncq_solve_min_conflicts(got)) {
        std::fprintf(stderr, "the walk found nothing on a 6-queens instance\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (got[i] < 0 || got[i] >= n) {
            std::fprintf(stderr, "the walk reported a value outside the domain\n");
            return 1;
        }
        for (int j = i + 1; j < n; ++j) {
            if (got[i] == got[j] || std::llabs(got[i] - got[j]) == j - i) {
                std::fprintf(stderr, "the walk reported an assignment that is not a solution\n");
                return 1;
            }
        }
    }
    return 0;
}
EOF
    compile_and_run_cu "csp_compile cuda/min_conflicts" "${gen}" "${WORK}/csp_mc_cu_driver.cu"
else
    fail "csp_compile cuda/minconflicts: emission refused"
fi

if ! "${BUILD}/csp_compile" 6 triton ncq > "${WORK}/csp_triton.py" 2>/dev/null; then
    fail "csp_compile triton: emission refused"
elif [[ -z "${FIRST}" ]]; then
    fail "csp_compile triton: the reference search reported no solution to check against"
else
    cat > "${WORK}/csp_tri_driver.py" <<EOF
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton

import csp_triton as k

N = k.ncq_NUM_VARS


def is_solution(row):
    """6-queens, stated from the problem rather than from the emitter's constraint list."""
    for i in range(N):
        for j in range(i + 1, N):
            if row[i] == row[j] or abs(row[i] - row[j]) == j - i:
                return False
    return True


# The whole 6**6 assignment space. The kernel counts VIOLATED CONSTRAINTS, and its all-different
# constraint is one constraint rather than fifteen, so the counts are not comparable term by
# term -- but a count of zero means a solution, and that set is comparable exactly.
rows = []
for a in range(N ** N):
    row = []
    rest = a
    for _ in range(N):
        row.append(rest % N)
        rest //= N
    rows.append(row)

assign = torch.tensor(rows, dtype=torch.int64, device="cuda")
conflicts = torch.zeros((len(rows),), dtype=torch.int64, device="cuda")
BLOCK = 256
k.ncq_conflict_kernel[(triton.cdiv(len(rows), BLOCK),)](
    assign, conflicts, len(rows), BLOCK=BLOCK
)

scored = conflicts.tolist()
kernel_says = {i for i, c in enumerate(scored) if c == 0}
truth = {i for i, row in enumerate(rows) if is_solution(row)}
if kernel_says != truth:
    wrong = sorted(kernel_says ^ truth)[:5]
    print(
        "the scorer and the problem disagree on {} of {} assignments, e.g. {}".format(
            len(kernel_says ^ truth), len(rows), [rows[i] for i in wrong]
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)
if not truth:
    print("6-queens has solutions; the check found none and so proves nothing", file=sys.stderr)
    raise SystemExit(1)

# And the solution the module's own reference search reported must score zero.
reference = [$(echo "${FIRST}" | tr ' ' ',')]
if rows.index(reference) not in kernel_says:
    print("the scorer found conflicts in the reference solution", file=sys.stderr)
    raise SystemExit(1)
EOF
    check_triton "csp_compile triton" "${WORK}/csp_triton.py" "${WORK}/csp_tri_driver.py"
fi

# ---------------------------------------------------------------------------
note "summary"
printf '%d checks, %d failed\n' "${checks}" "${failures}"
[[ "${failures}" -eq 0 ]]
