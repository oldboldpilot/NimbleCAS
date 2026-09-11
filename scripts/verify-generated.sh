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
    if "${BUILD}/prolog_compile" "${REPO}/examples/fib.pl" fib io cuda "${width}" \
            > "${gen}" 2>/dev/null; then
        compile_cu "logic_compile cuda/${width}" "${gen}"
    else
        fail "logic_compile cuda/${width}: emission refused"
    fi
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

for strategy in exhaustive walksat; do
    gen="${WORK}/sat_${strategy}.cu"
    if "${BUILD}/sat_compile" "${WORK}/f.cnf" cuda formula "${strategy}" > "${gen}" 2>/dev/null; then
        compile_cu "sat_compile cuda/${strategy}" "${gen}"
    else
        fail "sat_compile cuda/${strategy}: emission refused"
    fi
done

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

for strategy in exhaustive minconflicts; do
    gen="${WORK}/csp_${strategy}.cu"
    if "${BUILD}/csp_compile" 6 cuda ncq "${strategy}" > "${gen}" 2>/dev/null; then
        compile_cu "csp_compile cuda/${strategy}" "${gen}"
    else
        fail "csp_compile cuda/${strategy}: emission refused"
    fi
done

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
