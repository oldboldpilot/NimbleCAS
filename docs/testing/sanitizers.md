# Sanitizer & Memory-Safety Testing

Per Code Policy Rule 36, NimbleCAS is exercised under the full sanitizer set plus
valgrind. All runs are on the Linux/clang-22/libc++ build (the reference toolchain).

## Running

```bash
NIMBLECAS_SANITIZE=ON        scripts/build.sh   # AddressSanitizer + UBSan + LeakSanitizer
NIMBLECAS_SANITIZER=thread   scripts/build.sh   # ThreadSanitizer (TBB false positives suppressed)
NIMBLECAS_SANITIZER=undefined scripts/build.sh  # UndefinedBehaviorSanitizer only
NIMBLECAS_SANITIZER=memory   scripts/build.sh   # MemorySanitizer (see caveat below)
```

Each sanitizer uses its own build dir (`build-san-<name>`). Python bindings are skipped
under sanitizers. valgrind runs against the ordinary release `build/`.

## Status

| Tool | Result |
| :--- | :--- |
| AddressSanitizer (ASan) | clean |
| LeakSanitizer (LSan) | clean |
| UndefinedBehaviorSanitizer (UBSan) | clean |
| ThreadSanitizer (TSan) | **0 races in NimbleCAS code** (see TBB note) |
| valgrind memcheck (`--leak-check=full`) | no leaks, no errors |
| MemorySanitizer (MSan) | **clean** against an MSan-instrumented libc++ (see note) |

## ThreadSanitizer + oneTBB

The prebuilt system `libtbb.so` is **not** TSan-instrumented, so TSan cannot see TBB's
internal synchronization and reports its work-stealing (blocked_range / partitioner
splits) as data races. These are false positives — NimbleCAS frames appear only as the
*callers* into TBB. They are suppressed via `config/tsan.supp`
(`called_from_lib:libtbb.so`), applied automatically by `scripts/build.sh`. With that
suppression, TSan reports **zero** races across the concurrent suites (parallel,
simplify, diff, symbolic), confirming the parallel tree engine is race-free.

A fully-instrumented TSan run would require rebuilding oneTBB from source with
`-fsanitize=thread`; suppressing the prebuilt library is the standard alternative.

## MemorySanitizer

MSan requires that **all** code — including the C++ standard library — be instrumented.
The *system* libc++ is not, so against it MSan reports false "use-of-uninitialized-value"
inside ordinary `std::string`/`std::vector` operations. Building an MSan-instrumented
libc++ resolves this, and against it NimbleCAS is **MSan-clean**.

Build the MSan libc++ once (from the matching llvm-project source):

```bash
git clone --depth 1 --branch llvmorg-22.1.8 --filter=blob:none --sparse \
    https://github.com/llvm/llvm-project /scratch/llvm-project
cd /scratch/llvm-project && git sparse-checkout set runtimes libcxx libcxxabi libunwind cmake libc
cmake -S /scratch/llvm-project/runtimes -B /scratch/msan-libcxx -G Ninja \
  -DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 \
  -DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind' \
  -DLLVM_USE_SANITIZER=MemoryWithOrigins -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_TESTS=OFF -DLIBCXXABI_INCLUDE_TESTS=OFF \
  -DLIBUNWIND_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_BENCHMARKS=OFF -DLIBCXX_ENABLE_STD_MODULES=ON
ninja -C /scratch/msan-libcxx cxx cxxabi unwind
```

Then run MSan over the core symbolic/SIMD code:

```bash
MSAN_LIBCXX=/scratch/msan-libcxx scripts/build_msan.sh
```

Result: `core_tests`, `simd_tests`, and `symbolic_tests` all pass with **no MSan
reports**. Tests that exercise oneTBB (the large-tree `simplify`/`diff`/`parallel`
cases) are excluded because the prebuilt TBB is not MSan-instrumented; MSan-checking
those would additionally require an MSan-instrumented oneTBB.
