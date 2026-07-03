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
| MemorySanitizer (MSan) | not runnable with system libc++ (see note) |

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

## MemorySanitizer caveat

MSan requires that **all** code — including the C++ standard library — be instrumented.
The system libc++ (and TBB) are not, so MSan reports false "use-of-uninitialized-value"
inside ordinary `std::string`/`std::vector` operations (e.g. a `SymbolNode::name`
comparison whose bytes came from uninstrumented libc++). Running MSan meaningfully
requires an MSan-instrumented libc++ build; that is tracked as a follow-up. The
combination of ASan + LSan + UBSan + TSan + valgrind already covers memory-safety,
leaks, undefined behaviour, and data races.
