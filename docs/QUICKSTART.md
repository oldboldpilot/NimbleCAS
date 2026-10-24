# NimbleCAS Quickstart

## Prerequisites

NimbleCAS is cross-platform C++23 and builds under two clang toolchains:

- **Linux/macOS:** `clang++-23` with **libc++** (provides `share/libc++/v1/std.cppm`
  for `import std`), CMake ≥ 3.30, Ninja. Reference host `oluwasanmi-multigpu-server`.
- **Windows:** the **newest clang** (`clang++.exe` targeting `x86_64-pc-windows-msvc`)
  with the **MSVC STL**, using `import std` from the MSVC toolset's `std.ixx`. Install
  the latest LLVM release under `C:\tools\LLVM<major>` (or `C:\Program Files\LLVM`);
  Visual Studio supplies the STL, CMake and Ninja.

## Windows build

```bash
scripts/build_win.sh      # run in Git Bash — configures build-win/, builds, runs ctest
```

This picks the highest-versioned clang among `$NIMBLECAS_LLVM_BIN`, `C:\tools\LLVM*\bin`,
`C:\Program Files\LLVM\bin` and Visual Studio's bundled copy (which lags upstream, so it
is the last resort), and Visual Studio's CMake/Ninja. The whole engine and test suite
build and run identically to the Linux build. (Python bindings are Linux-only for now;
the C++ engine is fully cross-platform.)

**Visual Studio updates and module files.** clang stamps the MSVC version it finds in
`cl.exe` into every BMI, and a VS update patches `cl.exe` in place. `cmake/CanonicalFlags.cmake`
therefore pins `-fms-compatibility-version=<major>.<minor>.0` at configure time, so a patch
update leaves `build-win/` valid. If a build ever fails with *"Microsoft Visual C/C++ Version
differs in precompiled file"* (the first error; every later one reads as a missing `std`),
delete the `.pcm` files under `build-win/` and rebuild. `ToolchainBMIGuard` will not do it
for you here: it fingerprints the clang binary, and in this failure clang did not change.

## Python environment (uv)

All Python dependencies are managed with **uv** from `pyproject.toml`:

```bash
scripts/setup_python.sh      # uv sync -> creates .venv, installs nanobind, writes uv.lock
```

`scripts/build.sh` provisions this automatically on first use when building the bindings.

## Build & test

```bash
scripts/build.sh
```

This configures a Ninja build under `build/`, compiles the modules, builds the nanobind
Python extension (when the uv `.venv` is present), and runs the test suites via `ctest`.
It sources `scripts/build_common.sh`, which defines the authoritative `CANONICAL_FLAGS`
(Code Policy Rule 50) and resolves the repo root via git.

### Sanitizer build

```bash
NIMBLECAS_SANITIZE=ON scripts/build.sh
```

Adds AddressSanitizer + UndefinedBehaviorSanitizer + LeakSanitizer (Rules 36/56).

## Layout

| Path | Contents |
| :--- | :--- |
| `src/core/core.cppm` | `nimblecas.core` — types, `CowPtr`, `Result`/`MathError` |
| `src/testing/testing.cppm` | `nimblecas.testing` — internal test framework |
| `tests/` | test executables (registered with ctest) |
| `cmake/` | `CanonicalFlags.cmake`, `StdModule.cmake` |
| `config/toolchain.cmake` | clang-23 toolchain file |
| `scripts/` | `build_common.sh` (canonical flags), `build.sh` (driver) |

## Notes / deviations

The build currently uses the **system libc++-23** std module rather than a vendored
`external/libcxx-v1` tree (Code Policy Rule 51). This is fine on the fixed build server;
vendoring is required before multi-host / cloud builds for bit-reproducibility.
