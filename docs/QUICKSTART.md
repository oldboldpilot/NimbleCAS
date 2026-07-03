# NimbleCAS Quickstart

## Prerequisites

NimbleCAS is cross-platform C++23 and builds under two clang toolchains:

- **Linux/macOS:** `clang++-22` with **libc++** (provides `share/libc++/v1/std.cppm`
  for `import std`), CMake ≥ 3.30, Ninja. Reference host `oluwasanmi-multigpu-server`.
- **Windows:** the **clang + MSVC-STL** toolchain bundled with Visual Studio
  (`clang++.exe` targeting `x86_64-pc-windows-msvc`), using `import std` from the MSVC
  toolset's `std.ixx`. CMake + Ninja are also bundled with Visual Studio.

## Windows build

```bash
scripts/build_win.sh      # run in Git Bash — configures build-win/, builds, runs ctest
```

This finds the Visual Studio-bundled clang++/CMake/Ninja/llvm-rc automatically. The
whole engine and test suite build and run identically to the Linux build. (Python
bindings are Linux-only for now; the C++ engine is fully cross-platform.)

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
| `config/toolchain.cmake` | clang-22 toolchain file |
| `scripts/` | `build_common.sh` (canonical flags), `build.sh` (driver) |

## Notes / deviations

The build currently uses the **system libc++-22** std module rather than a vendored
`external/libcxx-v1` tree (Code Policy Rule 51). This is fine on the fixed build server;
vendoring is required before multi-host / cloud builds for bit-reproducibility.
