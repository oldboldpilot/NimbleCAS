# NimbleCAS Quickstart

## Prerequisites

- **clang++-22** with **libc++** (provides `share/libc++/v1/std.cppm` for `import std`)
- **CMake ≥ 3.30** and **Ninja**

The reference build host is `oluwasanmi-multigpu-server`, which has clang-22, libc++,
TBB, CMake 4.3, and Ninja installed. Windows dev boxes edit sources; builds run on the
clang-22 host.

## Build & test

```bash
scripts/build.sh
```

This configures a Ninja build under `build/`, compiles the modules, and runs the test
suite via `ctest`. It sources `scripts/build_common.sh`, which defines the authoritative
`CANONICAL_FLAGS` (Code Policy Rule 50) and resolves the repo root via git.

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
