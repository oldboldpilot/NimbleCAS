# NimbleCAS

A high-performance, modern **Computer Algebra System** in **C++23** — symbolic algebra
(Joel S. Cohen's algorithms), numeric solvers, multi-register SIMD, TBB/PPL parallelism,
and multi-GPU acceleration.

See [docs/PRD.md](docs/PRD.md) for the product scope and
[docs/ROADMAP.md](docs/ROADMAP.md) for the technical architecture and implementation plan.

## Status

Phase 1 (Core Framework) — in progress. Implemented so far:

- `nimblecas.core` — fundamental types, `CowPtr<T>` copy-on-write pointer, and
  `Result<T>` / `MathError` railway-oriented error handling (`std::expected`).
- `nimblecas.testing` — internal test framework (no external test dependency).

## Building

NimbleCAS targets **clang++-22 with libc++** and C++23 modules (`import std`). See
[docs/QUICKSTART.md](docs/QUICKSTART.md) for the full setup. In short, on a host with
clang-22, CMake ≥ 3.30, and Ninja:

```bash
scripts/build.sh                    # configure, build, run tests
NIMBLECAS_SANITIZE=ON scripts/build.sh   # ASan + UBSan + LSan build
```

## Engineering policy

All code adheres to [config/cpp_details.txt](config/cpp_details.txt) (the authoritative
code policy) and [config/update_policy.txt](config/update_policy.txt) (commit / update
policy).
