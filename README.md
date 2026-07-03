# NimbleCAS

A high-performance, modern **Computer Algebra System** in **C++23** — symbolic algebra
(Joel S. Cohen's algorithms), numeric solvers, multi-register SIMD, TBB/PPL parallelism,
and multi-GPU acceleration.

See [docs/PRD.md](docs/PRD.md) for the product scope and
[docs/ROADMAP.md](docs/ROADMAP.md) for the technical architecture and implementation plan.

## Status

Early implementation. Working today (each module built, tested, and adversarially
reviewed):

- `nimblecas.core` — fundamental types, `CowPtr<T>` copy-on-write pointer, and
  `Result<T>` / `MathError` railway-oriented error handling (`std::expected`).
- `nimblecas.symbolic` — immutable `Expr` expression trees (a `std::variant` of
  symbol / constant / add / mul / power / function nodes), structural equality,
  structural hashing, and the Cohen primitives `free_of` and `substitute`.
- `nimblecas.simplify` — Cohen automatic simplification: exact overflow-checked
  rational constant folding, algebraic identities, flattening, canonical ordering,
  and combination of like terms / like bases.
- `nimblecas.diff` — symbolic differentiation (sum, product/Leibniz, general power,
  and chain rules) with an elementary + special-function derivative table
  (trig, inverse-trig, hyperbolic, `erf`, `gamma`, `lambertW`, …).
- `nimblecas.testing` — internal test framework (no external test dependency).
- **Python bindings** via nanobind (`nimblecas_ext`), dependencies managed with uv.

## Architecture

Expressions are **immutable trees** shared through a copy-on-write pointer
(`CowPtr<T>`, backed by `shared_ptr`), so copies are O(1) refcount bumps and are safe
to read concurrently — the foundation for later parallel evaluation. Nodes are a
type-safe `std::variant` (no inheritance) to stay small and cache-friendly.

The engine is layered as one C++23 **module per concern**, with explicit dependency
edges:

```
core  ─►  symbolic  ─►  simplify  ─►  diff
                └────────────┴───────────┴──►  bindings (nanobind)
```

Errors use **railway-oriented programming** — fallible operations return
`Result<T> = std::expected<T, MathError>` rather than throwing (e.g. a zero
denominator, `0^0`, or integer overflow surfaces as a `MathError`). The Python layer
translates a `MathError` into an exception at the boundary.

`import std` is provided by compiling the standard library's std module source as an
ordinary module-library target (`cmake/StdModule.cmake`) — libc++'s `std.cppm` on
Linux, the MSVC toolset's `std.ixx` on Windows — which keeps module dependencies
explicit and reproducible without CMake's version-locked experimental `import std`.

## Building

NimbleCAS is cross-platform C++23 and builds under clang with two standard libraries:

```bash
# Linux/macOS — clang++-22 + libc++ + CMake >= 3.30 + Ninja
scripts/build.sh                         # configure, build, run tests
NIMBLECAS_SANITIZE=ON scripts/build.sh   # ASan + UBSan + LSan build

# Windows — Visual Studio's bundled clang + MSVC STL (run in Git Bash)
scripts/build_win.sh
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for details.

## Engineering policy

All code adheres to [config/cpp_details.txt](config/cpp_details.txt) (the authoritative
code policy) and [config/update_policy.txt](config/update_policy.txt) (commit / update
policy). Every major change is adversarially code-reviewed before it lands.
