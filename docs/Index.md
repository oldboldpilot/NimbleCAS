# NimbleCAS Documentation

**Author:** Olumuyiwa Oluwasanmi

Welcome to the documentation hub for **NimbleCAS** — a high-performance,
modern Computer Algebra System in **C++23**. It combines exact symbolic algebra
(Joel S. Cohen's algorithms) with a high-performance numeric path (multi-register
SIMD, dense polynomial arithmetic) on a foundation of immutable copy-on-write
expression trees, railway-oriented error handling (`std::expected`), and
deterministic TBB/PPL parallelism. This page links to every documentation topic
in the repository.

## Overview & product

| Document | What it covers |
| :--- | :--- |
| [Product Requirement Document](PRD.md) | Product scope, vision, and mathematical/algorithmic goals. |
| [Roadmap](ROADMAP.md) | Technical architecture, module structures, and the implementation plan. |
| [README](../README.md) | Project summary, current status, and build entry points. |

## Getting started

| Document | What it covers |
| :--- | :--- |
| [Quickstart](QUICKSTART.md) | Prerequisites, build & test on Linux/macOS and Windows, the uv Python environment. |

## Architecture

| Document | What it covers |
| :--- | :--- |
| [Architecture overview](architecture/overview.md) | The module graph, COW data model, error model, `import std` strategy, and parallel strategy. |
| [Parallel tree computation](architecture/parallel-tree-computation.md) | Why tree manipulation parallelizes by construction; fork–join, grain control, determinism, hash-consing. |

## Module reference

The symbolic chain (`core → symbolic → {simplify, cache} → diff`):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.core` | [core.md](reference/core.md) | `MathError`/`Result`, `make_error`, `CowPtr<T>`, `cache_line_size`. |
| `nimblecas.symbolic` | [symbolic.md](reference/symbolic.md) | Immutable `Expr` trees, node kinds, structural equality/hashing, `free_of`, `substitute`. |
| `nimblecas.simplify` | [simplify.md](reference/simplify.md) | Cohen automatic simplification (ASAE): folding, identities, canonical order, like-term combination. |
| `nimblecas.cache` | [cache.md](reference/cache.md) | `ExprMemo` sharded concurrent hash-consing / memoization. |
| `nimblecas.diff` | [diff.md](reference/diff.md) | Symbolic differentiation with an elementary + special-function derivative table. |

The runtime and numeric chain (`core → simd → polynomial → {polyexpr, ratpoly → {pfd → ratint, resultant → rothstein} → integrate}}`; `parallel`):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.parallel` | [parallel.md](reference/parallel.md) | Deterministic fork–join over TBB/PPL/serial; order-preserving tree combinators. |
| `nimblecas.simd` | [simd.md](reference/simd.md) | Runtime-dispatched elementwise `float32` SIMD kernels (AVX-512 → AVX2 → scalar). |
| `nimblecas.polynomial` | [polynomial.md](reference/polynomial.md) | Dense univariate `int64` polynomials: ring ops, gcd, square-free factorization, SIMD batch eval. |
| `nimblecas.ratpoly` | [ratpoly.md](reference/ratpoly.md) | Exact `Rational` and dense polynomials over `Q[x]`: division-with-remainder, monic Euclidean gcd. |
| `nimblecas.polyexpr` | [polyexpr.md](reference/polyexpr.md) | Bridge between `Expr` and `Polynomial`; polynomial gcd / square-free factor at the `Expr` level. |
| `nimblecas.pfd` | [pfd.md](reference/pfd.md) | Square-free partial-fraction decomposition over `Q[x]`: Yun factorization, Bezout split, base-`b` power expansion. |
| `nimblecas.ratint` | [ratint.md](reference/ratint.md) | Hermite reduction of `int A/B dx` over `Q`: exact rational part plus a square-free-denominator logarithmic integrand. |
| `nimblecas.resultant` | [resultant.md](reference/resultant.md) | Resultant and discriminant over `Q[x]` via the Euclidean remainder sequence: common-factor / repeated-root detection. |
| `nimblecas.rothstein` | [rothstein.md](reference/rothstein.md) | Rothstein–Trager logarithmic integration over `Q(x)`: the residue resultant `R(t) = res_x(D, A − t·D')`, rational-residue logarithms of a square-free-denominator integrand. |
| `nimblecas.integrate` | [integrate.md](reference/integrate.md) | Rational-function integration capstone over `Q(x)`: Hermite reduction then Rothstein–Trager, assembling `int A/B dx = rational part + sum of residue-weighted logarithms`. |

Tooling and integration:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.testing` | [testing.md](reference/testing.md) | Internal, dependency-free test framework (`TestSuite`/`TestContext`) wired to ctest. |
| `nimblecas_ext` (Python) | [python-bindings.md](reference/python-bindings.md) | nanobind bindings: the `Expr` API, module functions, and `MathError`→exception translation. |

## Testing

| Document | What it covers |
| :--- | :--- |
| [Sanitizers & memory-safety](testing/sanitizers.md) | ASan/UBSan/LSan/TSan/MSan + valgrind status and how to run each. |
| [Internal test framework](reference/testing.md) | The `nimblecas.testing` runner and ctest integration. |

## Module-dependency diagram

```
                nimblecas.core
                      │
      ┌───────────────┼───────────────────────────┐
      │               │                            │
nimblecas.simd  nimblecas.parallel          nimblecas.symbolic
      │               │        │                   │
      ▼               └────────┼──────────┐        │
nimblecas.polynomial           ▼          ▼        ▼
      │      │          nimblecas.cache  nimblecas.simplify
      ▼      ▼                 │                   │
nimblecas.  nimblecas.ratpoly  └────────┬──────────┤
 polyexpr                │              ▼          ▼
                         │       nimblecas.diff ◄──┘
              ┌──────────┴──────────┐
              ▼                     ▼
        nimblecas.pfd       nimblecas.resultant
              │                     │
              ▼                     ▼
        nimblecas.ratint    nimblecas.rothstein
              │                     │
              └──────────┬──────────┘
                         ▼
               nimblecas.integrate

nimblecas.testing   (stands alone)
nimblecas_ext       (nanobind: imports symbolic, simplify, diff, polyexpr)
```

See the [architecture overview](architecture/overview.md) for the exact `import`
edges and the rationale.

## Build & test quickstart

```bash
# Linux/macOS — clang++-22 + libc++ + CMake >= 3.30 + Ninja
scripts/build.sh                         # configure, build, run tests (ctest)
NIMBLECAS_SANITIZE=ON scripts/build.sh   # ASan + UBSan + LSan build

# Windows — Visual Studio's bundled clang + MSVC STL (run in Git Bash)
scripts/build_win.sh

# Python bindings (Linux) — uv-managed dependencies
scripts/setup_python.sh                  # uv sync -> .venv + nanobind
```

Full details are in the [Quickstart](QUICKSTART.md). All code adheres to
`config/cpp_details.txt` (the authoritative code policy); every major change is
adversarially code-reviewed before it lands.
