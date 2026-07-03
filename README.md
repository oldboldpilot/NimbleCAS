# NimbleCAS

A high-performance, modern **Computer Algebra System** in **C++23** — symbolic algebra
(Joel S. Cohen's algorithms), numeric solvers, multi-register SIMD, TBB/PPL parallelism,
and multi-GPU acceleration.

**Start here: [docs/Index.md](docs/Index.md) — the documentation hub** (architecture,
per-module reference, and testing guides). See [docs/PRD.md](docs/PRD.md) for the
product scope and [docs/ROADMAP.md](docs/ROADMAP.md) for the technical architecture
and implementation plan.

## Status

Early implementation. Working today (each module built, tested, and adversarially
reviewed):

- `nimblecas.core` — fundamental types, `CowPtr<T>` copy-on-write pointer, and
  `Result<T>` / `MathError` railway-oriented error handling (`std::expected`).
- `nimblecas.symbolic` — immutable `Expr` expression trees (a `std::variant` of
  symbol / constant / add / mul / power / function nodes), structural equality,
  structural hashing, and the Cohen primitives `free_of` and `substitute`.
- `nimblecas.parallel` — deterministic fork–join runtime and order-preserving tree
  combinators over Intel oneTBB (Linux/macOS), Microsoft PPL (Windows), or a serial
  fallback, with grain-size cost gating.
- `nimblecas.simplify` — Cohen automatic simplification: exact overflow-checked
  rational constant folding, algebraic identities, flattening, canonical ordering,
  and combination of like terms / like bases.
- `nimblecas.cache` — `ExprMemo`, a sharded concurrent hash-consing / memoization
  table keyed by structural identity (each unique subtree transformed once).
- `nimblecas.diff` — symbolic differentiation (sum, product/Leibniz, general power,
  and chain rules) with an elementary + special-function derivative table
  (trig, inverse-trig, hyperbolic, `erf`, `gamma`, `lambertW`, …).
- `nimblecas.vectorcalc` — vector calculus over `diff`: gradient, divergence, curl,
  Laplacian, Jacobian, Hessian, and directional / total derivatives as exact,
  automatically-simplified compositions of partial derivatives (so `curl(grad f)`
  and `div(curl F)` collapse to zero by Clairaut cancellation).
- `nimblecas.simd` — multi-register SIMD engine with runtime dynamic dispatch
  (AVX-512 → AVX2 → scalar) for bit-identical elementwise `float32` kernels.
- `nimblecas.polynomial` — dense univariate `int64` polynomials: overflow-checked
  ring operations, primitive-Euclidean gcd, Yun square-free factorization, exact
  and SIMD-batch evaluation.
- `nimblecas.ratpoly` — exact `Rational` fractions and dense polynomials over
  `Q[x]`: overflow-checked field arithmetic, division-with-remainder, and a monic
  Euclidean gcd (the field the integer `Polynomial` is not).
- `nimblecas.polyexpr` — bridge between symbolic `Expr` and dense `Polynomial`
  (`to_polynomial` / `from_polynomial`, polynomial gcd and square-free factoring).
- `nimblecas.pfd` — square-free partial-fraction decomposition over `Q[x]` (Yun
  factorization, Bézout distinct-factor split, base-`b` power expansion) — the
  substrate for Hermite reduction / rational-function integration.
- `nimblecas.ratint` — Hermite reduction of `int A/B dx` over `Q`: an exact rational
  part plus a square-free-denominator logarithmic integrand, without fully factoring
  `B` — the rational-part half of rational-function integration (Rothstein–Trager next).
- `nimblecas.resultant` — resultant and discriminant over `Q[x]` via the Euclidean
  remainder sequence: common-factor / repeated-root detection — the substrate for the
  subresultant PRS, multivariate GCD, and the Rothstein–Trager resultant.
- `nimblecas.rothstein` — Rothstein–Trager logarithmic integration over `Q(x)`: the
  residue resultant `R(t) = res_x(D, A − t·D')` sampled and interpolated, emitting the
  rational-residue logarithms of a square-free-denominator integrand — the logarithmic-part
  half of rational-function integration (completing it with Hermite reduction).
- `nimblecas.integrate` — rational-function integration capstone over `Q(x)`: runs Hermite
  reduction then Rothstein–Trager to assemble the complete `int A/B dx` as a rational part
  plus a sum of residue-weighted logarithms (`not_implemented` for the algebraic-residue case).
- `nimblecas.testing` — internal test framework (no external test dependency).
- **Python bindings** via nanobind (`nimblecas_ext`), dependencies managed with uv.

## Architecture

Expressions are **immutable trees** shared through a copy-on-write pointer
(`CowPtr<T>`, backed by `shared_ptr`), so copies are O(1) refcount bumps and are safe
to read concurrently — the foundation for later parallel evaluation. Nodes are a
type-safe `std::variant` (no inheritance) to stay small and cache-friendly.

The engine is layered as one C++23 **module per concern**, with explicit dependency
edges. Two chains sit on the common `core` foundation — a symbolic chain and a
numeric chain — joined by `polyexpr`:

```
                core
                  │
   ┌──────────────┼────────────────────────┐
   │              │                         │
  simd        parallel                  symbolic
   │              │      │                  │
   ▼              └──────┼────────┐         │
polynomial              cache   simplify    │
   │    │                 │        │        │
   ▼    ▼                 └────┬───┴────► diff ──► vectorcalc
polyexpr ratpoly               │
      ┌─────┴─────┐            │
      ▼           ▼            ▼
     pfd      resultant  bindings (nanobind: symbolic, simplify, diff, polyexpr)
      │           │
      ▼           ▼
    ratint    rothstein
      │           │
      └─────┬─────┘
            ▼
        integrate

testing  (stands alone)
```

See [docs/architecture/overview.md](docs/architecture/overview.md) for the exact
`import` edges and rationale.

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
