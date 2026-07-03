# NimbleCAS Architecture Overview

**Author:** Olumuyiwa Oluwasanmi

This document describes how NimbleCAS is put together: the C++23 module graph,
the immutable copy-on-write data model, the railway error model, the
`import std` strategy, and the cross-platform parallel runtime. For the detailed
per-module API, follow the links to the [module reference](../reference/core.md);
for the parallel tree design in depth, see
[parallel tree computation](parallel-tree-computation.md).

## 1. Layered module graph

NimbleCAS is built as **one C++23 module per concern**, with explicit dependency
edges enforced by the module system (no macro leakage, no hidden includes). The
current graph:

```
                nimblecas.core
                      │
      ┌───────────────┼───────────────────────────┐
      │               │                            │
nimblecas.simd  nimblecas.parallel          nimblecas.symbolic ──────┐
      │               │        │                   │                 │
      │               └────────┼──────────┐        │                 │
      ▼                        ▼           ▼        ▼                 ▼
nimblecas.polynomial     nimblecas.cache   nimblecas.simplify   (free_of,
      │      │                 │                   │             substitute)
      ▼      ▼                 └──────┬────────────┤
nimblecas.  nimblecas.ratpoly        ▼            ▼
 polyexpr                │     nimblecas.diff  ◄──┘
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

nimblecas.testing  (stands alone; depends only on std)
nimblecas_ext      (nanobind bindings; imports symbolic, simplify, diff, polyexpr)
```

Edges as declared in the sources (`import` statements):

| Module | Imports |
| :--- | :--- |
| [`core`](../reference/core.md) | `std` |
| [`parallel`](../reference/parallel.md) | `std` |
| [`simd`](../reference/simd.md) | `std` |
| [`testing`](../reference/testing.md) | `std` |
| [`symbolic`](../reference/symbolic.md) | `core`, `parallel` |
| [`cache`](../reference/cache.md) | `core`, `symbolic` |
| [`simplify`](../reference/simplify.md) | `core`, `symbolic`, `parallel`, `cache` |
| [`diff`](../reference/diff.md) | `core`, `symbolic`, `simplify`, `parallel`, `cache` |
| [`polynomial`](../reference/polynomial.md) | `core`, `simd` |
| [`ratpoly`](../reference/ratpoly.md) | `core`, `polynomial` |
| [`polyexpr`](../reference/polyexpr.md) | `core`, `symbolic`, `polynomial` |
| [`pfd`](../reference/pfd.md) | `core`, `ratpoly` |
| [`ratint`](../reference/ratint.md) | `core`, `ratpoly`, `pfd` |
| [`resultant`](../reference/resultant.md) | `core`, `ratpoly` |
| [`rothstein`](../reference/rothstein.md) | `core`, `ratpoly`, `resultant` |
| [`integrate`](../reference/integrate.md) | `core`, `ratpoly`, `ratint`, `rothstein` |
| bindings (`nimblecas_ext`) | `core`, `symbolic`, `simplify`, `diff`, `polyexpr` |

Two chains sit on the common `core` foundation:

- **The symbolic chain** — `core → parallel/symbolic → {simplify, cache} → diff`
  — does exact tree manipulation.
- **The numeric chain** — `core → simd → polynomial → {polyexpr, ratpoly}` —
  does dense polynomial arithmetic and the SIMD numeric fast path, with
  `polyexpr` bridging back to the symbolic `Expr` and `ratpoly` lifting `Z[x]`
  into the coefficient field `Q[x]` for exact division-with-remainder, on which
  `pfd` builds square-free partial-fraction decomposition, `ratint` builds
  Hermite reduction (the rational part of rational-function integration),
  `resultant` builds the resultant and discriminant (common-factor and
  repeated-root detection) via the Euclidean remainder sequence, `rothstein`
  builds Rothstein–Trager logarithmic integration (the logarithmic part, for
  rational residues) on top of that resultant, and `integrate` is the capstone
  that joins the two halves — running Hermite reduction then Rothstein–Trager — to
  assemble the complete `int A/B dx` as a rational part plus residue-weighted
  logarithms.

## 2. The immutable, copy-on-write data model

The central data type is [`Expr`](../reference/symbolic.md): a handle to an
**immutable** expression node shared through
[`CowPtr<T>`](../reference/core.md) (backed by `std::shared_ptr`).

- **Node kinds are a `std::variant`, not a class hierarchy.** An `ExprNode`
  holds one of `SymbolNode`, `ConstantNode`, `AddNode`, `MulNode`, `PowerNode`,
  `FunctionNode`. A type-safe union keeps nodes small and cache-friendly and
  makes visiting exhaustive (a `static_assert` breaks the build if a new
  alternative is left unhandled by any visitor). `ConstantNode` itself unifies
  `int64`, `double`, and exact rational (`pair<int64,int64>`).
- **Copies are O(1).** Copying an `Expr` is an atomic refcount bump, and shared
  subtrees are read-only, so any number of threads may traverse the same tree
  concurrently without synchronization.
- **Transformations are functional.** `simplify`, `differentiate`, and
  `substitute` never mutate; they build a *new* immutable tree. This is the
  precondition for the trivially-correct parallel recursion in §5.
- **Memoized `size_` and `hash_`.** Every `Expr` computes, once at construction,
  its subtree node count and a structural hash, so `size()` and
  `structural_hash()` are O(1). `size()` drives the parallel cost gate;
  `structural_hash()` keys hash-consing.
- **Structural equality and hashing.** `is_equivalent_to` is *syntactic*: two
  trees are equal when identical (doubles compared bitwise, so `NaN` equals
  itself and `±0` stay distinct). It short-circuits in O(1) on a shared COW node
  or a hash mismatch. The hash is consistent: `a == b ⇒ hash(a) == hash(b)`. It
  is automatic simplification that maps *mathematically*-equal expressions to
  identical trees.
- **Hash-consing.** Because interned nodes are immutable, the concurrent
  [`ExprMemo`](../reference/cache.md) can share one canonical result per unique
  subtree across all threads — collapsing the repeated work of DAG-shaped
  expressions to O(1) lookups.

## 3. The railway error model

Fallible operations never throw (Rule 32). They return
`Result<T> = std::expected<T, MathError>` and are threaded through the
`std::expected` monadic surface. `MathError` is a small enum
(`division_by_zero`, `undefined_value`, `overflow`, `domain_error`,
`syntax_error`, `not_implemented`) with a `to_string_view` renderer.

Representative surfaces: a zero denominator in `Expr::rational`
(`division_by_zero`); `0^0` in `simplify` (`undefined_value`); integer overflow
in rational folding or polynomial arithmetic (`overflow`); a non-polynomial
input to `to_polynomial` (`not_implemented`); an inexact polynomial division
(`domain_error`). The Python bindings translate a `MathError` into a
`ValueError` at the boundary — the one place exceptions appear.

## 4. The `import std` module strategy

NimbleCAS uses `import std;` everywhere instead of classic `#include`s. Rather
than CMake's version-locked experimental `import std` support, the build
compiles the standard library's own std module source as an ordinary
`CXX_MODULES` library target (`cmake/StdModule.cmake`), so CMake's module
scanner resolves `import std;` with explicit, reproducible dependency edges
(Rules 12/41/50–52):

| Platform | std module source |
| :--- | :--- |
| Linux/macOS (clang + libc++) | libc++'s `std.cppm` (e.g. `/usr/lib/llvm-22/share/libc++/v1/std.cppm`) |
| Windows (clang targeting `windows-msvc` + MSVC STL) | the MSVC toolset's `std.ixx` |

A handful of macro-only facilities that `import std` cannot provide (notably
`assert` via `<cassert>`) are pulled into the **global module fragment** of the
modules that need them.

## 5. Cross-platform parallel strategy

Symbolic manipulation is a **parallel tree-transformation** problem, not a serial
recursion (see [parallel tree computation](parallel-tree-computation.md)). The
[`nimblecas.parallel`](../reference/parallel.md) module provides a thin,
deterministic fork–join layer with a backend chosen at compile time:

| Platform | Backend |
| :--- | :--- |
| Windows | Microsoft PPL (Concurrency Runtime) |
| Linux/macOS (with oneTBB headers) | Intel oneTBB |
| otherwise | serial fallback (identical results) |

The TBB/PPL headers live in the global module fragment and are invisible to
importers; all runtime calls are confined to the concrete functions
`for_ranges` / `invoke2`. The exported combinators — `transform_index`,
`transform`, and the cost-gated `transform_index_if` — are thin `std`-only
templates over those.

Two design invariants make this safe and predictable:

- **Determinism (Rule 55).** The parallel maps are *order-preserving* (child *i*
  → result slot *i*), and constant folding is exact (no floating-point
  reduction non-associativity in the symbolic layer), so results are
  bit-identical to a serial run regardless of scheduling.
- **Grain control.** Forking small trees is pure overhead, so the symbolic
  layers only fan out when a subtree's `size()` reaches
  `parallel_cost_threshold` (512); below `default_grain` (256) items,
  `for_ranges` runs serially. Both are tunable, `perf`-profiled constants.

The numeric layer parallelizes differently: [`nimblecas.simd`](../reference/simd.md)
dispatches elementwise `float32` kernels to the best CPU ISA at runtime
(AVX-512 → AVX2 → AVX → scalar), keeping every path bit-identical via `std::fma`.

## See also

- [Parallel tree computation](parallel-tree-computation.md) — the parallel design in depth.
- Module reference: [core](../reference/core.md) · [symbolic](../reference/symbolic.md) · [simplify](../reference/simplify.md) · [cache](../reference/cache.md) · [diff](../reference/diff.md) · [parallel](../reference/parallel.md) · [simd](../reference/simd.md) · [polynomial](../reference/polynomial.md) · [ratpoly](../reference/ratpoly.md) · [polyexpr](../reference/polyexpr.md) · [pfd](../reference/pfd.md) · [ratint](../reference/ratint.md) · [resultant](../reference/resultant.md) · [rothstein](../reference/rothstein.md) · [integrate](../reference/integrate.md)
- [Documentation hub](../Index.md)
