# Parallel Tree Computation — Design

**Status:** design (foundation being implemented)
**Author:** Olumuyiwa Oluwasanmi

Symbolic manipulation is the core of NimbleCAS, and it must scale across cores (and
later GPUs and cluster nodes). Tree manipulation is *not* inherently serial — there is
a rich body of parallel tree algorithms, and NimbleCAS's data model is chosen
specifically to make them safe and cheap. This document sets the design so that every
symbolic operation parallelizes by construction rather than as an afterthought.

## 1. The enabling property: COW immutability

Expressions are **immutable** trees shared through a copy-on-write pointer
(`CowPtr<T>`, backed by `shared_ptr`). Two consequences drive everything below:

- **Reads are lock-free and data-race-free.** Any number of threads may traverse the
  same (sub)tree concurrently with no synchronization.
- **Transformations are functional.** `simplify`, `differentiate`, `substitute`, …
  never mutate; they build a *new* immutable tree. So independent subtrees can be
  transformed by different threads with **zero shared mutable state** — the classic
  precondition for trivially-correct parallelism.

This is why the tree layer, not just the numeric layer, is a first-class parallel
target.

## 2. Execution model: fork–join over tree structure

Every structural operation has the shape *"transform each child, then combine"*.
Children are independent, so they fork; the combine step joins. This recursive
divide-and-conquer maps directly onto a **work-stealing scheduler**:

- **Linux:** Intel **oneTBB** (`task_group` / `parallel_for` / `parallel_invoke`).
- **Windows:** Microsoft **PPL** (`concurrency::parallel_for` / `parallel_invoke`).
- **Fallback:** serial execution (identical results).

Work-stealing gives automatic load balancing for the irregular, unbalanced trees that
CAS produces (a sum with one huge term and many small ones must not stall on the big
one).

## 3. Grain control — parallelism only where it pays

Most expressions in a CAS session are small; forking them would be pure overhead. Every
combinator is gated by a **grain-size cutoff** based on a subtree *cost estimate*:

- Each node carries (or memoizes) a cheap **subtree weight** ≈ node count.
- Fork only when a child's weight exceeds the cutoff; otherwise recurse serially.

This keeps small-tree latency identical to the serial engine while letting large
expansions (high-order HAM/ADM series, big polynomial trees, deep Adomian polynomials)
saturate the cores. The cutoff is a single tunable constant, profiled with `perf`
(Rule 43/58), not guessed.

## 4. Determinism is mandatory (Rule 55)

Results must be **independent of thread scheduling** — bit-identical to the serial run.
This is achieved structurally, not by luck:

- **`parallel_transform` is order-preserving**: child *i* → result slot *i*. A parallel
  map over a sum's terms yields the same vector as the serial map.
- **Reductions are associative and then canonicalized.** Automatic simplification
  already sorts terms/factors into a canonical order and combines like terms by
  structural equality; doing the per-child work in parallel and the grouping/sort
  afterward is deterministic regardless of completion order.
- **No floating-point reductions in the symbolic layer.** Constant folding uses exact
  overflow-checked rationals, so there is no FP non-associativity to worry about (that
  concern is confined to the numeric/SIMD layer, which already forbids `-ffast-math`).

## 5. Shared-subexpression memoization: concurrent hash-consing

Real expressions are **DAGs**: the same subtree appears many times (e.g. every Adomian
polynomial reuses lower-order terms; Jacobian/Hessian assembly differentiates the same
operands repeatedly). A **concurrent hash-consing table** turns repeated work into O(1)
lookups:

- Key = structural hash (`hash_value`, already implemented) + `is_equivalent_to` on
  collision; value = the canonical interned subtree.
- Because nodes are immutable, an interned node is safely shared across all threads.
- Each unique subtree is simplified/differentiated **once**; identical subtrees dedupe.

This collapses exponential repeated work to linear and is the mechanism behind the
"distributed hash-consing" of ROADMAP §6.2 — the same table, made networked, at cluster
scale. Backing store: TBB `concurrent_hash_map` (Linux) / PPL `concurrent_unordered_map`
(Windows).

## 6. Linearization: the bridge to SIMD and GPU

For bulk, data-parallel work (parallel pattern matching, subterm search, and GPU kernel
evaluation), pointer trees are the wrong shape. NimbleCAS **flattens** a tree into a
contiguous array in Polish/prefix order (ROADMAP §5.2):

- Tree → array via an **Euler tour**; node offsets computed by **parallel prefix sum /
  list ranking** (classic O(log n)-depth parallel primitives).
- The flat token array enables **coalesced GPU scans** and SIMD-friendly traversal with
  no pointer chasing and no branch divergence.

This is the hand-off from the symbolic tree layer to the SIMD (`nimblecas.simd`) and GPU
(NVRTC/Triton, ROADMAP §5) layers.

## 7. Distributed extension (later)

The same fork–join decomposition extends across machines: subtree tasks become nodes of
a DAG scheduled by the `StochasticGraphExecutionEngine` (ROADMAP §6), with the
hash-consing table promoted to a distributed key–value memo. Nothing about the local
design has to change — only where a task runs.

## 8. Module & API shape

A dedicated module `nimblecas.parallel` provides the runtime abstraction and the
tree-specific combinators; the symbolic modules depend on it and express their
recursions through it:

```
core ─► parallel ─┐
core ─► symbolic ─┴─► simplify ─► diff
                        (all call the parallel combinators with a cost estimate)
```

Combinators (each with a serial fallback below the grain cutoff):

- `run_in_parallel(f, g, …)` — fork–join of independent tasks (e.g. differentiate a
  power's base and exponent at once).
- `parallel_transform(children, f, grain)` — order-preserving parallel map.
- `parallel_reduce(children, map, combine, identity, grain)` — associative fold.
- `intern(expr)` — concurrent hash-cons lookup/insert.
- `flatten(expr) -> tokens` / `unflatten(tokens) -> expr` — linearization for SIMD/GPU.

Symbolic operations become, schematically:

```cpp
// differentiate a sum: each term is independent -> parallel_transform
auto terms = parallel_transform(node.terms,
                                [&](const Expr& t){ return differentiate(t, x); },
                                grain);
return Expr::sum(std::move(terms));   // combine + canonicalize (deterministic)
```

## 9. Rollout

1. `nimblecas.parallel` runtime + `run_in_parallel` / `parallel_transform` (TBB / PPL /
   serial) with grain control. **← starting here**
2. Route `differentiate` and `substitute` (pure order-preserving maps) through it.
3. Subtree-weight memoization to drive the grain cutoff cheaply.
4. Concurrent hash-consing; route `simplify` and `differentiate` through the memo.
5. Linearization primitives for the SIMD/GPU bridge.
6. `perf`-profile the grain cutoff on representative large expressions.
