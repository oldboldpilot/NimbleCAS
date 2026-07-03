# `nimblecas.parallel` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/parallel/parallel.cppm`

A thin, deterministic **fork–join** layer over Intel oneTBB (Linux/macOS) and
Microsoft PPL (Windows), with a serial fallback below a grain-size cutoff.
Because NimbleCAS expressions are immutable (COW), independent subtrees can be
transformed concurrently with no locking; these combinators express that safely.
For the full design, see
[Parallel tree computation](../architecture/parallel-tree-computation.md).

```cpp
import nimblecas.parallel;
```

Namespace: `nimblecas::parallel`. Depends only on `std`.

## Backend selection

The backend is chosen at compile time by preprocessor logic in the global module
fragment (the TBB/PPL headers are *not* visible to importers — all runtime calls
are confined to the non-template functions `for_ranges` / `invoke2`):

| Platform | Backend | Macro |
| :--- | :--- | :--- |
| `_WIN32` | Microsoft PPL (`<ppl.h>`) | `NIMBLECAS_PARALLEL_BACKEND_PPL` |
| Linux/macOS with `<tbb/parallel_for.h>` | Intel oneTBB | `NIMBLECAS_PARALLEL_BACKEND_TBB` |
| otherwise | serial (identical results) | `NIMBLECAS_PARALLEL_BACKEND_SERIAL` |

## Constants

| Name | Value | Purpose |
| :--- | :--- | :--- |
| `default_grain` | `256` | Work below this many items runs serially (forking would cost more than it saves). |
| `parallel_cost_threshold` | `512` | Subtree node count at/above which recursing into a node's children in parallel is worthwhile. The symbolic layers gate on `Expr::size()` against this. |

Both are tunable constants, intended to be profiled with `perf` on
representative large expressions (Rule 43/58).

## Runtime primitives

```cpp
auto for_ranges(std::size_t n, std::size_t grain,
                const std::function<void(std::size_t, std::size_t)>& body) -> void;
auto invoke2(const std::function<void()>& f, const std::function<void()>& g) -> void;
[[nodiscard]] auto max_concurrency() noexcept -> unsigned;
[[nodiscard]] auto backend() noexcept -> std::string_view;
```

- **`for_ranges(n, grain, body)`** — invokes `body(begin, end)` over a partition
  of disjoint sub-ranges covering exactly `[0, n)`, in parallel (work-stealing)
  when `n >= grain`, else serially. Per-index work must be independent. Blocking:
  once it returns, all ranges have run.
- **`invoke2(f, g)`** — runs two independent tasks, possibly concurrently.
- **`max_concurrency()`** — worker threads available (informational; `>= 1`).
- **`backend()`** — `"ppl"`, `"tbb"`, or `"serial"`.

## Tree combinators

```cpp
template <typename Fn>
[[nodiscard]] auto transform_index(std::size_t n, Fn fn,
                                   std::size_t grain = default_grain)
    -> std::vector<std::invoke_result_t<Fn&, std::size_t>>;

template <typename T, typename Fn>
[[nodiscard]] auto transform(std::span<const T> in, Fn fn,
                             std::size_t grain = default_grain)
    -> std::vector<std::invoke_result_t<Fn&, const T&>>;

template <typename Fn>
[[nodiscard]] auto transform_index_if(bool parallel, std::size_t n, Fn fn)
    -> std::vector<std::invoke_result_t<Fn&, std::size_t>>;
```

- **`transform_index(n, fn, grain)`** — order-preserving parallel map by index:
  `result[i] = fn(i)` for `i` in `[0, n)`. **Deterministic** — the result
  depends only on `fn`, not on scheduling. Serial below `grain`.
  Results are staged in per-index `std::optional` slots so `R` need not be
  default-constructible (`Expr` is not); each index writes its own disjoint
  slot, so concurrent fills never race.
- **`transform(in, fn, grain)`** — the same, over a `std::span<const T>`:
  `result[i] = fn(in[i])`.
- **`transform_index_if(parallel, n, fn)`** — the **cost-gated** entry point the
  symbolic layers use. When `parallel` is `true`, each index becomes an
  independent task (grain 1; the backend auto-chunks) so both wide and deep
  trees fan out; when `false`, it runs fully serially. Deterministic either way.

**Precondition.** `fn` must be safe to invoke concurrently for distinct indices
(stateless or internally synchronising). In NimbleCAS this holds because the
tree is immutable (COW): `fn` only reads shared subexpressions and produces new
ones.

## Example

```cpp
import nimblecas.parallel;
using namespace nimblecas;

// Order-preserving parallel squares of 0..n.
auto squares = parallel::transform_index(1000, [](std::size_t i) { return i * i; });

// Cost-gated: parallel only when the caller decides the work is large enough.
const bool big = /* subtree size */ 4000 >= parallel::parallel_cost_threshold;
auto out = parallel::transform_index_if(big, terms.size(),
                                        [&](std::size_t i) { return f(terms[i]); });

std::println("backend={} workers={}", parallel::backend(), parallel::max_concurrency());
```

## See also

- [Parallel tree computation](../architecture/parallel-tree-computation.md) — the design this module implements.
- [`nimblecas.symbolic`](symbolic.md), [`simplify`](simplify.md), [`diff`](diff.md) — callers of these combinators.
- [Documentation hub](../Index.md)
