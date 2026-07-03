# `nimblecas.cache` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/cache/cache.cppm`

A thread-safe **concurrent hash-consing / memoization** table for pure
`Expr -> Result<Expr>` transforms (such as [`simplify`](simplify.md) and
[`differentiate`](diff.md)). Because `Expr` is immutable (COW), a memoized
result is safe to share across threads, so each unique subexpression is
transformed once even under parallel recursion.

```cpp
import nimblecas.cache;
```

Depends on [`core`](core.md) and [`symbolic`](symbolic.md).

## `class ExprMemo`

Keyed by **structural identity**: the O(1) memoized `structural_hash()` picks a
shard and a bucket, and `is_equivalent_to` resolves hash collisions within the
bucket. The table is **sharded** (default 64 shards) so concurrent lookups on
distinct keys rarely contend on the same mutex.

### Constructor

```cpp
explicit ExprMemo(std::size_t shard_count = 64);
```

Creates a memo with `shard_count` independently-locked shards (must be `> 0`).

### Methods

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `get_or_compute` | `auto get_or_compute(const Expr& key, const std::function<Result<Expr>()>& compute) -> Result<Expr>` | Returns the cached result for `key`, computing and caching it on first sight. |
| `get_or_compute_value` | `auto get_or_compute_value(const Expr& key, const std::function<Expr()>& compute) -> Expr` | Variant for **total** (non-failing) transforms, e.g. the raw differentiation pass. |
| `size` | `auto size() const -> std::size_t` | Number of distinct entries cached (for tests / introspection). |

`compute` **must be a pure function of `key`** — its result may be reused for
any structurally-equal expression.

### Compute-outside-lock discipline

`get_or_compute` runs `compute()` **without holding any shard lock**:

1. Fast path — look up `key` under the shard lock; return a hit.
2. On a miss, release the lock and run `compute()`. Because the lock is not
   held, `compute` may freely recurse into `get_or_compute` on *other* keys
   (other shards) without any risk of deadlock — this is exactly what the
   recursive simplify/differentiate passes do.
3. Re-acquire the lock, **double-check** in case another thread computed the
   same key concurrently, then insert.

`get_or_compute_value` wraps `get_or_compute` and asserts the cached value is
never an error branch — a value-only memo instance is only ever used with a
single total transform.

## Example

```cpp
import nimblecas.cache;
import nimblecas.simplify;   // for illustration
using namespace nimblecas;

ExprMemo memo;
const Expr big = /* a large expression with repeated subtrees */;

// Each unique subtree is simplified once; identical subtrees dedupe.
auto result = memo.get_or_compute(big, [&] { return simplify(big); });
std::println("distinct cached entries: {}", memo.size());
```

In practice you rarely construct an `ExprMemo` directly: `simplify` and
`differentiate` each create a per-call memo internally.

## See also

- [`nimblecas.simplify`](simplify.md) / [`nimblecas.diff`](diff.md) — the transforms that use it.
- [Parallel tree computation §5](../architecture/parallel-tree-computation.md) — concurrent hash-consing design.
- [Documentation hub](../Index.md)
